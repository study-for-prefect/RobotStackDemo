"""Stateless selection of one immutable physical edge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..clutter.target_options import TargetOption
from ..common.action_edges import PhysicalActionEdge
from ..common.scene_state import ClutterSceneState
from .qwen_client import QwenCallResult, QwenSelectionClient
from .schemas import EDGE_SELECTION_SCHEMA, PolicyOutputError, parse_edge_selection


@dataclass(frozen=True)
class EdgeSelectionOutcome:
    selected: PhysicalActionEdge | None
    backup_candidate_ids: tuple[str, ...]
    decision_source: str
    reason_codes: tuple[str, ...]
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    error: str | None = None


class QwenEdgeSelector:
    def __init__(self, client: QwenSelectionClient, recent_failure_limit: int = 5):
        self._client = client
        self._recent_failure_limit = max(0, int(recent_failure_limit))

    def select(
        self,
        scene: ClutterSceneState,
        task_state: Mapping[str, Any],
        target: TargetOption,
        edges: Sequence[PhysicalActionEdge],
        image_paths: Sequence[str] = (),
        artifact_dir: str | None = None,
    ) -> EdgeSelectionOutcome:
        eligible = [edge for edge in edges if edge.candidate_id in target.feasible_first_step_edge_ids]
        if not eligible:
            return EdgeSelectionOutcome(None, (), "reobserve", (), {}, {"skipped_reason": "no_edges_for_selected_target"})
        if len(eligible) == 1:
            return EdgeSelectionOutcome(
                eligible[0], (), "single_feasible_edge", ("only_physical_edge",),
                {"skipped_reason": "single_feasible_edge"},
                {"skipped_reason": "single_feasible_edge"},
            )
        request = _edge_request(scene, task_state, target, eligible, self._recent_failure_limit)
        call = self._client.call("edge_selection", request, EDGE_SELECTION_SCHEMA, image_paths, artifact_dir)
        if not call.success or call.parsed_output is None:
            return _invalid(call, request)
        try:
            parsed = parse_edge_selection(call.parsed_output, eligible, scene.scene_revision)
        except PolicyOutputError as exc:
            return EdgeSelectionOutcome(None, (), "policy_invalid_output", (), request, _response_log(call), str(exc))
        selected = next(edge for edge in eligible if edge.candidate_id == parsed["selected_candidate_id"])
        return EdgeSelectionOutcome(
            selected, tuple(parsed["backup_candidate_ids"]), "qwen_edge_selection",
            tuple(parsed["reason_codes"]), request, _response_log(call),
        )


def _edge_request(
    scene: ClutterSceneState,
    task_state: Mapping[str, Any],
    target: TargetOption,
    edges: Sequence[PhysicalActionEdge],
    failure_limit: int,
) -> dict[str, Any]:
    return {
        "protocol": "qwen_edge_selection_v1",
        "scene_revision": scene.scene_revision,
        "task_state": dict(task_state),
        "selected_target_option": target.to_dict(),
        "physical_edges": [edge.to_dict() for edge in edges],
        "protected_tracks": list(scene.protected_tracks),
        "recent_failures": [dict(item) for item in scene.recent_action_results[-failure_limit:]] if failure_limit else [],
        "forbidden_action_fingerprints": list(scene.forbidden_action_fingerprints),
        "selection_priority": [
            "all_physical_prechecks_passed",
            "larger_direct_task_progress",
            "larger_clearance_gain",
            "lower_protected_structure_risk",
            "lower_motion_planning_cost",
            "avoid_failed_fingerprint",
        ],
        "output_contract": {
            "selected_candidate_id": "one exact supplied candidate_id",
            "backup_candidate_ids": "zero to three distinct supplied ids",
            "reason_codes": "short comparison codes only",
        },
    }


def _invalid(call: QwenCallResult, request: Mapping[str, Any]) -> EdgeSelectionOutcome:
    return EdgeSelectionOutcome(
        None, (), "policy_invalid_output", (), request, _response_log(call),
        call.error_type or call.error_message or "invalid edge selection output",
    )


def _response_log(call: QwenCallResult) -> dict[str, Any]:
    return {
        "raw_output": call.raw_output,
        "parsed_output": call.parsed_output,
        "success": call.success,
        "error_type": call.error_type,
        "error_message": call.error_message,
        "format_repair_used": call.format_repair_used,
    }
