"""Stateless selection of one immutable physical edge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..clutter.target_options import TargetOption
from ..clutter.push_orientation_priority import prefer_axis_aligned_pushes
from ..common.action_edges import PhysicalActionEdge
from ..common.scene_state import ClutterSceneState
from .payload_views import (
    compact_task_state,
    physical_edge_policy_view,
    target_option_policy_view,
)
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
        eligible = list(prefer_axis_aligned_pushes(eligible))
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
        parsed_output, alias_repairs = _repair_redundant_action_aliases(
            call.parsed_output, eligible,
        )
        try:
            parsed = parse_edge_selection(parsed_output, eligible, scene.scene_revision)
        except PolicyOutputError as exc:
            return EdgeSelectionOutcome(None, (), "policy_invalid_output", (), request, _response_log(call), str(exc))
        selected = next(edge for edge in eligible if edge.candidate_id == parsed["selected_candidate_id"])
        response = _response_log(call)
        if alias_repairs:
            response["candidate_id_alias_repairs"] = alias_repairs
        return EdgeSelectionOutcome(
            selected, tuple(parsed["backup_candidate_ids"]),
            "qwen_edge_selection_alias_resolved" if alias_repairs else "qwen_edge_selection",
            tuple(parsed["reason_codes"]), request, response,
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
        "task_state": compact_task_state(task_state),
        "selected_target_option": target_option_policy_view(target),
        "physical_edges": [physical_edge_policy_view(edge, scene) for edge in edges],
        "protected_tracks": list(scene.protected_tracks),
        "recent_failures": [dict(item) for item in scene.recent_action_results[-failure_limit:]] if failure_limit else [],
        "forbidden_action_fingerprints": list(scene.forbidden_action_fingerprints),
        "selection_priority": [
            "all_physical_prechecks_passed",
            "larger_direct_task_progress",
            "larger_clearance_gain",
            "axis_aligned_0_or_90_push_before_diagonal_when_equivalent",
            "larger_prepush_clearance",
            "lower_protected_structure_risk",
            "lower_motion_planning_cost",
            "avoid_failed_fingerprint",
        ],
        "output_contract": {
            "selected_candidate_id": (
                "one exact supplied candidate_id; copy it verbatim and never insert action_type text"
            ),
            "backup_candidate_ids": "zero to three distinct supplied ids",
            "reason_codes": "short comparison codes only",
        },
    }


def _repair_redundant_action_aliases(
    parsed_output: Mapping[str, Any],
    edges: Sequence[PhysicalActionEdge],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Remove one known redundant action token only when the match is unique.

    Qwen occasionally copies the exact first staging ID but expands
    ``_staging_`` to ``_extract_to_staging_`` using the adjacent action_type.
    That is a serialization alias, not a different physical decision.  No
    edit-distance or prefix matching is allowed here: every other unknown ID
    remains invalid and triggers the normal reobserve path.
    """
    output = dict(parsed_output)
    exact_ids = {edge.candidate_id for edge in edges}
    aliases: dict[str, set[str]] = {}
    for edge in edges:
        if edge.action_type.value != "extract_to_staging":
            continue
        alias = edge.candidate_id.replace(
            "_staging_", "_extract_to_staging_", 1,
        )
        if alias != edge.candidate_id:
            aliases.setdefault(alias, set()).add(edge.candidate_id)

    repairs: list[dict[str, str]] = []

    def resolve(value: Any) -> str:
        candidate_id = str(value)
        if candidate_id in exact_ids:
            return candidate_id
        matches = aliases.get(candidate_id, set())
        if len(matches) != 1:
            return candidate_id
        repaired = next(iter(matches))
        repairs.append({"received": candidate_id, "resolved": repaired})
        return repaired

    output["selected_candidate_id"] = resolve(output.get("selected_candidate_id", ""))
    output["backup_candidate_ids"] = [
        resolve(value) for value in output.get("backup_candidate_ids", [])
    ]
    return output, repairs


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
