"""Stateless selection of one code-qualified target option."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..clutter.target_options import TargetOption
from ..common.scene_state import ClutterSceneState
from .payload_views import compact_task_state, target_option_policy_view
from .qwen_client import QwenCallResult, QwenSelectionClient
from .schemas import PolicyOutputError, TARGET_SELECTION_SCHEMA, parse_target_selection


@dataclass(frozen=True)
class TargetSelectionOutcome:
    selected: TargetOption | None
    decision_source: str
    reason_codes: tuple[str, ...]
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    error: str | None = None


class QwenTargetSelector:
    def __init__(self, client: QwenSelectionClient, recent_failure_limit: int = 5):
        self._client = client
        self._recent_failure_limit = max(0, int(recent_failure_limit))

    def select(
        self,
        scene: ClutterSceneState,
        task_state: Mapping[str, Any],
        options: Sequence[TargetOption],
        image_paths: Sequence[str] = (),
        artifact_dir: str | None = None,
    ) -> TargetSelectionOutcome:
        if not options:
            return TargetSelectionOutcome(None, "reobserve", (), {}, {"skipped_reason": "no_feasible_target_options"})
        if len(options) == 1:
            return TargetSelectionOutcome(
                options[0], "single_feasible_target_option", ("only_physical_target",),
                {"skipped_reason": "single_feasible_target_option"},
                {"skipped_reason": "single_feasible_target_option"},
            )
        request = _target_request(scene, task_state, options, self._recent_failure_limit)
        call = self._client.call("target_selection", request, TARGET_SELECTION_SCHEMA, image_paths, artifact_dir)
        if not call.success or call.parsed_output is None:
            return _invalid(call, request)
        try:
            parsed = parse_target_selection(call.parsed_output, options, scene.scene_revision)
        except PolicyOutputError as exc:
            return TargetSelectionOutcome(
                None, "policy_invalid_output", (), request,
                _response_log(call), str(exc),
            )
        selected = next(item for item in options if item.target_option_id == parsed["selected_target_option_id"])
        return TargetSelectionOutcome(
            selected, "qwen_target_selection", tuple(parsed["reason_codes"]),
            request, _response_log(call),
        )


def _target_request(
    scene: ClutterSceneState,
    task_state: Mapping[str, Any],
    options: Sequence[TargetOption],
    failure_limit: int,
) -> dict[str, Any]:
    return {
        "protocol": "qwen_target_selection_v1",
        "scene_revision": scene.scene_revision,
        "task_type": next(iter(options)).task_type,
        "task_state": compact_task_state(task_state),
        "scene_summary": {
            "visible_tracks": list(scene.visible_tracks),
            "missing_expected_tracks": list(scene.missing_expected_tracks),
            "completed_tracks": list(scene.completed_tracks),
            "protected_tracks": list(scene.protected_tracks),
        },
        "target_options": [target_option_policy_view(item) for item in options],
        "recent_failures": [dict(item) for item in scene.recent_action_results[-failure_limit:]] if failure_limit else [],
        "forbidden_action_fingerprints": list(scene.forbidden_action_fingerprints),
        "selection_priority": [
            "direct_task_progress_with_low_physical_risk",
            "wider_continuous_safe_grasp_yaw_interval",
            "larger_nearest_neighbor_clearance",
            "fewer_neighbors",
            "lower_clearance_cost",
            "releases_more_unfinished_tracks",
            "farther_from_protected_structure",
            "avoid_recently_failed_target",
            "never_use_object_id_array_or_image_order",
        ],
        "output_contract": {
            "selected_target_option_id": "one exact supplied target_option_id",
            "reason_codes": "short comparison codes only",
        },
    }


def _invalid(call: QwenCallResult, request: Mapping[str, Any]) -> TargetSelectionOutcome:
    return TargetSelectionOutcome(
        None, "policy_invalid_output", (), request, _response_log(call),
        call.error_type or call.error_message or "invalid target selection output",
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
