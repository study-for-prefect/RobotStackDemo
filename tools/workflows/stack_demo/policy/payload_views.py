"""Compact, decision-complete views for restricted Qwen ID selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..clutter.target_options import TargetOption
from ..common.action_edges import PhysicalActionEdge
from ..common.scene_state import ClutterSceneState


def target_option_policy_view(option: TargetOption) -> dict[str, Any]:
    """Keep only fields used by the declared target-selection priorities."""
    value = option.to_dict()
    keys = (
        "target_option_id", "scene_revision", "target_track_id", "task_type",
        "task_role", "target_color", "direct_graspable",
        "safe_grasp_interval_count", "best_safe_yaw_span_deg",
        "best_grasp_clearance_m", "neighbor_count",
        "minimum_neighbor_clearance_m", "estimated_clearance_cost",
        "task_progress_gain", "future_obstruction_risk",
        "protected_structure_risk", "recent_failure_count", "blocker_count",
        "physical_clearance_edge_count", "best_first_step_clearance_gain_m",
        "direct_edge_count", "feasible_transport_edge_count",
        "protected_structure_min_clearance_m", "released_track_count",
        "best_moveit_cost",
    )
    return {key: value.get(key) for key in keys}


def physical_edge_policy_view(
    edge: PhysicalActionEdge,
    scene: ClutterSceneState,
) -> dict[str, Any]:
    """Omit poses and collision traces already enforced by code-owned gates."""
    physical = edge.physical_parameters
    checks = edge.precheck_results
    chain_ids = list(physical.get("chain_track_ids") or checks.get("chain_track_ids") or ())
    return {
        "candidate_id": edge.candidate_id,
        "scene_revision": edge.scene_revision,
        "action_type": edge.action_type.value,
        "target_track_id": edge.primary_target_track_id,
        "acted_track_id": edge.acted_object_track_id,
        "target_region_id": edge.target_region_id,
        "grasp_yaw_deg": physical.get("grasp_yaw_deg"),
        "push_direction": physical.get("push_direction"),
        "contact_side": physical.get("contact_side"),
        "push_distance_m": physical.get("push_distance_m"),
        "push_wrist_yaw_deg": physical.get("push_wrist_yaw_deg"),
        "push_axis_alignment_error_deg": physical.get("push_axis_alignment_error_deg"),
        "prepush_clearance_m": physical.get("prepush_clearance_m"),
        "chain_object_count": physical.get("chain_object_count", len(chain_ids)),
        "secondary_contact_expected": bool(physical.get("secondary_contact_expected")),
        "task_progress_gain": edge.task_progress_gain,
        "clearance_gain_m": edge.expected_clearance_gain_m,
        "released_track_count": len(edge.expected_released_tracks),
        "risk_score": edge.risk_score,
        "moveit_cost": checks.get("moveit_cost"),
        "moveit_plan_only": bool(checks.get("moveit_plan_only")),
        "all_physical_prechecks_passed": bool(checks.get("passed")),
        "forbidden": edge.failure_fingerprint in scene.forbidden_action_fingerprints,
        "recent_failure_count": _recent_failure_count(edge, scene.recent_action_results),
    }


def compact_task_state(task_state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove regions, slots, and geometry already represented by candidate metrics."""
    keys = (
        "scene_revision", "expected_tracks", "visible_tracks",
        "missing_expected_tracks", "completed_tracks", "unresolved_tracks",
        "current_action_result",
    )
    return {key: task_state.get(key) for key in keys if key in task_state}


def _recent_failure_count(
    edge: PhysicalActionEdge,
    recent_results: Sequence[Mapping[str, Any]],
) -> int:
    return sum(
        not bool(item.get("success"))
        and str(item.get("failure_fingerprint") or "") == edge.failure_fingerprint
        for item in recent_results
    )
