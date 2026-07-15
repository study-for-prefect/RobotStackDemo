"""Restricted Qwen target choices derived only from feasible first-step edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ..common.action_edges import PhysicalActionEdge
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .grasp_edges import GraspScanResult


@dataclass(frozen=True)
class TargetOption:
    target_option_id: str
    scene_revision: int
    target_track_id: str
    object_ref: str
    task_type: str
    task_role: str | None
    target_color: str | None
    direct_graspable: bool
    safe_grasp_interval_count: int
    best_safe_yaw_span_deg: float
    best_grasp_clearance_m: float
    neighbor_count: int
    minimum_neighbor_clearance_m: float
    edge_clearances_m: Mapping[str, float]
    blocked_sides: tuple[str, ...]
    free_sides: tuple[str, ...]
    blocker_track_ids: tuple[str, ...]
    feasible_first_step_edge_ids: tuple[str, ...]
    estimated_clearance_cost: float
    releases_track_ids: tuple[str, ...]
    task_progress_gain: float
    future_obstruction_risk: float
    protected_structure_risk: float
    recent_failure_count: int
    selection_features: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_target_options(
    scene: ClutterSceneState,
    task_type: str,
    target_edges: Mapping[str, Sequence[PhysicalActionEdge]],
    grasp_scans: Mapping[str, GraspScanResult],
    *,
    task_roles: Mapping[str, str] | None = None,
) -> list[TargetOption]:
    """Exclude every target that has no code-generated feasible first step."""
    options = []
    for option_key, edges in target_edges.items():
        feasible = [edge for edge in edges if bool(edge.precheck_results.get("passed"))]
        if not feasible:
            continue
        track_id = feasible[0].primary_target_track_id
        obj = scene.object_by_track(track_id)
        if obj is None or obj.already_completed:
            continue
        scan = grasp_scans.get(track_id)
        intervals = list(scan.safe_intervals if scan else ())
        best = intervals[0] if intervals else None
        clearance = _minimum_neighbor_clearance(obj, scene)
        recent_failures = sum(
            str(item.get("primary_target_track_id")) == track_id and not bool(item.get("success"))
            for item in scene.recent_action_results
        )
        releases = sorted({track for edge in feasible for track in edge.expected_released_tracks})
        options.append(TargetOption(
            target_option_id=f"target_r{scene.scene_revision}_{_safe_id(option_key)}",
            scene_revision=scene.scene_revision,
            target_track_id=track_id,
            object_ref=obj.object_ref,
            task_type=task_type,
            task_role=feasible[0].task_role or (task_roles or {}).get(option_key) or (task_roles or {}).get(track_id),
            target_color=obj.color if task_type == "organize_blocks" else None,
            direct_graspable=bool(intervals),
            safe_grasp_interval_count=len(intervals),
            best_safe_yaw_span_deg=0.0 if best is None else best.span_deg,
            best_grasp_clearance_m=0.0 if best is None else float(best.selected_check.get("fingertip_clearance_m", 0.0)),
            neighbor_count=len(obj.neighbors),
            minimum_neighbor_clearance_m=clearance,
            edge_clearances_m=dict(obj.edge_clearances_m),
            blocked_sides=obj.blocked_sides,
            free_sides=obj.free_sides,
            blocker_track_ids=() if scan is None else scan.blocker_track_ids,
            feasible_first_step_edge_ids=tuple(edge.candidate_id for edge in feasible),
            estimated_clearance_cost=min((edge.risk_score for edge in feasible), default=0.0),
            releases_track_ids=tuple(releases),
            task_progress_gain=max(edge.task_progress_gain for edge in feasible),
            future_obstruction_risk=max((edge.risk_score for edge in feasible), default=0.0),
            protected_structure_risk=max((float(bool(edge.protected_tracks)) * edge.risk_score for edge in feasible), default=0.0),
            recent_failure_count=recent_failures,
            selection_features={
                "not_order_based": True,
                "edge_count": len(feasible),
                "direct_edge_count": sum(edge.acted_object_track_id == track_id for edge in feasible),
                "clearance_edge_count": sum(edge.acted_object_track_id != track_id for edge in feasible),
            },
        ))
    return sorted(options, key=lambda item: item.target_option_id)


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in str(value))


def _minimum_neighbor_clearance(obj: SceneObjectState, scene: ClutterSceneState) -> float:
    values = []
    for neighbor_id in obj.neighbors:
        other = scene.object_by_track(neighbor_id)
        if other is None:
            continue
        dx = obj.center_xyz_m[0] - other.center_xyz_m[0]
        dy = obj.center_xyz_m[1] - other.center_xyz_m[1]
        center_distance = (dx * dx + dy * dy) ** 0.5
        values.append(center_distance - 0.5 * max(obj.size_xyz_m[:2]) - 0.5 * max(other.size_xyz_m[:2]))
    return round(min(values), 6) if values else 1.0
