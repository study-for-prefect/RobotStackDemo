"""Closed physical action-edge vocabulary and immutable edge records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


class ActionType(str, Enum):
    PICK_PLACE = "pick_place"
    EXTRACT_THEN_PLACE = "extract_then_place"
    EXTRACT_TO_STAGING = "extract_to_staging"
    REGRASP_FOR_ORIENTATION = "regrasp_for_orientation"
    PICK_AWAY_BLOCKER = "pick_away_blocker"
    NUDGE_BLOCKER = "nudge_blocker"
    PLACE_HOUSE_ROLE = "place_house_role"
    REPAIR_STRUCTURE = "repair_structure"
    REOBSERVE = "reobserve"


@dataclass(frozen=True)
class PhysicalActionEdge:
    candidate_id: str
    scene_revision: int
    action_type: ActionType
    task_type: str
    primary_target_track_id: str
    acted_object_track_id: str
    task_role: str | None
    target_region_id: str | None
    physical_parameters: Mapping[str, Any]
    expected_effects: tuple[str, ...]
    expected_clearance_gain_m: float
    expected_released_tracks: tuple[str, ...]
    task_progress_gain: float
    risk_score: float
    protected_tracks: tuple[str, ...]
    precheck_results: Mapping[str, Any]
    failure_fingerprint: str
    decision_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action_type"] = self.action_type.value
        return value


def make_edge(
    *,
    candidate_id: str,
    scene_revision: int,
    action_type: ActionType,
    task_type: str,
    primary_target_track_id: str,
    acted_object_track_id: str,
    physical_parameters: Mapping[str, Any],
    task_role: str | None = None,
    target_region_id: str | None = None,
    expected_effects: tuple[str, ...] = (),
    expected_clearance_gain_m: float = 0.0,
    expected_released_tracks: tuple[str, ...] = (),
    task_progress_gain: float = 0.0,
    risk_score: float = 0.0,
    protected_tracks: tuple[str, ...] = (),
    precheck_results: Mapping[str, Any] | None = None,
    decision_metadata: Mapping[str, Any] | None = None,
) -> PhysicalActionEdge:
    """Create an edge and derive a direction-sensitive failure fingerprint."""
    fingerprint = build_failure_fingerprint(
        scene_revision=scene_revision,
        action_type=action_type,
        primary_target_track_id=primary_target_track_id,
        acted_object_track_id=acted_object_track_id,
        task_role=task_role,
        target_region_id=target_region_id,
        physical_parameters=physical_parameters,
    )
    return PhysicalActionEdge(
        candidate_id=candidate_id,
        scene_revision=scene_revision,
        action_type=action_type,
        task_type=task_type,
        primary_target_track_id=primary_target_track_id,
        acted_object_track_id=acted_object_track_id,
        task_role=task_role,
        target_region_id=target_region_id,
        physical_parameters=dict(physical_parameters),
        expected_effects=tuple(expected_effects),
        expected_clearance_gain_m=float(expected_clearance_gain_m),
        expected_released_tracks=tuple(expected_released_tracks),
        task_progress_gain=float(task_progress_gain),
        risk_score=float(risk_score),
        protected_tracks=tuple(protected_tracks),
        precheck_results=dict(precheck_results or {}),
        failure_fingerprint=fingerprint,
        decision_metadata=dict(decision_metadata or {}),
    )


def build_failure_fingerprint(
    *,
    scene_revision: int,
    action_type: ActionType,
    primary_target_track_id: str,
    acted_object_track_id: str,
    task_role: str | None,
    target_region_id: str | None,
    physical_parameters: Mapping[str, Any],
) -> str:
    """Fingerprint all material parameters, including signed push direction."""
    value = {
        "action_type": action_type.value,
        "primary_target_track_id": primary_target_track_id,
        "acted_object_track_id": acted_object_track_id,
        "task_role": task_role,
        "target_region_id": target_region_id,
        "grasp_yaw_deg": _quantize(physical_parameters.get("grasp_yaw_deg"), 5.0),
        "grasp_pose": _pose_bin(physical_parameters.get("grasp_pose")),
        "push_direction_base": _vector_bin(physical_parameters.get("push_direction_base")),
        "push_distance_m": _quantize(physical_parameters.get("push_distance_m"), 0.005),
        "push_start": _pose_bin(physical_parameters.get("push_start")),
        "push_end": _pose_bin(physical_parameters.get("push_end")),
        "place_pose": _pose_bin(physical_parameters.get("place_pose")),
    }
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"action_{hashlib.sha256(canonical.encode()).hexdigest()[:20]}:{canonical}"


def _quantize(value: Any, step: float) -> float | None:
    if value is None:
        return None
    return round(round(float(value) / step) * step, 6)


def _vector_bin(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    vector = [float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 0.0]
    return tuple(round(item, 3) for item in vector)


def _pose_bin(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, Mapping):
        return None
    position = value.get("position_m") or value.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    yaw = value.get("yaw_deg")
    return tuple([_quantize(item, 0.005) for item in position[:3]] + [_quantize(yaw, 5.0)])
