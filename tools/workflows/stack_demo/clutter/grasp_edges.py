"""Full-range grasp-yaw scanning and code-owned physical grasp checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from robot_scene_pipeline.grasp_yaw_search import evaluate_grasp_yaw

from ..common.config import StackDemoConfig
from ..common.scene_state import SceneObjectState


GeometryCheck = Callable[[SceneObjectState, Sequence[SceneObjectState], float], Mapping[str, Any]]


@dataclass(frozen=True)
class GraspInterval:
    start_deg: float
    end_deg: float
    selected_yaw_deg: float
    span_deg: float
    score: float
    selected_check: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_deg": self.start_deg,
            "end_deg": self.end_deg,
            "selected_yaw_deg": self.selected_yaw_deg,
            "span_deg": self.span_deg,
            "score": self.score,
            "selected_check": dict(self.selected_check),
        }


@dataclass(frozen=True)
class GraspScanResult:
    target_track_id: str
    samples: tuple[Mapping[str, Any], ...]
    safe_intervals: tuple[GraspInterval, ...]
    rejected_narrow_intervals_deg: tuple[tuple[float, float], ...]
    blocker_track_ids: tuple[str, ...]

    @property
    def graspable(self) -> bool:
        return bool(self.safe_intervals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_track_id": self.target_track_id,
            "graspable": self.graspable,
            "samples": [dict(item) for item in self.samples],
            "safe_intervals": [item.to_dict() for item in self.safe_intervals],
            "rejected_narrow_intervals_deg": [list(item) for item in self.rejected_narrow_intervals_deg],
            "blocker_track_ids": list(self.blocker_track_ids),
        }


def normalize_gripper_yaw_deg(yaw_deg: float) -> float:
    """Normalize a parallel-gripper axis to its unique half-turn interval."""
    return round((float(yaw_deg) + 90.0) % 180.0 - 90.0, 6)


def scan_grasp_yaws(
    target: SceneObjectState,
    objects: Sequence[SceneObjectState],
    config: StackDemoConfig,
    *,
    geometry_check: GeometryCheck | None = None,
    current_wrist_yaw_deg: float | None = None,
) -> GraspScanResult:
    """Scan every configured yaw without preferring a detected object axis."""
    grasp = config.section("grasp")
    step = float(grasp["yaw_scan_step_deg"])
    if step <= 0.0 or step > 10.0:
        raise ValueError("grasp yaw scan step must be in (0, 10] degrees")
    checker = geometry_check or (lambda item, scene, yaw: _default_geometry_check(item, scene, yaw, config))
    sample_count = int(math.ceil(180.0 / step))
    samples = []
    for index in range(sample_count):
        yaw = normalize_gripper_yaw_deg(-90.0 + index * step)
        result = dict(checker(target, objects, yaw))
        result["yaw_deg"] = round(yaw, 6)
        result["safe"] = all(
            bool(result.get(key))
            for key in (
                "opening_ok", "center_offset_ok", "contact_length_ok", "finger_safe",
                "palm_safe", "descent_safe", "lift_safe",
            )
        )
        result["score"] = _sample_score(result, yaw, current_wrist_yaw_deg)
        samples.append(result)
    intervals = _sample_intervals(samples, step)
    minimum_span = float(grasp["minimum_continuous_safe_yaw_span_deg"])
    robust: list[GraspInterval] = []
    narrow: list[tuple[float, float]] = []
    for start_index, end_index, start_deg, end_deg in intervals:
        span = end_deg - start_deg
        if span + 1e-9 < minimum_span:
            narrow.append((round(start_deg, 3), round(end_deg, 3)))
            continue
        selected = _select_inside_interval(samples, start_index, end_index, start_deg, end_deg, config)
        robust.append(GraspInterval(
            start_deg=round(start_deg, 3),
            end_deg=round(end_deg, 3),
            selected_yaw_deg=float(selected["yaw_deg"]),
            span_deg=round(span, 3),
            score=round(float(selected["score"]) + span / 180.0, 6),
            selected_check=selected,
        ))
    robust.sort(key=lambda item: (item.score, item.span_deg), reverse=True)
    blockers = sorted({
        str(track_id)
        for sample in samples if not sample["safe"]
        for track_id in sample.get("blocking_track_ids", [])
        if track_id
    })
    return GraspScanResult(
        target_track_id=target.track_id,
        samples=tuple(samples),
        safe_intervals=tuple(robust),
        rejected_narrow_intervals_deg=tuple(narrow),
        blocker_track_ids=tuple(blockers),
    )


def evaluate_gripper_pose_clearance(
    target: SceneObjectState,
    objects: Sequence[SceneObjectState],
    yaw_deg: float,
    config: StackDemoConfig,
) -> Mapping[str, Any]:
    """Evaluate GF225 finger and palm clearance at one code-owned pose."""
    return _default_geometry_check(target, objects, yaw_deg, config)


def _default_geometry_check(
    target: SceneObjectState,
    objects: Sequence[SceneObjectState],
    yaw_deg: float,
    config: StackDemoConfig,
) -> Mapping[str, Any]:
    gripper = config.section("gripper")
    safety = config.section("safety")
    target_raw = _geometry_object(target)
    raw_objects = [_geometry_object(obj) for obj in objects]
    fingers = evaluate_grasp_yaw(
        target_raw,
        raw_objects,
        yaw_deg,
        gripper_outer_width_m=float(gripper["open_outer_width_m"]),
        gripper_inner_width_m=float(gripper["open_inner_width_m"]),
        approach_length_m=float(config.section("grasp")["approach_envelope_length_m"]),
        side_clearance_m=float(safety["object_clearance_m"]),
    )
    closing_extent, contact_length = _target_contact_geometry(target, yaw_deg)
    center_offset = float(target.source.get("grasp_center_offset_m", 0.0))
    palm_blockers = _palm_blockers(target, objects, yaw_deg, config)
    finger_blockers = tuple(
        _blocker_track_id(item, objects)
        for item in fingers.get("blocking_objects", [])
    )
    blockers = tuple(sorted({item for item in finger_blockers + palm_blockers if item}))
    finger_safe = bool(fingers.get("feasible"))
    palm_safe = not palm_blockers
    return {
        "opening_ok": closing_extent + 2.0 * center_offset <= float(gripper["open_inner_width_m"]),
        "center_offset_ok": center_offset <= float(safety["grasp_center_tolerance_m"]),
        "contact_length_ok": contact_length >= float(safety["minimum_contact_length_m"]),
        "finger_safe": finger_safe,
        "palm_safe": palm_safe,
        "descent_safe": finger_safe and palm_safe,
        "lift_safe": finger_safe and palm_safe,
        "closing_extent_m": round(closing_extent, 6),
        "grasp_center_offset_m": round(center_offset, 6),
        "effective_contact_length_m": round(contact_length, 6),
        "fingertip_clearance_m": float(fingers.get("clearance_m", 0.0)),
        "palm_clearance_m": 0.0 if palm_blockers else float(safety["object_clearance_m"]),
        "vertical_lift_clearance_m": 0.0 if blockers else float(safety["observation_height_m"]),
        "blocking_track_ids": list(blockers),
    }


def _target_contact_geometry(target: SceneObjectState, gripper_yaw_deg: float) -> tuple[float, float]:
    delta = math.radians(target.yaw_deg - gripper_yaw_deg)
    size_x, size_y = target.size_xyz_m[:2]
    along_fingers = abs(size_x * math.cos(delta)) + abs(size_y * math.sin(delta))
    along_closing = abs(size_x * math.sin(delta)) + abs(size_y * math.cos(delta))
    explicit = target.source.get("effective_contact_length_m")
    if explicit is not None:
        along_fingers = float(explicit)
    return along_closing, along_fingers


def _palm_blockers(
    target: SceneObjectState,
    objects: Sequence[SceneObjectState],
    yaw_deg: float,
    config: StackDemoConfig,
) -> tuple[str, ...]:
    gripper = config.section("gripper")
    palm_width = float(gripper["palm_width_m"])
    palm_depth = float(gripper["palm_depth_m"])
    palm_min_z = target.center_xyz_m[2] + 0.5 * target.size_xyz_m[2] + float(gripper["upper_finger_height_m"])
    angle = math.radians(yaw_deg)
    blockers = []
    for other in objects:
        if other.track_id == target.track_id:
            continue
        other_top = other.center_xyz_m[2] + 0.5 * other.size_xyz_m[2]
        if other_top < palm_min_z:
            continue
        dx = other.center_xyz_m[0] - target.center_xyz_m[0]
        dy = other.center_xyz_m[1] - target.center_xyz_m[1]
        local_u = dx * math.cos(angle) + dy * math.sin(angle)
        local_v = -dx * math.sin(angle) + dy * math.cos(angle)
        radius = 0.5 * max(other.size_xyz_m[:2])
        if abs(local_u) <= palm_depth / 2.0 + radius and abs(local_v) <= palm_width / 2.0 + radius:
            blockers.append(other.track_id)
    return tuple(sorted(blockers))


def _sample_intervals(samples: Sequence[Mapping[str, Any]], step: float) -> list[tuple[int, int, float, float]]:
    intervals = []
    start: int | None = None
    for index, sample in enumerate(samples):
        if sample["safe"] and start is None:
            start = index
        next_safe = index + 1 < len(samples) and bool(samples[index + 1]["safe"])
        if start is not None and sample["safe"] and not next_safe:
            intervals.append((
                start,
                index,
                float(samples[start]["yaw_deg"]),
                min(90.0, float(sample["yaw_deg"]) + step),
            ))
            start = None
    return intervals


def _select_inside_interval(
    samples: Sequence[Mapping[str, Any]],
    start_index: int,
    end_index: int,
    start_deg: float,
    end_deg: float,
    config: StackDemoConfig,
) -> Mapping[str, Any]:
    inset = min(float(config.section("grasp")["interval_boundary_inset_deg"]), 0.25 * (end_deg - start_deg))
    interior = [
        item for item in samples[start_index:end_index + 1]
        if start_deg + inset <= float(item["yaw_deg"]) <= end_deg - inset
    ]
    if not interior:
        interior = list(samples[start_index:end_index + 1])
    return max(interior, key=lambda item: (float(item["score"]), -abs(float(item["yaw_deg"]) - (start_deg + end_deg) / 2.0)))


def _sample_score(result: Mapping[str, Any], yaw_deg: float, current_wrist_yaw_deg: float | None) -> float:
    wrist_cost = 0.0
    if current_wrist_yaw_deg is not None:
        wrist_cost = abs((yaw_deg - current_wrist_yaw_deg + 90.0) % 180.0 - 90.0) / 180.0
    return (
        2.0 * float(result.get("fingertip_clearance_m", 0.0))
        + float(result.get("palm_clearance_m", 0.0))
        + float(result.get("vertical_lift_clearance_m", 0.0))
        + float(result.get("effective_contact_length_m", 0.0))
        - float(result.get("grasp_center_offset_m", 0.0))
        - 0.05 * wrist_cost
    )


def _geometry_object(obj: SceneObjectState) -> dict[str, Any]:
    return {
        **dict(obj.source),
        "id": obj.detector_id,
        "track_id": obj.track_id,
        "label": obj.class_name,
        "geometry_center_m": list(obj.center_xyz_m),
        "dimensions_m": list(obj.size_xyz_m),
        "yaw_deg": obj.yaw_deg,
        "protected": obj.protected,
        "already_completed": obj.already_completed,
    }


def _blocker_track_id(blocker: Mapping[str, Any], objects: Sequence[SceneObjectState]) -> str | None:
    detector_id = str(blocker.get("id", ""))
    return next((obj.track_id for obj in objects if obj.detector_id == detector_id), None)
