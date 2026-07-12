"""Quaternion trajectory planning for held-object roll/pitch reorientation."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence


CollisionCheck = Callable[[dict], bool]


def plan_reorientation(
    current_object_orientation: Sequence[float],
    target_object_orientation: Sequence[float],
    grasp_transform_object_to_tool: Sequence[float],
    source_position_base_m: Sequence[float],
    target_position_base_m: Sequence[float],
    semantics_config: dict,
    collision_check: Optional[CollisionCheck] = None,
    joint_states: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    """Plan a safe-height SLERP path and validate every intermediate waypoint."""
    config = semantics_config.get("house_semantics", semantics_config)
    source_object = normalize_quaternion(current_object_orientation)
    target_object = normalize_quaternion(target_object_orientation)
    grasp = normalize_quaternion(grasp_transform_object_to_tool)
    source_tool = quaternion_multiply(source_object, grasp)
    target_tool = quaternion_multiply(target_object, grasp)
    relative = quaternion_multiply(quaternion_inverse(source_tool), target_tool)
    axis, total_angle_deg = quaternion_axis_angle(relative)
    maximum_total = float(config.get("maximum_total_reorientation_deg", 180.0))
    if total_angle_deg > maximum_total + 1e-6:
        raise ValueError("reorientation_angle_exceeds_configured_limit")
    maximum_step = max(1.0, float(config.get("maximum_single_reorientation_step_deg", 20.0)))
    segment_count = max(1, int(math.ceil(total_angle_deg / maximum_step)))
    safe_height = max(
        float(source_position_base_m[2]),
        float(target_position_base_m[2]),
    ) + float(config.get("safe_reorientation_height_m", 0.12))
    waypoints = []
    for index in range(segment_count + 1):
        ratio = index / float(segment_count)
        position = [
            (1.0 - ratio) * float(source_position_base_m[0]) + ratio * float(target_position_base_m[0]),
            (1.0 - ratio) * float(source_position_base_m[1]) + ratio * float(target_position_base_m[1]),
            safe_height,
        ]
        waypoint = {
            "index": index,
            "position_base_m": position,
            "orientation_xyzw": quaternion_slerp(source_tool, target_tool, ratio),
            "at_safe_reorientation_height": True,
        }
        waypoint["collision_free"] = True if collision_check is None else bool(collision_check(waypoint))
        waypoints.append(waypoint)
    joint_continuity = _joint_continuity(joint_states or [])
    collision_free = all(item["collision_free"] for item in waypoints)
    roll_pitch_component = math.hypot(axis[0], axis[1]) > 1e-3 and total_angle_deg > 1e-3
    return {
        "schema_version": "reorientation_plan_v1",
        "required": total_angle_deg > 1e-3,
        "rotation_type": "flip" if roll_pitch_component else "yaw_only",
        "rotation_axis_tool": axis,
        "rotation_angle_deg": total_angle_deg,
        "intermediate_pose_count": len(waypoints),
        "perform_above_safe_height": True,
        "safe_reorientation_height_m": safe_height,
        "source_orientation_xyzw": source_object,
        "target_orientation_xyzw": target_object,
        "source_tool_orientation_xyzw": source_tool,
        "target_tool_orientation_xyzw": target_tool,
        "intermediate_orientations": [item["orientation_xyzw"] for item in waypoints],
        "waypoints": waypoints,
        "all_waypoints_collision_checked": collision_check is not None,
        "collision_free": collision_free,
        "joint_continuity": joint_continuity,
        "roll_pitch_component": roll_pitch_component,
    }


def normalize_quaternion(quaternion: Sequence[float]) -> List[float]:
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("quaternion_must_have_four_values")
    values = [float(value) for value in quaternion]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid_quaternion")
    return [value / norm for value in values]


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> List[float]:
    lx, ly, lz, lw = normalize_quaternion(left)
    rx, ry, rz, rw = normalize_quaternion(right)
    return normalize_quaternion([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def quaternion_inverse(quaternion: Sequence[float]) -> List[float]:
    x, y, z, w = normalize_quaternion(quaternion)
    return [-x, -y, -z, w]


def quaternion_axis_angle(quaternion: Sequence[float]) -> tuple:
    x, y, z, w = normalize_quaternion(quaternion)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
    sine = math.sqrt(max(0.0, 1.0 - w * w))
    axis = [1.0, 0.0, 0.0] if sine < 1e-9 else [x / sine, y / sine, z / sine]
    return axis, math.degrees(angle)


def quaternion_slerp(start: Sequence[float], end: Sequence[float], ratio: float) -> List[float]:
    first, second = normalize_quaternion(start), normalize_quaternion(end)
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second = [-value for value in second]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion([(1.0 - ratio) * a + ratio * b for a, b in zip(first, second)])
    theta = math.acos(dot)
    sine = math.sin(theta)
    first_weight = math.sin((1.0 - ratio) * theta) / sine
    second_weight = math.sin(ratio * theta) / sine
    return normalize_quaternion([first_weight * a + second_weight * b for a, b in zip(first, second)])


def _joint_continuity(joint_states: Sequence[Sequence[float]]) -> Dict[str, Any]:
    if len(joint_states) < 2:
        return {"checked": False, "passed": None, "maximum_delta_rad": None}
    maximum = max(
        abs(float(current[index]) - float(previous[index]))
        for previous, current in zip(joint_states, joint_states[1:])
        for index in (2, 4, 5)
        if len(previous) > index and len(current) > index
    )
    return {"checked": True, "passed": maximum <= 1.2, "maximum_delta_rad": maximum}
