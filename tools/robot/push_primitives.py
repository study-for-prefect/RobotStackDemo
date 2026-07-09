"""Pure geometry for a four-stage tabletop push-clearing motion."""

import math
from typing import Any, Dict, List


def _finite_vector(value: Any, length: int, name: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        raise ValueError("{} must contain at least {} values.".format(name, length))
    try:
        output = [float(value[index]) for index in range(length)]
    except (TypeError, ValueError):
        raise ValueError("{} must contain numeric values.".format(name))
    if not all(math.isfinite(item) for item in output):
        raise ValueError("{} must contain finite values.".format(name))
    return output


def _finite_optional_float(value: Any, default: float, name: str) -> float:
    if value is None:
        return float(default)
    try:
        output = float(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be numeric.".format(name))
    if not math.isfinite(output):
        raise ValueError("{} must be finite.".format(name))
    return output


def push_contact_z_offset_m(push_plan: Dict[str, Any], obstacle_height_m: float) -> float:
    """Return the contact offset after applying the closed-gripper table clearance floor."""
    contact_z_offset_m = _finite_optional_float(
        push_plan.get("contact_z_offset_m"),
        0.0,
        "contact_z_offset_m",
    )
    min_contact_z_offset_m = _finite_optional_float(
        push_plan.get("min_contact_z_offset_m"),
        0.0,
        "min_contact_z_offset_m",
    )
    if not 0.0 <= contact_z_offset_m <= 0.10:
        raise ValueError("contact_z_offset_m must be in [0.0, 0.10].")
    if not 0.0 <= min_contact_z_offset_m <= 0.10:
        raise ValueError("min_contact_z_offset_m must be in [0.0, 0.10].")

    applied_offset_m = max(contact_z_offset_m, min_contact_z_offset_m)
    max_contact_offset = max(0.004, 0.75 * float(obstacle_height_m))
    if applied_offset_m > max_contact_offset:
        raise ValueError(
            "contact_z_offset_m {:.4f} is too high for obstacle height {:.4f}; "
            "closed-gripper push would need a contact above the safe push band. "
            "Use pick-away, raise only after verifying hardware clearance, or lower "
            "min_contact_z_offset_m explicitly.".format(applied_offset_m, obstacle_height_m)
        )
    return applied_offset_m


def build_push_targets(push_plan: Dict[str, Any]) -> Dict[str, List[float]]:
    """Validate one push plan and return pre-push, contact, push, and retreat targets."""
    if push_plan.get("schema_version") != "push_execution_plan_v1":
        raise ValueError("Unsupported push plan schema_version.")
    if push_plan.get("frame_id") != "base_link":
        raise ValueError("Push plan frame_id must be base_link.")

    obstacle = push_plan.get("obstacle")
    if not isinstance(obstacle, dict):
        raise ValueError("Push plan must contain an obstacle object.")
    center = _finite_vector(obstacle.get("geometry_center_m"), 3, "obstacle.geometry_center_m")
    size = _finite_vector(obstacle.get("dimensions_m"), 3, "obstacle.dimensions_m")
    if any(value <= 0.0 for value in size):
        raise ValueError("obstacle.dimensions_m must be positive.")

    direction = _finite_vector(push_plan.get("direction_base"), 2, "direction_base")
    direction_norm = math.hypot(direction[0], direction[1])
    if direction_norm < 1e-9:
        raise ValueError("direction_base XY norm must be non-zero.")
    direction_xy = [direction[0] / direction_norm, direction[1] / direction_norm]

    distance_m = float(push_plan.get("distance_m"))
    lift_m = float(push_plan.get("lift_m"))
    if not math.isfinite(distance_m) or not 0.0 < distance_m <= 0.20:
        raise ValueError("distance_m must be in (0.0, 0.20].")
    if not math.isfinite(lift_m) or not 0.02 <= lift_m <= 0.30:
        raise ValueError("lift_m must be in [0.02, 0.30].")
    contact_z_offset_m = push_contact_z_offset_m(push_plan, size[2])

    contact_margin = max(size[0], size[1]) * 0.5 + 0.015
    push_start_xy = [
        center[0] - direction_xy[0] * contact_margin,
        center[1] - direction_xy[1] * contact_margin,
    ]
    push_end_xy = [
        push_start_xy[0] + direction_xy[0] * distance_m,
        push_start_xy[1] + direction_xy[1] * distance_m,
    ]
    table_z = center[2] - size[2] / 2.0
    contact_z = table_z + contact_z_offset_m
    approach_z = contact_z + lift_m
    return {
        "pre_push": [push_start_xy[0], push_start_xy[1], approach_z],
        "contact": [push_start_xy[0], push_start_xy[1], contact_z],
        "push_end": [push_end_xy[0], push_end_xy[1], contact_z],
        "retreat": [push_end_xy[0], push_end_xy[1], approach_z],
    }
