"""Pure geometry for a staged tabletop push-clearing motion."""

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


def build_push_targets(push_plan: Dict[str, Any]) -> Dict[str, List[float]]:
    """Return a safe vertical entry, horizontal contact, push, and retreat path.

    New plans provide a center-to-TCP contact standoff that includes the
    projected fingertip envelope.  ``prepush_clearance_m`` is an additional
    horizontal entry clearance: the tool descends at ``pre_contact`` and only
    then approaches the object horizontally.  Missing fields keep legacy plan
    files readable without silently changing their old target positions.
    """
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
    retreat_lift_m = float(push_plan.get("retreat_lift_m", lift_m))
    contact_z_offset_m = float(push_plan.get("contact_z_offset_m"))
    if not math.isfinite(distance_m) or not 0.0 < distance_m <= 0.20:
        raise ValueError("distance_m must be in (0.0, 0.20].")
    if not math.isfinite(lift_m) or not 0.02 <= lift_m <= 0.30:
        raise ValueError("lift_m must be in [0.02, 0.30].")
    if not math.isfinite(retreat_lift_m) or not 0.02 <= retreat_lift_m <= 0.30:
        raise ValueError("retreat_lift_m must be in [0.02, 0.30].")
    if not math.isfinite(contact_z_offset_m) or not 0.0 <= contact_z_offset_m <= 0.10:
        raise ValueError("contact_z_offset_m must be in [0.0, 0.10].")
    min_contact_offset = max(0.004, min(0.012, 0.20 * size[2]))
    if contact_z_offset_m < min_contact_offset:
        raise ValueError(
            "contact_z_offset_m {:.4f} is too low for obstacle height {:.4f}; "
            "raise the push contact to avoid table contact.".format(contact_z_offset_m, size[2])
        )
    max_contact_offset = max(0.004, 0.75 * size[2])
    if contact_z_offset_m > max_contact_offset:
        raise ValueError(
            "contact_z_offset_m {:.4f} is too high for obstacle height {:.4f}; "
            "use pick-away or a lower push contact.".format(contact_z_offset_m, size[2])
        )

    legacy_contact_standoff = max(size[0], size[1]) * 0.5 + 0.015
    contact_standoff_m = float(push_plan.get("contact_standoff_m", legacy_contact_standoff))
    contact_clearance_m = float(push_plan.get("contact_clearance_m", 0.0))
    prepush_clearance_m = float(push_plan.get("prepush_clearance_m", 0.0))
    if not math.isfinite(contact_standoff_m) or not 0.0 < contact_standoff_m <= 0.15:
        raise ValueError("contact_standoff_m must be in (0.0, 0.15].")
    if not math.isfinite(contact_clearance_m) or not 0.0 <= contact_clearance_m <= 0.03:
        raise ValueError("contact_clearance_m must be in [0.0, 0.03].")
    if not math.isfinite(prepush_clearance_m) or not 0.0 <= prepush_clearance_m <= 0.10:
        raise ValueError("prepush_clearance_m must be in [0.0, 0.10].")

    contact_xy = [
        center[0] - direction_xy[0] * contact_standoff_m,
        center[1] - direction_xy[1] * contact_standoff_m,
    ]
    pre_contact_xy = [
        contact_xy[0] - direction_xy[0] * prepush_clearance_m,
        contact_xy[1] - direction_xy[1] * prepush_clearance_m,
    ]
    push_end_xy = [
        contact_xy[0] + direction_xy[0] * (distance_m + contact_clearance_m),
        contact_xy[1] + direction_xy[1] * (distance_m + contact_clearance_m),
    ]
    table_z = center[2] - size[2] / 2.0
    contact_z = table_z + contact_z_offset_m
    approach_z = contact_z + lift_m
    return {
        "pre_push": [pre_contact_xy[0], pre_contact_xy[1], approach_z],
        "pre_contact": [pre_contact_xy[0], pre_contact_xy[1], contact_z],
        "contact": [contact_xy[0], contact_xy[1], contact_z],
        "push_end": [push_end_xy[0], push_end_xy[1], contact_z],
        "retreat": [
            push_end_xy[0], push_end_xy[1], contact_z + retreat_lift_m,
        ],
    }
