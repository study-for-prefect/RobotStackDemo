"""Pure parameter, workspace, and movable-contact checks for VLM nudges."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .geometry_relations import get_center, object_xy_aabb, xy_aabb_overlap


MIN_PUSH_DISTANCE_M = 0.01
MAX_PUSH_DISTANCE_M = 0.05
DIRECTION_TOLERANCE = 1e-3


def validate_nudge_parameters(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Validate VLM-controlled distance, direction, contact side, and yaw."""
    distance_ok, distance = _valid_push_distance(decision.get("distance_m", decision.get("push_distance_m")))
    direction_ok, direction = _valid_push_direction(decision.get("direction_base", decision.get("push_direction_base")))
    contact_ok, contact_side = _valid_contact_side(decision.get("contact_side"), direction)
    yaw_ok, gripper_yaw_rad = _valid_yaw(decision.get("gripper_yaw_rad"))
    return {
        "passed": all((distance_ok, direction_ok, contact_ok, yaw_ok)),
        "distance_ok": distance_ok,
        "distance_m": distance,
        "direction_ok": direction_ok,
        "direction_base": direction,
        "contact_ok": contact_ok,
        "contact_side": contact_side,
        "yaw_ok": yaw_ok,
        "gripper_yaw_rad": gripper_yaw_rad,
    }


def push_stays_in_workspace(
    obj: Dict[str, Any], current_state: dict, direction: List[float], distance_m: float,
) -> Tuple[bool, dict]:
    """Check the pushed object's end footprint against configured table bounds."""
    end_aabb = object_xy_aabb(_translated_object(obj, direction, distance_m))
    bounds = current_state.get("table_bounds") or current_state.get("workspace_bounds")
    return _table_bounds_contain_aabb(end_aabb, bounds), {
        "end_aabb": end_aabb,
        "workspace_bounds": bounds,
    }


def recoverable_push_contacts(
    obj: Dict[str, Any],
    objects: Iterable[Dict[str, Any]],
    protected_ids: Iterable[Any],
    direction: List[float],
    distance_m: float,
) -> List[dict]:
    """Report pushed-object contacts with ordinary movable objects without rejecting them."""
    start = object_xy_aabb(obj)
    end = object_xy_aabb(_translated_object(obj, direction, distance_m))
    if not start or not end:
        return []
    swept = {
        "xmin": min(start["xmin"], end["xmin"]), "xmax": max(start["xmax"], end["xmax"]),
        "ymin": min(start["ymin"], end["ymin"]), "ymax": max(start["ymax"], end["ymax"]),
        "zmin": min(start["zmin"], end["zmin"]), "zmax": max(start["zmax"], end["zmax"]),
    }
    protected = {str(value) for value in protected_ids or []}
    contacts = []
    for other in objects or []:
        if not isinstance(other, dict):
            continue
        other_id = str(other.get("id"))
        if other_id == str(obj.get("id")) or other_id in protected:
            continue
        other_aabb = object_xy_aabb(other)
        if other_aabb and xy_aabb_overlap(swept, other_aabb)[2] > 0.0:
            contacts.append({
                "type": "pushed_object_contact_with_movable_object",
                "operated_object_id": obj.get("id"),
                "contacted_object_id": other.get("id"),
                "contacted_object_label": other.get("label"),
                "classification": "recoverable_contact",
            })
    return contacts


def _valid_push_distance(value: Any) -> Tuple[bool, Optional[float]]:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return False, None
    if not math.isfinite(distance):
        return False, None
    return MIN_PUSH_DISTANCE_M <= distance <= MAX_PUSH_DISTANCE_M, distance


def _valid_push_direction(value: Any) -> Tuple[bool, Optional[List[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False, None
    try:
        direction = [float(item) for item in value]
    except (TypeError, ValueError):
        return False, None
    if not all(math.isfinite(item) for item in direction):
        return False, None
    norm = math.sqrt(sum(item * item for item in direction))
    xy_norm = math.hypot(direction[0], direction[1])
    ok = abs(norm - 1.0) <= DIRECTION_TOLERANCE and abs(direction[2]) <= DIRECTION_TOLERANCE and xy_norm > 1e-6
    return ok, [direction[0], direction[1], 0.0] if ok else None


def _valid_contact_side(value: Any, direction: Optional[List[float]]) -> Tuple[bool, Optional[str]]:
    side = str(value or "").strip().lower()
    if side not in ("+x", "-x", "+y", "-y") or direction is None:
        return False, None
    if abs(direction[0]) >= abs(direction[1]):
        expected = "-x" if direction[0] >= 0.0 else "+x"
    else:
        expected = "-y" if direction[1] >= 0.0 else "+y"
    return side == expected, side


def _valid_yaw(value: Any) -> Tuple[bool, Optional[float]]:
    try:
        yaw = float(value)
    except (TypeError, ValueError):
        return False, None
    return math.isfinite(yaw), yaw if math.isfinite(yaw) else None


def _translated_object(obj: Dict[str, Any], direction: List[float], distance_m: float) -> Dict[str, Any]:
    output = copy.deepcopy(obj)
    center = get_center(output)
    if center is not None:
        output["geometry_center_m"] = [
            float(center[0]) + float(direction[0]) * float(distance_m),
            float(center[1]) + float(direction[1]) * float(distance_m),
            float(center[2]),
        ]
    return output


def _table_bounds_contain_aabb(aabb: Optional[dict], table_bounds: Optional[dict]) -> bool:
    if not aabb or not table_bounds:
        return True
    try:
        return (
            aabb["xmin"] >= float(table_bounds["xmin"])
            and aabb["xmax"] <= float(table_bounds["xmax"])
            and aabb["ymin"] >= float(table_bounds["ymin"])
            and aabb["ymax"] <= float(table_bounds["ymax"])
        )
    except (KeyError, TypeError, ValueError):
        return False
