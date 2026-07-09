"""Safety validation for VLM action-intent decisions."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .geometry_relations import get_center, get_size, object_xy_aabb, xy_aabb_overlap
from .grasp_yaw_search import select_best_grasp


ObjectDict = Dict[str, Any]
ActionDict = Dict[str, Any]
MIN_PUSH_DISTANCE_M = 0.01
MAX_PUSH_DISTANCE_M = 0.05
DIRECTION_TOLERANCE = 1e-3


def validate_vlm_action_decision(
    decision: ActionDict,
    current_state: dict,
    target_object: ObjectDict,
    protected_ids: Iterable[Any],
    analysis: Optional[dict] = None,
    protection_margin_m: float = 0.01,
    future_place_regions: Optional[Iterable[dict]] = None,
) -> Tuple[Optional[dict], dict]:
    """Validate VLM intent before MoveIt preflight."""
    safety = {
        "schema_version": "vlm_action_safety_report_v1",
        "accepted": False,
        "moveit_feasible": None,
        "failed_fields": [],
        "checks": {},
        "decision": decision,
    }
    action_type = decision.get("action_type")
    if action_type in ("stop", "reobserve"):
        safety["reason"] = "policy_requested_{}".format(action_type)
        return None, safety
    objects = [obj for obj in current_state.get("objects", []) if isinstance(obj, dict)]
    object_map = {str(obj.get("id")): obj for obj in objects if obj.get("id") is not None}
    obj = object_map.get(str(decision.get("object_id"))) if decision.get("object_id") is not None else None
    target_matches = str(decision.get("target_object_id")) == str(target_object.get("id"))
    _record(safety, "object_id_exists", obj is not None)
    _record(safety, "target_object_id_matches_current_target", target_matches)
    if obj is None or not target_matches:
        safety["reason"] = "object_or_target_validation_failed"
        return None, safety
    movable_ok = _object_can_move(obj, protected_ids)
    _record(safety, "object_not_base_placed_locked_protected", movable_ok)
    if not movable_ok:
        safety["reason"] = "object_is_not_movable"
        return None, safety
    if action_type == "pick":
        return _validate_pick(decision, obj, target_object, analysis or {}, safety)
    if action_type == "pick_away":
        return _validate_pick_away(
            decision,
            current_state,
            obj,
            target_object,
            protected_ids,
            safety,
            protection_margin_m,
            future_place_regions or [],
        )
    if action_type != "nudge":
        _record(safety, "action_type_supported", False)
        safety["reason"] = "unsupported_action_type"
        return None, safety
    return _validate_nudge(decision, current_state, obj, target_object, protected_ids, safety, protection_margin_m)


def mark_moveit_result(action: Optional[dict], safety: dict, feasible: bool, error: Optional[str] = None) -> dict:
    """Attach MoveIt preflight outcome to a previously validated action."""
    output = copy.deepcopy(safety)
    output["moveit_feasible"] = bool(feasible)
    _record(output, "moveit_preflight_passed", bool(feasible))
    if not feasible:
        output["accepted"] = False
        output["reason"] = "moveit_preflight_failed"
        if error:
            output["moveit_preflight_error"] = str(error)
    elif action is not None:
        output["accepted"] = True
        output["reason"] = "accepted"
    return output


def _validate_pick(decision: dict, obj: dict, target: dict, analysis: dict, safety: dict) -> Tuple[Optional[dict], dict]:
    object_is_target = str(obj.get("id")) == str(target.get("id"))
    grasp_feasible = bool(analysis.get("grasp_feasible"))
    _record(safety, "pick_object_is_current_target", object_is_target)
    _record(safety, "target_grasp_feasible", grasp_feasible)
    if not object_is_target or not grasp_feasible:
        safety["reason"] = "pick_intent_not_safe_for_current_target"
        return None, safety
    action = _base_action(decision, "pick", obj, target)
    action["selected_grasp_yaw_deg"] = analysis.get("selected_grasp_yaw_deg")
    action["executable_safe"] = True
    safety["accepted"] = True
    safety["reason"] = "accepted_pick_intent"
    return action, safety


def _validate_pick_away(
    decision: dict,
    current_state: dict,
    obj: dict,
    target: dict,
    protected_ids: Iterable[Any],
    safety: dict,
    protection_margin_m: float,
    future_place_regions: Iterable[dict],
) -> Tuple[Optional[dict], dict]:
    objects = [item for item in current_state.get("objects", []) if isinstance(item, dict)]
    grasp = select_best_grasp(obj, objects)
    grasp_ok = bool(grasp.get("grasp_feasible"))
    place_ok, safe_place = _valid_safe_place(decision.get("safe_place_center_base_m"))
    _record(safety, "pick_away_grasp_feasible", grasp_ok, {"selected_grasp_yaw_deg": grasp.get("selected_grasp_yaw_deg")})
    _record(safety, "safe_place_center_base_m_valid", place_ok)
    if not grasp_ok or not place_ok:
        safety["reason"] = "pick_away_grasp_or_place_invalid"
        return None, safety

    placed = _object_at_center(obj, safe_place)
    place_aabb = object_xy_aabb(placed, margin_m=protection_margin_m)
    table_ok = _table_bounds_contain_aabb(place_aabb, current_state.get("table_bounds"))
    collision_ok, collision_report = _place_avoids_scene(placed, obj, objects, protection_margin_m)
    protected_ok, protected_report = _aabb_clear_of_protected(
        place_aabb,
        _protected_objects(objects, protected_ids),
        protection_margin_m,
    )
    future_ok, future_report = _place_avoids_future_regions(placed, future_place_regions, protection_margin_m)
    _record(safety, "safe_place_inside_table_bounds", table_ok)
    _record(safety, "safe_place_avoids_visible_objects", collision_ok, collision_report)
    _record(safety, "safe_place_outside_protected_structure", protected_ok, protected_report)
    _record(safety, "safe_place_avoids_future_stack_region", future_ok, future_report)
    if not table_ok or not collision_ok or not protected_ok or not future_ok:
        safety["reason"] = "pick_away_safe_place_rejected"
        return None, safety

    action = _base_action(decision, "pick_away", obj, target)
    action.update({
        "action_id": "vlm_action_pick_away_obj_{}".format(obj.get("id")),
        "obstacle_id": obj.get("id"),
        "selected_grasp_yaw_deg": grasp.get("selected_grasp_yaw_deg"),
        "safe_place_center_m": safe_place,
        "geometry_feasible": True,
        "approach_path_safe": True,
        "push_swept_safe": True,
        "push_end_safe": True,
        "protected_structure_safe": True,
        "task_effective": True,
        "automatic_execution_allowed": True,
        "clearance_preflight_allowed": True,
        "collision_free": True,
        "sweep_collision_free": True,
        "workspace_feasible": True,
        "gripper_feasible": True,
        "executable_safe": False,
    })
    safety["accepted"] = True
    safety["reason"] = "accepted_pending_moveit_preflight"
    return action, safety


def _validate_nudge(
    decision: dict,
    current_state: dict,
    obj: dict,
    target: dict,
    protected_ids: Iterable[Any],
    safety: dict,
    protection_margin_m: float,
) -> Tuple[Optional[dict], dict]:
    distance_ok, distance = _valid_push_distance(decision.get("push_distance_m"))
    direction_ok, direction = _valid_push_direction(decision.get("push_direction_base"))
    _record(safety, "push_distance_in_range", distance_ok)
    _record(safety, "push_direction_base_unit_xy_vector", direction_ok)
    if not distance_ok or not direction_ok:
        safety["reason"] = "push_direction_or_distance_invalid"
        return None, safety
    protected_objects = _protected_objects(current_state.get("objects", []), protected_ids)
    end_ok, end_report = _push_end_outside_protected_zone(obj, protected_objects, direction, distance, protection_margin_m)
    sweep_ok, sweep_report = _push_sweep_avoids_protected_structure(obj, protected_objects, direction, distance, protection_margin_m)
    _record(safety, "push_end_outside_protected_structure", end_ok, end_report)
    _record(safety, "push_swept_path_avoids_protected_structure", sweep_ok, sweep_report)
    if not end_ok or not sweep_ok:
        safety["reason"] = "push_collides_with_protected_structure"
        return None, safety
    action = _base_action(decision, "nudge", obj, target)
    action.update({
        "action_id": "vlm_action_nudge_obj_{}".format(obj.get("id")),
        "obstacle_id": obj.get("id"),
        "direction_base": direction,
        "distance_m": distance,
        "geometry_feasible": True,
        "approach_path_safe": True,
        "push_swept_safe": True,
        "push_end_safe": True,
        "protected_structure_safe": True,
        "task_effective": True,
        "automatic_execution_allowed": True,
        "clearance_preflight_allowed": True,
        "collision_free": True,
        "sweep_collision_free": True,
        "workspace_feasible": True,
        "gripper_feasible": True,
        "executable_safe": False,
    })
    safety["accepted"] = True
    safety["reason"] = "accepted_pending_moveit_preflight"
    return action, safety


def _base_action(decision: dict, action_type: str, obj: dict, target: dict) -> dict:
    return {
        "action": action_type if action_type in ("pick", "pick_away") else "push_away",
        "action_type": action_type,
        "object_id": obj.get("id"),
        "target_object_id": target.get("id"),
        "reason": decision.get("reason"),
        "confidence": decision.get("confidence"),
        "selection_source": "vlm_action_policy",
    }


def _object_can_move(obj: ObjectDict, protected_ids: Iterable[Any]) -> bool:
    protected = {str(value) for value in protected_ids or []}
    if str(obj.get("id")) in protected:
        return False
    if obj.get("role") in ("base", "structure", "protected"):
        return False
    if obj.get("state") in ("locked", "placed", "protected"):
        return False
    return obj.get("pushable") is not False


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


def _valid_safe_place(value: Any) -> Tuple[bool, Optional[List[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False, None
    try:
        place = [float(item) for item in value]
    except (TypeError, ValueError):
        return False, None
    if not all(math.isfinite(item) for item in place):
        return False, None
    return True, place


def _protected_objects(objects: Iterable[ObjectDict], protected_ids: Iterable[Any]) -> List[ObjectDict]:
    protected = {str(value) for value in protected_ids or []}
    return [
        obj for obj in objects or []
        if isinstance(obj, dict) and str(obj.get("id")) in protected
    ]


def _translated_object(obj: ObjectDict, direction: List[float], distance_m: float) -> ObjectDict:
    output = copy.deepcopy(obj)
    center = get_center(output)
    if center is not None:
        output["geometry_center_m"] = [
            float(center[0]) + float(direction[0]) * float(distance_m),
            float(center[1]) + float(direction[1]) * float(distance_m),
            float(center[2]),
        ]
    return output


def _object_at_center(obj: ObjectDict, center_base_m: List[float]) -> ObjectDict:
    output = copy.deepcopy(obj)
    output["geometry_center_m"] = [float(value) for value in center_base_m[:3]]
    return output


def _expanded_aabb(aabb: dict, margin_m: float) -> dict:
    expanded = dict(aabb)
    expanded["xmin"] -= float(margin_m)
    expanded["xmax"] += float(margin_m)
    expanded["ymin"] -= float(margin_m)
    expanded["ymax"] += float(margin_m)
    return expanded


def _push_end_outside_protected_zone(
    obj: ObjectDict,
    protected_objects: Iterable[ObjectDict],
    direction: List[float],
    distance_m: float,
    margin_m: float,
) -> Tuple[bool, dict]:
    moved_aabb = object_xy_aabb(_translated_object(obj, direction, distance_m))
    return _aabb_clear_of_protected(moved_aabb, protected_objects, margin_m)


def _push_sweep_avoids_protected_structure(
    obj: ObjectDict,
    protected_objects: Iterable[ObjectDict],
    direction: List[float],
    distance_m: float,
    margin_m: float,
) -> Tuple[bool, dict]:
    start = object_xy_aabb(obj)
    end = object_xy_aabb(_translated_object(obj, direction, distance_m))
    if not start or not end:
        return False, {"reason": "missing_pushed_object_aabb"}
    swept = {
        "xmin": min(start["xmin"], end["xmin"]),
        "xmax": max(start["xmax"], end["xmax"]),
        "ymin": min(start["ymin"], end["ymin"]),
        "ymax": max(start["ymax"], end["ymax"]),
        "zmin": min(start["zmin"], end["zmin"]),
        "zmax": max(start["zmax"], end["zmax"]),
    }
    return _aabb_clear_of_protected(swept, protected_objects, margin_m)


def _aabb_clear_of_protected(aabb: dict, protected_objects: Iterable[ObjectDict], margin_m: float) -> Tuple[bool, dict]:
    if not aabb:
        return False, {"reason": "missing_pushed_object_aabb"}
    collisions = []
    for protected in protected_objects:
        protected_aabb = object_xy_aabb(protected)
        if not protected_aabb:
            continue
        expanded = _expanded_aabb(protected_aabb, margin_m)
        if xy_aabb_overlap(aabb, expanded)[2] > 0.0:
            collisions.append({"id": protected.get("id"), "label": protected.get("label")})
    return not collisions, {"collisions": collisions}


def _table_bounds_contain_aabb(aabb: dict, table_bounds: Optional[dict]) -> bool:
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


def _place_avoids_scene(
    placed: ObjectDict,
    original: ObjectDict,
    objects: Iterable[ObjectDict],
    margin_m: float,
) -> Tuple[bool, dict]:
    placed_aabb = object_xy_aabb(placed, margin_m=margin_m)
    if not placed_aabb:
        return False, {"reason": "missing_placed_object_aabb"}
    collisions = []
    for other in objects or []:
        if not isinstance(other, dict) or str(other.get("id")) == str(original.get("id")):
            continue
        other_aabb = object_xy_aabb(other, margin_m=margin_m)
        if other_aabb and xy_aabb_overlap(placed_aabb, other_aabb)[2] > 0.0:
            collisions.append({"id": other.get("id"), "label": other.get("label")})
    return not collisions, {"collisions": collisions}


def _place_avoids_future_regions(
    placed: ObjectDict,
    future_place_regions: Iterable[dict],
    margin_m: float,
) -> Tuple[bool, dict]:
    center = get_center(placed)
    size = get_size(placed)
    if center is None or size is None:
        return False, {"reason": "missing_placed_object_geometry"}
    half_diagonal = 0.5 * math.hypot(float(size[0]), float(size[1])) + float(margin_m)
    collisions = []
    for region in future_place_regions or []:
        region_center = region.get("center_base_m")
        if not isinstance(region_center, list) or len(region_center) < 2:
            continue
        try:
            radius_m = float(region.get("radius_m"))
            distance_m = math.hypot(
                float(center[0]) - float(region_center[0]),
                float(center[1]) - float(region_center[1]),
            )
        except (TypeError, ValueError):
            continue
        if distance_m <= radius_m + half_diagonal:
            collisions.append({"id": region.get("id"), "distance_m": round(distance_m, 6)})
    return not collisions, {"collisions": collisions}


def _record(report: dict, name: str, ok: bool, detail: Optional[dict] = None) -> None:
    report.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail or {}}
    if not ok and name not in report.setdefault("failed_fields", []):
        report["failed_fields"].append(name)
