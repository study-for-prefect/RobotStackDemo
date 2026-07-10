"""Safety validation for VLM action-intent decisions."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .geometry_relations import get_center, get_size, object_xy_aabb, xy_aabb_overlap
from .grasp_yaw_search import select_best_grasp
from .nudge_safety import push_stays_in_workspace, recoverable_push_contacts, validate_nudge_parameters


ObjectDict = Dict[str, Any]
ActionDict = Dict[str, Any]
GROUNDING_CENTER_TOLERANCE_M = 0.005


def validate_vlm_action_decision(
    decision: ActionDict,
    current_state: dict,
    protected_ids: Iterable[Any],
    grasp_options: Optional[dict] = None,
    protection_margin_m: float = 0.01,
    future_place_regions: Optional[Iterable[dict]] = None,
) -> Tuple[Optional[dict], dict]:
    """Validate only physical safety and executability of a VLM-selected intent."""
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
    duplicate_ids = _duplicate_object_ids(objects)
    _record(safety, "scene_object_ids_unique", not duplicate_ids, {"duplicate_object_ids": duplicate_ids})
    if duplicate_ids:
        safety["reason"] = "scene_object_ids_not_unique"
        return None, safety
    object_map = {str(obj.get("id")): obj for obj in objects if obj.get("id") is not None}
    obj = object_map.get(str(decision.get("object_id"))) if decision.get("object_id") is not None else None
    target = object_map.get(str(decision.get("target_object_id"))) if decision.get("target_object_id") is not None else None
    _record(safety, "object_id_exists", obj is not None, {"object_id": decision.get("object_id")})
    _record(
        safety,
        "target_object_id_exists",
        target is not None,
        {
            "target_object_id": decision.get("target_object_id"),
            "note": "The VLM chooses the task object advanced by this action; code only verifies that it exists.",
        },
    )
    if obj is None or target is None:
        safety["reason"] = "object_or_target_validation_failed"
        return None, safety
    object_label_ok = _label_matches(decision.get("object_label"), obj.get("label"))
    target_label_ok = _label_matches(decision.get("target_object_label"), target.get("label"))
    object_center_ok, object_center_detail = _center_matches(decision.get("object_center_base_m"), obj)
    target_center_ok, target_center_detail = _center_matches(decision.get("target_object_center_base_m"), target)
    _record(safety, "object_label_matches_selected_id", object_label_ok, {
        "reported": decision.get("object_label"), "observed": obj.get("label"),
    })
    _record(safety, "target_label_matches_selected_id", target_label_ok, {
        "reported": decision.get("target_object_label"), "observed": target.get("label"),
    })
    _record(safety, "object_center_matches_selected_id", object_center_ok, object_center_detail)
    _record(safety, "target_center_matches_selected_id", target_center_ok, target_center_detail)
    if not all((object_label_ok, target_label_ok, object_center_ok, target_center_ok)):
        safety["reason"] = "decision_object_grounding_mismatch"
        return None, safety
    movable_ok = _object_can_move(obj, protected_ids, require_pushable=action_type == "nudge")
    _record(safety, "object_not_base_placed_locked_protected", movable_ok)
    if not movable_ok:
        safety["reason"] = "object_is_not_movable"
        return None, safety
    if action_type == "pick":
        return _validate_pick(decision, current_state, obj, target, grasp_options or {}, safety)
    if action_type == "pick_away":
        return _validate_pick_away(
            decision,
            current_state,
            obj,
            target,
            protected_ids,
            safety,
            protection_margin_m,
            future_place_regions or [],
            grasp_options or {},
        )
    if action_type != "nudge":
        _record(safety, "action_type_supported", False)
        safety["reason"] = "unsupported_action_type"
        return None, safety
    return _validate_nudge(decision, current_state, obj, target, protected_ids, safety, protection_margin_m)


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


def _validate_pick(
    decision: dict,
    current_state: dict,
    obj: dict,
    target: dict,
    grasp_options: dict,
    safety: dict,
) -> Tuple[Optional[dict], dict]:
    grasp = _select_grasp(obj, current_state.get("objects", []), grasp_options)
    grasp_feasible = bool(grasp.get("grasp_feasible"))
    _record(safety, "selected_object_grasp_feasible", grasp_feasible, _compact_grasp_report(grasp))
    if not grasp_feasible:
        safety["reason"] = "selected_pick_not_grasp_feasible"
        return None, safety
    action = _base_action(decision, "pick", obj, target)
    action["selected_grasp_yaw_deg"] = grasp.get("selected_grasp_yaw_deg")
    action["grasp_validation"] = _compact_grasp_report(grasp)
    action["executable_safe"] = False
    safety["accepted"] = True
    safety["reason"] = "accepted_pending_moveit_preflight"
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
    grasp_options: dict,
) -> Tuple[Optional[dict], dict]:
    objects = [item for item in current_state.get("objects", []) if isinstance(item, dict)]
    grasp = _select_grasp(obj, objects, grasp_options)
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
        "grasp_validation": _compact_grasp_report(grasp),
        "safe_place_center_m": safe_place,
        "collision_free": True,
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
    parameters = validate_nudge_parameters(decision)
    direction, distance = parameters["direction_base"], parameters["distance_m"]
    _record(safety, "push_distance_in_range", parameters["distance_ok"])
    _record(safety, "push_direction_base_unit_xy_vector", parameters["direction_ok"])
    _record(safety, "contact_side_matches_push_direction", parameters["contact_ok"], {"contact_side": decision.get("contact_side")})
    _record(safety, "gripper_yaw_rad_finite", parameters["yaw_ok"], {"gripper_yaw_rad": decision.get("gripper_yaw_rad")})
    if not parameters["passed"]:
        safety["reason"] = "push_parameters_invalid"
        return None, safety
    protected_objects = _protected_objects(current_state.get("objects", []), protected_ids)
    end_ok, end_report = _push_end_outside_protected_zone(obj, protected_objects, direction, distance, protection_margin_m)
    sweep_ok, sweep_report = _push_sweep_avoids_protected_structure(obj, protected_objects, direction, distance, protection_margin_m)
    workspace_ok, workspace_report = push_stays_in_workspace(obj, current_state, direction, distance)
    recoverable_contacts = recoverable_push_contacts(
        obj, current_state.get("objects", []), protected_ids, direction, distance,
    )
    _record(safety, "push_end_outside_protected_structure", end_ok, end_report)
    _record(safety, "push_swept_path_avoids_protected_structure", sweep_ok, sweep_report)
    _record(safety, "push_stays_inside_workspace", workspace_ok, workspace_report)
    safety["recoverable_contacts"] = recoverable_contacts
    if not workspace_ok:
        safety["reason"] = "push_outside_workspace"
        return None, safety
    if not end_ok or not sweep_ok:
        safety["reason"] = "push_collides_with_protected_structure"
        return None, safety
    action = _base_action(decision, "nudge", obj, target)
    action.update({
        "action_id": "vlm_action_nudge_obj_{}".format(obj.get("id")),
        "obstacle_id": obj.get("id"),
        "direction_base": direction,
        "distance_m": distance,
        "contact_side": parameters["contact_side"],
        "gripper_yaw_rad": parameters["gripper_yaw_rad"],
        "recoverable_contacts": recoverable_contacts,
        "protected_object_ids": [value for value in protected_ids or []],
        "protected_structure_safe": True,
        "sweep_collision_free": True,
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
        "object_label": obj.get("label"),
        "object_center_base_m": get_center(obj),
        "target_object_id": target.get("id"),
        "target_object_label": target.get("label"),
        "target_object_center_base_m": get_center(target),
        "scene_problem": decision.get("scene_problem"),
        "predicted_scene_benefit": decision.get("predicted_scene_benefit"),
        "risk_assessment": decision.get("risk_assessment"),
        "reason": decision.get("reason"),
        "confidence": decision.get("confidence"),
        "selection_source": "vlm_action_policy",
    }


def _object_can_move(obj: ObjectDict, protected_ids: Iterable[Any], require_pushable: bool) -> bool:
    protected = {str(value) for value in protected_ids or []}
    if str(obj.get("id")) in protected:
        return False
    if obj.get("role") in ("base", "structure", "protected"):
        return False
    if obj.get("state") in ("locked", "placed", "protected"):
        return False
    return not require_pushable or obj.get("pushable") is not False


def _label_matches(reported: Any, observed: Any) -> bool:
    return str(reported or "").strip().lower() == str(observed or "").strip().lower()


def _center_matches(reported: Any, obj: ObjectDict) -> Tuple[bool, dict]:
    observed = get_center(obj)
    if observed is None or not isinstance(reported, (list, tuple)) or len(reported) != 3:
        return False, {"reported": reported, "observed": observed, "tolerance_m": GROUNDING_CENTER_TOLERANCE_M}
    try:
        report_center = [float(value) for value in reported]
        distance = math.sqrt(sum((report_center[index] - float(observed[index])) ** 2 for index in range(3)))
    except (TypeError, ValueError):
        return False, {"reported": reported, "observed": observed, "tolerance_m": GROUNDING_CENTER_TOLERANCE_M}
    return distance <= GROUNDING_CENTER_TOLERANCE_M, {
        "reported": report_center,
        "observed": [float(value) for value in observed[:3]],
        "distance_m": round(distance, 6),
        "tolerance_m": GROUNDING_CENTER_TOLERANCE_M,
    }


def _select_grasp(obj: ObjectDict, objects: Iterable[ObjectDict], options: dict) -> dict:
    keys = {
        "yaw_step_deg", "local_refine_step_deg", "gripper_outer_width_m",
        "gripper_inner_width_m", "current_wrist_yaw_deg", "approach_length_m",
        "z_tolerance_m", "side_clearance_m",
    }
    kwargs = {key: options[key] for key in keys if options.get(key) is not None}
    return select_best_grasp(obj, objects, **kwargs)


def _compact_grasp_report(grasp: dict) -> dict:
    return {
        key: grasp.get(key)
        for key in (
            "grasp_feasible", "selected_grasp_yaw_deg", "selected_grasp_axis_delta_deg",
            "selected_grasp_source", "feasible_yaw_intervals_deg", "blocked_yaw_intervals_deg",
            "all_grasps_blocked", "blocking_objects", "parameters",
        )
    }


def _duplicate_object_ids(objects: Iterable[ObjectDict]) -> List[Any]:
    seen = set()
    duplicates = []
    for obj in objects:
        object_id = obj.get("id")
        key = str(object_id)
        if key in seen and object_id not in duplicates:
            duplicates.append(object_id)
        seen.add(key)
    return duplicates


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
