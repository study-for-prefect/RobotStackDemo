"""Push plan construction and protected-structure annotations."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional

from robot_scene_pipeline.geometry_relations import get_center


ObjectDict = Dict[str, Any]
RelationDict = Dict[str, Any]


def object_by_string_id(objects: Iterable[ObjectDict], object_id: Any) -> ObjectDict:
    for obj in objects:
        if str(obj.get("id")) == str(object_id):
            return obj
    raise RuntimeError("Object id {} is absent from current_state.".format(object_id))


def locked_stack_object_ids(base_id: Any, previous_locked_stack: Optional[dict]) -> set:
    locked_ids = {str(base_id)}
    if previous_locked_stack:
        for obj in previous_locked_stack.get("stack_objects", []):
            if obj.get("id") is not None:
                locked_ids.add(str(obj["id"]))
    return locked_ids


def _xy_distance(first: ObjectDict, second: ObjectDict) -> float:
    first_center = get_center(first)
    second_center = get_center(second)
    if first_center is None or second_center is None:
        return math.inf
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def _matches_locked_template(obj: ObjectDict, template: ObjectDict, max_xy_m: float = 0.04) -> bool:
    if not isinstance(obj, dict) or not isinstance(template, dict):
        return False
    obj_label = obj.get("label")
    template_label = template.get("label")
    if obj_label is not None and template_label is not None and obj_label != template_label:
        return False
    return _xy_distance(obj, template) <= float(max_xy_m)


def current_protected_structure_ids(
    objects: Iterable[ObjectDict],
    base_id: Any,
    previous_locked_stack: Optional[dict],
    base_template: Optional[ObjectDict] = None,
) -> set:
    scene_objects = list(objects)
    protected_ids = set()
    templates: List[ObjectDict] = []
    if isinstance(base_template, dict):
        templates.append(base_template)
    if previous_locked_stack:
        templates.extend(previous_locked_stack.get("stack_objects", []) or [])
    for template in templates:
        matches = [obj for obj in scene_objects if _matches_locked_template(obj, template)]
        if matches:
            matches.sort(key=lambda obj: _xy_distance(obj, template))
            protected_ids.add(str(matches[0].get("id")))
    if not templates and base_id is not None:
        protected_ids.add(str(base_id))
    return protected_ids


def relation_objects_with_protected_structure(
    objects: Iterable[ObjectDict],
    base_id: Any,
    previous_locked_stack: Optional[dict],
    base_template: Optional[ObjectDict] = None,
) -> List[ObjectDict]:
    scene_objects = list(objects)
    locked_ids = current_protected_structure_ids(scene_objects, base_id, previous_locked_stack, base_template)
    if base_template is not None:
        base_ids = current_protected_structure_ids(scene_objects, base_id, None, base_template)
    else:
        base_ids = {str(base_id)}
    relation_objects = copy.deepcopy(scene_objects)
    for obj in relation_objects:
        if str(obj.get("id")) in locked_ids:
            obj["pushable"] = False
            obj["role"] = "base" if str(obj.get("id")) in base_ids else "structure"
            obj["state"] = "locked"
        elif obj.get("role") is None and obj.get("state") is None:
            obj["role"] = "loose_movable"
            obj["state"] = "free"
    return relation_objects


def build_push_execution_plan(
    current_state: dict,
    held_object: ObjectDict,
    selected_push: RelationDict,
    args: Any,
    direction_evaluations: Optional[List[dict]] = None,
) -> dict:
    obstacle = copy.deepcopy(
        object_by_string_id(current_state.get("objects", []), selected_push.get("subject"))
    )
    reference_yaw = selected_push.get("selected_grasp_yaw_deg")
    reference_yaw_source = "selected_clearance_grasp_yaw"
    if reference_yaw is None:
        reference_yaw = selected_push.get("predicted_selected_grasp_yaw_deg")
        reference_yaw_source = "predicted_post_push_grasp_yaw"
    if reference_yaw is None:
        reference_yaw = held_object.get("selected_grasp_yaw_deg")
        reference_yaw_source = held_object.get("grasp_yaw_source") or "target_selected_grasp_yaw"
    if reference_yaw is None:
        reference_yaw = held_object.get("table_yaw_deg")
        reference_yaw_source = held_object.get("table_yaw_source") or "target_table_yaw"
    reference_yaw_valid = reference_yaw is not None
    return {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "execution_status": "not_executed",
        "target_object_id": held_object["id"],
        "obstacle_object_id": selected_push["subject"],
        "action_id": selected_push.get("action_id"),
        "direction_base": selected_push["direction_base"],
        "distance_m": selected_push.get("distance_m", args.push_clearing_distance_m),
        "lift_m": args.push_clearing_lift_m,
        "contact_z_offset_m": args.push_clearing_contact_z_offset_m,
        "push_orientation_policy": selected_push.get("push_orientation_policy") or "preserve_current_tool_orientation",
        "target_yaw_deg": None,
        "target_yaw_valid": False,
        "target_yaw_source": "not_used_for_push_orientation",
        "reference_target_yaw_deg": None if reference_yaw is None else float(reference_yaw),
        "reference_target_yaw_valid": bool(reference_yaw_valid),
        "reference_target_yaw_source": reference_yaw_source if reference_yaw_valid else "missing_target_yaw",
        "target_table_yaw_valid": bool(held_object.get("table_yaw_valid")),
        "target_yaw_frame": "base_link",
        "target": copy.deepcopy(held_object),
        "obstacle": obstacle,
        "reason": selected_push.get("reason"),
        "direction_evaluations": direction_evaluations or [],
    }
