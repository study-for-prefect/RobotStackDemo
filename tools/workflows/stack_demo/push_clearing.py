"""Push-clearing plan construction and relation-scene safety annotations."""

import copy
from typing import Any, Dict, Iterable, List, Optional

from robot_scene_pipeline.geometry_relations import has_free_push_space


ObjectDict = Dict[str, Any]
RelationDict = Dict[str, Any]


def object_by_string_id(objects: Iterable[ObjectDict], object_id: Any) -> ObjectDict:
    for obj in objects:
        if str(obj.get("id")) == str(object_id):
            return obj
    raise RuntimeError("Push obstacle id {} is absent from current_state.".format(object_id))


def locked_stack_object_ids(base_id: Any, previous_locked_stack: Optional[dict]) -> set:
    locked_ids = {str(base_id)}
    if previous_locked_stack:
        for obj in previous_locked_stack.get("stack_objects", []):
            if obj.get("id") is not None:
                locked_ids.add(str(obj["id"]))
    return locked_ids


def relation_objects_with_protected_structure(
    objects: Iterable[ObjectDict],
    base_id: Any,
    previous_locked_stack: Optional[dict],
) -> List[ObjectDict]:
    locked_ids = locked_stack_object_ids(base_id, previous_locked_stack)
    relation_objects = copy.deepcopy(list(objects))
    for obj in relation_objects:
        if str(obj.get("id")) in locked_ids:
            obj["pushable"] = False
    return relation_objects


def validate_push_clearance_against_structure(
    current_state: dict,
    selected_push: RelationDict,
    base_id: Any,
    previous_locked_stack: Optional[dict],
) -> None:
    locked_ids = locked_stack_object_ids(base_id, previous_locked_stack)
    objects = copy.deepcopy(current_state.get("objects", []))
    obstacle = object_by_string_id(objects, selected_push.get("subject"))
    for obj in objects:
        if str(obj.get("id")) in locked_ids:
            obj["role"] = "base" if str(obj.get("id")) == str(base_id) else "structure"
            obj["state"] = "locked"
    if not has_free_push_space(
        obstacle,
        objects,
        selected_push["direction_base"],
        distance_m=selected_push.get("distance_m", 0.05),
    ):
        raise RuntimeError(
            "Push clearing refused: proposed obstacle motion intersects the locked "
            "base/stack structure or lacks free space."
        )


def build_push_execution_plan(
    current_state: dict,
    held_object: ObjectDict,
    selected_push: RelationDict,
    args: Any,
) -> dict:
    obstacle = copy.deepcopy(
        object_by_string_id(current_state.get("objects", []), selected_push.get("subject"))
    )
    return {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "execution_status": "not_executed",
        "target_object_id": held_object["id"],
        "obstacle_object_id": selected_push["subject"],
        "direction_base": selected_push["direction_base"],
        "distance_m": selected_push.get("distance_m", args.push_clearing_distance_m),
        "lift_m": args.push_clearing_lift_m,
        "contact_z_offset_m": args.push_clearing_contact_z_offset_m,
        "obstacle": obstacle,
        "reason": selected_push.get("reason"),
    }
