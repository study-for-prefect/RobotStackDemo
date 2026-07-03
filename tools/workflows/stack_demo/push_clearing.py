"""Push-clearing plan construction and relation-scene safety annotations."""

import copy
import math
from typing import Any, Dict, Iterable, List, Optional

from robot_scene_pipeline.geometry_relations import (
    get_center,
    object_xy_aabb,
    xy_aabb_overlap,
)
from robot_scene_pipeline.push_grasp_joint_evaluator import (
    evaluate_push_grasp_joint_candidates,
)


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


def _normalized_xy(direction: Iterable[float]) -> List[float]:
    values = [float(value) for value in list(direction)[:2]]
    norm = math.hypot(values[0], values[1])
    if norm < 1e-9:
        return [1.0, 0.0]
    return [values[0] / norm, values[1] / norm]


def candidate_push_directions(obstacle: ObjectDict, target: ObjectDict) -> List[dict]:
    raw_candidates = [
        ("base_positive_x", [1.0, 0.0]),
        ("base_negative_x", [-1.0, 0.0]),
        ("base_positive_y", [0.0, 1.0]),
        ("base_negative_y", [0.0, -1.0]),
    ]
    candidates = []
    for source, direction in raw_candidates:
        direction = _normalized_xy(direction)
        if any(
            abs(direction[0] - item["direction_base"][0]) < 1e-6
            and abs(direction[1] - item["direction_base"][1]) < 1e-6
            for item in candidates
        ):
            continue
        candidates.append(
            {
                "source": source,
                "direction_base": [direction[0], direction[1], 0.0],
            }
        )
    return candidates


def _translated_aabb(aabb: dict, dx: float, dy: float) -> dict:
    output = dict(aabb)
    output["xmin"] += dx
    output["xmax"] += dx
    output["ymin"] += dy
    output["ymax"] += dy
    return output


def _swept_aabb(start_aabb: dict, end_aabb: dict, clearance_m: float) -> dict:
    clearance = max(0.0, float(clearance_m))
    return {
        "xmin": min(start_aabb["xmin"], end_aabb["xmin"]) - clearance,
        "xmax": max(start_aabb["xmax"], end_aabb["xmax"]) + clearance,
        "ymin": min(start_aabb["ymin"], end_aabb["ymin"]) - clearance,
        "ymax": max(start_aabb["ymax"], end_aabb["ymax"]) + clearance,
        "zmin": start_aabb["zmin"],
        "zmax": start_aabb["zmax"],
    }


def _z_ranges_near(aabb_a: dict, aabb_b: dict, tolerance_m: float = 0.04) -> bool:
    if aabb_a["zmax"] < aabb_b["zmin"]:
        return aabb_b["zmin"] - aabb_a["zmax"] < tolerance_m
    if aabb_b["zmax"] < aabb_a["zmin"]:
        return aabb_a["zmin"] - aabb_b["zmax"] < tolerance_m
    return True


def evaluate_push_directions(
    obstacle: ObjectDict,
    target: ObjectDict,
    objects: Iterable[ObjectDict],
    distance_m: float = 0.05,
    table_bounds: Optional[dict] = None,
    clearance_m: float = 0.01,
) -> List[dict]:
    obstacle_aabb = object_xy_aabb(obstacle)
    target_center = get_center(target)
    if not obstacle_aabb or target_center is None:
        return []

    distance = float(distance_m)
    evaluations = []
    for candidate in candidate_push_directions(obstacle, target):
        direction = candidate["direction_base"]
        dx = direction[0] * distance
        dy = direction[1] * distance
        moved_aabb = _translated_aabb(obstacle_aabb, dx, dy)
        swept_aabb = _swept_aabb(obstacle_aabb, moved_aabb, clearance_m)
        out_of_bounds = False
        if table_bounds is not None:
            try:
                out_of_bounds = (
                    moved_aabb["xmin"] < float(table_bounds["xmin"])
                    or moved_aabb["xmax"] > float(table_bounds["xmax"])
                    or moved_aabb["ymin"] < float(table_bounds["ymin"])
                    or moved_aabb["ymax"] > float(table_bounds["ymax"])
                )
            except (KeyError, TypeError, ValueError):
                out_of_bounds = True

        collisions = []
        for other in objects:
            if not isinstance(other, dict) or str(other.get("id")) == str(obstacle.get("id")):
                continue
            if other.get("visible") is False:
                continue
            other_aabb = object_xy_aabb(other)
            if not other_aabb or not _z_ranges_near(swept_aabb, other_aabb):
                continue
            if xy_aabb_overlap(swept_aabb, other_aabb)[2] > 0.0:
                collisions.append(
                    {
                        "id": other.get("id"),
                        "label": other.get("label"),
                        "role": other.get("role"),
                        "state": other.get("state"),
                    }
                )

        moved_center = [
            (moved_aabb["xmin"] + moved_aabb["xmax"]) / 2.0,
            (moved_aabb["ymin"] + moved_aabb["ymax"]) / 2.0,
        ]
        target_distance_after = math.hypot(
            moved_center[0] - target_center[0],
            moved_center[1] - target_center[1],
        )
        feasible = not out_of_bounds and not collisions
        source_bonus = 0.02 if candidate["source"] == "away_from_target" else 0.0
        evaluations.append(
            {
                "source": candidate["source"],
                "direction_base": direction,
                "distance_m": distance,
                "feasible": feasible,
                "out_of_table_bounds": out_of_bounds,
                "collisions": collisions,
                "target_distance_after_m": round(target_distance_after, 6),
                "score": round(target_distance_after + source_bonus, 6),
            }
        )
    evaluations.sort(
        key=lambda item: (
            not item["feasible"],
            -float(item["score"]),
            item["source"],
        )
    )
    return evaluations


def select_push_direction(
    current_state: dict,
    held_object: ObjectDict,
    relation: RelationDict,
    distance_m: float,
    table_bounds: Optional[dict] = None,
) -> dict:
    obstacle = object_by_string_id(current_state.get("objects", []), relation.get("subject"))
    evaluations = evaluate_push_directions(
        obstacle,
        held_object,
        current_state.get("objects", []),
        distance_m=distance_m,
        table_bounds=table_bounds,
    )
    selected = next((item for item in evaluations if item["feasible"]), None)
    return {
        "relation": relation,
        "obstacle": obstacle,
        "evaluations": evaluations,
        "selected_direction": selected,
    }


def evaluate_push_candidates(
    current_state: dict,
    held_object: ObjectDict,
    push_candidates: Iterable[RelationDict],
    distance_m: float,
    table_bounds: Optional[dict] = None,
    qwen_candidates: Optional[Iterable[dict]] = None,
    future_targets: Optional[Iterable[ObjectDict]] = None,
    future_place_regions: Optional[Iterable[dict]] = None,
    protected_objects: Optional[Iterable[ObjectDict]] = None,
    memory: Optional[dict] = None,
    lift_m: float = 0.05,
    contact_z_offset_m: float = 0.015,
    gripper_outer_width_m: float = 0.112,
    gripper_inner_width_m: float = 0.048,
    grasp_approach_length_m: float = 0.02,
    push_tool_width_m: float = 0.035,
    push_tool_safety_margin_m: float = 0.005,
) -> dict:
    relations = list(push_candidates)
    obstacle_ids = []
    relation_by_obstacle = {}
    for relation in relations:
        obstacle_id = relation.get("subject")
        if obstacle_id is None or str(obstacle_id) in relation_by_obstacle:
            continue
        obstacle_ids.append(obstacle_id)
        relation_by_obstacle[str(obstacle_id)] = relation

    joint = evaluate_push_grasp_joint_candidates(
        current_state,
        held_object,
        obstacle_ids,
        qwen_candidates=qwen_candidates or [],
        future_targets=future_targets or [],
        future_place_regions=future_place_regions or [],
        protected_objects=protected_objects or [],
        memory=memory,
        table_bounds=table_bounds,
        lift_m=lift_m,
        contact_z_offset_m=contact_z_offset_m,
        gripper_outer_width_m=gripper_outer_width_m,
        gripper_inner_width_m=gripper_inner_width_m,
        grasp_approach_length_m=grasp_approach_length_m,
        push_tool_width_m=push_tool_width_m,
        push_tool_safety_margin_m=push_tool_safety_margin_m,
    )
    candidate_results = []
    for relation in relations:
        obstacle_id = str(relation.get("subject"))
        evaluations = [
            item for item in joint["candidates"]
            if str(item.get("obstacle_id")) == obstacle_id
        ]
        feasible = [item for item in evaluations if item.get("feasible")]
        feasible.sort(key=lambda item: -float(item.get("score", 0.0)))
        selected = None
        if feasible:
            best = feasible[0]
            selected = {
                "source": best.get("source"),
                "direction_base": best.get("direction_base"),
                "distance_m": best.get("distance_m", distance_m),
                "feasible": True,
                "score": best.get("score", 0.0),
                "reason": best.get("reason"),
                "predicted_selected_grasp_yaw_deg": best.get("predicted_selected_grasp_yaw_deg"),
            }
        candidate_results.append(
            {
                "relation": relation,
                "obstacle": object_by_string_id(current_state.get("objects", []), relation.get("subject")),
                "evaluations": evaluations,
                "selected_direction": selected,
            }
        )
    selected_candidate = joint.get("selected_candidate")
    selected_result = None
    if selected_candidate is not None:
        selected_relation = relation_by_obstacle.get(str(selected_candidate.get("obstacle_id")))
        if selected_relation is not None:
            selected_result = next(
                (
                    result for result in candidate_results
                    if str(result["relation"].get("subject")) == str(selected_candidate.get("obstacle_id"))
                ),
                None,
            )
    return {
        "candidate_results": candidate_results,
        "selected_result": selected_result,
        "joint_evaluation": joint,
    }


def blocking_relations_for_target(
    relations: Iterable[RelationDict],
    target_object_id: Any,
) -> List[RelationDict]:
    return [
        relation
        for relation in relations
        if relation.get("type") in ("blocking_grasp", "should_push_away")
        and str(relation.get("object")) == str(target_object_id)
    ]


def pushable_blocking_relations(
    current_state: dict,
    relations: Iterable[RelationDict],
    target_object_id: Any,
    base_id: Any,
    previous_locked_stack: Optional[dict],
    protected_object_ids: Optional[Iterable[Any]] = None,
) -> List[RelationDict]:
    protected_ids = {str(value) for value in protected_object_ids or []}
    if not protected_ids:
        protected_ids = locked_stack_object_ids(base_id, previous_locked_stack)
    output = []
    seen = set()
    for relation in relations:
        if relation.get("type") != "blocking_grasp":
            continue
        if str(relation.get("object")) != str(target_object_id):
            continue
        obstacle_id = relation.get("subject")
        if str(obstacle_id) in protected_ids or str(obstacle_id) in seen:
            continue
        obstacle = object_by_string_id(current_state.get("objects", []), obstacle_id)
        if obstacle.get("pushable") is False:
            continue
        if obstacle.get("role") in ("base", "structure"):
            continue
        if obstacle.get("state") in ("locked", "placed"):
            continue
        candidate = dict(relation)
        candidate["type"] = "should_push_away"
        candidate["reason"] = "blocking_grasp_requires_direction_evaluation"
        output.append(candidate)
        seen.add(str(obstacle_id))
    return output


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
        "direction_evaluations": direction_evaluations or [],
    }
