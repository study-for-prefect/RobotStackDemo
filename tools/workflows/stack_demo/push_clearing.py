"""Push-clearing plan construction and relation-scene safety annotations."""

import copy
import math
from typing import Any, Dict, Iterable, List, Optional

from robot_scene_pipeline.geometry_relations import (
    get_center,
    object_xy_aabb,
    xy_aabb_overlap,
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


def _normalized_xy(direction: Iterable[float]) -> List[float]:
    values = [float(value) for value in list(direction)[:2]]
    norm = math.hypot(values[0], values[1])
    if norm < 1e-9:
        return [1.0, 0.0]
    return [values[0] / norm, values[1] / norm]


def candidate_push_directions(obstacle: ObjectDict, target: ObjectDict) -> List[dict]:
    obstacle_center = get_center(obstacle)
    target_center = get_center(target)
    if obstacle_center is None or target_center is None:
        away = [1.0, 0.0]
    else:
        away = _normalized_xy(
            [
                obstacle_center[0] - target_center[0],
                obstacle_center[1] - target_center[1],
            ]
        )
    raw_candidates = [
        ("away_from_target", away),
        ("perpendicular_left", [-away[1], away[0]]),
        ("perpendicular_right", [away[1], -away[0]]),
        ("toward_opposite_side", [-away[0], -away[1]]),
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
) -> dict:
    candidate_results = [
        select_push_direction(
            current_state,
            held_object,
            relation,
            distance_m=distance_m,
            table_bounds=table_bounds,
        )
        for relation in push_candidates
    ]
    feasible = [
        result
        for result in candidate_results
        if result.get("selected_direction") is not None
    ]
    feasible.sort(
        key=lambda result: -float(result["selected_direction"]["score"])
    )
    return {
        "candidate_results": candidate_results,
        "selected_result": feasible[0] if feasible else None,
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
) -> List[RelationDict]:
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
