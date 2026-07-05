"""Obstruction graph and frontier clearance candidate generation."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from robot_scene_pipeline.geometry_relations import get_center, get_size, object_xy_aabb, xy_aabb_overlap
from robot_scene_pipeline.grasp_yaw_search import select_best_grasp

from .push_clearing import evaluate_push_candidates


ObjectDict = Dict[str, Any]


def _object_id(obj: ObjectDict) -> str:
    return str(obj.get("id"))


def _find_object(objects: Iterable[ObjectDict], object_id: Any) -> Optional[ObjectDict]:
    for obj in objects:
        if _object_id(obj) == str(object_id):
            return obj
    return None


def _is_protected(obj: ObjectDict, protected_ids: Iterable[Any]) -> bool:
    protected = {str(value) for value in protected_ids or []}
    return (
        _object_id(obj) in protected
        or obj.get("role") in ("base", "structure")
        or obj.get("state") in ("locked", "placed")
        or obj.get("pushable") is False
    )


def _top_z(obj: ObjectDict) -> float:
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return 0.0
    return float(center[2]) + 0.5 * float(size[2])


def _feasible_yaw_count(grasp: dict) -> int:
    return len([item for item in grasp.get("candidate_results", []) if item.get("feasible")])


def _target_yaw_gain(target: ObjectDict, objects: List[ObjectDict], removed_id: Any, args: Any) -> dict:
    before = select_best_grasp(
        target,
        objects,
        gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
        gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
        approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
    )
    after_objects = [obj for obj in objects if _object_id(obj) != str(removed_id)]
    after = select_best_grasp(
        target,
        after_objects,
        gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
        gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
        approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
    )
    before_count = _feasible_yaw_count(before)
    after_count = _feasible_yaw_count(after)
    return {
        "before_feasible_yaw_count": before_count,
        "after_feasible_yaw_count": after_count,
        "gain": max(0, after_count - before_count),
        "after_grasp_feasible": bool(after.get("grasp_feasible")),
    }


def build_obstruction_graph(
    current_state: dict,
    target: ObjectDict,
    protected_ids: Iterable[Any],
    args: Any,
    max_depth: int = 3,
) -> dict:
    objects = [obj for obj in current_state.get("objects", []) if isinstance(obj, dict)]
    queue: List[Tuple[ObjectDict, int, Optional[str]]] = [(target, 0, None)]
    expanded = set()
    nodes: Dict[str, dict] = {}
    edges: List[dict] = []
    frontier: Dict[str, dict] = {}

    while queue:
        blocked_obj, depth, parent_id = queue.pop(0)
        blocked_id = _object_id(blocked_obj)
        if (blocked_id, depth) in expanded or depth > int(max_depth):
            continue
        expanded.add((blocked_id, depth))
        grasp = select_best_grasp(
            blocked_obj,
            objects,
            gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
            gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
            approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
        )
        nodes.setdefault(
            blocked_id,
            {
                "object_id": blocked_obj.get("id"),
                "label": blocked_obj.get("label"),
                "depth": depth,
                "parent_blocked_object_id": parent_id,
                "grasp_feasible": bool(grasp.get("grasp_feasible")),
                "selected_grasp_yaw_deg": grasp.get("selected_grasp_yaw_deg"),
                "feasible_yaw_count": _feasible_yaw_count(grasp),
                "blocking_objects": grasp.get("blocking_objects", []),
            },
        )
        if grasp.get("grasp_feasible"):
            continue
        for blocker in grasp.get("blocking_objects", []):
            blocker_id = blocker.get("id")
            blocker_obj = _find_object(objects, blocker_id)
            if blocker_obj is None:
                continue
            blocker_key = str(blocker_id)
            edges.append(
                {
                    "subject": blocker_id,
                    "object": blocked_obj.get("id"),
                    "depth": depth + 1,
                    "relation": "blocks_grasp",
                    "blocker_category": blocker.get("blocker_category"),
                }
            )
            if _is_protected(blocker_obj, protected_ids):
                continue
            item = frontier.setdefault(
                blocker_key,
                {
                    "object_id": blocker_obj.get("id"),
                    "label": blocker_obj.get("label"),
                    "min_depth": depth + 1,
                    "blocks": [],
                },
            )
            item["min_depth"] = min(int(item["min_depth"]), depth + 1)
            if str(blocked_obj.get("id")) not in {str(value) for value in item["blocks"]}:
                item["blocks"].append(blocked_obj.get("id"))
            if depth + 1 < int(max_depth):
                queue.append((blocker_obj, depth + 1, blocked_id))

    return {
        "schema_version": "obstruction_graph_v1",
        "target_object_id": target.get("id"),
        "nodes": list(nodes.values()),
        "edges": edges,
        "frontier": list(frontier.values()),
    }


def _translated_object(obj: ObjectDict, center_xy: Iterable[float]) -> ObjectDict:
    output = copy.deepcopy(obj)
    center = get_center(output)
    if center is None:
        return output
    xy = [float(value) for value in list(center_xy)[:2]]
    output["geometry_center_m"] = [xy[0], xy[1], float(center[2])]
    return output


def _safe_place_for_object(
    obj: ObjectDict,
    target: ObjectDict,
    objects: List[ObjectDict],
    protected_ids: Iterable[Any],
    table_bounds: Optional[dict],
    margin_m: float = 0.02,
) -> Optional[List[float]]:
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None or not table_bounds:
        return None
    try:
        xmin = float(table_bounds["xmin"]) + 0.5 * size[0] + margin_m
        xmax = float(table_bounds["xmax"]) - 0.5 * size[0] - margin_m
        ymin = float(table_bounds["ymin"]) + 0.5 * size[1] + margin_m
        ymax = float(table_bounds["ymax"]) - 0.5 * size[1] - margin_m
    except (KeyError, TypeError, ValueError):
        return None
    if xmin >= xmax or ymin >= ymax:
        return None
    target_center = get_center(target) or center
    samples = [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
        [(xmin + xmax) / 2.0, ymin],
        [(xmin + xmax) / 2.0, ymax],
        [xmin, (ymin + ymax) / 2.0],
        [xmax, (ymin + ymax) / 2.0],
    ]
    protected = {str(value) for value in protected_ids or []}
    samples.sort(key=lambda xy: -math.hypot(xy[0] - target_center[0], xy[1] - target_center[1]))
    for xy in samples:
        placed = _translated_object(obj, xy)
        placed_aabb = object_xy_aabb(placed, margin_m=0.005)
        if not placed_aabb:
            continue
        blocked = False
        for other in objects:
            if _object_id(other) == _object_id(obj):
                continue
            if _object_id(other) not in protected and _object_id(other) != _object_id(target):
                continue
            other_aabb = object_xy_aabb(other, margin_m=0.005)
            if other_aabb and xy_aabb_overlap(placed_aabb, other_aabb)[2] > 0.0:
                blocked = True
                break
        if not blocked:
            return [round(float(xy[0]), 5), round(float(xy[1]), 5), round(float(center[2]), 5)]
    return None


def _score_candidate(candidate: dict) -> dict:
    utility = float(candidate.get("utility_score", 0.0))
    easiness = float(candidate.get("easiness_score", 0.0))
    risk = float(candidate.get("risk_score", 0.0))
    candidate["score"] = round(utility + easiness - risk, 6)
    return candidate


def _make_pick_away_candidate(
    obj: ObjectDict,
    target: ObjectDict,
    objects: List[ObjectDict],
    protected_ids: Iterable[Any],
    graph_item: dict,
    args: Any,
    index: int,
    table_bounds: Optional[dict],
) -> Optional[dict]:
    blocked_ids = {str(value) for value in graph_item.get("blocks", [])}
    grasp_objects = [item for item in objects if _object_id(item) not in blocked_ids]
    grasp = select_best_grasp(
        obj,
        grasp_objects,
        gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
        gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
        approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
    )
    if not grasp.get("grasp_feasible"):
        return None
    safe_place = _safe_place_for_object(obj, target, objects, protected_ids, table_bounds)
    if safe_place is None:
        return None
    yaw_gain = _target_yaw_gain(target, objects, obj.get("id"), args)
    direct = int(graph_item.get("min_depth", 99)) == 1
    utility = 2.0 if direct else 1.0
    utility += 0.3 * len(set(str(value) for value in graph_item.get("blocks", [])))
    utility += 0.2 * float(yaw_gain["gain"])
    if yaw_gain["after_grasp_feasible"]:
        utility += 1.0
    easiness = 1.0 + 0.05 * _feasible_yaw_count(grasp)
    risk = 0.1 * max(0, int(graph_item.get("min_depth", 1)) - 1)
    return _score_candidate(
        {
            "candidate_id": "clear_{:03d}_pick_away_{}".format(index, obj.get("id")),
            "action": "pick_away",
            "action_type": "pick_away",
            "obstacle_id": obj.get("id"),
            "target_object_id": target.get("id"),
            "frontier_depth": graph_item.get("min_depth"),
            "blocks": graph_item.get("blocks", []),
            "selected_grasp_yaw_deg": grasp.get("selected_grasp_yaw_deg"),
            "feasible_yaw_count": _feasible_yaw_count(grasp),
            "safe_place_center_m": safe_place,
            "utility_score": round(utility, 6),
            "easiness_score": round(easiness, 6),
            "risk_score": round(risk, 6),
            "target_yaw_gain": yaw_gain,
            "reason": "frontier_obstacle_graspable_pick_away_first",
            "requires_moveit_preflight": True,
            "ignored_blocked_objects_for_grasp_check": sorted(blocked_ids),
        }
    )


def build_frontier_clearance_plan(
    current_state: dict,
    target: ObjectDict,
    protected_ids: Iterable[Any],
    args: Any,
    future_place_regions: Optional[Iterable[dict]] = None,
    evaluate_push_fn: Callable[..., dict] = evaluate_push_candidates,
) -> dict:
    objects = [obj for obj in current_state.get("objects", []) if isinstance(obj, dict)]
    graph = build_obstruction_graph(
        current_state,
        target,
        protected_ids,
        args,
        max_depth=getattr(args, "obstruction_graph_max_depth", 3),
    )
    all_candidates: List[dict] = []
    safe_candidates: List[dict] = []
    candidate_index = 1
    nudge_max = float(getattr(args, "clearance_nudge_distance_m", 0.025))

    for frontier_item in graph.get("frontier", []):
        obstacle = _find_object(objects, frontier_item.get("object_id"))
        if obstacle is None or _is_protected(obstacle, protected_ids):
            continue
        pick_candidate = _make_pick_away_candidate(
            obstacle,
            target,
            objects,
            protected_ids,
            frontier_item,
            args,
            candidate_index,
            current_state.get("table_bounds"),
        )
        if pick_candidate is not None:
            all_candidates.append(pick_candidate)
            safe_candidates.append(pick_candidate)
            candidate_index += 1

        relation_target_id = frontier_item.get("blocks", [target.get("id")])[0]
        relation_target = _find_object(objects, relation_target_id) or target
        relation = {
            "type": "should_push_away",
            "subject": obstacle.get("id"),
            "object": relation_target.get("id"),
            "source": "obstruction_frontier",
            "reason": "frontier_obstacle_not_yet_cleared",
        }
        push_report = evaluate_push_fn(
            current_state,
            relation_target,
            [relation],
            distance_m=nudge_max,
            table_bounds=current_state.get("table_bounds"),
            future_targets=[],
            future_place_regions=future_place_regions or [],
            protected_objects=[
                obj for obj in objects
                if _object_id(obj) in {str(value) for value in protected_ids or []}
            ],
            memory=None,
            lift_m=getattr(args, "push_clearing_lift_m", 0.05),
            contact_z_offset_m=getattr(args, "push_clearing_contact_z_offset_m", 0.015),
            gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
            gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
            grasp_approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
            push_tool_width_m=getattr(args, "push_tool_width_m", 0.035),
            push_tool_safety_margin_m=getattr(args, "push_tool_safety_margin_m", 0.005),
        )
        for result in push_report.get("candidate_results", []):
            for evaluation in result.get("evaluations", []):
                if float(evaluation.get("distance_m") or 0.0) > nudge_max + 1e-9:
                    continue
                yaw_gain = _target_yaw_gain(target, objects, obstacle.get("id"), args)
                direct = int(frontier_item.get("min_depth", 99)) == 1
                utility = (1.2 if direct else 0.6) + 0.1 * float(yaw_gain["gain"])
                if yaw_gain["after_grasp_feasible"]:
                    utility += 0.5
                easiness = 0.4 + max(0.0, float(evaluation.get("score", 0.0)))
                risk = 0.4 + 0.2 * max(0, int(frontier_item.get("min_depth", 1)) - 1)
                candidate = _score_candidate(
                    {
                        "candidate_id": evaluation.get("candidate_id") or "clear_{:03d}_nudge_{}".format(candidate_index, obstacle.get("id")),
                        "action": "nudge",
                        "action_type": "nudge",
                        "obstacle_id": obstacle.get("id"),
                        "target_object_id": target.get("id"),
                        "frontier_depth": frontier_item.get("min_depth"),
                        "blocks": frontier_item.get("blocks", []),
                        "direction_base": evaluation.get("direction_base"),
                        "distance_m": evaluation.get("distance_m"),
                        "direction_source": evaluation.get("source"),
                        "utility_score": round(utility, 6),
                        "easiness_score": round(easiness, 6),
                        "risk_score": round(risk, 6),
                        "target_yaw_gain": yaw_gain,
                        "reason": evaluation.get("reason"),
                        "feasible": bool(evaluation.get("feasible")),
                        "push_evaluation": evaluation,
                        "push_relation": relation,
                        "push_selected_result": result,
                        "requires_moveit_preflight": True,
                    }
                )
                all_candidates.append(candidate)
                if evaluation.get("feasible"):
                    safe_candidates.append(candidate)
                candidate_index += 1

    safe_candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("frontier_depth", 99)), str(item.get("candidate_id"))))
    all_candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("frontier_depth", 99)), str(item.get("candidate_id"))))
    return {
        "schema_version": "obstruction_frontier_clearance_v1",
        "obstruction_graph": graph,
        "obstacle_frontier_candidates": graph.get("frontier", []),
        "all_clearance_action_candidates": all_candidates,
        "safe_clearance_candidates": safe_candidates,
        "selected_clearance_action": safe_candidates[0] if safe_candidates else None,
    }
