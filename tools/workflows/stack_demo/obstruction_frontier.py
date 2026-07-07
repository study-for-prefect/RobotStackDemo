"""Obstruction graph and frontier clearance candidate generation."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from robot_scene_pipeline.detection_merge import merge_duplicate_objects_3d
from robot_scene_pipeline.geometry_relations import get_center
from robot_scene_pipeline.grasp_yaw_search import select_best_grasp

from .clearance_placement import safe_place_for_object
from .clearance_policy import annotate_nudge_preflight_policy, refresh_executable_safe
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


def _status_tags(obj: ObjectDict, protected_ids: Iterable[Any]) -> dict:
    return {
        "protected_structure_safe": not _is_protected(obj, protected_ids),
        "role": obj.get("role"),
        "state": obj.get("state"),
        "pushable": obj.get("pushable"),
    }


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
    target_id = str(target.get("id"))
    queue: List[Tuple[ObjectDict, int, Optional[str]]] = [(target, 0, None)]
    expanded = set()
    nodes: Dict[str, dict] = {}
    edges: List[dict] = []
    frontier: Dict[str, dict] = {}

    while queue:
        blocked_obj, depth, parent_id = queue.pop(0)
        blocked_id = _object_id(blocked_obj)
        if blocked_id in expanded or depth > int(max_depth):
            continue
        expanded.add(blocked_id)
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
                **_status_tags(blocked_obj, protected_ids),
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
            if blocker_key == target_id:
                continue
            if _is_protected(blocker_obj, protected_ids):
                continue
            item = frontier.setdefault(
                blocker_key,
                {
                    "object_id": blocker_obj.get("id"),
                    "label": blocker_obj.get("label"),
                    "min_depth": depth + 1,
                    "blocks": [],
                    **_status_tags(blocker_obj, protected_ids),
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


def _score_candidate(candidate: dict) -> dict:
    direct_gain = float(candidate.get("direct_target_gain", 0.0))
    direct_progress_gain = float(candidate.get("direct_progress_gain", 0.0))
    enabling_gain = float(candidate.get("enabling_gain", 0.0))
    free_space_gain = float(candidate.get("free_space_gain", 0.0))
    candidate["task_effective"] = bool(
        direct_gain > 0.0 or direct_progress_gain > 0.0 or enabling_gain > 0.0 or free_space_gain > 0.0
    )
    yaw_gain = candidate.get("target_yaw_gain", {})
    has_direct_target_gain = bool(
        float(yaw_gain.get("gain", 0.0)) > 0.0
        or yaw_gain.get("after_grasp_feasible")
        or direct_gain > 0.0
    )
    has_direct_progress = bool(direct_progress_gain > 0.0)
    has_enabling_gain = bool(enabling_gain > 0.0)
    candidate["direct_clearance_candidate"] = has_direct_target_gain
    candidate["direct_progress_candidate"] = has_direct_progress
    candidate["enabling_clearance_candidate"] = has_enabling_gain
    candidate["exploratory"] = not (has_direct_target_gain or has_direct_progress or has_enabling_gain)
    candidate["automatic_execution_allowed"] = bool(
        candidate["task_effective"] and (has_direct_target_gain or has_direct_progress or has_enabling_gain)
    )
    utility = (3.0 * direct_gain) + (2.2 * direct_progress_gain) + (1.8 * enabling_gain) + (0.9 * free_space_gain)
    if candidate["exploratory"]:
        utility *= 0.35
    easiness = float(candidate.get("easiness_score", 0.0))
    risk = float(candidate.get("risk_score", 0.0))
    candidate["utility_score"] = round(utility, 6)
    candidate["score"] = round(utility + easiness - risk, 6)
    return refresh_executable_safe(candidate)


def _direct_target_gain(yaw_gain: dict) -> float:
    gain = float(yaw_gain.get("gain", 0.0))
    if yaw_gain.get("after_grasp_feasible"):
        gain += 1.0
    return gain


def _enabling_gain(
    obj: ObjectDict,
    target: ObjectDict,
    objects: List[ObjectDict],
    graph_item: dict,
    args: Any,
) -> float:
    gains = []
    target_id = str(target.get("id"))
    for blocked_id in graph_item.get("blocks", []):
        if str(blocked_id) == target_id:
            continue
        blocked_obj = _find_object(objects, blocked_id)
        if blocked_obj is None:
            continue
        yaw_gain = _target_yaw_gain(blocked_obj, objects, obj.get("id"), args)
        gain = float(yaw_gain.get("gain", 0.0))
        if yaw_gain.get("after_grasp_feasible"):
            gain += 1.0
        gains.append(gain)
    return max(gains) if gains else 0.0


def _evaluation_direct_target_gain(
    evaluation: dict,
    relation_target: ObjectDict,
    target: ObjectDict,
) -> float:
    if _object_id(relation_target) != _object_id(target):
        return 0.0
    return 1.0 if evaluation.get("post_push_grasp_feasible") else 0.0


def _evaluation_direct_progress_gain(
    evaluation: dict,
    relation_target: ObjectDict,
    target: ObjectDict,
) -> Tuple[float, Optional[str]]:
    if _object_id(relation_target) != _object_id(target):
        return 0.0, None
    blocker_count_reduction = max(0, int(evaluation.get("blocker_count_reduction") or 0))
    grasp_gain = max(0.0, float(evaluation.get("current_grasp_gain") or 0.0))
    distance_delta = max(0.0, float(evaluation.get("target_distance_delta_m") or 0.0))
    if blocker_count_reduction > 0:
        return 0.5 + 0.25 * blocker_count_reduction + grasp_gain + distance_delta, "target_blocker_count_reduction"
    if grasp_gain > 0.0:
        return grasp_gain, "target_grasp_clearance_gain"
    if distance_delta >= 0.005:
        return 0.35 + min(0.25, distance_delta * 5.0), "direct_blocker_moved_away_from_target"
    return 0.0, None


def _evaluation_enabling_gain(
    evaluation: dict,
    relation_target: ObjectDict,
    target: ObjectDict,
) -> Tuple[float, Optional[str]]:
    if _object_id(relation_target) == _object_id(target):
        return 0.0, None
    blocker_count_reduction = max(0, int(evaluation.get("blocker_count_reduction") or 0))
    grasp_gain = max(0.0, float(evaluation.get("current_grasp_gain") or 0.0))
    post_push_grasp_feasible = bool(evaluation.get("post_push_grasp_feasible"))
    if post_push_grasp_feasible:
        return 1.0 + grasp_gain + 0.25 * blocker_count_reduction, "post_push_blocker_grasp_feasible"
    if blocker_count_reduction > 0:
        return 0.5 + 0.25 * blocker_count_reduction + grasp_gain, "blocker_count_reduction"
    if grasp_gain > 0.0:
        return grasp_gain, "blocker_grasp_clearance_gain"
    return 0.0, None


def _candidate_safety_fields(
    candidate: dict,
    *,
    geometry_feasible: bool,
    protected_structure_safe: bool,
    moveit_feasible: bool = False,
) -> dict:
    candidate["geometry_feasible"] = bool(geometry_feasible)
    candidate["moveit_feasible"] = bool(moveit_feasible)
    candidate["protected_structure_safe"] = bool(protected_structure_safe)
    candidate.setdefault("feasible", bool(geometry_feasible))
    candidate.setdefault("approach_path_safe", bool(geometry_feasible))
    candidate.setdefault("push_swept_safe", bool(geometry_feasible))
    candidate.setdefault("push_end_safe", bool(geometry_feasible))
    return refresh_executable_safe(candidate)


def _frontier_item_priority(item: dict, target: ObjectDict, objects: List[ObjectDict]) -> Tuple[float, int, str]:
    obstacle = _find_object(objects, item.get("object_id")) or {}
    target_center = get_center(target)
    obstacle_center = get_center(obstacle)
    distance = 1.0
    if target_center is not None and obstacle_center is not None:
        distance = math.hypot(float(target_center[0]) - float(obstacle_center[0]), float(target_center[1]) - float(obstacle_center[1]))
    depth = max(1, int(item.get("min_depth", 99)))
    blocks_count = len(item.get("blocks", []) or [])
    return (
        float(depth) - 0.25 * float(blocks_count) + 0.5 * distance,
        depth,
        str(item.get("object_id")),
    )


def _rank_frontier_items(items: Iterable[dict], target: ObjectDict, objects: List[ObjectDict], limit: int) -> List[dict]:
    ranked = sorted(list(items or []), key=lambda item: _frontier_item_priority(item, target, objects))
    if int(limit) <= 0:
        return ranked
    return ranked[: int(limit)]


def _evaluation_priority(evaluation: dict) -> Tuple[int, int, int, float, str]:
    return (
        0 if evaluation.get("feasible") else 1,
        0 if evaluation.get("approach_path_safe") else 1,
        0 if evaluation.get("push_swept_safe") else 1,
        -float(evaluation.get("score", 0.0) or 0.0),
        str(evaluation.get("candidate_id")),
    )


def _make_pick_away_candidate(
    obj: ObjectDict,
    target: ObjectDict,
    objects: List[ObjectDict],
    protected_ids: Iterable[Any],
    graph_item: dict,
    args: Any,
    index: int,
    table_bounds: Optional[dict],
    future_place_regions: Iterable[dict] = (),
) -> Optional[dict]:
    grasp = _pick_away_grasp(obj, objects, args)
    if not grasp.get("grasp_feasible"):
        return None
    safe_place = safe_place_for_object(
        obj,
        target,
        objects,
        protected_ids,
        table_bounds,
        future_place_regions=future_place_regions,
    )
    if safe_place is None:
        return None
    yaw_gain = _target_yaw_gain(target, objects, obj.get("id"), args)
    direct_gain = _direct_target_gain(yaw_gain)
    enabling_gain = _enabling_gain(obj, target, objects, graph_item, args)
    free_space_gain = 0.5
    easiness = 1.0 + 0.05 * _feasible_yaw_count(grasp)
    risk = 0.1 * max(0, int(graph_item.get("min_depth", 1)) - 1)
    candidate = _score_candidate(
        _candidate_safety_fields(
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
                "direct_target_gain": round(direct_gain, 6),
                "enabling_gain": round(enabling_gain, 6),
                "free_space_gain": round(free_space_gain, 6),
                "easiness_score": round(easiness, 6),
                "risk_score": round(risk, 6),
                "target_yaw_gain": yaw_gain,
                "reason": "frontier_obstacle_graspable_pick_away_first",
                "grasp_policy": grasp.get("grasp_policy", "full_scene_grasp"),
                "relaxed_pick_away_grasp": bool(grasp.get("relaxed_pick_away_grasp")),
                "ignored_grasp_blockers": grasp.get("ignored_grasp_blockers", []),
                "requires_moveit_preflight": True,
            },
            geometry_feasible=True,
            protected_structure_safe=not _is_protected(obj, protected_ids),
        )
    )
    if candidate.get("relaxed_pick_away_grasp"):
        candidate["automatic_execution_allowed"] = False
        candidate["automatic_execution_reason"] = "relaxed_pick_away_requires_more_clearance"
        candidate["clearance_preflight_allowed"] = False
        refresh_executable_safe(candidate)
    return candidate


def _pick_away_grasp(obj: ObjectDict, objects: List[ObjectDict], args: Any) -> dict:
    grasp_args = {
        "gripper_outer_width_m": getattr(args, "grasp_gripper_outer_width_m", 0.112),
        "gripper_inner_width_m": getattr(args, "grasp_gripper_inner_width_m", 0.048),
        "approach_length_m": getattr(args, "grasp_approach_length_m", 0.02),
    }
    full_scene_grasp = select_best_grasp(obj, objects, **grasp_args)
    if full_scene_grasp.get("grasp_feasible"):
        full_scene_grasp["grasp_policy"] = "full_scene_grasp"
        return full_scene_grasp
    object_only_grasp = select_best_grasp(obj, [obj], **grasp_args)
    if not object_only_grasp.get("grasp_feasible"):
        return full_scene_grasp
    object_only_grasp["grasp_policy"] = "relaxed_top_pick_away_grasp"
    object_only_grasp["relaxed_pick_away_grasp"] = True
    object_only_grasp["ignored_grasp_blockers"] = [
        {"id": item.get("id"), "label": item.get("label")}
        for item in full_scene_grasp.get("blocking_objects", [])
    ]
    return object_only_grasp


def build_frontier_clearance_plan(
    current_state: dict,
    target: ObjectDict,
    protected_ids: Iterable[Any],
    args: Any,
    future_place_regions: Optional[Iterable[dict]] = None,
    evaluate_push_fn: Callable[..., dict] = evaluate_push_candidates,
) -> dict:
    objects = merge_duplicate_objects_3d(
        [obj for obj in current_state.get("objects", []) if isinstance(obj, dict)],
        preferred_id=target.get("id"),
    )
    target = _find_object(objects, target.get("id")) or target
    graph_state = copy.deepcopy(current_state)
    graph_state["objects"] = objects
    graph = build_obstruction_graph(
        graph_state,
        target,
        protected_ids,
        args,
        max_depth=getattr(args, "obstruction_graph_max_depth", 3),
    )
    all_candidates: List[dict] = []
    safe_candidates: List[dict] = []
    candidate_index = 1
    nudge_max = float(getattr(args, "clearance_nudge_distance_m", 0.025))
    frontier_top_k = int(getattr(args, "clearance_frontier_top_k", 6))
    candidate_top_n = int(getattr(args, "clearance_candidate_top_n_per_obstacle", 8))
    target_yaw_gain_cache: Dict[str, dict] = {}

    def target_yaw_gain_for(obstacle_id: Any) -> dict:
        key = str(obstacle_id)
        if key not in target_yaw_gain_cache:
            target_yaw_gain_cache[key] = _target_yaw_gain(target, objects, obstacle_id, args)
        return target_yaw_gain_cache[key]

    evaluated_frontier = _rank_frontier_items(
        graph.get("frontier", []),
        target,
        objects,
        frontier_top_k,
    )

    for frontier_item in evaluated_frontier:
        obstacle = _find_object(objects, frontier_item.get("object_id"))
        if obstacle is None or _object_id(obstacle) == _object_id(target) or _is_protected(obstacle, protected_ids):
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
            future_place_regions=future_place_regions or [],
        )
        if pick_candidate is not None:
            if (
                not pick_candidate.get("relaxed_pick_away_grasp")
                and pick_candidate.get("automatic_execution_allowed")
            ):
                pick_candidate["clearance_preflight_allowed"] = True
            refresh_executable_safe(pick_candidate)
            all_candidates.append(pick_candidate)
            if pick_candidate.get("clearance_preflight_allowed"):
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
        evaluation_items = []
        for result in push_report.get("candidate_results", []):
            for evaluation in result.get("evaluations", []) or []:
                evaluation_items.append((result, evaluation))
        evaluation_items.sort(key=lambda item: _evaluation_priority(item[1]))
        if candidate_top_n > 0:
            evaluation_items = evaluation_items[:candidate_top_n]
        for result, evaluation in evaluation_items:
                if float(evaluation.get("distance_m") or 0.0) > nudge_max + 1e-9:
                    continue
                yaw_gain = target_yaw_gain_for(obstacle.get("id"))
                direct_gain = _direct_target_gain(yaw_gain) + _evaluation_direct_target_gain(
                    evaluation,
                    relation_target,
                    target,
                )
                direct_progress_gain, direct_progress_reason = _evaluation_direct_progress_gain(
                    evaluation,
                    relation_target,
                    target,
                )
                eval_enabling_gain, eval_enabling_reason = _evaluation_enabling_gain(
                    evaluation,
                    relation_target,
                    target,
                )
                enabling_gain = eval_enabling_gain
                free_space_gain = 0.2 if float(evaluation.get("target_distance_after_m") or 0.0) > 0.0 else 0.0
                easiness = 0.4 + max(0.0, float(evaluation.get("score", 0.0)))
                risk = 0.4 + 0.2 * max(0, int(frontier_item.get("min_depth", 1)) - 1)
                enabling_reason = eval_enabling_reason
                candidate = _score_candidate(
                    _candidate_safety_fields(
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
                            "direct_target_gain": round(direct_gain, 6),
                            "direct_progress_gain": round(direct_progress_gain, 6),
                            "direct_progress_reason": direct_progress_reason,
                            "enabling_gain": round(enabling_gain, 6),
                            "free_space_gain": round(free_space_gain, 6),
                            "current_grasp_gain": evaluation.get("current_grasp_gain", 0.0),
                            "current_blocker_count": evaluation.get("current_blocker_count"),
                            "predicted_blocker_count": evaluation.get("predicted_blocker_count"),
                            "blocker_count_reduction": evaluation.get("blocker_count_reduction", 0),
                            "target_distance_before_m": evaluation.get("target_distance_before_m"),
                            "target_distance_after_m": evaluation.get("target_distance_after_m"),
                            "target_distance_delta_m": evaluation.get("target_distance_delta_m"),
                            "post_push_grasp_feasible": bool(evaluation.get("post_push_grasp_feasible")),
                            "enables_blocker_object_id": evaluation.get("enables_blocker_object_id")
                            or relation_target.get("id"),
                            "enabling_reason": enabling_reason,
                            "easiness_score": round(easiness, 6),
                            "risk_score": round(risk, 6),
                            "target_yaw_gain": yaw_gain,
                            "reason": evaluation.get("reason"),
                            "feasible": bool(evaluation.get("feasible")),
                            "approach_path_safe": bool(evaluation.get("approach_path_safe")),
                            "push_swept_safe": bool(evaluation.get("push_swept_safe")),
                            "push_end_safe": bool(evaluation.get("push_end_safe")),
                            "future_task_feasible": bool((evaluation.get("future_task_impact") or {}).get("feasible", True)),
                            "push_evaluation": evaluation,
                            "push_relation": relation,
                            "push_selected_result": result,
                            "requires_moveit_preflight": True,
                        },
                        geometry_feasible=bool(evaluation.get("feasible")),
                        protected_structure_safe=not _is_protected(obstacle, protected_ids),
                    )
                )
                candidate = annotate_nudge_preflight_policy(candidate, nudge_max)
                all_candidates.append(candidate)
                if candidate.get("clearance_preflight_allowed"):
                    safe_candidates.append(candidate)
                candidate_index += 1

    safe_candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("frontier_depth", 99)), str(item.get("candidate_id"))))
    all_candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), int(item.get("frontier_depth", 99)), str(item.get("candidate_id"))))
    executable_safe_candidates = [candidate for candidate in safe_candidates if candidate.get("executable_safe")]
    return {
        "schema_version": "obstruction_frontier_clearance_v1",
        "obstruction_graph": graph,
        "obstacle_frontier_candidates": graph.get("frontier", []),
        "evaluated_frontier_candidates": evaluated_frontier,
        "candidate_pruning": {
            "frontier_top_k": frontier_top_k,
            "candidate_top_n_per_obstacle": candidate_top_n,
            "frontier_count": len(graph.get("frontier", [])),
            "evaluated_frontier_count": len(evaluated_frontier),
        },
        "all_clearance_action_candidates": all_candidates,
        "preflight_clearance_candidates": safe_candidates,
        "safe_clearance_candidates": executable_safe_candidates,
        "selected_clearance_action": executable_safe_candidates[0] if executable_safe_candidates else None,
    }
