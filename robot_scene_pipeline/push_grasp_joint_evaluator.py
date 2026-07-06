"""Jointly evaluate push candidates by post-push grasp and future impact."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .future_task_impact import evaluate_future_task_impact
from .geometry_relations import is_locked, object_xy_aabb, xy_aabb_overlap, xy_distance
from .grasp_yaw_search import select_best_grasp
from .push_candidate_generation import build_joint_push_candidates, normalize_xy
from .push_direction_safety import direction_safety_defaults, push_end_collisions, table_bounds_ok
from .tool_swept_volume import check_tool_swept_volume


ObjectDict = Dict[str, Any]
CandidateDict = Dict[str, Any]


MIN_PUSH_DISTANCE_M = 0.025
MAX_PUSH_DISTANCE_M = 0.07
DEFAULT_SAFETY_MARGIN_M = 0.01
DEFAULT_GRIPPER_OUTER_WIDTH_M = 0.112
DEFAULT_GRIPPER_INNER_WIDTH_M = 0.048
DEFAULT_GRASP_APPROACH_LENGTH_M = 0.02
DEFAULT_PUSH_TOOL_WIDTH_M = 0.035
DEFAULT_PUSH_TOOL_SAFETY_MARGIN_M = 0.005
DEFAULT_PROGRESS_PUSH_DISTANCE_M = 0.025
DEFAULT_SOFT_TARGET_APPROACH_OVERLAP_M2 = 1e-6
DEFAULT_SOFT_TARGET_PUSH_OVERLAP_M2 = 3e-5


def _object_id(obj: ObjectDict) -> str:
    return str(obj.get("id"))


def _objects(scene: Dict[str, Any]) -> List[ObjectDict]:
    return [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]


def _find_object(objects: Iterable[ObjectDict], object_id: Any) -> Optional[ObjectDict]:
    for obj in objects:
        if _object_id(obj) == str(object_id):
            return obj
    return None


def _expanded_target_zone(target: ObjectDict, gripper_outer_width_m: float, safety_margin_m: float) -> Dict[str, float]:
    target_aabb = object_xy_aabb(target)
    if not target_aabb:
        return {}
    margin = 0.5 * float(gripper_outer_width_m) + float(safety_margin_m)
    return {
        "xmin": target_aabb["xmin"] - margin,
        "xmax": target_aabb["xmax"] + margin,
        "ymin": target_aabb["ymin"] - margin,
        "ymax": target_aabb["ymax"] + margin,
        "zmin": target_aabb["zmin"],
        "zmax": target_aabb["zmax"],
    }


def _translate_aabb(aabb: Dict[str, float], direction: List[float], distance_m: float) -> Dict[str, float]:
    dx = direction[0] * distance_m
    dy = direction[1] * distance_m
    return {
        "xmin": aabb["xmin"] + dx,
        "xmax": aabb["xmax"] + dx,
        "ymin": aabb["ymin"] + dy,
        "ymax": aabb["ymax"] + dy,
        "zmin": aabb["zmin"],
        "zmax": aabb["zmax"],
    }


def estimate_required_push_distance_m(
    obstacle: ObjectDict,
    target: ObjectDict,
    direction_base: Iterable[float],
    gripper_outer_width_m: float = DEFAULT_GRIPPER_OUTER_WIDTH_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    min_push_distance_m: float = MIN_PUSH_DISTANCE_M,
    max_push_distance_m: float = MAX_PUSH_DISTANCE_M,
) -> Dict[str, Any]:
    direction = normalize_xy(direction_base)
    obstacle_aabb = object_xy_aabb(obstacle)
    target_zone = _expanded_target_zone(target, gripper_outer_width_m, safety_margin_m)
    if direction is None or not obstacle_aabb or not target_zone:
        return {"feasible": False, "reason": "missing_geometry", "distance_m": None}
    if xy_aabb_overlap(obstacle_aabb, target_zone)[2] <= 0.0:
        return {"feasible": True, "reason": "minimum_push_distance", "distance_m": float(min_push_distance_m)}

    step = 0.005
    distance = float(min_push_distance_m)
    while distance <= float(max_push_distance_m) + 1e-9:
        moved = _translate_aabb(obstacle_aabb, direction, distance)
        if xy_aabb_overlap(moved, target_zone)[2] <= 0.0:
            return {
                "feasible": True,
                "reason": "estimated_clearance_distance",
                "distance_m": round(distance, 6),
            }
        distance += step
    return {"feasible": True, "reason": "required_push_distance_exceeds_max", "distance_m": round(float(max_push_distance_m), 6), "clamped_to_max": True}


def predict_scene_after_push(
    scene: Dict[str, Any],
    obstacle_id: Any,
    direction_base: Iterable[float],
    distance_m: float,
) -> Dict[str, Any]:
    predicted = copy.deepcopy(scene)
    direction = normalize_xy(direction_base)
    if direction is None:
        return predicted
    for obj in predicted.get("objects", []):
        if str(obj.get("id")) != str(obstacle_id):
            continue
        for key in ("geometry_center_m", "last_pose_base", "center_base_m", "center_3d_base_m"):
            center = obj.get(key)
            if isinstance(center, list) and len(center) >= 2:
                center[0] = float(center[0]) + direction[0] * float(distance_m)
                center[1] = float(center[1]) + direction[1] * float(distance_m)
        obj["predicted_push_delta_m"] = [direction[0] * float(distance_m), direction[1] * float(distance_m), 0.0]
        break
    return predicted


def _push_plan(target: ObjectDict, obstacle: ObjectDict, candidate: CandidateDict, lift_m: float, contact_z_offset_m: float) -> Dict[str, Any]:
    return {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "target_object_id": target.get("id"),
        "obstacle_object_id": obstacle.get("id"),
        "direction_base": candidate["direction_base"],
        "distance_m": candidate["distance_m"],
        "lift_m": lift_m,
        "contact_z_offset_m": contact_z_offset_m,
        "obstacle": obstacle,
    }


def _memory_direction_bonus(memory: Optional[dict], obstacle: ObjectDict, direction_base: Iterable[float]) -> Tuple[float, Optional[str]]:
    if not memory:
        return 0.0, None
    direction = normalize_xy(direction_base)
    if direction is None:
        return 0.0, None
    bonus = 0.0
    reason = None
    for action in memory.get("action_history", [])[-20:]:
        if action.get("action") != "push":
            continue
        hist_dir = normalize_xy(action.get("direction_base"))
        if hist_dir is None:
            continue
        dot = direction[0] * hist_dir[0] + direction[1] * hist_dir[1]
        if action.get("result") == "success" and dot > 0.95:
            bonus += 0.05
            reason = "similar_successful_push_direction"
        if action.get("result") == "failure" and dot > 0.95:
            bonus -= 0.20
            reason = "recent_failed_push_direction"
    return bonus, reason


def _candidate_output(candidate: CandidateDict, obstacle: ObjectDict) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": candidate.get("action", "push_away"),
        "obstacle_id": obstacle.get("id"),
        "direction_base": candidate.get("direction_base"),
        "distance_m": candidate.get("distance_m"),
        "feasible": False,
        "score": 0.0,
        "reason": "not_evaluated",
        "current_grasp_gain": 0.0,
        "current_blocker_count": None,
        "predicted_blocker_count": None,
        "blocker_count_reduction": 0,
        "post_push_grasp_feasible": False,
        "enables_blocker_object_id": None,
        "enabling_reason": None,
        "future_blocking_cost": 0.0,
        "place_blocking_cost": 0.0,
        "collision_risk": 0.0,
        "predicted_selected_grasp_yaw_deg": None,
        "source": candidate.get("source"),
        **direction_safety_defaults(),
    }


def _prepare_push_context(
    scene: Dict[str, Any],
    target: ObjectDict,
    obstacle: ObjectDict,
    candidate: CandidateDict,
    output: Dict[str, Any],
    table_bounds: Optional[dict],
    gripper_outer_width_m: float,
    gripper_inner_width_m: float,
    grasp_approach_length_m: float,
    safety_margin_m: float,
    push_tool_width_m: float,
    push_tool_safety_margin_m: float,
    lift_m: float,
    contact_z_offset_m: float,
    protected_object_ids: Iterable[Any] = (),
) -> Optional[Dict[str, Any]]:
    distance_report = estimate_required_push_distance_m(
        obstacle,
        target,
        candidate.get("direction_base", []),
        gripper_outer_width_m=gripper_outer_width_m,
        safety_margin_m=safety_margin_m,
    )
    output["required_distance"] = distance_report
    distance_m = max(MIN_PUSH_DISTANCE_M, min(MAX_PUSH_DISTANCE_M, float(candidate.get("distance_m") or distance_report["distance_m"])))
    candidate["distance_m"] = distance_m
    output["distance_m"] = distance_m
    current_grasp = select_best_grasp(
        target,
        _objects(scene),
        gripper_outer_width_m=gripper_outer_width_m,
        gripper_inner_width_m=gripper_inner_width_m,
        approach_length_m=grasp_approach_length_m,
    )
    predicted_scene = predict_scene_after_push(scene, obstacle.get("id"), candidate["direction_base"], distance_m)
    predicted_objects = _objects(predicted_scene)
    predicted_target = _find_object(predicted_objects, target.get("id")) or target
    predicted_obstacle = _find_object(predicted_objects, obstacle.get("id")) or obstacle
    table_ok, table_reason = table_bounds_ok(predicted_obstacle, table_bounds, edge_margin_m=0.02)
    if not table_ok:
        output["push_end_safe"] = False
        output["reason"] = table_reason
        return None
    end_collisions = push_end_collisions(predicted_obstacle, predicted_objects)
    output["push_end_collisions"] = end_collisions
    output["push_end_safe"] = not end_collisions
    if end_collisions:
        protected_ids = {str(value) for value in protected_object_ids or []}
        soft_end_collision = (
            float(distance_m) <= DEFAULT_PROGRESS_PUSH_DISTANCE_M + 1e-9
            and all(
                str(item.get("id")) != str(target.get("id"))
                and str(item.get("id")) not in protected_ids
                and not is_locked(_find_object(predicted_objects, item.get("id")) or {})
                for item in end_collisions
            )
        )
        output["push_end_soft_collision"] = bool(soft_end_collision)
        if not soft_end_collision:
            output["reason"] = "push_end_collision"
            return None
    push_plan = _push_plan(target, obstacle, candidate, lift_m=lift_m, contact_z_offset_m=contact_z_offset_m)
    swept = check_tool_swept_volume(
        push_plan,
        _objects(scene),
        ignore_object_ids=[obstacle.get("id")],
        gripper_outer_width_m=push_tool_width_m,
        safety_margin_m=push_tool_safety_margin_m,
    )
    output["tool_swept_volume"] = swept
    approach_collisions = [
        item for item in swept.get("collisions", [])
        if item.get("stage") in ("pre_push_pose", "vertical_approach", "contact_pose")
    ]
    push_collisions = [
        item for item in swept.get("collisions", [])
        if item.get("stage") in ("horizontal_push", "retreat")
    ]
    output["approach_path_safe"] = not approach_collisions
    output["push_swept_safe"] = not push_collisions
    if not swept["feasible"]:
        soft_swept_collision = _is_soft_swept_collision(
            swept.get("collisions", []),
            _objects(scene),
            target,
            protected_object_ids,
            distance_m,
        )
        output["tool_swept_soft_collision"] = bool(soft_swept_collision)
        if not soft_swept_collision:
            output["collision_risk"] = 1.0
            output["reason"] = swept["reason"]
            return None
    return {
        "current_grasp": current_grasp,
        "predicted_scene": predicted_scene,
        "predicted_objects": predicted_objects,
        "predicted_target": predicted_target,
        "predicted_obstacle": predicted_obstacle,
    }


def _mark_feasible_candidate(
    output: Dict[str, Any],
    predicted_grasp: Dict[str, Any],
    current_grasp: Dict[str, Any],
    future: Dict[str, Any],
    memory: Optional[dict],
    obstacle: ObjectDict,
    candidate: CandidateDict,
    predicted_target: ObjectDict,
    predicted_obstacle: ObjectDict,
) -> None:
    predicted_clearances = [
        float(item.get("clearance_m", 0.0))
        for item in predicted_grasp.get("candidate_results", [])
        if item.get("feasible")
    ]
    clearance_after = max(predicted_clearances) if predicted_clearances else 0.0
    current_gain = max(0.0, clearance_after - float(current_grasp.get("selected_clearance_m") or 0.0))
    history_bonus, history_reason = _memory_direction_bonus(memory, obstacle, candidate["direction_base"])
    score = (
        1.0
        + current_gain
        + history_bonus
        + 0.02 * float(candidate.get("qwen_confidence", 0.0))
        - 0.05 * output["future_blocking_cost"]
        - 0.05 * output["place_blocking_cost"]
        - 0.02 * float(future.get("congestion_cost", 0.0))
        - 0.5 * float(candidate["distance_m"])
    )
    output.update(
        {
            "feasible": True,
            "score": round(score, 6),
            "reason": "push_enables_current_grasp_and_preserves_future_tasks",
            "current_grasp_gain": round(current_gain, 6),
            "enables_blocker_object_id": predicted_target.get("id"),
            "enabling_reason": "post_push_grasp_feasible",
            "history_reason": history_reason,
            "target_distance_after_m": round(xy_distance(predicted_obstacle, predicted_target), 6),
        }
    )


def _mark_progress_candidate(
    output: Dict[str, Any],
    predicted_grasp: Dict[str, Any],
    current_grasp: Dict[str, Any],
    future: Dict[str, Any],
    candidate: CandidateDict,
    predicted_target: ObjectDict,
    predicted_obstacle: ObjectDict,
) -> None:
    before = _blocker_count(current_grasp)
    after = _blocker_count(predicted_grasp)
    progress = max(0, before - after)
    score = (
        0.45
        + 0.15 * progress
        - 0.05 * output["future_blocking_cost"]
        - 0.05 * output["place_blocking_cost"]
        - 0.02 * float(future.get("congestion_cost", 0.0))
        - 0.5 * float(candidate["distance_m"])
    )
    output.update(
        {
            "feasible": True,
            "score": round(score, 6),
            "reason": "push_reduces_current_blockers_and_preserves_future_tasks",
            "current_blocker_count": before,
            "predicted_blocker_count": after,
            "blocker_count_reduction": progress,
            "enables_blocker_object_id": predicted_target.get("id"),
            "enabling_reason": "blocker_count_reduction",
            "target_distance_after_m": round(xy_distance(predicted_obstacle, predicted_target), 6),
        }
    )


def _blocker_count(grasp: Dict[str, Any]) -> int:
    blockers = {
        str(item.get("id"))
        for item in grasp.get("blocking_objects", [])
        if item.get("id") is not None
    }
    return len(blockers)


def _is_loose_movable_obstacle(obstacle: ObjectDict) -> bool:
    role = obstacle.get("role")
    state = obstacle.get("state")
    if role in ("base", "structure") or state in ("locked", "placed"):
        return False
    if obstacle.get("pushable") is False or is_locked(obstacle):
        return False
    if role is None and state is None:
        return True
    return role == "loose_movable" and state not in ("locked", "placed")


def _is_soft_swept_collision(
    collisions: Iterable[Dict[str, Any]],
    objects: Iterable[ObjectDict],
    target: ObjectDict,
    protected_ids: Iterable[str],
    distance_m: float,
) -> bool:
    if float(distance_m) > DEFAULT_PROGRESS_PUSH_DISTANCE_M + 1e-9:
        return False
    object_list = list(objects)
    protected = {str(value) for value in protected_ids or []}
    target_id = str(target.get("id"))
    for collision in collisions or []:
        object_id = str(collision.get("id"))
        obj = _find_object(object_list, object_id) or {}
        if object_id in protected or is_locked(obj):
            return False
        if object_id != target_id:
            if not _is_loose_movable_obstacle(obj):
                return False
            continue
        overlap_area = float(collision.get("overlap_area_m2") or 0.0)
        stage = str(collision.get("stage") or "")
        if stage in ("pre_push_pose", "vertical_approach", "contact_pose"):
            if overlap_area > DEFAULT_SOFT_TARGET_APPROACH_OVERLAP_M2:
                return False
        elif stage in ("horizontal_push", "retreat"):
            if overlap_area > DEFAULT_SOFT_TARGET_PUSH_OVERLAP_M2:
                return False
        else:
            return False
    return True


def _with_short_progress_variants(candidates: Iterable[CandidateDict]) -> List[CandidateDict]:
    output: List[CandidateDict] = []
    for candidate in candidates:
        output.append(candidate)
        if candidate.get("distance_m") is not None:
            continue
        short_candidate = dict(candidate)
        short_candidate["distance_m"] = DEFAULT_PROGRESS_PUSH_DISTANCE_M
        short_candidate["source"] = "{}.short_progress".format(candidate.get("source", "rule"))
        output.append(short_candidate)
    return output


def _assign_candidate_ids(candidates: Iterable[CandidateDict]) -> List[CandidateDict]:
    output: List[CandidateDict] = []
    for index, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        if item.get("candidate_id") is None:
            item["candidate_id"] = "push_{:03d}_obj_{}".format(index, item.get("obstacle_id"))
        output.append(item)
    return output


def evaluate_one_push_grasp_candidate(
    scene: Dict[str, Any], target: ObjectDict, obstacle: ObjectDict, candidate: CandidateDict,
    future_targets: Iterable[ObjectDict] = (),
    future_place_regions: Iterable[Dict[str, Any]] = (),
    protected_objects: Iterable[ObjectDict] = (),
    memory: Optional[dict] = None, table_bounds: Optional[dict] = None,
    gripper_outer_width_m: float = DEFAULT_GRIPPER_OUTER_WIDTH_M,
    gripper_inner_width_m: float = DEFAULT_GRIPPER_INNER_WIDTH_M,
    grasp_approach_length_m: float = DEFAULT_GRASP_APPROACH_LENGTH_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    push_tool_width_m: float = DEFAULT_PUSH_TOOL_WIDTH_M,
    push_tool_safety_margin_m: float = DEFAULT_PUSH_TOOL_SAFETY_MARGIN_M,
    lift_m: float = 0.05,
    contact_z_offset_m: float = 0.015,
    protect_future_targets: bool = False,
) -> Dict[str, Any]:
    output = _candidate_output(candidate, obstacle)
    protected_ids = {str(obj.get("id")) for obj in protected_objects or []}
    loose_movable = _is_loose_movable_obstacle(obstacle)
    output["pushed_object_loose_movable"] = bool(loose_movable)
    if (
        not loose_movable
        or str(obstacle.get("id")) in protected_ids
    ):
        output["reason"] = "pushed_object_protected_or_not_pushable"
        return output

    context = _prepare_push_context(
        scene,
        target,
        obstacle,
        candidate,
        output,
        table_bounds,
        gripper_outer_width_m=gripper_outer_width_m,
        gripper_inner_width_m=gripper_inner_width_m,
        grasp_approach_length_m=grasp_approach_length_m,
        safety_margin_m=safety_margin_m,
        push_tool_width_m=push_tool_width_m,
        push_tool_safety_margin_m=push_tool_safety_margin_m,
        lift_m=lift_m,
        contact_z_offset_m=contact_z_offset_m,
        protected_object_ids=protected_ids,
    )
    if context is None:
        return output

    predicted_grasp = select_best_grasp(
        context["predicted_target"],
        context["predicted_objects"],
        gripper_outer_width_m=gripper_outer_width_m,
        gripper_inner_width_m=gripper_inner_width_m,
        approach_length_m=grasp_approach_length_m,
    )
    output["predicted_selected_grasp_yaw_deg"] = predicted_grasp.get("selected_grasp_yaw_deg")
    output["predicted_grasp"] = predicted_grasp
    output["post_push_grasp_feasible"] = bool(predicted_grasp.get("grasp_feasible"))

    future = evaluate_future_task_impact(
        context["predicted_scene"],
        current_target=target,
        future_targets=future_targets if protect_future_targets else [],
        future_place_regions=future_place_regions,
        protected_objects=protected_objects,
        moved_object_id=obstacle.get("id"),
        table_bounds=table_bounds,
        gripper_outer_width_m=gripper_outer_width_m,
        gripper_inner_width_m=gripper_inner_width_m,
    )
    output["future_task_impact"] = future
    output["future_blocking_cost"] = float(future.get("future_blocking_cost", 0.0))
    output["place_blocking_cost"] = float(future.get("place_blocking_cost", 0.0))
    if not future["feasible"]:
        output["reason"] = future["reason"]
        return output

    if not predicted_grasp["grasp_feasible"]:
        output["current_blocker_count"] = _blocker_count(context["current_grasp"])
        output["predicted_blocker_count"] = _blocker_count(predicted_grasp)
        if output["predicted_blocker_count"] < output["current_blocker_count"]:
            _mark_progress_candidate(
                output,
                predicted_grasp,
                context["current_grasp"],
                future,
                candidate,
                context["predicted_target"],
                context["predicted_obstacle"],
            )
            return output
        output["reason"] = "push_after_current_target_still_blocked"
        return output

    _mark_feasible_candidate(
        output,
        predicted_grasp,
        context["current_grasp"],
        future,
        memory,
        obstacle,
        candidate,
        context["predicted_target"],
        context["predicted_obstacle"],
    )
    return output


def evaluate_push_grasp_joint_candidates(
    scene: Dict[str, Any],
    target: ObjectDict,
    obstacle_ids: Iterable[Any],
    qwen_candidates: Iterable[dict] = (),
    future_targets: Iterable[ObjectDict] = (),
    future_place_regions: Iterable[Dict[str, Any]] = (),
    protected_objects: Iterable[ObjectDict] = (),
    memory: Optional[dict] = None,
    table_bounds: Optional[dict] = None,
    gripper_outer_width_m: float = DEFAULT_GRIPPER_OUTER_WIDTH_M,
    gripper_inner_width_m: float = DEFAULT_GRIPPER_INNER_WIDTH_M,
    grasp_approach_length_m: float = DEFAULT_GRASP_APPROACH_LENGTH_M,
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M,
    push_tool_width_m: float = DEFAULT_PUSH_TOOL_WIDTH_M,
    push_tool_safety_margin_m: float = DEFAULT_PUSH_TOOL_SAFETY_MARGIN_M,
    lift_m: float = 0.05,
    contact_z_offset_m: float = 0.015,
    protect_future_targets: bool = False,
) -> Dict[str, Any]:
    objects = _objects(scene)
    candidates = _assign_candidate_ids(
        _with_short_progress_variants(
            build_joint_push_candidates(objects, target, obstacle_ids, qwen_candidates=qwen_candidates)
        )
    )
    evaluations: List[Dict[str, Any]] = []
    for candidate in candidates:
        obstacle = _find_object(objects, candidate.get("obstacle_id"))
        if obstacle is None:
            continue
        evaluations.append(
            evaluate_one_push_grasp_candidate(
                scene,
                target,
                obstacle,
                candidate,
                future_targets=future_targets,
                future_place_regions=future_place_regions,
                protected_objects=protected_objects,
                memory=memory,
                table_bounds=table_bounds,
                gripper_outer_width_m=gripper_outer_width_m,
                gripper_inner_width_m=gripper_inner_width_m,
                grasp_approach_length_m=grasp_approach_length_m,
                safety_margin_m=safety_margin_m,
                push_tool_width_m=push_tool_width_m,
                push_tool_safety_margin_m=push_tool_safety_margin_m,
                lift_m=lift_m,
                contact_z_offset_m=contact_z_offset_m,
                protect_future_targets=protect_future_targets,
            )
        )
    feasible = [item for item in evaluations if item.get("feasible")]
    feasible.sort(key=lambda item: (-float(item.get("score", 0.0)), float(item.get("distance_m") or 0.0)))
    return {
        "schema_version": "push_grasp_joint_candidates_v1",
        "target_object_id": target.get("id"),
        "all_grasps_blocked": True,
        "candidates": evaluations,
        "selected_candidate": feasible[0] if feasible else None,
    }
