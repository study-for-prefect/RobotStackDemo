"""Push-direction evaluation, optional execution, and manual clearing flow."""

import copy
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from robot_scene_pipeline.geometry_relations import build_geometry_relations
from robot_scene_pipeline.scene_memory import mark_pushed, save_memory, update_from_detections
from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, close_gripper_command, load_json, push_clear_command, run
from .obstruction_frontier import build_frontier_clearance_plan
from .observation_scope import observe_empty_with_scope
from .pick import build_offline_pick_plan, pick_command, place_command, plan_envelope
from .push_context import (
    observed_push_delta_m, protected_stack_templates,
)
from .push_clearing import (
    build_push_execution_plan, current_protected_structure_ids, evaluate_push_candidates, object_by_string_id,
    pushable_blocking_relations, relation_objects_with_protected_structure,
)
from .scene import memory_id_for_scene_object, reacquire_target
from .push_selection import choose_clearance_action
from .target_recovery import missing_target_clearance_relations, state_with_missing_target

def _relations_for_target(
    current_state: dict,
    held_object: dict,
    base_id: Any,
    previous_locked_stack: dict,
    args: Any,
    base_template: Optional[dict] = None,
) -> list:
    relation_objects = relation_objects_with_protected_structure(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    return build_geometry_relations(
        relation_objects,
        target_id=held_object["id"],
        gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
        gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
        grasp_approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
    )


def _target_grasp_analysis(relations: list, target_id: Any) -> dict:
    for relation in relations:
        if (
            relation.get("type") == "target_grasp_analysis"
            and str(relation.get("object")) == str(target_id)
        ):
            return relation
    return {}


def _apply_selected_grasp_to_target(held_object: dict, analysis: dict) -> dict:
    selected_yaw = analysis.get("selected_grasp_yaw_deg")
    if selected_yaw is None:
        return held_object
    held_object["selected_grasp_yaw_deg"] = float(selected_yaw)
    held_object["grasp_feasible"] = bool(analysis.get("grasp_feasible"))
    held_object["selected_grasp_axis_delta_deg"] = analysis.get("selected_grasp_axis_delta_deg")
    held_object["feasible_yaw_intervals_deg"] = analysis.get("feasible_yaw_intervals_deg", [])
    held_object["blocked_yaw_intervals_deg"] = analysis.get("blocked_yaw_intervals_deg", [])
    held_object["grasp_yaw_source"] = analysis.get("selected_grasp_source") or "adaptive_grasp_yaw_search"
    return held_object


def _raise_if_non_push_action(analysis: dict, has_push_candidates: bool = False) -> None:
    action = analysis.get("action")
    if action in (None, "pick", "push_clearing", "pick_away"):
        return
    if action in ("remove_top_object", "pick_away_top_object"):
        raise RuntimeError(
            "Target is under another object; do not solve this by rotating the gripper. "
            "Recommended action={} object_above_target={}.".format(
                action,
                analysis.get("object_above_target"),
            )
        )
    if action == "replan_required" and has_push_candidates:
        return
    if action == "replan_required":
        raise RuntimeError(
            "All grasp yaws are blocked by protected structure; push clearing is refused. "
            "blocked_by_base={} blocked_by_locked_structure={} blocked_by_placed_structure={}.".format(
                analysis.get("blocked_by_base"),
                analysis.get("blocked_by_locked_structure"),
                analysis.get("blocked_by_placed_structure"),
            )
        )


def _write_frontier_debug_files(cycle_dir: str, frontier_plan: dict) -> None:
    write_json(os.path.join(cycle_dir, "obstruction_graph.json"), frontier_plan.get("obstruction_graph", {}))
    write_json(
        os.path.join(cycle_dir, "obstacle_frontier_candidates.json"),
        frontier_plan.get("obstacle_frontier_candidates", []),
    )
    write_json(
        os.path.join(cycle_dir, "all_clearance_action_candidates.json"),
        frontier_plan.get("all_clearance_action_candidates", []),
    )
    write_json(
        os.path.join(cycle_dir, "safe_clearance_candidates.json"),
        frontier_plan.get("safe_clearance_candidates", []),
    )
    write_json(
        os.path.join(cycle_dir, "selected_clearance_action.json"),
        frontier_plan.get("selected_clearance_action") or {"action": "no_feasible_clearance_action"},
    )


def _build_pick_away_place_plan(current_state: dict, obstacle: dict, selected_action: dict, args: Any) -> dict:
    safe_place = selected_action.get("safe_place_center_m")
    if not isinstance(safe_place, list) or len(safe_place) < 3:
        raise RuntimeError("pick_away selected action is missing safe_place_center_m.")
    release_z = float(safe_place[2]) + float(getattr(args, "release_gap_m", 0.010))
    step = {
        "step": 1,
        "action": "place_relative",
        "object_id": obstacle.get("id"),
        "object_label": obstacle.get("label"),
        "reference_object_id": None,
        "reference_label": None,
        "relative_position": "frontier_safe_place",
        "reason": "Place cleared obstacle at a safe table location.",
        "status": "planned",
        "coordinate_frame": "base_frame",
        "coordinate_source": "obstruction_frontier_safe_place",
        "target_position_m": [round(float(safe_place[0]), 5), round(float(safe_place[1]), 5), round(release_z, 5)],
        "approach_position_m": [
            round(float(safe_place[0]), 5),
            round(float(safe_place[1]), 5),
            round(release_z + float(args.approach_height_m), 5),
        ],
        "target_yaw_deg": float(selected_action.get("selected_grasp_yaw_deg") or 0.0),
        "chosen_grasp_yaw_deg": float(selected_action.get("selected_grasp_yaw_deg") or 0.0),
        "target_yaw_valid": True,
        "exact_tool_yaw_required": True,
        "yaw_frame": "base_link",
        "yaw_source": "obstruction_frontier_pick_away",
    }
    return plan_envelope(current_state, step)


def _execute_pick_away_and_reobserve(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    selected_action: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict],
    step_index: int,
) -> Tuple[dict, dict, dict]:
    obstacle = object_by_string_id(current_state.get("objects", []), selected_action.get("obstacle_id"))
    obstacle_for_pick = copy.deepcopy(obstacle)
    obstacle_for_pick["selected_grasp_yaw_deg"] = selected_action.get("selected_grasp_yaw_deg")
    obstacle_for_pick["grasp_yaw_source"] = "obstruction_frontier_pick_away"
    pick_plan_path = os.path.join(cycle_dir, "pick_away_step_{:02d}_pick_plan.json".format(step_index))
    place_plan_path = os.path.join(cycle_dir, "pick_away_step_{:02d}_place_plan.json".format(step_index))
    build_offline_pick_plan(current_state, obstacle_for_pick, pick_plan_path, args)
    place_plan = _build_pick_away_place_plan(current_state, obstacle_for_pick, selected_action, args)
    write_json(place_plan_path, place_plan)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "pending_moveit_preflight",
            "action": "pick_away",
            "pick_plan": pick_plan_path,
            "place_plan": place_plan_path,
            "note": "MoveIt path planning is performed before real pick_away and place execution.",
        },
    )
    if not (args.execute and args.execute_push_clearing):
        write_json(
            os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
            {
                "action": "pick_away",
                "result": "dry_run_only",
                "selected_clearance_action": selected_action,
            },
        )
        return current_state, memory, held_template

    runtime["current_stage"] = "pick_away_obstacle_step_{:02d}".format(step_index)
    runtime["held_object_id"] = obstacle.get("id")
    run(pick_command(args, pick_plan_path))
    runtime["current_stage"] = "place_away_obstacle_step_{:02d}".format(step_index)
    run(place_command(args, place_plan_path))
    runtime["held_object_id"] = None
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "passed_and_executed",
            "action": "pick_away",
            "pick_plan": pick_plan_path,
            "place_plan": place_plan_path,
        },
    )
    runtime["current_stage"] = "scoped_observation_after_pick_away"
    observed_state, memory, post_observation = observe_empty_with_scope(
        args,
        os.path.join(cycle_dir, "observation_after_pick_away"),
        runtime,
        memory,
        critical_templates=(
            [held_template]
            + protected_stack_templates(
                current_state,
                base_id,
                previous_locked_stack,
                base_template=base_template,
            )
        ),
        noncritical_templates=[obstacle],
        scope_name="after_pick_away_clearance",
        description="Post-pick-away scoped observation",
    )
    if observed_state is None:
        raise RuntimeError("pick_away completed but no live observation was produced afterward.")
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(observed_state, held_template))
    return observed_state, memory, held_object


def _selected_push_from_clearance_action(selected_action: dict) -> dict:
    return {
        "type": "should_push_away",
        "subject": selected_action.get("obstacle_id"),
        "object": selected_action.get("blocks", [selected_action.get("target_object_id")])[0],
        "source": "obstruction_frontier",
        "reason": selected_action.get("reason"),
        "candidate_id": selected_action.get("candidate_id"),
        "direction_base": selected_action.get("direction_base"),
        "distance_m": selected_action.get("distance_m"),
        "direction_source": selected_action.get("direction_source"),
        "direction_score": selected_action.get("score"),
    }


def _execute_nudge_and_reobserve(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    held_object: dict,
    selected_action: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict],
    step_index: int,
) -> Tuple[dict, dict, dict]:
    selected_push = _selected_push_from_clearance_action(selected_action)
    obstacle = object_by_string_id(current_state.get("objects", []), selected_push["subject"])
    obstacle_memory_id = memory_id_for_scene_object(memory, obstacle)
    push_execution_plan = build_push_execution_plan(
        current_state,
        held_object,
        selected_push,
        args,
        direction_evaluations=[selected_action.get("push_evaluation", {})],
    )
    push_execution_plan["action_type"] = "nudge"
    push_execution_plan_path = os.path.join(cycle_dir, "push_execution_plan.json")
    write_json(push_execution_plan_path, push_execution_plan)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "pending_moveit_preflight" if (args.execute and args.execute_push_clearing) else "not_requested_dry_run",
            "action": "nudge",
            "push_plan": push_execution_plan_path,
            "gripper_policy": "close_before_push_use_as_rigid_paddle",
        },
    )
    if not (args.execute and args.execute_push_clearing):
        write_json(os.path.join(cycle_dir, "scene_state_after_action.json"), current_state)
        write_json(
            os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
            {
                "action": "nudge",
                "result": "dry_run_only",
                "selected_clearance_action": selected_action,
                "post_push_requirement": "reobserve_and_rerun_clearance_loop",
            },
        )
        return current_state, memory, held_object

    runtime["current_stage"] = "close_gripper_as_rigid_push_paddle_step_{:02d}".format(step_index)
    run(close_gripper_command(args))
    runtime["current_stage"] = "nudge_clearing_before_pick_step_{:02d}".format(step_index)
    run(push_clear_command(args, push_execution_plan_path))
    push_execution_plan["execution_status"] = "executed"
    write_json(push_execution_plan_path, push_execution_plan)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "passed_and_executed",
            "action": "nudge",
            "source": "moveit_plan_preview_push_preflight",
            "gripper_policy": "closed_rigid_paddle",
        },
    )
    runtime["current_stage"] = "scoped_observation_after_nudge_clearing"
    pushed_state, memory, post_observation = observe_empty_with_scope(
        args,
        os.path.join(cycle_dir, "observation_after_nudge"),
        runtime,
        memory,
        critical_templates=(
            [held_template]
            + protected_stack_templates(
                current_state,
                base_id,
                previous_locked_stack,
                base_template=base_template,
            )
        ),
        noncritical_templates=[obstacle],
        scope_name="after_nudge_clearing",
        description="Post-nudge scoped observation",
    )
    if pushed_state is None:
        raise RuntimeError("Nudge clearing completed but no live observation was produced afterward.")
    observed_delta_m = observed_push_delta_m(obstacle, pushed_state)
    if obstacle_memory_id is not None:
        memory = mark_pushed(
            memory,
            obstacle_memory_id,
            selected_push["direction_base"],
            selected_push.get("distance_m", args.push_clearing_distance_m),
            reason=selected_push.get("reason") or "frontier_nudge_clearance",
            result="success",
            observed_delta_m=observed_delta_m,
        )
    save_memory(memory, args.memory_json)
    try:
        held_object = copy.deepcopy(reacquire_target(pushed_state, held_template))
    except RuntimeError:
        pushed_state, held_object = state_with_missing_target(pushed_state, held_template)
    write_json(
        os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
        {
            "action": "nudge",
            "result": "executed_and_reobserved",
            "selected_clearance_action": selected_action,
            "post_observation": post_observation,
            "observed_delta_m": observed_delta_m,
        },
    )
    return pushed_state, memory, held_object


def _manual_clear_and_reobserve(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    held_template: dict,
    base_id: Any,
    previous_locked_stack: dict,
    reason: str,
    direction_assessment: dict,
    base_template: Optional[dict] = None,
    future_targets: Optional[Iterable[dict]] = None,
    future_place_regions: Optional[Iterable[dict]] = None,
    automatic_push_attempt: int = 0,
) -> Tuple[dict, dict, dict]:
    request_path = os.path.join(cycle_dir, "manual_clearance_required.json")
    write_json(
        request_path,
        {
            "schema_version": "manual_clearance_request_v1",
            "execution_status": "waiting_for_operator",
            "reason": reason,
            "target_object_id": held_template.get("id"),
            "target_label": held_template.get("label"),
            "direction_assessment": direction_assessment,
            "note": (
                "No safe automatic push direction is available. Clear the obstacle "
                "manually, then confirm so the scene can be observed again."
            ),
        },
    )
    print(
        "\nNo safe automatic push direction is available. "
        "Please clear the obstacle manually without moving the stack base.",
        flush=True,
    )
    try:
        input("After manual clearing is complete, press Enter to observe again...")
    except EOFError:
        raise RuntimeError(
            "Manual clearing requires interactive confirmation, but stdin is unavailable."
        )

    runtime["current_stage"] = "observation_after_manual_clearing"
    observed_state = capture_empty_observation(
        args,
        os.path.join(cycle_dir, "observation_after_manual_clearing"),
        runtime["held_object_id"],
    )
    if observed_state is None:
        raise RuntimeError("Manual clearing completed but no live observation was produced.")
    memory = update_from_detections(memory, observed_state.get("objects", []))
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(observed_state, held_template))
    relations = _relations_for_target(
        observed_state,
        held_object,
        base_id,
        previous_locked_stack,
        args,
        base_template=base_template,
    )
    write_json(os.path.join(cycle_dir, "geometry_relations_after_manual_clearing.json"), relations)
    protected_ids = current_protected_structure_ids(
        observed_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    remaining_push = pushable_blocking_relations(
        observed_state,
        relations,
        held_object["id"],
        base_id,
        previous_locked_stack,
        protected_object_ids=protected_ids,
    )
    after_analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(os.path.join(cycle_dir, "grasp_yaw_analysis_after_manual_clearing.json"), after_analysis)
    if after_analysis.get("action") == "pick" and after_analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, after_analysis)
    elif remaining_push:
        request = load_json(request_path)
        request["execution_status"] = "operator_confirmed_and_reobserved"
        request["post_manual_action"] = "rerun_automatic_push_planning"
        request["remaining_push_relations"] = remaining_push
        write_json(request_path, request)
        return handle_push_clearing_before_pick(
            args,
            cycle_dir,
            runtime,
            memory,
            observed_state,
            held_template,
            held_object,
            base_id,
            previous_locked_stack,
            base_template=base_template,
            future_targets=future_targets,
            future_place_regions=future_place_regions,
            automatic_push_attempt=automatic_push_attempt,
        )
    else:
        raise RuntimeError(
            "Manual clearing did not produce a feasible grasp. action={} base={} locked={} placed={}.".format(
                after_analysis.get("action"),
                after_analysis.get("blocked_by_base"),
                after_analysis.get("blocked_by_locked_structure"),
                after_analysis.get("blocked_by_placed_structure"),
            )
        )
    request = load_json(request_path)
    request["execution_status"] = "operator_confirmed_and_reobserved"
    write_json(request_path, request)
    return observed_state, memory, held_object


def handle_push_clearing_before_pick(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    held_object: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict] = None,
    future_targets: Optional[Iterable[dict]] = None,
    future_place_regions: Optional[Iterable[dict]] = None,
    automatic_push_attempt: int = 0,
) -> Tuple[dict, dict, dict]:
    step_index = int(automatic_push_attempt) + 1
    write_json(os.path.join(cycle_dir, "scene_state_before_action.json"), current_state)
    relations = _relations_for_target(
        current_state,
        held_object,
        base_id,
        previous_locked_stack,
        args,
        base_template=base_template,
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_before_pick.json"),
        relations,
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_before_action.json"),
        relations,
    )
    protected_ids = current_protected_structure_ids(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    push_candidates = pushable_blocking_relations(
        current_state,
        relations,
        held_object["id"],
        base_id,
        previous_locked_stack,
        protected_object_ids=protected_ids,
    )
    if held_object.get("target_detection_status") == "missing_in_current_observation":
        missing_candidates = missing_target_clearance_relations(
            current_state,
            held_object,
            protected_ids,
            radius_m=getattr(args, "missing_target_clearance_radius_m", 0.10),
            min_top_z_delta_m=getattr(args, "high_block_min_top_z_delta_m", 0.01),
        )
        seen_subjects = {str(item.get("subject")) for item in push_candidates}
        for candidate in missing_candidates:
            if str(candidate.get("subject")) in seen_subjects:
                continue
            push_candidates.append(candidate)
            seen_subjects.add(str(candidate.get("subject")))
    analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_pick.json"),
        analysis,
    )
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_action.json"),
        analysis,
    )
    _raise_if_non_push_action(analysis, has_push_candidates=True)
    if analysis.get("action") == "pick" and analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, analysis)
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "pick",
                "reason": "direct_or_rotated_grasp_yaw_feasible",
                "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
                "priority": "direct_pick_then_rotated_yaw_pick",
                "clearance_step_index": step_index,
                "selection_source": "direct_geometry_grasp",
                "multi_step_status": "target_graspable",
            },
        )
        write_json(
            os.path.join(cycle_dir, "multi_step_clearance_summary.json"),
            {
                "schema_version": "multi_step_clearance_summary_v1",
                "status": "target_graspable",
                "steps_completed": max(0, step_index - 1),
                "target_object_id": held_object.get("id"),
                "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
            },
        )
        return current_state, memory, held_object

    frontier_plan = build_frontier_clearance_plan(
        current_state,
        held_object,
        protected_ids,
        args,
        future_place_regions=future_place_regions or [],
        evaluate_push_fn=evaluate_push_candidates,
    )
    selected_clearance, llm_selection = choose_clearance_action(
        args,
        cycle_dir,
        held_object,
        frontier_plan.get("safe_clearance_candidates", []),
        memory,
        step_index,
    )
    frontier_plan["selected_clearance_action"] = selected_clearance
    frontier_plan["llm_selection"] = llm_selection
    _write_frontier_debug_files(cycle_dir, frontier_plan)

    if selected_clearance is None:
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "replan_required",
                "reason": "no_feasible_clearance_action",
                "grasp_action": analysis.get("action"),
                "clearance_step_index": step_index,
                "multi_step_status": "no_feasible_clearance_action",
            },
        )
        write_json(
            os.path.join(cycle_dir, "clearance_verification.json"),
            {
                "status": "not_requested",
                "reason": "no_feasible_clearance_action",
                "safe_candidate_count": 0,
            },
        )
        return current_state, memory, held_object

    write_json(
        os.path.join(cycle_dir, "selected_action.json"),
        {
            "action": selected_clearance.get("action"),
            "reason": selected_clearance.get("reason"),
            "selected_clearance_action": selected_clearance,
            "clearance_step_index": step_index,
            "selection_source": llm_selection.get("selection_source"),
            "multi_step_status": "selected_one_frontier_action",
            "post_action_requirement": "reobserve_and_rebuild_obstruction_graph",
        },
    )
    if selected_clearance.get("action_type") == "pick_away":
        next_state, memory, next_held = _execute_pick_away_and_reobserve(
            args,
            cycle_dir,
            runtime,
            memory,
            current_state,
            held_template,
            selected_clearance,
            base_id,
            previous_locked_stack,
            base_template,
            step_index,
        )
    else:
        next_state, memory, next_held = _execute_nudge_and_reobserve(
            args,
            cycle_dir,
            runtime,
            memory,
            current_state,
            held_template,
            held_object,
            selected_clearance,
            base_id,
            previous_locked_stack,
            base_template,
            step_index,
        )
    if not (args.execute and args.execute_push_clearing):
        return next_state, memory, next_held
    max_attempts = max(1, int(getattr(args, "max_automatic_push_clearing_attempts", 2)))
    if automatic_push_attempt + 1 >= max_attempts:
        write_json(
            os.path.join(cycle_dir, "multi_step_clearance_summary.json"),
            {
                "schema_version": "multi_step_clearance_summary_v1",
                "status": "multi_step_clearance_exhausted",
                "steps_completed": step_index,
                "max_attempts": max_attempts,
            },
        )
        return next_state, memory, next_held
    return handle_push_clearing_before_pick(
        args,
        cycle_dir,
        runtime,
        memory,
        next_state,
        held_template,
        next_held,
        base_id,
        previous_locked_stack,
        base_template=base_template,
        future_targets=future_targets,
        future_place_regions=future_place_regions,
        automatic_push_attempt=automatic_push_attempt + 1,
    )
