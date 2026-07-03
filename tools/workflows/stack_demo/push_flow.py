"""Push-direction evaluation, optional execution, and manual clearing flow."""

import copy
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from robot_scene_pipeline.geometry_relations import build_geometry_relations
from robot_scene_pipeline.qwen_action_candidate_parser import parse_qwen_action_candidates
from robot_scene_pipeline.scene_memory import mark_pushed, save_memory, update_from_detections
from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, load_json, push_clear_command, run
from .observation_scope import observe_empty_with_scope
from .push_context import (
    observed_push_delta_m, protected_objects, protected_stack_templates, qwen_forbidden_objects,
)
from .push_clearing import (
    build_push_execution_plan, current_protected_structure_ids, evaluate_push_candidates, object_by_string_id,
    pushable_blocking_relations, relation_objects_with_protected_structure,
)
from .scene import memory_id_for_scene_object, reacquire_target

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
    analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_pick.json"),
        analysis,
    )
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_action.json"),
        analysis,
    )
    _raise_if_non_push_action(analysis, has_push_candidates=bool(push_candidates))
    if analysis.get("action") == "pick" and analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, analysis)
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "pick",
                "reason": "direct_or_rotated_grasp_yaw_feasible",
                "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
                "priority": "direct_pick_then_rotated_yaw_pick",
            },
        )
        return current_state, memory, held_object
    if not push_candidates:
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "reobserve",
                "reason": "no_pushable_blocking_relation",
                "grasp_action": analysis.get("action"),
            },
        )
        return current_state, memory, held_object

    qwen_path = os.path.join(cycle_dir, "qwen_candidate_actions.json")
    qwen_report = parse_qwen_action_candidates(qwen_path)
    if os.path.exists(qwen_path):
        write_json(os.path.join(cycle_dir, "qwen_candidate_actions_parsed.json"), qwen_report)
    else:
        write_json(qwen_path, qwen_report)
    protected = protected_objects(
        current_state,
        base_id,
        previous_locked_stack,
        protected_object_ids=protected_ids,
    )
    protected.extend(qwen_forbidden_objects(current_state, qwen_report))
    direction_assessment = evaluate_push_candidates(
        current_state,
        held_object,
        push_candidates,
        distance_m=args.push_clearing_distance_m,
        table_bounds=current_state.get("table_bounds"),
        qwen_candidates=qwen_report.get("candidate_actions", []),
        future_targets=[],
        future_place_regions=future_place_regions or [],
        protected_objects=protected,
        memory=memory,
        lift_m=args.push_clearing_lift_m,
        contact_z_offset_m=args.push_clearing_contact_z_offset_m,
        gripper_outer_width_m=args.grasp_gripper_outer_width_m,
        gripper_inner_width_m=args.grasp_gripper_inner_width_m,
        grasp_approach_length_m=args.grasp_approach_length_m,
        push_tool_width_m=args.push_tool_width_m,
        push_tool_safety_margin_m=args.push_tool_safety_margin_m,
    )
    write_json(
        os.path.join(cycle_dir, "push_grasp_joint_candidates.json"),
        direction_assessment.get("joint_evaluation", {}),
    )
    selected_result = direction_assessment["selected_result"]
    selected_push = None
    if selected_result is not None:
        selected_push = dict(selected_result["relation"])
        selected_direction = selected_result["selected_direction"]
        selected_push["direction_base"] = selected_direction["direction_base"]
        selected_push["distance_m"] = selected_direction["distance_m"]
        selected_push["direction_source"] = selected_direction["source"]
        selected_push["direction_score"] = selected_direction["score"]

    write_json(
        os.path.join(cycle_dir, "push_plan_before_pick.json"),
        {
            "schema_version": "push_plan_v1",
            "execution_status": "dry_run_only",
            "target_object_id": held_object.get("id"),
            "target_label": held_object.get("label"),
            "push_candidates": push_candidates,
            "selected_push": selected_push,
            "direction_assessment": direction_assessment,
            "note": (
                "Geometry relation suggests clearing obstacle before pick. "
                "Automatic execution requires one feasible evaluated direction."
            ),
        },
    )
    rejected = [
        {
            "obstacle_id": item.get("obstacle_id"),
            "direction_base": item.get("direction_base"),
            "reason": item.get("reason"),
            "score": item.get("score"),
        }
        for item in direction_assessment.get("joint_evaluation", {}).get("candidates", [])
        if not item.get("feasible")
    ]

    if selected_push is None:
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "replan_required",
                "reason": "all_push_directions_unsafe" if push_candidates else "no_pushable_blocking_relation",
                "rejected_candidates": rejected,
            },
        )
        write_json(
            os.path.join(cycle_dir, "moveit_verification.json"),
            {"status": "not_requested", "reason": "no_feasible_joint_push_candidate"},
        )
        write_json(os.path.join(cycle_dir, "scene_state_after_action.json"), current_state)
        write_json(
            os.path.join(cycle_dir, "action_result.json"),
            {
                "action": "replan_required",
                "result": "not_selected",
                "reason": "all_push_directions_unsafe" if push_candidates else "no_pushable_blocking_relation",
            },
        )
        write_json(
            os.path.join(cycle_dir, "manual_clearance_required.json"),
            {
                "schema_version": "manual_clearance_request_v1",
                "execution_status": "dry_run_only",
                "reason": "no_feasible_automatic_push_direction",
                "target_object_id": held_object.get("id"),
                "target_label": held_object.get("label"),
                "direction_assessment": direction_assessment,
            },
        )
        print(
            "No feasible automatic push direction; manual clearing is required.",
            flush=True,
        )
        if args.execute and args.execute_push_clearing:
            return _manual_clear_and_reobserve(
                args,
                cycle_dir,
                runtime,
                memory,
                held_template,
                base_id,
                previous_locked_stack,
                "no_feasible_automatic_push_direction",
                direction_assessment,
                base_template=base_template,
                future_targets=future_targets,
                future_place_regions=future_place_regions,
                automatic_push_attempt=automatic_push_attempt,
            )
        return current_state, memory, held_object

    obstacle = object_by_string_id(
        current_state.get("objects", []),
        selected_push["subject"],
    )
    obstacle_memory_id = memory_id_for_scene_object(memory, obstacle)
    push_execution_plan = build_push_execution_plan(
        current_state,
        held_object,
        selected_push,
        args,
        direction_evaluations=selected_result["evaluations"],
    )
    push_execution_plan_path = os.path.join(cycle_dir, "push_execution_plan.json")
    write_json(push_execution_plan_path, push_execution_plan)
    write_json(
        os.path.join(cycle_dir, "selected_action.json"),
        {
            "action": "short_safe_push",
            "reason": selected_push.get("reason"),
            "selected_push": selected_push,
            "rejected_candidates": rejected,
            "post_push_requirement": "reobserve_and_rerun_grasp_yaw_search",
        },
    )
    write_json(
        os.path.join(cycle_dir, "moveit_verification.json"),
        {
            "status": "pending_execute_mode" if (args.execute and args.execute_push_clearing) else "not_requested_dry_run",
            "reason": "MoveIt verification is performed by tools/robot/moveit_plan_preview.py before execution.",
            "max_pre_rotate_joint_delta": getattr(args, "max_pre_rotate_joint_delta", None),
        },
    )
    print(
        "Geometry relation suggests push before pick: "
        "obstacle={} target={} direction={} source={} distance={}".format(
            selected_push.get("subject"),
            selected_push.get("object"),
            selected_push.get("direction_base"),
            selected_push.get("direction_source"),
            selected_push.get("distance_m"),
        ),
        flush=True,
    )
    if not (args.execute and args.execute_push_clearing):
        write_json(os.path.join(cycle_dir, "scene_state_after_action.json"), current_state)
        write_json(
            os.path.join(cycle_dir, "action_result.json"),
            {"action": "push_clearing", "result": "dry_run_only", "selected_push": selected_push},
        )
        return current_state, memory, held_object

    runtime["current_stage"] = "push_clearing_before_pick"
    run(push_clear_command(args, push_execution_plan_path))
    push_execution_plan["execution_status"] = "executed"
    write_json(push_execution_plan_path, push_execution_plan)
    write_json(
        os.path.join(cycle_dir, "moveit_verification.json"),
        {"status": "passed_and_executed", "source": "moveit_plan_preview_push_preflight"},
    )

    runtime["current_stage"] = "scoped_observation_after_push_clearing"
    pushed_state, memory, post_push_observation = observe_empty_with_scope(
        args,
        os.path.join(cycle_dir, "observation_after_push"),
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
        scope_name="after_push_clearing",
        description="Post-push scoped observation",
    )
    if pushed_state is None:
        raise RuntimeError(
            "Push clearing completed but no live observation was produced afterward."
        )
    current_state = pushed_state
    write_json(os.path.join(cycle_dir, "scene_state_after_action.json"), current_state)
    observed_delta_m = observed_push_delta_m(obstacle, current_state)
    if obstacle_memory_id is not None:
        memory = mark_pushed(
            memory,
            obstacle_memory_id,
            selected_push["direction_base"],
            selected_push.get("distance_m", args.push_clearing_distance_m),
            reason=selected_push.get("reason") or "clear_obstacle",
            result="success",
            observed_delta_m=observed_delta_m,
        )
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(current_state, held_template))
    relations_after_push = _relations_for_target(
        current_state,
        held_object,
        base_id,
        previous_locked_stack,
        args,
        base_template=base_template,
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_after_push.json"),
        relations_after_push,
    )
    after_analysis = _target_grasp_analysis(relations_after_push, held_object["id"])
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_after_push.json"),
        after_analysis,
    )
    write_json(
        os.path.join(cycle_dir, "action_result.json"),
        {
            "action": "push_clearing",
            "result": "executed_and_reobserved",
            "post_push_observation": post_push_observation,
            "observed_delta_m": observed_delta_m,
            "post_push_grasp_action": after_analysis.get("action"),
            "post_push_selected_grasp_yaw_deg": after_analysis.get("selected_grasp_yaw_deg"),
        },
    )
    protected_ids_after_push = current_protected_structure_ids(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    remaining_push = pushable_blocking_relations(
        current_state,
        relations_after_push,
        held_object["id"],
        base_id,
        previous_locked_stack,
        protected_object_ids=protected_ids_after_push,
    )
    if after_analysis.get("action") == "pick" and after_analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, after_analysis)
    if remaining_push:
        max_attempts = max(1, int(getattr(args, "max_automatic_push_clearing_attempts", 2)))
        if automatic_push_attempt + 1 < max_attempts:
            write_json(
                os.path.join(cycle_dir, "automatic_push_replan_{:02d}.json".format(automatic_push_attempt + 1)),
                {
                    "reason": "target_still_blocked_after_automatic_push",
                    "attempt_completed": automatic_push_attempt + 1,
                    "max_attempts": max_attempts,
                    "remaining_push_relations": remaining_push,
                },
            )
            return handle_push_clearing_before_pick(
                args,
                cycle_dir,
                runtime,
                memory,
                current_state,
                held_template,
                held_object,
                base_id,
                previous_locked_stack,
                base_template=base_template,
                future_targets=future_targets,
                future_place_regions=future_place_regions,
                automatic_push_attempt=automatic_push_attempt + 1,
            )
        return _manual_clear_and_reobserve(
            args,
            cycle_dir,
            runtime,
            memory,
            held_template,
            base_id,
            previous_locked_stack,
            "target_still_blocked_after_one_automatic_push",
            {"remaining_push_relations": remaining_push},
            base_template=base_template,
            future_targets=future_targets,
            future_place_regions=future_place_regions,
            automatic_push_attempt=automatic_push_attempt + 1,
        )
    return current_state, memory, held_object
