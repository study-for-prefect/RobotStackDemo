"""Push-direction evaluation, optional execution, and manual clearing flow."""

import copy
import os
from typing import Any, Dict, Tuple

from robot_scene_pipeline.geometry_relations import build_geometry_relations
from robot_scene_pipeline.scene_memory import (
    mark_pushed,
    save_memory,
    update_from_detections,
)
from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, load_json, push_clear_command, run
from .push_clearing import (
    build_push_execution_plan,
    evaluate_push_candidates,
    object_by_string_id,
    pushable_blocking_relations,
    relation_objects_with_protected_structure,
)
from .scene import memory_id_for_scene_object, reacquire_target


def _relations_for_target(
    current_state: dict,
    held_object: dict,
    base_id: Any,
    previous_locked_stack: dict,
) -> list:
    relation_objects = relation_objects_with_protected_structure(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
    )
    return build_geometry_relations(
        relation_objects,
        target_id=int(held_object["id"]),
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
    held_object["feasible_yaw_intervals_deg"] = analysis.get("feasible_yaw_intervals_deg", [])
    held_object["blocked_yaw_intervals_deg"] = analysis.get("blocked_yaw_intervals_deg", [])
    held_object["grasp_yaw_source"] = analysis.get("selected_grasp_source") or "adaptive_grasp_yaw_search"
    return held_object


def _raise_if_non_push_action(analysis: dict) -> None:
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
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_after_manual_clearing.json"),
        relations,
    )
    remaining_push = pushable_blocking_relations(
        observed_state,
        relations,
        held_object["id"],
        base_id,
        previous_locked_stack,
    )
    if remaining_push:
        raise RuntimeError(
            "Target remains blocked after one manual clearing observation."
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
) -> Tuple[dict, dict, dict]:
    relations = _relations_for_target(
        current_state,
        held_object,
        base_id,
        previous_locked_stack,
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_before_pick.json"),
        relations,
    )
    push_candidates = pushable_blocking_relations(
        current_state,
        relations,
        held_object["id"],
        base_id,
        previous_locked_stack,
    )
    analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_pick.json"),
        analysis,
    )
    _raise_if_non_push_action(analysis)
    if analysis.get("action") == "pick" and analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, analysis)
        return current_state, memory, held_object
    if not push_candidates:
        return current_state, memory, held_object

    direction_assessment = evaluate_push_candidates(
        current_state,
        held_object,
        push_candidates,
        distance_m=args.push_clearing_distance_m,
        table_bounds=current_state.get("table_bounds"),
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

    if selected_push is None:
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
        return current_state, memory, held_object

    runtime["current_stage"] = "push_clearing_before_pick"
    run(push_clear_command(args, push_execution_plan_path))
    push_execution_plan["execution_status"] = "executed"
    write_json(push_execution_plan_path, push_execution_plan)

    runtime["current_stage"] = "observation_after_push_clearing"
    pushed_state = capture_empty_observation(
        args,
        os.path.join(cycle_dir, "observation_after_push"),
        runtime["held_object_id"],
    )
    if pushed_state is None:
        raise RuntimeError(
            "Push clearing completed but no live observation was produced afterward."
        )
    current_state = pushed_state
    memory = update_from_detections(memory, current_state.get("objects", []))
    if obstacle_memory_id is not None:
        memory = mark_pushed(
            memory,
            obstacle_memory_id,
            selected_push["direction_base"],
            selected_push.get("distance_m", args.push_clearing_distance_m),
            reason=selected_push.get("reason") or "clear_obstacle",
            result="executed",
        )
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(current_state, held_template))
    relations_after_push = _relations_for_target(
        current_state,
        held_object,
        base_id,
        previous_locked_stack,
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_after_push.json"),
        relations_after_push,
    )
    remaining_push = pushable_blocking_relations(
        current_state,
        relations_after_push,
        held_object["id"],
        base_id,
        previous_locked_stack,
    )
    if remaining_push:
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
        )
    return current_state, memory, held_object
