"""Top-level orchestration for the closed-loop stack demo."""

import copy
import os
import sys

from robot_scene_pipeline.scene_memory import (
    load_memory,
    mark_placed,
    save_memory,
    set_base,
    set_role,
    update_from_detections,
)
from robot_scene_pipeline.stack_state import verify_stack_growth
from robot_scene_pipeline.xy_correction import load_xy_correction
from robot_scene_pipeline.vlm_replanning import advance_scene_revision
from robot_scene_pipeline.ollama_policy_client import unload_model, warm_model
from robot_scene_pipeline.task_routing import route_task_type
from tools.planning.decision_to_execution import write_json
from tools.workflows.two_stage_visual_pick import build_tcp_error_corrected_plan

from .arguments import parse_args
from .commands import (
    capture_empty_observation,
    failure_state_path,
    init_ready_pose,
    load_json,
    plan_only_command,
    retry_close_observation,
    run,
)
from .observation_scope import expected_placed_template
from .pick import (
    build_offline_pick_plan,
    pick_approach_command,
    pick_command,
    place_approach_command,
    place_command,
    plan_envelope,
    simulate_held_state,
    simulate_placed_state,
)
from .placement import (
    build_frozen_place_step,
    future_stack_place_regions,
    reject_held_observation_and_keep_locked,
    validate_pick_place_separation,
    validate_place_second_snapshot,
)
from .post_place_observation import handle_post_place_observation
from .push_flow import handle_vlm_action_before_pick
from .scene import (
    estimate_current_stack,
    held_object_exclusion,
    load_or_capture_initial,
    locked_object_geometry,
    memory_id_for_scene_object,
    object_by_id,
    print_decision_summary,
    reacquire_pick_target_for_second_observation,
    reacquire_target,
    target_exclusion_for_pre_pick,
    validate_decision,
)
from .target_recovery import recover_or_lock_missing_target
from .task_workflow import run_semantic_task_workflow
from .execution_safety import validate_execution_source

SECOND_PICK_OBSERVATION_ERROR_MARKERS = (
    "Second observation center z",
    "Second observation center XY delta",
    "Second observation center is not finite",
    "Second-snapshot XY correction",
    "Second-snapshot TCP-target correction",
)

def _is_second_pick_observation_error(message: str) -> bool:
    return any(marker in message for marker in SECOND_PICK_OBSERVATION_ERROR_MARKERS)

def _write_first_pick_plan_without_second_xy(
    first_pick_plan_path: str,
    output_path: str,
    report_path: str,
    reason: str,
    args,
) -> None:
    plan = load_json(first_pick_plan_path)
    step = plan["steps"][0]
    step["coordinate_source"] = "locked_first_observation_without_second_xy_correction"
    step["second_snapshot_xy_correction_applied"] = False
    step["second_snapshot_rejected_reason"] = str(reason)
    write_json(output_path, plan)
    write_json(
        report_path,
        {
            "first_plan": first_pick_plan_path,
            "corrected_plan": output_path,
            "correction_applied": False,
            "fallback": "use_locked_first_observation_without_second_xy_correction",
            "reason": str(reason),
            "max_second_snapshot_correction_m": args.max_second_snapshot_correction_m,
            "second_snapshot_max_z_error_m": args.second_snapshot_max_z_error_m,
        },
    )

def _write_second_snapshot_hover_plan(first_pick_plan_path: str, output_path: str, hover_above_object_m: float) -> None:
    plan = load_json(first_pick_plan_path)
    step = plan["steps"][0]
    target = [float(value) for value in step["target_position_m"][:3]]
    hover_z = target[2] + float(hover_above_object_m)
    step["approach_position_m"] = [round(target[0], 5), round(target[1], 5), round(hover_z, 5)]
    step["second_snapshot_hover_above_object_m"] = float(hover_above_object_m)
    step["coordinate_source"] = "{}.second_snapshot_hover".format(step.get("coordinate_source", "locked_first"))
    write_json(output_path, plan)

def main() -> int:
    args = parse_args()
    validate_execution_source(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.memory_json is None:
        args.memory_json = os.path.join(args.output_dir, "scene_memory.json")

    calibration_path = args.calibration_json or args.xy_correction_json
    calibration = load_xy_correction(calibration_path)
    if calibration:
        write_json(os.path.join(args.output_dir, "active_calibration.json"), calibration)
    if calibration.get("tcp_offset_tool_m") is not None:
        tcp_offset = calibration.get("tcp_offset_tool_m")
        if not isinstance(tcp_offset, (list, tuple)) or len(tcp_offset) != 3:
            raise RuntimeError("tcp_offset_tool_m must contain exactly three values.")
        args.tcp_offset_tool = [float(value) for value in tcp_offset]

    warm_result = warm_model(args, args.output_dir)
    if warm_result.transport_status != "ok":
        raise RuntimeError("VLM_BACKEND_FAILED: {}: {}".format(
            warm_result.error_type or "MODEL_WARMUP_FAILED",
            warm_result.error_message or "model warmup failed",
        ))

    if route_task_type(args.instruction) == "stack_blocks":
        args.legacy_linear_stack = True

    if not args.legacy_linear_stack:
        try:
            return run_semantic_task_workflow(args)
        finally:
            if args.unload_model_after_task:
                unload_model(args, args.output_dir)

    memory = load_memory(args.memory_json, task="stack_blocks")

    if not args.resume_memory:
        memory = {
            "version": 1,
            "task": "stack_blocks",
            "step_index": 0,
            "objects": {},
            "structure": {
                "base": None,
                "placed_order": [],
                "current_top": None,
                "top_center_base": None,
                "top_z": None,
            },
            "action_history": [],
        }
        save_memory(memory, args.memory_json)
    runtime = {
        "current_stage": "startup",
        "held_object_id": None,
        "last_pick_pose": None,
        "last_place_pose": None,
        "scene_revision": 1,
    }
    try:
        runtime["current_stage"] = "init_ready_pose"
        init_ready_pose(args)

        runtime["current_stage"] = "initial_snapshot_and_decision"
        initial_state, raw_decision = load_or_capture_initial(args)
        decision, base_id, order = validate_decision(raw_decision, initial_state)
        print_decision_summary(initial_state, decision)
        write_json(os.path.join(args.output_dir, "stack_blocks_decision.json"), decision)
        write_json(os.path.join(args.output_dir, "initial_scene_state.json"), initial_state)

        memory = update_from_detections(memory, initial_state.get("objects", []), scene_revision=runtime["scene_revision"])
        write_json(os.path.join(args.output_dir, "track_assignment.json"), memory.get("last_track_assignment", []))
        write_json(os.path.join(args.output_dir, "track_history.json"), memory.get("track_history", []))
        memory["task_plan"] = {
            "instruction": args.instruction,
            "base_object_id": base_id,
            "full_stack_order": decision.get("full_stack_order", [base_id] + list(order)),
            "stack_order": list(order),
            "structure_plan": decision.get("structure_plan"),
            "decision_source": "vlm_stack_policy",
        }

        base_object = copy.deepcopy(object_by_id(initial_state, base_id))
        base_mem_id = memory_id_for_scene_object(memory, base_object)
        if base_mem_id is not None:
            memory = set_base(memory, base_mem_id)

        for target_id in order:
            target_object = object_by_id(initial_state, target_id)
            target_mem_id = memory_id_for_scene_object(memory, target_object)
            if target_mem_id is not None:
                memory = set_role(memory, target_mem_id, role="target", state="free")

        write_json(os.path.join(args.output_dir, "role_binding_history.json"), memory.get("role_binding_history", []))
        save_memory(memory, args.memory_json)

        held_templates = {object_id: copy.deepcopy(object_by_id(initial_state, object_id)) for object_id in order}
        write_json(
            os.path.join(args.output_dir, "locked_initial_geometry.json"),
            {
                "base": locked_object_geometry(base_object),
                "stack_order": [locked_object_geometry(held_templates[object_id]) for object_id in order],
                "table_plane": initial_state.get("table_plane"),
                "source": "initial_safe_observation_before_any_pick",
            },
        )

        current_state = copy.deepcopy(initial_state)
        previous_stack_xy = None
        previous_locked_stack = None
        protected_locked_stack = None

        for index, object_id in enumerate(order, start=1):
            cycle_dir = os.path.join(args.output_dir, "cycle_{:02d}_object_{}".format(index, object_id))
            os.makedirs(cycle_dir, exist_ok=True)

            if index > 1:
                runtime["current_stage"] = "empty_gripper_ready_observation"
                observed = capture_empty_observation(
                    args,
                    os.path.join(cycle_dir, "observation_before_pick"),
                    runtime["held_object_id"],
                )
                if observed is not None:
                    current_state = observed
                    advance_scene_revision(runtime, current_state)
                    memory = update_from_detections(memory, current_state.get("objects", []), scene_revision=runtime["scene_revision"])
                    save_memory(memory, args.memory_json)

            runtime["current_stage"] = "detect_target_and_freeze_place"
            try:
                held_object = copy.deepcopy(reacquire_target(current_state, held_templates[object_id]))
            except RuntimeError:
                runtime["current_stage"] = "target_recovery_before_clearance"
                current_state, memory, held_object, _target_recovery = recover_or_lock_missing_target(
                    args,
                    cycle_dir,
                    runtime,
                    memory,
                    current_state,
                    held_templates[object_id],
                    protected_templates=[base_object] + (
                        (protected_locked_stack or previous_locked_stack or {}).get("stack_objects", [])
                        if isinstance(protected_locked_stack or previous_locked_stack, dict)
                        else []
                    ),
                )
            current_state, memory, held_object = handle_vlm_action_before_pick(
                args,
                cycle_dir,
                runtime,
                memory,
                current_state,
                held_templates[object_id],
                held_object,
                base_id,
                protected_locked_stack or previous_locked_stack,
                base_template=base_object,
                future_place_regions=future_stack_place_regions(base_object, previous_stack_xy, args),
            )
            planned_clearance = current_state.pop("_moveit_plan_only_clearance_action", None)
            if planned_clearance is not None:
                summary = {
                    "task_type": "stack_blocks",
                    "execution_status": "planned_only",
                    "planned_action": planned_clearance,
                    "moveit_feasible": bool(planned_clearance.get("moveit_feasible")),
                    "scene_changed": False,
                    "gripper_enabled": False,
                    "next_required_step": "execute_clearance_then_reobserve_before_pick",
                }
                write_json(os.path.join(args.output_dir, "stack_demo_summary.json"), summary)
                if os.path.exists(failure_state_path(args)):
                    os.remove(failure_state_path(args))
                return 0
            selected_pick_template = copy.deepcopy(held_object)
            pre_pick_excluded_ids, pre_pick_excluded_xy = target_exclusion_for_pre_pick(held_object)
            current_base_object, stack_state = estimate_current_stack(
                current_state,
                base_object,
                previous_stack_xy,
                args,
                search_radius_m=args.search_radius_m,
                excluded_object_ids=pre_pick_excluded_ids,
                excluded_xy=pre_pick_excluded_xy,
                exclusion_radius_m=args.target_exclusion_radius_m,
            )
            if current_base_object.get("reacquire_source") == "current_observation":
                base_object = current_base_object
            if not stack_state.get("valid"):
                raise RuntimeError("Cycle {} pre-pick stack estimate failed: {}".format(index, stack_state.get("reason")))
            if previous_locked_stack is not None:
                verification = verify_stack_growth(previous_locked_stack, stack_state)
                write_json(os.path.join(cycle_dir, "previous_place_verification.json"), verification)
                if not verification["valid"]:
                    recheck_state = capture_empty_observation(
                        args,
                        os.path.join(cycle_dir, "previous_place_verification_recheck"),
                        runtime["held_object_id"],
                    )
                    if recheck_state is not None:
                        current_state = recheck_state
                        memory = update_from_detections(memory, current_state.get("objects", []), scene_revision=runtime["scene_revision"])
                        save_memory(memory, args.memory_json)
                        held_object = copy.deepcopy(reacquire_target(current_state, selected_pick_template))
                        pre_pick_excluded_ids, pre_pick_excluded_xy = target_exclusion_for_pre_pick(held_object)
                        current_base_object, stack_state = estimate_current_stack(
                            current_state,
                            base_object,
                            previous_stack_xy,
                            args,
                            search_radius_m=args.search_radius_m,
                            excluded_object_ids=pre_pick_excluded_ids,
                            excluded_xy=pre_pick_excluded_xy,
                            exclusion_radius_m=args.target_exclusion_radius_m,
                        )
                        if current_base_object.get("reacquire_source") == "current_observation":
                            base_object = current_base_object
                        verification = verify_stack_growth(previous_locked_stack, stack_state)
                        write_json(
                            os.path.join(cycle_dir, "previous_place_verification_recheck.json"),
                            verification,
                        )
                    if not verification["valid"]:
                        raise RuntimeError(
                            "Cycle {} previous placement verification failed after recheck: {}".format(
                                index, verification["reason"]
                            )
                        )
            write_json(os.path.join(cycle_dir, "stack_state_locked_before_pick.json"), stack_state)
            previous_stack_xy = stack_state["stack_xy_base_m"]
            provisional_place_step = build_frozen_place_step(
                current_state, stack_state, current_base_object, held_object, args
            )
            provisional_place_plan_path = os.path.join(cycle_dir, "place_on_top_plan_first_observation.json")
            write_json(provisional_place_plan_path, plan_envelope(current_state, provisional_place_step))
            first_pick_plan_path = os.path.join(cycle_dir, "pick_plan_first_observation.json")
            build_offline_pick_plan(current_state, held_object, first_pick_plan_path, args)

            runtime["current_stage"] = "empty_gripper_first_pick_approach"
            if args.execute:
                approach_plan_path = first_pick_plan_path
                if args.enable_second_pick_snapshot:
                    approach_plan_path = os.path.join(cycle_dir, "pick_plan_second_snapshot_hover.json")
                    _write_second_snapshot_hover_plan(
                        first_pick_plan_path,
                        approach_plan_path,
                        args.second_snapshot_hover_above_object_m,
                    )
                run(pick_approach_command(args, approach_plan_path))

            second_object = copy.deepcopy(held_object)
            pick_plan_path = first_pick_plan_path
            correction_report_path = os.path.join(cycle_dir, "pick_second_xy_correction.json")
            if not args.enable_second_pick_snapshot:
                write_json(
                    correction_report_path,
                    {
                        "first_plan": first_pick_plan_path,
                        "corrected_plan": first_pick_plan_path,
                        "correction_applied": False,
                        "fallback": "use_locked_first_observation_without_second_snapshot",
                        "reason": "second pick snapshot disabled",
                    },
                )
            else:
                runtime["current_stage"] = "empty_gripper_second_pick_snapshot"
                second_dir = os.path.join(cycle_dir, "pick_second_observation")

                def parse_second_pick_target(state):
                    state["_second_snapshot_max_xy_m"] = float(args.max_second_snapshot_correction_m)
                    state["_second_snapshot_max_z_error_m"] = float(args.second_snapshot_max_z_error_m)
                    return copy.deepcopy(reacquire_pick_target_for_second_observation(state, held_object))

                try:
                    second_state, second_object = retry_close_observation(
                        args,
                        second_dir,
                        runtime["held_object_id"],
                        parse_second_pick_target,
                        "Second target observation",
                        retry_offset_camera=args.second_snapshot_retry_offset_camera,
                        refresh_tf=True,
                    )
                    second_unreliable_reason = None
                except RuntimeError as exc:
                    message = str(exc)
                    if not _is_second_pick_observation_error(message):
                        raise
                    print(
                        "Second target observation was visible but had unreliable 3D center; "
                        "using locked first observation without XY correction: {}".format(message),
                        flush=True,
                    )
                    write_json(
                        os.path.join(cycle_dir, "pick_second_observation_unreliable_use_first.json"),
                        {
                            "reason": message,
                            "fallback": "use_locked_first_observation_without_second_xy_correction",
                            "max_second_snapshot_correction_m": args.max_second_snapshot_correction_m,
                            "second_snapshot_max_z_error_m": args.second_snapshot_max_z_error_m,
                        },
                    )
                    second_unreliable_reason = message
                    second_state = current_state
                    second_object = copy.deepcopy(held_object)
                if second_state is None:
                    second_unreliable_reason = "Second target observation produced no scene state."
                    second_state = current_state
                    second_object = copy.deepcopy(held_object)
                second_pick_plan_path = os.path.join(cycle_dir, "pick_plan_second_observation.json")
                build_offline_pick_plan(second_state, second_object, second_pick_plan_path, args)
                pick_plan_path = os.path.join(cycle_dir, "pick_plan_second_xy_corrected.json")
                if second_unreliable_reason is None:
                    try:
                        build_tcp_error_corrected_plan(
                            first_pick_plan_path,
                            second_pick_plan_path,
                            args.tf_json,
                            pick_plan_path,
                            correction_report_path,
                            args.max_second_snapshot_correction_m,
                            args.max_grasp_offset_m,
                            tcp_offset_tool=args.tcp_offset_tool,
                        )
                    except RuntimeError as exc:
                        message = str(exc)
                        if not _is_second_pick_observation_error(message):
                            raise
                        print(
                            "Second target correction rejected; using locked first observation "
                            "without XY correction: {}".format(message),
                            flush=True,
                        )
                        _write_first_pick_plan_without_second_xy(
                            first_pick_plan_path,
                            pick_plan_path,
                            correction_report_path,
                            message,
                            args,
                        )
                else:
                    _write_first_pick_plan_without_second_xy(
                        first_pick_plan_path,
                        pick_plan_path,
                        correction_report_path,
                        second_unreliable_reason,
                        args,
                    )
            pick_plan = load_json(pick_plan_path)
            pick_step = pick_plan["steps"][0]

            # Do not leave the selected pick approach before grasping. The base is
            # reacquired only after the object is held and the robot is above it.
            final_stack_state = stack_state
            place_step = build_frozen_place_step(
                current_state, final_stack_state, current_base_object, held_object, args
            )
            place_step["coordinate_source"] = "pre_pick_stack_observation.approach_only"
            validate_pick_place_separation(pick_step, place_step, args.min_pick_place_xy_distance_m)
            place_plan_path = os.path.join(cycle_dir, "place_on_top_plan_locked_before_pick.json")
            write_json(place_plan_path, plan_envelope(current_state, place_step))
            runtime["last_place_pose"] = {
                "position_m": place_step["target_position_m"],
                "pre_place_z_base_m": place_step["pre_place_z_base_m"],
                "release_z_base_m": place_step["release_z_base_m"],
                "detected_base_center_xy_m": final_stack_state.get("placement_base_center_xy_m"),
                "placement_reference_center_xy_m": place_step.get("placement_reference_center_xy_m"),
                "placement_reference_source": place_step.get("placement_reference_source"),
                "observed_top_center_xy_m": place_step.get("observed_top_center_xy_m"),
                "top_center_offset_from_reference_m": place_step.get("top_center_offset_from_reference_m"),
                "detected_base_top_z_m": final_stack_state.get("placement_base_top_z_m"),
                "place_top_z_bias_m": place_step.get("place_top_z_bias_m"),
                "release_gap_m": place_step.get("release_gap_m"),
                "yaw_deg": place_step["chosen_place_yaw_deg"],
                "source": place_step["coordinate_source"],
            }

            if args.moveit_plan_only and not args.execute:
                runtime["current_stage"] = "linear_stack_moveit_plan_only"
                run(plan_only_command(pick_command(args, pick_plan_path)))
                run(plan_only_command(place_command(args, place_plan_path)))
                summary = {
                    "task_type": "stack_blocks",
                    "execution_status": "planned_only",
                    "base_object_id": base_id,
                    "full_stack_order": decision.get("full_stack_order", [base_id] + list(order)),
                    "stack_order": order,
                    "planned_object_id": held_object.get("id"),
                    "pick_plan": pick_plan_path,
                    "place_plan": place_plan_path,
                    "moveit_feasible": True,
                    "gripper_enabled": False,
                }
                write_json(os.path.join(args.output_dir, "stack_demo_summary.json"), summary)
                if os.path.exists(failure_state_path(args)):
                    os.remove(failure_state_path(args))
                return 0

            print(
                "\nCycle {} pick plan ready; grasp immediately before base approach: "
                "target_id={} label={} first_center={} second_center={} "
                "first_object_yaw={} second_object_yaw={} final_grasp_yaw={} pick_plan_source={} "
                "detected_base_id={} detected_base_center={} "
                "detected_base_top_z={} stack_average_center={} stack_yaw={} place_pose={}".format(
                    index,
                    held_object.get("id"),
                    held_object.get("label"),
                    held_object.get("geometry_center_m"),
                    second_object.get("geometry_center_m"),
                    held_object.get("table_yaw_deg"),
                    second_object.get("table_yaw_deg"),
                    pick_step.get("chosen_grasp_yaw_deg"),
                    pick_step.get("coordinate_source"),
                    final_stack_state.get("placement_base_object_id"),
                    final_stack_state.get("placement_base_center_xy_m"),
                    final_stack_state.get("placement_base_top_z_m"),
                    final_stack_state.get("stack_xy_base_m"),
                    final_stack_state.get("stack_yaw_deg"),
                    runtime["last_place_pose"],
                ),
                flush=True,
            )

            runtime["current_stage"] = "pick_from_locked_visual_plan"
            runtime["last_pick_pose"] = {
                "first_center_base_m": held_object.get("geometry_center_m"),
                "second_center_base_m": second_object.get("geometry_center_m"),
                "corrected_target_position_m": pick_step.get("target_position_m"),
                "first_estimated_yaw_deg": held_object.get("table_yaw_deg"),
                "second_estimated_yaw_deg": second_object.get("table_yaw_deg"),
                "chosen_grasp_yaw_deg": pick_step.get("chosen_grasp_yaw_deg"),
            }
            if args.execute:
                runtime["held_object_id"] = held_object.get("id")
                run(pick_command(args, pick_plan_path))
            else:
                runtime["held_object_id"] = held_object.get("id")

            runtime["current_stage"] = "holding_object_move_above_locked_base"
            if args.execute:
                run(place_approach_command(args, place_plan_path))

                runtime["current_stage"] = "holding_object_final_base_snapshot"
                held_close_dir = os.path.join(cycle_dir, "place_final_observation_while_holding")

                def parse_held_base(state):
                    excluded_ids, excluded_xy = held_object_exclusion(state, held_object)
                    anchor_xy = final_stack_state.get("stack_xy_base_m") or final_stack_state.get("placement_base_center_xy_m")
                    _, estimate = estimate_current_stack(
                        state,
                        current_base_object,
                        anchor_xy,
                        args,
                        search_radius_m=args.close_stack_search_radius_m,
                        excluded_object_ids=excluded_ids,
                        excluded_xy=excluded_xy,
                        exclusion_radius_m=args.target_exclusion_radius_m,
                    )
                    if not estimate.get("valid"):
                        _, estimate = estimate_current_stack(
                            state,
                            current_base_object,
                            anchor_xy,
                            args,
                            search_radius_m=max(args.close_stack_search_radius_m, args.search_radius_m),
                            excluded_object_ids=excluded_ids,
                            excluded_xy=excluded_xy,
                            exclusion_radius_m=args.target_exclusion_radius_m,
                        )
                    if not estimate.get("valid"):
                        raise RuntimeError(estimate.get("reason"))
                    estimate["held_observation_excluded_held_object_ids"] = excluded_ids
                    estimate["held_observation_excluded_held_object_xy_m"] = excluded_xy
                    estimate["held_observation_anchor_xy_m"] = anchor_xy
                    return estimate

                try:
                    held_close_state, held_final_stack = retry_close_observation(
                        args,
                        held_close_dir,
                        runtime["held_object_id"],
                        parse_held_base,
                        "Final held-object base observation",
                        allow_holding=True,
                        retry_count=args.close_observation_retry_count,
                        retry_offset_camera=args.place_observation_offset_camera,
                    )
                    if held_close_state is None or held_final_stack is None:
                        raise RuntimeError("Final held-object base observation produced no scene state.")
                    write_json(
                        os.path.join(cycle_dir, "stack_state_final_held_before_place.json"),
                        held_final_stack,
                    )
                    held_place_correction = validate_place_second_snapshot(
                        final_stack_state,
                        held_final_stack,
                        args.max_place_second_snapshot_correction_m,
                        top_z_tolerance_m=args.held_base_top_z_tolerance_m,
                    )
                    write_json(
                        os.path.join(cycle_dir, "place_final_held_xy_z_correction.json"),
                        held_place_correction,
                    )
                    write_json(
                        os.path.join(cycle_dir, "place_final_held_observation_verified_keep_locked.json"),
                        {
                            "validated": True,
                            "execution_plan": place_plan_path,
                            "policy": "keep_locked_before_pick_place_plan",
                            "reason": (
                                "Held-object close observation is verification-only; "
                                "it must not overwrite locked place center or yaw."
                            ),
                            "held_place_correction": held_place_correction,
                            "locked_place_pose": runtime["last_place_pose"],
                        },
                    )
                except RuntimeError as exc:
                    reject_held_observation_and_keep_locked(
                        cycle_dir,
                        exc,
                        place_plan_path,
                        locals().get("held_final_stack"),
                        args,
                    )

            runtime["current_stage"] = "place_from_final_base_geometry"
            if args.execute:
                run(place_command(args, place_plan_path))
            runtime["held_object_id"] = None

            placed_mem_id = memory_id_for_scene_object(memory, held_object)
            if placed_mem_id is not None:
                memory = mark_placed(
                    memory,
                    placed_mem_id,
                    target_id=memory["structure"].get("current_top"),
                    result="executed" if args.execute else "dry_run",
                )
                write_json(os.path.join(args.output_dir, "role_binding_history.json"), memory.get("role_binding_history", []))
                placed_template = expected_placed_template(held_object, place_step)
                placed_center = placed_template.get("geometry_center_m")
                if isinstance(placed_center, list) and len(placed_center) >= 3:
                    memory["objects"][placed_mem_id]["last_pose_base"] = [float(value) for value in placed_center[:3]]
                    held_size = held_object.get("dimensions_m") or memory["objects"][placed_mem_id].get("size_m")
                    if isinstance(held_size, list) and len(held_size) >= 3:
                        memory["objects"][placed_mem_id]["top_z"] = float(placed_center[2]) + float(held_size[2]) / 2.0
                    memory["structure"]["top_center_base"] = memory["objects"][placed_mem_id]["last_pose_base"]
                    memory["structure"]["top_z"] = memory["objects"][placed_mem_id].get("top_z")
                save_memory(memory, args.memory_json)

            if args.execute:
                current_state, memory, protected_locked_stack = handle_post_place_observation(
                    args,
                    cycle_dir,
                    runtime,
                    memory,
                    current_state,
                    base_object,
                    current_base_object,
                    held_object,
                    place_step,
                    final_stack_state,
                    [held_templates[target_id] for target_id in order[index:]],
                    index,
                )
                save_memory(memory, args.memory_json)
            else:
                held_state = simulate_held_state(current_state, held_object.get("id"))
                current_state = simulate_placed_state(held_state, held_object, place_step)
                write_json(os.path.join(cycle_dir, "after_place_scene_state.json"), current_state)
                memory = update_from_detections(memory, current_state.get("objects", []), scene_revision=runtime["scene_revision"])
                _, protected_locked_stack = estimate_current_stack(
                    current_state,
                    base_object,
                    final_stack_state.get("stack_xy_base_m"),
                    args,
                    search_radius_m=args.search_radius_m,
                )
                save_memory(memory, args.memory_json)

            previous_locked_stack = final_stack_state
            previous_stack_xy = (
                protected_locked_stack.get("stack_xy_base_m")
                if protected_locked_stack and protected_locked_stack.get("valid")
                else final_stack_state["stack_xy_base_m"]
            )

        runtime["current_stage"] = "final_empty_gripper_ready_verification"
        observed = capture_empty_observation(
            args,
            os.path.join(args.output_dir, "final_observation"),
            runtime["held_object_id"],
        )
        if observed is not None:
            current_state = observed
            memory = update_from_detections(memory, current_state.get("objects", []), scene_revision=runtime["scene_revision"])
            save_memory(memory, args.memory_json)
        final_base_object, final_stack_state = estimate_current_stack(
            current_state,
            base_object,
            previous_stack_xy,
            args,
            search_radius_m=args.search_radius_m,
        )
        final_verification = verify_stack_growth(previous_locked_stack, final_stack_state)
        write_json(os.path.join(args.output_dir, "final_place_verification.json"), final_verification)
        if not final_verification["valid"]:
            raise RuntimeError("Final placement verification failed: {}".format(final_verification["reason"]))

        summary = {
            "task_type": "stack_blocks",
            "execution_status": "executed" if args.execute else "dry_run_complete",
            "base_object_id": base_id,
            "full_stack_order": decision.get("full_stack_order", [base_id] + list(order)),
            "stack_order": order,
            "stack_order_semantics": decision.get("stack_order_semantics", "place_order_excludes_base"),
            "structure_plan": decision.get("structure_plan"),
            "cycles_completed": len(order),
            "final_stack_state": final_stack_state,
            "pre_vlm_geometry_action_analysis_enabled": False,
            "post_vlm_physical_validation_enabled": True,
            "push_clearing_enabled": args.execute_push_clearing,
            "scene_memory_json": args.memory_json,
        }
        write_json(os.path.join(args.output_dir, "stack_demo_summary.json"), summary)
        if os.path.exists(failure_state_path(args)):
            os.remove(failure_state_path(args))
        print("\nStack demo {}: {}".format(summary["execution_status"], args.output_dir))
        if args.unload_model_after_task:
            unload_model(args, args.output_dir)
        return 0
    except Exception as exc:
        held_at_failure = runtime.get("held_object_id")
        failure = dict(runtime)
        failure.update(
            {
                "error": str(exc),
                "held_object_id_at_failure": held_at_failure,
            }
        )
        exception_history = getattr(exc, "failure_history", None)
        if exception_history:
            failure["vlm_failure_history"] = exception_history
        write_json(failure_state_path(args), failure)
        print(
            "\nSTACK DEMO FAILED at stage {}: {}\nSaved failure state: {}".format(
                runtime.get("current_stage"), exc, failure_state_path(args)
            ),
            file=sys.stderr,
            flush=True,
        )
        if args.unload_model_after_task:
            unload_model(args, args.output_dir)
        return 1
