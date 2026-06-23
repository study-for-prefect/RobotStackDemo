"""Top-level two-stage visual pick orchestration."""

import os

from .arguments import parse_args
from .commands import (
    build_plan_command,
    capture_first_snapshot_until_target_visible,
    capture_second_snapshot_and_plan,
    moveit_common,
    relative_translate_command,
    snapshot_command,
)
from .io import load_json, run, write_json
from .planning import (
    build_corrected_plan,
    has_planned_motion,
    plan_has_reliable_yaw,
    reject_additional_pick_steps,
    reject_incomplete_remaining_motion_steps,
    remaining_plan_after_step,
    safe_name,
    select_llm_pick_step,
)
from .scene import camera_optical_vector_to_base, matching_target_objects, target_description

def main() -> int:
    args = parse_args()
    execute = bool(args.execute)
    first_private = os.path.join(args.first_dir, "private_scene_state.json")
    second_private = os.path.join(args.second_dir, "private_scene_state.json")
    correction_report = os.path.join(args.first_dir, "second_snapshot_xy_correction.json")
    llm_plan_path = os.path.join(args.first_dir, "robot_execution_plan.json")
    remaining_plan_path = os.path.join(args.first_dir, "robot_execution_plan_after_two_stage_pick.json")
    llm_plan = None
    selected_pick_index = None

    if args.instruction:
        print("LLM + two-stage visual pick: ready -> LLM snapshot -> yaw/approach -> second snapshot -> XY correction -> pick -> remaining LLM plan")
    else:
        print("Two-stage visual pick: ready -> snapshot1 -> yaw/approach -> snapshot2 -> XY correction -> pick")
    if not execute:
        print("DRY RUN: add --execute to move the robot and capture snapshots.")

    ready = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--ready-only",
        "--ready-joint-pose-json",
        args.ready_pose_json,
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--tf-timeout",
        str(args.moveit_tf_timeout),
    ]
    if args.execute:
        ready.append("--execute")
    if args.enable_gripper:
        ready.extend([
            "--enable-gripper",
            "--gripper-port",
            args.gripper_port,
            "--open-gripper-at-start",
        ])
    if args.yes:
        ready.append("--yes")
    run(ready, execute)

    if not capture_first_snapshot_until_target_visible(args, first_private, run_llm=bool(args.instruction)):
        print(
            "Stop before motion. First snapshot did not detect {} after {} retry/retries.".format(
                target_description(args),
                args.first_snapshot_retry_count,
            ),
            flush=True,
        )
        return 2
    if args.instruction and execute:
        llm_plan = load_json(llm_plan_path)
        selected_pick_index, selected_pick = select_llm_pick_step(llm_plan)
        reject_additional_pick_steps(llm_plan, selected_pick_index)
        reject_incomplete_remaining_motion_steps(llm_plan, selected_pick_index)
        args.object_id = int(selected_pick["object_id"])
        if selected_pick.get("object_label"):
            args.object_label = str(selected_pick["object_label"])
        print(
            "\nLLM selected pick target: object_id={} label={}".format(args.object_id, args.object_label),
            flush=True,
        )

    object_name = "id_{}".format(args.object_id) if args.object_id is not None else safe_name(args.object_label)
    if args.instruction and not execute:
        object_name = "llm_selected"
    first_plan = os.path.join(args.first_dir, "{}_pick_plan_target_test.json".format(object_name))
    second_plan = os.path.join(args.second_dir, "{}_pick_plan_second.json".format(object_name))
    corrected_plan = os.path.join(args.first_dir, "{}_pick_plan_second_xy_corrected.json".format(object_name))

    if execute and not target_visible(args, first_private):
        print("Stop before motion. Move the object into view or change --object-label/--object-id, then rerun.", flush=True)
        return 2
    run(build_plan_command(args, first_private, first_plan), execute)
    if args.instruction and execute:
        selected_objects = matching_target_objects(args, first_private)
        selected_point = object_base_point(selected_objects[0]) if selected_objects else None
        if selected_point is not None:
            args.target_reference_base_xy = selected_point[:2]

    approach = moveit_common(args, first_plan)
    use_object_yaw = plan_has_reliable_yaw(first_plan) if execute else True
    if use_object_yaw:
        approach.extend(
            [
                "--orientation-mode",
                "object-yaw",
                "--grasp-axis",
                args.grasp_axis,
                "--yaw-offset-deg",
                str(args.yaw_offset_deg),
                "--pre-rotate-before-translation",
                "--pre-rotate-strategy",
                "joint-wrist3",
                "--pre-rotate-wrist-direction",
                args.pre_rotate_wrist_direction,
                "--max-pre-rotate-joint-delta",
                str(args.max_pre_rotate_joint_delta),
                "--path-mode",
                "approach",
            ]
        )
    else:
        print("\nFirst plan has no reliable yaw; keeping ready/current orientation for approach.", flush=True)
        approach.extend(["--orientation-mode", "current", "--path-mode", "approach"])
    run(approach, execute)

    if args.instruction and execute:
        # Detector IDs may change after the camera moves. The selected public label
        # is the stable target key for the second snapshot.
        args.object_id = None

    if not capture_second_snapshot_and_plan(args, second_private, second_plan):
        print(
            "\nSecond snapshot could not find/build a plan for {} after {} retry/retries. Stop before pick.".format(
                target_description(args),
                args.second_snapshot_retry_count,
            ),
            flush=True,
        )
        return 3
    if execute:
        build_corrected_plan(
            first_plan,
            second_plan,
            corrected_plan,
            correction_report,
            args.max_correction_m,
            args.max_grasp_offset_m,
        )
    else:
        print("\nWould calculate second_target_xy - first_target_xy and write {}".format(corrected_plan))

    pick = moveit_common(args, corrected_plan)
    pick.extend(["--orientation-mode", "current", "--path-mode", "full"])
    if args.enable_gripper:
        pick.extend([
            "--enable-gripper",
            "--gripper-port",
            args.gripper_port,
            "--skip-gripper-init",
            "--post-close-wait",
            str(args.post_close_wait),
        ])
        if args.release_after_pick and not args.instruction:
            pick.append("--release-after-pick")
    run(pick, execute)

    if args.instruction:
        if execute:
            remaining_plan = remaining_plan_after_step(llm_plan, selected_pick_index)
            write_json(remaining_plan_path, remaining_plan)
            print("\nSaved remaining LLM execution plan: {}".format(remaining_plan_path), flush=True)
            if args.execute_remaining_plan and has_planned_motion(remaining_plan):
                remaining = moveit_common(args, remaining_plan_path)
                remaining.extend(["--all-approaches", "--orientation-mode", "current", "--path-mode", "full"])
                if args.enable_gripper:
                    remaining.extend(
                        [
                            "--enable-gripper",
                            "--gripper-port",
                            args.gripper_port,
                            "--skip-gripper-init",
                            "--post-close-wait",
                            str(args.post_close_wait),
                        ]
                    )
                run(remaining, execute)
            elif not has_planned_motion(remaining_plan):
                print("\nLLM plan has no executable motion after pick; keeping the object held.", flush=True)
            else:
                print("\nRemaining LLM plan execution disabled; keeping the object held.", flush=True)
        else:
            print("\nWould extract the LLM pick target, preserve the remaining plan, and execute it after the corrected pick.")
    return 0
