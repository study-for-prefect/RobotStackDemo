"""Top-level MoveIt preview workflow orchestration."""

import math
import sys
import time
from typing import Optional

import rclpy
from sensor_msgs.msg import JointState

from robot_scene_pipeline.grasp_orientation import quaternion_distance_rad

from .arguments import parse_args
from .execution import (
    plan_and_maybe_execute_joint_motion,
    plan_and_maybe_execute_motion,
    run_hover_only,
    select_best_pre_rotate_plan,
    settle_orientation_before_descent,
)
from .orientation import (
    add_base_offset,
    estimate_downward_family_yaw_deg,
    gripper_yaw_error_deg,
    orientation_for_step,
    shortest_yaw_delta_deg,
    tool0_goal_from_tcp,
    transform_position_quat,
)
from .plan_io import load_joint_pose, load_plan, load_tcp_offset
from .push import run_push_plan
from .steps import (
    command_sequence_for_step,
    diagnostic_steps,
    selected_steps,
    validate_complete_plan,
)
from .tf_node import MoveItPreviewNode
from .trajectory import (
    gripper_position_for_command,
    joint_position_map_from_state,
    max_joint_error_to_goal,
    maybe_confirm,
    setup_gripper,
)

def main() -> Optional[int]:
    args = parse_args()
    push_only = bool(args.push_plan_json)
    exclusive_modes = sum(
        bool(value)
        for value in (
            args.ready_only,
            args.gripper_open_only,
            args.gripper_close_only,
            args.relative_tool_translation_base is not None,
            args.hover_only,
            push_only,
        )
    )
    if exclusive_modes > 1:
        raise RuntimeError(
            "--push-plan-json, --ready-only, --gripper-open-only, --gripper-close-only, "
            "--relative-tool-translation-base, and --hover-only are mutually exclusive."
        )
    if args.ready_only and not args.ready_joint_pose_json:
        raise RuntimeError("--ready-only requires --ready-joint-pose-json.")
    if args.gripper_open_only and not args.enable_gripper:
        raise RuntimeError("--gripper-open-only requires --enable-gripper.")
    if args.gripper_close_only and not args.enable_gripper:
        raise RuntimeError("--gripper-close-only requires --enable-gripper.")
    if args.hover_only:
        if args.hover_target_base is None:
            raise RuntimeError("--hover-only requires --hover-target-base X Y Z.")
        if args.hover_orientation_xyzw is None:
            raise RuntimeError("--hover-only requires --hover-orientation-xyzw QX QY QZ QW.")
        if args.enable_gripper:
            raise RuntimeError("--hover-only refuses --enable-gripper.")
    if push_only and args.enable_gripper:
        raise RuntimeError("--push-plan-json refuses --enable-gripper.")
    relative_only = args.relative_tool_translation_base is not None
    plan = (
        None
        if args.ready_only
        or relative_only
        or args.gripper_open_only
        or args.gripper_close_only
        or args.hover_only
        or push_only
        else load_plan(args.plan_json)
    )
    calibrated_tcp_offset_tool = load_tcp_offset(args.tcp_calibration_json)
    tcp_offset_tool = calibrated_tcp_offset_tool or [float(value) for value in args.tcp_offset_tool]
    ready_joint_pose = load_joint_pose(args.ready_joint_pose_json)
    if args.diagnostic_yaw_deg:
        if args.path_mode != "approach":
            raise RuntimeError("--diagnostic-yaw-deg only supports --path-mode approach.")
        if args.orientation_mode != "object-yaw":
            raise RuntimeError("--diagnostic-yaw-deg requires --orientation-mode object-yaw.")
        if args.execute and args.yes:
            raise RuntimeError("--diagnostic-yaw-deg execution refuses --yes; confirm every diagnostic pose.")
    if plan is not None and plan.get("coordinate_convention", {}).get("frame") != "base_link":
        print("ERROR: execution plan is not in base_link frame.", file=sys.stderr)
        return 2
    if args.execute and plan is not None:
        validate_complete_plan(plan, args)

    steps = [] if plan is None else diagnostic_steps(selected_steps(plan, args), args.diagnostic_yaw_deg)
    if plan is not None:
        print("Loaded {} planned step(s) from {}".format(len(steps), args.plan_json))
    print("Mode: {}".format("EXECUTE" if args.execute else "PLAN ONLY"))
    if args.hover_only:
        print("Hover-only: enabled")
    if push_only:
        print("Push clearing plan: {}".format(args.push_plan_json))
    print("Path mode: {}".format(args.path_mode))
    print("Gripper: {}".format("enabled" if args.enable_gripper else "disabled"))
    print("Planner: {}".format("Cartesian" if args.cartesian else "Joint-space pose"))
    print("Pre-rotate before translation: {}".format("enabled" if args.pre_rotate_before_translation else "disabled"))
    print("Pre-rotate strategy: {}".format(args.pre_rotate_strategy))
    print("Pre-rotate wrist direction: {}".format(args.pre_rotate_wrist_direction))
    print("Max joint delta: motion={:.3f}, pre_rotate={:.3f}".format(args.max_joint_delta, args.max_pre_rotate_joint_delta))
    if ready_joint_pose is not None:
        print("Ready joint pose: {}".format(args.ready_joint_pose_json))
    if calibrated_tcp_offset_tool is not None:
        print("TCP calibration: {} offset_tool={}".format(args.tcp_calibration_json, tcp_offset_tool))
    else:
        print("TCP offset tool0->TCP: {}".format(tcp_offset_tool))

    rclpy.init(args=None)
    node = MoveItPreviewNode(args)
    gripper = None
    try:
        if args.execute:
            gripper = setup_gripper(args)
            if args.gripper_open_only:
                position = gripper_position_for_command(args, "open")
                gripper.set_position(position)
                status = gripper.wait_until_done(timeout=args.gripper_wait)
                current = gripper.get_position()
                node.get_logger().info("Recovery gripper open done: status={}, position={}".format(status, current))
                return 0
            if args.gripper_close_only:
                position = gripper_position_for_command(args, "close")
                gripper.set_position(position)
                status = gripper.wait_until_done(timeout=args.gripper_wait)
                current = gripper.get_position()
                node.get_logger().info("Rigid paddle gripper close done: status={}, position={}".format(status, current))
                return 0
            if args.open_gripper_at_start:
                if gripper is None:
                    node.get_logger().warning("--open-gripper-at-start ignored because --enable-gripper is not active.")
                elif maybe_confirm(args, "Open gripper at start?"):
                    position = gripper_position_for_command(args, "open")
                    gripper.set_position(position)
                    status = gripper.wait_until_done(timeout=args.gripper_wait)
                    current = gripper.get_position()
                    node.get_logger().info(
                        "Initial gripper open done: status={}, position={}".format(status, current)
                    )
                else:
                    node.get_logger().info("Initial gripper open skipped by user.")

        if not node.wait_for_joint_state(timeout=5.0):
            node.get_logger().error("No /joint_states received. Start UR driver and MoveIt first.")
            return 2

        planning_start_state = node.latest_joint_state
        if ready_joint_pose is not None:
            ready_error = max_joint_error_to_goal(
                joint_position_map_from_state(planning_start_state),
                ready_joint_pose["joint_names"],
                ready_joint_pose["joint_positions"],
            )
            if ready_error is not None and ready_error <= args.ready_joint_tolerance:
                node.get_logger().info(
                    "Ready pose already reached: max_joint_error={:.4f} rad <= {:.4f} rad; skipping ready motion.".format(
                        ready_error,
                        args.ready_joint_tolerance,
                    )
                )
            else:
                if ready_error is not None:
                    node.get_logger().info("Ready pose max_joint_error={:.4f} rad; planning ready motion.".format(ready_error))
                ready_result = plan_and_maybe_execute_joint_motion(
                    node,
                    args,
                    "ready_pose {}".format(ready_joint_pose["name"]),
                    ready_joint_pose["joint_names"],
                    ready_joint_pose["joint_positions"],
                    start_joint_state=planning_start_state,
                )
                if not ready_result:
                    node.get_logger().error("Ready joint pose did not complete; selected steps will not run.")
                    return 2
                if isinstance(ready_result, JointState):
                    planning_start_state = ready_result
            if args.ready_only:
                node.get_logger().info("Ready-only motion complete.")
                return 0

        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, current_quat = transform_position_quat(current_tool)
        node.get_logger().info(
            "Current tool0 position=[{:.4f}, {:.4f}, {:.4f}], quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                current_pos[0], current_pos[1], current_pos[2],
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
            )
        )
        if relative_only:
            offset = [float(value) for value in args.relative_tool_translation_base]
            goal = [current_pos[i] + offset[i] for i in range(3)]
            result = plan_and_maybe_execute_motion(
                node,
                args,
                {"step": "relative", "action": "relative_translate"},
                "relative_translate",
                goal,
                current_quat,
                current_pos,
                "Relative tool translation base_offset={} -> tool0_goal={}".format(
                    [round(v, 4) for v in offset],
                    [round(v, 4) for v in goal],
                ),
                cartesian=args.cartesian,
                start_joint_state=planning_start_state,
            )
            if not result:
                return 2
            return 0

        if push_only:
            if not run_push_plan(node, args, planning_start_state):
                return 2
            return 0

        if args.hover_only:
            if not run_hover_only(node, args, planning_start_state):
                return 2
            return 0

        for step in steps:
            if step.get("action") == "place_on_top":
                node.moveit2.max_velocity = min(float(args.velocity), float(args.place_on_top_velocity))
                node.moveit2.max_acceleration = min(float(args.acceleration), float(args.place_on_top_acceleration))
            else:
                node.moveit2.max_velocity = args.velocity
                node.moveit2.max_acceleration = args.acceleration
            if step.get("diagnostic_yaw_deg") is not None:
                print(
                    "\nDIAGNOSTIC YAW {:.2f} deg: measure gripper-center error before continuing.".format(
                        step["diagnostic_yaw_deg"]
                    ),
                    flush=True,
                )
            commands = command_sequence_for_step(
                step,
                args.path_mode,
                args.enable_gripper,
                release_after_pick=args.release_after_pick,
            )
            if not commands:
                node.get_logger().warning("Step {} has no executable commands.".format(step.get("step")))
                continue
            step_quat_xyzw = None
            step_selected_yaw_deg = None
            step_orientation_source = None
            pre_rotated = bool(step.get("preserve_current_yaw"))
            step_start_state = planning_start_state

            for command in commands:
                if command["type"] == "wait":
                    print(
                        "Step {} {} wait {:.2f}s after release".format(
                            step.get("step"),
                            step.get("action"),
                            args.place_on_top_release_wait,
                        )
                    )
                    if args.execute:
                        time.sleep(float(args.place_on_top_release_wait))
                    continue
                if command["type"] == "gripper":
                    position = gripper_position_for_command(args, command["name"])
                    print(
                        "Step {} {} gripper {} -> position {}".format(
                            step.get("step"),
                            step.get("action"),
                            command["name"],
                            position,
                        )
                    )
                    if args.execute:
                        if gripper is None:
                            node.get_logger().warning("Gripper command skipped because --enable-gripper is not active.")
                            continue
                        if not maybe_confirm(
                            args,
                            "Execute step {} {} gripper {}?".format(
                                step.get("step"),
                                step.get("action"),
                                command["name"],
                            ),
                        ):
                            node.get_logger().info("Gripper command skipped by user.")
                            continue
                        gripper.set_position(position)
                        status = gripper.wait_until_done(timeout=args.gripper_wait)
                        current = gripper.get_position()
                        node.get_logger().info(
                            "Gripper {} done: status={}, position={}".format(command["name"], status, current)
                        )
                        if command["name"] == "close" and step.get("action") == "pick" and args.post_close_wait > 0:
                            node.get_logger().info(
                                "Waiting {:.2f}s after gripper close before retreat.".format(args.post_close_wait)
                            )
                            time.sleep(float(args.post_close_wait))
                    continue

                current_tool = node.current_tool_transform(timeout=args.tf_timeout)
                current_pos, current_quat = transform_position_quat(current_tool)
                pre_rotate_selects_orientation = (
                    args.pre_rotate_before_translation
                    and not pre_rotated
                    and args.orientation_mode == "object-yaw"
                )
                if step_quat_xyzw is None and not pre_rotate_selects_orientation:
                    step_quat_xyzw, step_selected_yaw_deg, step_orientation_source = orientation_for_step(
                        step, args, current_quat
                    )
                    node.get_logger().info(
                        "Step {} orientation source={} object_yaw={} selected_yaw={} quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                            step.get("step"),
                            step_orientation_source,
                            step.get("target_yaw_deg"),
                            step_selected_yaw_deg,
                            step_quat_xyzw[0],
                            step_quat_xyzw[1],
                            step_quat_xyzw[2],
                            step_quat_xyzw[3],
                        )
                    )
                if args.pre_rotate_before_translation and not pre_rotated:
                    pre_rotated = True
                    selected_pre_rotate_trajectory = None
                    if step_quat_xyzw is None:
                        motion_velocity = node.moveit2.max_velocity
                        motion_acceleration = node.moveit2.max_acceleration
                        node.moveit2.max_velocity = args.pre_rotate_velocity
                        node.moveit2.max_acceleration = args.pre_rotate_acceleration
                        try:
                            selected = select_best_pre_rotate_plan(
                                node,
                                args,
                                step,
                                current_pos,
                                current_quat,
                                start_joint_state=step_start_state,
                            )
                        finally:
                            node.moveit2.max_velocity = motion_velocity
                            node.moveit2.max_acceleration = motion_acceleration
                        if selected is None:
                            node.get_logger().error(
                                "Skipping step {} remaining motions because no pre-rotate candidate planned.".format(
                                    step.get("step")
                                )
                            )
                            break
                        selected_candidate, selected_pre_rotate_trajectory = selected
                        step_quat_xyzw = selected_candidate["quat_xyzw"]
                        step_selected_yaw_deg = selected_candidate["selected_yaw_deg"]
                        step_orientation_source = selected_candidate["source"]
                        node.get_logger().info(
                            "Step {} orientation source={} object_yaw={} selected_yaw={} quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                                step.get("step"),
                                step_orientation_source,
                                step.get("target_yaw_deg"),
                                step_selected_yaw_deg,
                                step_quat_xyzw[0],
                                step_quat_xyzw[1],
                                step_quat_xyzw[2],
                                step_quat_xyzw[3],
                            )
                        )
                    quat_xyzw = step_quat_xyzw
                    pre_log = (
                        "Step {} {} pre_rotate: object={} current_tool0={} -> quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]"
                        .format(
                            step.get("step"),
                            step.get("action"),
                            step.get("object_label"),
                            [round(v, 4) for v in current_pos],
                            quat_xyzw[0],
                            quat_xyzw[1],
                            quat_xyzw[2],
                            quat_xyzw[3],
                        )
                    )
                    pre_rotate_result = plan_and_maybe_execute_motion(
                        node,
                        args,
                        step,
                        "pre_rotate",
                        current_pos,
                        quat_xyzw,
                        current_pos,
                        pre_log,
                        cartesian=False,
                        trajectory=selected_pre_rotate_trajectory,
                        start_joint_state=step_start_state,
                        max_joint_delta_limit=args.max_pre_rotate_joint_delta,
                    )
                    if not pre_rotate_result:
                        node.get_logger().error(
                            "Skipping step {} remaining motions because pre-rotate did not complete.".format(
                                step.get("step")
                            )
                        )
                        break
                    if isinstance(pre_rotate_result, JointState):
                        step_start_state = pre_rotate_result
                    if args.execute:
                        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
                        current_pos, current_quat = transform_position_quat(current_tool)
                        step_start_state = node.latest_joint_state
                        if step_selected_yaw_deg is not None:
                            actual_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
                            yaw_error = (
                                abs(shortest_yaw_delta_deg(step_selected_yaw_deg, actual_yaw))
                                if step.get("exact_tool_yaw_required")
                                else gripper_yaw_error_deg(step_selected_yaw_deg, actual_yaw)
                            )
                            node.get_logger().info(
                                "Pre-rotate yaw verification: target_yaw={:.2f} actual_tcp_yaw={:.2f} "
                                "yaw_error_deg={:.2f}".format(
                                    step_selected_yaw_deg,
                                    actual_yaw,
                                    yaw_error,
                                )
                            )
                            if yaw_error > float(args.max_grasp_yaw_error_deg):
                                raise RuntimeError(
                                    "Refusing translation after pre-rotate: yaw_error_deg {:.2f} exceeds {:.2f}. "
                                    "Check --pre-rotate-wrist-yaw-sign.".format(
                                        yaw_error, args.max_grasp_yaw_error_deg
                                    )
                                )

                quat_xyzw = step_quat_xyzw
                plan_position = command["position"]
                if (
                    args.execute
                    and step.get("action") == "pick"
                    and command["name"] == "target"
                    and step_selected_yaw_deg is not None
                ):
                    current_pos, current_quat, settled_state = settle_orientation_before_descent(
                        node,
                        args,
                        step,
                        step_quat_xyzw,
                        current_pos,
                        current_quat,
                    )
                    if settled_state is not None:
                        step_start_state = settled_state
                    orientation_error_deg = math.degrees(
                        quaternion_distance_rad(current_quat, step_quat_xyzw)
                    )
                    current_tcp_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
                    yaw_error = (
                        abs(shortest_yaw_delta_deg(step_selected_yaw_deg, current_tcp_yaw))
                        if step.get("exact_tool_yaw_required")
                        else gripper_yaw_error_deg(step_selected_yaw_deg, current_tcp_yaw)
                    )
                    print(
                        "Pick yaw gate: object_yaw={:.2f} grasp_yaw={:.2f} current_tcp_yaw={:.2f} "
                        "yaw_error_deg={:.2f} orientation_error_deg={:.2f}".format(
                            float(step.get("object_yaw_deg", step.get("estimated_yaw_deg", step_selected_yaw_deg))),
                            float(step_selected_yaw_deg),
                            float(current_tcp_yaw),
                            float(yaw_error),
                            float(orientation_error_deg),
                        ),
                        flush=True,
                    )
                    if yaw_error > float(args.max_grasp_yaw_error_deg):
                        raise RuntimeError(
                            "Refusing pick descent: yaw_error_deg {:.2f} exceeds {:.2f}.".format(
                                yaw_error, args.max_grasp_yaw_error_deg
                            )
                        )
                    if orientation_error_deg > float(args.max_grasp_orientation_error_deg):
                        raise RuntimeError(
                            "Refusing pick descent: full orientation_error_deg {:.2f} exceeds {:.2f}. "
                            "Gripper plane is not level enough.".format(
                                orientation_error_deg,
                                args.max_grasp_orientation_error_deg,
                            )
                        )
                tcp_target = add_base_offset(plan_position, args.tcp_target_offset_base)
                tool_goal = tool0_goal_from_tcp(tcp_target, quat_xyzw, tcp_offset_tool)

                log_message = (
                    "Step {} {} {}: object={} ref={} plan_position={} tcp_target={} tcp_offset_tool={} -> tool0_goal={}".format(
                        step.get("step"),
                        step.get("action"),
                        command["name"],
                        step.get("object_label"),
                        step.get("reference_label"),
                        [round(v, 4) for v in plan_position],
                        [round(v, 4) for v in tcp_target],
                        [round(v, 4) for v in tcp_offset_tool],
                        [round(v, 4) for v in tool_goal],
                    )
                )
                motion_result = plan_and_maybe_execute_motion(
                    node,
                    args,
                    step,
                    command["name"],
                    tool_goal,
                    quat_xyzw,
                    current_pos,
                    log_message,
                    start_joint_state=step_start_state,
                )
                if not motion_result:
                    continue
                if isinstance(motion_result, JointState):
                    step_start_state = motion_result
            planning_start_state = step_start_state
    finally:
        if gripper is not None:
            gripper.close()
        rclpy.shutdown()
