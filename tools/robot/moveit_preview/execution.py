"""MoveIt planning and optional execution primitives."""

import math
import time

from pymoveit2.robots import ur
from sensor_msgs.msg import JointState

from robot_scene_pipeline.grasp_orientation import (
    downward_quaternion_for_yaw,
    normalize_quaternion_xyzw,
    quaternion_distance_rad,
)

from .orientation import (
    estimate_downward_family_yaw_deg,
    make_pose,
    object_yaw_candidate_values,
    orientation_candidates_for_pre_rotate,
    orientation_command_quaternion,
    pose_payload,
    shortest_yaw_delta_deg,
    transform_position_quat,
    validate_goal,
    xyz_delta,
)
from .trajectory import (
    joint_position_map,
    joint_state_from_trajectory,
    max_joint_delta,
    maybe_confirm,
    print_trajectory_summary,
    stretch_trajectory_timing,
    unwrap_continuous_joint_trajectory,
)

def plan_motion_trajectory(node, args, tool_goal, quat_xyzw, cartesian=None, start_joint_state=None):
    validate_goal(tool_goal, args)
    command_quat = orientation_command_quaternion(args, quat_xyzw)
    pose = make_pose(tool_goal, command_quat)
    node.get_logger().info(
        "Orientation target={} command_after_compensation={}".format(
            pose_payload(tool_goal, quat_xyzw),
            pose_payload(tool_goal, command_quat),
        )
    )
    trajectory = node.moveit2.plan(
        pose=pose,
        cartesian=args.cartesian if cartesian is None else cartesian,
        max_step=args.cartesian_max_step,
        cartesian_fraction_threshold=args.cartesian_fraction_threshold,
        start_joint_state=start_joint_state if start_joint_state is not None else node.latest_joint_state,
    )
    if trajectory is None:
        return None
    unwrap_continuous_joint_trajectory(trajectory, joint_position_map(node, start_joint_state))
    stretch_trajectory_timing(
        trajectory,
        min_duration=args.min_trajectory_duration,
        min_point_dt=args.min_point_dt,
    )
    return trajectory


def plan_joint_trajectory(node, args, joint_positions, joint_names, start_joint_state=None):
    trajectory = node.moveit2.plan(
        joint_positions=joint_positions,
        joint_names=joint_names,
        start_joint_state=start_joint_state if start_joint_state is not None else node.latest_joint_state,
    )
    if trajectory is None:
        return None
    unwrap_continuous_joint_trajectory(trajectory, joint_position_map(node, start_joint_state))
    stretch_trajectory_timing(
        trajectory,
        min_duration=args.min_trajectory_duration,
        min_point_dt=args.min_point_dt,
    )
    return trajectory


def wrist_yaw_sign_values(args):
    if args.pre_rotate_wrist_yaw_sign == "positive":
        return [1.0]
    if args.pre_rotate_wrist_yaw_sign == "negative":
        return [-1.0]
    return [1.0, -1.0]


def wrist_direction_allowed(args, wrist_delta):
    if args.pre_rotate_wrist_direction == "positive":
        return wrist_delta > 0.0
    if args.pre_rotate_wrist_direction == "negative":
        return wrist_delta < 0.0
    return True


def select_best_joint_wrist3_pre_rotate_plan(node, args, step, current_quat, start_joint_state=None):
    current_positions = joint_position_map(node, start_joint_state)
    joint_names = ur.joint_names()
    missing = [name for name in joint_names if name not in current_positions]
    if missing:
        raise RuntimeError("Missing current joint positions for pre-rotate: {}".format(", ".join(missing)))
    current_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
    yaws = object_yaw_candidate_values(step, args)
    planned = []
    total_candidates = len(yaws) * len(wrist_yaw_sign_values(args))
    for yaw in yaws:
        yaw_delta_deg = shortest_yaw_delta_deg(yaw, current_yaw)
        for wrist_sign in wrist_yaw_sign_values(args):
            target_positions = [current_positions[name] for name in joint_names]
            wrist_index = joint_names.index("wrist_3_joint")
            wrist_delta = wrist_sign * math.radians(yaw_delta_deg)
            if not wrist_direction_allowed(args, wrist_delta):
                continue
            target_positions[wrist_index] += wrist_delta
            label = (
                "joint_wrist3 current_yaw={:.2f} selected_yaw={:.2f} yaw_delta={:.2f} "
                "wrist_sign={:+.0f} wrist_delta={:.3f} rad"
            ).format(current_yaw, yaw, yaw_delta_deg, wrist_sign, wrist_delta)
            trajectory = plan_joint_trajectory(
                node,
                args,
                target_positions,
                joint_names,
                start_joint_state=start_joint_state,
            )
            if trajectory is None:
                node.get_logger().warning(
                    "Pre-rotate joint candidate failed: {}".format(label)
                )
                continue
            delta = max_joint_delta(trajectory)
            if delta is None:
                delta_text = "none"
                delta_sort = 0.0
            else:
                delta_text = "{} {:.3f} rad".format(delta[0], delta[1])
                delta_sort = float(delta[1])
            node.get_logger().info(
                "Pre-rotate joint candidate max_delta={} {}".format(delta_text, label)
            )
            quaternion = downward_quaternion_for_yaw(args.quat_xyzw, yaw)
            planned.append(
                (
                    delta_sort,
                    {
                        "quat_xyzw": quaternion,
                        "selected_yaw_deg": yaw,
                        "source": "object_yaw_joint_wrist3_selected",
                        "label": label,
                    },
                    trajectory,
                )
            )
    if not planned:
        return None
    planned.sort(key=lambda item: item[0])
    best_delta, best_candidate, best_trajectory = planned[0]
    node.get_logger().info(
        "Selected pre-rotate joint candidate max_delta={:.3f} {} from {} candidates".format(
            best_delta,
            best_candidate["label"],
            total_candidates,
        )
    )
    return best_candidate, best_trajectory


def select_best_pose_pre_rotate_plan(node, args, step, tool_goal, current_quat, start_joint_state=None):
    candidates = orientation_candidates_for_pre_rotate(step, args, current_quat)
    planned = []
    for index, candidate in enumerate(candidates, start=1):
        trajectory = plan_motion_trajectory(
            node,
            args,
            tool_goal,
            candidate["quat_xyzw"],
            cartesian=False,
            start_joint_state=start_joint_state,
        )
        if trajectory is None:
            node.get_logger().warning(
                "Pre-rotate candidate {}/{} failed: {}".format(index, len(candidates), candidate["label"])
            )
            continue
        delta = max_joint_delta(trajectory)
        if delta is None:
            delta_text = "none"
            delta_sort = 0.0
        else:
            delta_text = "{} {:.3f} rad".format(delta[0], delta[1])
            delta_sort = float(delta[1])
        node.get_logger().info(
            "Pre-rotate candidate {}/{} max_delta={} {}".format(
                index,
                len(candidates),
                delta_text,
                candidate["label"],
            )
        )
        planned.append((delta_sort, candidate, trajectory))

    if not planned:
        return None
    planned.sort(key=lambda item: item[0])
    best_delta, best_candidate, best_trajectory = planned[0]
    node.get_logger().info(
        "Selected pre-rotate candidate max_delta={:.3f} {}".format(
            best_delta,
            best_candidate["label"],
        )
    )
    return best_candidate, best_trajectory


def select_best_pre_rotate_plan(node, args, step, tool_goal, current_quat, start_joint_state=None):
    if step.get("preserve_current_yaw"):
        return select_best_pose_pre_rotate_plan(
            node,
            args,
            step,
            tool_goal,
            current_quat,
            start_joint_state=start_joint_state,
        )
    if args.pre_rotate_strategy == "joint-wrist3" and args.orientation_mode == "object-yaw":
        return select_best_joint_wrist3_pre_rotate_plan(
            node,
            args,
            step,
            current_quat,
            start_joint_state=start_joint_state,
        )
    return select_best_pose_pre_rotate_plan(
        node,
        args,
        step,
        tool_goal,
        current_quat,
        start_joint_state=start_joint_state,
    )


def plan_and_maybe_execute_motion(
    node,
    args,
    step,
    motion_name,
    tool_goal,
    quat_xyzw,
    current_pos,
    log_message,
    cartesian=None,
    trajectory=None,
    start_joint_state=None,
    max_joint_delta_limit=None,
):
    validate_goal(tool_goal, args)
    delta_xyz = [tool_goal[i] - current_pos[i] for i in range(3)]
    node.get_logger().info("{} delta_xyz={}".format(log_message, [round(v, 4) for v in delta_xyz]))

    if trajectory is None:
        trajectory = plan_motion_trajectory(
            node,
            args,
            tool_goal,
            quat_xyzw,
            cartesian=cartesian,
            start_joint_state=start_joint_state,
        )
    if trajectory is None:
        node.get_logger().error(
            "MoveIt planning failed for step {} {}.".format(step.get("step"), motion_name)
        )
        return False
    print_trajectory_summary(node, trajectory)
    delta = max_joint_delta(trajectory)
    delta_too_large = False
    joint_delta_limit = args.max_joint_delta if max_joint_delta_limit is None else float(max_joint_delta_limit)
    if delta:
        joint_name, value = delta
        if value > joint_delta_limit:
            delta_too_large = True
            node.get_logger().warning(
                "Large joint change detected: {} delta={:.3f} rad > {:.3f}. Do not execute until reviewed.".format(
                    joint_name,
                    value,
                    joint_delta_limit,
                )
            )

    if not args.execute:
        return joint_state_from_trajectory(trajectory) or True
    if delta_too_large:
        node.get_logger().error("Execution refused because joint delta exceeds safety threshold.")
        return False
    if not maybe_confirm(
        args,
        "Execute step {} {} motion {}?".format(
            step.get("step"),
            step.get("action"),
            motion_name,
        ),
    ):
        node.get_logger().info("Execution skipped by user.")
        return False
    node.moveit2.execute(trajectory)
    ok = node.moveit2.wait_until_executed()
    node.get_logger().info("Execution result: {}".format(ok))
    final_tool = node.current_tool_transform(timeout=args.tf_timeout)
    final_pos, final_quat = transform_position_quat(final_tool)
    err = xyz_delta(final_pos, tool_goal)
    orientation_error_deg = math.degrees(quaternion_distance_rad(final_quat, quat_xyzw))
    node.get_logger().info(
        "Final tool0 position={}, goal={}, error={}, orientation_error_deg={:.3f}".format(
            [round(v, 4) for v in final_pos],
            [round(v, 4) for v in tool_goal],
            [round(v, 4) for v in err],
            orientation_error_deg,
        )
    )
    node.get_logger().info(
        "Final orientation target={} actual={}".format(
            pose_payload(tool_goal, quat_xyzw),
            pose_payload(final_pos, final_quat),
        )
    )
    if not ok:
        return False
    return joint_state_from_trajectory(trajectory) or True


def settle_orientation_before_descent(node, args, step, target_quat, current_pos, current_quat):
    attempts = max(0, int(args.orientation_settle_attempts))
    threshold = max(0.0, float(args.orientation_settle_error_deg))
    start_state = node.latest_joint_state
    for attempt in range(attempts):
        error_deg = math.degrees(quaternion_distance_rad(current_quat, target_quat))
        if error_deg <= threshold:
            break
        node.get_logger().warning(
            "Pick orientation settle {}/{}: full error {:.3f} deg > {:.3f} deg.".format(
                attempt + 1,
                attempts,
                error_deg,
                threshold,
            )
        )
        result = plan_and_maybe_execute_motion(
            node,
            args,
            step,
            "orientation_settle_{}".format(attempt + 1),
            current_pos,
            target_quat,
            current_pos,
            "In-place orientation correction before descent",
            cartesian=False,
            start_joint_state=start_state,
            max_joint_delta_limit=args.max_pre_rotate_joint_delta,
        )
        if not result:
            break
        if args.execute and args.orientation_settle_wait_s > 0:
            time.sleep(float(args.orientation_settle_wait_s))
        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, current_quat = transform_position_quat(current_tool)
        start_state = node.latest_joint_state
    return current_pos, current_quat, start_state


def run_hover_only(node, args, planning_start_state):
    if args.hover_target_base is None:
        raise RuntimeError("--hover-only requires --hover-target-base X Y Z.")
    if args.hover_orientation_xyzw is None:
        raise RuntimeError("--hover-only requires --hover-orientation-xyzw QX QY QZ QW.")

    hover_target = [float(value) for value in args.hover_target_base]
    tool_offset = [float(value) for value in args.tool_offset_base]
    effective_tool_z_offset = float(args.tool_z_offset) + tool_offset[2]
    if args.execute and effective_tool_z_offset < 0.08:
        raise RuntimeError(
            "Hover-only refused: effective tool Z offset {:.3f} m is too small. "
            "Use the same --tool-z-offset as real grasp execution.".format(effective_tool_z_offset)
        )
    tool_goal = [
        hover_target[0] + tool_offset[0],
        hover_target[1] + tool_offset[1],
        hover_target[2] + effective_tool_z_offset,
    ]
    hover_quat = normalize_quaternion_xyzw(args.hover_orientation_xyzw)
    validate_goal(tool_goal, args)

    current_tool = node.current_tool_transform(timeout=args.tf_timeout)
    current_pos, current_quat = transform_position_quat(current_tool)
    orientation_error_rad = quaternion_distance_rad(current_quat, hover_quat)
    orientation_error_deg = math.degrees(orientation_error_rad)
    node.get_logger().info("Hover-only current_tool0_pose={}".format(pose_payload(current_pos, current_quat)))
    node.get_logger().info(
        "Hover-only target_base={} tool_z_offset={:.4f} tool_offset_base={} -> tool0_goal={} target_orientation={} orientation_error_deg={:.2f}".format(
            [round(v, 4) for v in hover_target],
            float(args.tool_z_offset),
            [round(v, 4) for v in tool_offset],
            [round(v, 4) for v in tool_goal],
            [round(v, 5) for v in hover_quat],
            orientation_error_deg,
        )
    )

    start_state = planning_start_state
    threshold_deg = float(args.max_grasp_yaw_error_deg)
    if orientation_error_deg > threshold_deg:
        safe_position = [
            current_pos[0],
            current_pos[1],
            max(float(current_pos[2]), float(tool_goal[2]), float(args.safe_pre_rotate_height)),
        ]
        node.get_logger().info(
            "Hover-only pre-rotate required: orientation_error_deg={:.2f} > {:.2f}; safe_position={}".format(
                orientation_error_deg,
                threshold_deg,
                [round(v, 4) for v in safe_position],
            )
        )
        if (
            args.pre_rotate_strategy == "joint-wrist3"
            and abs(safe_position[2] - current_pos[2]) > 1e-4
        ):
            lift_result = plan_and_maybe_execute_motion(
                node,
                args,
                {"step": "hover", "action": "hover_only"},
                "safe_pre_rotate_lift",
                safe_position,
                current_quat,
                current_pos,
                "Hover-only lift before wrist pre-rotate",
                start_joint_state=start_state,
            )
            if not lift_result:
                node.get_logger().error(
                    "Hover-only stopped because the safe pre-rotate lift did not complete."
                )
                return False
            if isinstance(lift_result, JointState):
                start_state = lift_result
            if args.execute:
                current_tool = node.current_tool_transform(timeout=args.tf_timeout)
                current_pos, current_quat = transform_position_quat(current_tool)
                start_state = node.latest_joint_state

        motion_velocity = node.moveit2.max_velocity
        motion_acceleration = node.moveit2.max_acceleration
        node.moveit2.max_velocity = args.pre_rotate_velocity
        node.moveit2.max_acceleration = args.pre_rotate_acceleration
        try:
            selected_pre_rotate_trajectory = None
            if args.pre_rotate_strategy == "joint-wrist3":
                target_yaw = estimate_downward_family_yaw_deg(hover_quat, args.quat_xyzw)
                selected = select_best_joint_wrist3_pre_rotate_plan(
                    node,
                    args,
                    {
                        "target_yaw_deg": target_yaw,
                        "target_yaw_valid": True,
                        "exact_tool_yaw_required": True,
                    },
                    current_quat,
                    start_joint_state=start_state,
                )
                if selected is None:
                    node.get_logger().error(
                        "Hover-only stopped because no safe wrist_3 pre-rotate candidate planned."
                    )
                    return False
                selected_candidate, selected_pre_rotate_trajectory = selected
                node.get_logger().info(
                    "Hover-only selected explicit-yaw pre-rotate: {}".format(
                        selected_candidate["label"]
                    )
                )
            pre_rotate_result = plan_and_maybe_execute_motion(
                node,
                args,
                {"step": "hover", "action": "hover_only"},
                "safe_pre_rotate",
                safe_position,
                hover_quat,
                current_pos,
                "Hover-only safe pre-rotate",
                cartesian=False,
                trajectory=selected_pre_rotate_trajectory,
                start_joint_state=start_state,
                max_joint_delta_limit=args.max_pre_rotate_joint_delta,
            )
        finally:
            node.moveit2.max_velocity = motion_velocity
            node.moveit2.max_acceleration = motion_acceleration
        if not pre_rotate_result:
            node.get_logger().error("Hover-only stopped because safe pre-rotate did not complete.")
            return False
        if isinstance(pre_rotate_result, JointState):
            start_state = pre_rotate_result
        if args.execute:
            current_tool = node.current_tool_transform(timeout=args.tf_timeout)
            current_pos, current_quat = transform_position_quat(current_tool)
            start_state = node.latest_joint_state
            if args.pre_rotate_strategy == "joint-wrist3":
                target_yaw = estimate_downward_family_yaw_deg(hover_quat, args.quat_xyzw)
                actual_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
                yaw_error = abs(shortest_yaw_delta_deg(target_yaw, actual_yaw))
                node.get_logger().info(
                    "Hover-only pre-rotate yaw verification: target_yaw={:.2f} "
                    "actual_tcp_yaw={:.2f} yaw_error_deg={:.2f}".format(
                        target_yaw,
                        actual_yaw,
                        yaw_error,
                    )
                )
                if yaw_error > float(args.max_grasp_yaw_error_deg):
                    node.get_logger().error(
                        "Hover-only translation refused: pre-rotate yaw error {:.2f} deg "
                        "exceeds {:.2f} deg. Check --pre-rotate-wrist-yaw-sign.".format(
                            yaw_error,
                            float(args.max_grasp_yaw_error_deg),
                        )
                    )
                    return False

    hover_result = plan_and_maybe_execute_motion(
        node,
        args,
        {"step": "hover", "action": "hover_only"},
        "hover_pose",
        tool_goal,
        hover_quat,
        current_pos,
        "Hover-only move to hover pose",
        start_joint_state=start_state,
    )
    if not hover_result:
        return False
    final_tool = node.current_tool_transform(timeout=args.tf_timeout)
    final_pos, final_quat = transform_position_quat(final_tool)
    if args.execute:
        final_pos, final_quat, _settled_state = settle_orientation_before_descent(
            node,
            args,
            {"step": "hover", "action": "hover_only"},
            hover_quat,
            final_pos,
            final_quat,
        )
    node.get_logger().info("Hover-only final_tool0_pose={}".format(pose_payload(final_pos, final_quat)))
    return True


def plan_and_maybe_execute_joint_motion(
    node,
    args,
    motion_name,
    joint_names,
    joint_positions,
    start_joint_state=None,
):
    trajectory = plan_joint_trajectory(
        node,
        args,
        joint_positions,
        joint_names,
        start_joint_state=start_joint_state,
    )
    if trajectory is None:
        node.get_logger().error("MoveIt planning failed for joint motion {}.".format(motion_name))
        return False
    node.get_logger().info(
        "{} joint goal: {}".format(
            motion_name,
            ["{}={:.4f}".format(name, position) for name, position in zip(joint_names, joint_positions)],
        )
    )
    print_trajectory_summary(node, trajectory)
    delta = max_joint_delta(trajectory)
    delta_too_large = False
    if delta:
        joint_name, value = delta
        if value > args.max_joint_delta:
            delta_too_large = True
            node.get_logger().warning(
                "Large joint change detected: {} delta={:.3f} rad > {:.3f}. Do not execute until reviewed.".format(
                    joint_name,
                    value,
                    args.max_joint_delta,
                )
            )
    if not args.execute:
        return joint_state_from_trajectory(trajectory) or True
    if delta_too_large:
        node.get_logger().error("Execution refused because joint delta exceeds safety threshold.")
        return False
    if not maybe_confirm(args, "Execute joint motion {}?".format(motion_name)):
        node.get_logger().info("Execution skipped by user.")
        return False
    node.moveit2.execute(trajectory)
    ok = node.moveit2.wait_until_executed()
    node.get_logger().info("Execution result: {}".format(ok))
    if not ok:
        return False
    return joint_state_from_trajectory(trajectory) or True
