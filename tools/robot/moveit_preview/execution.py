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
    tool0_goal_from_tcp,
    transform_position_quat,
    validate_goal,
    xyz_delta,
)
from .trajectory import (
    joint_start_goal_delta,
    joint_position_map,
    joint_position_map_from_state,
    joint_state_from_trajectory,
    max_joint_delta,
    max_joint_start_goal_delta,
    maybe_confirm,
    print_trajectory_summary,
    stretch_trajectory_timing,
    unwrap_continuous_joint_trajectory,
    weighted_joint_start_goal_cost,
)


def plan_motion_trajectory(node, args, tool_goal, quat_xyzw, cartesian=None, start_joint_state=None):
    validate_goal(tool_goal, args)
    # A post-grasp translation must not apply the normal command calibration a
    # second time: quat_xyzw is already the measured tool0 orientation that we
    # intend to preserve exactly.
    command_quat = (
        normalize_quaternion_xyzw(quat_xyzw)
        if bool(getattr(args, "hover_preserve_current_orientation", False))
        else orientation_command_quaternion(args, quat_xyzw)
    )
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
            wrist_delta = joint_start_goal_delta(trajectory, "wrist_3_joint")
            if (
                wrist_delta is not None
                and wrist_delta > float(args.max_wrist_3_start_goal_delta)
            ):
                node.get_logger().warning(
                    "Pre-rotate joint candidate rejected: wrist_3 start_goal_delta "
                    "{:.3f} rad > {:.3f}; {}".format(
                        wrist_delta,
                        float(args.max_wrist_3_start_goal_delta),
                        label,
                    )
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
            cost = weighted_joint_start_goal_cost(trajectory)
            planned.append(
                (
                    float(cost if cost is not None else delta_sort),
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
    best_cost, best_candidate, best_trajectory = planned[0]
    node.get_logger().info(
        "Selected pre-rotate joint candidate weighted_joint_delta_cost={:.3f} {} from {} candidates".format(
            best_cost,
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
        wrist_delta = joint_start_goal_delta(trajectory, "wrist_3_joint")
        if (
            wrist_delta is not None
            and wrist_delta > float(args.max_wrist_3_start_goal_delta)
        ):
            node.get_logger().warning(
                "Pre-rotate candidate {}/{} rejected: wrist_3 start_goal_delta "
                "{:.3f} rad > {:.3f} {}".format(
                    index,
                    len(candidates),
                    wrist_delta,
                    float(args.max_wrist_3_start_goal_delta),
                    candidate["label"],
                )
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
        cost = weighted_joint_start_goal_cost(trajectory)
        planned.append((float(cost if cost is not None else delta_sort), candidate, trajectory))

    if not planned:
        return None
    planned.sort(key=lambda item: item[0])
    best_cost, best_candidate, best_trajectory = planned[0]
    node.get_logger().info(
        "Selected pre-rotate candidate weighted_joint_delta_cost={:.3f} {}".format(
            best_cost,
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

    # A dense trajectory can hide a remote IK branch: every adjacent sample is
    # small even though one or more joints make an almost complete revolution.
    # Apply the same safety limit to the complete unwrapped start-to-goal
    # travel, not only to adjacent samples and wrist_3.
    start_goal_delta = max_joint_start_goal_delta(trajectory)
    if start_goal_delta:
        start_goal_joint, start_goal_value = start_goal_delta
        start_goal_limit = (
            max(joint_delta_limit, float(args.max_wrist_3_start_goal_delta))
            if start_goal_joint == "wrist_3_joint"
            else joint_delta_limit
        )
        if start_goal_value > start_goal_limit:
            delta_too_large = True
            node.get_logger().warning(
                "Large joint start-goal travel detected: {} delta={:.3f} rad > {:.3f}; "
                "rejecting remote IK branch.".format(
                    start_goal_joint,
                    start_goal_value,
                    start_goal_limit,
                )
            )

    wrist_3_start_goal_delta = joint_start_goal_delta(trajectory, "wrist_3_joint")
    wrist_3_delta_too_large = bool(
        wrist_3_start_goal_delta is not None
        and wrist_3_start_goal_delta > float(args.max_wrist_3_start_goal_delta)
    )
    if wrist_3_delta_too_large:
        node.get_logger().warning(
            "wrist_3 start_goal_delta {:.3f} rad > {:.3f}; rejecting IK/plan branch.".format(
                wrist_3_start_goal_delta,
                float(args.max_wrist_3_start_goal_delta),
            )
        )

    if not args.execute:
        if delta_too_large or wrist_3_delta_too_large:
            return False
        return joint_state_from_trajectory(trajectory) or True
    if delta_too_large or wrist_3_delta_too_large:
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
    position_error_m = math.sqrt(sum(float(value) ** 2 for value in err))
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
    if position_error_m > float(args.max_final_position_error_m):
        node.get_logger().error(
            "Executed motion final position error {:.4f} m exceeds {:.4f} m; "
            "treating the stage as failed.".format(
                position_error_m,
                float(args.max_final_position_error_m),
            )
        )
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
    tcp_offset_tool = [float(value) for value in args.tcp_offset_tool]
    if args.execute and abs(tcp_offset_tool[2]) < 0.08:
        raise RuntimeError(
            "Hover-only refused: TCP tool Z offset {:.3f} m is too small. "
            "Use the same --tcp-offset-tool as real grasp execution.".format(tcp_offset_tool[2])
        )
    current_tool = node.current_tool_transform(timeout=args.tf_timeout)
    current_pos, current_quat = transform_position_quat(current_tool)
    if bool(getattr(args, "hover_preserve_current_orientation", False)):
        hover_quat = normalize_quaternion_xyzw(current_quat)
        node.get_logger().info(
            "Hover-only fixed-posture mode: preserving the measured tool0 quaternion; "
            "pre-rotation and orientation settling are disabled."
        )
    else:
        hover_quat = normalize_quaternion_xyzw(args.hover_orientation_xyzw)
    tool_goal = tool0_goal_from_tcp(hover_target, hover_quat, tcp_offset_tool)
    validate_goal(tool_goal, args)
    orientation_error_rad = quaternion_distance_rad(current_quat, hover_quat)
    orientation_error_deg = math.degrees(orientation_error_rad)
    node.get_logger().info("Hover-only current_tool0_pose={}".format(pose_payload(current_pos, current_quat)))
    node.get_logger().info(
        "Hover-only tcp_target_base={} tcp_offset_tool={} -> tool0_goal={} target_orientation={} orientation_error_deg={:.2f}".format(
            [round(v, 4) for v in hover_target],
            [round(v, 4) for v in tcp_offset_tool],
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
            target_yaw = estimate_downward_family_yaw_deg(hover_quat, args.quat_xyzw)
            equivalent_step = {
                "target_yaw_deg": target_yaw,
                "target_yaw_valid": True,
                "parallel_gripper_axis_equivalent": True,
                "yaw_equivalence_period_deg": 180.0,
            }
            if args.pre_rotate_strategy == "joint-wrist3":
                selected = select_best_joint_wrist3_pre_rotate_plan(
                    node,
                    args,
                    equivalent_step,
                    current_quat,
                    start_joint_state=start_state,
                )
            else:
                selected = select_best_pose_pre_rotate_plan(
                    node,
                    args,
                    equivalent_step,
                    safe_position,
                    current_quat,
                    start_joint_state=start_state,
                )
            if selected is None:
                node.get_logger().error(
                    "Hover-only stopped because no safe 180-degree equivalent pre-rotate candidate planned."
                )
                return False
            selected_candidate, selected_pre_rotate_trajectory = selected
            hover_quat = selected_candidate["quat_xyzw"]
            target_yaw = float(selected_candidate["selected_yaw_deg"])
            tool_goal = tool0_goal_from_tcp(hover_target, hover_quat, tcp_offset_tool)
            node.get_logger().info(
                "Hover-only selected parallel-gripper equivalent: {}".format(
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
    if (
        args.execute
        and not bool(getattr(args, "hover_preserve_current_orientation", False))
        and not bool(getattr(args, "hover_disable_orientation_settle", False))
    ):
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
    start_map = joint_position_map_from_state(start_joint_state or node.latest_joint_state)
    wrist_delta = _joint_start_goal_delta(
        start_map, joint_names, joint_positions, "wrist_3_joint",
    )
    if wrist_delta is not None and wrist_delta > args.max_wrist_3_start_goal_delta:
        node.get_logger().error(
            "Joint motion {} rejected: wrist_3 start_goal_delta {:.3f} rad > {:.3f}.".format(
                motion_name, wrist_delta, args.max_wrist_3_start_goal_delta,
            )
        )
        return False
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


def stage_wrist_3_before_ready(
    node,
    args,
    start_joint_state,
    ready_joint_names,
    ready_joint_positions,
):
    """Split a large empty-gripper wrist return into bounded safe-height stages."""
    current_state = start_joint_state
    ready = dict(zip(ready_joint_names, ready_joint_positions))
    max_step = min(
        0.9 * float(args.max_joint_delta),
        0.9 * float(args.max_wrist_3_start_goal_delta),
    )
    if max_step <= 0.0:
        return False
    stage_index = 0
    while True:
        current = joint_position_map_from_state(current_state)
        if "wrist_3_joint" not in current or "wrist_3_joint" not in ready:
            return current_state
        delta = float(ready["wrist_3_joint"]) - float(current["wrist_3_joint"])
        if abs(delta) <= float(args.max_wrist_3_start_goal_delta):
            return current_state
        stage_index += 1
        stage_names = list(ready_joint_names)
        stage_positions = [float(current[name]) for name in stage_names]
        wrist_index = stage_names.index("wrist_3_joint")
        stage_positions[wrist_index] += math.copysign(max_step, delta)
        result = plan_and_maybe_execute_joint_motion(
            node,
            args,
            "ready_wrist_3_stage_{}".format(stage_index),
            stage_names,
            stage_positions,
            start_joint_state=current_state,
        )
        if not result or not hasattr(result, "name"):
            return False
        current_state = result


def _joint_start_goal_delta(
    start_map,
    joint_names,
    joint_positions,
    target_joint_name,
):
    if target_joint_name not in start_map or target_joint_name not in joint_names:
        return None
    index = list(joint_names).index(target_joint_name)
    return abs(float(joint_positions[index]) - float(start_map[target_joint_name]))
