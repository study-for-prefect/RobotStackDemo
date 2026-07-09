"""Yaw verification and in-place repair after pre-rotation."""

from typing import Any, Callable, Iterable, Optional, Tuple

from sensor_msgs.msg import JointState

from .orientation import (
    estimate_downward_family_yaw_deg,
    gripper_yaw_error_deg,
    shortest_yaw_delta_deg,
    transform_position_quat,
)


def pre_rotate_yaw_error_deg(
    step: dict,
    selected_yaw_deg: Optional[float],
    actual_quat: Iterable[float],
    args: Any,
) -> Tuple[Optional[float], Optional[float]]:
    if selected_yaw_deg is None:
        return None, None
    actual_yaw = estimate_downward_family_yaw_deg(actual_quat, args.quat_xyzw)
    if step.get("exact_tool_yaw_required"):
        error = abs(shortest_yaw_delta_deg(selected_yaw_deg, actual_yaw))
    else:
        error = gripper_yaw_error_deg(selected_yaw_deg, actual_yaw)
    return actual_yaw, error


def repair_pre_rotate_yaw_if_needed(
    node: Any,
    args: Any,
    step: dict,
    target_quat: Iterable[float],
    selected_yaw_deg: Optional[float],
    current_pos: Iterable[float],
    current_quat: Iterable[float],
    start_state: Any,
    select_pose_pre_rotate_plan: Callable[..., Any],
    execute_motion: Callable[..., Any],
) -> Tuple[bool, Iterable[float], Iterable[float], Any]:
    threshold = float(args.max_grasp_yaw_error_deg)
    actual_yaw, yaw_error = pre_rotate_yaw_error_deg(step, selected_yaw_deg, current_quat, args)
    if yaw_error is None:
        return True, current_pos, current_quat, start_state
    node.get_logger().info(
        "Pre-rotate yaw verification: target_yaw={:.2f} actual_tcp_yaw={:.2f} yaw_error_deg={:.2f}".format(
            float(selected_yaw_deg),
            float(actual_yaw),
            float(yaw_error),
        )
    )
    if yaw_error <= threshold:
        return True, current_pos, current_quat, start_state

    attempts = max(0, int(getattr(args, "pre_rotate_yaw_repair_attempts", 1)))
    for attempt in range(attempts):
        node.get_logger().warning(
            "Pre-rotate yaw repair {}/{}: yaw_error_deg {:.2f} > {:.2f}; planning in-place pose correction.".format(
                attempt + 1,
                attempts,
                yaw_error,
                threshold,
            )
        )
        repair_quat, repair_trajectory = _plan_pose_repair(
            node,
            args,
            step,
            target_quat,
            current_pos,
            current_quat,
            start_state,
            select_pose_pre_rotate_plan,
        )
        result = execute_motion(
            node,
            args,
            step,
            "pre_rotate_yaw_repair_{}".format(attempt + 1),
            current_pos,
            repair_quat,
            current_pos,
            "In-place pre-rotate yaw repair",
            cartesian=False,
            trajectory=repair_trajectory,
            start_joint_state=start_state,
            max_joint_delta_limit=args.max_pre_rotate_joint_delta,
        )
        if not result:
            return False, current_pos, current_quat, start_state
        if isinstance(result, JointState):
            start_state = result
        if args.execute:
            current_tool = node.current_tool_transform(timeout=args.tf_timeout)
            current_pos, current_quat = transform_position_quat(current_tool)
            start_state = node.latest_joint_state
        actual_yaw, yaw_error = pre_rotate_yaw_error_deg(step, selected_yaw_deg, current_quat, args)
        node.get_logger().info(
            "Pre-rotate yaw repair result: target_yaw={:.2f} actual_tcp_yaw={:.2f} yaw_error_deg={:.2f}".format(
                float(selected_yaw_deg),
                float(actual_yaw),
                float(yaw_error),
            )
        )
        if yaw_error <= threshold:
            return True, current_pos, current_quat, start_state
    return False, current_pos, current_quat, start_state


def _plan_pose_repair(
    node: Any,
    args: Any,
    step: dict,
    target_quat: Iterable[float],
    current_pos: Iterable[float],
    current_quat: Iterable[float],
    start_state: Any,
    select_pose_pre_rotate_plan: Callable[..., Any],
) -> Tuple[Iterable[float], Any]:
    motion_velocity = node.moveit2.max_velocity
    motion_acceleration = node.moveit2.max_acceleration
    node.moveit2.max_velocity = args.pre_rotate_velocity
    node.moveit2.max_acceleration = args.pre_rotate_acceleration
    try:
        selected = select_pose_pre_rotate_plan(
            node,
            args,
            step,
            current_pos,
            current_quat,
            start_joint_state=start_state,
        )
    finally:
        node.moveit2.max_velocity = motion_velocity
        node.moveit2.max_acceleration = motion_acceleration
    if selected is None:
        return target_quat, None
    candidate, repair_trajectory = selected
    return candidate["quat_xyzw"], repair_trajectory
