"""MoveIt-backed execution of a validated four-stage push-clearing plan."""

import json
from typing import Any, Dict, List, Tuple

from sensor_msgs.msg import JointState

from robot_scene_pipeline.grasp_orientation import normalize_quaternion_xyzw
from tools.robot.push_primitives import build_push_targets

from .execution import plan_and_maybe_execute_motion, plan_motion_trajectory
from .orientation import tool0_goal_from_tcp, transform_position_quat
from .trajectory import gripper_position_for_command, joint_state_from_trajectory, max_joint_delta


def load_push_plan(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _push_stage_goals(push_plan: dict, args: Any) -> List[Tuple[str, List[float], bool]]:
    targets = build_push_targets(push_plan)
    push_quat = normalize_quaternion_xyzw(args.quat_xyzw)
    tcp_offset_tool = [float(value) for value in args.tcp_offset_tool]
    stages = [
        ("pre_push", targets["pre_push"], False),
        ("contact", targets["contact"], True),
        ("push_end", targets["push_end"], True),
        ("retreat", targets["retreat"], True),
    ]
    return [
        (
            name,
            tool0_goal_from_tcp(target, push_quat, tcp_offset_tool),
            cartesian,
        )
        for name, target, cartesian in stages
    ]


def _set_gripper(node: Any, args: Any, gripper: Any, command: str) -> bool:
    if gripper is None:
        node.get_logger().error("Push gripper {} requested but gripper is not initialized.".format(command))
        return False
    position = gripper_position_for_command(args, command)
    gripper.set_position(position)
    status = gripper.wait_until_done(timeout=args.gripper_wait)
    current = gripper.get_position()
    node.get_logger().info(
        "Push gripper {} done: status={}, position={}".format(command, status, current)
    )
    return bool(status)


def _recover_open_gripper(node: Any, args: Any, gripper: Any, reason: str) -> None:
    if gripper is None:
        return
    node.get_logger().warning("Opening gripper after push failure: {}".format(reason))
    _set_gripper(node, args, gripper, "open")


def run_push_plan(node: Any, args: Any, planning_start_state: Any, gripper: Any = None) -> bool:
    """Preflight all four MoveIt trajectories, then execute them once in order."""
    push_plan = load_push_plan(args.push_plan_json)
    gripper_closed_for_push = False
    if args.execute and getattr(args, "close_gripper_for_push", False):
        if not _set_gripper(node, args, gripper, "close"):
            node.get_logger().error("failed_before_motion: could not close gripper for rigid-paddle push.")
            return False
        gripper_closed_for_push = True
        if not node.wait_for_joint_state(timeout=2.0):
            _recover_open_gripper(node, args, gripper, "failed_before_motion_no_joint_state_after_gripper_close")
            return False
        planning_start_state = node.latest_joint_state

    try:
        stage_goals = _push_stage_goals(push_plan, args)
        push_quat = normalize_quaternion_xyzw(args.quat_xyzw)
        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, _current_quat = transform_position_quat(current_tool)

        preflight = []
        start_state = planning_start_state
        previous_position = current_pos
        for stage_name, tool_goal, cartesian in stage_goals:
            trajectory = plan_motion_trajectory(
                node,
                args,
                tool_goal,
                push_quat,
                cartesian=cartesian,
                start_joint_state=start_state,
            )
            if trajectory is None:
                if gripper_closed_for_push:
                    _recover_open_gripper(node, args, gripper, "failed_before_motion_preflight_stage_{}".format(stage_name))
                node.get_logger().error(
                    "failed_before_motion: Push clearing preflight failed to plan stage '{}'.".format(stage_name)
                )
                return False
            delta = max_joint_delta(trajectory)
            if delta and delta[1] > float(args.max_joint_delta):
                if gripper_closed_for_push:
                    _recover_open_gripper(node, args, gripper, "failed_before_motion_joint_delta_stage_{}".format(stage_name))
                node.get_logger().error(
                    "Push clearing preflight refused stage '{}': {} delta {:.3f} rad "
                    "exceeds {:.3f}.".format(
                        stage_name,
                        delta[0],
                        delta[1],
                        float(args.max_joint_delta),
                    )
                )
                return False
            preflight.append(
                (stage_name, tool_goal, cartesian, trajectory, start_state, previous_position)
            )
            start_state = joint_state_from_trajectory(trajectory) or start_state
            previous_position = tool_goal
    except Exception as exc:
        if gripper_closed_for_push:
            _recover_open_gripper(node, args, gripper, "failed_before_motion_exception")
        node.get_logger().error("failed_before_motion: push preflight exception: {}".format(exc))
        return False

    node.get_logger().info(
        "Push clearing preflight passed for stages: {}".format(
            ", ".join(item[0] for item in preflight)
        )
    )
    for stage_name, tool_goal, cartesian, trajectory, start_state, previous_position in preflight:
        result = plan_and_maybe_execute_motion(
            node,
            args,
            {
                "step": "push",
                "action": "push_clearing",
                "object_id": push_plan.get("obstacle_object_id"),
            },
            stage_name,
            tool_goal,
            push_quat,
            previous_position,
            "Push clearing stage {} -> tool0_goal={}".format(
                stage_name,
                [round(value, 4) for value in tool_goal],
            ),
            cartesian=cartesian,
            trajectory=trajectory,
            start_joint_state=start_state,
        )
        if not result:
            if gripper_closed_for_push:
                _recover_open_gripper(node, args, gripper, "failed_during_motion_stage_{}".format(stage_name))
            node.get_logger().error("failed_during_motion: push stage '{}' did not complete.".format(stage_name))
            return False
        if isinstance(result, JointState):
            planning_start_state = result
    if gripper_closed_for_push:
        if not _set_gripper(node, args, gripper, "open"):
            node.get_logger().error("failed_after_motion: gripper open recovery failed after push success.")
            return False
    return True
