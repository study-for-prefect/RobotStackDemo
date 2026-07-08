"""MoveIt-backed execution of a validated four-stage push-clearing plan."""

import json
import math
from typing import Any, Dict, List, Tuple

from sensor_msgs.msg import JointState

from robot_scene_pipeline.grasp_orientation import normalize_quaternion_xyzw, quaternion_distance_rad
from robot_scene_pipeline.grasp_orientation import downward_quaternion_for_yaw
from tools.robot.push_primitives import build_push_targets

from .execution import plan_and_maybe_execute_motion, plan_motion_trajectory
from .orientation import (
    estimate_downward_family_yaw_deg,
    normalize_yaw_deg,
    shortest_yaw_delta_deg,
    tool0_goal_from_tcp,
    transform_position_quat,
)
from .trajectory import (
    gripper_command_accepted,
    gripper_position_for_command,
    joint_state_from_trajectory,
    max_joint_delta,
)


def load_push_plan(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_float(value: Any) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError("Expected a finite float, got {!r}.".format(value))
    return output


def _push_target_yaw(push_plan: dict) -> Tuple[float, str]:
    for key, source in (
        ("target_yaw_deg", push_plan.get("target_yaw_source") or "target_yaw_deg"),
        ("selected_grasp_yaw_deg", "selected_grasp_yaw_deg"),
        ("target_table_yaw_deg", "target_table_yaw_deg"),
    ):
        if push_plan.get(key) is not None:
            return _finite_float(push_plan[key]), str(source)
    target = push_plan.get("target")
    if isinstance(target, dict):
        if target.get("selected_grasp_yaw_deg") is not None:
            return _finite_float(target["selected_grasp_yaw_deg"]), str(
                target.get("grasp_yaw_source") or "target.selected_grasp_yaw_deg"
            )
        if target.get("table_yaw_deg") is not None:
            return _finite_float(target["table_yaw_deg"]), str(
                target.get("table_yaw_source") or "target.table_yaw_deg"
            )
    raise ValueError("Push plan does not contain target yaw.")


def _resolve_push_orientation(push_plan: dict, args: Any, current_quat_xyzw: List[float]) -> Tuple[List[float], Dict[str, Any]]:
    current_quat = normalize_quaternion_xyzw(current_quat_xyzw)
    try:
        target_yaw, yaw_source = _push_target_yaw(push_plan)
    except (TypeError, ValueError):
        return current_quat, {
            "source": "current_tool_orientation_no_target_yaw",
            "target_yaw_deg": None,
            "target_yaw_source": "missing_target_yaw",
            "selected_yaw_deg": None,
            "current_yaw_deg": None,
            "yaw_delta_deg": 0.0,
        }

    current_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
    candidates = [target_yaw, target_yaw + 180.0, target_yaw - 180.0]
    selected_yaw = min(
        candidates,
        key=lambda yaw: abs(shortest_yaw_delta_deg(yaw, current_yaw)),
    )
    selected_yaw = normalize_yaw_deg(selected_yaw)
    yaw_delta = shortest_yaw_delta_deg(selected_yaw, current_yaw)
    return downward_quaternion_for_yaw(current_quat, yaw_delta), {
        "source": "current_tool_orientation_plus_target_yaw",
        "target_yaw_deg": normalize_yaw_deg(target_yaw),
        "target_yaw_source": yaw_source,
        "selected_yaw_deg": selected_yaw,
        "current_yaw_deg": current_yaw,
        "yaw_delta_deg": yaw_delta,
    }


def _push_stage_goals(push_plan: dict, args: Any, push_quat: List[float]) -> List[Tuple[str, List[float], bool]]:
    targets = build_push_targets(push_plan)
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
    accepted = gripper_command_accepted(args, command, status, current)
    if accepted and not status:
        node.get_logger().warning(
            "Push gripper close accepted by contact position: position={} target={}.".format(
                current,
                gripper_position_for_command(args, command),
            )
        )
    return accepted


def _recover_open_gripper(node: Any, args: Any, gripper: Any, reason: str) -> None:
    if gripper is None:
        return
    node.get_logger().warning("Opening gripper after push failure: {}".format(reason))
    _set_gripper(node, args, gripper, "open")


def _push_orientation_ok(node: Any, args: Any, target_quat_xyzw: List[float]) -> bool:
    current_tool = node.current_tool_transform(timeout=args.tf_timeout)
    _current_pos, current_quat = transform_position_quat(current_tool)
    error_deg = math.degrees(quaternion_distance_rad(current_quat, target_quat_xyzw))
    limit_deg = float(getattr(args, "max_grasp_orientation_error_deg", 0.5))
    node.get_logger().info(
        "Push pre-contact orientation gate: error_deg={:.3f} limit_deg={:.3f}".format(
            error_deg,
            limit_deg,
        )
    )
    return error_deg <= limit_deg


def run_push_plan(node: Any, args: Any, planning_start_state: Any, gripper: Any = None) -> bool:
    """Preflight all four MoveIt trajectories, then execute them once in order."""
    push_plan = load_push_plan(args.push_plan_json)
    gripper_closed_for_push = False
    close_at_pre_push = bool(args.execute and getattr(args, "close_gripper_for_push", False))

    try:
        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, current_quat = transform_position_quat(current_tool)
        push_quat, orientation_report = _resolve_push_orientation(push_plan, args, current_quat)
        node.get_logger().info(
            "Push orientation source={source} target_yaw={target_yaw_deg} "
            "target_yaw_source={target_yaw_source} current_yaw={current_yaw_deg} "
            "selected_yaw={selected_yaw_deg} yaw_delta={yaw_delta_deg} "
            "quat_xyzw=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]".format(
                qx=push_quat[0],
                qy=push_quat[1],
                qz=push_quat[2],
                qw=push_quat[3],
                **orientation_report,
            )
        )
        stage_goals = _push_stage_goals(push_plan, args, push_quat)

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
        if stage_name == "pre_push" and close_at_pre_push and not gripper_closed_for_push:
            if not _push_orientation_ok(node, args, push_quat):
                node.get_logger().error(
                    "failed_before_contact: push tool orientation is not level enough for descent."
                )
                return False
            if not _set_gripper(node, args, gripper, "close"):
                _recover_open_gripper(node, args, gripper, "failed_before_contact_gripper_close")
                node.get_logger().error("failed_before_contact: could not close gripper for rigid-paddle push.")
                return False
            gripper_closed_for_push = True
            if not node.wait_for_joint_state(timeout=2.0):
                _recover_open_gripper(node, args, gripper, "failed_before_contact_no_joint_state_after_gripper_close")
                return False
    if gripper_closed_for_push:
        if not _set_gripper(node, args, gripper, "open"):
            node.get_logger().error("failed_after_motion: gripper open recovery failed after push success.")
            return False
    return True
