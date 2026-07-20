"""Continuous full-SE(3) TCP waypoint planning for staged manipulation."""

from __future__ import annotations

import json
from typing import Any

from sensor_msgs.msg import JointState

from robot_scene_pipeline.grasp_orientation import normalize_quaternion_xyzw

from .execution import plan_and_maybe_execute_motion
from .orientation import tool0_goal_from_tcp, transform_position_quat


def load_pose_sequence(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "tcp_pose_sequence_v1":
        raise RuntimeError("Pose sequence must use schema_version tcp_pose_sequence_v1.")
    if payload.get("frame_id") != "base_link":
        raise RuntimeError("Pose sequence frame_id must be base_link.")
    raw_waypoints = payload.get("waypoints")
    if not isinstance(raw_waypoints, list) or not raw_waypoints:
        raise RuntimeError("Pose sequence must contain at least one waypoint.")
    waypoints = []
    for index, raw in enumerate(raw_waypoints, start=1):
        if not isinstance(raw, dict):
            raise RuntimeError("Pose sequence waypoint {} must be an object.".format(index))
        position = raw.get("position_m")
        orientation = raw.get("orientation_xyzw")
        if not isinstance(position, list) or len(position) != 3:
            raise RuntimeError("Pose sequence waypoint {} requires position_m[3].".format(index))
        if not isinstance(orientation, list) or len(orientation) != 4:
            raise RuntimeError(
                "Pose sequence waypoint {} requires the full orientation_xyzw[4]; yaw-only is refused."
                .format(index)
            )
        waypoints.append({
            "name": str(raw.get("name") or "waypoint_{}".format(index)),
            "position_m": [float(value) for value in position],
            "orientation_xyzw": normalize_quaternion_xyzw(orientation),
        })
    return waypoints


def run_pose_sequence(node, args, planning_start_state, tcp_offset_tool, waypoints) -> bool:
    """Plan each pose from the previous trajectory endpoint; stop on the first failure."""
    current_tool = node.current_tool_transform(timeout=args.tf_timeout)
    previous_tool_position, _ = transform_position_quat(current_tool)
    start_state = planning_start_state
    for index, waypoint in enumerate(waypoints, start=1):
        quaternion = waypoint["orientation_xyzw"]
        tool_goal = tool0_goal_from_tcp(
            waypoint["position_m"], quaternion, tcp_offset_tool,
        )
        result = plan_and_maybe_execute_motion(
            node,
            args,
            {"step": index, "action": "tcp_pose_sequence"},
            waypoint["name"],
            tool_goal,
            quaternion,
            previous_tool_position,
            "Continuous TCP pose sequence {}/{} {}".format(
                index, len(waypoints), waypoint["name"],
            ),
            start_joint_state=start_state,
        )
        if not result:
            node.get_logger().error(
                "Continuous TCP pose sequence stopped at {}/{} {}.".format(
                    index, len(waypoints), waypoint["name"],
                )
            )
            return False
        if isinstance(result, JointState):
            start_state = result
        previous_tool_position = tool_goal
        if args.execute:
            # Execution verification has already checked the measured pose. Use
            # the measured joints for the next request so small controller error
            # cannot accumulate across the sequence.
            start_state = node.latest_joint_state
            current_tool = node.current_tool_transform(timeout=args.tf_timeout)
            previous_tool_position, _ = transform_position_quat(current_tool)
    return True
