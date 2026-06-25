"""MoveIt command construction for code-driven yaw probe motions."""

import subprocess
from argparse import Namespace
from typing import Dict, Iterable, List

from tools.monitoring.realtime_monitor.constants import PROJECT_ROOT


def hover_target_from_tool0_position(
    tool0_position_m: Iterable[float],
    tool_z_offset_m: float,
    tool_offset_base_m: Iterable[float],
) -> List[float]:
    """Return a hover target that makes moveit_preview keep tool0 XYZ unchanged."""
    position = [float(value) for value in tool0_position_m]
    offset = [float(value) for value in tool_offset_base_m]
    return [
        position[0] - offset[0],
        position[1] - offset[1],
        position[2] - float(tool_z_offset_m) - offset[2],
    ]


def build_yaw_motion_command(args: Namespace, target_pose: Dict[str, object]) -> List[str]:
    tool0_position = [float(value) for value in target_pose["tool0_position"]]  # type: ignore[index]
    tool0_quat = [float(value) for value in target_pose["tool0_quat"]]  # type: ignore[index]
    hover_target = hover_target_from_tool0_position(
        tool0_position,
        args.motion_tool_z_offset,
        args.motion_tool_offset_base,
    )
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--hover-only",
        "--hover-target-base",
        *[str(value) for value in hover_target],
        "--hover-orientation-xyzw",
        *[str(value) for value in tool0_quat],
        "--tool-z-offset",
        str(args.motion_tool_z_offset),
        "--tool-offset-base",
        *[str(value) for value in args.motion_tool_offset_base],
        "--velocity",
        str(args.velocity),
        "--acceleration",
        str(args.acceleration),
        "--pre-rotate-velocity",
        str(args.pre_rotate_velocity),
        "--pre-rotate-acceleration",
        str(args.pre_rotate_acceleration),
        "--planning-time",
        str(args.planning_time),
        "--tf-timeout",
        str(args.tf_timeout),
        "--base-link",
        args.base_frame,
        "--end-effector",
        args.tool_frame,
        "--joint-space",
        "--pre-rotate-before-translation",
        "--pre-rotate-strategy",
        args.pre_rotate_strategy,
        "--pre-rotate-wrist-yaw-sign",
        args.pre_rotate_wrist_yaw_sign,
        "--pre-rotate-wrist-direction",
        args.pre_rotate_wrist_direction,
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--max-pre-rotate-joint-delta",
        str(args.max_pre_rotate_joint_delta),
        "--max-grasp-yaw-error-deg",
        str(args.max_grasp_yaw_error_deg),
        "--orientation-settle-error-deg",
        str(args.orientation_settle_error_deg),
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command


def run_yaw_motion_command(args: Namespace, target_pose: Dict[str, object]) -> None:
    command = build_yaw_motion_command(args, target_pose)
    print("\n$ {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
