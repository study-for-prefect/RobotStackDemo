"""MoveIt command construction for code-driven yaw probe motions."""

import subprocess
import time
from argparse import Namespace
from typing import Dict, Iterable, List

from tools.monitoring.realtime_monitor.constants import PROJECT_ROOT

from .pose_math import quaternion_xyzw_to_matrix


def cli_float(value: float) -> str:
    """Format floats so argparse never confuses tiny negative values for flags."""
    text = "{:.12f}".format(float(value)).rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def cli_floats(values: Iterable[float]) -> List[str]:
    return [cli_float(float(value)) for value in values]


def hover_target_from_tool0_position(
    tool0_position_m: Iterable[float],
    tool0_quat_xyzw: Iterable[float],
    tcp_offset_tool_m: Iterable[float],
) -> List[float]:
    """Return the TCP target that makes moveit_preview keep tool0 pose unchanged."""
    position = [float(value) for value in tool0_position_m]
    matrix = quaternion_xyzw_to_matrix(tool0_quat_xyzw)
    offset = [float(value) for value in tcp_offset_tool_m]
    tcp_offset_base = matrix.dot(offset).astype(float).tolist()
    return [position[i] + tcp_offset_base[i] for i in range(3)]


def build_yaw_motion_command(args: Namespace, target_pose: Dict[str, object]) -> List[str]:
    tool0_position = [float(value) for value in target_pose["tool0_position"]]  # type: ignore[index]
    tool0_quat = [float(value) for value in target_pose["tool0_quat"]]  # type: ignore[index]
    hover_target = hover_target_from_tool0_position(
        tool0_position,
        tool0_quat,
        args.motion_tcp_offset_tool,
    )
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--hover-only",
        "--hover-target-base",
        *cli_floats(hover_target),
        "--hover-orientation-xyzw",
        *cli_floats(tool0_quat),
        "--tcp-offset-tool",
        *cli_floats(args.motion_tcp_offset_tool),
        "--velocity",
        cli_float(args.velocity),
        "--acceleration",
        cli_float(args.acceleration),
        "--pre-rotate-velocity",
        cli_float(args.pre_rotate_velocity),
        "--pre-rotate-acceleration",
        cli_float(args.pre_rotate_acceleration),
        "--planning-time",
        cli_float(args.planning_time),
        "--tf-timeout",
        cli_float(args.tf_timeout),
        "--base-link",
        args.base_frame,
        "--end-effector",
        args.tool_frame,
        "--pre-rotate-before-translation",
        "--pre-rotate-strategy",
        args.pre_rotate_strategy,
        "--pre-rotate-wrist-yaw-sign",
        args.pre_rotate_wrist_yaw_sign,
        "--pre-rotate-wrist-direction",
        args.pre_rotate_wrist_direction,
        "--max-joint-delta",
        cli_float(args.max_joint_delta),
        "--max-pre-rotate-joint-delta",
        cli_float(args.max_pre_rotate_joint_delta),
        "--max-grasp-yaw-error-deg",
        cli_float(args.max_grasp_yaw_error_deg),
        "--orientation-settle-error-deg",
        cli_float(args.orientation_settle_error_deg),
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command


def build_ready_motion_command(args: Namespace) -> List[str]:
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--ready-only",
        "--ready-joint-pose-json",
        args.ready_joint_pose_json,
        "--velocity",
        cli_float(args.velocity),
        "--acceleration",
        cli_float(args.acceleration),
        "--planning-time",
        cli_float(args.planning_time),
        "--tf-timeout",
        cli_float(args.tf_timeout),
        "--base-link",
        args.base_frame,
        "--end-effector",
        args.tool_frame,
        "--max-joint-delta",
        cli_float(args.max_joint_delta),
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command


def pump_gui_events(args: Namespace) -> None:
    if getattr(args, "no_window", False):
        return
    try:
        import cv2

        cv2.waitKey(1)
    except Exception:
        pass


def run_command_with_gui_pump(args: Namespace, command: List[str]) -> None:
    print("\n$ {}".format(" ".join(command)), flush=True)
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                return
            pump_gui_events(args)
            time.sleep(0.05)
    except KeyboardInterrupt:
        process.terminate()
        raise


def run_ready_motion_command(args: Namespace) -> None:
    command = build_ready_motion_command(args)
    run_command_with_gui_pump(args, command)


def run_yaw_motion_command(args: Namespace, target_pose: Dict[str, object]) -> None:
    command = build_yaw_motion_command(args, target_pose)
    run_command_with_gui_pump(args, command)
