"""Subprocess execution and hover MoveIt command construction."""

import subprocess

from .constants import PROJECT_ROOT

def run(command):
    print("$ {}".format(" ".join(str(part) for part in command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def moveit_hover_command(args, hover_position, hover_quat):
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--hover-only",
        "--hover-target-base", *[str(value) for value in hover_position],
        "--hover-orientation-xyzw", *[str(value) for value in hover_quat],
        "--tcp-offset-tool", *[str(value) for value in args.tcp_offset_tool],
        "--safe-pre-rotate-height", str(args.safe_pre_rotate_height),
        "--base-link", args.base_frame,
        "--end-effector", args.tool_frame,
        "--tf-timeout", str(args.tf_timeout),
        "--velocity", str(args.velocity),
        "--acceleration", str(args.acceleration),
        "--pre-rotate-velocity", str(args.pre_rotate_velocity),
        "--pre-rotate-acceleration", str(args.pre_rotate_acceleration),
        "--pre-rotate-strategy", args.pre_rotate_strategy,
        "--pre-rotate-wrist-yaw-sign", args.pre_rotate_wrist_yaw_sign,
        "--orientation-settle-error-deg", str(args.orientation_settle_error_deg),
        "--orientation-settle-attempts", str(args.orientation_settle_attempts),
        "--max-grasp-orientation-error-deg", str(args.max_grasp_orientation_error_deg),
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command
