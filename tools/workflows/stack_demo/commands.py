"""Existing observation, TF, MoveIt, and GF225 command boundary."""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

from .constants import PROJECT_ROOT
from .workspace import attach_configured_workspace


DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run(command: Sequence[str]) -> None:
    print("\n$ {}".format(" ".join(command)), flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def plan_only_command(command: Sequence[str]) -> list[str]:
    """Remove every trajectory and gripper execution option."""
    output = list(command)
    for flag, value_count in (
        ("--execute", 0), ("--enable-gripper", 0), ("--skip-gripper-init", 0),
        ("--close-gripper-for-push", 0), ("--gripper-open-only", 0),
        ("--gripper-close-only", 0), ("--gripper-port", 1),
    ):
        while flag in output:
            index = output.index(flag)
            del output[index:index + value_count + 1]
    return output


def perception_camera_frame(args: Any) -> str:
    frame = str(getattr(args, "camera_frame", DEFAULT_CAMERA_FRAME) or DEFAULT_CAMERA_FRAME).lstrip("/")
    return DEFAULT_CAMERA_FRAME if frame == "camera_link" else frame


def tf_lookup_command(args: Any, require_tool: bool = True) -> list[str]:
    command = [
        args.ros_python, "tools/robot/tf_lookup_json.py",
        "--output", args.tf_json,
        "--base-frame", args.base_frame,
        "--camera-frame", perception_camera_frame(args),
        "--tool-frame", args.tool_frame,
        "--timeout", str(args.tf_timeout),
        "--once",
    ]
    if require_tool:
        command.append("--require-tool")
    return command


def moveit_frame_args(args: Any) -> list[str]:
    return ["--base-link", args.base_frame, "--end-effector", args.tool_frame]


def pose_command(args: Any, pose_json: str) -> list[str]:
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--ready-only", "--ready-joint-pose-json", pose_json,
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        *moveit_frame_args(args), "--tf-timeout", str(args.tf_timeout), "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def open_gripper_command(args: Any) -> list[str]:
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--gripper-open-only", "--enable-gripper",
        "--gripper-port", args.gripper_port, "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def close_gripper_command(args: Any) -> list[str]:
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--gripper-close-only", "--enable-gripper",
        "--gripper-port", args.gripper_port, "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def init_ready_pose(args: Any) -> None:
    if not args.execute:
        return
    if not args.ready_pose_json or not os.path.isfile(args.ready_pose_json):
        raise RuntimeError(f"ready pose JSON not found: {args.ready_pose_json}")
    run(pose_command(args, args.ready_pose_json))
    run(open_gripper_command(args))
    if args.init_stable_wait_s > 0:
        time.sleep(float(args.init_stable_wait_s))


def capture_empty_observation(args: Any, output_dir: str, held_object_id: str | None) -> dict[str, Any] | None:
    if held_object_id is not None:
        raise RuntimeError(f"ready observation forbidden while holding {held_object_id}")
    init_ready_pose(args)
    if args.offline_scene_state:
        return None
    capture_scene_observation(args, output_dir)
    return attach_configured_workspace(load_json(os.path.join(output_dir, "private_scene_state.json")), args)


def capture_empty_current_pose(
    args: Any,
    output_dir: str,
    held_object_id: str | None,
    *,
    allow_holding: bool = False,
    refresh_tf: bool = False,
) -> dict[str, Any] | None:
    if held_object_id is not None and not allow_holding:
        raise RuntimeError(f"observation forbidden while holding {held_object_id}")
    if args.offline_scene_state:
        return None
    if args.second_snapshot_stable_wait_s > 0:
        time.sleep(float(args.second_snapshot_stable_wait_s))
    if refresh_tf:
        run(tf_lookup_command(args))
    capture_scene_observation(args, output_dir)
    return attach_configured_workspace(load_json(os.path.join(output_dir, "private_scene_state.json")), args)


def perception_server_snapshot(args: Any, output_dir: str) -> dict[str, Any]:
    base_url = str(args.perception_server_url or "").rstrip("/")
    if not base_url:
        raise RuntimeError("no perception server URL configured")
    query = urlencode({
        "output_dir": output_dir,
        "camera_frame": perception_camera_frame(args),
        "score_thresh": float(args.score_thresh),
    })
    with urlopen(f"{base_url}/snapshot?{query}", timeout=float(args.perception_server_timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "perception server snapshot failed")
    return payload


def capture_scene_observation(args: Any, output_dir: str) -> None:
    """Force fresh live TF before every wrist-camera scene acquisition."""
    run(tf_lookup_command(args))
    try:
        perception_server_snapshot(args, output_dir)
        return
    except Exception as exc:
        if not args.allow_snapshot_subprocess_fallback:
            raise RuntimeError(f"perception server failed and fallback is disabled: {exc}") from exc
    run(tf_lookup_command(args))
    run(snapshot_command(args, output_dir))


def snapshot_command(args: Any, output_dir: str) -> list[str]:
    return [
        "conda", "run", "-n", args.conda_env, "python", "-m", "robot_scene_pipeline.snapshot_pipeline",
        "--output-dir", output_dir,
        "--use-tf", "--tf-json", args.tf_json,
        "--base-frame", args.base_frame,
        "--camera-frame", perception_camera_frame(args),
        "--tf-point-mode", "direct",
        "--estimate-tabletop",
        "--detector-weight", args.detector_weight,
        "--score-thresh", str(args.score_thresh),
        "--detector-imgsz", str(args.detector_imgsz),
        "--detector-iou", str(args.detector_iou),
        "--detector-device", args.detector_device,
    ]
