"""Existing observation, TF, MoveIt, and GF225 command boundary."""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Sequence

from robot_scene_pipeline.perception_contract import PerceptionServerError

from .constants import PROJECT_ROOT
from .perception_client import (
    build_perception_snapshot_request,
    perception_camera_frame,
    perception_server_health,
    perception_server_snapshot,
)
from .workspace import attach_configured_workspace


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run(command: Sequence[str]) -> None:
    print("\n$ {}".format(" ".join(command)), flush=True)
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


NON_ACTUATING_FORBIDDEN_FLAGS = frozenset({
    "--execute", "--yes", "--enable-gripper", "--gripper-open-only",
    "--gripper-close-only", "--close-gripper-for-push", "--execute-push-clearing",
})


def assert_non_actuating_command(command: Sequence[str]) -> None:
    forbidden = sorted(NON_ACTUATING_FORBIDDEN_FLAGS.intersection(str(item) for item in command))
    if forbidden:
        raise RuntimeError(f"non-actuating safety guard rejected command flags: {forbidden}")


def run_non_actuating(command: Sequence[str]) -> None:
    assert_non_actuating_command(command)
    run(command)


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


def close_gripper_command(args: Any, result_json: str = "") -> list[str]:
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--gripper-close-only", "--enable-gripper",
        "--gripper-port", args.gripper_port, "--execute",
    ]
    if result_json:
        command.extend(["--gripper-result-json", result_json])
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


def return_to_ready_observation(args: Any) -> None:
    """Return an executing workflow to the same joint pose used for observation."""
    if not args.execute:
        return
    if not args.ready_pose_json or not os.path.isfile(args.ready_pose_json):
        raise RuntimeError(f"ready pose JSON not found: {args.ready_pose_json}")
    run(pose_command(args, args.ready_pose_json))
    if args.init_stable_wait_s > 0:
        time.sleep(float(args.init_stable_wait_s))


def capture_empty_observation(args: Any, output_dir: str, held_object_id: str | None) -> dict[str, Any] | None:
    if held_object_id is not None:
        raise RuntimeError(f"ready observation forbidden while holding {held_object_id}")
    init_ready_pose(args)
    if args.offline_scene_state:
        return None
    capture_scene_observation(args, output_dir, capture_reason="initial_observation", scene_revision=1)
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
    capture_scene_observation(args, output_dir, capture_reason="fresh_observation")
    return attach_configured_workspace(load_json(os.path.join(output_dir, "private_scene_state.json")), args)


def capture_scene_observation(
    args: Any,
    output_dir: str,
    *,
    capture_reason: str = "unspecified",
    scene_revision: int = 1,
) -> None:
    """Force fresh live TF before every wrist-camera scene acquisition."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_non_actuating(tf_lookup_command(args))
    request = build_perception_snapshot_request(
        args, output_dir, capture_reason=capture_reason, scene_revision=scene_revision,
    )
    _write_json(os.path.join(output_dir, "perception_snapshot_request.json"), request.to_dict())
    try:
        health = perception_server_health(args)
        _write_json(os.path.join(output_dir, "perception_server_health.json"), health.to_dict())
        response = perception_server_snapshot(
            args, output_dir, capture_reason=capture_reason, scene_revision=scene_revision,
        )
        _write_json(os.path.join(output_dir, "perception_snapshot_response.json"), response)
        _write_json(os.path.join(output_dir, "perception_capture.json"), {"perception_source": "perception_server"})
        return
    except Exception as exc:
        error = exc.to_dict() if isinstance(exc, PerceptionServerError) else {
            "error_type": type(exc).__name__, "error": str(exc),
        }
        _write_json(os.path.join(output_dir, "perception_server_error.json"), error)
        if not args.allow_snapshot_subprocess_fallback:
            detail = exc.concise_message() if isinstance(exc, PerceptionServerError) else str(exc)
            raise RuntimeError(f"perception server failed and fallback is disabled: {detail}") from exc
    command = snapshot_command(args, output_dir)
    _write_json(os.path.join(output_dir, "perception_capture.json"), {
        "perception_source": "snapshot_subprocess_fallback",
        "server_error": error,
        "fallback_command": command,
    })
    run_non_actuating(tf_lookup_command(args))
    run_non_actuating(command)


def snapshot_command(args: Any, output_dir: str) -> list[str]:
    return [
        "conda", "run", "-n", args.conda_env, "python", "-m", "robot_scene_pipeline.snapshot_pipeline",
        "--output-dir", output_dir,
        "--use-tf", "--tf-json", args.tf_json,
        "--base-frame", args.base_frame,
        "--camera-frame", perception_camera_frame(args),
        "--tf-point-mode", args.tf_point_mode,
        "--estimate-tabletop",
        "--detector-weight", args.detector_weight,
        "--score-thresh", str(args.score_thresh),
        "--detector-imgsz", str(args.detector_imgsz),
        "--detector-iou", str(args.detector_iou),
        "--detector-device", args.detector_device,
        "--color-topic", args.color_topic,
        "--depth-topic", args.depth_topic,
        "--camera-info-topic", args.camera_info_topic,
    ]


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
