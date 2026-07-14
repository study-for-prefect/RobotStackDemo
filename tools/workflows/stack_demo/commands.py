"""External command construction and observation capture."""

import json
import os
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import urlopen

from tools.workflows.two_stage_visual_pick import camera_optical_vector_to_base

from .constants import PROJECT_ROOT
from .workspace import attach_configured_workspace

DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_TF_POINT_MODE = "direct"


def format_motion_float(value):
    return "{:.6f}".format(float(value))


def perception_camera_frame(args):
    camera_frame = str(getattr(args, "camera_frame", DEFAULT_CAMERA_FRAME) or DEFAULT_CAMERA_FRAME).lstrip("/")
    if camera_frame == "camera_link":
        return DEFAULT_CAMERA_FRAME
    return camera_frame


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(command):
    print("\n$ {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def plan_only_command(command):
    """Strip every trajectory/gripper execution option from a MoveIt command."""
    output = list(command)
    for flag, value_count in (
        ("--execute", 0),
        ("--enable-gripper", 0),
        ("--skip-gripper-init", 0),
        ("--close-gripper-for-push", 0),
        ("--gripper-open-only", 0),
        ("--gripper-close-only", 0),
        ("--gripper-port", 1),
    ):
        while flag in output:
            index = output.index(flag)
            del output[index:index + value_count + 1]
    return output


def tf_lookup_command(args, require_tool=True):
    command = [
        args.ros_python,
        "tools/robot/tf_lookup_json.py",
        "--output",
        args.tf_json,
        "--base-frame",
        getattr(args, "base_frame", "base_link"),
        "--camera-frame",
        perception_camera_frame(args),
        "--tool-frame",
        getattr(args, "tool_frame", "tool0"),
        "--timeout",
        str(getattr(args, "tf_timeout", 8.0)),
        "--once",
    ]
    if require_tool:
        command.append("--require-tool")
    return command


def moveit_frame_args(args):
    return [
        "--base-link",
        getattr(args, "base_frame", "base_link"),
        "--end-effector",
        getattr(args, "tool_frame", "tool0"),
    ]


def failure_state_path(args):
    return os.path.join(args.output_dir, "failure_state.json")


def pose_command(args, pose_json):
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--ready-only", "--ready-joint-pose-json", pose_json,
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        "--max-joint-delta", str(getattr(args, "ready_max_joint_delta", 1.30)),
        "--ready-joint-tolerance", str(getattr(args, "ready_joint_tolerance", 0.15)),
        *moveit_frame_args(args),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def open_gripper_command(args):
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--gripper-open-only", "--enable-gripper",
        "--gripper-port", args.gripper_port, "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def close_gripper_command(args):
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--gripper-close-only", "--enable-gripper",
        "--gripper-port", args.gripper_port, "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def init_ready_pose(args):
    """Put the empty-gripper robot in the configured observation pose."""
    if args.execute:
        print("\ninit_ready_pose: moving to ready pose", flush=True)
        if not args.ready_pose_json or not os.path.isfile(args.ready_pose_json):
            raise RuntimeError(
                "Ready pose JSON not found: {}. Set --ready-pose-json/READY_POSE_JSON to an existing init pose.".format(
                    args.ready_pose_json
                )
            )
        run(pose_command(args, args.ready_pose_json))
    else:
        print("\ninit_ready_pose: motion skipped (execution disabled)", flush=True)

    if args.execute:
        print("init_ready_pose: opening gripper", flush=True)
        run(open_gripper_command(args))
        if args.init_stable_wait_s > 0:
            time.sleep(float(args.init_stable_wait_s))
    else:
        print("init_ready_pose: gripper open skipped (execution disabled)", flush=True)
    print("init_ready_pose: stable, start snapshot", flush=True)


def capture_empty_observation(args, output_dir, held_object_id):
    if held_object_id is not None:
        raise RuntimeError("Snapshot forbidden while holding object {}.".format(held_object_id))
    init_ready_pose(args)
    if args.offline_scene_state:
        return None
    capture_scene_observation(args, output_dir)
    return attach_configured_workspace(
        load_json(os.path.join(output_dir, "private_scene_state.json")), args,
    )


def capture_empty_current_pose(args, output_dir, held_object_id, allow_holding=False, refresh_tf=False):
    if held_object_id is not None and not allow_holding:
        raise RuntimeError("Second snapshot forbidden while holding object {}.".format(held_object_id))
    if args.offline_scene_state:
        return None
    if args.second_snapshot_stable_wait_s > 0:
        time.sleep(float(args.second_snapshot_stable_wait_s))
    if refresh_tf:
        run(tf_lookup_command(args))
    capture_scene_observation(args, output_dir)
    return attach_configured_workspace(
        load_json(os.path.join(output_dir, "private_scene_state.json")), args,
    )


def perception_server_snapshot(args, output_dir, score_thresh=None):
    base_url = str(getattr(args, "perception_server_url", "") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("No --perception-server-url configured.")
    query_values = {
        "output_dir": output_dir,
        "instruction": getattr(args, "instruction", ""),
        "camera_frame": perception_camera_frame(args),
    }
    if score_thresh is not None:
        query_values["score_thresh"] = float(score_thresh)
    query = urlencode(query_values)
    url = "{}/snapshot?{}".format(base_url, query)
    with urlopen(url, timeout=float(getattr(args, "perception_server_timeout_s", 10.0))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "perception server snapshot failed")
    return payload


def capture_scene_observation(args, output_dir, score_thresh=None):
    # The wrist camera moves with every robot trajectory.  The background TF
    # bridge may still contain a transform sampled during the return motion,
    # so force one post-motion lookup before the persistent server consumes it.
    run(tf_lookup_command(args))
    try:
        payload = perception_server_snapshot(args, output_dir, score_thresh=score_thresh)
        print(
            "Perception server snapshot: output={} objects={} frame_seq={}".format(
                output_dir,
                payload.get("object_count"),
                payload.get("frame_color_seq"),
            ),
            flush=True,
        )
        return
    except Exception as exc:
        if not getattr(args, "allow_snapshot_subprocess_fallback", False):
            raise RuntimeError(
                "Persistent perception server snapshot failed and subprocess fallback is disabled: {}".format(exc)
            )
        print("Perception server failed; using legacy snapshot subprocess: {}".format(exc), flush=True)
    run(tf_lookup_command(args))
    run(snapshot_command(args, output_dir))


def relative_translate_command(args, offset_base):
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--relative-tool-translation-base", *[format_motion_float(value) for value in offset_base],
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        *moveit_frame_args(args),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def push_clear_command(args, push_plan_path):
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--push-plan-json",
        push_plan_path,
        "--enable-gripper",
        "--close-gripper-for-push",
        "--gripper-port",
        args.gripper_port,
        "--tcp-offset-tool",
        *[str(value) for value in args.tcp_offset_tool],
        "--velocity",
        str(args.velocity),
        "--acceleration",
        str(args.acceleration),
        "--tf-timeout",
        str(getattr(args, "tf_timeout", 8.0)),
        *moveit_frame_args(args),
        "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def push_preflight_command(args, push_plan_path):
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--push-plan-json",
        push_plan_path,
        "--tcp-offset-tool",
        *[str(value) for value in getattr(args, "tcp_offset_tool", [0.0, 0.0, 0.0])],
        "--velocity",
        str(getattr(args, "velocity", 0.20)),
        "--acceleration",
        str(getattr(args, "acceleration", 0.20)),
        "--tf-timeout",
        str(getattr(args, "tf_timeout", 8.0)),
        *moveit_frame_args(args),
    ]
    if args.yes:
        command.append("--yes")
    return command


def retry_close_observation(
    args,
    output_dir,
    held_object_id,
    parse_state,
    description,
    allow_holding=False,
    retry_count=None,
    retry_offset_camera=None,
    refresh_tf=False,
):
    retries = args.close_observation_retry_count if retry_count is None else retry_count
    attempts = max(0, int(retries)) + 1
    last_error = None
    for attempt in range(attempts):
        attempt_dir = os.path.join(output_dir, "attempt_{:02d}".format(attempt + 1))
        state = capture_empty_current_pose(
            args,
            attempt_dir,
            held_object_id,
            allow_holding=allow_holding,
            refresh_tf=bool(refresh_tf),
        )
        if state is None:
            return None, None
        try:
            return state, parse_state(state)
        except RuntimeError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            if retry_offset_camera is not None:
                camera_offset = [float(value) for value in retry_offset_camera]
                offset = camera_optical_vector_to_base(args.tf_json, camera_offset)
                print(
                    "{} detection failed: {}. Retry after optical camera offset {} -> base_link offset {}.".format(
                        description, last_error, camera_offset, offset
                    ),
                    flush=True,
                )
            else:
                offset = [float(value) for value in args.close_observation_retry_base_offset]
                print(
                    "{} detection failed: {}. Retry after base_link offset {}.".format(
                        description, last_error, offset
                    ),
                    flush=True,
                )
            run(relative_translate_command(args, offset))
    raise RuntimeError("{} detection failed after {} attempts: {}".format(description, attempts, last_error))


def snapshot_command(args, output_dir):
    command = [
        "conda", "run", "-n", args.conda_env, "python", "-m",
        "robot_scene_pipeline.snapshot_pipeline",
        "--output-dir", output_dir,
        "--use-tf", "--tf-json", args.tf_json,
        "--base-frame", getattr(args, "base_frame", "base_link"),
        "--camera-frame", perception_camera_frame(args),
        "--tf-point-mode", DEFAULT_TF_POINT_MODE,
        "--estimate-tabletop",
        "--detector-weight", args.detector_weight,
        "--score-thresh", str(args.score_thresh),
        "--detector-imgsz", str(args.detector_imgsz),
        "--detector-iou", str(args.detector_iou),
        "--detector-device", args.detector_device,
    ]
    command.append("--skip-llm")
    return command
