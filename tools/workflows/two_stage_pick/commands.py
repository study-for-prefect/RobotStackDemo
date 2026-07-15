"""Command builders and snapshot retry helpers."""

import sys
import time

from .io import run, run_checked
from .scene import (
    camera_optical_vector_to_base,
    print_missing_target,
    target_visible,
)

DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_TF_POINT_MODE = "direct"

def perception_camera_frame(args):
    camera_frame = str(getattr(args, "camera_frame", DEFAULT_CAMERA_FRAME) or DEFAULT_CAMERA_FRAME).lstrip("/")
    if camera_frame == "camera_link":
        return DEFAULT_CAMERA_FRAME
    return camera_frame


def tf_lookup_command(args):
    return [
        args.ros_python,
        "tools/robot/tf_lookup_json.py",
        "--output",
        args.tf_json,
        "--base-frame",
        getattr(args, "base_frame", "base_link"),
        "--camera-frame",
        perception_camera_frame(args),
        "--once",
    ]


def snapshot_command(args, output_dir):
    command = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "-m",
        "robot_scene_pipeline.snapshot_pipeline",
        "--output-dir",
        output_dir,
        "--use-tf",
        "--tf-json",
        args.tf_json,
        "--base-frame",
        getattr(args, "base_frame", "base_link"),
        "--camera-frame",
        perception_camera_frame(args),
        "--tf-point-mode",
        DEFAULT_TF_POINT_MODE,
        "--estimate-tabletop",
        "--detector-weight",
        args.detector_weight,
        "--score-thresh",
        str(args.score_thresh),
        "--detector-imgsz",
        str(args.detector_imgsz),
        "--detector-iou",
        str(args.detector_iou),
        "--detector-device",
        args.detector_device,
    ]
    return command


def build_plan_command(args, private_state, output):
    command = [
        sys.executable,
        "tools/planning/build_geometry_pick_plan.py",
        "--private-state-json",
        private_state,
    ]
    if args.object_id is not None:
        command.extend(["--object-id", str(args.object_id)])
    else:
        command.extend(["--object-label", args.object_label])
    command.extend([
        "--output",
        output,
        "--approach-height-m",
        str(args.approach_height_m),
        "--pick-target-lift-m",
        str(args.pick_target_lift_m),
        "--force-yaw-labels",
        args.force_yaw_labels,
    ])
    reference_xy = getattr(args, "target_reference_base_xy", None)
    if reference_xy is None:
        reference_xy = args.nearest_base_xy
    if reference_xy is not None:
        command.extend(["--nearest-base-xy", str(reference_xy[0]), str(reference_xy[1])])
    if args.stack_demo_mode:
        command.extend([
            "--stack-demo-mode",
            "--require-geometry-center",
            "--fixed-square-yaw-deg",
            str(args.fixed_square_yaw_deg),
            "--stack-square-yaw-mode",
            args.stack_square_yaw_mode,
            "--square-yaw-snap-tolerance-deg",
            str(args.square_yaw_snap_tolerance_deg),
            "--stack-grasp-axis",
            args.grasp_axis,
        ])
    if args.xy_correction_json:
        command.extend(["--xy-correction-json", args.xy_correction_json])
    return command


def moveit_common(args, plan_path):
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--plan-json",
        plan_path,
        "--tcp-offset-tool",
        *[str(value) for value in args.tcp_offset_tool],
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--max-grasp-orientation-error-deg",
        str(args.max_grasp_orientation_error_deg),
        "--velocity",
        str(args.velocity),
        "--acceleration",
        str(args.acceleration),
        "--tf-timeout",
        str(args.moveit_tf_timeout),
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command


def relative_translate_command(args, offset_base):
    command = [
        args.ros_python,
        "tools/robot/moveit_plan_preview.py",
        "--relative-tool-translation-base",
        *[str(value) for value in offset_base],
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--velocity",
        str(args.velocity),
        "--acceleration",
        str(args.acceleration),
        "--tf-timeout",
        str(args.moveit_tf_timeout),
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command


def capture_second_snapshot_and_plan(args, second_private, second_plan):
    attempts = max(0, int(args.second_snapshot_retry_count)) + 1
    for attempt in range(attempts):
        if attempt > 0:
            run(tf_lookup_command(args), args.execute)
            base_offset = camera_optical_vector_to_base(args.tf_json, args.second_snapshot_retry_offset_camera)
            print(
                "\nSecond snapshot target missing; retry {}/{} after optical camera offset {} -> base offset {}.".format(
                    attempt,
                    attempts - 1,
                    [round(value, 4) for value in args.second_snapshot_retry_offset_camera],
                    [round(value, 4) for value in base_offset],
                ),
                flush=True,
            )
            run(relative_translate_command(args, base_offset), args.execute)
        run(tf_lookup_command(args), args.execute)
        run(snapshot_command(args, args.second_dir), args.execute)
        if args.execute and not target_visible(args, second_private):
            print_missing_target("Second snapshot", args, second_private)
            if attempt + 1 < attempts:
                continue
            return False
        if run_checked(build_plan_command(args, second_private, second_plan), args.execute):
            return True
        if attempt + 1 >= attempts:
            return False
    return False


def capture_first_snapshot_until_target_visible(args, first_private):
    attempts = max(0, int(args.first_snapshot_retry_count)) + 1
    for attempt in range(attempts):
        run(tf_lookup_command(args), args.execute)
        run(snapshot_command(args, args.first_dir), args.execute)
        if not args.execute or target_visible(args, first_private):
            return True
        print_missing_target("First snapshot", args, first_private)
        if attempt + 1 < attempts and float(args.first_snapshot_stable_wait_s) > 0.0:
            print(
                "First snapshot retry {}/{} after {:.2f}s stable wait.".format(
                    attempt + 1,
                    attempts - 1,
                    float(args.first_snapshot_stable_wait_s),
                ),
                flush=True,
            )
            import time

            time.sleep(float(args.first_snapshot_stable_wait_s))
    return False
