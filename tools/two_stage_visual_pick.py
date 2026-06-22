#!/usr/bin/env python3
"""Run a two-snapshot pick: ready -> detect -> yaw/approach -> detect -> XY correct -> pick."""

import argparse
import json
import math
import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage visual pick with pure base-link XY correction.")
    parser.add_argument("--object-label", default="rectangle")
    parser.add_argument("--object-id", type=int, default=None)
    parser.add_argument("--nearest-base-xy", nargs=2, type=float, default=None)
    parser.add_argument(
        "--instruction",
        default="",
        help="Run LLM reasoning on the first snapshot and use its first pick target. Empty keeps deterministic target mode.",
    )
    parser.add_argument("--model", default="qwen2.5vl:7b-q4_K_M")
    parser.add_argument("--first-dir", default="/tmp/current_scene")
    parser.add_argument("--second-dir", default="/tmp/current_scene_second")
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json")
    parser.add_argument("--ready-pose-json", default="config/rectangle_ready_pose.json")
    parser.add_argument("--conda-env", default="scene_graph_benchmark")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--detector-weight", default="models/yolo/weights/best.pt")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--approach-height-m", type=float, default=0.05)
    parser.add_argument("--pick-target-lift-m", type=float, default=0.005)
    parser.add_argument("--place-target-lift-m", type=float, default=0.03)
    parser.add_argument("--place-offset-m", type=float, default=0.08)
    parser.add_argument("--left-right-axis", choices=("x", "y"), default="y")
    parser.add_argument("--left-direction-sign", choices=("positive", "negative"), default="positive")
    parser.add_argument("--front-back-axis", choices=("x", "y"), default="x")
    parser.add_argument("--front-direction-sign", choices=("positive", "negative"), default="positive")
    parser.add_argument("--tool-z-offset", type=float, default=0.15)
    parser.add_argument("--tool-offset-base", nargs=3, type=float, default=[-0.015, 0.0, 0.0])
    parser.add_argument("--grasp-axis", choices=("long", "short"), default="long")
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--force-yaw-labels",
        default="square",
        help="Comma-separated label substrings that should use detected yaw even if aspect-ratio yaw validity is false.",
    )
    parser.add_argument("--pre-rotate-wrist-direction", choices=("positive", "negative"), default="negative")
    parser.add_argument("--max-joint-delta", type=float, default=2.0)
    parser.add_argument("--max-pre-rotate-joint-delta", type=float, default=3.1416)
    parser.add_argument("--max-grasp-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--max-correction-m", type=float, default=0.05)
    parser.add_argument("--max-grasp-offset-m", type=float, default=0.05)
    parser.add_argument("--first-snapshot-retry-count", type=int, default=1)
    parser.add_argument("--first-snapshot-stable-wait-s", type=float, default=0.5)
    parser.add_argument(
        "--pick-observation-offset-camera",
        nargs=3,
        type=float,
        default=[0.0, 0.04, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Optical-camera offset used only when the close pick observation misses the target.",
    )
    parser.add_argument("--stack-demo-mode", action="store_true")
    parser.add_argument("--fixed-square-yaw-deg", type=float, default=0.0)
    parser.add_argument("--stack-square-yaw-mode", choices=("detected", "fixed"), default="detected")
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--xy-correction-json", default="")
    parser.add_argument("--velocity", type=float, default=0.1)
    parser.add_argument("--acceleration", type=float, default=0.1)
    parser.add_argument("--moveit-tf-timeout", type=float, default=10.0)
    parser.add_argument("--second-snapshot-retry-count", type=int, default=2)
    parser.add_argument(
        "--second-snapshot-retry-offset-camera",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Override the retry offset for a failed second snapshot. Defaults to --pick-observation-offset-camera.",
    )
    parser.add_argument(
        "--disable-gripper",
        action="store_false",
        dest="enable_gripper",
        default=True,
        help="Run the final descent without opening/closing the gripper.",
    )
    parser.add_argument(
        "--no-release-after-pick",
        action="store_false",
        dest="release_after_pick",
        default=True,
        help="Keep holding the object after pick instead of putting it back down and opening.",
    )
    parser.add_argument(
        "--no-execute-remaining-plan",
        action="store_false",
        dest="execute_remaining_plan",
        default=True,
        help="In LLM mode, stop while holding the object instead of executing steps after the selected pick.",
    )
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--post-close-wait", type=float, default=1.0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to move the robot. Without this flag, only print the workflow commands.",
    )
    args = parser.parse_args()
    if args.second_snapshot_retry_offset_camera is None:
        args.second_snapshot_retry_offset_camera = list(args.pick_observation_offset_camera)
    return args


def run(command, execute=True):
    print("\n$ {}".format(" ".join(command)), flush=True)
    if execute:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_checked(command, execute=True):
    print("\n$ {}".format(" ".join(command)), flush=True)
    if not execute:
        return True
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode == 0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def object_id_value(obj):
    try:
        return int(obj.get("id", -1))
    except (TypeError, ValueError):
        return -1


def object_base_point(obj):
    geometry_point = obj.get("geometry_center_m") if obj.get("geometry_frame") == "base_link" else None
    point = geometry_point or obj.get("center_3d_base_m")
    if not point or len(point) < 2:
        return None
    try:
        return [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
    except (TypeError, ValueError):
        return None


def object_sort_key(obj):
    point = object_base_point(obj)
    if point is None:
        return (float("inf"), float("inf"), object_id_value(obj))
    return (point[0], point[1], object_id_value(obj))


def describe_object(obj):
    point = object_base_point(obj)
    xy = "unknown"
    if point is not None:
        xy = "[{:.4f}, {:.4f}]".format(point[0], point[1])
    return "id={} label={} base_xy={}".format(obj.get("id"), obj.get("label"), xy)


def detected_object_summary(private_state_path):
    try:
        state = load_json(private_state_path)
    except (IOError, ValueError) as exc:
        return "failed to read {}: {}".format(private_state_path, exc)
    objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
    if not objects:
        return "no non-workspace objects detected"
    objects = sorted(objects, key=object_sort_key)
    return "; ".join(describe_object(obj) for obj in objects)


def target_description(args):
    if args.object_id is not None:
        return "object_id={}".format(args.object_id)
    return "object_label='{}'".format(args.object_label)


def matching_target_objects(args, private_state_path):
    state = load_json(private_state_path)
    objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
    if args.object_id is not None:
        matches = [obj for obj in objects if object_id_value(obj) == int(args.object_id)]
    else:
        requested = str(args.object_label).strip().lower()
        matches = [obj for obj in objects if requested in str(obj.get("label", "")).lower()]
    return sorted(matches, key=object_sort_key)


def target_visible(args, private_state_path):
    try:
        return bool(matching_target_objects(args, private_state_path))
    except (IOError, ValueError):
        return False


def print_missing_target(stage, args, private_state_path):
    print(
        "\n{}: did not detect requested target {} in this snapshot; it may be outside the camera frame or missed by detection.".format(
            stage,
            target_description(args),
        ),
        flush=True,
    )
    print("Detected objects: {}".format(detected_object_summary(private_state_path)), flush=True)


def camera_vector_to_base(tf_json_path, camera_vector):
    payload = load_json(tf_json_path)
    matrix = payload.get("matrix_4x4")
    if not isinstance(matrix, list) or len(matrix) < 3:
        raise RuntimeError("TF JSON missing matrix_4x4: {}".format(tf_json_path))
    return [
        sum(float(matrix[row][col]) * float(camera_vector[col]) for col in range(3))
        for row in range(3)
    ]


def camera_optical_vector_to_base(tf_json_path, optical_vector):
    x_opt, y_opt, z_opt = [float(value) for value in optical_vector]
    camera_link_vector = [z_opt, -x_opt, -y_opt]
    return camera_vector_to_base(tf_json_path, camera_link_vector)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def first_planned_step(plan):
    for step in plan.get("steps", []):
        if step.get("status") == "planned" and step.get("target_position_m"):
            return step
    raise RuntimeError("No planned target step found.")


def select_llm_pick_step(plan):
    for index, step in enumerate(plan.get("steps", [])):
        if step.get("action") == "pick" and step.get("status") == "planned" and step.get("object_id") is not None:
            return index, step
    raise RuntimeError("LLM execution plan has no planned pick step with an object_id.")


def remaining_plan_after_step(plan, selected_index):
    remaining = json.loads(json.dumps(plan))
    remaining["steps"] = remaining.get("steps", [])[selected_index + 1 :]
    remaining["execution_status"] = "not_executed"
    return remaining


def reject_additional_pick_steps(plan, selected_index):
    additional = [
        step
        for step in plan.get("steps", [])[selected_index + 1 :]
        if step.get("action") == "pick" and step.get("status") == "planned"
    ]
    if additional:
        raise RuntimeError(
            "Integrated two-stage execution supports one pick per run; found another planned pick at step {}.".format(
                additional[0].get("step")
            )
        )


def reject_incomplete_remaining_motion_steps(plan, selected_index):
    for step in plan.get("steps", [])[selected_index + 1 :]:
        if step.get("action") in ("ask_user", "stop"):
            continue
        if step.get("status") != "planned" or not step.get("approach_position_m") or not step.get("target_position_m"):
            raise RuntimeError(
                "LLM step {} after pick is not executable: action={} status={}.".format(
                    step.get("step"),
                    step.get("action"),
                    step.get("status"),
                )
            )


def has_planned_motion(plan):
    return any(
        step.get("status") == "planned" and step.get("approach_position_m")
        for step in plan.get("steps", [])
    )


def plan_has_reliable_yaw(path):
    step = first_planned_step(load_json(path))
    return bool(step.get("target_yaw_valid") and step.get("target_yaw_deg") is not None)


def valid_geometry_center(step, name):
    center = step.get("geometry_center_m")
    if (
        step.get("pointcloud_geometry_valid") is not True
        or
        step.get("geometry_frame") != "base_link"
        or not isinstance(center, list)
        or len(center) < 3
    ):
        raise RuntimeError("{} plan step is missing base_link geometry_center_m.".format(name))
    try:
        center = [float(value) for value in center[:3]]
    except (TypeError, ValueError):
        raise RuntimeError("{} geometry_center_m is not numeric.".format(name))
    if not all(math.isfinite(value) for value in center):
        raise RuntimeError("{} geometry_center_m is not finite.".format(name))
    return center


def safe_name(value):
    output = []
    for char in str(value).strip().lower():
        if char.isalnum():
            output.append(char)
        elif output and output[-1] != "_":
            output.append("_")
    return "".join(output).strip("_") or "object"


def build_corrected_plan(
    first_plan_path,
    second_plan_path,
    output_path,
    report_path,
    max_correction_m,
    max_grasp_offset_m=0.05,
    use_second_yaw=False,
    use_second_grasp_offset=False,
):
    first_plan = load_json(first_plan_path)
    second_plan = load_json(second_plan_path)
    first_step = first_planned_step(first_plan)
    second_step = first_planned_step(second_plan)
    first_target = [float(value) for value in first_step["target_position_m"]]
    second_target = [float(value) for value in second_step["target_position_m"]]
    first_geometry = valid_geometry_center(first_step, "First")
    second_geometry = valid_geometry_center(second_step, "Second")
    first_grasp_offset = [
        first_target[0] - first_geometry[0],
        first_target[1] - first_geometry[1],
    ]
    second_grasp_offset = [
        second_target[0] - second_geometry[0],
        second_target[1] - second_geometry[1],
    ]
    grasp_offset = second_grasp_offset if use_second_grasp_offset else first_grasp_offset
    grasp_offset_norm = math.hypot(grasp_offset[0], grasp_offset[1])
    if grasp_offset_norm > float(max_grasp_offset_m):
        raise RuntimeError(
            "Selected grasp offset {:.4f} m exceeds limit {:.4f} m.".format(
                grasp_offset_norm, max_grasp_offset_m
            )
        )
    corrected_xy = [
        second_geometry[0] + grasp_offset[0],
        second_geometry[1] + grasp_offset[1],
    ]
    delta = [corrected_xy[0] - first_target[0], corrected_xy[1] - first_target[1], 0.0]
    correction_norm = (delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5
    if correction_norm > float(max_correction_m):
        raise RuntimeError(
            "Second-snapshot XY correction {:.4f} m exceeds limit {:.4f} m.".format(
                correction_norm, max_correction_m
            )
        )

    corrected = json.loads(json.dumps(first_plan))
    corrected_step = first_planned_step(corrected)
    for key in ("target_position_m", "approach_position_m"):
        position = corrected_step.get(key)
        if position:
            corrected_step[key] = [
                round(corrected_xy[0], 5),
                round(corrected_xy[1], 5),
                round(float(position[2]), 5),
            ]
    corrected_step["second_snapshot_delta_base_xy_m"] = [round(delta[0], 5), round(delta[1], 5)]
    corrected_step["second_geometry_center_m"] = second_geometry
    corrected_step["grasp_offset_from_geometry_center_m"] = [round(value, 6) for value in grasp_offset]
    corrected_step["coordinate_source"] = (
        "second_geometry_center_plus_second_snapshot_grasp_offset"
        if use_second_grasp_offset
        else "second_geometry_center_plus_first_grasp_offset"
    )
    if use_second_yaw:
        for key in (
            "estimated_yaw_deg",
            "object_yaw_deg",
            "target_yaw_deg",
            "chosen_grasp_yaw_deg",
            "gripper_yaw_offset_deg",
            "requested_grasp_axis",
            "grasp_axis",
            "target_yaw_valid",
            "exact_tool_yaw_required",
            "yaw_frame",
            "yaw_source",
        ):
            if key in second_step:
                corrected_step[key] = second_step[key]
        corrected_step["yaw_source"] = "second_snapshot_{}".format(second_step.get("yaw_source") or "detected")
    local_offset = corrected_step.get("grasp_tool_offset_local_xy_m")
    final_yaw = corrected_step.get("chosen_grasp_yaw_deg")
    if isinstance(local_offset, list) and len(local_offset) >= 2 and final_yaw is not None:
        yaw_rad = math.radians(float(final_yaw))
        rotated_offset = [
            math.cos(yaw_rad) * float(local_offset[0]) - math.sin(yaw_rad) * float(local_offset[1]),
            math.sin(yaw_rad) * float(local_offset[0]) + math.cos(yaw_rad) * float(local_offset[1]),
        ]
        corrected_step["grasp_tool_offset_base_xy_m"] = rotated_offset
        corrected_step["expected_tool0_grasp_xy_base_m"] = [
            corrected_xy[0] + rotated_offset[0],
            corrected_xy[1] + rotated_offset[1],
        ]
    write_json(output_path, corrected)

    report = {
        "first_plan": first_plan_path,
        "second_plan": second_plan_path,
        "corrected_plan": output_path,
        "first_target_position_m": first_target,
        "second_target_position_m": second_target,
        "first_geometry_center_m": first_geometry,
        "second_geometry_center_m": second_geometry,
        "first_grasp_offset_xy_m": first_grasp_offset,
        "second_grasp_offset_xy_m": second_grasp_offset,
        "selected_grasp_offset_xy_m": grasp_offset,
        "grasp_offset_norm_m": grasp_offset_norm,
        "delta_base_xy_m": [delta[0], delta[1]],
        "correction_norm_m": correction_norm,
        "first_target_yaw_deg": first_step.get("target_yaw_deg"),
        "second_target_yaw_deg": second_step.get("target_yaw_deg"),
        "final_target_yaw_deg": corrected_step.get("target_yaw_deg"),
        "used_second_yaw": bool(use_second_yaw),
        "used_second_grasp_offset": bool(use_second_grasp_offset),
    }
    write_json(report_path, report)
    print(
        "\nSecond-snapshot correction: delta_base_xy=[{:.4f}, {:.4f}] m norm={:.4f} m".format(
            delta[0], delta[1], correction_norm
        ),
        flush=True,
    )


def snapshot_command(args, output_dir, run_llm=False):
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
        "--tf-point-mode",
        "optical-to-camera-link",
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
    if run_llm:
        command.extend(
            [
                "--model",
                args.model,
                "--instruction",
                args.instruction,
                "--compile-execution-plan",
                "--place-offset-m",
                str(args.place_offset_m),
                "--approach-height-m",
                str(args.approach_height_m),
                "--pick-target-lift-m",
                str(args.pick_target_lift_m),
                "--place-target-lift-m",
                str(args.place_target_lift_m),
                "--left-right-axis",
                args.left_right_axis,
                "--left-direction-sign",
                args.left_direction_sign,
                "--front-back-axis",
                args.front_back_axis,
                "--front-direction-sign",
                args.front_direction_sign,
            ]
        )
    else:
        command.append("--skip-llm")
    return command


def build_plan_command(args, private_state, output):
    command = [
        sys.executable,
        "tools/build_geometry_pick_plan.py",
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
        "tools/moveit_plan_preview.py",
        "--plan-json",
        plan_path,
        "--tool-z-offset",
        str(args.tool_z_offset),
        "--tool-offset-base",
        *[str(value) for value in args.tool_offset_base],
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
        "tools/moveit_plan_preview.py",
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
            run([args.ros_python, "tools/tf_lookup_json.py", "--output", args.tf_json, "--once"], args.execute)
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
        run([args.ros_python, "tools/tf_lookup_json.py", "--output", args.tf_json, "--once"], args.execute)
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


def capture_first_snapshot_until_target_visible(args, first_private, run_llm=False):
    attempts = max(0, int(args.first_snapshot_retry_count)) + 1
    for attempt in range(attempts):
        run([args.ros_python, "tools/tf_lookup_json.py", "--output", args.tf_json, "--once"], args.execute)
        run(snapshot_command(args, args.first_dir, run_llm=run_llm), args.execute)
        if run_llm or not args.execute or target_visible(args, first_private):
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


def main():
    args = parse_args()
    execute = bool(args.execute)
    first_private = os.path.join(args.first_dir, "private_scene_state.json")
    second_private = os.path.join(args.second_dir, "private_scene_state.json")
    correction_report = os.path.join(args.first_dir, "second_snapshot_xy_correction.json")
    llm_plan_path = os.path.join(args.first_dir, "robot_execution_plan.json")
    remaining_plan_path = os.path.join(args.first_dir, "robot_execution_plan_after_two_stage_pick.json")
    llm_plan = None
    selected_pick_index = None

    if args.instruction:
        print("LLM + two-stage visual pick: ready -> LLM snapshot -> yaw/approach -> second snapshot -> XY correction -> pick -> remaining LLM plan")
    else:
        print("Two-stage visual pick: ready -> snapshot1 -> yaw/approach -> snapshot2 -> XY correction -> pick")
    if not execute:
        print("DRY RUN: add --execute to move the robot and capture snapshots.")

    ready = [
        args.ros_python,
        "tools/moveit_plan_preview.py",
        "--ready-only",
        "--ready-joint-pose-json",
        args.ready_pose_json,
        "--max-joint-delta",
        str(args.max_joint_delta),
        "--tf-timeout",
        str(args.moveit_tf_timeout),
    ]
    if args.execute:
        ready.append("--execute")
    if args.enable_gripper:
        ready.extend([
            "--enable-gripper",
            "--gripper-port",
            args.gripper_port,
            "--open-gripper-at-start",
        ])
    if args.yes:
        ready.append("--yes")
    run(ready, execute)

    if not capture_first_snapshot_until_target_visible(args, first_private, run_llm=bool(args.instruction)):
        print(
            "Stop before motion. First snapshot did not detect {} after {} retry/retries.".format(
                target_description(args),
                args.first_snapshot_retry_count,
            ),
            flush=True,
        )
        return 2
    if args.instruction and execute:
        llm_plan = load_json(llm_plan_path)
        selected_pick_index, selected_pick = select_llm_pick_step(llm_plan)
        reject_additional_pick_steps(llm_plan, selected_pick_index)
        reject_incomplete_remaining_motion_steps(llm_plan, selected_pick_index)
        args.object_id = int(selected_pick["object_id"])
        if selected_pick.get("object_label"):
            args.object_label = str(selected_pick["object_label"])
        print(
            "\nLLM selected pick target: object_id={} label={}".format(args.object_id, args.object_label),
            flush=True,
        )

    object_name = "id_{}".format(args.object_id) if args.object_id is not None else safe_name(args.object_label)
    if args.instruction and not execute:
        object_name = "llm_selected"
    first_plan = os.path.join(args.first_dir, "{}_pick_plan_target_test.json".format(object_name))
    second_plan = os.path.join(args.second_dir, "{}_pick_plan_second.json".format(object_name))
    corrected_plan = os.path.join(args.first_dir, "{}_pick_plan_second_xy_corrected.json".format(object_name))

    if execute and not target_visible(args, first_private):
        print("Stop before motion. Move the object into view or change --object-label/--object-id, then rerun.", flush=True)
        return 2
    run(build_plan_command(args, first_private, first_plan), execute)
    if args.instruction and execute:
        selected_objects = matching_target_objects(args, first_private)
        selected_point = object_base_point(selected_objects[0]) if selected_objects else None
        if selected_point is not None:
            args.target_reference_base_xy = selected_point[:2]

    approach = moveit_common(args, first_plan)
    use_object_yaw = plan_has_reliable_yaw(first_plan) if execute else True
    if use_object_yaw:
        approach.extend(
            [
                "--orientation-mode",
                "object-yaw",
                "--grasp-axis",
                args.grasp_axis,
                "--yaw-offset-deg",
                str(args.yaw_offset_deg),
                "--pre-rotate-before-translation",
                "--pre-rotate-strategy",
                "joint-wrist3",
                "--pre-rotate-wrist-direction",
                args.pre_rotate_wrist_direction,
                "--max-pre-rotate-joint-delta",
                str(args.max_pre_rotate_joint_delta),
                "--path-mode",
                "approach",
            ]
        )
    else:
        print("\nFirst plan has no reliable yaw; keeping ready/current orientation for approach.", flush=True)
        approach.extend(["--orientation-mode", "current", "--path-mode", "approach"])
    run(approach, execute)

    if args.instruction and execute:
        # Detector IDs may change after the camera moves. The selected public label
        # is the stable target key for the second snapshot.
        args.object_id = None

    if not capture_second_snapshot_and_plan(args, second_private, second_plan):
        print(
            "\nSecond snapshot could not find/build a plan for {} after {} retry/retries. Stop before pick.".format(
                target_description(args),
                args.second_snapshot_retry_count,
            ),
            flush=True,
        )
        return 3
    if execute:
        build_corrected_plan(
            first_plan,
            second_plan,
            corrected_plan,
            correction_report,
            args.max_correction_m,
            args.max_grasp_offset_m,
        )
    else:
        print("\nWould calculate second_target_xy - first_target_xy and write {}".format(corrected_plan))

    pick = moveit_common(args, corrected_plan)
    pick.extend(["--orientation-mode", "current", "--path-mode", "full"])
    if args.enable_gripper:
        pick.extend([
            "--enable-gripper",
            "--gripper-port",
            args.gripper_port,
            "--skip-gripper-init",
            "--post-close-wait",
            str(args.post_close_wait),
        ])
        if args.release_after_pick and not args.instruction:
            pick.append("--release-after-pick")
    run(pick, execute)

    if args.instruction:
        if execute:
            remaining_plan = remaining_plan_after_step(llm_plan, selected_pick_index)
            write_json(remaining_plan_path, remaining_plan)
            print("\nSaved remaining LLM execution plan: {}".format(remaining_plan_path), flush=True)
            if args.execute_remaining_plan and has_planned_motion(remaining_plan):
                remaining = moveit_common(args, remaining_plan_path)
                remaining.extend(["--all-approaches", "--orientation-mode", "current", "--path-mode", "full"])
                if args.enable_gripper:
                    remaining.extend(
                        [
                            "--enable-gripper",
                            "--gripper-port",
                            args.gripper_port,
                            "--skip-gripper-init",
                            "--post-close-wait",
                            str(args.post_close_wait),
                        ]
                    )
                run(remaining, execute)
            elif not has_planned_motion(remaining_plan):
                print("\nLLM plan has no executable motion after pick; keeping the object held.", flush=True)
            else:
                print("\nRemaining LLM plan execution disabled; keeping the object held.", flush=True)
        else:
            print("\nWould extract the LLM pick target, preserve the remaining plan, and execute it after the corrected pick.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
