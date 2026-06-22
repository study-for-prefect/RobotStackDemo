#!/usr/bin/env python3
"""Hover-only tool offset calibration.

The flow observes one block, moves tool0 to a hover pose above it, and records
the current and reached tool0 poses. It never descends, closes, picks, or places.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robot_scene_pipeline.grasp_orientation import (
    downward_quaternion_for_yaw,
    normalize_quaternion_xyzw,
    quaternion_distance_rad,
)
from robot_scene_pipeline.io_utils import project_path, write_json


DEFAULT_OUTPUT_DIR = os.path.join(
    "runtime",
    "tool_offset_calibration",
    "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
)
# Exact level downward pose. Detected yaw is applied around base Z separately,
# so both gripper fingers remain at the same height for every grasp yaw.
DEFAULT_DOWNWARD_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Move to a hover-only calibration pose above a detected block.")
    parser.add_argument("--label", required=True, help="Target label substring, e.g. green, red, yellow.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hover-height", type=float, default=0.08)
    parser.add_argument("--use-tf", action="store_true")
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--tool-z-offset",
        type=float,
        default=0.15,
        help="Existing stack-demo tool0 Z offset from the hover/TCP target. Keep this consistent with real grasp execution.",
    )
    parser.add_argument("--tool-offset-base", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--tool-offset-yaw-local",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Tool-local offset rotated into base_link by the selected grasp yaw.",
    )
    parser.add_argument(
        "--hover-orientation-mode",
        choices=("grasp_downward", "current", "fixed-yaw"),
        default="grasp_downward",
    )
    parser.add_argument("--fixed-hover-yaw", type=float, default=0.0)
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--safe-pre-rotate-height", type=float, default=0.12)
    parser.add_argument("--conda-env", default="yolo")
    parser.add_argument("--ros-python", default=sys.executable)
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--detector-config", default="config/yolo_detector.json")
    parser.add_argument("--detector-weight", default="")
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--detector-imgsz", type=int, default=None)
    parser.add_argument("--detector-iou", type=float, default=None)
    parser.add_argument("--detector-device", default="")
    parser.add_argument("--object-mask-erode-px", type=int, default=2)
    parser.add_argument("--object-mask-dilate-fallback-px", type=int, default=4)
    parser.add_argument("--object-min-points", type=int, default=40)
    parser.add_argument(
        "--known-object-height-m",
        type=float,
        default=0.0,
        help="Optional measured height prior. Keep 0 for unknown objects; use e.g. 0.0235 for this known square block.",
    )
    parser.add_argument("--velocity", type=float, default=0.08)
    parser.add_argument("--acceleration", type=float, default=0.08)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.20)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.20)
    parser.add_argument(
        "--pre-rotate-strategy",
        choices=("joint-wrist3", "pose"),
        default="joint-wrist3",
    )
    parser.add_argument(
        "--pre-rotate-wrist-yaw-sign",
        choices=("auto", "positive", "negative"),
        default="negative",
        help="UR setup yaw-to-wrist mapping used for safe hover orientation staging.",
    )
    parser.add_argument("--orientation-settle-error-deg", type=float, default=0.2)
    parser.add_argument("--orientation-settle-attempts", type=int, default=2)
    parser.add_argument("--max-grasp-orientation-error-deg", type=float, default=0.5)
    return parser.parse_args()


def run(command):
    print("$ {}".format(" ".join(str(part) for part in command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def quaternion_to_rpy_xyzw(quat_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def quaternion_to_rotvec_xyzw(quat_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(0.0, 1.0 - w * w))
    if scale < 1e-9:
        return [0.0, 0.0, 0.0]
    return [angle * x / scale, angle * y / scale, angle * z / scale]


def pose_payload(position, quat_xyzw):
    rpy = quaternion_to_rpy_xyzw(quat_xyzw)
    return {
        "position": [float(value) for value in position],
        "orientation_xyzw": normalize_quaternion_xyzw(quat_xyzw),
        "rpy_rad": rpy,
        "rpy_deg": [math.degrees(value) for value in rpy],
        "rotation_vector_rad": quaternion_to_rotvec_xyzw(quat_xyzw),
    }


def tf_frame_names(frames_yaml):
    try:
        payload = yaml.safe_load(frames_yaml) or {}
    except Exception:
        return []
    return sorted(str(name).lstrip("/") for name in payload if name)


def tf_name_matches(frame, requested):
    frame = str(frame).lstrip("/")
    requested = str(requested).lstrip("/")
    return frame == requested or frame.endswith("/" + requested) or frame.endswith("_" + requested)


def tf_candidates(frames, requested):
    requested = str(requested).lstrip("/")
    return [requested] + [frame for frame in frames if tf_name_matches(frame, requested) and frame != requested]


def lookup_tool_pose(base_frame, tool_frame, timeout):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    started_here = not rclpy.ok()
    if started_here:
        rclpy.init(args=None)
    node = rclpy.create_node("hover_tool_offset_tf_lookup")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa F841
    deadline = time.time() + float(timeout)
    last_error = None
    try:
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            frames = tf_frame_names(buffer.all_frames_as_yaml())
            for base in tf_candidates(frames, base_frame):
                for tool in tf_candidates(frames, tool_frame):
                    try:
                        if not buffer.can_transform(base, tool, Time(), timeout=Duration(seconds=0.05)):
                            continue
                        transform = buffer.lookup_transform(base, tool, Time(), timeout=Duration(seconds=0.05))
                        t = transform.transform.translation
                        q = transform.transform.rotation
                        pose = pose_payload(
                            [float(t.x), float(t.y), float(t.z)],
                            [float(q.x), float(q.y), float(q.z), float(q.w)],
                        )
                        pose["parent_frame"] = base
                        pose["child_frame"] = tool
                        return pose
                    except Exception as exc:
                        last_error = exc
            time.sleep(0.05)
        raise RuntimeError(
            "Failed to lookup TF {} <- {} within {:.1f}s: {}".format(
                base_frame, tool_frame, float(timeout), last_error
            )
        )
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()


def snapshot_command(args, snapshot_dir):
    command = [
        "conda", "run", "-n", args.conda_env, "python", "-m",
        "robot_scene_pipeline.snapshot_pipeline",
        "--output-dir", snapshot_dir,
        "--skip-llm",
        "--estimate-tabletop",
        "--detector-config", args.detector_config,
        "--base-frame", args.base_frame,
        "--camera-frame", args.camera_frame,
        "--tf-point-mode", "optical-to-camera-link",
        "--object-mask-erode-px", str(args.object_mask_erode_px),
        "--object-mask-dilate-fallback-px", str(args.object_mask_dilate_fallback_px),
        "--object-min-points", str(args.object_min_points),
        "--known-object-height-m", str(args.known_object_height_m),
    ]
    if args.use_tf:
        command.extend(["--use-tf", "--tf-json", args.tf_json])
    if args.detector_weight:
        command.extend(["--detector-weight", args.detector_weight])
    if args.score_thresh is not None:
        command.extend(["--score-thresh", str(args.score_thresh)])
    if args.detector_imgsz is not None:
        command.extend(["--detector-imgsz", str(args.detector_imgsz)])
    if args.detector_iou is not None:
        command.extend(["--detector-iou", str(args.detector_iou)])
    if args.detector_device:
        command.extend(["--detector-device", args.detector_device])
    return command


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def matching_objects(state, label_filter):
    label_filter = str(label_filter).strip().lower()
    objects = []
    for obj in state.get("objects", []):
        if obj.get("is_workspace") or obj.get("label") == "workspace":
            continue
        if label_filter and label_filter not in str(obj.get("label", "")).lower():
            continue
        if not obj.get("pointcloud_geometry_valid"):
            continue
        if obj.get("geometry_frame") != "base_link":
            continue
        center = obj.get("geometry_center_m")
        top_z = obj.get("top_z_base_m")
        if not isinstance(center, list) or len(center) < 3 or top_z is None:
            continue
        objects.append(obj)
    return objects


def choose_target(objects, state=None, label_filter=""):
    if not objects:
        diagnostics = []
        label_filter = str(label_filter).strip().lower()
        if isinstance(state, dict):
            for obj in state.get("objects", []):
                label = str(obj.get("label", ""))
                if label_filter and label_filter not in label.lower():
                    continue
                diagnostics.append(
                    "{}: geometry_valid={} base_valid={} frame={} source={} points(raw/table/outlier/final)="
                    "{}/{}/{}/{} method={} depth_observable={}".format(
                        label or "<unknown>",
                        obj.get("pointcloud_geometry_valid"),
                        obj.get("base_coordinate_valid"),
                        obj.get("geometry_frame"),
                        obj.get("pointcloud_source"),
                        obj.get("pointcloud_raw_point_count"),
                        obj.get("pointcloud_table_filtered_count"),
                        obj.get("pointcloud_outlier_filtered_count"),
                        obj.get("pointcloud_point_count"),
                        obj.get("geometry_estimation_method"),
                        obj.get("depth_geometry_observable"),
                    )
                )
        detail = "; ".join(diagnostics) if diagnostics else "no detector object matched the label"
        raise RuntimeError("No matching geometry-valid base_link object was found. " + detail)
    return max(objects, key=lambda item: float(item.get("confidence", 0.0)))


def normalize_yaw_deg(yaw_deg):
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def closest_equivalent_yaw(yaw_deg, period_deg, current_quaternion_xyzw):
    candidates = [
        normalize_yaw_deg(float(yaw_deg) + index * float(period_deg))
        for index in range(-4, 5)
    ]
    return min(
        candidates,
        key=lambda candidate: quaternion_distance_rad(
            downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, candidate),
            current_quaternion_xyzw,
        ),
    )


def hover_orientation(args, target, current_tool0_pose):
    label = str(target.get("label") or "").lower()
    if args.hover_orientation_mode == "current":
        return (
            normalize_quaternion_xyzw(current_tool0_pose["orientation_xyzw"]),
            None,
            "current_tool0_orientation_debug_only",
        )
    if args.hover_orientation_mode == "fixed-yaw":
        yaw = normalize_yaw_deg(args.fixed_hover_yaw)
        return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, "fixed_hover_yaw"

    detected_yaw = target.get("table_yaw_deg")
    if "square" in label and detected_yaw is not None:
        yaw = closest_equivalent_yaw(
            detected_yaw,
            90.0,
            current_tool0_pose["orientation_xyzw"],
        )
        if abs(yaw) <= max(0.0, float(args.square_yaw_snap_tolerance_deg)):
            yaw = 0.0
            source = "square_detected_90deg_equivalent_snapped"
        else:
            source = "square_detected_90deg_equivalent"
        return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, source

    if target.get("table_yaw_valid") and detected_yaw is not None:
        yaw = closest_equivalent_yaw(
            detected_yaw,
            180.0,
            current_tool0_pose["orientation_xyzw"],
        )
        return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, "table_yaw_180deg_equivalent"

    yaw = normalize_yaw_deg(args.fixed_hover_yaw)
    return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, "invalid_yaw_fixed_hover_yaw"


def yaw_local_offset_to_base(offset, yaw_deg):
    yaw_rad = math.radians(float(yaw_deg))
    dx, dy, dz = [float(value) for value in offset]
    return [
        math.cos(yaw_rad) * dx - math.sin(yaw_rad) * dy,
        math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy,
        dz,
    ]


def effective_tool_offset_base(args, yaw_used):
    base_offset = [float(value) for value in args.tool_offset_base]
    local_offset = [float(value) for value in args.tool_offset_yaw_local]
    if not any(abs(value) > 0.0 for value in local_offset):
        return base_offset, [0.0, 0.0, 0.0]
    if yaw_used is None:
        raise RuntimeError("--tool-offset-yaw-local requires a selected fixed/detected yaw.")
    rotated = yaw_local_offset_to_base(local_offset, yaw_used)
    return [base_offset[i] + rotated[i] for i in range(3)], rotated


def moveit_hover_command(args, hover_position, hover_quat, tool_offset_base):
    command = [
        args.ros_python,
        "tools/moveit_plan_preview.py",
        "--hover-only",
        "--hover-target-base", *[str(value) for value in hover_position],
        "--hover-orientation-xyzw", *[str(value) for value in hover_quat],
        "--tool-z-offset", str(args.tool_z_offset),
        "--tool-offset-base", *[str(value) for value in tool_offset_base],
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


def main():
    args = parse_args()
    if not args.use_tf:
        raise RuntimeError("Hover calibration requires --use-tf so geometry_center_m is in base_link.")
    args.output_dir = project_path(args.output_dir)
    args.tf_json = project_path(args.tf_json)
    args.detector_config = project_path(args.detector_config)
    if args.detector_weight:
        args.detector_weight = project_path(args.detector_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    snapshot_dir = os.path.join(args.output_dir, "snapshot")
    run(snapshot_command(args, snapshot_dir))
    state_path = os.path.join(snapshot_dir, "private_scene_state.json")
    state = load_json(state_path)
    target = choose_target(matching_objects(state, args.label), state=state, label_filter=args.label)
    if target.get("geometry_estimation_method") == "known_height_prior":
        print(
            "WARNING: target geometry uses the explicit {:.4f} m height prior; object height was not observable in depth.".format(
                float(args.known_object_height_m)
            ),
            flush=True,
        )

    current_tool0_pose_before = lookup_tool_pose(args.base_frame, args.tool_frame, args.tf_timeout)
    hover_quat, yaw_used, yaw_source = hover_orientation(args, target, current_tool0_pose_before)
    effective_offset, rotated_local_offset = effective_tool_offset_base(args, yaw_used)
    center = [float(value) for value in target["geometry_center_m"]]
    top_z = float(target["top_z_base_m"])
    hover_position = [center[0], center[1], top_z + float(args.hover_height)]
    tool0_goal_position = [
        hover_position[0] + effective_offset[0],
        hover_position[1] + effective_offset[1],
        hover_position[2] + float(args.tool_z_offset) + effective_offset[2],
    ]

    record = {
        "label": target.get("label"),
        "requested_label": args.label,
        "object_id": target.get("id"),
        "target_geometry_center_base": center,
        "target_top_z_base_m": top_z,
        "target_yaw_used": yaw_used,
        "target_yaw_source": yaw_source,
        "target_table_yaw_deg": target.get("table_yaw_deg"),
        "target_table_yaw_valid": target.get("table_yaw_valid"),
        "target_pointcloud_source": target.get("pointcloud_source"),
        "target_geometry_estimation_method": target.get("geometry_estimation_method"),
        "target_depth_geometry_observable": target.get("depth_geometry_observable"),
        "target_pointcloud_point_count": target.get("pointcloud_point_count"),
        "hover_target_position_base": hover_position,
        "hover_target_orientation_xyzw": hover_quat,
        "hover_target_pose": pose_payload(hover_position, hover_quat),
        "tool0_goal_position_base": tool0_goal_position,
        "current_tool0_pose_before": current_tool0_pose_before,
        "actual_tool0_pose_after": None,
        "current_tool_offset_base": [float(value) for value in args.tool_offset_base],
        "tool_offset_yaw_local": [float(value) for value in args.tool_offset_yaw_local],
        "rotated_tool_offset_yaw_local_base": rotated_local_offset,
        "effective_tool_offset_base": effective_offset,
        "tool_z_offset": float(args.tool_z_offset),
        "hover_height": float(args.hover_height),
        "hover_orientation_mode": args.hover_orientation_mode,
        "safe_pre_rotate_height": float(args.safe_pre_rotate_height),
        "snapshot_path": state.get("snapshot_image") or os.path.join(snapshot_dir, "snapshot.jpg"),
        "annotated_image_path": state.get("annotated_image") or os.path.join(snapshot_dir, "annotated_detector.jpg"),
        "private_scene_state_path": state_path,
        "timestamp": time.time(),
        "execute": bool(args.execute),
    }
    write_json(os.path.join(args.output_dir, "hover_record_before_move.json"), record)

    command = moveit_hover_command(args, hover_position, hover_quat, effective_offset)
    record["moveit_command"] = command
    try:
        run(command)
        record["moveit_status"] = "ok"
    except subprocess.CalledProcessError as exc:
        record["moveit_status"] = "failed"
        record["moveit_returncode"] = int(exc.returncode)
        try:
            record["actual_tool0_pose_after"] = lookup_tool_pose(args.base_frame, args.tool_frame, args.tf_timeout)
        except Exception as tf_exc:
            record["actual_tool0_pose_after_error"] = str(tf_exc)
        summary = {
            "schema_version": "hover_tool_offset_calibration_v1",
            "output_dir": args.output_dir,
            "hover_records": [record],
        }
        write_json(os.path.join(args.output_dir, "summary.json"), summary)
        print(
            "Hover motion failed safely; summary saved to {}".format(
                os.path.join(args.output_dir, "summary.json")
            ),
            file=sys.stderr,
            flush=True,
        )
        return int(exc.returncode or 1)

    record["actual_tool0_pose_after"] = lookup_tool_pose(args.base_frame, args.tool_frame, args.tf_timeout)
    record["timestamp_after"] = time.time()

    summary = {
        "schema_version": "hover_tool_offset_calibration_v1",
        "output_dir": args.output_dir,
        "hover_records": [record],
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("Saved hover calibration summary: {}".format(os.path.join(args.output_dir, "summary.json")), flush=True)
    print(
        "Target {} hover_position={} orientation={} yaw_source={}".format(
            target.get("label"),
            [round(value, 4) for value in hover_position],
            [round(value, 5) for value in hover_quat],
            yaw_source,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
