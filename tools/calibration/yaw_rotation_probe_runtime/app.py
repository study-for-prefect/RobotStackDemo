"""Top-level yaw-only rotation probe orchestration."""

import json
import os
from argparse import Namespace
from typing import Any, List, Tuple

import cv2

from robot_scene_pipeline.io_utils import project_path, write_json
from robot_scene_pipeline.ros_topic_capture import RosRgbdSubscriber
from tools.monitoring.realtime_monitor.display import auto_ignore_zone

from .arguments import parse_args
from .motion import run_ready_motion_command
from .pose_math import build_yaw_targets, vertical_down_quaternion_for_yaw
from .record_modes import run_auto_record_sequence, run_manual_record_loop
from .recording import lookup_pose

Pose = Tuple[List[float], List[float]]


def _load_probe_pose(path: str) -> dict:
    path = project_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{} not found. Create it with tool0_position XYZ in base_link, or pass --tool0-position.".format(path)
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _target_tool_pose(args: Namespace) -> Pose:
    payload = {} if args.tool0_position is not None else _load_probe_pose(args.probe_pose_json)
    if args.tool0_position is not None:
        position = [float(v) for v in args.tool0_position]
    elif args.use_current_tool0_position:
        raise RuntimeError("--use-current-tool0-position is handled after TF lookup.")
    else:
        position = [float(v) for v in payload["tool0_position"]]
    yaw_deg = args.initial_yaw_deg
    if yaw_deg is None:
        yaw_deg = float(payload.get("initial_yaw_deg", 0.0))
    return position, vertical_down_quaternion_for_yaw(float(yaw_deg))


def _current_tool_pose(args: Namespace, subscriber: Any, tf_buffer: Any) -> Pose:
    return lookup_pose(
        tf_buffer,
        subscriber.node,
        subscriber._rclpy,
        args.base_frame,
        args.tool_frame,
        args.tf_timeout,
    )


def main() -> None:
    args = parse_args()
    if getattr(args, "camera_source", "ros-topic") != "ros-topic":
        raise RuntimeError("yaw_rotation_probe only subscribes ROS topics; start the camera driver outside the project.")
    if args.record_mode == "manual" and args.no_window:
        raise RuntimeError("--record-mode manual requires an OpenCV window; remove --no-window.")

    args.output_dir = project_path(args.output_dir)
    args.weight = project_path(args.weight)
    args.probe_pose_json = project_path(args.probe_pose_json)
    args.ready_joint_pose_json = project_path(args.ready_joint_pose_json)
    os.makedirs(args.output_dir, exist_ok=True)
    target_path = project_path(args.target_poses_json) if args.target_poses_json else os.path.join(args.output_dir, "yaw_target_poses.json")

    from tf2_ros import Buffer, TransformListener
    from ultralytics import YOLO

    if not os.path.exists(args.weight):
        raise FileNotFoundError(args.weight)

    model = YOLO(args.weight)
    print("[INFO] loaded:", args.weight)
    print("[INFO] topics:", args.color_topic, args.depth_topic, args.camera_info_topic)
    print("[INFO] record mode:", args.record_mode)
    print("[INFO] motion mode:", "EXECUTE" if args.execute else "PLAN ONLY")

    subscriber = RosRgbdSubscriber(
        args.color_topic,
        args.depth_topic,
        args.camera_info_topic,
        depth_scale_m=args.ros_depth_scale_m,
        node_name="yaw_rotation_probe",
    )
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, subscriber.node)  # noqa F841

    try:
        last_seq = None
        for _ in range(max(0, int(args.warmup_frames))):
            frame = subscriber.wait_for_frame(float(args.frame_timeout_ms) / 1000.0, last_seq, require_depth=True)
            last_seq = frame.color_seq

        first_frame = subscriber.wait_for_frame(float(args.frame_timeout_ms) / 1000.0, last_seq, require_depth=True)
        first_seq = int(first_frame.color_seq)
        ignore_zone = args.ignore_zone or auto_ignore_zone(first_frame.frame_bgr.shape[1], first_frame.frame_bgr.shape[0])

        if args.record_mode == "auto" and not args.skip_initial_ready and args.ready_joint_pose_json:
            print("[INFO] moving/planning initial ready pose:", args.ready_joint_pose_json, flush=True)
            run_ready_motion_command(args)
            if args.execute and args.initial_ready_settle_s > 0:
                subscriber._rclpy.spin_once(subscriber.node, timeout_sec=float(args.initial_ready_settle_s))

        current_tool_pose = _current_tool_pose(args, subscriber, tf_buffer)
        if args.use_current_tool0_position:
            yaw_deg = 0.0 if args.initial_yaw_deg is None else float(args.initial_yaw_deg)
            tool_pose = (current_tool_pose[0], vertical_down_quaternion_for_yaw(yaw_deg))
        else:
            tool_pose = _target_tool_pose(args)
        camera_pose = lookup_pose(
            tf_buffer,
            subscriber.node,
            subscriber._rclpy,
            args.base_frame,
            args.camera_frame,
            args.tf_timeout,
        )
        targets = build_yaw_targets(
            tool_pose[0],
            tool_pose[1],
            camera_pose[0],
            camera_pose[1],
            yaw_values_deg=args.yaw_values_deg,
            reference_tool0_position_m=current_tool_pose[0],
            reference_tool0_quat_xyzw=current_tool_pose[1],
        )
        targets.update(
            {
                "base_frame": args.base_frame,
                "tool_frame": args.tool_frame,
                "camera_frame": args.camera_frame,
                "output_dir": args.output_dir,
                "record_mode": args.record_mode,
                "code_driven_motion": args.record_mode == "auto",
                "probe_pose_json": args.probe_pose_json,
                "ready_joint_pose_json": args.ready_joint_pose_json,
            }
        )
        write_json(target_path, targets)
        print("[INFO] saved yaw target poses:", target_path)

        if args.record_mode == "auto":
            run_auto_record_sequence(
                args,
                model,
                subscriber,
                tf_buffer,
                first_seq,
                ignore_zone,
                tool_pose,
                camera_pose,
                targets,
            )
        else:
            run_manual_record_loop(
                args,
                model,
                subscriber,
                tf_buffer,
                first_frame,
                first_seq,
                ignore_zone,
                tool_pose,
                camera_pose,
                targets,
            )
    finally:
        subscriber.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
