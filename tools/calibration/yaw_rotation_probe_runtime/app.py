"""Top-level yaw-only rotation probe orchestration."""

import os
from argparse import Namespace
from typing import Any, List, Tuple

import cv2

from robot_scene_pipeline.io_utils import project_path, write_json
from robot_scene_pipeline.ros_topic_capture import RosRgbdSubscriber
from tools.monitoring.realtime_monitor.display import auto_ignore_zone

from .arguments import parse_args
from .pose_math import build_yaw_targets
from .record_modes import run_auto_record_sequence, run_manual_record_loop
from .recording import lookup_pose

Pose = Tuple[List[float], List[float]]


def _source_tool_pose(args: Namespace, subscriber: Any, tf_buffer: Any) -> Pose:
    if args.tool0_position is not None or args.tool0_quat is not None:
        if args.tool0_position is None or args.tool0_quat is None:
            raise RuntimeError("--tool0-position and --tool0-quat must be provided together.")
        return ([float(v) for v in args.tool0_position], [float(v) for v in args.tool0_quat])
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

        tool_pose = _source_tool_pose(args, subscriber, tf_buffer)
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
        )
        targets.update(
            {
                "base_frame": args.base_frame,
                "tool_frame": args.tool_frame,
                "camera_frame": args.camera_frame,
                "output_dir": args.output_dir,
                "record_mode": args.record_mode,
                "code_driven_motion": args.record_mode == "auto",
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
