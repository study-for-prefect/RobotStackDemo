import argparse
import json
import os
import sys
import time

import cv2

from .depth_geometry import add_depth_args, attach_3d, build_private_state, coordinate_convention, draw_annotated
from .detector_runtime import DetectorModel, add_detector_args
from .io_utils import project_path, write_json
from .realsense_capture import add_realsense_args, capture_rgbd
from .ros_topic_capture import capture_rgbd_from_ros_topics
from .tabletop_geometry import add_tabletop_args, attach_tabletop_geometry
from .tf_transform import (
    DEFAULT_TF_POINT_MODE,
    add_tf_args,
    attach_base_coordinates,
    frame_matches,
    resolved_transform_matrix,
    transform_summary,
)
from .visual_color import attach_visual_colors


DEFAULT_OUT = "/tmp/robot_scene_pipeline"


def parse_args():
    parser = argparse.ArgumentParser(description="Single-frame RGB-D robot scene perception pipeline.")
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument(
        "--capture-trigger",
        choices=("immediate", "enter"),
        default="immediate",
        help="When to capture the RGB-D snapshot.",
    )
    parser.add_argument("--pre-capture-delay", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    add_realsense_args(parser)
    add_detector_args(parser)
    parser.add_argument("--detector-config", default="config/yolo_detector.json")
    add_depth_args(parser)
    add_tf_args(parser)
    add_tabletop_args(parser)
    return parser.parse_args()


def cli_flag_present(*flags):
    present = set(sys.argv[1:])
    return any(flag in present for flag in flags)


def apply_detector_config(args):
    config_path = project_path(getattr(args, "detector_config", ""))
    if not config_path or not os.path.exists(config_path):
        return args
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not cli_flag_present("--detector-weight") and config.get("weights"):
        args.detector_weight = config["weights"]
    if not cli_flag_present("--score-thresh") and config.get("conf") is not None:
        args.score_thresh = float(config["conf"])
    if not cli_flag_present("--detector-iou") and config.get("iou") is not None:
        args.detector_iou = float(config["iou"])
    if not cli_flag_present("--detector-imgsz") and config.get("imgsz") is not None:
        args.detector_imgsz = int(config["imgsz"])
    if not cli_flag_present("--detector-device") and config.get("device"):
        args.detector_device = config["device"]
    return args


def wait_for_capture_trigger(args):
    if args.image_in:
        return
    if args.capture_trigger == "enter":
        input("Move the robot to an observe pose, then press Enter to capture RGB-D...")
    if args.pre_capture_delay > 0:
        print("Waiting {:.2f}s before capture...".format(args.pre_capture_delay), flush=True)
        time.sleep(args.pre_capture_delay)


def capture_or_load_snapshot(args):
    if args.image_in:
        args.image_in = project_path(args.image_in)
        frame_bgr = cv2.imread(args.image_in, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError("Failed to read image: {}".format(args.image_in))
        used_profile = {
            "color_width": int(frame_bgr.shape[1]),
            "color_height": int(frame_bgr.shape[0]),
            "depth_width": None,
            "depth_height": None,
            "fps": None,
            "depth_available": False,
            "source": args.image_in,
        }
        return frame_bgr, None, None, used_profile

    wait_for_capture_trigger(args)
    if getattr(args, "camera_source", "ros-topic") == "ros-topic":
        return capture_rgbd_from_ros_topics(args)
    return capture_rgbd(args)


def _profile_coordinate_frame(used_profile, intrinsics):
    frame = str((used_profile or {}).get("coordinate_frame") or "").strip()
    if frame:
        return frame
    return str(getattr(intrinsics, "frame_id", "") or "").strip()


def _sync_camera_frame_with_source(args, used_profile, intrinsics):
    source_frame = _profile_coordinate_frame(used_profile, intrinsics)
    if not source_frame or not getattr(args, "use_tf", False):
        return
    if getattr(args, "tf_point_mode", DEFAULT_TF_POINT_MODE) != "direct":
        return
    if not cli_flag_present("--tf-json"):
        if frame_matches(source_frame, "camera_depth_optical_frame"):
            args.tf_json = "/tmp/scene_tf_base_depth_optical.json"
        elif frame_matches(source_frame, "camera_color_optical_frame"):
            args.tf_json = "/tmp/scene_tf_base_color_optical.json"
    camera_frame_was_explicit = cli_flag_present("--camera-frame")
    if camera_frame_was_explicit:
        if not frame_matches(source_frame, args.camera_frame):
            raise RuntimeError(
                "Point source frame is {}, but --camera-frame is {}. "
                "With --tf-point-mode direct, regenerate/pass TF JSON for the same optical frame.".format(
                    source_frame, args.camera_frame
                )
            )
        return
    if not frame_matches(source_frame, args.camera_frame):
        print(
            "[TF] camera_frame inferred from RGB-D source: {} -> {}".format(
                args.camera_frame, source_frame
            ),
            flush=True,
        )
        args.camera_frame = source_frame


def drop_transient_detection_fields(detections):
    for det in detections:
        det.pop("_mask_bool", None)
        det.pop("_mask", None)
        det.pop("_object_points_base", None)
    return detections


def main():
    args = parse_args()
    args = apply_detector_config(args)
    args.output_dir = project_path(args.output_dir)
    if hasattr(args, "detector_weight"):
        args.detector_weight = project_path(args.detector_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    snapshot_path = os.path.join(args.output_dir, "snapshot.jpg")
    annotated_path = os.path.join(args.output_dir, "annotated_detector.jpg")
    objects_path = os.path.join(args.output_dir, "detector_objects_3d.json")
    candidates_path = os.path.join(args.output_dir, "detector_candidates.json")
    private_state_path = os.path.join(args.output_dir, "private_scene_state.json")
    tf_status_path = os.path.join(args.output_dir, "tf_status.json")
    tabletop_path = os.path.join(args.output_dir, "tabletop_geometry.json")

    frame_bgr, depth_frame, intrinsics, used_profile = capture_or_load_snapshot(args)
    _sync_camera_frame_with_source(args, used_profile, intrinsics)
    cv2.imwrite(snapshot_path, frame_bgr)
    print("Saved snapshot: {}".format(snapshot_path), flush=True)

    detector = DetectorModel(args)
    detections, candidates = detector.predict(
        frame_bgr,
        args.score_thresh,
        args.detector_scale,
        args.max_detections,
    )
    detections = attach_visual_colors(detections, frame_bgr, candidates)
    detector_metadata = {
        "weight": args.detector_weight,
        "score_threshold": args.score_thresh,
        "iou": args.detector_iou,
        "imgsz": args.detector_imgsz,
        "device": args.detector_device,
        "detector_scale": args.detector_scale,
    }
    write_json(candidates_path, {**detector_metadata, "candidates": candidates[: args.debug_topk]})
    print("Saved detector candidates: {}".format(candidates_path), flush=True)

    detections = attach_3d(detections, depth_frame, intrinsics, args.depth_window)
    tf_payload = {
        "enabled": bool(args.use_tf),
        "base_frame": args.base_frame,
        "camera_frame": args.camera_frame,
        "status": "not_requested",
    }
    transform_matrix = None
    if args.use_tf:
        detections, transform = attach_base_coordinates(
            detections,
            args.base_frame,
            args.camera_frame,
            args.tf_timeout,
            getattr(args, "tf_json", ""),
            getattr(args, "tf_point_mode", DEFAULT_TF_POINT_MODE)
        )
        transform_matrix = resolved_transform_matrix(transform)
        tf_payload.update({"status": "ok", "transform": transform_summary(transform)})
    write_json(tf_status_path, tf_payload)

    table_plane = None
    tabletop_payload = {"enabled": bool(args.estimate_tabletop), "status": "not_requested"}
    if args.estimate_tabletop:
        try:
            detections, table_plane = attach_tabletop_geometry(
                detections,
                depth_frame,
                intrinsics,
                args,
                transform_matrix=transform_matrix,
            )
            valid_objects = sum(bool(det.get("pointcloud_geometry_valid")) for det in detections)
            tabletop_payload.update(
                {
                    "status": "ok",
                    "table_plane": table_plane,
                    "valid_object_count": valid_objects,
                }
            )
        except Exception as exc:
            tabletop_payload.update({"status": "error", "error": str(exc)})
            write_json(tabletop_path, tabletop_payload)
            raise
    write_json(tabletop_path, tabletop_payload)
    detections = drop_transient_detection_fields(detections)

    write_json(
        objects_path,
        {
            **detector_metadata,
            "max_detections": args.max_detections,
            "coordinate_convention": coordinate_convention(args.camera_frame),
            "tf": tf_payload,
            "tabletop": tabletop_payload,
            "objects": detections,
        },
    )

    annotated = draw_annotated(frame_bgr, detections)
    cv2.imwrite(annotated_path, annotated)
    print("Saved annotated detector image: {}".format(annotated_path), flush=True)

    private_state = build_private_state(
        args,
        detections,
        snapshot_path,
        annotated_path,
        used_profile,
        table_plane=table_plane,
    )
    write_json(private_state_path, private_state)
    print("Saved private scene state: {}".format(private_state_path), flush=True)

    if args.show:
        cv2.imshow("robot scene pipeline", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
