"""Image and RealSense session capture helpers."""

import cv2
import numpy as np

from robot_scene_pipeline.io_utils import project_path
from robot_scene_pipeline.realsense_capture import (
    active_usb_type,
    configure_color_sensor,
    configure_depth_sensor,
    depth_scale_from_profile,
    hardware_reset,
    list_realsense_devices,
    make_depth_filters,
    read_aligned_rgbd,
    stream_profiles,
)

def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def load_image_snapshot(args):
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


def start_realsense_session(args):
    import pyrealsense2 as rs

    list_realsense_devices(rs)
    if args.rs_reset:
        hardware_reset(rs)

    errors = []
    for width, height, depth_width, depth_height, fps in stream_profiles(args):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        align = rs.align(rs.stream.color)
        try:
            print(
                "Starting RealSense color={}x{}@{} depth={}x{}@{}".format(
                    width, height, fps, depth_width, depth_height, fps
                ),
                flush=True,
            )
            profile = pipeline.start(config)
            usb_type = active_usb_type(rs, profile)
            print("Active RealSense USB descriptor: {}".format(usb_type), flush=True)
            if getattr(args, "require_usb3", False) and not usb_type.startswith("3"):
                raise RuntimeError("RealSense is not running on USB3.x: {}".format(usb_type))

            configure_color_sensor(rs, profile, args.rs_exposure, args.rs_gain)
            configure_depth_sensor(rs, profile, args)
            depth_scale = depth_scale_from_profile(rs, profile)
            depth_filters = make_depth_filters(rs) if getattr(args, "rs_depth_postprocess", True) else []
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = color_stream.get_intrinsics()
            return {
                "rs": rs,
                "pipeline": pipeline,
                "align": align,
                "intrinsics": intrinsics,
                "depth_scale": depth_scale,
                "depth_filters": depth_filters,
                "profile": {
                    "color_width": width,
                    "color_height": height,
                    "depth_width": depth_width,
                    "depth_height": depth_height,
                    "fps": fps,
                    "usb_type_descriptor": usb_type,
                    "depth_available": True,
                    "depth_scale_m_per_unit": depth_scale,
                    "depth_postprocess": bool(depth_filters),
                    "depth_profile_requested": [depth_width, depth_height, fps],
                },
            }
        except RuntimeError as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            error = "profile color={}x{}@{} depth={}x{}@{} failed: {}".format(
                width, height, fps, depth_width, depth_height, fps, exc
            )
            print(error, flush=True)
            errors.append(error)
            time.sleep(1.0)
    raise RuntimeError("All RealSense RGB-D profiles failed:\n{}".format("\n".join(errors)))


def stop_realsense_session(session):
    if not session:
        return
    session["pipeline"].stop()


def capture_from_session(args, session, sample_index):
    if args.image_in:
        return load_image_snapshot(args)

    warmup_frames = args.warmup_frames if sample_index == 1 else 0
    frame_bgr, depth_frame, sharpness, sample_count = read_aligned_rgbd(
        session["rs"],
        session["pipeline"],
        session["align"],
        args.frame_timeout_ms,
        warmup_frames,
        getattr(args, "stable_frames", 1),
        session["depth_scale"],
        session["depth_filters"],
    )
    used_profile = dict(session["profile"])
    used_profile.update(
        {
            "stable_frames": sample_count,
            "best_color_sharpness": round(float(sharpness), 3),
            "aligned_depth_width": depth_frame.get_width(),
            "aligned_depth_height": depth_frame.get_height(),
        }
    )
    return frame_bgr, depth_frame, session["intrinsics"], used_profile


def detector_metadata(args):
    return {
        "weight": args.detector_weight,
        "score_threshold": args.score_thresh,
        "iou": args.detector_iou,
        "imgsz": args.detector_imgsz,
        "device": args.detector_device,
        "detector_scale": args.detector_scale,
    }


def should_record_object(args, obj):
    if args.target_object_id is not None and int(obj.get("id", -1)) != int(args.target_object_id):
        return False
    label_filter = str(args.target_label_contains or "").strip().lower()
    if label_filter and label_filter not in str(obj.get("label", "")).lower():
        return False
    return obj.get("label") != "workspace" and not obj.get("is_workspace")
