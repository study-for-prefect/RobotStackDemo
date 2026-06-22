#!/usr/bin/env python3
"""Static RGB-D perception repeatability probe.

This tool keeps the robot untouched, repeatedly samples the current scene, and
reports whether geometry is stable enough before tuning robot compensation.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robot_scene_pipeline.depth_geometry import (
    add_depth_args,
    attach_3d,
    build_private_state,
    coordinate_convention,
    draw_annotated,
)
from robot_scene_pipeline.detector_runtime import DetectorModel, add_detector_args
from robot_scene_pipeline.io_utils import project_path, write_json
from robot_scene_pipeline.realsense_capture import (
    active_usb_type,
    add_realsense_args,
    configure_color_sensor,
    configure_depth_sensor,
    depth_scale_from_profile,
    hardware_reset,
    list_realsense_devices,
    make_depth_filters,
    read_aligned_rgbd,
    stream_profiles,
)
from robot_scene_pipeline.snapshot_pipeline import apply_detector_config, drop_transient_detection_fields
from robot_scene_pipeline.tabletop_geometry import add_tabletop_args, attach_tabletop_geometry
from robot_scene_pipeline.tf_transform import (
    add_tf_args,
    attach_base_coordinates,
    resolved_transform_matrix,
    transform_summary,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    "runtime",
    "perception_stability",
    "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture a fixed scene repeatedly and summarize perception stability."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--instruction", default="perception stability probe")
    parser.add_argument("--detector-config", default="config/yolo_detector.json")
    parser.add_argument("--target-object-id", type=int, default=None)
    parser.add_argument(
        "--target-label-contains",
        default="",
        help="Optional case-insensitive label substring filter for recorded objects.",
    )
    parser.add_argument("--pre-capture-delay", type=float, default=0.0)
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--xy-std-threshold-m", type=float, default=0.003)
    parser.add_argument("--top-z-std-threshold-m", type=float, default=0.003)
    parser.add_argument("--dimension-range-threshold-m", type=float, default=0.004)
    parser.add_argument("--yaw-large-jitter-deg", type=float, default=5.0)
    parser.add_argument(
        "--spatial-cluster-radius-m",
        type=float,
        default=0.02,
        help="XY radius used to summarize records as physical object clusters.",
    )
    add_realsense_args(parser)
    add_detector_args(parser)
    add_depth_args(parser)
    add_tf_args(parser)
    add_tabletop_args(parser)
    parser.add_argument("--no-use-tf", action="store_false", dest="use_tf")
    parser.add_argument("--no-estimate-tabletop", action="store_false", dest="estimate_tabletop")
    parser.set_defaults(use_tf=True, estimate_tabletop=True)
    return parser.parse_args()


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


def run_one_sample(args, detector, sample_index, session):
    sample_dir = os.path.join(args.output_dir, "sample_{:03d}".format(sample_index))
    os.makedirs(sample_dir, exist_ok=True)

    frame_bgr, depth_frame, intrinsics, used_profile = capture_from_session(args, session, sample_index)
    snapshot_path = os.path.join(sample_dir, "snapshot.jpg")
    annotated_path = os.path.join(sample_dir, "annotated_detector.jpg")
    objects_path = os.path.join(sample_dir, "detector_objects_3d.json")
    candidates_path = os.path.join(sample_dir, "detector_candidates.json")
    private_state_path = os.path.join(sample_dir, "private_scene_state.json")
    tf_status_path = os.path.join(sample_dir, "tf_status.json")
    tabletop_path = os.path.join(sample_dir, "tabletop_geometry.json")

    if args.save_images:
        cv2.imwrite(snapshot_path, frame_bgr)

    detections, candidates = detector.predict(
        frame_bgr,
        args.score_thresh,
        args.detector_scale,
        args.max_detections,
    )
    write_json(candidates_path, {**detector_metadata(args), "candidates": candidates[: args.debug_topk]})

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
            getattr(args, "tf_point_mode", "optical-to-camera-link"),
        )
        transform_matrix = resolved_transform_matrix(transform)
        tf_payload.update({"status": "ok", "transform": transform_summary(transform)})
    write_json(tf_status_path, tf_payload)

    table_plane = None
    tabletop_payload = {"enabled": bool(args.estimate_tabletop), "status": "not_requested"}
    if args.estimate_tabletop:
        detections, table_plane = attach_tabletop_geometry(
            detections,
            depth_frame,
            intrinsics,
            args,
            transform_matrix=transform_matrix,
        )
        tabletop_payload.update(
            {
                "status": "ok",
                "table_plane": table_plane,
                "valid_object_count": sum(bool(det.get("pointcloud_geometry_valid")) for det in detections),
            }
        )
    write_json(tabletop_path, jsonable(tabletop_payload))

    clean_detections = drop_transient_detection_fields(detections)
    objects_payload = {
        **detector_metadata(args),
        "max_detections": args.max_detections,
        "coordinate_convention": coordinate_convention(args.camera_frame),
        "tf": tf_payload,
        "tabletop": tabletop_payload,
        "objects": clean_detections,
    }
    write_json(objects_path, jsonable(objects_payload))

    if args.save_images:
        annotated = draw_annotated(frame_bgr, clean_detections)
        cv2.imwrite(annotated_path, annotated)

    private_state = build_private_state(
        args,
        clean_detections,
        snapshot_path if args.save_images else None,
        annotated_path if args.save_images else None,
        used_profile,
        table_plane=table_plane,
    )
    write_json(private_state_path, jsonable(private_state))

    records = []
    for obj in private_state.get("objects", []):
        if not should_record_object(args, obj):
            continue
        records.append(
            {
                "sample_index": sample_index,
                "timestamp": private_state.get("timestamp"),
                "object_id": obj.get("id"),
                "label": obj.get("label"),
                "pointcloud_geometry_valid": obj.get("pointcloud_geometry_valid"),
                "geometry_center_m": obj.get("geometry_center_m"),
                "top_z_base_m": obj.get("top_z_base_m"),
                "dimensions_m": obj.get("dimensions_m"),
                "table_yaw_deg": obj.get("table_yaw_deg"),
                "table_yaw_valid": obj.get("table_yaw_valid"),
                "pointcloud_point_count": obj.get("pointcloud_point_count"),
                "sample_dir": sample_dir,
            }
        )
    return records


def finite_array(values, width=None):
    output = []
    for value in values:
        if value is None:
            continue
        if width is None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                output.append(number)
        else:
            if not isinstance(value, (list, tuple)) or len(value) < width:
                continue
            row = []
            ok = True
            for item in value[:width]:
                try:
                    number = float(item)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not math.isfinite(number):
                    ok = False
                    break
                row.append(number)
            if ok:
                output.append(row)
    if not output:
        return np.empty((0, width), dtype=float) if width is not None else np.empty((0,), dtype=float)
    return np.asarray(output, dtype=float)


def sample_std(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    return float(np.std(values, ddof=1))


def axial_yaw_std_deg(yaws_deg):
    yaws = finite_array(yaws_deg)
    if yaws.size < 2:
        return None
    doubled = np.deg2rad(yaws * 2.0)
    resultant = np.abs(np.mean(np.exp(1j * doubled)))
    resultant = max(resultant, 1e-12)
    circular_std = math.sqrt(max(0.0, -2.0 * math.log(resultant)))
    return math.degrees(circular_std) / 2.0


def summarize_group(args, key, rows):
    centers = finite_array([row.get("geometry_center_m") for row in rows], width=3)
    top_z = finite_array([row.get("top_z_base_m") for row in rows])
    dimensions = finite_array([row.get("dimensions_m") for row in rows], width=3)
    point_counts = finite_array([row.get("pointcloud_point_count") for row in rows])
    yaw_values = [row.get("table_yaw_deg") for row in rows]

    xy_std = [sample_std(centers[:, axis]) for axis in (0, 1)] if len(centers) >= 2 else [None, None]
    xy_std_max = None if any(value is None for value in xy_std) else max(xy_std)
    top_z_std = sample_std(top_z)
    dim_std = [sample_std(dimensions[:, axis]) for axis in range(3)] if len(dimensions) >= 2 else [None, None, None]
    dim_range = (
        (np.max(dimensions, axis=0) - np.min(dimensions, axis=0)).astype(float).tolist()
        if len(dimensions) >= 2
        else [None, None, None]
    )
    dim_range_max = None if any(value is None for value in dim_range) else float(max(dim_range))
    point_count_std = sample_std(point_counts)
    point_count_range = (
        float(np.max(point_counts) - np.min(point_counts)) if len(point_counts) >= 2 else None
    )
    point_count_cv = None
    if point_count_std is not None and len(point_counts) and float(np.mean(point_counts)) > 0:
        point_count_cv = float(point_count_std / float(np.mean(point_counts)))
    yaw_std = axial_yaw_std_deg(yaw_values)
    label = key[1]

    checks = {
        "xy_std_pass": xy_std_max is not None and xy_std_max < args.xy_std_threshold_m,
        "top_z_std_pass": top_z_std is not None and top_z_std < args.top_z_std_threshold_m,
        "dimension_range_pass": dim_range_max is not None and dim_range_max < args.dimension_range_threshold_m,
    }
    notes = []
    if xy_std_max is not None and xy_std_max > args.xy_std_threshold_m:
        notes.append("center XY std > {:.3f} m: perception is unstable".format(args.xy_std_threshold_m))
    if top_z_std is not None and top_z_std > args.top_z_std_threshold_m:
        notes.append("top_z std > {:.3f} m: depth/table estimation is unstable".format(args.top_z_std_threshold_m))
    if dim_range_max is not None and dim_range_max > args.dimension_range_threshold_m:
        notes.append("dimension range > {:.3f} m: size estimate is unstable".format(args.dimension_range_threshold_m))
    if yaw_std is not None and yaw_std > args.yaw_large_jitter_deg:
        notes.append("yaw jitter is large")
    if "square" in str(label).lower():
        notes.append("square-like label: do not rely on yaw for compensation")
    if point_count_cv is not None and point_count_cv > 0.20:
        notes.append("point_count coefficient of variation > 20%: check mask/depth alignment or depth noise")

    return {
        "object_id": key[0],
        "label": label,
        "sample_count": len(rows),
        "valid_geometry_count": int(sum(bool(row.get("pointcloud_geometry_valid")) for row in rows)),
        "center_xy_std_m": xy_std,
        "center_xy_std_max_m": xy_std_max,
        "top_z_std_m": top_z_std,
        "dimensions_std_m": dim_std,
        "dimensions_range_m": dim_range,
        "dimensions_range_max_m": dim_range_max,
        "yaw_axial_std_deg": yaw_std,
        "pointcloud_point_count_std": point_count_std,
        "pointcloud_point_count_range": point_count_range,
        "pointcloud_point_count_cv": point_count_cv,
        "checks": checks,
        "pass_acceptance": all(checks.values()),
        "notes": notes,
    }


def summarize_spatial_clusters(args, rows):
    clusters = []
    radius_m = float(getattr(args, "spatial_cluster_radius_m", 0.02))
    for row in rows:
        center = row.get("geometry_center_m")
        if not isinstance(center, list) or len(center) < 2:
            continue
        center_np = np.asarray(center[:3], dtype=float)
        best = None
        best_distance = float("inf")
        for cluster in clusters:
            distance = float(np.linalg.norm(center_np[:2] - cluster["mean_center_m"][:2]))
            if distance < best_distance:
                best = cluster
                best_distance = distance
        if best is not None and best_distance <= radius_m:
            best["rows"].append(row)
            best["mean_center_m"] = np.mean(
                [np.asarray(item["geometry_center_m"][:3], dtype=float) for item in best["rows"]],
                axis=0,
            )
        else:
            clusters.append({"rows": [row], "mean_center_m": center_np})

    summaries = []
    for index, cluster in enumerate(sorted(clusters, key=lambda item: (item["mean_center_m"][0], item["mean_center_m"][1])), 1):
        label_counts = defaultdict(int)
        object_id_counts = defaultdict(int)
        sample_indices = set()
        for row in cluster["rows"]:
            label_counts[str(row.get("label"))] += 1
            object_id_counts[str(row.get("object_id"))] += 1
            sample_indices.add(int(row.get("sample_index")))
        summary = summarize_group(
            args,
            (index, max(label_counts.items(), key=lambda item: item[1])[0]),
            cluster["rows"],
        )
        summary.update(
            {
                "cluster_id": index,
                "mean_geometry_center_m": cluster["mean_center_m"].astype(float).tolist(),
                "unique_sample_count": len(sample_indices),
                "record_count": len(cluster["rows"]),
                "label_counts": dict(label_counts),
                "object_id_counts": dict(object_id_counts),
                "duplicate_records_detected": len(cluster["rows"]) > len(sample_indices),
                "missed_samples": max(0, int(args.samples) - len(sample_indices)),
            }
        )
        summaries.append(summary)
    return summaries


def write_records_csv(path, rows):
    fieldnames = [
        "sample_index",
        "timestamp",
        "object_id",
        "label",
        "pointcloud_geometry_valid",
        "geometry_center_m",
        "top_z_base_m",
        "dimensions_m",
        "table_yaw_deg",
        "table_yaw_valid",
        "pointcloud_point_count",
        "sample_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("geometry_center_m", "dimensions_m"):
                encoded[key] = json.dumps(encoded.get(key), ensure_ascii=False)
            writer.writerow(encoded)


def main():
    args = parse_args()
    args = apply_detector_config(args)
    args.output_dir = project_path(args.output_dir)
    args.image_in = project_path(args.image_in) if args.image_in else ""
    args.detector_weight = project_path(args.detector_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pre_capture_delay > 0:
        print("Waiting {:.2f}s before first capture...".format(args.pre_capture_delay), flush=True)
        time.sleep(args.pre_capture_delay)

    detector = DetectorModel(args)
    session = None
    all_records = []
    sample_errors = []
    try:
        if not args.image_in:
            session = start_realsense_session(args)
        for sample_index in range(1, int(args.samples) + 1):
            print("Capturing sample {}/{}...".format(sample_index, args.samples), flush=True)
            try:
                records = run_one_sample(args, detector, sample_index, session)
            except Exception as exc:
                sample_dir = os.path.join(args.output_dir, "sample_{:03d}".format(sample_index))
                os.makedirs(sample_dir, exist_ok=True)
                error = {
                    "sample_index": sample_index,
                    "timestamp": time.time(),
                    "error": str(exc),
                    "sample_dir": sample_dir,
                }
                sample_errors.append(error)
                write_json(os.path.join(sample_dir, "sample_error.json"), error)
                print("  sample failed: {}".format(exc), flush=True)
                if sample_index < int(args.samples) and args.sleep_s > 0:
                    time.sleep(float(args.sleep_s))
                continue
            all_records.extend(records)
            print("  recorded {} object(s)".format(len(records)), flush=True)
            if sample_index < int(args.samples) and args.sleep_s > 0:
                time.sleep(float(args.sleep_s))
    finally:
        stop_realsense_session(session)

    records_json_path = os.path.join(args.output_dir, "stability_records.json")
    records_csv_path = os.path.join(args.output_dir, "stability_records.csv")
    summary_path = os.path.join(args.output_dir, "stability_summary.json")
    write_json(records_json_path, jsonable({"records": all_records}))
    write_records_csv(records_csv_path, all_records)

    groups = defaultdict(list)
    for row in all_records:
        groups[(row.get("object_id"), row.get("label"))].append(row)
    summaries = [summarize_group(args, key, rows) for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))]
    payload = {
        "schema_version": "perception_stability_probe_v1",
        "output_dir": args.output_dir,
        "sample_count_requested": int(args.samples),
        "sample_error_count": len(sample_errors),
        "record_count": len(all_records),
        "thresholds": {
            "xy_std_m": args.xy_std_threshold_m,
            "top_z_std_m": args.top_z_std_threshold_m,
            "dimension_range_m": args.dimension_range_threshold_m,
            "yaw_large_jitter_deg": args.yaw_large_jitter_deg,
        },
        "summaries": summaries,
        "spatial_cluster_radius_m": float(getattr(args, "spatial_cluster_radius_m", 0.02)),
        "spatial_cluster_summaries": summarize_spatial_clusters(args, all_records),
        "records_json": records_json_path,
        "records_csv": records_csv_path,
        "sample_errors": sample_errors,
        "recommendation": (
            "Do not tune robot compensation until the target object passes XY, top_z, and dimension stability."
        ),
    }
    write_json(summary_path, jsonable(payload))

    print("\nSaved records: {}".format(records_csv_path), flush=True)
    print("Saved summary: {}".format(summary_path), flush=True)
    for item in summaries:
        print(
            "object_id={} label='{}' pass={} xy_std_max={} top_z_std={} dim_range_max={}".format(
                item["object_id"],
                item["label"],
                item["pass_acceptance"],
                item["center_xy_std_max_m"],
                item["top_z_std_m"],
                item["dimensions_range_max_m"],
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
