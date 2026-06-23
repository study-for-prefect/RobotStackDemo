"""Snapshot command and calibration-target selection."""

import json

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
