import time

import cv2
import numpy as np


def add_depth_args(parser):
    parser.add_argument("--depth-window", type=int, default=7)


def coordinate_convention(camera_frame="camera_color_optical_frame"):
    return {
        "frame": camera_frame,
        "unit": "meter",
        "x": "positive camera-right, negative camera-left",
        "y": "positive downward in optical frame",
        "z": "positive forward from camera; smaller z is closer/in front of larger z",
    }


def color_for_label(label_id):
    rng = np.random.default_rng(label_id * 9973)
    return tuple(int(v) for v in rng.integers(60, 240, size=3))


def robust_depth(depth_frame, cx, cy, window):
    half = max(0, int(window) // 2)
    values = []
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    for y in range(max(0, cy - half), min(height, cy + half + 1)):
        for x in range(max(0, cx - half), min(width, cx + half + 1)):
            value = float(depth_frame.get_distance(x, y))
            if value > 0:
                values.append(value)
    if not values:
        return 0.0
    return float(np.median(values))


def robust_depth_in_bbox(depth_frame, bbox, max_samples=1600):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    x1, y1, x2, y2 = bbox
    x1 = clamp_int(x1, 0, width - 1)
    x2 = clamp_int(x2, 0, width - 1)
    y1 = clamp_int(y1, 0, height - 1)
    y2 = clamp_int(y2, 0, height - 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0, None

    # Prefer the middle of the box to avoid sampling table/background at edges.
    bw = x2 - x1
    bh = y2 - y1
    ix1 = clamp_int(x1 + 0.2 * bw, 0, width - 1)
    ix2 = clamp_int(x2 - 0.2 * bw, 0, width - 1)
    iy1 = clamp_int(y1 + 0.2 * bh, 0, height - 1)
    iy2 = clamp_int(y2 - 0.2 * bh, 0, height - 1)
    if ix2 <= ix1 or iy2 <= iy1:
        ix1, ix2, iy1, iy2 = x1, x2, y1, y2

    area = max(1, (ix2 - ix1 + 1) * (iy2 - iy1 + 1))
    stride = max(1, int(np.sqrt(area / float(max_samples))))
    values = []
    weighted_x = 0.0
    weighted_y = 0.0
    for y in range(iy1, iy2 + 1, stride):
        for x in range(ix1, ix2 + 1, stride):
            value = float(depth_frame.get_distance(x, y))
            if value > 0:
                values.append(value)
                weighted_x += x
                weighted_y += y
    if not values:
        return 0.0, None
    sample_px = [weighted_x / len(values), weighted_y / len(values)]
    return float(np.median(values)), sample_px


def clamp_int(value, low, high):
    return max(low, min(high, int(round(value))))


def deproject_pixel_to_point(intrinsics: object, pixel: object, depth_m: float) -> object:
    if depth_m <= 0:
        return None
    try:
        import pyrealsense2 as rs

        return rs.rs2_deproject_pixel_to_point(intrinsics, pixel, float(depth_m))
    except Exception:
        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        ppx = float(intrinsics.ppx)
        ppy = float(intrinsics.ppy)
        x, y = [float(value) for value in pixel]
        return [(x - ppx) * float(depth_m) / fx, (y - ppy) * float(depth_m) / fy, float(depth_m)]


def attach_3d(detections, depth_frame, intrinsics, depth_window):
    if depth_frame is None:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            det["center_px"] = [cx, cy]
            det["depth_m"] = 0.0
            det["center_3d_m"] = None
            det["coordinate_valid"] = False
        return detections

    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = clamp_int((x1 + x2) / 2.0, 0, depth_width - 1)
        cy = clamp_int((y1 + y2) / 2.0, 0, depth_height - 1)
        depth_m = robust_depth(depth_frame, cx, cy, depth_window)
        sample_px = [float(cx), float(cy)]
        depth_source = "center_window"
        if depth_m <= 0:
            depth_m, bbox_sample_px = robust_depth_in_bbox(depth_frame, det["bbox"])
            if bbox_sample_px is not None:
                sample_px = bbox_sample_px
                depth_source = "bbox_valid_median"
        point = deproject_pixel_to_point(intrinsics, sample_px, float(depth_m)) if depth_m > 0 else None
        det["center_px"] = [cx, cy]
        det["depth_sample_px"] = [round(float(sample_px[0]), 2), round(float(sample_px[1]), 2)]
        det["depth_source"] = depth_source if depth_m > 0 else "missing"
        det["depth_m"] = depth_m
        det["center_3d_m"] = [float(v) for v in point] if point is not None else None
        det["coordinate_valid"] = point is not None
    return detections


def draw_annotated(frame_bgr, detections):
    output = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        color = color_for_label(det["label_id"])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        # The image is policy-facing: keep only a short current-frame ID.
        # Labels, confidence, and geometry remain in detector JSON artifacts.
        text = str(det["id"])
        cv2.putText(output, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return output


def public_object(det):
    geometry_center = det.get("geometry_center_m")
    dimensions = det.get("dimensions_m")
    object_height = float(dimensions[2]) if dimensions and len(dimensions) >= 3 else None
    top_z_base = det.get("top_z_base_m")
    if top_z_base is None and geometry_center and len(geometry_center) >= 3 and object_height is not None and det.get("geometry_frame") == "base_link":
        top_z_base = float(geometry_center[2]) + object_height / 2.0
    return {
        "id": det["id"],
        "label": det["label"],
        "visual_color": det.get("visual_color"),
        "visual_color_confidence": det.get("visual_color_confidence"),
        "visual_color_source": det.get("visual_color_source"),
        "visual_color_pixel_count": det.get("visual_color_pixel_count"),
        "confidence": round(det["confidence"], 4),
        "bbox_xyxy_px": [round(v, 2) for v in det["bbox"]],
        "center_px": det.get("center_px"),
        "depth_m": round(det.get("depth_m", 0.0), 4),
        "center_3d_m": [round(v, 4) for v in det["center_3d_m"]] if det.get("center_3d_m") else None,
        "center_3d_base_m": [round(v, 4) for v in det["center_3d_base_m"]] if det.get("center_3d_base_m") else None,
        "coordinate_valid": bool(det.get("coordinate_valid")),
        "base_coordinate_valid": bool(det.get("base_coordinate_valid")),
        "pointcloud_geometry_valid": bool(det.get("pointcloud_geometry_valid")),
        "pointcloud_source": det.get("pointcloud_source"),
        "geometry_estimation_method": det.get("geometry_estimation_method"),
        "depth_geometry_observable": bool(det.get("depth_geometry_observable")),
        "pointcloud_point_count": det.get("pointcloud_point_count"),
        "pointcloud_raw_point_count": det.get("pointcloud_raw_point_count"),
        "pointcloud_table_filtered_count": det.get("pointcloud_table_filtered_count"),
        "pointcloud_outlier_filtered_count": det.get("pointcloud_outlier_filtered_count"),
        "geometry_frame": det.get("geometry_frame"),
        "geometry_center_m": [round(v, 4) for v in det["geometry_center_m"]] if det.get("geometry_center_m") else None,
        "center_on_table_m": [round(v, 4) for v in det["center_on_table_m"]] if det.get("center_on_table_m") else None,
        "top_surface_center_m": [round(v, 4) for v in det["top_surface_center_m"]] if det.get("top_surface_center_m") else None,
        "dimensions_m": [round(v, 4) for v in det["dimensions_m"]] if det.get("dimensions_m") else None,
        "object_height_m": round(object_height, 4) if object_height is not None else None,
        "top_z_base_m": round(top_z_base, 4) if top_z_base is not None else None,
        "top_z_source": det.get("top_z_source"),
        "local_support_surface": det.get("local_support_surface"),
        "height_estimation_method": det.get("height_estimation_method"),
        "height_surface_cluster": det.get("height_surface_cluster"),
        "height_surface_candidate_count": det.get("height_surface_candidate_count"),
        "table_yaw_deg": round(det["table_yaw_deg"], 2) if det.get("table_yaw_deg") is not None else None,
        "table_yaw_valid": bool(det.get("table_yaw_valid")),
        "table_yaw_source": det.get("table_yaw_source"),
        "footprint_aspect_ratio": round(det["footprint_aspect_ratio"], 3)
        if det.get("footprint_aspect_ratio") is not None
        else None,
    }


def build_private_state(args, detections, snapshot_path, annotated_path, used_profile, table_plane=None):
    """Build perception facts without depending on any task or policy input."""
    camera_frame = getattr(args, "camera_frame", "camera_color_optical_frame")
    base_frame = getattr(args, "base_frame", None)
    coordinate_frame = base_frame if bool(getattr(args, "use_tf", False)) else camera_frame
    return {
        "schema_version": "private_scene_state_v1",
        "scene_id": "snapshot_{}".format(int(time.time() * 1000)),
        "frame_id": coordinate_frame,
        "coordinate_frame": coordinate_frame,
        "timestamp": time.time(),
        "snapshot_image": snapshot_path,
        "annotated_image": annotated_path,
        "camera_profile": used_profile,
        "coordinate_convention": coordinate_convention(camera_frame),
        "base_frame": base_frame,
        "table_plane": table_plane,
        "objects": [
            dict(
                public_object(det),
                label_id=det.get("label_id"),
                is_workspace=det.get("label") == "workspace",
            )
            for det in detections
        ],
    }
