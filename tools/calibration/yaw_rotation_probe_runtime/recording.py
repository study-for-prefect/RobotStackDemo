"""Detection, TF lookup, drawing, and record payload helpers."""

import time
from argparse import Namespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from robot_scene_pipeline.depth_geometry import deproject_pixel_to_point
from robot_scene_pipeline.tf_transform import validate_point_mode_for_frame
from tools.monitoring.realtime_monitor.display import (
    box_intersection_area,
    draw_lines,
)
from tools.monitoring.realtime_monitor.transforms import optical_to_camera_link, transform_point

from .pose_math import (
    DEFAULT_YAWS_DEG,
    pose_payload,
    quaternion_to_rpy_xyzw,
    yaw_file_stem,
)


def transform_to_pose(transform: Any) -> Tuple[List[float], List[float]]:
    t = transform.transform.translation
    q = transform.transform.rotation
    return [float(t.x), float(t.y), float(t.z)], [float(q.x), float(q.y), float(q.z), float(q.w)]


def lookup_pose(
    buffer: Any,
    node: Any,
    rclpy: Any,
    base_frame: str,
    child_frame: str,
    timeout_sec: float,
) -> Tuple[List[float], List[float]]:
    from rclpy.duration import Duration
    from rclpy.time import Time

    deadline = time.time() + float(timeout_sec)
    last_error = None
    while time.time() < deadline:
        try:
            try:
                rclpy.spin_once(node, timeout_sec=0.02)
            except Exception as exc:
                last_error = exc
            transform = buffer.lookup_transform(
                base_frame,
                child_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            return transform_to_pose(transform)
        except Exception as exc:
            last_error = exc
            try:
                rclpy.spin_once(node, timeout_sec=0.02)
            except Exception as spin_exc:
                last_error = spin_exc
    raise RuntimeError("TF lookup failed: {} <- {}: {}".format(base_frame, child_frame, last_error))


def label_name(names: object, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(cls_id), str(cls_id)))
    if 0 <= int(cls_id) < len(names):  # type: ignore[arg-type]
        return str(names[int(cls_id)])  # type: ignore[index]
    return str(cls_id)


def finite_list(value: Optional[Iterable[float]]) -> Optional[List[float]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape[0] < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return [float(v) for v in array[:3]]


def sample_depth_m(depth_frame: Any, cx: int, cy: int, radius: int, min_depth_m: float, max_depth_m: float) -> float:
    values = []
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    for y in range(max(0, cy - radius), min(height - 1, cy + radius) + 1):
        for x in range(max(0, cx - radius), min(width - 1, cx + radius) + 1):
            depth = float(depth_frame.get_distance(x, y))
            if float(min_depth_m) <= depth <= float(max_depth_m):
                values.append(depth)
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def detect_objects(
    model: Any,
    args: Namespace,
    frame_bgr: np.ndarray,
    depth_frame: Any,
    intrinsics: Any,
    transform_base_camera: Optional[np.ndarray],
    ignore_zone: List[int],
) -> List[Dict[str, object]]:
    tf_point_mode = getattr(args, "tf_point_mode", "direct")
    validate_point_mode_for_frame(args.camera_frame, tf_point_mode)
    results = model.predict(
        source=frame_bgr,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    detections = []
    if not results:
        return detections

    result = results[0]
    if result.boxes is None or len(result.boxes) <= 0:
        return detections

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    masks_np = None
    mask_polygons = None
    if getattr(result, "masks", None) is not None and result.masks is not None:
        masks_np = result.masks.data.detach().cpu().numpy()
        mask_polygons = result.masks.xy

    for index, (box, confidence, cls_id) in enumerate(zip(boxes, confs, clss)):
        x1, y1, x2, y2 = box.astype(int).tolist()
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < int(args.min_area) or area > int(args.max_area):
            continue
        if ignore_zone and box_intersection_area([x1, y1, x2, y2], ignore_zone) > 0:
            continue

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        depth_m = sample_depth_m(depth_frame, cx, cy, args.depth_sample_radius, args.min_depth_m, args.max_depth_m)
        point_optical = None
        point_camera = None
        point_base = None
        if depth_m > 0:
            point_optical = np.asarray(
                deproject_pixel_to_point(intrinsics, [float(cx), float(cy)], float(depth_m)),
                dtype=float,
            )
            point_camera = optical_to_camera_link(point_optical) if tf_point_mode == "optical-to-camera-link" else point_optical
            if transform_base_camera is not None:
                point_base = transform_point(transform_base_camera, point_camera)

        det = {
            "id": len(detections),
            "label": label_name(model.names, int(cls_id)),
            "label_id": int(cls_id),
            "confidence": float(confidence),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "center_px": [int(cx), int(cy)],
            "area_px": int(area),
            "depth_m": float(depth_m),
            "point_camera_frame": args.camera_frame,
            "tf_point_mode": tf_point_mode,
            "point_optical_xyz": finite_list(point_optical),
            "point_camera_xyz": finite_list(point_camera),
            "point_base_xyz": finite_list(point_base),
        }
        if masks_np is not None and index < masks_np.shape[0]:
            mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            polygon = mask_polygons[index] if mask_polygons is not None and index < len(mask_polygons) else None
            if polygon is not None and len(polygon) >= 3:
                polygon = np.asarray(polygon, dtype=np.float32)
                polygon[:, 0] = np.clip(polygon[:, 0], 0, frame_bgr.shape[1] - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, frame_bgr.shape[0] - 1)
                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
                det["mask_source"] = "polygon"
            else:
                mask = cv2.resize(
                    masks_np[index].astype(np.float32),
                    (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
                det["mask_source"] = "resized_tensor"
            det["_mask_bool"] = mask.astype(bool)
            det["has_mask"] = True
            det["mask_area_px"] = int(np.count_nonzero(det["_mask_bool"]))
        else:
            det["has_mask"] = False
        detections.append(det)
    return detections


def select_detection(detections: List[Dict[str, object]], label_filter: str) -> Optional[Dict[str, object]]:
    label_filter = str(label_filter or "").strip().lower()
    geometry_candidates = [
        det for det in detections
        if det.get("pointcloud_geometry_valid") and det.get("geometry_center_m") is not None
    ]
    base_candidates = geometry_candidates or [det for det in detections if det.get("point_base_xyz") is not None]
    candidates = base_candidates
    if label_filter:
        candidates = [det for det in base_candidates if label_filter in str(det.get("label", "")).lower()]
        if not candidates and len(base_candidates) == 1:
            fallback = dict(base_candidates[0])
            fallback["selection_warning"] = (
                "target label filter '{}' matched no detections; selected the only base-valid detection".format(
                    label_filter
                )
            )
            return fallback
    if not candidates:
        return None
    return max(candidates, key=lambda det: (float(det.get("confidence", 0.0)), int(det.get("area_px", 0))))


def selection_failure_reason(detections: List[Dict[str, object]], label_filter: str) -> str:
    label_filter = str(label_filter or "").strip().lower()
    base_candidates = [det for det in detections if det.get("point_base_xyz") is not None]
    labels = sorted({str(det.get("label", "")) for det in detections})
    if not detections:
        return "no detections from model"
    if not base_candidates:
        return "detections exist but none has a valid base_link point; labels={}".format(labels)
    if label_filter:
        matching = [det for det in base_candidates if label_filter in str(det.get("label", "")).lower()]
        if not matching:
            return (
                "base-valid detections exist but none matches target label filter '{}'; labels={}".format(
                    label_filter,
                    labels,
                )
            )
    return "no selected detection with base_link point"


def draw_detections(image: np.ndarray, detections: List[Dict[str, object]], selected_id: Optional[int]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        cx, cy = det["center_px"]
        selected = det["id"] == selected_id
        color = (0, 255, 255) if selected else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (cx, cy), 4, color, -1)
        lines = ["{}#{} {:.2f}".format(det["label"], det["id"], det["confidence"])]
        if det.get("point_base_xyz") is not None:
            x, y, z = det["point_base_xyz"]
            lines.append("base[{:+.3f},{:+.3f},{:+.3f}]".format(x, y, z))
        if det.get("point_camera_xyz") is not None:
            x, y, z = det["point_camera_xyz"]
            lines.append("camera[{:+.3f},{:+.3f},{:+.3f}]".format(x, y, z))
        draw_lines(image, lines, x1, max(20, y1 - 8 - 20 * (len(lines) - 1)), color)


def parse_record_yaw(text: str) -> Optional[int]:
    value = str(text or "").strip().lower().replace("deg", "")
    if value.startswith("p"):
        value = value[1:]
    elif value.startswith("m"):
        value = "-" + value[1:]
    try:
        yaw = int(round(float(value)))
    except ValueError:
        return None
    if yaw == -180:
        yaw = 180
    allowed = {int(v) for v in DEFAULT_YAWS_DEG}
    return yaw if yaw in allowed else None


def angle_delta_deg(current_deg: float, reference_deg: float) -> float:
    return ((float(current_deg) - float(reference_deg) + 180.0) % 360.0) - 180.0


def pose_checks(
    position: Iterable[float],
    quat: Iterable[float],
    reference_position: Iterable[float],
    reference_quat: Iterable[float],
) -> Dict[str, object]:
    rpy = quaternion_to_rpy_xyzw(quat)
    reference_rpy = quaternion_to_rpy_xyzw(reference_quat)
    return {
        "position_delta_from_start_m": [float(position[i]) - float(reference_position[i]) for i in range(3)],
        "rpy_delta_from_start_deg": [
            angle_delta_deg(np.degrees(rpy[0]), np.degrees(reference_rpy[0])),
            angle_delta_deg(np.degrees(rpy[1]), np.degrees(reference_rpy[1])),
            angle_delta_deg(np.degrees(rpy[2]), np.degrees(reference_rpy[2])),
        ],
    }


def build_record(
    args: Namespace,
    yaw_deg: float,
    selected: Dict[str, object],
    tool_pose: Tuple[List[float], List[float]],
    camera_pose: Tuple[List[float], List[float]],
    camera_link_pose: Tuple[List[float], List[float]],
    source_tool_pose: Tuple[List[float], List[float]],
    source_camera_pose: Tuple[List[float], List[float]],
    source_camera_link_pose: Tuple[List[float], List[float]],
    targets: Dict[str, object],
) -> Dict[str, object]:
    tool_position, tool_quat = tool_pose
    camera_position, camera_quat = camera_pose
    camera_link_position, camera_link_quat = camera_link_pose
    stem = yaw_file_stem(yaw_deg)
    tf_point_mode = getattr(args, "tf_point_mode", "direct")
    return {
        "schema_version": "yaw_rotation_record_v1",
        "recorded_at_unix": time.time(),
        "yaw_deg": float(yaw_deg),
        "target_pose_key": stem,
        "base_frame": args.base_frame,
        "tool_frame": args.tool_frame,
        "camera_frame": args.camera_frame,
        "tf_point_mode": tf_point_mode,
        "point_camera_frame": args.camera_frame,
        "tool0_position": [float(v) for v in tool_position],
        "tool0_quat": [float(v) for v in tool_quat],
        "tool0_pose": pose_payload(tool_position, tool_quat),
        "camera_pose_frame": args.camera_frame,
        "camera_frame_position": [float(v) for v in camera_position],
        "camera_frame_quat": [float(v) for v in camera_quat],
        "camera_frame_pose": pose_payload(camera_position, camera_quat),
        "camera_link_position": [float(v) for v in camera_link_position],
        "camera_link_quat": [float(v) for v in camera_link_quat],
        "camera_link_pose": pose_payload(camera_link_position, camera_link_quat),
        "point_camera_xyz": selected.get("point_camera_xyz"),
        "point_base_xyz": selected.get("point_base_xyz"),
        "point_base_source": selected.get("point_base_source"),
        "sample_base_xyz": selected.get("sample_base_xyz"),
        "point_optical_xyz": selected.get("point_optical_xyz"),
        "geometry_center_m": selected.get("geometry_center_m"),
        "center_on_table_m": selected.get("center_on_table_m"),
        "top_surface_center_m": selected.get("top_surface_center_m"),
        "top_z_base_m": selected.get("top_z_base_m"),
        "dimensions_m": selected.get("dimensions_m"),
        "pointcloud_geometry_valid": bool(selected.get("pointcloud_geometry_valid")),
        "pointcloud_source": selected.get("pointcloud_source"),
        "pointcloud_point_count": selected.get("pointcloud_point_count"),
        "pointcloud_point_count_min": selected.get("pointcloud_point_count_min"),
        "geometry_frame": selected.get("geometry_frame"),
        "aggregate_sample_count": selected.get("aggregate_sample_count"),
        "aggregate_samples": selected.get("aggregate_samples"),
        "detection": {
            key: selected.get(key)
            for key in (
                "id",
                "label",
                "label_id",
                "confidence",
                "bbox",
                "center_px",
                "area_px",
                "depth_m",
                "point_camera_frame",
                "tf_point_mode",
                "point_base_source",
                "sample_base_xyz",
                "geometry_center_m",
                "center_on_table_m",
                "top_surface_center_m",
                "top_z_base_m",
                "dimensions_m",
                "pointcloud_geometry_valid",
                "pointcloud_source",
                "pointcloud_point_count",
                "pointcloud_point_count_min",
                "geometry_frame",
                "aggregate_sample_count",
                "selection_warning",
            )
        },
        "target_pose": targets["targets"].get(stem),
        "checks": {
            "tool0": pose_checks(tool_position, tool_quat, source_tool_pose[0], source_tool_pose[1]),
            "camera_frame": pose_checks(camera_position, camera_quat, source_camera_pose[0], source_camera_pose[1]),
            "camera_link": pose_checks(
                camera_link_position,
                camera_link_quat,
                source_camera_link_pose[0],
                source_camera_link_pose[1],
            ),
        },
    }
