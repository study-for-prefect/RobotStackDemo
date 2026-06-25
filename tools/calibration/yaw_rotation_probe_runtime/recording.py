"""Detection, TF lookup, drawing, and record payload helpers."""

import time
from argparse import Namespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from robot_scene_pipeline.depth_geometry import deproject_pixel_to_point
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
    for box, confidence, cls_id in zip(boxes, confs, clss):
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
            point_camera = optical_to_camera_link(point_optical) if args.tf_point_mode == "optical-to-camera-link" else point_optical
            if transform_base_camera is not None:
                point_base = transform_point(transform_base_camera, point_camera)

        detections.append(
            {
                "id": len(detections),
                "label": label_name(model.names, int(cls_id)),
                "label_id": int(cls_id),
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center_px": [int(cx), int(cy)],
                "area_px": int(area),
                "depth_m": float(depth_m),
                "point_optical_xyz": finite_list(point_optical),
                "point_camera_xyz": finite_list(point_camera),
                "point_base_xyz": finite_list(point_base),
            }
        )
    return detections


def select_detection(detections: List[Dict[str, object]], label_filter: str) -> Optional[Dict[str, object]]:
    label_filter = str(label_filter or "").strip().lower()
    candidates = detections
    if label_filter:
        candidates = [det for det in detections if label_filter in str(det.get("label", "")).lower()]
    candidates = [det for det in candidates if det.get("point_base_xyz") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda det: (float(det.get("confidence", 0.0)), int(det.get("area_px", 0))))


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
    source_tool_pose: Tuple[List[float], List[float]],
    source_camera_pose: Tuple[List[float], List[float]],
    targets: Dict[str, object],
) -> Dict[str, object]:
    tool_position, tool_quat = tool_pose
    camera_position, camera_quat = camera_pose
    stem = yaw_file_stem(yaw_deg)
    return {
        "schema_version": "yaw_rotation_record_v1",
        "recorded_at_unix": time.time(),
        "yaw_deg": float(yaw_deg),
        "target_pose_key": stem,
        "base_frame": args.base_frame,
        "tool_frame": args.tool_frame,
        "camera_frame": args.camera_frame,
        "point_camera_frame": args.camera_frame if args.tf_point_mode == "optical-to-camera-link" else "camera_optical_frame",
        "tool0_position": [float(v) for v in tool_position],
        "tool0_quat": [float(v) for v in tool_quat],
        "tool0_pose": pose_payload(tool_position, tool_quat),
        "camera_link_position": [float(v) for v in camera_position],
        "camera_link_quat": [float(v) for v in camera_quat],
        "camera_link_pose": pose_payload(camera_position, camera_quat),
        "point_camera_xyz": selected.get("point_camera_xyz"),
        "point_base_xyz": selected.get("point_base_xyz"),
        "point_optical_xyz": selected.get("point_optical_xyz"),
        "detection": {
            key: selected.get(key)
            for key in ("id", "label", "label_id", "confidence", "bbox", "center_px", "area_px", "depth_m")
        },
        "target_pose": targets["targets"].get(stem),
        "checks": {
            "tool0": pose_checks(tool_position, tool_quat, source_tool_pose[0], source_tool_pose[1]),
            "camera_link": pose_checks(camera_position, camera_quat, source_camera_pose[0], source_camera_pose[1]),
        },
    }
