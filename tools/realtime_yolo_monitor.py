#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("Realtime YOLO-seg monitor for RealSense with base_link 3D coordinates")
    parser.add_argument("--weight", default=str(root / "runs" / "segment" / "building_block5" / "weights" / "best.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--rs-exposure", type=float, default=-1.0)
    parser.add_argument("--rs-gain", type=float, default=-1.0)
    parser.add_argument(
        "--rs-depth-preset",
        choices=("none", "default", "high_accuracy", "high_density", "medium_density"),
        default="high_accuracy",
    )
    parser.add_argument("--rs-emitter-enabled", type=int, default=1)
    parser.add_argument("--rs-laser-power", type=float, default=-1.0)
    parser.add_argument("--require-usb3", action="store_true", dest="require_usb3", default=True)
    parser.add_argument("--allow-usb2", action="store_false", dest="require_usb3")
    parser.add_argument("--allow-low-fps-fallback", action="store_true", default=False)

    # 只过滤夹爪区域。默认按当前分辨率自动取右下角区域。
    parser.add_argument("--ignore-zone", type=int, nargs=4, default=None)

    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument("--max-area", type=int, default=180000)
    parser.add_argument("--show-depth", action="store_true")

    # TF bridge output:
    # python3 tools/tf_lookup_json.py --base-frame base_link --camera-frame camera_link --output /tmp/scene_tf_base_camera.json
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json")
    parser.add_argument("--tf-reload-s", type=float, default=0.2)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument(
        "--tf-point-mode",
        choices=("optical-to-camera-link", "direct"),
        default="optical-to-camera-link",
        help="RealSense deprojection is optical-frame XYZ. Use optical-to-camera-link when tf-json is base_link<-camera_link.",
    )
    parser.add_argument("--depth-sample-radius", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--hide-camera-coord", action="store_true")
    parser.add_argument("--estimate-tabletop", dest="estimate_tabletop", action="store_true", default=True)
    parser.add_argument("--no-estimate-tabletop", dest="estimate_tabletop", action="store_false")
    parser.add_argument(
        "--known-block-height-m",
        type=float,
        default=0.0,
        help="Optional physical-height prior. Default 0 disables inferred height/topZ for unknown objects.",
    )
    parser.add_argument("--plane-point-stride", type=int, default=8)
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.006)
    parser.add_argument("--plane-ransac-iterations", type=int, default=120)
    parser.add_argument("--plane-min-inliers", type=int, default=300)
    parser.add_argument("--plane-max-depth-m", type=float, default=2.0)
    parser.add_argument("--plane-min-up-alignment", type=float, default=0.70)
    parser.add_argument("--object-point-stride", type=int, default=2)
    parser.add_argument("--object-min-height-m", type=float, default=0.004)
    parser.add_argument("--object-max-height-m", type=float, default=0.20)
    parser.add_argument("--object-min-points", type=int, default=25)
    parser.add_argument("--object-mask-erode-px", type=int, default=2)
    parser.add_argument("--object-mask-dilate-fallback-px", type=int, default=4)
    parser.add_argument("--object-outlier-percentile", type=float, default=2.0)
    parser.add_argument("--yaw-min-aspect-ratio", type=float, default=1.20)
    return parser.parse_args()


def option_range(sensor, option):
    try:
        if sensor.supports(option):
            return sensor.get_option_range(option)
    except Exception:
        pass
    return None


def set_option(sensor, option, value, name):
    try:
        if not sensor.supports(option):
            return False
        r = sensor.get_option_range(option)
        value = max(float(r.min), min(float(r.max), float(value)))
        sensor.set_option(option, value)
        print(f"[RS] {name}={value}", flush=True)
        return True
    except Exception as exc:
        print(f"[RS] set {name} failed: {exc}", flush=True)
        return False


def configure_sensors(rs, profile, args):
    preset_values = {
        "default": 1,
        "high_accuracy": 3,
        "high_density": 4,
        "medium_density": 5,
    }
    for sensor in profile.get_device().query_sensors():
        name = sensor.get_info(rs.camera_info.name)
        lower = name.lower()
        if "rgb" in lower:
            if args.rs_exposure >= 0 and sensor.supports(rs.option.enable_auto_exposure):
                set_option(sensor, rs.option.enable_auto_exposure, 0, "rgb_auto_exposure")
                set_option(sensor, rs.option.exposure, args.rs_exposure, "rgb_exposure")
            elif sensor.supports(rs.option.enable_auto_exposure):
                set_option(sensor, rs.option.enable_auto_exposure, 1, "rgb_auto_exposure")
            if args.rs_gain >= 0:
                set_option(sensor, rs.option.gain, args.rs_gain, "rgb_gain")
            if sensor.supports(rs.option.enable_auto_white_balance):
                set_option(sensor, rs.option.enable_auto_white_balance, 1, "rgb_auto_white_balance")
        if "stereo" in lower or "depth" in lower:
            if args.rs_depth_preset != "none" and sensor.supports(rs.option.visual_preset):
                set_option(sensor, rs.option.visual_preset, preset_values[args.rs_depth_preset], f"visual_preset({args.rs_depth_preset})")
            if sensor.supports(rs.option.enable_auto_exposure):
                set_option(sensor, rs.option.enable_auto_exposure, 1, "depth_auto_exposure")
            if sensor.supports(rs.option.emitter_enabled):
                set_option(sensor, rs.option.emitter_enabled, args.rs_emitter_enabled, "emitter_enabled")
            if sensor.supports(rs.option.laser_power):
                laser_power = args.rs_laser_power
                if laser_power < 0:
                    r = option_range(sensor, rs.option.laser_power)
                    laser_power = r.max if r is not None else laser_power
                if laser_power >= 0:
                    set_option(sensor, rs.option.laser_power, laser_power, "laser_power")


def get_usb_type(rs, profile):
    dev = profile.get_device()
    try:
        if dev.supports(rs.camera_info.usb_type_descriptor):
            return str(dev.get_info(rs.camera_info.usb_type_descriptor))
    except Exception:
        pass
    return "unknown"


def stream_profiles(args):
    requested = (args.width, args.height, args.depth_width, args.depth_height, args.fps)
    candidates = [
        requested,
        (1280, 720, 1280, 720, 30),
        (1280, 720, 848, 480, 30),
        (960, 540, 848, 480, 30),
        (640, 480, 848, 480, 30),
        (640, 480, 640, 480, 30),
    ]
    if args.allow_low_fps_fallback:
        candidates.extend(
            [
                (640, 480, 640, 480, 15),
                (640, 480, 640, 360, 15),
            ]
        )
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        yield item


def start_camera(rs, args):
    errors = []
    for width, height, depth_width, depth_height, fps in stream_profiles(args):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        try:
            print(f"[RS] starting color={width}x{height}@{fps} depth={depth_width}x{depth_height}@{fps}", flush=True)
            profile = pipeline.start(config)
            usb_type = get_usb_type(rs, profile)
            print(f"[RS] USB descriptor: {usb_type}", flush=True)
            if args.require_usb3 and not usb_type.startswith("3"):
                pipeline.stop()
                raise RuntimeError(f"RealSense is not USB3.x: {usb_type}")
            configure_sensors(rs, profile, args)
            return pipeline, profile, (width, height, depth_width, depth_height, fps), usb_type
        except RuntimeError as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            msg = f"profile color={width}x{height}@{fps} depth={depth_width}x{depth_height}@{fps} failed: {exc}"
            print("[RS] " + msg, flush=True)
            errors.append(msg)
    raise RuntimeError("All RealSense profiles failed:\n" + "\n".join(errors))


def box_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def auto_ignore_zone(width, height):
    return [int(width * 0.86), int(height * 0.84), int(width), int(height)]


def quat_xyzw_to_rot(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("zero quaternion")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def as_vec3(v):
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float64)
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    raise ValueError(f"cannot parse vec3: {v}")


def as_quat_xyzw(v):
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"]), float(v["w"])], dtype=np.float64)
    if isinstance(v, (list, tuple)) and len(v) == 4:
        return np.array([float(v[0]), float(v[1]), float(v[2]), float(v[3])], dtype=np.float64)
    raise ValueError(f"cannot parse quat xyzw: {v}")


def matrix_from_translation_quat(t, q_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(q_xyzw)
    T[:3, 3] = as_vec3(t)
    return T


def is_4x4_matrix(v):
    try:
        a = np.asarray(v, dtype=np.float64)
        return a.shape == (4, 4)
    except Exception:
        return False


def find_transform_matrix(obj):
    """
    Accept several common tf_lookup_json formats:
    - {"translation": {"x":..}, "rotation": {"x":..,"y":..,"z":..,"w":..}}
    - {"transform": {"translation": ..., "rotation": ...}}
    - {"matrix": [[...]*4]*4}
    - nested equivalents.
    """
    if isinstance(obj, dict):
        for key in (
            "matrix",
            "transform_matrix",
            "T",
            "T_base_camera",
            "base_T_camera",
            "tf_matrix",
            "homogeneous_matrix",
        ):
            if key in obj and is_4x4_matrix(obj[key]):
                return np.asarray(obj[key], dtype=np.float64)

        node = obj.get("transform", obj)
        if isinstance(node, dict):
            trans = None
            rot = None
            for k in ("translation", "trans", "position", "xyz"):
                if k in node:
                    trans = node[k]
                    break
            for k in ("rotation", "quaternion", "quat", "q", "quaternion_xyzw", "rotation_xyzw", "q_xyzw"):
                if k in node:
                    rot = node[k]
                    break
            if trans is not None and rot is not None:
                return matrix_from_translation_quat(trans, as_quat_xyzw(rot))

            # Some tools flatten fields.
            flat_t_keys = ("x", "y", "z")
            flat_q_keys = ("qx", "qy", "qz", "qw")
            if all(k in node for k in flat_t_keys) and all(k in node for k in flat_q_keys):
                t = [node["x"], node["y"], node["z"]]
                q = [node["qx"], node["qy"], node["qz"], node["qw"]]
                return matrix_from_translation_quat(t, q)

        for value in obj.values():
            try:
                return find_transform_matrix(value)
            except Exception:
                pass

    if is_4x4_matrix(obj):
        return np.asarray(obj, dtype=np.float64)

    raise ValueError("no transform matrix found in json")


class TfJsonCache:
    def __init__(self, path, reload_s):
        self.path = Path(path) if path else None
        self.reload_s = float(reload_s)
        self.last_check = 0.0
        self.last_mtime = None
        self.T = None
        self.error = "not loaded"

    def update(self, force=False):
        if self.path is None:
            self.error = "disabled"
            return self.T

        now = time.time()
        if not force and now - self.last_check < self.reload_s:
            return self.T
        self.last_check = now

        try:
            if not self.path.exists():
                self.error = f"missing {self.path}"
                return self.T

            mtime = self.path.stat().st_mtime
            if not force and self.last_mtime == mtime and self.T is not None:
                self.error = None
                return self.T

            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self.T = find_transform_matrix(data)
            self.last_mtime = mtime
            self.error = None
            print(f"[TF] loaded {self.path}", flush=True)
        except Exception as exc:
            self.error = str(exc)

        return self.T


def sample_depth_m(depth_frame, cx, cy, radius, min_depth_m, max_depth_m):
    if depth_frame is None:
        return 0.0

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    xs = range(max(0, cx - radius), min(width - 1, cx + radius) + 1)
    ys = range(max(0, cy - radius), min(height - 1, cy + radius) + 1)

    vals = []
    for y in ys:
        for x in xs:
            d = float(depth_frame.get_distance(int(x), int(y)))
            if min_depth_m <= d <= max_depth_m:
                vals.append(d)

    if not vals:
        return 0.0
    return float(np.median(np.asarray(vals, dtype=np.float32)))


def optical_to_camera_link(p_optical):
    # ROS optical frame: x right, y down, z forward
    # ROS camera_link convention: x forward, y left, z up
    x, y, z = p_optical
    return np.array([z, -x, -y], dtype=np.float64)


def transform_point(T_dst_src, p_src):
    p = np.ones(4, dtype=np.float64)
    p[:3] = p_src
    q = T_dst_src @ p
    return q[:3]


def finite_xyz(p):
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.shape[0] < 3 or not np.all(np.isfinite(a[:3])):
        return None
    return a[:3]


def known_height_m(args):
    h = float(getattr(args, "known_block_height_m", 0.0) or 0.0)
    return h if h > 0 else None


def object_height_for_display(det, args):
    known = known_height_m(args)
    dims = det.get("dimensions_m")
    measured = None
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        try:
            measured = float(dims[2])
        except Exception:
            measured = None
    if measured is not None and np.isfinite(measured) and measured > 0:
        return max(measured, known) if known is not None else measured
    return known


def top_z_from_geometry(det, args):
    center = finite_xyz(det.get("geometry_center_m"))
    height = object_height_for_display(det, args)
    if center is None or height is None:
        return None
    return float(max(center[2] + height / 2.0, height))


def top_z_fallback_from_sample(det, args):
    p_base = finite_xyz(det.get("p_base"))
    height = known_height_m(args)
    if p_base is None or height is None:
        return None
    # The bbox center is only a surface sample. Treat it as an approximate
    # object-center fallback so the displayed top Z is not below a 25 mm block.
    return float(max(p_base[2] + height / 2.0, height))


def draw_lines(img, lines, x, y, color, scale=0.52, thickness=2, line_h=20):
    yy = y
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        yy += line_h


def main():
    args = parse_args()
    args.known_object_height_m = args.known_block_height_m

    import pyrealsense2 as rs

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from robot_scene_pipeline.tabletop_geometry import attach_tabletop_geometry

    weight = Path(args.weight)
    if not weight.exists():
        raise FileNotFoundError(weight)

    model = YOLO(str(weight))
    print("[INFO] loaded:", weight)
    print("[INFO] names:", model.names)
    print("[INFO] tf_json:", args.tf_json)
    print("[INFO] tf_point_mode:", args.tf_point_mode)

    pipeline, profile, used, usb_type = start_camera(rs, args)
    color_width, color_height, depth_width, depth_height, fps = used
    ignore_zone = args.ignore_zone or auto_ignore_zone(color_width, color_height)
    print("[INFO] ignore_zone:", ignore_zone)
    print("[INFO] weight:", args.weight)
    print("[INFO] conf/iou:", args.conf, args.iou)
    print("[INFO] imgsz:", args.imgsz)
    print("[INFO] estimate_tabletop:", args.estimate_tabletop)
    print("[INFO] known_block_height_m:", args.known_block_height_m)

    align = rs.align(rs.stream.color)
    tf_cache = TfJsonCache(args.tf_json, args.tf_reload_s)
    tf_cache.update(force=True)

    last_t = time.time()
    fps_smooth = 0.0

    try:
        for _ in range(max(0, int(args.warmup_frames))):
            pipeline.wait_for_frames(args.frame_timeout_ms)

        while True:
            frames = pipeline.wait_for_frames(args.frame_timeout_ms)
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                continue

            tf_cache.update(force=False)
            T_base_camera = tf_cache.T

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            frame = np.asanyarray(color_frame.get_data())
            annotated = frame.copy()

            ix1, iy1, ix2, iy2 = ignore_zone

            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                "IGNORE",
                (ix1, max(20, iy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            detections = []
            geom_error = None

            if results:
                r = results[0]

                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.detach().cpu().numpy()
                    confs = r.boxes.conf.detach().cpu().numpy()
                    clss = r.boxes.cls.detach().cpu().numpy().astype(int)
                    masks_np = None
                    mask_polygons = None
                    if getattr(r, "masks", None) is not None and r.masks is not None:
                        masks_np = r.masks.data.detach().cpu().numpy()
                        mask_polygons = r.masks.xy

                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, clss)):
                        x1, y1, x2, y2 = box.astype(int).tolist()

                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        if box_intersection_area([x1, y1, x2, y2], ignore_zone) > 0:
                            continue

                        if area < args.min_area or area > args.max_area:
                            continue

                        label = model.names.get(int(cls_id), str(cls_id))

                        depth_m = sample_depth_m(
                            depth_frame,
                            cx,
                            cy,
                            args.depth_sample_radius,
                            args.min_depth_m,
                            args.max_depth_m,
                        )

                        p_optical = None
                        p_camera_link = None
                        p_base = None

                        if depth_m > 0:
                            p_optical = np.asarray(
                                rs.rs2_deproject_pixel_to_point(intr, [float(cx), float(cy)], float(depth_m)),
                                dtype=np.float64,
                            )

                            if args.tf_point_mode == "optical-to-camera-link":
                                p_camera_link = optical_to_camera_link(p_optical)
                            else:
                                p_camera_link = p_optical

                            if T_base_camera is not None:
                                p_base = transform_point(T_base_camera, p_camera_link)

                        det = {
                            "id": len(detections),
                            "label": label,
                            "label_id": int(cls_id),
                            "confidence": float(conf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "center_px": [int(cx), int(cy)],
                            "area_px": int(area),
                            "depth_m": float(depth_m),
                            "p_optical": p_optical,
                            "p_camera_link": p_camera_link,
                            "p_base": p_base,
                        }
                        if masks_np is not None and i < masks_np.shape[0]:
                            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            polygon = mask_polygons[i] if mask_polygons is not None and i < len(mask_polygons) else None
                            if polygon is not None and len(polygon) >= 3:
                                polygon = np.asarray(polygon, dtype=np.float32)
                                polygon[:, 0] = np.clip(polygon[:, 0], 0, frame.shape[1] - 1)
                                polygon[:, 1] = np.clip(polygon[:, 1], 0, frame.shape[0] - 1)
                                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
                                mask = mask.astype(bool)
                                mask_source = "polygon"
                            else:
                                mask = cv2.resize(
                                    masks_np[i].astype(np.float32),
                                    (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                ) > 0.5
                                mask_source = "resized_tensor"
                            det["has_mask"] = True
                            det["mask_shape"] = list(masks_np[i].shape)
                            det["mask_area_px"] = int(np.count_nonzero(mask))
                            det["mask_source"] = mask_source
                            det["_mask_bool"] = mask
                        else:
                            det["has_mask"] = False
                        detections.append(det)

            if args.estimate_tabletop and detections and depth_frame and T_base_camera is not None:
                try:
                    detections, _table_plane = attach_tabletop_geometry(
                        detections,
                        depth_frame,
                        intr,
                        args,
                        transform_matrix=T_base_camera,
                    )
                except Exception as exc:
                    geom_error = str(exc)

            kept = len(detections)

            for det in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                cx, cy = det["center_px"]
                label = det["label"]
                conf = det["confidence"]
                depth_m = det["depth_m"]
                p_optical = finite_xyz(det.get("p_optical"))
                p_base = finite_xyz(det.get("p_base"))
                geometry_center = finite_xyz(det.get("geometry_center_m"))
                height_m = object_height_for_display(det, args)
                top_z = top_z_from_geometry(det, args)
                top_z_source = "geom"
                if top_z is None:
                    top_z = top_z_fallback_from_sample(det, args)
                    top_z_source = "known"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

                if top_z is not None and height_m is not None:
                    lines = [f"{label} {conf:.2f} topZ:{top_z:.3f}m h:{height_m * 100:.1f}cm"]
                else:
                    lines = [f"{label} {conf:.2f} depth:{depth_m:.3f}m"]

                if geometry_center is not None:
                    lines.append(
                        f"center[{geometry_center[0]:+.3f},{geometry_center[1]:+.3f},{geometry_center[2]:+.3f}] {top_z_source}"
                    )
                elif p_base is not None:
                    lines.append(f"sample[{p_base[0]:+.3f},{p_base[1]:+.3f},{p_base[2]:+.3f}]")
                elif depth_m > 0:
                    lines.append("baselink[no tf]")
                else:
                    lines.append("baselink[no depth]")

                if det.get("pointcloud_geometry_valid") and det.get("dimensions_m"):
                    dims = det["dimensions_m"]
                    lines.append(f"dims[{dims[0] * 100:.1f},{dims[1] * 100:.1f},{dims[2] * 100:.1f}]cm")

                # Display RealSense optical camera coordinates in x,y,z order.
                # Here z is depth/forward distance, so it appears as the third value.
                if (not args.hide_camera_coord) and p_optical is not None:
                    lines.append(f"camera[{p_optical[0]:+.3f},{p_optical[1]:+.3f},{p_optical[2]:+.3f}]")

                draw_lines(
                    annotated,
                    lines,
                    x1,
                    max(20, y1 - 8 - 20 * (len(lines) - 1)),
                    (0, 255, 0),
                )

            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_now = 1.0 / dt
                fps_smooth = fps_now if fps_smooth <= 0 else 0.9 * fps_smooth + 0.1 * fps_now

            status = f"FPS:{fps_smooth:.1f} kept:{kept} {color_width}x{color_height}@{fps} USB:{usb_type}"
            if tf_cache.error:
                status += " TF:ERR"
            else:
                status += f" TF:{args.base_frame}<-{args.camera_frame}"

            cv2.putText(
                annotated,
                status,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if tf_cache.error:
                cv2.putText(
                    annotated,
                    f"TF error: {tf_cache.error[:100]}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif geom_error:
                cv2.putText(
                    annotated,
                    f"geometry fallback: {geom_error[:100]}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            if args.show_depth and depth_frame:
                depth = np.asanyarray(depth_frame.get_data())
                depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                view = np.hstack([annotated, cv2.resize(depth_vis, (annotated.shape[1], annotated.shape[0]))])
            else:
                view = annotated

            cv2.imshow("RobotStackDemo YOLO realtime monitor", view)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
