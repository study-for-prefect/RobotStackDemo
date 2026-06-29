"""Realtime YOLO, depth, tabletop, and TF monitoring loop."""

import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from robot_scene_pipeline.depth_geometry import deproject_pixel_to_point

from .arguments import parse_args
from .constants import PROJECT_ROOT
from .display import (
    auto_ignore_zone,
    box_intersection_area,
    draw_lines,
    finite_xyz,
    object_height_for_display,
    top_z_fallback_from_sample,
    top_z_from_geometry,
)
from .transforms import (
    TfJsonCache,
    optical_to_camera_link,
    sample_depth_m,
    transform_point,
)
from .stream import start_realtime_stream


def _tf_status_text(tf_cache: TfJsonCache, args: Namespace) -> str:
    if tf_cache.error:
        return "TF error: {}".format(tf_cache.error)
    age = " age={:.2f}s".format(tf_cache.json_age_s) if tf_cache.json_age_s is not None else ""
    return "TF ok: {}<-{}{} path={}".format(args.base_frame, args.camera_frame, age, args.tf_json)


def _tf_status_key(tf_cache: TfJsonCache, args: Namespace) -> str:
    if not tf_cache.error:
        return "ok:{}<-{}:{}".format(args.base_frame, args.camera_frame, args.tf_json)
    return "error:" + str(tf_cache.error).split(": age ", 1)[0]


def _print_tf_status_if_changed(tf_cache: TfJsonCache, args: Namespace, last_status: str) -> str:
    status_key = _tf_status_key(tf_cache, args)
    status = _tf_status_text(tf_cache, args)
    if status_key != last_status:
        prefix = "[WARN]" if tf_cache.error else "[INFO]"
        print("{} {}".format(prefix, status), flush=True)
    return status_key


def wait_for_required_tf(tf_cache: TfJsonCache, args: Namespace) -> bool:
    deadline = time.time() + max(0.0, float(args.tf_startup_timeout_s))
    last_status = ""
    while True:
        tf_cache.update(force=True)
        last_status = _print_tf_status_if_changed(tf_cache, args, last_status)
        if tf_cache.T is not None and not tf_cache.error:
            return True
        if getattr(args, "allow_missing_tf", False):
            print("[WARN] continuing without base_link TF because --allow-missing-tf is set", flush=True)
            return False
        if time.time() >= deadline:
            raise RuntimeError(
                "Realtime monitor requires a valid TF JSON for base coordinates. "
                "{}. Expected {}<-{} from path {}".format(
                    tf_cache.error,
                    args.base_frame,
                    args.camera_frame,
                    args.tf_json,
                )
            )
        time.sleep(min(0.2, max(0.01, float(args.tf_reload_s))))


def main() -> None:
    args = parse_args()
    args.known_object_height_m = args.known_block_height_m

    from ultralytics import YOLO

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
    print("[INFO] tf_expected:", "{}<-{}".format(args.base_frame, args.camera_frame))
    print("[INFO] tf_point_mode:", args.tf_point_mode)

    tf_cache = TfJsonCache(
        args.tf_json,
        args.tf_reload_s,
        base_frame=args.base_frame,
        camera_frame=args.camera_frame,
        max_age_s=args.tf_max_age_s,
    )
    last_tf_status = ""
    wait_for_required_tf(tf_cache, args)
    last_tf_status = _tf_status_key(tf_cache, args)

    stream = start_realtime_stream(args)
    try:
        first_frame = stream.read(args)
        if first_frame is None:
            raise RuntimeError("No RGB-D frame received.")
    except Exception:
        stream.close()
        raise
    color_width = int(first_frame.profile.get("color_width") or first_frame.frame_bgr.shape[1])
    color_height = int(first_frame.profile.get("color_height") or first_frame.frame_bgr.shape[0])
    ignore_zone = args.ignore_zone or auto_ignore_zone(color_width, color_height)
    print("[INFO] ignore_zone:", ignore_zone)
    print("[INFO] weight:", args.weight)
    print("[INFO] conf/iou:", args.conf, args.iou)
    print("[INFO] imgsz:", args.imgsz)
    print("[INFO] estimate_tabletop:", args.estimate_tabletop)
    print("[INFO] known_block_height_m:", args.known_block_height_m)

    last_t = time.time()
    fps_smooth = 0.0

    try:
        pending_frame = first_frame
        while True:
            frame_data = pending_frame if pending_frame is not None else stream.read(args)
            pending_frame = None
            if frame_data is None:
                continue

            tf_cache.update(force=False)
            last_tf_status = _print_tf_status_if_changed(tf_cache, args, last_tf_status)
            T_base_camera = tf_cache.T

            intr = frame_data.intrinsics
            depth_frame = frame_data.depth_frame
            frame = frame_data.frame_bgr
            annotated = frame.copy()
            profile = frame_data.profile
            color_width = int(profile.get("color_width") or frame.shape[1])
            color_height = int(profile.get("color_height") or frame.shape[0])
            depth_width = profile.get("depth_width")
            depth_height = profile.get("depth_height")
            fps = profile.get("fps")
            source = profile.get("source", "camera")
            usb_type = profile.get("usb_type_descriptor", "topic")

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
                                deproject_pixel_to_point(intr, [float(cx), float(cy)], float(depth_m)),
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

            size_text = f"{color_width}x{color_height}"
            if depth_width and depth_height:
                size_text += f"/D{int(depth_width)}x{int(depth_height)}"
            rate_text = f"@{fps}" if fps else ""
            status = f"FPS:{fps_smooth:.1f} kept:{kept} {size_text}{rate_text} SRC:{source}"
            if usb_type != "topic":
                status += f" USB:{usb_type}"
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
                alpha = 0.03 if depth.dtype != np.float32 else 120.0
                depth_vis = cv2.convertScaleAbs(depth, alpha=alpha)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                view = np.hstack([annotated, cv2.resize(depth_vis, (annotated.shape[1], annotated.shape[0]))])
            else:
                view = annotated

            cv2.imshow("RobotStackDemo YOLO realtime monitor", view)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:
                break

    finally:
        stream.close()
        cv2.destroyAllWindows()
