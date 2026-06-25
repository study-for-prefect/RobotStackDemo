"""Manual and code-driven recording loops for yaw rotation probe."""

import math
import os
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from robot_scene_pipeline.io_utils import write_json

from .motion import build_yaw_motion_command, run_yaw_motion_command
from .pose_math import pose_to_matrix, yaw_file_stem
from .pose_math import tool_z_axis_base
from .recording import (
    build_record,
    detect_objects,
    draw_detections,
    lookup_pose,
    parse_record_yaw,
    select_detection,
)


Pose = Tuple[List[float], List[float]]


def target_sequence(targets: Dict[str, object]) -> List[Dict[str, object]]:
    output = []
    for yaw_deg in targets.get("yaw_values_deg", []):
        target = targets["targets"].get(yaw_file_stem(float(yaw_deg)))  # type: ignore[index]
        if target is not None:
            output.append(target)
    return output


def show_view(args: Namespace, view: np.ndarray, delay_ms: int) -> int:
    if args.no_window:
        return -1
    cv2.imshow("Yaw rotation probe", view)
    return cv2.waitKey(max(1, int(delay_ms))) & 0xFF


def maybe_depth_view(args: Namespace, annotated: np.ndarray, depth_frame: Any) -> np.ndarray:
    if not args.show_depth:
        return annotated
    depth = np.asarray(depth_frame.get_data())
    depth_vis = cv2.convertScaleAbs(depth, alpha=120.0)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return np.hstack([annotated, cv2.resize(depth_vis, (annotated.shape[1], annotated.shape[0]))])


def current_pose_and_detection(
    args: Namespace,
    model: Any,
    subscriber: Any,
    tf_buffer: Any,
    frame: Any,
    ignore_zone: List[int],
    tf_timeout: float,
) -> Tuple[Optional[Pose], Optional[Pose], Optional[Dict[str, object]], np.ndarray, Optional[str]]:
    tf_error = None
    latest_tool_pose = None
    latest_camera_pose = None
    transform_base_camera = None
    try:
        latest_tool_pose = lookup_pose(
            tf_buffer,
            subscriber.node,
            subscriber._rclpy,
            args.base_frame,
            args.tool_frame,
            tf_timeout,
        )
        latest_camera_pose = lookup_pose(
            tf_buffer,
            subscriber.node,
            subscriber._rclpy,
            args.base_frame,
            args.camera_frame,
            tf_timeout,
        )
        transform_base_camera = pose_to_matrix(latest_camera_pose[0], latest_camera_pose[1])
    except Exception as exc:
        tf_error = str(exc)

    detections = detect_objects(
        model,
        args,
        frame.frame_bgr,
        frame.depth_frame,
        frame.intrinsics,
        transform_base_camera,
        ignore_zone,
    )
    selected = select_detection(detections, args.target_label_contains)
    selected_id = None if selected is None else int(selected["id"])

    annotated = frame.frame_bgr.copy()
    cv2.rectangle(annotated, (ignore_zone[0], ignore_zone[1]), (ignore_zone[2], ignore_zone[3]), (0, 0, 255), 2)
    draw_detections(annotated, detections, selected_id)
    if tf_error:
        cv2.putText(
            annotated,
            "TF error: {}".format(tf_error[:110]),
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 255),
            2,
        )
    return latest_tool_pose, latest_camera_pose, selected, annotated, tf_error


def write_motion_command_preview(args: Namespace, targets: Dict[str, object]) -> str:
    commands = []
    for target in target_sequence(targets):
        commands.append(
            {
                "yaw_deg": target.get("yaw_deg"),
                "file_stem": target.get("file_stem"),
                "command": build_yaw_motion_command(args, target),
            }
        )
    output_path = os.path.join(args.output_dir, "yaw_motion_commands.json")
    write_json(
        output_path,
        {
            "schema_version": "yaw_motion_commands_v1",
            "mode": "EXECUTE" if args.execute else "PLAN_ONLY",
            "note": "Commands drive tool0 to the generated yaw-only poses; no manual yaw rotation is used.",
            "commands": commands,
        },
    )
    return output_path


def angle_between_deg(left: List[float], right: List[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 180.0
    dot = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return math.degrees(math.acos(dot))


def quaternion_distance_deg(left: List[float], right: List[float]) -> float:
    def normalized(quat: List[float]) -> List[float]:
        norm = math.sqrt(sum(float(value) * float(value) for value in quat))
        if norm <= 0.0:
            raise ValueError("Quaternion norm is zero.")
        return [float(value) / norm for value in quat]

    left_q = normalized(left)
    right_q = normalized(right)
    dot = abs(sum(a * b for a, b in zip(left_q, right_q)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def position_error_m(actual: List[float], target: List[float]) -> float:
    return math.sqrt(sum((float(actual[i]) - float(target[i])) ** 2 for i in range(3)))


def pose_check(args: Namespace, tool_pose: Pose, target_pose: Dict[str, object]) -> Dict[str, object]:
    actual_position, actual_quat = tool_pose
    target_position = [float(value) for value in target_pose["tool0_position"]]  # type: ignore[index]
    target_quat = [float(value) for value in target_pose["tool0_quat"]]  # type: ignore[index]
    actual_z_axis = tool_z_axis_base(actual_quat)
    target_z_axis = tool_z_axis_base(target_quat)
    orientation_error = quaternion_distance_deg(actual_quat, target_quat)
    z_axis_error = angle_between_deg(actual_z_axis, target_z_axis)
    position_error = position_error_m(actual_position, target_position)
    ok = (
        orientation_error <= float(args.pose_check_orientation_deg)
        and z_axis_error <= float(args.pose_check_z_axis_deg)
        and position_error <= float(args.pose_check_position_m)
    )
    return {
        "ok": ok,
        "orientation_error_deg": orientation_error,
        "z_axis_error_deg": z_axis_error,
        "position_error_m": position_error,
        "actual_tool0_z_axis_base": actual_z_axis,
        "target_tool0_z_axis_base": target_z_axis,
        "thresholds": {
            "orientation_error_deg": float(args.pose_check_orientation_deg),
            "z_axis_error_deg": float(args.pose_check_z_axis_deg),
            "position_error_m": float(args.pose_check_position_m),
        },
    }


def build_pose_failed_record(
    args: Namespace,
    yaw_deg: float,
    tool_pose: Pose,
    camera_pose: Pose,
    targets: Dict[str, object],
    check: Dict[str, object],
    attempts: int,
) -> Dict[str, object]:
    stem = yaw_file_stem(yaw_deg)
    return {
        "schema_version": "yaw_rotation_pose_failed_record_v1",
        "recorded_at_unix": time.time(),
        "yaw_deg": float(yaw_deg),
        "target_pose_key": stem,
        "status": "pose_failed",
        "reason": "tool0 pose did not reach target tolerance",
        "attempts": int(attempts),
        "base_frame": args.base_frame,
        "tool_frame": args.tool_frame,
        "camera_frame": args.camera_frame,
        "target_pose": targets["targets"].get(stem),
        "tool0_position": [float(v) for v in tool_pose[0]],
        "tool0_quat": [float(v) for v in tool_pose[1]],
        "camera_link_position": [float(v) for v in camera_pose[0]],
        "camera_link_quat": [float(v) for v in camera_pose[1]],
        "pose_check": check,
        "point_camera_xyz": None,
        "point_base_xyz": None,
    }


def build_missed_record(
    args: Namespace,
    yaw_deg: float,
    reason: str,
    tool_pose: Optional[Pose],
    camera_pose: Optional[Pose],
    targets: Dict[str, object],
    attempts: int,
) -> Dict[str, object]:
    stem = yaw_file_stem(yaw_deg)
    record = {
        "schema_version": "yaw_rotation_missed_record_v1",
        "recorded_at_unix": time.time(),
        "yaw_deg": float(yaw_deg),
        "target_pose_key": stem,
        "status": "missed_detection",
        "reason": str(reason),
        "attempts": int(attempts),
        "base_frame": args.base_frame,
        "tool_frame": args.tool_frame,
        "camera_frame": args.camera_frame,
        "target_pose": targets["targets"].get(stem),
        "tool0_position": None,
        "tool0_quat": None,
        "camera_link_position": None,
        "camera_link_quat": None,
        "point_camera_xyz": None,
        "point_base_xyz": None,
    }
    if tool_pose is not None:
        record["tool0_position"] = [float(v) for v in tool_pose[0]]
        record["tool0_quat"] = [float(v) for v in tool_pose[1]]
    if camera_pose is not None:
        record["camera_link_position"] = [float(v) for v in camera_pose[0]]
        record["camera_link_quat"] = [float(v) for v in camera_pose[1]]
    return record


def capture_yaw_record(
    args: Namespace,
    model: Any,
    subscriber: Any,
    tf_buffer: Any,
    last_seq: Optional[int],
    ignore_zone: List[int],
    yaw_deg: float,
    source_tool_pose: Pose,
    source_camera_pose: Pose,
    targets: Dict[str, object],
) -> Tuple[int, bool]:
    last_error = "no attempt"
    last_tool_pose = None
    last_camera_pose = None
    last_annotated = None
    attempts = max(1, int(args.record_attempts))
    for attempt in range(attempts):
        frame = subscriber.wait_for_frame(float(args.frame_timeout_ms) / 1000.0, last_seq, require_depth=True)
        last_seq = frame.color_seq
        tool_pose, camera_pose, selected, annotated, tf_error = current_pose_and_detection(
            args,
            model,
            subscriber,
            tf_buffer,
            frame,
            ignore_zone,
            0.50,
        )
        last_tool_pose = tool_pose
        last_camera_pose = camera_pose
        last_annotated = annotated
        status = "AUTO yaw={:+.1f} attempt={}/{} selected={}".format(
            float(yaw_deg),
            attempt + 1,
            attempts,
            "none" if selected is None else "{}#{}".format(selected["label"], selected["id"]),
        )
        cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
        view = maybe_depth_view(args, annotated, frame.depth_frame)
        key = show_view(args, view, args.preview_ms if selected is not None else 1)
        if key in (27, ord("q")):
            raise KeyboardInterrupt("yaw probe interrupted by user")
        if selected is None or tool_pose is None or camera_pose is None:
            last_error = tf_error or "no selected detection with base_link point"
            continue

        target_pose = targets["targets"].get(yaw_file_stem(yaw_deg))  # type: ignore[index]
        check = pose_check(args, tool_pose, target_pose)
        if not check["ok"]:
            record = build_pose_failed_record(
                args,
                yaw_deg,
                tool_pose,
                camera_pose,
                targets,
                check,
                attempt + 1,
            )
            stem = yaw_file_stem(yaw_deg)
            output_path = os.path.join(args.output_dir, stem + "_pose_failed.json")
            image_path = os.path.join(args.output_dir, stem + "_pose_failed.png")
            write_json(output_path, record)
            cv2.imwrite(image_path, annotated)
            print(
                "[WARN] pose failed yaw {:.1f}: orientation_error={:.2f}deg "
                "z_axis_error={:.2f}deg position_error={:.4f}m; saved {} and {}".format(
                    float(yaw_deg),
                    float(check["orientation_error_deg"]),
                    float(check["z_axis_error_deg"]),
                    float(check["position_error_m"]),
                    output_path,
                    image_path,
                ),
                flush=True,
            )
            return int(last_seq), False

        record = build_record(
            args,
            yaw_deg,
            selected,
            tool_pose,
            camera_pose,
            source_tool_pose,
            source_camera_pose,
            targets,
        )
        record["pose_check"] = check
        stem = yaw_file_stem(yaw_deg)
        output_path = os.path.join(args.output_dir, stem + ".json")
        image_path = os.path.join(args.output_dir, stem + ".png")
        write_json(output_path, record)
        cv2.imwrite(image_path, annotated)
        print("[INFO] saved record: {} and {}".format(output_path, image_path), flush=True)
        return int(last_seq), True

    stem = yaw_file_stem(yaw_deg)
    record = build_missed_record(
        args,
        yaw_deg,
        last_error,
        last_tool_pose,
        last_camera_pose,
        targets,
        attempts,
    )
    output_path = os.path.join(args.output_dir, stem + "_missed.json")
    write_json(output_path, record)
    if last_annotated is not None:
        image_path = os.path.join(args.output_dir, stem + "_missed.png")
        cv2.imwrite(image_path, last_annotated)
        print("[WARN] missed yaw {:.1f}; saved {} and {}".format(float(yaw_deg), output_path, image_path), flush=True)
    else:
        print("[WARN] missed yaw {:.1f}; saved {}".format(float(yaw_deg), output_path), flush=True)
    return int(last_seq), False


def run_auto_record_sequence(
    args: Namespace,
    model: Any,
    subscriber: Any,
    tf_buffer: Any,
    first_seq: int,
    ignore_zone: List[int],
    source_tool_pose: Pose,
    source_camera_pose: Pose,
    targets: Dict[str, object],
) -> None:
    command_path = write_motion_command_preview(args, targets)
    print("[INFO] saved yaw motion commands:", command_path, flush=True)
    last_seq = first_seq
    valid_count = 0
    invalid_count = 0
    for target in target_sequence(targets):
        yaw_deg = float(target["yaw_deg"])
        print("[INFO] code-driven yaw target: {:+.1f} deg".format(yaw_deg), flush=True)
        run_yaw_motion_command(args, target)
        if not args.execute:
            continue
        if args.motion_settle_s > 0:
            time.sleep(float(args.motion_settle_s))
        last_seq, recorded = capture_yaw_record(
            args,
            model,
            subscriber,
            tf_buffer,
            last_seq,
            ignore_zone,
            yaw_deg,
            source_tool_pose,
            source_camera_pose,
            targets,
        )
        if recorded:
            valid_count += 1
        else:
            invalid_count += 1
    if not args.execute:
        print("[INFO] plan-only run finished; add --execute to move the robot and write yaw_p*.json records.", flush=True)
    else:
        print("[INFO] yaw probe finished: valid={} invalid={}".format(valid_count, invalid_count), flush=True)


def run_manual_record_loop(
    args: Namespace,
    model: Any,
    subscriber: Any,
    tf_buffer: Any,
    first_frame: Any,
    first_seq: int,
    ignore_zone: List[int],
    source_tool_pose: Pose,
    source_camera_pose: Pose,
    targets: Dict[str, object],
) -> None:
    pending_frame = first_frame
    last_seq = first_seq
    fps_smooth = 0.0
    last_time = time.time()
    latest_selected = None
    latest_tool_pose = source_tool_pose
    latest_camera_pose = source_camera_pose
    print("[INFO] manual compatibility mode: press r to record a typed yaw, q/esc to quit")

    while True:
        frame = pending_frame if pending_frame is not None else subscriber.wait_for_frame(
            float(args.frame_timeout_ms) / 1000.0,
            last_seq,
            require_depth=True,
        )
        pending_frame = None
        last_seq = frame.color_seq
        tool_pose, camera_pose, selected, annotated, _tf_error = current_pose_and_detection(
            args,
            model,
            subscriber,
            tf_buffer,
            frame,
            ignore_zone,
            0.20,
        )
        latest_selected = selected
        if tool_pose is not None:
            latest_tool_pose = tool_pose
        if camera_pose is not None:
            latest_camera_pose = camera_pose

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_now = 1.0 / dt
            fps_smooth = fps_now if fps_smooth <= 0 else 0.9 * fps_smooth + 0.1 * fps_now

        status = "FPS:{:.1f} det selected:{}  r:record yaw  q:quit".format(
            fps_smooth,
            "none" if latest_selected is None else "{}#{}".format(latest_selected["label"], latest_selected["id"]),
        )
        cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
        view = maybe_depth_view(args, annotated, frame.depth_frame)
        key = show_view(args, view, 1)
        if key in (27, ord("q")):
            break
        if key != ord("r"):
            continue

        yaw_text = input("Record yaw deg (0, +/-45, +/-90, +/-135, 180): ")
        yaw_deg = parse_record_yaw(yaw_text)
        if yaw_deg is None:
            print("[WARN] unsupported yaw:", yaw_text, flush=True)
            continue
        if latest_selected is None:
            print("[WARN] no selected detection with base_link point; not recording", flush=True)
            continue
        record = build_record(
            args,
            yaw_deg,
            latest_selected,
            latest_tool_pose,
            latest_camera_pose,
            source_tool_pose,
            source_camera_pose,
            targets,
        )
        output_path = os.path.join(args.output_dir, yaw_file_stem(yaw_deg) + ".json")
        write_json(output_path, record)
        print("[INFO] saved record:", output_path, flush=True)
