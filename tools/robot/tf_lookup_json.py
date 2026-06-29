"""Write a ROS2 TF lookup to JSON for non-ROS Python environments.

Run this with the ROS2 Python that has rclpy, usually system Python 3.10 on
Ubuntu 22.04 / ROS2 Humble. Detector scripts in the Python 3.7 environment can
read the JSON file without importing rclpy.
"""

import argparse
import json
import math
import os
import time

import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


DEFAULT_OUTPUT = "/tmp/scene_tf_base_color_optical.json"
DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"


def parse_args():
    parser = argparse.ArgumentParser(description="Write base<-camera TF as JSON.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--require-tool", action="store_true")
    parser.add_argument(
        "--no-auto-resolve-frames",
        action="store_false",
        dest="auto_resolve_frames",
        default=True,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def quaternion_to_matrix(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform_payload(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    rot = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    matrix = [
        [rot[0][0], rot[0][1], rot[0][2], float(t.x)],
        [rot[1][0], rot[1][1], rot[1][2], float(t.y)],
        [rot[2][0], rot[2][1], rot[2][2], float(t.z)],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "timestamp": time.time(),
        "parent_frame": transform.header.frame_id,
        "child_frame": transform.child_frame_id,
        "translation": [float(t.x), float(t.y), float(t.z)],
        "rotation_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
        "matrix_4x4": matrix,
    }


def write_json_atomic(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def frame_matches(frame, requested):
    frame = str(frame).lstrip("/")
    requested = str(requested).lstrip("/")
    return frame == requested or frame.endswith("/" + requested) or frame.endswith("_" + requested)


def frame_candidates(buffer, requested):
    try:
        frames = (yaml.safe_load(buffer.all_frames_as_yaml()) or {}).keys()
    except Exception:
        return [requested]
    matches = [str(frame).lstrip("/") for frame in frames if frame_matches(frame, requested)]
    requested = str(requested).lstrip("/")
    return [requested] + [frame for frame in matches if frame != requested]


def resolve_frames(buffer, args):
    bases = frame_candidates(buffer, args.base_frame) if args.auto_resolve_frames else [args.base_frame]
    cameras = frame_candidates(buffer, args.camera_frame) if args.auto_resolve_frames else [args.camera_frame]
    tools = frame_candidates(buffer, args.tool_frame) if args.auto_resolve_frames else [args.tool_frame]
    last_error = None
    for base_frame in bases:
        for camera_frame in cameras:
            try:
                camera_transform = buffer.lookup_transform(
                    base_frame,
                    camera_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
            except Exception as exc:
                last_error = exc
                continue
            if args.require_tool:
                connected_tool = None
                for tool_frame in tools:
                    try:
                        buffer.lookup_transform(
                            base_frame,
                            tool_frame,
                            Time(),
                            timeout=Duration(seconds=0.1),
                        )
                        connected_tool = tool_frame
                        break
                    except Exception as exc:
                        last_error = exc
                if connected_tool is None:
                    continue
            else:
                connected_tool = args.tool_frame
            return base_frame, camera_frame, connected_tool, camera_transform
    raise RuntimeError(
        "TF trees are not connected for base={} camera={} tool={}: {}\nKnown TF frames:\n{}".format(
            args.base_frame,
            args.camera_frame,
            args.tool_frame,
            last_error,
            buffer.all_frames_as_yaml(),
        )
    )


def main():
    args = parse_args()
    rclpy.init(args=None)
    node = rclpy.create_node("tf_lookup_json")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa F841
    period = 1.0 / max(args.rate, 0.1)

    print(
        "Writing TF {} <- {} to {}".format(args.base_frame, args.camera_frame, args.output),
        flush=True,
    )
    try:
        while rclpy.ok():
            deadline = time.time() + args.timeout
            last_error = None
            resolved = None
            while time.time() < deadline and resolved is None:
                rclpy.spin_once(node, timeout_sec=0.05)
                try:
                    resolved = resolve_frames(buffer, args)
                except Exception as exc:
                    last_error = exc
            if resolved is None:
                message = "TF lookup failed: base={} camera={} tool={}: {}".format(
                    args.base_frame, args.camera_frame, args.tool_frame, last_error
                )
                if args.once:
                    raise RuntimeError(message)
                print(message, flush=True)
            else:
                base_frame, camera_frame, tool_frame, transform = resolved
                payload = transform_payload(transform)
                payload["requested_frames"] = {
                    "base_frame": args.base_frame,
                    "camera_frame": args.camera_frame,
                    "tool_frame": args.tool_frame,
                }
                payload["resolved_frames"] = {
                    "base_frame": base_frame,
                    "camera_frame": camera_frame,
                    "tool_frame": tool_frame,
                }
                payload["tool_frame"] = tool_frame
                payload["tool_transform_verified"] = bool(args.require_tool)
                write_json_atomic(args.output, payload)
                print(
                    "ok t=[{:.4f},{:.4f},{:.4f}]".format(*payload["translation"]),
                    flush=True,
                )
                if args.once:
                    break
            time.sleep(period)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
