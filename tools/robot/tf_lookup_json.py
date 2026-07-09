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


DEFAULT_OUTPUT = "/tmp/scene_tf_base_color_optical.json"
DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
UR_DYNAMIC_CHAIN_FRAMES = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
)


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
    parser.add_argument(
        "--print-known-frames",
        action="store_true",
        help="Append the full TF frame YAML to connection errors.",
    )
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


def _simple_tf_frame_payload(frames_yaml: str) -> dict:
    payload = {}
    current = None
    for raw_line in str(frames_yaml or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current = line[:-1].strip().strip("'\"").lstrip("/")
            if current:
                payload.setdefault(current, {})
            continue
        if current and line.strip().startswith("parent:"):
            parent = line.split(":", 1)[1].strip().strip("'\"").lstrip("/")
            payload.setdefault(current, {})["parent"] = parent
    return payload


def tf_frame_payload(frames_yaml: str) -> dict:
    try:
        import yaml

        payload = yaml.safe_load(frames_yaml) or {}
    except Exception:
        return _simple_tf_frame_payload(frames_yaml)
    return payload if isinstance(payload, dict) else {}


def tf_frame_names(frames_yaml: str) -> list:
    return sorted(str(name).lstrip("/") for name in tf_frame_payload(frames_yaml) if name)


def _frame_parent(payload: dict, frame: str):
    entry = payload.get(frame) or payload.get("/" + frame)
    if not isinstance(entry, dict):
        return None
    parent = entry.get("parent")
    return str(parent).lstrip("/") if parent else None


def _component_for_frame(payload: dict, frame: str) -> list:
    frame = str(frame).lstrip("/")
    graph = {}
    for child in payload:
        child_name = str(child).lstrip("/")
        graph.setdefault(child_name, set())
        parent = _frame_parent(payload, child_name)
        if parent:
            graph.setdefault(parent, set())
            graph[child_name].add(parent)
            graph[parent].add(child_name)
    if frame not in graph:
        return []
    seen = set()
    stack = [frame]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(graph.get(current, set()) - seen))
    return sorted(seen)


def _component_summary(component: list, limit: int = 10) -> str:
    if not component:
        return "[]"
    shown = component[:limit]
    suffix = "" if len(component) <= limit else "...(+{} more)".format(len(component) - limit)
    return "[{}{}]".format(", ".join(shown), suffix)


def _candidate_frames(frames: list, requested: str) -> list:
    return [frame for frame in frames if frame_matches(frame, requested)]


def tf_tree_diagnosis(frames_yaml: str, base_frame: str, camera_frame: str, tool_frame: str, require_tool: bool) -> str:
    payload = tf_frame_payload(frames_yaml)
    frames = tf_frame_names(frames_yaml)
    if not frames:
        return (
            "No TF frames were received. Start the UR driver / robot_state_publisher and check "
            "ROS_DOMAIN_ID/RMW environment consistency."
        )

    base_matches = _candidate_frames(frames, base_frame)
    camera_matches = _candidate_frames(frames, camera_frame)
    tool_matches = _candidate_frames(frames, tool_frame)
    missing = []
    if not base_matches:
        missing.append("base_frame={}".format(base_frame))
    if not camera_matches:
        missing.append("camera_frame={}".format(camera_frame))
    if require_tool and not tool_matches:
        missing.append("tool_frame={}".format(tool_frame))
    if missing:
        return "Missing requested TF frame candidates: {}.".format(", ".join(missing))

    base_component = _component_for_frame(payload, base_matches[0])
    camera_component = _component_for_frame(payload, camera_matches[0])
    tool_component = _component_for_frame(payload, tool_matches[0]) if tool_matches else []
    dynamic_chain_seen = [frame for frame in UR_DYNAMIC_CHAIN_FRAMES if frame in frames]
    dynamic_chain_missing = [frame for frame in UR_DYNAMIC_CHAIN_FRAMES if frame not in frames]

    details = [
        "base component {}".format(_component_summary(base_component)),
        "camera component {}".format(_component_summary(camera_component)),
    ]
    if require_tool:
        details.append("tool component {}".format(_component_summary(tool_component)))
    if dynamic_chain_missing:
        details.append("missing UR dynamic chain frames {}".format(dynamic_chain_missing))
    if dynamic_chain_seen == ["wrist_3_link"] or (camera_matches[0] in camera_component and not set(base_component) & set(camera_component)):
        details.append(
            "camera/tool static frames exist, but the live joint TF chain from base_link to wrist_3_link is not connected"
        )

    checks = (
        "Check `ros2 topic echo /joint_states --once`, "
        "`ros2 run tf2_ros tf2_echo base_link wrist_3_link`, and that UR driver, "
        "robot_state_publisher, MoveIt, hand-eye static TF, and this process share the same ROS_DOMAIN_ID."
    )
    return "{}. {}".format("; ".join(details), checks)


def frame_candidates(buffer, requested):
    try:
        frames = tf_frame_payload(buffer.all_frames_as_yaml()).keys()
    except Exception:
        return [requested]
    matches = [str(frame).lstrip("/") for frame in frames if frame_matches(frame, requested)]
    requested = str(requested).lstrip("/")
    return [requested] + [frame for frame in matches if frame != requested]


def resolve_frames(buffer, args):
    from rclpy.duration import Duration
    from rclpy.time import Time

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
                tool_transform = None
                for tool_frame in tools:
                    try:
                        tool_transform = buffer.lookup_transform(
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
                tool_transform = None
            return base_frame, camera_frame, connected_tool, camera_transform, tool_transform
    frames_yaml = buffer.all_frames_as_yaml()
    message = (
        "TF trees are not connected for base={} camera={} tool={}: {}\nDiagnosis: {}".format(
            args.base_frame,
            args.camera_frame,
            args.tool_frame,
            last_error,
            tf_tree_diagnosis(frames_yaml, args.base_frame, args.camera_frame, args.tool_frame, args.require_tool),
        )
    )
    if getattr(args, "print_known_frames", False):
        message += "\nKnown TF frames:\n{}".format(frames_yaml)
    raise RuntimeError(message)


def main():
    import rclpy
    from tf2_ros import Buffer, TransformListener

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
                base_frame, camera_frame, tool_frame, transform, tool_transform = resolved
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
                payload["tool_transform"] = transform_payload(tool_transform) if tool_transform is not None else None
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
