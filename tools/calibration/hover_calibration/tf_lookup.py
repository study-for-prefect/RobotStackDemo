"""ROS 2 TF lookup for the current tool pose."""

import time

from .pose import pose_payload

def tf_frame_names(frames_yaml):
    import yaml

    try:
        payload = yaml.safe_load(frames_yaml) or {}
    except Exception:
        return []
    return sorted(str(name).lstrip("/") for name in payload if name)


def tf_name_matches(frame, requested):
    frame = str(frame).lstrip("/")
    requested = str(requested).lstrip("/")
    return frame == requested or frame.endswith("/" + requested) or frame.endswith("_" + requested)


def tf_candidates(frames, requested):
    requested = str(requested).lstrip("/")
    return [requested] + [frame for frame in frames if tf_name_matches(frame, requested) and frame != requested]


def lookup_tool_pose(base_frame, tool_frame, timeout):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    started_here = not rclpy.ok()
    if started_here:
        rclpy.init(args=None)
    node = rclpy.create_node("hover_tool_offset_tf_lookup")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa F841
    deadline = time.time() + float(timeout)
    last_error = None
    try:
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            frames = tf_frame_names(buffer.all_frames_as_yaml())
            for base in tf_candidates(frames, base_frame):
                for tool in tf_candidates(frames, tool_frame):
                    try:
                        if not buffer.can_transform(base, tool, Time(), timeout=Duration(seconds=0.05)):
                            continue
                        transform = buffer.lookup_transform(base, tool, Time(), timeout=Duration(seconds=0.05))
                        t = transform.transform.translation
                        q = transform.transform.rotation
                        pose = pose_payload(
                            [float(t.x), float(t.y), float(t.z)],
                            [float(q.x), float(q.y), float(q.z), float(q.w)],
                        )
                        pose["parent_frame"] = base
                        pose["child_frame"] = tool
                        return pose
                    except Exception as exc:
                        last_error = exc
            time.sleep(0.05)
        raise RuntimeError(
            "Failed to lookup TF {} <- {} within {:.1f}s: {}".format(
                base_frame, tool_frame, float(timeout), last_error
            )
        )
    finally:
        node.destroy_node()
        if started_here and rclpy.ok():
            rclpy.shutdown()
