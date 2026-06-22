import argparse
import json
import math
import os
import time

import numpy as np


def add_tf_args(parser):
    parser.add_argument("--use-tf", action="store_true", help="Transform camera 3D points into the robot base frame.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    # 新增绕过 ROS2 动态库的 JSON 桥接参数与坐标系修正模式
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json", help="Bypass ROS2 rclpy by reading static JSON.")
    parser.add_argument("--tf-point-mode", choices=("direct", "optical-to-camera-link"), default="optical-to-camera-link")


def quaternion_to_matrix(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def transform_to_matrix(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = quaternion_to_matrix(q.x, q.y, q.z, q.w)
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat


def resolved_transform_matrix(transform):
    if isinstance(transform, dict):
        return np.asarray(transform["matrix_4x4"], dtype=float)
    return transform_to_matrix(transform)


def apply_transform(matrix, point):
    homogeneous = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=float)
    output = matrix.dot(homogeneous)
    return [float(output[0]), float(output[1]), float(output[2])]


def lookup_transform_matrix(base_frame, camera_frame, timeout_sec):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=None)
    node = rclpy.create_node("scene_pipeline_tf_lookup")
    buffer = Buffer()
    listener = TransformListener(buffer, node)  # noqa F841
    deadline = time.time() + timeout_sec
    last_error = None
    try:
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                transform = buffer.lookup_transform(
                    base_frame,
                    camera_frame,
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
                return transform_to_matrix(transform), transform
            except Exception as exc:  # tf2 exceptions vary across ROS2 distros
                last_error = exc
        raise RuntimeError(
            "Failed to lookup TF {} <- {} within {:.2f}s: {}".format(
                base_frame, camera_frame, timeout_sec, last_error
            )
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def attach_base_coordinates(detections, base_frame, camera_frame, timeout_sec, tf_json="", point_mode="optical-to-camera-link"):
    if tf_json:
        if not os.path.exists(tf_json):
            raise RuntimeError(
                "TF JSON file does not exist: {}. Start tools/tf_lookup_json.py with ROS2 python first, "
                "or pass an existing --tf-json path.".format(tf_json)
            )
        with open(tf_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        matrix = np.array(payload["matrix_4x4"], dtype=float)
        transform_obj = payload
    else:
        matrix, transform_obj = lookup_transform_matrix(base_frame, camera_frame, timeout_sec)

    for det in detections:
        point = det.get("center_3d_m")
        if not point:
            det["center_3d_base_m"] = None
            det["base_coordinate_valid"] = False
            continue

        if point_mode == "optical-to-camera-link":
            x_opt, y_opt, z_opt = point
            point_tf = [z_opt, -x_opt, -y_opt]
        else:
            point_tf = point

        det["center_3d_base_m"] = apply_transform(matrix, point_tf)
        det["base_coordinate_valid"] = True

    return detections, transform_obj


def transform_summary(transform):
    if isinstance(transform, dict):
        return {
            "parent_frame": transform.get("parent_frame", "base_link"),
            "child_frame": transform.get("child_frame", "camera_link"),
            "translation": transform.get("translation", [0.0, 0.0, 0.0]),
            "matrix_4x4": transform.get("matrix_4x4", [])
        }
    t = transform.transform.translation
    q = transform.transform.rotation
    return {
        "parent_frame": transform.header.frame_id,
        "child_frame": transform.child_frame_id,
        "translation": [float(t.x), float(t.y), float(t.z)],
        "rotation_xyzw": [float(q.x), float(q.y), float(q.z), float(q.w)],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the TF chain used by robot_scene_pipeline.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--tf-timeout", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    _, transform = lookup_transform_matrix(args.base_frame, args.camera_frame, args.tf_timeout)
    print("TF lookup OK: {} <- {}".format(args.base_frame, args.camera_frame))
    print(transform_summary(transform))


if __name__ == "__main__":
    main()
