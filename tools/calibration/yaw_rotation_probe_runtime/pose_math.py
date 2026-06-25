"""Pure pose helpers for yaw-only target generation."""

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from robot_scene_pipeline.grasp_orientation import normalize_quaternion_xyzw


DEFAULT_YAWS_DEG: Tuple[int, ...] = (0, 45, -45, 90, -90, 135, -135, 180)


def normalize_yaw_deg(yaw_deg: float) -> float:
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def yaw_file_stem(yaw_deg: float) -> str:
    rounded = int(round(float(yaw_deg)))
    prefix = "p" if rounded >= 0 else "m"
    return "yaw_{}{}deg".format(prefix, abs(rounded))


def rpy_to_quaternion_xyzw(roll: float, pitch: float, yaw: float) -> List[float]:
    cr = math.cos(float(roll) * 0.5)
    sr = math.sin(float(roll) * 0.5)
    cp = math.cos(float(pitch) * 0.5)
    sp = math.sin(float(pitch) * 0.5)
    cy = math.cos(float(yaw) * 0.5)
    sy = math.sin(float(yaw) * 0.5)
    return normalize_quaternion_xyzw(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def quaternion_to_rpy_xyzw(quat_xyzw: Iterable[float]) -> List[float]:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def quaternion_to_rotvec_xyzw(quat_xyzw: Iterable[float]) -> List[float]:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(0.0, 1.0 - w * w))
    if scale < 1e-9:
        return [0.0, 0.0, 0.0]
    return [angle * x / scale, angle * y / scale, angle * z / scale]


def quaternion_xyzw_to_matrix(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> List[float]:
    r = np.asarray(matrix, dtype=float)[:3, :3]
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion_xyzw([x, y, z, w])


def pose_to_matrix(position_m: Iterable[float], quat_xyzw: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(quat_xyzw)
    matrix[:3, 3] = np.asarray(list(position_m), dtype=float)[:3]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> Tuple[List[float], List[float]]:
    matrix = np.asarray(matrix, dtype=float)
    return matrix[:3, 3].astype(float).tolist(), matrix_to_quaternion_xyzw(matrix)


def pose_payload(position_m: Iterable[float], quat_xyzw: Iterable[float]) -> Dict[str, object]:
    quat = normalize_quaternion_xyzw(quat_xyzw)
    rpy = quaternion_to_rpy_xyzw(quat)
    return {
        "position": [float(value) for value in position_m],
        "quat_xyzw": quat,
        "rpy_rad": rpy,
        "rpy_deg": [math.degrees(value) for value in rpy],
        "rotation_vector_rad": quaternion_to_rotvec_xyzw(quat),
    }


def build_yaw_targets(
    tool0_position_m: Iterable[float],
    tool0_quat_xyzw: Iterable[float],
    camera_position_m: Optional[Iterable[float]] = None,
    camera_quat_xyzw: Optional[Iterable[float]] = None,
    yaw_values_deg: Iterable[float] = DEFAULT_YAWS_DEG,
) -> Dict[str, object]:
    tool0_position = [float(value) for value in tool0_position_m]
    source_quat = normalize_quaternion_xyzw(tool0_quat_xyzw)
    source_roll, source_pitch, source_yaw = quaternion_to_rpy_xyzw(source_quat)
    source_tool_matrix = pose_to_matrix(tool0_position, source_quat)

    tool_to_camera = None
    if camera_position_m is not None and camera_quat_xyzw is not None:
        camera_matrix = pose_to_matrix(camera_position_m, camera_quat_xyzw)
        tool_to_camera = np.linalg.inv(source_tool_matrix).dot(camera_matrix)

    targets = {}
    for yaw_deg in yaw_values_deg:
        target_quat = rpy_to_quaternion_xyzw(source_roll, source_pitch, math.radians(float(yaw_deg)))
        target_tool_matrix = pose_to_matrix(tool0_position, target_quat)
        stem = yaw_file_stem(yaw_deg)
        item = {
            "yaw_deg": float(yaw_deg),
            "file_stem": stem,
            "tool0_position": list(tool0_position),
            "tool0_quat": target_quat,
            "tool0_pose": pose_payload(tool0_position, target_quat),
        }
        if tool_to_camera is not None:
            camera_position, camera_quat = matrix_to_pose(target_tool_matrix.dot(tool_to_camera))
            item.update(
                {
                    "camera_link_position": camera_position,
                    "camera_link_quat": camera_quat,
                    "camera_link_pose": pose_payload(camera_position, camera_quat),
                }
            )
        targets[stem] = item

    return {
        "schema_version": "yaw_only_target_poses_v1",
        "yaw_values_deg": [float(value) for value in yaw_values_deg],
        "euler_convention": "ROS quaternion <-> roll/pitch/yaw; target keeps source roll/pitch and replaces base yaw.",
        "source_tool0_position": tool0_position,
        "source_tool0_quat": source_quat,
        "source_tool0_pose": pose_payload(tool0_position, source_quat),
        "source_yaw_deg": math.degrees(source_yaw),
        "targets": targets,
    }
