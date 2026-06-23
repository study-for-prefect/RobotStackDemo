"""Hover orientation, pose, and tool-offset calculations."""

import math

from robot_scene_pipeline.grasp_orientation import (
    downward_quaternion_for_yaw,
    normalize_quaternion_xyzw,
    quaternion_distance_rad,
)

from .constants import DEFAULT_DOWNWARD_QUAT_XYZW

def quaternion_to_rpy_xyzw(quat_xyzw):
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


def quaternion_to_rotvec_xyzw(quat_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(0.0, 1.0 - w * w))
    if scale < 1e-9:
        return [0.0, 0.0, 0.0]
    return [angle * x / scale, angle * y / scale, angle * z / scale]


def pose_payload(position, quat_xyzw):
    rpy = quaternion_to_rpy_xyzw(quat_xyzw)
    return {
        "position": [float(value) for value in position],
        "orientation_xyzw": normalize_quaternion_xyzw(quat_xyzw),
        "rpy_rad": rpy,
        "rpy_deg": [math.degrees(value) for value in rpy],
        "rotation_vector_rad": quaternion_to_rotvec_xyzw(quat_xyzw),
    }


def normalize_yaw_deg(yaw_deg):
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def closest_equivalent_yaw(yaw_deg, period_deg, current_quaternion_xyzw):
    candidates = [
        normalize_yaw_deg(float(yaw_deg) + index * float(period_deg))
        for index in range(-4, 5)
    ]
    return min(
        candidates,
        key=lambda candidate: quaternion_distance_rad(
            downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, candidate),
            current_quaternion_xyzw,
        ),
    )


def hover_orientation(args, target, current_tool0_pose):
    label = str(target.get("label") or "").lower()
    if args.hover_orientation_mode == "current":
        return (
            normalize_quaternion_xyzw(current_tool0_pose["orientation_xyzw"]),
            None,
            "current_tool0_orientation_debug_only",
        )
    if args.hover_orientation_mode == "fixed-yaw":
        yaw = normalize_yaw_deg(args.fixed_hover_yaw)
        return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, "fixed_hover_yaw"

    detected_yaw = target.get("table_yaw_deg")
    if "square" in label and detected_yaw is not None:
        yaw = closest_equivalent_yaw(
            detected_yaw,
            90.0,
            current_tool0_pose["orientation_xyzw"],
        )
        if abs(yaw) <= max(0.0, float(args.square_yaw_snap_tolerance_deg)):
            yaw = 0.0
            source = "square_detected_90deg_equivalent_snapped"
        else:
            source = "square_detected_90deg_equivalent"
        return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, source

    if target.get("table_yaw_valid") and detected_yaw is not None:
        yaw = closest_equivalent_yaw(
            detected_yaw,
            180.0,
            current_tool0_pose["orientation_xyzw"],
        )
        return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, "table_yaw_180deg_equivalent"

    yaw = normalize_yaw_deg(args.fixed_hover_yaw)
    return downward_quaternion_for_yaw(DEFAULT_DOWNWARD_QUAT_XYZW, yaw), yaw, "invalid_yaw_fixed_hover_yaw"


def yaw_local_offset_to_base(offset, yaw_deg):
    yaw_rad = math.radians(float(yaw_deg))
    dx, dy, dz = [float(value) for value in offset]
    return [
        math.cos(yaw_rad) * dx - math.sin(yaw_rad) * dy,
        math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy,
        dz,
    ]


def effective_tool_offset_base(args, yaw_used):
    base_offset = [float(value) for value in args.tool_offset_base]
    local_offset = [float(value) for value in args.tool_offset_yaw_local]
    if not any(abs(value) > 0.0 for value in local_offset):
        return base_offset, [0.0, 0.0, 0.0]
    if yaw_used is None:
        raise RuntimeError("--tool-offset-yaw-local requires a selected fixed/detected yaw.")
    rotated = yaw_local_offset_to_base(local_offset, yaw_used)
    return [base_offset[i] + rotated[i] for i in range(3)], rotated
