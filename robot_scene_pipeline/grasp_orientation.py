"""Pure quaternion helpers for downward object-yaw grasp orientations."""

import math


def normalize_quaternion_xyzw(quaternion):
    quaternion = [float(value) for value in quaternion]
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    return [value / norm for value in quaternion]


def quaternion_multiply_xyzw(left, right):
    lx, ly, lz, lw = normalize_quaternion_xyzw(left)
    rx, ry, rz, rw = normalize_quaternion_xyzw(right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def downward_quaternion_for_yaw(base_quaternion_xyzw, yaw_deg):
    half = math.radians(float(yaw_deg)) / 2.0
    yaw_quaternion = [0.0, 0.0, math.sin(half), math.cos(half)]
    return normalize_quaternion_xyzw(quaternion_multiply_xyzw(yaw_quaternion, base_quaternion_xyzw))


def quaternion_distance_rad(left, right):
    left = normalize_quaternion_xyzw(left)
    right = normalize_quaternion_xyzw(right)
    dot = min(1.0, abs(sum(a * b for a, b in zip(left, right))))
    return 2.0 * math.acos(dot)


def object_yaw_orientation(
    target_yaw_deg,
    target_yaw_valid,
    yaw_frame,
    base_quaternion_xyzw,
    current_quaternion_xyzw,
    grasp_axis="long",
    yaw_sign="positive",
    yaw_offset_deg=0.0,
):
    if not target_yaw_valid or target_yaw_deg is None:
        raise ValueError("Object yaw is not reliable.")
    if yaw_frame not in (None, "", "base_link"):
        raise ValueError("Object yaw frame must be base_link, got {}.".format(yaw_frame))
    if yaw_sign == "positive":
        yaw_deg = float(target_yaw_deg)
    elif yaw_sign == "negative":
        yaw_deg = -float(target_yaw_deg)
    else:
        raise ValueError("Unsupported yaw sign: {}".format(yaw_sign))
    if grasp_axis == "short":
        yaw_deg += 90.0
    elif grasp_axis != "long":
        raise ValueError("Unsupported grasp axis: {}".format(grasp_axis))
    yaw_deg += float(yaw_offset_deg)

    candidates = [yaw_deg, yaw_deg + 180.0]
    candidate_quaternions = [downward_quaternion_for_yaw(base_quaternion_xyzw, value) for value in candidates]
    best_index = min(
        range(len(candidates)),
        key=lambda index: quaternion_distance_rad(candidate_quaternions[index], current_quaternion_xyzw),
    )
    selected_yaw = ((candidates[best_index] + 180.0) % 360.0) - 180.0
    return candidate_quaternions[best_index], selected_yaw
