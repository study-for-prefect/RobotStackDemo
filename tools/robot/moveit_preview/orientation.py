"""Tool pose, yaw, quaternion, and offset calculations."""

import math
from typing import Iterable, List

from geometry_msgs.msg import Point, Pose

from robot_scene_pipeline.grasp_orientation import (
    downward_quaternion_for_yaw,
    normalize_quaternion_xyzw,
    object_yaw_orientation as solve_object_yaw_orientation,
    quaternion_distance_rad,
)

def quaternion_to_matrix_xyzw(quat_xyzw: Iterable[float]) -> List[List[float]]:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    return [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]


def rotate_tool_vector_to_base(quat_xyzw: Iterable[float], vector_tool_m: Iterable[float]) -> List[float]:
    matrix = quaternion_to_matrix_xyzw(quat_xyzw)
    vector = [float(value) for value in vector_tool_m]
    return [
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    ]


def tool0_goal_from_tcp(
    tcp_position_m: Iterable[float],
    quat_xyzw: Iterable[float],
    tcp_offset_tool_m: Iterable[float],
) -> List[float]:
    tcp_position = [float(value) for value in tcp_position_m]
    tcp_offset_base = rotate_tool_vector_to_base(quat_xyzw, tcp_offset_tool_m)
    return [tcp_position[i] - tcp_offset_base[i] for i in range(3)]


def add_base_offset(position_m, offset_m):
    return [float(position_m[i]) + float(offset_m[i]) for i in range(3)]


def object_yaw_orientation(step, args, current_quaternion_xyzw):
    if step.get("preserve_current_yaw"):
        current_yaw = estimate_downward_family_yaw_deg(current_quaternion_xyzw, args.quat_xyzw)
        return normalize_quaternion_xyzw(current_quaternion_xyzw), current_yaw, "preserve_current_yaw"
    if not step.get("target_yaw_valid") or step.get("target_yaw_deg") is None:
        if args.invalid_yaw_fallback == "error":
            raise RuntimeError(
                "Step {} object={} has no reliable target yaw.".format(
                    step.get("step"), step.get("object_label")
                )
            )
        if args.invalid_yaw_fallback == "current":
            return normalize_quaternion_xyzw(current_quaternion_xyzw), None, "invalid_yaw_current_fallback"
        return normalize_quaternion_xyzw(args.quat_xyzw), None, "invalid_yaw_fixed_fallback"

    try:
        grasp_axis = effective_grasp_axis_for_step(step, args.grasp_axis)
        if step.get("exact_tool_yaw_required"):
            selected_yaw = normalize_yaw_deg(
                object_yaw_base_value(step, args.grasp_axis, args.yaw_sign, args.yaw_offset_deg)
            )
            return downward_quaternion_for_yaw(args.quat_xyzw, selected_yaw), selected_yaw, "exact_object_yaw"
        quaternion, selected_yaw = solve_object_yaw_orientation(
            step.get("target_yaw_deg"),
            step.get("target_yaw_valid"),
            step.get("yaw_frame"),
            args.quat_xyzw,
            current_quaternion_xyzw,
            grasp_axis=grasp_axis,
            yaw_sign=args.yaw_sign,
            yaw_offset_deg=args.yaw_offset_deg,
        )
    except ValueError as exc:
        raise RuntimeError("Step {} object yaw error: {}".format(step.get("step"), exc))
    return quaternion, selected_yaw, "object_yaw"


def orientation_for_step(step, args, current_quaternion_xyzw):
    if args.orientation_mode == "current":
        return normalize_quaternion_xyzw(current_quaternion_xyzw), None, "current"
    if args.orientation_mode == "fixed":
        return normalize_quaternion_xyzw(args.quat_xyzw), None, "fixed"
    return object_yaw_orientation(step, args, current_quaternion_xyzw)


def normalize_yaw_deg(yaw_deg):
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def effective_grasp_axis_for_step(step, grasp_axis):
    label = str(step.get("object_label") or "").lower()
    if step.get("yaw_forced_from_invalid_aspect_ratio") and "square" in label:
        return "long"
    return grasp_axis


def object_yaw_base_value(step, grasp_axis, yaw_sign, yaw_offset_deg):
    if not step.get("target_yaw_valid") or step.get("target_yaw_deg") is None:
        raise ValueError("Object yaw is not reliable.")
    if step.get("yaw_frame") not in (None, "", "base_link"):
        raise ValueError("Object yaw frame must be base_link, got {}.".format(step.get("yaw_frame")))

    if yaw_sign == "positive":
        yaw_deg = float(step.get("target_yaw_deg"))
    elif yaw_sign == "negative":
        yaw_deg = -float(step.get("target_yaw_deg"))
    else:
        raise ValueError("Unsupported yaw sign: {}".format(yaw_sign))

    grasp_axis = effective_grasp_axis_for_step(step, grasp_axis)
    if grasp_axis == "short":
        yaw_deg += 90.0
    elif grasp_axis != "long":
        raise ValueError("Unsupported grasp axis: {}".format(grasp_axis))
    return yaw_deg + float(yaw_offset_deg)


def unique_values(values):
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def shortest_yaw_delta_deg(target_yaw_deg, current_yaw_deg):
    return normalize_yaw_deg(float(target_yaw_deg) - float(current_yaw_deg))


def gripper_yaw_error_deg(target_yaw_deg, current_yaw_deg):
    """Return the smallest yaw error for a parallel gripper's 180-degree symmetry."""
    return min(
        abs(shortest_yaw_delta_deg(float(target_yaw_deg) + equivalent, current_yaw_deg))
        for equivalent in (0.0, 180.0, -180.0)
    )


def estimate_downward_family_yaw_deg(quaternion_xyzw, base_quaternion_xyzw):
    best_yaw = 0.0
    best_distance = None
    for yaw in range(-180, 181, 2):
        candidate = downward_quaternion_for_yaw(base_quaternion_xyzw, yaw)
        distance = quaternion_distance_rad(candidate, quaternion_xyzw)
        if best_distance is None or distance < best_distance:
            best_yaw = float(yaw)
            best_distance = distance
    start = best_yaw - 2.0
    for index in range(81):
        yaw = start + index * 0.05
        candidate = downward_quaternion_for_yaw(base_quaternion_xyzw, yaw)
        distance = quaternion_distance_rad(candidate, quaternion_xyzw)
        if distance < best_distance:
            best_yaw = yaw
            best_distance = distance
    return normalize_yaw_deg(best_yaw)


def object_yaw_candidate_values(step, args):
    if step.get("exact_tool_yaw_required"):
        return [
            normalize_yaw_deg(
                object_yaw_base_value(step, args.grasp_axis, args.yaw_sign, args.yaw_offset_deg)
            )
        ]
    yaw_signs = [args.yaw_sign]
    period = float(step.get("yaw_equivalence_period_deg", 180.0))
    equivalents = [index * period for index in range(-4, 5)]
    yaws = []
    for yaw_sign in yaw_signs:
        base_yaw = object_yaw_base_value(step, args.grasp_axis, yaw_sign, args.yaw_offset_deg)
        for equivalent in equivalents:
            selected_yaw = normalize_yaw_deg(base_yaw + equivalent)
            if selected_yaw not in yaws:
                yaws.append(selected_yaw)
    return yaws


def orientation_candidates_for_pre_rotate(step, args, current_quaternion_xyzw):
    if step.get("preserve_current_yaw"):
        quaternion, selected_yaw, source = object_yaw_orientation(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": source,
            }
        ]
    if args.orientation_mode != "object-yaw":
        quaternion, selected_yaw, source = orientation_for_step(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": source,
            }
        ]

    if not step.get("target_yaw_valid") or step.get("target_yaw_deg") is None:
        quaternion, selected_yaw, source = object_yaw_orientation(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": source,
            }
        ]
    if step.get("exact_tool_yaw_required"):
        quaternion, selected_yaw, source = object_yaw_orientation(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": "exact_object_yaw selected_yaw={:.2f}".format(selected_yaw),
            }
        ]

    candidates = []
    period = float(step.get("yaw_equivalence_period_deg", 180.0))
    yaw_equivalents = [index * period for index in range(-4, 5)]
    effective_axis = effective_grasp_axis_for_step(step, args.grasp_axis)
    for yaw_sign in [args.yaw_sign]:
        base_yaw = object_yaw_base_value(step, args.grasp_axis, yaw_sign, args.yaw_offset_deg)
        for equivalent in yaw_equivalents:
            raw_yaw = base_yaw + equivalent
            quaternion = downward_quaternion_for_yaw(args.quat_xyzw, raw_yaw)
            selected_yaw = normalize_yaw_deg(raw_yaw)
            label = "object_yaw axis={} sign={} raw_yaw={:.2f} selected_yaw={:.2f}".format(
                effective_axis,
                yaw_sign,
                raw_yaw,
                selected_yaw,
            )
            if any(
                existing["selected_yaw_deg"] == selected_yaw
                and all(abs(a - b) < 1e-9 for a, b in zip(existing["quat_xyzw"], quaternion))
                for existing in candidates
            ):
                continue
            candidates.append(
                {
                    "quat_xyzw": quaternion,
                    "selected_yaw_deg": selected_yaw,
                    "source": "object_yaw_joint_delta_selected",
                    "label": label,
                }
            )
    return candidates


def validate_goal(goal, args):
    x, y, z = goal
    radius = (x * x + y * y) ** 0.5
    if z < args.min_z:
        raise RuntimeError("Rejected goal z={:.3f} below min-z={:.3f}".format(z, args.min_z))
    if z > args.max_z:
        raise RuntimeError("Rejected goal z={:.3f} above max-z={:.3f}".format(z, args.max_z))
    if radius > args.max_radius:
        raise RuntimeError("Rejected goal xy radius={:.3f} above max-radius={:.3f}".format(radius, args.max_radius))


def xyz_delta(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def make_pose(position, quat_xyzw):
    pose = Pose()
    pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def quaternion_multiply_xyzw(left, right):
    lx, ly, lz, lw = normalize_quaternion_xyzw(left)
    rx, ry, rz, rw = normalize_quaternion_xyzw(right)
    return normalize_quaternion_xyzw(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def orientation_command_quaternion(args, target_quat_xyzw):
    return quaternion_multiply_xyzw(
        args.orientation_command_correction_xyzw,
        target_quat_xyzw,
    )


def transform_position_quat(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    return [float(t.x), float(t.y), float(t.z)], [float(q.x), float(q.y), float(q.z), float(q.w)]


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
