"""Small, dependency-free SE(3) helpers for base_link/GF225 pose planning.

Quaternions are always ordered ``xyzw`` and transforms map local coordinates
into their parent frame.  The explicit naming keeps object, grasp-TCP and
tool0 frames from collapsing into one ambiguous yaw value.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
Matrix4 = tuple[tuple[float, float, float, float], ...]


def normalize_vector(value: Sequence[float]) -> Vector3:
    norm = math.sqrt(sum(float(item) ** 2 for item in value[:3]))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(item) / norm for item in value[:3])  # type: ignore[return-value]


def quaternion_normalize(value: Sequence[float]) -> Quaternion:
    if len(value) != 4:
        raise ValueError("quaternion must contain xyzw")
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero quaternion")
    return tuple(float(item) / norm for item in value)  # type: ignore[return-value]


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> Quaternion:
    x1, y1, z1, w1 = quaternion_normalize(left)
    x2, y2, z2, w2 = quaternion_normalize(right)
    return quaternion_normalize((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))


def quaternion_inverse(value: Sequence[float]) -> Quaternion:
    x, y, z, w = quaternion_normalize(value)
    return (-x, -y, -z, w)


def quaternion_from_axis_angle(axis: Sequence[float], angle_deg: float) -> Quaternion:
    unit = normalize_vector(axis)
    half = math.radians(float(angle_deg)) / 2.0
    sine = math.sin(half)
    return quaternion_normalize((*[item * sine for item in unit], math.cos(half)))


def quaternion_to_matrix(value: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = quaternion_normalize(value)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def matrix_to_quaternion(matrix: Sequence[Sequence[float]]) -> Quaternion:
    m = matrix
    trace = float(m[0][0]) + float(m[1][1]) + float(m[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return quaternion_normalize(((m[2][1] - m[1][2]) / scale,
            (m[0][2] - m[2][0]) / scale, (m[1][0] - m[0][1]) / scale, scale / 4.0))
    index = max(range(3), key=lambda item: float(m[item][item]))
    if index == 0:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        value = (scale / 4.0, (m[0][1] + m[1][0]) / scale,
                 (m[0][2] + m[2][0]) / scale, (m[2][1] - m[1][2]) / scale)
    elif index == 1:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        value = ((m[0][1] + m[1][0]) / scale, scale / 4.0,
                 (m[1][2] + m[2][1]) / scale, (m[0][2] - m[2][0]) / scale)
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        value = ((m[0][2] + m[2][0]) / scale, (m[1][2] + m[2][1]) / scale,
                 scale / 4.0, (m[1][0] - m[0][1]) / scale)
    return quaternion_normalize(value)


def pose_to_transform(pose: Mapping[str, Any]) -> Matrix4:
    position = pose.get("position_m")
    orientation = pose.get("orientation_xyzw")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("pose requires position_m[3]")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise ValueError("pose requires orientation_xyzw[4]")
    rotation = quaternion_to_matrix(orientation)
    return tuple(
        tuple(rotation[row][column] for column in range(3)) + (float(position[row]),)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def transform_to_pose(transform: Sequence[Sequence[float]], frame_id: str = "base_link") -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "position_m": [float(transform[row][3]) for row in range(3)],
        "orientation_xyzw": list(matrix_to_quaternion(transform)),
    }


def transform_multiply(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix4:
    return tuple(tuple(sum(float(left[row][k]) * float(right[k][column]) for k in range(4))
                       for column in range(4)) for row in range(4))


def transform_inverse(transform: Sequence[Sequence[float]]) -> Matrix4:
    rotation = [[float(transform[row][column]) for column in range(3)] for row in range(3)]
    position = [float(transform[row][3]) for row in range(3)]
    inverse_rotation = [[rotation[column][row] for column in range(3)] for row in range(3)]
    inverse_position = [-sum(inverse_rotation[row][k] * position[k] for k in range(3)) for row in range(3)]
    return tuple(tuple(inverse_rotation[row]) + (inverse_position[row],) for row in range(3)) + ((0.0, 0.0, 0.0, 1.0),)


def slerp(start: Sequence[float], target: Sequence[float], fraction: float) -> Quaternion:
    first, second = quaternion_normalize(start), quaternion_normalize(target)
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second, dot = tuple(-item for item in second), -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return quaternion_normalize(tuple(a + fraction * (b - a) for a, b in zip(first, second)))
    theta = math.acos(dot)
    sine = math.sin(theta)
    return quaternion_normalize(tuple(
        (math.sin((1.0 - fraction) * theta) * a + math.sin(fraction * theta) * b) / sine
        for a, b in zip(first, second)
    ))


def rotation_error_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(quaternion_normalize(first), quaternion_normalize(second))))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def transform_vector(quaternion: Sequence[float], vector: Sequence[float]) -> Vector3:
    matrix = quaternion_to_matrix(quaternion)
    return tuple(sum(matrix[row][column] * float(vector[column]) for column in range(3)) for row in range(3))  # type: ignore[return-value]


def transform_point(transform: Sequence[Sequence[float]], point: Sequence[float]) -> Vector3:
    return tuple(sum(float(transform[row][column]) * float(point[column]) for column in range(3)) + float(transform[row][3]) for row in range(3))  # type: ignore[return-value]


def axis_alignment_error_deg(first: Sequence[float], second: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(normalize_vector(first), normalize_vector(second))))
    return math.degrees(math.acos(min(1.0, max(-1.0, dot))))

