"""Latest-observation geometric predicates for the six house roles."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .orientation import roof_orientation_from_object, triangle_orientation_from_object


def role_observation_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    role: str,
    config: StackDemoConfig,
    *,
    role_object: SceneObjectState | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Evaluate one bound role solely from current base_link geometry."""
    obj = role_object or scene.object_by_track(bindings.get(role, ""))
    checks: dict[str, Any] = {"role_object_visible": obj is not None}
    if obj is None:
        return False, checks
    if role.endswith("_lower"):
        checks.update(_lower_checks(obj, role, config))
    elif role.endswith("_upper"):
        checks.update(_upper_checks(scene, bindings, obj, role, config))
    elif role == "roof":
        checks.update(_roof_checks(scene, bindings, obj, config))
    elif role == "triangle_top":
        checks.update(_triangle_checks(scene, bindings, obj, config))
    else:
        checks["known_role"] = False
    booleans = [value for value in checks.values() if isinstance(value, bool)]
    return bool(booleans and all(booleans)), checks


def object_at_pose(obj: SceneObjectState, pose: Mapping[str, Any]) -> SceneObjectState:
    """Build a non-persistent planned object used only for prechecking a role."""
    position = pose.get("position_m")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError("house place pose needs three base_link coordinates")
    return replace(
        obj,
        center_xyz_m=tuple(float(value) for value in position[:3]),
        yaw_deg=float(pose.get("yaw_deg", obj.yaw_deg)),
    )


def _lower_checks(obj: SceneObjectState, role: str, config: StackDemoConfig) -> dict[str, Any]:
    house = config.section("house")
    origin_x, origin_y, origin_z = (float(value) for value in house["origin_center_base_m"])
    center_spacing = float(obj.size_xyz_m[0]) + float(house["support_inner_gap_m"])
    expected_x = origin_x + (-0.5 if role.startswith("left_") else 0.5) * center_spacing
    offset = math.hypot(obj.center_xyz_m[0] - expected_x, obj.center_xyz_m[1] - origin_y)
    return {
        "center_offset_valid": offset <= float(house["center_tolerance_m"]),
        "layer_height_valid": abs(obj.center_xyz_m[2] - origin_z) <= float(house["support_height_tolerance_m"]),
        "center_offset_m": offset,
    }


def _upper_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
) -> dict[str, Any]:
    lower_role = role.replace("upper", "lower")
    lower = scene.object_by_track(bindings.get(lower_role, ""))
    if lower is None:
        return {"lower_support_visible": False}
    tolerance = float(config.section("house")["support_height_tolerance_m"])
    center_tolerance = float(config.section("house")["center_tolerance_m"])
    offset = math.hypot(obj.center_xyz_m[0] - lower.center_xyz_m[0], obj.center_xyz_m[1] - lower.center_xyz_m[1])
    vertical_gap = _bottom(obj) - _top(lower)
    return {
        "lower_support_visible": True,
        "column_center_offset_valid": offset <= center_tolerance,
        "vertical_contact_valid": abs(vertical_gap) <= tolerance,
        "column_center_offset_m": offset,
        "vertical_contact_gap_m": vertical_gap,
    }


def _roof_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    roof: SceneObjectState,
    config: StackDemoConfig,
) -> dict[str, Any]:
    house = config.section("house")
    left = scene.object_by_track(bindings.get("left_support_upper", ""))
    right = scene.object_by_track(bindings.get("right_support_upper", ""))
    if left is None or right is None:
        return {"both_upper_supports_visible": False}
    span = (right.center_xyz_m[0] - left.center_xyz_m[0], right.center_xyz_m[1] - left.center_xyz_m[1])
    spacing = math.hypot(*span)
    if spacing <= 1e-9:
        return {"both_upper_supports_visible": True, "support_spacing_valid": False}
    span_yaw = math.degrees(math.atan2(span[1], span[0]))
    left_along_half, _ = _projected_half_extents(left, span_yaw)
    right_along_half, _ = _projected_half_extents(right, span_yaw)
    support_inner_gap = spacing - left_along_half - right_along_half
    midpoint = (
        0.5 * (left.center_xyz_m[0] + right.center_xyz_m[0]),
        0.5 * (left.center_xyz_m[1] + right.center_xyz_m[1]),
    )
    center_offset = math.hypot(roof.center_xyz_m[0] - midpoint[0], roof.center_xyz_m[1] - midpoint[1])
    orientation = roof_orientation_from_object(roof, config)
    yaw_error = _axis_error_deg(orientation.long_axis_yaw_deg, span_yaw)
    along_half, across_half = _projected_half_extents(roof, span_yaw)
    minimum_margin = float(house["minimum_support_margin_m"])
    left_margin = along_half - 0.5 * spacing
    right_margin = along_half - 0.5 * spacing
    across_required = 0.5 * max(left.size_xyz_m[1], right.size_xyz_m[1])
    height_difference = abs(_top(left) - _top(right))
    vertical_gap = _bottom(roof) - max(_top(left), _top(right))
    return {
        "both_upper_supports_visible": True,
        "groove_face_correct": (
            orientation.groove_face_state == house["required_roof_groove_face_state"]
            and orientation.face_up
            and orientation.satisfies_roof_orientation
        ),
        "long_axis_matches_support_span": yaw_error <= float(house["orientation_tolerance_deg"]),
        "support_spacing_valid": abs(
            support_inner_gap - float(house["support_inner_gap_m"])
        ) <= 2.0 * float(house["center_tolerance_m"]),
        "support_height_difference_valid": height_difference <= float(house["support_height_tolerance_m"]),
        "roof_center_offset_valid": center_offset <= float(house["center_tolerance_m"]),
        "covers_left_support": left_margin >= minimum_margin,
        "covers_right_support": right_margin >= minimum_margin,
        "roof_width_covers_supports": across_half >= across_required,
        "vertical_contact_valid": abs(vertical_gap) <= float(house["support_height_tolerance_m"]),
        "support_spacing_m": spacing,
        "support_inner_gap_m": support_inner_gap,
        "support_height_difference_m": height_difference,
        "roof_center_offset_m": center_offset,
        "left_support_margin_m": left_margin,
        "right_support_margin_m": right_margin,
        "roof_vertical_gap_m": vertical_gap,
    }


def _triangle_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    triangle: SceneObjectState,
    config: StackDemoConfig,
) -> dict[str, Any]:
    house = config.section("house")
    roof = scene.object_by_track(bindings.get("roof", ""))
    if roof is None:
        return {"roof_visible": False}
    orientation = triangle_orientation_from_object(triangle)
    center_offset = math.hypot(
        triangle.center_xyz_m[0] - roof.center_xyz_m[0],
        triangle.center_xyz_m[1] - roof.center_xyz_m[1],
    )
    roof_support_half = 0.5 * min(roof.size_xyz_m[:2])
    support_margin = roof_support_half - center_offset
    vertical_gap = _bottom(triangle) - _top(roof)
    return {
        "roof_visible": True,
        "apex_up": orientation.apex_direction == house["required_triangle_apex_direction"],
        "base_edge_down": orientation.base_edge_direction in {"down", "roof_aligned"},
        "face_state_valid": orientation.face_state in {"front", "correct", "upright"},
        "base_contact_valid": abs(vertical_gap) <= float(house["support_height_tolerance_m"]),
        "center_of_mass_supported": abs(orientation.center_of_mass_projection_m) <= support_margin,
        "support_margin_valid": support_margin >= float(house["minimum_support_margin_m"]),
        "roof_center_offset_valid": center_offset <= roof_support_half,
        "triangle_center_offset_m": center_offset,
        "support_margin_m": support_margin,
        "base_contact_gap_m": vertical_gap,
    }


def _projected_half_extents(obj: SceneObjectState, axis_yaw_deg: float) -> tuple[float, float]:
    delta = math.radians(obj.yaw_deg - axis_yaw_deg)
    half_x, half_y = 0.5 * obj.size_xyz_m[0], 0.5 * obj.size_xyz_m[1]
    return (
        abs(half_x * math.cos(delta)) + abs(half_y * math.sin(delta)),
        abs(half_x * math.sin(delta)) + abs(half_y * math.cos(delta)),
    )


def _axis_error_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _top(obj: SceneObjectState) -> float:
    return obj.center_xyz_m[2] + 0.5 * obj.size_xyz_m[2]


def _bottom(obj: SceneObjectState) -> float:
    return obj.center_xyz_m[2] - 0.5 * obj.size_xyz_m[2]
