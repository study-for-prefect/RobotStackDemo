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
    obj = role_object or _scene_object(scene, bindings.get(role, ""))
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
        source={**dict(obj.source), "planned_house_pose": True},
    )


def _lower_checks(obj: SceneObjectState, role: str, config: StackDemoConfig) -> dict[str, Any]:
    house = config.section("house")
    origin_x, origin_y, origin_z = (float(value) for value in house["origin_center_base_m"])
    center_spacing = float(obj.size_xyz_m[0]) + float(house["support_inner_gap_m"])
    expected_x = origin_x + (-0.5 if role.startswith("left_") else 0.5) * center_spacing
    offset = math.hypot(obj.center_xyz_m[0] - expected_x, obj.center_xyz_m[1] - origin_y)
    vertical_gap, height_mode = _lower_vertical_gap(obj, config)
    return {
        "center_offset_valid": offset <= float(house["center_tolerance_m"]),
        "layer_height_valid": abs(vertical_gap) <= float(house["support_height_tolerance_m"]),
        "center_offset_m": offset,
        "vertical_contact_gap_m": vertical_gap,
        "layer_height_verification_mode": height_mode,
    }


def _lower_vertical_gap(
    obj: SceneObjectState,
    config: StackDemoConfig,
) -> tuple[float, str]:
    """Compare observations to the local table, not an absolute calibrated z.

    The commanded release pose intentionally includes a real-machine downward
    compensation.  Once released, perception reports the settled object center
    relative to the locally observed tabletop, whose base_link z is not fixed.
    """
    house = config.section("house")
    if bool(obj.source.get("planned_house_pose")):
        expected = (
            float(house["origin_center_base_m"][2])
            + float(house.get("final_place_z_offset_m", 0.0))
        )
        return obj.center_xyz_m[2] - expected, "planned_release_pose"

    local_support = obj.source.get("local_support_surface")
    support_z = None
    if isinstance(local_support, Mapping):
        value = local_support.get("support_z_base_m")
        if value is not None:
            support_z = float(value)
    if support_z is None:
        center_on_table = obj.source.get("center_on_table_m")
        if isinstance(center_on_table, (list, tuple)) and len(center_on_table) >= 3:
            support_z = float(center_on_table[2])
    if support_z is not None:
        return _bottom(obj) - support_z, "observed_local_support_contact"

    # Offline fixtures and legacy observations may not carry local support
    # evidence.  Preserve the nominal absolute-height fallback for those only.
    return (
        obj.center_xyz_m[2] - float(house["origin_center_base_m"][2]),
        "nominal_center_height_fallback",
    )


def _upper_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
) -> dict[str, Any]:
    lower_role = role.replace("upper", "lower")
    lower = _scene_object(scene, bindings.get(lower_role, ""))
    if lower is None:
        return {"lower_support_visible": False}
    tolerance = float(config.section("house")["support_height_tolerance_m"])
    center_tolerance = float(config.section("house")["center_tolerance_m"])
    offset = math.hypot(obj.center_xyz_m[0] - lower.center_xyz_m[0], obj.center_xyz_m[1] - lower.center_xyz_m[1])
    vertical_gap = _bottom(obj) - _top(lower)
    merged_column = _merged_two_support_column(obj, lower, offset, config)
    return {
        "lower_support_visible": True,
        "column_center_offset_valid": offset <= center_tolerance,
        # Two same-colour cubes in direct contact are frequently returned by
        # instance segmentation as one roughly 2H-tall column.  In that case
        # comparing the merged blob's bottom against the remembered lower
        # cube's top creates a false positive gap.  Accept only the tightly
        # constrained two-cube column geometry; ordinary rectangles and loose
        # objects cannot satisfy this predicate.
        "vertical_contact_valid": abs(vertical_gap) <= tolerance or merged_column,
        "vertical_contact_verification_mode": (
            "merged_two_support_column" if merged_column else "separate_support_surfaces"
        ),
        "column_center_offset_m": offset,
        "vertical_contact_gap_m": vertical_gap,
    }


def _merged_two_support_column(
    obj: SceneObjectState,
    lower: SceneObjectState,
    center_offset_m: float,
    config: StackDemoConfig,
) -> bool:
    """Recognize a detector blob formed by two vertically touching cubes."""
    house = config.section("house")
    x_size, y_size, height = (float(value) for value in obj.size_xyz_m)
    lower_height = float(lower.size_xyz_m[2])
    footprint_min = min(x_size, y_size)
    footprint_max = max(x_size, y_size)
    expected_height = 2.0 * lower_height
    return bool(
        obj.shape in set(house["support_classes"])
        and footprint_min >= 0.015
        and footprint_max / footprint_min <= 1.35
        and 1.65 * footprint_min <= height <= 2.35 * footprint_max
        and abs(height - expected_height) <= max(
            0.008, 2.0 * float(house["support_height_tolerance_m"])
        )
        and center_offset_m <= float(house["center_tolerance_m"])
    )


def _roof_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    roof: SceneObjectState,
    config: StackDemoConfig,
) -> dict[str, Any]:
    house = config.section("house")
    left = _scene_object(scene, bindings.get("left_support_upper", ""))
    right = _scene_object(scene, bindings.get("right_support_upper", ""))
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
    support_top = max(_top(left), _top(right))
    vertical_gap = _bottom(roof) - support_top
    roof_top = _observed_top(roof)
    top_surface_gap = (
        roof_top - support_top - float(house["roof_nominal_thickness_m"])
    )
    direct_contact = abs(vertical_gap) <= float(house["support_height_tolerance_m"])
    top_surface_contact = bool(
        not roof.source.get("planned_house_pose")
        and roof.source.get("top_z_base_m") is not None
        and abs(top_surface_gap) <= float(house["support_height_tolerance_m"])
    )
    return {
        "both_upper_supports_visible": True,
        "roof_semantic_face_correct": (
            orientation.face_up and orientation.satisfies_roof_orientation
            and (
                roof.shape == "rectangle"
                or orientation.groove_face_state == house["required_roof_groove_face_state"]
            )
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
        # On an assembled house the roof mask can include the visible support
        # sides.  Perception then reports size_z as table-to-roof-top total
        # height, so _bottom(roof) is the table rather than the roof underside.
        # The independently clustered top surface remains valid; subtract the
        # configured physical roof thickness before comparing with support tops.
        "vertical_contact_valid": direct_contact or top_surface_contact,
        "vertical_contact_verification_mode": (
            "direct_object_bottom" if direct_contact else
            "top_surface_minus_nominal_roof_thickness" if top_surface_contact else
            "contact_not_verified"
        ),
        "support_spacing_m": spacing,
        "support_inner_gap_m": support_inner_gap,
        "support_height_difference_m": height_difference,
        "roof_center_offset_m": center_offset,
        "left_support_margin_m": left_margin,
        "right_support_margin_m": right_margin,
        "roof_vertical_gap_m": vertical_gap,
        "roof_top_surface_contact_gap_m": top_surface_gap,
    }


def _triangle_checks(
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    triangle: SceneObjectState,
    config: StackDemoConfig,
) -> dict[str, Any]:
    house = config.section("house")
    roof = _scene_object(scene, bindings.get("roof", ""))
    if roof is None:
        return {"roof_visible": False}
    orientation = triangle_orientation_from_object(triangle)
    # The triangle occludes a large part of the roof after placement and can
    # shift its visible point-cloud centroid by centimetres.  The verified roof
    # was placed at the code-owned structure center, which remains the stable
    # reference for the top role.
    origin_x, origin_y = (float(value) for value in house["origin_center_base_m"][:2])
    center_offset = math.hypot(
        triangle.center_xyz_m[0] - origin_x,
        triangle.center_xyz_m[1] - origin_y,
    )
    roof_support_half = 0.5 * min(roof.size_xyz_m[:2])
    support_margin = roof_support_half - center_offset
    vertical_gap = _bottom(triangle) - _top(roof)
    triangle_axis = triangle.source.get("long_axis_base")
    roof_axis = roof.source.get("long_axis_base")
    use_metric_axis = bool(
        float(triangle.orientation_confidence) >= 0.70
        and float(roof.orientation_confidence) >= 0.70
    )
    long_axis_error = (
        _vector_axis_error_deg(triangle_axis, roof_axis)
        if use_metric_axis and _vector3(triangle_axis) and _vector3(roof_axis)
        else _axis_error_deg(
            orientation.target_yaw_deg,
            roof_orientation_from_object(roof, config).long_axis_yaw_deg,
        )
    )
    return {
        "roof_visible": True,
        "apex_up": orientation.apex_direction == house["required_triangle_apex_direction"],
        "base_edge_down": orientation.base_edge_direction in {"down", "roof_aligned"},
        "face_state_valid": orientation.face_state in {"front", "correct", "upright"},
        "long_edge_matches_roof_axis": long_axis_error <= float(house["orientation_tolerance_deg"]),
        "base_contact_valid": abs(vertical_gap) <= float(house["support_height_tolerance_m"]),
        "center_of_mass_supported": center_offset <= roof_support_half,
        "support_margin_valid": support_margin >= float(house["minimum_support_margin_m"]),
        "roof_center_offset_valid": center_offset <= roof_support_half,
        "triangle_center_offset_m": center_offset,
        "support_margin_m": support_margin,
        "base_contact_gap_m": vertical_gap,
        "long_edge_alignment_error_deg": long_axis_error,
    }


def _scene_object(
    scene: ClutterSceneState,
    track_id: str,
) -> SceneObjectState | None:
    return scene.object_by_track(track_id) or next(
        (obj for obj in scene.collision_obstacles if obj.track_id == track_id),
        None,
    )


def _projected_half_extents(obj: SceneObjectState, axis_yaw_deg: float) -> tuple[float, float]:
    delta = math.radians(obj.yaw_deg - axis_yaw_deg)
    half_x, half_y = 0.5 * obj.size_xyz_m[0], 0.5 * obj.size_xyz_m[1]
    return (
        abs(half_x * math.cos(delta)) + abs(half_y * math.sin(delta)),
        abs(half_x * math.sin(delta)) + abs(half_y * math.cos(delta)),
    )


def _axis_error_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def _vector3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3


def _vector_axis_error_deg(first: Any, second: Any) -> float:
    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        return float("inf")
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second)) / (first_norm * second_norm))
    return math.degrees(math.acos(min(1.0, max(-1.0, dot))))


def _top(obj: SceneObjectState) -> float:
    return obj.center_xyz_m[2] + 0.5 * obj.size_xyz_m[2]


def _observed_top(obj: SceneObjectState) -> float:
    value = obj.source.get("top_z_base_m")
    return float(value) if value is not None else _top(obj)


def _bottom(obj: SceneObjectState) -> float:
    return obj.center_xyz_m[2] - 0.5 * obj.size_xyz_m[2]
