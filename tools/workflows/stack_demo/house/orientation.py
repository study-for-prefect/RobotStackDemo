"""Explicit concave-roof and triangle-top pose state and predicates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from ..common.config import StackDemoConfig
from ..common.scene_state import SceneObjectState


@dataclass(frozen=True)
class RoofOrientationState:
    track_id: str
    groove_face_state: str
    face_up: bool
    long_axis_yaw_deg: float
    current_grasp_pose: Mapping[str, Any] | None
    satisfies_roof_orientation: bool
    left_support_coverage_m: float
    right_support_coverage_m: float
    center_offset_m: float
    left_support_margin_m: float
    right_support_margin_m: float
    placement_stable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriangleOrientationState:
    track_id: str
    apex_direction: str
    base_edge_direction: str
    face_state: str
    target_yaw_deg: float
    base_contact: bool
    center_of_mass_projection_m: float
    support_margin_m: float
    roof_relative_pose: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def roof_orientation_from_object(
    obj: SceneObjectState,
    config: StackDemoConfig,
) -> RoofOrientationState:
    source = obj.source
    groove_normal = source.get("groove_opening_normal_base")
    broad_normal = source.get("broad_face_normal_base")
    groove = str(source.get("groove_face_state") or source.get("face_state") or "unknown")
    if isinstance(groove_normal, (list, tuple)) and len(groove_normal) == 3:
        groove = "opening_down" if float(groove_normal[2]) < 0.0 else "opening_up"
    required = str(config.section("house")["required_roof_groove_face_state"])
    long_axis_vector = source.get("long_axis_base")
    long_axis = (
        math.degrees(math.atan2(float(long_axis_vector[1]), float(long_axis_vector[0])))
        if isinstance(long_axis_vector, (list, tuple)) and len(long_axis_vector) == 3
        else float(source.get("long_axis_yaw_deg", obj.yaw_deg))
    )
    semantic_face_correct = bool(
        groove == required
        or isinstance(broad_normal, (list, tuple)) and len(broad_normal) == 3
        and float(broad_normal[2]) > 0.0
    )
    face_up = bool(source.get("face_up", semantic_face_correct))
    return RoofOrientationState(
        track_id=obj.track_id,
        groove_face_state=groove,
        face_up=face_up,
        long_axis_yaw_deg=long_axis,
        current_grasp_pose=source.get("current_grasp_pose"),
        satisfies_roof_orientation=bool(source.get("satisfies_roof_orientation", semantic_face_correct and face_up)),
        left_support_coverage_m=float(source.get("left_support_coverage_m", 0.0)),
        right_support_coverage_m=float(source.get("right_support_coverage_m", 0.0)),
        center_offset_m=float(source.get("roof_center_offset_m", 0.0)),
        left_support_margin_m=float(source.get("left_support_margin_m", 0.0)),
        right_support_margin_m=float(source.get("right_support_margin_m", 0.0)),
        placement_stable=bool(source.get("placement_stable", False)),
    )


def triangle_orientation_from_object(obj: SceneObjectState) -> TriangleOrientationState:
    source = obj.source
    designated = source.get("designated_right_angle_edge_base")
    long_axis = source.get("long_axis_base")
    designated_up = bool(
        isinstance(designated, (list, tuple)) and len(designated) == 3
        and float(designated[2]) > 0.0
    )
    return TriangleOrientationState(
        track_id=obj.track_id,
        apex_direction=str(source.get("apex_direction") or ("up" if designated_up else "unknown")),
        base_edge_direction=str(source.get("base_edge_direction") or ("roof_aligned" if designated_up else "unknown")),
        face_state=str(source.get("face_state") or ("upright" if designated_up else "unknown")),
        target_yaw_deg=(
            math.degrees(math.atan2(float(long_axis[1]), float(long_axis[0])))
            if isinstance(long_axis, (list, tuple)) and len(long_axis) == 3
            else float(source.get("target_yaw_deg", obj.yaw_deg))
        ),
        base_contact=bool(source.get("base_contact", False)),
        center_of_mass_projection_m=float(source.get("center_of_mass_projection_m", float("inf"))),
        support_margin_m=float(source.get("support_margin_m", 0.0)),
        roof_relative_pose=dict(source.get("roof_relative_pose") or {}),
    )


def roof_direct_place_valid(
    state: RoofOrientationState,
    target_long_axis_yaw_deg: float,
    config: StackDemoConfig,
) -> tuple[bool, dict[str, bool]]:
    house = config.section("house")
    yaw_error = abs((state.long_axis_yaw_deg - target_long_axis_yaw_deg + 90.0) % 180.0 - 90.0)
    minimum_margin = float(house["minimum_support_margin_m"])
    checks = {
        "groove_face_correct": state.groove_face_state == house["required_roof_groove_face_state"] and state.face_up,
        "long_axis_matches_support_span": yaw_error <= float(house["orientation_tolerance_deg"]),
        "covers_left_support": state.left_support_coverage_m >= minimum_margin,
        "covers_right_support": state.right_support_coverage_m >= minimum_margin,
        "center_offset_valid": state.center_offset_m <= float(house["center_tolerance_m"]),
        "left_support_margin_valid": state.left_support_margin_m >= minimum_margin,
        "right_support_margin_valid": state.right_support_margin_m >= minimum_margin,
        "placement_stable": state.placement_stable,
    }
    return all(checks.values()), checks


def triangle_direct_place_valid(
    state: TriangleOrientationState,
    config: StackDemoConfig,
) -> tuple[bool, dict[str, bool]]:
    house = config.section("house")
    checks = {
        "apex_up": state.apex_direction == house["required_triangle_apex_direction"],
        "base_edge_down": state.base_edge_direction in {"down", "roof_aligned"},
        "face_state_valid": state.face_state in {"front", "correct", "upright"},
        "base_contact_valid": state.base_contact,
        "center_of_mass_supported": abs(state.center_of_mass_projection_m) <= state.support_margin_m,
        "support_margin_valid": state.support_margin_m >= float(house["minimum_support_margin_m"]),
        "roof_relative_pose_available": bool(state.roof_relative_pose),
    }
    return all(checks.values()), checks
