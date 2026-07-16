"""Pure target-region and direction geometry for code-generated nudge edges."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..common.scene_state import ClutterSceneState, SceneObjectState
from .grasp_edges import GraspScanResult


def object_separation(first: SceneObjectState, second: SceneObjectState) -> float:
    return point_separation(first.center_xyz_m, second.center_xyz_m)


def point_separation(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def nudge_region_effects(
    blocker: SceneObjectState,
    moved_center: Sequence[float],
    scene: ClutterSceneState,
) -> dict[str, Any]:
    destination = next((
        region for region in scene.target_regions
        if _point_inside_bounds(moved_center, region.get("bounds_base_m") or region.get("bounds"))
    ), None)
    if destination is None:
        return {
            "destination_color_region": None,
            "destination_region_id": None,
            "destination_region_occupants": [],
            "expected_effects": (),
            "task_progress_gain": 0.0,
            "risk_delta": 0.0,
        }
    bounds = destination.get("bounds_base_m") or destination.get("bounds")
    occupants = sorted(
        obj.track_id for obj in scene.current_objects
        if obj.track_id != blocker.track_id and _point_inside_bounds(obj.center_xyz_m, bounds)
    )
    destination_color = str(destination.get("color") or "")
    correct_color = destination_color == blocker.color and _footprint_inside_bounds(
        blocker, moved_center, bounds,
    )
    return {
        "destination_color_region": destination_color or None,
        "destination_region_id": destination.get("region_id"),
        "destination_region_occupants": occupants,
        "expected_effects": (
            ("enter_correct_color_region",) if correct_color
            else ("enter_other_color_region",)
        ),
        "task_progress_gain": 1.0 if correct_color else 0.0,
        "risk_delta": 0.0 if correct_color else 0.10,
    }


def direction_name(direction: Sequence[float]) -> str:
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "px" if float(direction[0]) > 0 else "nx"
    return "py" if float(direction[1]) > 0 else "ny"


def opposite_side(direction: Sequence[float]) -> str:
    """Return the block contact side opposite a code-owned push direction."""
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "-x" if float(direction[0]) > 0 else "+x"
    return "-y" if float(direction[1]) > 0 else "+y"


def blocked_intervals_for(scan: GraspScanResult, blocker_id: str) -> list[float]:
    return [
        float(sample["yaw_deg"])
        for sample in scan.samples if blocker_id in sample.get("blocking_track_ids", [])
    ]


def _point_inside_bounds(point: Sequence[float], bounds: Any) -> bool:
    return bool(
        isinstance(bounds, Mapping)
        and float(bounds["xmin"]) <= float(point[0]) <= float(bounds["xmax"])
        and float(bounds["ymin"]) <= float(point[1]) <= float(bounds["ymax"])
    )


def _footprint_inside_bounds(
    obj: SceneObjectState,
    center: Sequence[float],
    bounds: Any,
) -> bool:
    if not isinstance(bounds, Mapping):
        return False
    angle = math.radians(obj.yaw_deg)
    half_x = 0.5 * (
        abs(math.cos(angle)) * obj.size_xyz_m[0]
        + abs(math.sin(angle)) * obj.size_xyz_m[1]
    )
    half_y = 0.5 * (
        abs(math.sin(angle)) * obj.size_xyz_m[0]
        + abs(math.cos(angle)) * obj.size_xyz_m[1]
    )
    return (
        float(center[0]) - half_x >= float(bounds["xmin"])
        and float(center[0]) + half_x <= float(bounds["xmax"])
        and float(center[1]) - half_y >= float(bounds["ymin"])
        and float(center[1]) + half_y <= float(bounds["ymax"])
    )
