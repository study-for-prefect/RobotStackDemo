"""Code-owned organization regions, occupancy, and collision-aware slots."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState, SceneObjectState


def build_color_target_regions(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    track_colors: Mapping[str, str],
) -> dict[str, Mapping[str, Any]]:
    """Build four fixed-width rows separated by code-verified clearing channels."""
    configured = [str(item) for item in config.section("organize")["colors"]]
    if not configured:
        return {}
    bounds = config.organize_layout
    region_width = float(config.section("organize")["target_region_width_y_m"])
    channel_width = (
        float(bounds["ymax"]) - float(bounds["ymin"]) - len(configured) * region_width
    ) / (len(configured) - 1)
    regions = {}
    for index, color in enumerate(configured):
        ymin = float(bounds["ymin"]) + index * (region_width + channel_width)
        ymax = ymin + region_width
        regions[color] = {
            "region_id": f"organize_{color}",
            "color": color,
            "frame_id": "base_link",
            "target_region_width_y_m": region_width,
            "clearance_channel_width_y_m": channel_width,
            "bounds_base_m": {
                "xmin": float(bounds["xmin"]), "xmax": float(bounds["xmax"]),
                "ymin": round(ymin, 9), "ymax": round(ymax, 9),
            },
        }
    return regions


def region_occupancy(
    scene: ClutterSceneState,
    regions: Mapping[str, Mapping[str, Any]],
    tolerance_m: float,
) -> dict[str, tuple[str, ...]]:
    output = {}
    for color, region in regions.items():
        bounds = region["bounds_base_m"]
        output[color] = tuple(sorted(
            obj.track_id for obj in scene.current_objects
            if (
                obj.color == color
                and _center_inside(obj.center_xyz_m, bounds)
                and _footprint_inside(obj, bounds, tolerance_m=tolerance_m)
            )
        ))
    return output


def build_safe_slots(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    regions: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Generate only slots clearing both the object and open GF225 descent."""
    spacing = float(config.section("organize")["minimum_spacing_m"])
    slots: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for color, region in regions.items():
        members = [obj for obj in scene.current_objects if obj.color == color]
        width = max((obj.size_xyz_m[0] for obj in members), default=0.03)
        depth = max((obj.size_xyz_m[1] for obj in members), default=0.03)
        height = max((obj.size_xyz_m[2] for obj in members), default=0.04)
        bounds = region["bounds_base_m"]
        y = 0.5 * (float(bounds["ymin"]) + float(bounds["ymax"]))
        step = width + spacing
        count = max(0, int(math.floor((float(bounds["xmax"]) - float(bounds["xmin"]) - width) / step)) + 1)
        values = []
        for index in range(count):
            x = float(bounds["xmin"]) + width / 2.0 + index * step
            pose = [x, y, _table_center_z(scene, height)]
            if _slot_safe(scene, pose, (width, depth, height), config):
                values.append({
                    "slot_id": f"{region['region_id']}_slot_{index + 1}",
                    "region_id": region["region_id"],
                    "position_m": pose,
                    "yaw_deg": 0.0,
                    "open_gripper_descent_safe": True,
                })
        slots[color] = tuple(values)
    return slots


def _slot_safe(
    scene: ClutterSceneState,
    position: list[float],
    size: tuple[float, float, float],
    config: StackDemoConfig,
) -> bool:
    clearance = float(config.section("safety")["object_clearance_m"])
    outer = float(config.section("gripper")["open_outer_width_m"])
    for obj in scene.current_objects:
        dx = abs(position[0] - obj.center_xyz_m[0])
        dy = abs(position[1] - obj.center_xyz_m[1])
        object_overlap = dx < 0.5 * (size[0] + obj.size_xyz_m[0]) + clearance and dy < 0.5 * (size[1] + obj.size_xyz_m[1]) + clearance
        open_gripper_overlap = dx < 0.5 * (size[0] + obj.size_xyz_m[0]) + clearance and dy < 0.5 * (outer + obj.size_xyz_m[1]) + clearance
        if object_overlap or open_gripper_overlap:
            return False
    return True


def _footprint_inside(obj: SceneObjectState, bounds: Mapping[str, float], tolerance_m: float) -> bool:
    half_extent_x, half_extent_y = _rotated_half_extents(obj)
    x, y = obj.center_xyz_m[:2]
    return (
        float(bounds["xmin"]) - tolerance_m <= x - half_extent_x
        and x + half_extent_x <= float(bounds["xmax"]) + tolerance_m
        and float(bounds["ymin"]) - tolerance_m <= y - half_extent_y
        and y + half_extent_y <= float(bounds["ymax"]) + tolerance_m
    )


def _center_inside(center_xyz_m: tuple[float, float, float], bounds: Mapping[str, float]) -> bool:
    return (
        float(bounds["xmin"]) <= center_xyz_m[0] <= float(bounds["xmax"])
        and float(bounds["ymin"]) <= center_xyz_m[1] <= float(bounds["ymax"])
    )


def _rotated_half_extents(obj: SceneObjectState) -> tuple[float, float]:
    angle = math.radians(obj.yaw_deg)
    half_x, half_y = 0.5 * obj.size_xyz_m[0], 0.5 * obj.size_xyz_m[1]
    return (
        abs(half_x * math.cos(angle)) + abs(half_y * math.sin(angle)),
        abs(half_x * math.sin(angle)) + abs(half_y * math.cos(angle)),
    )


def _table_center_z(scene: ClutterSceneState, height: float) -> float:
    bottoms = [obj.center_xyz_m[2] - obj.size_xyz_m[2] / 2.0 for obj in scene.current_objects]
    return (min(bottoms) if bottoms else 0.0) + height / 2.0
