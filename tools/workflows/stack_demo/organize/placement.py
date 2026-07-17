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
    """Build four calibrated image quadrants as square base_link regions."""
    configured = [str(item) for item in config.section("organize")["colors"]]
    if not configured:
        return {}
    bounds = config.organize_layout
    organize = config.section("organize")
    size = float(organize["target_region_size_m"])
    quadrant_by_color = organize["quadrant_by_color"]
    x_ranges = {
        "image_bottom": (float(bounds["xmin"]), float(bounds["xmin"]) + size),
        "image_top": (float(bounds["xmax"]) - size, float(bounds["xmax"])),
    }
    y_ranges = {
        "image_right": (float(bounds["ymin"]), float(bounds["ymin"]) + size),
        "image_left": (float(bounds["ymax"]) - size, float(bounds["ymax"])),
    }
    channel_x = float(bounds["xmax"]) - float(bounds["xmin"]) - 2.0 * size
    channel_y = float(bounds["ymax"]) - float(bounds["ymin"]) - 2.0 * size
    regions = {}
    for color in configured:
        quadrant = str(quadrant_by_color[color])
        vertical, horizontal = quadrant.rsplit("_", 1)
        xmin, xmax = x_ranges[vertical]
        ymin, ymax = y_ranges[f"image_{horizontal}"]
        regions[color] = {
            "region_id": f"organize_{color}",
            "color": color,
            "frame_id": "base_link",
            "image_quadrant": quadrant,
            "target_region_size_m": size,
            "clearance_channel_width_x_m": channel_x,
            "clearance_channel_width_y_m": channel_y,
            "bounds_base_m": {
                "xmin": round(xmin, 9), "xmax": round(xmax, 9),
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
    """Generate dense object-clear anchors; edge checks own yaw-specific tool clearance."""
    organize = config.section("organize")
    spacing = float(organize["minimum_spacing_m"])
    # Spend only one third of the completion tolerance on fitting the regular
    # grid.  The remaining two thirds absorb detector/placement drift instead
    # of letting nominal slots consume the entire tolerance at a region edge.
    edge_allowance = float(organize["observation_region_tolerance_m"]) / 3.0
    slots: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for color, region in regions.items():
        members = [obj for obj in scene.current_objects if obj.color == color]
        width = max((obj.size_xyz_m[0] for obj in members), default=0.03)
        depth = max((obj.size_xyz_m[1] for obj in members), default=0.03)
        height = max((obj.size_xyz_m[2] for obj in members), default=0.04)
        yaw_independent_extent = math.hypot(width, depth)
        bounds = region["bounds_base_m"]
        x_values = _regular_axis_centers(
            float(bounds["xmin"]) - edge_allowance,
            float(bounds["xmax"]) + edge_allowance,
            yaw_independent_extent,
            spacing,
        )
        y_values = _regular_axis_centers(
            float(bounds["ymin"]) - edge_allowance,
            float(bounds["ymax"]) + edge_allowance,
            yaw_independent_extent,
            spacing,
        )
        values = []
        for row, x in enumerate(reversed(x_values), start=1):
            for column, y in enumerate(reversed(y_values), start=1):
                pose = [x, y, _table_center_z(scene, height)]
                if _slot_safe(scene, pose, (width, depth, height), config):
                    values.append({
                        "slot_id": f"{region['region_id']}_r{row}_c{column}",
                        "region_id": region["region_id"],
                        "position_m": pose,
                        "yaw_deg": 0.0,
                        "open_gripper_descent_safe": True,
                    })
        slots[color] = tuple(values)
    return slots


def rank_slots_by_clearance(
    scene: ClutterSceneState,
    slots: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    """Try the least obstructed color slot first, while retaining every slot."""
    return tuple(sorted(
        slots,
        key=lambda slot: (-_minimum_xy_clearance(scene, slot["position_m"]), str(slot["slot_id"])),
    ))


def table_contact_center_z(scene: ClutterSceneState, obj: SceneObjectState) -> float:
    """Object-center height that puts the object's bottom on the observed table."""
    bottoms = [
        item.center_xyz_m[2] - item.size_xyz_m[2] / 2.0
        for item in scene.current_objects
    ]
    table_z = min(bottoms) if bottoms else obj.center_xyz_m[2] - obj.size_xyz_m[2] / 2.0
    return table_z + obj.size_xyz_m[2] / 2.0


def select_staging_position(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    obj: SceneObjectState,
    reservations: tuple[Mapping[str, Any], ...],
) -> list[float] | None:
    """Choose an unoccupied staging point; never reuse a released placement."""
    candidates = _staging_candidates(config)
    object_radius = 0.5 * math.hypot(obj.size_xyz_m[0], obj.size_xyz_m[1])
    gripper_radius = 0.5 * float(config.section("gripper")["open_outer_width_m"])
    margin = float(config.section("safety")["object_clearance_m"])
    occupied = [
        (item.center_xyz_m[:2], 0.5 * math.hypot(item.size_xyz_m[0], item.size_xyz_m[1]))
        for item in scene.current_objects if item.track_id != obj.track_id
    ]
    occupied.extend(_reservation_footprints(reservations))
    for x, y in candidates:
        if all(
            math.dist((x, y), center) >= max(object_radius, gripper_radius) + radius + margin
            for center, radius in occupied
        ):
            return [x, y, table_contact_center_z(scene, obj)]
    return None


def _slot_safe(
    scene: ClutterSceneState,
    position: list[float],
    size: tuple[float, float, float],
    config: StackDemoConfig,
) -> bool:
    clearance = float(config.section("safety")["object_clearance_m"])
    for obj in scene.current_objects:
        dx = abs(position[0] - obj.center_xyz_m[0])
        dy = abs(position[1] - obj.center_xyz_m[1])
        object_overlap = dx < 0.5 * (size[0] + obj.size_xyz_m[0]) + clearance and dy < 0.5 * (size[1] + obj.size_xyz_m[1]) + clearance
        # Tool clearance depends on the release yaw.  Reject physical object
        # overlap here and let placement_path_checks evaluate the actual GF225
        # finger/palm envelope for every release-yaw candidate.
        if object_overlap:
            return False
    return True


def _minimum_xy_clearance(scene: ClutterSceneState, position: list[float]) -> float:
    distances = [
        math.dist(position[:2], item.center_xyz_m[:2])
        for item in scene.current_objects
    ]
    return min(distances, default=float("inf"))


def _regular_axis_centers(
    low: float,
    high: float,
    object_extent: float,
    minimum_spacing: float,
) -> tuple[float, ...]:
    """Return an edge-balanced regular grid satisfying the requested clear gap."""
    span = high - low
    count = max(1, int(math.floor((span + minimum_spacing) / (object_extent + minimum_spacing))))
    if count == 1:
        return (0.5 * (low + high),)
    first = low + object_extent / 2.0
    last = high - object_extent / 2.0
    pitch = (last - first) / (count - 1)
    return tuple(first + index * pitch for index in range(count))


def _staging_candidates(config: StackDemoConfig) -> tuple[tuple[float, float], ...]:
    workspace, clutter = config.workspace, config.default_initial_clutter
    x_values = (
        float(workspace["xmax"]) - 0.06,
        float(workspace["xmax"]) - 0.13,
        max(float(clutter["xmax"]) + 0.07, float(workspace["xmin"]) + 0.07),
    )
    y_values = (
        float(workspace["ymin"]) + 0.06,
        float(workspace["ymin"]) + 0.15,
        float(workspace["ymin"]) + 0.24,
    )
    return tuple((round(x, 6), round(y, 6)) for x in x_values for y in y_values)


def _reservation_footprints(
    reservations: tuple[Mapping[str, Any], ...],
) -> list[tuple[tuple[float, float], float]]:
    output = []
    for item in reservations:
        position = item.get("position_m")
        size = item.get("size_m")
        if not (
            isinstance(position, (list, tuple)) and len(position) >= 2
            and isinstance(size, (list, tuple)) and len(size) >= 2
        ):
            continue
        output.append((
            (float(position[0]), float(position[1])),
            0.5 * math.hypot(float(size[0]), float(size[1])),
        ))
    return output


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
