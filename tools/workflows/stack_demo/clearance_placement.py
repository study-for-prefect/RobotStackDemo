"""Safe temporary placement helpers for stack-demo clearance."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional

from robot_scene_pipeline.geometry_relations import get_center, get_size, object_xy_aabb, xy_aabb_overlap


ObjectDict = Dict[str, Any]


def safe_place_for_object(
    obj: ObjectDict,
    target: ObjectDict,
    objects: List[ObjectDict],
    protected_ids: Iterable[Any],
    table_bounds: Optional[dict],
    future_place_regions: Iterable[dict] = (),
    margin_m: float = 0.02,
    max_distance_from_target_m: Optional[float] = 0.14,
    avoid_all_visible_objects: bool = True,
) -> Optional[List[float]]:
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return None
    target_center = get_center(target) or center
    samples = _safe_place_samples(size, target_center, objects, table_bounds, margin_m)
    if not samples:
        return None
    protected = {str(value) for value in protected_ids or []}
    avoid_scene_objects = bool(avoid_all_visible_objects or not table_bounds)
    max_distance = None
    if max_distance_from_target_m is not None:
        try:
            max_distance = float(max_distance_from_target_m)
        except (TypeError, ValueError):
            max_distance = None
    samples.sort(key=lambda xy: _place_sample_priority(xy, target_center))
    for xy in samples:
        if max_distance is not None:
            distance_m = math.hypot(float(xy[0]) - float(target_center[0]), float(xy[1]) - float(target_center[1]))
            if distance_m > max_distance:
                continue
        placed = _translated_object(obj, xy)
        placed_aabb = object_xy_aabb(placed, margin_m=0.005)
        if not placed_aabb:
            continue
        if _overlaps_future_place_region(placed, future_place_regions, margin_m=0.005):
            continue
        if _placement_overlaps_scene(
            placed_aabb,
            obj,
            target,
            objects,
            protected,
            avoid_scene_objects,
        ):
            continue
        return [round(float(xy[0]), 5), round(float(xy[1]), 5), round(float(center[2]), 5)]
    return None


def _translated_object(obj: ObjectDict, center_xy: Iterable[float]) -> ObjectDict:
    output = copy.deepcopy(obj)
    center = get_center(output)
    if center is None:
        return output
    xy = [float(value) for value in list(center_xy)[:2]]
    output["geometry_center_m"] = [xy[0], xy[1], float(center[2])]
    return output


def _placement_overlaps_scene(
    placed_aabb: dict,
    obj: ObjectDict,
    target: ObjectDict,
    objects: List[ObjectDict],
    protected: set,
    avoid_all_visible_objects: bool,
) -> bool:
    for other in objects:
        if _object_id(other) == _object_id(obj):
            continue
        if (
            not avoid_all_visible_objects
            and _object_id(other) not in protected
            and _object_id(other) != _object_id(target)
        ):
            continue
        other_aabb = object_xy_aabb(other, margin_m=0.005)
        if other_aabb and xy_aabb_overlap(placed_aabb, other_aabb)[2] > 0.0:
            return True
    return False


def _overlaps_future_place_region(
    obj: ObjectDict,
    future_place_regions: Iterable[dict],
    margin_m: float,
) -> bool:
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return False
    half_diagonal = 0.5 * math.hypot(float(size[0]), float(size[1])) + float(margin_m)
    for region in future_place_regions or []:
        region_center = region.get("center_base_m")
        if not isinstance(region_center, list) or len(region_center) < 2:
            continue
        try:
            radius_m = float(region.get("radius_m"))
            distance_m = math.hypot(
                float(center[0]) - float(region_center[0]),
                float(center[1]) - float(region_center[1]),
            )
        except (TypeError, ValueError):
            continue
        if distance_m <= radius_m + half_diagonal:
            return True
    return False


def _safe_place_samples(
    size: List[float],
    target_center: List[float],
    objects: List[ObjectDict],
    table_bounds: Optional[dict],
    margin_m: float,
) -> List[List[float]]:
    observed = _observed_scene_place_samples(target_center, objects)
    bounded = _table_bound_samples(size, target_center, table_bounds, margin_m)
    samples = []
    for sample in observed + bounded:
        if sample not in samples:
            samples.append(sample)
    return samples


def _place_sample_priority(sample_xy: List[float], target_center: List[float]) -> tuple:
    distance = math.hypot(float(sample_xy[0]) - float(target_center[0]), float(sample_xy[1]) - float(target_center[1]))
    preferred_m = 0.11
    return (abs(distance - preferred_m), distance)


def _table_bound_samples(
    size: List[float],
    target_center: List[float],
    table_bounds: Optional[dict],
    margin_m: float,
) -> List[List[float]]:
    if not table_bounds:
        return []
    try:
        xmin = float(table_bounds["xmin"]) + 0.5 * size[0] + margin_m
        xmax = float(table_bounds["xmax"]) - 0.5 * size[0] - margin_m
        ymin = float(table_bounds["ymin"]) + 0.5 * size[1] + margin_m
        ymax = float(table_bounds["ymax"]) - 0.5 * size[1] - margin_m
    except (KeyError, TypeError, ValueError):
        return []
    if xmin >= xmax or ymin >= ymax:
        return []
    return [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
        [(xmin + xmax) / 2.0, ymin],
        [(xmin + xmax) / 2.0, ymax],
        [xmin, (ymin + ymax) / 2.0],
        [xmax, (ymin + ymax) / 2.0],
        [target_center[0], ymin],
        [target_center[0], ymax],
    ]


def _observed_scene_place_samples(target_center: List[float], objects: List[ObjectDict]) -> List[List[float]]:
    centers = [get_center(obj) for obj in objects]
    xy_centers = [center for center in centers if center is not None]
    if not xy_centers:
        return []
    xmin = min(float(center[0]) for center in xy_centers) - 0.10
    xmax = max(float(center[0]) for center in xy_centers) + 0.10
    ymin = min(float(center[1]) for center in xy_centers) - 0.10
    ymax = max(float(center[1]) for center in xy_centers) + 0.10
    samples: List[List[float]] = []
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (0.7071, -0.7071),
        (-0.7071, 0.7071), (-0.7071, -0.7071),
    ]
    for radius_m in (0.10, 0.13, 0.16, 0.19):
        for dx, dy in directions:
            sample = [
                max(xmin, min(xmax, float(target_center[0]) + radius_m * dx)),
                max(ymin, min(ymax, float(target_center[1]) + radius_m * dy)),
            ]
            if sample not in samples:
                samples.append(sample)
    return samples


def _object_id(obj: ObjectDict) -> str:
    return str(obj.get("id"))
