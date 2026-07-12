"""Reusable tabletop footprint geometry for semantic task validation."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from .geometry_relations import get_center, get_size


Point2D = Tuple[float, float]
Polygon2D = List[Point2D]


def object_footprint_polygon(obj: dict) -> Optional[Polygon2D]:
    """Return the object's yaw-oriented rectangular footprint in base_link."""
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return None
    yaw = object_yaw_rad(obj)
    return rectangle_footprint(center[:2], size[:2], yaw)


def rectangle_footprint(center_xy: Sequence[float], size_xy: Sequence[float], yaw_rad: float) -> Polygon2D:
    """Build a counter-clockwise rectangle polygon."""
    half_x = 0.5 * float(size_xy[0])
    half_y = 0.5 * float(size_xy[1])
    cosine = math.cos(float(yaw_rad))
    sine = math.sin(float(yaw_rad))
    output = []
    for local_x, local_y in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)):
        output.append((
            float(center_xy[0]) + local_x * cosine - local_y * sine,
            float(center_xy[1]) + local_x * sine + local_y * cosine,
        ))
    return output


def footprint_inside_region(footprint: Optional[Polygon2D], region: Optional[dict], margin_m: float = 0.0) -> bool:
    """Require every footprint vertex to remain within the rectangular region."""
    if not footprint or not region:
        return False
    try:
        return all(
            float(region["xmin"]) + margin_m <= x <= float(region["xmax"]) - margin_m
            and float(region["ymin"]) + margin_m <= y <= float(region["ymax"]) - margin_m
            for x, y in footprint
        )
    except (KeyError, TypeError, ValueError):
        return False


def footprint_overlap(first: Optional[Polygon2D], second: Optional[Polygon2D]) -> bool:
    """Test convex polygon overlap with the separating-axis theorem."""
    if not first or not second:
        return False
    for axis in _polygon_axes(first) + _polygon_axes(second):
        first_range = _projection_range(first, axis)
        second_range = _projection_range(second, axis)
        if min(first_range[1], second_range[1]) <= max(first_range[0], second_range[0]):
            return False
    return True


def footprint_overlap_area(first: Optional[Polygon2D], second: Optional[Polygon2D]) -> float:
    """Return exact convex intersection area using polygon clipping."""
    if not first or not second:
        return 0.0
    clipped = list(first)
    for start, end in _edges(second):
        clipped = _clip_polygon(clipped, start, end)
        if not clipped:
            return 0.0
    return abs(_signed_area(clipped))


def footprint_boundary_distance(first: Optional[Polygon2D], second: Optional[Polygon2D]) -> float:
    """Return zero for overlap, otherwise the minimum edge-to-edge distance."""
    if not first or not second:
        return math.inf
    if footprint_overlap(first, second):
        return 0.0
    return min(
        _segment_distance(first_start, first_end, second_start, second_end)
        for first_start, first_end in _edges(first)
        for second_start, second_end in _edges(second)
    )


def footprint_area(footprint: Optional[Polygon2D]) -> float:
    return 0.0 if not footprint else abs(_signed_area(footprint))


def object_yaw_rad(obj: dict) -> float:
    for key in ("yaw_rad", "table_yaw_rad"):
        if obj.get(key) is not None:
            return float(obj[key])
    if obj.get("table_yaw_deg") is not None:
        return math.radians(float(obj["table_yaw_deg"]))
    return 0.0


def object_inside_workspace(obj: dict, workspace: Optional[dict]) -> bool:
    """Require full XY footprint and vertical extent inside available workspace bounds."""
    footprint = object_footprint_polygon(obj)
    if not footprint_inside_region(footprint, workspace):
        return False
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None or not workspace:
        return False
    zmin = workspace.get("zmin")
    zmax = workspace.get("zmax")
    if zmin is not None and center[2] - 0.5 * size[2] < float(zmin):
        return False
    if zmax is not None and center[2] + 0.5 * size[2] > float(zmax):
        return False
    return True


def _polygon_axes(polygon: Polygon2D) -> List[Point2D]:
    axes = []
    for start, end in _edges(polygon):
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = math.hypot(dx, dy)
        if norm > 1e-12:
            axes.append((-dy / norm, dx / norm))
    return axes


def _projection_range(polygon: Polygon2D, axis: Point2D) -> Tuple[float, float]:
    values = [point[0] * axis[0] + point[1] * axis[1] for point in polygon]
    return min(values), max(values)


def _edges(polygon: Polygon2D) -> Iterable[Tuple[Point2D, Point2D]]:
    for index, point in enumerate(polygon):
        yield point, polygon[(index + 1) % len(polygon)]


def _clip_polygon(polygon: Polygon2D, edge_start: Point2D, edge_end: Point2D) -> Polygon2D:
    output = []
    for current, following in _edges(polygon):
        current_inside = _inside_edge(current, edge_start, edge_end)
        following_inside = _inside_edge(following, edge_start, edge_end)
        if current_inside:
            output.append(current)
        if current_inside != following_inside:
            output.append(_line_intersection(current, following, edge_start, edge_end))
    return output


def _inside_edge(point: Point2D, start: Point2D, end: Point2D) -> bool:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]) >= -1e-12


def _line_intersection(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> Point2D:
    ab_x, ab_y = b[0] - a[0], b[1] - a[1]
    cd_x, cd_y = d[0] - c[0], d[1] - c[1]
    denominator = ab_x * cd_y - ab_y * cd_x
    if abs(denominator) < 1e-12:
        return b
    t = ((c[0] - a[0]) * cd_y - (c[1] - a[1]) * cd_x) / denominator
    return a[0] + t * ab_x, a[1] + t * ab_y


def _signed_area(polygon: Polygon2D) -> float:
    return 0.5 * sum(start[0] * end[1] - end[0] * start[1] for start, end in _edges(polygon))


def _segment_distance(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> float:
    return min(_point_segment_distance(point, start, end) for point, start, end in ((a, c, d), (b, c, d), (c, a, b), (d, a, b)))


def _point_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared))
    return math.hypot(point[0] - (start[0] + ratio * dx), point[1] - (start[1] + ratio * dy))
