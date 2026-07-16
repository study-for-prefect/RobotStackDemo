"""Stage-specific object geometry for a predicted tabletop push chain."""

from __future__ import annotations

import math
from typing import Any, Mapping

from robot_scene_pipeline.geometry_relations import object_xy_aabb


def object_bounds_for_push_stage(
    obj: Mapping[str, Any],
    stage: str,
    chain_analysis: Mapping[str, Any],
    push_plan: Mapping[str, Any],
) -> dict[str, float] | None:
    """Use a chain object's predicted final pose for the vertical retreat stage."""
    if stage != "retreat":
        return object_xy_aabb(dict(obj))
    object_id = str(obj.get("id") or "")
    displacements = chain_analysis.get("estimated_displacements_m") or {}
    if object_id not in displacements:
        return object_xy_aabb(dict(obj))
    center = obj.get("geometry_center_m") or obj.get("center_base_m")
    direction = push_plan.get("direction_base")
    if not (
        isinstance(center, (list, tuple)) and len(center) >= 3
        and isinstance(direction, (list, tuple)) and len(direction) >= 2
    ):
        return object_xy_aabb(dict(obj))
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return object_xy_aabb(dict(obj))
    distance = float(displacements[object_id])
    moved = dict(obj)
    moved["geometry_center_m"] = [
        float(center[0]) + dx / norm * distance,
        float(center[1]) + dy / norm * distance,
        float(center[2]),
    ]
    return object_xy_aabb(moved)
