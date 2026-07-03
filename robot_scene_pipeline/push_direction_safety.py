"""Safety checks for individual push directions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .geometry_relations import object_xy_aabb, xy_aabb_overlap


ObjectDict = Dict[str, Any]


def direction_safety_defaults() -> Dict[str, bool]:
    return {
        "pushed_object_loose_movable": False,
        "approach_path_safe": False,
        "push_swept_safe": False,
        "push_end_safe": False,
        "post_push_grasp_feasible": False,
    }


def table_bounds_ok(obj: ObjectDict, table_bounds: dict, edge_margin_m: float) -> Tuple[bool, str]:
    if table_bounds is None:
        return True, "table_bounds_not_available"
    aabb = object_xy_aabb(obj)
    if not aabb:
        return False, "missing_pushed_object_aabb"
    try:
        if (
            aabb["xmin"] < float(table_bounds["xmin"])
            or aabb["xmax"] > float(table_bounds["xmax"])
            or aabb["ymin"] < float(table_bounds["ymin"])
            or aabb["ymax"] > float(table_bounds["ymax"])
        ):
            return False, "push_end_out_of_table_bounds"
        if (
            aabb["xmin"] < float(table_bounds["xmin"]) + edge_margin_m
            or aabb["xmax"] > float(table_bounds["xmax"]) - edge_margin_m
            or aabb["ymin"] < float(table_bounds["ymin"]) + edge_margin_m
            or aabb["ymax"] > float(table_bounds["ymax"]) - edge_margin_m
        ):
            return False, "push_end_near_table_edge"
    except (KeyError, TypeError, ValueError):
        return False, "invalid_table_bounds"
    return True, "table_bounds_clear"


def push_end_collisions(predicted_obstacle: ObjectDict, predicted_objects: Iterable[ObjectDict]) -> List[Dict[str, Any]]:
    obstacle_aabb = object_xy_aabb(predicted_obstacle)
    if not obstacle_aabb:
        return [{"id": predicted_obstacle.get("id"), "reason": "missing_pushed_object_aabb"}]
    collisions: List[Dict[str, Any]] = []
    obstacle_id = str(predicted_obstacle.get("id"))
    for other in predicted_objects:
        if not isinstance(other, dict) or str(other.get("id")) == obstacle_id:
            continue
        other_aabb = object_xy_aabb(other)
        if not other_aabb or xy_aabb_overlap(obstacle_aabb, other_aabb)[2] <= 0.0:
            continue
        collisions.append({
            "id": other.get("id"), "label": other.get("label"),
            "role": other.get("role"), "state": other.get("state"),
            "reason": "push_end_collision",
        })
    return collisions
