"""Conservative contact-chain analysis for tabletop push motions."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from robot_scene_pipeline.geometry_relations import object_xy_aabb


def analyze_push_contact_chain(
    push_plan: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    ignored_ids: Iterable[str],
    protected_ids: Iterable[str],
    safety_margin_m: float,
) -> dict[str, Any]:
    """Build a direction-consistent movable chain and validate its joint sweep."""
    obstacle = dict(push_plan.get("obstacle") or {})
    primary_id = str(obstacle.get("id") or "")
    direction = _unit_direction(push_plan.get("direction_base"))
    distance_m = float(push_plan.get("distance_m", 0.0))
    if not primary_id or direction is None or distance_m <= 0.0:
        return _invalid_analysis(primary_id)

    protected = {str(value) for value in protected_ids}
    ignored = {str(value) for value in ignored_ids}
    scene_by_id = {
        str(item.get("id")): dict(item)
        for item in objects
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    scene_by_id.setdefault(primary_id, obstacle)
    displacements = {primary_id: distance_m}
    chain_ids = [primary_id]
    frontier = [primary_id]
    controlled_contacts: list[dict[str, Any]] = []
    hard_collisions: list[dict[str, Any]] = []

    while frontier:
        current_id = frontier.pop(0)
        current = scene_by_id.get(current_id)
        if current is None:
            continue
        current_distance = displacements[current_id]
        for other_id, other in scene_by_id.items():
            if other_id == current_id or other_id in ignored or other_id in chain_ids:
                continue
            gap = _forward_contact_gap_m(
                current,
                other,
                direction,
                safety_margin_m,
            )
            if gap is None or gap > current_distance + 1e-9:
                continue
            propagated = max(0.0, current_distance - max(0.0, gap))
            classification = _object_classification(other, other_id, protected)
            contact = {
                "operated_object_id": current_id,
                "contacted_object_id": other_id,
                "contact_gap_m": round(max(0.0, gap), 6),
                "estimated_displacement_upper_bound_m": round(propagated, 6),
                "stage": "horizontal_push",
                "collision_source": "pushed_object",
                "entity_type": classification,
            }
            if classification != "ordinary_object":
                reason = (
                    "protected_completed_object_collision"
                    if classification == "protected_completed_object"
                    else "fixed_obstacle_collision"
                )
                hard_collisions.append({
                    **contact,
                    "reason": reason,
                    "classification": "hard_collision",
                })
                continue
            chain_ids.append(other_id)
            displacements[other_id] = propagated
            frontier.append(other_id)
            controlled_contacts.append({
                **contact,
                "reason": "controlled_secondary_contact",
                "classification": "controlled_contact",
            })

    workspace = push_plan.get("workspace_bounds") or push_plan.get("table_bounds")
    if isinstance(workspace, Mapping):
        for track_id in chain_ids:
            obj = scene_by_id.get(track_id)
            if obj is None or _translated_inside_workspace(
                obj,
                direction,
                displacements[track_id],
                workspace,
            ):
                continue
            hard_collisions.append({
                "operated_object_id": primary_id,
                "contacted_object_id": track_id,
                "entity_type": "workspace_boundary",
                "collision_source": "push_chain",
                "stage": "horizontal_push",
                "reason": "push_chain_workspace_violation",
                "classification": "hard_collision",
            })

    robot_exclusions = push_plan.get("robot_exclusion_geometry") or ()
    for track_id in chain_ids:
        obj = scene_by_id.get(track_id)
        if obj is None:
            continue
        swept = _translated_sweep_aabb(obj, direction, displacements[track_id])
        if any(_aabb_overlap(swept, region) for region in robot_exclusions if isinstance(region, Mapping)):
            hard_collisions.append({
                "operated_object_id": primary_id,
                "contacted_object_id": track_id,
                "entity_type": "robot_exclusion_geometry",
                "collision_source": "push_chain",
                "stage": "horizontal_push",
                "reason": "fixed_obstacle_collision",
                "classification": "hard_collision",
            })

    ordered_displacements = {
        track_id: round(displacements[track_id], 6)
        for track_id in chain_ids
    }
    return {
        "valid": True,
        "passed": not hard_collisions,
        "chain_track_ids": chain_ids,
        "chain_object_count": len(chain_ids),
        "secondary_contact_expected": len(chain_ids) > 1,
        "estimated_displacements_m": ordered_displacements,
        "estimated_displacements_are_upper_bounds": True,
        "estimation_method": "directional_aabb_gap_conservative_upper_bound",
        "controlled_contacts": controlled_contacts,
        "hard_collisions": hard_collisions,
    }


def pushed_object_sweep_collisions(
    push_plan: dict[str, Any],
    objects: list[dict[str, Any]],
    ignored_ids: Iterable[str],
    protected_ids: Iterable[str],
    safety_margin_m: float,
) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only genuine hard chain failures."""
    return list(analyze_push_contact_chain(
        push_plan,
        objects,
        ignored_ids,
        protected_ids,
        safety_margin_m,
    )["hard_collisions"])


def _invalid_analysis(primary_id: str) -> dict[str, Any]:
    return {
        "valid": False,
        "passed": False,
        "chain_track_ids": [primary_id] if primary_id else [],
        "chain_object_count": 1 if primary_id else 0,
        "secondary_contact_expected": False,
        "estimated_displacements_m": {},
        "estimated_displacements_are_upper_bounds": True,
        "estimation_method": "invalid_push_plan",
        "controlled_contacts": [],
        "hard_collisions": [{
            "entity_type": "invalid_push_plan",
            "collision_source": "push_chain",
            "stage": "generation",
            "reason": "invalid_push_plan",
            "classification": "hard_collision",
        }],
    }


def _unit_direction(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    dx, dy = float(value[0]), float(value[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return dx / norm, dy / norm


def _object_classification(
    obj: Mapping[str, Any],
    object_id: str,
    protected_ids: set[str],
) -> str:
    kind = str(obj.get("protection_kind") or "")
    if object_id in protected_ids and (
        kind == "completed_object" or obj.get("is_completed") is True
    ):
        return "protected_completed_object"
    if (
        object_id in protected_ids
        or kind in {"fixed_obstacle", "protected_structure"}
        or obj.get("is_fixed") is True
        or obj.get("movable") is False
        or str(obj.get("role") or "") in {"fixed_obstacle", "robot_exclusion"}
    ):
        return "fixed_obstacle"
    return "ordinary_object"


def _forward_contact_gap_m(
    current: Mapping[str, Any],
    other: Mapping[str, Any],
    direction: tuple[float, float],
    safety_margin_m: float,
) -> float | None:
    current_box = object_xy_aabb(current, margin_m=0.5 * safety_margin_m)
    other_box = object_xy_aabb(other, margin_m=0.5 * safety_margin_m)
    if not current_box or not other_box or not _z_overlap(current_box, other_box):
        return None
    dx, dy = direction
    if abs(dx) >= abs(dy):
        if min(current_box["ymax"], other_box["ymax"]) <= max(current_box["ymin"], other_box["ymin"]):
            return None
        current_center = 0.5 * (current_box["xmin"] + current_box["xmax"])
        other_center = 0.5 * (other_box["xmin"] + other_box["xmax"])
        if dx > 0.0 and other_center <= current_center:
            return None
        if dx < 0.0 and other_center >= current_center:
            return None
        return (
            other_box["xmin"] - current_box["xmax"]
            if dx > 0.0 else current_box["xmin"] - other_box["xmax"]
        )
    if min(current_box["xmax"], other_box["xmax"]) <= max(current_box["xmin"], other_box["xmin"]):
        return None
    current_center = 0.5 * (current_box["ymin"] + current_box["ymax"])
    other_center = 0.5 * (other_box["ymin"] + other_box["ymax"])
    if dy > 0.0 and other_center <= current_center:
        return None
    if dy < 0.0 and other_center >= current_center:
        return None
    return (
        other_box["ymin"] - current_box["ymax"]
        if dy > 0.0 else current_box["ymin"] - other_box["ymax"]
    )


def _translated_inside_workspace(
    obj: Mapping[str, Any],
    direction: tuple[float, float],
    distance_m: float,
    workspace: Mapping[str, Any],
) -> bool:
    end = object_xy_aabb(obj)
    if not end:
        return False
    dx, dy = direction[0] * distance_m, direction[1] * distance_m
    return (
        end["xmin"] + dx >= float(workspace.get("xmin", -math.inf))
        and end["xmax"] + dx <= float(workspace.get("xmax", math.inf))
        and end["ymin"] + dy >= float(workspace.get("ymin", -math.inf))
        and end["ymax"] + dy <= float(workspace.get("ymax", math.inf))
    )


def _translated_sweep_aabb(
    obj: Mapping[str, Any],
    direction: tuple[float, float],
    distance_m: float,
) -> dict[str, float]:
    bounds = object_xy_aabb(obj)
    dx, dy = direction[0] * distance_m, direction[1] * distance_m
    return {
        **bounds,
        "xmin": min(bounds["xmin"], bounds["xmin"] + dx),
        "xmax": max(bounds["xmax"], bounds["xmax"] + dx),
        "ymin": min(bounds["ymin"], bounds["ymin"] + dy),
        "ymax": max(bounds["ymax"], bounds["ymax"] + dy),
    }


def _aabb_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    required = {"xmin", "xmax", "ymin", "ymax"}
    if not required.issubset(first) or not required.issubset(second):
        return False
    return (
        min(float(first["xmax"]), float(second["xmax"]))
        > max(float(first["xmin"]), float(second["xmin"]))
        and min(float(first["ymax"]), float(second["ymax"]))
        > max(float(first["ymin"]), float(second["ymin"]))
    )


def _z_overlap(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    return min(first["zmax"], second["zmax"]) > max(first["zmin"], second["zmin"])
