"""Evaluate whether a predicted push harms later grasping or placement."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Union

from .geometry_relations import is_locked, object_xy_aabb, xy_aabb_overlap, xy_distance
from .grasp_yaw_search import select_best_grasp


ObjectDict = Dict[str, Any]


def _object_id(obj: ObjectDict) -> str:
    return str(obj.get("id"))


def _objects(scene: Union[Dict[str, Any], Iterable[ObjectDict]]) -> List[ObjectDict]:
    if isinstance(scene, dict):
        return [obj for obj in scene.get("objects", []) if isinstance(obj, dict)]
    return [obj for obj in scene if isinstance(obj, dict)]


def _find_object(objects: Iterable[ObjectDict], object_id: Any) -> Optional[ObjectDict]:
    for obj in objects:
        if _object_id(obj) == str(object_id):
            return obj
    return None


def _place_region_aabb(region: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if not isinstance(region, dict):
        return None
    if all(key in region for key in ("xmin", "xmax", "ymin", "ymax")):
        return {
            "xmin": float(region["xmin"]),
            "xmax": float(region["xmax"]),
            "ymin": float(region["ymin"]),
            "ymax": float(region["ymax"]),
            "zmin": -math.inf,
            "zmax": math.inf,
        }
    center = region.get("center_base_m") or region.get("center_xy_base_m")
    radius = region.get("radius_m")
    if isinstance(center, (list, tuple)) and len(center) >= 2 and radius is not None:
        radius_value = float(radius)
        return {
            "xmin": float(center[0]) - radius_value,
            "xmax": float(center[0]) + radius_value,
            "ymin": float(center[1]) - radius_value,
            "ymax": float(center[1]) + radius_value,
            "zmin": -math.inf,
            "zmax": math.inf,
        }
    return None


def _table_edge_violation(obj: ObjectDict, table_bounds: Optional[dict], edge_margin_m: float) -> bool:
    if table_bounds is None:
        return False
    aabb = object_xy_aabb(obj)
    if not aabb:
        return True
    try:
        return (
            aabb["xmin"] < float(table_bounds["xmin"]) + edge_margin_m
            or aabb["xmax"] > float(table_bounds["xmax"]) - edge_margin_m
            or aabb["ymin"] < float(table_bounds["ymin"]) + edge_margin_m
            or aabb["ymax"] > float(table_bounds["ymax"]) - edge_margin_m
        )
    except (KeyError, TypeError, ValueError):
        return True


def _report(
    feasible: bool,
    reason: str,
    future_blocking_cost: float,
    place_blocking_cost: float,
    protected_risk: float,
    congestion_cost: float,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "feasible": feasible,
        "reason": reason,
        "future_blocking_cost": future_blocking_cost,
        "place_blocking_cost": place_blocking_cost,
        "protected_structure_risk": protected_risk,
        "congestion_cost": congestion_cost,
        "details": details,
    }


def _future_target_report(
    scene_objects: List[ObjectDict],
    current_target: ObjectDict,
    future_targets: Iterable[ObjectDict],
    gripper_outer_width_m: float,
    gripper_inner_width_m: float,
) -> Dict[str, Any]:
    results = []
    for target in future_targets:
        target_id = target.get("id")
        if str(target_id) == str(current_target.get("id")):
            continue
        predicted_target = _find_object(scene_objects, target_id) or target
        grasp = select_best_grasp(
            predicted_target,
            scene_objects,
            gripper_outer_width_m=gripper_outer_width_m,
            gripper_inner_width_m=gripper_inner_width_m,
        )
        results.append(
            {
                "object_id": target_id,
                "grasp_feasible": grasp["grasp_feasible"],
                "selected_grasp_yaw_deg": grasp.get("selected_grasp_yaw_deg"),
                "blocking_objects": grasp.get("blocking_objects", []),
            }
        )
        if not grasp["grasp_feasible"]:
            return {
                "ok": False,
                "reason": "blocks_future_target",
                "cost": 10.0,
                "details": {"future_targets": results, "blocked_future_target": target_id},
            }
    return {"ok": True, "reason": "future_targets_clear", "cost": 0.0, "details": {"future_targets": results}}


def _place_region_report(
    scene_objects: List[ObjectDict],
    current_target: ObjectDict,
    future_place_regions: Iterable[Dict[str, Any]],
    protected_ids: set,
) -> Dict[str, Any]:
    for region in future_place_regions:
        region_aabb = _place_region_aabb(region)
        if not region_aabb:
            continue
        for obj in scene_objects:
            if str(obj.get("id")) == str(current_target.get("id")):
                continue
            if str(obj.get("id")) in protected_ids or is_locked(obj):
                continue
            obj_aabb = object_xy_aabb(obj)
            if not obj_aabb:
                continue
            overlap_area = xy_aabb_overlap(obj_aabb, region_aabb)[2]
            if overlap_area > 0.0:
                return {
                    "ok": False,
                    "reason": "blocks_future_place_region",
                    "cost": overlap_area,
                    "details": {"blocked_place_region": region},
                }
    return {"ok": True, "reason": "future_place_regions_clear", "cost": 0.0, "details": {}}


def _protected_structure_report(
    scene_objects: List[ObjectDict],
    moved_obj: Optional[ObjectDict],
    protected_objects: Iterable[ObjectDict],
) -> Dict[str, Any]:
    if moved_obj is None:
        return {"ok": True, "reason": "no_moved_object", "risk": 0.0}
    for protected in protected_objects:
        predicted_protected = _find_object(scene_objects, protected.get("id")) or protected
        if xy_distance(moved_obj, predicted_protected) < 0.055:
            return {"ok": False, "reason": "near_protected_structure", "risk": 10.0}
    for obj in scene_objects:
        if is_locked(obj) and xy_distance(moved_obj, obj) < 0.055:
            return {"ok": False, "reason": "near_protected_structure", "risk": 10.0}
    return {"ok": True, "reason": "protected_structures_clear", "risk": 0.0}


def evaluate_future_task_impact(
    predicted_scene: Union[Dict[str, Any], Iterable[ObjectDict]],
    current_target: ObjectDict,
    future_targets: Optional[Iterable[ObjectDict]] = None,
    future_place_regions: Optional[Iterable[Dict[str, Any]]] = None,
    protected_objects: Optional[Iterable[ObjectDict]] = None,
    moved_object_id: Any = None,
    table_bounds: Optional[dict] = None,
    gripper_outer_width_m: float = 0.112,
    gripper_inner_width_m: float = 0.048,
    edge_margin_m: float = 0.02,
    congestion_radius_m: float = 0.06,
) -> Dict[str, Any]:
    scene_objects = _objects(predicted_scene)
    details: Dict[str, Any] = {}
    future_blocking_cost = 0.0
    place_blocking_cost = 0.0
    protected_risk = 0.0
    congestion_cost = 0.0
    protected_objects_list = list(protected_objects or [])

    future = _future_target_report(
        scene_objects,
        current_target,
        future_targets or [],
        gripper_outer_width_m,
        gripper_inner_width_m,
    )
    details.update(future["details"])
    if not future["ok"]:
        future_blocking_cost = float(future["cost"])
        return _report(False, future["reason"], future_blocking_cost, place_blocking_cost, protected_risk, congestion_cost, details)

    moved_obj = _find_object(scene_objects, moved_object_id)
    moved_aabb = object_xy_aabb(moved_obj) if moved_obj is not None else {}
    protected_ids = {str(obj.get("id")) for obj in protected_objects_list}
    place = _place_region_report(scene_objects, current_target, future_place_regions or [], protected_ids)
    if not place["ok"]:
        place_blocking_cost = float(place["cost"])
        details.update(place["details"])
        return _report(False, place["reason"], future_blocking_cost, place_blocking_cost, protected_risk, congestion_cost, details)

    if moved_obj is not None and _table_edge_violation(moved_obj, table_bounds, edge_margin_m):
        return _report(False, "pushed_object_near_table_edge", future_blocking_cost, place_blocking_cost, protected_risk, congestion_cost, details)

    protected = _protected_structure_report(scene_objects, moved_obj, protected_objects_list)
    if not protected["ok"]:
        protected_risk = float(protected["risk"])
        return _report(False, protected["reason"], future_blocking_cost, place_blocking_cost, protected_risk, congestion_cost, details)

    if moved_obj is not None:
        neighbors = [
            obj for obj in scene_objects
            if str(obj.get("id")) != str(moved_object_id) and xy_distance(moved_obj, obj) < congestion_radius_m
        ]
        congestion_cost = float(len(neighbors))

    details["moved_object_aabb"] = moved_aabb
    return _report(True, "future_task_impact_clear", future_blocking_cost, place_blocking_cost, protected_risk, congestion_cost, details)
