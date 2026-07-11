"""Geometry-predicate evaluation for task completion after every observation."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .geometry_relations import get_center, get_size, object_xy_aabb, xy_aabb_overlap
from .llm_stack_blocks import object_label_contains
from .task_semantic_validation import object_matches_role


def evaluate_task_goal_progress(
    state: dict, task_contract: dict, grounded_task_plan: dict, semantics_config: dict,
) -> dict:
    """Recompute goal predicates from current geometry, never from prior object ids."""
    task_type = task_contract["task_type"]
    if task_type == "build_house":
        satisfied, unsatisfied, observations = _evaluate_house(state, task_contract, grounded_task_plan, semantics_config)
    else:
        satisfied, unsatisfied, observations = _evaluate_organize(state, task_contract, grounded_task_plan)
    return {
        "schema_version": "task_goal_progress_v1", "task_type": task_type,
        "scene_revision": state.get("scene_revision"), "satisfied_predicates": satisfied,
        "unsatisfied_predicates": unsatisfied, "task_complete": not unsatisfied,
        "repair_required": bool(unsatisfied), "role_observations": observations,
    }


def _evaluate_house(state: dict, contract: dict, plan: dict, config: dict) -> Tuple[List[str], List[str], dict]:
    role_specs = contract["goal_spec"].get("roles") or contract["goal_spec"].get("required_roles") or []
    roles = {item["role_id"]: item for item in role_specs}
    candidates = {role_id: _role_candidates(state, role, _expected_region(plan, role_id, config)) for role_id, role in roles.items()}
    best = next((combo for combo in itertools.product(candidates.get("left_support", []), candidates.get("right_support", []), candidates.get("roof", [])) if len({str(item.get("id")) for item in combo}) == 3), None)
    expected = ["left_support.on_table", "right_support.on_table", "left_support.left_of.right_support", "left_support.supports.roof", "right_support.supports.roof", "roof.bridges.left_support.right_support"]
    if best is None:
        return [], expected, {}
    left, right, roof = best; semantics = config.get("house_semantics", {})
    table_z = _table_z(state); contact_tol = float(semantics.get("vertical_contact_tolerance_m", 0.006)); min_overlap = float(semantics.get("minimum_support_overlap_ratio", 0.20)); separation = float(semantics.get("minimum_support_separation_m", 0.025))
    predicates = {
        "left_support.on_table": _on_table(left, table_z, contact_tol),
        "right_support.on_table": _on_table(right, table_z, contact_tol),
        "left_support.left_of.right_support": get_center(left)[0] + separation <= get_center(right)[0],
        "left_support.supports.roof": _supports(left, roof, contact_tol, min_overlap),
        "right_support.supports.roof": _supports(right, roof, contact_tol, min_overlap),
    }
    predicates["roof.bridges.left_support.right_support"] = predicates["left_support.supports.roof"] and predicates["right_support.supports.roof"] and get_center(left)[0] <= get_center(roof)[0] <= get_center(right)[0]
    satisfied = [name for name in expected if predicates.get(name)]
    unsatisfied = [name for name in expected if not predicates.get(name)]
    observations = {"left_support": _role_observation(left), "right_support": _role_observation(right), "roof": _role_observation(roof)}
    return satisfied, unsatisfied, observations


def _evaluate_organize(state: dict, contract: dict, plan: dict) -> Tuple[List[str], List[str], dict]:
    goal = contract["goal_spec"]; regions = {item.get("region_id"): item.get("bounds_base_m") for item in plan.get("target_regions", [])}; expected, satisfied, observations = [], [], {}
    for group in plan.get("groups", []):
        group_id, value, region = group.get("group_id"), group.get("group_value"), regions.get(group.get("target_region_id"))
        members = [obj for obj in _objects(state) if _group_match(obj, value, goal.get("grouping_key"))]
        inside = all(_inside_region(obj, region) for obj in members)
        spacing_ok = _minimum_spacing(members, float(goal.get("minimum_spacing_m", 0.015)))
        aligned = _row_aligned(members, float(goal.get("alignment_tolerance_m", 0.012)))
        for suffix, result in (("all_inside", inside), ("spacing", spacing_ok), ("aligned", aligned)):
            predicate = "{}.{}".format(group_id, suffix); expected.append(predicate)
            if result: satisfied.append(predicate)
        observations[group_id] = [_role_observation(obj) for obj in members]
    unsatisfied = [item for item in expected if item not in satisfied]
    return satisfied, unsatisfied, observations


def _role_candidates(state: dict, role: dict, expected_region: Optional[dict]) -> List[dict]:
    candidates = [obj for obj in _objects(state) if object_matches_role(obj, role)]
    if expected_region:
        regional = [obj for obj in candidates if _near_region(obj, expected_region)]
        return regional or candidates
    return candidates


def _expected_region(plan: dict, role_id: str, config: dict) -> Optional[dict]:
    for assignment in plan.get("role_assignments", []):
        if assignment.get("role_id") == role_id and assignment.get("expected_region"):
            return assignment["expected_region"]
    return None


def _supports(support: dict, roof: dict, tolerance: float, minimum_overlap: float) -> bool:
    support_center, support_size, roof_center, roof_size = get_center(support), get_size(support), get_center(roof), get_size(roof)
    if None in (support_center, support_size, roof_center, roof_size): return False
    support_top = support_center[2] + 0.5 * support_size[2]; roof_bottom = roof_center[2] - 0.5 * roof_size[2]
    support_aabb, roof_aabb = object_xy_aabb(support), object_xy_aabb(roof)
    if not support_aabb or not roof_aabb or abs(roof_bottom - support_top) > tolerance: return False
    _, _, area = xy_aabb_overlap(support_aabb, roof_aabb)
    return area / max(1e-9, (support_aabb["xmax"] - support_aabb["xmin"]) * (support_aabb["ymax"] - support_aabb["ymin"])) >= minimum_overlap


def _on_table(obj: dict, table_z: float, tolerance: float) -> bool:
    center, size = get_center(obj), get_size(obj)
    return center is not None and size is not None and abs((center[2] - 0.5 * size[2]) - table_z) <= tolerance


def _table_z(state: dict) -> float:
    plane = state.get("table_plane") or {}; normal = plane.get("normal_base") or []
    if plane.get("z_base_m") is not None: return float(plane["z_base_m"])
    objects = _objects(state); bottoms = [get_center(obj)[2] - 0.5 * get_size(obj)[2] for obj in objects if get_center(obj) and get_size(obj)]
    return min(bottoms) if bottoms else 0.0


def _inside_region(obj: dict, region: Optional[dict]) -> bool:
    center = get_center(obj)
    return bool(center and region and float(region["xmin"]) <= center[0] <= float(region["xmax"]) and float(region["ymin"]) <= center[1] <= float(region["ymax"]))


def _near_region(obj: dict, region: dict) -> bool:
    center = get_center(obj); tolerance = float(region.get("position_tolerance_m", 0.015)); target = region.get("center_base_m")
    return bool(center and target and math.hypot(center[0] - target[0], center[1] - target[1]) <= tolerance)


def _minimum_spacing(objects: List[dict], minimum: float) -> bool:
    return all(math.hypot(get_center(a)[0] - get_center(b)[0], get_center(a)[1] - get_center(b)[1]) >= minimum for a, b in itertools.combinations(objects, 2) if get_center(a) and get_center(b))


def _row_aligned(objects: List[dict], tolerance: float) -> bool:
    centers = [get_center(obj) for obj in objects if get_center(obj)]
    return len(centers) <= 1 or max(point[1] for point in centers) - min(point[1] for point in centers) <= tolerance


def _group_match(obj: dict, value: Any, grouping_key: str) -> bool:
    return object_label_contains(obj, str(value)) if grouping_key == "color" else str(value).lower() in str(obj.get("label") or "").lower()


def _objects(state: dict) -> List[dict]: return [obj for obj in state.get("objects", []) if isinstance(obj, dict) and not obj.get("is_workspace")]
def _role_observation(obj: dict) -> dict: return {"observed_object_id": obj.get("id"), "label": obj.get("label"), "center_base_m": get_center(obj)}
