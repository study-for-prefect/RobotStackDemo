"""Geometry-predicate evaluation for task completion after every observation."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geometry_relations import get_center, get_size
from .llm_stack_blocks import object_label_contains
from .task_geometry import (
    footprint_area,
    footprint_boundary_distance,
    footprint_inside_region,
    footprint_overlap,
    footprint_overlap_area,
    object_footprint_polygon,
    object_inside_workspace,
    object_yaw_rad,
)
from .task_semantic_validation import object_matches_role


HOUSE_PREDICATES = [
    "left_support.on_table",
    "right_support.on_table",
    "left_support.left_of.right_support",
    "supports.height_aligned",
    "supports.separated",
    "left_support.supports.roof",
    "right_support.supports.roof",
    "roof.bridges.left_support.right_support",
    "roof.tilt_within_limit",
    "house.inside_workspace",
]


def evaluate_task_goal_progress(
    state: dict,
    task_contract: dict,
    grounded_task_plan: dict,
    semantics_config: dict,
    previous_progress: Optional[dict] = None,
) -> dict:
    """Recompute task completion from current geometry; VLM completion text is ignored."""
    task_type = task_contract["task_type"]
    if task_type == "build_house":
        details = evaluate_house_assignments(
            state, task_contract, grounded_task_plan, semantics_config, previous_progress,
        )
    else:
        details = evaluate_organize_groups(state, task_contract, grounded_task_plan)
    scene_events = list(details.get("scene_events", []))
    scene_events.extend(_progress_events(state, details, previous_progress))
    return {
        "schema_version": "task_goal_progress_v2",
        "task_type": task_type,
        "scene_revision": state.get("scene_revision"),
        "satisfied_predicates": details["satisfied_predicates"],
        "unsatisfied_predicates": details["unsatisfied_predicates"],
        "task_complete": bool(details["task_complete"]),
        "repair_required": bool(details["unsatisfied_predicates"]),
        "role_observations": details.get("role_observations", {}),
        "role_assignment_candidates": details.get("role_assignment_candidates", []),
        "selected_role_assignment": details.get("selected_role_assignment"),
        "group_diagnostics": details.get("group_diagnostics", []),
        "scene_events": scene_events,
    }


def evaluate_house_assignments(
    state: dict,
    contract: dict,
    grounded_plan: dict,
    config: dict,
    previous_progress: Optional[dict] = None,
) -> dict:
    """Evaluate every distinct legal role combination and select the highest predicate score."""
    roles = _house_roles(contract)
    role_candidates = {
        role_id: [obj for obj in _objects(state) if object_matches_role(obj, role)]
        for role_id, role in roles.items()
    }
    preferred_ids = _preferred_role_ids(grounded_plan, previous_progress)
    evaluated = []
    combinations = itertools.product(
        role_candidates.get("left_support", []),
        role_candidates.get("right_support", []),
        role_candidates.get("roof", []),
    )
    for left_support, right_support, roof in combinations:
        assignment = {
            "left_support": left_support,
            "right_support": right_support,
            "roof": roof,
        }
        if len({str(obj.get("id")) for obj in assignment.values()}) != 3:
            continue
        result = evaluate_house_assignment(state, assignment, config)
        consistency = sum(
            str(obj.get("id")) == str(preferred_ids.get(role_id))
            for role_id, obj in assignment.items()
        )
        result["score"] = _house_progress_score(result, consistency)
        evaluated.append(result)
    if not evaluated:
        return {
            "satisfied_predicates": [],
            "unsatisfied_predicates": list(HOUSE_PREDICATES),
            "task_complete": False,
            "role_observations": {},
            "role_assignment_candidates": [],
            "selected_role_assignment": None,
            "scene_events": [{"type": "insufficient_role_resources"}],
        }
    selected = max(evaluated, key=lambda item: item["score"])
    return {
        **selected,
        "role_assignment_candidates": [_candidate_log(item) for item in evaluated],
        "selected_role_assignment": selected["assignment"],
    }


def evaluate_house_assignment(state: dict, assignment: Dict[str, dict], config: dict) -> dict:
    """Evaluate all configured house geometry predicates for one role assignment."""
    semantics = config.get("house_semantics", {})
    left_support = assignment["left_support"]
    right_support = assignment["right_support"]
    roof = assignment["roof"]
    table_z = _table_z(state)
    vertical_tolerance = float(semantics.get("vertical_contact_tolerance_m", 0.006))
    height_tolerance = float(semantics.get("support_height_tolerance_m", 0.008))
    overlap_ratio = float(semantics.get("minimum_support_overlap_ratio", 0.20))
    minimum_separation = float(semantics.get("minimum_support_separation_m", 0.025))
    maximum_tilt = float(semantics.get("maximum_structure_tilt_deg", 10.0))
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    left_top = _top_z(left_support)
    right_top = _top_z(right_support)
    support_distance = footprint_boundary_distance(
        object_footprint_polygon(left_support), object_footprint_polygon(right_support),
    )
    left_supports, left_detail = _supports(left_support, roof, vertical_tolerance, overlap_ratio)
    right_supports, right_detail = _supports(right_support, roof, vertical_tolerance, overlap_ratio)
    tilt_deg = _roof_tilt_deg(left_support, right_support, roof)
    predicates = {
        "left_support.on_table": _on_table(left_support, table_z, vertical_tolerance),
        "right_support.on_table": _on_table(right_support, table_z, vertical_tolerance),
        "left_support.left_of.right_support": get_center(left_support)[0] < get_center(right_support)[0],
        "supports.height_aligned": abs(left_top - right_top) <= height_tolerance,
        "supports.separated": support_distance >= minimum_separation,
        "left_support.supports.roof": left_supports,
        "right_support.supports.roof": right_supports,
        "roof.bridges.left_support.right_support": _bridges(left_support, right_support, roof, left_supports, right_supports),
        "roof.tilt_within_limit": tilt_deg <= maximum_tilt,
        "house.inside_workspace": all(object_inside_workspace(obj, workspace) for obj in assignment.values()),
    }
    satisfied = [name for name in HOUSE_PREDICATES if predicates[name]]
    unsatisfied = [name for name in HOUSE_PREDICATES if not predicates[name]]
    role_observations = {
        role_id: _role_observation(obj) for role_id, obj in assignment.items()
    }
    return {
        "assignment": {
            role_id: {"current_object_id": obj.get("id"), "label": obj.get("label")}
            for role_id, obj in assignment.items()
        },
        "satisfied_predicates": satisfied,
        "unsatisfied_predicates": unsatisfied,
        "task_complete": not unsatisfied,
        "role_observations": role_observations,
        "geometry_details": {
            "left_support": left_detail,
            "right_support": right_detail,
            "support_boundary_distance_m": support_distance,
            "support_top_height_delta_m": abs(left_top - right_top),
            "roof_tilt_deg": tilt_deg,
        },
    }


def evaluate_organize_groups(state: dict, contract: dict, plan: dict) -> dict:
    """Evaluate non-empty groups using full footprints, spacing, regions, and selected layout."""
    goal_spec = contract["goal_spec"]
    regions = {
        item.get("region_id"): item.get("bounds_base_m")
        for item in plan.get("target_regions", [])
    }
    all_objects = _objects(state)
    diagnostics = []
    satisfied = []
    unsatisfied = []
    grouped_ids = set()
    for group in plan.get("groups", []):
        group_id = str(group.get("group_id"))
        members = [obj for obj in all_objects if _group_match(obj, group, goal_spec.get("grouping_key"))]
        minimum_count = int(group.get("minimum_required_count", 1))
        region = regions.get(group.get("target_region_id"))
        inside = [obj for obj in members if footprint_inside_region(object_footprint_polygon(obj), region)]
        outside = [obj for obj in members if obj not in inside]
        overlapping_pairs = _overlapping_pairs(members)
        spacing_violations = _spacing_violations(members, float(goal_spec.get("minimum_spacing_m", 0.015)))
        layout_satisfied = evaluate_layout(
            members,
            str(goal_spec.get("layout_type")),
            float(goal_spec.get("alignment_tolerance_m", 0.012)),
        )
        predicates = {
            "{}.minimum_count".format(group_id): len(members) >= minimum_count,
            "{}.all_inside".format(group_id): len(inside) == len(members) and len(members) >= minimum_count,
            "{}.no_overlap".format(group_id): not overlapping_pairs and len(members) >= minimum_count,
            "{}.spacing".format(group_id): not spacing_violations and len(members) >= minimum_count,
            "{}.layout".format(group_id): layout_satisfied and len(members) >= minimum_count,
        }
        satisfied.extend(name for name, passed in predicates.items() if passed)
        unsatisfied.extend(name for name, passed in predicates.items() if not passed)
        grouped_ids.update(str(obj.get("id")) for obj in members)
        diagnostics.append({
            "group_id": group_id,
            "detected_member_count": len(members),
            "minimum_required_count": minimum_count,
            "temporarily_unobserved": len(members) < minimum_count,
            "inside_region": [obj.get("id") for obj in inside],
            "outside_region": [obj.get("id") for obj in outside],
            "overlapping_pairs": overlapping_pairs,
            "spacing_violations": spacing_violations,
            "layout_satisfied": layout_satisfied,
        })
    ungrouped = [obj.get("id") for obj in all_objects if str(obj.get("id")) not in grouped_ids]
    if ungrouped:
        unsatisfied.append("organize.all_objects_grouped")
    else:
        satisfied.append("organize.all_objects_grouped")
    return {
        "satisfied_predicates": satisfied,
        "unsatisfied_predicates": unsatisfied,
        "task_complete": not unsatisfied,
        "role_observations": {},
        "group_diagnostics": diagnostics,
        "scene_events": ([{"type": "temporarily_unobserved"}] if any(item["temporarily_unobserved"] for item in diagnostics) else []),
    }


def evaluate_layout(objects: Sequence[dict], layout_type: str, tolerance_m: float) -> bool:
    if layout_type == "rows":
        return evaluate_rows_layout(objects, tolerance_m)
    if layout_type == "columns":
        return evaluate_columns_layout(objects, tolerance_m)
    if layout_type == "grid":
        return evaluate_grid_layout(objects, tolerance_m)
    return False


def evaluate_rows_layout(objects: Sequence[dict], tolerance_m: float) -> bool:
    centers = _centers(objects)
    return bool(centers) and max(point[1] for point in centers) - min(point[1] for point in centers) <= tolerance_m


def evaluate_columns_layout(objects: Sequence[dict], tolerance_m: float) -> bool:
    centers = _centers(objects)
    return bool(centers) and max(point[0] for point in centers) - min(point[0] for point in centers) <= tolerance_m


def evaluate_grid_layout(objects: Sequence[dict], tolerance_m: float) -> bool:
    centers = _centers(objects)
    if not centers:
        return False
    x_clusters = _coordinate_clusters([point[0] for point in centers], tolerance_m)
    y_clusters = _coordinate_clusters([point[1] for point in centers], tolerance_m)
    occupied = set()
    for x, y, _z in centers:
        cell = (_nearest_cluster(x, x_clusters), _nearest_cluster(y, y_clusters))
        if cell in occupied:
            return False
        occupied.add(cell)
    return (
        len(occupied) == len(x_clusters) * len(y_clusters)
        and _stable_cluster_spacing(x_clusters, tolerance_m)
        and _stable_cluster_spacing(y_clusters, tolerance_m)
    )


def _supports(support: dict, roof: dict, tolerance: float, minimum_overlap: float) -> Tuple[bool, dict]:
    support_top = _top_z(support)
    roof_bottom = _bottom_z(roof)
    support_footprint = object_footprint_polygon(support)
    roof_footprint = object_footprint_polygon(roof)
    overlap_area = footprint_overlap_area(support_footprint, roof_footprint)
    ratio = overlap_area / max(1e-9, footprint_area(support_footprint))
    detail = {
        "vertical_gap_m": roof_bottom - support_top,
        "overlap_area_m2": overlap_area,
        "overlap_ratio": ratio,
    }
    return abs(roof_bottom - support_top) <= tolerance and ratio >= minimum_overlap, detail


def _bridges(left: dict, right: dict, roof: dict, left_supported: bool, right_supported: bool) -> bool:
    if not left_supported or not right_supported:
        return False
    left_center = get_center(left)
    right_center = get_center(right)
    support_angle = math.atan2(right_center[1] - left_center[1], right_center[0] - left_center[0])
    roof_angle = object_yaw_rad(roof)
    difference = abs(math.atan2(math.sin(roof_angle - support_angle), math.cos(roof_angle - support_angle)))
    difference = min(difference, abs(math.pi - difference))
    return difference <= math.radians(30.0)


def _roof_tilt_deg(left: dict, right: dict, roof: dict) -> float:
    for key in ("tilt_deg", "object_tilt_deg"):
        if roof.get(key) is not None:
            return abs(float(roof[key]))
    roll = float(roof.get("roll_rad", 0.0))
    pitch = float(roof.get("pitch_rad", 0.0))
    reported_tilt = math.degrees(math.hypot(roll, pitch))
    left_center = get_center(left)
    right_center = get_center(right)
    horizontal = math.hypot(right_center[0] - left_center[0], right_center[1] - left_center[1])
    support_tilt = math.degrees(math.atan2(abs(_top_z(left) - _top_z(right)), max(horizontal, 1e-9)))
    return max(reported_tilt, support_tilt)


def _on_table(obj: dict, table_z: float, tolerance: float) -> bool:
    return abs(_bottom_z(obj) - table_z) <= tolerance


def _table_z(state: dict) -> float:
    plane = state.get("table_plane") or {}
    if plane.get("z_base_m") is not None:
        return float(plane["z_base_m"])
    bottoms = [_bottom_z(obj) for obj in _objects(state) if get_center(obj) and get_size(obj)]
    return min(bottoms) if bottoms else 0.0


def _house_roles(contract: dict) -> Dict[str, dict]:
    role_specs = contract["goal_spec"].get("roles") or contract["goal_spec"].get("required_roles") or []
    return {item["role_id"]: item for item in role_specs}


def _preferred_role_ids(plan: dict, previous_progress: Optional[dict]) -> Dict[str, Any]:
    preferred = {
        item.get("role_id"): item.get("selected_object_id", item.get("object_id"))
        for item in plan.get("role_assignments", [])
    }
    for role_id, observation in (previous_progress or {}).get("role_observations", {}).items():
        if isinstance(observation, dict) and observation.get("observed_object_id") is not None:
            preferred[role_id] = observation["observed_object_id"]
    return preferred


def _house_progress_score(result: dict, consistency: int) -> float:
    support_count = sum(name.endswith("supports.roof") for name in result["satisfied_predicates"])
    bridge_bonus = 2.0 if "roof.bridges.left_support.right_support" in result["satisfied_predicates"] else 0.0
    workspace_bonus = 1.0 if "house.inside_workspace" in result["satisfied_predicates"] else -5.0
    return len(result["satisfied_predicates"]) + support_count + bridge_bonus + workspace_bonus + 0.1 * consistency


def _candidate_log(result: dict) -> dict:
    return {
        "assignment": result["assignment"],
        "satisfied_predicates": result["satisfied_predicates"],
        "unsatisfied_predicates": result["unsatisfied_predicates"],
        "score": result["score"],
        "task_complete": result["task_complete"],
    }


def _group_match(obj: dict, group: dict, grouping_key: str) -> bool:
    rule = group.get("matching_rule") or {}
    value = rule.get(grouping_key, group.get("group_value"))
    if value is None:
        return False
    if grouping_key == "color":
        return object_label_contains(obj, str(value))
    return str(value).lower() in str(obj.get("label") or "").lower()


def _overlapping_pairs(objects: Sequence[dict]) -> List[List[Any]]:
    return [
        [first.get("id"), second.get("id")]
        for first, second in itertools.combinations(objects, 2)
        if footprint_overlap(object_footprint_polygon(first), object_footprint_polygon(second))
    ]


def _spacing_violations(objects: Sequence[dict], minimum_spacing_m: float) -> List[dict]:
    violations = []
    for first, second in itertools.combinations(objects, 2):
        distance = footprint_boundary_distance(
            object_footprint_polygon(first), object_footprint_polygon(second),
        )
        if distance < minimum_spacing_m:
            violations.append({
                "object_ids": [first.get("id"), second.get("id")],
                "boundary_distance_m": distance,
            })
    return violations


def _coordinate_clusters(values: Sequence[float], tolerance_m: float) -> List[float]:
    clusters = []
    for value in sorted(values):
        matching = next((index for index, center in enumerate(clusters) if abs(value - center) <= tolerance_m), None)
        if matching is None:
            clusters.append(value)
        else:
            clusters[matching] = 0.5 * (clusters[matching] + value)
    return clusters


def _nearest_cluster(value: float, clusters: Sequence[float]) -> int:
    return min(range(len(clusters)), key=lambda index: abs(value - clusters[index]))


def _stable_cluster_spacing(clusters: Sequence[float], tolerance_m: float) -> bool:
    if len(clusters) <= 2:
        return True
    distances = [clusters[index + 1] - clusters[index] for index in range(len(clusters) - 1)]
    return max(distances) - min(distances) <= tolerance_m


def _centers(objects: Sequence[dict]) -> List[List[float]]:
    return [get_center(obj) for obj in objects if get_center(obj) is not None]


def _top_z(obj: dict) -> float:
    center, size = get_center(obj), get_size(obj)
    return float(center[2]) + 0.5 * float(size[2])


def _bottom_z(obj: dict) -> float:
    center, size = get_center(obj), get_size(obj)
    return float(center[2]) - 0.5 * float(size[2])


def _objects(state: dict) -> List[dict]:
    return [obj for obj in state.get("objects", []) if isinstance(obj, dict) and not obj.get("is_workspace")]


def _role_observation(obj: dict) -> dict:
    return {
        "observed_object_id": obj.get("id"),
        "label": obj.get("label"),
        "center_base_m": get_center(obj),
    }


def _progress_events(state: dict, details: dict, previous_progress: Optional[dict]) -> List[dict]:
    if not previous_progress:
        return []
    events = []
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    previous_observations = previous_progress.get("role_observations") or {}
    current_observations = details.get("role_observations") or {}
    for role_id, previous in previous_observations.items():
        current = current_observations.get(role_id)
        if not current:
            continue
        previous_id = previous.get("observed_object_id")
        current_id = current.get("observed_object_id")
        if str(previous_id) != str(current_id):
            events.append({"type": "object_id_changed", "role_id": role_id, "previous_object_id": previous_id, "current_object_id": current_id})
            events.append({"type": "role_reassigned", "role_id": role_id, "current_object_id": current_id})
        previous_center = previous.get("center_base_m")
        current_center = current.get("center_base_m")
        if previous_center and current_center and math.dist(previous_center[:3], current_center[:3]) > 0.06:
            events.append({"type": "object_moved_far", "role_id": role_id, "distance_m": math.dist(previous_center[:3], current_center[:3])})
        obj = next((item for item in _objects(state) if str(item.get("id")) == str(current_id)), None)
        if obj is not None and workspace and not object_inside_workspace(obj, workspace):
            events.append({"type": "object_left_workspace", "role_id": role_id, "current_object_id": current_id})
    lost = set(previous_progress.get("satisfied_predicates") or []) - set(details.get("satisfied_predicates") or [])
    events.extend({"type": "goal_predicate_lost", "predicate": predicate} for predicate in sorted(lost))
    return events
