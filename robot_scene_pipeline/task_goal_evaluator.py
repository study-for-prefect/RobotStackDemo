"""Code-owned geometry and orientation predicates for semantic task completion."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence

from .geometry_relations import get_center, get_size
from .house_task_definition import HOUSE_ROLE_IDS
from .llm_stack_blocks import object_label_contains
from .task_geometry import (
    footprint_area,
    footprint_boundary_distance,
    footprint_inside_region,
    footprint_overlap,
    footprint_overlap_area,
    object_footprint_polygon,
    object_inside_workspace,
)
from .task_semantic_validation import infer_object_shape, object_matches_role


HOUSE_PREDICATES = [
    "left_support_lower.on_table",
    "right_support_lower.on_table",
    "left_support_lower.supports.left_support_upper",
    "right_support_lower.supports.right_support_upper",
    "left_column.vertical_aligned",
    "right_column.vertical_aligned",
    "columns.height_aligned",
    "left_support_upper.supports.roof",
    "right_support_upper.supports.roof",
    "roof.bridges.upper_supports",
    "roof.correct_face_up",
    "roof.opening_down",
    "roof.straight_edge_up",
    "roof.orientation_correct",
    "roof.supports.triangle_top",
    "triangle_top.correct_face",
    "triangle_top.apex_up",
    "triangle_top.not_side_lying",
    "triangle_top.centered_on_roof",
    "house.inside_workspace",
]


def evaluate_task_goal_progress(
    state: dict,
    task_contract: dict,
    grounded_task_plan: dict,
    semantics_config: dict,
    previous_progress: Optional[dict] = None,
) -> dict:
    """Recompute completion from the latest scene; VLM completion claims are ignored."""
    task_type = task_contract["task_type"]
    details = (
        evaluate_house_assignments(state, task_contract, grounded_task_plan, semantics_config, previous_progress)
        if task_type == "build_house"
        else evaluate_organize_groups(state, task_contract, grounded_task_plan)
    )
    scene_events = list(details.get("scene_events", []))
    scene_events.extend(_progress_events(state, details, previous_progress))
    return {
        "schema_version": "task_goal_progress_v3",
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
        "house_frame": grounded_task_plan.get("house_frame"),
        "orientation_results": grounded_task_plan.get("fused_orientation_results", []),
        "scene_events": scene_events,
    }


def evaluate_house_assignments(
    state: dict,
    contract: dict,
    grounded_plan: dict,
    config: dict,
    previous_progress: Optional[dict] = None,
) -> dict:
    """Search every distinct four-square/roof/triangle assignment."""
    roles = _house_roles(contract)
    candidates = {
        role_id: [obj for obj in _objects(state) if object_matches_role(obj, roles[role_id])]
        for role_id in HOUSE_ROLE_IDS
    }
    candidates["roof"] = sorted(
        candidates["roof"], key=lambda obj: infer_object_shape(obj) != "concave_rectangle",
    )
    preferred_ids = _preferred_role_ids(grounded_plan, previous_progress)
    orientation_by_id = {
        (item.get("role_id"), str(item.get("selected_object_id"))): item
        for item in grounded_plan.get("fused_orientation_results", [])
    }
    evaluated = []
    pools = [candidates[role_id] for role_id in HOUSE_ROLE_IDS]
    for selected in itertools.product(*pools):
        if len({str(obj.get("id")) for obj in selected}) != len(HOUSE_ROLE_IDS):
            continue
        assignment = dict(zip(HOUSE_ROLE_IDS, selected))
        result = evaluate_house_assignment(state, assignment, config, orientation_by_id)
        consistency = sum(
            str(obj.get("id")) == str(preferred_ids.get(role_id))
            for role_id, obj in assignment.items()
        )
        concave_bonus = 0.25 if infer_object_shape(assignment["roof"]) == "concave_rectangle" else 0.0
        result["score"] = len(result["satisfied_predicates"]) + 0.1 * consistency + concave_bonus
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


def evaluate_house_assignment(
    state: dict,
    assignment: Dict[str, dict],
    config: dict,
    orientation_by_id: Optional[dict] = None,
) -> dict:
    """Evaluate the fixed two-column/two-level/roof/triangle structure."""
    semantics = config.get("house_semantics", {})
    contact_tolerance = float(semantics.get("vertical_contact_tolerance_m", 0.006))
    overlap_ratio = float(semantics.get("minimum_support_overlap_ratio", 0.20))
    column_tolerance = float(semantics.get("column_center_tolerance_m", 0.008))
    height_tolerance = float(semantics.get("column_height_tolerance_m", 0.008))
    triangle_tolerance = float(semantics.get("triangle_center_tolerance_m", 0.010))
    roof_position_tolerance = float(semantics.get("roof_position_tolerance_m", 0.010))
    roof_orientation_tolerance = max(
        float(semantics.get("roof_yaw_tolerance_deg", 8.0)),
        float(semantics.get("roof_flip_tolerance_deg", 10.0)),
    )
    triangle_orientation_tolerance = max(
        float(semantics.get("triangle_yaw_tolerance_deg", 8.0)),
        float(semantics.get("triangle_tilt_tolerance_deg", 10.0)),
    )
    lower_left, lower_right = assignment["left_support_lower"], assignment["right_support_lower"]
    upper_left, upper_right = assignment["left_support_upper"], assignment["right_support_upper"]
    roof, triangle = assignment["roof"], assignment["triangle_top"]
    roof_orientation = _orientation_result(orientation_by_id, "roof", roof)
    triangle_orientation = _orientation_result(orientation_by_id, "triangle_top", triangle)
    left_stack = _supports(lower_left, upper_left, contact_tolerance, overlap_ratio)
    right_stack = _supports(lower_right, upper_right, contact_tolerance, overlap_ratio)
    left_roof = _supports(upper_left, roof, contact_tolerance, overlap_ratio)
    right_roof = _supports(upper_right, roof, contact_tolerance, overlap_ratio)
    roof_triangle = _supports(roof, triangle, contact_tolerance, overlap_ratio)
    predicates = {
        "left_support_lower.on_table": _on_table(lower_left, _table_z(state), contact_tolerance),
        "right_support_lower.on_table": _on_table(lower_right, _table_z(state), contact_tolerance),
        "left_support_lower.supports.left_support_upper": left_stack,
        "right_support_lower.supports.right_support_upper": right_stack,
        "left_column.vertical_aligned": _xy_distance(lower_left, upper_left) <= column_tolerance,
        "right_column.vertical_aligned": _xy_distance(lower_right, upper_right) <= column_tolerance,
        "columns.height_aligned": abs(_top_z(upper_left) - _top_z(upper_right)) <= height_tolerance,
        "left_support_upper.supports.roof": left_roof,
        "right_support_upper.supports.roof": right_roof,
        "roof.bridges.upper_supports": left_roof and right_roof and _centered_between(roof, upper_left, upper_right, roof_position_tolerance),
        "roof.correct_face_up": bool(roof_orientation.get("correct_face_up")),
        "roof.opening_down": bool(roof_orientation.get("opening_down")),
        "roof.straight_edge_up": bool(roof_orientation.get("straight_edge_up")),
        "roof.orientation_correct": _roof_orientation_correct(roof_orientation) and _orientation_matches_candidates(roof, roof_orientation, roof_orientation_tolerance),
        "roof.supports.triangle_top": roof_triangle,
        "triangle_top.correct_face": bool(triangle_orientation.get("correct_face")) and _orientation_matches_candidates(triangle, triangle_orientation, triangle_orientation_tolerance),
        "triangle_top.apex_up": bool(triangle_orientation.get("apex_up")),
        "triangle_top.not_side_lying": bool(triangle_orientation.get("not_side_lying")),
        "triangle_top.centered_on_roof": _xy_distance(roof, triangle) <= triangle_tolerance,
        "house.inside_workspace": all(
            object_inside_workspace(obj, state.get("table_bounds") or state.get("workspace_bounds"))
            for obj in assignment.values()
        ),
    }
    satisfied = [name for name in HOUSE_PREDICATES if predicates[name]]
    unsatisfied = [name for name in HOUSE_PREDICATES if not predicates[name]]
    return {
        "assignment": {
            role_id: {"current_object_id": obj.get("id"), "label": obj.get("label")}
            for role_id, obj in assignment.items()
        },
        "satisfied_predicates": satisfied,
        "unsatisfied_predicates": unsatisfied,
        "task_complete": not unsatisfied,
        "role_observations": {role_id: _role_observation(obj) for role_id, obj in assignment.items()},
        "geometry_details": {
            "left_column_xy_error_m": _xy_distance(lower_left, upper_left),
            "right_column_xy_error_m": _xy_distance(lower_right, upper_right),
            "upper_height_delta_m": abs(_top_z(upper_left) - _top_z(upper_right)),
            "triangle_roof_center_error_m": _xy_distance(roof, triangle),
        },
    }


def evaluate_organize_groups(state: dict, contract: dict, plan: dict) -> dict:
    """Evaluate non-empty groups using full footprints, spacing, regions, and layout."""
    goal_spec = contract["goal_spec"]
    regions = {item.get("region_id"): item.get("bounds_base_m") for item in plan.get("target_regions", [])}
    all_objects, diagnostics, satisfied, unsatisfied, grouped_ids = _objects(state), [], [], [], set()
    for group in plan.get("groups", []):
        group_id = str(group.get("group_id"))
        members = [obj for obj in all_objects if _group_match(obj, group, goal_spec.get("grouping_key"))]
        minimum_count = int(group.get("minimum_required_count", 1))
        region = regions.get(group.get("target_region_id"))
        inside = [obj for obj in members if footprint_inside_region(object_footprint_polygon(obj), region)]
        overlapping_pairs = _overlapping_pairs(members)
        spacing_violations = _spacing_violations(members, float(goal_spec.get("minimum_spacing_m", 0.015)))
        layout_satisfied = evaluate_layout(members, str(goal_spec.get("layout_type")), float(goal_spec.get("alignment_tolerance_m", 0.012)))
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
            "group_id": group_id, "detected_member_count": len(members), "minimum_required_count": minimum_count,
            "temporarily_unobserved": len(members) < minimum_count,
            "inside_region": [obj.get("id") for obj in inside],
            "outside_region": [obj.get("id") for obj in members if obj not in inside],
            "overlapping_pairs": overlapping_pairs, "spacing_violations": spacing_violations,
            "layout_satisfied": layout_satisfied,
        })
    ungrouped = [obj.get("id") for obj in all_objects if str(obj.get("id")) not in grouped_ids]
    (unsatisfied if ungrouped else satisfied).append("organize.all_objects_grouped")
    return {
        "satisfied_predicates": satisfied, "unsatisfied_predicates": unsatisfied,
        "task_complete": not unsatisfied, "role_observations": {}, "group_diagnostics": diagnostics,
        "scene_events": ([{"type": "temporarily_unobserved"}] if any(item["temporarily_unobserved"] for item in diagnostics) else []),
    }


def evaluate_layout(objects: Sequence[dict], layout_type: str, tolerance_m: float) -> bool:
    if layout_type == "rows": return evaluate_rows_layout(objects, tolerance_m)
    if layout_type == "columns": return evaluate_columns_layout(objects, tolerance_m)
    if layout_type == "grid": return evaluate_grid_layout(objects, tolerance_m)
    return False


def evaluate_rows_layout(objects: Sequence[dict], tolerance_m: float) -> bool:
    centers = _centers(objects)
    return bool(centers) and max(p[1] for p in centers) - min(p[1] for p in centers) <= tolerance_m


def evaluate_columns_layout(objects: Sequence[dict], tolerance_m: float) -> bool:
    centers = _centers(objects)
    return bool(centers) and max(p[0] for p in centers) - min(p[0] for p in centers) <= tolerance_m


def evaluate_grid_layout(objects: Sequence[dict], tolerance_m: float) -> bool:
    centers = _centers(objects)
    if not centers: return False
    xs, ys = _coordinate_clusters([p[0] for p in centers], tolerance_m), _coordinate_clusters([p[1] for p in centers], tolerance_m)
    occupied = {(_nearest_cluster(x, xs), _nearest_cluster(y, ys)) for x, y, _z in centers}
    return len(occupied) == len(centers) == len(xs) * len(ys) and _stable_cluster_spacing(xs, tolerance_m) and _stable_cluster_spacing(ys, tolerance_m)


def _supports(support: dict, upper: dict, tolerance: float, minimum_overlap: float) -> bool:
    overlap = footprint_overlap_area(object_footprint_polygon(support), object_footprint_polygon(upper))
    reference_area = min(
        footprint_area(object_footprint_polygon(support)),
        footprint_area(object_footprint_polygon(upper)),
    )
    ratio = overlap / max(reference_area, 1e-9)
    return abs(_bottom_z(upper) - _top_z(support)) <= tolerance and ratio >= minimum_overlap


def _orientation_result(results: Optional[dict], role_id: str, obj: dict) -> dict:
    if not results: return {}
    return results.get((role_id, str(obj.get("id"))), {})


def _roof_orientation_correct(result: dict) -> bool:
    return (
        all(bool(result.get(key)) for key in ("correct_face_up", "opening_down", "straight_edge_up"))
        and not result.get("flip_required")
        and not result.get("reobserve_required")
    )


def _centered_between(item: dict, first: dict, second: dict, tolerance: float) -> bool:
    center, a, b = get_center(item), get_center(first), get_center(second)
    midpoint = [(a[index] + b[index]) * 0.5 for index in range(2)]
    return math.hypot(center[0] - midpoint[0], center[1] - midpoint[1]) <= tolerance


def _orientation_matches_candidates(obj: dict, result: dict, tolerance_deg: float) -> bool:
    current = obj.get("orientation_xyzw") or obj.get("object_orientation_xyzw")
    candidates = result.get("candidate_target_orientations") or []
    if not isinstance(current, (list, tuple)) or len(current) != 4 or not candidates:
        return False
    return min(_quaternion_distance_deg(current, candidate) for candidate in candidates) <= tolerance_deg


def _quaternion_distance_deg(first: Sequence[float], second: Sequence[float]) -> float:
    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return math.inf
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second)) / (first_norm * second_norm))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _on_table(obj: dict, table_z: float, tolerance: float) -> bool:
    return abs(_bottom_z(obj) - table_z) <= tolerance


def _table_z(state: dict) -> float:
    plane = state.get("table_plane") or {}
    if plane.get("z_base_m") is not None: return float(plane["z_base_m"])
    bottoms = [_bottom_z(obj) for obj in _objects(state) if get_center(obj) and get_size(obj)]
    return min(bottoms) if bottoms else 0.0


def _top_z(obj: dict) -> float:
    return float(get_center(obj)[2]) + 0.5 * float(get_size(obj)[2])


def _bottom_z(obj: dict) -> float:
    return float(get_center(obj)[2]) - 0.5 * float(get_size(obj)[2])


def _xy_distance(first: dict, second: dict) -> float:
    a, b = get_center(first), get_center(second)
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _house_roles(contract: dict) -> Dict[str, dict]:
    return {item["role_id"]: item for item in contract["goal_spec"].get("roles", [])}


def _preferred_role_ids(plan: dict, previous: Optional[dict]) -> Dict[str, Any]:
    preferred = {item.get("role_id"): item.get("selected_object_id") for item in plan.get("role_assignments", [])}
    for role_id, observation in (previous or {}).get("role_observations", {}).items():
        if observation.get("observed_object_id") is not None: preferred[role_id] = observation["observed_object_id"]
    return preferred


def _candidate_log(result: dict) -> dict:
    return {key: result[key] for key in ("assignment", "satisfied_predicates", "unsatisfied_predicates", "score", "task_complete")}


def _group_match(obj: dict, group: dict, key: str) -> bool:
    rule, value = group.get("matching_rule") or {}, None
    value = rule.get(key, group.get("group_value"))
    return value is not None and object_label_contains(obj, str(value))


def _overlapping_pairs(objects: Sequence[dict]) -> List[List[Any]]:
    return [[a.get("id"), b.get("id")] for a, b in itertools.combinations(objects, 2) if footprint_overlap(object_footprint_polygon(a), object_footprint_polygon(b))]


def _spacing_violations(objects: Sequence[dict], minimum: float) -> List[dict]:
    output = []
    for first, second in itertools.combinations(objects, 2):
        distance = footprint_boundary_distance(object_footprint_polygon(first), object_footprint_polygon(second))
        if distance < minimum: output.append({"object_ids": [first.get("id"), second.get("id")], "boundary_distance_m": distance})
    return output


def _coordinate_clusters(values: Sequence[float], tolerance: float) -> List[float]:
    clusters = []
    for value in sorted(values):
        index = next((i for i, center in enumerate(clusters) if abs(value - center) <= tolerance), None)
        if index is None: clusters.append(value)
        else: clusters[index] = 0.5 * (clusters[index] + value)
    return clusters


def _nearest_cluster(value: float, clusters: Sequence[float]) -> int:
    return min(range(len(clusters)), key=lambda index: abs(value - clusters[index]))


def _stable_cluster_spacing(clusters: Sequence[float], tolerance: float) -> bool:
    if len(clusters) <= 2: return True
    distances = [clusters[i + 1] - clusters[i] for i in range(len(clusters) - 1)]
    return max(distances) - min(distances) <= tolerance


def _centers(objects: Sequence[dict]) -> List[List[float]]:
    return [get_center(obj) for obj in objects if get_center(obj) is not None]


def _objects(state: dict) -> List[dict]:
    return [obj for obj in state.get("objects", []) if isinstance(obj, dict) and not obj.get("is_workspace")]


def _role_observation(obj: dict) -> dict:
    return {"observed_object_id": obj.get("id"), "label": obj.get("label"), "center_base_m": get_center(obj)}


def _progress_events(state: dict, details: dict, previous: Optional[dict]) -> List[dict]:
    if not previous: return []
    events, workspace = [], state.get("table_bounds") or state.get("workspace_bounds")
    for role_id, old in (previous.get("role_observations") or {}).items():
        current = (details.get("role_observations") or {}).get(role_id)
        if not current: continue
        if str(old.get("observed_object_id")) != str(current.get("observed_object_id")):
            events.extend(({"type": "object_id_changed", "role_id": role_id}, {"type": "role_reassigned", "role_id": role_id}))
        if old.get("center_base_m") and current.get("center_base_m") and math.dist(old["center_base_m"][:3], current["center_base_m"][:3]) > 0.06:
            events.append({"type": "object_moved_far", "role_id": role_id})
        obj = next((item for item in _objects(state) if str(item.get("id")) == str(current.get("observed_object_id"))), None)
        if obj and workspace and not object_inside_workspace(obj, workspace): events.append({"type": "object_left_workspace", "role_id": role_id})
    lost = set(previous.get("satisfied_predicates") or []) - set(details.get("satisfied_predicates") or [])
    events.extend({"type": "goal_predicate_lost", "predicate": item} for item in sorted(lost))
    return events
