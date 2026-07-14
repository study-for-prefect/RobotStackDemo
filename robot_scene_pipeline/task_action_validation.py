"""Task-semantic validation for one VLM-selected pick/place action."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from .geometry_relations import get_center, get_size
from .llm_stack_blocks import object_label_contains
from .reorientation_planner import plan_reorientation, quaternion_multiply
from .task_geometry import (
    footprint_area,
    footprint_boundary_distance,
    footprint_inside_region,
    footprint_overlap,
    footprint_overlap_area,
    object_footprint_polygon,
    object_inside_workspace,
)
from .task_goal_evaluator import evaluate_layout
from .task_semantic_validation import object_matches_role
from .task_schemas import TASK_ACTION_SCHEMA, validate_against_schema


def validate_task_action(
    proposal: dict,
    state: dict,
    task_contract: dict,
    grounded_plan: dict,
    goal_progress: dict,
    dynamic_protection: Optional[dict] = None,
    semantics_config: Optional[dict] = None,
) -> Tuple[Optional[dict], dict]:
    """Dispatch pick_place validation by task type without changing the VLM pose."""
    action_type = str(proposal.get("action_type") or "").lower()
    if action_type in {"reobserve", "stop"}:
        return None, _report("task_action_validation", proposal, "policy_requested_{}".format(action_type))
    if action_type not in {"pick_place", "pick_reorient_place"}:
        return None, _report("task_action_validation", proposal, "unsupported_task_action_type", ["action_type"])
    common_failure = _validate_common(proposal, state)
    if common_failure is not None:
        return None, common_failure
    task_type = task_contract.get("task_type")
    if task_type == "build_house":
        return validate_house_pick_place(
            proposal, state, task_contract, grounded_plan, goal_progress,
            dynamic_protection or {}, semantics_config or {},
        )
    if task_type == "organize_blocks":
        return validate_organize_pick_place(
            proposal, state, task_contract, grounded_plan, goal_progress,
        )
    return None, _report("task_action_validation", proposal, "unsupported_task_type", ["task_type"])


def validate_house_pick_place(
    proposal: dict,
    state: dict,
    contract: dict,
    grounded_plan: dict,
    progress: dict,
    dynamic_protection: dict,
    semantics_config: dict,
) -> Tuple[Optional[dict], dict]:
    """Validate a house pose against the selected role and current structure."""
    report = _report("task_pose_semantic_validation", proposal, "semantic_validation_failed")
    role_id = proposal.get("role_id")
    role = _roles(contract).get(role_id)
    obj = _object_by_id(state, proposal.get("selected_object_id"))
    if role is None:
        return None, _fail(report, "unknown_house_role", "role_id")
    if obj is None:
        return None, _fail(report, "selected_object_no_longer_exists", "selected_object_id")
    if not object_matches_role(obj, role):
        return None, _fail(report, "selected_object_no_longer_matches_role", "role_match")
    binding = next((item for item in grounded_plan.get("role_assignments", []) if item.get("role_id") == role_id), None)
    if binding is None or str(binding.get("selected_object_id")) != str(obj.get("id")):
        return None, _fail(report, "selected_object_does_not_match_current_role_binding", "selected_object_id")
    grounding_failure = _grounding_failure(obj, proposal, report)
    if grounding_failure:
        return None, grounding_failure
    missing_prerequisites = _missing_role_prerequisites(role_id, progress)
    if missing_prerequisites:
        failed = _fail(report, "role_prerequisites_not_satisfied", "role_id")
        failed["checks"]["role_id"]["detail"].update({
            "rejected_role_id": role_id,
            "missing_prerequisites": missing_prerequisites,
            "eligible_role_ids": [
                candidate for candidate in (
                    "left_support_lower", "right_support_lower",
                    "left_support_upper", "right_support_upper", "roof", "triangle_top",
                )
                if not _missing_role_prerequisites(candidate, progress)
            ],
            "correction": "choose an eligible role; a role assignment does not mean its structural predicates are satisfied",
        })
        return None, failed
    orientation = next(
        (item for item in grounded_plan.get("fused_orientation_results", []) if item.get("role_id") == role_id and str(item.get("selected_object_id")) == str(obj.get("id"))),
        None,
    )
    action_type = str(proposal.get("action_type"))
    if role_id in {"roof", "triangle_top"}:
        if orientation is None or orientation.get("reobserve_required"):
            report["required_response"] = "reobserve"
            return None, _fail(report, "orientation_reobserve_required", "orientation_observation")
        threshold = float(semantics_config.get("house_semantics", {}).get("orientation_confidence_threshold", 0.75))
        if float(orientation.get("orientation_confidence", 0.0)) < threshold:
            return None, _fail(report, "orientation_confidence_below_threshold", "orientation_confidence")
        if orientation.get("flip_required") and action_type != "pick_reorient_place":
            return None, _fail(report, "flip_requires_pick_reorient_place", "action_type")
        if not orientation.get("flip_required") and action_type == "pick_reorient_place":
            return None, _fail(report, "unnecessary_three_dimensional_reorientation", "action_type")
    action = _build_code_oriented_action(proposal, obj, orientation, semantics_config)
    if isinstance(action, dict) and action.get("validation_error"):
        return None, _fail(report, action["validation_error"], "reorientation_plan")
    moved = _object_at_pose(obj, action["target_pose_base"])
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    failed_predicates: List[dict] = []
    if not object_inside_workspace(moved, workspace):
        failed_predicates.append(_predicate("house.inside_workspace", "target_full_footprint_outside_workspace"))
    semantics = semantics_config.get("house_semantics") or {}
    vertical_tolerance = float(semantics.get("vertical_contact_tolerance_m", 0.006))
    minimum_overlap = float(semantics.get("minimum_support_overlap_ratio", 0.20))
    observations = progress.get("role_observations") or {}
    current = {name: _object_by_id(state, item.get("observed_object_id")) for name, item in observations.items()}
    current[role_id] = moved
    if role_id in {"left_support_lower", "right_support_lower"}:
        if abs(_bottom_z(moved) - _table_z(state)) > vertical_tolerance:
            failed_predicates.append(_predicate("{}.on_table".format(role_id), "target_bottom_not_on_table"))
    elif role_id in {"left_support_upper", "right_support_upper"}:
        lower_role = role_id.replace("upper", "lower")
        lower = current.get(lower_role)
        if lower is None:
            failed_predicates.append(_predicate(
                "{}.supports.{}".format(lower_role, role_id),
                "current_lower_support_unavailable",
            ))
        elif not _supports_target(lower, moved, vertical_tolerance, minimum_overlap):
            expected_position = [
                float(get_center(lower)[0]),
                float(get_center(lower)[1]),
                _top_z(lower) + 0.5 * float(get_size(moved)[2]),
            ]
            failed = _fail(report, "upper_support_not_aligned_on_lower_support", "target_pose_base")
            failed["failed_predicates"] = [_predicate(
                "{}.supports.{}".format(lower_role, role_id),
                "upper_support_not_aligned_on_lower_support",
            )]
            failed["checks"]["target_pose_base"]["detail"].update({
                "received_position_m": get_center(moved),
                "support_role_id": lower_role,
                "support_center_base_m": get_center(lower),
                "support_dimensions_m": get_size(lower),
                "moved_dimensions_m": get_size(moved),
                "expected_aligned_position_m": expected_position,
                "correction": "place the upper center directly above its bound lower support at contact height",
            })
            return None, failed
    elif role_id == "roof":
        left, right = current.get("left_support_upper"), current.get("right_support_upper")
        if left is None or right is None:
            failed_predicates.append(_predicate("roof.supported_by_both_supports", "current_supports_unavailable"))
        else:
            for support, name in ((left, "left_support_upper"), (right, "right_support_upper")):
                overlap = footprint_overlap_area(object_footprint_polygon(support), object_footprint_polygon(moved))
                ratio = overlap / max(footprint_area(object_footprint_polygon(support)), 1e-9)
                if ratio < minimum_overlap:
                    failed_predicates.append(_predicate("roof.supported_by_both_supports", "insufficient_overlap_with_{}".format(name)))
                if abs(_bottom_z(moved) - _top_z(support)) > vertical_tolerance:
                    failed_predicates.append(_predicate("{}.supports.roof".format(name), "roof_contact_height_mismatch"))
    elif role_id == "triangle_top":
        roof = current.get("roof")
        tolerance = float(semantics.get("triangle_center_tolerance_m", 0.010))
        if roof is None or not _supports_target(roof, moved, vertical_tolerance, minimum_overlap):
            failed_predicates.append(_predicate("roof.supports.triangle_top", "triangle_bottom_not_supported_by_roof"))
        elif math.hypot(get_center(roof)[0] - get_center(moved)[0], get_center(roof)[1] - get_center(moved)[1]) > tolerance:
            failed_predicates.append(_predicate("triangle_top.centered_on_roof", "triangle_not_centered_on_roof"))
    failed_predicates.extend(_protected_pose_failures(moved, obj, dynamic_protection, state, role_id))
    if failed_predicates:
        report["failed_predicates"] = failed_predicates
        report["reason"] = failed_predicates[0]["reason"]
        return None, report
    return _accepted_action(action, obj, report)


def validate_organize_pick_place(
    proposal: dict,
    state: dict,
    contract: dict,
    grounded_plan: dict,
    progress: Optional[dict] = None,
) -> Tuple[Optional[dict], dict]:
    """Validate group binding, full target footprint, spacing, and layout."""
    report = _report("organize_action_validation", proposal, "organize_action_invalid")
    group_id = proposal.get("group_id")
    region_id = proposal.get("target_region_id")
    groups = {item.get("group_id"): item for item in grounded_plan.get("groups", [])}
    regions = {item.get("region_id"): item.get("bounds_base_m") for item in grounded_plan.get("target_regions", [])}
    group = groups.get(group_id)
    if group is None:
        return None, _fail(report, "unknown_group_id", "group_id")
    if region_id not in regions:
        return None, _fail(report, "unknown_target_region_id", "target_region_id")
    if str(group.get("target_region_id")) != str(region_id):
        return None, _fail(report, "target_region_does_not_match_group", "target_region_id")
    obj = _object_by_id(state, proposal.get("selected_object_id"))
    if obj is None:
        return None, _fail(report, "selected_object_no_longer_exists", "selected_object_id")
    grouping_key = contract.get("goal_spec", {}).get("grouping_key")
    matching = group.get("matching_rule") or {}
    group_value = matching.get(grouping_key, group.get("group_value"))
    if group_value is None or not object_label_contains(obj, str(group_value)):
        return None, _fail(report, "selected_object_does_not_match_group", "group_match")
    for other_group in grounded_plan.get("groups", []):
        if other_group is group:
            continue
        if any(str(value) == str(obj.get("id")) for value in other_group.get("object_ids", [])):
            return None, _fail(report, "selected_object_assigned_to_mutually_exclusive_group", "group_membership")
    grounding_failure = _grounding_failure(obj, proposal, report)
    if grounding_failure:
        return None, grounding_failure
    group_progress = next((
        item for item in (progress or {}).get("group_diagnostics", [])
        if str(item.get("group_id")) == str(group_id)
    ), None)
    outside_ids = {
        str(value) for value in (group_progress or {}).get("outside_region", [])
    }
    if outside_ids and str(obj.get("id")) not in outside_ids:
        failed = _fail(
            report,
            "selected_object_already_inside_while_group_has_outside_members",
            "selected_object_id",
        )
        failed["checks"]["selected_object_id"]["detail"].update({
            "selected_object_id": obj.get("id"),
            "required_outside_object_ids": sorted(outside_ids),
            "correction": "select a member from current_goal_progress.group_diagnostics[].outside_region",
        })
        return None, failed
    moved = _object_at_pose(obj, proposal["target_pose_base"])
    source_center = get_center(obj)
    target_center = get_center(moved)
    xy_displacement = math.hypot(
        float(target_center[0]) - float(source_center[0]),
        float(target_center[1]) - float(source_center[1]),
    )
    alignment_tolerance = float(contract.get("goal_spec", {}).get("alignment_tolerance_m", 0.012))
    minimum_reposition = max(0.005, min(0.010, 0.5 * alignment_tolerance))
    if xy_displacement < minimum_reposition:
        failed = _fail(report, "target_pose_too_close_to_source", "target_pose_base")
        detail = {
            "source_position_m": source_center,
            "received_position_m": target_center,
            "xy_displacement_m": round(xy_displacement, 6),
            "minimum_reposition_distance_m": round(minimum_reposition, 6),
            "correction": "choose an unsatisfied group member and a materially different collision-free XY",
        }
        suggested = _suggest_organize_target_position(
            obj=obj,
            state=state,
            group=group,
            region=regions[region_id],
            grouping_key=grouping_key,
            contract=contract,
            target_pose=proposal["target_pose_base"],
            minimum_reposition_m=minimum_reposition,
        )
        if suggested is not None:
            detail["suggested_collision_free_position_m"] = suggested
            detail["correction"] = (
                "use this geometry-checked position or another position satisfying the same constraints"
            )
        failed["checks"]["target_pose_base"]["detail"].update(detail)
        return None, failed
    if not footprint_inside_region(object_footprint_polygon(moved), regions[region_id]):
        region = regions[region_id]
        size = get_size(moved)
        yaw = float(moved.get("yaw_rad", 0.0))
        half_x = 0.5 * (
            abs(math.cos(yaw)) * float(size[0]) + abs(math.sin(yaw)) * float(size[1])
        )
        half_y = 0.5 * (
            abs(math.sin(yaw)) * float(size[0]) + abs(math.cos(yaw)) * float(size[1])
        )
        failed = _fail(report, "target_pose_outside_group_region", "target_pose_base")
        failed["checks"]["target_pose_base"]["detail"].update({
            "received_position_m": get_center(moved),
            "target_region_bounds_base_m": region,
            "object_dimensions_m": size,
            "target_yaw_rad": yaw,
            "allowed_center_x_m": [float(region["xmin"]) + half_x, float(region["xmax"]) - half_x],
            "allowed_center_y_m": [float(region["ymin"]) + half_y, float(region["ymax"]) - half_y],
            "suggested_interval_midpoint_position_m": [
                round(0.5 * (float(region["xmin"]) + float(region["xmax"])), 6),
                round(0.5 * (float(region["ymin"]) + float(region["ymax"])), 6),
                round(float(get_center(moved)[2]), 6),
            ],
            "correction": "choose a new XY inside both allowed center intervals; do not copy the source center",
        })
        return None, failed
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    if not object_inside_workspace(moved, workspace):
        return None, _fail(report, "target_pose_outside_workspace", "target_pose_base")
    minimum_spacing = float(contract.get("goal_spec", {}).get("minimum_spacing_m", 0.015))
    group_members = [moved]
    for other in state.get("objects", []):
        if str(other.get("id")) == str(obj.get("id")):
            continue
        other_footprint = object_footprint_polygon(other)
        if footprint_overlap(object_footprint_polygon(moved), other_footprint):
            detail = {
                "reason": "target_pose_overlaps_planned_object",
                "blocking_object_id": other.get("id"),
                "blocking_object_label": other.get("label"),
                "blocking_object_center_base_m": get_center(other),
            }
            suggested = _suggest_organize_target_position(
                obj=obj,
                state=state,
                group=group,
                region=regions[region_id],
                grouping_key=grouping_key,
                contract=contract,
                target_pose=proposal["target_pose_base"],
                minimum_reposition_m=minimum_reposition,
            )
            if suggested is not None:
                detail["suggested_collision_free_position_m"] = suggested
                detail["correction"] = (
                    "use this geometry-checked position or another position satisfying the same constraints"
                )
            report.setdefault("checks", {})["target_pose_base"] = {"detail": detail}
            return None, _fail(report, "target_pose_overlaps_planned_object", "target_pose_base")
        if _object_belongs_to_group(other, group, grouping_key):
            distance = footprint_boundary_distance(object_footprint_polygon(moved), other_footprint)
            if distance < minimum_spacing:
                report.setdefault("checks", {})["target_pose_base"] = {"detail": {
                    "reason": "target_pose_group_spacing_too_small",
                    "near_object_id": other.get("id"),
                    "boundary_distance_m": distance,
                    "minimum_spacing_m": minimum_spacing,
                }}
                return None, _fail(report, "target_pose_group_spacing_too_small", "target_pose_base")
            if footprint_inside_region(other_footprint, regions[region_id]):
                group_members.append(other)
    layout_type = str(contract.get("goal_spec", {}).get("layout_type") or "")
    tolerance = alignment_tolerance
    if not evaluate_layout(group_members, layout_type, tolerance):
        return None, _fail(report, "target_pose_violates_group_layout", "target_pose_base")
    return _accepted_action(proposal, obj, report)


def _validate_common(proposal: dict, state: dict) -> Optional[dict]:
    report = _report("action_schema_validation", proposal, "action_schema_invalid")
    schema_errors = validate_against_schema(proposal, TASK_ACTION_SCHEMA)
    if schema_errors:
        report["schema_errors"] = schema_errors
        return _fail(report, "task_action_json_schema_invalid", "action_schema")
    if "object_id" in proposal:
        if str(proposal.get("object_id")) != str(proposal.get("selected_object_id")):
            return _fail(report, "conflicting_object_id_fields", "selected_object_id")
        return _fail(report, "legacy_object_id_forbidden", "object_id")
    if proposal.get("selected_object_id") is None:
        return _fail(report, "missing_selected_object_id", "selected_object_id")
    try:
        if int(proposal.get("scene_revision", -1)) != int(state.get("scene_revision")):
            return _fail(report, "stale_scene_revision", "scene_revision")
    except (TypeError, ValueError):
        return _fail(report, "stale_scene_revision", "scene_revision")
    pose = proposal.get("target_pose_base") or {}
    position = pose.get("position_m", pose.get("position"))
    try:
        valid = isinstance(position, list) and len(position) == 3 and all(math.isfinite(float(v)) for v in position)
        orientation = pose.get("orientation_xyzw")
        yaw_valid = pose.get("yaw_rad") is not None and math.isfinite(float(pose.get("yaw_rad")))
        quaternion_valid = isinstance(orientation, list) and len(orientation) == 4 and all(math.isfinite(float(v)) for v in orientation)
        valid = valid and (yaw_valid or quaternion_valid or proposal.get("role_id") in {"roof", "triangle_top"})
    except (TypeError, ValueError):
        valid = False
    if valid:
        return None
    failed = _fail(report, "invalid_target_pose_base", "target_pose_base")
    failed["checks"]["target_pose_base"]["detail"].update({
        "received_target_pose_base": proposal.get("target_pose_base"),
        "required_format": {
            "position_m": ["x_m", "y_m", "z_m"],
            "yaw_rad": "finite number; roof/triangle may omit because code supplies orientation",
        },
        "correction": "output a complete target_pose_base; the source object center is not a placement target",
    })
    return failed


def _accepted_action(proposal: dict, obj: dict, report: dict) -> Tuple[dict, dict]:
    action = copy.deepcopy(proposal)
    action["selected_object_id"] = obj.get("id")
    action["selection_source"] = "vlm_task_action_policy"
    action.pop("object_id", None)
    report.update({"accepted": True, "passed": True, "reason": "accepted_pending_moveit_preflight"})
    return action, report


def _grounding_failure(obj: dict, proposal: dict, report: dict) -> Optional[dict]:
    if str(proposal.get("object_label") or "").lower() != str(obj.get("label") or "").lower():
        failed = _fail(report, "object_binding_grounding_mismatch", "object_label")
        failed["checks"]["object_label"]["detail"].update({
            "expected_object_label": obj.get("label"),
            "received_object_label": proposal.get("object_label"),
            "correction": "copy the selected object's label exactly from objects",
        })
        return failed
    observed = get_center(obj)
    proposed = proposal.get("object_center_base_m")
    try:
        valid = isinstance(proposed, list) and len(proposed) == 3 and math.dist(proposed, observed[:3]) <= 0.005
    except (TypeError, ValueError):
        valid = False
    if valid:
        return None
    failed = _fail(report, "object_binding_grounding_mismatch", "object_center_base_m")
    failed["checks"]["object_center_base_m"]["detail"].update({
        "expected_object_center_base_m": observed[:3],
        "received_object_center_base_m": proposed,
        "correction": "copy the selected object's geometry_center_base_m exactly; do not copy target_pose_base",
    })
    return failed


def _protected_pose_failures(moved: dict, source: dict, protection: dict, state: dict, role_id: str) -> List[dict]:
    allowed_roles = {
        "left_support_upper": {"left_support_lower"},
        "right_support_upper": {"right_support_lower"},
        "roof": {"left_support_upper", "right_support_upper"},
        "triangle_top": {"roof"},
    }.get(role_id, set())
    allowed = {item.get("current_object_id") for item in protection.get("protected_roles", []) if item.get("role_id") in allowed_roles}
    protected = {str(value) for value in protection.get("protected_object_ids", []) if value not in allowed}
    for other in state.get("objects", []):
        if str(other.get("id")) == str(source.get("id")) or str(other.get("id")) not in protected:
            continue
        if footprint_overlap(object_footprint_polygon(moved), object_footprint_polygon(other)):
            return [_predicate("protected_structure.collision_free", "target_pose_collides_with_protected_role")]
    return []


def _object_at_pose(obj: dict, pose: dict) -> dict:
    moved = copy.deepcopy(obj)
    moved["geometry_center_m"] = [float(value) for value in pose.get("position_m", pose.get("position"))]
    if pose.get("yaw_rad") is not None:
        moved["yaw_rad"] = float(pose["yaw_rad"])
    if pose.get("orientation_xyzw") is not None:
        moved["orientation_xyzw"] = list(pose["orientation_xyzw"])
    return moved


def _suggest_organize_target_position(
    obj: dict,
    state: dict,
    group: dict,
    region: dict,
    grouping_key: str,
    contract: dict,
    target_pose: dict,
    minimum_reposition_m: float,
) -> Optional[List[float]]:
    """Find one geometry-checked recovery point after a VLM target collision."""
    source = get_center(obj)
    size = get_size(obj)
    yaw = float(target_pose.get("yaw_rad", obj.get("yaw_rad", 0.0)))
    half_x = 0.5 * (abs(math.cos(yaw)) * float(size[0]) + abs(math.sin(yaw)) * float(size[1]))
    half_y = 0.5 * (abs(math.sin(yaw)) * float(size[0]) + abs(math.cos(yaw)) * float(size[1]))
    xmin, xmax = float(region["xmin"]) + half_x, float(region["xmax"]) - half_x
    ymin, ymax = float(region["ymin"]) + half_y, float(region["ymax"]) - half_y
    if xmin > xmax or ymin > ymax:
        return None
    same_group_inside = [
        other for other in state.get("objects", [])
        if str(other.get("id")) != str(obj.get("id"))
        and _object_belongs_to_group(other, group, grouping_key)
        and footprint_inside_region(object_footprint_polygon(other), region)
    ]
    layout = str(contract.get("goal_spec", {}).get("layout_type") or "")
    tolerance = float(contract.get("goal_spec", {}).get("alignment_tolerance_m", 0.012))
    spacing = float(contract.get("goal_spec", {}).get("minimum_spacing_m", 0.015))
    x_values = _candidate_axis_values(xmin, xmax, float(source[0]))
    y_values = _candidate_axis_values(ymin, ymax, float(source[1]))
    if same_group_inside and layout == "rows":
        y_values = [sum(get_center(item)[1] for item in same_group_inside) / len(same_group_inside)]
    elif same_group_inside and layout == "columns":
        x_values = [sum(get_center(item)[0] for item in same_group_inside) / len(same_group_inside)]
    positions = [(x, y) for x in x_values for y in y_values]
    positions.sort(key=lambda point: math.hypot(point[0] - source[0], point[1] - source[1]))
    target_position = target_pose.get("position_m", target_pose.get("position"))
    target_z = float(target_position[2])
    rejected_destination_bin = (
        round(float(target_position[0]), 2),
        round(float(target_position[1]), 2),
    )
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    for x, y in positions:
        if math.hypot(x - source[0], y - source[1]) < minimum_reposition_m:
            continue
        if (round(float(x), 2), round(float(y), 2)) == rejected_destination_bin:
            continue
        candidate = _object_at_pose(obj, {"position_m": [x, y, target_z], "yaw_rad": yaw})
        footprint = object_footprint_polygon(candidate)
        if not footprint_inside_region(footprint, region) or not object_inside_workspace(candidate, workspace):
            continue
        collision = False
        group_members = [candidate]
        for other in state.get("objects", []):
            if str(other.get("id")) == str(obj.get("id")):
                continue
            other_footprint = object_footprint_polygon(other)
            if footprint_overlap(footprint, other_footprint):
                collision = True
                break
            if _object_belongs_to_group(other, group, grouping_key):
                if footprint_boundary_distance(footprint, other_footprint) < spacing:
                    collision = True
                    break
                if footprint_inside_region(other_footprint, region):
                    group_members.append(other)
        if collision or not evaluate_layout(group_members, layout, tolerance):
            continue
        return [round(float(x), 6), round(float(y), 6), round(target_z, 6)]
    return None


def _candidate_axis_values(low: float, high: float, preferred: float) -> List[float]:
    preferred = min(max(preferred, low), high)
    values = [preferred, 0.5 * (low + high), low, high]
    steps = max(1, int(math.ceil((high - low) / 0.01)))
    values.extend(low + (high - low) * index / steps for index in range(steps + 1))
    unique = []
    for value in values:
        if not any(abs(value - seen) < 1e-6 for seen in unique):
            unique.append(value)
    return unique


def _dependency_failure(role_id: str, progress: dict) -> Optional[str]:
    return "role_prerequisites_not_satisfied" if _missing_role_prerequisites(role_id, progress) else None


def _missing_role_prerequisites(role_id: str, progress: dict) -> List[str]:
    satisfied = set(progress.get("satisfied_predicates") or [])
    required = {
        "left_support_upper": {"left_support_lower.on_table"},
        "right_support_upper": {"right_support_lower.on_table"},
        "roof": {
            "left_support_lower.supports.left_support_upper",
            "right_support_lower.supports.right_support_upper",
            "left_column.vertical_aligned", "right_column.vertical_aligned", "columns.height_aligned",
        },
        "triangle_top": {
            "left_support_upper.supports.roof", "right_support_upper.supports.roof",
            "roof.bridges.upper_supports", "roof.correct_face_up", "roof.opening_down",
            "roof.straight_edge_up", "roof.orientation_correct",
        },
    }.get(role_id, set())
    return sorted(required - satisfied)


def _build_code_oriented_action(
    proposal: dict, obj: dict, orientation: Optional[dict], semantics_config: dict,
) -> dict:
    action = copy.deepcopy(proposal)
    pose = copy.deepcopy(action.get("target_pose_base") or {})
    position = pose.get("position_m", pose.get("position"))
    pose["position_m"] = [float(value) for value in position]
    pose.pop("position", None)
    if orientation is None:
        action["target_pose_base"] = pose
        return action
    candidates = orientation.get("candidate_target_orientations") or []
    if not candidates:
        return {"validation_error": "code_target_orientation_unavailable"}
    target_orientation = list(candidates[0])
    grasp_transform = obj.get("grasp_transform_object_to_tool_xyzw", [0.0, 0.0, 0.0, 1.0])
    pose["orientation_xyzw"] = target_orientation
    pose["orientation_source"] = "house_frame_code_geometry"
    action["target_pose_base"] = pose
    action["orientation_confidence"] = orientation.get("orientation_confidence")
    action["fused_orientation_result"] = copy.deepcopy(orientation)
    action["target_tool_orientation_xyzw"] = quaternion_multiply(target_orientation, grasp_transform)
    observed_orientation = _object_orientation(obj)
    if observed_orientation is not None:
        action["grasp_tool_orientation_xyzw"] = quaternion_multiply(observed_orientation, grasp_transform)
    if action.get("action_type") == "pick_reorient_place":
        current = observed_orientation
        if current is None:
            return {"validation_error": "current_object_orientation_unavailable_reobserve"}
        reorientation = plan_reorientation(
            current, target_orientation,
            grasp_transform,
            get_center(obj), pose["position_m"], semantics_config,
        )
        if not reorientation.get("roll_pitch_component"):
            return {"validation_error": "flip_plan_must_include_roll_or_pitch"}
        action["source_pose_base"] = {"position": list(get_center(obj)), "orientation_xyzw": current}
        action["reorientation_plan"] = reorientation
        action["grasp_target_center_base_m"] = _safe_reorientation_grasp_center(obj, proposal.get("role_id"))
        action["grasp_constraints"] = (
            ["avoid_concave_opening", "avoid_roof_support_legs", "retain_object_during_roll_pitch"]
            if proposal.get("role_id") == "roof"
            else ["avoid_triangle_apex", "prefer_wide_lower_region", "preserve_stable_bottom_edge"]
        )
    return action


def _object_orientation(obj: dict) -> Optional[List[float]]:
    orientation = obj.get("orientation_xyzw") or obj.get("object_orientation_xyzw")
    if isinstance(orientation, (list, tuple)) and len(orientation) == 4:
        return [float(value) for value in orientation]
    if any(obj.get(key) is not None for key in ("roll_rad", "pitch_rad", "yaw_rad")):
        roll, pitch, yaw = (float(obj.get(key, 0.0)) for key in ("roll_rad", "pitch_rad", "yaw_rad"))
        cr, sr, cp, sp, cy, sy = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2)
        return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    return None


def _supports_target(support: dict, upper: dict, tolerance: float, minimum_overlap: float) -> bool:
    overlap = footprint_overlap_area(object_footprint_polygon(support), object_footprint_polygon(upper))
    ratio = overlap / max(min(
        footprint_area(object_footprint_polygon(support)),
        footprint_area(object_footprint_polygon(upper)),
    ), 1e-9)
    return abs(_bottom_z(upper) - _top_z(support)) <= tolerance and ratio >= minimum_overlap


def _safe_reorientation_grasp_center(obj: dict, role_id: str) -> List[float]:
    explicit = obj.get("reorientation_grasp_center_base_m")
    if isinstance(explicit, (list, tuple)) and len(explicit) == 3:
        return [float(value) for value in explicit]
    center, size = get_center(obj), get_size(obj)
    if role_id == "roof":
        yaw = float(obj.get("yaw_rad", 0.0))
        offset = 0.25 * max(float(size[0]), float(size[1]))
        return [center[0] + offset * math.cos(yaw), center[1] + offset * math.sin(yaw), center[2]]
    return [center[0], center[1], center[2] - 0.15 * float(size[2])]


def _object_belongs_to_group(obj: dict, group: dict, grouping_key: str) -> bool:
    matching = group.get("matching_rule") or {}
    value = matching.get(grouping_key, group.get("group_value"))
    return value is not None and object_label_contains(obj, str(value))


def _roles(contract: dict) -> Dict[str, dict]:
    role_specs = contract.get("goal_spec", {}).get("roles") or contract.get("goal_spec", {}).get("required_roles") or []
    return {item.get("role_id"): item for item in role_specs if isinstance(item, dict)}


def _object_by_id(state: dict, object_id: Any) -> Optional[dict]:
    return next((obj for obj in state.get("objects", []) if str(obj.get("id")) == str(object_id)), None)


def _table_z(state: dict) -> float:
    plane = state.get("table_plane") or {}
    if plane.get("z_base_m") is not None:
        return float(plane["z_base_m"])
    bottoms = [_bottom_z(obj) for obj in state.get("objects", []) if get_center(obj) and get_size(obj)]
    return min(bottoms) if bottoms else 0.0


def _top_z(obj: dict) -> float:
    return float(get_center(obj)[2]) + 0.5 * float(get_size(obj)[2])


def _bottom_z(obj: dict) -> float:
    return float(get_center(obj)[2]) - 0.5 * float(get_size(obj)[2])


def _report(stage: str, proposal: dict, reason: str, fields: Optional[List[str]] = None) -> dict:
    return {
        "validation_stage": stage,
        "passed": False,
        "accepted": False,
        "reason": reason,
        "failed_fields": list(fields or []),
        "decision": proposal,
        "selected_object_id": proposal.get("selected_object_id"),
    }


def _fail(report: dict, reason: str, field: str) -> dict:
    report["reason"] = reason
    report.setdefault("failed_fields", []).append(field)
    report.setdefault("checks", {}).setdefault(field, {"detail": {"reason": reason}})
    return report


def _predicate(predicate: str, reason: str) -> dict:
    return {"predicate": predicate, "reason": reason}
