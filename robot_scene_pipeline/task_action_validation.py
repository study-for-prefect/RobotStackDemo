"""Task-semantic validation for one VLM-selected pick/place action."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

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
)
from .task_goal_evaluator import evaluate_layout
from .task_semantic_validation import object_matches_role


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
    if action_type != "pick_place":
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
        return validate_organize_pick_place(proposal, state, task_contract, grounded_plan)
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
    grounding_failure = _grounding_failure(obj, proposal, report)
    if grounding_failure:
        return None, grounding_failure
    moved = _object_at_pose(obj, proposal["target_pose_base"])
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    failed_predicates: List[dict] = []
    if not object_inside_workspace(moved, workspace):
        failed_predicates.append(_predicate("house.inside_workspace", "target_full_footprint_outside_workspace"))
    semantics = semantics_config.get("house_semantics") or {}
    vertical_tolerance = float(semantics.get("vertical_contact_tolerance_m", 0.006))
    minimum_separation = float(semantics.get("minimum_support_separation_m", 0.025))
    minimum_overlap = float(semantics.get("minimum_support_overlap_ratio", 0.20))
    observations = progress.get("role_observations") or {}
    current = {name: _object_by_id(state, item.get("observed_object_id")) for name, item in observations.items()}
    current[role_id] = moved
    if role_id in {"left_support", "right_support"}:
        other_role = "right_support" if role_id == "left_support" else "left_support"
        other = current.get(other_role)
        if abs(_bottom_z(moved) - _table_z(state)) > vertical_tolerance:
            failed_predicates.append(_predicate("{}.on_table".format(role_id), "target_bottom_not_on_table"))
        if other is not None:
            distance = footprint_boundary_distance(object_footprint_polygon(moved), object_footprint_polygon(other))
            if distance < minimum_separation:
                failed_predicates.append(_predicate("supports.separated", "support_boundary_spacing_too_small"))
            left = moved if role_id == "left_support" else other
            right = other if role_id == "left_support" else moved
            if get_center(left)[0] >= get_center(right)[0]:
                failed_predicates.append(_predicate("left_support.left_of.right_support", "support_side_order_reversed"))
    elif role_id == "roof":
        left, right = current.get("left_support"), current.get("right_support")
        if left is None or right is None:
            failed_predicates.append(_predicate("roof.supported_by_both_supports", "current_supports_unavailable"))
        else:
            for support, name in ((left, "left_support"), (right, "right_support")):
                overlap = footprint_overlap_area(object_footprint_polygon(support), object_footprint_polygon(moved))
                ratio = overlap / max(footprint_area(object_footprint_polygon(support)), 1e-9)
                if ratio < minimum_overlap:
                    failed_predicates.append(_predicate("roof.supported_by_both_supports", "insufficient_overlap_with_{}".format(name)))
                if abs(_bottom_z(moved) - _top_z(support)) > vertical_tolerance:
                    failed_predicates.append(_predicate("{}.supports.roof".format(name), "roof_contact_height_mismatch"))
    failed_predicates.extend(_protected_pose_failures(moved, obj, dynamic_protection, state, role_id))
    if failed_predicates:
        report["failed_predicates"] = failed_predicates
        report["reason"] = failed_predicates[0]["reason"]
        return None, report
    return _accepted_action(proposal, obj, report)


def validate_organize_pick_place(
    proposal: dict,
    state: dict,
    contract: dict,
    grounded_plan: dict,
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
    moved = _object_at_pose(obj, proposal["target_pose_base"])
    if not footprint_inside_region(object_footprint_polygon(moved), regions[region_id]):
        return None, _fail(report, "target_pose_outside_group_region", "target_pose_base")
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
            return None, _fail(report, "target_pose_overlaps_planned_object", "target_pose_base")
        if _object_belongs_to_group(other, group, grouping_key):
            group_members.append(other)
            distance = footprint_boundary_distance(object_footprint_polygon(moved), other_footprint)
            if distance < minimum_spacing:
                return None, _fail(report, "target_pose_group_spacing_too_small", "target_pose_base")
    layout_type = str(contract.get("goal_spec", {}).get("layout_type") or "")
    tolerance = float(contract.get("goal_spec", {}).get("alignment_tolerance_m", 0.012))
    if not evaluate_layout(group_members, layout_type, tolerance):
        return None, _fail(report, "target_pose_violates_group_layout", "target_pose_base")
    return _accepted_action(proposal, obj, report)


def _validate_common(proposal: dict, state: dict) -> Optional[dict]:
    report = _report("action_schema_validation", proposal, "action_schema_invalid")
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
    position = pose.get("position_m")
    try:
        valid = isinstance(position, list) and len(position) == 3 and all(math.isfinite(float(v)) for v in position)
        valid = valid and math.isfinite(float(pose.get("yaw_rad")))
    except (TypeError, ValueError):
        valid = False
    return None if valid else _fail(report, "invalid_target_pose_base", "target_pose_base")


def _accepted_action(proposal: dict, obj: dict, report: dict) -> Tuple[dict, dict]:
    action = copy.deepcopy(proposal)
    action["action_type"] = "pick_place"
    action["selected_object_id"] = obj.get("id")
    action["selection_source"] = "vlm_task_action_policy"
    action.pop("object_id", None)
    report.update({"accepted": True, "passed": True, "reason": "accepted_pending_moveit_preflight"})
    return action, report


def _grounding_failure(obj: dict, proposal: dict, report: dict) -> Optional[dict]:
    if str(proposal.get("object_label") or "").lower() != str(obj.get("label") or "").lower():
        return _fail(report, "object_binding_grounding_mismatch", "object_label")
    observed = get_center(obj)
    proposed = proposal.get("object_center_base_m")
    try:
        valid = isinstance(proposed, list) and len(proposed) == 3 and math.dist(proposed, observed[:3]) <= 0.005
    except (TypeError, ValueError):
        valid = False
    return None if valid else _fail(report, "object_binding_grounding_mismatch", "object_center_base_m")


def _protected_pose_failures(moved: dict, source: dict, protection: dict, state: dict, role_id: str) -> List[dict]:
    allowed = set()
    if role_id == "roof":
        allowed = {item.get("current_object_id") for item in protection.get("protected_roles", []) if item.get("role_id") in {"left_support", "right_support"}}
    protected = {str(value) for value in protection.get("protected_object_ids", []) if value not in allowed}
    for other in state.get("objects", []):
        if str(other.get("id")) == str(source.get("id")) or str(other.get("id")) not in protected:
            continue
        if footprint_overlap(object_footprint_polygon(moved), object_footprint_polygon(other)):
            return [_predicate("protected_structure.collision_free", "target_pose_collides_with_protected_role")]
    return []


def _object_at_pose(obj: dict, pose: dict) -> dict:
    moved = copy.deepcopy(obj)
    moved["geometry_center_m"] = [float(value) for value in pose["position_m"]]
    moved["yaw_rad"] = float(pose["yaw_rad"])
    return moved


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
    return report


def _predicate(predicate: str, reason: str) -> dict:
    return {"predicate": predicate, "reason": reason}
