"""Task-aware validation for one VLM-selected pick_place action."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from .geometry_relations import get_center, object_xy_aabb, xy_aabb_overlap
from .task_semantic_validation import object_matches_role


def validate_task_action(
    proposal: dict, state: dict, task_contract: dict, grounded_plan: dict, goal_progress: dict,
) -> Tuple[Optional[dict], dict]:
    """Validate current object binding and target pose without choosing replacements."""
    report = {"accepted": False, "checks": {}, "failed_fields": [], "decision": proposal}
    action_type = str(proposal.get("action_type") or "").lower()
    if action_type in {"reobserve", "stop"}:
        report["reason"] = "policy_requested_{}".format(action_type)
        return None, report
    if action_type != "pick_place":
        report["reason"] = "unsupported_task_action_type"; report["failed_fields"].append("action_type")
        return None, report
    expected_revision = state.get("scene_revision")
    if int(proposal.get("scene_revision", -1)) != int(expected_revision):
        return None, _fail(report, "scene_revision", "stale_scene_revision", {"expected": expected_revision, "actual": proposal.get("scene_revision")})
    role_id = proposal.get("role_id"); role = _roles(task_contract).get(role_id)
    object_id = proposal.get("selected_object_id", proposal.get("object_id")); obj = _object_by_id(state, object_id)
    if role is None or obj is None:
        return None, _fail(report, "role_or_object", "unknown_role_or_object", {"role_id": role_id, "object_id": object_id})
    if not object_matches_role(obj, role):
        return None, _fail(report, "role_match", "selected_object_no_longer_matches_role", {"role_id": role_id, "object_id": object_id})
    if not _grounded(obj, proposal):
        return None, _fail(report, "object_grounding", "object_binding_grounding_mismatch", {"object_id": object_id})
    pose = proposal.get("target_pose_base") or {}; position = pose.get("position_m")
    if not _valid_pose(position, pose.get("yaw_rad")):
        return None, _fail(report, "target_pose", "invalid_target_pose_base", {})
    if not _inside_workspace(position, state.get("table_bounds") or state.get("workspace_bounds")):
        return None, _fail(report, "workspace", "target_pose_outside_workspace", {"position_m": position})
    if _target_hits_protected(obj, position, role_id, grounded_plan, goal_progress, state):
        return None, _fail(report, "protected_structure", "target_pose_collides_with_protected_role", {"role_id": role_id})
    action = {
        "action_type": "pick_place", "object_id": obj.get("id"), "selected_object_id": obj.get("id"),
        "role_id": role_id, "scene_revision": int(expected_revision), "target_pose_base": copy.deepcopy(pose),
        "expected_goal_predicates": list(proposal.get("expected_goal_predicates") or []),
        "reason": proposal.get("reason"), "confidence": proposal.get("confidence"), "selection_source": "vlm_task_action_policy",
    }
    report["accepted"] = True; report["reason"] = "accepted_pending_moveit_preflight"
    return action, report


def _roles(contract: dict) -> Dict[str, dict]:
    role_specs = contract.get("goal_spec", {}).get("roles") or contract.get("goal_spec", {}).get("required_roles") or []
    return {item.get("role_id"): item for item in role_specs if isinstance(item, dict)}
def _object_by_id(state: dict, object_id: Any) -> Optional[dict]:
    return next((obj for obj in state.get("objects", []) if str(obj.get("id")) == str(object_id)), None)
def _grounded(obj: dict, proposal: dict) -> bool:
    if str(proposal.get("object_label") or "").lower() != str(obj.get("label") or "").lower(): return False
    proposed, actual = proposal.get("object_center_base_m"), get_center(obj)
    return isinstance(proposed, list) and actual is not None and len(proposed) >= 3 and math.dist(proposed[:3], actual[:3]) <= 0.005
def _valid_pose(position: Any, yaw: Any) -> bool:
    try: return isinstance(position, list) and len(position) == 3 and all(math.isfinite(float(item)) for item in position) and math.isfinite(float(yaw))
    except (TypeError, ValueError): return False
def _inside_workspace(position: list, bounds: Optional[dict]) -> bool:
    if not bounds: return True
    try: return float(bounds["xmin"]) <= position[0] <= float(bounds["xmax"]) and float(bounds["ymin"]) <= position[1] <= float(bounds["ymax"])
    except (KeyError, TypeError, ValueError): return False
def _target_hits_protected(obj: dict, position: list, action_role_id: str, plan: dict, progress: dict, state: dict) -> bool:
    observations = progress.get("role_observations") or {}
    protected_ids = {str(item.get("observed_object_id")) for item in observations.values() if isinstance(item, dict)}
    allowed_contacts = set()
    if action_role_id == "roof":
        allowed_contacts = {
            str(observations.get(role, {}).get("observed_object_id"))
            for role in ("left_support", "right_support")
        }
    moved = copy.deepcopy(obj); moved["geometry_center_m"] = list(position); moved_aabb = object_xy_aabb(moved)
    if not moved_aabb: return True
    for other in state.get("objects", []):
        if str(other.get("id")) in protected_ids and str(other.get("id")) not in allowed_contacts and str(other.get("id")) != str(obj.get("id")):
            other_aabb = object_xy_aabb(other)
            if other_aabb and xy_aabb_overlap(moved_aabb, other_aabb)[2] > 0: return True
    return False
def _fail(report: dict, field: str, reason: str, detail: dict) -> dict:
    report["reason"] = reason; report["failed_fields"].append(field); report["checks"][field] = {"ok": False, "detail": detail}; return report
