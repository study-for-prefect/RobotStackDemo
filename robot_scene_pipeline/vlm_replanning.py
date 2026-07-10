"""Structured feedback and duplicate detection for VLM replanning loops."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional


ACTION_CHANGE_FIELDS = [
    "object_id",
    "action_type",
    "direction_base",
    "distance_m",
    "contact_side",
    "gripper_yaw_rad",
]


class StackSemanticValidationError(ValueError):
    """A stack proposal that can be corrected by asking the VLM again."""

    def __init__(self, message: str, feedback: Dict[str, Any]):
        super().__init__(message)
        self.feedback = feedback


class VlmReplanningExhausted(RuntimeError):
    """A bounded VLM loop exhausted with all proposal feedback preserved."""

    def __init__(self, message: str, failure_history: List[Dict[str, Any]]):
        super().__init__(message)
        self.failure_history = failure_history


def action_fingerprint(proposal: Dict[str, Any], scene_revision: int) -> Dict[str, Any]:
    """Return a tolerance-bucketed fingerprint for one VLM action proposal."""
    direction = proposal.get("direction_base") or proposal.get("push_direction_base")
    distance = proposal.get("distance_m", proposal.get("push_distance_m"))
    yaw = proposal.get("gripper_yaw_rad")
    return {
        "scene_revision": int(scene_revision),
        "object_id": proposal.get("object_id"),
        "action_type": proposal.get("action_type"),
        "direction_sector": _direction_sector(direction),
        "distance_range_m": _distance_range(distance),
        "contact_side": proposal.get("contact_side"),
        "gripper_yaw_sector": _yaw_sector(yaw),
    }


def find_duplicate_failed_proposal(
    proposal: Dict[str, Any],
    scene_revision: int,
    failure_history: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find an approximately identical failed proposal in the same scene revision."""
    fingerprint = action_fingerprint(proposal, scene_revision)
    for failure in failure_history:
        previous = failure.get("action_fingerprint") if isinstance(failure, dict) else None
        if isinstance(previous, dict) and previous == fingerprint:
            return failure
    return None


def duplicate_proposal_feedback(
    proposal: Dict[str, Any], scene_revision: int, attempt: int,
) -> Dict[str, Any]:
    """Build feedback that requires a materially different next action."""
    return {
        "proposal_id": "attempt_{}".format(attempt),
        "scene_revision": int(scene_revision),
        "validation_stage": "duplicate_detection",
        "passed": False,
        "hard_failure": False,
        "failed_checks": [{"type": "duplicate_failed_proposal"}],
        "rejected_action": compact_action(proposal),
        "action_fingerprint": action_fingerprint(proposal, scene_revision),
        "constraints_for_next_proposal": {
            "must_change_at_least_one": list(ACTION_CHANGE_FIELDS),
        },
    }


def action_validation_feedback(
    proposal: Dict[str, Any],
    safety_report: Dict[str, Any],
    scene_revision: int,
    attempt: int,
) -> Dict[str, Any]:
    """Convert safety and MoveIt reports into a VLM-facing validation result."""
    failed_checks: List[Dict[str, Any]] = []
    for name in safety_report.get("failed_fields", []):
        check = (safety_report.get("checks") or {}).get(name) or {}
        detail = copy.deepcopy(check.get("detail") or {})
        if name == "tool_swept_volume_clear":
            failed_checks.append({"type": name, "reason": detail.get("reason")})
        else:
            failed_checks.append({"type": name, **detail})
    tool_report = safety_report.get("tool_swept_volume_report") or {}
    for collision in tool_report.get("hard_collisions", tool_report.get("collisions", [])):
        failed_checks.append({
            "type": "tool_swept_volume_collision",
            "phase": collision.get("stage"),
            "colliding_entity_type": collision.get("entity_type", "scene_object"),
            "colliding_object_id": collision.get("id"),
            "colliding_object_label": collision.get("label"),
            "minimum_clearance_m": collision.get("minimum_clearance_m"),
        })
    if not failed_checks:
        failed_checks.append({"type": safety_report.get("reason") or "proposal_rejected"})
    stage = _validation_stage(safety_report)
    feedback = {
        "proposal_id": "attempt_{}".format(attempt),
        "scene_revision": int(scene_revision),
        "validation_stage": stage,
        "passed": False,
        "hard_failure": stage in ("geometry_validation", "moveit_validation"),
        "failed_checks": failed_checks,
        "rejected_action": compact_action(proposal),
        "action_fingerprint": action_fingerprint(proposal, scene_revision),
        "constraints_for_next_proposal": {
            "must_change_at_least_one": list(ACTION_CHANGE_FIELDS),
        },
    }
    recoverable = tool_report.get("recoverable_contacts") or safety_report.get("recoverable_contacts")
    if recoverable:
        feedback["recoverable_contacts"] = copy.deepcopy(recoverable)
    return feedback


def compact_action(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Return only VLM-controlled action fields for logs and feedback."""
    return {
        key: proposal.get(key)
        for key in (
            "action_type", "object_id", "target_object_id", "contact_side",
            "direction_base", "push_direction_base", "distance_m", "push_distance_m",
            "gripper_yaw_rad", "safe_place_center_base_m", "reason",
        )
        if proposal.get(key) is not None
    }


def advance_scene_revision(runtime: Dict[str, Any], observed_state: Dict[str, Any]) -> int:
    """Increment the revision only after a new observation has been produced."""
    revision = int(runtime.get("scene_revision", observed_state.get("scene_revision", 0))) + 1
    runtime["scene_revision"] = revision
    observed_state["scene_revision"] = revision
    return revision


def stack_feedback(
    errors: List[Dict[str, Any]],
    instruction: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Build stack semantic feedback without choosing replacement object ids."""
    from .llm_stack_blocks import color_mentions, object_label_contains

    required_colors = color_mentions(instruction or "")
    objects = [
        {"id": obj.get("id"), "label": obj.get("label")}
        for obj in state.get("objects", [])
        if isinstance(obj, dict) and not obj.get("is_workspace")
    ]
    required_labels = []
    for color in required_colors:
        required_labels.append(next(
            (
                str(obj.get("label"))
                for obj in state.get("objects", [])
                if isinstance(obj, dict) and object_label_contains(obj, color)
            ),
            color,
        ))
    return {
        "validation_stage": "stack_semantic_validation",
        "passed": False,
        "errors": errors,
        "required_label_order": required_labels,
        "detected_objects": objects,
    }


def _direction_sector(value: Any) -> Optional[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        angle = math.atan2(float(value[1]), float(value[0]))
    except (TypeError, ValueError):
        return None
    return int(round(angle / (math.pi / 8.0))) % 16


def _distance_range(value: Any) -> Optional[List[float]]:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return None
    lower = math.floor(distance / 0.01 + 1e-9) * 0.01
    return [round(lower, 3), round(lower + 0.01, 3)]


def _yaw_sector(value: Any) -> Optional[int]:
    try:
        yaw = float(value)
    except (TypeError, ValueError):
        return None
    return int(round(yaw / (math.pi / 12.0))) % 24


def _validation_stage(safety_report: Dict[str, Any]) -> str:
    if safety_report.get("moveit_feasible") is False:
        return "moveit_validation"
    if safety_report.get("tool_swept_volume_report") is not None:
        return "geometry_validation"
    return "action_semantic_validation"
