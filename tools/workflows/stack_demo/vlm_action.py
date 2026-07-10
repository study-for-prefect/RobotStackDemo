"""Workflow glue for VLM action-intent decisions."""

from __future__ import annotations

import os
import shutil
from typing import Any, Iterable, Optional, Tuple

from robot_scene_pipeline.vlm_action_policy import (
    build_vlm_action_decision_input,
    call_vlm_action_policy,
    mark_moveit_result,
    validate_vlm_action_decision,
)
from robot_scene_pipeline.vlm_replanning import (
    duplicate_proposal_feedback,
    find_duplicate_failed_proposal,
)
from tools.planning.decision_to_execution import write_json

from .clearance_execution import preflight_nudge_action, preflight_pick_away_action
from .pick_preflight import preflight_pick_action


def evaluate_autonomous_vlm_action_attempt(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    task_focus_object: dict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: dict,
    step_index: int,
    future_place_regions: Optional[Iterable[dict]] = None,
    failure_history: Optional[list] = None,
    scene_revision: int = 1,
    attempt_index: int = 1,
) -> Tuple[Optional[dict], dict, dict, dict]:
    """Evaluate one VLM-proposed action without choosing a replacement in code."""
    scene_rgb_path, depth_path, minimal_overlay_path = materialize_vlm_images(cycle_dir, current_state)
    policy_input = build_vlm_action_decision_input(
        scene_rgb_path,
        minimal_overlay_path,
        current_state,
        task_focus_object,
        protected_ids,
        base_id,
        memory,
        step_index,
        manipulator_geometry={
            "closed_gripper_outer_width_m": float(getattr(args, "grasp_gripper_outer_width_m", 0.112)),
            "finger_length_m": float(getattr(args, "push_tool_finger_length_m", 0.12)),
            "safety_margin_m": float(getattr(args, "push_tool_safety_margin_m", 0.005)),
            "contact_z_offset_m": float(getattr(args, "push_clearing_contact_z_offset_m", 0.015)),
            "push_tool_yaw_offset_deg": float(getattr(args, "push_tool_yaw_offset_deg", 0.0)),
            "contact_rule": (
                "The VLM selects contact_side and gripper_yaw_rad. The gripper approaches vertically, "
                "then moves along direction_base. The full tool corridor must be clear."
            ),
            "tool0_to_tcp_offset_m": list(getattr(args, "tcp_offset_tool", [0.0, 0.0, 0.0])),
            "tool_depth_m": float(getattr(args, "push_tool_depth_m", 0.04)),
            "fingertip_thickness_m": float(getattr(args, "push_tool_fingertip_thickness_m", 0.01)),
        },
        failure_history=failure_history,
        scene_revision=scene_revision,
        depth_visualization_path=depth_path,
    )
    raw_output = call_vlm_action_policy(args, policy_input)
    decision = raw_output.get("decision") or {}
    if find_duplicate_failed_proposal(decision, scene_revision, failure_history or []):
        safety_report = duplicate_proposal_feedback(decision, scene_revision, attempt_index)
        validated_output = _validated_output(raw_output, None, safety_report)
        write_vlm_action_artifacts(
            cycle_dir, policy_input, raw_output, validated_output, safety_report, attempt_index,
        )
        return None, validated_output, safety_report, {"run_moveit_preflight": False, "failures": []}
    selected, safety_report = validate_vlm_action_decision(
        decision,
        current_state,
        protected_ids,
        grasp_options={
            "gripper_outer_width_m": getattr(args, "grasp_gripper_outer_width_m", 0.112),
            "gripper_inner_width_m": getattr(args, "grasp_gripper_inner_width_m", 0.048),
            "side_clearance_m": getattr(args, "grasp_gripper_side_clearance_m", 0.006),
            "approach_length_m": getattr(args, "grasp_approach_length_m", 0.02),
        },
        protection_margin_m=float(getattr(args, "push_tool_safety_margin_m", 0.005)),
        future_place_regions=future_place_regions or [],
    )
    preflight_report = {
        "schema_version": "vlm_action_moveit_preflight_v1",
        "run_moveit_preflight": False,
        "failures": [],
    }
    run_clearance_preflight = bool(
        getattr(args, "execute", False)
        and getattr(args, "execute_push_clearing", False)
    )
    if selected is not None and selected.get("action_type") in ("nudge", "pick_away"):
        preflight_report["run_moveit_preflight"] = run_clearance_preflight
        if run_clearance_preflight:
            checked = (
                preflight_nudge_action(args, cycle_dir, current_state, selected, step_index)
                if selected.get("action_type") == "nudge"
                else preflight_pick_away_action(args, cycle_dir, current_state, selected, step_index)
            )
            selected, safety_report = _apply_preflight_result(checked, decision, safety_report, preflight_report)
        else:
            selected["moveit_feasible"] = False
            selected["executable_safe"] = False
            safety_report["reason"] = "accepted_without_moveit_preflight_dry_run"
    elif selected is not None and selected.get("action_type") == "pick":
        run_pick_preflight = bool(getattr(args, "execute", False))
        preflight_report["run_moveit_preflight"] = run_pick_preflight
        if run_pick_preflight:
            checked = preflight_pick_action(args, cycle_dir, current_state, selected, step_index)
            selected, safety_report = _apply_preflight_result(checked, decision, safety_report, preflight_report)
        else:
            selected["moveit_feasible"] = False
            selected["executable_safe"] = False
            safety_report["reason"] = "accepted_without_moveit_preflight_dry_run"

    validated_output = _validated_output(raw_output, selected, safety_report)
    write_vlm_action_artifacts(
        cycle_dir, policy_input, raw_output, validated_output, safety_report, attempt_index,
    )
    return selected if safety_report.get("accepted") else None, validated_output, safety_report, preflight_report


def write_vlm_action_artifacts(
    cycle_dir: str,
    policy_input: dict,
    raw_output: dict,
    validated_output: dict,
    safety_report: dict,
    attempt_index: int = 1,
) -> None:
    prefix = "vlm_action_attempt_{:02d}".format(attempt_index)
    write_json(os.path.join(cycle_dir, "{}_input.json".format(prefix)), policy_input)
    write_json(os.path.join(cycle_dir, "{}_output.json".format(prefix)), raw_output)
    write_json(os.path.join(cycle_dir, "{}_validation.json".format(prefix)), safety_report)
    if attempt_index == 1:
        write_json(os.path.join(cycle_dir, "vlm_action_decision_input.json"), policy_input)
        write_json(os.path.join(cycle_dir, "vlm_action_decision_raw.json"), raw_output)
    write_json(os.path.join(cycle_dir, "vlm_action_decision_validated.json"), validated_output)
    write_json(os.path.join(cycle_dir, "vlm_action_safety_report.json"), safety_report)


def materialize_vlm_images(
    cycle_dir: str, current_state: dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    scene_path = _resolve_image_path(
        current_state.get("snapshot_image"),
        cycle_dir,
        fallback_names=("snapshot.jpg", "scene_rgb.png"),
    )
    overlay_path = _resolve_image_path(
        current_state.get("annotated_image"),
        cycle_dir,
        fallback_names=("annotated_detector.jpg", "minimal_overlay.png"),
    )
    scene_output = _copy_image_if_available(scene_path, os.path.join(cycle_dir, "scene_rgb.png"))
    overlay_output = _copy_image_if_available(overlay_path, os.path.join(cycle_dir, "minimal_overlay.png"))
    depth_path = _resolve_image_path(
        current_state.get("depth_visualization") or current_state.get("depth_visualization_image"),
        cycle_dir,
        fallback_names=("depth_visualization.png", "depth_colormap.png"),
    )
    depth_output = _copy_image_if_available(depth_path, os.path.join(cycle_dir, "depth_visualization.png"))
    return scene_output, depth_output, overlay_output


def _apply_preflight_result(
    checked: dict,
    decision: dict,
    safety_report: dict,
    preflight_report: dict,
) -> Tuple[Optional[dict], dict]:
    selected = checked if checked.get("moveit_feasible") else None
    preflight_report["action"] = checked
    if checked.get("moveit_feasible"):
        checked["executable_safe"] = True
        passed = mark_moveit_result(checked, safety_report, True)
        tool_report = checked.get("tool_swept_volume_report")
        if tool_report is not None:
            passed["tool_swept_volume_report"] = tool_report
            passed.setdefault("checks", {})["tool_swept_volume_clear"] = {
                "ok": bool(tool_report.get("feasible")),
                "detail": tool_report,
            }
        return selected, passed
    preflight_report["failures"].append(
        {
            "action_type": checked.get("action_type"),
            "object_id": decision.get("object_id"),
            "stage": checked.get("preflight_failure_stage") or "moveit",
            "reason": checked.get("moveit_preflight_error") or "moveit_preflight_failed",
        }
    )
    if checked.get("preflight_failure_stage") == "tool_swept_volume":
        failed = dict(safety_report)
        failed["accepted"] = False
        failed["moveit_feasible"] = None
        failed["reason"] = "tool_swept_volume_rejected"
        failed["tool_swept_volume_report"] = checked.get("tool_swept_volume_report")
        failed.setdefault("checks", {})["tool_swept_volume_clear"] = {
            "ok": False,
            "detail": checked.get("tool_swept_volume_report") or {},
        }
        if "tool_swept_volume_clear" not in failed.setdefault("failed_fields", []):
            failed["failed_fields"].append("tool_swept_volume_clear")
        return None, failed
    return selected, mark_moveit_result(
        checked,
        safety_report,
        False,
        error=checked.get("moveit_preflight_error"),
    )


def _validated_output(raw_output: dict, selected: Optional[dict], safety_report: dict) -> dict:
    decision = raw_output.get("decision") or {}
    accepted = bool(safety_report.get("accepted"))
    return {
        "schema_version": "vlm_action_decision_validated_v1",
        "selection_source": "vlm_action_policy",
        "selection_status": "selected" if accepted else "fail_safe_stop",
        "scene_problem": decision.get("scene_problem"),
        "action_type": decision.get("action_type", "stop"),
        "object_id": decision.get("object_id"),
        "object_label": decision.get("object_label"),
        "object_center_base_m": decision.get("object_center_base_m"),
        "target_object_id": decision.get("target_object_id"),
        "target_object_label": decision.get("target_object_label"),
        "target_object_center_base_m": decision.get("target_object_center_base_m"),
        "contact_side": decision.get("contact_side"),
        "direction_base": decision.get("direction_base", decision.get("push_direction_base")),
        "distance_m": decision.get("distance_m", decision.get("push_distance_m")),
        "gripper_yaw_rad": decision.get("gripper_yaw_rad"),
        "safe_place_center_base_m": decision.get("safe_place_center_base_m"),
        "predicted_scene_benefit": decision.get("predicted_scene_benefit"),
        "risk_assessment": decision.get("risk_assessment"),
        "reason": decision.get("reason") if accepted else safety_report.get("reason"),
        "vlm_reason": decision.get("reason"),
        "confidence": decision.get("confidence", 0.0),
        "selected_action": selected,
        "safety_report": safety_report,
    }


def _copy_image_if_available(source_path: Optional[str], output_path: str) -> Optional[str]:
    if source_path is None or not os.path.isfile(source_path):
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.abspath(source_path) != os.path.abspath(output_path):
        shutil.copyfile(source_path, output_path)
    return output_path


def _resolve_image_path(path: Any, cycle_dir: str, fallback_names: Tuple[str, ...]) -> Optional[str]:
    if isinstance(path, str) and os.path.isfile(path):
        return path
    search_dirs = [
        cycle_dir,
        os.path.dirname(cycle_dir),
        os.path.join(os.path.dirname(cycle_dir), "initial_order"),
    ]
    if isinstance(path, str) and path:
        fallback_names = (os.path.basename(path),) + tuple(fallback_names)
    for directory in search_dirs:
        for name in fallback_names:
            image_path = os.path.join(directory, name)
            if os.path.isfile(image_path):
                return image_path
    return None
