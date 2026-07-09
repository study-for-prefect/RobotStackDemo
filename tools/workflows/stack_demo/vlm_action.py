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
from tools.planning.decision_to_execution import write_json

from .clearance_execution import preflight_nudge_action, preflight_pick_away_action


def select_clearance_with_vlm_action_policy(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    held_object: dict,
    analysis: dict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: dict,
    step_index: int,
    future_place_regions: Optional[Iterable[dict]] = None,
) -> Tuple[Optional[dict], dict, dict, dict]:
    """Select a VLM-proposed action intent after code safety validation."""
    scene_rgb_path, minimal_overlay_path = materialize_vlm_images(cycle_dir, current_state)
    policy_input = build_vlm_action_decision_input(
        scene_rgb_path,
        minimal_overlay_path,
        current_state,
        held_object,
        analysis,
        protected_ids,
        base_id,
        memory,
        step_index,
    )
    raw_output = call_vlm_action_policy(args, policy_input)
    decision = raw_output.get("decision") or {}
    selected, safety_report = validate_vlm_action_decision(
        decision,
        current_state,
        held_object,
        protected_ids,
        analysis=analysis,
        protection_margin_m=float(getattr(args, "push_tool_safety_margin_m", 0.005)),
        future_place_regions=future_place_regions or [],
    )
    preflight_report = {
        "schema_version": "vlm_action_moveit_preflight_v1",
        "run_moveit_preflight": False,
        "failures": [],
    }
    run_moveit_preflight = bool(
        getattr(args, "execute", False)
        and getattr(args, "execute_push_clearing", False)
    )
    if selected is not None and selected.get("action_type") in ("nudge", "pick_away"):
        preflight_report["run_moveit_preflight"] = run_moveit_preflight
        if run_moveit_preflight:
            checked = (
                preflight_nudge_action(args, cycle_dir, current_state, held_object, selected, step_index)
                if selected.get("action_type") == "nudge"
                else preflight_pick_away_action(args, cycle_dir, current_state, selected, step_index)
            )
            selected, safety_report = _apply_preflight_result(checked, decision, safety_report, preflight_report)
        else:
            selected["moveit_feasible"] = False
            selected["executable_safe"] = False
            safety_report["reason"] = "accepted_without_moveit_preflight_dry_run"
    elif selected is not None and selected.get("action_type") == "pick":
        safety_report = mark_moveit_result(selected, safety_report, True)

    validated_output = _validated_output(raw_output, selected, safety_report)
    write_vlm_action_artifacts(cycle_dir, policy_input, raw_output, validated_output, safety_report)
    return selected if safety_report.get("accepted") else None, validated_output, safety_report, preflight_report


def write_vlm_action_artifacts(
    cycle_dir: str,
    policy_input: dict,
    raw_output: dict,
    validated_output: dict,
    safety_report: dict,
) -> None:
    write_json(os.path.join(cycle_dir, "vlm_action_decision_input.json"), policy_input)
    write_json(os.path.join(cycle_dir, "vlm_action_decision_raw.json"), raw_output)
    write_json(os.path.join(cycle_dir, "vlm_action_decision_validated.json"), validated_output)
    write_json(os.path.join(cycle_dir, "vlm_action_safety_report.json"), safety_report)


def materialize_vlm_images(cycle_dir: str, current_state: dict) -> Tuple[Optional[str], Optional[str]]:
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
    return scene_output, overlay_output


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
        return selected, mark_moveit_result(checked, safety_report, True)
    preflight_report["failures"].append(
        {
            "action_type": checked.get("action_type"),
            "object_id": decision.get("object_id"),
            "reason": checked.get("moveit_preflight_error") or "moveit_preflight_failed",
        }
    )
    return selected, mark_moveit_result(
        checked,
        safety_report,
        False,
        error=checked.get("moveit_preflight_error"),
    )


def _validated_output(raw_output: dict, selected: Optional[dict], safety_report: dict) -> dict:
    decision = raw_output.get("decision") or {}
    return {
        "schema_version": "vlm_action_decision_validated_v1",
        "selection_source": "vlm_action_policy",
        "selection_status": "selected" if safety_report.get("accepted") else "fail_safe_stop",
        "action_type": decision.get("action_type", "stop"),
        "object_id": decision.get("object_id"),
        "target_object_id": decision.get("target_object_id"),
        "push_direction_base": decision.get("push_direction_base"),
        "push_distance_m": decision.get("push_distance_m"),
        "safe_place_center_base_m": decision.get("safe_place_center_base_m"),
        "reason": decision.get("reason") or safety_report.get("reason"),
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
