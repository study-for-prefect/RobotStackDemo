"""Workflow glue for VLM-based clearance selection."""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Optional, Tuple

from robot_scene_pipeline.vlm_clearance_policy import (
    build_objects_for_vlm,
    build_physical_clearance_candidates,
    build_policy_input,
    build_task_state_for_vlm,
    call_vlm_clearance_policy,
    evaluate_final_safety_gate,
    materialize_vlm_images,
    write_vlm_policy_artifacts,
)
from tools.planning.decision_to_execution import write_json

from .clearance_execution import preflight_nudge_candidate, preflight_pick_away_candidate


def dedupe_candidates_by_id(candidates: Iterable[dict]) -> list:
    output = []
    seen = set()
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = candidate.get("candidate_id")
        if candidate_id is None or str(candidate_id) in seen:
            continue
        output.append(candidate)
        seen.add(str(candidate_id))
    return output


def select_clearance_with_vlm_policy(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    held_object: dict,
    analysis: dict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: dict,
    step_index: int,
    candidates: Iterable[dict],
) -> Tuple[Optional[dict], dict, dict, dict]:
    preflight_report = _preflight_physical_candidates_for_vlm(
        args,
        cycle_dir,
        current_state,
        held_object,
        candidates,
        step_index,
    )
    objects_for_vlm = build_objects_for_vlm(current_state)
    task_state_for_vlm = build_task_state_for_vlm(
        current_state,
        held_object,
        analysis,
        protected_ids,
        base_id,
        memory,
        step_index,
    )
    scene_rgb_path, minimal_overlay_path = materialize_vlm_images(cycle_dir, current_state)
    physical_candidates = build_physical_clearance_candidates(preflight_report.get("candidates", []))
    policy_input = build_policy_input(
        scene_rgb_path,
        minimal_overlay_path,
        objects_for_vlm,
        task_state_for_vlm,
        physical_candidates,
    )
    policy_output = call_vlm_clearance_policy(args, policy_input)
    final_safety_gate = evaluate_final_safety_gate(policy_output, physical_candidates)
    write_vlm_policy_artifacts(
        cycle_dir,
        objects_for_vlm,
        task_state_for_vlm,
        physical_candidates,
        policy_input,
        policy_output,
        final_safety_gate,
    )
    selected = None
    if final_safety_gate.get("accepted"):
        selected_id = str(policy_output.get("selected_candidate_id"))
        selected = next(
            (
                candidate for candidate in preflight_report.get("candidates", [])
                if str(candidate.get("candidate_id")) == selected_id
            ),
            None,
        )
    return selected, policy_output, final_safety_gate, preflight_report


def _annotate_physical_gate_fields(candidate: dict) -> dict:
    geometry_ok = bool(candidate.get("geometry_feasible", candidate.get("feasible", False)))
    push_end_safe = bool(candidate.get("push_end_safe", geometry_ok))
    approach_safe = bool(candidate.get("approach_path_safe", geometry_ok))
    swept_safe = bool(candidate.get("push_swept_safe", geometry_ok))
    protected_safe = bool(candidate.get("protected_structure_safe", True))
    action_type = candidate.get("action_type") or candidate.get("action")
    candidate["collision_free"] = bool(push_end_safe and protected_safe)
    candidate["sweep_collision_free"] = bool(approach_safe and swept_safe)
    candidate["workspace_feasible"] = bool(
        geometry_ok
        and push_end_safe
        and candidate.get("future_task_feasible", True)
    )
    candidate["gripper_feasible"] = bool(
        action_type != "pick_away" or not candidate.get("relaxed_pick_away_grasp")
    )
    return candidate


def _pre_moveit_gate(candidate: dict) -> Tuple[bool, list]:
    _annotate_physical_gate_fields(candidate)
    required_fields = (
        "collision_free",
        "sweep_collision_free",
        "workspace_feasible",
        "gripper_feasible",
    )
    failed = [field for field in required_fields if not candidate.get(field)]
    return not failed, failed


def _preflight_physical_candidates_for_vlm(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    held_object: dict,
    candidates: Iterable[dict],
    step_index: int,
) -> dict:
    checked_candidates = []
    failures = []
    run_moveit_preflight = bool(
        getattr(args, "execute", False)
        and getattr(args, "execute_push_clearing", False)
    )
    for raw_candidate in candidates or []:
        if not isinstance(raw_candidate, dict) or raw_candidate.get("candidate_id") is None:
            continue
        candidate = copy.deepcopy(raw_candidate)
        action_type = candidate.get("action_type") or candidate.get("action")
        gate_ok, failed_fields = _pre_moveit_gate(candidate)
        if not gate_ok:
            candidate["moveit_feasible"] = False
            failures.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "action_type": action_type,
                    "reason": "pre_moveit_physical_gate_failed",
                    "failed_fields": failed_fields,
                }
            )
            checked_candidates.append(candidate)
            continue
        if not run_moveit_preflight:
            candidate["moveit_feasible"] = bool(candidate.get("moveit_feasible", False))
            failures.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "action_type": action_type,
                    "reason": "moveit_preflight_not_run_without_execute_push_clearing",
                    "moveit_feasible": candidate.get("moveit_feasible"),
                }
            )
            checked_candidates.append(candidate)
            continue
        if action_type == "nudge":
            candidate = preflight_nudge_candidate(
                args,
                cycle_dir,
                current_state,
                held_object,
                candidate,
                step_index,
            )
        elif action_type == "pick_away":
            candidate = preflight_pick_away_candidate(
                args,
                cycle_dir,
                current_state,
                candidate,
                step_index,
            )
        else:
            candidate["moveit_feasible"] = False
            candidate["moveit_preflight_error"] = "unsupported_clearance_action_type"
        _annotate_physical_gate_fields(candidate)
        if not candidate.get("moveit_feasible"):
            failures.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "action_type": action_type,
                    "reason": candidate.get("moveit_preflight_error") or "moveit_preflight_failed",
                    "moveit_feasible": candidate.get("moveit_feasible"),
                }
            )
        checked_candidates.append(candidate)
    write_json(
        os.path.join(cycle_dir, "clearance_step_{:02d}_moveit_preflight.json".format(step_index)),
        {
            "schema_version": "vlm_physical_moveit_preflight_v1",
            "run_moveit_preflight": run_moveit_preflight,
            "candidate_ids": [candidate.get("candidate_id") for candidate in checked_candidates],
            "moveit_feasible_candidate_ids": [
                candidate.get("candidate_id")
                for candidate in checked_candidates
                if candidate.get("moveit_feasible")
            ],
            "failures": failures,
        },
    )
    return {"candidates": checked_candidates, "failures": failures}

