"""Finite VLM proposal/rejection/reproposal loop for one observed scene."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Tuple

from robot_scene_pipeline.vlm_replanning import action_validation_feedback
from tools.planning.decision_to_execution import write_json

from .vlm_action import evaluate_autonomous_vlm_action_attempt


def select_autonomous_vlm_action(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    task_focus_object: dict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: dict,
    step_index: int,
    future_place_regions: Optional[Iterable[dict]] = None,
    scene_revision: int = 1,
) -> Tuple[Optional[dict], dict, dict, dict]:
    """Ask the VLM repeatedly until one proposal passes every safety gate."""
    failure_history = []
    attempts = []
    max_attempts = max(1, int(getattr(args, "max_vlm_action_attempts", 5)))
    for attempt in range(1, max_attempts + 1):
        artifact_index = (max(1, int(step_index)) - 1) * max_attempts + attempt
        selected, action_report, safety_report, preflight_report = evaluate_autonomous_vlm_action_attempt(
            args,
            cycle_dir,
            current_state,
            task_focus_object,
            protected_ids,
            base_id,
            memory,
            step_index,
            future_place_regions=future_place_regions,
            failure_history=failure_history,
            scene_revision=scene_revision,
            attempt_index=artifact_index,
        )
        proposal = (safety_report.get("decision") or action_report) if isinstance(safety_report, dict) else action_report
        control_action_type = str(
            action_report.get("action_type") or proposal.get("action_type") or ""
        ).strip().lower()
        if control_action_type in {"reobserve", "stop"}:
            attempts.append({
                "attempt": attempt,
                "proposal": proposal,
                "validation": safety_report,
                "result_type": "control_action",
            })
            _write_history(
                cycle_dir,
                scene_revision,
                attempts,
                None,
                {
                    "status": control_action_type,
                    "reason": action_report.get("reason"),
                },
            )
            return None, action_report, safety_report, {
                "run_moveit_preflight": False,
                "control_action": control_action_type,
            }
        if selected is not None and safety_report.get("accepted"):
            attempts.append({"attempt": attempt, "proposal": proposal, "validation": safety_report})
            _write_history(cycle_dir, scene_revision, attempts, selected, None)
            return selected, action_report, safety_report, preflight_report
        if safety_report.get("validation_stage") == "duplicate_detection":
            feedback = safety_report
        else:
            feedback = action_validation_feedback(proposal, safety_report, scene_revision, attempt)
        failure_history.append(feedback)
        attempts.append({"attempt": attempt, "proposal": proposal, "validation": feedback})
        write_json(os.path.join(cycle_dir, "vlm_action_attempt_{:02d}_validation.json".format(artifact_index)), feedback)
    stop_report = {
        "selection_status": "fail_safe_stop",
        "action_type": "stop",
        "reason": "no_valid_vlm_action_after_replanning",
        "failure_history": failure_history,
    }
    safety = {
        "accepted": False,
        "reason": "no_valid_vlm_action_after_replanning",
        "failed_fields": ["max_vlm_action_attempts"],
        "checks": {},
        "failure_history": failure_history,
    }
    _write_history(cycle_dir, scene_revision, attempts, None, {"status": "fail_safe_stop"})
    return None, stop_report, safety, {"run_moveit_preflight": False, "failures": failure_history}


def mark_autonomous_execution_result(cycle_dir: str, execution_result: dict) -> None:
    """Attach execution/reobservation outcome to the scene's complete action history."""
    path = os.path.join(cycle_dir, "autonomous_action_history.json")
    try:
        import json
        with open(path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
    except (OSError, ValueError):
        history = {"scene_revision": None, "attempts": [], "selected_action": None}
    history["execution_result"] = execution_result
    scene_histories = history.get("scene_histories") or []
    if scene_histories:
        scene_histories[-1]["execution_result"] = execution_result
        history["scene_histories"] = scene_histories
    write_json(path, history)


def _write_history(
    cycle_dir: str,
    scene_revision: int,
    attempts: list,
    selected_action: Optional[dict],
    execution_result: Optional[dict],
) -> None:
    path = os.path.join(cycle_dir, "autonomous_action_history.json")
    try:
        import json
        with open(path, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        previous = {}
    scene_entry = {
        "scene_revision": int(scene_revision),
        "attempts": attempts,
        "selected_action": selected_action,
        "execution_result": execution_result or {},
    }
    scene_histories = list(previous.get("scene_histories") or [])
    if scene_histories and scene_histories[-1].get("scene_revision") == int(scene_revision):
        scene_histories[-1] = scene_entry
    else:
        scene_histories.append(scene_entry)
    write_json(path, {**scene_entry, "scene_histories": scene_histories})
