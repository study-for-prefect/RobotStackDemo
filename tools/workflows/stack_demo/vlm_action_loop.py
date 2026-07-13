"""Finite VLM proposal/rejection/reproposal loop for one observed scene."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Tuple

from robot_scene_pipeline.vlm_replanning import action_validation_feedback
from robot_scene_pipeline.action_fingerprint import (
    build_replanning_context, ledger_entry, normalize_action_fingerprint,
)
from robot_scene_pipeline.object_tracking import resolve_action_references
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
    if current_state.get("table_bounds") is None and current_state.get("workspace_bounds") is None:
        raise RuntimeError("WORKSPACE_CONFIGURATION_MISSING")
    failure_history = []
    failed_action_ledger = []
    fallback_pool = []
    use_fallback_next = False
    attempts = []
    max_attempts = max(1, int(getattr(args, "max_vlm_action_attempts", 5)))
    write_json(os.path.join(cycle_dir, "track_assignment.json"), memory.get("last_track_assignment", []))
    write_json(os.path.join(cycle_dir, "track_history.json"), memory.get("track_history", []))
    write_json(os.path.join(cycle_dir, "role_binding_history.json"), memory.get("role_binding_history", []))
    for attempt in range(1, max_attempts + 1):
        replanning_context = build_replanning_context(failed_action_ledger, attempt, max_attempts)
        artifact_index = (max(1, int(step_index)) - 1) * max_attempts + attempt
        forced_decision = None
        if use_fallback_next:
            forced_decision = _take_untried_vlm_fallback(
                fallback_pool, current_state, task_focus_object,
                scene_revision, replanning_context,
            )
            use_fallback_next = False
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
            replanning_context=replanning_context,
            forced_decision=forced_decision,
        )
        if preflight_report.get("backend_failure"):
            error_type = preflight_report.get("error_type") or "UNKNOWN"
            if error_type in {"TOKEN_BUDGET_EXHAUSTED", "budget_exhausted"}:
                raise RuntimeError("TOKEN_BUDGET_EXHAUSTED")
            if error_type in {"FINALIZATION_FAILED", "finalization_failed"}:
                raise RuntimeError("FINALIZATION_FAILED")
            raise RuntimeError("VLM_BACKEND_FAILED: {}".format(error_type))
        print("Action Attempt {}/{} forbidden_fingerprints={} required_strategy_change={}".format(
            attempt, max_attempts,
            len(replanning_context["hard_constraints"]["forbidden_action_fingerprints"]),
            replanning_context["hard_constraints"]["required_strategy_change"],
        ), flush=True)
        proposal = (safety_report.get("decision") or action_report) if isinstance(safety_report, dict) else action_report
        for candidate in proposal.get("alternative_actions") or []:
            if isinstance(candidate, dict):
                fallback_pool.append(candidate)
        control_action_type = str(
            action_report.get("action_type") or proposal.get("action_type") or ""
        ).strip().lower()
        is_explicit_safe_stop = control_action_type == "stop"
        if is_explicit_safe_stop and not replanning_context["hard_constraints"]["safe_stop_allowed"]:
            safety_report = {"accepted": False, "validation_stage": "graded_replanning", "reason": "premature_safe_stop", "failed_fields": ["action_type"], "checks": {}, "decision": proposal}
            control_action_type = ""
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
            write_json(os.path.join(cycle_dir, "failure_ledger.json"), failed_action_ledger)
            write_json(os.path.join(cycle_dir, "replanning_context.json"), replanning_context)
            return None, action_report, safety_report, {
                "run_moveit_preflight": False,
                "control_action": control_action_type,
            }
        if selected is not None and safety_report.get("accepted"):
            fingerprint = safety_report.get("normalized_action_fingerprint") or normalize_action_fingerprint(proposal, current_state, scene_revision)
            print("strategy={} action={} operated={} fingerprint={} result=accepted".format(
                fingerprint.get("strategy_id"), fingerprint.get("action_type"),
                fingerprint.get("operated_track_id"), fingerprint.get("fingerprint"),
            ), flush=True)
            attempts.append({"attempt": attempt, "proposal": proposal, "validation": safety_report})
            _write_history(cycle_dir, scene_revision, attempts, selected, None)
            write_json(os.path.join(cycle_dir, "failure_ledger.json"), failed_action_ledger)
            write_json(os.path.join(cycle_dir, "replanning_context.json"), replanning_context)
            return selected, action_report, safety_report, preflight_report
        if safety_report.get("validation_stage") == "duplicate_detection":
            feedback = safety_report
        else:
            feedback = action_validation_feedback(proposal, safety_report, scene_revision, attempt)
        failure_history.append(feedback)
        fingerprint = safety_report.get("normalized_action_fingerprint") or normalize_action_fingerprint(proposal, current_state, scene_revision)
        reason_codes = [item.get("type") for item in feedback.get("failed_checks", []) if item.get("type")]
        if not reason_codes: reason_codes = [str(safety_report.get("reason") or "proposal_rejected")]
        failed_action_ledger.append(ledger_entry(
            attempt, fingerprint, proposal, str(feedback.get("validation_stage") or "semantic"),
            reason_codes, scene_revision,
        ))
        use_fallback_next = safety_report.get("reason") in {
            "duplicate_failed_action", "selected_object_reference_not_found",
            "selected_track_not_visible", "stale_or_invalid_object_ref",
            "legacy_object_id_requires_scene_revision", "strategy_change_required",
            "premature_safe_stop",
        }
        print("strategy={} action={} operated={} fingerprint={} result={}".format(
            fingerprint.get("strategy_id"), fingerprint.get("action_type"),
            fingerprint.get("operated_track_id"), fingerprint.get("fingerprint"),
            safety_report.get("reason") or "rejected",
        ), flush=True)
        attempts.append({"attempt": attempt, "proposal": proposal, "validation": feedback})
        write_json(os.path.join(cycle_dir, "vlm_action_attempt_{:02d}_validation.json".format(artifact_index)), feedback)
        write_json(os.path.join(cycle_dir, "failure_ledger.json"), failed_action_ledger)
        write_json(os.path.join(cycle_dir, "replanning_context.json"), build_replanning_context(failed_action_ledger, min(attempt + 1, max_attempts), max_attempts))
        write_json(os.path.join(cycle_dir, "action_fingerprint.json"), fingerprint)
    unique_fingerprints = {item.get("fingerprint") for item in failed_action_ledger if item.get("fingerprint")}
    unique_strategies = {item.get("strategy_id") for item in failed_action_ledger if item.get("strategy_id")}
    if len(unique_fingerprints) < 2 or len(unique_strategies) < 2:
        reobserve_report = {
            "selection_status": "reobserve_required",
            "action_type": "reobserve",
            "reason": "reobserve_after_nondiverse_replanning",
            "failure_history": failure_history,
        }
        safety = {
            "accepted": False,
            "reason": "insufficient_unique_actions_for_safe_stop",
            "failed_fields": ["unique_fingerprints", "unique_strategies"],
            "checks": {},
            "failure_history": failure_history,
        }
        _write_history(cycle_dir, scene_revision, attempts, None, {"status": "reobserve_required"})
        return None, reobserve_report, safety, {
            "run_moveit_preflight": False,
            "control_action": "reobserve",
            "failures": failure_history,
        }
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


def _take_untried_vlm_fallback(
    pool: list, state: dict, task_focus: dict, scene_revision: int, context: dict,
) -> Optional[dict]:
    forbidden = set(context["hard_constraints"]["forbidden_action_fingerprints"])
    candidates = []
    while pool:
        candidate = pool.pop(0)
        resolved, error = resolve_action_references(candidate, state, scene_revision)
        if error or _fallback_semantic_error(resolved, task_focus):
            continue
        fingerprint = normalize_action_fingerprint(resolved, state, scene_revision)
        if fingerprint["fingerprint"] in forbidden:
            continue
        candidates.append((float(resolved.get("confidence", 0.0)), resolved))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    pool.extend(item[1] for item in candidates[1:])
    return candidates[0][1]


def _fallback_semantic_error(action: dict, task_focus: dict) -> bool:
    if str(action.get("action_type")) != "nudge":
        return False
    same_object = (
        str(action.get("object_id")) == str(task_focus.get("id"))
        or action.get("object_track_id") == task_focus.get("track_id")
    )
    text = " ".join(str(action.get(key) or "") for key in ("reason", "predicted_scene_benefit")).lower()
    return bool(same_object and any(token in text for token in ("上方", "堆叠", "on_top", "on top", "stack")))


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
