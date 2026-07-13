"""Shared semantic workflow for build_house and organize_blocks tasks."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Tuple

from robot_scene_pipeline.task_action_adapter import TaskActionSchemaError, adapt_task_action_to_legacy_action
from robot_scene_pipeline.task_action_validation import validate_task_action
from robot_scene_pipeline.task_dynamic_protection import derive_dynamic_protection
from robot_scene_pipeline.task_goal_evaluator import evaluate_task_goal_progress
from robot_scene_pipeline.orientation_fusion import fuse_house_orientation_observations
from robot_scene_pipeline.orientation_assets import prepare_orientation_candidate_assets
from robot_scene_pipeline.task_semantic_validation import validate_grounded_task_plan, validate_task_contract
from robot_scene_pipeline.vlm_replanning import action_validation_feedback, advance_scene_revision
from robot_scene_pipeline.action_fingerprint import build_replanning_context, ledger_entry, normalize_action_fingerprint
from robot_scene_pipeline.object_tracking import resolve_action_references
from robot_scene_pipeline.task_routing import route_task_type
from robot_scene_pipeline.vlm_task_policy import build_grounded_task_plan_input, build_task_action_input, build_task_contract_input, call_vlm_task_policy
from robot_scene_pipeline.scene_memory import default_memory, save_memory, update_from_detections
from robot_scene_pipeline.vlm_action_validation import validate_vlm_action_decision
from tools.planning.decision_to_execution import write_json

from .commands import capture_scene_observation, init_ready_pose, load_json
from .execution_safety import validate_execution_source
from .task_execution import execute_pick_place_and_reobserve, preflight_pick_place_action
from .clearance_execution import (
    execute_nudge_and_reobserve,
    execute_pick_away_and_reobserve,
    preflight_nudge_action,
    preflight_pick_away_action,
)
from .push_clearing import object_by_string_id
from .workspace import attach_configured_workspace


def run_semantic_task_workflow(args: Any) -> int:
    """Run contract -> temporary binding -> predicate progress -> action loop."""
    validate_execution_source(args)
    os.makedirs(args.output_dir, exist_ok=True)
    config = _load_semantics_config(args)
    runtime = {"current_stage": "semantic_task_startup", "scene_revision": 1}
    try:
        if getattr(args, "memory_json", None) is None:
            args.memory_json = os.path.join(args.output_dir, "scene_memory.json")
        init_ready_pose(args)
        state = _initial_state(args)
        state["scene_revision"] = runtime["scene_revision"]
        contract = _obtain_contract(args, state, config)
        memory = update_from_detections(default_memory(task=contract["task_type"]), state.get("objects", []), scene_revision=state["scene_revision"])
        save_memory(memory, args.memory_json)
        write_json(os.path.join(args.output_dir, "task_contract_validated.json"), contract)
        previous_progress = None
        previous_plan = None
        pending_post_place_log = None
        for step_index in range(1, max(1, int(getattr(args, "max_task_steps", 12))) + 1):
            cycle_dir = os.path.join(args.output_dir, "task_cycle_{:02d}".format(step_index)); os.makedirs(cycle_dir, exist_ok=True)
            state = prepare_orientation_candidate_assets(state, os.path.join(cycle_dir, "orientation_assets"))
            plan = _obtain_grounded_plan(
                args, state, contract, config, runtime["scene_revision"], cycle_dir,
                previous_plan, previous_progress,
            )
            if pending_post_place_log:
                prior_execution = load_json(pending_post_place_log)
                prior_execution["post_place_orientation_result"] = plan.get("fused_orientation_results", [])
                prior_execution["post_place_scene_revision"] = state.get("scene_revision")
                write_json(pending_post_place_log, prior_execution)
                pending_post_place_log = None
            progress = evaluate_task_goal_progress(state, contract, plan, config, previous_progress)
            _update_role_binding_history(
                memory,
                progress.get("selected_role_assignment"),
                runtime["scene_revision"],
            )
            protection = derive_dynamic_protection(
                state, contract, progress, progress.get("selected_role_assignment"),
            )
            write_json(os.path.join(args.output_dir, "task_goal_progress_revision_{:02d}.json".format(runtime["scene_revision"])), progress)
            write_json(os.path.join(cycle_dir, "role_assignment_candidates.json"), progress.get("role_assignment_candidates", []))
            write_json(os.path.join(cycle_dir, "selected_role_assignment.json"), progress.get("selected_role_assignment"))
            write_json(os.path.join(cycle_dir, "dynamic_protection.json"), protection)
            write_json(os.path.join(cycle_dir, "orientation_fusion.json"), plan.get("fused_orientation_results", []))
            write_json(os.path.join(cycle_dir, "track_assignment.json"), memory.get("last_track_assignment", []))
            write_json(os.path.join(args.output_dir, "track_history.json"), memory.get("track_history", []))
            write_json(os.path.join(args.output_dir, "role_binding_history.json"), memory.get("role_binding_history", []))
            if progress["task_complete"]:
                write_json(os.path.join(args.output_dir, "task_demo_summary.json"), {"execution_status": "success", "task_contract": contract, "task_goal_progress": progress})
                return 0
            action, action_report = _select_task_action(
                args, state, contract, plan, progress, protection, config, cycle_dir, step_index,
            )
            action_history = {
                "task_type": contract["task_type"],
                "scene_revision": runtime["scene_revision"],
                "targeted_unsatisfied_predicates": list(progress["unsatisfied_predicates"]),
                "predicates_before": list(progress["satisfied_predicates"]),
                "selected_action": action or action_report,
                "predicates_after": [],
            }
            write_json(os.path.join(cycle_dir, "autonomous_action_history.json"), action_history)
            action_type = action_report.get("action_type")
            if action_type == "reobserve":
                previous_progress = progress
                previous_plan = plan
                state = _reobserve(args, cycle_dir, runtime)
                memory = update_from_detections(memory, state.get("objects", []), scene_revision=state["scene_revision"]); save_memory(memory, args.memory_json)
                continue
            if action_type == "stop":
                raise RuntimeError("VLM requested safe stop: {}".format(action_report.get("reason") or "unspecified"))
            if action is None:
                raise RuntimeError("No valid VLM task action after replanning.")
            checked = action
            write_json(os.path.join(cycle_dir, "task_action_preflight.json"), checked)
            state, result = _execute_task_action(args, cycle_dir, runtime, memory, state, checked, step_index)
            execution_log_path = os.path.join(cycle_dir, "task_action_execution.json")
            write_json(execution_log_path, result)
            if checked.get("action_type") == "pick_reorient_place" and getattr(args, "execute", False):
                pending_post_place_log = execution_log_path
            if not getattr(args, "execute", False):
                plan_only = bool(getattr(args, "moveit_plan_only", False))
                write_json(os.path.join(args.output_dir, "task_demo_summary.json"), {
                    "execution_status": "planned_only" if plan_only else "dry_run_only",
                    "task_contract": contract,
                    "task_goal_progress": progress,
                    "selected_action": checked,
                    "moveit_feasible": bool(checked.get("moveit_feasible")) if plan_only else None,
                    "gripper_enabled": False,
                })
                return 0
            advance_scene_revision(runtime, state)
            memory = update_from_detections(memory, state.get("objects", []), scene_revision=state["scene_revision"]); save_memory(memory, args.memory_json)
            predicates_after = evaluate_task_goal_progress(state, contract, plan, config, progress)
            action_history["predicates_after"] = list(predicates_after["satisfied_predicates"])
            action_history["execution_result"] = result
            write_json(os.path.join(cycle_dir, "autonomous_action_history.json"), action_history)
            previous_progress = progress
            previous_plan = plan
        raise RuntimeError("Task did not satisfy goal before max_task_steps.")
    except Exception as exc:
        write_json(os.path.join(args.output_dir, "failure_state.json"), {**runtime, "error": str(exc)})
        return 1


def _obtain_contract(args: Any, state: dict, config: dict) -> dict:
    expected_task_type = route_task_type(args.instruction)
    print('Task route:\ninstruction="{}"\nexpected_task_type={}\nschema={}'.format(
        args.instruction,
        expected_task_type,
        "BUILD_HOUSE_CONTRACT_SCHEMA" if expected_task_type == "build_house" else "ORGANIZE_BLOCKS_CONTRACT_SCHEMA",
    ), flush=True)
    if expected_task_type == "stack_blocks":
        raise RuntimeError("STACK_TASK_REQUIRES_LINEAR_STACK_WORKFLOW")
    if getattr(args, "task_contract_json", ""):
        return validate_task_contract(
            load_json(args.task_contract_json), config,
            expected_task_type=expected_task_type,
        )
    history: List[dict] = []
    for attempt in range(1, max(1, int(getattr(args, "max_vlm_task_plan_attempts", 5))) + 1):
        policy_input = build_task_contract_input(
            state, args.instruction, config,
            expected_task_type=expected_task_type,
        )
        policy_input["failure_history"] = history
        raw = call_vlm_task_policy(args, policy_input, "task_contract", artifact_dir=args.output_dir)
        prefix = "task_contract_attempt_{:02d}".format(attempt)
        write_json(os.path.join(args.output_dir, "{}_input.json".format(prefix)), policy_input)
        write_json(os.path.join(args.output_dir, "{}_raw.json".format(prefix)), raw)
        if raw.get("call_status") != "parsed":
            _raise_policy_generation_failure(raw)
        try:
            contract = validate_task_contract(
                raw.get("decision"), config,
                expected_task_type=expected_task_type,
            )
        except Exception as exc:
            feedback = getattr(exc, "feedback", None) or {"validation_stage": "task_semantic_validation", "passed": False, "errors": [{"type": "task_contract_invalid", "message": str(exc)}]}
            history.append(feedback); write_json(os.path.join(args.output_dir, "{}_validation.json".format(prefix)), feedback); continue
        write_json(os.path.join(args.output_dir, "{}_validation.json".format(prefix)), {"passed": True})
        write_json(os.path.join(args.output_dir, "task_contract_input.json"), policy_input)
        write_json(os.path.join(args.output_dir, "task_contract_raw.json"), raw)
        return contract
    raise RuntimeError("No valid task_contract after replanning.")


def _obtain_grounded_plan(
    args: Any, state: dict, contract: dict, config: dict, revision: int, cycle_dir: str,
    previous_plan: Optional[dict] = None, goal_progress: Optional[dict] = None,
) -> dict:
    if getattr(args, "grounded_task_plan_json", ""):
        raw = load_json(args.grounded_task_plan_json)
        validated = validate_grounded_task_plan(raw, contract, state, revision, config)
        return _fuse_plan_orientation(validated, state, contract, config, previous_plan)
    history: List[dict] = []
    for attempt in range(1, max(1, int(getattr(args, "max_vlm_task_plan_attempts", 5))) + 1):
        policy_input = build_grounded_task_plan_input(
            state, contract, revision, config, goal_progress=goal_progress,
            previous_plan=previous_plan, failure_history=history,
        )
        raw = call_vlm_task_policy(args, policy_input, "grounded_task_plan", artifact_dir=cycle_dir)
        prefix = "grounded_task_plan_attempt_{:02d}".format(attempt)
        write_json(os.path.join(cycle_dir, "{}_input.json".format(prefix)), policy_input)
        write_json(os.path.join(cycle_dir, "{}_raw.json".format(prefix)), raw)
        if raw.get("call_status") != "parsed":
            _raise_policy_generation_failure(raw)
        try:
            plan = validate_grounded_task_plan(raw.get("decision"), contract, state, revision, config)
            plan = _fuse_plan_orientation(plan, state, contract, config, previous_plan)
        except Exception as exc:
            feedback = getattr(exc, "feedback", None) or {"validation_stage": "task_semantic_validation", "passed": False, "errors": [{"type": "grounded_plan_invalid", "message": str(exc)}]}
            history.append(feedback); write_json(os.path.join(cycle_dir, "{}_validation.json".format(prefix)), feedback); continue
        write_json(os.path.join(cycle_dir, "{}_validation.json".format(prefix)), {"passed": True})
        write_json(os.path.join(args.output_dir, "grounded_task_plan_input.json"), policy_input)
        write_json(os.path.join(args.output_dir, "grounded_task_plan_raw.json"), raw)
        write_json(os.path.join(args.output_dir, "grounded_task_plan_validated.json"), plan)
        return plan
    raise RuntimeError("No valid grounded_task_plan after replanning.")


def _fuse_plan_orientation(
    plan: dict, state: dict, contract: dict, config: dict, previous_plan: Optional[dict],
) -> dict:
    if contract.get("task_type") != "build_house":
        return plan
    return fuse_house_orientation_observations(
        plan, state, config, (previous_plan or {}).get("fused_orientation_results"),
    )


def _select_task_action(
    args: Any, state: dict, contract: dict, plan: dict, progress: dict,
    protection: dict, semantics_config: dict, cycle_dir: str, step_index: int,
) -> Tuple[Optional[dict], dict]:
    if state.get("table_bounds") is None and state.get("workspace_bounds") is None:
        raise RuntimeError("WORKSPACE_CONFIGURATION_MISSING")
    history = []
    ledger = []
    max_attempts = max(1, int(getattr(args, "max_vlm_action_attempts", 5)))
    for attempt in range(1, max_attempts + 1):
        replanning = build_replanning_context(ledger, attempt, max_attempts)
        policy_input = build_task_action_input(
            state, contract, plan, progress, state["scene_revision"], history,
            replanning_context=replanning,
        )
        raw = call_vlm_task_policy(args, policy_input, "task_action", artifact_dir=cycle_dir)
        if raw.get("call_status") != "parsed" or not isinstance(raw.get("decision"), dict):
            _raise_policy_generation_failure(raw)
        proposal = {
            key: value for key, value in (raw.get("decision") or {}).items()
            if value is not None
        }
        raw["decision"] = proposal
        action_type = str(proposal.get("action_type") or "").lower()
        print("Action Attempt {}/{}".format(attempt, max_attempts), flush=True)
        preflight_state = state
        fingerprint = None
        try:
            if action_type not in {"reobserve", "stop"}:
                proposal = _resolve_task_action_reference(proposal, state)
                raw["decision"] = proposal
                fingerprint = normalize_action_fingerprint(proposal, state, state["scene_revision"])
                constraints = replanning["hard_constraints"]
                if fingerprint["fingerprint"] in set(constraints["forbidden_action_fingerprints"]):
                    raise TaskActionSchemaError("duplicate_failed_action", proposal)
                if action_type in set(constraints["forbidden_action_types"]):
                    raise TaskActionSchemaError("repeated_action_type_forbidden", proposal)
                prior_strategies = {item.get("strategy_id") for item in replanning["failed_actions"]}
                if constraints["required_strategy_change"] and proposal.get("strategy_id") in prior_strategies:
                    raise TaskActionSchemaError("strategy_change_required", proposal)
                semantic_error = _task_nudge_semantic_error(proposal, progress)
                if semantic_error:
                    raise TaskActionSchemaError(semantic_error, proposal)
            if action_type in {"nudge", "pick_away"}:
                adapted = adapt_task_action_to_legacy_action(proposal)
                protected_state, protected_ids = _state_with_dynamic_protection(state, protection)
                preflight_state = protected_state
                action, report = validate_vlm_action_decision(adapted, protected_state, protected_ids=protected_ids)
                if action is not None:
                    action["selected_object_id"] = proposal["selected_object_id"]
            else:
                action, report = validate_task_action(
                    proposal, state, contract, plan, progress, protection, semantics_config,
                )
        except TaskActionSchemaError as exc:
            action, report = None, exc.feedback
        write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_input.json".format(attempt)), policy_input)
        write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_output.json".format(attempt)), raw)
        write_json(os.path.join(cycle_dir, "task_action_semantic_validation.json"), report)
        if action_type == "reobserve":
            write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_validation.json".format(attempt)), report); return None, proposal
        if action_type == "stop" and replanning["hard_constraints"]["safe_stop_allowed"]:
            write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_validation.json".format(attempt)), report); return None, proposal
        if action_type == "stop":
            report = {"accepted": False, "validation_stage": "graded_replanning", "reason": "safe_stop_not_yet_allowed"}
        if action is not None:
            legacy_action = (
                action if action_type in {"nudge", "pick_away"}
                else adapt_task_action_to_legacy_action(action)
            )
            checked = _preflight_task_action(args, cycle_dir, preflight_state, legacy_action, step_index)
            plan_only = bool(getattr(args, "moveit_plan_only", False))
            if (not getattr(args, "execute", False) and not plan_only) or checked.get("moveit_feasible"):
                write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_validation.json".format(attempt)), report)
                return checked, proposal
            report = {
                **report,
                "passed": False,
                "accepted": False,
                "validation_stage": "moveit_preflight",
                "reason": "moveit_preflight_failed",
                "moveit_preflight_error": checked.get("moveit_preflight_error"),
            }
        feedback = action_validation_feedback(proposal, report, state["scene_revision"], attempt); history.append(feedback)
        correctable_protocol_reasons = {
            "object_binding_grounding_mismatch", "task_action_json_schema_invalid",
            "invalid_target_pose_base",
        }
        if str(report.get("reason")) not in correctable_protocol_reasons:
            fingerprint = fingerprint or normalize_action_fingerprint(proposal, state, state["scene_revision"])
            ledger.append(ledger_entry(
                attempt, fingerprint, proposal, str(feedback.get("validation_stage") or "semantic"),
                [str(report.get("reason") or "proposal_rejected")], state["scene_revision"],
            ))
        write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_validation.json".format(attempt)), feedback)
        write_json(os.path.join(cycle_dir, "failure_ledger.json"), ledger)
        write_json(os.path.join(cycle_dir, "replanning_context.json"), build_replanning_context(ledger, min(attempt + 1, max_attempts), max_attempts))
        write_json(os.path.join(cycle_dir, "action_fingerprint.json"), fingerprint)
    unique_fingerprints = {item["fingerprint"] for item in ledger}
    unique_strategies = {item["strategy_id"] for item in ledger}
    if len(unique_fingerprints) < 2 or len(unique_strategies) < 2:
        return None, {"action_type": "reobserve", "reason": "reobserve_after_nondiverse_replanning"}
    return None, {"action_type": "stop", "reason": "no_valid_vlm_action_after_replanning"}


def _resolve_task_action_reference(proposal: dict, state: dict) -> dict:
    reference_action = dict(proposal)
    reference_action["object_ref"] = proposal.get("selected_object_ref")
    reference_action["object_track_id"] = proposal.get("selected_track_id")
    reference_action["object_id"] = proposal.get("selected_object_id")
    resolved, error = resolve_action_references(reference_action, state, int(state["scene_revision"]))
    if error:
        raise TaskActionSchemaError(error["reason"], proposal)
    output = dict(proposal)
    output["selected_object_id"] = resolved["object_id"]
    output["selected_object_ref"] = resolved["object_ref"]
    if resolved.get("object_track_id"):
        output["selected_track_id"] = resolved["object_track_id"]
    else:
        output.pop("selected_track_id", None)
    output.setdefault("strategy_id", "{}_{}".format(proposal.get("action_type", "action"), proposal.get("role_id") or proposal.get("group_id") or "task"))
    return output


def _task_nudge_semantic_error(proposal: dict, progress: dict) -> Optional[str]:
    if str(proposal.get("action_type")) != "nudge":
        return None
    text = " ".join(str(proposal.get(key) or "") for key in ("reason", "predicted_scene_benefit")).lower()
    if any(token in text for token in ("上方", "堆叠", "on_top", "on top", "stack")):
        return "nudge_cannot_satisfy_vertical_stack_relation"
    if proposal.get("role_id") and any("on_top" in str(value) for value in progress.get("unsatisfied_predicates", [])):
        return "nudge_cannot_satisfy_vertical_stack_relation"
    return None


def _state_with_dynamic_protection(state: dict, protection: dict) -> Tuple[dict, List[Any]]:
    """Expose id-backed and region-only protection to existing physical validators."""
    protected_state = {**state, "objects": list(state.get("objects", []))}
    protected_ids: List[Any] = list(protection.get("protected_object_ids") or [])
    for index, region in enumerate(protection.get("protected_regions") or []):
        if not region.get("center_base_m") or not region.get("dimensions_m"):
            continue
        synthetic_id = "dynamic_protected_region_{}".format(index)
        protected_state["objects"].append({
            "id": synthetic_id,
            "label": "protected task structure region",
            "geometry_center_m": list(region["center_base_m"]),
            "dimensions_m": list(region["dimensions_m"]),
            "yaw_rad": float(region.get("yaw_rad", 0.0)),
            "state": "protected",
        })
        protected_ids.append(synthetic_id)
    return protected_state, protected_ids


def _initial_state(args: Any) -> dict:
    if getattr(args, "offline_scene_state", ""):
        return attach_configured_workspace(load_json(args.offline_scene_state), args)
    initial_dir = os.path.join(args.output_dir, "initial_task_scene"); capture_scene_observation(args, initial_dir)
    return attach_configured_workspace(
        load_json(os.path.join(initial_dir, "private_scene_state.json")), args,
    )


def _reobserve(args: Any, cycle_dir: str, runtime: dict) -> dict:
    if getattr(args, "offline_scene_state", ""):
        raise RuntimeError(
            "OFFLINE_REOBSERVE_UNAVAILABLE: offline_scene_state is a fixed replay; "
            "rerun with live perception or provide a newer offline scene"
        )
    observation_dir = os.path.join(cycle_dir, "observation_after_vlm_reobserve"); capture_scene_observation(args, observation_dir)
    state = attach_configured_workspace(
        load_json(os.path.join(observation_dir, "private_scene_state.json")), args,
    )
    advance_scene_revision(runtime, state)
    return state


def _load_semantics_config(args: Any) -> dict:
    path = getattr(args, "task_semantics_config", "config/task_semantics.json")
    with open(path, "r", encoding="utf-8") as handle: return json.load(handle)


def _raise_policy_generation_failure(raw: dict) -> None:
    error_type = str(raw.get("error_type") or raw.get("call_status") or "UNKNOWN")
    if error_type in {"TOKEN_BUDGET_EXHAUSTED", "budget_exhausted"}:
        raise RuntimeError("TOKEN_BUDGET_EXHAUSTED")
    if error_type in {"FINALIZATION_FAILED", "finalization_failed"}:
        raise RuntimeError("FINALIZATION_FAILED")
    raise RuntimeError("VLM_BACKEND_FAILED: {}".format(error_type))


def _update_role_binding_history(memory: dict, assignment: Any, scene_revision: int) -> None:
    if not isinstance(assignment, dict):
        return
    history = memory.setdefault("role_binding_history", [])
    for role_id, obj in assignment.items():
        if not isinstance(obj, dict) or not obj.get("track_id"):
            continue
        previous = next((item for item in reversed(history) if item.get("role_id") == role_id), None)
        track_id = str(obj["track_id"])
        object_ref = obj.get("object_ref")
        if previous and previous.get("track_id") == track_id and previous.get("object_ref") == object_ref:
            continue
        history.append({
            "event": "role_rebound" if previous and previous.get("track_id") != track_id else "role_bound",
            "role_id": role_id,
            "track_id": track_id,
            "previous_track_id": None if previous is None else previous.get("track_id"),
            "object_ref": object_ref,
            "scene_revision": int(scene_revision),
        })


def _preflight_task_action(args: Any, cycle_dir: str, state: dict, action: dict, step_index: int) -> dict:
    if action["action_type"] in {"pick_place", "pick_reorient_place"}:
        return preflight_pick_place_action(args, cycle_dir, state, action, step_index)
    if action["action_type"] == "nudge":
        return preflight_nudge_action(args, cycle_dir, state, action, step_index)
    return preflight_pick_away_action(args, cycle_dir, state, action, step_index)


def _execute_task_action(args: Any, cycle_dir: str, runtime: dict, memory: dict, state: dict, action: dict, step_index: int) -> Tuple[dict, dict]:
    if action["action_type"] in {"pick_place", "pick_reorient_place"}:
        return execute_pick_place_and_reobserve(args, cycle_dir, runtime, state, action, step_index)
    if getattr(args, "execute", False) and not getattr(args, "execute_push_clearing", False):
        raise RuntimeError(
            "CLEARANCE_EXECUTION_NOT_AUTHORIZED: nudge/pick_away requires --execute-push-clearing"
        )
    target = object_by_string_id(state.get("objects", []), action.get("target_object_id"))
    if action["action_type"] == "nudge":
        next_state, _memory, _held = execute_nudge_and_reobserve(args, cycle_dir, runtime, memory, state, target, target, action, None, {}, None, step_index)
    else:
        next_state, _memory, _held = execute_pick_away_and_reobserve(args, cycle_dir, runtime, memory, state, target, action, None, {}, None, step_index)
    if not getattr(args, "execute", False):
        return next_state, {
            "status": "planned_only" if getattr(args, "moveit_plan_only", False) else "dry_run_only",
            "action_type": action["action_type"],
            "moveit_feasible": bool(action.get("moveit_feasible")),
            "scene_changed": False,
        }
    return next_state, {"status": "executed_and_reobserved", "action_type": action["action_type"]}
