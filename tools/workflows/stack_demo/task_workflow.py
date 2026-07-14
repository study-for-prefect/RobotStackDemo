"""Shared semantic workflow for build_house and organize_blocks tasks."""

from __future__ import annotations

import json
import math
import os
from typing import Any, List, Optional, Tuple

from robot_scene_pipeline.task_action_adapter import TaskActionSchemaError, adapt_task_action_to_legacy_action
from robot_scene_pipeline.task_action_validation import validate_task_action
from robot_scene_pipeline.grasp_yaw_search import select_best_grasp
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
from robot_scene_pipeline.vlm_perception_review import (
    review_and_recover_expected_misses,
    review_and_recover_low_confidence_candidates,
    review_and_recover_static_misses,
)
from robot_scene_pipeline.detection_merge import merge_duplicate_objects_3d
from robot_scene_pipeline.organize_scope import color_value_from_object
from robot_scene_pipeline.ollama_policy_client import unload_model
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
                state = _reobserve(args, cycle_dir, runtime, previous_state=state)
                memory = update_from_detections(memory, state.get("objects", []), scene_revision=state["scene_revision"]); save_memory(memory, args.memory_json)
                continue
            if action_type == "stop":
                raise RuntimeError("VLM requested safe stop: {}".format(action_report.get("reason") or "unspecified"))
            if action is None:
                raise RuntimeError("No valid VLM task action after replanning.")
            checked = action
            write_json(os.path.join(cycle_dir, "task_action_preflight.json"), checked)
            if (
                getattr(args, "execute", False)
                and getattr(args, "unload_vlm_before_execution", True)
            ):
                runtime["current_stage"] = "unload_vlm_before_physical_execution"
                unload_result = unload_model(args, cycle_dir)
                write_json(
                    os.path.join(cycle_dir, "vlm_unload_before_execution.json"),
                    unload_result.to_dict(),
                )
                if unload_result.transport_status != "ok":
                    raise RuntimeError(
                        "VLM_UNLOAD_BEFORE_EXECUTION_FAILED: {}: {}".format(
                            unload_result.error_type or "unknown",
                            unload_result.error_message or "model was not released",
                        )
                    )
            state_before_action = state
            state, result = _execute_task_action(args, cycle_dir, runtime, memory, state, checked, step_index)
            state = _deduplicate_perception_state(state)
            if getattr(args, "execute", False):
                expected = _expected_state_after_action(state_before_action, checked)
                state, review = review_and_recover_expected_misses(
                    args,
                    expected,
                    state,
                    os.path.join(cycle_dir, "observation_after_action_perception_review"),
                    _known_action_description(checked),
                )
                write_json(os.path.join(cycle_dir, "post_action_vlm_perception_review.json"), review)
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
        decision = raw.get("decision")
        if contract.get("task_type") == "organize_blocks":
            decision, normalization = _normalize_organize_grounded_plan(
                decision, policy_input, revision, previous_plan,
            )
            raw["decision"] = decision
            if normalization:
                raw["protocol_normalization"] = normalization
        try:
            plan = validate_grounded_task_plan(decision, contract, state, revision, config)
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


def _normalize_organize_grounded_plan(
    decision: object, policy_input: dict, revision: int,
    previous_plan: Optional[dict] = None,
) -> Tuple[object, Optional[dict]]:
    """Bind observed colors to the already computed non-overlapping row slots.

    The VLM still supplies the task-level rationale, while current detector IDs,
    resolved visual colors, and code-generated free row bands remain authoritative.
    This prevents a contract color-order hint from creating absent color groups or
    assigning one object to two colors.
    """
    if not isinstance(decision, dict):
        return decision, None
    objects = [item for item in policy_input.get("objects") or [] if isinstance(item, dict)]
    slots = [item for item in policy_input.get("layout_slot_candidates") or [] if isinstance(item, dict)]
    if not objects or not slots:
        return decision, None

    ids_by_color = {}
    for obj in objects:
        color = color_value_from_object(obj)
        object_id = obj.get("detector_object_id", obj.get("id"))
        if color is None or object_id is None:
            continue
        ids_by_color.setdefault(str(color), []).append(object_id)

    previous_regions_by_color = {}
    if isinstance(previous_plan, dict) and previous_plan.get("task_type") == "organize_blocks":
        previous_regions = {
            item.get("region_id"): item.get("bounds_base_m")
            for item in previous_plan.get("target_regions") or []
            if isinstance(item, dict)
        }
        for group in previous_plan.get("groups") or []:
            if not isinstance(group, dict):
                continue
            color = str(group.get("group_value") or "")
            bounds = previous_regions.get(group.get("target_region_id"))
            if color and isinstance(bounds, dict):
                previous_regions_by_color[color] = bounds

    groups = []
    regions = []
    for slot in slots:
        color = str(slot.get("assigned_color") or "")
        object_ids = ids_by_color.get(color) or []
        # A destination layout is a task-level commitment.  Recomputing it
        # after every observation shifts row boundaries by millimetres and can
        # make a just-placed block look outside its row.  Preserve the first
        # valid row bounds for every still-observed color.
        bounds = previous_regions_by_color.get(color) or slot.get("bounds_base_m")
        if not color or not object_ids or not isinstance(bounds, dict):
            continue
        group_id = "group_{}".format(color)
        region_id = "target_region_{}".format(color)
        groups.append({
            "group_id": group_id,
            "group_value": color,
            "object_ids": list(object_ids),
            "target_region_id": region_id,
            "minimum_required_count": len(object_ids),
        })
        regions.append({
            "region_id": region_id,
            "bounds_base_m": {
                key: float(bounds[key]) for key in ("xmin", "xmax", "ymin", "ymax")
            },
        })
    if not groups or sum(len(group["object_ids"]) for group in groups) != len(objects):
        return decision, None

    normalized = {
        "schema_version": "grounded_task_plan_v1",
        "task_type": "organize_blocks",
        "scene_revision": int(revision),
        "groups": groups,
        "target_regions": regions,
    }
    if isinstance(decision.get("reason"), str):
        normalized["reason"] = decision["reason"]
    if isinstance(decision.get("confidence"), (int, float)):
        normalized["confidence"] = float(decision["confidence"])
    if normalized == decision:
        return normalized, None
    return normalized, {
        "type": "authoritative_organize_color_slot_binding",
        "replaced_fields": ["scene_revision", "groups", "target_regions"],
        "preserved_previous_target_regions": bool(previous_regions_by_color),
    }


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
        physical_options = _physical_action_options(args, state, contract, progress)
        policy_input["physical_action_options"] = physical_options
        raw = call_vlm_task_policy(args, policy_input, "task_action", artifact_dir=cycle_dir)
        if raw.get("call_status") != "parsed" or not isinstance(raw.get("decision"), dict):
            _raise_policy_generation_failure(raw)
        proposal = {
            key: value for key, value in (raw.get("decision") or {}).items()
            if value is not None
        }
        original_proposal = dict(proposal)
        proposal, grasp_first_fields = _normalize_organize_grasp_first_proposal(
            proposal, contract.get("task_type"), state, physical_options,
        )
        proposal, normalized_fields = _normalize_task_action_protocol(
            proposal, contract.get("task_type"), state.get("scene_revision"),
        )
        normalized_fields = list(grasp_first_fields) + list(normalized_fields)
        if normalized_fields:
            raw["protocol_normalization"] = {
                "removed_or_corrected_fields": normalized_fields,
                "original_decision": original_proposal,
            }
        raw["decision"] = proposal
        action_type = str(proposal.get("action_type") or "").lower()
        print("Action Attempt {}/{}".format(attempt, max_attempts), flush=True)
        preflight_state = state
        fingerprint = None
        try:
            if action_type not in {"reobserve", "stop"}:
                proposal = _resolve_task_action_reference(proposal, state)
                proposal, group_fields = _normalize_organize_selected_group(
                    proposal, contract.get("task_type"), state, plan,
                )
                if group_fields:
                    normalization = raw.setdefault("protocol_normalization", {
                        "removed_or_corrected_fields": [],
                        "original_decision": original_proposal,
                    })
                    normalization.setdefault("removed_or_corrected_fields", []).extend(
                        key for key in group_fields
                        if key not in normalization["removed_or_corrected_fields"]
                    )
                raw["decision"] = proposal
                fingerprint = normalize_action_fingerprint(proposal, state, state["scene_revision"])
                constraints = replanning["hard_constraints"]
                if fingerprint["fingerprint"] in set(constraints["forbidden_action_fingerprints"]):
                    raise TaskActionSchemaError("duplicate_failed_action", proposal)
                if action_type in set(constraints["forbidden_action_types"]):
                    raise TaskActionSchemaError("repeated_action_type_forbidden", proposal)
                if _repeats_physically_failed_strategy(
                    proposal, fingerprint, replanning,
                ):
                    raise TaskActionSchemaError("strategy_change_required", proposal)
                semantic_error = _task_nudge_semantic_error(
                    proposal, progress, history, plan, physical_options,
                )
                if semantic_error:
                    raise TaskActionSchemaError(semantic_error, proposal)
                priority_error = _task_action_priority_error(
                    proposal, contract.get("task_type"), physical_options,
                )
                if priority_error:
                    raise TaskActionSchemaError(priority_error, proposal)
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
                corrected_action, corrected_proposal, corrected_report = (
                    _retry_geometry_checked_organize_target(
                        proposal, report, state, contract, plan, progress,
                        protection, semantics_config,
                    )
                )
                if corrected_action is not None:
                    corrected_fingerprint = normalize_action_fingerprint(
                        corrected_proposal, state, state["scene_revision"],
                    )
                    if corrected_fingerprint["fingerprint"] in set(
                        constraints["forbidden_action_fingerprints"]
                    ):
                        raise TaskActionSchemaError(
                            "duplicate_failed_action", corrected_proposal,
                        )
                    proposal = corrected_proposal
                    raw["decision"] = proposal
                    raw["geometry_checked_target_correction"] = corrected_report.get(
                        "automatic_target_correction"
                    )
                    fingerprint = corrected_fingerprint
                    action, report = corrected_action, corrected_report
        except TaskActionSchemaError as exc:
            action, report = None, exc.feedback
        write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_input.json".format(attempt)), policy_input)
        write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_output.json".format(attempt)), raw)
        write_json(os.path.join(cycle_dir, "task_action_semantic_validation.json"), report)
        if (
            action is None
            and contract.get("task_type") == "organize_blocks"
            and not physical_options.get("direct_grasp_available")
        ):
            # No current block has a valid grasp angle.  Do not spend eight
            # identical model calls on stop/reobserve/a blocked pick: search a
            # bounded, fully preflighted side push immediately.
            alternative = _automatic_organize_clearance_when_all_grasps_blocked(
                args, cycle_dir, state, plan, progress, protection, step_index,
                history, physical_options,
            )
            if alternative is not None:
                checked, proposal, alternative_report = alternative
                raw["decision"] = proposal
                raw["automatic_clearance_search"] = alternative_report
                write_json(
                    os.path.join(
                        cycle_dir,
                        "task_action_attempt_{:02d}_output.json".format(attempt),
                    ),
                    raw,
                )
                write_json(
                    os.path.join(
                        cycle_dir,
                        "task_action_attempt_{:02d}_validation.json".format(attempt),
                    ),
                    {"passed": True, "accepted": True, **alternative_report},
                )
                return checked, proposal
        if action_type == "reobserve":
            write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_validation.json".format(attempt)), report); return None, proposal
        if (
            action_type == "stop"
            and contract.get("task_type") != "organize_blocks"
            and replanning["hard_constraints"]["safe_stop_allowed"]
        ):
            write_json(os.path.join(cycle_dir, "task_action_attempt_{:02d}_validation.json".format(attempt)), report); return None, proposal
        if action_type == "stop":
            report = {
                "accepted": False,
                "validation_stage": "graded_replanning",
                "reason": (
                    "incomplete_organize_must_reobserve_or_try_another_object"
                    if contract.get("task_type") == "organize_blocks"
                    else "safe_stop_not_yet_allowed"
                ),
            }
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
            physical_grasp = checked.get("physical_grasp_validation") or {}
            grasp_blocked = checked.get("preflight_failure_type") == "selected_pick_not_grasp_feasible"
            tool_sweep_blocked = checked.get("preflight_failure_stage") == "tool_swept_volume"
            report = {**report, "passed": False, "accepted": False}
            if grasp_blocked:
                report.update({
                    "validation_stage": "physical_grasp_preflight",
                    "reason": "selected_pick_not_grasp_feasible",
                    "moveit_feasible": False,
                    "moveit_preflight_error": checked.get("moveit_preflight_error"),
                })
            elif tool_sweep_blocked:
                tool_report = checked.get("tool_swept_volume_report") or {}
                report.update({
                    "validation_stage": "geometry_preflight",
                    "reason": "tool_swept_volume_rejected",
                    "moveit_feasible": None,
                    "moveit_preflight_error": checked.get("moveit_preflight_error"),
                    "tool_swept_volume_report": tool_report,
                })
                if "tool_swept_volume_clear" not in report.setdefault("failed_fields", []):
                    report["failed_fields"].append("tool_swept_volume_clear")
                report.setdefault("checks", {})["tool_swept_volume_clear"] = {
                    "ok": False,
                    "detail": tool_report,
                }
                if contract.get("task_type") == "organize_blocks":
                    alternative = _search_geometry_checked_organize_nudge(
                        args, cycle_dir, state, proposal, plan, progress,
                        protection, step_index, checked, history, physical_options,
                    )
                    if alternative is not None:
                        checked, proposal, alternative_report = alternative
                        raw["decision"] = proposal
                        raw["automatic_clearance_search"] = alternative_report
                        write_json(
                            os.path.join(
                                cycle_dir,
                                "task_action_attempt_{:02d}_output.json".format(attempt),
                            ),
                            raw,
                        )
                        write_json(
                            os.path.join(
                                cycle_dir,
                                "task_action_attempt_{:02d}_validation.json".format(attempt),
                            ),
                            {"passed": True, "accepted": True, **alternative_report},
                        )
                        return checked, proposal
            else:
                report.update({
                    "validation_stage": "moveit_preflight",
                    "reason": "moveit_preflight_failed",
                    "moveit_feasible": False,
                    "moveit_preflight_error": checked.get("moveit_preflight_error"),
                })
            if grasp_blocked:
                report["failed_fields"] = ["selected_object_grasp_feasible"]
                report["checks"] = {
                    **(report.get("checks") or {}),
                    "selected_object_grasp_feasible": {
                        "ok": False,
                        "detail": {
                            **physical_grasp,
                            "required_next_action": (
                                "choose nudge or pick_away clearance, then reobserve before pick_place"
                            ),
                        },
                    },
                }
        feedback = action_validation_feedback(proposal, report, state["scene_revision"], attempt); history.append(feedback)
        correctable_protocol_reasons = {
            "object_binding_grounding_mismatch", "task_action_json_schema_invalid",
            "invalid_target_pose_base", "push_parameters_invalid",
            "organize_clearance_requires_physical_grasp_blockage",
            "organize_clearance_avoid_target_regions",
            "direct_grasp_available_before_clearance",
            "selected_object_not_grasp_feasible_while_alternative_available",
            "duplicate_failed_action",
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
    if contract.get("task_type") == "organize_blocks":
        return None, {"action_type": "reobserve", "reason": "reobserve_incomplete_organize_after_action_replanning"}
    if len(unique_fingerprints) < 2 or len(unique_strategies) < 2:
        return None, {"action_type": "reobserve", "reason": "reobserve_after_nondiverse_replanning"}
    return None, {"action_type": "stop", "reason": "no_valid_vlm_action_after_replanning"}


def _physical_action_options(
    args: Any, state: dict, contract: dict, progress: dict,
) -> dict:
    """Expose code-checked graspability so the VLM does not guess from RGB."""
    outside_ids = None
    if contract.get("task_type") == "organize_blocks":
        outside_ids = {
            str(object_id)
            for group in progress.get("group_diagnostics") or []
            for object_id in group.get("outside_region") or []
        }
    candidates = []
    for obj in state.get("objects", []):
        if not isinstance(obj, dict) or obj.get("id") is None or not obj.get("visible", True):
            continue
        if outside_ids is not None and str(obj.get("id")) not in outside_ids:
            continue
        report = select_best_grasp(
            obj,
            state.get("objects", []),
            gripper_outer_width_m=float(getattr(args, "grasp_gripper_outer_width_m", 0.112)),
            gripper_inner_width_m=float(getattr(args, "grasp_gripper_inner_width_m", 0.049)),
            side_clearance_m=float(getattr(args, "grasp_gripper_side_clearance_m", 0.006)),
            approach_length_m=float(getattr(args, "grasp_approach_length_m", 0.02)),
            min_feasible_yaw_span_deg=float(
                getattr(args, "grasp_min_feasible_yaw_span_deg", 10.0)
            ),
        )
        feasible_span = sum(
            max(0.0, float(interval[1]) - float(interval[0]))
            for interval in report.get("feasible_yaw_intervals_deg") or []
            if isinstance(interval, (list, tuple)) and len(interval) >= 2
        )
        candidates.append({
            "selected_object_id": obj.get("id"),
            "selected_object_ref": obj.get("object_ref"),
            "selected_track_id": obj.get("track_id"),
            "label": obj.get("label"),
            "visual_color": color_value_from_object(obj),
            "grasp_feasible": bool(report.get("grasp_feasible")),
            "feasible_yaw_span_deg": round(feasible_span, 3),
            "selected_grasp_yaw_deg": report.get("selected_grasp_yaw_deg"),
            "blocking_object_ids": [
                blocker.get("id")
                for blocker in report.get("blocking_objects") or []
                if isinstance(blocker, dict) and blocker.get("id") is not None
            ],
        })
    feasible = sorted(
        (item for item in candidates if item["grasp_feasible"]),
        key=lambda item: -float(item.get("feasible_yaw_span_deg") or 0.0),
    )
    return {
        "priority_rule": "grasp_feasible_first_then_clearance",
        "candidates": candidates,
        "direct_grasp_available": bool(feasible),
        "direct_grasp_object_refs": [item.get("selected_object_ref") for item in feasible],
        "direct_grasp_track_ids": [item.get("selected_track_id") for item in feasible],
        "recommended_direct_grasp_object_ref": (
            feasible[0].get("selected_object_ref") if feasible else None
        ),
        "recommendation_basis": "largest_feasible_yaw_span_deg",
    }


def _task_action_priority_error(
    proposal: dict, task_type: object, physical_options: dict,
) -> Optional[str]:
    """For organize, moving any graspable outside block is task progress."""
    if str(task_type or "") != "organize_blocks":
        return None
    feasible = [
        item for item in physical_options.get("candidates") or []
        if item.get("grasp_feasible")
    ]
    if not feasible:
        return None
    if proposal.get("action_type") != "pick_place":
        return "direct_grasp_available_before_clearance"
    selected_id = proposal.get("selected_object_id")
    selected_ref = proposal.get("selected_object_ref")
    selected_track = proposal.get("selected_track_id")
    if not any(
        str(item.get("selected_object_id")) == str(selected_id)
        or (selected_ref and item.get("selected_object_ref") == selected_ref)
        or (selected_track and item.get("selected_track_id") == selected_track)
        for item in feasible
    ):
        return "selected_object_not_grasp_feasible_while_alternative_available"
    return None


def _normalize_organize_grasp_first_proposal(
    proposal: dict, task_type: object, state: dict, physical_options: dict,
) -> Tuple[dict, List[str]]:
    """Turn a poor VLM choice into the best code-checked progress grasp.

    RGB reasoning may rank an occluded object above an easy one, or request
    reobserve/stop even though the grasp-yaw scan found an executable block.
    The model still supplies task intent, but factual physical feasibility is
    authoritative: when at least one outside block is graspable, select a
    feasible candidate and let semantic geometry generate/validate its row
    destination.
    """
    if str(task_type or "") != "organize_blocks":
        return proposal, []
    feasible = [
        item for item in physical_options.get("candidates") or []
        if isinstance(item, dict) and item.get("grasp_feasible")
    ]
    if not feasible:
        return proposal, []
    selected = next((
        item for item in feasible
        if (
            str(item.get("selected_object_id")) == str(proposal.get("selected_object_id"))
            or (
                proposal.get("selected_object_ref")
                and item.get("selected_object_ref") == proposal.get("selected_object_ref")
            )
            or (
                proposal.get("selected_track_id")
                and item.get("selected_track_id") == proposal.get("selected_track_id")
            )
        )
    ), feasible[0])
    obj = object_by_string_id(state.get("objects", []), selected.get("selected_object_id"))
    if not isinstance(obj, dict):
        return proposal, []
    center = _object_center(obj)
    if center is None:
        return proposal, []
    output = dict(proposal)
    previous_identity = (
        proposal.get("action_type"), proposal.get("selected_object_id"),
        proposal.get("selected_object_ref"), proposal.get("selected_track_id"),
    )
    new_identity = (
        "pick_place", obj.get("id"), obj.get("object_ref"), obj.get("track_id"),
    )
    output.update({
        "strategy_id": "organize_blocks",
        "action_type": "pick_place",
        "selected_object_id": obj.get("id"),
        "selected_object_ref": obj.get("object_ref"),
        "selected_track_id": obj.get("track_id"),
        "object_label": obj.get("label"),
        "object_center_base_m": list(center),
        "scene_revision": int(state.get("scene_revision", 0)),
    })
    changed = []
    if previous_identity != new_identity:
        # Do not retain a pose meant for the model's rejected object/action.
        output["target_pose_base"] = {
            "position_m": list(center),
            "yaw_rad": 0.0,
        }
        changed.extend([
            "action_type", "selected_object_id", "selected_object_ref",
            "selected_track_id", "object_label", "object_center_base_m",
            "target_pose_base",
        ])
    for key in ("strategy_id", "scene_revision"):
        if proposal.get(key) != output.get(key):
            changed.append(key)
    return output, list(dict.fromkeys(changed))


def _search_geometry_checked_organize_nudge(
    args: Any, cycle_dir: str, state: dict, failed_proposal: dict, plan: dict,
    progress: dict, protection: dict, step_index: int, failed_checked: dict,
    failure_history: List[dict], physical_options: dict,
) -> Optional[Tuple[dict, dict, dict]]:
    """Try other loose objects only after a VLM nudge fails physical geometry.

    This is a bounded repair, not a second planner: cardinal directions and two
    tool yaw orientations are enumerated, then the existing semantic, swept-
    volume, and MoveIt preflights remain authoritative.
    """
    target = object_by_string_id(
        state.get("objects", []), failed_proposal.get("target_object_id"),
    )
    target_center = _object_center(target or {})
    if target is None or target_center is None:
        return None
    failed_selected_id = failed_proposal.get("selected_object_id")
    colliding_id = (
        ((failed_checked.get("tool_swept_volume_report") or {}).get("collision") or {})
        .get("object_id")
    )
    objects = [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict)
        and obj.get("id") is not None
        and str(obj.get("id")) not in {str(target.get("id")), str(failed_selected_id)}
        and obj.get("visible", True)
    ]
    objects.sort(key=lambda obj: (
        0 if colliding_id is not None and str(obj.get("id")) == str(colliding_id) else 1,
        math.dist((_object_center(obj) or [99.0, 99.0])[:2], target_center[:2]),
    ))
    protected_state, protected_ids = _state_with_dynamic_protection(state, protection)
    attempts = []
    cardinal = (
        ([1.0, 0.0, 0.0], "-x"),
        ([-1.0, 0.0, 0.0], "+x"),
        ([0.0, 1.0, 0.0], "-y"),
        ([0.0, -1.0, 0.0], "+y"),
    )
    for obj in objects:
        center = _object_center(obj)
        if center is None:
            continue
        # Directions that increase separation from the blocked target first;
        # the semantic check below rejects any non-improving remainder.
        delta = [center[0] - target_center[0], center[1] - target_center[1]]
        directions = sorted(
            cardinal,
            key=lambda item: -(delta[0] * item[0][0] + delta[1] * item[0][1]),
        )
        for direction, contact_side in directions:
            for yaw in (0.0, math.pi / 2.0):
                candidate = {
                    **failed_proposal,
                    "strategy_id": "clear_blocker_by_nudge",
                    "action_type": "nudge",
                    "selected_object_id": obj.get("id"),
                    "selected_object_ref": obj.get("object_ref"),
                    "selected_track_id": obj.get("track_id"),
                    "object_label": obj.get("label"),
                    "object_center_base_m": center,
                    "target_object_id": target.get("id"),
                    "target_object_ref": target.get("object_ref"),
                    "target_object_track_id": target.get("track_id"),
                    "target_object_label": target.get("label"),
                    "target_object_center_base_m": target_center,
                    "contact_side": contact_side,
                    "direction_base": list(direction),
                    "distance_m": 0.04,
                    "gripper_yaw_rad": yaw,
                    "scene_revision": int(state.get("scene_revision", 0)),
                    "reason": "automatic bounded clearance search after VLM nudge geometry failure",
                }
                semantic_error = _task_nudge_semantic_error(
                    candidate, progress, failure_history, plan, physical_options,
                )
                if semantic_error:
                    attempts.append({
                        "selected_object_id": obj.get("id"),
                        "direction_base": list(direction),
                        "gripper_yaw_rad": yaw,
                        "result": semantic_error,
                    })
                    continue
                adapted = adapt_task_action_to_legacy_action(candidate)
                action, safety = validate_vlm_action_decision(
                    adapted, protected_state, protected_ids=protected_ids,
                )
                if action is None:
                    attempts.append({
                        "selected_object_id": obj.get("id"),
                        "direction_base": list(direction),
                        "gripper_yaw_rad": yaw,
                        "result": safety.get("reason"),
                    })
                    continue
                action["selected_object_id"] = obj.get("id")
                candidate_dir = os.path.join(
                    cycle_dir,
                    "automatic_clearance_candidate_obj_{}_{}_yaw_{}".format(
                        obj.get("id"), contact_side.replace("+", "p").replace("-", "m"),
                        int(round(math.degrees(yaw))),
                    ),
                )
                os.makedirs(candidate_dir, exist_ok=True)
                checked = _preflight_task_action(
                    args, candidate_dir, protected_state, action, step_index,
                )
                attempts.append({
                    "selected_object_id": obj.get("id"),
                    "direction_base": list(direction),
                    "gripper_yaw_rad": yaw,
                    "result": "accepted" if checked.get("moveit_feasible") else (
                        checked.get("preflight_failure_stage")
                        or checked.get("moveit_preflight_error")
                        or "preflight_failed"
                    ),
                })
                if checked.get("moveit_feasible"):
                    report = {
                        "validation_stage": "automatic_clearance_search",
                        "source_failed_selected_object_id": failed_selected_id,
                        "selected_object_id": obj.get("id"),
                        "direction_base": list(direction),
                        "contact_side": contact_side,
                        "gripper_yaw_rad": yaw,
                        "candidate_attempts": attempts,
                    }
                    write_json(
                        os.path.join(cycle_dir, "automatic_clearance_search.json"),
                        report,
                    )
                    return checked, candidate, report
    write_json(os.path.join(cycle_dir, "automatic_clearance_search.json"), {
        "validation_stage": "automatic_clearance_search",
        "result": "no_executable_candidate",
        "candidate_attempts": attempts,
    })
    return None


def _automatic_organize_clearance_when_all_grasps_blocked(
    args: Any, cycle_dir: str, state: dict, plan: dict, progress: dict,
    protection: dict, step_index: int, failure_history: List[dict],
    physical_options: dict,
) -> Optional[Tuple[dict, dict, dict]]:
    """Seed the bounded side-push search from the easiest blocked grasp."""
    candidates = [
        item for item in physical_options.get("candidates") or []
        if isinstance(item, dict) and not item.get("grasp_feasible")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        len(item.get("blocking_object_ids") or []) or 999,
        -float(item.get("feasible_yaw_span_deg") or 0.0),
    ))
    target_option = candidates[0]
    target = object_by_string_id(
        state.get("objects", []), target_option.get("selected_object_id"),
    )
    target_center = _object_center(target or {})
    if not isinstance(target, dict) or target_center is None:
        return None
    blocker_ids = target_option.get("blocking_object_ids") or []
    seed = {
        "strategy_id": "clear_blocker_by_nudge",
        "action_type": "nudge",
        "selected_object_id": target.get("id"),
        "selected_object_ref": target.get("object_ref"),
        "selected_track_id": target.get("track_id"),
        "object_label": target.get("label"),
        "object_center_base_m": list(target_center),
        "target_object_id": target.get("id"),
        "target_object_ref": target.get("object_ref"),
        "target_object_track_id": target.get("track_id"),
        "target_object_label": target.get("label"),
        "target_object_center_base_m": list(target_center),
        "scene_revision": int(state.get("scene_revision", 0)),
        "reason": "all current organize grasps are physically blocked; enter bounded clearance",
    }
    failed_checked = {}
    if blocker_ids:
        failed_checked = {
            "tool_swept_volume_report": {
                "collision": {"object_id": blocker_ids[0]},
            },
        }
    return _search_geometry_checked_organize_nudge(
        args, cycle_dir, state, seed, plan, progress, protection, step_index,
        failed_checked, failure_history, physical_options,
    )


def _repeats_physically_failed_strategy(
    proposal: dict, fingerprint: dict, replanning: dict,
) -> bool:
    """Reject only the same physical tactic, not a newly selected object/action.

    Organize pick/place proposals intentionally share the stable strategy id
    ``organize_blocks``.  Requiring that id itself to change after two blocked
    grasps made every later object look like the same failed tactic and
    prevented both trying another object and switching to clearance.
    """
    constraints = replanning.get("hard_constraints") or {}
    if not constraints.get("required_strategy_change"):
        return False
    strategy_id = proposal.get("strategy_id")
    action_type = proposal.get("action_type")
    operated_track = (fingerprint or {}).get("operated_track_id")
    for failed in replanning.get("failed_actions") or []:
        reason_codes = {str(value) for value in failed.get("reason_codes") or []}
        physically_blocked = bool(reason_codes & {
            "selected_pick_not_grasp_feasible",
            "tool_swept_volume_rejected",
            "moveit_preflight_failed",
        })
        if not physically_blocked:
            continue
        if not (
            failed.get("strategy_id") == strategy_id
            and failed.get("action_type") == action_type
            and failed.get("operated_object") == operated_track
        ):
            continue
        # A push from the opposite contact side is a different physical
        # tactic: its vertical entry column and horizontal swept volume are
        # different.  Likewise, a changed pick destination or grasp yaw can
        # resolve the prior physical failure.  Keep the conservative legacy
        # behavior only when old context lacks a structured fingerprint.
        try:
            failed_fingerprint = json.loads(str(failed.get("fingerprint") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            failed_fingerprint = {}
        if not failed_fingerprint:
            return True
        if action_type == "nudge":
            if failed_fingerprint.get("direction_bin") != fingerprint.get("direction_bin"):
                continue
            if failed_fingerprint.get("grasp_yaw_bin_deg") != fingerprint.get("grasp_yaw_bin_deg"):
                continue
            return True
        if action_type in {"pick_place", "pick_reorient_place", "pick_away"}:
            if failed_fingerprint.get("destination_bin") != fingerprint.get("destination_bin"):
                continue
            if failed_fingerprint.get("grasp_yaw_bin_deg") != fingerprint.get("grasp_yaw_bin_deg"):
                continue
            return True
        if failed.get("fingerprint") == fingerprint.get("fingerprint"):
            return True
    return False


def _normalize_task_action_protocol(
    proposal: dict, task_type: object, current_scene_revision: object = None,
) -> Tuple[dict, List[str]]:
    """Drop union-schema fields that cannot affect this concrete action."""
    output = dict(proposal)
    changed: List[str] = []
    if (
        str(task_type or "") != "organize_blocks"
        or str(output.get("action_type") or "") != "pick_place"
    ):
        return output, changed
    irrelevant = (
        "target_object_ref", "target_object_track_id", "target_object_id",
        "target_object_label", "target_object_center_base_m",
        "contact_side", "direction_base", "push_direction_base",
        "distance_m", "push_distance_m", "gripper_yaw_rad",
        "safe_place_center_base_m", "safe_place_center_m", "role_id",
    )
    for key in irrelevant:
        if key in output:
            output.pop(key, None)
            changed.append(key)
    if output.get("strategy_id") != "organize_blocks":
        output["strategy_id"] = "organize_blocks"
        changed.append("strategy_id")
    expected_ref_prefix = "scene_{}:".format(current_scene_revision)
    if (
        current_scene_revision is not None
        and str(output.get("selected_object_ref") or "").startswith(expected_ref_prefix)
        and str(output.get("scene_revision")) != str(current_scene_revision)
    ):
        output["scene_revision"] = int(current_scene_revision)
        changed.append("scene_revision")
    return output, changed


def _retry_geometry_checked_organize_target(
    proposal: dict,
    report: dict,
    state: dict,
    contract: dict,
    plan: dict,
    progress: dict,
    protection: dict,
    semantics_config: dict,
) -> Tuple[Optional[dict], dict, dict]:
    """Retry validator-generated organize targets and fully revalidate each."""
    if (
        contract.get("task_type") != "organize_blocks"
        or proposal.get("action_type") != "pick_place"
    ):
        return None, proposal, report
    original_pose = proposal.get("target_pose_base") or {}
    corrected = proposal
    corrected_report = report
    corrections = []
    for _retry in range(2):
        detail = (
            ((corrected_report.get("checks") or {}).get("target_pose_base") or {})
            .get("detail") or {}
        )
        source = None
        suggested = detail.get("suggested_collision_free_position_m")
        if isinstance(suggested, (list, tuple)) and len(suggested) >= 3:
            source = "validator_suggested_collision_free_position_m"
        else:
            suggested = detail.get("suggested_interval_midpoint_position_m")
            if isinstance(suggested, (list, tuple)) and len(suggested) >= 3:
                source = "validator_suggested_interval_midpoint_position_m"
        if source is None:
            return None, proposal, report
        previous_position = (
            (corrected.get("target_pose_base") or {}).get("position_m")
        )
        corrected = {
            **corrected,
            "target_pose_base": {
                **(corrected.get("target_pose_base") or original_pose),
                "position_m": [float(value) for value in suggested[:3]],
            },
        }
        corrections.append({
            "source": source,
            "original_position_m": previous_position,
            "corrected_position_m": list(suggested[:3]),
        })
        action, corrected_report = validate_task_action(
            corrected, state, contract, plan, progress, protection,
            semantics_config,
        )
        if action is not None:
            corrected_report = {
                **corrected_report,
                "automatic_target_correction": {
                    **corrections[-1],
                    "correction_chain": corrections,
                    "revalidated": True,
                },
            }
            return action, corrected, corrected_report
    return None, proposal, report


def _normalize_organize_selected_group(
    proposal: dict, task_type: object, state: dict, plan: dict,
) -> Tuple[dict, List[str]]:
    """Ground an organize pick to the selected object's factual color group."""
    if (
        str(task_type or "") != "organize_blocks"
        or proposal.get("action_type") != "pick_place"
    ):
        return proposal, []
    selected = object_by_string_id(
        state.get("objects", []), proposal.get("selected_object_id"),
    )
    color = color_value_from_object(selected)
    if color is None:
        return proposal, []
    group = next((
        item for item in plan.get("groups", [])
        if str(item.get("group_value") or "").lower() == color
    ), None)
    if group is None:
        return proposal, []
    output = dict(proposal)
    changed = []
    expected = {
        "group_id": group.get("group_id"),
        "target_region_id": group.get("target_region_id"),
    }
    for key, value in expected.items():
        if value is not None and output.get(key) != value:
            output[key] = value
            changed.append(key)
    return output, changed


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
    if resolved.get("target_object_id") is not None:
        output["target_object_id"] = resolved["target_object_id"]
        output["target_object_ref"] = resolved.get("target_object_ref")
        if resolved.get("target_object_track_id"):
            output["target_object_track_id"] = resolved["target_object_track_id"]
        else:
            output.pop("target_object_track_id", None)
    if output.get("action_type") == "nudge":
        normalized_direction = _normalize_nudge_direction(output.get("direction_base"))
        if normalized_direction is not None and list(output.get("direction_base") or []) != normalized_direction:
            output["model_reported_direction_base"] = output.get("direction_base")
            output["direction_base"] = normalized_direction
            output["direction_base_source"] = "normalized_model_xy_direction"
        expected_contact = _contact_side_for_direction(output.get("direction_base"))
        if expected_contact is not None and output.get("contact_side") != expected_contact:
            output["model_reported_contact_side"] = output.get("contact_side")
            output["contact_side"] = expected_contact
            output["contact_side_source"] = "derived_opposite_to_valid_direction_base"
    output.setdefault("strategy_id", "{}_{}".format(proposal.get("action_type", "action"), proposal.get("role_id") or proposal.get("group_id") or "task"))
    return output


def _normalize_nudge_direction(direction: Any) -> Optional[List[float]]:
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        return None
    try:
        x, y, z = [float(value) for value in direction]
    except (TypeError, ValueError):
        return None
    xy_norm = (x * x + y * y) ** 0.5
    if xy_norm <= 1e-9 or abs(z) > 1e-3:
        return None
    return [x / xy_norm, y / xy_norm, 0.0]


def _contact_side_for_direction(direction: Any) -> Optional[str]:
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        return None
    try:
        x, y, z = [float(value) for value in direction]
    except (TypeError, ValueError):
        return None
    norm = (x * x + y * y + z * z) ** 0.5
    if abs(norm - 1.0) > 1e-3 or abs(z) > 1e-3:
        return None
    if abs(x) >= abs(y):
        return "-x" if x >= 0.0 else "+x"
    return "-y" if y >= 0.0 else "+y"


def _task_nudge_semantic_error(
    proposal: dict, progress: dict, failure_history: Optional[List[dict]] = None,
    grounded_plan: Optional[dict] = None,
    physical_options: Optional[dict] = None,
) -> Optional[str]:
    action_type = str(proposal.get("action_type"))
    if action_type not in {"nudge", "pick_away"}:
        return None
    if progress.get("task_type") == "organize_blocks":
        blocked_entry, blocked_check = _latest_blocked_grasp_failure(failure_history or [])
        all_current_picks_blocked = bool(
            isinstance(physical_options, dict)
            and physical_options.get("candidates")
            and not physical_options.get("direct_grasp_available")
        )
        if blocked_check is None and not all_current_picks_blocked:
            return "organize_clearance_requires_physical_grasp_blockage"
        blocked_target = (blocked_entry.get("rejected_action") or {}).get("selected_object_ref")
        if blocked_target and str(proposal.get("target_object_ref")) != str(blocked_target):
            return "organize_clearance_target_not_blocked_pick_object"
    if action_type != "nudge":
        return None
    if progress.get("task_type") == "organize_blocks" and not _nudge_increases_target_separation(proposal):
        return "organize_clearance_must_increase_target_separation"
    if (
        progress.get("task_type") == "organize_blocks"
        and _nudge_end_inside_target_region(proposal, grounded_plan or {})
    ):
        return "organize_clearance_avoid_target_regions"
    text = " ".join(str(proposal.get(key) or "") for key in ("reason", "predicted_scene_benefit")).lower()
    if any(token in text for token in ("上方", "堆叠", "on_top", "on top", "stack")):
        return "nudge_cannot_satisfy_vertical_stack_relation"
    if proposal.get("role_id") and any("on_top" in str(value) for value in progress.get("unsatisfied_predicates", [])):
        return "nudge_cannot_satisfy_vertical_stack_relation"
    return None


def _nudge_end_inside_target_region(proposal: dict, grounded_plan: dict) -> bool:
    center = proposal.get("object_center_base_m")
    direction = proposal.get("direction_base")
    try:
        end_x = float(center[0]) + float(direction[0]) * float(proposal.get("distance_m"))
        end_y = float(center[1]) + float(direction[1]) * float(proposal.get("distance_m"))
    except (TypeError, ValueError, IndexError):
        return False
    for region in grounded_plan.get("target_regions") or []:
        bounds = region.get("bounds_base_m") or {}
        try:
            if (
                float(bounds["xmin"]) <= end_x <= float(bounds["xmax"])
                and float(bounds["ymin"]) <= end_y <= float(bounds["ymax"])
            ):
                return True
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _nudge_increases_target_separation(proposal: dict, minimum_gain_m: float = 0.005) -> bool:
    """Reject clearance pushes that move the blocker toward the blocked pick target."""
    operated = proposal.get("object_center_base_m")
    target = proposal.get("target_object_center_base_m")
    direction = proposal.get("direction_base")
    try:
        ox, oy = float(operated[0]), float(operated[1])
        tx, ty = float(target[0]), float(target[1])
        dx, dy = float(direction[0]), float(direction[1])
        distance = float(proposal.get("distance_m"))
    except (TypeError, ValueError, IndexError):
        # Schema/grounding validation owns malformed fields.
        return True
    start_distance = ((ox - tx) ** 2 + (oy - ty) ** 2) ** 0.5
    end_distance = (
        (ox + dx * distance - tx) ** 2
        + (oy + dy * distance - ty) ** 2
    ) ** 0.5
    return end_distance >= start_distance + float(minimum_gain_m)


def _latest_blocked_grasp_failure(history: List[dict]) -> Tuple[dict, Optional[dict]]:
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        check = next((
            item for item in entry.get("failed_checks", [])
            if isinstance(item, dict)
            and item.get("type") == "selected_object_grasp_feasible"
            and (item.get("all_grasps_blocked") or not item.get("grasp_feasible", True))
        ), None)
        if check is not None:
            return entry, check
    return {}, None


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
        return _deduplicate_perception_state(
            attach_configured_workspace(load_json(args.offline_scene_state), args)
        )
    initial_dir = os.path.join(args.output_dir, "initial_task_scene"); capture_scene_observation(args, initial_dir)
    state = _deduplicate_perception_state(
        attach_configured_workspace(
            load_json(os.path.join(initial_dir, "private_scene_state.json")), args,
        )
    )
    state = _recover_low_confidence_observation(
        args,
        state,
        os.path.join(args.output_dir, "initial_task_scene_low_confidence"),
    )
    reference_state_path = str(getattr(args, "resume_reference_state_json", "") or "")
    reference_action_path = str(getattr(args, "resume_reference_action_json", "") or "")
    reference_execution_path = str(
        getattr(args, "resume_reference_execution_json", "") or ""
    )
    if reference_state_path or reference_action_path or reference_execution_path:
        if not reference_state_path or not reference_action_path:
            raise RuntimeError("RESUME_REFERENCE_REQUIRES_STATE_AND_ACTION")
        reference = _deduplicate_perception_state(load_json(reference_state_path))
        action = load_json(reference_action_path)
        expected = _expected_state_after_action(reference, action)
        if reference_execution_path:
            state, review = _recover_trusted_executed_object(
                reference,
                expected,
                action,
                load_json(reference_execution_path),
                state,
            )
        else:
            state, review = review_and_recover_expected_misses(
                args,
                expected,
                state,
                os.path.join(initial_dir, "resume_perception_review"),
                _known_action_description(action),
            )
        write_json(os.path.join(initial_dir, "resume_vlm_perception_review.json"), review)
        if (
            not reference_execution_path
            and not review.get("recovered")
            and len(state.get("objects", [])) < len(expected.get("objects", []))
        ):
            raise RuntimeError("RESUME_PERCEPTION_REVIEW_DID_NOT_CONFIRM_EXPECTED_SCENE")
    return state


def _reobserve(
    args: Any, cycle_dir: str, runtime: dict, previous_state: Optional[dict] = None,
) -> dict:
    if getattr(args, "offline_scene_state", ""):
        raise RuntimeError(
            "OFFLINE_REOBSERVE_UNAVAILABLE: offline_scene_state is a fixed replay; "
            "rerun with live perception or provide a newer offline scene"
        )
    observation_dir = os.path.join(cycle_dir, "observation_after_vlm_reobserve"); capture_scene_observation(args, observation_dir)
    state = attach_configured_workspace(
        load_json(os.path.join(observation_dir, "private_scene_state.json")), args,
    )
    state = _deduplicate_perception_state(state)
    state = _recover_low_confidence_observation(
        args,
        state,
        os.path.join(cycle_dir, "observation_after_vlm_reobserve_low_confidence"),
    )
    if previous_state is not None:
        state, review = review_and_recover_static_misses(
            args, previous_state, state, observation_dir,
        )
        write_json(os.path.join(observation_dir, "vlm_perception_review.json"), review)
    advance_scene_revision(runtime, state)
    return state


def _recover_low_confidence_observation(args: Any, state: dict, recovery_dir: str) -> dict:
    recovery_threshold = float(getattr(args, "initial_detector_recovery_score_thresh", 0.0) or 0.0)
    if not 0.0 < recovery_threshold < float(getattr(args, "score_thresh", 0.5)):
        return state
    capture_scene_observation(args, recovery_dir, score_thresh=recovery_threshold)
    low_state = _deduplicate_perception_state(
        attach_configured_workspace(
            load_json(os.path.join(recovery_dir, "private_scene_state.json")), args,
        )
    )
    state, review = review_and_recover_low_confidence_candidates(
        args,
        state,
        low_state,
        os.path.join(recovery_dir, "vlm_candidate_review"),
    )
    write_json(os.path.join(recovery_dir, "vlm_perception_review.json"), review)
    return state


def _deduplicate_perception_state(state: dict) -> dict:
    """Remove only near-identical 3D duplicates, not adjacent block instances."""
    objects = list(state.get("objects", []))
    merged = merge_duplicate_objects_3d(
        objects,
        center_threshold_m=0.003,
        dimension_threshold_m=0.006,
        bbox_iou_threshold=0.70,
    )
    if len(merged) == len(objects):
        return state
    output = {**state, "objects": merged}
    output["perception_duplicate_merge"] = {
        "raw_object_count": len(objects),
        "deduplicated_object_count": len(merged),
        "merged_objects": [
            {
                "kept_id": item.get("id"),
                "merged_duplicate_ids": item.get("merged_duplicate_ids", []),
                "reason": item.get("merge_reason"),
            }
            for item in merged if item.get("merged_duplicate_ids")
        ],
    }
    return output


def _expected_state_after_action(state: dict, action: dict) -> dict:
    """Apply only a validated action's known planar result to a scene copy."""
    output = {**state, "objects": [dict(item) for item in state.get("objects", [])]}
    selected_id = action.get("selected_object_id", action.get("object_id"))
    selected = object_by_string_id(output["objects"], selected_id)
    if selected is None:
        return output
    action_type = str(action.get("action_type") or "")
    expected_center = None
    if action_type in {"pick_place", "pick_reorient_place"}:
        expected_center = (action.get("target_pose_base") or {}).get("position_m")
    elif action_type == "pick_away":
        expected_center = action.get("safe_place_center_m") or action.get("safe_place_center_base_m")
    elif action_type == "nudge":
        center = selected.get("geometry_center_m") or selected.get("center_3d_base_m")
        direction = action.get("direction_base")
        try:
            distance = float(action.get("distance_m"))
            expected_center = [
                float(center[index]) + float(direction[index]) * distance for index in range(3)
            ]
        except (TypeError, ValueError, IndexError):
            expected_center = None
    if isinstance(expected_center, (list, tuple)) and len(expected_center) >= 3:
        selected["geometry_center_m"] = [float(value) for value in expected_center[:3]]
        selected["center_3d_base_m"] = [float(value) for value in expected_center[:3]]
        selected.pop("bbox_xyxy_px", None)
        selected.pop("bbox", None)
        selected["expected_pose_source"] = "validated_executed_action"
    output["expected_action"] = _known_action_description(action)
    return output


def _recover_trusted_executed_object(
    reference_state: dict,
    expected_state: dict,
    action: dict,
    execution: dict,
    current_state: dict,
    match_tolerance_m: float = 0.030,
) -> Tuple[dict, dict]:
    """Restore only the selected object proven moved by a completed execution log.

    The current camera frame remains authoritative for every other object.  This is
    intentionally narrower than static-scene recovery because the robot may have
    moved one block outside the standard observation frame before a host reboot.
    """
    if str(execution.get("status") or "") != "executed_and_reobserved":
        raise RuntimeError("TRUSTED_RESUME_EXECUTION_NOT_COMPLETE")

    selected_id = action.get("selected_object_id", action.get("object_id"))
    reference_object = object_by_string_id(reference_state.get("objects", []), selected_id)
    expected_object = object_by_string_id(expected_state.get("objects", []), selected_id)
    if reference_object is None or expected_object is None:
        raise RuntimeError("TRUSTED_RESUME_SELECTED_OBJECT_NOT_FOUND")

    color = color_value_from_object(reference_object)
    source_center = _object_center(reference_object)
    target_center = _object_center(expected_object)
    if color is None or source_center is None or target_center is None:
        raise RuntimeError("TRUSTED_RESUME_SELECTED_OBJECT_GEOMETRY_INVALID")

    current_objects = list(current_state.get("objects", []))
    target_match = _same_color_xy_match(
        current_objects, color, target_center, match_tolerance_m,
    )
    if target_match is not None:
        return current_state, {
            "mode": "trusted_executed_action",
            "execution_status": execution.get("status"),
            "selected_object_id": selected_id,
            "recovered": [],
            "already_observed_at_target": target_match.get("id"),
        }

    source_match = _same_color_xy_match(
        current_objects, color, source_center, match_tolerance_m,
    )
    if source_match is not None:
        raise RuntimeError("TRUSTED_RESUME_ACTION_NOT_REFLECTED_AT_SOURCE")

    recovered = dict(expected_object)
    recovered["id"] = _next_unique_numeric_object_id(current_objects)
    recovered.pop("object_ref", None)
    recovered.pop("bbox_xyxy_px", None)
    recovered.pop("bbox", None)
    recovered["visible"] = False
    recovered["trusted_executed_recovery"] = True
    recovered["geometry_source"] = "executed_and_reobserved_action_target"
    output = {**current_state, "objects": current_objects + [recovered]}
    return output, {
        "mode": "trusted_executed_action",
        "execution_status": execution.get("status"),
        "selected_object_id": selected_id,
        "source_center_base_m": source_center,
        "target_center_base_m": target_center,
        "recovered": [{
            "id": recovered["id"],
            "label": recovered.get("label"),
            "visual_color": color,
            "geometry_center_m": target_center,
            "reason": "executed action target is outside current observation",
        }],
    }


def _object_center(obj: dict) -> Optional[List[float]]:
    center = obj.get("geometry_center_m") or obj.get("center_3d_base_m")
    if not isinstance(center, (list, tuple)) or len(center) < 3:
        return None
    try:
        return [float(value) for value in center[:3]]
    except (TypeError, ValueError):
        return None


def _same_color_xy_match(
    objects: List[dict], color: str, center: List[float], tolerance_m: float,
) -> Optional[dict]:
    limit_squared = float(tolerance_m) ** 2
    for obj in objects:
        if color_value_from_object(obj) != color:
            continue
        candidate = _object_center(obj)
        if candidate is None:
            continue
        distance_squared = sum(
            (candidate[index] - center[index]) ** 2 for index in range(2)
        )
        if distance_squared <= limit_squared:
            return obj
    return None


def _next_unique_numeric_object_id(objects: List[dict]) -> int:
    numeric_ids = []
    for obj in objects:
        try:
            numeric_ids.append(int(obj.get("id")))
        except (TypeError, ValueError):
            continue
    return max(numeric_ids, default=-1) + 1


def _known_action_description(action: dict) -> str:
    return json.dumps({
        "action_type": action.get("action_type"),
        "selected_object_id": action.get("selected_object_id", action.get("object_id")),
        "selected_track_id": action.get("selected_track_id"),
        "target_pose_base": action.get("target_pose_base"),
        "direction_base": action.get("direction_base"),
        "distance_m": action.get("distance_m"),
        "safe_place_center_m": action.get("safe_place_center_m") or action.get("safe_place_center_base_m"),
    }, ensure_ascii=False, sort_keys=True)


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
