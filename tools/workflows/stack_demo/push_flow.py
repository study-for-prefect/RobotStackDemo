"""VLM-first clearance action flow before picking the current target."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from robot_scene_pipeline.detection_merge import merge_duplicate_objects_3d
from robot_scene_pipeline.geometry_relations import build_geometry_relations
from robot_scene_pipeline.scene_memory import save_memory, update_from_detections
from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, failure_state_path
from .clearance_execution import execute_nudge_and_reobserve, execute_pick_away_and_reobserve
from .push_clearing import current_protected_structure_ids, relation_objects_with_protected_structure
from .scene import reacquire_target
from .vlm_action import select_clearance_with_vlm_action_policy


def _relations_for_target(
    current_state: dict,
    held_object: dict,
    base_id: Any,
    previous_locked_stack: dict,
    args: Any,
    base_template: Optional[dict] = None,
) -> list:
    relation_objects = relation_objects_with_protected_structure(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    return build_geometry_relations(
        relation_objects,
        target_id=held_object["id"],
        target_object=held_object,
        gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
        gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
        gripper_side_clearance_m=getattr(args, "grasp_gripper_side_clearance_m", 0.006),
        grasp_approach_length_m=getattr(args, "grasp_approach_length_m", 0.02),
    )


def _target_grasp_analysis(relations: list, target_id: Any) -> dict:
    for relation in relations:
        if (
            relation.get("type") == "target_grasp_analysis"
            and str(relation.get("object")) == str(target_id)
        ):
            return relation
    return {}


def _apply_selected_grasp_to_target(held_object: dict, analysis: dict) -> dict:
    selected_yaw = analysis.get("selected_grasp_yaw_deg")
    if selected_yaw is None:
        return held_object
    held_object["selected_grasp_yaw_deg"] = float(selected_yaw)
    held_object["grasp_feasible"] = bool(analysis.get("grasp_feasible"))
    held_object["selected_grasp_axis_delta_deg"] = analysis.get("selected_grasp_axis_delta_deg")
    held_object["feasible_yaw_intervals_deg"] = analysis.get("feasible_yaw_intervals_deg", [])
    held_object["blocked_yaw_intervals_deg"] = analysis.get("blocked_yaw_intervals_deg", [])
    held_object["grasp_yaw_source"] = analysis.get("selected_grasp_source") or "adaptive_grasp_yaw_search"
    return held_object


def _merge_scene_duplicates(current_state: dict, held_object: dict, cycle_dir: str) -> Tuple[dict, dict]:
    objects = current_state.get("objects", [])
    merged = merge_duplicate_objects_3d(objects, preferred_id=held_object.get("id"))
    if len(merged) == len(objects) and all(
        str(before.get("id")) == str(after.get("id"))
        and before.get("merged_duplicate_ids") == after.get("merged_duplicate_ids")
        for before, after in zip(objects, merged)
        if isinstance(before, dict) and isinstance(after, dict)
    ):
        return current_state, held_object
    merged_state = copy.deepcopy(current_state)
    merged_state["objects"] = merged
    try:
        merged_target = next(obj for obj in merged if str(obj.get("id")) == str(held_object.get("id")))
    except StopIteration:
        merged_target = held_object
    write_json(
        os.path.join(cycle_dir, "scene_state_before_action_merged.json"),
        {
            "schema_version": "scene_state_duplicate_merge_v1",
            "merge_policy": "same_label_near_3d_center_similar_dimensions",
            "objects_before": len(objects),
            "objects_after": len(merged),
            "state": merged_state,
        },
    )
    return merged_state, merged_target


def _action_summary(action: dict) -> dict:
    keys = (
        "action_id", "action", "action_type", "object_id", "obstacle_id", "target_object_id",
        "direction_base", "distance_m", "safe_place_center_m", "moveit_feasible", "executable_safe",
        "collision_free", "sweep_collision_free", "workspace_feasible", "gripper_feasible",
        "selected_grasp_yaw_deg", "geometry_feasible", "approach_path_safe", "push_swept_safe",
        "push_end_safe", "protected_structure_safe", "reason", "confidence",
    )
    return {key: action.get(key) for key in keys if key in action}


def _write_blocked_failure(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    held_object: dict,
    analysis: dict,
    step_index: int,
    reason: str,
    action_report: dict,
    safety_report: dict,
    preflight_report: dict,
) -> None:
    failure = {
        "failure_state": "manual_required",
        "reason": reason,
        "target_object_id": held_object.get("id"),
        "target_label": held_object.get("label"),
        "grasp_action": analysis.get("action"),
        "clearance_step_index": step_index,
        "required_next_step": "reobserve_after_manual_or_replan_before_any_pick",
        "blocked_pick_policy": "do_not_generate_pick_plan_after_rejected_vlm_action",
    }
    runtime.update(failure)
    write_json(os.path.join(cycle_dir, "failure_state.json"), failure)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "rejected_by_vlm_action_safety_gate",
            "reason": reason,
            "failure_state": "manual_required",
            "vlm_action_decision": action_report,
            "safety_report": safety_report,
            "moveit_preflight": preflight_report,
        },
    )
    if getattr(args, "output_dir", None):
        write_json(failure_state_path(args), {**runtime, **failure})


def _reobserve_and_retry(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict],
    future_targets: Optional[Iterable[dict]],
    future_place_regions: Optional[Iterable[dict]],
    automatic_push_attempt: int,
) -> Tuple[dict, dict, dict]:
    runtime["current_stage"] = "vlm_requested_reobserve_before_pick"
    observed_state = capture_empty_observation(
        args,
        os.path.join(cycle_dir, "observation_after_vlm_reobserve"),
        runtime["held_object_id"],
    )
    if observed_state is None:
        raise RuntimeError("VLM requested reobserve, but no live observation was produced.")
    memory = update_from_detections(memory, observed_state.get("objects", []))
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(observed_state, held_template))
    return handle_push_clearing_before_pick(
        args,
        cycle_dir,
        runtime,
        memory,
        observed_state,
        held_template,
        held_object,
        base_id,
        previous_locked_stack,
        base_template=base_template,
        future_targets=future_targets,
        future_place_regions=future_place_regions,
        automatic_push_attempt=automatic_push_attempt + 1,
    )


def handle_push_clearing_before_pick(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    held_object: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict] = None,
    future_targets: Optional[Iterable[dict]] = None,
    future_place_regions: Optional[Iterable[dict]] = None,
    automatic_push_attempt: int = 0,
) -> Tuple[dict, dict, dict]:
    step_index = int(automatic_push_attempt) + 1
    max_attempts = max(1, int(getattr(args, "max_automatic_push_clearing_attempts", 2)))
    if step_index > max_attempts:
        raise RuntimeError("VLM action loop exceeded max attempts before picking target {}.".format(held_object.get("id")))

    current_state, held_object = _merge_scene_duplicates(current_state, held_object, cycle_dir)
    write_json(os.path.join(cycle_dir, "scene_state_before_action.json"), current_state)
    relations = _relations_for_target(
        current_state,
        held_object,
        base_id,
        previous_locked_stack,
        args,
        base_template=base_template,
    )
    analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(os.path.join(cycle_dir, "geometry_relations_before_action.json"), relations)
    write_json(os.path.join(cycle_dir, "grasp_yaw_analysis_before_action.json"), analysis)
    protected_ids = current_protected_structure_ids(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    selected_action, action_report, safety_report, preflight_report = select_clearance_with_vlm_action_policy(
        args,
        cycle_dir,
        current_state,
        held_object,
        analysis,
        protected_ids,
        base_id,
        memory,
        step_index,
        future_place_regions=future_place_regions,
    )
    action_type = action_report.get("action_type")
    if action_type == "reobserve":
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "reobserve",
                "reason": action_report.get("reason"),
                "clearance_step_index": step_index,
                "selection_source": "vlm_action_policy",
            },
        )
        return _reobserve_and_retry(
            args,
            cycle_dir,
            runtime,
            memory,
            current_state,
            held_template,
            base_id,
            previous_locked_stack,
            base_template,
            future_targets,
            future_place_regions,
            automatic_push_attempt,
        )
    if selected_action is None:
        reason = action_report.get("reason") or safety_report.get("reason") or "vlm_action_policy_rejected"
        _write_blocked_failure(
            args,
            cycle_dir,
            runtime,
            held_object,
            analysis,
            step_index,
            reason,
            action_report,
            safety_report,
            preflight_report,
        )
        raise RuntimeError(
            "VLM action policy did not produce a validated executable action for target {}; stopped fail-safe: {}".format(
                held_object.get("id"),
                reason,
            )
        )

    write_json(
        os.path.join(cycle_dir, "selected_action.json"),
        {
            "action": selected_action.get("action_type"),
            "reason": selected_action.get("reason"),
            "selected_action": _action_summary(selected_action),
            "clearance_step_index": step_index,
            "selection_source": "vlm_action_policy",
            "multi_step_status": "selected_vlm_action_intent",
            "post_action_requirement": "reobserve_and_rebuild_if_not_pick",
            "vlm_action_policy_output": action_report,
            "final_safety_gate": safety_report,
        },
    )
    if selected_action.get("action_type") == "pick":
        held_object = _apply_selected_grasp_to_target(held_object, analysis)
        write_json(
            os.path.join(cycle_dir, "multi_step_clearance_summary.json"),
            {
                "schema_version": "multi_step_clearance_summary_v1",
                "status": "target_graspable_by_vlm_action_policy",
                "steps_completed": max(0, step_index - 1),
                "target_object_id": held_object.get("id"),
                "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
            },
        )
        return current_state, memory, held_object

    if selected_action.get("action_type") == "pick_away":
        next_state, memory, next_held = execute_pick_away_and_reobserve(
            args,
            cycle_dir,
            runtime,
            memory,
            current_state,
            held_template,
            selected_action,
            base_id,
            previous_locked_stack,
            base_template,
            step_index,
        )
    else:
        next_state, memory, next_held = execute_nudge_and_reobserve(
            args,
            cycle_dir,
            runtime,
            memory,
            current_state,
            held_template,
            held_object,
            selected_action,
            base_id,
            previous_locked_stack,
            base_template,
            step_index,
        )
    if not (args.execute and args.execute_push_clearing):
        return next_state, memory, next_held
    return handle_push_clearing_before_pick(
        args,
        cycle_dir,
        runtime,
        memory,
        next_state,
        held_template,
        next_held,
        base_id,
        previous_locked_stack,
        base_template=base_template,
        future_targets=future_targets,
        future_place_regions=future_place_regions,
        automatic_push_attempt=automatic_push_attempt + 1,
    )
