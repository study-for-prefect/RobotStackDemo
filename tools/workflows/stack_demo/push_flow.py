"""VLM-first clearance action flow before picking the current target."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from robot_scene_pipeline.detection_merge import merge_duplicate_objects_3d
from robot_scene_pipeline.scene_memory import save_memory, update_from_detections
from robot_scene_pipeline.vlm_replanning import advance_scene_revision
from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, failure_state_path
from .clearance_execution import execute_nudge_and_reobserve, execute_pick_away_and_reobserve
from .push_clearing import current_protected_structure_ids, object_by_string_id
from .scene import reacquire_target
from .vlm_action_loop import mark_autonomous_execution_result, select_autonomous_vlm_action


def _selected_pick_object(current_state: dict, selected_action: dict) -> dict:
    selected = copy.deepcopy(
        object_by_string_id(current_state.get("objects", []), selected_action.get("object_id"))
    )
    selected_yaw = selected_action.get("selected_grasp_yaw_deg")
    if selected_yaw is None:
        return selected
    grasp = selected_action.get("grasp_validation") or {}
    selected["selected_grasp_yaw_deg"] = float(selected_yaw)
    selected["grasp_feasible"] = bool(grasp.get("grasp_feasible"))
    selected["selected_grasp_axis_delta_deg"] = grasp.get("selected_grasp_axis_delta_deg")
    selected["feasible_yaw_intervals_deg"] = grasp.get("feasible_yaw_intervals_deg", [])
    selected["blocked_yaw_intervals_deg"] = grasp.get("blocked_yaw_intervals_deg", [])
    selected["grasp_yaw_source"] = grasp.get("selected_grasp_source") or "post_vlm_physical_validation"
    return selected


def _center_xyz(obj: dict) -> Optional[list]:
    value = obj.get("geometry_center_m")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        center = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return center if all(math.isfinite(item) for item in center) else None


def _label_center_distance(first: dict, second: dict) -> float:
    if first.get("label") != second.get("label"):
        return math.inf
    first_center = _center_xyz(first)
    second_center = _center_xyz(second)
    if first_center is None or second_center is None:
        return math.inf
    return math.sqrt(sum((first_center[index] - second_center[index]) ** 2 for index in range(3)))


def _nearest_same_label_object(objects: Iterable[dict], template: dict) -> Optional[dict]:
    matches = [obj for obj in objects if isinstance(obj, dict) and obj.get("label") == template.get("label")]
    if not matches:
        return None
    matches.sort(key=lambda obj: _label_center_distance(obj, template))
    return matches[0]


def _ensure_unique_scene_object_ids(current_state: dict, held_object: dict, cycle_dir: str) -> Tuple[dict, dict]:
    objects = [obj for obj in current_state.get("objects", []) if isinstance(obj, dict)]
    seen = set()
    numeric_ids = []
    for obj in objects:
        try:
            numeric_ids.append(int(obj.get("id")))
        except (TypeError, ValueError):
            continue
    next_id = max(numeric_ids or [-1]) + 1
    output = []
    remapped = []
    for index, obj in enumerate(objects):
        item = copy.deepcopy(obj)
        old_id = item.get("id")
        old_key = str(old_id)
        if old_id is None or old_key in seen:
            while str(next_id) in seen:
                next_id += 1
            item["id"] = next_id
            item["original_detection_id"] = old_id
            item["snapshot_id_reassigned"] = True
            remapped.append(
                {
                    "object_index": index,
                    "label": item.get("label"),
                    "old_id": old_id,
                    "new_id": next_id,
                    "geometry_center_m": item.get("geometry_center_m"),
                }
            )
            seen.add(str(next_id))
            next_id += 1
        else:
            seen.add(old_key)
        output.append(item)
    if not remapped:
        return current_state, held_object
    unique_state = copy.deepcopy(current_state)
    unique_state["objects"] = output
    unique_state["snapshot_id_reassignment"] = {
        "schema_version": "snapshot_id_reassignment_v1",
        "reason": "duplicate_or_missing_snapshot_object_ids",
        "remapped_objects": remapped,
    }
    unique_target = _nearest_same_label_object(output, held_object) or held_object
    write_json(
        os.path.join(cycle_dir, "scene_state_unique_object_ids.json"),
        {
            "schema_version": "scene_state_unique_object_ids_v1",
            "remapped_objects": remapped,
            "state": unique_state,
            "target_after_reassignment": unique_target,
        },
    )
    return unique_state, copy.deepcopy(unique_target)


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
        "action_id", "action", "action_type", "object_id", "object_label", "obstacle_id",
        "target_object_id", "target_object_label",
        "direction_base", "distance_m", "safe_place_center_m", "moveit_feasible", "executable_safe",
        "contact_side", "gripper_yaw_rad", "recoverable_contacts",
        "collision_free", "sweep_collision_free", "selected_grasp_yaw_deg",
        "protected_structure_safe", "reason", "confidence",
        "scene_problem", "predicted_scene_benefit", "risk_assessment",
    )
    return {key: action.get(key) for key in keys if key in action}


def _write_blocked_failure(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    held_object: dict,
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
        "clearance_step_index": step_index,
        "required_next_step": "reobserve_after_manual_or_replan_before_any_pick",
        "blocked_pick_policy": "do_not_generate_pick_plan_after_rejected_vlm_action",
        "vlm_failure_history": (
            action_report.get("failure_history")
            or safety_report.get("failure_history")
            or []
        ),
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
    future_place_regions: Optional[Iterable[dict]],
    action_attempt: int,
) -> Tuple[dict, dict, dict]:
    runtime["current_stage"] = "vlm_requested_reobserve_before_pick"
    observed_state = capture_empty_observation(
        args,
        os.path.join(cycle_dir, "observation_after_vlm_reobserve"),
        runtime["held_object_id"],
    )
    if observed_state is None:
        raise RuntimeError("VLM requested reobserve, but no live observation was produced.")
    advance_scene_revision(runtime, observed_state)
    memory = update_from_detections(memory, observed_state.get("objects", []))
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(observed_state, held_template))
    return handle_vlm_action_before_pick(
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
        future_place_regions=future_place_regions,
        action_attempt=action_attempt + 1,
    )


def handle_vlm_action_before_pick(
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
    future_place_regions: Optional[Iterable[dict]] = None,
    action_attempt: int = 0,
) -> Tuple[dict, dict, dict]:
    step_index = int(action_attempt) + 1
    max_attempts = max(1, int(getattr(args, "max_vlm_action_attempts", 4)))
    if step_index > max_attempts:
        raise RuntimeError("VLM action loop exceeded max attempts before picking target {}.".format(held_object.get("id")))

    current_state, held_object = _ensure_unique_scene_object_ids(current_state, held_object, cycle_dir)
    current_state, held_object = _merge_scene_duplicates(current_state, held_object, cycle_dir)
    current_state["instruction"] = getattr(args, "instruction", current_state.get("instruction"))
    current_state["scene_revision"] = int(runtime.get("scene_revision", current_state.get("scene_revision", 1)))
    write_json(os.path.join(cycle_dir, "scene_state_before_action.json"), current_state)
    protected_ids = current_protected_structure_ids(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    selected_action, action_report, safety_report, preflight_report = select_autonomous_vlm_action(
        args,
        cycle_dir,
        current_state,
        held_object,
        protected_ids,
        base_id,
        memory,
        step_index,
        future_place_regions=future_place_regions,
        scene_revision=current_state["scene_revision"],
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
            future_place_regions,
            action_attempt,
        )
    if action_type == "stop":
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "stop",
                "reason": action_report.get("reason"),
                "selection_source": "vlm_action_policy",
                "status": "policy_requested_safe_stop",
            },
        )
        mark_autonomous_execution_result(
            cycle_dir,
            {
                "status": "policy_requested_safe_stop",
                "reason": action_report.get("reason"),
            },
        )
        raise RuntimeError(
            "VLM requested safe stop: {}".format(action_report.get("reason") or "unspecified")
        )
    if selected_action is None:
        reason = action_report.get("reason") or safety_report.get("reason") or "vlm_action_policy_rejected"
        _write_blocked_failure(
            args,
            cycle_dir,
            runtime,
            held_object,
            step_index,
            reason,
            action_report,
            safety_report,
            preflight_report,
        )
        raise RuntimeError(
            "VLM autonomous action did not pass physical validation; stopped fail-safe: {}".format(
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
        held_object = _selected_pick_object(current_state, selected_action)
        write_json(
            os.path.join(cycle_dir, "multi_step_clearance_summary.json"),
            {
                "schema_version": "multi_step_clearance_summary_v1",
                "status": "vlm_selected_pick_passed_physical_validation",
                "steps_completed": max(0, step_index - 1),
                "target_object_id": held_object.get("id"),
                "selected_grasp_yaw_deg": selected_action.get("selected_grasp_yaw_deg"),
                "planned_focus_object_id": held_template.get("id"),
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
    if args.execute and args.execute_push_clearing:
        advance_scene_revision(runtime, next_state)
        mark_autonomous_execution_result(cycle_dir, {
            "status": "executed_and_reobserved",
            "action_type": selected_action.get("action_type"),
            "next_scene_revision": runtime["scene_revision"],
        })
    if not (args.execute and args.execute_push_clearing):
        return next_state, memory, next_held
    return handle_vlm_action_before_pick(
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
        future_place_regions=future_place_regions,
        action_attempt=action_attempt + 1,
    )
