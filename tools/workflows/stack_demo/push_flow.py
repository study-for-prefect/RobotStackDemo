"""Push-direction evaluation, optional execution, and manual clearing flow."""

import copy
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from robot_scene_pipeline.detection_merge import merge_duplicate_objects_3d
from robot_scene_pipeline.geometry_relations import build_geometry_relations
from robot_scene_pipeline.scene_memory import save_memory, update_from_detections
from tools.planning.decision_to_execution import write_json

from .commands import (
    capture_empty_observation,
    failure_state_path,
    load_json,
)
from .clearance_execution import (
    ClearancePreflightFailed,
    execute_nudge_and_reobserve,
    execute_pick_away_and_reobserve,
    preflight_nudge_candidate,
)
from .obstruction_frontier import build_frontier_clearance_plan, refresh_executable_safe
from .push_clearing import (
    current_protected_structure_ids,
    evaluate_push_candidates,
    object_by_string_id,
    pushable_blocking_relations,
    relation_objects_with_protected_structure,
)
from .scene import reacquire_target
from .push_selection import choose_clearance_action
from .target_recovery import missing_target_clearance_relations

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
        gripper_outer_width_m=getattr(args, "grasp_gripper_outer_width_m", 0.112),
        gripper_inner_width_m=getattr(args, "grasp_gripper_inner_width_m", 0.048),
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


def _raise_if_non_push_action(analysis: dict, has_push_candidates: bool = False) -> None:
    action = analysis.get("action")
    if action in (None, "pick", "push_clearing", "pick_away"):
        return
    if action in ("remove_top_object", "pick_away_top_object"):
        raise RuntimeError(
            "Target is under another object; do not solve this by rotating the gripper. "
            "Recommended action={} object_above_target={}.".format(
                action,
                analysis.get("object_above_target"),
            )
        )
    if action == "replan_required" and has_push_candidates:
        return
    if action == "replan_required":
        raise RuntimeError(
            "All grasp yaws are blocked by protected structure; push clearing is refused. "
            "blocked_by_base={} blocked_by_locked_structure={} blocked_by_placed_structure={}.".format(
                analysis.get("blocked_by_base"),
                analysis.get("blocked_by_locked_structure"),
                analysis.get("blocked_by_placed_structure"),
            )
        )


def _write_frontier_debug_files(cycle_dir: str, frontier_plan: dict) -> None:
    def candidate_summary(candidate: dict) -> dict:
        keys = (
            "candidate_id", "action", "action_type", "obstacle_id", "target_object_id",
            "frontier_depth", "blocks", "direction_base", "distance_m", "direction_source",
            "selected_grasp_yaw_deg", "safe_place_center_m", "feasible",
            "geometry_feasible", "approach_path_safe", "push_swept_safe", "push_end_safe",
            "future_task_feasible", "protected_structure_safe", "moveit_feasible",
            "task_effective", "direct_clearance_candidate", "exploratory",
            "automatic_execution_allowed", "executable_safe", "direct_target_gain",
            "enabling_gain", "free_space_gain", "utility_score", "easiness_score",
            "risk_score", "score", "target_yaw_gain", "reason",
            "moveit_preflight_error", "push_execution_plan_path",
        )
        return {key: candidate.get(key) for key in keys if key in candidate}

    def summarize(candidates: Iterable[dict]) -> list:
        return [candidate_summary(candidate) for candidate in candidates or []]

    write_json(os.path.join(cycle_dir, "obstruction_graph.json"), frontier_plan.get("obstruction_graph", {}))
    write_json(
        os.path.join(cycle_dir, "obstacle_frontier_candidates.json"),
        frontier_plan.get("obstacle_frontier_candidates", []),
    )
    write_json(
        os.path.join(cycle_dir, "all_clearance_action_candidates.json"),
        summarize(frontier_plan.get("all_clearance_action_candidates", [])),
    )
    write_json(
        os.path.join(cycle_dir, "safe_clearance_candidates.json"),
        summarize(frontier_plan.get("safe_clearance_candidates", [])),
    )
    write_json(
        os.path.join(cycle_dir, "preflight_clearance_candidates.json"),
        summarize(frontier_plan.get("preflight_clearance_candidates", [])),
    )
    write_json(
        os.path.join(cycle_dir, "selected_clearance_action.json"),
        candidate_summary(frontier_plan.get("selected_clearance_action") or {})
        or {"action": "no_feasible_clearance_action"},
    )
    if frontier_plan.get("debug_dump_full_candidates"):
        write_json(
            os.path.join(cycle_dir, "debug_full_clearance_candidates.json"),
            {
                "all_clearance_action_candidates": frontier_plan.get("all_clearance_action_candidates", []),
                "preflight_clearance_candidates": frontier_plan.get("preflight_clearance_candidates", []),
                "safe_clearance_candidates": frontier_plan.get("safe_clearance_candidates", []),
            },
        )


def _candidate_summary(candidate: dict) -> dict:
    keys = (
        "candidate_id", "action", "action_type", "obstacle_id", "target_object_id",
        "direction_base", "distance_m", "moveit_feasible", "executable_safe",
        "geometry_feasible", "approach_path_safe", "push_swept_safe", "push_end_safe",
        "protected_structure_safe", "task_effective", "exploratory",
        "automatic_execution_allowed", "target_yaw_gain", "score", "reason",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def _preflight_safe_candidates(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    held_object: dict,
    frontier_plan: dict,
    step_index: int,
) -> dict:
    candidates = list(frontier_plan.get("preflight_clearance_candidates", []))
    safe_candidates = []
    failures = []
    if not (args.execute and args.execute_push_clearing):
        frontier_plan["safe_clearance_candidates"] = []
        return {"safe_candidates": safe_candidates, "failures": failures}
    for candidate in candidates:
        if candidate.get("action_type") != "nudge":
            continue
        if candidate.get("exploratory"):
            failures.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": "exploratory_candidate_requires_manual_confirmation",
                }
            )
            continue
        checked = preflight_nudge_candidate(args, cycle_dir, current_state, held_object, candidate, step_index)
        refresh_executable_safe(checked)
        if checked.get("executable_safe"):
            safe_candidates.append(checked)
        else:
            failures.append(
                {
                    "candidate_id": checked.get("candidate_id"),
                    "reason": checked.get("moveit_preflight_error") or "hard_safety_filter_failed",
                    "moveit_feasible": checked.get("moveit_feasible"),
                    "executable_safe": checked.get("executable_safe"),
                }
            )
    safe_candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item.get("frontier_depth", 99)),
            str(item.get("candidate_id")),
        )
    )
    frontier_plan["safe_clearance_candidates"] = safe_candidates
    write_json(
        os.path.join(cycle_dir, "clearance_step_{:02d}_moveit_preflight.json".format(step_index)),
        {
            "safe_candidate_ids": [item.get("candidate_id") for item in safe_candidates],
            "failures": failures,
        },
    )
    return {"safe_candidates": safe_candidates, "failures": failures}


def _assert_selection_consistency(selected_clearance: Optional[dict], llm_selection: dict) -> None:
    if selected_clearance is None:
        return
    candidate_id = selected_clearance.get("candidate_id")
    if candidate_id is None:
        raise RuntimeError("Selected clearance action is missing candidate_id.")
    if llm_selection.get("selection_status") == "selected":
        llm_id = llm_selection.get("selected_candidate_id")
        if str(llm_id) != str(candidate_id):
            raise RuntimeError(
                "LLM selected_candidate_id {} does not match selected_clearance_action {}.".format(
                    llm_id,
                    candidate_id,
                )
            )


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
        merged_target = object_by_string_id(merged, held_object.get("id"))
    except RuntimeError:
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


def _write_blocked_no_clearance_failure(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    held_object: dict,
    analysis: dict,
    step_index: int,
) -> None:
    failure = {
        "failure_state": "manual_required",
        "reason": "grasp_blocked_no_executable_clearance",
        "target_object_id": held_object.get("id"),
        "target_label": held_object.get("label"),
        "all_grasps_blocked": bool(analysis.get("all_grasps_blocked")),
        "grasp_action": analysis.get("action"),
        "clearance_step_index": step_index,
        "required_next_step": "reobserve_after_manual_or_replan_before_any_pick",
        "blocked_pick_policy": "do_not_generate_pick_plan_do_not_fallback_to_min_area_rect_yaw",
    }
    runtime.update(failure)
    write_json(os.path.join(cycle_dir, "failure_state.json"), failure)
    if getattr(args, "output_dir", None):
        write_json(failure_state_path(args), {**runtime, **failure})


def _manual_clear_and_reobserve(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    held_template: dict,
    base_id: Any,
    previous_locked_stack: dict,
    reason: str,
    direction_assessment: dict,
    base_template: Optional[dict] = None,
    future_targets: Optional[Iterable[dict]] = None,
    future_place_regions: Optional[Iterable[dict]] = None,
    automatic_push_attempt: int = 0,
) -> Tuple[dict, dict, dict]:
    request_path = os.path.join(cycle_dir, "manual_clearance_required.json")
    write_json(
        request_path,
        {
            "schema_version": "manual_clearance_request_v1",
            "execution_status": "waiting_for_operator",
            "reason": reason,
            "target_object_id": held_template.get("id"),
            "target_label": held_template.get("label"),
            "direction_assessment": direction_assessment,
            "note": (
                "No safe automatic push direction is available. Clear the obstacle "
                "manually, then confirm so the scene can be observed again."
            ),
        },
    )
    print(
        "\nNo safe automatic push direction is available. "
        "Please clear the obstacle manually without moving the stack base.",
        flush=True,
    )
    try:
        input("After manual clearing is complete, press Enter to observe again...")
    except EOFError:
        raise RuntimeError(
            "Manual clearing requires interactive confirmation, but stdin is unavailable."
        )

    runtime["current_stage"] = "observation_after_manual_clearing"
    observed_state = capture_empty_observation(
        args,
        os.path.join(cycle_dir, "observation_after_manual_clearing"),
        runtime["held_object_id"],
    )
    if observed_state is None:
        raise RuntimeError("Manual clearing completed but no live observation was produced.")
    memory = update_from_detections(memory, observed_state.get("objects", []))
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(observed_state, held_template))
    relations = _relations_for_target(
        observed_state,
        held_object,
        base_id,
        previous_locked_stack,
        args,
        base_template=base_template,
    )
    write_json(os.path.join(cycle_dir, "geometry_relations_after_manual_clearing.json"), relations)
    protected_ids = current_protected_structure_ids(
        observed_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    remaining_push = pushable_blocking_relations(
        observed_state,
        relations,
        held_object["id"],
        base_id,
        previous_locked_stack,
        protected_object_ids=protected_ids,
    )
    after_analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(os.path.join(cycle_dir, "grasp_yaw_analysis_after_manual_clearing.json"), after_analysis)
    if after_analysis.get("action") == "pick" and after_analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, after_analysis)
    elif remaining_push:
        request = load_json(request_path)
        request["execution_status"] = "operator_confirmed_and_reobserved"
        request["post_manual_action"] = "rerun_automatic_push_planning"
        request["remaining_push_relations"] = remaining_push
        write_json(request_path, request)
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
            automatic_push_attempt=automatic_push_attempt,
        )
    else:
        raise RuntimeError(
            "Manual clearing did not produce a feasible grasp. action={} base={} locked={} placed={}.".format(
                after_analysis.get("action"),
                after_analysis.get("blocked_by_base"),
                after_analysis.get("blocked_by_locked_structure"),
                after_analysis.get("blocked_by_placed_structure"),
            )
        )
    request = load_json(request_path)
    request["execution_status"] = "operator_confirmed_and_reobserved"
    write_json(request_path, request)
    return observed_state, memory, held_object


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
    write_json(
        os.path.join(cycle_dir, "geometry_relations_before_pick.json"),
        relations,
    )
    write_json(
        os.path.join(cycle_dir, "geometry_relations_before_action.json"),
        relations,
    )
    protected_ids = current_protected_structure_ids(
        current_state.get("objects", []),
        base_id,
        previous_locked_stack,
        base_template=base_template,
    )
    push_candidates = pushable_blocking_relations(
        current_state,
        relations,
        held_object["id"],
        base_id,
        previous_locked_stack,
        protected_object_ids=protected_ids,
    )
    if held_object.get("target_detection_status") == "missing_in_current_observation":
        missing_candidates = missing_target_clearance_relations(
            current_state,
            held_object,
            protected_ids,
            radius_m=getattr(args, "missing_target_clearance_radius_m", 0.10),
            min_top_z_delta_m=getattr(args, "high_block_min_top_z_delta_m", 0.01),
        )
        seen_subjects = {str(item.get("subject")) for item in push_candidates}
        for candidate in missing_candidates:
            if str(candidate.get("subject")) in seen_subjects:
                continue
            push_candidates.append(candidate)
            seen_subjects.add(str(candidate.get("subject")))
    analysis = _target_grasp_analysis(relations, held_object["id"])
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_pick.json"),
        analysis,
    )
    write_json(
        os.path.join(cycle_dir, "grasp_yaw_analysis_before_action.json"),
        analysis,
    )
    _raise_if_non_push_action(analysis, has_push_candidates=True)
    if analysis.get("action") == "pick" and analysis.get("grasp_feasible"):
        held_object = _apply_selected_grasp_to_target(held_object, analysis)
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "pick",
                "reason": "direct_or_rotated_grasp_yaw_feasible",
                "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
                "priority": "direct_pick_then_rotated_yaw_pick",
                "clearance_step_index": step_index,
                "selection_source": "direct_geometry_grasp",
                "multi_step_status": "target_graspable",
            },
        )
        write_json(
            os.path.join(cycle_dir, "multi_step_clearance_summary.json"),
            {
                "schema_version": "multi_step_clearance_summary_v1",
                "status": "target_graspable",
                "steps_completed": max(0, step_index - 1),
                "target_object_id": held_object.get("id"),
                "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
            },
        )
        return current_state, memory, held_object

    frontier_plan = build_frontier_clearance_plan(
        current_state,
        held_object,
        protected_ids,
        args,
        future_place_regions=future_place_regions or [],
        evaluate_push_fn=evaluate_push_candidates,
    )
    frontier_plan["debug_dump_full_candidates"] = bool(getattr(args, "debug_dump_full_candidates", False))
    preflight_report = _preflight_safe_candidates(
        args,
        cycle_dir,
        current_state,
        held_object,
        frontier_plan,
        step_index,
    )
    selectable_clearance_candidates = frontier_plan.get("safe_clearance_candidates", [])
    selected_clearance, llm_selection = choose_clearance_action(
        args,
        cycle_dir,
        held_object,
        selectable_clearance_candidates,
        memory,
        step_index,
    )
    _assert_selection_consistency(selected_clearance, llm_selection)
    frontier_plan["selected_clearance_action"] = selected_clearance
    frontier_plan["llm_selection"] = llm_selection
    _write_frontier_debug_files(cycle_dir, frontier_plan)

    if selected_clearance is None:
        _write_blocked_no_clearance_failure(args, cycle_dir, runtime, held_object, analysis, step_index)
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": "replan_required",
                "reason": "no_feasible_clearance_action",
                "grasp_action": analysis.get("action"),
                "clearance_step_index": step_index,
                "multi_step_status": "no_feasible_clearance_action",
                "moveit_preflight_failures": preflight_report.get("failures", []),
            },
        )
        write_json(
            os.path.join(cycle_dir, "clearance_verification.json"),
            {
                "status": "not_requested",
                "reason": "no_feasible_clearance_action",
                "safe_candidate_count": 0,
                "failure_state": "manual_required",
                "failure_reason": "grasp_blocked_no_executable_clearance",
                "moveit_preflight_failures": preflight_report.get("failures", []),
            },
        )
        raise RuntimeError(
            "Target {} has all grasps blocked and no executable clearance action; pick planning is stopped.".format(
                held_object.get("id")
            )
        )

    ordered_candidates = [selected_clearance] + [
        candidate for candidate in selectable_clearance_candidates
        if candidate.get("candidate_id") != selected_clearance.get("candidate_id")
    ]
    preflight_failures = []
    next_state = None
    next_held = None
    for clearance_action in ordered_candidates:
        write_json(
            os.path.join(cycle_dir, "selected_action.json"),
            {
                "action": clearance_action.get("action"),
                "reason": clearance_action.get("reason"),
                "selected_clearance_action": _candidate_summary(clearance_action),
                "clearance_step_index": step_index,
                "selection_source": llm_selection.get("selection_source"),
                "selected_candidate_id": clearance_action.get("candidate_id"),
                "multi_step_status": "selected_one_frontier_action",
                "post_action_requirement": "reobserve_and_rebuild_obstruction_graph",
            },
        )
        try:
            if clearance_action.get("action_type") == "pick_away":
                next_state, memory, next_held = execute_pick_away_and_reobserve(
                    args,
                    cycle_dir,
                    runtime,
                    memory,
                    current_state,
                    held_template,
                    clearance_action,
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
                    clearance_action,
                    base_id,
                    previous_locked_stack,
                    base_template,
                    step_index,
                )
            selected_clearance = clearance_action
            if selected_clearance.get("executable_safe"):
                frontier_plan["safe_clearance_candidates"] = [selected_clearance]
                _write_frontier_debug_files(cycle_dir, frontier_plan)
            break
        except ClearancePreflightFailed as exc:
            preflight_failures.append(
                {
                    "candidate_id": clearance_action.get("candidate_id"),
                    "action_type": clearance_action.get("action_type"),
                    "error": str(exc),
                }
            )
            continue
    if next_state is None:
        _write_blocked_no_clearance_failure(args, cycle_dir, runtime, held_object, analysis, step_index)
        write_json(
            os.path.join(cycle_dir, "clearance_verification.json"),
            {
                "status": "no_feasible_clearance_action",
                "reason": "all_selected_candidates_failed_moveit_preflight",
                "preflight_failures": preflight_failures,
                "failure_state": "manual_required",
            },
        )
        raise RuntimeError(
            "All clearance candidates failed MoveIt preflight before gripper close; pick planning is stopped."
        )
    if not (args.execute and args.execute_push_clearing):
        return next_state, memory, next_held
    max_attempts = max(1, int(getattr(args, "max_automatic_push_clearing_attempts", 2)))
    if automatic_push_attempt + 1 >= max_attempts:
        write_json(
            os.path.join(cycle_dir, "multi_step_clearance_summary.json"),
            {
                "schema_version": "multi_step_clearance_summary_v1",
                "status": "multi_step_clearance_exhausted",
                "steps_completed": step_index,
                "max_attempts": max_attempts,
            },
        )
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
