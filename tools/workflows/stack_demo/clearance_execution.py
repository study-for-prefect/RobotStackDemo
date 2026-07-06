"""Execution helpers for frontier clearance actions."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

from tools.planning.decision_to_execution import write_json

from .commands import open_gripper_command, push_clear_command, push_preflight_command, run
from .observation_scope import observe_empty_with_scope
from .pick import build_offline_pick_plan, pick_command, place_command, plan_envelope
from .push_context import observed_push_delta_m, protected_stack_templates
from .push_clearing import build_push_execution_plan, object_by_string_id
from .scene import memory_id_for_scene_object, reacquire_target
from .target_recovery import state_with_missing_target
from robot_scene_pipeline.scene_memory import mark_pushed, save_memory


class ClearancePreflightFailed(RuntimeError):
    """A selected clearance action failed MoveIt preflight before hardware motion."""


def _candidate_summary(candidate: dict) -> dict:
    keys = (
        "candidate_id", "action", "action_type", "obstacle_id", "target_object_id",
        "direction_base", "distance_m", "moveit_feasible", "executable_safe",
        "geometry_feasible", "approach_path_safe", "push_swept_safe", "push_end_safe",
        "protected_structure_safe", "task_effective", "direct_clearance_candidate",
        "enabling_clearance_candidate", "exploratory", "automatic_execution_allowed",
        "target_yaw_gain", "direct_target_gain", "enabling_gain", "free_space_gain",
        "blocker_count_reduction", "current_grasp_gain", "post_push_grasp_feasible",
        "enables_blocker_object_id", "enabling_reason", "score", "reason",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def _direction_matches(first: object, second: object, tolerance: float = 1e-6) -> bool:
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        return False
    if len(first) < 2 or len(second) < 2:
        return False
    return all(abs(float(first[index]) - float(second[index])) <= tolerance for index in range(2))


def _assert_nudge_candidate_consistency(selected_action: dict, push_execution_plan: dict) -> None:
    candidate_id = selected_action.get("candidate_id")
    plan_candidate_id = push_execution_plan.get("candidate_id")
    if candidate_id is None:
        raise RuntimeError("Selected nudge action is missing candidate_id.")
    if plan_candidate_id is not None and str(plan_candidate_id) != str(candidate_id):
        raise RuntimeError(
            "Nudge candidate_id mismatch: selected={} plan={}.".format(candidate_id, plan_candidate_id)
        )
    if not _direction_matches(selected_action.get("direction_base"), push_execution_plan.get("direction_base")):
        raise RuntimeError(
            "Nudge direction mismatch for candidate {}: selected={} plan={}.".format(
                candidate_id,
                selected_action.get("direction_base"),
                push_execution_plan.get("direction_base"),
            )
        )


def _build_nudge_execution_plan(args: Any, current_state: dict, held_object: dict, selected_action: dict) -> dict:
    selected_push = _selected_push_from_clearance_action(selected_action)
    push_execution_plan = build_push_execution_plan(
        current_state,
        held_object,
        selected_push,
        args,
        direction_evaluations=[selected_action.get("push_evaluation", {})],
    )
    push_execution_plan["action_type"] = "nudge"
    push_execution_plan["candidate_id"] = selected_action.get("candidate_id")
    _assert_nudge_candidate_consistency(selected_action, push_execution_plan)
    return push_execution_plan


def preflight_nudge_candidate(
    args: Any,
    cycle_dir: str,
    current_state: dict,
    held_object: dict,
    selected_action: dict,
    step_index: int,
) -> dict:
    push_execution_plan = _build_nudge_execution_plan(args, current_state, held_object, selected_action)
    plan_path = os.path.join(
        cycle_dir,
        "clearance_step_{:02d}_candidate_{}_push_plan.json".format(
            step_index,
            str(selected_action.get("candidate_id")).replace(os.sep, "_"),
        ),
    )
    write_json(plan_path, push_execution_plan)
    try:
        run(push_preflight_command(args, plan_path))
    except Exception as exc:
        output = dict(selected_action)
        output["moveit_feasible"] = False
        output["executable_safe"] = False
        output["moveit_preflight_error"] = str(exc)
        output["push_execution_plan_path"] = plan_path
        return output
    output = dict(selected_action)
    output["moveit_feasible"] = True
    output["push_execution_plan_path"] = plan_path
    return output


def _recover_open_gripper(args: Any, cycle_dir: str, reason: str) -> dict:
    report = {"requested": True, "reason": reason, "status": "not_attempted"}
    try:
        run(open_gripper_command(args))
        report["status"] = "opened"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    write_json(os.path.join(cycle_dir, "gripper_open_recovery.json"), report)
    return report


def _build_pick_away_place_plan(current_state: dict, obstacle: dict, selected_action: dict, args: Any) -> dict:
    safe_place = selected_action.get("safe_place_center_m")
    if not isinstance(safe_place, list) or len(safe_place) < 3:
        raise RuntimeError("pick_away selected action is missing safe_place_center_m.")
    release_z = float(safe_place[2]) + float(getattr(args, "release_gap_m", 0.010))
    step = {
        "step": 1,
        "action": "place_relative",
        "object_id": obstacle.get("id"),
        "object_label": obstacle.get("label"),
        "reference_object_id": None,
        "reference_label": None,
        "relative_position": "frontier_safe_place",
        "reason": "Place cleared obstacle at a safe table location.",
        "status": "planned",
        "coordinate_frame": "base_frame",
        "coordinate_source": "obstruction_frontier_safe_place",
        "target_position_m": [round(float(safe_place[0]), 5), round(float(safe_place[1]), 5), round(release_z, 5)],
        "approach_position_m": [
            round(float(safe_place[0]), 5),
            round(float(safe_place[1]), 5),
            round(release_z + float(args.approach_height_m), 5),
        ],
        "target_yaw_deg": float(selected_action.get("selected_grasp_yaw_deg") or 0.0),
        "chosen_grasp_yaw_deg": float(selected_action.get("selected_grasp_yaw_deg") or 0.0),
        "target_yaw_valid": True,
        "exact_tool_yaw_required": True,
        "yaw_frame": "base_link",
        "yaw_source": "obstruction_frontier_pick_away",
    }
    return plan_envelope(current_state, step)


def execute_pick_away_and_reobserve(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    selected_action: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict],
    step_index: int,
) -> Tuple[dict, dict, dict]:
    obstacle = object_by_string_id(current_state.get("objects", []), selected_action.get("obstacle_id"))
    obstacle_for_pick = copy.deepcopy(obstacle)
    obstacle_for_pick["selected_grasp_yaw_deg"] = selected_action.get("selected_grasp_yaw_deg")
    obstacle_for_pick["grasp_yaw_source"] = "obstruction_frontier_pick_away"
    pick_plan_path = os.path.join(cycle_dir, "pick_away_step_{:02d}_pick_plan.json".format(step_index))
    place_plan_path = os.path.join(cycle_dir, "pick_away_step_{:02d}_place_plan.json".format(step_index))
    build_offline_pick_plan(current_state, obstacle_for_pick, pick_plan_path, args)
    place_plan = _build_pick_away_place_plan(current_state, obstacle_for_pick, selected_action, args)
    write_json(place_plan_path, place_plan)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "pending_moveit_preflight",
            "action": "pick_away",
            "pick_plan": pick_plan_path,
            "place_plan": place_plan_path,
            "note": "MoveIt path planning is performed before real pick_away and place execution.",
        },
    )
    if not (args.execute and args.execute_push_clearing):
        write_json(
            os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
            {
                "action": "pick_away",
                "result": "dry_run_only",
                "selected_clearance_action": _candidate_summary(selected_action),
            },
        )
        return current_state, memory, held_template

    runtime["current_stage"] = "pick_away_obstacle_step_{:02d}".format(step_index)
    runtime["held_object_id"] = obstacle.get("id")
    run(pick_command(args, pick_plan_path))
    runtime["current_stage"] = "place_away_obstacle_step_{:02d}".format(step_index)
    run(place_command(args, place_plan_path))
    runtime["held_object_id"] = None
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "passed_and_executed",
            "action": "pick_away",
            "pick_plan": pick_plan_path,
            "place_plan": place_plan_path,
        },
    )
    runtime["current_stage"] = "scoped_observation_after_pick_away"
    observed_state, memory, post_observation = observe_empty_with_scope(
        args,
        os.path.join(cycle_dir, "observation_after_pick_away"),
        runtime,
        memory,
        critical_templates=(
            [held_template]
            + protected_stack_templates(
                current_state,
                base_id,
                previous_locked_stack,
                base_template=base_template,
            )
        ),
        noncritical_templates=[obstacle],
        scope_name="after_pick_away_clearance",
        description="Post-pick-away scoped observation",
    )
    if observed_state is None:
        raise RuntimeError("pick_away completed but no live observation was produced afterward.")
    save_memory(memory, args.memory_json)
    held_object = copy.deepcopy(reacquire_target(observed_state, held_template))
    return observed_state, memory, held_object


def _selected_push_from_clearance_action(selected_action: dict) -> dict:
    return {
        "type": "should_push_away",
        "subject": selected_action.get("obstacle_id"),
        "object": selected_action.get("blocks", [selected_action.get("target_object_id")])[0],
        "source": "obstruction_frontier",
        "reason": selected_action.get("reason"),
        "candidate_id": selected_action.get("candidate_id"),
        "direction_base": selected_action.get("direction_base"),
        "distance_m": selected_action.get("distance_m"),
        "direction_source": selected_action.get("direction_source"),
        "direction_score": selected_action.get("score"),
    }


def execute_nudge_and_reobserve(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: dict,
    held_object: dict,
    selected_action: dict,
    base_id: Any,
    previous_locked_stack: dict,
    base_template: Optional[dict],
    step_index: int,
) -> Tuple[dict, dict, dict]:
    selected_push = _selected_push_from_clearance_action(selected_action)
    obstacle = object_by_string_id(current_state.get("objects", []), selected_push["subject"])
    obstacle_memory_id = memory_id_for_scene_object(memory, obstacle)
    push_execution_plan = _build_nudge_execution_plan(args, current_state, held_object, selected_action)
    push_execution_plan_path = os.path.join(cycle_dir, "push_execution_plan.json")
    write_json(push_execution_plan_path, push_execution_plan)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "pending_moveit_preflight" if (args.execute and args.execute_push_clearing) else "not_requested_dry_run",
            "action": "nudge",
            "push_plan": push_execution_plan_path,
            "gripper_policy": "close_only_after_push_preflight_as_rigid_paddle",
        },
    )
    if not (args.execute and args.execute_push_clearing):
        write_json(os.path.join(cycle_dir, "scene_state_after_action.json"), current_state)
        write_json(
            os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
            {
                "action": "nudge",
                "result": "dry_run_only",
                "selected_clearance_action": _candidate_summary(selected_action),
                "post_push_requirement": "reobserve_and_rerun_clearance_loop",
            },
        )
        return current_state, memory, held_object

    if not selected_action.get("executable_safe"):
        raise ClearancePreflightFailed(
            "Selected nudge candidate {} is not executable_safe.".format(selected_action.get("candidate_id"))
        )

    runtime["current_stage"] = "nudge_single_process_preflight_execute_step_{:02d}".format(step_index)
    try:
        run(push_clear_command(args, push_execution_plan_path))
    except Exception as exc:
        selected_action["moveit_feasible"] = False
        selected_action["executable_safe"] = False
        runtime["current_stage"] = "nudge_failed_before_or_during_motion_step_{:02d}".format(step_index)
        recovery = _recover_open_gripper(args, cycle_dir, "nudge_push_execute_failed")
        write_json(
            os.path.join(cycle_dir, "clearance_verification.json"),
            {
                "status": "push_execute_failed",
                "action": "nudge",
                "source": "moveit_plan_preview_single_process_push_execute",
                "push_plan": push_execution_plan_path,
                "gripper_policy": "open_recovery_requested",
                "error": str(exc),
                "gripper_open_recovery": recovery,
            },
        )
        write_json(
            os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
            {
                "action": "nudge",
                "result": "failed_before_or_during_motion",
                "selected_candidate_id": selected_action.get("candidate_id"),
                "selected_clearance_action": _candidate_summary(selected_action),
                "gripper_open_recovery": recovery,
                "error": str(exc),
            },
        )
        raise RuntimeError("Nudge single-process push execution failed: {}".format(exc))
    push_execution_plan["execution_status"] = "executed"
    write_json(push_execution_plan_path, push_execution_plan)
    write_json(
        os.path.join(cycle_dir, "clearance_verification.json"),
        {
            "status": "passed_and_executed",
            "action": "nudge",
            "source": "moveit_plan_preview_push_preflight",
            "gripper_policy": "closed_rigid_paddle",
        },
    )
    runtime["current_stage"] = "scoped_observation_after_nudge_clearing"
    pushed_state, memory, post_observation = observe_empty_with_scope(
        args,
        os.path.join(cycle_dir, "observation_after_nudge"),
        runtime,
        memory,
        critical_templates=(
            [held_template]
            + protected_stack_templates(
                current_state,
                base_id,
                previous_locked_stack,
                base_template=base_template,
            )
        ),
        noncritical_templates=[obstacle],
        scope_name="after_nudge_clearing",
        description="Post-nudge scoped observation",
    )
    if pushed_state is None:
        raise RuntimeError("Nudge clearing completed but no live observation was produced afterward.")
    observed_delta_m = observed_push_delta_m(obstacle, pushed_state)
    if obstacle_memory_id is not None:
        memory = mark_pushed(
            memory,
            obstacle_memory_id,
            selected_push["direction_base"],
            selected_push.get("distance_m", args.push_clearing_distance_m),
            reason=selected_push.get("reason") or "frontier_nudge_clearance",
            result="success",
            observed_delta_m=observed_delta_m,
        )
    save_memory(memory, args.memory_json)
    try:
        held_object = copy.deepcopy(reacquire_target(pushed_state, held_template))
    except RuntimeError:
        pushed_state, held_object = state_with_missing_target(pushed_state, held_template)
    write_json(
        os.path.join(cycle_dir, "clearance_step_{:02d}_result.json".format(step_index)),
        {
            "action": "nudge",
            "result": "executed_and_reobserved",
            "selected_clearance_action": _candidate_summary(selected_action),
            "post_observation": post_observation,
            "observed_delta_m": observed_delta_m,
        },
    )
    return pushed_state, memory, held_object
