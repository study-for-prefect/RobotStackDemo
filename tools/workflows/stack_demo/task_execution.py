"""Pick-place plan construction and execution for semantic task actions."""

from __future__ import annotations

import copy
import os
from typing import Any, Tuple

from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, run
from .pick import build_offline_pick_plan, pick_command, place_command, plan_envelope
from .push_clearing import object_by_string_id


def preflight_pick_place_action(args: Any, cycle_dir: str, state: dict, action: dict, step_index: int) -> dict:
    """Use existing pick/place MoveIt commands in plan-only mode for a VLM pose."""
    pick_path, place_path = build_pick_place_plans(args, cycle_dir, state, action, step_index)
    output = dict(action); output.update({"pick_plan_path": pick_path, "place_plan_path": place_path})
    if not getattr(args, "execute", False):
        output.update({"moveit_feasible": False, "executable_safe": False}); return output
    try:
        run(_plan_only(pick_command(args, pick_path))); run(_plan_only(place_command(args, place_path)))
    except Exception as exc:
        output.update({"moveit_feasible": False, "executable_safe": False, "moveit_preflight_error": str(exc)}); return output
    output.update({"moveit_feasible": True, "executable_safe": True}); return output


def execute_pick_place_and_reobserve(
    args: Any, cycle_dir: str, runtime: dict, state: dict, action: dict, step_index: int,
) -> Tuple[dict, dict]:
    """Execute a preflighted semantic pick_place, then require a fresh scene."""
    pick_path, place_path = build_pick_place_plans(args, cycle_dir, state, action, step_index)
    if not getattr(args, "execute", False):
        return state, {"status": "dry_run_only", "pick_plan": pick_path, "place_plan": place_path}
    run(pick_command(args, pick_path)); run(place_command(args, place_path))
    runtime["current_stage"] = "semantic_task_observation_after_pick_place"
    observed = capture_empty_observation(args, os.path.join(cycle_dir, "observation_after_pick_place"), None)
    if observed is None:
        raise RuntimeError("pick_place executed but no post-action observation was produced.")
    return observed, {"status": "executed_and_reobserved", "pick_plan": pick_path, "place_plan": place_path}


def build_pick_place_plans(args: Any, cycle_dir: str, state: dict, action: dict, step_index: int) -> Tuple[str, str]:
    """Build existing pick plan plus a VLM-coordinate place plan without stack assumptions."""
    obj = copy.deepcopy(object_by_string_id(state.get("objects", []), action["object_id"]))
    pick_path = os.path.join(cycle_dir, "task_step_{:02d}_pick_plan.json".format(step_index))
    place_path = os.path.join(cycle_dir, "task_step_{:02d}_place_plan.json".format(step_index))
    build_offline_pick_plan(state, obj, pick_path, args)
    pose = action["target_pose_base"]; position = [float(value) for value in pose["position_m"]]
    step = {
        "step": 1, "action": "place_relative", "object_id": obj.get("id"), "object_label": obj.get("label"),
        "coordinate_frame": "base_link", "coordinate_source": "vlm_task_action_target_pose_base",
        "target_position_m": position, "approach_position_m": [position[0], position[1], position[2] + float(args.approach_height_m)],
        "target_yaw_deg": float(pose["yaw_rad"]) * 180.0 / 3.141592653589793,
        "target_yaw_valid": True, "chosen_grasp_yaw_deg": float(pose["yaw_rad"]) * 180.0 / 3.141592653589793,
        "exact_tool_yaw_required": True, "yaw_frame": "base_link", "yaw_source": "vlm_task_action",
    }
    write_json(place_path, plan_envelope(state, step))
    return pick_path, place_path


def _plan_only(command: list) -> list:
    output = list(command)
    for flag, count in (("--execute", 0), ("--enable-gripper", 0), ("--skip-gripper-init", 0), ("--gripper-port", 1)):
        while flag in output:
            index = output.index(flag); del output[index:index + count + 1]
    return output
