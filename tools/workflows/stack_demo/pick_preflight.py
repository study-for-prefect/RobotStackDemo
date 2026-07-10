"""MoveIt plan-only validation for a VLM-selected normal pick."""

from __future__ import annotations

import copy
import os
from typing import Any

from .commands import run
from .pick import build_offline_pick_plan, pick_command
from .push_clearing import object_by_string_id


def preflight_pick_action(
    args: Any, cycle_dir: str, current_state: dict, selected_action: dict, step_index: int,
) -> dict:
    """Run the existing pick planner in plan-only mode for the VLM-selected object."""
    selected_object = object_by_string_id(current_state.get("objects", []), selected_action.get("object_id"))
    object_for_pick = copy.deepcopy(selected_object)
    object_for_pick["selected_grasp_yaw_deg"] = selected_action.get("selected_grasp_yaw_deg")
    object_for_pick["grasp_yaw_source"] = "post_vlm_physical_validation"
    plan_path = os.path.join(cycle_dir, "action_step_{:02d}_vlm_pick_plan.json".format(step_index))
    build_offline_pick_plan(current_state, object_for_pick, plan_path, args)
    output = dict(selected_action)
    output["pick_plan_path"] = plan_path
    try:
        run(_plan_only_pick_command(pick_command(args, plan_path)))
    except Exception as exc:
        output["moveit_feasible"] = False
        output["executable_safe"] = False
        output["moveit_preflight_error"] = str(exc)
        return output
    output["moveit_feasible"] = True
    output["executable_safe"] = True
    return output


def _plan_only_pick_command(command: list) -> list:
    output = list(command)
    for flag, value_count in (("--execute", 0), ("--enable-gripper", 0), ("--skip-gripper-init", 0), ("--gripper-port", 1)):
        while flag in output:
            index = output.index(flag)
            del output[index:index + 1 + value_count]
    return output
