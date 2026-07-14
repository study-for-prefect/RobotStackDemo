"""Pick-place plan construction and execution for semantic task actions."""

from __future__ import annotations

import copy
import os
from typing import Any, List, Tuple

from robot_scene_pipeline.grasp_yaw_search import select_best_grasp
from tools.planning.decision_to_execution import write_json

from .commands import capture_empty_observation, moveit_frame_args, plan_only_command, run
from .pick import build_offline_pick_plan, pick_command, place_command, plan_envelope
from .push_clearing import object_by_string_id


def preflight_pick_place_action(args: Any, cycle_dir: str, state: dict, action: dict, step_index: int) -> dict:
    """Use existing pick/place MoveIt commands in plan-only mode for a VLM pose."""
    output = dict(action)
    if output.get("action_type") == "pick_place":
        output, grasp_report = _validate_pick_grasp_geometry(args, state, output)
        write_json(
            os.path.join(cycle_dir, "task_step_{:02d}_grasp_validation.json".format(step_index)),
            grasp_report,
        )
        if not grasp_report.get("grasp_feasible"):
            output.update({
                "moveit_feasible": False,
                "executable_safe": False,
                "preflight_failure_type": "selected_pick_not_grasp_feasible",
                "moveit_preflight_error": (
                    "selected_pick_not_grasp_feasible: clear blocking objects before retrying pick_place"
                ),
            })
            return output
    pick_path, place_path = build_pick_place_plans(args, cycle_dir, state, output, step_index)
    output.update({"pick_plan_path": pick_path, "place_plan_path": place_path})
    if not getattr(args, "execute", False) and not getattr(args, "moveit_plan_only", False):
        output.update({"moveit_feasible": False, "executable_safe": False}); return output
    try:
        run(plan_only_command(_pick_command(args, pick_path, action)))
        if action.get("action_type") == "pick_reorient_place":
            for waypoint in action.get("reorientation_plan", {}).get("waypoints", []):
                run(plan_only_command(_reorientation_waypoint_command(args, waypoint)))
        run(plan_only_command(_place_command(args, place_path, action)))
    except Exception as exc:
        output.update({"moveit_feasible": False, "executable_safe": False, "moveit_preflight_error": str(exc)}); return output
    if output.get("reorientation_plan"):
        output["reorientation_plan"] = copy.deepcopy(output["reorientation_plan"])
        output["reorientation_plan"]["all_waypoints_collision_checked"] = True
        output["reorientation_plan"]["collision_free"] = True
        for waypoint in output["reorientation_plan"].get("waypoints", []):
            waypoint["collision_free"] = True
            waypoint["moveit_plan_only_checked"] = True
    output.update({"moveit_feasible": True, "executable_safe": True}); return output


def _validate_pick_grasp_geometry(args: Any, state: dict, action: dict) -> Tuple[dict, dict]:
    """Select only a gripper yaw whose open fingers clear every visible scene object."""
    obj = object_by_string_id(state.get("objects", []), action["object_id"])
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
    output = dict(action)
    output["physical_grasp_validation"] = _compact_grasp_report(report)
    selected_yaw = report.get("selected_grasp_yaw_deg")
    if selected_yaw is not None:
        output["selected_grasp_yaw_deg"] = float(selected_yaw)
        output["grasp_yaw_source"] = report.get("selected_grasp_source") or "task_pick_physical_validation"
    return output, report


def _compact_grasp_report(report: dict) -> dict:
    return {
        key: report.get(key)
        for key in (
            "grasp_feasible", "selected_grasp_yaw_deg", "selected_grasp_axis_delta_deg",
            "selected_grasp_source", "feasible_yaw_intervals_deg", "blocked_yaw_intervals_deg",
            "all_grasps_blocked", "blocking_objects", "parameters",
            "robust_feasible_yaw_intervals_deg",
            "rejected_narrow_feasible_yaw_intervals_deg",
        )
    }


def execute_pick_place_and_reobserve(
    args: Any, cycle_dir: str, runtime: dict, state: dict, action: dict, step_index: int,
) -> Tuple[dict, dict]:
    """Execute a preflighted semantic pick_place, then require a fresh scene."""
    pick_path, place_path = build_pick_place_plans(args, cycle_dir, state, action, step_index)
    if not getattr(args, "execute", False):
        return state, {"status": "dry_run_only", "pick_plan": pick_path, "place_plan": place_path}
    run(_pick_command(args, pick_path, action))
    if action.get("action_type") == "pick_reorient_place":
        for waypoint in action.get("reorientation_plan", {}).get("waypoints", []):
            run(_reorientation_waypoint_command(args, waypoint))
    run(_place_command(args, place_path, action))
    runtime["current_stage"] = "semantic_task_observation_after_pick_place"
    observed = capture_empty_observation(args, os.path.join(cycle_dir, "observation_after_pick_place"), None)
    if observed is None:
        raise RuntimeError("pick_place executed but no post-action observation was produced.")
    return observed, {
        "status": "executed_and_reobserved", "pick_plan": pick_path, "place_plan": place_path,
        "action_type": action.get("action_type"),
        "source_orientation_xyzw": (action.get("source_pose_base") or {}).get("orientation_xyzw"),
        "target_orientation_xyzw": (action.get("target_pose_base") or {}).get("orientation_xyzw"),
        "rotation_axis": (action.get("reorientation_plan") or {}).get("rotation_axis_tool"),
        "rotation_angle_deg": (action.get("reorientation_plan") or {}).get("rotation_angle_deg"),
        "intermediate_orientations": (action.get("reorientation_plan") or {}).get("intermediate_orientations", []),
        "flip_required": bool((action.get("fused_orientation_result") or {}).get("flip_required")),
        "fused_orientation_result": action.get("fused_orientation_result"),
        "gripper_stability_check": "GF225 close feedback handled by existing pick command",
        "post_place_orientation_result": "pending_fresh_observation",
    }


def build_pick_place_plans(args: Any, cycle_dir: str, state: dict, action: dict, step_index: int) -> Tuple[str, str]:
    """Build existing pick plan plus a VLM-coordinate place plan without stack assumptions."""
    obj = copy.deepcopy(object_by_string_id(state.get("objects", []), action["object_id"]))
    if action.get("selected_grasp_yaw_deg") is not None:
        obj["selected_grasp_yaw_deg"] = float(action["selected_grasp_yaw_deg"])
        obj["grasp_yaw_source"] = action.get("grasp_yaw_source") or "task_pick_physical_validation"
        grasp_report = action.get("physical_grasp_validation") or {}
        obj["selected_grasp_axis_delta_deg"] = grasp_report.get("selected_grasp_axis_delta_deg")
        obj["feasible_yaw_intervals_deg"] = grasp_report.get("feasible_yaw_intervals_deg", [])
        obj["blocked_yaw_intervals_deg"] = grasp_report.get("blocked_yaw_intervals_deg", [])
    if action.get("action_type") == "pick_reorient_place" and action.get("grasp_target_center_base_m"):
        obj["geometry_center_m"] = list(action["grasp_target_center_base_m"])
        obj["center_3d_base_m"] = list(action["grasp_target_center_base_m"])
        obj["grasp_target_source"] = "task_reorientation_semantic_grasp_region"
    pick_path = os.path.join(cycle_dir, "task_step_{:02d}_pick_plan.json".format(step_index))
    place_path = os.path.join(cycle_dir, "task_step_{:02d}_place_plan.json".format(step_index))
    pick_plan = build_offline_pick_plan(state, obj, pick_path, args)
    source_tool_orientation = (
        (action.get("reorientation_plan") or {}).get("source_tool_orientation_xyzw")
        or action.get("grasp_tool_orientation_xyzw")
    )
    if source_tool_orientation is not None:
        pick_plan["steps"][0]["target_orientation_xyzw"] = list(source_tool_orientation)
        pick_plan["steps"][0]["orientation_source"] = "object_orientation_and_grasp_transform"
        write_json(pick_path, pick_plan)
    pose = action["target_pose_base"]
    nominal_position, position, release_gap = _task_place_release_position(args, action)
    target_yaw_deg = (
        float(pose["yaw_rad"]) * 180.0 / 3.141592653589793
        if pose.get("yaw_rad") is not None else None
    )
    step = {
        "step": 1, "action": "place_relative", "status": "planned",
        "object_id": obj.get("id"), "object_label": obj.get("label"),
        "coordinate_frame": "base_link",
        "coordinate_source": (
            "vlm_task_action_target_pose_base+organize_release_gap"
            if release_gap > 0.0 else "vlm_task_action_target_pose_base"
        ),
        "nominal_target_position_m": nominal_position,
        "release_gap_m": release_gap,
        "target_position_m": position, "approach_position_m": [position[0], position[1], position[2] + float(args.approach_height_m)],
        "target_yaw_deg": target_yaw_deg,
        "target_yaw_valid": target_yaw_deg is not None,
        "chosen_grasp_yaw_deg": target_yaw_deg,
        "target_orientation_xyzw": (
            (action.get("reorientation_plan") or {}).get("target_tool_orientation_xyzw")
            or action.get("target_tool_orientation_xyzw")
        ),
        "exact_tool_yaw_required": target_yaw_deg is not None,
        "yaw_frame": "base_link",
        "yaw_source": "vlm_task_action" if pose.get("yaw_rad") is not None else "house_frame_code_geometry",
    }
    write_json(place_path, plan_envelope(state, step))
    return pick_path, place_path


def _task_place_release_position(args: Any, action: dict) -> Tuple[List[float], List[float], float]:
    """Raise organize releases above the nominal settled center to avoid table contact."""
    pose = action["target_pose_base"]
    nominal = [float(value) for value in pose["position_m"]]
    release_gap = 0.0
    if action.get("group_id") is not None:
        release_gap = max(0.0, float(getattr(args, "release_gap_m", 0.010)))
    release = list(nominal)
    release[2] += release_gap
    return nominal, release, release_gap


def _place_command(args: Any, place_path: str, action: dict) -> List[str]:
    command = place_command(args, place_path)
    code_tool_orientation = (
        (action.get("reorientation_plan") or {}).get("target_tool_orientation_xyzw")
        or action.get("target_tool_orientation_xyzw")
    )
    if code_tool_orientation is not None:
        command[command.index("--orientation-mode") + 1] = "step-quaternion"
    return command


def _pick_command(args: Any, pick_path: str, action: dict) -> List[str]:
    command = pick_command(args, pick_path)
    if action.get("grasp_tool_orientation_xyzw") is not None:
        command[command.index("--orientation-mode") + 1] = "step-quaternion"
    return command


def _reorientation_waypoint_command(args: Any, waypoint: dict) -> List[str]:
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--hover-only", "--hover-target-base", *[str(value) for value in waypoint["position_base_m"]],
        "--hover-orientation-xyzw", *[str(value) for value in waypoint["orientation_xyzw"]],
        "--tcp-offset-tool", *[str(value) for value in args.tcp_offset_tool],
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        "--max-joint-delta", str(getattr(args, "max_pre_rotate_joint_delta", 1.2)),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        *moveit_frame_args(args), "--execute",
    ]
    if getattr(args, "yes", False):
        command.append("--yes")
    return command
