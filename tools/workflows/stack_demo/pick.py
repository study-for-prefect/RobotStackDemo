"""Pick-plan construction, dry-run scene simulation, and pick motion commands."""

import copy
import math
from types import SimpleNamespace

from robot_scene_pipeline.xy_correction import apply_step_xyz_correction, load_xy_correction
from tools.planning.build_geometry_pick_plan import normalize_equivalent_yaw, set_stack_demo_yaw
from tools.planning.decision_to_execution import compile_plan, write_json
from tools.workflows.two_stage_visual_pick import camera_vector_to_base

from .commands import moveit_frame_args
from .scene import require_geometry_object

def plan_envelope(state, step):
    return {
        "schema_version": "robot_execution_plan_v1",
        "frame_id": state.get("frame_id"),
        "timestamp": state.get("timestamp"),
        "execution_status": "not_executed",
        "coordinate_convention": {
            "frame": "base_link",
            "unit": "meter",
            "x": "robot base x axis",
            "y": "robot base y axis",
            "z": "positive upward from robot base",
        },
        "steps": [step],
        "safety_checks": [
            "lock this one-block place target before closing the gripper",
            "do not reacquire the base while an object is held",
            "stop if geometry_center_m or dynamic stack top is invalid",
        ],
    }


def apply_selected_grasp_yaw(step, obj):
    selected_yaw = obj.get("selected_grasp_yaw_deg")
    if selected_yaw is None:
        return
    yaw = float(selected_yaw)
    label = str(step.get("object_label") or obj.get("label") or "").lower()
    if "square" in label and step.get("target_yaw_deg") is not None:
        period = float(step.get("yaw_equivalence_period_deg", 90.0))
        base_yaw = float(step["target_yaw_deg"])
        delta = abs(normalize_equivalent_yaw(yaw - base_yaw, period))
        step["adaptive_selected_grasp_yaw_deg"] = yaw
        step["adaptive_grasp_yaw_delta_deg"] = float(delta)
        step["target_yaw_deg"] = yaw
        step["chosen_grasp_yaw_deg"] = yaw
        step["selected_grasp_yaw_deg"] = yaw
        step["target_yaw_valid"] = True
        step["exact_tool_yaw_required"] = True
        step["yaw_equivalence_period_deg"] = 180.0
        step["yaw_frame"] = "base_link"
        step["yaw_source"] = obj.get("grasp_yaw_source") or "adaptive_grasp_yaw_search"
        step["selected_grasp_axis_delta_deg"] = obj.get("selected_grasp_axis_delta_deg")
        step["feasible_yaw_intervals_deg"] = obj.get("feasible_yaw_intervals_deg", [])
        step["blocked_yaw_intervals_deg"] = obj.get("blocked_yaw_intervals_deg", [])
        return
    step["target_yaw_deg"] = yaw
    step["chosen_grasp_yaw_deg"] = yaw
    step["selected_grasp_yaw_deg"] = yaw
    step["target_yaw_valid"] = True
    step["exact_tool_yaw_required"] = True
    step["yaw_equivalence_period_deg"] = 180.0
    step["yaw_frame"] = "base_link"
    step["yaw_source"] = obj.get("grasp_yaw_source") or "adaptive_grasp_yaw_search"
    step["selected_grasp_axis_delta_deg"] = obj.get("selected_grasp_axis_delta_deg")
    step["feasible_yaw_intervals_deg"] = obj.get("feasible_yaw_intervals_deg", [])
    step["blocked_yaw_intervals_deg"] = obj.get("blocked_yaw_intervals_deg", [])


def _finite_float(value, default=None):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _clamp_pick_target_height(step, obj, args):
    center = require_geometry_object(obj)["geometry_center_m"]
    size = obj.get("dimensions_m")
    target = step.get("target_position_m")
    if not isinstance(size, list) or len(size) < 3 or not isinstance(target, list) or len(target) < 3:
        return
    height = abs(float(size[2]))
    if height <= 0.0:
        return
    top_z = _finite_float(obj.get("top_z_base_m"), None)
    if top_z is None:
        top_z = float(center[2]) + 0.5 * height
    bottom_z = top_z - height
    margin = min(
        max(0.0, _finite_float(getattr(args, "pick_target_z_margin_m", 0.002), 0.002)),
        0.25 * height,
    )
    min_z = bottom_z + margin
    max_z = top_z - margin
    raw_z = float(target[2])
    clamped_z = min(max(raw_z, min_z), max_z)
    if abs(clamped_z - raw_z) > 1e-9:
        step["target_position_m"][2] = round(clamped_z, 5)
        step["approach_position_m"][2] = round(clamped_z + float(args.approach_height_m), 5)
    step["grasp_final_xyz_m"] = [round(float(value), 5) for value in step["target_position_m"][:3]]
    step["pick_grasp_height"] = {
        "object_bottom_z_base_m": round(bottom_z, 6),
        "object_top_z_base_m": round(top_z, 6),
        "safe_min_z_base_m": round(min_z, 6),
        "safe_max_z_base_m": round(max_z, 6),
        "raw_target_z_base_m": round(raw_z, 6),
        "final_target_z_base_m": round(float(step["target_position_m"][2]), 6),
        "margin_m": round(margin, 6),
        "clamped": bool(abs(clamped_z - raw_z) > 1e-9),
    }
    if step["pick_grasp_height"]["clamped"]:
        step["grasp_height_adjustment_reason"] = "target_z_outside_object_safe_band"


def build_offline_pick_plan(state, obj, output_path, args):
    decision = {"action_plan": [{"step": 1, "action": "pick", "object_id": int(obj["id"])}]}
    plan_args = SimpleNamespace(
        place_offset_m=0.08,
        approach_height_m=args.approach_height_m,
        pick_target_lift_m=args.pick_target_lift_m,
        place_target_lift_m=0.0,
        left_right_axis="y",
        left_direction_sign="positive",
        front_back_axis="x",
        front_direction_sign="positive",
    )
    plan = compile_plan(decision, state, plan_args)
    step = plan["steps"][0]
    center = require_geometry_object(obj)["geometry_center_m"]
    if step.get("status") != "planned":
        raise RuntimeError("Could not build offline stack pick plan for object {}.".format(obj["id"]))
    step["geometry_center_m"] = [round(float(value), 5) for value in center[:3]]
    step["geometry_frame"] = "base_link"
    step["pointcloud_geometry_valid"] = bool(obj.get("pointcloud_geometry_valid"))
    step["object_dimensions_m"] = [float(value) for value in obj["dimensions_m"][:3]]
    for key in ("reacquire_source", "geometry_fallback_source", "pointcloud_geometry_reason"):
        if obj.get(key) is not None:
            step[key] = obj.get(key)
    target_z = float(center[2]) + float(args.pick_target_lift_m)
    step["target_position_m"] = [
        round(float(center[0]), 5),
        round(float(center[1]), 5),
        round(target_z, 5),
    ]
    step["approach_position_m"] = [
        round(float(center[0]), 5),
        round(float(center[1]), 5),
        round(target_z + float(args.approach_height_m), 5),
    ]
    step["pick_target_lift_m"] = float(args.pick_target_lift_m)
    step["coordinate_source"] = "required_geometry_center_m"
    correction_path = getattr(args, "calibration_json", "") or args.xy_correction_json
    correction = load_xy_correction(correction_path)
    configured_bias = correction.get("grasp_base_bias_m", correction.get("grasp_xy_bias_m", [0.0, 0.0, 0.0]))
    if len(configured_bias) == 2:
        configured_bias = [float(configured_bias[0]), float(configured_bias[1]), 0.0]
    grasp_bias_base = getattr(args, "grasp_bias_base", [0.0, 0.0])
    grasp_bias_camera = getattr(args, "grasp_bias_camera", [0.0, 0.0])
    grasp_bias_camera_base = [0.0, 0.0, 0.0]
    if any(float(value) != 0.0 for value in grasp_bias_camera):
        grasp_bias_camera_base = camera_vector_to_base(
            args.tf_json,
            [float(grasp_bias_camera[0]), float(grasp_bias_camera[1]), 0.0],
        )
    correction["grasp_base_bias_m"] = [
        float(configured_bias[0]) + float(grasp_bias_base[0]) + float(grasp_bias_camera_base[0]),
        float(configured_bias[1]) + float(grasp_bias_base[1]) + float(grasp_bias_camera_base[1]),
        float(configured_bias[2]),
    ]
    apply_step_xyz_correction(step, step["target_position_m"], correction, kind="grasp")
    _clamp_pick_target_height(step, obj, args)
    step["grasp_offset_from_geometry_center_m"] = step["grasp_correction_delta_m"]
    step["grasp_bias_camera_xy_m"] = [float(value) for value in grasp_bias_camera]
    step["grasp_bias_camera_base_xy_m"] = [float(value) for value in grasp_bias_camera_base[:2]]
    set_stack_demo_yaw(
        step,
        obj,
        args.fixed_square_yaw_deg,
        args.stack_square_yaw_mode,
        args.grasp_axis,
        args.gripper_yaw_offset_deg,
        args.square_yaw_snap_tolerance_deg,
    )
    apply_selected_grasp_yaw(step, obj)
    write_json(output_path, plan)
    return plan


def simulate_held_state(state, held_id):
    output = copy.deepcopy(state)
    output["objects"] = [obj for obj in output.get("objects", []) if int(obj.get("id", -1)) != int(held_id)]
    return output


def simulate_placed_state(held_state, held_object, place_step):
    output = copy.deepcopy(held_state)
    placed = copy.deepcopy(held_object)
    tcp_xy = place_step["tcp_place_xy_base_m"]
    offset_xy = place_step["object_offset_base_xy_m"]
    placed["geometry_center_m"] = [
        float(tcp_xy[0]) + float(offset_xy[0]),
        float(tcp_xy[1]) + float(offset_xy[1]),
        float(place_step["release_z_base_m"]),
    ]
    placed["center_3d_base_m"] = list(placed["geometry_center_m"])
    placed["pointcloud_geometry_valid"] = True
    placed["geometry_frame"] = "base_link"
    output.setdefault("objects", []).append(placed)
    return output


def pick_motion_command(args, plan_path, path_mode, enable_gripper):
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--plan-json", plan_path, "--path-mode", path_mode,
        "--orientation-mode", "object-yaw",
        "--grasp-axis", "long",
        "--pre-rotate-before-translation",
        "--pre-rotate-strategy", "joint-wrist3",
        "--pre-rotate-wrist-yaw-sign", args.pre_rotate_wrist_yaw_sign,
        "--pre-rotate-wrist-direction", "auto",
        "--max-pre-rotate-joint-delta", str(args.max_pre_rotate_joint_delta),
        "--pre-rotate-velocity", str(getattr(args, "pre_rotate_velocity", 0.20)),
        "--pre-rotate-acceleration", str(getattr(args, "pre_rotate_acceleration", 0.20)),
        "--max-grasp-yaw-error-deg", str(args.max_grasp_yaw_error_deg),
        "--max-grasp-orientation-error-deg", str(args.max_grasp_orientation_error_deg),
        "--tcp-offset-tool", *[str(value) for value in args.tcp_offset_tool],
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        *moveit_frame_args(args),
        "--execute",
    ]
    if enable_gripper:
        command.extend(["--enable-gripper", "--gripper-port", args.gripper_port, "--skip-gripper-init"])
    if args.yes:
        command.append("--yes")
    return command


def pick_approach_command(args, plan_path):
    return pick_motion_command(args, plan_path, "approach", enable_gripper=False)


def pick_command(args, plan_path):
    return pick_motion_command(args, plan_path, "full", enable_gripper=True)


def place_command(args, plan_path):
    command = [
        args.ros_python, "tools/robot/moveit_plan_preview.py",
        "--plan-json", plan_path, "--path-mode", "full",
        "--orientation-mode", "object-yaw",
        "--grasp-axis", "long",
        "--pre-rotate-before-translation",
        "--pre-rotate-strategy", "joint-wrist3",
        "--pre-rotate-wrist-yaw-sign", args.pre_rotate_wrist_yaw_sign,
        "--pre-rotate-wrist-direction", "auto",
        "--max-pre-rotate-joint-delta", str(args.max_pre_rotate_joint_delta),
        "--pre-rotate-velocity", str(getattr(args, "pre_rotate_velocity", 0.20)),
        "--pre-rotate-acceleration", str(getattr(args, "pre_rotate_acceleration", 0.20)),
        "--tcp-offset-tool", *[str(value) for value in args.tcp_offset_tool],
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        "--place-on-top-velocity", str(args.place_velocity),
        "--place-on-top-acceleration", str(args.place_acceleration),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        *moveit_frame_args(args),
        "--enable-gripper", "--gripper-port", args.gripper_port,
        "--skip-gripper-init", "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def place_approach_command(args, plan_path):
    command = place_command(args, plan_path)
    command[command.index("--path-mode") + 1] = "approach"
    for value in ("--enable-gripper", "--gripper-port", args.gripper_port, "--skip-gripper-init"):
        if value in command:
            command.remove(value)
    return command
