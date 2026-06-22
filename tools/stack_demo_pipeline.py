#!/usr/bin/env python3
"""Closed-loop 3-5 block stacking demo built on the existing scripts."""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from types import SimpleNamespace


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robot_scene_pipeline.llm_scene_reasoner import rule_stack_blocks_decision, validate_stack_blocks_decision
from robot_scene_pipeline.stack_state import estimate_stack_state, verify_stack_growth
from tools.build_geometry_pick_plan import find_object, set_stack_demo_yaw
from tools.decision_to_execution import compile_place_on_top, compile_plan, write_json
from tools.two_stage_visual_pick import build_corrected_plan, camera_optical_vector_to_base, camera_vector_to_base
from tools.xy_correction import apply_step_xy_correction, load_xy_correction


def parse_args():
    parser = argparse.ArgumentParser(description="Run a per-block closed-loop stack demo.")
    parser.add_argument("--instruction", default="把积木按指定顺序叠起来")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runtime"))
    parser.add_argument("--stack-decision-json", default="")
    parser.add_argument(
        "--force-llm-decision",
        action="store_true",
        help=(
            "Always call the VLM/LLM to parse the initial stack order. "
            "Detected ids are still repaired from explicit color instructions before execution."
        ),
    )
    parser.add_argument("--offline-scene-state", default="")
    parser.add_argument("--base-object-id", type=int, default=None)
    parser.add_argument("--stack-order", nargs="+", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--conda-env", default="yolo")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--model", default="qwen2.5vl:7b-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--detector-weight", default="models/yolo/weights/best.pt")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--ready-pose-json", default="config/rectangle_ready_pose.json")
    parser.add_argument("--init-stable-wait-s", type=float, default=1.0)
    parser.add_argument("--initial-observation-retry-count", type=int, default=1)
    parser.add_argument("--initial-observation-stable-wait-s", type=float, default=0.5)
    parser.add_argument("--xy-correction-json", default="")
    parser.add_argument("--search-radius-m", type=float, default=0.06)
    parser.add_argument("--close-stack-search-radius-m", type=float, default=0.04)
    parser.add_argument("--target-exclusion-radius-m", type=float, default=0.035)
    parser.add_argument("--min-pick-place-xy-distance-m", type=float, default=0.04)
    parser.add_argument("--max-place-second-snapshot-correction-m", type=float, default=0.015)
    parser.add_argument("--held-base-top-z-tolerance-m", type=float, default=0.015)
    parser.add_argument("--approach-height-m", type=float, default=0.05)
    parser.add_argument(
        "--pick-observation-offset-base",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Deprecated compatibility option; close-observation retry now uses optical camera offsets only.",
    )
    parser.add_argument(
        "--pick-observation-offset-camera",
        nargs=3,
        type=float,
        default=[0.0, 0.04, 0.0],
        help="Optical-camera offset used only when the close target observation misses the object.",
    )
    parser.add_argument(
        "--place-observation-offset-camera",
        nargs=3,
        type=float,
        default=[0.0, 0.04, 0.0],
        help="Optical-camera offset used only when the held-object close base observation misses the base.",
    )
    parser.add_argument("--grasp-bias-base", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--grasp-bias-camera", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--pick-target-lift-m", type=float, default=0.005)
    parser.add_argument("--release-gap-m", type=float, default=0.001)
    parser.add_argument("--place-top-z-bias-m", type=float, default=0.0)
    parser.add_argument("--fixed-square-yaw-deg", type=float, default=0.0)
    parser.add_argument("--stack-square-yaw-mode", choices=("detected", "fixed"), default="detected")
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--grasp-axis", choices=("long", "short"), default="long")
    parser.add_argument("--gripper-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--max-grasp-yaw-error-deg", type=float, default=5.0)
    parser.add_argument("--max-grasp-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--pre-rotate-wrist-yaw-sign", choices=("positive", "negative"), default="negative")
    parser.add_argument("--max-pre-rotate-joint-delta", type=float, default=3.1416)
    parser.add_argument("--max-second-snapshot-correction-m", type=float, default=0.02)
    parser.add_argument("--second-snapshot-max-z-error-m", type=float, default=0.06)
    parser.add_argument("--max-grasp-offset-m", type=float, default=0.05)
    parser.add_argument("--second-snapshot-stable-wait-s", type=float, default=0.5)
    parser.add_argument("--second-snapshot-retry-offset-camera", nargs=3, type=float, default=[0.0, 0.04, 0.0])
    parser.add_argument("--close-observation-retry-count", type=int, default=1)
    parser.add_argument("--close-observation-retry-base-offset", nargs=3, type=float, default=[-0.02, 0.0, 0.0])
    parser.add_argument("--place-yaw-strategy", choices=("stack", "base", "held"), default="base")
    parser.add_argument("--place-center-strategy", choices=("top", "base"), default="top")
    parser.add_argument("--max-stack-top-center-offset-m", type=float, default=0.015)
    parser.add_argument("--object-offset-tool", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--place-bias-base", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--max-place-bias-base-m", type=float, default=0.01)
    parser.add_argument("--tool-z-offset", type=float, default=0.15)
    parser.add_argument("--tool-offset-base", nargs=3, type=float, default=[-0.015, 0.0, 0.0])
    parser.add_argument(
        "--grasp-tool-offset-local",
        nargs=2,
        type=float,
        default=None,
        help="Tool0 XY offset from the visual grasp center in the selected grasp-yaw frame.",
    )
    parser.add_argument("--velocity", type=float, default=0.08)
    parser.add_argument("--acceleration", type=float, default=0.08)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.20)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.20)
    parser.add_argument("--ready-max-joint-delta", type=float, default=1.30)
    parser.add_argument("--place-velocity", type=float, default=0.03)
    parser.add_argument("--place-acceleration", type=float, default=0.03)
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(command):
    print("\n$ {}".format(" ".join(command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def tf_lookup_command(args, require_tool=True):
    command = [
        args.ros_python,
        "tools/tf_lookup_json.py",
        "--output",
        args.tf_json,
        "--base-frame",
        getattr(args, "base_frame", "base_link"),
        "--camera-frame",
        getattr(args, "camera_frame", "camera_link"),
        "--tool-frame",
        getattr(args, "tool_frame", "tool0"),
        "--timeout",
        str(getattr(args, "tf_timeout", 8.0)),
        "--once",
    ]
    if require_tool:
        command.append("--require-tool")
    return command


def moveit_frame_args(args):
    return [
        "--base-link",
        getattr(args, "base_frame", "base_link"),
        "--end-effector",
        getattr(args, "tool_frame", "tool0"),
    ]


def failure_state_path(args):
    return os.path.join(args.output_dir, "failure_state.json")


def pose_command(args, pose_json):
    command = [
        args.ros_python, "tools/moveit_plan_preview.py",
        "--ready-only", "--ready-joint-pose-json", pose_json,
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        "--max-joint-delta", str(getattr(args, "ready_max_joint_delta", 1.30)),
        *moveit_frame_args(args),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def open_gripper_command(args):
    command = [
        args.ros_python, "tools/moveit_plan_preview.py",
        "--gripper-open-only", "--enable-gripper",
        "--gripper-port", args.gripper_port, "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def init_ready_pose(args):
    """Put the empty-gripper robot in the configured observation pose."""
    print("\ninit_ready_pose: moving to ready pose", flush=True)
    if args.execute:
        if not args.ready_pose_json or not os.path.isfile(args.ready_pose_json):
            raise RuntimeError(
                "Ready pose JSON not found: {}. Set --ready-pose-json/READY_POSE_JSON to an existing init pose.".format(
                    args.ready_pose_json
                )
            )
        run(pose_command(args, args.ready_pose_json))

    print("init_ready_pose: opening gripper", flush=True)
    if args.execute:
        run(open_gripper_command(args))
        if args.init_stable_wait_s > 0:
            time.sleep(float(args.init_stable_wait_s))
    print("init_ready_pose: stable, start snapshot", flush=True)


def capture_empty_observation(args, output_dir, held_object_id):
    if held_object_id is not None:
        raise RuntimeError("Snapshot forbidden while holding object {}.".format(held_object_id))
    init_ready_pose(args)
    if args.offline_scene_state:
        return None
    run(tf_lookup_command(args))
    run(snapshot_command(args, output_dir, stack_reasoning=False))
    return load_json(os.path.join(output_dir, "private_scene_state.json"))


def capture_empty_current_pose(args, output_dir, held_object_id, allow_holding=False):
    if held_object_id is not None and not allow_holding:
        raise RuntimeError("Second snapshot forbidden while holding object {}.".format(held_object_id))
    if args.offline_scene_state:
        return None
    if args.second_snapshot_stable_wait_s > 0:
        time.sleep(float(args.second_snapshot_stable_wait_s))
    run(tf_lookup_command(args))
    run(snapshot_command(args, output_dir, stack_reasoning=False))
    return load_json(os.path.join(output_dir, "private_scene_state.json"))


def relative_translate_command(args, offset_base):
    command = [
        args.ros_python, "tools/moveit_plan_preview.py",
        "--relative-tool-translation-base", *[str(value) for value in offset_base],
        "--velocity", str(args.velocity), "--acceleration", str(args.acceleration),
        *moveit_frame_args(args),
        "--tf-timeout", str(getattr(args, "tf_timeout", 8.0)),
        "--execute",
    ]
    if args.yes:
        command.append("--yes")
    return command


def retry_close_observation(
    args,
    output_dir,
    held_object_id,
    parse_state,
    description,
    allow_holding=False,
    retry_count=None,
    retry_offset_camera=None,
):
    retries = args.close_observation_retry_count if retry_count is None else retry_count
    attempts = max(0, int(retries)) + 1
    last_error = None
    for attempt in range(attempts):
        attempt_dir = os.path.join(output_dir, "attempt_{:02d}".format(attempt + 1))
        state = capture_empty_current_pose(args, attempt_dir, held_object_id, allow_holding=allow_holding)
        if state is None:
            return None, None
        try:
            return state, parse_state(state)
        except RuntimeError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            if retry_offset_camera is not None:
                camera_offset = [float(value) for value in retry_offset_camera]
                offset = camera_optical_vector_to_base(args.tf_json, camera_offset)
                print(
                    "{} detection failed: {}. Retry after optical camera offset {} -> base_link offset {}.".format(
                        description, last_error, camera_offset, offset
                    ),
                    flush=True,
                )
            else:
                offset = [float(value) for value in args.close_observation_retry_base_offset]
                print(
                    "{} detection failed: {}. Retry after base_link offset {}.".format(
                        description, last_error, offset
                    ),
                    flush=True,
                )
            run(relative_translate_command(args, offset))
    raise RuntimeError("{} detection failed after {} attempts: {}".format(description, attempts, last_error))


def snapshot_command(args, output_dir, stack_reasoning=False):
    command = [
        "conda", "run", "-n", args.conda_env, "python", "-m",
        "robot_scene_pipeline.snapshot_pipeline",
        "--output-dir", output_dir,
        "--use-tf", "--tf-json", args.tf_json,
        "--base-frame", getattr(args, "base_frame", "base_link"),
        "--camera-frame", getattr(args, "camera_frame", "camera_link"),
        "--tf-point-mode", "optical-to-camera-link",
        "--estimate-tabletop",
        "--detector-weight", args.detector_weight,
        "--score-thresh", str(args.score_thresh),
        "--detector-imgsz", str(args.detector_imgsz),
        "--detector-iou", str(args.detector_iou),
        "--detector-device", args.detector_device,
    ]
    if stack_reasoning:
        command.extend(
            [
                "--llm-task", "stack_blocks",
                "--instruction", args.instruction,
                "--model", args.model,
                "--ollama-url", args.ollama_url,
                "--timeout", str(args.timeout),
                "--num-predict", str(args.num_predict),
            ]
        )
        if args.no_image:
            command.append("--no-image")
        if args.force_llm_decision:
            command.append("--force-llm-stack-decision")
    else:
        command.append("--skip-llm")
    return command


def object_by_id(state, object_id):
    for obj in state.get("objects", []):
        if int(obj.get("id", -1)) == int(object_id):
            return obj
    raise RuntimeError("Object id {} is absent from the initial scene state.".format(object_id))


def require_geometry_object(obj):
    center = obj.get("geometry_center_m")
    dimensions = obj.get("dimensions_m")
    if obj.get("geometry_frame") != "base_link" or not center or len(center) < 3 or not dimensions or len(dimensions) < 3:
        raise RuntimeError("Stack demo object {} requires base_link geometry_center_m and dimensions_m.".format(obj.get("id")))
    return obj


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


def yaw_local_xy_to_base(offset_xy, yaw_deg):
    yaw_rad = math.radians(float(yaw_deg))
    return [
        math.cos(yaw_rad) * float(offset_xy[0]) - math.sin(yaw_rad) * float(offset_xy[1]),
        math.sin(yaw_rad) * float(offset_xy[0]) + math.cos(yaw_rad) * float(offset_xy[1]),
    ]


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
    step["target_position_m"][:2] = [round(float(value), 5) for value in center[:2]]
    step["approach_position_m"][:2] = [round(float(value), 5) for value in center[:2]]
    step["coordinate_source"] = "required_geometry_center_m"
    correction = load_xy_correction(args.xy_correction_json)
    configured_bias = correction.get("grasp_xy_bias_m", [0.0, 0.0])
    grasp_bias_base = getattr(args, "grasp_bias_base", [0.0, 0.0])
    grasp_bias_camera = getattr(args, "grasp_bias_camera", [0.0, 0.0])
    grasp_bias_camera_base = [0.0, 0.0, 0.0]
    if any(float(value) != 0.0 for value in grasp_bias_camera):
        grasp_bias_camera_base = camera_vector_to_base(
            args.tf_json,
            [float(grasp_bias_camera[0]), float(grasp_bias_camera[1]), 0.0],
        )
    correction["grasp_xy_bias_m"] = [
        float(configured_bias[0]) + float(grasp_bias_base[0]) + float(grasp_bias_camera_base[0]),
        float(configured_bias[1]) + float(grasp_bias_base[1]) + float(grasp_bias_camera_base[1]),
    ]
    apply_step_xy_correction(step, center[:2], correction, kind="grasp")
    step["grasp_offset_from_geometry_center_m"] = step["grasp_xy_bias_m"]
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
    local_offset = args.grasp_tool_offset_local
    if local_offset is None:
        local_offset = [0.0, 0.0]
    rotated_offset = yaw_local_xy_to_base(local_offset, step["chosen_grasp_yaw_deg"])
    step["grasp_tool_offset_local_xy_m"] = [float(local_offset[0]), float(local_offset[1])]
    step["grasp_tool_offset_base_xy_m"] = rotated_offset
    step["expected_tool0_grasp_xy_base_m"] = [
        float(step["target_position_m"][0]) + rotated_offset[0],
        float(step["target_position_m"][1]) + rotated_offset[1],
    ]
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


def load_or_capture_initial(args):
    if args.offline_scene_state:
        state = load_json(args.offline_scene_state)
        if args.stack_decision_json:
            decision = load_json(args.stack_decision_json)
        elif args.base_object_id is not None and args.stack_order:
            decision = {
                "task_type": "stack_blocks",
                "base_object_id": args.base_object_id,
                "stack_order": args.stack_order,
                "reason": "Explicit offline stack order.",
            }
        else:
            raise RuntimeError("Offline mode requires --stack-decision-json or --base-object-id with --stack-order.")
        return state, decision

    state = None
    initial_attempts = max(0, int(args.initial_observation_retry_count)) + 1
    for attempt in range(initial_attempts):
        initial_dir = os.path.join(args.output_dir, "initial_order")
        if attempt > 0:
            initial_dir = os.path.join(args.output_dir, "initial_order_retry_{:02d}".format(attempt))
            if float(args.initial_observation_stable_wait_s) > 0.0:
                time.sleep(float(args.initial_observation_stable_wait_s))
        run(tf_lookup_command(args))
        run(snapshot_command(args, initial_dir, stack_reasoning=False))
        state = load_json(os.path.join(initial_dir, "private_scene_state.json"))
        objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
        if objects or attempt + 1 >= initial_attempts:
            break
        print(
            "Initial observation detected no non-workspace objects; retry {}/{}.".format(
                attempt + 1,
                initial_attempts - 1,
            ),
            flush=True,
        )
    if args.stack_decision_json:
        decision = load_json(args.stack_decision_json)
    else:
        decision = None if args.force_llm_decision else rule_stack_blocks_decision(args.instruction, state.get("objects", []))
        if decision is None:
            reasoning_dir = os.path.join(args.output_dir, "initial_order_llm")
            run(tf_lookup_command(args))
            run(snapshot_command(args, reasoning_dir, stack_reasoning=True))
            state = load_json(os.path.join(reasoning_dir, "private_scene_state.json"))
            decision = load_json(os.path.join(reasoning_dir, "llm_scene_graph_decision.json"))
    return state, decision


def validate_decision(decision, initial_state, instruction="", prefer_explicit_rule=True):
    validated = validate_stack_blocks_decision(
        decision,
        initial_state.get("objects", []),
        instruction,
        prefer_explicit_rule=prefer_explicit_rule,
    )
    base_id = int(validated["base_object_id"])
    order = [int(value) for value in validated["stack_order"]]
    require_geometry_object(object_by_id(initial_state, base_id))
    for object_id in order:
        require_geometry_object(object_by_id(initial_state, object_id))
    return validated, base_id, order


def selected_stack_yaw(obj, args):
    step = {}
    set_stack_demo_yaw(
        step,
        obj,
        args.fixed_square_yaw_deg,
        args.stack_square_yaw_mode,
        args.grasp_axis,
        args.gripper_yaw_offset_deg,
        args.square_yaw_snap_tolerance_deg,
    )
    return float(step["chosen_grasp_yaw_deg"])


def print_decision_summary(initial_state, decision):
    base = object_by_id(initial_state, decision["base_object_id"])
    ordered = [object_by_id(initial_state, object_id) for object_id in decision["stack_order"]]
    print(
        "\nValidated stack decision: base={} id={} stack_order={} ids={} source={}".format(
            base.get("label"),
            base.get("id"),
            [obj.get("label") for obj in ordered],
            [obj.get("id") for obj in ordered],
            decision.get("decision_source"),
        ),
        flush=True,
    )
    for selection in decision.get("color_candidate_selections", []):
        if len(selection.get("candidate_ids", [])) > 1:
            print(
                "Color candidate selection: color={} candidates={} eligible={} selected={} "
                "xy={} strategy={}".format(
                    selection.get("color"),
                    selection.get("candidate_ids"),
                    selection.get("eligible_ids"),
                    selection.get("selected_id"),
                    selection.get("selected_xy_base_m"),
                    selection.get("strategy"),
                ),
                flush=True,
            )


def pick_motion_command(args, plan_path, path_mode, enable_gripper):
    local_offset = args.grasp_tool_offset_local
    if local_offset is None:
        local_offset = [0.0, 0.0]
    command = [
        args.ros_python, "tools/moveit_plan_preview.py",
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
        "--tool-z-offset", str(args.tool_z_offset),
        "--tool-offset-base", *[str(value) for value in args.tool_offset_base],
        "--tool-offset-yaw-local", str(local_offset[0]), str(local_offset[1]), "0",
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
        args.ros_python, "tools/moveit_plan_preview.py",
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
        "--tool-z-offset", str(args.tool_z_offset),
        "--tool-offset-base", *[str(value) for value in args.tool_offset_base],
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


def locked_object_geometry(obj):
    obj = require_geometry_object(obj)
    center = [float(value) for value in obj["geometry_center_m"][:3]]
    height = float(obj["dimensions_m"][2])
    return {
        "id": int(obj["id"]),
        "label": obj.get("label"),
        "center_base_m": center,
        "dimensions_m": [float(value) for value in obj["dimensions_m"][:3]],
        "height_m": height,
        "top_z_base_m": center[2] + height / 2.0,
        "estimated_yaw_deg": obj.get("table_yaw_deg"),
        "yaw_source": obj.get("table_yaw_source"),
    }


def reacquire_target(current_state, initial_template, allow_locked_fallback=False):
    center = require_geometry_object(initial_template)["geometry_center_m"]
    try:
        target = find_object(
            current_state,
            object_label=initial_template.get("label"),
            nearest_base_xy=center[:2],
        )
        target = copy.deepcopy(require_geometry_object(target))
        target["reacquire_source"] = "current_observation"
        return target
    except RuntimeError:
        if not allow_locked_fallback:
            raise
        target = copy.deepcopy(require_geometry_object(initial_template))
        target["reacquire_source"] = "locked_pre_pick_template"
        return target


def reacquire_pick_target_for_second_observation(current_state, initial_template):
    center = require_geometry_object(initial_template)["geometry_center_m"]
    target = find_object(
        current_state,
        object_label=initial_template.get("label"),
        nearest_base_xy=center[:2],
    )
    try:
        target = copy.deepcopy(require_geometry_object(target))
        validate_second_observation_center(
            target["geometry_center_m"],
            center,
            max_xy_m=current_state.get("_second_snapshot_max_xy_m"),
            max_z_error_m=current_state.get("_second_snapshot_max_z_error_m"),
        )
        target["reacquire_source"] = "current_observation"
        return target
    except RuntimeError as exc:
        observed_center = target.get("center_3d_base_m")
        if (
            target.get("base_coordinate_valid") is not True
            or not isinstance(observed_center, list)
            or len(observed_center) < 3
        ):
            raise
        validate_second_observation_center(
            observed_center,
            center,
            max_xy_m=current_state.get("_second_snapshot_max_xy_m"),
            max_z_error_m=current_state.get("_second_snapshot_max_z_error_m"),
        )
        fallback = copy.deepcopy(target)
        template = require_geometry_object(initial_template)
        fallback["geometry_frame"] = "base_link"
        fallback["geometry_center_m"] = [float(value) for value in observed_center[:3]]
        fallback["dimensions_m"] = [float(value) for value in template["dimensions_m"][:3]]
        fallback["table_yaw_deg"] = template.get("table_yaw_deg")
        fallback["table_yaw_source"] = "locked_template_after_second_observation_geometry_missing"
        fallback["table_yaw_valid"] = True
        fallback["pointcloud_geometry_valid"] = True
        fallback["pointcloud_geometry_reason"] = "fallback_center_3d_base_with_locked_dimensions"
        fallback["geometry_fallback_source"] = str(exc)
        fallback["reacquire_source"] = "current_detection_center_with_locked_geometry"
        return fallback


def validate_second_observation_center(center, locked_center, max_xy_m=None, max_z_error_m=None):
    observed = [float(value) for value in center[:3]]
    locked = [float(value) for value in locked_center[:3]]
    if not all(math.isfinite(value) for value in observed):
        raise RuntimeError("Second observation center is not finite: {}".format(observed))
    if max_z_error_m is not None and abs(observed[2] - locked[2]) > float(max_z_error_m):
        raise RuntimeError(
            "Second observation center z {:.4f} differs from locked z {:.4f} by more than {:.4f} m.".format(
                observed[2],
                locked[2],
                float(max_z_error_m),
            )
        )
    if max_xy_m is not None:
        delta = math.hypot(observed[0] - locked[0], observed[1] - locked[1])
        if delta > float(max_xy_m):
            raise RuntimeError(
                "Second observation center XY delta {:.4f} m exceeds correction limit {:.4f} m.".format(
                    delta,
                    float(max_xy_m),
                )
            )


def reacquire_stack_anchor(current_state, base_template, previous_stack_xy=None):
    """Prefer current stack observations; use the original base only as an initial anchor."""
    try:
        observed_base = reacquire_target(current_state, base_template, allow_locked_fallback=False)
        base_xy = require_geometry_object(observed_base)["geometry_center_m"][:2]
        return observed_base, int(observed_base["id"]), base_xy
    except RuntimeError:
        if previous_stack_xy is not None:
            return copy.deepcopy(require_geometry_object(base_template)), None, [float(v) for v in previous_stack_xy[:2]]
        observed_base = reacquire_target(current_state, base_template, allow_locked_fallback=True)
        base_xy = require_geometry_object(observed_base)["geometry_center_m"][:2]
        base_id = int(observed_base["id"]) if observed_base.get("reacquire_source") == "current_observation" else None
        return observed_base, base_id, base_xy


def estimate_current_stack(current_state, base_template, previous_stack_xy, args, search_radius_m=None, **kwargs):
    base_for_yaw, base_id, anchor_xy = reacquire_stack_anchor(current_state, base_template, previous_stack_xy)
    stack = estimate_stack_state(
        current_state,
        base_object_id=base_id,
        previous_stack_xy=anchor_xy,
        search_radius_m=args.search_radius_m if search_radius_m is None else search_radius_m,
        **kwargs,
    )
    stack["requested_base_anchor_object_id"] = base_id
    stack["requested_base_anchor_xy_m"] = anchor_xy
    stack["requested_base_anchor_source"] = base_for_yaw.get("reacquire_source")
    return base_for_yaw, stack


def held_object_exclusion(current_state, held_object):
    try:
        observed = reacquire_target(current_state, held_object, allow_locked_fallback=False)
        observed = require_geometry_object(observed)
        return [int(observed["id"])], observed["geometry_center_m"][:2]
    except RuntimeError:
        return [], None


def target_exclusion_for_pre_pick(held_object):
    obj = require_geometry_object(held_object)
    return [int(obj["id"])], obj["geometry_center_m"][:2]


def stack_yaw_object(current_state, stack_state, base_template, held_object, args, placement_base=None):
    if args.place_yaw_strategy == "held":
        return held_object
    if args.place_yaw_strategy == "base":
        return placement_base or reacquire_target(current_state, base_template, allow_locked_fallback=True)
    return require_geometry_object(object_by_id(current_state, stack_state["top_object_id"]))


def build_frozen_place_step(current_state, stack_state, base_object, held_object, args, locked_place_step=None):
    placement_base = None
    if args.place_center_strategy == "base" or args.place_yaw_strategy == "base":
        placement_base = reacquire_target(current_state, base_object, allow_locked_fallback=True)
        if placement_base.get("reacquire_source") == "locked_pre_pick_template" and locked_place_step is not None:
            locked_center = locked_place_step.get("placement_reference_center_xy_m")
            if locked_center is not None:
                placement_base["geometry_center_m"][:2] = [float(value) for value in locked_center[:2]]
            locked_dimensions = locked_place_step.get("placement_reference_dimensions_m")
            if locked_dimensions is not None:
                placement_base["dimensions_m"][:2] = [float(value) for value in locked_dimensions[:2]]
            locked_yaw = locked_place_step.get(
                "placement_reference_yaw_deg",
                locked_place_step.get("chosen_place_yaw_deg"),
            )
            if locked_yaw is not None:
                placement_base["table_yaw_deg"] = float(locked_yaw)
            placement_base["reacquire_source"] = "locked_pre_pick_place_reference"
    place_yaw_object = stack_yaw_object(
        current_state,
        stack_state,
        base_object,
        held_object,
        args,
        placement_base=placement_base,
    )
    place_yaw = selected_stack_yaw(place_yaw_object, args)
    placement_stack_state = copy.deepcopy(stack_state)
    top_xy = [float(value) for value in stack_state["placement_base_center_xy_m"][:2]]
    safety_top_xy = top_xy
    safety_top_xy_source = "current_observation"
    if (
        placement_base is not None
        and placement_base.get("reacquire_source") == "locked_pre_pick_place_reference"
        and locked_place_step is not None
    ):
        locked_top_xy = locked_place_step.get("observed_top_center_xy_m")
        if locked_top_xy is not None:
            safety_top_xy = [float(value) for value in locked_top_xy[:2]]
            safety_top_xy_source = "locked_pre_pick_place_reference"
    if args.place_center_strategy == "base":
        base_xy = [float(value) for value in placement_base["geometry_center_m"][:2]]
        top_center_offset = math.hypot(safety_top_xy[0] - base_xy[0], safety_top_xy[1] - base_xy[1])
        base_dimensions = [float(value) for value in placement_base["dimensions_m"][:2]]
        max_top_center_offset = min(
            float(args.max_stack_top_center_offset_m),
            0.20 * min(base_dimensions),
        )
        if top_center_offset > max_top_center_offset:
            raise RuntimeError(
                "Refusing placement: stack top center offset {:.4f} m from base center exceeds {:.4f} m.".format(
                    top_center_offset, max_top_center_offset
                )
            )
        placement_stack_state["placement_base_center_xy_m"] = base_xy
    else:
        base_xy = top_xy
        top_center_offset = 0.0
        max_top_center_offset = float(args.max_stack_top_center_offset_m)
    place_step = compile_place_on_top(
        held_object,
        placement_stack_state,
        {
            "step": 1,
            "approach_height_m": args.approach_height_m,
            "release_gap_m": args.release_gap_m,
            "place_top_z_bias_m": args.place_top_z_bias_m,
            "fixed_yaw_deg": place_yaw,
            "object_offset_tool_xy_m": args.object_offset_tool,
            "place_bias_base_xy_m": args.place_bias_base,
            "max_place_bias_base_m": args.max_place_bias_base_m,
        },
    )
    place_step["target_yaw_deg"] = place_yaw
    place_step["chosen_place_yaw_deg"] = place_yaw
    place_step["yaw_source"] = "locked_close_{}_yaw".format(args.place_yaw_strategy)
    place_step["place_center_strategy"] = args.place_center_strategy
    place_step["observed_top_center_xy_m"] = top_xy
    place_step["placement_reference_center_xy_m"] = base_xy
    place_step["placement_reference_dimensions_m"] = [
        float(value) for value in placement_base["dimensions_m"][:2]
    ] if placement_base is not None else None
    place_step["placement_reference_yaw_deg"] = place_yaw
    place_step["placement_reference_source"] = (
        placement_base.get("reacquire_source") if placement_base is not None else "observed_top_center"
    )
    place_step["top_center_offset_check_xy_m"] = safety_top_xy
    place_step["top_center_offset_check_source"] = safety_top_xy_source
    place_step["top_center_offset_from_reference_m"] = top_center_offset
    place_step["max_stack_top_center_offset_m"] = max_top_center_offset
    if "square" in str(place_yaw_object.get("label") or "").lower():
        place_step["exact_tool_yaw_required"] = False
        place_step["yaw_equivalence_period_deg"] = 90.0
        place_step["preserve_current_yaw"] = False
    return place_step


def validate_pick_place_separation(pick_step, place_step, min_distance_m):
    pick_xy = [float(value) for value in pick_step["target_position_m"][:2]]
    place_xy = [float(value) for value in place_step["stack_center_xy_base_m"][:2]]
    distance = math.hypot(place_xy[0] - pick_xy[0], place_xy[1] - pick_xy[1])
    if distance < float(min_distance_m):
        raise RuntimeError(
            "Refusing place target near pick source: distance {:.4f} m is below {:.4f} m. "
            "Close stack observation likely selected the held target instead of the base.".format(
                distance, min_distance_m
            )
        )
    place_step["pick_source_xy_base_m"] = pick_xy
    place_step["pick_place_xy_distance_m"] = distance
    place_step["min_pick_place_xy_distance_m"] = float(min_distance_m)
    return place_step


def validate_place_second_snapshot(first_stack_state, second_stack_state, max_correction_m, top_z_tolerance_m=None):
    first_xy = [float(value) for value in first_stack_state["placement_base_center_xy_m"][:2]]
    second_xy = [float(value) for value in second_stack_state["placement_base_center_xy_m"][:2]]
    delta = [second_xy[0] - first_xy[0], second_xy[1] - first_xy[1]]
    distance = math.hypot(delta[0], delta[1])
    first_top_z = float(first_stack_state["placement_base_top_z_m"])
    second_top_z = float(second_stack_state["placement_base_top_z_m"])
    if distance > float(max_correction_m):
        raise RuntimeError(
            "Refusing second base correction: XY delta {:.4f} m exceeds {:.4f} m.".format(
                distance, max_correction_m
            )
        )
    if top_z_tolerance_m is not None and abs(second_top_z - first_top_z) > float(top_z_tolerance_m):
        raise RuntimeError(
            "Refusing second base correction: top_z delta {:.4f} m exceeds {:.4f} m.".format(
                abs(second_top_z - first_top_z),
                float(top_z_tolerance_m),
            )
        )
    return {
        "first_base_center_xy_m": first_xy,
        "second_base_center_xy_m": second_xy,
        "second_snapshot_delta_base_xy_m": delta,
        "second_snapshot_correction_norm_m": distance,
        "max_second_snapshot_correction_m": float(max_correction_m),
        "first_base_top_z_m": first_top_z,
        "second_base_top_z_m": second_top_z,
        "held_base_top_z_tolerance_m": None if top_z_tolerance_m is None else float(top_z_tolerance_m),
        "coordinate_source": "second_close_base_observation",
    }


def reject_held_observation_and_keep_locked(cycle_dir, reason, place_plan_path, held_final_stack, args):
    write_json(
        os.path.join(cycle_dir, "place_final_held_observation_rejected_use_locked.json"),
        {
            "reason": str(reason),
            "fallback": "use_locked_before_pick_place_plan",
            "locked_place_plan": place_plan_path,
            "held_stack_state": held_final_stack,
            "max_place_second_snapshot_correction_m": args.max_place_second_snapshot_correction_m,
            "held_base_top_z_tolerance_m": args.held_base_top_z_tolerance_m,
        },
    )
    print(
        "Final held-object base observation rejected; using locked before-pick place plan: {}".format(reason),
        flush=True,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    runtime = {
        "current_stage": "startup",
        "held_object_id": None,
        "last_pick_pose": None,
        "last_place_pose": None,
    }
    try:
        runtime["current_stage"] = "init_ready_pose"
        init_ready_pose(args)

        runtime["current_stage"] = "initial_snapshot_and_decision"
        initial_state, raw_decision = load_or_capture_initial(args)
        decision, base_id, order = validate_decision(
            raw_decision,
            initial_state,
            args.instruction,
            prefer_explicit_rule=True,
        )
        print_decision_summary(initial_state, decision)
        write_json(os.path.join(args.output_dir, "stack_blocks_decision.json"), decision)
        write_json(os.path.join(args.output_dir, "initial_scene_state.json"), initial_state)

        base_object = copy.deepcopy(object_by_id(initial_state, base_id))
        held_templates = {object_id: copy.deepcopy(object_by_id(initial_state, object_id)) for object_id in order}
        write_json(
            os.path.join(args.output_dir, "locked_initial_geometry.json"),
            {
                "base": locked_object_geometry(base_object),
                "stack_order": [locked_object_geometry(held_templates[object_id]) for object_id in order],
                "table_plane": initial_state.get("table_plane"),
                "source": "initial_safe_observation_before_any_pick",
            },
        )

        xy_correction = load_xy_correction(args.xy_correction_json)
        current_state = copy.deepcopy(initial_state)
        previous_stack_xy = None
        previous_locked_stack = None

        for index, object_id in enumerate(order, start=1):
            cycle_dir = os.path.join(args.output_dir, "cycle_{:02d}_object_{}".format(index, object_id))
            os.makedirs(cycle_dir, exist_ok=True)

            if index > 1:
                runtime["current_stage"] = "empty_gripper_ready_observation"
                observed = capture_empty_observation(
                    args,
                    os.path.join(cycle_dir, "observation_before_pick"),
                    runtime["held_object_id"],
                )
                if observed is not None:
                    current_state = observed

            runtime["current_stage"] = "detect_target_and_freeze_place"
            held_object = copy.deepcopy(reacquire_target(current_state, held_templates[object_id]))
            pre_pick_excluded_ids, pre_pick_excluded_xy = target_exclusion_for_pre_pick(held_object)
            current_base_object, stack_state = estimate_current_stack(
                current_state,
                base_object,
                previous_stack_xy,
                args,
                search_radius_m=args.search_radius_m,
                excluded_object_ids=pre_pick_excluded_ids,
                excluded_xy=pre_pick_excluded_xy,
                exclusion_radius_m=args.target_exclusion_radius_m,
            )
            if current_base_object.get("reacquire_source") == "current_observation":
                base_object = current_base_object
            if not stack_state.get("valid"):
                raise RuntimeError("Cycle {} pre-pick stack estimate failed: {}".format(index, stack_state.get("reason")))
            if previous_locked_stack is not None:
                verification = verify_stack_growth(previous_locked_stack, stack_state)
                write_json(os.path.join(cycle_dir, "previous_place_verification.json"), verification)
                if not verification["valid"]:
                    recheck_state = capture_empty_observation(
                        args,
                        os.path.join(cycle_dir, "previous_place_verification_recheck"),
                        runtime["held_object_id"],
                    )
                    if recheck_state is not None:
                        current_state = recheck_state
                        held_object = copy.deepcopy(reacquire_target(current_state, held_templates[object_id]))
                        pre_pick_excluded_ids, pre_pick_excluded_xy = target_exclusion_for_pre_pick(held_object)
                        current_base_object, stack_state = estimate_current_stack(
                            current_state,
                            base_object,
                            previous_stack_xy,
                            args,
                            search_radius_m=args.search_radius_m,
                            excluded_object_ids=pre_pick_excluded_ids,
                            excluded_xy=pre_pick_excluded_xy,
                            exclusion_radius_m=args.target_exclusion_radius_m,
                        )
                        if current_base_object.get("reacquire_source") == "current_observation":
                            base_object = current_base_object
                        verification = verify_stack_growth(previous_locked_stack, stack_state)
                        write_json(
                            os.path.join(cycle_dir, "previous_place_verification_recheck.json"),
                            verification,
                        )
                    if not verification["valid"]:
                        raise RuntimeError(
                            "Cycle {} previous placement verification failed after recheck: {}".format(
                                index, verification["reason"]
                            )
                        )
            write_json(os.path.join(cycle_dir, "stack_state_locked_before_pick.json"), stack_state)
            previous_stack_xy = stack_state["stack_xy_base_m"]
            provisional_place_step = build_frozen_place_step(
                current_state, stack_state, current_base_object, held_object, args
            )
            provisional_place_plan_path = os.path.join(cycle_dir, "place_on_top_plan_first_observation.json")
            write_json(provisional_place_plan_path, plan_envelope(current_state, provisional_place_step))
            first_pick_plan_path = os.path.join(cycle_dir, "pick_plan_first_observation.json")
            build_offline_pick_plan(current_state, held_object, first_pick_plan_path, args)

            runtime["current_stage"] = "empty_gripper_first_pick_approach"
            if args.execute:
                run(pick_approach_command(args, first_pick_plan_path))

            runtime["current_stage"] = "empty_gripper_second_pick_snapshot"
            second_dir = os.path.join(cycle_dir, "pick_second_observation")
            def parse_second_pick_target(state):
                state["_second_snapshot_max_xy_m"] = float(args.max_second_snapshot_correction_m)
                state["_second_snapshot_max_z_error_m"] = float(args.second_snapshot_max_z_error_m)
                return copy.deepcopy(reacquire_pick_target_for_second_observation(state, held_object))

            try:
                second_state, second_object = retry_close_observation(
                    args,
                    second_dir,
                    runtime["held_object_id"],
                    parse_second_pick_target,
                    "Second target observation",
                    retry_offset_camera=args.second_snapshot_retry_offset_camera,
                )
            except RuntimeError as exc:
                message = str(exc)
                if (
                    "Second observation center z" not in message
                    and "Second observation center XY delta" not in message
                    and "Second observation center is not finite" not in message
                ):
                    raise
                print(
                    "Second target observation was visible but had unreliable 3D center; "
                    "using locked first observation without XY correction: {}".format(message),
                    flush=True,
                )
                write_json(
                    os.path.join(cycle_dir, "pick_second_observation_unreliable_use_first.json"),
                    {
                        "reason": message,
                        "fallback": "use_locked_first_observation_without_second_xy_correction",
                        "max_second_snapshot_correction_m": args.max_second_snapshot_correction_m,
                        "second_snapshot_max_z_error_m": args.second_snapshot_max_z_error_m,
                    },
                )
                second_state = current_state
                second_object = copy.deepcopy(held_object)
            if second_state is None:
                second_state = current_state
                second_object = copy.deepcopy(held_object)
            second_pick_plan_path = os.path.join(cycle_dir, "pick_plan_second_observation.json")
            build_offline_pick_plan(second_state, second_object, second_pick_plan_path, args)
            pick_plan_path = os.path.join(cycle_dir, "pick_plan_second_xy_corrected.json")
            correction_report_path = os.path.join(cycle_dir, "pick_second_xy_correction.json")
            build_corrected_plan(
                first_pick_plan_path,
                second_pick_plan_path,
                pick_plan_path,
                correction_report_path,
                args.max_second_snapshot_correction_m,
                args.max_grasp_offset_m,
                use_second_yaw=False,
                use_second_grasp_offset=True,
            )
            pick_plan = load_json(pick_plan_path)
            pick_step = pick_plan["steps"][0]

            # Do not leave the corrected pick approach before grasping. The base is
            # reacquired only after the object is held and the robot is above it.
            final_stack_state = stack_state
            place_step = build_frozen_place_step(
                current_state, final_stack_state, current_base_object, held_object, args
            )
            place_step["coordinate_source"] = "pre_pick_stack_observation.approach_only"
            validate_pick_place_separation(pick_step, place_step, args.min_pick_place_xy_distance_m)
            place_plan_path = os.path.join(cycle_dir, "place_on_top_plan_locked_before_pick.json")
            write_json(place_plan_path, plan_envelope(current_state, place_step))
            runtime["last_place_pose"] = {
                "position_m": place_step["target_position_m"],
                "pre_place_z_base_m": place_step["pre_place_z_base_m"],
                "release_z_base_m": place_step["release_z_base_m"],
                "detected_base_center_xy_m": final_stack_state.get("placement_base_center_xy_m"),
                "placement_reference_center_xy_m": place_step.get("placement_reference_center_xy_m"),
                "placement_reference_source": place_step.get("placement_reference_source"),
                "observed_top_center_xy_m": place_step.get("observed_top_center_xy_m"),
                "top_center_offset_from_reference_m": place_step.get("top_center_offset_from_reference_m"),
                "detected_base_top_z_m": final_stack_state.get("placement_base_top_z_m"),
                "place_top_z_bias_m": place_step.get("place_top_z_bias_m"),
                "release_gap_m": place_step.get("release_gap_m"),
                "yaw_deg": place_step["chosen_place_yaw_deg"],
                "source": place_step["coordinate_source"],
            }

            print(
                "\nCycle {} pick corrected by second target snapshot; grasp immediately before base approach: "
                "target_id={} label={} first_center={} second_center={} "
                "first_object_yaw={} second_object_yaw={} final_grasp_yaw={} "
                "grasp_tool_offset_local={} grasp_tool_offset_base={} "
                "expected_tool0_grasp_xy={} detected_base_id={} detected_base_center={} "
                "detected_base_top_z={} stack_average_center={} stack_yaw={} place_pose={}".format(
                    index,
                    held_object.get("id"),
                    held_object.get("label"),
                    held_object.get("geometry_center_m"),
                    second_object.get("geometry_center_m"),
                    held_object.get("table_yaw_deg"),
                    second_object.get("table_yaw_deg"),
                    pick_step.get("chosen_grasp_yaw_deg"),
                    pick_step.get("grasp_tool_offset_local_xy_m"),
                    pick_step.get("grasp_tool_offset_base_xy_m"),
                    pick_step.get("expected_tool0_grasp_xy_base_m"),
                    final_stack_state.get("placement_base_object_id"),
                    final_stack_state.get("placement_base_center_xy_m"),
                    final_stack_state.get("placement_base_top_z_m"),
                    final_stack_state.get("stack_xy_base_m"),
                    final_stack_state.get("stack_yaw_deg"),
                    runtime["last_place_pose"],
                ),
                flush=True,
            )

            runtime["current_stage"] = "pick_from_second_visual_correction"
            runtime["last_pick_pose"] = {
                "first_center_base_m": held_object.get("geometry_center_m"),
                "second_center_base_m": second_object.get("geometry_center_m"),
                "corrected_target_position_m": pick_step.get("target_position_m"),
                "first_estimated_yaw_deg": held_object.get("table_yaw_deg"),
                "second_estimated_yaw_deg": second_object.get("table_yaw_deg"),
                "chosen_grasp_yaw_deg": pick_step.get("chosen_grasp_yaw_deg"),
            }
            if args.execute:
                runtime["held_object_id"] = object_id
                run(pick_command(args, pick_plan_path))
            else:
                runtime["held_object_id"] = object_id

            runtime["current_stage"] = "holding_object_move_above_locked_base"
            if args.execute:
                run(place_approach_command(args, place_plan_path))

                runtime["current_stage"] = "holding_object_final_base_snapshot"
                held_close_dir = os.path.join(cycle_dir, "place_final_observation_while_holding")

                def parse_held_base(state):
                    excluded_ids, excluded_xy = held_object_exclusion(state, held_object)
                    anchor_xy = final_stack_state.get("stack_xy_base_m") or final_stack_state.get("placement_base_center_xy_m")
                    _, estimate = estimate_current_stack(
                        state,
                        current_base_object,
                        anchor_xy,
                        args,
                        search_radius_m=args.close_stack_search_radius_m,
                        excluded_object_ids=excluded_ids,
                        excluded_xy=excluded_xy,
                        exclusion_radius_m=args.target_exclusion_radius_m,
                    )
                    if not estimate.get("valid"):
                        _, estimate = estimate_current_stack(
                            state,
                            current_base_object,
                            anchor_xy,
                            args,
                            search_radius_m=max(args.close_stack_search_radius_m, args.search_radius_m),
                            excluded_object_ids=excluded_ids,
                            excluded_xy=excluded_xy,
                            exclusion_radius_m=args.target_exclusion_radius_m,
                        )
                    if not estimate.get("valid"):
                        raise RuntimeError(estimate.get("reason"))
                    estimate["held_observation_excluded_held_object_ids"] = excluded_ids
                    estimate["held_observation_excluded_held_object_xy_m"] = excluded_xy
                    estimate["held_observation_anchor_xy_m"] = anchor_xy
                    return estimate

                try:
                    held_close_state, held_final_stack = retry_close_observation(
                        args,
                        held_close_dir,
                        runtime["held_object_id"],
                        parse_held_base,
                        "Final held-object base observation",
                        allow_holding=True,
                        retry_count=args.close_observation_retry_count,
                        retry_offset_camera=args.place_observation_offset_camera,
                    )
                    if held_close_state is None or held_final_stack is None:
                        raise RuntimeError("Final held-object base observation produced no scene state.")
                    write_json(
                        os.path.join(cycle_dir, "stack_state_final_held_before_place.json"),
                        held_final_stack,
                    )
                    held_place_correction = validate_place_second_snapshot(
                        final_stack_state,
                        held_final_stack,
                        args.max_place_second_snapshot_correction_m,
                        top_z_tolerance_m=args.held_base_top_z_tolerance_m,
                    )
                    write_json(
                        os.path.join(cycle_dir, "place_final_held_xy_z_correction.json"),
                        held_place_correction,
                    )
                    place_step = build_frozen_place_step(
                        held_close_state,
                        held_final_stack,
                        current_base_object,
                        held_object,
                        args,
                        locked_place_step=place_step,
                    )
                    place_step["coordinate_source"] = (
                        "held_object_final_observation.placement_base_center_xy_and_top_z"
                    )
                    place_step["pre_holding_base_center_xy_m"] = held_place_correction["first_base_center_xy_m"]
                    place_step["final_held_base_center_xy_m"] = held_place_correction["second_base_center_xy_m"]
                    place_step["final_held_delta_base_xy_m"] = held_place_correction["second_snapshot_delta_base_xy_m"]
                    place_step["pre_holding_base_top_z_m"] = held_place_correction["first_base_top_z_m"]
                    place_step["final_held_base_top_z_m"] = held_place_correction["second_base_top_z_m"]
                    validate_pick_place_separation(pick_step, place_step, args.min_pick_place_xy_distance_m)
                    place_plan_path = os.path.join(cycle_dir, "place_on_top_plan_final_held_observation.json")
                    write_json(place_plan_path, plan_envelope(held_close_state, place_step))
                    final_stack_state = held_final_stack
                    runtime["last_place_pose"] = {
                        "position_m": place_step["target_position_m"],
                        "pre_place_z_base_m": place_step["pre_place_z_base_m"],
                        "release_z_base_m": place_step["release_z_base_m"],
                        "detected_base_center_xy_m": held_final_stack.get("placement_base_center_xy_m"),
                        "placement_reference_center_xy_m": place_step.get("placement_reference_center_xy_m"),
                        "placement_reference_source": place_step.get("placement_reference_source"),
                        "observed_top_center_xy_m": place_step.get("observed_top_center_xy_m"),
                        "top_center_offset_from_reference_m": place_step.get("top_center_offset_from_reference_m"),
                        "detected_base_top_z_m": held_final_stack.get("placement_base_top_z_m"),
                        "place_top_z_bias_m": place_step.get("place_top_z_bias_m"),
                        "release_gap_m": place_step.get("release_gap_m"),
                        "yaw_deg": place_step["chosen_place_yaw_deg"],
                        "source": place_step["coordinate_source"],
                    }
                except RuntimeError as exc:
                    reject_held_observation_and_keep_locked(
                        cycle_dir,
                        exc,
                        place_plan_path,
                        locals().get("held_final_stack"),
                        args,
                    )

            runtime["current_stage"] = "place_from_final_base_geometry"
            if args.execute:
                run(place_command(args, place_plan_path))
            runtime["held_object_id"] = None

            if not args.execute:
                held_state = simulate_held_state(current_state, object_id)
                current_state = simulate_placed_state(held_state, held_object, place_step)
                write_json(os.path.join(cycle_dir, "after_place_scene_state.json"), current_state)
            previous_locked_stack = final_stack_state
            previous_stack_xy = final_stack_state["stack_xy_base_m"]

        runtime["current_stage"] = "final_empty_gripper_ready_verification"
        observed = capture_empty_observation(
            args,
            os.path.join(args.output_dir, "final_observation"),
            runtime["held_object_id"],
        )
        if observed is not None:
            current_state = observed
        final_base_object, final_stack_state = estimate_current_stack(
            current_state,
            base_object,
            previous_stack_xy,
            args,
            search_radius_m=args.search_radius_m,
        )
        final_verification = verify_stack_growth(previous_locked_stack, final_stack_state)
        write_json(os.path.join(args.output_dir, "final_place_verification.json"), final_verification)
        if not final_verification["valid"]:
            raise RuntimeError("Final placement verification failed: {}".format(final_verification["reason"]))

        summary = {
            "task_type": "stack_blocks",
            "execution_status": "executed" if args.execute else "dry_run_complete",
            "base_object_id": base_id,
            "stack_order": order,
            "cycles_completed": len(order),
            "final_stack_state": final_stack_state,
        }
        write_json(os.path.join(args.output_dir, "stack_demo_summary.json"), summary)
        if os.path.exists(failure_state_path(args)):
            os.remove(failure_state_path(args))
        print("\nStack demo {}: {}".format(summary["execution_status"], args.output_dir))
        return 0
    except Exception as exc:
        held_at_failure = runtime.get("held_object_id")
        failure = dict(runtime)
        failure.update(
            {
                "error": str(exc),
                "held_object_id_at_failure": held_at_failure,
            }
        )
        write_json(failure_state_path(args), failure)
        print(
            "\nSTACK DEMO FAILED at stage {}: {}\nSaved failure state: {}".format(
                runtime.get("current_stage"), exc, failure_state_path(args)
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
