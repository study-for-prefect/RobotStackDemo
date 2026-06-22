#!/usr/bin/env python3
"""Safely preview MoveIt motions from robot_execution_plan.json.

Default behavior is planning only. In approach mode it does not descend to
target points and does not control the gripper. Use --path-mode full and
--enable-gripper to test a full pick/place sequence step by step.
Run with ROS2 Python 3.10 after ur_robot_driver and ur_moveit are running.
"""

import argparse
import json
import math
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rclpy
from geometry_msgs.msg import Pose, Point
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from robot_scene_pipeline.grasp_orientation import (
    downward_quaternion_for_yaw,
    normalize_quaternion_xyzw,
    object_yaw_orientation as solve_object_yaw_orientation,
    quaternion_distance_rad,
)


DEFAULT_PLAN = "/tmp/robot_scene_pipeline/robot_execution_plan.json"

# Verified downward TCP/tool0 orientation copied from the existing yolo_grasp
# MoveIt control script in this workspace.
# Exact level downward pose. Object yaw is applied around base Z separately,
# so both gripper fingers remain at the same height throughout descent.
DEFAULT_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]
# Measured inverse of the repeatable tool0 execution bias. The public target
# remains level; this correction is applied only to the pose sent to MoveIt.
DEFAULT_ORIENTATION_COMMAND_CORRECTION_XYZW = [
    0.0037191751,
    0.0001823352,
    -0.0028525390,
    0.9999889987,
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plan or execute MoveIt motions from robot_execution_plan.json.")
    parser.add_argument("--plan-json", default=DEFAULT_PLAN)
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="Only plan/execute --ready-joint-pose-json, then exit without loading or running a pick plan.",
    )
    parser.add_argument(
        "--gripper-open-only",
        action="store_true",
        help="Open the gripper with no arm motion or plan loading, then exit.",
    )
    parser.add_argument(
        "--relative-tool-translation-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Move tool0 by this relative base_link translation with current orientation, then exit.",
    )
    parser.add_argument(
        "--hover-only",
        action="store_true",
        help="Move tool0 to a hover target only. No gripper, descent, lift, pick, or place actions are run.",
    )
    parser.add_argument(
        "--hover-target-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Hover target point in base_link. --tool-offset-base is added to produce the tool0 goal.",
    )
    parser.add_argument(
        "--hover-orientation-xyzw",
        nargs=4,
        type=float,
        default=None,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Hover tool0 target orientation in xyzw order.",
    )
    parser.add_argument("--step", type=int, default=1, help="Plan one step number. Ignored by --all-approaches.")
    parser.add_argument("--all-approaches", action="store_true", help="Plan all steps that have approach_position_m.")
    parser.add_argument(
        "--path-mode",
        choices=("approach", "full"),
        default="approach",
        help="approach: only move above targets. full: approach -> target -> retreat, with optional gripper actions.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually execute selected motions.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for interactive confirmation before execution.")
    parser.add_argument(
        "--allow-partial-plan",
        action="store_true",
        help="Allow execution when some plan steps are not planned. Default refuses partial execution.",
    )
    parser.add_argument("--cartesian", dest="cartesian", action="store_true", default=True, help="Use Cartesian planning. This is the default.")
    parser.add_argument("--joint-space", dest="cartesian", action="store_false", help="Use normal MoveIt joint-space pose planning.")
    parser.add_argument("--cartesian-max-step", type=float, default=0.005)
    parser.add_argument("--cartesian-fraction-threshold", type=float, default=0.90)
    parser.add_argument(
        "--min-trajectory-duration",
        type=float,
        default=0.0,
        help="Stretch planned trajectory timestamps to be at least this many seconds before execution.",
    )
    parser.add_argument(
        "--min-point-dt",
        type=float,
        default=0.0,
        help="Stretch trajectory timestamps so consecutive points are at least this many seconds apart.",
    )
    parser.add_argument("--tool-z-offset", type=float, default=0.15, help="Add this to plan z for tool0/flange pose.")
    parser.add_argument(
        "--tool-offset-base",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Additional base_link offset added to tool0 goal after z offset. Use this to compensate gripper/TCP XY offset.",
    )
    parser.add_argument(
        "--tool-offset-yaw-local",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help=(
            "Additional offset in the selected yaw local XY frame. It is rotated into base_link "
            "by the selected grasp yaw, then added to the legacy tool0 goal."
        ),
    )
    parser.add_argument(
        "--tcp-calibration-json",
        default="",
        help="Use calibrated tool0->TCP translation. This overrides --tool-z-offset and --tool-offset-base.",
    )
    parser.add_argument(
        "--tcp-target-offset-base",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Add a fixed base_link offset to the requested TCP target before applying calibrated TCP conversion.",
    )
    parser.add_argument("--min-z", type=float, default=0.05, help="Reject tool0 goals below this base_link z.")
    parser.add_argument("--max-z", type=float, default=0.80, help="Reject tool0 goals above this base_link z.")
    parser.add_argument("--max-radius", type=float, default=0.90, help="Reject xy radius beyond this value.")
    parser.add_argument("--quat-xyzw", nargs=4, type=float, default=DEFAULT_QUAT_XYZW)
    parser.add_argument(
        "--orientation-command-correction-xyzw",
        nargs=4,
        type=float,
        default=DEFAULT_ORIENTATION_COMMAND_CORRECTION_XYZW,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Base-frame quaternion pre-compensation applied only to MoveIt pose commands.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("current", "fixed", "object-yaw"),
        default="current",
        help="Use current orientation, fixed --quat-xyzw, or valid object yaw from the execution plan.",
    )
    parser.add_argument(
        "--grasp-axis",
        choices=("long", "short"),
        default="long",
        help="With object-yaw, align the tool yaw reference with the object's long or short footprint axis.",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="Fixed calibration offset added after object yaw and grasp-axis selection.",
    )
    parser.add_argument(
        "--yaw-sign",
        choices=("positive", "negative"),
        default="positive",
        help="Use object yaw as reported or invert its sign before applying the grasp offset.",
    )
    parser.add_argument(
        "--invalid-yaw-fallback",
        choices=("fixed", "current", "error"),
        default="fixed",
        help="Behavior for square/round objects whose object yaw is not reliable.",
    )
    parser.add_argument(
        "--diagnostic-yaw-deg",
        nargs="+",
        type=float,
        default=None,
        help="Repeat each selected approach-only step at explicit base-link yaw values for offset diagnosis.",
    )
    parser.add_argument(
        "--pre-rotate-before-translation",
        action="store_true",
        help=(
            "Before the first Cartesian/pose translation of each step, first plan the selected "
            "orientation at the current tool0 position. This stages yaw change before XY/Z motion."
        ),
    )
    parser.add_argument(
        "--pre-rotate-strategy",
        choices=("joint-wrist3", "pose"),
        default="joint-wrist3",
        help="joint-wrist3 changes only wrist_3_joint for yaw staging; pose uses MoveIt pose IK candidates.",
    )
    parser.add_argument(
        "--pre-rotate-wrist-yaw-sign",
        choices=("auto", "positive", "negative"),
        default="auto",
        help="Mapping from base yaw delta to wrist_3_joint delta for --pre-rotate-strategy joint-wrist3.",
    )
    parser.add_argument(
        "--pre-rotate-wrist-direction",
        choices=("auto", "positive", "negative"),
        default="auto",
        help="Restrict the actual wrist_3_joint pre-rotation direction. Use this to avoid cable wrap.",
    )
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--group-name", default=ur.MOVE_GROUP_ARM)
    parser.add_argument("--base-link", default=ur.base_link_name())
    parser.add_argument("--end-effector", default=ur.end_effector_name())
    parser.add_argument(
        "--no-auto-resolve-tf-frames",
        action="store_false",
        dest="auto_resolve_tf_frames",
        default=True,
        help="Disable suffix-based recovery for prefixed base/tool frame names.",
    )
    parser.add_argument("--velocity", type=float, default=0.2)
    parser.add_argument("--acceleration", type=float, default=0.2)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.4)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.4)
    parser.add_argument(
        "--safe-pre-rotate-height",
        type=float,
        default=0.12,
        help="Minimum base_link z used before hover-only orientation pre-rotation.",
    )
    parser.add_argument("--place-on-top-velocity", type=float, default=0.03)
    parser.add_argument("--place-on-top-acceleration", type=float, default=0.03)
    parser.add_argument("--place-on-top-release-wait", type=float, default=1.0)
    parser.add_argument(
        "--max-grasp-yaw-error-deg",
        type=float,
        default=5.0,
        help="Refuse the final pick descent when current TCP yaw differs from grasp_yaw by more than this.",
    )
    parser.add_argument(
        "--max-grasp-orientation-error-deg",
        type=float,
        default=0.5,
        help="Refuse final pick descent when the full tool orientation differs from the target by more than this.",
    )
    parser.add_argument(
        "--orientation-settle-error-deg",
        type=float,
        default=0.2,
        help="At approach height, retry an in-place orientation correction above this full orientation error.",
    )
    parser.add_argument("--orientation-settle-attempts", type=int, default=2)
    parser.add_argument("--orientation-settle-wait-s", type=float, default=0.25)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--max-joint-delta", type=float, default=1.2, help="Warn if any joint changes more than this radian value.")
    parser.add_argument(
        "--max-pre-rotate-joint-delta",
        type=float,
        default=math.pi,
        help="Separate joint-delta limit for the intentional wrist_3 pre-rotation stage.",
    )
    parser.add_argument(
        "--ready-joint-pose-json",
        default="",
        help="Optional fixed joint pose JSON to plan/execute before selected steps.",
    )
    parser.add_argument(
        "--ready-joint-tolerance",
        type=float,
        default=0.03,
        help="Skip ready-pose planning when every joint is already within this many radians.",
    )
    parser.add_argument("--enable-gripper", action="store_true", help="Enable DH PGC gripper actions in --path-mode full.")
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-slave-id", type=int, default=1)
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--skip-gripper-init", action="store_true")
    parser.add_argument("--gripper-full-calibration", action="store_true")
    parser.add_argument("--gripper-force", type=int, default=50)
    parser.add_argument("--gripper-speed", type=int, default=30)
    parser.add_argument("--gripper-open-position", type=int, default=1000)
    parser.add_argument("--gripper-close-position", type=int, default=0)
    parser.add_argument("--gripper-wait", type=float, default=2.0)
    parser.add_argument(
        "--open-gripper-at-start",
        action="store_true",
        help="Open the gripper after initialization before any motion/plan steps.",
    )
    parser.add_argument(
        "--post-close-wait",
        type=float,
        default=1.0,
        help="Extra pause after a pick close command before retreating.",
    )
    parser.add_argument(
        "--release-after-pick",
        action="store_true",
        help="After a pick retreat, descend back to target, open gripper, then retreat again.",
    )
    return parser.parse_args()


def load_plan(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tcp_offset(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    offset = payload.get("tcp_offset_tool_m")
    if not isinstance(offset, list) or len(offset) != 3:
        raise RuntimeError("TCP calibration JSON must contain tcp_offset_tool_m with 3 values.")
    if payload.get("quality_pass") is not True:
        raise RuntimeError("TCP calibration quality_pass must be true: {}".format(path))
    return [float(value) for value in offset]


def load_joint_pose(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    joint_names = payload.get("joint_names")
    joint_positions = payload.get("joint_positions") or payload.get("positions")
    if not isinstance(joint_names, list) or not isinstance(joint_positions, list):
        raise RuntimeError("Joint pose JSON must contain joint_names and joint_positions.")
    if len(joint_names) != len(joint_positions):
        raise RuntimeError("Joint pose JSON joint_names and joint_positions lengths differ.")
    required = ur.joint_names()
    position_by_name = {str(name): float(position) for name, position in zip(joint_names, joint_positions)}
    missing = [name for name in required if name not in position_by_name]
    if missing:
        raise RuntimeError("Joint pose JSON missing required joints: {}".format(", ".join(missing)))
    return {
        "name": payload.get("name") or os.path.basename(path),
        "joint_names": required,
        "joint_positions": [position_by_name[name] for name in required],
    }


def tool0_goal_from_tcp(tcp_position_m, quat_xyzw, tcp_offset_tool):
    try:
        from tools.tcp_pivot_calibration import tool0_position_for_tcp
    except ImportError:
        from tcp_pivot_calibration import tool0_position_for_tcp

    return tool0_position_for_tcp(tcp_position_m, quat_xyzw, tcp_offset_tool)


def add_base_offset(position_m, offset_m):
    return [float(position_m[i]) + float(offset_m[i]) for i in range(3)]


def object_yaw_orientation(step, args, current_quaternion_xyzw):
    if step.get("preserve_current_yaw"):
        current_yaw = estimate_downward_family_yaw_deg(current_quaternion_xyzw, args.quat_xyzw)
        return normalize_quaternion_xyzw(current_quaternion_xyzw), current_yaw, "preserve_current_yaw"
    if not step.get("target_yaw_valid") or step.get("target_yaw_deg") is None:
        if args.invalid_yaw_fallback == "error":
            raise RuntimeError(
                "Step {} object={} has no reliable target yaw.".format(
                    step.get("step"), step.get("object_label")
                )
            )
        if args.invalid_yaw_fallback == "current":
            return normalize_quaternion_xyzw(current_quaternion_xyzw), None, "invalid_yaw_current_fallback"
        return normalize_quaternion_xyzw(args.quat_xyzw), None, "invalid_yaw_fixed_fallback"

    try:
        grasp_axis = effective_grasp_axis_for_step(step, args.grasp_axis)
        if step.get("exact_tool_yaw_required"):
            selected_yaw = normalize_yaw_deg(
                object_yaw_base_value(step, args.grasp_axis, args.yaw_sign, args.yaw_offset_deg)
            )
            return downward_quaternion_for_yaw(args.quat_xyzw, selected_yaw), selected_yaw, "exact_object_yaw"
        quaternion, selected_yaw = solve_object_yaw_orientation(
            step.get("target_yaw_deg"),
            step.get("target_yaw_valid"),
            step.get("yaw_frame"),
            args.quat_xyzw,
            current_quaternion_xyzw,
            grasp_axis=grasp_axis,
            yaw_sign=args.yaw_sign,
            yaw_offset_deg=args.yaw_offset_deg,
        )
    except ValueError as exc:
        raise RuntimeError("Step {} object yaw error: {}".format(step.get("step"), exc))
    return quaternion, selected_yaw, "object_yaw"


def orientation_for_step(step, args, current_quaternion_xyzw):
    if args.orientation_mode == "current":
        return normalize_quaternion_xyzw(current_quaternion_xyzw), None, "current"
    if args.orientation_mode == "fixed":
        return normalize_quaternion_xyzw(args.quat_xyzw), None, "fixed"
    return object_yaw_orientation(step, args, current_quaternion_xyzw)


def normalize_yaw_deg(yaw_deg):
    return ((float(yaw_deg) + 180.0) % 360.0) - 180.0


def effective_grasp_axis_for_step(step, grasp_axis):
    label = str(step.get("object_label") or "").lower()
    if step.get("yaw_forced_from_invalid_aspect_ratio") and "square" in label:
        return "long"
    return grasp_axis


def object_yaw_base_value(step, grasp_axis, yaw_sign, yaw_offset_deg):
    if not step.get("target_yaw_valid") or step.get("target_yaw_deg") is None:
        raise ValueError("Object yaw is not reliable.")
    if step.get("yaw_frame") not in (None, "", "base_link"):
        raise ValueError("Object yaw frame must be base_link, got {}.".format(step.get("yaw_frame")))

    if yaw_sign == "positive":
        yaw_deg = float(step.get("target_yaw_deg"))
    elif yaw_sign == "negative":
        yaw_deg = -float(step.get("target_yaw_deg"))
    else:
        raise ValueError("Unsupported yaw sign: {}".format(yaw_sign))

    grasp_axis = effective_grasp_axis_for_step(step, grasp_axis)
    if grasp_axis == "short":
        yaw_deg += 90.0
    elif grasp_axis != "long":
        raise ValueError("Unsupported grasp axis: {}".format(grasp_axis))
    return yaw_deg + float(yaw_offset_deg)


def unique_values(values):
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def shortest_yaw_delta_deg(target_yaw_deg, current_yaw_deg):
    return normalize_yaw_deg(float(target_yaw_deg) - float(current_yaw_deg))


def gripper_yaw_error_deg(target_yaw_deg, current_yaw_deg):
    """Return the smallest yaw error for a parallel gripper's 180-degree symmetry."""
    return min(
        abs(shortest_yaw_delta_deg(float(target_yaw_deg) + equivalent, current_yaw_deg))
        for equivalent in (0.0, 180.0, -180.0)
    )


def estimate_downward_family_yaw_deg(quaternion_xyzw, base_quaternion_xyzw):
    best_yaw = 0.0
    best_distance = None
    for yaw in range(-180, 181, 2):
        candidate = downward_quaternion_for_yaw(base_quaternion_xyzw, yaw)
        distance = quaternion_distance_rad(candidate, quaternion_xyzw)
        if best_distance is None or distance < best_distance:
            best_yaw = float(yaw)
            best_distance = distance
    start = best_yaw - 2.0
    for index in range(81):
        yaw = start + index * 0.05
        candidate = downward_quaternion_for_yaw(base_quaternion_xyzw, yaw)
        distance = quaternion_distance_rad(candidate, quaternion_xyzw)
        if distance < best_distance:
            best_yaw = yaw
            best_distance = distance
    return normalize_yaw_deg(best_yaw)


def object_yaw_candidate_values(step, args):
    if step.get("exact_tool_yaw_required"):
        return [
            normalize_yaw_deg(
                object_yaw_base_value(step, args.grasp_axis, args.yaw_sign, args.yaw_offset_deg)
            )
        ]
    yaw_signs = [args.yaw_sign]
    period = float(step.get("yaw_equivalence_period_deg", 180.0))
    equivalents = [index * period for index in range(-4, 5)]
    yaws = []
    for yaw_sign in yaw_signs:
        base_yaw = object_yaw_base_value(step, args.grasp_axis, yaw_sign, args.yaw_offset_deg)
        for equivalent in equivalents:
            selected_yaw = normalize_yaw_deg(base_yaw + equivalent)
            if selected_yaw not in yaws:
                yaws.append(selected_yaw)
    return yaws


def orientation_candidates_for_pre_rotate(step, args, current_quaternion_xyzw):
    if step.get("preserve_current_yaw"):
        quaternion, selected_yaw, source = object_yaw_orientation(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": source,
            }
        ]
    if args.orientation_mode != "object-yaw":
        quaternion, selected_yaw, source = orientation_for_step(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": source,
            }
        ]

    if not step.get("target_yaw_valid") or step.get("target_yaw_deg") is None:
        quaternion, selected_yaw, source = object_yaw_orientation(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": source,
            }
        ]
    if step.get("exact_tool_yaw_required"):
        quaternion, selected_yaw, source = object_yaw_orientation(step, args, current_quaternion_xyzw)
        return [
            {
                "quat_xyzw": quaternion,
                "selected_yaw_deg": selected_yaw,
                "source": source,
                "label": "exact_object_yaw selected_yaw={:.2f}".format(selected_yaw),
            }
        ]

    candidates = []
    period = float(step.get("yaw_equivalence_period_deg", 180.0))
    yaw_equivalents = [index * period for index in range(-4, 5)]
    effective_axis = effective_grasp_axis_for_step(step, args.grasp_axis)
    for yaw_sign in [args.yaw_sign]:
        base_yaw = object_yaw_base_value(step, args.grasp_axis, yaw_sign, args.yaw_offset_deg)
        for equivalent in yaw_equivalents:
            raw_yaw = base_yaw + equivalent
            quaternion = downward_quaternion_for_yaw(args.quat_xyzw, raw_yaw)
            selected_yaw = normalize_yaw_deg(raw_yaw)
            label = "object_yaw axis={} sign={} raw_yaw={:.2f} selected_yaw={:.2f}".format(
                effective_axis,
                yaw_sign,
                raw_yaw,
                selected_yaw,
            )
            if any(
                existing["selected_yaw_deg"] == selected_yaw
                and all(abs(a - b) < 1e-9 for a, b in zip(existing["quat_xyzw"], quaternion))
                for existing in candidates
            ):
                continue
            candidates.append(
                {
                    "quat_xyzw": quaternion,
                    "selected_yaw_deg": selected_yaw,
                    "source": "object_yaw_joint_delta_selected",
                    "label": label,
                }
            )
    return candidates


def selected_steps(plan, args):
    steps = [step for step in plan.get("steps", []) if step.get("status") == "planned"]
    steps = [step for step in steps if step.get("approach_position_m")]
    if args.all_approaches:
        return steps
    for step in steps:
        if int(step.get("step", -1)) == args.step:
            return [step]
    raise RuntimeError("No planned step {} with approach_position_m.".format(args.step))


def validate_complete_plan(plan, args):
    if args.allow_partial_plan:
        return
    bad_steps = []
    for step in plan.get("steps", []):
        status = step.get("status")
        action = step.get("action")
        if action in ("ask_user", "stop"):
            continue
        if status != "planned":
            bad_steps.append((step.get("step"), action, status, "status is not planned"))
            continue
        if not step.get("approach_position_m"):
            bad_steps.append((step.get("step"), action, status, "missing approach_position_m"))
            continue
        if args.path_mode == "full" and not step.get("target_position_m"):
            bad_steps.append((step.get("step"), action, status, "missing target_position_m"))
    if not bad_steps:
        return
    lines = ["Execution refused because the plan is incomplete:"]
    for step_id, action, status, reason in bad_steps:
        lines.append("  step {} {} status={} {}".format(step_id, action, status, reason))
    lines.append("Regenerate the snapshot/plan after making sure all referenced objects are detected with depth.")
    raise RuntimeError("\n".join(lines))


def command_sequence_for_step(step, path_mode, enable_gripper, release_after_pick=False):
    approach = step.get("approach_position_m")
    target = step.get("release_position_m") or step.get("target_position_m")
    action = step.get("action")
    commands = []
    if path_mode == "approach":
        if approach:
            commands.append({"type": "motion", "name": "approach", "position": approach})
        return commands

    if action == "place_on_top" and approach and target:
        if abs(float(approach[0]) - float(target[0])) > 1e-6 or abs(float(approach[1]) - float(target[1])) > 1e-6:
            raise RuntimeError("place_on_top approach and target must have identical XY for vertical descent.")
    if action == "pick" and enable_gripper:
        commands.append({"type": "gripper", "name": "open"})
    if approach:
        commands.append({"type": "motion", "name": "approach", "position": approach})
    if target:
        commands.append(
            {
                "type": "motion",
                "name": "release_z" if action == "place_on_top" else "target",
                "position": target,
            }
        )
    if action == "pick" and enable_gripper:
        commands.append({"type": "gripper", "name": "close"})
    elif action in ("place_relative", "place_on_top") and enable_gripper:
        commands.append({"type": "gripper", "name": "open"})
        if action == "place_on_top":
            commands.append({"type": "wait", "name": "release_wait"})
    if approach and target:
        commands.append({"type": "motion", "name": "retreat", "position": approach})
    if action == "pick" and enable_gripper and release_after_pick and approach and target:
        commands.append({"type": "motion", "name": "release_target", "position": target})
        commands.append({"type": "gripper", "name": "open"})
        commands.append({"type": "motion", "name": "release_retreat", "position": approach})
    return commands


def diagnostic_steps(steps, diagnostic_yaw_deg):
    if not diagnostic_yaw_deg:
        return steps
    output = []
    for step in steps:
        for yaw_deg in diagnostic_yaw_deg:
            diagnostic = dict(step)
            diagnostic["diagnostic_yaw_deg"] = float(yaw_deg)
            diagnostic["target_yaw_deg"] = float(yaw_deg)
            diagnostic["target_yaw_valid"] = True
            diagnostic["yaw_frame"] = "base_link"
            diagnostic["reason"] = "Offset diagnosis at explicit base-link yaw."
            output.append(diagnostic)
    return output


def tool0_goal_from_approach(approach_position_m, tool_z_offset, tool_offset_base):
    x, y, z = [float(v) for v in approach_position_m]
    return [
        x + float(tool_offset_base[0]),
        y + float(tool_offset_base[1]),
        z + float(tool_z_offset) + float(tool_offset_base[2]),
    ]


def yaw_local_offset_to_base(offset_m, yaw_deg):
    dx, dy, dz = [float(v) for v in offset_m]
    yaw_rad = math.radians(float(yaw_deg))
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return [
        c * dx - s * dy,
        s * dx + c * dy,
        dz,
    ]


def tool_offset_for_step(args, selected_yaw_deg):
    offset = [float(value) for value in args.tool_offset_base]
    local_offset = [float(value) for value in args.tool_offset_yaw_local]
    if any(value != 0.0 for value in local_offset):
        if selected_yaw_deg is None:
            raise RuntimeError("--tool-offset-yaw-local requires --orientation-mode object-yaw with valid selected yaw.")
        rotated = yaw_local_offset_to_base(local_offset, selected_yaw_deg)
        offset = [offset[i] + rotated[i] for i in range(3)]
    return offset


def validate_goal(goal, args):
    x, y, z = goal
    radius = (x * x + y * y) ** 0.5
    if z < args.min_z:
        raise RuntimeError("Rejected goal z={:.3f} below min-z={:.3f}".format(z, args.min_z))
    if z > args.max_z:
        raise RuntimeError("Rejected goal z={:.3f} above max-z={:.3f}".format(z, args.max_z))
    if radius > args.max_radius:
        raise RuntimeError("Rejected goal xy radius={:.3f} above max-radius={:.3f}".format(radius, args.max_radius))


def xyz_delta(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def make_pose(position, quat_xyzw):
    pose = Pose()
    pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def quaternion_multiply_xyzw(left, right):
    lx, ly, lz, lw = normalize_quaternion_xyzw(left)
    rx, ry, rz, rw = normalize_quaternion_xyzw(right)
    return normalize_quaternion_xyzw(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def orientation_command_quaternion(args, target_quat_xyzw):
    return quaternion_multiply_xyzw(
        args.orientation_command_correction_xyzw,
        target_quat_xyzw,
    )


def transform_position_quat(transform):
    t = transform.transform.translation
    q = transform.transform.rotation
    return [float(t.x), float(t.y), float(t.z)], [float(q.x), float(q.y), float(q.z), float(q.w)]


def quaternion_to_rpy_xyzw(quat_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def quaternion_to_rotvec_xyzw(quat_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    scale = math.sqrt(max(0.0, 1.0 - w * w))
    if scale < 1e-9:
        return [0.0, 0.0, 0.0]
    return [angle * x / scale, angle * y / scale, angle * z / scale]


def pose_payload(position, quat_xyzw):
    rpy = quaternion_to_rpy_xyzw(quat_xyzw)
    return {
        "position": [float(value) for value in position],
        "orientation_xyzw": normalize_quaternion_xyzw(quat_xyzw),
        "rpy_rad": rpy,
        "rpy_deg": [math.degrees(value) for value in rpy],
        "rotation_vector_rad": quaternion_to_rotvec_xyzw(quat_xyzw),
    }


def trajectory_points(trajectory):
    if trajectory is None:
        return None
    joint_traj = trajectory.joint_trajectory if hasattr(trajectory, "joint_trajectory") else trajectory
    return joint_traj.joint_names, joint_traj.points


def time_from_start_seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec / 1e9


def set_time_from_start_seconds(point, seconds):
    seconds = max(0.0, float(seconds))
    point.time_from_start.sec = int(math.floor(seconds))
    point.time_from_start.nanosec = int(round((seconds - point.time_from_start.sec) * 1e9))
    if point.time_from_start.nanosec >= 1000000000:
        point.time_from_start.sec += 1
        point.time_from_start.nanosec -= 1000000000


def stretch_trajectory_timing(trajectory, min_duration=0.0, min_point_dt=0.0):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return
    _, points = parsed
    if not points:
        return
    original_times = [time_from_start_seconds(point) for point in points]
    total = original_times[-1]
    scale = 1.0
    if min_duration > 0.0 and total > 0.0 and total < min_duration:
        scale = max(scale, float(min_duration) / total)
    scaled_times = [value * scale for value in original_times]
    if min_point_dt > 0.0:
        for index in range(1, len(scaled_times)):
            scaled_times[index] = max(scaled_times[index], scaled_times[index - 1] + float(min_point_dt))
    effective_scale = 1.0
    if total > 0.0 and scaled_times[-1] > 0.0:
        effective_scale = scaled_times[-1] / total
    for point, value in zip(points, scaled_times):
        set_time_from_start_seconds(point, value)
        if effective_scale > 1.0 and point.velocities:
            point.velocities = [float(v) / effective_scale for v in point.velocities]
        if effective_scale > 1.0 and point.accelerations:
            point.accelerations = [float(a) / (effective_scale * effective_scale) for a in point.accelerations]


def print_trajectory_summary(node, trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        node.get_logger().error("No trajectory returned.")
        return
    joint_names, points = parsed
    node.get_logger().info("Trajectory points: {}".format(len(points)))
    if not points:
        return
    for idx in sorted(set([0, len(points) - 1])):
        point = points[idx]
        t = time_from_start_seconds(point)
        values = ", ".join(
            "{}={:.4f}".format(name, pos)
            for name, pos in zip(joint_names, point.positions)
        )
        node.get_logger().info("  point {} t={:.3f}: {}".format(idx, t, values))


def max_joint_delta(trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return None
    joint_names, points = parsed
    if len(points) < 2:
        return None
    start = points[0].positions
    max_name = None
    max_delta = None
    for point in points[1:]:
        for name, start_value, value in zip(joint_names, start, point.positions):
            delta = abs(float(value) - float(start_value))
            if max_delta is None or delta > max_delta:
                max_name = name
                max_delta = delta
    if max_delta is None:
        return None
    return max_name, max_delta


def setup_gripper(args):
    if not args.enable_gripper:
        return None
    try:
        from tools.robot.dh_gripper_runtime import DHPGCGripper
    except ImportError:
        from dh_gripper_runtime import DHPGCGripper

    gripper = DHPGCGripper(
        port=args.gripper_port,
        slave_id=args.gripper_slave_id,
        baudrate=args.gripper_baudrate,
    )
    print("Connected DH gripper: port={}, slave_id={}".format(args.gripper_port, args.gripper_slave_id))
    if not args.skip_gripper_init:
        if not gripper.init_gripper(full_calibration=args.gripper_full_calibration):
            gripper.close()
            raise RuntimeError("Gripper initialization failed.")
        print("Gripper initialization OK.")
    else:
        print("Gripper initialization skipped; preserving the current grip state.")
    gripper.set_force(args.gripper_force)
    gripper.set_speed(args.gripper_speed)
    print("Gripper force={}, speed={}".format(args.gripper_force, args.gripper_speed))
    return gripper


def gripper_position_for_command(args, name):
    if name == "open":
        return int(args.gripper_open_position)
    if name == "close":
        return int(args.gripper_close_position)
    raise ValueError("Unsupported gripper command: {}".format(name))


def maybe_confirm(args, prompt):
    if args.yes:
        return True
    confirm = input("{} Type yes: ".format(prompt))
    return confirm.strip().lower() == "yes"


def latest_joint_position_map(node):
    msg = node.latest_joint_state
    if msg is None:
        return {}
    return {name: float(position) for name, position in zip(msg.name, msg.position)}


def joint_position_map_from_state(joint_state):
    if joint_state is None:
        return {}
    return {name: float(position) for name, position in zip(joint_state.name, joint_state.position)}


def joint_position_map(node, start_joint_state=None):
    if start_joint_state is not None:
        return joint_position_map_from_state(start_joint_state)
    return latest_joint_position_map(node)


def joint_state_from_trajectory(trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return None
    joint_names, points = parsed
    if not points:
        return None
    state = JointState()
    state.name = list(joint_names)
    state.position = list(points[-1].positions)
    return state


def max_joint_error_to_goal(current_positions, joint_names, joint_positions):
    errors = []
    for name, target in zip(joint_names, joint_positions):
        if name not in current_positions:
            return None
        errors.append(abs(closest_angle_equivalent(target, current_positions[name]) - current_positions[name]))
    return max(errors) if errors else 0.0


def closest_angle_equivalent(value, reference):
    value = float(value)
    reference = float(reference)
    while value - reference > math.pi:
        value -= 2.0 * math.pi
    while value - reference < -math.pi:
        value += 2.0 * math.pi
    return value


def unwrap_continuous_joint_trajectory(trajectory, reference_positions):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return
    joint_names, points = parsed
    previous = [float(reference_positions.get(name, points[0].positions[index])) for index, name in enumerate(joint_names)]
    for point in points:
        positions = list(point.positions)
        for index, _name in enumerate(joint_names):
            positions[index] = closest_angle_equivalent(positions[index], previous[index])
        point.positions = positions
        previous = positions


def plan_motion_trajectory(node, args, tool_goal, quat_xyzw, cartesian=None, start_joint_state=None):
    validate_goal(tool_goal, args)
    command_quat = orientation_command_quaternion(args, quat_xyzw)
    pose = make_pose(tool_goal, command_quat)
    node.get_logger().info(
        "Orientation target={} command_after_compensation={}".format(
            pose_payload(tool_goal, quat_xyzw),
            pose_payload(tool_goal, command_quat),
        )
    )
    trajectory = node.moveit2.plan(
        pose=pose,
        cartesian=args.cartesian if cartesian is None else cartesian,
        max_step=args.cartesian_max_step,
        cartesian_fraction_threshold=args.cartesian_fraction_threshold,
        start_joint_state=start_joint_state if start_joint_state is not None else node.latest_joint_state,
    )
    if trajectory is None:
        return None
    unwrap_continuous_joint_trajectory(trajectory, joint_position_map(node, start_joint_state))
    stretch_trajectory_timing(
        trajectory,
        min_duration=args.min_trajectory_duration,
        min_point_dt=args.min_point_dt,
    )
    return trajectory


def plan_joint_trajectory(node, args, joint_positions, joint_names, start_joint_state=None):
    trajectory = node.moveit2.plan(
        joint_positions=joint_positions,
        joint_names=joint_names,
        start_joint_state=start_joint_state if start_joint_state is not None else node.latest_joint_state,
    )
    if trajectory is None:
        return None
    unwrap_continuous_joint_trajectory(trajectory, joint_position_map(node, start_joint_state))
    stretch_trajectory_timing(
        trajectory,
        min_duration=args.min_trajectory_duration,
        min_point_dt=args.min_point_dt,
    )
    return trajectory


def wrist_yaw_sign_values(args):
    if args.pre_rotate_wrist_yaw_sign == "positive":
        return [1.0]
    if args.pre_rotate_wrist_yaw_sign == "negative":
        return [-1.0]
    return [1.0, -1.0]


def wrist_direction_allowed(args, wrist_delta):
    if args.pre_rotate_wrist_direction == "positive":
        return wrist_delta > 0.0
    if args.pre_rotate_wrist_direction == "negative":
        return wrist_delta < 0.0
    return True


def select_best_joint_wrist3_pre_rotate_plan(node, args, step, current_quat, start_joint_state=None):
    current_positions = joint_position_map(node, start_joint_state)
    joint_names = ur.joint_names()
    missing = [name for name in joint_names if name not in current_positions]
    if missing:
        raise RuntimeError("Missing current joint positions for pre-rotate: {}".format(", ".join(missing)))
    current_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
    yaws = object_yaw_candidate_values(step, args)
    planned = []
    total_candidates = len(yaws) * len(wrist_yaw_sign_values(args))
    for yaw in yaws:
        yaw_delta_deg = shortest_yaw_delta_deg(yaw, current_yaw)
        for wrist_sign in wrist_yaw_sign_values(args):
            target_positions = [current_positions[name] for name in joint_names]
            wrist_index = joint_names.index("wrist_3_joint")
            wrist_delta = wrist_sign * math.radians(yaw_delta_deg)
            if not wrist_direction_allowed(args, wrist_delta):
                continue
            target_positions[wrist_index] += wrist_delta
            label = (
                "joint_wrist3 current_yaw={:.2f} selected_yaw={:.2f} yaw_delta={:.2f} "
                "wrist_sign={:+.0f} wrist_delta={:.3f} rad"
            ).format(current_yaw, yaw, yaw_delta_deg, wrist_sign, wrist_delta)
            trajectory = plan_joint_trajectory(
                node,
                args,
                target_positions,
                joint_names,
                start_joint_state=start_joint_state,
            )
            if trajectory is None:
                node.get_logger().warning(
                    "Pre-rotate joint candidate failed: {}".format(label)
                )
                continue
            delta = max_joint_delta(trajectory)
            if delta is None:
                delta_text = "none"
                delta_sort = 0.0
            else:
                delta_text = "{} {:.3f} rad".format(delta[0], delta[1])
                delta_sort = float(delta[1])
            node.get_logger().info(
                "Pre-rotate joint candidate max_delta={} {}".format(delta_text, label)
            )
            quaternion = downward_quaternion_for_yaw(args.quat_xyzw, yaw)
            planned.append(
                (
                    delta_sort,
                    {
                        "quat_xyzw": quaternion,
                        "selected_yaw_deg": yaw,
                        "source": "object_yaw_joint_wrist3_selected",
                        "label": label,
                    },
                    trajectory,
                )
            )
    if not planned:
        return None
    planned.sort(key=lambda item: item[0])
    best_delta, best_candidate, best_trajectory = planned[0]
    node.get_logger().info(
        "Selected pre-rotate joint candidate max_delta={:.3f} {} from {} candidates".format(
            best_delta,
            best_candidate["label"],
            total_candidates,
        )
    )
    return best_candidate, best_trajectory


def select_best_pose_pre_rotate_plan(node, args, step, tool_goal, current_quat, start_joint_state=None):
    candidates = orientation_candidates_for_pre_rotate(step, args, current_quat)
    planned = []
    for index, candidate in enumerate(candidates, start=1):
        trajectory = plan_motion_trajectory(
            node,
            args,
            tool_goal,
            candidate["quat_xyzw"],
            cartesian=False,
            start_joint_state=start_joint_state,
        )
        if trajectory is None:
            node.get_logger().warning(
                "Pre-rotate candidate {}/{} failed: {}".format(index, len(candidates), candidate["label"])
            )
            continue
        delta = max_joint_delta(trajectory)
        if delta is None:
            delta_text = "none"
            delta_sort = 0.0
        else:
            delta_text = "{} {:.3f} rad".format(delta[0], delta[1])
            delta_sort = float(delta[1])
        node.get_logger().info(
            "Pre-rotate candidate {}/{} max_delta={} {}".format(
                index,
                len(candidates),
                delta_text,
                candidate["label"],
            )
        )
        planned.append((delta_sort, candidate, trajectory))

    if not planned:
        return None
    planned.sort(key=lambda item: item[0])
    best_delta, best_candidate, best_trajectory = planned[0]
    node.get_logger().info(
        "Selected pre-rotate candidate max_delta={:.3f} {}".format(
            best_delta,
            best_candidate["label"],
        )
    )
    return best_candidate, best_trajectory


def select_best_pre_rotate_plan(node, args, step, tool_goal, current_quat, start_joint_state=None):
    if step.get("preserve_current_yaw"):
        return select_best_pose_pre_rotate_plan(
            node,
            args,
            step,
            tool_goal,
            current_quat,
            start_joint_state=start_joint_state,
        )
    if args.pre_rotate_strategy == "joint-wrist3" and args.orientation_mode == "object-yaw":
        return select_best_joint_wrist3_pre_rotate_plan(
            node,
            args,
            step,
            current_quat,
            start_joint_state=start_joint_state,
        )
    return select_best_pose_pre_rotate_plan(
        node,
        args,
        step,
        tool_goal,
        current_quat,
        start_joint_state=start_joint_state,
    )


def plan_and_maybe_execute_motion(
    node,
    args,
    step,
    motion_name,
    tool_goal,
    quat_xyzw,
    current_pos,
    log_message,
    cartesian=None,
    trajectory=None,
    start_joint_state=None,
    max_joint_delta_limit=None,
):
    validate_goal(tool_goal, args)
    delta_xyz = [tool_goal[i] - current_pos[i] for i in range(3)]
    node.get_logger().info("{} delta_xyz={}".format(log_message, [round(v, 4) for v in delta_xyz]))

    if trajectory is None:
        trajectory = plan_motion_trajectory(
            node,
            args,
            tool_goal,
            quat_xyzw,
            cartesian=cartesian,
            start_joint_state=start_joint_state,
        )
    if trajectory is None:
        node.get_logger().error(
            "MoveIt planning failed for step {} {}.".format(step.get("step"), motion_name)
        )
        return False
    print_trajectory_summary(node, trajectory)
    delta = max_joint_delta(trajectory)
    delta_too_large = False
    joint_delta_limit = args.max_joint_delta if max_joint_delta_limit is None else float(max_joint_delta_limit)
    if delta:
        joint_name, value = delta
        if value > joint_delta_limit:
            delta_too_large = True
            node.get_logger().warning(
                "Large joint change detected: {} delta={:.3f} rad > {:.3f}. Do not execute until reviewed.".format(
                    joint_name,
                    value,
                    joint_delta_limit,
                )
            )

    if not args.execute:
        return joint_state_from_trajectory(trajectory) or True
    if delta_too_large:
        node.get_logger().error("Execution refused because joint delta exceeds safety threshold.")
        return False
    if not maybe_confirm(
        args,
        "Execute step {} {} motion {}?".format(
            step.get("step"),
            step.get("action"),
            motion_name,
        ),
    ):
        node.get_logger().info("Execution skipped by user.")
        return False
    node.moveit2.execute(trajectory)
    ok = node.moveit2.wait_until_executed()
    node.get_logger().info("Execution result: {}".format(ok))
    final_tool = node.current_tool_transform(timeout=args.tf_timeout)
    final_pos, final_quat = transform_position_quat(final_tool)
    err = xyz_delta(final_pos, tool_goal)
    orientation_error_deg = math.degrees(quaternion_distance_rad(final_quat, quat_xyzw))
    node.get_logger().info(
        "Final tool0 position={}, goal={}, error={}, orientation_error_deg={:.3f}".format(
            [round(v, 4) for v in final_pos],
            [round(v, 4) for v in tool_goal],
            [round(v, 4) for v in err],
            orientation_error_deg,
        )
    )
    node.get_logger().info(
        "Final orientation target={} actual={}".format(
            pose_payload(tool_goal, quat_xyzw),
            pose_payload(final_pos, final_quat),
        )
    )
    if not ok:
        return False
    return joint_state_from_trajectory(trajectory) or True


def settle_orientation_before_descent(node, args, step, target_quat, current_pos, current_quat):
    attempts = max(0, int(args.orientation_settle_attempts))
    threshold = max(0.0, float(args.orientation_settle_error_deg))
    start_state = node.latest_joint_state
    for attempt in range(attempts):
        error_deg = math.degrees(quaternion_distance_rad(current_quat, target_quat))
        if error_deg <= threshold:
            break
        node.get_logger().warning(
            "Pick orientation settle {}/{}: full error {:.3f} deg > {:.3f} deg.".format(
                attempt + 1,
                attempts,
                error_deg,
                threshold,
            )
        )
        result = plan_and_maybe_execute_motion(
            node,
            args,
            step,
            "orientation_settle_{}".format(attempt + 1),
            current_pos,
            target_quat,
            current_pos,
            "In-place orientation correction before descent",
            cartesian=False,
            start_joint_state=start_state,
            max_joint_delta_limit=args.max_pre_rotate_joint_delta,
        )
        if not result:
            break
        if args.execute and args.orientation_settle_wait_s > 0:
            time.sleep(float(args.orientation_settle_wait_s))
        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, current_quat = transform_position_quat(current_tool)
        start_state = node.latest_joint_state
    return current_pos, current_quat, start_state


def run_hover_only(node, args, planning_start_state):
    if args.hover_target_base is None:
        raise RuntimeError("--hover-only requires --hover-target-base X Y Z.")
    if args.hover_orientation_xyzw is None:
        raise RuntimeError("--hover-only requires --hover-orientation-xyzw QX QY QZ QW.")

    hover_target = [float(value) for value in args.hover_target_base]
    tool_offset = [float(value) for value in args.tool_offset_base]
    effective_tool_z_offset = float(args.tool_z_offset) + tool_offset[2]
    if args.execute and effective_tool_z_offset < 0.08:
        raise RuntimeError(
            "Hover-only refused: effective tool Z offset {:.3f} m is too small. "
            "Use the same --tool-z-offset as real grasp execution.".format(effective_tool_z_offset)
        )
    tool_goal = [
        hover_target[0] + tool_offset[0],
        hover_target[1] + tool_offset[1],
        hover_target[2] + effective_tool_z_offset,
    ]
    hover_quat = normalize_quaternion_xyzw(args.hover_orientation_xyzw)
    validate_goal(tool_goal, args)

    current_tool = node.current_tool_transform(timeout=args.tf_timeout)
    current_pos, current_quat = transform_position_quat(current_tool)
    orientation_error_rad = quaternion_distance_rad(current_quat, hover_quat)
    orientation_error_deg = math.degrees(orientation_error_rad)
    node.get_logger().info("Hover-only current_tool0_pose={}".format(pose_payload(current_pos, current_quat)))
    node.get_logger().info(
        "Hover-only target_base={} tool_z_offset={:.4f} tool_offset_base={} -> tool0_goal={} target_orientation={} orientation_error_deg={:.2f}".format(
            [round(v, 4) for v in hover_target],
            float(args.tool_z_offset),
            [round(v, 4) for v in tool_offset],
            [round(v, 4) for v in tool_goal],
            [round(v, 5) for v in hover_quat],
            orientation_error_deg,
        )
    )

    start_state = planning_start_state
    threshold_deg = float(args.max_grasp_yaw_error_deg)
    if orientation_error_deg > threshold_deg:
        safe_position = [
            current_pos[0],
            current_pos[1],
            max(float(current_pos[2]), float(tool_goal[2]), float(args.safe_pre_rotate_height)),
        ]
        node.get_logger().info(
            "Hover-only pre-rotate required: orientation_error_deg={:.2f} > {:.2f}; safe_position={}".format(
                orientation_error_deg,
                threshold_deg,
                [round(v, 4) for v in safe_position],
            )
        )
        if (
            args.pre_rotate_strategy == "joint-wrist3"
            and abs(safe_position[2] - current_pos[2]) > 1e-4
        ):
            lift_result = plan_and_maybe_execute_motion(
                node,
                args,
                {"step": "hover", "action": "hover_only"},
                "safe_pre_rotate_lift",
                safe_position,
                current_quat,
                current_pos,
                "Hover-only lift before wrist pre-rotate",
                start_joint_state=start_state,
            )
            if not lift_result:
                node.get_logger().error(
                    "Hover-only stopped because the safe pre-rotate lift did not complete."
                )
                return False
            if isinstance(lift_result, JointState):
                start_state = lift_result
            if args.execute:
                current_tool = node.current_tool_transform(timeout=args.tf_timeout)
                current_pos, current_quat = transform_position_quat(current_tool)
                start_state = node.latest_joint_state

        motion_velocity = node.moveit2.max_velocity
        motion_acceleration = node.moveit2.max_acceleration
        node.moveit2.max_velocity = args.pre_rotate_velocity
        node.moveit2.max_acceleration = args.pre_rotate_acceleration
        try:
            selected_pre_rotate_trajectory = None
            if args.pre_rotate_strategy == "joint-wrist3":
                target_yaw = estimate_downward_family_yaw_deg(hover_quat, args.quat_xyzw)
                selected = select_best_joint_wrist3_pre_rotate_plan(
                    node,
                    args,
                    {
                        "target_yaw_deg": target_yaw,
                        "target_yaw_valid": True,
                        "exact_tool_yaw_required": True,
                    },
                    current_quat,
                    start_joint_state=start_state,
                )
                if selected is None:
                    node.get_logger().error(
                        "Hover-only stopped because no safe wrist_3 pre-rotate candidate planned."
                    )
                    return False
                selected_candidate, selected_pre_rotate_trajectory = selected
                node.get_logger().info(
                    "Hover-only selected explicit-yaw pre-rotate: {}".format(
                        selected_candidate["label"]
                    )
                )
            pre_rotate_result = plan_and_maybe_execute_motion(
                node,
                args,
                {"step": "hover", "action": "hover_only"},
                "safe_pre_rotate",
                safe_position,
                hover_quat,
                current_pos,
                "Hover-only safe pre-rotate",
                cartesian=False,
                trajectory=selected_pre_rotate_trajectory,
                start_joint_state=start_state,
                max_joint_delta_limit=args.max_pre_rotate_joint_delta,
            )
        finally:
            node.moveit2.max_velocity = motion_velocity
            node.moveit2.max_acceleration = motion_acceleration
        if not pre_rotate_result:
            node.get_logger().error("Hover-only stopped because safe pre-rotate did not complete.")
            return False
        if isinstance(pre_rotate_result, JointState):
            start_state = pre_rotate_result
        if args.execute:
            current_tool = node.current_tool_transform(timeout=args.tf_timeout)
            current_pos, current_quat = transform_position_quat(current_tool)
            start_state = node.latest_joint_state
            if args.pre_rotate_strategy == "joint-wrist3":
                target_yaw = estimate_downward_family_yaw_deg(hover_quat, args.quat_xyzw)
                actual_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
                yaw_error = abs(shortest_yaw_delta_deg(target_yaw, actual_yaw))
                node.get_logger().info(
                    "Hover-only pre-rotate yaw verification: target_yaw={:.2f} "
                    "actual_tcp_yaw={:.2f} yaw_error_deg={:.2f}".format(
                        target_yaw,
                        actual_yaw,
                        yaw_error,
                    )
                )
                if yaw_error > float(args.max_grasp_yaw_error_deg):
                    node.get_logger().error(
                        "Hover-only translation refused: pre-rotate yaw error {:.2f} deg "
                        "exceeds {:.2f} deg. Check --pre-rotate-wrist-yaw-sign.".format(
                            yaw_error,
                            float(args.max_grasp_yaw_error_deg),
                        )
                    )
                    return False

    hover_result = plan_and_maybe_execute_motion(
        node,
        args,
        {"step": "hover", "action": "hover_only"},
        "hover_pose",
        tool_goal,
        hover_quat,
        current_pos,
        "Hover-only move to hover pose",
        start_joint_state=start_state,
    )
    if not hover_result:
        return False
    final_tool = node.current_tool_transform(timeout=args.tf_timeout)
    final_pos, final_quat = transform_position_quat(final_tool)
    if args.execute:
        final_pos, final_quat, _settled_state = settle_orientation_before_descent(
            node,
            args,
            {"step": "hover", "action": "hover_only"},
            hover_quat,
            final_pos,
            final_quat,
        )
    node.get_logger().info("Hover-only final_tool0_pose={}".format(pose_payload(final_pos, final_quat)))
    return True


def plan_and_maybe_execute_joint_motion(
    node,
    args,
    motion_name,
    joint_names,
    joint_positions,
    start_joint_state=None,
):
    trajectory = plan_joint_trajectory(
        node,
        args,
        joint_positions,
        joint_names,
        start_joint_state=start_joint_state,
    )
    if trajectory is None:
        node.get_logger().error("MoveIt planning failed for joint motion {}.".format(motion_name))
        return False
    node.get_logger().info(
        "{} joint goal: {}".format(
            motion_name,
            ["{}={:.4f}".format(name, position) for name, position in zip(joint_names, joint_positions)],
        )
    )
    print_trajectory_summary(node, trajectory)
    delta = max_joint_delta(trajectory)
    delta_too_large = False
    if delta:
        joint_name, value = delta
        if value > args.max_joint_delta:
            delta_too_large = True
            node.get_logger().warning(
                "Large joint change detected: {} delta={:.3f} rad > {:.3f}. Do not execute until reviewed.".format(
                    joint_name,
                    value,
                    args.max_joint_delta,
                )
            )
    if not args.execute:
        return joint_state_from_trajectory(trajectory) or True
    if delta_too_large:
        node.get_logger().error("Execution refused because joint delta exceeds safety threshold.")
        return False
    if not maybe_confirm(args, "Execute joint motion {}?".format(motion_name)):
        node.get_logger().info("Execution skipped by user.")
        return False
    node.moveit2.execute(trajectory)
    ok = node.moveit2.wait_until_executed()
    node.get_logger().info("Execution result: {}".format(ok))
    if not ok:
        return False
    return joint_state_from_trajectory(trajectory) or True


def tf_frame_names(frames_yaml):
    try:
        payload = yaml.safe_load(frames_yaml) or {}
    except Exception:
        return []
    return sorted(str(name).lstrip("/") for name in payload if name)


def tf_name_matches(frame, requested):
    frame = str(frame).lstrip("/")
    requested = str(requested).lstrip("/")
    return frame == requested or frame.endswith("/" + requested) or frame.endswith("_" + requested)


def tf_candidate_pairs(frames, requested_base, requested_tool):
    bases = [frame for frame in frames if tf_name_matches(frame, requested_base)]
    tools = [frame for frame in frames if tf_name_matches(frame, requested_tool)]
    pairs = []
    for base_frame in bases:
        for tool_frame in tools:
            pair = (base_frame, tool_frame)
            if pair != (requested_base, requested_tool) and pair not in pairs:
                pairs.append(pair)
    return pairs


def tf_tree_diagnosis(frames_yaml, requested_base, requested_tool):
    frames = tf_frame_names(frames_yaml)
    base_matches = [frame for frame in frames if tf_name_matches(frame, requested_base)]
    tool_matches = [frame for frame in frames if tf_name_matches(frame, requested_tool)]
    if not frames:
        return (
            "No TF frames were received. Start robot_state_publisher and the UR driver, "
            "and verify ROS_DOMAIN_ID/RMW settings."
        )
    if not base_matches or not tool_matches:
        return "Missing TF frame candidates: base_matches={} tool_matches={}.".format(base_matches, tool_matches)
    return (
        "Base and tool frames exist but are disconnected. This usually means multiple robot_state_publisher "
        "instances, inconsistent UR prefixes, or a missing fixed joint between the robot and tool. "
        "base_matches={} tool_matches={}.".format(base_matches, tool_matches)
    )


class MoveItPreviewNode(Node):
    def __init__(self, args):
        super().__init__("moveit_plan_preview")
        self.joint_state_seen = False
        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True,
        )  # noqa F841
        self.base_frame = str(args.base_link).lstrip("/")
        self.tool_frame = str(args.end_effector).lstrip("/")
        if not args.gripper_open_only:
            self.base_frame, self.tool_frame = self.resolve_tool_frames(
                self.base_frame,
                self.tool_frame,
                timeout=args.tf_timeout,
                auto_resolve=args.auto_resolve_tf_frames,
            )
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=self.base_frame,
            end_effector_name=self.tool_frame,
            group_name=args.group_name,
            use_move_group_action=True,
        )
        self.moveit2.max_velocity = args.velocity
        self.moveit2.max_acceleration = args.acceleration
        self.moveit2.allowed_planning_time = args.planning_time

    def joint_state_cb(self, msg):
        self.joint_state_seen = True
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout=5.0):
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_state_seen:
                return True
        return False

    def current_tool_quat_xyzw(self, timeout=3.0):
        transform = self.current_tool_transform(timeout=timeout)
        q = transform.transform.rotation
        return [float(q.x), float(q.y), float(q.z), float(q.w)]

    def resolve_tool_frames(self, requested_base, requested_tool, timeout=8.0, auto_resolve=True):
        deadline = time.time() + float(timeout)
        last_error = "not attempted"
        known_frames = []
        while rclpy.ok() and time.time() < deadline:
            try:
                if self.tf_buffer.can_transform(
                    requested_base,
                    requested_tool,
                    Time(),
                    timeout=Duration(seconds=0.2),
                ):
                    return requested_base, requested_tool
                last_error = "can_transform returned False"
            except Exception as exc:
                last_error = repr(exc)
            if auto_resolve:
                known_frames = tf_frame_names(self.tf_buffer.all_frames_as_yaml())
                for base_frame, tool_frame in tf_candidate_pairs(known_frames, requested_base, requested_tool):
                    try:
                        if self.tf_buffer.can_transform(
                            base_frame,
                            tool_frame,
                            Time(),
                            timeout=Duration(seconds=0.05),
                        ):
                            self.get_logger().warning(
                                "Resolved configured TF frames {} -> {} to connected frames {} -> {}.".format(
                                    requested_base, requested_tool, base_frame, tool_frame
                                )
                            )
                            return base_frame, tool_frame
                    except Exception as exc:
                        last_error = repr(exc)
            time.sleep(0.1)
        frames_yaml = self.tf_buffer.all_frames_as_yaml()
        raise RuntimeError(
            "Robot TF preflight failed: cannot connect configured base/tool frames {} -> {} after {:.1f}s: {}\n"
            "{}\nKnown TF frames:\n{}".format(
                requested_base,
                requested_tool,
                float(timeout),
                last_error,
                tf_tree_diagnosis(frames_yaml, requested_base, requested_tool),
                frames_yaml,
            )
        )

    def current_tool_transform(self, timeout=8.0):
        deadline = time.time() + float(timeout)

        target_frame = self.base_frame
        source_frame = self.tool_frame
        last_error = "not attempted"

        while rclpy.ok() and time.time() < deadline:
            try:
                ok = self.tf_buffer.can_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )

                if ok:
                    return self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        Time(),
                        timeout=Duration(seconds=0.2),
                    )

                last_error = "can_transform returned False"

            except Exception as exc:
                last_error = repr(exc)

            time.sleep(0.1)

        try:
            frames = self.tf_buffer.all_frames_as_yaml()
        except Exception:
            frames = "<failed to dump TF frames>"

        raise RuntimeError(
            "Failed to read current {}->{} transform after {:.1f}s: {}\n"
            "Known TF frames:\n{}".format(
                target_frame,
                source_frame,
                float(timeout),
                last_error,
                frames,
            )
        )


def main():
    args = parse_args()
    if args.ready_only and not args.ready_joint_pose_json:
        raise RuntimeError("--ready-only requires --ready-joint-pose-json.")
    if args.gripper_open_only and not args.enable_gripper:
        raise RuntimeError("--gripper-open-only requires --enable-gripper.")
    if args.hover_only:
        if args.hover_target_base is None:
            raise RuntimeError("--hover-only requires --hover-target-base X Y Z.")
        if args.hover_orientation_xyzw is None:
            raise RuntimeError("--hover-only requires --hover-orientation-xyzw QX QY QZ QW.")
        if args.enable_gripper:
            raise RuntimeError("--hover-only refuses --enable-gripper.")
    relative_only = args.relative_tool_translation_base is not None
    plan = None if args.ready_only or relative_only or args.gripper_open_only or args.hover_only else load_plan(args.plan_json)
    tcp_offset_tool = load_tcp_offset(args.tcp_calibration_json)
    ready_joint_pose = load_joint_pose(args.ready_joint_pose_json)
    if args.diagnostic_yaw_deg:
        if args.path_mode != "approach":
            raise RuntimeError("--diagnostic-yaw-deg only supports --path-mode approach.")
        if args.orientation_mode != "object-yaw":
            raise RuntimeError("--diagnostic-yaw-deg requires --orientation-mode object-yaw.")
        if args.execute and args.yes:
            raise RuntimeError("--diagnostic-yaw-deg execution refuses --yes; confirm every diagnostic pose.")
    if plan is not None and plan.get("coordinate_convention", {}).get("frame") != "base_link":
        print("ERROR: execution plan is not in base_link frame.", file=sys.stderr)
        return 2
    if args.execute and plan is not None:
        validate_complete_plan(plan, args)

    steps = [] if plan is None else diagnostic_steps(selected_steps(plan, args), args.diagnostic_yaw_deg)
    if plan is not None:
        print("Loaded {} planned step(s) from {}".format(len(steps), args.plan_json))
    print("Mode: {}".format("EXECUTE" if args.execute else "PLAN ONLY"))
    if args.hover_only:
        print("Hover-only: enabled")
    print("Path mode: {}".format(args.path_mode))
    print("Gripper: {}".format("enabled" if args.enable_gripper else "disabled"))
    print("Planner: {}".format("Cartesian" if args.cartesian else "Joint-space pose"))
    print("Pre-rotate before translation: {}".format("enabled" if args.pre_rotate_before_translation else "disabled"))
    print("Pre-rotate strategy: {}".format(args.pre_rotate_strategy))
    print("Pre-rotate wrist direction: {}".format(args.pre_rotate_wrist_direction))
    print("Max joint delta: motion={:.3f}, pre_rotate={:.3f}".format(args.max_joint_delta, args.max_pre_rotate_joint_delta))
    if ready_joint_pose is not None:
        print("Ready joint pose: {}".format(args.ready_joint_pose_json))
    if tcp_offset_tool is not None:
        print("TCP calibration: {} offset_tool={}".format(args.tcp_calibration_json, tcp_offset_tool))
    else:
        print("TCP calibration: disabled; using legacy base-frame tool offsets")

    rclpy.init(args=None)
    node = MoveItPreviewNode(args)
    gripper = None
    try:
        if args.execute:
            gripper = setup_gripper(args)
            if args.gripper_open_only:
                position = gripper_position_for_command(args, "open")
                gripper.set_position(position)
                status = gripper.wait_until_done(timeout=args.gripper_wait)
                current = gripper.get_position()
                node.get_logger().info("Recovery gripper open done: status={}, position={}".format(status, current))
                return 0
            if args.open_gripper_at_start:
                if gripper is None:
                    node.get_logger().warning("--open-gripper-at-start ignored because --enable-gripper is not active.")
                elif maybe_confirm(args, "Open gripper at start?"):
                    position = gripper_position_for_command(args, "open")
                    gripper.set_position(position)
                    status = gripper.wait_until_done(timeout=args.gripper_wait)
                    current = gripper.get_position()
                    node.get_logger().info(
                        "Initial gripper open done: status={}, position={}".format(status, current)
                    )
                else:
                    node.get_logger().info("Initial gripper open skipped by user.")

        if not node.wait_for_joint_state(timeout=5.0):
            node.get_logger().error("No /joint_states received. Start UR driver and MoveIt first.")
            return 2

        planning_start_state = node.latest_joint_state
        if ready_joint_pose is not None:
            ready_error = max_joint_error_to_goal(
                joint_position_map_from_state(planning_start_state),
                ready_joint_pose["joint_names"],
                ready_joint_pose["joint_positions"],
            )
            if ready_error is not None and ready_error <= args.ready_joint_tolerance:
                node.get_logger().info(
                    "Ready pose already reached: max_joint_error={:.4f} rad <= {:.4f} rad; skipping ready motion.".format(
                        ready_error,
                        args.ready_joint_tolerance,
                    )
                )
            else:
                if ready_error is not None:
                    node.get_logger().info("Ready pose max_joint_error={:.4f} rad; planning ready motion.".format(ready_error))
                ready_result = plan_and_maybe_execute_joint_motion(
                    node,
                    args,
                    "ready_pose {}".format(ready_joint_pose["name"]),
                    ready_joint_pose["joint_names"],
                    ready_joint_pose["joint_positions"],
                    start_joint_state=planning_start_state,
                )
                if not ready_result:
                    node.get_logger().error("Ready joint pose did not complete; selected steps will not run.")
                    return 2
                if isinstance(ready_result, JointState):
                    planning_start_state = ready_result
            if args.ready_only:
                node.get_logger().info("Ready-only motion complete.")
                return 0

        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
        current_pos, current_quat = transform_position_quat(current_tool)
        node.get_logger().info(
            "Current tool0 position=[{:.4f}, {:.4f}, {:.4f}], quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                current_pos[0], current_pos[1], current_pos[2],
                current_quat[0], current_quat[1], current_quat[2], current_quat[3],
            )
        )
        if relative_only:
            offset = [float(value) for value in args.relative_tool_translation_base]
            goal = [current_pos[i] + offset[i] for i in range(3)]
            result = plan_and_maybe_execute_motion(
                node,
                args,
                {"step": "relative", "action": "relative_translate"},
                "relative_translate",
                goal,
                current_quat,
                current_pos,
                "Relative tool translation base_offset={} -> tool0_goal={}".format(
                    [round(v, 4) for v in offset],
                    [round(v, 4) for v in goal],
                ),
                cartesian=args.cartesian,
                start_joint_state=planning_start_state,
            )
            if not result:
                return 2
            return 0

        if args.hover_only:
            if not run_hover_only(node, args, planning_start_state):
                return 2
            return 0

        for step in steps:
            if step.get("action") == "place_on_top":
                node.moveit2.max_velocity = min(float(args.velocity), float(args.place_on_top_velocity))
                node.moveit2.max_acceleration = min(float(args.acceleration), float(args.place_on_top_acceleration))
            else:
                node.moveit2.max_velocity = args.velocity
                node.moveit2.max_acceleration = args.acceleration
            if step.get("diagnostic_yaw_deg") is not None:
                print(
                    "\nDIAGNOSTIC YAW {:.2f} deg: measure gripper-center error before continuing.".format(
                        step["diagnostic_yaw_deg"]
                    ),
                    flush=True,
                )
            commands = command_sequence_for_step(
                step,
                args.path_mode,
                args.enable_gripper,
                release_after_pick=args.release_after_pick,
            )
            if not commands:
                node.get_logger().warning("Step {} has no executable commands.".format(step.get("step")))
                continue
            step_quat_xyzw = None
            step_selected_yaw_deg = None
            step_orientation_source = None
            pre_rotated = bool(step.get("preserve_current_yaw"))
            step_start_state = planning_start_state

            for command in commands:
                if command["type"] == "wait":
                    print(
                        "Step {} {} wait {:.2f}s after release".format(
                            step.get("step"),
                            step.get("action"),
                            args.place_on_top_release_wait,
                        )
                    )
                    if args.execute:
                        time.sleep(float(args.place_on_top_release_wait))
                    continue
                if command["type"] == "gripper":
                    position = gripper_position_for_command(args, command["name"])
                    print(
                        "Step {} {} gripper {} -> position {}".format(
                            step.get("step"),
                            step.get("action"),
                            command["name"],
                            position,
                        )
                    )
                    if args.execute:
                        if gripper is None:
                            node.get_logger().warning("Gripper command skipped because --enable-gripper is not active.")
                            continue
                        if not maybe_confirm(
                            args,
                            "Execute step {} {} gripper {}?".format(
                                step.get("step"),
                                step.get("action"),
                                command["name"],
                            ),
                        ):
                            node.get_logger().info("Gripper command skipped by user.")
                            continue
                        gripper.set_position(position)
                        status = gripper.wait_until_done(timeout=args.gripper_wait)
                        current = gripper.get_position()
                        node.get_logger().info(
                            "Gripper {} done: status={}, position={}".format(command["name"], status, current)
                        )
                        if command["name"] == "close" and step.get("action") == "pick" and args.post_close_wait > 0:
                            node.get_logger().info(
                                "Waiting {:.2f}s after gripper close before retreat.".format(args.post_close_wait)
                            )
                            time.sleep(float(args.post_close_wait))
                    continue

                current_tool = node.current_tool_transform(timeout=args.tf_timeout)
                current_pos, current_quat = transform_position_quat(current_tool)
                pre_rotate_selects_orientation = (
                    args.pre_rotate_before_translation
                    and not pre_rotated
                    and args.orientation_mode == "object-yaw"
                )
                if step_quat_xyzw is None and not pre_rotate_selects_orientation:
                    step_quat_xyzw, step_selected_yaw_deg, step_orientation_source = orientation_for_step(
                        step, args, current_quat
                    )
                    node.get_logger().info(
                        "Step {} orientation source={} object_yaw={} selected_yaw={} quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                            step.get("step"),
                            step_orientation_source,
                            step.get("target_yaw_deg"),
                            step_selected_yaw_deg,
                            step_quat_xyzw[0],
                            step_quat_xyzw[1],
                            step_quat_xyzw[2],
                            step_quat_xyzw[3],
                        )
                    )
                if args.pre_rotate_before_translation and not pre_rotated:
                    pre_rotated = True
                    selected_pre_rotate_trajectory = None
                    if step_quat_xyzw is None:
                        motion_velocity = node.moveit2.max_velocity
                        motion_acceleration = node.moveit2.max_acceleration
                        node.moveit2.max_velocity = args.pre_rotate_velocity
                        node.moveit2.max_acceleration = args.pre_rotate_acceleration
                        try:
                            selected = select_best_pre_rotate_plan(
                                node,
                                args,
                                step,
                                current_pos,
                                current_quat,
                                start_joint_state=step_start_state,
                            )
                        finally:
                            node.moveit2.max_velocity = motion_velocity
                            node.moveit2.max_acceleration = motion_acceleration
                        if selected is None:
                            node.get_logger().error(
                                "Skipping step {} remaining motions because no pre-rotate candidate planned.".format(
                                    step.get("step")
                                )
                            )
                            break
                        selected_candidate, selected_pre_rotate_trajectory = selected
                        step_quat_xyzw = selected_candidate["quat_xyzw"]
                        step_selected_yaw_deg = selected_candidate["selected_yaw_deg"]
                        step_orientation_source = selected_candidate["source"]
                        node.get_logger().info(
                            "Step {} orientation source={} object_yaw={} selected_yaw={} quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                                step.get("step"),
                                step_orientation_source,
                                step.get("target_yaw_deg"),
                                step_selected_yaw_deg,
                                step_quat_xyzw[0],
                                step_quat_xyzw[1],
                                step_quat_xyzw[2],
                                step_quat_xyzw[3],
                            )
                        )
                    quat_xyzw = step_quat_xyzw
                    pre_log = (
                        "Step {} {} pre_rotate: object={} current_tool0={} -> quat_xyzw=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]"
                        .format(
                            step.get("step"),
                            step.get("action"),
                            step.get("object_label"),
                            [round(v, 4) for v in current_pos],
                            quat_xyzw[0],
                            quat_xyzw[1],
                            quat_xyzw[2],
                            quat_xyzw[3],
                        )
                    )
                    pre_rotate_result = plan_and_maybe_execute_motion(
                        node,
                        args,
                        step,
                        "pre_rotate",
                        current_pos,
                        quat_xyzw,
                        current_pos,
                        pre_log,
                        cartesian=False,
                        trajectory=selected_pre_rotate_trajectory,
                        start_joint_state=step_start_state,
                        max_joint_delta_limit=args.max_pre_rotate_joint_delta,
                    )
                    if not pre_rotate_result:
                        node.get_logger().error(
                            "Skipping step {} remaining motions because pre-rotate did not complete.".format(
                                step.get("step")
                            )
                        )
                        break
                    if isinstance(pre_rotate_result, JointState):
                        step_start_state = pre_rotate_result
                    if args.execute:
                        current_tool = node.current_tool_transform(timeout=args.tf_timeout)
                        current_pos, current_quat = transform_position_quat(current_tool)
                        step_start_state = node.latest_joint_state
                        if step_selected_yaw_deg is not None:
                            actual_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
                            yaw_error = (
                                abs(shortest_yaw_delta_deg(step_selected_yaw_deg, actual_yaw))
                                if step.get("exact_tool_yaw_required")
                                else gripper_yaw_error_deg(step_selected_yaw_deg, actual_yaw)
                            )
                            node.get_logger().info(
                                "Pre-rotate yaw verification: target_yaw={:.2f} actual_tcp_yaw={:.2f} "
                                "yaw_error_deg={:.2f}".format(
                                    step_selected_yaw_deg,
                                    actual_yaw,
                                    yaw_error,
                                )
                            )
                            if yaw_error > float(args.max_grasp_yaw_error_deg):
                                raise RuntimeError(
                                    "Refusing translation after pre-rotate: yaw_error_deg {:.2f} exceeds {:.2f}. "
                                    "Check --pre-rotate-wrist-yaw-sign.".format(
                                        yaw_error, args.max_grasp_yaw_error_deg
                                    )
                                )

                quat_xyzw = step_quat_xyzw
                plan_position = command["position"]
                if (
                    args.execute
                    and step.get("action") == "pick"
                    and command["name"] == "target"
                    and step_selected_yaw_deg is not None
                ):
                    current_pos, current_quat, settled_state = settle_orientation_before_descent(
                        node,
                        args,
                        step,
                        step_quat_xyzw,
                        current_pos,
                        current_quat,
                    )
                    if settled_state is not None:
                        step_start_state = settled_state
                    orientation_error_deg = math.degrees(
                        quaternion_distance_rad(current_quat, step_quat_xyzw)
                    )
                    current_tcp_yaw = estimate_downward_family_yaw_deg(current_quat, args.quat_xyzw)
                    yaw_error = (
                        abs(shortest_yaw_delta_deg(step_selected_yaw_deg, current_tcp_yaw))
                        if step.get("exact_tool_yaw_required")
                        else gripper_yaw_error_deg(step_selected_yaw_deg, current_tcp_yaw)
                    )
                    print(
                        "Pick yaw gate: object_yaw={:.2f} grasp_yaw={:.2f} current_tcp_yaw={:.2f} "
                        "yaw_error_deg={:.2f} orientation_error_deg={:.2f}".format(
                            float(step.get("object_yaw_deg", step.get("estimated_yaw_deg", step_selected_yaw_deg))),
                            float(step_selected_yaw_deg),
                            float(current_tcp_yaw),
                            float(yaw_error),
                            float(orientation_error_deg),
                        ),
                        flush=True,
                    )
                    if yaw_error > float(args.max_grasp_yaw_error_deg):
                        raise RuntimeError(
                            "Refusing pick descent: yaw_error_deg {:.2f} exceeds {:.2f}.".format(
                                yaw_error, args.max_grasp_yaw_error_deg
                            )
                        )
                    if orientation_error_deg > float(args.max_grasp_orientation_error_deg):
                        raise RuntimeError(
                            "Refusing pick descent: full orientation_error_deg {:.2f} exceeds {:.2f}. "
                            "Gripper plane is not level enough.".format(
                                orientation_error_deg,
                                args.max_grasp_orientation_error_deg,
                            )
                        )
                if tcp_offset_tool is not None:
                    tcp_target = add_base_offset(plan_position, args.tcp_target_offset_base)
                    tool_goal = tool0_goal_from_tcp(tcp_target, quat_xyzw, tcp_offset_tool)
                    effective_tool_offset = None
                else:
                    tcp_target = plan_position
                    effective_tool_offset = tool_offset_for_step(args, step_selected_yaw_deg)
                    tool_goal = tool0_goal_from_approach(plan_position, args.tool_z_offset, effective_tool_offset)

                log_message = (
                    "Step {} {} {}: object={} ref={} plan_position={} tcp_target={} tool_offset={} -> tool0_goal={}".format(
                        step.get("step"),
                        step.get("action"),
                        command["name"],
                        step.get("object_label"),
                        step.get("reference_label"),
                        [round(v, 4) for v in plan_position],
                        [round(v, 4) for v in tcp_target],
                        None if effective_tool_offset is None else [round(v, 4) for v in effective_tool_offset],
                        [round(v, 4) for v in tool_goal],
                    )
                )
                motion_result = plan_and_maybe_execute_motion(
                    node,
                    args,
                    step,
                    command["name"],
                    tool_goal,
                    quat_xyzw,
                    current_pos,
                    log_message,
                    start_joint_state=step_start_state,
                )
                if not motion_result:
                    continue
                if isinstance(motion_result, JointState):
                    step_start_state = motion_result
            planning_start_state = step_start_state
    finally:
        if gripper is not None:
            gripper.close()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
