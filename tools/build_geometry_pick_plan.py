#!/usr/bin/env python3
"""Build a deterministic one-object pick plan from private_scene_state.json."""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.decision_to_execution import compile_plan, write_json
from tools.xy_correction import apply_step_xy_correction, load_xy_correction


DEFAULT_PRIVATE = "/tmp/robot_scene_geometry/private_scene_state.json"
DEFAULT_OUTPUT = "/tmp/robot_scene_geometry/geometry_pick_plan.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a pick plan without LLM reasoning.")
    parser.add_argument("--private-state-json", default=DEFAULT_PRIVATE)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--object-id", type=int)
    group.add_argument("--object-label")
    parser.add_argument(
        "--nearest-base-xy",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="When a label matches multiple objects, select the one nearest this base_link XY point.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--approach-height-m", type=float, default=0.10)
    parser.add_argument("--pick-target-lift-m", type=float, default=0.0)
    parser.add_argument(
        "--force-yaw-labels",
        default="square",
        help="Comma-separated label substrings that should use table_yaw_deg even when aspect-ratio yaw validity is false.",
    )
    parser.add_argument("--stack-demo-mode", action="store_true")
    parser.add_argument("--require-geometry-center", action="store_true")
    parser.add_argument("--fixed-square-yaw-deg", type=float, default=0.0)
    parser.add_argument("--stack-square-yaw-mode", choices=("detected", "fixed"), default="detected")
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--stack-grasp-axis", choices=("long", "short"), default="long")
    parser.add_argument("--gripper-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--xy-correction-json", default="")
    return parser.parse_args()


def object_id_value(obj):
    try:
        return int(obj.get("id", -1))
    except (TypeError, ValueError):
        return -1


def object_base_point(obj, require_geometry_center=False):
    geometry_point = obj.get("geometry_center_m") if obj.get("geometry_frame") == "base_link" else None
    if require_geometry_center and not obj.get("pointcloud_geometry_valid"):
        return None
    point = geometry_point if require_geometry_center else (geometry_point or obj.get("center_3d_base_m"))
    if not point or len(point) < 2:
        return None
    try:
        output = [float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0]
        return output if all(math.isfinite(value) for value in output) else None
    except (TypeError, ValueError):
        return None


def object_sort_key(obj):
    point = object_base_point(obj)
    if point is None:
        return (float("inf"), float("inf"), object_id_value(obj))
    return (point[0], point[1], object_id_value(obj))


def describe_object(obj):
    point = object_base_point(obj)
    xy = "unknown"
    if point is not None:
        xy = "[{:.4f}, {:.4f}]".format(point[0], point[1])
    return "id={} label={} base_xy={}".format(obj.get("id"), obj.get("label"), xy)


def label_matches_any(label, substrings):
    label = str(label or "").strip().lower()
    return any(value and value in label for value in substrings)


def force_object_yaw_if_requested(step, obj, force_yaw_labels):
    labels = [value.strip().lower() for value in str(force_yaw_labels or "").split(",")]
    yaw = obj.get("table_yaw_deg")
    if step.get("target_yaw_valid") or yaw is None:
        return
    if not label_matches_any(obj.get("label"), labels):
        return
    step["target_yaw_deg"] = float(yaw)
    step["target_yaw_valid"] = True
    step["yaw_frame"] = obj.get("geometry_frame") or "base_link"
    step["yaw_forced_from_invalid_aspect_ratio"] = True
    print(
        "Forced object yaw for label '{}': yaw={:.2f} deg was table_yaw_valid=false.".format(
            obj.get("label"),
            float(yaw),
        ),
        flush=True,
    )


def normalize_equivalent_yaw(yaw_deg, period_deg):
    half = float(period_deg) / 2.0
    return ((float(yaw_deg) + half) % float(period_deg)) - half


def set_stack_demo_yaw(
    step,
    obj,
    fixed_square_yaw_deg=0.0,
    square_yaw_mode="detected",
    grasp_axis="long",
    gripper_yaw_offset_deg=0.0,
    square_yaw_snap_tolerance_deg=5.0,
):
    label = str(obj.get("label", "")).lower()
    estimated_yaw = obj.get("table_yaw_deg")
    is_square = "square" in label
    is_rectangle = "rectangle" in label
    if not (is_square or is_rectangle):
        if not obj.get("table_yaw_valid") or estimated_yaw is None:
            raise RuntimeError("Stack demo object '{}' has no reliable grasp yaw.".format(obj.get("label")))
        chosen_yaw = normalize_equivalent_yaw(estimated_yaw, 180.0)
    elif is_square and square_yaw_mode == "fixed":
        chosen_yaw = float(fixed_square_yaw_deg)
    else:
        if estimated_yaw is None:
            raise RuntimeError("Stack demo block '{}' is missing minAreaRect/PCA yaw.".format(obj.get("label")))
        chosen_yaw = normalize_equivalent_yaw(estimated_yaw, 90.0 if is_square else 180.0)
        if is_square and abs(chosen_yaw) <= max(0.0, float(square_yaw_snap_tolerance_deg)):
            chosen_yaw = 0.0

    grasp_yaw = chosen_yaw + (90.0 if grasp_axis == "short" else 0.0) + float(gripper_yaw_offset_deg)
    grasp_yaw = normalize_equivalent_yaw(grasp_yaw, 180.0)
    step["estimated_yaw_deg"] = None if estimated_yaw is None else float(estimated_yaw)
    step["object_yaw_deg"] = float(chosen_yaw)
    step["target_yaw_deg"] = float(grasp_yaw)
    step["chosen_grasp_yaw_deg"] = float(grasp_yaw)
    step["requested_grasp_axis"] = grasp_axis
    step["gripper_yaw_offset_deg"] = float(gripper_yaw_offset_deg)
    step["grasp_axis"] = "long"
    step["target_yaw_valid"] = True
    step["exact_tool_yaw_required"] = not is_square
    step["yaw_equivalence_period_deg"] = 90.0 if is_square else 180.0
    step["preserve_current_yaw"] = False
    step["yaw_frame"] = "base_link"
    step["yaw_source"] = (
        "stack_demo_fixed_square"
        if is_square and square_yaw_mode == "fixed"
        else "stack_demo_{}_90deg_equivalent".format(obj.get("table_yaw_source") or "detected")
    )
    step.pop("yaw_forced_from_invalid_aspect_ratio", None)


def find_object(state, object_id=None, object_label=None, nearest_base_xy=None):
    objects = state.get("objects", [])
    if object_id is not None:
        matches = [obj for obj in objects if object_id_value(obj) == int(object_id)]
    else:
        requested = str(object_label).strip().lower()
        matches = [obj for obj in objects if requested in str(obj.get("label", "")).lower()]
    if not matches:
        raise RuntimeError("Requested object was not found in private scene state.")
    if len(matches) > 1 and object_id is None and nearest_base_xy is not None:
        reference = [float(nearest_base_xy[0]), float(nearest_base_xy[1])]
        matches = sorted(
            matches,
            key=lambda obj: (
                sum((object_base_point(obj)[index] - reference[index]) ** 2 for index in range(2))
                if object_base_point(obj) is not None
                else float("inf"),
                object_sort_key(obj),
            ),
        )
        print(
            "Multiple objects matched label '{}'; selected nearest base_link XY to {}: {}.".format(
                object_label,
                reference,
                describe_object(matches[0]),
            ),
            flush=True,
        )
        return matches[0]
    if len(matches) > 1 and object_id is None:
        matches = sorted(matches, key=object_sort_key)
        print(
            "Multiple objects matched label '{}'; selected smallest base_link x/y: {}. Candidates: {}".format(
                object_label,
                describe_object(matches[0]),
                "; ".join(describe_object(obj) for obj in matches),
            ),
            flush=True,
        )
    return sorted(matches, key=object_sort_key)[0]


def main():
    args = parse_args()
    with open(args.private_state_json, "r", encoding="utf-8") as f:
        state = json.load(f)
    obj = find_object(
        state,
        object_id=args.object_id,
        object_label=args.object_label,
        nearest_base_xy=args.nearest_base_xy,
    )
    require_geometry_center = bool(args.require_geometry_center or args.stack_demo_mode)
    geometry_center = object_base_point(obj, require_geometry_center=True)
    if require_geometry_center and geometry_center is None:
        raise RuntimeError("Stack demo requires a valid base_link geometry_center_m; bbox/depth-center fallback is disabled.")
    decision = {
        "action_plan": [
            {
                "step": 1,
                "action": "pick",
                "object_id": int(obj["id"]),
                "reference_object_id": None,
                "relative_position": None,
                "reason": "Deterministic geometry pick test.",
            }
        ]
    }
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
    if plan.get("steps"):
        step = plan["steps"][0]
        if require_geometry_center:
            step["target_position_m"][:2] = [round(value, 5) for value in geometry_center[:2]]
            step["approach_position_m"][:2] = [round(value, 5) for value in geometry_center[:2]]
            step["coordinate_source"] = "required_geometry_center_m"
        correction = load_xy_correction(args.xy_correction_json)
        if geometry_center is not None:
            apply_step_xy_correction(step, geometry_center[:2], correction=correction, kind="grasp")
            step["grasp_offset_from_geometry_center_m"] = step["grasp_xy_bias_m"]
        if args.stack_demo_mode:
            set_stack_demo_yaw(
                step,
                obj,
                args.fixed_square_yaw_deg,
                args.stack_square_yaw_mode,
                args.stack_grasp_axis,
                args.gripper_yaw_offset_deg,
                args.square_yaw_snap_tolerance_deg,
            )
        else:
            force_object_yaw_if_requested(step, obj, args.force_yaw_labels)
    write_json(args.output, plan)
    step = plan["steps"][0]
    print("Saved geometry pick plan: {}".format(args.output))
    print(
        "object_id={} color_class={} center_base={} estimated_yaw={} chosen_grasp_yaw={} grasp_pose={} valid={} frame={}".format(
            step.get("object_id"),
            step.get("object_label"),
            step.get("geometry_center_m"),
            step.get("estimated_yaw_deg"),
            step.get("chosen_grasp_yaw_deg") or step.get("target_yaw_deg"),
            step.get("target_position_m"),
            step.get("target_yaw_valid"),
            step.get("yaw_frame"),
        )
    )


if __name__ == "__main__":
    main()
