"""Convert an LLM symbolic decision into coordinate targets for execution.

This script does not drive the robot.  It creates a checked execution plan that
an arm-specific controller can consume.
"""

import argparse
import json
import math
import os


DEFAULT_DECISION = "/tmp/scene_reasoning_decision.json"
DEFAULT_PRIVATE = "/tmp/scene_state_private.json"
DEFAULT_OUT = "/tmp/robot_execution_plan.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Compile symbolic LLM decision into robot execution targets.")
    parser.add_argument("--decision-json", default=DEFAULT_DECISION)
    parser.add_argument("--private-state-json", default=DEFAULT_PRIVATE)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--place-offset-m", type=float, default=0.08)
    parser.add_argument("--approach-height-m", type=float, default=0.08)
    parser.add_argument(
        "--pick-target-lift-m",
        type=float,
        default=0.0,
        help="Extra z lift for pick target points in base frame.",
    )
    parser.add_argument(
        "--place-target-lift-m",
        type=float,
        default=0.0,
        help="Extra z lift for place_relative target points in base frame.",
    )
    parser.add_argument(
        "--left-right-axis",
        choices=("x", "y"),
        default="y",
        help="Base-frame axis used for left_of/right_of placement.",
    )
    parser.add_argument(
        "--left-direction-sign",
        choices=("positive", "negative"),
        default="positive",
        help="Direction for left_of along --left-right-axis. right_of uses the opposite direction.",
    )
    parser.add_argument(
        "--front-back-axis",
        choices=("x", "y"),
        default="x",
        help="Base-frame axis used for in_front_of/behind placement.",
    )
    parser.add_argument(
        "--front-direction-sign",
        choices=("positive", "negative"),
        default="positive",
        help="Direction for in_front_of along --front-back-axis. behind uses the opposite direction.",
    )
    return parser.parse_args()


def load_json_or_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def object_map(private_state):
    return {int(obj["id"]): obj for obj in private_state.get("objects", [])}


def workspace_object(objects):
    candidates = [obj for obj in objects.values() if obj.get("is_workspace") or obj.get("label") == "workspace"]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.get("confidence", 0.0))


def point_for_object(objects, object_id):
    if object_id is None:
        return None
    obj = objects.get(int(object_id))
    if not obj:
        return None
    geometry_point = obj.get("geometry_center_m") if obj.get("geometry_frame") == "base_link" else None
    point = geometry_point or obj.get("center_3d_base_m") or obj.get("center_3d_m")
    if not point or len(point) < 3:
        return None
    return [float(point[0]), float(point[1]), float(point[2])]


def object_label(objects, object_id):
    obj = objects.get(int(object_id)) if object_id is not None else None
    return obj.get("label") if obj else None


def object_geometry(objects, object_id):
    obj = objects.get(int(object_id)) if object_id is not None else None
    if not obj:
        return {}
    return {
        "geometry_center_m": obj.get("geometry_center_m") if obj.get("geometry_frame") == "base_link" else None,
        "geometry_frame": obj.get("geometry_frame"),
        "pointcloud_geometry_valid": bool(obj.get("pointcloud_geometry_valid")),
        "object_dimensions_m": obj.get("dimensions_m"),
        "target_yaw_deg": obj.get("table_yaw_deg") if obj.get("table_yaw_valid") else None,
        "target_yaw_valid": bool(obj.get("table_yaw_valid")),
        "estimated_yaw_deg": obj.get("table_yaw_deg"),
        "yaw_source": obj.get("table_yaw_source"),
        "yaw_frame": obj.get("geometry_frame"),
    }


def _config_value(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def compile_place_on_top(held_object, stack_state, config):
    """Compile one dynamic place action from the latest observed stack state."""
    if not stack_state.get("valid"):
        raise ValueError("Cannot place on an invalid stack state.")
    stack_xy = stack_state.get("placement_base_center_xy_m") or stack_state.get("stack_xy_base_m")
    top_z = stack_state.get("placement_base_top_z_m")
    if top_z is None:
        top_z = stack_state.get("top_z_base_m")
    dimensions = held_object.get("dimensions_m") if isinstance(held_object, dict) else None
    if not isinstance(stack_xy, (list, tuple)) or len(stack_xy) < 2 or top_z is None:
        raise ValueError("Stack state must contain stack_xy_base_m and top_z_base_m.")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3:
        raise ValueError("Held object must contain dimensions_m for dynamic placement Z.")
    values = [float(stack_xy[0]), float(stack_xy[1]), float(top_z), float(dimensions[2])]
    if not all(math.isfinite(value) for value in values) or values[3] <= 0.0:
        raise ValueError("Place-on-top geometry values must be finite and object height must be positive.")

    place_yaw = float(_config_value(config, "fixed_yaw_deg", 0.0))
    object_offset_tool = _config_value(config, "object_offset_tool_xy_m", [0.0, 0.0])
    place_bias_base = _config_value(config, "place_bias_base_xy_m", [0.0, 0.0])
    if (
        not isinstance(object_offset_tool, (list, tuple))
        or len(object_offset_tool) != 2
        or not isinstance(place_bias_base, (list, tuple))
        or len(place_bias_base) != 2
    ):
        raise ValueError("object_offset_tool_xy_m and place_bias_base_xy_m must each contain two values.")
    offset_values = [float(value) for value in object_offset_tool]
    bias_values = [float(value) for value in place_bias_base]
    if not all(math.isfinite(value) for value in offset_values + bias_values):
        raise ValueError("Place XY offset values must be finite.")
    max_place_bias = float(_config_value(config, "max_place_bias_base_m", 0.01))
    place_bias_norm = math.hypot(bias_values[0], bias_values[1])
    if place_bias_norm > max_place_bias:
        raise ValueError(
            "place_bias_base_xy_m norm {:.4f} exceeds small final compensation limit {:.4f}.".format(
                place_bias_norm, max_place_bias
            )
        )
    yaw_rad = math.radians(place_yaw)
    rotated_offset = [
        math.cos(yaw_rad) * offset_values[0] - math.sin(yaw_rad) * offset_values[1],
        math.sin(yaw_rad) * offset_values[0] + math.cos(yaw_rad) * offset_values[1],
    ]
    target_xy = [
        float(stack_xy[0]) - rotated_offset[0] + bias_values[0],
        float(stack_xy[1]) - rotated_offset[1] + bias_values[1],
    ]
    top_z_bias = float(_config_value(config, "place_top_z_bias_m", 0.0))
    if not math.isfinite(top_z_bias) or abs(top_z_bias) > 0.01:
        raise ValueError("place_top_z_bias_m must be finite and within +/-0.010 m.")
    corrected_top_z = float(top_z) + top_z_bias
    release_gap = float(_config_value(config, "release_gap_m", 0.001))
    if not math.isfinite(release_gap) or release_gap < 0.0:
        raise ValueError("release_gap_m must be finite and non-negative.")
    if release_gap > 0.01:
        raise ValueError("release_gap_m {:.4f} exceeds the 0.0100 m no-drop limit.".format(release_gap))
    release_z = corrected_top_z + values[3] / 2.0 + release_gap
    approach_height = float(_config_value(config, "approach_height_m", 0.05))
    pre_place_z = release_z + approach_height
    object_id = held_object.get("id") if isinstance(held_object, dict) else held_object
    return {
        "step": int(_config_value(config, "step", 1)),
        "action": "place_on_top",
        "object_id": object_id,
        "object_label": held_object.get("label") if isinstance(held_object, dict) else None,
        "reference_object_id": stack_state.get("top_object_id"),
        "reference_label": None,
        "relative_position": "on_top",
        "reason": "Place held object on the currently observed stack top.",
        "status": "planned",
        "coordinate_frame": "base_frame",
        "coordinate_source": "close_observation.placement_base_center_xy_and_top_z",
        "target_position_m": [round(target_xy[0], 5), round(target_xy[1], 5), round(release_z, 5)],
        "release_position_m": [round(target_xy[0], 5), round(target_xy[1], 5), round(release_z, 5)],
        "approach_position_m": [
            round(target_xy[0], 5),
            round(target_xy[1], 5),
            round(pre_place_z, 5),
        ],
        "pre_place_z_base_m": float(pre_place_z),
        "release_z_base_m": float(release_z),
        "stack_top_z_base_m": float(top_z),
        "corrected_stack_top_z_base_m": corrected_top_z,
        "place_top_z_bias_m": top_z_bias,
        "held_object_height_m": values[3],
        "release_gap_m": release_gap,
        "object_offset_tool_xy_m": offset_values,
        "object_offset_base_xy_m": rotated_offset,
        "place_bias_base_xy_m": bias_values,
        "place_bias_base_norm_m": place_bias_norm,
        "tcp_place_xy_base_m": target_xy,
        "stack_center_xy_base_m": [float(stack_xy[0]), float(stack_xy[1])],
        "placement_base_object_id": stack_state.get("placement_base_object_id") or stack_state.get("top_object_id"),
        "target_yaw_deg": place_yaw,
        "target_yaw_valid": True,
        "exact_tool_yaw_required": True,
        "yaw_frame": "base_link",
        "yaw_source": "stack_demo_fixed",
    }


def axis_index(axis):
    return 0 if axis == "x" else 1


def direction_sign(sign):
    return 1.0 if sign == "positive" else -1.0


def target_for_relative(
    reference_point,
    relative_position,
    offset_m,
    left_right_axis="y",
    left_direction_sign="positive",
    front_back_axis="x",
    front_direction_sign="positive",
):
    if reference_point is None:
        return None
    target = list(reference_point)
    left_axis_index = axis_index(left_right_axis)
    left_sign = direction_sign(left_direction_sign)
    front_axis_index = axis_index(front_back_axis)
    front_sign = direction_sign(front_direction_sign)
    if relative_position == "left_of":
        target[left_axis_index] += left_sign * offset_m
    elif relative_position == "right_of":
        target[left_axis_index] -= left_sign * offset_m
    elif relative_position == "in_front_of":
        target[front_axis_index] += front_sign * offset_m
    elif relative_position == "behind":
        target[front_axis_index] -= front_sign * offset_m
    elif relative_position in ("near", "on_surface", "center_of_workspace", None):
        pass
    else:
        return None
    return target


def normalize_steps(decision):
    if isinstance(decision.get("action_plan"), list):
        return decision["action_plan"]
    if decision.get("action"):
        return [decision]
    raise ValueError("Decision JSON must contain action_plan or action.")


def compile_step(step, objects, args):
    action = step.get("action")
    object_id = step.get("object_id")
    reference_id = step.get("reference_object_id")
    relative_position = step.get("relative_position")
    workspace = workspace_object(objects)

    compiled = {
        "step": step.get("step"),
        "action": action,
        "object_id": object_id,
        "object_label": object_label(objects, object_id),
        "reference_object_id": reference_id,
        "reference_label": object_label(objects, reference_id),
        "relative_position": relative_position,
        "reason": step.get("reason", ""),
        "status": "planned",
    }
    compiled.update(object_geometry(objects, object_id))

    if action in ("ask_user", "stop"):
        compiled["status"] = "requires_no_motion"
        return compiled

    object_point = point_for_object(objects, object_id)
    reference_point = point_for_object(objects, reference_id)

    if relative_position == "center_of_workspace" and workspace:
        reference_point = point_for_object(objects, workspace["id"])
        compiled["reference_object_id"] = workspace["id"]
        compiled["reference_label"] = workspace.get("label")

    if action in ("pick", "move_above"):
        target = object_point
    elif action == "place_relative":
        target = target_for_relative(
            reference_point,
            relative_position,
            args.place_offset_m,
            getattr(args, "left_right_axis", "y"),
            getattr(args, "left_direction_sign", "positive"),
            getattr(args, "front_back_axis", "x"),
            getattr(args, "front_direction_sign", "positive"),
        )
    else:
        compiled["status"] = "unsupported_action"
        return compiled

    if target is None:
        compiled["status"] = "missing_coordinate_or_reference"
        return compiled

    coordinate_frame = private_state_frame(objects, object_id, reference_id, workspace)
    if action == "pick" and coordinate_frame == "base_frame":
        target[2] += float(getattr(args, "pick_target_lift_m", 0.0))
    if action == "place_relative" and coordinate_frame == "base_frame":
        target[2] += float(getattr(args, "place_target_lift_m", 0.0))
    if coordinate_frame == "base_frame":
        approach = [target[0], target[1], target[2] + args.approach_height_m]
    else:
        approach = [target[0], target[1] - args.approach_height_m, target[2]]
    compiled["target_position_m"] = [round(value, 5) for value in target]
    compiled["approach_position_m"] = [round(value, 5) for value in approach]
    compiled["coordinate_source"] = "private_scene_state"
    compiled["coordinate_frame"] = coordinate_frame
    return compiled


def private_state_frame(objects, object_id, reference_id, workspace):
    for candidate_id in (object_id, reference_id, workspace.get("id") if workspace else None):
        if candidate_id is None:
            continue
        obj = objects.get(int(candidate_id))
        if not obj:
            continue
        if obj.get("center_3d_base_m"):
            return "base_frame"
    return "camera_frame"


def compile_plan(decision, private_state, args):
    objects = object_map(private_state)
    steps = normalize_steps(decision)
    compiled_steps = [compile_step(step, objects, args) for step in steps]
    uses_base_frame = any(step.get("coordinate_frame") == "base_frame" for step in compiled_steps)
    coordinate_convention = private_state.get("coordinate_convention")
    if uses_base_frame:
        coordinate_convention = {
            "frame": private_state.get("base_frame") or "base_link",
            "unit": "meter",
            "x": "robot base x axis",
            "y": "robot base y axis",
            "z": "positive upward from robot base",
        }
    return {
        "schema_version": "robot_execution_plan_v1",
        "frame_id": private_state.get("frame_id"),
        "timestamp": private_state.get("timestamp"),
        "execution_status": "not_executed",
        "coordinate_convention": coordinate_convention,
        "steps": compiled_steps,
        "safety_checks": [
            "verify target object is still visible before motion",
            "verify depth and hand-eye transform before sending robot command",
            "use approach_position_m before descending to target_position_m",
            "stop if compiled step status is not planned",
        ],
    }


def main():
    args = parse_args()
    decision = load_json_or_text(args.decision_json)
    private_state = load_json_or_text(args.private_state_json)
    plan = compile_plan(decision, private_state, args)
    write_json(args.output, plan)
    print("Saved robot execution plan: {}".format(args.output))


if __name__ == "__main__":
    main()
