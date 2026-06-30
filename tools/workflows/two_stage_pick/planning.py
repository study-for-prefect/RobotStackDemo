"""Execution-plan validation and second-snapshot correction."""

import json
import math
import numpy as np

from .io import load_json, write_json

def first_planned_step(plan):
    for step in plan.get("steps", []):
        if step.get("status") == "planned" and step.get("target_position_m"):
            return step
    raise RuntimeError("No planned target step found.")


def select_llm_pick_step(plan):
    for index, step in enumerate(plan.get("steps", [])):
        if step.get("action") == "pick" and step.get("status") == "planned" and step.get("object_id") is not None:
            return index, step
    raise RuntimeError("LLM execution plan has no planned pick step with an object_id.")


def remaining_plan_after_step(plan, selected_index):
    remaining = json.loads(json.dumps(plan))
    remaining["steps"] = remaining.get("steps", [])[selected_index + 1 :]
    remaining["execution_status"] = "not_executed"
    return remaining


def reject_additional_pick_steps(plan, selected_index):
    additional = [
        step
        for step in plan.get("steps", [])[selected_index + 1 :]
        if step.get("action") == "pick" and step.get("status") == "planned"
    ]
    if additional:
        raise RuntimeError(
            "Integrated two-stage execution supports one pick per run; found another planned pick at step {}.".format(
                additional[0].get("step")
            )
        )


def reject_incomplete_remaining_motion_steps(plan, selected_index):
    for step in plan.get("steps", [])[selected_index + 1 :]:
        if step.get("action") in ("ask_user", "stop"):
            continue
        if step.get("status") != "planned" or not step.get("approach_position_m") or not step.get("target_position_m"):
            raise RuntimeError(
                "LLM step {} after pick is not executable: action={} status={}.".format(
                    step.get("step"),
                    step.get("action"),
                    step.get("status"),
                )
            )


def has_planned_motion(plan):
    return any(
        step.get("status") == "planned" and step.get("approach_position_m")
        for step in plan.get("steps", [])
    )


def plan_has_reliable_yaw(path):
    step = first_planned_step(load_json(path))
    return bool(step.get("target_yaw_valid") and step.get("target_yaw_deg") is not None)


def current_tcp_position_from_tf_json(tf_json_path, tcp_offset_tool):
    payload = load_json(tf_json_path)
    tool_transform = payload.get("tool_transform")
    if not tool_transform:
        raise RuntimeError(
            "{} has no tool_transform. Regenerate it with tools/robot/tf_lookup_json.py --require-tool.".format(
                tf_json_path
            )
        )
    matrix = np.asarray(tool_transform["matrix_4x4"], dtype=float)
    offset = np.asarray([float(value) for value in tcp_offset_tool[:3]] + [1.0], dtype=float)
    tcp_base = matrix.dot(offset)
    return [float(tcp_base[index]) for index in range(3)]


def valid_geometry_center(step, name):
    center = step.get("geometry_center_m")
    if (
        step.get("pointcloud_geometry_valid") is not True
        or
        step.get("geometry_frame") != "base_link"
        or not isinstance(center, list)
        or len(center) < 3
    ):
        raise RuntimeError("{} plan step is missing base_link geometry_center_m.".format(name))
    try:
        center = [float(value) for value in center[:3]]
    except (TypeError, ValueError):
        raise RuntimeError("{} geometry_center_m is not numeric.".format(name))
    if not all(math.isfinite(value) for value in center):
        raise RuntimeError("{} geometry_center_m is not finite.".format(name))
    return center


def safe_name(value):
    output = []
    for char in str(value).strip().lower():
        if char.isalnum():
            output.append(char)
        elif output and output[-1] != "_":
            output.append("_")
    return "".join(output).strip("_") or "object"


def build_corrected_plan(
    first_plan_path,
    second_plan_path,
    output_path,
    report_path,
    max_correction_m,
    max_grasp_offset_m=0.05,
    use_second_yaw=False,
    use_second_grasp_offset=False,
):
    first_plan = load_json(first_plan_path)
    second_plan = load_json(second_plan_path)
    first_step = first_planned_step(first_plan)
    second_step = first_planned_step(second_plan)
    first_target = [float(value) for value in first_step["target_position_m"]]
    second_target = [float(value) for value in second_step["target_position_m"]]
    first_geometry = valid_geometry_center(first_step, "First")
    second_geometry = valid_geometry_center(second_step, "Second")
    first_grasp_offset = [
        first_target[0] - first_geometry[0],
        first_target[1] - first_geometry[1],
    ]
    second_grasp_offset = [
        second_target[0] - second_geometry[0],
        second_target[1] - second_geometry[1],
    ]
    grasp_offset = second_grasp_offset if use_second_grasp_offset else first_grasp_offset
    grasp_offset_norm = math.hypot(grasp_offset[0], grasp_offset[1])
    if grasp_offset_norm > float(max_grasp_offset_m):
        raise RuntimeError(
            "Selected grasp offset {:.4f} m exceeds limit {:.4f} m.".format(
                grasp_offset_norm, max_grasp_offset_m
            )
        )
    corrected_xy = [
        second_geometry[0] + grasp_offset[0],
        second_geometry[1] + grasp_offset[1],
    ]
    delta = [corrected_xy[0] - first_target[0], corrected_xy[1] - first_target[1], 0.0]
    correction_norm = (delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5
    if correction_norm > float(max_correction_m):
        raise RuntimeError(
            "Second-snapshot XY correction {:.4f} m exceeds limit {:.4f} m.".format(
                correction_norm, max_correction_m
            )
        )

    corrected = json.loads(json.dumps(first_plan))
    corrected_step = first_planned_step(corrected)
    for key in ("target_position_m", "approach_position_m"):
        position = corrected_step.get(key)
        if position:
            corrected_step[key] = [
                round(corrected_xy[0], 5),
                round(corrected_xy[1], 5),
                round(float(position[2]), 5),
            ]
    corrected_step["second_snapshot_delta_base_xy_m"] = [round(delta[0], 5), round(delta[1], 5)]
    corrected_step["second_geometry_center_m"] = second_geometry
    corrected_step["grasp_offset_from_geometry_center_m"] = [round(value, 6) for value in grasp_offset]
    corrected_step["coordinate_source"] = (
        "second_geometry_center_plus_second_snapshot_grasp_offset"
        if use_second_grasp_offset
        else "second_geometry_center_plus_first_grasp_offset"
    )
    if use_second_yaw:
        for key in (
            "estimated_yaw_deg",
            "object_yaw_deg",
            "target_yaw_deg",
            "chosen_grasp_yaw_deg",
            "gripper_yaw_offset_deg",
            "requested_grasp_axis",
            "grasp_axis",
            "target_yaw_valid",
            "exact_tool_yaw_required",
            "yaw_frame",
            "yaw_source",
        ):
            if key in second_step:
                corrected_step[key] = second_step[key]
        corrected_step["yaw_source"] = "second_snapshot_{}".format(second_step.get("yaw_source") or "detected")
    write_json(output_path, corrected)

    report = {
        "first_plan": first_plan_path,
        "second_plan": second_plan_path,
        "corrected_plan": output_path,
        "first_target_position_m": first_target,
        "second_target_position_m": second_target,
        "first_geometry_center_m": first_geometry,
        "second_geometry_center_m": second_geometry,
        "first_grasp_offset_xy_m": first_grasp_offset,
        "second_grasp_offset_xy_m": second_grasp_offset,
        "selected_grasp_offset_xy_m": grasp_offset,
        "grasp_offset_norm_m": grasp_offset_norm,
        "delta_base_xy_m": [delta[0], delta[1]],
        "correction_norm_m": correction_norm,
        "first_target_yaw_deg": first_step.get("target_yaw_deg"),
        "second_target_yaw_deg": second_step.get("target_yaw_deg"),
        "final_target_yaw_deg": corrected_step.get("target_yaw_deg"),
        "used_second_yaw": bool(use_second_yaw),
        "used_second_grasp_offset": bool(use_second_grasp_offset),
    }
    write_json(report_path, report)
    print(
        "\nSecond-snapshot correction: delta_base_xy=[{:.4f}, {:.4f}] m norm={:.4f} m".format(
            delta[0], delta[1], correction_norm
        ),
        flush=True,
    )


def build_tcp_error_corrected_plan(
    first_plan_path,
    second_plan_path,
    tf_json_path,
    output_path,
    report_path,
    max_correction_m,
    max_grasp_offset_m=0.05,
    tcp_offset_tool=None,
):
    first_plan = load_json(first_plan_path)
    second_plan = load_json(second_plan_path)
    first_step = first_planned_step(first_plan)
    second_step = first_planned_step(second_plan)
    first_target = [float(value) for value in first_step["target_position_m"]]
    first_geometry = valid_geometry_center(first_step, "First")
    second_geometry = valid_geometry_center(second_step, "Second")
    desired_tcp_to_target_xy = [
        first_target[0] - first_geometry[0],
        first_target[1] - first_geometry[1],
    ]
    desired_norm = math.hypot(desired_tcp_to_target_xy[0], desired_tcp_to_target_xy[1])
    if desired_norm > float(max_grasp_offset_m):
        raise RuntimeError(
            "Selected TCP-to-target offset {:.4f} m exceeds limit {:.4f} m.".format(
                desired_norm, max_grasp_offset_m
            )
        )
    current_tcp = current_tcp_position_from_tf_json(tf_json_path, tcp_offset_tool or [0.0, 0.0, 0.15])
    observed_tcp_to_target_xy = [
        current_tcp[0] - second_geometry[0],
        current_tcp[1] - second_geometry[1],
    ]
    delta = [
        desired_tcp_to_target_xy[0] - observed_tcp_to_target_xy[0],
        desired_tcp_to_target_xy[1] - observed_tcp_to_target_xy[1],
        0.0,
    ]
    correction_norm = math.hypot(delta[0], delta[1])
    if correction_norm > float(max_correction_m):
        raise RuntimeError(
            "Second-snapshot TCP-target correction {:.4f} m exceeds limit {:.4f} m.".format(
                correction_norm, max_correction_m
            )
        )

    corrected = json.loads(json.dumps(first_plan))
    corrected_step = first_planned_step(corrected)
    for key in ("target_position_m", "approach_position_m"):
        position = corrected_step.get(key)
        if position:
            corrected_step[key] = [
                round(float(position[0]) + delta[0], 5),
                round(float(position[1]) + delta[1], 5),
                round(float(position[2]), 5),
            ]
    corrected_step["second_snapshot_delta_base_xy_m"] = [round(delta[0], 5), round(delta[1], 5)]
    corrected_step["second_geometry_center_m"] = second_geometry
    corrected_step["current_tcp_position_base_m"] = [round(value, 6) for value in current_tcp]
    corrected_step["desired_tcp_to_target_xy_m"] = [round(value, 6) for value in desired_tcp_to_target_xy]
    corrected_step["observed_tcp_to_target_xy_m"] = [round(value, 6) for value in observed_tcp_to_target_xy]
    corrected_step["coordinate_source"] = "locked_plan_plus_second_snapshot_tcp_target_error"
    write_json(output_path, corrected)

    report = {
        "first_plan": first_plan_path,
        "second_plan": second_plan_path,
        "tf_json": tf_json_path,
        "corrected_plan": output_path,
        "first_target_position_m": first_target,
        "first_geometry_center_m": first_geometry,
        "second_geometry_center_m": second_geometry,
        "current_tcp_position_base_m": current_tcp,
        "desired_tcp_to_target_xy_m": desired_tcp_to_target_xy,
        "observed_tcp_to_target_xy_m": observed_tcp_to_target_xy,
        "delta_base_xy_m": [delta[0], delta[1]],
        "correction_norm_m": correction_norm,
        "max_correction_m": float(max_correction_m),
        "max_tcp_to_target_offset_m": float(max_grasp_offset_m),
        "correction_model": "tcp_target_error_from_second_snapshot",
        "used_second_yaw": False,
        "used_second_absolute_target": False,
    }
    write_json(report_path, report)
    print(
        "\nSecond-snapshot TCP-target correction: delta_base_xy=[{:.4f}, {:.4f}] m norm={:.4f} m".format(
            delta[0], delta[1], correction_norm
        ),
        flush=True,
    )
