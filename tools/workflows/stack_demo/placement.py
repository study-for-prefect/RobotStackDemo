"""Stack placement geometry and safety validation."""

import copy
import math
import os

from robot_scene_pipeline.grasp_yaw_search import equivalent_yaw_delta_deg
from robot_scene_pipeline.xy_correction import apply_step_xyz_correction, load_xy_correction
from tools.planning.decision_to_execution import compile_place_on_top, write_json

from .scene import (
    object_by_id,
    reacquire_target,
    require_geometry_object,
    selected_stack_yaw,
)

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
        place_step["exact_tool_yaw_required"] = True
        place_step["yaw_equivalence_period_deg"] = 180.0
        place_step["preserve_current_yaw"] = False
    correction_path = getattr(args, "calibration_json", "") or getattr(args, "xy_correction_json", "")
    correction = load_xy_correction(correction_path)
    apply_step_xyz_correction(place_step, place_step["target_position_m"], correction, kind="place")
    return place_step


def validate_pick_place_separation(pick_step, place_step, min_distance_m):
    pick_xy = [float(value) for value in pick_step["target_position_m"][:2]]
    place_reference = place_step.get("tcp_place_xy_base_m") or place_step.get("target_position_m")
    place_xy = [float(value) for value in place_reference[:2]]
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


def _placement_base_stack_object(stack_state):
    base_id = stack_state.get("placement_base_object_id")
    for obj in stack_state.get("stack_objects") or []:
        if base_id is not None and str(obj.get("id")) == str(base_id):
            return obj
    objects = stack_state.get("stack_objects") or []
    return objects[-1] if objects else {}


def _validate_same_base_identity(first_stack_state, second_stack_state):
    first_id = first_stack_state.get("placement_base_object_id")
    second_id = second_stack_state.get("placement_base_object_id")
    if first_id is not None and second_id is not None and str(first_id) != str(second_id):
        raise RuntimeError(
            "Refusing second base correction: observed base id {} does not match locked base id {}.".format(
                second_id,
                first_id,
            )
        )

    first_obj = _placement_base_stack_object(first_stack_state)
    second_obj = _placement_base_stack_object(second_stack_state)
    first_label = first_obj.get("label")
    second_label = second_obj.get("label")
    if first_label and second_label and str(first_label).lower() != str(second_label).lower():
        raise RuntimeError(
            "Refusing second base correction: observed base label '{}' does not match locked label '{}'.".format(
                second_label,
                first_label,
            )
        )

    first_dims = first_obj.get("dimensions_m")
    second_dims = second_obj.get("dimensions_m")
    if first_dims is None or second_dims is None:
        return
    deltas = [
        abs(float(second_dims[index]) - float(first_dims[index]))
        for index in range(min(3, len(first_dims), len(second_dims)))
    ]
    if deltas and max(deltas) > 0.012:
        raise RuntimeError(
            "Refusing second base correction: observed base dimensions changed by {} m.".format(
                [round(value, 5) for value in deltas]
            )
        )


def validate_place_second_snapshot(first_stack_state, second_stack_state, max_correction_m, top_z_tolerance_m=None):
    first_xy = [float(value) for value in first_stack_state["placement_base_center_xy_m"][:2]]
    second_xy = [float(value) for value in second_stack_state["placement_base_center_xy_m"][:2]]
    delta = [second_xy[0] - first_xy[0], second_xy[1] - first_xy[1]]
    distance = math.hypot(delta[0], delta[1])
    first_top_z = float(first_stack_state["placement_base_top_z_m"])
    second_top_z = float(second_stack_state["placement_base_top_z_m"])
    _validate_same_base_identity(first_stack_state, second_stack_state)
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
    first_yaw = first_stack_state.get("stack_yaw_deg")
    second_yaw = second_stack_state.get("stack_yaw_deg")
    if first_yaw is not None and second_yaw is not None:
        yaw_delta = equivalent_yaw_delta_deg(float(second_yaw), float(first_yaw))
        if yaw_delta > 10.0:
            raise RuntimeError(
                "Refusing second base correction: yaw delta {:.2f} deg exceeds 10.00 deg.".format(yaw_delta)
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
