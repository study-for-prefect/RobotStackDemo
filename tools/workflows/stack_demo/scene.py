"""Scene lookup, target reacquisition, memory matching, and stack estimation."""

import copy
import math
import os
import time

from robot_scene_pipeline.llm_scene_reasoner import (
    StackColorSelectionError,
    build_stack_blocks_prompt,
    call_ollama,
    normalize_stack_blocks_decision,
    rule_stack_blocks_decision,
    stack_color_requirement_report,
    validate_stack_blocks_decision,
)
from tools.planning.decision_to_execution import write_json
from robot_scene_pipeline.stack_state import estimate_stack_state
from tools.planning.build_geometry_pick_plan import find_object, set_stack_demo_yaw

from .commands import capture_scene_observation, load_json, relative_translate_command, run


def _parse_base_offsets(raw_value):
    offsets = []
    for item in str(raw_value or "").split(";"):
        if not item.strip():
            continue
        values = [float(value.strip()) for value in item.split(",")]
        if len(values) != 3:
            raise RuntimeError("Initial observation recovery offsets must contain XYZ triples.")
        offsets.append(values)
    return offsets


def _move_to_initial_recovery_offset(args, attempt):
    offsets = _parse_base_offsets(getattr(args, "initial_observation_recovery_offsets_base", ""))
    if not offsets or not getattr(args, "execute", False):
        return
    offset = offsets[min(max(0, attempt - 1), len(offsets) - 1)]
    if any(abs(value) > 1e-9 for value in offset):
        print("Initial observation retry: moving by base_link offset {}.".format(offset), flush=True)
        run(relative_translate_command(args, offset))


def _initial_decision_valid(args, state):
    if args.stack_decision_json or args.force_llm_decision:
        return True, None
    decision = rule_stack_blocks_decision(args.instruction, state.get("objects", []))
    if decision is None:
        objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
        if len(objects) >= 2:
            return True, "structure planner will choose roles"
        return False, "not enough detected blocks for structure planning"
    try:
        validate_stack_blocks_decision(
            decision,
            state.get("objects", []),
            args.instruction,
            prefer_explicit_rule=True,
        )
    except (RuntimeError, ValueError) as exc:
        return False, str(exc)
    return True, None


def _write_initial_required_objects_report(args, output_dir, state, decision_error=None):
    report = stack_color_requirement_report(args.instruction, state.get("objects", []))
    report["snapshot_image"] = state.get("snapshot_image")
    report["annotated_image"] = state.get("annotated_image")
    report["decision_error"] = decision_error
    write_json(os.path.join(output_dir, "initial_required_objects_report.json"), report)
    return report


def write_stack_object_selection_debug(output_dir: str, decision: dict) -> None:
    selections = decision.get("color_candidate_selections") if isinstance(decision, dict) else None
    if not selections:
        return
    write_json(
        os.path.join(output_dir, "stack_object_selection_debug.json"),
        {
            "schema_version": "stack_object_selection_debug_v1",
            "base_object_id": decision.get("base_object_id"),
            "stack_order": decision.get("stack_order"),
            "full_stack_order": decision.get("full_stack_order"),
            "selections": selections,
        },
    )


def _raise_initial_required_objects_missing(args, state, decision_error):
    report = _write_initial_required_objects_report(
        args,
        args.output_dir,
        state,
        decision_error=decision_error,
    )
    missing = report.get("missing_colors", [])
    raise RuntimeError(
        "Initial required objects are still missing after recovery observation: "
        "missing_colors={} error={}. See initial_required_objects_report.json. "
        "Object id is a per-snapshot instance id; label is the detector class name; "
        "label_id/class_id are YOLO class ids and are not unique object ids.".format(
            missing,
            decision_error,
        )
    )

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
        write_stack_object_selection_debug(args.output_dir, decision)
        return state, decision

    state = None
    decision_error = None
    initial_attempts = max(0, int(args.initial_observation_retry_count)) + 1
    for attempt in range(initial_attempts):
        initial_dir = os.path.join(args.output_dir, "initial_order")
        if attempt > 0:
            initial_dir = os.path.join(args.output_dir, "initial_order_retry_{:02d}".format(attempt))
            if float(args.initial_observation_stable_wait_s) > 0.0:
                time.sleep(float(args.initial_observation_stable_wait_s))
            _move_to_initial_recovery_offset(args, attempt)
        capture_scene_observation(args, initial_dir, stack_reasoning=False)
        state = load_json(os.path.join(initial_dir, "private_scene_state.json"))
        objects = [obj for obj in state.get("objects", []) if not obj.get("is_workspace")]
        decision_ok, decision_error = _initial_decision_valid(args, state)
        _write_initial_required_objects_report(args, initial_dir, state, decision_error=decision_error)
        if objects and decision_ok:
            break
        if attempt + 1 >= initial_attempts:
            break
        print(
            "Initial observation incomplete: objects={} decision_error={}. Retry {}/{}.".format(
                len(objects),
                decision_error,
                attempt + 1,
                initial_attempts - 1,
            ),
            flush=True,
        )
    if args.stack_decision_json:
        decision = load_json(args.stack_decision_json)
    else:
        try:
            decision = None if args.force_llm_decision else rule_stack_blocks_decision(args.instruction, state.get("objects", []))
        except StackColorSelectionError as exc:
            _raise_initial_required_objects_missing(args, state, str(exc))
        if decision is None:
            reasoning_dir = os.path.join(args.output_dir, "initial_order_llm")
            capture_scene_observation(args, reasoning_dir, stack_reasoning=False)
            state = load_json(os.path.join(reasoning_dir, "private_scene_state.json"))
            _write_initial_required_objects_report(args, reasoning_dir, state, decision_error=None)
            llm_input = load_json(os.path.join(reasoning_dir, "llm_input.json"))
            prompt = build_stack_blocks_prompt(llm_input)
            raw_result = call_ollama(args, prompt, state.get("snapshot_image", ""))
            decision = normalize_stack_blocks_decision(
                raw_result,
                hard_prior_objects=llm_input.get("hard_priors", {}).get("objects", []),
                instruction=args.instruction,
                prefer_explicit_rule=True,
            )
            write_json(os.path.join(reasoning_dir, "llm_scene_graph_decision_raw.json"), {"raw": raw_result})
            write_json(os.path.join(reasoning_dir, "llm_scene_graph_decision.json"), decision)
    write_stack_object_selection_debug(args.output_dir, decision)
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
    full_ordered = [
        object_by_id(initial_state, object_id)
        for object_id in decision.get("full_stack_order", [decision["base_object_id"]] + list(decision["stack_order"]))
    ]
    structure_plan = decision.get("structure_plan") if isinstance(decision.get("structure_plan"), dict) else {}
    print(
        "\nValidated stack decision: structure={} strategy={} base={} id={} "
        "full_stack_order={} full_ids={} place_order={} place_ids={} semantics={} source={}".format(
            structure_plan.get("structure_type", "stack"),
            structure_plan.get("execution_strategy", "vertical_stack"),
            base.get("label"),
            base.get("id"),
            [obj.get("label") for obj in full_ordered],
            [obj.get("id") for obj in full_ordered],
            [obj.get("label") for obj in ordered],
            [obj.get("id") for obj in ordered],
            decision.get("stack_order_semantics", "place_order_excludes_base"),
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
    for role in structure_plan.get("roles", []) if isinstance(structure_plan.get("roles"), list) else []:
        print(
            "Structure role: role={} object_id={} reason={}".format(
                role.get("role"),
                role.get("object_id", role.get("object_ids")),
                role.get("reason"),
            ),
            flush=True,
        )


def memory_id_for_scene_object(memory, scene_obj, max_dist_m=0.05):
    label = scene_obj.get("label") or scene_obj.get("class_name") or scene_obj.get("name")
    center = (
        scene_obj.get("geometry_center_m")
        or scene_obj.get("center_base")
        or scene_obj.get("center_base_m")
    )

    if label is None or center is None:
        return None

    best_id = None
    best_dist = float("inf")

    for obj_id, obj in memory.get("objects", {}).items():
        if obj.get("label") != label:
            continue

        old_center = obj.get("last_pose_base")
        if not old_center:
            continue

        dist = math.hypot(
            float(old_center[0]) - float(center[0]),
            float(old_center[1]) - float(center[1]),
        )

        if dist < best_dist:
            best_dist = dist
            best_id = obj_id

    if best_id is not None and best_dist <= float(max_dist_m):
        return best_id

    return None


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
