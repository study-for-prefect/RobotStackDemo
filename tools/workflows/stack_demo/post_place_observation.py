"""Post-place scoped observation and stack-growth verification."""

import os
from typing import Any, Dict, Iterable, Tuple

from robot_scene_pipeline.stack_state import verify_stack_growth
from tools.planning.decision_to_execution import write_json

from .observation_scope import expected_placed_template, observe_empty_with_scope
from .scene import estimate_current_stack


def _restore_geometry_confirmed_placed_memory(memory: dict, post_place_observation: dict) -> dict:
    if post_place_observation.get("status") != "critical_missing":
        return memory
    for obj_id in post_place_observation.get("critical_memory_ids", []):
        obj = memory.get("objects", {}).get(str(obj_id))
        if obj is None:
            continue
        obj["state"] = "placed"
        obj["visible"] = False
        obj["graspable"] = False
        obj["pushable"] = False
        obj["observation_confidence"] = max(float(obj.get("observation_confidence", 0.0)), 0.7)
        obj["post_place_confirmed_by"] = "stack_growth_geometry"
        obj.pop("missing_observation_scope", None)
        obj.pop("unconfirmed_missing_count", None)
    return memory


def _planned_height_fallback(
    held_object: dict,
    place_step: dict,
    final_stack_state: dict,
    post_place_stack: dict,
    post_place_observation: dict,
) -> dict:
    placed_template = expected_placed_template(held_object, place_step)
    placed_id = placed_template.get("id")
    observed = []
    for attempt in post_place_observation.get("attempts", []):
        observed.extend(attempt.get("observed_critical", []))
    if not any(str(item.get("template_id")) == str(placed_id) for item in observed):
        return post_place_stack
    center = placed_template.get("geometry_center_m")
    size = held_object.get("dimensions_m")
    previous_top = final_stack_state.get("top_z_base_m")
    if (
        not isinstance(center, list)
        or len(center) < 3
        or not isinstance(size, list)
        or len(size) < 3
        or previous_top is None
    ):
        return post_place_stack
    expected_top = float(center[2]) + 0.5 * float(size[2])
    if expected_top < float(previous_top) + 0.005:
        return post_place_stack
    corrected = dict(post_place_stack)
    corrected["valid"] = True
    corrected["reason"] = "Accepted planned place height because scoped observation confirmed the placed object XY."
    corrected["top_z_base_m"] = round(expected_top, 6)
    corrected["placement_base_top_z_m"] = round(expected_top, 6)
    corrected["top_z_source"] = "planned_place_height_after_noisy_depth"
    corrected["height_estimation_method"] = "planned_place_height_fallback"
    corrected["post_place_height_fallback"] = {
        "placed_object_id": placed_id,
        "expected_top_z_base_m": round(expected_top, 6),
        "observed_top_z_base_m": post_place_stack.get("top_z_base_m"),
        "previous_top_z_base_m": previous_top,
    }
    return corrected


def handle_post_place_observation(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    base_object: dict,
    current_base_object: dict,
    held_object: dict,
    place_step: dict,
    final_stack_state: dict,
    future_templates: Iterable[dict],
    cycle_index: int,
) -> Tuple[dict, dict, dict]:
    runtime["current_stage"] = "scoped_observation_after_place"
    post_place_state, memory, post_place_observation = observe_empty_with_scope(
        args,
        os.path.join(cycle_dir, "observation_after_place"),
        runtime,
        memory,
        critical_templates=[expected_placed_template(held_object, place_step)],
        noncritical_templates=[current_base_object] + list(future_templates),
        scope_name="after_place",
        description="Post-place scoped observation",
        critical_match_z_tolerance_m=getattr(args, "post_place_match_z_tolerance_m", 0.025),
        fail_on_missing_critical=False,
    )
    if post_place_state is not None:
        current_state = post_place_state
    write_json(
        os.path.join(cycle_dir, "post_place_scoped_observation.json"),
        post_place_observation,
    )
    _, post_place_stack = estimate_current_stack(
        current_state,
        base_object,
        final_stack_state.get("stack_xy_base_m"),
        args,
        search_radius_m=args.search_radius_m,
    )
    write_json(os.path.join(cycle_dir, "stack_state_after_place_observation.json"), post_place_stack)
    post_place_verification = verify_stack_growth(final_stack_state, post_place_stack)
    if not post_place_verification["valid"]:
        corrected_stack = _planned_height_fallback(
            held_object,
            place_step,
            final_stack_state,
            post_place_stack,
            post_place_observation,
        )
        if corrected_stack is not post_place_stack:
            post_place_stack = corrected_stack
            write_json(os.path.join(cycle_dir, "stack_state_after_place_height_fallback.json"), post_place_stack)
            post_place_verification = verify_stack_growth(final_stack_state, post_place_stack)
    write_json(
        os.path.join(cycle_dir, "post_place_growth_verification.json"),
        post_place_verification,
    )
    if not post_place_verification["valid"]:
        raise RuntimeError(
            "Cycle {} post-place verification failed: {}".format(
                cycle_index,
                post_place_verification["reason"],
            )
        )
    memory = _restore_geometry_confirmed_placed_memory(memory, post_place_observation)
    return current_state, memory, post_place_stack
