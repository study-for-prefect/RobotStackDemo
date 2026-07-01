"""Post-place scoped observation and stack-growth verification."""

import os
from typing import Any, Dict, Iterable, Tuple

from robot_scene_pipeline.stack_state import verify_stack_growth
from tools.planning.decision_to_execution import write_json

from .observation_scope import expected_placed_template, observe_empty_with_scope
from .scene import estimate_current_stack


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
        critical_templates=[expected_placed_template(held_object, place_step)] + list(future_templates),
        noncritical_templates=[current_base_object],
        scope_name="after_place",
        description="Post-place scoped observation",
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
    return current_state, memory, post_place_stack
