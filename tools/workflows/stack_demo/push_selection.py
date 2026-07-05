"""Safe push candidate selection helpers for stack-demo clearing."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from robot_scene_pipeline.llm_safe_action_selector import select_safe_push_candidate
from tools.planning.decision_to_execution import write_json

from .push_clearing import select_result_by_candidate_id


def push_history(memory: dict) -> list:
    return [
        action for action in memory.get("action_history", [])
        if action.get("action") == "push"
    ]


def choose_selected_result(
    args: Any,
    cycle_dir: str,
    held_object: dict,
    direction_assessment: dict,
    memory: dict,
    step_index: int,
) -> Tuple[Optional[dict], dict]:
    joint = direction_assessment.get("joint_evaluation", {})
    llm_report = select_safe_push_candidate(
        args,
        getattr(args, "instruction", ""),
        held_object,
        joint,
        step_index,
        history=push_history(memory),
    )
    write_json(
        os.path.join(cycle_dir, "clearance_step_{:02d}_llm_selection.json".format(step_index)),
        llm_report,
    )
    selected_result = None
    if llm_report.get("selection_status") == "selected":
        selected_result = select_result_by_candidate_id(
            direction_assessment,
            llm_report.get("selected_candidate_id"),
        )
    if selected_result is None:
        selected_result = direction_assessment.get("selected_result")
        llm_report["selection_source"] = "geometry_score_fallback"
    return selected_result, llm_report


def selected_push_from_result(selected_result: Optional[dict]) -> Optional[Dict[str, Any]]:
    if selected_result is None:
        return None
    selected_direction = selected_result["selected_direction"]
    selected_push = dict(selected_result["relation"])
    selected_push["candidate_id"] = selected_direction.get("candidate_id")
    selected_push["direction_base"] = selected_direction["direction_base"]
    selected_push["distance_m"] = selected_direction["distance_m"]
    selected_push["direction_source"] = selected_direction["source"]
    selected_push["direction_score"] = selected_direction["score"]
    selected_push["reason"] = selected_direction.get("reason") or selected_push.get("reason")
    return selected_push


def choose_clearance_action(
    args: Any,
    cycle_dir: str,
    held_object: dict,
    safe_candidates: list,
    memory: dict,
    step_index: int,
) -> Tuple[Optional[dict], dict]:
    llm_report = select_safe_push_candidate(
        args,
        getattr(args, "instruction", ""),
        held_object,
        {"candidates": safe_candidates},
        step_index,
        history=push_history(memory),
    )
    write_json(
        os.path.join(cycle_dir, "clearance_step_{:02d}_llm_selection.json".format(step_index)),
        llm_report,
    )
    selected = None
    if llm_report.get("selection_status") == "selected":
        selected_id = str(llm_report.get("selected_candidate_id"))
        selected = next(
            (candidate for candidate in safe_candidates if str(candidate.get("candidate_id")) == selected_id),
            None,
        )
    if selected is None and safe_candidates:
        selected = safe_candidates[0]
        llm_report["selection_source"] = "geometry_score_fallback"
    return selected, llm_report
