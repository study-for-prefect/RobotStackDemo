"""LLM ranking for already-safe push candidates."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import requests


CandidateDict = Dict[str, Any]


def _compact_candidate(candidate: CandidateDict) -> CandidateDict:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action": candidate.get("action") or candidate.get("action_type"),
        "action_type": candidate.get("action_type"),
        "obstacle_id": candidate.get("obstacle_id"),
        "direction_base": candidate.get("direction_base"),
        "distance_m": candidate.get("distance_m"),
        "selected_grasp_yaw_deg": candidate.get("selected_grasp_yaw_deg"),
        "safe_place_center_m": candidate.get("safe_place_center_m"),
        "frontier_depth": candidate.get("frontier_depth"),
        "utility_score": candidate.get("utility_score"),
        "easiness_score": candidate.get("easiness_score"),
        "risk_score": candidate.get("risk_score"),
        "score": candidate.get("score"),
        "reason": candidate.get("reason"),
        "source": candidate.get("source"),
        "post_push_grasp_feasible": candidate.get("post_push_grasp_feasible"),
        "current_grasp_gain": candidate.get("current_grasp_gain"),
        "blocker_count_reduction": candidate.get("blocker_count_reduction"),
        "future_blocking_cost": candidate.get("future_blocking_cost"),
        "place_blocking_cost": candidate.get("place_blocking_cost"),
    }


def build_selection_request(
    instruction: str,
    target: Dict[str, Any],
    joint_evaluation: Dict[str, Any],
    step_index: int,
    history: Optional[Iterable[dict]] = None,
) -> Dict[str, Any]:
    feasible = [
        _compact_candidate(candidate)
        for candidate in joint_evaluation.get("candidates", [])
        if candidate.get("feasible") and candidate.get("candidate_id") is not None
    ]
    return {
        "schema_version": "llm_safe_push_selection_request_v1",
        "instruction": instruction,
        "clearance_step_index": int(step_index),
        "target_object": {
            "id": target.get("id"),
            "label": target.get("label"),
            "geometry_center_m": target.get("geometry_center_m"),
            "dimensions_m": target.get("dimensions_m"),
            "reacquire_source": target.get("reacquire_source"),
        },
        "safe_candidates": feasible,
        "recent_clearance_history": list(history or [])[-6:],
        "selection_rule": (
            "Choose exactly one candidate_id from safe_candidates. Prefer actions that make "
            "the current target graspable soon, avoid future stack placement, and use shorter "
            "progress pushes when full clearance is not yet possible."
        ),
    }


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM selection response must be a JSON object.")
    return payload


def _fallback(report: Dict[str, Any], reason: str) -> Dict[str, Any]:
    output = dict(report)
    output.update(
        {
            "selection_status": "fallback_geometry_score",
            "selection_source": "geometry_score_fallback",
            "reason": reason,
            "selected_candidate_id": None,
        }
    )
    return output


def select_safe_push_candidate(
    args: Any,
    instruction: str,
    target: Dict[str, Any],
    joint_evaluation: Dict[str, Any],
    step_index: int,
    history: Optional[Iterable[dict]] = None,
) -> Dict[str, Any]:
    request = build_selection_request(
        instruction,
        target,
        joint_evaluation,
        step_index,
        history=history,
    )
    safe_ids = {
        str(candidate["candidate_id"])
        for candidate in request["safe_candidates"]
        if candidate.get("candidate_id") is not None
    }
    base_report = {
        "schema_version": "llm_safe_push_selection_result_v1",
        "request": request,
    }
    if not safe_ids:
        return _fallback(base_report, "no_safe_candidates")
    if not bool(getattr(args, "enable_llm_push_selection", True)):
        return _fallback(base_report, "llm_push_selection_disabled")

    prompt = (
        "你是机器人积木清障动作选择器。输入中的 safe_candidates 已通过几何、碰撞、"
        "桌面边界和未来任务安全检查。你只能从 safe_candidates 里选择一个 candidate_id，"
        "不能创造新动作。只输出严格 JSON: "
        "{\"selected_candidate_id\":\"...\",\"reason\":\"...\"}\n\n输入:\n"
        + json.dumps(request, ensure_ascii=False, indent=2)
    )
    payload = {
        "model": getattr(args, "model", "qwen2.5vl:7b-q4_K_M"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": min(256, int(getattr(args, "num_predict", 256)))},
    }
    try:
        response = requests.post(
            getattr(args, "ollama_url", "http://127.0.0.1:11434/api/chat"),
            json=payload,
            timeout=float(getattr(args, "timeout", 600)),
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        decision = _parse_json(content)
    except Exception as exc:
        return _fallback(base_report, "llm_selection_error: {}".format(exc))

    selected_id = decision.get("selected_candidate_id")
    if selected_id is None:
        selected_id = decision.get("candidate_id")
    if str(selected_id) not in safe_ids:
        report = _fallback(base_report, "llm_selected_unknown_or_unsafe_candidate")
        report["raw_decision"] = decision
        return report
    output = dict(base_report)
    output.update(
        {
            "selection_status": "selected",
            "selection_source": "llm_safe_candidate_selector",
            "selected_candidate_id": selected_id,
            "reason": decision.get("reason") or "selected_by_llm",
            "raw_decision": decision,
        }
    )
    return output
