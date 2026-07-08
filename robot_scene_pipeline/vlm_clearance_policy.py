"""VLM policy for choosing among physically checked clearance candidates."""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .io_utils import image_to_base64, parse_json_or_embedded, write_json


ObjectDict = Dict[str, Any]
CandidateDict = Dict[str, Any]

DECISION_TYPES = {"pick", "place", "nudge", "pick_away", "reobserve", "none"}
SAFETY_GATE_FIELDS = (
    "collision_free",
    "moveit_feasible",
    "sweep_collision_free",
    "workspace_feasible",
    "gripper_feasible",
)


def compact_object_for_vlm(obj: ObjectDict) -> ObjectDict:
    """Return detector/depth facts useful to a VLM policy."""
    return {
        "id": obj.get("id"),
        "label": obj.get("label"),
        "confidence": obj.get("confidence"),
        "bbox_xyxy_px": obj.get("bbox_xyxy_px"),
        "center_px": obj.get("center_px"),
        "geometry_center_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
        "dimensions_m": obj.get("dimensions_m"),
        "top_z_base_m": obj.get("top_z_base_m"),
        "role": obj.get("role"),
        "state": obj.get("state"),
        "pushable": obj.get("pushable"),
    }


def build_objects_for_vlm(current_state: dict) -> dict:
    return {
        "schema_version": "objects_for_vlm_v1",
        "base_frame": current_state.get("base_frame", "base_link"),
        "coordinate_convention": current_state.get("coordinate_convention"),
        "objects": [
            compact_object_for_vlm(obj)
            for obj in current_state.get("objects", [])
            if isinstance(obj, dict)
        ],
    }


def build_task_state_for_vlm(
    current_state: dict,
    held_object: ObjectDict,
    analysis: dict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: Optional[dict],
    step_index: int,
) -> dict:
    return {
        "schema_version": "task_state_for_vlm_v1",
        "instruction": current_state.get("instruction"),
        "clearance_step_index": int(step_index),
        "target_object": compact_object_for_vlm(held_object),
        "target_grasp_state": {
            "action": analysis.get("action"),
            "grasp_feasible": bool(analysis.get("grasp_feasible")),
            "all_grasps_blocked": bool(analysis.get("all_grasps_blocked")),
            "selected_grasp_yaw_deg": analysis.get("selected_grasp_yaw_deg"),
            "blocking_objects": analysis.get("blocking_objects", []),
            "blocking_objects_by_interval": analysis.get("blocking_objects_by_interval", []),
        },
        "base_object_id": base_id,
        "protected_object_ids": [value for value in protected_ids or []],
        "stack_memory": {
            "base": ((memory or {}).get("structure") or {}).get("base"),
            "placed_order": ((memory or {}).get("structure") or {}).get("placed_order", []),
            "current_top": ((memory or {}).get("structure") or {}).get("current_top"),
        },
        "policy_role_split": {
            "llm_vlm": "task_rule_policy",
            "code": "perception_candidate_generation_collision_safety_gate_execution",
            "moveit": "trajectory_planning",
        },
    }


def materialize_vlm_images(cycle_dir: str, current_state: dict) -> Tuple[Optional[str], Optional[str]]:
    """Copy scene and overlay images into stable per-cycle VLM file names."""
    scene_path = _resolve_image_path(
        current_state.get("snapshot_image"),
        cycle_dir,
        fallback_names=("snapshot.jpg", "scene_rgb.png"),
    )
    overlay_path = _resolve_image_path(
        current_state.get("annotated_image"),
        cycle_dir,
        fallback_names=("annotated_detector.jpg", "minimal_overlay.png"),
    )
    scene_output = _copy_image_if_available(scene_path, os.path.join(cycle_dir, "scene_rgb.png"))
    overlay_output = _copy_image_if_available(overlay_path, os.path.join(cycle_dir, "minimal_overlay.png"))
    return scene_output, overlay_output


def build_physical_clearance_candidates(candidates: Iterable[CandidateDict]) -> dict:
    physical = [
        physical_candidate_from_clearance_candidate(candidate)
        for candidate in candidates or []
        if isinstance(candidate, dict) and candidate.get("candidate_id") is not None
    ]
    physical.sort(key=lambda item: str(item.get("candidate_id")))
    return {
        "schema_version": "physical_clearance_candidates_v1",
        "selection_contract": (
            "These are physical action candidates only. Final action selection is made by the VLM policy."
        ),
        "candidates": physical,
    }


def physical_candidate_from_clearance_candidate(candidate: CandidateDict) -> CandidateDict:
    action_type = candidate.get("action_type") or candidate.get("action")
    obstacle_id = candidate.get("obstacle_id")
    object_id = candidate.get("object_id")
    if object_id is None:
        object_id = candidate.get("target_object_id") if action_type in ("pick", "place") else obstacle_id
    geometry_ok = _bool(candidate.get("geometry_feasible", candidate.get("feasible", False)))
    push_end_safe = _bool(candidate.get("push_end_safe", geometry_ok))
    approach_safe = _bool(candidate.get("approach_path_safe", geometry_ok))
    swept_safe = _bool(candidate.get("push_swept_safe", geometry_ok))
    protected_safe = _bool(candidate.get("protected_structure_safe", True))
    workspace_ok = _bool(candidate.get("workspace_feasible", geometry_ok and push_end_safe))
    gripper_ok = _bool(
        candidate.get(
            "gripper_feasible",
            action_type != "pick_away" or not candidate.get("relaxed_pick_away_grasp"),
        )
    )
    moveit_ok = _bool(candidate.get("moveit_feasible", False))
    collision_free = _bool(candidate.get("collision_free", push_end_safe and protected_safe))
    return {
        "candidate_id": candidate.get("candidate_id"),
        "action_type": action_type,
        "object_id": object_id,
        "obstacle_object_id": obstacle_id,
        "target_object_id": candidate.get("target_object_id"),
        "direction_base": candidate.get("direction_base"),
        "distance_m": candidate.get("distance_m"),
        "collision_free": collision_free,
        "moveit_feasible": moveit_ok,
        "sweep_collision_free": bool(approach_safe and swept_safe),
        "workspace_feasible": workspace_ok,
        "gripper_feasible": gripper_ok,
        "brief_physical_description": _brief_physical_description(candidate),
        "physical_facts": _physical_facts(candidate),
    }


def build_policy_input(
    scene_rgb_path: Optional[str],
    minimal_overlay_path: Optional[str],
    objects_for_vlm: dict,
    task_state_for_vlm: dict,
    physical_clearance_candidates: dict,
) -> dict:
    return {
        "schema_version": "vlm_clearance_policy_input_v1",
        "scene_rgb": scene_rgb_path,
        "minimal_overlay": minimal_overlay_path,
        "objects_for_vlm": objects_for_vlm,
        "task_state_for_vlm": task_state_for_vlm,
        "physical_clearance_candidates": physical_clearance_candidates,
        "output_schema": {
            "selected_candidate_id": "string|null",
            "decision_type": "pick|place|nudge|pick_away|reobserve|none",
            "object_id": "int|null",
            "target_object_id": "int|null",
            "reason": "string",
            "confidence": "number between 0.0 and 1.0",
            "need_reobserve_after_action": "boolean",
        },
    }


def call_vlm_clearance_policy(args: Any, policy_input: dict) -> dict:
    base_report = {
        "schema_version": "vlm_clearance_policy_output_v1",
        "selection_status": "fail_safe",
        "selection_source": "vlm_clearance_policy",
    }
    candidate_ids = _candidate_ids(policy_input)
    if not candidate_ids:
        return {
            **base_report,
            "decision_type": "reobserve",
            "selected_candidate_id": None,
            "reason": "no_physical_clearance_candidates",
            "confidence": 0.0,
            "need_reobserve_after_action": True,
        }
    prompt = build_vlm_clearance_prompt(policy_input)
    message = {"role": "user", "content": prompt}
    images = _image_payloads(policy_input, include_images=not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    payload = {
        "model": getattr(args, "model", "qwen2.5vl:7b-q4_K_M"),
        "messages": [message],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": min(512, int(getattr(args, "num_predict", 512)))},
    }
    try:
        response = requests.post(
            getattr(args, "ollama_url", "http://127.0.0.1:11434/api/chat"),
            json=payload,
            timeout=float(getattr(args, "timeout", 600)),
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        decision = normalize_vlm_decision(parse_json_or_embedded(content))
    except Exception as exc:
        return {
            **base_report,
            "decision_type": "reobserve",
            "selected_candidate_id": None,
            "reason": "vlm_json_or_call_failed: {}".format(exc),
            "confidence": 0.0,
            "need_reobserve_after_action": True,
        }
    if decision.get("decision_type") in ("none", "reobserve"):
        return {**base_report, **decision, "selection_status": "no_action_selected"}
    selected_id = decision.get("selected_candidate_id")
    if selected_id is None or str(selected_id) not in candidate_ids:
        return {
            **base_report,
            **decision,
            "selection_status": "rejected_unknown_candidate",
            "reason": "vlm_selected_candidate_not_in_physical_candidates",
            "need_reobserve_after_action": True,
        }
    return {**base_report, **decision, "selection_status": "selected"}


def build_vlm_clearance_prompt(policy_input: dict) -> str:
    text_input = {
        key: value
        for key, value in policy_input.items()
        if key not in ("scene_rgb", "minimal_overlay")
    }
    return (
        "你是 UR5 桌面积木堆叠任务的 LLM/VLM 清障策略模块。\n"
        "图片是当前原始场景，可选 overlay 只用于核对物体 id 和位置。"
        "结构化 JSON 是硬约束，候选动作已经由代码生成；你只能做任务规则决策。\n\n"
        "任务规则：\n"
        "- 当前目标可抓则优先抓当前目标。\n"
        "- 当前目标不可抓则选择最合理的清障对象。\n"
        "- 不移动已经完成堆叠结构。\n"
        "- 不移动底座。\n"
        "- 不破坏后续任务所需物体。\n"
        "- 优先最小扰动。\n"
        "- 优先释放当前目标抓取空间。\n"
        "- nudge 与 pick_away 由你根据场景选择。\n"
        "- 若没有合理动作，输出 none 或 reobserve。\n"
        "- 只能选择 physical_clearance_candidates.candidates 中的 candidate_id。\n"
        "- 不允许输出任意坐标、关节角、速度、ROS topic 或新动作。\n\n"
        "只输出严格 JSON，格式必须是：\n"
        "{\n"
        '  "selected_candidate_id": "...",\n'
        '  "decision_type": "pick|place|nudge|pick_away|reobserve|none",\n'
        '  "object_id": 0,\n'
        '  "target_object_id": 0,\n'
        '  "reason": "...",\n'
        '  "confidence": 0.0,\n'
        '  "need_reobserve_after_action": true\n'
        "}\n\n"
        "输入：\n"
        + json.dumps(text_input, ensure_ascii=False, indent=2)
    )


def normalize_vlm_decision(decision: dict) -> dict:
    if not isinstance(decision, dict):
        raise ValueError("VLM output must be a JSON object.")
    output = {
        "selected_candidate_id": decision.get("selected_candidate_id"),
        "decision_type": str(decision.get("decision_type") or "none"),
        "object_id": decision.get("object_id"),
        "target_object_id": decision.get("target_object_id"),
        "reason": str(decision.get("reason") or ""),
        "confidence": _clamp_float(decision.get("confidence"), 0.0, 1.0),
        "need_reobserve_after_action": bool(decision.get("need_reobserve_after_action", True)),
        "raw_decision": decision,
    }
    if output["decision_type"] not in DECISION_TYPES:
        raise ValueError("Unsupported decision_type: {}".format(output["decision_type"]))
    return output


def evaluate_final_safety_gate(policy_output: dict, physical_candidates: dict) -> dict:
    candidate_map = {
        str(item.get("candidate_id")): item
        for item in physical_candidates.get("candidates", [])
        if item.get("candidate_id") is not None
    }
    selected_id = policy_output.get("selected_candidate_id")
    if policy_output.get("decision_type") in ("none", "reobserve"):
        return {
            "schema_version": "final_safety_gate_v1",
            "accepted": False,
            "rejected_by_safety_gate": False,
            "selected_candidate_id": selected_id,
            "decision_type": policy_output.get("decision_type"),
            "reason": "policy_requested_{}".format(policy_output.get("decision_type")),
            "failed_fields": [],
        }
    candidate = candidate_map.get(str(selected_id)) if selected_id is not None else None
    if candidate is None:
        return {
            "schema_version": "final_safety_gate_v1",
            "accepted": False,
            "rejected_by_safety_gate": True,
            "selected_candidate_id": selected_id,
            "decision_type": policy_output.get("decision_type"),
            "reason": "selected_candidate_id_not_found",
            "failed_fields": ["selected_candidate_id"],
        }
    failed_fields = [field for field in SAFETY_GATE_FIELDS if not bool(candidate.get(field))]
    return {
        "schema_version": "final_safety_gate_v1",
        "accepted": not failed_fields,
        "rejected_by_safety_gate": bool(failed_fields),
        "selected_candidate_id": selected_id,
        "decision_type": policy_output.get("decision_type"),
        "candidate": candidate,
        "failed_fields": failed_fields,
        "reason": "accepted" if not failed_fields else "safety_fields_failed",
    }


def write_vlm_policy_artifacts(
    cycle_dir: str,
    objects_for_vlm: dict,
    task_state_for_vlm: dict,
    physical_candidates: dict,
    policy_input: dict,
    policy_output: dict,
    final_safety_gate: dict,
) -> None:
    write_json(os.path.join(cycle_dir, "objects_for_vlm.json"), objects_for_vlm)
    write_json(os.path.join(cycle_dir, "task_state_for_vlm.json"), task_state_for_vlm)
    write_json(os.path.join(cycle_dir, "physical_clearance_candidates.json"), physical_candidates)
    write_json(os.path.join(cycle_dir, "vlm_clearance_policy_input.json"), policy_input)
    write_json(os.path.join(cycle_dir, "vlm_clearance_policy_output.json"), policy_output)
    write_json(os.path.join(cycle_dir, "final_safety_gate.json"), final_safety_gate)


def _physical_facts(candidate: CandidateDict) -> dict:
    facts = {
        "frontier_depth": candidate.get("frontier_depth"),
        "blocks": candidate.get("blocks", []),
        "safe_place_center_m": candidate.get("safe_place_center_m"),
        "selected_grasp_yaw_deg": candidate.get("selected_grasp_yaw_deg"),
        "push_end_safe": candidate.get("push_end_safe"),
        "approach_path_safe": candidate.get("approach_path_safe"),
        "push_swept_safe": candidate.get("push_swept_safe"),
        "protected_structure_safe": candidate.get("protected_structure_safe"),
        "future_task_feasible": candidate.get("future_task_feasible"),
        "distance_reference_object_id": candidate.get("distance_reference_object_id"),
        "distance_reference_is_current_target": candidate.get("distance_reference_is_current_target"),
        "target_distance_before_m": candidate.get("target_distance_before_m"),
        "target_distance_after_m": candidate.get("target_distance_after_m"),
        "target_distance_delta_m": candidate.get("target_distance_delta_m"),
        "contact_z_offset_m": candidate.get("contact_z_offset_m"),
        "push_execution_plan_path": candidate.get("push_execution_plan_path"),
        "moveit_preflight_error": candidate.get("moveit_preflight_error"),
    }
    push_eval = candidate.get("push_evaluation") or {}
    if isinstance(push_eval, dict):
        tool_swept = push_eval.get("tool_swept_volume") or {}
        facts["tool_swept_reason"] = tool_swept.get("reason")
        facts["push_end_collisions"] = push_eval.get("push_end_collisions", [])
        facts["tool_swept_collisions"] = tool_swept.get("collisions", [])
        facts["required_distance"] = push_eval.get("required_distance")
    return {key: value for key, value in facts.items() if value is not None}


def _brief_physical_description(candidate: CandidateDict) -> str:
    action_type = candidate.get("action_type") or candidate.get("action")
    obstacle_id = candidate.get("obstacle_id")
    target_id = candidate.get("target_object_id")
    if action_type == "nudge":
        return "Nudge obstacle {} away from target {} by {} m along base direction {}.".format(
            obstacle_id,
            target_id,
            candidate.get("distance_m"),
            candidate.get("direction_base"),
        )
    if action_type == "pick_away":
        return "Pick obstacle {} away from target {} and place it at safe_place_center_m {}.".format(
            obstacle_id,
            target_id,
            candidate.get("safe_place_center_m"),
        )
    return "{} object {} for target {}.".format(action_type, candidate.get("object_id"), target_id)


def _candidate_ids(policy_input: dict) -> set:
    return {
        str(candidate.get("candidate_id"))
        for candidate in (
            policy_input.get("physical_clearance_candidates", {}).get("candidates", [])
        )
        if candidate.get("candidate_id") is not None
    }


def _image_payloads(policy_input: dict, include_images: bool) -> List[str]:
    if not include_images:
        return []
    images = []
    for key in ("scene_rgb", "minimal_overlay"):
        path = policy_input.get(key)
        if path and os.path.isfile(path):
            images.append(image_to_base64(path))
    return images


def _copy_image_if_available(source_path: Optional[str], output_path: str) -> Optional[str]:
    if source_path is None or not os.path.isfile(source_path):
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.abspath(source_path) != os.path.abspath(output_path):
        shutil.copyfile(source_path, output_path)
    return output_path


def _resolve_image_path(path: Any, cycle_dir: str, fallback_names: Tuple[str, ...]) -> Optional[str]:
    if isinstance(path, str) and os.path.isfile(path):
        return path
    search_dirs = [
        cycle_dir,
        os.path.dirname(cycle_dir),
        os.path.join(os.path.dirname(cycle_dir), "initial_order"),
    ]
    if isinstance(path, str) and path:
        search_dirs.append(os.path.join(os.path.dirname(cycle_dir), os.path.dirname(path).split(os.sep)[-1]))
        fallback_names = (os.path.basename(path),) + tuple(fallback_names)
    for directory in search_dirs:
        for name in fallback_names:
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
    return None


def _bool(value: Any) -> bool:
    return bool(value)


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(low)
    return max(float(low), min(float(high), numeric))
