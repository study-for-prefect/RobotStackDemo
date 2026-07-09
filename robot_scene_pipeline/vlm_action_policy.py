"""VLM action-intent policy and safety validation for stack clearing."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import requests

from .io_utils import image_to_base64, parse_json_or_embedded
from .vlm_action_validation import mark_moveit_result, validate_vlm_action_decision


ObjectDict = Dict[str, Any]
ActionDict = Dict[str, Any]

ACTION_TYPES = {"pick", "nudge", "pick_away", "reobserve", "stop"}


def compact_object_for_action_policy(obj: ObjectDict) -> ObjectDict:
    """Return scene facts for VLM action intent without camera intrinsics."""
    return {
        "id": obj.get("id"),
        "label": obj.get("label"),
        "confidence": obj.get("confidence"),
        "bbox_xyxy_px": obj.get("bbox_xyxy_px"),
        "center_px": obj.get("center_px"),
        "geometry_center_base_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
        "dimensions_m": obj.get("dimensions_m"),
        "top_z_base_m": obj.get("top_z_base_m"),
        "role": obj.get("role"),
        "state": obj.get("state"),
        "pushable": obj.get("pushable"),
        "visible": obj.get("visible", True),
    }


def build_vlm_action_decision_input(
    scene_rgb_path: Optional[str],
    minimal_overlay_path: Optional[str],
    current_state: dict,
    target_object: ObjectDict,
    analysis: dict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: Optional[dict],
    step_index: int,
) -> dict:
    """Build the high-level action-intent prompt payload."""
    structure = (memory or {}).get("structure") or {}
    return {
        "schema_version": "vlm_action_decision_input_v1",
        "scene_rgb": scene_rgb_path,
        "minimal_overlay": minimal_overlay_path,
        "base_frame": current_state.get("base_frame", "base_link"),
        "instruction": current_state.get("instruction"),
        "clearance_step_index": int(step_index),
        "objects": [
            compact_object_for_action_policy(obj)
            for obj in current_state.get("objects", [])
            if isinstance(obj, dict)
        ],
        "target_object": compact_object_for_action_policy(target_object),
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
            "base": structure.get("base"),
            "placed_order": structure.get("placed_order", []),
            "current_top": structure.get("current_top"),
        },
        "policy_role_split": {
            "vlm": "high_level_action_intent_only",
            "code": "physical_feasibility_safety_validation_and_execution",
            "moveit": "ik_collision_and_trajectory_preflight",
        },
        "output_schema": {
            "action_type": "pick|nudge|pick_away|reobserve|stop",
            "object_id": "int|string|null",
            "target_object_id": "int|string|null",
            "push_direction_base": "[x,y,z] unit vector in base_link, required for nudge",
            "push_distance_m": "0.01..0.05, required for nudge",
            "safe_place_center_base_m": "[x,y,z] in base_link, required for pick_away",
            "reason": "string",
            "confidence": "0.0..1.0",
        },
    }


def call_vlm_action_policy(args: Any, policy_input: dict) -> dict:
    """Call the VLM and return raw plus parsed action decision."""
    prompt = build_vlm_action_prompt(policy_input)
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
        raw_content = response.json().get("message", {}).get("content", "")
        decision = parse_vlm_action_decision_text(raw_content)
        return {
            "schema_version": "vlm_action_decision_raw_v1",
            "call_status": "parsed",
            "raw_content": raw_content,
            "decision": decision,
        }
    except Exception as exc:
        return {
            "schema_version": "vlm_action_decision_raw_v1",
            "call_status": "fail_safe_stop",
            "raw_content": locals().get("raw_content", ""),
            "error": str(exc),
            "decision": {
                "action_type": "stop",
                "object_id": None,
                "target_object_id": None,
                "push_direction_base": None,
                "push_distance_m": None,
                "safe_place_center_base_m": None,
                "reason": "vlm_json_or_call_failed: {}".format(exc),
                "confidence": 0.0,
                "raw_decision": None,
            },
        }


def build_vlm_action_prompt(policy_input: dict) -> str:
    text_input = {key: value for key, value in policy_input.items() if key not in ("scene_rgb", "minimal_overlay")}
    return (
        "你是 UR5 桌面积木堆叠任务的 VLM 高层动作意图决策模块。\n"
        "图片只用于核对带编号物体；结构化输入中的 base_link 坐标、尺寸、bbox 和任务状态是决策依据。\n"
        "不要输出关节角、轨迹、速度、ROS 控制命令或相机内参。\n\n"
        "你必须自己决定动作意图：\n"
        "- pick: 当前任务目标已经适合抓取。\n"
        "- nudge: 推开某个松散物体；你必须给出 base_link 下的单位方向向量和 0.01 到 0.05 m 的距离。\n"
        "- pick_away: 抓走某个松散物体，并给出 base_link 下的安全临时放置中心 safe_place_center_base_m。\n"
        "- reobserve: 视觉信息不足，需要重新观察。\n"
        "- stop: 没有安全/合理动作。\n\n"
        "约束：不移动 base、placed、locked、protected 物体；优先最小扰动和保护已堆叠结构。\n"
        "代码会独立验证物理可行性、安全和 MoveIt，不安全会停止。\n\n"
        "只输出严格 JSON：\n"
        "{\n"
        '  "action_type": "pick|nudge|pick_away|reobserve|stop",\n'
        '  "object_id": 0,\n'
        '  "target_object_id": 0,\n'
        '  "push_direction_base": [1.0, 0.0, 0.0],\n'
        '  "push_distance_m": 0.025,\n'
        '  "safe_place_center_base_m": [0.0, 0.0, 0.0],\n'
        '  "reason": "...",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "输入：\n"
        + json.dumps(text_input, ensure_ascii=False, indent=2)
    )


def parse_vlm_action_decision_text(text: str) -> ActionDict:
    decision = parse_json_or_embedded(text)
    if not isinstance(decision, dict):
        raise ValueError("VLM action output must be a JSON object.")
    action_type = str(decision.get("action_type") or "").strip()
    if action_type not in ACTION_TYPES:
        raise ValueError("Unsupported action_type: {}".format(action_type))
    return {
        "action_type": action_type,
        "object_id": decision.get("object_id"),
        "target_object_id": decision.get("target_object_id"),
        "push_direction_base": decision.get("push_direction_base"),
        "push_distance_m": decision.get("push_distance_m"),
        "safe_place_center_base_m": decision.get("safe_place_center_base_m"),
        "reason": str(decision.get("reason") or ""),
        "confidence": _clamp_float(decision.get("confidence"), 0.0, 1.0),
        "raw_decision": decision,
    }


def _image_payloads(policy_input: dict, include_images: bool) -> List[str]:
    if not include_images:
        return []
    images = []
    for key in ("scene_rgb", "minimal_overlay"):
        path = policy_input.get(key)
        if path:
            images.append(image_to_base64(path))
    return images


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(low)
    return max(float(low), min(float(high), numeric))
