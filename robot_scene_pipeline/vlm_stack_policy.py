"""VLM-first stack-order decision policy."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import requests

from .io_utils import image_to_base64, parse_json_or_embedded
from .llm_stack_blocks import rule_stack_blocks_decision, validate_stack_blocks_decision


ObjectDict = Dict[str, Any]


def compact_stack_object(obj: ObjectDict) -> ObjectDict:
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


def build_vlm_stack_decision_input(state: dict, instruction: str) -> dict:
    return {
        "schema_version": "vlm_stack_decision_input_v1",
        "instruction": instruction,
        "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"),
        "base_frame": state.get("base_frame", "base_link"),
        "objects": [
            compact_stack_object(obj)
            for obj in state.get("objects", [])
            if isinstance(obj, dict)
            and not obj.get("is_workspace")
            and str(obj.get("label", "")).lower() != "workspace"
        ],
        "output_schema": {
            "base_object_id": "int",
            "stack_order": "list[int], place order excluding base_object_id",
            "full_stack_order": "list[int], starts with base_object_id",
            "structure_plan": "object",
            "reason": "string",
            "confidence": "0.0..1.0",
        },
        "policy_role_split": {
            "vlm": "structure_and_stack_order_decision",
            "code": "id_geometry_validation_and_execution",
        },
    }


def call_vlm_stack_policy(args: Any, policy_input: dict) -> dict:
    prompt = build_vlm_stack_prompt(policy_input)
    message = {"role": "user", "content": prompt}
    images = _image_payloads(policy_input, include_images=not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    payload = {
        "model": getattr(args, "model", "qwen2.5vl:7b-q4_K_M"),
        "messages": [message],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": min(1024, int(getattr(args, "num_predict", 1024)))},
    }
    try:
        response = requests.post(
            getattr(args, "ollama_url", "http://127.0.0.1:11434/api/chat"),
            json=payload,
            timeout=float(getattr(args, "timeout", 600)),
        )
        response.raise_for_status()
        raw_content = response.json().get("message", {}).get("content", "")
        decision = parse_json_or_embedded(raw_content)
        return {
            "schema_version": "vlm_stack_decision_raw_v1",
            "call_status": "parsed",
            "raw_content": raw_content,
            "decision": decision,
        }
    except Exception as exc:
        return {
            "schema_version": "vlm_stack_decision_raw_v1",
            "call_status": "fail_safe_stop",
            "raw_content": locals().get("raw_content", ""),
            "error": str(exc),
            "decision": None,
        }


def validate_vlm_stack_decision(raw_output: dict, state: dict, instruction: str) -> dict:
    if raw_output.get("call_status") != "parsed":
        raise RuntimeError("VLM stack decision failed: {}".format(raw_output.get("error")))
    decision = raw_output.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("VLM stack decision must be a JSON object.")
    base_id = int(decision.get("base_object_id"))
    stack_order = [int(value) for value in decision.get("stack_order", [])]
    full_stack_order = [int(value) for value in decision.get("full_stack_order", [])]
    if base_id in stack_order:
        raise ValueError("VLM stack_order must exclude base_object_id.")
    if not full_stack_order or full_stack_order[0] != base_id:
        raise ValueError("VLM full_stack_order must start with base_object_id.")
    if len(stack_order) != len(set(stack_order)) or len(full_stack_order) != len(set(full_stack_order)):
        raise ValueError("VLM stack decision ids must be unique.")
    validated = validate_stack_blocks_decision(
        decision,
        state.get("objects", []),
        instruction,
        prefer_explicit_rule=False,
    )
    if decision and decision.get("confidence") is not None:
        try:
            validated["confidence"] = max(0.0, min(1.0, float(decision.get("confidence"))))
        except (TypeError, ValueError):
            validated["confidence"] = 0.0
    rule_diagnostic = rule_stack_blocks_decision(instruction, state.get("objects", [])) if instruction else None
    if rule_diagnostic is not None:
        _validate_explicit_rule_coverage(validated, rule_diagnostic)
        validated["explicit_rule_diagnostic"] = rule_diagnostic
    validated["decision_source"] = "vlm_stack_policy"
    return validated


def _validate_explicit_rule_coverage(validated: dict, rule_diagnostic: dict) -> None:
    """Reject VLM stack decisions that omit explicit instruction objects."""
    expected_ids = {int(value) for value in rule_diagnostic.get("full_stack_order", [])}
    actual_ids = {int(value) for value in validated.get("full_stack_order", [])}
    if int(validated.get("base_object_id")) != int(rule_diagnostic.get("base_object_id")):
        raise ValueError("VLM base_object_id conflicts with explicit instruction target.")
    missing_ids = expected_ids - actual_ids
    if missing_ids:
        raise ValueError("VLM stack decision omits explicit instruction object ids: {}".format(sorted(missing_ids)))


def build_vlm_stack_prompt(policy_input: dict) -> str:
    text_input = {key: value for key, value in policy_input.items() if key not in ("scene_rgb", "minimal_overlay")}
    return (
        "你是 UR5 桌面积木堆叠任务的 VLM 结构决策模块。\n"
        "你必须根据用户指令、快照图和检测后的精简 JSON 决定底座和堆叠顺序。\n"
        "只能选择 objects 中已有的 object id；不要编造物体。不要输出相机内参、关节角、轨迹、速度或 ROS 命令。\n"
        "stack_order 必须是不含 base_object_id 的放置顺序，full_stack_order 必须以 base_object_id 开头。\n\n"
        "只输出严格 JSON：\n"
        "{\n"
        '  "task_type": "stack_blocks",\n'
        '  "structure_plan": {"structure_type": "tower", "roles": [], "assembly_steps": [], "limitations": []},\n'
        '  "base_object_id": 0,\n'
        '  "full_stack_order": [0, 1, 2],\n'
        '  "stack_order": [1, 2],\n'
        '  "stack_order_semantics": "place_order_excludes_base",\n'
        '  "reason": "...",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "输入：\n"
        + json.dumps(text_input, ensure_ascii=False, indent=2)
    )


def _image_payloads(policy_input: dict, include_images: bool) -> List[str]:
    if not include_images:
        return []
    images = []
    for key in ("scene_rgb", "minimal_overlay"):
        path = policy_input.get(key)
        if path:
            images.append(image_to_base64(path))
    return images
