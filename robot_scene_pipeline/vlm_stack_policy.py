"""VLM-first stack-order decision policy."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

import requests

from .io_utils import image_to_base64, parse_json_or_embedded


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
    objects = [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict)
        and not obj.get("is_workspace")
        and str(obj.get("label", "")).lower() != "workspace"
    ]
    return {
        "schema_version": "vlm_stack_decision_input_v1",
        "instruction": instruction,
        "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"),
        "base_frame": state.get("base_frame", "base_link"),
        "objects": [
            compact_stack_object(obj)
            for obj in objects
        ],
        "scene_integrity": {
            "object_ids_unique": not _duplicate_object_ids(objects),
            "duplicate_object_ids": _duplicate_object_ids(objects),
            "same_label_instance_groups": _label_instance_groups(objects),
            "same_label_rule": "If several objects share a label/color, choose by object id, bbox, and base_link center; never by color alone.",
        },
        "output_schema": {
            "base_object_id": "int",
            "stack_order": "list[int], place order excluding base_object_id",
            "full_stack_order": "list[int], starts with base_object_id",
            "object_bindings": "list of selected id + exact observed label + base_link center",
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


def validate_vlm_stack_decision(raw_output: dict, state: dict) -> dict:
    """Validate stack JSON shape and referenced scene ids without task-rule repair."""
    if raw_output.get("call_status") != "parsed":
        raise RuntimeError("VLM stack decision failed: {}".format(raw_output.get("error")))
    decision = raw_output.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("VLM stack decision must be a JSON object.")
    if decision.get("task_type") not in ("stack_blocks", "structure_plan"):
        raise ValueError("VLM stack decision task_type must be stack_blocks.")
    duplicate_ids = _duplicate_object_ids(state.get("objects", []))
    if duplicate_ids:
        raise ValueError("Scene object ids must be unique before VLM stack decision: {}".format(duplicate_ids))
    base_id = int(decision.get("base_object_id"))
    stack_order = [int(value) for value in decision.get("stack_order", [])]
    full_stack_order = [int(value) for value in decision.get("full_stack_order", [])]
    if base_id in stack_order:
        raise ValueError("VLM stack_order must exclude base_object_id.")
    if not full_stack_order or full_stack_order[0] != base_id:
        raise ValueError("VLM full_stack_order must start with base_object_id.")
    if len(stack_order) != len(set(stack_order)) or len(full_stack_order) != len(set(full_stack_order)):
        raise ValueError("VLM stack decision ids must be unique.")
    if full_stack_order != [base_id] + stack_order:
        raise ValueError("VLM full_stack_order must equal [base_object_id] + stack_order.")
    if not isinstance(decision.get("structure_plan"), dict):
        raise ValueError("VLM stack decision requires structure_plan object.")
    if not str(decision.get("reason") or "").strip():
        raise ValueError("VLM stack decision requires non-empty reason.")
    valid_ids = {
        int(obj["id"])
        for obj in state.get("objects", [])
        if isinstance(obj, dict) and obj.get("id") is not None and not obj.get("is_workspace")
    }
    unknown_ids = set(full_stack_order) - valid_ids
    if unknown_ids:
        raise ValueError("VLM stack decision references unknown object ids: {}".format(sorted(unknown_ids)))
    object_bindings = _validate_object_bindings(decision.get("object_bindings"), full_stack_order, state)
    validated = dict(decision)
    validated["task_type"] = "stack_blocks"
    validated["base_object_id"] = base_id
    validated["stack_order"] = stack_order
    validated["full_stack_order"] = full_stack_order
    validated["object_bindings"] = object_bindings
    validated["stack_order_semantics"] = "place_order_excludes_base"
    validated["confidence"] = _strict_confidence(decision.get("confidence"))
    validated["decision_source"] = "vlm_stack_policy"
    return validated


def build_vlm_stack_prompt(policy_input: dict) -> str:
    text_input = {key: value for key, value in policy_input.items() if key not in ("scene_rgb", "minimal_overlay")}
    return (
        "你是 UR5 桌面积木堆叠任务的 VLM 结构决策模块。\n"
        "你必须根据用户指令、快照图和检测后的精简 JSON 决定底座和堆叠顺序。\n"
        "只能选择 objects 中已有的 object id；不要编造物体。不要输出相机内参、关节角、轨迹、速度或 ROS 命令。\n"
        "stack_order 必须是不含 base_object_id 的放置顺序，full_stack_order 必须以 base_object_id 开头。\n\n"
        "你负责理解用户任务语义并保证 JSON 顺序与目标一致；代码只检查 JSON 结构、物体 id 和几何是否可执行。\n\n"
        "如果同一种颜色/label 有多个实例，必须用编号图、object id、bbox 和 base_link 坐标区分；不要只按颜色猜。\n\n"
        "先为 full_stack_order 中每个 id 输出 object_bindings，并从 objects 原样复制 observed_label 和 "
        "geometry_center_base_m；任何 id/label/中心不一致都会停止。\n\n"
        "只输出严格 JSON：\n"
        "{\n"
        '  "task_type": "stack_blocks",\n'
        '  "structure_plan": {"structure_type": "tower", "roles": [], "assembly_steps": [], "limitations": []},\n'
        '  "base_object_id": 0,\n'
        '  "full_stack_order": [0, 1, 2],\n'
        '  "stack_order": [1, 2],\n'
        '  "object_bindings": [\n'
        '    {"object_id": 0, "observed_label": "exact detector label", "geometry_center_base_m": [0.0, 0.0, 0.0]}\n'
        '  ],\n'
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


def _duplicate_object_ids(objects: Iterable[ObjectDict]) -> List[Any]:
    seen = set()
    duplicates = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("id")
        key = str(object_id)
        if key in seen and object_id not in duplicates:
            duplicates.append(object_id)
        seen.add(key)
    return duplicates


def _label_instance_groups(objects: Iterable[ObjectDict]) -> List[dict]:
    groups: Dict[str, List[ObjectDict]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        groups.setdefault(str(obj.get("label")), []).append(obj)
    output = []
    for label, items in sorted(groups.items()):
        if len(items) <= 1:
            continue
        output.append({
            "label": label,
            "instances": [
                {
                    "id": obj.get("id"),
                    "bbox_xyxy_px": obj.get("bbox_xyxy_px"),
                    "geometry_center_base_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
                }
                for obj in items
            ],
        })
    return output


def _validate_object_bindings(bindings: Any, full_order: List[int], state: dict) -> List[dict]:
    if not isinstance(bindings, list):
        raise ValueError("VLM stack decision requires object_bindings list.")
    object_map = {
        int(obj["id"]): obj
        for obj in state.get("objects", [])
        if isinstance(obj, dict) and obj.get("id") is not None
    }
    binding_map = {}
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("VLM stack object_bindings entries must be objects.")
        object_id = int(binding.get("object_id"))
        if object_id in binding_map:
            raise ValueError("VLM stack object_bindings ids must be unique.")
        binding_map[object_id] = binding
    if set(binding_map) != set(full_order):
        raise ValueError("VLM stack object_bindings must cover exactly full_stack_order ids.")
    validated = []
    for object_id in full_order:
        binding = binding_map[object_id]
        observed = object_map[object_id]
        reported_label = str(binding.get("observed_label") or "").strip().lower()
        observed_label = str(observed.get("label") or "").strip().lower()
        if reported_label != observed_label:
            raise ValueError("VLM stack object binding label does not match id {}.".format(object_id))
        reported_center = binding.get("geometry_center_base_m")
        observed_center = observed.get("geometry_center_m") or observed.get("center_3d_base_m")
        if not isinstance(reported_center, (list, tuple)) or len(reported_center) != 3 or not observed_center:
            raise ValueError("VLM stack object binding center is missing for id {}.".format(object_id))
        try:
            center = [float(value) for value in reported_center]
            distance = math.sqrt(sum((center[index] - float(observed_center[index])) ** 2 for index in range(3)))
        except (TypeError, ValueError):
            raise ValueError("VLM stack object binding center is invalid for id {}.".format(object_id))
        if not math.isfinite(distance) or distance > 0.005:
            raise ValueError("VLM stack object binding center does not match id {}.".format(object_id))
        validated.append({
            "object_id": object_id,
            "observed_label": observed.get("label"),
            "geometry_center_base_m": [float(value) for value in observed_center[:3]],
        })
    return validated


def _strict_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ValueError("VLM stack confidence must be numeric.")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("VLM stack confidence must be within 0.0..1.0.")
    return confidence
