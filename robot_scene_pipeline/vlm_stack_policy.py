"""VLM-first stack-order decision policy."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import image_to_base64
from .llm_stack_blocks import color_mentions, object_label_contains
from .ollama_policy_client import call_policy
from .stack_binding import STACK_BINDING_SCHEMA, required_color_order, validate_stack_binding
from .vlm_replanning import StackSemanticValidationError, stack_feedback


ObjectDict = Dict[str, Any]


def compact_stack_object(obj: ObjectDict) -> ObjectDict:
    return {
        "object_ref": obj.get("object_ref"),
        "track_id": obj.get("track_id"),
        "detector_object_id": obj.get("id"),
        "label": obj.get("label"),
        "visual_color": obj.get("visual_color"),
        "visual_color_confidence": obj.get("visual_color_confidence"),
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


def build_vlm_stack_decision_input(
    state: dict,
    instruction: str,
    failure_history: Optional[List[dict]] = None,
    forbidden_order_fingerprints: Optional[List[str]] = None,
) -> dict:
    objects = [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict)
        and not obj.get("is_workspace")
        and str(obj.get("label", "")).lower() != "workspace"
    ]
    return {
        "schema_version": "vlm_stack_decision_input_v1",
        "instruction": instruction,
        "scene_revision": int(state.get("scene_revision", 1)),
        "required_color_order": required_color_order(instruction),
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
            "same_label_rule": "If several objects share a color, choose by object_ref, track_id, bbox, and base_link center.",
        },
        "output_schema": STACK_BINDING_SCHEMA,
        "policy_role_split": {
            "vlm": "structure_and_stack_order_decision",
            "code": "id_geometry_validation_and_execution",
        },
        "failure_history": list(failure_history or []),
        "forbidden_order_fingerprints": list(forbidden_order_fingerprints or []),
    }


def call_vlm_stack_policy(args: Any, policy_input: dict, artifact_dir: Optional[str] = None) -> dict:
    prompt = build_vlm_stack_prompt(policy_input)
    message = {"role": "user", "content": prompt}
    images = _image_payloads(policy_input, include_images=not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    result = call_policy(
        args, "stack_order", [message], STACK_BINDING_SCHEMA,
        artifact_dir=artifact_dir,
        reasoning_attempt=len(policy_input.get("failure_history") or []) + 1,
    )
    return {
        "schema_version": "vlm_stack_binding_raw_v2",
        "call_status": "parsed" if result.parsed_decision is not None else result.generation_status,
        "raw_content": result.content,
        "thinking": result.thinking,
        "error_type": result.error_type,
        "error": result.error_message,
        "diagnostics": result.to_dict(),
        "decision": result.parsed_decision,
    }


def validate_vlm_stack_decision(raw_output: dict, state: dict, instruction: str = "") -> dict:
    """Validate stack JSON shape and referenced scene ids without task-rule repair."""
    if raw_output.get("call_status") != "parsed":
        raise RuntimeError("VLM stack decision failed: {}".format(raw_output.get("error")))
    decision = raw_output.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("VLM stack decision must be a JSON object.")
    if decision.get("schema_version") == "stack_binding_v2":
        return validate_stack_binding(decision, state, instruction)
    if decision.get("task_type") not in ("stack_blocks", "structure_plan"):
        raise ValueError("VLM stack decision task_type must be stack_blocks.")
    duplicate_ids = _duplicate_object_ids(state.get("objects", []))
    if duplicate_ids:
        raise ValueError("Scene object ids must be unique before VLM stack decision: {}".format(duplicate_ids))
    try:
        full_stack_order = [int(value) for value in decision.get("full_stack_order", [])]
    except (TypeError, ValueError):
        _raise_stack_error("VLM full_stack_order must contain integer ids.", [{"type": "invalid_id"}], instruction, state)
    if not full_stack_order:
        _raise_stack_error("VLM full_stack_order must not be empty.", [{"type": "empty_full_stack_order"}], instruction, state)
    if len(full_stack_order) != len(set(full_stack_order)):
        duplicates = sorted({value for value in full_stack_order if full_stack_order.count(value) > 1})
        _raise_stack_error(
            "VLM full_stack_order ids must be unique.",
            [{"type": "duplicate_object_id", "object_id": value} for value in duplicates],
            instruction,
            state,
        )
    base_id = full_stack_order[0]
    stack_order = full_stack_order[1:]
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
        _raise_stack_error(
            "VLM stack decision references unknown object ids: {}".format(sorted(unknown_ids)),
            [{"type": "unknown_object_id", "object_id": value} for value in sorted(unknown_ids)],
            instruction,
            state,
        )
    _validate_instruction_label_order(full_stack_order, state, instruction)
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
        "你必须为红、绿、蓝、黄四个固定颜色槽位选择当前场景实例。\n"
        "只能使用 objects 中的 object_ref 和 track_id，不得输出裸 object id。\n"
        "颜色顺序由 required_color_order 固定；颜色优先使用 objects.visual_color，缺失时再结合 label 和图像；"
        "若同色有多个实例，由你结合图像和几何自主选择。\n"
        "禁止输出 forbidden_order_fingerprints 中相同的四 track 组合。\n"
        "只输出严格 JSON：\n"
        "{\n"
        '  "schema_version": "stack_binding_v2",\n'
        '  "structure": "tower",\n'
        '  "strategy": "vertical_stack",\n'
        '  "selected_by_color": {\n'
        '    "red": {"object_ref": "scene_1:obj_1", "track_id": "track_red_01"},\n'
        '    "green": {"object_ref": "scene_1:obj_2", "track_id": "track_green_01"},\n'
        '    "blue": {"object_ref": "scene_1:obj_0", "track_id": "track_blue_01"},\n'
        '    "yellow": {"object_ref": "scene_1:obj_4", "track_id": "track_yellow_01"}\n'
        '  },\n'
        '  "reason": "...",\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "输入：\n"
        + json.dumps(text_input, ensure_ascii=False, indent=2)
    )


def _raise_stack_error(message: str, errors: List[dict], instruction: str, state: dict) -> None:
    raise StackSemanticValidationError(message, stack_feedback(errors, instruction, state))


def _validate_instruction_label_order(full_order: List[int], state: dict, instruction: str) -> None:
    required_colors = color_mentions(instruction or "")
    if not required_colors:
        return
    object_map = {
        int(obj["id"]): obj
        for obj in state.get("objects", [])
        if isinstance(obj, dict) and obj.get("id") is not None
    }
    selected_colors = []
    for object_id in full_order:
        obj = object_map[object_id]
        color = next((item for item in required_colors if object_label_contains(obj, item)), None)
        selected_colors.append(color or str(obj.get("label") or ""))
    errors = []
    for color in required_colors:
        required_label = next(
            (
                str(obj.get("label"))
                for obj in object_map.values()
                if object_label_contains(obj, color)
            ),
            color,
        )
        count = selected_colors.count(color)
        if count == 0:
            errors.append({"type": "missing_required_label", "label": required_label})
        elif count > 1:
            errors.append({"type": "duplicate_label", "label": required_label})
    if selected_colors != required_colors:
        errors.append({
            "type": "label_order_mismatch",
            "required_label_order": required_colors,
            "proposed_label_order": selected_colors,
        })
    if errors:
        _raise_stack_error("VLM full_stack_order does not match the requested label order.", errors, instruction, state)


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
