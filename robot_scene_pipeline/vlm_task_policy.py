"""VLM policies for task-level contracts and temporary scene bindings."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import requests

from .io_utils import image_to_base64, parse_json_or_embedded


TASK_TYPES = {"build_house", "organize_blocks"}


def compact_task_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return stable scene facts; detection ids remain explicitly temporary."""
    return {
        "object_id": obj.get("id"),
        "label": obj.get("label"),
        "confidence": obj.get("confidence"),
        "geometry_center_base_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
        "dimensions_m": obj.get("dimensions_m"),
        "yaw_rad": obj.get("yaw_rad"),
        "visible": obj.get("visible", True),
    }


def build_task_contract_input(
    state: dict, instruction: str, semantics_config: dict,
) -> dict:
    """Build task-only VLM input without permanent object assignments."""
    return {
        "schema_version": "task_contract_input_v1",
        "instruction": instruction,
        "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"),
        "supported_task_types": sorted(TASK_TYPES),
        "organize_defaults": semantics_config.get("organize_defaults", {}),
        "contract_rules": [
            "Task contract must not contain object_id, selected_object_id, or temporary detection bindings.",
            "Use roles, category/label requirements, size ranges, and spatial relations only.",
            "For an underspecified organize request, use organize_defaults exactly.",
        ],
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "output_schema": {
            "schema_version": "task_contract_v1",
            "task_type": "build_house|organize_blocks",
            "goal_spec": "object containing role/group/relation requirements only",
            "reason": "string",
            "confidence": "0.0..1.0",
        },
    }


def build_grounded_task_plan_input(
    state: dict,
    task_contract: dict,
    scene_revision: int,
    semantics_config: dict,
    goal_progress: Optional[dict] = None,
    previous_plan: Optional[dict] = None,
    failure_history: Optional[List[dict]] = None,
) -> dict:
    """Build current-scene VLM input for temporary role/group bindings."""
    return {
        "schema_version": "grounded_task_plan_input_v1",
        "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"),
        "scene_revision": int(scene_revision),
        "task_contract": task_contract,
        "goal_progress": goal_progress or {},
        "previous_plan": previous_plan or {},
        "failure_history": list(failure_history or []),
        "workspace_bounds": state.get("table_bounds") or state.get("workspace_bounds"),
        "semantics_config": semantics_config,
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "binding_rules": [
            "All selected object ids are valid only for this scene_revision.",
            "Assignments must be temporary and replaceable when the contract allows it.",
            "Do not bind a task role permanently to a detection object id.",
        ],
    }


def build_task_action_input(
    state: dict, task_contract: dict, grounded_plan: dict, goal_progress: dict,
    scene_revision: int, failure_history: Optional[List[dict]] = None,
) -> dict:
    """Build VLM input for exactly one current-scene task action."""
    return {
        "schema_version": "task_action_input_v1", "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"), "scene_revision": int(scene_revision),
        "task_contract": task_contract, "grounded_task_plan": grounded_plan,
        "current_goal_progress": goal_progress, "failure_history": list(failure_history or []),
        "workspace_bounds": state.get("table_bounds") or state.get("workspace_bounds"),
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "output_schema": {
            "action_type": "pick_place|nudge|pick_away|reobserve|stop",
            "role_id": "required for pick_place", "selected_object_id": "current scene id",
            "object_label": "exact detector label", "object_center_base_m": "exact base_link XYZ",
            "scene_revision": "must equal input scene_revision",
            "target_pose_base": {"position_m": "XYZ in base_link", "yaw_rad": "finite"},
            "target_object_id": "required for nudge/pick_away", "target_object_label": "exact label",
            "target_object_center_base_m": "exact XYZ", "contact_side": "nudge contact side",
            "direction_base": "nudge unit base_link XY direction", "distance_m": "nudge distance",
            "gripper_yaw_rad": "nudge yaw", "safe_place_center_base_m": "pick_away temporary place",
            "expected_goal_predicates": "list[string]", "reason": "string", "confidence": "0.0..1.0",
        },
    }


def call_vlm_task_policy(args: Any, policy_input: dict, policy_kind: str) -> dict:
    """Call the configured VLM for a contract or grounded plan JSON response."""
    prompt = _task_prompt(policy_input, policy_kind)
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    images = _images(policy_input, not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    try:
        response = requests.post(
            getattr(args, "ollama_url", "http://127.0.0.1:11434/api/chat"),
            json={
                "model": getattr(args, "model", "qwen2.5vl:7b-q4_K_M"),
                "messages": [message], "stream": False, "format": "json",
                "options": {"temperature": 0.0, "num_predict": min(1536, int(getattr(args, "num_predict", 1536)))},
            },
            timeout=float(getattr(args, "timeout", 600)),
        )
        response.raise_for_status()
        raw_content = response.json().get("message", {}).get("content", "")
        return {"call_status": "parsed", "raw_content": raw_content, "decision": parse_json_or_embedded(raw_content)}
    except Exception as exc:
        return {"call_status": "fail_safe_stop", "raw_content": locals().get("raw_content", ""), "error": str(exc), "decision": None}


def _task_prompt(policy_input: dict, policy_kind: str) -> str:
    text = {key: value for key, value in policy_input.items() if key not in ("scene_rgb", "minimal_overlay")}
    if policy_kind == "task_contract":
        instruction = (
            "输出一次且仅一次任务合同。只能选择 build_house 或 organize_blocks。"
            "合同定义可替代角色、分组规则和空间关系，绝不能写任何 object_id。"
        )
    elif policy_kind == "grounded_task_plan":
        instruction = (
            "根据固定 task_contract、当前 scene_revision 和未满足谓词输出临时 grounded_task_plan。"
            "你决定当前 object_id，但必须标记 temporary；不能把检测 id 当作永久身份。"
        )
    else:
        instruction = (
            "基于固定任务合同和当前未满足谓词输出一个动作。pick_place 必须给 role_id、当前 selected_object_id、"
            "scene_revision、准确 object grounding 和 target_pose_base。nudge/pick_away 使用当前检测 id 和完整物理参数。"
            "只能选择当前检测 id；不要自动宣称任务完成。"
        )
    return "你是 UR5 桌面积木任务语义规划器。\n{}\n只输出 JSON。\n输入：\n{}".format(
        instruction, json.dumps(text, ensure_ascii=False, indent=2),
    )


def _scene_objects(state: dict) -> Iterable[dict]:
    return [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict) and not obj.get("is_workspace") and str(obj.get("label", "")).lower() != "workspace"
    ]


def _images(policy_input: dict, include: bool) -> List[str]:
    if not include:
        return []
    return [image_to_base64(path) for key in ("scene_rgb", "minimal_overlay") if (path := policy_input.get(key))]
