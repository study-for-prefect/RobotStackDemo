"""VLM policies for task-level contracts and temporary scene bindings."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

import requests

from .io_utils import image_to_base64, parse_json_or_embedded
from .house_task_definition import HOUSE_DEFINITION_PROMPT, HOUSE_ROLE_IDS, canonical_house_goal_spec
from .task_schemas import schema_for_policy


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
        "bbox_xyxy_px": obj.get("bbox_xyxy_px") or obj.get("bbox"),
        "pointcloud_height_range_m": obj.get("pointcloud_height_range_m"),
        "pca_axes_base": obj.get("pca_axes_base") or obj.get("pca_main_axes"),
        "local_plane_normal_base": obj.get("local_plane_normal_base"),
        "image_up_direction_base_xy": obj.get("image_up_direction_base_xy"),
        "image_right_direction_base_xy": obj.get("image_right_direction_base_xy"),
        "contour_vertices_px": obj.get("contour_vertices_px"),
        "contour_inner_angles_deg": obj.get("contour_inner_angles_deg"),
        "rgb_crop": obj.get("rgb_crop"),
        "depth_crop": obj.get("depth_crop") or obj.get("depth_colormap_crop"),
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
            "For build_house use the canonical structure_variant and exactly the six canonical roles.",
            "For an underspecified organize request, use organize_defaults exactly.",
        ],
        "canonical_build_house_goal_spec": canonical_house_goal_spec(),
        "build_house_definition": HOUSE_DEFINITION_PROMPT,
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "output_schema": schema_for_policy("task_contract"),
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
        "house_frame": _house_frame_description(semantics_config),
        "build_house_definition": HOUSE_DEFINITION_PROMPT,
        "allowed_house_roles": list(HOUSE_ROLE_IDS),
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "binding_rules": [
            "All selected object ids are valid only for this scene_revision.",
            "Assignments must be temporary and replaceable when the contract allows it.",
            "Do not bind a task role permanently to a detection object id.",
            "For roof and triangle_top output semantic orientation_observations only; code computes target orientation.",
        ],
        "output_schema": schema_for_policy("grounded_task_plan", task_contract.get("task_type", "")),
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
        "build_house_definition": HOUSE_DEFINITION_PROMPT,
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "output_schema": schema_for_policy("task_action", task_contract.get("task_type", "")),
    }


def call_vlm_task_policy(args: Any, policy_input: dict, policy_kind: str) -> dict:
    """Call the configured VLM for a contract or grounded plan JSON response."""
    prompt = _task_prompt(policy_input, policy_kind)
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    system_message = {
        "role": "system",
        "content": "你是 UR5 桌面积木任务语义规划器。以下 build_house 本体不可改写：\n{}".format(HOUSE_DEFINITION_PROMPT),
    }
    images = _images(policy_input, not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    try:
        task_type = str((policy_input.get("task_contract") or {}).get("task_type") or "")
        response_schema = schema_for_policy(policy_kind, task_type)
        response = requests.post(
            getattr(args, "ollama_url", "http://127.0.0.1:11434/api/chat"),
            json={
                "model": getattr(args, "model", "qwen2.5vl:7b-q4_K_M"),
                "messages": [system_message, message], "stream": False, "format": response_schema,
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
            "build_house 必须逐字遵守下方固定六角色结构，不得生成其他角色。"
        )
    elif policy_kind == "grounded_task_plan":
        instruction = (
            "根据固定 task_contract、当前 scene_revision 和未满足谓词输出临时 grounded_task_plan。"
            "你决定当前 object_id，但必须标记 temporary；不能把检测 id 当作永久身份。"
            "roof 与 triangle_top 只输出图像语义姿态观察，不输出最终旋转轴或角度。"
        )
    else:
        instruction = (
            "基于固定任务合同和当前未满足谓词输出一个动作。所有可执行动作必须且只能用 selected_object_id，禁止输出 object_id。"
            "房子 pick_place 给 role_id；整理 pick_place 给 group_id 和 target_region_id；并给 scene_revision、准确 grounding 和目标位姿。"
            "需要改变 roof/triangle 正反面时选择 pick_reorient_place；仅 yaw 不能代替翻面，具体轴角由代码计算。"
            "nudge/pick_away 也使用 selected_object_id 和完整物理参数。"
            "只能选择当前检测 id；不要自动宣称任务完成。"
        )
    return "你是 UR5 桌面积木任务语义规划器。\n{}\n\n{}\n只输出符合 output_schema 的 JSON。\n输入：\n{}".format(
        HOUSE_DEFINITION_PROMPT, instruction, json.dumps(text, ensure_ascii=False, indent=2),
    )


def _scene_objects(state: dict) -> Iterable[dict]:
    return [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict) and not obj.get("is_workspace") and str(obj.get("label", "")).lower() != "workspace"
    ]


def _images(policy_input: dict, include: bool) -> List[str]:
    if not include:
        return []
    paths = [policy_input.get(key) for key in ("scene_rgb", "minimal_overlay", "depth_image")]
    for obj in policy_input.get("objects", []):
        if any(token in str(obj.get("label") or "").lower() for token in ("rectangle", "triangle")):
            paths.extend((obj.get("rgb_crop"), obj.get("depth_crop")))
    return [image_to_base64(path) for path in paths if path]


def _house_frame_description(config: dict) -> dict:
    up = config.get("house_semantics", {}).get("house_up_direction_base_xy", [0.0, 1.0])
    return {
        "house_x": "current left-column center to current right-column center",
        "house_y": "table-plane perpendicular selected to agree with configured house_up",
        "house_z": "table normal",
        "house_up_direction_base_xy": up,
        "warning": "Image up is not a fixed base_link direction.",
    }
