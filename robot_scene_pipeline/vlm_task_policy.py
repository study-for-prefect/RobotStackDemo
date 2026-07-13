"""VLM policies for task-level contracts and temporary scene bindings."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import image_to_base64
from .house_task_definition import HOUSE_DEFINITION_PROMPT, HOUSE_ROLE_IDS, canonical_house_goal_spec
from .ollama_policy_client import call_policy
from .task_schemas import schema_for_policy
from .task_routing import route_task_type


TASK_TYPES = {"build_house", "organize_blocks"}


def compact_task_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return stable scene facts; detection ids remain explicitly temporary."""
    return {
        "object_ref": obj.get("object_ref"),
        "track_id": obj.get("track_id"),
        "detector_object_id": obj.get("id"),
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
    state: dict, instruction: str, semantics_config: dict, expected_task_type: str = "",
) -> dict:
    """Build task-only VLM input without permanent object assignments."""
    expected = expected_task_type or route_task_type(instruction)
    value = {
        "schema_version": "task_contract_input_v1",
        "instruction": instruction,
        "expected_task_type": expected,
        "contract_rules": [
            "Task contract must not contain object_id, selected_object_id, or temporary detection bindings.",
            "Use roles, category/label requirements, size ranges, and spatial relations only.",
        ],
        "output_schema": schema_for_policy("task_contract", expected),
    }
    if expected == "build_house":
        value.update({
            "canonical_build_house_goal_spec": canonical_house_goal_spec(),
            "build_house_definition": HOUSE_DEFINITION_PROMPT,
        })
    elif expected == "organize_blocks":
        value["organize_defaults"] = semantics_config.get("organize_defaults", {})
    return value


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
    value = {
        "schema_version": "grounded_task_plan_input_v1",
        "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"),
        "scene_revision": int(scene_revision),
        "task_contract": task_contract,
        "goal_progress": goal_progress or {},
        "previous_plan": previous_plan or {},
        "failure_history": list(failure_history or []),
        "workspace_bounds": state.get("table_bounds") or state.get("workspace_bounds"),
        "semantics_config": (
            semantics_config
            if task_contract.get("task_type") == "build_house"
            else {"organize_defaults": semantics_config.get("organize_defaults", {})}
        ),
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "binding_rules": [
            "All object_ref values are valid only for this scene_revision.",
            "Assignments must be temporary and replaceable when the contract allows it.",
            "Do not bind a task role permanently to a detection object id.",
        ],
        "output_schema": schema_for_policy("grounded_task_plan", task_contract.get("task_type", "")),
    }
    if task_contract.get("task_type") == "build_house":
        value.update({
            "house_frame": _house_frame_description(semantics_config),
            "build_house_definition": HOUSE_DEFINITION_PROMPT,
            "allowed_house_roles": list(HOUSE_ROLE_IDS),
        })
        value["binding_rules"].append(
            "For roof and triangle_top output semantic orientation_observations only; code computes target orientation."
        )
    return value


def build_task_action_input(
    state: dict, task_contract: dict, grounded_plan: dict, goal_progress: dict,
    scene_revision: int, failure_history: Optional[List[dict]] = None,
    replanning_context: Optional[dict] = None,
) -> dict:
    """Build VLM input for exactly one current-scene task action."""
    value = {
        "schema_version": "task_action_input_v1", "scene_rgb": state.get("snapshot_image"),
        "minimal_overlay": state.get("annotated_image"), "scene_revision": int(scene_revision),
        "task_contract": task_contract, "grounded_task_plan": grounded_plan,
        "current_goal_progress": goal_progress, "failure_history": list(failure_history or []),
        "replanning_context": replanning_context or {},
        "workspace_bounds": state.get("table_bounds") or state.get("workspace_bounds"),
        "objects": [compact_task_object(obj) for obj in _scene_objects(state)],
        "output_schema": schema_for_policy("task_action", task_contract.get("task_type", "")),
    }
    if task_contract.get("task_type") == "build_house":
        value["build_house_definition"] = HOUSE_DEFINITION_PROMPT
    return value


def call_vlm_task_policy(
    args: Any, policy_input: dict, policy_kind: str, artifact_dir: Optional[str] = None,
) -> dict:
    """Call the configured VLM for a contract or grounded plan JSON response."""
    prompt = _task_prompt(policy_input, policy_kind)
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    task_type = str(
        policy_input.get("expected_task_type")
        or (policy_input.get("task_contract") or {}).get("task_type")
        or ""
    )
    system_content = "你是 UR5 桌面积木任务语义规划器。"
    if task_type == "build_house":
        system_content += "以下 build_house 本体不可改写：\n{}".format(HOUSE_DEFINITION_PROMPT)
    system_message = {"role": "system", "content": system_content}
    images = _images(policy_input, not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    response_schema = schema_for_policy(policy_kind, task_type)
    attempt = int(
        (policy_input.get("replanning_context") or {}).get("attempt")
        or len(policy_input.get("failure_history") or []) + 1
    )
    client_kind = {
        "task_contract": "task_contract",
        "grounded_task_plan": "grounded_task_plan",
        "task_action": "action_proposal" if attempt <= 1 else "action_replan",
    }[policy_kind]
    result = call_policy(
        args, client_kind, [system_message, message], response_schema,
        artifact_dir=artifact_dir, reasoning_attempt=attempt,
        temperature=(
            0.0 if policy_kind != "task_action" or attempt <= 1
            else float(getattr(args, "replanning_temperature", 0.15))
        ),
        top_p=(
            float(getattr(args, "replanning_top_p", 0.85))
            if policy_kind == "task_action" else 0.85
        ),
    )
    return {
        "call_status": "parsed" if result.parsed_decision is not None else result.generation_status,
        "raw_content": result.content,
        "thinking": result.thinking,
        "error_type": result.error_type,
        "error": result.error_message,
        "diagnostics": result.to_dict(),
        "decision": result.parsed_decision,
    }


def _task_prompt(policy_input: dict, policy_kind: str) -> str:
    text = {key: value for key, value in policy_input.items() if key not in ("scene_rgb", "minimal_overlay")}
    if policy_kind == "task_contract":
        expected = policy_input.get("expected_task_type")
        instruction = "输出一次且仅一次 {} 任务合同，绝不能写任何 object_id。".format(expected)
        if expected == "build_house":
            instruction += "必须逐字遵守输入中的固定六角色结构，不得生成其他角色。"
        else:
            instruction += "只定义按颜色整理的分组、布局、范围和间距语义。"
    elif policy_kind == "grounded_task_plan":
        if (policy_input.get("task_contract") or {}).get("task_type") == "build_house":
            instruction = (
                "输出精简 grounded_house_plan_v1，只包含 role_bindings、orientation_observations、reason 和 confidence。"
                "role_bindings 必须使用当前 object_ref/track_id；不要重复 label、中心、尺寸、固定步骤或完成状态。"
                "roof 与 triangle_top 只输出图像语义姿态观察，不输出最终旋转轴或角度。"
            )
        else:
            instruction = "根据按颜色整理合同输出当前场景的分组绑定和目标区域，不得输出任何房屋角色或房屋结构。"
    else:
        instruction = (
            "基于固定任务合同和当前未满足谓词输出一个动作。所有可执行动作必须使用 selected_object_ref 或 selected_track_id，禁止裸整数 object_id/selected_object_id。"
            "房子 pick_place 给 role_id；整理 pick_place 给 group_id 和 target_region_id；并给 scene_revision、准确 grounding 和目标位姿。"
            "需要改变 roof/triangle 正反面时选择 pick_reorient_place；仅 yaw 不能代替翻面，具体轴角由代码计算。"
            "nudge/pick_away 也使用统一引用和完整物理参数。禁止输出 replanning_context 中的失败指纹，不能只修改 reason、confidence 或小数尾数。"
            "nudge 只能用于桌面平面清障，不能形成 on_top_of；堆叠必须 pick_place。只能选择当前可见引用；不要自动宣称任务完成。"
        )
    ontology = "\n{}\n".format(HOUSE_DEFINITION_PROMPT) if (
        policy_input.get("expected_task_type") == "build_house"
        or (policy_input.get("task_contract") or {}).get("task_type") == "build_house"
    ) else "\n"
    return "你是 UR5 桌面积木任务语义规划器。{}\n{}\n只输出符合 output_schema 的 JSON。\n输入：\n{}".format(
        ontology, instruction, json.dumps(text, ensure_ascii=False, indent=2),
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
