"""VLM action-intent policy and safety validation for stack clearing."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import image_to_base64, parse_json_or_embedded
from .ollama_policy_client import call_policy
from .task_schemas import VLM_ACTION_SCHEMA
from .vlm_action_validation import mark_moveit_result, validate_vlm_action_decision


ObjectDict = Dict[str, Any]
ActionDict = Dict[str, Any]

ACTION_TYPES = {"pick", "nudge", "pick_away", "reobserve", "stop"}


def compact_object_for_action_policy(obj: ObjectDict) -> ObjectDict:
    """Return scene facts for VLM action intent without camera intrinsics."""
    return {
        "object_ref": obj.get("object_ref"),
        "track_id": obj.get("track_id"),
        "detector_object_id": obj.get("id"),
        "label": obj.get("label"),
        "confidence": obj.get("confidence"),
        "bbox_xyxy_px": obj.get("bbox_xyxy_px"),
        "center_px": obj.get("center_px"),
        "depth_m": obj.get("depth_m"),
        "geometry_center_base_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
        "dimensions_m": obj.get("dimensions_m"),
        "semantic_shape": obj.get("semantic_shape"),
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
    task_focus_object: ObjectDict,
    protected_ids: Iterable[Any],
    base_id: Any,
    memory: Optional[dict],
    step_index: int,
    manipulator_geometry: Optional[dict] = None,
    failure_history: Optional[List[dict]] = None,
    scene_revision: int = 1,
    depth_visualization_path: Optional[str] = None,
    replanning_context: Optional[dict] = None,
) -> dict:
    """Build VLM input from images, objective scene facts, and task state."""
    structure = (memory or {}).get("structure") or {}
    objects = [obj for obj in current_state.get("objects", []) if isinstance(obj, dict)]
    duplicate_ids = _duplicate_object_ids(objects)
    label_groups = _label_instance_groups(objects)
    protected_id_keys = {str(value) for value in protected_ids or []}
    base_object = _object_by_detector_id(objects, base_id)
    protected_objects = [obj for obj in objects if str(obj.get("id")) in protected_id_keys]
    return {
        "schema_version": "vlm_autonomous_action_input_v2",
        "scene_rgb": scene_rgb_path,
        "depth_visualization": depth_visualization_path,
        "minimal_overlay": minimal_overlay_path,
        "base_frame": current_state.get("base_frame", "base_link"),
        "camera_frame": current_state.get("camera_frame", "camera_color_optical_frame"),
        "frame_convention": {
            "base_link": "+X forward, +Y left, +Z up; all action vectors use this frame",
            "camera_optical": "+X right, +Y down, +Z forward; image/depth observations use this frame",
        },
        "scene_revision": int(scene_revision),
        "task_goal": {
            "instruction": current_state.get("instruction"),
            "action_step_index": int(step_index),
            "current_plan_focus": compact_object_for_action_policy(task_focus_object),
            "current_plan_focus_is_advisory": True,
            "structure_plan": (memory or {}).get("task_plan") or structure.get("plan"),
            "stack_progress": {
                "base": structure.get("base"),
                "placed_order": structure.get("placed_order", []),
                "current_top": structure.get("current_top"),
            },
            "rule": (
                "Decide the next useful action from the whole scene. The current plan focus is context, "
                "not a forced object choice."
            ),
        },
        "objects": [
            compact_object_for_action_policy(obj)
            for obj in objects
        ],
        "scene_integrity": {
            "object_ids_unique": not duplicate_ids,
            "duplicate_object_ids": duplicate_ids,
            "same_label_instance_groups": label_groups,
        },
        "physical_context": {
            "base_object_ref": (base_object or {}).get("object_ref"),
            "base_track_id": (base_object or {}).get("track_id"),
            "protected_objects": [
                {"object_ref": obj.get("object_ref"), "track_id": obj.get("track_id")}
                for obj in protected_objects
            ],
            "note": "These identify the physical structure that must not be moved.",
            "workspace_bounds": current_state.get("table_bounds") or current_state.get("workspace_bounds"),
        },
        "manipulator_geometry": manipulator_geometry or {
            "closed_gripper_outer_width_m": 0.112,
            "finger_length_m": 0.12,
            "contact_rule": "For nudge, contact_side opposes direction_base and the tool approaches vertically.",
        },
        "policy_role_split": {
            "vlm": "scene_diagnosis_object_action_direction_distance_and_benefit_prediction",
            "code": "collision_grasp_and_execution_feasibility_validation_only",
            "moveit": "ik_collision_and_trajectory_preflight",
        },
        "output_schema": {
            "strategy_id": "high-level strategy string",
            "scene_problem": "string; VLM diagnosis of the current scene",
            "action_type": "pick|nudge|pick_away|reobserve|stop",
            "object_ref": "scene_<revision>:obj_<id>",
            "object_track_id": "stable track id",
            "object_label": "exact objects[].label for the selected reference; required for executable actions",
            "object_center_base_m": "exact objects[].geometry_center_base_m for the selected reference",
            "target_object_ref": "scene-local task target reference",
            "target_object_track_id": "stable task target track id",
            "target_object_label": "exact objects[].label for the selected target reference",
            "target_object_center_base_m": "exact objects[].geometry_center_base_m for the selected target reference",
            "contact_side": "+x|-x|+y|-y on the operated object, required for nudge",
            "direction_base": "[x,y,z] unit vector in base_link, required for nudge",
            "distance_m": "0.01..0.05, required for nudge",
            "gripper_yaw_rad": "finite yaw in base_link, required for nudge",
            "safe_place_center_base_m": "[x,y,z] in base_link, required for pick_away",
            "predicted_scene_benefit": "string; expected observable scene change and task gain",
            "risk_assessment": "string; main uncertainty or downside",
            "reason": "string",
            "confidence": "0.0..1.0",
            "alternative_actions": "optional VLM-generated action objects for deterministic anti-loop fallback",
        },
        "failure_history": list(failure_history or []),
        "replanning_context": replanning_context or {},
    }


def call_vlm_action_policy(args: Any, policy_input: dict, artifact_dir: Optional[str] = None) -> dict:
    """Call the VLM and return raw plus parsed action decision."""
    prompt = build_vlm_action_prompt(policy_input)
    message = {"role": "user", "content": prompt}
    images = _image_payloads(policy_input, include_images=not bool(getattr(args, "no_image", False)))
    if images:
        message["images"] = images
    attempt = int((policy_input.get("replanning_context") or {}).get("attempt", 1))
    result = call_policy(
        args,
        "action_proposal" if attempt <= 1 else "action_replan",
        [message],
        VLM_ACTION_SCHEMA,
        artifact_dir=artifact_dir,
        reasoning_attempt=attempt,
        temperature=0.0 if attempt <= 1 else float(getattr(args, "replanning_temperature", 0.15)),
        top_p=float(getattr(args, "replanning_top_p", 0.85)),
    )
    decision = None
    error = result.error_message
    if result.parsed_decision is not None:
        try:
            decision = parse_vlm_action_decision_text(json.dumps(result.parsed_decision, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            error = str(exc)
    error_type = result.error_type
    call_status = "parsed" if decision is not None else result.generation_status
    if result.parsed_decision is not None and decision is None:
        error_type = "DECISION_PROTOCOL_INVALID"
        call_status = "generation_failed"
    return {
        "schema_version": "vlm_action_decision_raw_v2",
        "call_status": call_status,
        "raw_content": result.content,
        "thinking": result.thinking,
        "error_type": error_type,
        "error": error,
        "diagnostics": result.to_dict(),
        "decision": decision,
    }


def build_vlm_action_prompt(policy_input: dict) -> str:
    text_input = {
        key: value for key, value in policy_input.items()
        if key not in ("scene_rgb", "depth_visualization", "minimal_overlay")
    }
    return (
        "你是 UR5 桌面积木任务的自主视觉动作决策模块。\n"
        "你直接根据原始快照、带编号图、base_link 场景几何和任务目标判断当前问题并提出下一步动作。\n"
        "不要输出关节角、轨迹、速度、ROS 控制命令或相机内参。\n\n"
        "必须独立完成五项判断：当前问题、抓还是推、操作哪个物体、推的方向和距离、操作后的场景收益。\n"
        "输入中没有代码生成的候选动作、阻挡物结论、抓取可行性结论或推荐方向。\n\n"
        "动作语义：\n"
        "- pick: 抓取你自主选择、最能推进任务的物体。\n"
        "- nudge: 推动你自主选择的松散物体；给出 base_link 下单位方向和 0.01 到 0.05 m 距离。\n"
        "- pick_away: 抓走你自主选择的松散物体，并给出 base_link 下临时放置中心。\n"
        "- reobserve: 视觉信息不足，需要重新观察。\n"
        "- stop: 没有安全/合理动作。\n\n"
        "物理约束：不移动 base、placed、locked、protected 物体；保护已堆叠结构。\n"
        "只能使用object_ref或track_id引用物体，禁止裸整数ID和数组index。\n"
        "pick 时两者通常相同；nudge/pick_away 时 operated reference 可是障碍物，target reference 可是受益对象。\n"
        "不要因为 current_plan_focus 存在就机械选择它；先看全图和全部几何，再说明你的判断。\n"
        "同颜色/label 多实例必须用track_id、object_ref、bbox和base_link中心区分，禁止只按颜色猜。\n"
        "对于可执行动作，object_label/object_center_base_m 和 target_object_label/target_object_center_base_m "
        "必须从 objects 对应 reference 原样复制；reference、label、中心不一致会被拒绝。\n"
        "nudge 必须自主给出 contact_side、direction_base、distance_m 和 gripper_yaw_rad。"
        "接触侧应位于 direction_base 反方向，并垂直接近；必须根据 manipulator_geometry "
        "确认该接触侧和工具扫掠走廊没有其他物体。不要把推动当成精确堆叠手段。\n"
        "nudge仅表示桌面平面清障，不能实现放到另一个物体上方；竖直堆叠必须pick。当前待堆叠目标默认不应nudge。\n"
        "禁止输出replanning_context中的forbidden_action_fingerprints；只改reason、confidence或小数尾数仍是重复动作。\n"
        "新动作至少实质改变strategy_id、action_type、操作物体、方向、grasp_yaw或target_role之一。\n"
        "如果 scene_integrity.object_ids_unique=false，输出 stop 并说明检测 id 不唯一，不能执行。\n"
        "代码只会验证碰撞、抓取/放置几何和 MoveIt 可行性，不会替你选择另一个动作。\n\n"
        "只输出严格 JSON：\n"
        "{\n"
        '  "strategy_id": "direct_pick_target",\n'
        '  "scene_problem": "...",\n'
        '  "action_type": "pick|nudge|pick_away|reobserve|stop",\n'
        '  "object_ref": "scene_1:obj_0",\n'
        '  "object_track_id": "track_green_01",\n'
        '  "object_label": "exact detector label",\n'
        '  "object_center_base_m": [0.0, 0.0, 0.0],\n'
        '  "target_object_ref": "scene_1:obj_1",\n'
        '  "target_object_track_id": "track_red_01",\n'
        '  "target_object_label": "exact detector label",\n'
        '  "target_object_center_base_m": [0.0, 0.0, 0.0],\n'
        '  "contact_side": "-x",\n'
        '  "direction_base": [1.0, 0.0, 0.0],\n'
        '  "distance_m": 0.025,\n'
        '  "gripper_yaw_rad": 0.0,\n'
        '  "safe_place_center_base_m": [0.0, 0.0, 0.0],\n'
        '  "predicted_scene_benefit": "...",\n'
        '  "risk_assessment": "...",\n'
        '  "reason": "...",\n'
        '  "confidence": 0.0,\n'
        '  "alternative_actions": []\n'
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
    scene_problem = _required_text(decision, "scene_problem")
    predicted_benefit = _required_text(decision, "predicted_scene_benefit")
    risk_assessment = _required_text(decision, "risk_assessment")
    reason = _required_text(decision, "reason")
    executable = action_type in ("pick", "nudge", "pick_away")
    object_center = _required_center(decision, "object_center_base_m") if executable else decision.get("object_center_base_m")
    target_center = (
        _required_center(decision, "target_object_center_base_m")
        if executable else decision.get("target_object_center_base_m")
    )
    alternatives = []
    for item in decision.get("alternative_actions") or []:
        if isinstance(item, dict):
            candidate = dict(item)
            candidate.pop("alternative_actions", None)
            try:
                alternatives.append(parse_vlm_action_decision_text(json.dumps(candidate, ensure_ascii=False)))
            except (TypeError, ValueError):
                continue
    return {
        "strategy_id": str(decision.get("strategy_id") or {"pick": "direct_pick_target", "nudge": "clear_blocker_by_nudge", "pick_away": "clear_blocker_by_pick_away", "reobserve": "reobserve_scene", "stop": "safe_stop"}.get(action_type, "unspecified")),
        "scene_problem": scene_problem,
        "action_type": action_type,
        "object_ref": decision.get("object_ref"),
        "object_track_id": decision.get("object_track_id") or decision.get("track_id"),
        "object_id": decision.get("object_id"),
        "object_label": _required_text(decision, "object_label") if executable else decision.get("object_label"),
        "object_center_base_m": object_center,
        "target_object_ref": decision.get("target_object_ref"),
        "target_object_track_id": decision.get("target_object_track_id"),
        "target_object_id": decision.get("target_object_id"),
        "target_object_label": (
            _required_text(decision, "target_object_label") if executable else decision.get("target_object_label")
        ),
        "target_object_center_base_m": target_center,
        "contact_side": decision.get("contact_side"),
        "direction_base": decision.get("direction_base", decision.get("push_direction_base")),
        "distance_m": decision.get("distance_m", decision.get("push_distance_m")),
        "gripper_yaw_rad": decision.get("gripper_yaw_rad"),
        "safe_place_center_base_m": decision.get("safe_place_center_base_m"),
        "predicted_scene_benefit": predicted_benefit,
        "risk_assessment": risk_assessment,
        "reason": reason,
        "confidence": _strict_confidence(decision.get("confidence")),
        "raw_decision": decision,
        "alternative_actions": alternatives,
    }


def _duplicate_object_ids(objects: Iterable[ObjectDict]) -> List[Any]:
    seen = set()
    duplicates = []
    for obj in objects:
        object_id = obj.get("id")
        key = str(object_id)
        if key in seen and object_id not in duplicates:
            duplicates.append(object_id)
        seen.add(key)
    return duplicates


def _label_instance_groups(objects: Iterable[ObjectDict]) -> List[dict]:
    groups: Dict[str, List[ObjectDict]] = {}
    for obj in objects:
        label = str(obj.get("label"))
        groups.setdefault(label, []).append(obj)
    output = []
    for label, items in sorted(groups.items()):
        if len(items) <= 1:
            continue
        output.append({
            "label": label,
            "instances": [
                {
                    "object_ref": obj.get("object_ref"),
                    "track_id": obj.get("track_id"),
                    "bbox_xyxy_px": obj.get("bbox_xyxy_px"),
                    "geometry_center_base_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
                }
                for obj in items
            ],
            "rule": "Use object_ref, track_id, and geometry_center_base_m to distinguish these same-label objects.",
        })
    return output


def _object_by_detector_id(objects: Iterable[ObjectDict], object_id: Any) -> Optional[ObjectDict]:
    return next((obj for obj in objects if str(obj.get("id")) == str(object_id)), None)


def _image_payloads(policy_input: dict, include_images: bool) -> List[str]:
    if not include_images:
        return []
    images = []
    for key in ("scene_rgb", "depth_visualization", "minimal_overlay"):
        path = policy_input.get(key)
        if path:
            images.append(image_to_base64(path))
    return images


def _required_text(decision: ActionDict, key: str) -> str:
    value = str(decision.get(key) or "").strip()
    if not value:
        raise ValueError("VLM action output requires non-empty {}.".format(key))
    return value


def _strict_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ValueError("VLM action output confidence must be numeric.")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("VLM action output confidence must be within 0.0..1.0.")
    return confidence


def _required_center(decision: ActionDict, key: str) -> List[float]:
    value = decision.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("VLM action output requires {} as XYZ vector.".format(key))
    try:
        center = [float(item) for item in value]
    except (TypeError, ValueError):
        raise ValueError("VLM action output {} must be numeric.".format(key))
    if not all(math.isfinite(item) for item in center):
        raise ValueError("VLM action output {} must be finite.".format(key))
    return center
