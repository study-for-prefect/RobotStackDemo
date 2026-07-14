"""VLM policies for task-level contracts and temporary scene bindings."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .io_utils import image_to_base64
from .house_task_definition import HOUSE_DEFINITION_PROMPT, HOUSE_ROLE_IDS, canonical_house_goal_spec
from .ollama_policy_client import call_policy
from .organize_scope import color_value_from_object, organize_scope_objects
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
        "visual_color": obj.get("visual_color"),
        "visual_color_confidence": obj.get("visual_color_confidence"),
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
        "goal_progress": compact_goal_progress(goal_progress),
        "previous_plan": compact_grounded_task_plan(previous_plan),
        "failure_history": list(failure_history or []),
        "workspace_bounds": state.get("table_bounds") or state.get("workspace_bounds"),
        "semantics_config": (
            semantics_config
            if task_contract.get("task_type") == "build_house"
            else {"organize_defaults": semantics_config.get("organize_defaults", {})}
        ),
        "objects": [
            compact_task_object(obj) for obj in (
                organize_scope_objects(state)
                if task_contract.get("task_type") == "organize_blocks"
                else _scene_objects(state)
            )
        ],
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
    else:
        organize_objects = organize_scope_objects(state)
        value["objects"] = [compact_task_object(obj) for obj in organize_objects]
        value["layout_slot_candidates"] = _organize_layout_slots(
            state.get("organize_layout_bounds")
            or state.get("table_bounds")
            or state.get("workspace_bounds"),
            organize_objects,
        )
        value["organize_plan_rules"] = [
            "Create exactly one group for each resolved visual_color (fall back to detector label when absent).",
            "object_ids are current-scene detector_object_id values, not persistent identities.",
            "Assign every resolved-color block exactly once and never include a non-block false detection.",
            "Create exactly one target region per group and link it with target_region_id.",
            "Every region must be inside workspace_bounds and regions must not overlap.",
            "For rows, allocate separated horizontal bands: members share y within alignment_tolerance_m and spread along x.",
            "Size each region for all member footprints plus minimum_spacing_m; do not use the current group centroid as the region definition.",
            "Finish the bounded construction once; do not repeatedly reconsider what target_regions means.",
            "Target regions are future destinations and do not need to contain the objects at their current positions.",
            "Copy each layout_slot_candidates bounds exactly once for its assigned_color; do not change assignments or recalculate bounds.",
        ]
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
        "task_contract": task_contract,
        "grounded_task_plan": compact_grounded_task_plan(grounded_plan),
        "current_goal_progress": compact_goal_progress(goal_progress),
        "failure_history": _compact_action_failure_history(failure_history or []),
        "replanning_context": replanning_context or {},
        "workspace_bounds": state.get("table_bounds") or state.get("workspace_bounds"),
        "objects": [
            compact_task_object(obj) for obj in (
                organize_scope_objects(state)
                if task_contract.get("task_type") == "organize_blocks"
                else _scene_objects(state)
            )
        ],
        "output_schema": schema_for_policy("task_action", task_contract.get("task_type", "")),
    }
    if task_contract.get("task_type") == "build_house":
        value["build_house_definition"] = HOUSE_DEFINITION_PROMPT
    return value


def compact_goal_progress(goal_progress: Optional[dict]) -> dict:
    """Keep policy-relevant progress without the combinatorial assignment audit log."""
    if not isinstance(goal_progress, dict):
        return {}
    policy_keys = {
        "schema_version", "task_type", "scene_revision",
        "satisfied_predicates", "unsatisfied_predicates",
        "task_complete", "repair_required", "selected_role_assignment",
        "group_diagnostics", "scene_events",
    }
    compact = {key: value for key, value in goal_progress.items() if key in policy_keys}
    if compact.get("task_type") == "build_house":
        compact["eligible_role_ids_for_next_action"] = _eligible_house_action_roles(compact)
    return compact


def _compact_action_failure_history(history: List[dict], tail_count: int = 1) -> List[dict]:
    """Keep the latest corrections plus the latest physical grasp blockage."""
    values = [item for item in history if isinstance(item, dict)]
    selected = values[-max(1, int(tail_count)):]
    blocked = next((
        entry for entry in reversed(values)
        if any(
            isinstance(check, dict)
            and check.get("type") == "selected_object_grasp_feasible"
            and (check.get("all_grasps_blocked") or not check.get("grasp_feasible", True))
            for check in entry.get("failed_checks", [])
        )
    ), None)
    if blocked is not None and blocked not in selected:
        selected.insert(0, blocked)
    compact = []
    for entry in selected:
        output = {
            key: entry.get(key) for key in (
                "proposal_id", "scene_revision", "validation_stage", "passed",
                "rejected_action", "constraints_for_next_proposal",
            ) if key in entry
        }
        output["failed_checks"] = []
        for check in entry.get("failed_checks", []):
            if not isinstance(check, dict):
                continue
            output["failed_checks"].append({
                key: check.get(key) for key in (
                    "type", "reason", "all_grasps_blocked", "grasp_feasible",
                    "blocking_objects", "suggested_collision_free_position_m",
                    "suggested_interval_midpoint_position_m", "expected_object_center_base_m",
                    "expected_object_label", "colliding_object_id", "colliding_object_label",
                    "phase", "phases", "minimum_clearance_m",
                ) if key in check
            })
        compact.append(output)
    return compact


def _eligible_house_action_roles(progress: dict) -> List[str]:
    """Expose validator-derived role eligibility without choosing an action."""
    satisfied = set(progress.get("satisfied_predicates") or [])
    unsatisfied = set(progress.get("unsatisfied_predicates") or [])
    prerequisites = {
        "left_support_lower": set(),
        "right_support_lower": set(),
        "left_support_upper": {"left_support_lower.on_table"},
        "right_support_upper": {"right_support_lower.on_table"},
        "roof": {
            "left_support_lower.supports.left_support_upper",
            "right_support_lower.supports.right_support_upper",
            "left_column.vertical_aligned", "right_column.vertical_aligned",
            "columns.height_aligned",
        },
        "triangle_top": {
            "left_support_upper.supports.roof", "right_support_upper.supports.roof",
            "roof.bridges.upper_supports", "roof.correct_face_up", "roof.opening_down",
            "roof.straight_edge_up", "roof.orientation_correct",
        },
    }
    role_predicate_prefixes = {
        "left_support_lower": ("left_support_lower.",),
        "right_support_lower": ("right_support_lower.",),
        "left_support_upper": (
            "left_support_lower.supports.left_support_upper", "left_column.vertical_aligned",
        ),
        "right_support_upper": (
            "right_support_lower.supports.right_support_upper", "right_column.vertical_aligned",
        ),
        "roof": (
            "left_support_upper.supports.roof", "right_support_upper.supports.roof", "roof.",
        ),
        "triangle_top": ("roof.supports.triangle_top", "triangle_top."),
    }
    eligible = []
    for role_id, required in prerequisites.items():
        prefixes = role_predicate_prefixes[role_id]
        if role_id in {"left_support_lower", "right_support_lower"}:
            needs_work = "{}.on_table".format(role_id) in unsatisfied
        else:
            needs_work = any(any(item.startswith(prefix) for prefix in prefixes) for item in unsatisfied)
        if needs_work and required.issubset(satisfied):
            eligible.append(role_id)
    return eligible


def compact_grounded_task_plan(plan: Optional[dict]) -> dict:
    """Remove fixed or duplicated house-plan fields before the next policy call."""
    if not isinstance(plan, dict):
        return {}
    redundant = {
        "assembly_steps", "orientation_observations", "reason", "confidence",
    }
    compact = {key: value for key, value in plan.items() if key not in redundant}
    if compact.get("task_type") == "organize_blocks":
        compact["groups"] = [
            {key: value for key, value in group.items() if key != "object_ids"}
            for group in compact.get("groups", []) if isinstance(group, dict)
        ]
        compact["note"] = "previous detector object_ids were removed because they are stale after reobservation"
    return compact


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
    # The exact schema is already passed to Ollama through the structured
    # ``format`` field.  Repeating it in the natural-language prompt wastes
    # several thousand tokens and can crowd out physical failure feedback.
    text = {
        key: value for key, value in policy_input.items()
        if key not in ("scene_rgb", "minimal_overlay", "output_schema")
    }
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
            instruction = (
                "根据按颜色整理合同一次性输出当前场景分组和目标区域，不得输出任何房屋角色或房屋结构。"
                "严格执行 organize_plan_rules：按标签颜色分组；object_ids 直接复制当前 detector_object_id；"
                "在 workspace_bounds 内为每组分配互不重叠且容量足够的矩形。rows 表示每个颜色组各占一条水平带。"
                "target_regions 是未来放置区，不要求覆盖积木当前坐标。直接复制 layout_slot_candidates 的边界，"
                "严格使用每个 slot 的 assigned_color；禁止根据当前物体 y 坐标重排颜色或重算区域。"
            )
    else:
        recovery_directive = _task_action_recovery_directive(policy_input)
        instruction = (
            recovery_directive
            +
            "基于固定任务合同和当前未满足谓词输出一个动作。所有可执行动作必须使用 selected_object_ref 或 selected_track_id，禁止裸整数 object_id/selected_object_id。"
            "房子 pick_place 给 role_id；整理 pick_place 给 group_id 和 target_region_id；并给 scene_revision、准确 grounding 和目标位姿。"
            "object_label 和 object_center_base_m 必须逐值复制所选 objects 条目的 label 和 geometry_center_base_m，绝不能复制 target_pose_base。"
            "需要改变 roof/triangle 正反面时选择 pick_reorient_place；仅 yaw 不能代替翻面，具体轴角由代码计算。"
            "nudge/pick_away 也使用统一引用和完整物理参数，strategy_id 必须改成独立清障策略名（例如 clear_blocker_by_nudge），"
            "不能继续使用 organize_blocks。nudge 必须把被移动的阻挡物写入 selected_object_ref/selected_track_id，"
            "把原本受阻的抓取目标写入 target_object_ref/target_object_track_id，并逐值复制双方 label/center；还必须给出"
            "contact_side(+x|-x|+y|-y)、direction_base(三维单位 XY 向量)、distance_m(0.01~0.05) 和 gripper_yaw_rad，"
            "此时 target_pose_base 与 safe_place_center_base_m 必须为 null。pick_away 使用相同的双方引用并给出"
            "safe_place_center_base_m，此时 contact_side/direction_base/distance_m/gripper_yaw_rad/target_pose_base 必须为 null。"
            "禁止输出 replanning_context 中的失败指纹，不能只修改 reason、confidence 或小数尾数。"
            "nudge 只能用于桌面平面清障，不能形成 on_top_of；堆叠必须 pick_place。只能选择当前可见引用；不要自动宣称任务完成。"
        )
        if (policy_input.get("task_contract") or {}).get("task_type") == "build_house":
            instruction += (
                "严格按 current_goal_progress.satisfied_predicates 判断装配先决条件，角色已绑定不等于结构已完成。"
                "先放 left_support_lower/right_support_lower，再放各自 upper，之后 roof，最后 triangle_top；"
                "只选择其全部先决谓词已在 satisfied_predicates 中的角色，不得提前选择屋顶或三角形。"
                "role_id 必须从 current_goal_progress.eligible_role_ids_for_next_action 中选择；"
                "每个 pick_place/pick_reorient_place 都必须完整输出 selected_object_id、object_label、"
                "object_center_base_m、scene_revision 和 target_pose_base；target_pose_base 至少包含"
                "position_m:[x,y,z]，普通支撑块还必须包含 yaw_rad。它是结构中的放置中心，绝不能省略，"
                "也不能直接照抄源物体 object_center_base_m。放置 upper 时，目标 XY 等于其绑定 lower 的中心 XY，"
                "目标 Z=lower_center_z+(lower_height+upper_height)/2，尺寸从 objects.dimensions_m 获取。"
            )
        if (policy_input.get("task_contract") or {}).get("task_type") == "organize_blocks":
            instruction += (
                "最高优先级执行 physical_action_options：只要 direct_grasp_available=true，本次动作必须是 pick_place，"
                "selected_object_ref 必须从 direct_grasp_object_refs 中选择，严禁 nudge、pick_away、reobserve 或 stop。"
                "首选 recommended_direct_grasp_object_ref（它具有最大的可行抓取 yaw 区间）；只有失败历史已明确拒绝它时"
                "才选择 direct_grasp_object_refs 中的下一项。"
                "整理不按颜色顺序执行。抓取任一可抓的 outside_region 积木并放入它自己的同色行，既是整理推进也是清障；"
                "因此不要为了抓某个受阻目标而跳过另一块可抓的红/绿/蓝/黄积木。只有当 direct_grasp_available=false 时"
                "才进入清障：先检查阻挡物是否可抓；整理任务中可抓阻挡物仍用 pick_place 放入其同色 target_region，"
                "所有相关松散块都不可抓时才允许 nudge。"
                "换 selected_object_ref 时必须同时切换到该物体自身颜色的 group_id 和 target_region_id，"
                "禁止沿用上一个受阻物体的组或目标区域。"
                "清障可选择 blocking_objects 或同一散乱簇内任一 loose_movable 积木，优先 nudge 3~5cm 来创造夹爪角度；"
                "被推积木碰到其他未保护松散积木属于 recoverable contact，不得因此停止。推动终点必须避开所有"
                "target_regions 和已整理/保护区域，并在动作后重新观测。"
                "pick_place 必须完整输出 strategy_id、"
                "selected_object_ref/selected_track_id、group_id、target_region_id、object_label、object_center_base_m、"
                "scene_revision 和 target_pose_base。目标完整足迹不得与任何当前物体重叠；必须检查所有物体中心，"
                "不要机械地选择区域中心。失败反馈给出 blocking_object 时必须更换 XY。"
                "整理 pick_place 的 role_id、所有 target_object_*、contact_side、direction_base、distance_m、"
                "gripper_yaw_rad 和 safe_place_center_base_m 必须为 null；这些字段只属于房屋或清障动作。"
                "若某组 group_diagnostics.outside_region 非空，只能优先选择其中的物体，禁止搬动已经在目标区内的成员。"
                "target_pose_base 的 XY 必须与源中心有实质距离，禁止原地抓起再原地放下。"
                "每种颜色独占一个 target_region，同色积木必须在该区域内沿同一行并排且保持 minimum_spacing_m；"
                "放置 yaw 必须兼顾完整积木足迹、夹爪空间和碰撞约束；只在避碰需要时改变姿态，不固定旋转角度。"
                "若失败反馈给出 allowed_center_x_m/allowed_center_y_m，下一次 target_pose_base.position_m 的 XY"
                "必须直接选在这两个闭区间内，禁止再次复制源物体中心。"
            )
        if (policy_input.get("task_contract") or {}).get("task_type") == "build_house":
            instruction += (
                "清障也必须遵守 grasp-first：目标角色可抓就直接装配；目标受阻时，阻挡物可抓则先 pick_away，"
                "只有阻挡物也不可抓时才 nudge。"
            )
    ontology = "\n{}\n".format(HOUSE_DEFINITION_PROMPT) if (
        policy_input.get("expected_task_type") == "build_house"
        or (policy_input.get("task_contract") or {}).get("task_type") == "build_house"
    ) else "\n"
    return "你是 UR5 桌面积木任务语义规划器。{}\n{}\n只输出符合 output_schema 的 JSON。\n输入：\n{}".format(
        ontology, instruction, json.dumps(text, ensure_ascii=False, indent=2),
    )


def _task_action_recovery_directive(policy_input: dict) -> str:
    history = policy_input.get("failure_history") or []
    if not history:
        return ""
    latest = history[-1] if isinstance(history[-1], dict) else {}
    latest_checks = [item for item in latest.get("failed_checks", []) if isinstance(item, dict)]
    if any(item.get("type") == "organize_clearance_requires_physical_grasp_blockage" for item in latest_checks):
        return (
            "最高优先级动作类型纠错：当前没有任何物理抓取全角度受阻证据，禁止继续 nudge 或 pick_away。"
            "下一动作必须改为 pick_place，选择 current_goal_progress.group_diagnostics 中 outside_region 的积木，"
            "并在其同色 target_region 内给出与源中心至少相距0.006m的无碰撞 target_pose_base。"
            "只有后续 physical_grasp_preflight 明确返回 all_grasps_blocked=true 才允许清障。"
        )
    if any(item.get("type") == "push_direction_base_unit_xy_vector" for item in latest_checks):
        return (
            "最高优先级清障参数纠错：保持上次选择的阻挡物和 target_object 不变，但 direction_base 必须是长度为1的"
            "三维单位 XY 向量，不是位移量；只能例如 [1,0,0]、[-1,0,0]、[0,1,0] 或 [0,-1,0]。"
            "distance_m 单独填写[0.01,0.05]。contact_side 必须与方向相反：+X方向用-x，-X方向用+x，+Y方向用-y，-Y方向用+y。"
        )
    if any(item.get("type") == "contact_side_matches_push_direction" for item in latest_checks):
        return (
            "最高优先级清障接触侧纠错：保持上次阻挡物、target_object、单位 direction_base 和 distance_m；"
            "仅把 contact_side 改到方向反侧：+X=>-x，-X=>+x，+Y=>-y，-Y=>+y。"
        )
    if any(item.get("type") == "organize_clearance_must_increase_target_separation" for item in latest_checks):
        return (
            "最高优先级清障方向纠错：上次推动会让阻挡物更靠近原受阻目标，禁止重复。"
            "保持原受阻 target_object，并从当前 objects 重新选择阻挡物或方向；推动终点到 target_object 中心的"
            "XY距离必须比起点至少增加0.005m。可以改为相反方向，若夹爪空间不足则改用 pick_away。"
        )
    if any(item.get("type") == "organize_clearance_avoid_target_regions" for item in latest_checks):
        return (
            "最高优先级清障方向纠错：保持松散清障物和原受阻 target_object，改用另一个单位 XY 方向并推动0.03~0.05m；"
            "推动终点不得落入 grounded_task_plan.target_regions。优先把散乱物推离所有最终颜色行。"
        )
    entry_collision = next((
        item for item in latest_checks
        if item.get("type") == "tool_swept_volume_collision"
        and item.get("colliding_entity_type") == "movable_object"
        and any(
            phase in {"approach_to_contact", "contact_pose"}
            for phase in (item.get("phases") or [item.get("phase")])
        )
    ), None)
    if entry_collision is not None:
        return (
            "最高优先级清障进场纠错：上次夹爪侧推接触位被松散积木 id={!r}, label={!r} 占据，"
            "禁止从该积木正上方下降，也不能只修改 distance_m。必须换另一个 direction_base/contact_side，"
            "或换散乱簇边缘的 selected_object，使 pre_push 到 contact_pose 的垂直下降列完全无积木；"
            "只有 horizontal_push 阶段才允许与其他未保护松散积木发生 recoverable contact。"
        ).format(
            entry_collision.get("colliding_object_id"),
            entry_collision.get("colliding_object_label"),
        )
    invalid_pose = next((
        item for item in latest_checks
        if item.get("type") == "target_pose_base" and item.get("reason") == "invalid_target_pose_base"
    ), None)
    if invalid_pose is not None:
        return (
            "最高优先级位姿格式纠错：普通整理 pick_place 的 target_pose_base 必须同时包含"
            "position_m:[x,y,z] 和有限 yaw_rad（方块可先用0.0）；不得省略 yaw_rad。"
        )
    blocked_entry = None
    blocked_check = None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        blocked_check = next((
            item for item in entry.get("failed_checks", [])
            if isinstance(item, dict)
            and item.get("type") == "selected_object_grasp_feasible"
            and (item.get("all_grasps_blocked") or not item.get("grasp_feasible", True))
        ), None)
        if blocked_check is not None:
            blocked_entry = entry
            break
    if blocked_check is not None:
        rejected = (blocked_entry or {}).get("rejected_action") or {}
        blockers = blocked_check.get("blocking_objects") or []
        outside_ids = {
            str(object_id)
            for group in (policy_input.get("current_goal_progress") or {}).get("group_diagnostics", [])
            for object_id in group.get("outside_region", [])
            if object_id is not None
        }
        blocked_id = rejected.get("selected_object_id")
        return (
            "最高优先级抓取清障：物理夹爪扫描确认原目标 selected_object_ref={!r}, selected_track_id={!r}, "
            "label={!r}, center={} 在所有角度均被阻挡。禁止再次对它输出 pick_place。"
            "立即清障，不需要再逐个尝试其他颜色。可从 blocking_objects={} 或同一散乱簇中选择任一 loose_movable，并用输入 objects 中匹配 detector_object_id 的"
            "object_ref/track_id/label/geometry_center_base_m 作为 selected_*；原受阻目标必须作为 target_object_*。"
            "strategy_id 必须改为 clear_blocker_by_nudge（pick_away 则 clear_blocker_by_pick_away），禁止继续用 organize_blocks。"
            "优先输出完整 nudge：target_pose_base=null, safe_place_center_base_m=null, direction_base 为三维单位 XY 向量，"
            "distance_m 在[0.03,0.05]，contact_side 必须位于 direction_base 反方向，gripper_yaw_rad 为有限数。"
            "方向要远离 grounded_task_plan.target_regions；被推物体接触其他未保护松散积木是 recoverable contact，不得 stop。"
            "若无安全推移空间则输出完整 pick_away 和无碰撞 safe_place_center_base_m。"
        ).format(
            rejected.get("selected_object_ref"),
            rejected.get("selected_track_id"),
            rejected.get("object_label"),
            rejected.get("object_center_base_m"),
            blockers,
        )
    target_check = next((
        item for item in latest.get("failed_checks", [])
        if isinstance(item, dict) and item.get("type") == "target_pose_base"
    ), None)
    if target_check and target_check.get("suggested_collision_free_position_m"):
        return (
            "最高优先级碰撞纠错：上次 target_pose_base 与现有物体重叠。下一动作必须保持正确的区外目标物体，"
            "并把 target_pose_base.position_m 直接改为 geometry-checked position={}；"
            "绝不能再次使用 blocking_object_center_base_m 或 rejected_action 的旧位置。".format(
                target_check["suggested_collision_free_position_m"]
            )
        )
    if target_check and target_check.get("suggested_interval_midpoint_position_m"):
        return (
            "最高优先级位姿纠错：上次 target_pose_base 已被拒绝。下一动作必须改变 target_pose_base，"
            "先直接采用 suggested_interval_midpoint_position_m={}；绝不能再次输出 rejected_action 中的旧位置。"
            "该点仍会接受重叠、间距和布局安全校验。".format(
                target_check["suggested_interval_midpoint_position_m"]
            )
        )
    grounding_check = next((
        item for item in latest.get("failed_checks", [])
        if isinstance(item, dict) and item.get("type") in {"object_center_base_m", "object_label"}
    ), None)
    if grounding_check:
        if grounding_check.get("expected_object_center_base_m") is not None:
            return (
                "最高优先级 grounding 纠错：保持上次物理动作和 target_pose_base，但 "
                "object_center_base_m 必须逐值改为源物体中心 {}，绝不能填目标中心。".format(
                    grounding_check["expected_object_center_base_m"]
                )
            )
        if grounding_check.get("expected_object_label") is not None:
            return (
                "最高优先级 grounding 纠错：保持上次物理动作和 target_pose_base，但 "
                "object_label 必须逐字改为源物体标签 {!r}。".format(
                    grounding_check["expected_object_label"]
                )
            )
    return ""


def _scene_objects(state: dict) -> Iterable[dict]:
    return [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict) and not obj.get("is_workspace") and str(obj.get("label", "")).lower() != "workspace"
    ]


def _organize_layout_slots(workspace: object, objects: List[dict]) -> List[dict]:
    """Offer compact row slots in the part of the layout area farthest from clutter."""
    present = {color_value_from_object(obj) for obj in objects} - {None}
    canonical_order = ("red", "green", "blue", "yellow")
    colors = [color for color in canonical_order if color in present]
    if not isinstance(workspace, dict) or not colors:
        return []
    try:
        xmin, xmax = float(workspace["xmin"]), float(workspace["xmax"])
        ymin, ymax = float(workspace["ymin"]), float(workspace["ymax"])
    except (KeyError, TypeError, ValueError):
        return []
    available_height = ymax - ymin
    row_heights = []
    for color in colors:
        color_objects = [obj for obj in objects if color_value_from_object(obj) == color]
        footprint = max((
            max(float(value) for value in (obj.get("dimensions_m") or [])[:2])
            for obj in color_objects
            if len(obj.get("dimensions_m") or []) >= 2
        ), default=0.022)
        row_heights.append(max(0.03, footprint + 0.008))
    total_height = sum(row_heights)
    if total_height > available_height:
        row_heights = [available_height / len(colors)] * len(colors)
        total_height = available_height

    # Candidate windows include both edges and positions just beyond each
    # observed footprint.  Maximize the minimum vertical gap so destinations
    # do not begin inside the current scattered cluster.
    starts = [ymin, ymax - total_height]
    obstacle_intervals = []
    for obj in objects:
        center = obj.get("geometry_center_m") or obj.get("center_3d_base_m")
        size = obj.get("dimensions_m") or []
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        half_y = 0.5 * float(size[1]) if len(size) >= 2 else 0.012
        lower, upper = float(center[1]) - half_y, float(center[1]) + half_y
        obstacle_intervals.append((lower, upper))
        starts.extend((upper + 0.025, lower - 0.025 - total_height))
    starts = [max(ymin, min(ymax - total_height, value)) for value in starts]

    def clearance(start: float) -> float:
        end = start + total_height
        gaps = []
        for lower, upper in obstacle_intervals:
            if upper < start:
                gaps.append(start - upper)
            elif lower > end:
                gaps.append(lower - end)
            else:
                gaps.append(-min(end, upper) + max(start, lower))
        return min(gaps) if gaps else available_height

    start_y = max(starts, key=lambda value: (clearance(value), value))
    slots = []
    cursor = start_y
    for index, (color, height) in enumerate(zip(colors, row_heights)):
        slots.append({
            "slot_id": "row_slot_{:02d}".format(index + 1),
            "assigned_color": color,
            "bounds_base_m": {
                "xmin": round(xmin, 6), "xmax": round(xmax, 6),
                "ymin": round(cursor, 6), "ymax": round(cursor + height, 6),
            },
            "selection_reason": "free_band_farthest_from_current_scattered_footprints",
        })
        cursor += height
    return slots


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
