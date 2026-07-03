import json
import math
from typing import Any, Dict, Iterable, List, Optional


def build_stack_blocks_prompt(llm_input):
    return """你是机器人积木任务的大脑：负责理解任务、分解结构、分配物体角色，并给当前执行器一个安全的可执行顺序。

输入包含用户指令、RGB 图像和 hard_priors.objects。只能选择 hard_priors.objects 中已有的 object id，不要编造物体。

你的职责：
1. 理解普通叠放、房子、桥、塔、门、平台等目标结构。
2. 根据 label、dimensions_m、geometry_center_m、置信度和形状分配 base/pillar/wall/beam/roof/top/unused 等角色。
3. 输出 structure_plan，说明物体选择、约束、执行器限制和失败后重规划策略。
4. 输出 full_stack_order=[base,...]、base_object_id，以及执行器可抓取的 stack_order（不含 base_object_id）。
5. 若当前执行器只能竖直叠放，桥/房子需输出 vertical_stack_approximation 并在 limitations 说明缺少 side-by-side/跨放能力。

安全和一致性规则：
- 颜色和类别必须与 hard_priors.objects 的 label 严格一致；用户要求 green 时禁止选择 yellow。
- 如果你自然地把完整结构写成 stack_order=[base,...]，系统会兼容；但推荐明确写 full_stack_order=[base,...] 且 stack_order 不含 base。
- base_object_id 禁止出现在推荐的 stack_order 中，stack_order 内 id 不得重复。
- 不要输出机器人坐标、抓取点、关节角或底层运动命令。
- 若选择桥/房子等非纯竖叠结构，必须仍然提供当前执行器可跑的降级顺序，并明确 limitations。
- 同色多个候选时，可结合几何和任务角色选择，而不是只能按 id；但 reason 里必须解释。

请只输出严格 JSON，schema 如下：
{{
  "task_type": "stack_blocks",
  "structure_plan": {{
    "structure_type": "tower|house|bridge|platform|custom",
    "user_goal": "short restatement",
    "execution_strategy": "vertical_stack|vertical_stack_approximation|needs_side_by_side_capability",
    "roles": [{{"role": "base|pillar|wall|beam|roof|top|unused", "object_id": 0, "reason": "why"}}],
    "assembly_steps": [{{"step": 1, "intent": "place roof on walls", "object_id": 2, "target_role": "top"}}],
    "limitations": [],
    "replan_policy": []
  }},
  "base_object_id": 0,
  "full_stack_order": [0, 1, 2],
  "stack_order": [1, 2],
  "stack_order_semantics": "place_order_excludes_base",
  "reason": "short explanation"
}}

输入：
{}""".format(json.dumps(llm_input, ensure_ascii=False, indent=2))


COLOR_ALIASES = [
    ("green", ("green", "绿色", "绿")), ("red", ("red", "红色", "红")),
    ("blue", ("blue", "蓝色", "蓝")), ("yellow", ("yellow", "黄色", "黄")),
]


class StackColorSelectionError(ValueError):
    def __init__(self, color: str, reason: str, message: str):
        super().__init__(message)
        self.color = color
        self.reason = reason

def color_mentions(text: str) -> List[str]:
    mentions = []
    lowered = text.lower()
    for color, aliases in COLOR_ALIASES:
        positions = []
        for alias in aliases:
            pos = lowered.find(alias.lower())
            if pos >= 0:
                positions.append(pos)
        if positions:
            mentions.append((min(positions), color))
    return [color for _, color in sorted(mentions)]


def object_label_contains(obj: Dict[str, Any], color: str) -> bool:
    label = str(obj.get("label", "")).lower()
    aliases = next((values for name, values in COLOR_ALIASES if name == color), (color,))
    return any(str(alias).lower() in label for alias in aliases)


def object_id_for_color(objects: Iterable[Dict[str, Any]], color: str) -> Optional[int]:
    matches = [obj for obj in objects if object_label_contains(obj, color)]
    if not matches:
        return None
    return int(max(matches, key=lambda item: item.get("confidence", 0.0))["id"])


def _parse_json_or_embedded(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _stack_candidate_xy(obj):
    center = obj.get("geometry_center_m")
    if (
        obj.get("geometry_frame") != "base_link"
        or not obj.get("pointcloud_geometry_valid", True)
        or not isinstance(center, (list, tuple))
        or len(center) < 2
    ):
        return None
    try:
        xy = (float(center[0]), float(center[1]))
    except (TypeError, ValueError):
        return None
    return xy if all(math.isfinite(value) for value in xy) else None


def select_stack_object_for_color(objects: Iterable[Dict[str, Any]], color: str):
    """Select one color match deterministically, using only stack-safe geometry for ambiguity."""
    matches = [
        obj
        for obj in objects
        if not obj.get("is_workspace")
        and str(obj.get("label", "")).lower() != "workspace"
        and object_label_contains(obj, color)
    ]
    if not matches:
        raise StackColorSelectionError(
            color,
            "missing_required_color",
            "Color-rule stack instruction requires a {} block, but none was detected.".format(color),
        )
    if len(matches) == 1:
        eligible_ids = [int(matches[0]["id"])] if _stack_candidate_xy(matches[0]) is not None else []
        return matches[0], {
            "color": color,
            "candidate_ids": [int(matches[0]["id"])],
            "eligible_ids": eligible_ids,
            "selected_id": int(matches[0]["id"]),
            "strategy": "only_color_match",
        }

    eligible = [(obj, _stack_candidate_xy(obj)) for obj in matches]
    eligible = [(obj, xy) for obj, xy in eligible if xy is not None]
    if not eligible:
        raise StackColorSelectionError(
            color,
            "ambiguous_color_without_geometry",
            "Color-rule stack instruction found {} {} blocks, but none has valid base_link "
            "geometry_center_m for deterministic x/y selection.".format(len(matches), color),
        )
    selected, selected_xy = min(
        eligible,
        key=lambda item: (item[1][0], item[1][1], int(item[0]["id"])),
    )
    return selected, {
        "color": color,
        "candidate_ids": sorted(int(obj["id"]) for obj in matches),
        "eligible_ids": sorted(int(obj["id"]) for obj, _ in eligible),
        "selected_id": int(selected["id"]),
        "selected_xy_base_m": [selected_xy[0], selected_xy[1]],
        "strategy": "min_base_link_geometry_x_then_y_then_id",
    }


BASE_ROLE_WORDS = ("底", "底座", "基底", "底层", "底部", "最底层", "最下面", "base", "bottom")
BOTTOM_TO_TOP_MARKERS = ("从下到上", "自下而上", "由下到上", "bottom to top")
ON_TOP_MARKERS = ("放到", "放在", "叠到", "叠在", "堆到", "堆在", "摞到", "摞在", "on top of", "onto", "stack")


def _explicit_base_color(text):
    for color, aliases in COLOR_ALIASES:
        for alias in aliases:
            alias = str(alias).lower()
            alias_forms = (alias, "{}方块".format(alias), "{}积木".format(alias), "{}块".format(alias))
            for alias_form in alias_forms:
                for role_word in BASE_ROLE_WORDS:
                    if (
                        "以{}为{}".format(alias_form, role_word) in text
                        or "{}为{}".format(alias_form, role_word) in text
                        or "{}作为{}".format(alias_form, role_word) in text
                        or "用{}作为{}".format(alias_form, role_word) in text
                        or "把{}作为{}".format(alias_form, role_word) in text
                        or "{}当{}".format(alias_form, role_word) in text
                        or "{} as {}".format(alias_form, role_word) in text
                        or "{} is {}".format(alias_form, role_word) in text
                    ):
                        return color
    return None


def _implied_stack_order_from_instruction(text, mentioned):
    if len(mentioned) < 2:
        return None, None, None
    if any(marker in text for marker in BOTTOM_TO_TOP_MARKERS):
        return mentioned[0], mentioned[1:], "bottom_to_top_order"
    if any(marker in text for marker in ON_TOP_MARKERS) and ("上" in text or "top" in text or "onto" in text):
        return mentioned[1], [mentioned[0]] + mentioned[2:], "target_on_reference_order"
    return None, None, None


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def stack_color_requirement_report(instruction: str, hard_prior_objects: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    required_colors = _unique_preserve_order(color_mentions(instruction or ""))
    observed = []
    missing = []
    objects = list(hard_prior_objects or [])
    for obj in objects:
        observed.append({
            "id": obj.get("id"), "label": obj.get("label"),
            "label_id": obj.get("label_id"), "class_id": obj.get("class_id"),
            "confidence": obj.get("confidence", obj.get("score")),
        })
    for color in required_colors:
        if object_id_for_color(objects, color) is None:
            missing.append(color)
    return {
        "schema_version": "initial_required_objects_report_v1",
        "required_colors": required_colors,
        "missing_colors": missing,
        "observed_objects": observed,
        "id_semantics": {
            "id": "per-snapshot object instance id used by planning and execution; unique only inside one scene state.",
            "label": "detector class name, for example square green.",
            "label_id": "YOLO class id; shared by all detections of the same class, not an object instance id.",
            "class_id": "YOLO class id alias; may be null in public hard-prior JSON.",
        },
    }


def rule_stack_blocks_decision(instruction: str, hard_prior_objects: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Resolve color-specified stack instructions without LLM freedom."""
    text = str(instruction or "").lower()
    mentioned = color_mentions(text)
    if not mentioned:
        return None

    base_color = _explicit_base_color(text)
    decision_rule = "explicit_base_role"
    if base_color is None:
        base_color, ordered_colors, decision_rule = _implied_stack_order_from_instruction(text, mentioned)
        if base_color is None:
            return None
    else:
        ordered_colors = [color for color in mentioned if color != base_color]

    if not ordered_colors:
        raise ValueError("Color-rule stack instruction has no block to place above the base.")
    color_ids = {}
    selections = []
    for color in [base_color] + ordered_colors:
        selected, selection = select_stack_object_for_color(hard_prior_objects, color)
        color_ids[color] = int(selected["id"])
        selections.append(selection)
    full_stack_order = [color_ids[base_color]] + [color_ids[color] for color in ordered_colors]
    place_order = full_stack_order[1:]
    return {
        "task_type": "stack_blocks",
        "base_object_id": color_ids[base_color],
        "full_stack_order": full_stack_order,
        "stack_order": place_order,
        "stack_order_semantics": "place_order_excludes_base",
        "reason": "Deterministic color-rule parse ({}): base={} stack_order={}.".format(
            decision_rule, base_color, ordered_colors
        ),
        "base_color": base_color,
        "stack_colors": ordered_colors,
        "decision_source": "instruction_color_rule",
        "instruction_parse_rule": decision_rule,
        "color_candidate_selections": selections,
    }


def validate_stack_blocks_decision(
    decision: Any,
    hard_prior_objects: Iterable[Dict[str, Any]],
    instruction: str = "",
    prefer_explicit_rule: bool = True,
) -> Dict[str, Any]:
    normalized = normalize_stack_blocks_decision(
        decision,
        hard_prior_objects,
        instruction,
        prefer_explicit_rule=prefer_explicit_rule,
    )
    rule_decision = (
        rule_stack_blocks_decision(instruction, hard_prior_objects)
        if prefer_explicit_rule
        else None
    )
    if rule_decision is not None:
        if (
            normalized["base_object_id"] != rule_decision["base_object_id"]
            or normalized["stack_order"] != rule_decision["stack_order"]
        ):
            normalized["base_object_id"] = rule_decision["base_object_id"]
            normalized["stack_order"] = rule_decision["stack_order"]
            normalized["full_stack_order"] = rule_decision["full_stack_order"]
            normalized["stack_order_semantics"] = "place_order_excludes_base"
            normalized["reason"] = (
                "{} Overrode conflicting stack decision with the explicit instruction parse.".format(
                    normalized.get("reason") or rule_decision["reason"]
                )
            )
            normalized["decision_source"] = "validated_llm_with_explicit_rule_ids"
            normalized["explicit_rule_repair"] = rule_decision
        return normalized
    normalized["decision_source"] = "validated_llm_or_explicit"
    return normalized


def _with_structure_extras(normalized: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(normalized)
    for key in ("structure_plan", "role_assignments", "assembly_steps", "limitations", "replan_policy"):
        if key in decision:
            output[key] = decision[key]
    plan = output.get("structure_plan")
    if isinstance(plan, dict):
        plan = dict(plan)
        if "roles" not in plan and isinstance(output.get("role_assignments"), list):
            plan["roles"] = output["role_assignments"]
        if "assembly_steps" not in plan and isinstance(output.get("assembly_steps"), list):
            plan["assembly_steps"] = output["assembly_steps"]
        if "replan_policy" not in plan and isinstance(output.get("replan_policy"), list):
            plan["replan_policy"] = output["replan_policy"]
        output["structure_plan"] = plan
    return output


def _structure_plan_ids(decision: Dict[str, Any]) -> set:
    ids = set()
    plan = decision.get("structure_plan")
    containers = []
    if isinstance(plan, dict):
        containers.extend(plan.get("roles", []) or [])
        containers.extend(plan.get("assembly_steps", []) or [])
    containers.extend(decision.get("role_assignments", []) or [])
    containers.extend(decision.get("assembly_steps", []) or [])
    for item in containers:
        if not isinstance(item, dict):
            continue
        for key in ("object_id", "reference_object_id"):
            value = item.get(key)
            if value is not None:
                ids.add(int(value))
        values = item.get("object_ids")
        if isinstance(values, list):
            ids.update(int(value) for value in values if value is not None)
    return ids


def _decision_from_any(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return _parse_json_or_embedded(value)
    raise ValueError("Stack decision must be a dict or JSON text.")


def _normalize_orders(decision: Dict[str, Any]) -> tuple:
    try:
        base_object_id = int(decision["base_object_id"])
    except (KeyError, TypeError, ValueError):
        full_order = decision.get("full_stack_order")
        if isinstance(full_order, list) and full_order:
            return int(full_order[0]), [int(value) for value in full_order[1:]], [int(value) for value in full_order]
        raise

    raw_order = decision.get("stack_order")
    full_order = decision.get("full_stack_order")
    if raw_order is None and full_order is None:
        raise KeyError("stack_order")
    if raw_order is None:
        normalized_full = [int(value) for value in full_order]
        if not normalized_full or normalized_full[0] != base_object_id:
            raise ValueError("full_stack_order must start with base_object_id.")
        return base_object_id, normalized_full[1:], normalized_full

    raw_place_order = [int(value) for value in raw_order]
    if raw_place_order and raw_place_order[0] == base_object_id:
        if len(raw_place_order) != len(set(raw_place_order)):
            raise ValueError("Stack decision must contain unique ids.")
        return base_object_id, raw_place_order[1:], raw_place_order
    if base_object_id in raw_place_order:
        raise ValueError("base_object_id may only appear as the first item of a full stack order.")
    if full_order is not None:
        normalized_full = [int(value) for value in full_order]
        if not normalized_full or normalized_full[0] != base_object_id:
            raise ValueError("full_stack_order must start with base_object_id.")
    else:
        normalized_full = [base_object_id] + raw_place_order
    return base_object_id, raw_place_order, normalized_full


def normalize_stack_blocks_decision(
    decision_or_text: Any,
    hard_prior_objects: Optional[Iterable[Dict[str, Any]]] = None,
    instruction: str = "",
    prefer_explicit_rule: bool = True,
) -> Dict[str, Any]:
    decision = _decision_from_any(decision_or_text)
    if decision.get("task_type") not in ("stack_blocks", "structure_plan"):
        raise ValueError("Stack decision task_type must be stack_blocks.")
    rule_decision = None
    if instruction and prefer_explicit_rule:
        rule_decision = rule_stack_blocks_decision(instruction, hard_prior_objects or [])
    valid_ids = {int(obj["id"]) for obj in (hard_prior_objects or [])}
    non_block_ids = {
        int(obj["id"])
        for obj in (hard_prior_objects or [])
        if obj.get("is_workspace") or str(obj.get("label", "")).lower() == "workspace"
    }
    try:
        base_object_id, stack_order, full_stack_order = _normalize_orders(decision)
    except (KeyError, TypeError, ValueError):
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired malformed LLM stack decision with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return _with_structure_extras(rule_decision, decision)
        raise ValueError("Stack decision must contain integer base_object_id and stack_order.")
    if base_object_id in stack_order or len(stack_order) != len(set(stack_order)) or len(full_stack_order) != len(set(full_stack_order)):
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired non-unique LLM stack ids with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return _with_structure_extras(rule_decision, decision)
        raise ValueError("Stack decision must contain unique ids and exclude the base from stack_order.")
    if valid_ids and set(full_stack_order) - valid_ids:
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired LLM ids absent from detector hard priors with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return _with_structure_extras(rule_decision, decision)
        raise ValueError("Stack decision references object ids absent from detector hard priors.")
    structure_ids = _structure_plan_ids(decision)
    if valid_ids and structure_ids - valid_ids:
        raise ValueError("Structure plan references object ids absent from detector hard priors.")
    if ({base_object_id} | set(stack_order)) & non_block_ids:
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired LLM workspace/non-block id selection with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return _with_structure_extras(rule_decision, decision)
        raise ValueError("Stack decision must not use the workspace as a block.")
    normalized = {
        "task_type": "stack_blocks",
        "base_object_id": base_object_id,
        "full_stack_order": full_stack_order,
        "stack_order": stack_order,
        "stack_order_semantics": "place_order_excludes_base",
        "reason": str(decision.get("reason", "")),
    }
    normalized = _with_structure_extras(normalized, decision)
    if rule_decision is not None:
        if base_object_id != rule_decision["base_object_id"] or stack_order != rule_decision["stack_order"]:
            normalized["base_object_id"] = rule_decision["base_object_id"]
            normalized["stack_order"] = rule_decision["stack_order"]
            normalized["full_stack_order"] = rule_decision["full_stack_order"]
            normalized["stack_order_semantics"] = "place_order_excludes_base"
            normalized["reason"] = (
                "{} Overrode conflicting LLM stack decision with deterministic instruction color parse.".format(
                    normalized.get("reason") or rule_decision["reason"]
                )
            )
            normalized["explicit_rule_repair"] = rule_decision
            return normalized
        normalized["reason"] = "{} {}".format(normalized["reason"], rule_decision["reason"]).strip()
    return normalized


def normalize_stack_blocks_text(
    text: str,
    hard_prior_objects: Optional[Iterable[Dict[str, Any]]] = None,
    instruction: str = "",
    prefer_explicit_rule: bool = True,
) -> str:
    normalized = normalize_stack_blocks_decision(
        text,
        hard_prior_objects=hard_prior_objects,
        instruction=instruction,
        prefer_explicit_rule=prefer_explicit_rule,
    )
    return json.dumps(normalized, ensure_ascii=False, indent=2)
