import json
import math


def build_stack_blocks_prompt(llm_input):
    return """你是机器人积木任务的大脑：负责理解任务、分解结构、分配物体角色，并给当前执行器一个安全的可执行顺序。

输入包含用户指令、RGB 图像和 hard_priors.objects。只能选择 hard_priors.objects 中已有的 object id，不要编造物体。

你的职责：
1. 理解目标结构，例如普通叠放、房子、桥、塔、门、平台等。
2. 根据物体 label、尺寸 dimensions_m、位置 geometry_center_m、置信度和形状，分配角色：
   - base/foundation/support/pillar/wall/beam/roof/top/decorative/unused 等。
3. 输出结构计划 structure_plan，说明为什么选这些物体、哪些约束重要、当前执行器有什么限制。
4. 同时给出现有执行器可执行的 base_object_id 和 stack_order：
   - base_object_id 是先作为底部参照、不在本轮抓取顺序里的物体。
   - stack_order 是从下到上依次抓取并放到当前顶部的对象 id。
   - 如果目标是桥或房子但当前执行器只支持竖直叠放，就输出最接近且安全的 vertical_stack_approximation，并在 limitations 里说明缺少 side-by-side/跨放能力。
5. 失败后重规划时，应优先说明：换哪个同类物体、先清哪个障碍、是否降低目标结构复杂度，或为什么必须人工介入。

安全和一致性规则：
- 颜色和类别必须与 hard_priors.objects 的 label 严格一致；用户要求 green 时禁止选择 yellow。
- base_object_id 禁止出现在 stack_order 中，stack_order 内 id 不得重复。
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
    "roles": [
      {{"role": "base|pillar|wall|beam|roof|top|unused", "object_id": 0, "reason": "why this object fits"}}
    ],
    "assembly_steps": [
      {{"step": 1, "intent": "place roof on walls", "object_id": 2, "target_role": "top"}}
    ],
    "limitations": [],
    "replan_policy": []
  }},
  "base_object_id": 0,
  "stack_order": [1, 2],
  "reason": "short explanation of the parsed order"
}}

输入：
{}""".format(json.dumps(llm_input, ensure_ascii=False, indent=2))


COLOR_ALIASES = [
    ("green", ("green", "绿色", "绿")),
    ("red", ("red", "红色", "红")),
    ("blue", ("blue", "蓝色", "蓝")),
    ("yellow", ("yellow", "黄色", "黄")),
]


def color_mentions(text):
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


def object_label_contains(obj, color):
    label = str(obj.get("label", "")).lower()
    aliases = next((values for name, values in COLOR_ALIASES if name == color), (color,))
    return any(str(alias).lower() in label for alias in aliases)


def object_id_for_color(objects, color):
    matches = [obj for obj in objects if object_label_contains(obj, color)]
    if not matches:
        return None
    return int(max(matches, key=lambda item: item.get("confidence", 0.0))["id"])


def _parse_json_or_embedded(text):
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


def select_stack_object_for_color(objects, color):
    """Select one color match deterministically, using only stack-safe geometry for ambiguity."""
    matches = [
        obj
        for obj in objects
        if not obj.get("is_workspace")
        and str(obj.get("label", "")).lower() != "workspace"
        and object_label_contains(obj, color)
    ]
    if not matches:
        raise ValueError(
            "Color-rule stack instruction requires exactly one {} block, found 0.".format(color)
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
        raise ValueError(
            "Color-rule stack instruction found {} {} blocks, but none has valid base_link "
            "geometry_center_m for deterministic x/y selection.".format(len(matches), color)
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


def rule_stack_blocks_decision(instruction, hard_prior_objects):
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
    return {
        "task_type": "stack_blocks",
        "base_object_id": color_ids[base_color],
        "stack_order": [color_ids[color] for color in ordered_colors],
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
    decision,
    hard_prior_objects,
    instruction="",
    prefer_explicit_rule=True,
):
    normalized = json.loads(
        normalize_stack_blocks_text(
            json.dumps(decision),
            hard_prior_objects,
            instruction,
            prefer_explicit_rule=prefer_explicit_rule,
        )
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


def _with_structure_extras(normalized, decision):
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


def _structure_plan_ids(decision):
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


def normalize_stack_blocks_text(
    text,
    hard_prior_objects=None,
    instruction="",
    prefer_explicit_rule=True,
):
    decision = _parse_json_or_embedded(text)
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
        base_object_id = int(decision["base_object_id"])
        stack_order = [int(value) for value in decision["stack_order"]]
    except (KeyError, TypeError, ValueError):
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired malformed LLM stack decision with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(_with_structure_extras(rule_decision, decision), ensure_ascii=False, indent=2)
        raise ValueError("Stack decision must contain integer base_object_id and stack_order.")
    if base_object_id in stack_order or len(stack_order) != len(set(stack_order)):
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired non-unique LLM stack ids with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(_with_structure_extras(rule_decision, decision), ensure_ascii=False, indent=2)
        raise ValueError("Stack decision must contain unique ids and exclude the base from stack_order.")
    if valid_ids and ({base_object_id} | set(stack_order)) - valid_ids:
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired LLM ids absent from detector hard priors with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(_with_structure_extras(rule_decision, decision), ensure_ascii=False, indent=2)
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
            return json.dumps(_with_structure_extras(rule_decision, decision), ensure_ascii=False, indent=2)
        raise ValueError("Stack decision must not use the workspace as a block.")
    normalized = {
        "task_type": "stack_blocks",
        "base_object_id": base_object_id,
        "stack_order": stack_order,
        "reason": str(decision.get("reason", "")),
    }
    normalized = _with_structure_extras(normalized, decision)
    if rule_decision is not None:
        if base_object_id != rule_decision["base_object_id"] or stack_order != rule_decision["stack_order"]:
            normalized["base_object_id"] = rule_decision["base_object_id"]
            normalized["stack_order"] = rule_decision["stack_order"]
            normalized["reason"] = (
                "{} Overrode conflicting LLM stack decision with deterministic instruction color parse.".format(
                    normalized.get("reason") or rule_decision["reason"]
                )
            )
            normalized["explicit_rule_repair"] = rule_decision
            return json.dumps(normalized, ensure_ascii=False, indent=2)
        normalized["reason"] = "{} {}".format(normalized["reason"], rule_decision["reason"]).strip()
    return json.dumps(normalized, ensure_ascii=False, indent=2)
