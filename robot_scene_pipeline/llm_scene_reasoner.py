import json
import math

import requests

from .depth_geometry import coordinate_convention, public_object
from .io_utils import image_to_base64


def add_llm_args(parser):
    parser.add_argument("--model", default="qwen2.5vl:7b-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")


def build_llm_input(args, detections, snapshot_path, used_profile):
    return {
        "schema_version": "detector_3d_llm_input_v1",
        "instruction": args.instruction,
        "visual_context": {
            "snapshot_image": snapshot_path,
            "usage": "The attached original RGB image is global visual context. Use it to verify layout and appearance, but do not invent objects that are absent from hard_priors.objects.",
        },
        "camera_profile": used_profile,
        "hard_priors": {
            "description": "Detector boxes/classes/confidences and RealSense depth-derived 3D centers. Treat these as primary structured evidence for relation and action decisions.",
            "coordinate_convention": coordinate_convention(args.camera_frame),
            "base_frame": getattr(args, "base_frame", None),
            "objects": [public_object(det) for det in detections],
        },
    }


def build_prompt(llm_input):
    return """你是机器人场景理解和任务决策模块。

输入由两部分组成：
1. 原始 RGB 图片：作为全局上下文视觉特征，用来辅助核对场景布局、遮挡和外观。
2. Hard Prior 文本：目标检测类别、置信度、bbox、深度值、相机坐标和可选的机械臂 base 坐标。关系判断和动作目标必须优先依据这些硬先验。

空间关系规则：
- 只允许引用 hard_priors.objects 中已有的 object id，不要编造不存在的物体。
- 相机坐标中 x 更小表示更靠左，x 更大表示更靠右。
- 相机坐标中 z 更小表示更靠近相机/in_front_of，z 更大表示更远/behind。
- near/far 需要结合 3D 欧氏距离判断。
- 若存在 center_3d_base_m，动作目标优先使用 base 坐标；否则动作计划应保守，使用 ask_user 或 stop。
- on_surface 只有在有 workspace/table 类参照物，且图片上下文与 bbox/3D 坐标共同支持时才输出；不确定则输出 unknown。
- center_3d_m 为 null 或 coordinate_valid 为 false 的对象，关系只能保守判断，并把原因写入 uncertainties。

动作计划规则：
- move_named_pose 只能用于机器人预设姿态，例如 named_pose 为 ready、home、observe；禁止用 move_named_pose 表示“把物体放到左边/右边/前面/后面”。
- 对“把 A 放到 B 左边/右边/前面/后面/附近”这类物体操作，必须输出两步：
  1) pick，object_id=A，reference_object_id=null，relative_position=null
  2) place_relative，object_id=A，reference_object_id=B，relative_position=left_of|right_of|in_front_of|behind|near
- 对“移动到某物上方”输出 move_above。
- 如果目标物体、参考物体或 base 坐标缺失，输出 ask_user 或 stop，并把原因写入 uncertainties。
- target_position_3d_m 可为 null；后续执行模块会用 hard prior 中的 center_3d_base_m 和 relative_position 计算目标点。

请先生成场景图，再结合用户指令生成给后续执行模块使用的动作计划。
不要输出关节角、速度、底层电机命令或 ROS topic。若指令无法安全执行，使用 ask_user 或 stop。

请只输出严格 JSON，schema 如下：
{{
  "scene_graph": {{
    "objects": [{{"id": 0, "label": "object name", "confidence": 0.95}}],
    "relations": [
      {{
        "subject_id": 0,
        "predicate": "left_of|right_of|in_front_of|behind|near|far|on_surface|unknown",
        "object_id": 1,
        "reason": "short coordinate-based reason"
      }}
    ],
    "summary": "short scene summary"
  }},
  "action_plan": [
    {{
      "step": 1,
      "action": "move_named_pose|open_gripper|close_gripper|pick|place_relative|move_above|ask_user|stop",
      "object_id": null,
      "reference_object_id": null,
      "named_pose": null,
      "relative_position": "left_of|right_of|in_front_of|behind|on_surface|near|center_of_workspace|null",
      "target_position_3d_m": null,
      "reason": "short reason"
    }}
  ],
  "safety_checks": [],
  "uncertainties": []
}}

输入：
{}""".format(json.dumps(llm_input, ensure_ascii=False, indent=2))


def build_stack_blocks_prompt(llm_input):
    return """你是积木叠放任务的符号指令解析模块。

你只负责解析用户自然语言中的积木叠放顺序。对象必须来自 hard_priors.objects。
不要输出机器人坐标、抓取点、放置点、关节角、动作轨迹或完整多步坐标计划。
base_object_id 是最底层且不需要抓取的积木；stack_order 是从下到上依次需要抓取并放置的对象 id。
颜色和类别必须与 hard_priors.objects 的 label 严格一致；例如用户要求 green 时禁止选择 yellow。
base_object_id 禁止出现在 stack_order 中。缺少用户指定颜色时，不得用其他颜色替代。同色存在多个积木时，按有效 base_link geometry_center_m 的 x、y、id 升序选择。
若指令不明确，reason 必须说明歧义；仍然只输出以下严格 JSON schema：
{{
  "task_type": "stack_blocks",
  "base_object_id": 0,
  "stack_order": [1, 2],
  "reason": "short explanation of the parsed order"
}}

输入：
{}""".format(json.dumps(llm_input, ensure_ascii=False, indent=2))


def call_ollama(args, prompt, snapshot_path):
    message = {"role": "user", "content": prompt}
    if not args.no_image:
        message["images"] = [image_to_base64(snapshot_path)]
    payload = {
        "model": args.model,
        "messages": [message],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": args.num_predict},
    }
    response = requests.post(args.ollama_url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", json.dumps(data, ensure_ascii=False))


def parse_json_or_embedded(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


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
            rule_decision["reason"] = (
                "{} Overrode conflicting stack decision with the explicit instruction parse.".format(
                    rule_decision["reason"]
                )
            )
        return rule_decision
    normalized["decision_source"] = "validated_llm_or_explicit"
    return normalized


def remap_step_ids_from_instruction(step, hard_prior_objects, instruction):
    if not hard_prior_objects:
        return step, False
    colors = color_mentions(instruction or "")
    if not colors:
        return step, False

    changed = False
    step = dict(step)
    target_color = colors[0] if len(colors) >= 1 else None
    reference_color = colors[1] if len(colors) >= 2 else None

    if target_color and step.get("object_id") is not None:
        target_id = object_id_for_color(hard_prior_objects, target_color)
        if target_id is not None and int(step["object_id"]) != target_id:
            step["object_id"] = target_id
            changed = True

    if reference_color and step.get("reference_object_id") is not None:
        reference_id = object_id_for_color(hard_prior_objects, reference_color)
        if reference_id is not None and int(step["reference_object_id"]) != reference_id:
            step["reference_object_id"] = reference_id
            changed = True

    return step, changed


def normalize_scene_graph(decision, hard_prior_objects, instruction):
    scene_graph = decision.get("scene_graph")
    if not isinstance(scene_graph, dict) or not hard_prior_objects:
        return False

    scene_graph["objects"] = [
        {
            "id": int(obj["id"]),
            "label": obj.get("label"),
            "confidence": obj.get("confidence"),
        }
        for obj in hard_prior_objects
    ]

    colors = color_mentions(instruction or "")
    target_color = colors[0] if len(colors) >= 1 else None
    reference_color = colors[1] if len(colors) >= 2 else None
    target_id = object_id_for_color(hard_prior_objects, target_color) if target_color else None
    reference_id = object_id_for_color(hard_prior_objects, reference_color) if reference_color else None

    for relation in scene_graph.get("relations", []):
        if not isinstance(relation, dict):
            continue
        if target_id is not None:
            relation["subject_id"] = target_id
        if reference_id is not None:
            relation["object_id"] = reference_id
    return True


def normalize_decision_text(text, hard_prior_objects=None, instruction=""):
    """Fix common LLM action-schema mistakes before execution planning."""
    decision = parse_json_or_embedded(text)
    action_plan = decision.get("action_plan")
    if not isinstance(action_plan, list):
        if hard_prior_objects:
            normalize_scene_graph(decision, hard_prior_objects, instruction)
        return json.dumps(decision, ensure_ascii=False, indent=2)

    normalized = []
    changed = False
    relation_values = {"left_of", "right_of", "in_front_of", "behind", "on_surface", "near", "center_of_workspace"}
    valid_named_poses = {"ready", "home", "observe"}

    for step in action_plan:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        object_id = step.get("object_id")
        reference_id = step.get("reference_object_id")
        relative_position = step.get("relative_position")
        named_pose = step.get("named_pose")

        invalid_named_relative = (
            action == "move_named_pose"
            and object_id is not None
            and reference_id is not None
            and relative_position in relation_values
        )
        invalid_named_pose = action == "move_named_pose" and named_pose not in valid_named_poses

        if invalid_named_relative or invalid_named_pose:
            changed = True
            reason = step.get("reason", "")
            pick_step = {
                "step": len(normalized) + 1,
                "action": "pick",
                "object_id": object_id,
                "reference_object_id": None,
                "named_pose": None,
                "relative_position": None,
                "target_position_3d_m": None,
                "reason": "Pick target object before relative placement. {}".format(reason).strip(),
            }
            pick_step, remapped = remap_step_ids_from_instruction(pick_step, hard_prior_objects, instruction)
            changed = changed or remapped
            normalized.append(pick_step)

            place_step = {
                "step": len(normalized) + 1,
                "action": "place_relative",
                "object_id": object_id,
                "reference_object_id": reference_id,
                "named_pose": None,
                "relative_position": relative_position,
                "target_position_3d_m": None,
                "reason": "Place target object at requested relative position. {}".format(reason).strip(),
            }
            place_step, remapped = remap_step_ids_from_instruction(place_step, hard_prior_objects, instruction)
            changed = changed or remapped
            normalized.append(place_step)
            continue

        step = dict(step)
        step, remapped = remap_step_ids_from_instruction(step, hard_prior_objects, instruction)
        changed = changed or remapped
        step["step"] = len(normalized) + 1
        normalized.append(step)

    if changed:
        decision["action_plan"] = normalized
        decision.setdefault("uncertainties", [])
        if "Normalized invalid move_named_pose object-placement step into pick + place_relative." not in decision["uncertainties"]:
            decision["uncertainties"].append(
                "Normalized invalid move_named_pose object-placement step into pick + place_relative."
            )
        if hard_prior_objects:
            decision["uncertainties"].append(
                "Validated action object ids against detector hard priors."
            )
    if hard_prior_objects and normalize_scene_graph(decision, hard_prior_objects, instruction):
        decision.setdefault("uncertainties", [])
        decision["uncertainties"].append(
            "Validated scene graph object ids against detector hard priors."
        )
    return json.dumps(decision, ensure_ascii=False, indent=2)


def normalize_stack_blocks_text(
    text,
    hard_prior_objects=None,
    instruction="",
    prefer_explicit_rule=True,
):
    decision = parse_json_or_embedded(text)
    if decision.get("task_type") != "stack_blocks":
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
            return json.dumps(rule_decision, ensure_ascii=False, indent=2)
        raise ValueError("Stack decision must contain integer base_object_id and stack_order.")
    if base_object_id in stack_order or len(stack_order) != len(set(stack_order)):
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired non-unique LLM stack ids with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(rule_decision, ensure_ascii=False, indent=2)
        raise ValueError("Stack decision must contain unique ids and exclude the base from stack_order.")
    if valid_ids and ({base_object_id} | set(stack_order)) - valid_ids:
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired LLM ids absent from detector hard priors with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(rule_decision, ensure_ascii=False, indent=2)
        raise ValueError("Stack decision references object ids absent from detector hard priors.")
    if ({base_object_id} | set(stack_order)) & non_block_ids:
        if rule_decision is not None:
            rule_decision["reason"] = (
                "{} Repaired LLM workspace/non-block id selection with explicit instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(rule_decision, ensure_ascii=False, indent=2)
        raise ValueError("Stack decision must not use the workspace as a block.")
    normalized = {
        "task_type": "stack_blocks",
        "base_object_id": base_object_id,
        "stack_order": stack_order,
        "reason": str(decision.get("reason", "")),
    }
    if rule_decision is not None:
        if base_object_id != rule_decision["base_object_id"] or stack_order != rule_decision["stack_order"]:
            rule_decision["reason"] = (
                "{} Overrode conflicting LLM stack decision with deterministic instruction color parse.".format(
                    rule_decision["reason"]
                )
            )
            return json.dumps(rule_decision, ensure_ascii=False, indent=2)
        normalized["reason"] = "{} {}".format(normalized["reason"], rule_decision["reason"]).strip()
    return json.dumps(normalized, ensure_ascii=False, indent=2)
