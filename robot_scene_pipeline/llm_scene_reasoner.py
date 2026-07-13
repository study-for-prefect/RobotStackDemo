import json
import os

from .depth_geometry import coordinate_convention, public_object
from .io_utils import image_to_base64
from .ollama_policy_client import call_policy
from .llm_stack_blocks import (
    StackColorSelectionError,
    build_stack_blocks_prompt,
    color_mentions,
    normalize_stack_blocks_decision,
    normalize_stack_blocks_text,
    object_id_for_color,
    rule_stack_blocks_decision,
    stack_color_requirement_report,
    validate_stack_blocks_decision,
)


def add_llm_args(parser):
    parser.add_argument("--model", default="qwen2.5vl:7b-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--vlm-think-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--vlm-num-ctx", type=int, default=0)
    parser.add_argument("--vlm-num-predict", type=int, default=0)
    parser.add_argument("--vlm-finalizer-num-predict", type=int, default=4096)
    parser.add_argument("--vlm-read-timeout-sec", type=float, default=1200.0)
    parser.add_argument("--vlm-keep-alive", default="1h")
    parser.add_argument("--vlm-max-backend-retries", type=int, default=3)
    parser.add_argument("--vlm-max-budget-retries", type=int, default=3)
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


def call_ollama(args, prompt, snapshot_path):
    message = {"role": "user", "content": prompt}
    if not args.no_image:
        message["images"] = [image_to_base64(snapshot_path)]
    result = call_policy(
        args, "orientation_analysis", [message], {"type": "object"},
        artifact_dir=os.path.dirname(snapshot_path), temperature=0.1,
    )
    if result.parsed_decision is None:
        raise RuntimeError("VLM_BACKEND_FAILED: {}".format(result.error_type or result.generation_status))
    return json.dumps(result.parsed_decision, ensure_ascii=False)


def parse_json_or_embedded(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


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
