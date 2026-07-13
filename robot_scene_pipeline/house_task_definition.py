"""Canonical six-role definition for the only supported build_house variant."""

from __future__ import annotations

from typing import Dict, List


HOUSE_STRUCTURE_VARIANT = "two_column_two_level_roof_triangle"
HOUSE_ROLE_IDS = (
    "left_support_lower",
    "right_support_lower",
    "left_support_upper",
    "right_support_upper",
    "roof",
    "triangle_top",
)
SUPPORT_ROLE_IDS = HOUSE_ROLE_IDS[:4]

HOUSE_DEFINITION_PROMPT = """“搭房子”固定表示：
- 两个正方形作为第一层左右支撑；
- 两个正方形分别叠放在左右第一层正方形上；
- 一个凹槽矩形或普通矩形横跨左右第二层正方形作为屋顶；
- 一个三角形放在屋顶中央，正确底面向下，目标锐角尖端朝 house_up；
- 三角形不得侧躺、倒置，也不得把直角顶点当作目标尖端；
- 凹槽屋顶的长平整边朝 house_up，凹槽开口朝 house_down；
- 凹槽屋顶错误面朝上时必须抓取并在安全高度通过 roll/pitch 翻面，仅 yaw 无效；
- 禁止解释成 wall、door、single_support 或任意自由结构。
合法角色只能是：left_support_lower、right_support_lower、left_support_upper、
right_support_upper、roof、triangle_top。"""


def canonical_house_goal_spec() -> dict:
    """Return a fresh immutable-task goal specification."""
    return {
        "structure_variant": HOUSE_STRUCTURE_VARIANT,
        "roles": [
            *[
                {
                    "role_id": role_id,
                    "requirements": {"shape": "square"},
                    "replaceable": True,
                }
                for role_id in SUPPORT_ROLE_IDS
            ],
            {
                "role_id": "roof",
                "requirements": {
                    "shape_any": ["concave_rectangle", "rectangle"],
                    "preferred_shape": "concave_rectangle",
                },
                "replaceable": True,
            },
            {
                "role_id": "triangle_top",
                "requirements": {"shape": "triangle"},
                "replaceable": True,
            },
        ],
        "required_relations": [
            {"type": "on_table", "subject_role": "left_support_lower"},
            {"type": "on_table", "subject_role": "right_support_lower"},
            {"type": "supports", "subject_role": "left_support_lower", "object_role": "left_support_upper"},
            {"type": "supports", "subject_role": "right_support_lower", "object_role": "right_support_upper"},
            {"type": "supports", "subject_role": "left_support_upper", "object_role": "roof"},
            {"type": "supports", "subject_role": "right_support_upper", "object_role": "roof"},
            {"type": "supports", "subject_role": "roof", "object_role": "triangle_top"},
        ],
    }


def canonical_house_assembly_steps() -> List[dict]:
    """Return the fixed dependency graph; same-level left/right may be interchanged."""
    return [
        {"step_id": "step_01_left_lower", "role_id": "left_support_lower", "prerequisites": []},
        {"step_id": "step_02_right_lower", "role_id": "right_support_lower", "prerequisites": []},
        {"step_id": "step_03_left_upper", "role_id": "left_support_upper", "prerequisites": ["left_support_lower"]},
        {"step_id": "step_04_right_upper", "role_id": "right_support_upper", "prerequisites": ["right_support_lower"]},
        {"step_id": "step_05_roof", "role_id": "roof", "prerequisites": ["left_support_upper", "right_support_upper"]},
        {"step_id": "step_06_triangle", "role_id": "triangle_top", "prerequisites": ["roof"]},
    ]


def assembly_prerequisites_by_role() -> Dict[str, List[str]]:
    """Map each role to prerequisite role ids."""
    return {
        step["role_id"]: list(step["prerequisites"])
        for step in canonical_house_assembly_steps()
    }
