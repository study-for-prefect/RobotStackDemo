#!/usr/bin/env python3

import json

from robot_scene_pipeline.llm_scene_reasoner import normalize_stack_blocks_text, validate_stack_blocks_decision


OBJECTS = [
    {"id": 0, "label": "square red", "confidence": 0.99, "geometry_center_m": [0.30, 0.10, 0.0]},
    {"id": 1, "label": "rectangle blue", "confidence": 0.98, "geometry_center_m": [0.35, 0.10, 0.0]},
    {"id": 2, "label": "triangle yellow", "confidence": 0.97, "geometry_center_m": [0.40, 0.10, 0.0]},
]


def test_structure_plan_is_preserved_with_executable_stack_order():
    raw = {
        "task_type": "stack_blocks",
        "structure_plan": {
            "structure_type": "house",
            "user_goal": "build a small house",
            "execution_strategy": "vertical_stack_approximation",
            "roles": [
                {"role": "foundation", "object_id": 0, "reason": "stable square base"},
                {"role": "wall", "object_id": 1, "reason": "taller middle block"},
                {"role": "roof", "object_id": 2, "reason": "roof-like triangle"},
            ],
            "assembly_steps": [
                {"step": 1, "intent": "place wall on foundation", "object_id": 1},
                {"step": 2, "intent": "place roof on wall", "object_id": 2},
            ],
            "limitations": ["current executor approximates house as vertical stack"],
            "replan_policy": ["if roof is blocked, use another top-like object"],
        },
        "base_object_id": 0,
        "stack_order": [1, 2],
        "reason": "Use square as foundation, rectangle as wall, triangle as roof.",
    }
    normalized = json.loads(normalize_stack_blocks_text(json.dumps(raw), OBJECTS, "搭一个房子", prefer_explicit_rule=False))
    assert normalized["base_object_id"] == 0
    assert normalized["stack_order"] == [1, 2]
    assert normalized["structure_plan"]["structure_type"] == "house"
    assert normalized["structure_plan"]["roles"][2]["role"] == "roof"


def test_explicit_color_rule_repairs_ids_but_keeps_structure_plan():
    raw = {
        "task_type": "stack_blocks",
        "structure_plan": {
            "structure_type": "tower",
            "roles": [{"role": "base", "object_id": 1, "reason": "LLM chose wrong base"}],
        },
        "base_object_id": 1,
        "stack_order": [0, 2],
        "reason": "Wrong order from model.",
    }
    validated = validate_stack_blocks_decision(raw, OBJECTS, "以红色为底，蓝色放到红色上，黄色放最上面")
    assert validated["base_object_id"] == 0
    assert validated["stack_order"] == [1, 2]
    assert validated["structure_plan"]["structure_type"] == "tower"
    assert validated["explicit_rule_repair"]["base_color"] == "red"


def test_structure_plan_rejects_unknown_object_ids():
    raw = {
        "task_type": "stack_blocks",
        "structure_plan": {"structure_type": "bridge", "roles": [{"role": "beam", "object_id": 99}]},
        "base_object_id": 0,
        "stack_order": [1],
        "reason": "Invalid role id.",
    }
    try:
        normalize_stack_blocks_text(json.dumps(raw), OBJECTS, "搭一座桥", prefer_explicit_rule=False)
        raise AssertionError("Expected unknown structure role object id to be rejected.")
    except ValueError as exc:
        assert "Structure plan references object ids" in str(exc)


if __name__ == "__main__":
    test_structure_plan_is_preserved_with_executable_stack_order()
    test_explicit_color_rule_repairs_ids_but_keeps_structure_plan()
    test_structure_plan_rejects_unknown_object_ids()
    print("llm structure planner tests passed")
