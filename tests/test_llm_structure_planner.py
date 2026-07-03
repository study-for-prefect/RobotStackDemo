#!/usr/bin/env python3

import json

from robot_scene_pipeline.llm_scene_reasoner import (
    StackColorSelectionError,
    normalize_stack_blocks_decision,
    normalize_stack_blocks_text,
    stack_color_requirement_report,
    validate_stack_blocks_decision,
)


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
    assert normalized["full_stack_order"] == [0, 1, 2]
    assert normalized["stack_order_semantics"] == "place_order_excludes_base"
    assert normalized["structure_plan"]["structure_type"] == "house"
    assert normalized["structure_plan"]["roles"][2]["role"] == "roof"


def test_normalize_stack_blocks_decision_returns_dict_for_scene_flow():
    raw = {
        "task_type": "stack_blocks",
        "structure_plan": {"structure_type": "house"},
        "base_object_id": 0,
        "stack_order": [1, 2],
        "reason": "Scene flow should receive a dict, not JSON text.",
    }
    normalized = normalize_stack_blocks_decision(json.dumps(raw), OBJECTS, "叠一个房子", prefer_explicit_rule=False)
    assert isinstance(normalized, dict)
    assert normalized["stack_order"] == [1, 2]


def test_full_stack_order_with_base_is_normalized_to_place_order():
    raw = {
        "task_type": "stack_blocks",
        "structure_plan": {"structure_type": "house"},
        "base_object_id": 0,
        "stack_order": [0, 2, 1],
        "reason": "LLM used stack_order as the full bottom-to-top order.",
    }
    normalized = normalize_stack_blocks_decision(raw, OBJECTS, "叠一个房子", prefer_explicit_rule=False)
    assert normalized["base_object_id"] == 0
    assert normalized["full_stack_order"] == [0, 2, 1]
    assert normalized["stack_order"] == [2, 1]
    assert normalized["stack_order_semantics"] == "place_order_excludes_base"


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


def test_missing_required_color_is_reported_without_inventing_ids():
    objects = [
        {"id": 0, "label": "square red", "confidence": 0.99},
        {"id": 1, "label": "square blue", "confidence": 0.98},
        {"id": 2, "label": "square yellow", "confidence": 0.97},
    ]
    report = stack_color_requirement_report("以红色为底，把绿色放到红色上面", objects)
    assert report["missing_colors"] == ["green"]
    assert report["id_semantics"]["label_id"].startswith("YOLO class id")
    try:
        validate_stack_blocks_decision(
            {"task_type": "stack_blocks", "base_object_id": 0, "stack_order": [1], "reason": ""},
            objects,
            "以红色为底，把绿色放到红色上面",
        )
        raise AssertionError("Expected missing green to fail before execution.")
    except StackColorSelectionError as exc:
        assert exc.color == "green"
        assert exc.reason == "missing_required_color"


if __name__ == "__main__":
    test_structure_plan_is_preserved_with_executable_stack_order()
    test_normalize_stack_blocks_decision_returns_dict_for_scene_flow()
    test_full_stack_order_with_base_is_normalized_to_place_order()
    test_explicit_color_rule_repairs_ids_but_keeps_structure_plan()
    test_structure_plan_rejects_unknown_object_ids()
    test_missing_required_color_is_reported_without_inventing_ids()
    print("llm structure planner tests passed")
