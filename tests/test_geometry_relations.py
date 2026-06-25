#!/usr/bin/env python3

import json
import os
import tempfile
from types import SimpleNamespace

from robot_scene_pipeline.geometry_relations import (
    blocks_grasp,
    build_geometry_relations,
    is_near,
    is_on,
    is_supporting,
    safe_to_push,
)
from tools.workflows.stack_demo.pick import build_offline_pick_plan
from tools.workflows.stack_demo.push_clearing import pushable_blocking_relations


def make_object(object_id, center, size=(0.04, 0.04, 0.03), **extra):
    obj = {
        "id": object_id,
        "label": object_id,
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "table_yaw_deg": 0.0,
        "visible": True,
    }
    obj.update(extra)
    return obj


def target_analysis(relations):
    matches = [relation for relation in relations if relation.get("type") == "target_grasp_analysis"]
    assert len(matches) == 1
    return matches[0]


def push_candidates(objects, relations, target_id):
    return pushable_blocking_relations(
        {"objects": objects},
        relations,
        target_id,
        base_id=None,
        previous_locked_stack=None,
    )


def test_near_relation():
    first = make_object("red_1", (0.40, 0.00, 0.015))
    second = make_object("green_1", (0.43, 0.00, 0.015))
    assert is_near(first, second) is True


def test_on_relation():
    red = make_object("square_red_1", (0.40, 0.00, 0.015))
    green = make_object("square_green_1", (0.40, 0.00, 0.045))
    assert is_on(green, red) is True
    assert is_supporting(red, green) is True


def test_blocking_grasp():
    target = make_object("square_green_1", (0.40, 0.00, 0.015))
    obstacle = make_object(
        "rectangle_1",
        (0.47, 0.00, 0.015),
        size=(0.06, 0.03, 0.03),
    )
    assert blocks_grasp(obstacle, target) is True


def test_locked_object_not_safe_to_push():
    target = make_object("square_green_1", (0.40, 0.00, 0.015))
    red = make_object(
        "square_red_1",
        (0.47, 0.00, 0.015),
        role="base",
        state="locked",
        pushable=True,
    )
    assert safe_to_push(red, [red, target], [1.0, 0.0, 0.0]) is False


def test_should_push_away_relation():
    target = make_object("square_green_1", (0.40, 0.00, 0.015))
    obstacles = [
        make_object("rectangle_1", (0.45, 0.00, 0.015), pushable=True),
        make_object("rectangle_2", (0.35, 0.00, 0.015), pushable=True),
        make_object("rectangle_3", (0.40, 0.05, 0.015), pushable=True),
        make_object("rectangle_4", (0.40, -0.05, 0.015), pushable=True),
    ]
    relations = build_geometry_relations(
        [target] + obstacles,
        target_id="square_green_1",
        table_bounds={"xmin": 0.20, "xmax": 0.80, "ymin": -0.30, "ymax": 0.30},
    )
    analysis = target_analysis(relations)
    assert analysis["action"] == "push_clearing"
    assert analysis["all_grasps_blocked"] is True
    push_relations = [
        relation
        for relation in relations
        if relation["type"] == "should_push_away"
    ]
    assert len(push_relations) >= 1
    assert push_relations[0]["subject"] == "rectangle_1"
    assert push_relations[0]["object"] == "square_green_1"
    assert push_relations[0]["direction_base"] == [1.0, 0.0, 0.0]
    assert push_relations[0]["reason"] == "blocking_grasp_and_safe_to_push"


def test_side_block_in_row_uses_continuous_pick_yaw_without_push():
    target = make_object("target", (0.40, 0.00, 0.015), table_yaw_deg=23.0)
    middle = make_object("middle", (0.47364, 0.03126, 0.015), table_yaw_deg=23.0)
    far = make_object("far", (0.54728, 0.06252, 0.015), table_yaw_deg=23.0)
    objects = [target, middle, far]
    relations = build_geometry_relations(objects, target_id="target")
    analysis = target_analysis(relations)
    fixed_yaws = {0.0, 45.0, 90.0, 135.0}
    assert analysis["action"] == "pick"
    assert analysis["grasp_feasible"] is True
    assert analysis["selected_grasp_yaw_deg"] not in fixed_yaws
    assert not push_candidates(objects, relations, "target")


def test_target_under_other_object_returns_remove_top_action():
    target = make_object("target", (0.40, 0.00, 0.015))
    top = make_object("top", (0.40, 0.00, 0.045))
    relations = build_geometry_relations([target, top], target_id="target")
    analysis = target_analysis(relations)
    assert analysis["target_under_other_object"] is True
    assert analysis["action"] in ("remove_top_object", "pick_away_top_object")
    assert analysis["object_above_target"]["id"] == "top"
    assert analysis["selected_grasp_yaw_deg"] is None


def test_loose_objects_block_all_yaws_returns_push_clearing():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacles = [
        make_object("east", (0.45, 0.00, 0.015), pushable=True),
        make_object("west", (0.35, 0.00, 0.015), pushable=True),
        make_object("north", (0.40, 0.05, 0.015), pushable=True),
        make_object("south", (0.40, -0.05, 0.015), pushable=True),
    ]
    objects = [target] + obstacles
    relations = build_geometry_relations(objects, target_id="target")
    analysis = target_analysis(relations)
    assert analysis["action"] in ("push_clearing", "pick_away")
    assert analysis["all_grasps_blocked"] is True
    assert push_candidates(objects, relations, "target")


def test_base_blocks_all_yaws_returns_replan_without_push():
    target = make_object("target", (0.40, 0.00, 0.015))
    objects = [
        target,
        make_object("base", (0.45, 0.00, 0.015), role="base"),
        make_object("west", (0.35, 0.00, 0.015)),
        make_object("north", (0.40, 0.05, 0.015)),
        make_object("south", (0.40, -0.05, 0.015)),
    ]
    relations = build_geometry_relations(objects, target_id="target")
    analysis = target_analysis(relations)
    assert analysis["action"] == "replan_required"
    assert analysis["blocked_by_base"] is True
    assert not push_candidates(objects, relations, "target")


def test_placed_or_locked_structure_blocks_all_yaws_returns_replan_without_push():
    for object_id, extra, field in (
        ("locked", {"state": "locked"}, "blocked_by_locked_structure"),
        ("placed", {"state": "placed"}, "blocked_by_placed_structure"),
    ):
        target = make_object("target", (0.40, 0.00, 0.015))
        objects = [
            target,
            make_object(object_id, (0.45, 0.00, 0.015), **extra),
            make_object("west", (0.35, 0.00, 0.015)),
            make_object("north", (0.40, 0.05, 0.015)),
            make_object("south", (0.40, -0.05, 0.015)),
        ]
        relations = build_geometry_relations(objects, target_id="target")
        analysis = target_analysis(relations)
        assert analysis["action"] == "replan_required"
        assert analysis[field] is True
        assert not push_candidates(objects, relations, "target")


def test_selected_grasp_yaw_is_written_to_pick_plan():
    target = make_object(
        1,
        (0.40, 0.00, 0.015),
        label="square green",
        pointcloud_geometry_valid=True,
        geometry_frame="base_link",
        selected_grasp_yaw_deg=113.0,
        grasp_yaw_source="adaptive_grasp_yaw_search",
        feasible_yaw_intervals_deg=[[104.0, 123.0]],
    )
    state = {"objects": [target], "base_frame": "base_link"}
    args = SimpleNamespace(
        approach_height_m=0.05,
        pick_target_lift_m=0.0,
        xy_correction_json="",
        grasp_bias_base=[0.0, 0.0],
        grasp_bias_camera=[0.0, 0.0],
        tf_json="",
        fixed_square_yaw_deg=0.0,
        stack_square_yaw_mode="detected",
        grasp_axis="long",
        gripper_yaw_offset_deg=0.0,
        square_yaw_snap_tolerance_deg=5.0,
        grasp_tool_offset_local=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "pick_plan.json")
        build_offline_pick_plan(state, target, output_path, args)
        with open(output_path, "r", encoding="utf-8") as f:
            step = json.load(f)["steps"][0]
    assert step["selected_grasp_yaw_deg"] == 113.0
    assert step["chosen_grasp_yaw_deg"] == 113.0
    assert step["target_yaw_deg"] == 113.0
    assert step["yaw_source"] == "adaptive_grasp_yaw_search"


if __name__ == "__main__":
    test_near_relation()
    test_on_relation()
    test_blocking_grasp()
    test_locked_object_not_safe_to_push()
    test_should_push_away_relation()
    test_side_block_in_row_uses_continuous_pick_yaw_without_push()
    test_target_under_other_object_returns_remove_top_action()
    test_loose_objects_block_all_yaws_returns_push_clearing()
    test_base_blocks_all_yaws_returns_replan_without_push()
    test_placed_or_locked_structure_blocks_all_yaws_returns_replan_without_push()
    test_selected_grasp_yaw_is_written_to_pick_plan()
    print("geometry_relations tests passed")
