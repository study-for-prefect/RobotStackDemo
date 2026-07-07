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
from robot_scene_pipeline.grasp_yaw_search import normalize_yaw_signed_180
from tools.workflows.stack_demo.pick import build_offline_pick_plan
from tools.workflows.stack_demo.placement import build_frozen_place_step, validate_place_second_snapshot
from tools.workflows.stack_demo.push_clearing import (
    current_protected_structure_ids,
    pushable_blocking_relations,
    relation_objects_with_protected_structure,
)


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


def test_grasp_yaw_prefers_target_axis_over_extra_clearance():
    target = make_object(
        0,
        (0.3402, 0.0764, -0.0025),
        size=(0.0246, 0.0233, 0.0233),
        label="square green",
        table_yaw_deg=-89.51,
    )
    red_base = make_object(
        1,
        (0.3010, 0.1585, -0.0017),
        size=(0.0221, 0.0208, 0.0248),
        label="square red",
        table_yaw_deg=89.67,
        pushable=False,
    )
    relations = build_geometry_relations([target, red_base], target_id=0)
    analysis = target_analysis(relations)
    assert analysis["action"] == "pick"
    assert analysis["selected_grasp_source"] == "target_principal_axis"
    assert analysis["selected_grasp_axis_delta_deg"] == 0.0


def test_close_square_row_uses_top_grasp_equivalent_yaw_without_push():
    target = make_object(
        0,
        (0.3222, 0.1138, -0.0017),
        size=(0.023, 0.0221, 0.025),
        label="square green",
        table_yaw_deg=-3.03,
    )
    blue = make_object(
        1,
        (0.3236, 0.0562, -0.0022),
        size=(0.0241, 0.0239, 0.023),
        label="square blue",
        table_yaw_deg=89.78,
        pushable=True,
    )
    red_base = make_object(
        2,
        (0.3183, 0.2186, -0.0024),
        size=(0.0249, 0.0242, 0.0246),
        label="square red",
        table_yaw_deg=89.39,
        role="base",
        state="locked",
        pushable=False,
    )
    yellow = make_object(
        3,
        (0.3232, 0.1658, -0.0013),
        size=(0.0221, 0.0203, 0.0232),
        label="square yellow",
        table_yaw_deg=89.59,
        pushable=True,
    )
    objects = [target, blue, red_base, yellow]
    relations = build_geometry_relations(objects, target_id=0)
    analysis = target_analysis(relations)
    assert analysis["action"] == "pick"
    assert analysis["grasp_feasible"] is True
    assert analysis["selected_grasp_yaw_deg"] is not None
    assert not push_candidates(objects, relations, 0)


def test_adjacent_center_gap_blocker_allows_perpendicular_green_grasp():
    target = make_object(
        0,
        (0.3289, 0.0888, -0.002),
        size=(0.0226, 0.0222, 0.025),
        label="square green",
        table_yaw_deg=-0.8,
    )
    blue = make_object(
        1,
        (0.3304, 0.1223, 0.0),
        size=(0.0225, 0.0216, 0.0257),
        label="square blue",
        table_yaw_deg=0.12,
        pushable=True,
    )
    relations = build_geometry_relations([target, blue], target_id=0)
    analysis = target_analysis(relations)
    assert analysis["action"] == "pick"
    assert analysis["grasp_feasible"] is True
    assert abs(normalize_yaw_signed_180(float(analysis["selected_grasp_yaw_deg"]) - 89.2)) <= 2.0
    assert not push_candidates([target, blue], relations, 0)


def test_signed_yaw_normalization_keeps_small_negative_equivalent():
    assert normalize_yaw_signed_180(176.0) == -4.0
    assert normalize_yaw_signed_180(-184.0) == -4.0


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


def test_protected_structure_rebinds_by_template_when_detector_id_changes():
    base_template = make_object(
        6,
        (0.36, 0.19, 0.015),
        label="square red",
        role="base",
        state="locked",
    )
    current_objects = [
        make_object(6, (0.37, 0.10, 0.015), label="square yellow"),
        make_object(5, (0.357, 0.194, 0.015), label="square red"),
        make_object(0, (0.40, 0.08, 0.015), label="square green"),
    ]

    protected_ids = current_protected_structure_ids(
        current_objects,
        base_id=6,
        previous_locked_stack=None,
        base_template=base_template,
    )
    relation_objects = relation_objects_with_protected_structure(
        current_objects,
        base_id=6,
        previous_locked_stack=None,
        base_template=base_template,
    )

    current_base = next(obj for obj in relation_objects if obj["id"] == 5)
    stale_id_object = next(obj for obj in relation_objects if obj["id"] == 6)
    assert protected_ids == {"5"}
    assert current_base["role"] == "base"
    assert current_base["state"] == "locked"
    assert current_base["pushable"] is False
    assert stale_id_object["role"] == "loose_movable"


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


def test_final_held_place_snapshot_rejects_mismatched_base_id_and_yaw():
    locked = {
        "placement_base_object_id": 1,
        "placement_base_center_xy_m": [0.30, 0.16],
        "placement_base_top_z_m": 0.011,
        "stack_yaw_deg": 0.0,
    }
    wrong_id = {
        "placement_base_object_id": 0,
        "placement_base_center_xy_m": [0.311, 0.158],
        "placement_base_top_z_m": 0.0108,
        "stack_yaw_deg": 28.0,
    }
    try:
        validate_place_second_snapshot(locked, wrong_id, max_correction_m=0.015, top_z_tolerance_m=0.015)
        raise AssertionError("Expected mismatched base id to be rejected.")
    except RuntimeError as exc:
        assert "does not match locked base id" in str(exc)

    wrong_yaw = dict(wrong_id)
    wrong_yaw["placement_base_object_id"] = 1
    try:
        validate_place_second_snapshot(locked, wrong_yaw, max_correction_m=0.015, top_z_tolerance_m=0.015)
        raise AssertionError("Expected yaw jump to be rejected.")
    except RuntimeError as exc:
        assert "yaw delta" in str(exc)


def test_calibration_json_applies_grasp_and_place_xyz_biases():
    target = make_object(
        1,
        (0.40, 0.00, 0.015),
        label="square green",
        pointcloud_geometry_valid=True,
        geometry_frame="base_link",
        dimensions_m=[0.024, 0.024, 0.024],
    )
    base = make_object(
        2,
        (0.30, 0.10, 0.012),
        label="square red",
        pointcloud_geometry_valid=True,
        geometry_frame="base_link",
        dimensions_m=[0.024, 0.024, 0.024],
    )
    calibration = {
        "grasp_base_bias_m": [0.002, -0.003, 0.004],
        "place_base_bias_m": [-0.001, 0.002, -0.003],
        "max_correction_m": {"xy_m": 0.02, "z_m": 0.015},
    }
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
        place_yaw_strategy="base",
        place_center_strategy="top",
        release_gap_m=0.01,
        place_top_z_bias_m=0.0,
        object_offset_tool=[0.0, 0.0],
        place_bias_base=[0.0, 0.0],
        max_place_bias_base_m=0.01,
        max_stack_top_center_offset_m=0.015,
    )
    state = {"objects": [target, base], "base_frame": "base_link"}
    stack_state = {
        "valid": True,
        "placement_base_center_xy_m": [0.30, 0.10],
        "placement_base_top_z_m": 0.024,
        "top_object_id": 2,
        "placement_base_object_id": 2,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        calibration_path = os.path.join(tmpdir, "calibration.json")
        with open(calibration_path, "w", encoding="utf-8") as f:
            json.dump(calibration, f)
        args.calibration_json = calibration_path
        pick_path = os.path.join(tmpdir, "pick.json")
        build_offline_pick_plan(state, target, pick_path, args)
        with open(pick_path, "r", encoding="utf-8") as f:
            pick_step = json.load(f)["steps"][0]
        place_step = build_frozen_place_step(state, stack_state, base, target, args)

    assert pick_step["target_position_m"] == [0.402, -0.003, 0.019]
    assert pick_step["grasp_base_bias_m"] == [0.002, -0.003, 0.004]
    assert place_step["target_position_m"] == [0.299, 0.102, 0.043]
    assert abs(place_step["release_z_base_m"] - 0.043) < 1e-12
    assert place_step["place_base_bias_m"] == [-0.001, 0.002, -0.003]


def test_calibration_correction_limit_rejects_large_bias():
    target = make_object(
        1,
        (0.40, 0.00, 0.015),
        label="square green",
        pointcloud_geometry_valid=True,
        geometry_frame="base_link",
    )
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
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        calibration_path = os.path.join(tmpdir, "calibration.json")
        with open(calibration_path, "w", encoding="utf-8") as f:
            json.dump({"grasp_base_bias_m": [0.03, 0.0, 0.0], "max_correction_m": {"xy_m": 0.02}}, f)
        args.calibration_json = calibration_path
        try:
            build_offline_pick_plan({"objects": [target]}, target, os.path.join(tmpdir, "pick.json"), args)
            raise AssertionError("Expected large correction to be rejected.")
        except ValueError as exc:
            assert "correction XY norm" in str(exc)


if __name__ == "__main__":
    test_near_relation()
    test_on_relation()
    test_blocking_grasp()
    test_locked_object_not_safe_to_push()
    test_should_push_away_relation()
    test_side_block_in_row_uses_continuous_pick_yaw_without_push()
    test_grasp_yaw_prefers_target_axis_over_extra_clearance()
    test_close_square_row_uses_top_grasp_equivalent_yaw_without_push()
    test_adjacent_center_gap_blocker_allows_perpendicular_green_grasp()
    test_signed_yaw_normalization_keeps_small_negative_equivalent()
    test_target_under_other_object_returns_remove_top_action()
    test_loose_objects_block_all_yaws_returns_push_clearing()
    test_base_blocks_all_yaws_returns_replan_without_push()
    test_placed_or_locked_structure_blocks_all_yaws_returns_replan_without_push()
    test_protected_structure_rebinds_by_template_when_detector_id_changes()
    test_selected_grasp_yaw_is_written_to_pick_plan()
    test_final_held_place_snapshot_rejects_mismatched_base_id_and_yaw()
    test_calibration_json_applies_grasp_and_place_xyz_biases()
    test_calibration_correction_limit_rejects_large_bias()
    print("geometry_relations tests passed")
