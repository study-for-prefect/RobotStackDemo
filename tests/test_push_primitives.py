#!/usr/bin/env python3

from types import SimpleNamespace

from tools.robot.push_primitives import build_push_targets
from robot_scene_pipeline.grasp_yaw_search import select_best_grasp
from robot_scene_pipeline.push_grasp_joint_evaluator import (
    _is_soft_swept_collision,
    evaluate_one_push_grasp_candidate,
    evaluate_push_grasp_joint_candidates,
)
from robot_scene_pipeline.tool_swept_volume import check_tool_swept_volume
from tools.workflows.stack_demo.push_clearing import build_push_execution_plan, evaluate_push_directions
from tools.workflows.stack_demo.push_flow import _raise_if_non_push_action


def test_push_targets():
    plan = {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "direction_base": [-1.0, 0.0, 0.0],
        "distance_m": 0.05,
        "lift_m": 0.05,
        "contact_z_offset_m": 0.015,
        "obstacle": {
            "geometry_center_m": [0.46, 0.10, 0.02],
            "dimensions_m": [0.06, 0.03, 0.03],
        },
    }
    targets = build_push_targets(plan)
    assert targets["pre_push"] == [0.505, 0.1, 0.07]
    assert targets["contact"] == [0.505, 0.1, 0.02]
    assert targets["push_end"] == [0.455, 0.1, 0.02]
    assert targets["retreat"] == [0.455, 0.1, 0.07]


def test_push_targets_reject_contact_above_thin_obstacle():
    plan = {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "direction_base": [1.0, 0.0, 0.0],
        "distance_m": 0.025,
        "lift_m": 0.05,
        "contact_z_offset_m": 0.015,
        "obstacle": {
            "geometry_center_m": [0.42, 0.10, -0.0059],
            "dimensions_m": [0.055, 0.031, 0.0145],
        },
    }
    try:
        build_push_targets(plan)
        raise AssertionError("expected thin obstacle contact height rejection")
    except ValueError as exc:
        assert "too high for obstacle height" in str(exc)


def make_object(object_id, center, size=(0.04, 0.04, 0.03)):
    return {
        "id": object_id,
        "label": str(object_id),
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "visible": True,
        "role": "loose_movable",
        "state": "free",
    }


def make_scene(objects):
    return {
        "objects": objects,
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }


def test_direction_evaluation_selects_open_side():
    target = make_object(2, (0.50, 0.10, 0.02))
    obstacle = make_object(3, (0.44, 0.10, 0.02))
    base = make_object(1, (0.37, 0.10, 0.02))
    evaluations = evaluate_push_directions(
        obstacle,
        target,
        [base, target, obstacle],
        distance_m=0.05,
    )
    feasible = [item for item in evaluations if item["feasible"]]
    assert feasible
    assert feasible[0]["collisions"] == []
    assert feasible[0]["source"] in (
        "tangent_left_from_target",
        "tangent_right_from_target",
        "rule_16dir_112deg",
        "rule_16dir_248deg",
    )
    assert len(evaluations) == 16


def test_direction_evaluation_reports_no_safe_direction():
    target = make_object(2, (0.50, 0.10, 0.02))
    obstacle = make_object(3, (0.46, 0.10, 0.02), size=(0.06, 0.03, 0.03))
    base = make_object(1, (0.40, 0.10, 0.02))
    evaluations = evaluate_push_directions(
        obstacle,
        target,
        [base, target, obstacle],
        distance_m=0.05,
    )
    assert evaluations
    assert not any(item["feasible"] for item in evaluations)


def test_all_yaws_blocked_selects_safe_joint_push():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacle = make_object("wide_obstacle", (0.46, 0.00, 0.015), size=(0.08, 0.12, 0.03))
    scene = make_scene([target, obstacle])
    grasp = select_best_grasp(
        target,
        scene["objects"],
        gripper_outer_width_m=0.04,
    )
    assert grasp["all_grasps_blocked"] is True

    report = evaluate_push_grasp_joint_candidates(
        scene,
        target,
        ["wide_obstacle"],
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    selected = report["selected_candidate"]
    assert selected is not None
    assert selected["feasible"] is True
    assert selected["candidate_id"] is not None
    assert selected["reason"] == "push_enables_current_grasp_and_preserves_future_tasks"
    assert selected["predicted_selected_grasp_yaw_deg"] is not None


def test_unannotated_detector_object_can_be_pushed_as_loose_movable():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacle = make_object("wide_obstacle", (0.46, 0.00, 0.015), size=(0.08, 0.12, 0.03))
    obstacle.pop("role")
    obstacle.pop("state")
    scene = make_scene([target, obstacle])
    report = evaluate_push_grasp_joint_candidates(
        scene,
        target,
        ["wide_obstacle"],
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    assert report["selected_candidate"] is not None
    assert any(item["pushed_object_loose_movable"] is True for item in report["candidates"])


def test_locked_structure_push_candidate_is_rejected():
    target = make_object("target", (0.40, 0.00, 0.015))
    base = make_object("red_base", (0.46, 0.00, 0.015), size=(0.08, 0.12, 0.03))
    base.update({"role": "base", "state": "locked", "pushable": True})
    scene = make_scene([target, base])
    report = evaluate_push_grasp_joint_candidates(
        scene,
        target,
        ["red_base"],
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    assert report["selected_candidate"] is None
    assert any(
        item["reason"] == "pushed_object_protected_or_not_pushable"
        for item in report["candidates"]
    )


def test_non_loose_movable_push_candidate_is_rejected():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacle = make_object("obstacle", (0.46, 0.00, 0.015), size=(0.08, 0.12, 0.03))
    obstacle["role"] = "unknown"
    scene = make_scene([target, obstacle])
    report = evaluate_push_grasp_joint_candidates(
        scene,
        target,
        ["obstacle"],
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    assert report["selected_candidate"] is None
    assert all(item["pushed_object_loose_movable"] is False for item in report["candidates"])


def test_replan_required_with_push_candidates_is_not_hard_refused():
    _raise_if_non_push_action(
        {
            "action": "replan_required",
            "blocked_by_base": True,
            "blocked_by_locked_structure": False,
            "blocked_by_placed_structure": False,
        },
        has_push_candidates=True,
    )


def test_push_reducing_blockers_through_target_is_rejected():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacle = make_object("obstacle", (0.46, 0.00, 0.015), size=(0.08, 0.12, 0.03))
    second_obstacle = make_object("second", (0.40, 0.06, 0.015), size=(0.08, 0.08, 0.03))
    scene = make_scene([target, obstacle, second_obstacle])
    result = evaluate_one_push_grasp_candidate(
        scene,
        target,
        obstacle,
        {
            "action": "push_away",
            "obstacle_id": "obstacle",
            "direction_base": [1.0, 0.0, 0.0],
            "distance_m": 0.05,
            "source": "test",
        },
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    assert result["feasible"] is False
    assert result["reason"] == "tool_swept_collision"


def test_soft_swept_collision_only_allows_tiny_short_progress_overlap():
    target = make_object("target", (0.40, 0.00, 0.015))
    loose = make_object("loose", (0.45, 0.00, 0.015))
    protected = make_object("base", (0.50, 0.00, 0.015))
    protected.update({"role": "base", "state": "locked"})
    tiny_target_collision = [
        {
            "id": "target",
            "stage": "vertical_approach",
            "overlap_area_m2": 0.0000008,
        },
        {
            "id": "loose",
            "stage": "horizontal_push",
            "overlap_area_m2": 0.00002,
        },
    ]
    large_target_collision = [
        {
            "id": "target",
            "stage": "vertical_approach",
            "overlap_area_m2": 0.000002,
        }
    ]
    protected_collision = [
        {
            "id": "base",
            "stage": "horizontal_push",
            "overlap_area_m2": 0.000001,
        }
    ]

    assert _is_soft_swept_collision(tiny_target_collision, [target, loose], target, [], 0.025)
    assert not _is_soft_swept_collision(large_target_collision, [target], target, [], 0.025)
    assert not _is_soft_swept_collision(protected_collision, [target, protected], target, ["base"], 0.025)
    assert not _is_soft_swept_collision(tiny_target_collision, [target, loose], target, [], 0.05)


def test_future_grasp_target_is_not_a_hard_push_clearing_constraint():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacle = make_object("obstacle", (0.55, 0.00, 0.015))
    future_target = make_object("future", (0.62, 0.00, 0.015))
    scene = make_scene([target, obstacle, future_target])
    result = evaluate_one_push_grasp_candidate(
        scene,
        target,
        obstacle,
        {
            "action": "push_away",
            "obstacle_id": "obstacle",
            "direction_base": [1.0, 0.0, 0.0],
            "distance_m": 0.05,
            "source": "test",
        },
        future_targets=[future_target],
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    assert result["reason"] != "blocks_future_target"


def test_push_candidate_rejected_when_it_blocks_future_place_region():
    target = make_object("target", (0.40, 0.00, 0.015))
    obstacle = make_object("obstacle", (0.55, 0.00, 0.015))
    scene = make_scene([target, obstacle])
    result = evaluate_one_push_grasp_candidate(
        scene,
        target,
        obstacle,
        {
            "action": "push_away",
            "obstacle_id": "obstacle",
            "direction_base": [1.0, 0.0, 0.0],
            "distance_m": 0.05,
            "source": "test",
        },
        future_place_regions=[{"center_base_m": [0.60, 0.00, 0.0], "radius_m": 0.04}],
        table_bounds=scene["table_bounds"],
        gripper_outer_width_m=0.04,
    )
    assert result["feasible"] is False
    assert result["reason"] == "blocks_future_place_region"


def test_tool_vertical_approach_collision_rejects_candidate():
    obstacle = make_object("obstacle", (0.50, 0.00, 0.015))
    target = make_object("target", (0.465, 0.00, 0.015))
    plan = {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "direction_base": [1.0, 0.0, 0.0],
        "distance_m": 0.05,
        "lift_m": 0.05,
        "contact_z_offset_m": 0.015,
        "obstacle": obstacle,
    }
    result = check_tool_swept_volume(plan, [target, obstacle], ignore_object_ids=["obstacle"])
    assert result["feasible"] is False
    assert result["reason"] in ("vertical_approach_collision", "tool_swept_collision")


def test_push_execution_plan_carries_target_yaw_for_motion_orientation():
    target = make_object("target", (0.40, 0.00, 0.015))
    target.update(
        {
            "table_yaw_deg": -88.0,
            "table_yaw_valid": False,
            "table_yaw_source": "min_area_rect",
        }
    )
    obstacle = make_object("obstacle", (0.46, 0.00, 0.015))
    scene = make_scene([target, obstacle])
    plan = build_push_execution_plan(
        scene,
        target,
        {
            "subject": "obstacle",
            "direction_base": [1.0, 0.0, 0.0],
            "distance_m": 0.025,
            "reason": "test",
        },
        SimpleNamespace(
            push_clearing_distance_m=0.05,
            push_clearing_lift_m=0.05,
            push_clearing_contact_z_offset_m=0.015,
        ),
    )
    assert plan["target_yaw_deg"] == -88.0
    assert plan["target_yaw_valid"] is True
    assert plan["target_yaw_source"] == "min_area_rect"
    assert plan["target"]["id"] == "target"


if __name__ == "__main__":
    test_push_targets()
    test_push_targets_reject_contact_above_thin_obstacle()
    test_direction_evaluation_selects_open_side()
    test_direction_evaluation_reports_no_safe_direction()
    test_all_yaws_blocked_selects_safe_joint_push()
    test_unannotated_detector_object_can_be_pushed_as_loose_movable()
    test_locked_structure_push_candidate_is_rejected()
    test_non_loose_movable_push_candidate_is_rejected()
    test_replan_required_with_push_candidates_is_not_hard_refused()
    test_push_reducing_blockers_through_target_is_rejected()
    test_soft_swept_collision_only_allows_tiny_short_progress_overlap()
    test_future_grasp_target_is_not_a_hard_push_clearing_constraint()
    test_push_candidate_rejected_when_it_blocks_future_place_region()
    test_tool_vertical_approach_collision_rejects_candidate()
    test_push_execution_plan_carries_target_yaw_for_motion_orientation()
    print("push_primitives tests passed")
