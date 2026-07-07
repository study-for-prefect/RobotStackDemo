#!/usr/bin/env python3

from types import SimpleNamespace

import tools.workflows.stack_demo.obstruction_frontier as frontier_module
from tools.workflows.stack_demo.obstruction_frontier import _score_candidate, build_frontier_clearance_plan


def obj(object_id, center, size=(0.035, 0.035, 0.03), role="loose_movable", state="free"):
    return {
        "id": object_id,
        "label": str(object_id),
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "visible": True,
        "role": role,
        "state": state,
    }


def args():
    return SimpleNamespace(
        grasp_gripper_outer_width_m=0.04,
        grasp_gripper_inner_width_m=0.05,
        grasp_approach_length_m=0.02,
        obstruction_graph_max_depth=3,
        clearance_nudge_distance_m=0.025,
        push_clearing_lift_m=0.05,
        push_clearing_contact_z_offset_m=0.015,
        push_tool_width_m=0.035,
        push_tool_safety_margin_m=0.005,
    )


def test_frontier_generates_preflight_nudge_for_direct_obstacle():
    target = obj("target", [0.40, 0.00, 0.015])
    obstacle = obj("obstacle", [0.435, 0.00, 0.015])
    state = {
        "objects": [target, obstacle],
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }
    def fake_push(*_args, **_kwargs):
        return {
            "candidate_results": [
                {
                    "relation": {},
                    "evaluations": [
                        {
                            "candidate_id": "nudge_obstacle",
                            "source": "away_from_target",
                            "direction_base": [1.0, 0.0, 0.0],
                            "distance_m": 0.025,
                            "feasible": True,
                            "approach_path_safe": True,
                            "push_swept_safe": True,
                            "push_end_safe": True,
                            "future_task_impact": {"feasible": True},
                            "score": 0.5,
                            "target_distance_after_m": 0.06,
                            "current_blocker_count": 2,
                            "predicted_blocker_count": 1,
                            "blocker_count_reduction": 1,
                        }
                    ],
                }
            ]
        }

    plan = build_frontier_clearance_plan(
        state,
        target,
        protected_ids=[],
        args=args(),
        evaluate_push_fn=fake_push,
    )
    assert plan["obstruction_graph"]["edges"]
    assert plan["obstruction_graph"]["frontier"][0]["object_id"] == "obstacle"
    assert plan["safe_clearance_candidates"] == []
    selected = plan["preflight_clearance_candidates"][0]
    assert selected["action_type"] == "nudge"
    assert selected["geometry_feasible"] is True
    assert selected["moveit_feasible"] is False
    assert selected["task_effective"] is True
    assert selected["direct_progress_candidate"] is True
    assert selected["exploratory"] is False


def test_protected_obstacle_is_not_frontier_candidate():
    target = obj("target", [0.40, 0.00, 0.015])
    protected = obj("base", [0.435, 0.00, 0.015], role="base", state="locked")
    state = {
        "objects": [target, protected],
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }
    plan = build_frontier_clearance_plan(
        state,
        target,
        protected_ids=["base"],
        args=args(),
        evaluate_push_fn=lambda *_args, **_kwargs: {"candidate_results": []},
    )
    assert plan["obstruction_graph"]["edges"]
    assert plan["obstacle_frontier_candidates"] == []
    assert plan["safe_clearance_candidates"] == []
    assert plan["preflight_clearance_candidates"] == []


def test_duplicate_target_detection_is_merged_before_frontier():
    target = obj("target", [0.40, 0.00, 0.015])
    duplicate_target = obj("target_dup", [0.402, 0.001, 0.015])
    duplicate_target["label"] = "target"
    obstacle = obj("obstacle", [0.435, 0.00, 0.015])
    state = {
        "objects": [target, duplicate_target, obstacle],
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }
    plan = build_frontier_clearance_plan(
        state,
        target,
        protected_ids=[],
        args=args(),
        evaluate_push_fn=lambda *_args, **_kwargs: {"candidate_results": []},
    )
    frontier_ids = {item["object_id"] for item in plan["obstacle_frontier_candidates"]}
    assert "target" not in frontier_ids
    assert "target_dup" not in frontier_ids
    assert "obstacle" in frontier_ids


def test_second_level_nudge_with_blocker_progress_is_enabling():
    target = obj("target", [0.40, 0.00, 0.015])
    blocker = obj("blocker", [0.435, 0.00, 0.015])
    enabler = obj("enabler", [0.470, 0.00, 0.015])
    state = {
        "objects": [target, blocker, enabler],
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }
    original_select = frontier_module.select_best_grasp

    def fake_select_best_grasp(target_obj, objects, **_kwargs):
        object_id = str(target_obj.get("id"))
        object_ids = {str(item.get("id")) for item in objects}
        if object_id == "target":
            blockers = [{"id": "blocker"}] if "blocker" in object_ids else []
            return {"grasp_feasible": False, "candidate_results": [], "blocking_objects": blockers}
        if object_id == "blocker":
            blockers = [{"id": "enabler"}] if "enabler" in object_ids else []
            return {"grasp_feasible": False, "candidate_results": [], "blocking_objects": blockers}
        return {"grasp_feasible": False, "candidate_results": [], "blocking_objects": []}

    def fake_push(_state, relation_target, relations, **_kwargs):
        if relation_target["id"] != "blocker":
            return {"candidate_results": []}
        return {
            "candidate_results": [
                {
                    "relation": relations[0],
                    "evaluations": [
                        {
                            "candidate_id": "push_018_obj_enabler",
                            "source": "progress.short",
                            "direction_base": [1.0, 0.0, 0.0],
                            "distance_m": 0.025,
                            "feasible": True,
                            "approach_path_safe": True,
                            "push_swept_safe": True,
                            "push_end_safe": True,
                            "future_task_impact": {"feasible": True},
                            "score": 0.5,
                            "target_distance_after_m": 0.06,
                            "current_blocker_count": 2,
                            "predicted_blocker_count": 1,
                            "blocker_count_reduction": 1,
                            "current_grasp_gain": 0.0,
                            "post_push_grasp_feasible": False,
                            "enables_blocker_object_id": "blocker",
                            "reason": "push_reduces_current_blockers_and_preserves_future_tasks",
                        }
                    ],
                }
            ]
        }

    try:
        frontier_module.select_best_grasp = fake_select_best_grasp
        plan = build_frontier_clearance_plan(
            state,
            target,
            protected_ids=[],
            args=args(),
            evaluate_push_fn=fake_push,
        )
    finally:
        frontier_module.select_best_grasp = original_select

    candidate = next(
        item for item in plan["preflight_clearance_candidates"]
        if item["candidate_id"] == "push_018_obj_enabler"
    )
    assert candidate["candidate_id"] == "push_018_obj_enabler"
    assert candidate["target_yaw_gain"]["gain"] == 0
    assert candidate["enabling_clearance_candidate"] is True
    assert candidate["exploratory"] is False
    assert candidate["automatic_execution_allowed"] is True
    assert candidate["blocker_count_reduction"] == 1


def test_blocker_count_reduction_is_enabling_candidate():
    candidate = _score_candidate(
        {
            "candidate_id": "push_018_obj_0",
            "action_type": "nudge",
            "feasible": True,
            "geometry_feasible": True,
            "approach_path_safe": True,
            "push_swept_safe": True,
            "push_end_safe": True,
            "future_task_feasible": True,
            "protected_structure_safe": True,
            "moveit_feasible": True,
            "direct_target_gain": 0.0,
            "enabling_gain": 0.75,
            "free_space_gain": 0.2,
            "target_yaw_gain": {"gain": 0, "after_grasp_feasible": False},
            "blocker_count_reduction": 1,
            "easiness_score": 0.5,
            "risk_score": 0.1,
        }
    )
    assert candidate["direct_clearance_candidate"] is False
    assert candidate["enabling_clearance_candidate"] is True
    assert candidate["exploratory"] is False
    assert candidate["automatic_execution_allowed"] is True
    assert candidate["executable_safe"] is True


def test_free_space_gain_alone_stays_exploratory():
    candidate = _score_candidate(
        {
            "candidate_id": "free_space_only",
            "action_type": "nudge",
            "feasible": True,
            "geometry_feasible": True,
            "approach_path_safe": True,
            "push_swept_safe": True,
            "push_end_safe": True,
            "future_task_feasible": True,
            "protected_structure_safe": True,
            "moveit_feasible": True,
            "direct_target_gain": 0.0,
            "enabling_gain": 0.0,
            "free_space_gain": 0.2,
            "target_yaw_gain": {"gain": 0, "after_grasp_feasible": False},
            "easiness_score": 0.5,
            "risk_score": 0.1,
        }
    )
    assert candidate["task_effective"] is True
    assert candidate["enabling_clearance_candidate"] is False
    assert candidate["exploratory"] is True
    assert candidate["automatic_execution_allowed"] is False
    assert candidate["executable_safe"] is False


def test_failed_swept_path_is_not_preflight_candidate():
    target = obj("target", [0.40, 0.00, 0.015])
    obstacle = obj("obstacle", [0.435, 0.00, 0.015])
    state = {
        "objects": [target, obstacle],
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }

    def fake_push(*_args, **_kwargs):
        return {
            "candidate_results": [
                {
                    "relation": {},
                    "evaluations": [
                        {
                            "candidate_id": "unsafe_swept",
                            "source": "test",
                            "direction_base": [1.0, 0.0, 0.0],
                            "distance_m": 0.025,
                            "feasible": True,
                            "approach_path_safe": True,
                            "push_swept_safe": False,
                            "push_end_safe": True,
                            "future_task_impact": {"feasible": True},
                            "score": 0.5,
                            "target_distance_after_m": 0.06,
                            "current_blocker_count": 2,
                            "predicted_blocker_count": 1,
                            "blocker_count_reduction": 1,
                        }
                    ],
                }
            ]
        }

    plan = build_frontier_clearance_plan(
        state,
        target,
        protected_ids=[],
        args=args(),
        evaluate_push_fn=fake_push,
    )
    assert any(item["candidate_id"] == "unsafe_swept" for item in plan["all_clearance_action_candidates"])
    assert plan["preflight_clearance_candidates"] == []


if __name__ == "__main__":
    test_frontier_generates_preflight_nudge_for_direct_obstacle()
    test_protected_obstacle_is_not_frontier_candidate()
    test_duplicate_target_detection_is_merged_before_frontier()
    test_second_level_nudge_with_blocker_progress_is_enabling()
    test_blocker_count_reduction_is_enabling_candidate()
    test_free_space_gain_alone_stays_exploratory()
    test_failed_swept_path_is_not_preflight_candidate()
    print("obstruction frontier tests passed")
