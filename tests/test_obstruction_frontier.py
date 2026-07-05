#!/usr/bin/env python3

from types import SimpleNamespace

from tools.workflows.stack_demo.obstruction_frontier import build_frontier_clearance_plan


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


def test_frontier_prefers_pick_away_for_graspable_direct_obstacle():
    target = obj("target", [0.40, 0.00, 0.015])
    obstacle = obj("obstacle", [0.435, 0.00, 0.015])
    state = {
        "objects": [target, obstacle],
        "table_bounds": {"xmin": 0.10, "xmax": 0.90, "ymin": -0.50, "ymax": 0.50},
    }
    plan = build_frontier_clearance_plan(
        state,
        target,
        protected_ids=[],
        args=args(),
        evaluate_push_fn=lambda *_args, **_kwargs: {"candidate_results": []},
    )
    assert plan["obstruction_graph"]["edges"]
    assert plan["obstruction_graph"]["frontier"][0]["object_id"] == "obstacle"
    selected = plan["safe_clearance_candidates"][0]
    assert selected["action_type"] == "pick_away"
    assert selected["safe_place_center_m"] is not None
    assert selected["target_yaw_gain"]["after_grasp_feasible"] is True


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


if __name__ == "__main__":
    test_frontier_prefers_pick_away_for_graspable_direct_obstacle()
    test_protected_obstacle_is_not_frontier_candidate()
    print("obstruction frontier tests passed")
