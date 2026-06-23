#!/usr/bin/env python3

from tools.robot.push_primitives import build_push_targets
from tools.workflows.stack_demo.push_clearing import evaluate_push_directions


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


def make_object(object_id, center, size=(0.04, 0.04, 0.03)):
    return {
        "id": object_id,
        "label": str(object_id),
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "visible": True,
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
    assert feasible[0]["source"] in ("perpendicular_left", "perpendicular_right")


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


if __name__ == "__main__":
    test_push_targets()
    test_direction_evaluation_selects_open_side()
    test_direction_evaluation_reports_no_safe_direction()
    print("push_primitives tests passed")
