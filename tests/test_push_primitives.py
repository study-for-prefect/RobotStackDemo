#!/usr/bin/env python3

from tools.robot.push_primitives import build_push_targets


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


if __name__ == "__main__":
    test_push_targets()
    print("push_primitives tests passed")
