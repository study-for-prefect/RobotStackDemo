#!/usr/bin/env python3

from robot_scene_pipeline.stack_state import estimate_stack_state


def obj(object_id, label, center, size=(0.024, 0.024, 0.024)):
    return {
        "id": object_id,
        "label": label,
        "geometry_frame": "base_link",
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "pointcloud_geometry_valid": True,
    }


def test_initial_stack_can_be_limited_to_requested_base_only():
    base = obj(3, "square red", [0.3288, 0.2342, -0.0009])
    nearby_loose = obj(5, "square blue", [0.3271, 0.2090, -0.0014])
    state = {"objects": [base, nearby_loose]}
    stack = estimate_stack_state(
        state,
        base_object_id=3,
        previous_stack_xy=[0.3288, 0.2342],
        search_radius_m=0.06,
        allowed_stack_object_ids=[3],
    )
    assert stack["valid"] is True
    assert stack["top_object_id"] == 3
    assert stack["placement_base_object_id"] == 3
    assert [item["id"] for item in stack["stack_objects"]] == [3]


if __name__ == "__main__":
    test_initial_stack_can_be_limited_to_requested_base_only()
    print("stack state tests passed")
