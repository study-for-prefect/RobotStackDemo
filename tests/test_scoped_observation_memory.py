#!/usr/bin/env python3

from robot_scene_pipeline.scene_memory import update_from_detections, update_from_scoped_detections
from tools.workflows.stack_demo.observation_scope import expected_placed_template


def detection(label, center, size=(0.04, 0.04, 0.03)):
    return {
        "label": label,
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
    }


def test_scoped_update_keeps_missing_objects_low_confidence_not_operable():
    memory = {
        "version": 1,
        "task": "stack_blocks",
        "step_index": 0,
        "objects": {},
        "structure": {"base": None, "placed_order": [], "current_top": None, "top_center_base": None, "top_z": None},
        "action_history": [],
    }
    memory = update_from_detections(
        memory,
        [
            detection("target", [0.40, 0.00, 0.015]),
            detection("future", [0.55, 0.00, 0.015]),
        ],
    )
    target_id = "target_1"
    future_id = "future_1"
    memory = update_from_scoped_detections(
        memory,
        [detection("target", [0.40, 0.00, 0.015])],
        scoped_object_ids=[target_id, future_id],
        critical_object_ids=[target_id],
        observation_scope="after_push_clearing",
    )
    assert memory["objects"][target_id]["visible"] is True
    assert memory["objects"][future_id]["visible"] is False
    assert memory["objects"][future_id]["state"] == "free"
    assert memory["objects"][future_id]["pushable"] is False
    assert memory["objects"][future_id]["graspable"] is False
    assert memory["objects"][future_id]["observation_confidence"] == 0.35


def test_scoped_update_marks_missing_critical_as_unconfirmed_missing():
    memory = {
        "version": 1,
        "task": "stack_blocks",
        "step_index": 0,
        "objects": {},
        "structure": {"base": None, "placed_order": [], "current_top": None, "top_center_base": None, "top_z": None},
        "action_history": [],
    }
    memory = update_from_detections(memory, [detection("target", [0.40, 0.00, 0.015])])
    memory = update_from_scoped_detections(
        memory,
        [],
        scoped_object_ids=["target_1"],
        critical_object_ids=["target_1"],
        observation_scope="after_place",
    )
    target = memory["objects"]["target_1"]
    assert target["state"] == "unconfirmed_missing"
    assert target["visible"] is False
    assert target["pushable"] is False
    assert target["graspable"] is False
    assert target["observation_confidence"] == 0.15


def test_expected_placed_template_uses_tcp_place_center():
    held = {"id": 3, "label": "square green", "geometry_center_m": [0.1, 0.2, 0.015]}
    place_step = {
        "tcp_place_xy_base_m": [0.30, 0.10],
        "object_offset_base_xy_m": [0.005, -0.002],
        "release_z_base_m": 0.052,
        "target_position_m": [9.0, 9.0, 9.0],
    }
    placed = expected_placed_template(held, place_step)
    assert placed["geometry_center_m"] == [0.305, 0.098, 0.052]
    assert placed["reacquire_source"] == "expected_post_place_pose"


if __name__ == "__main__":
    test_scoped_update_keeps_missing_objects_low_confidence_not_operable()
    test_scoped_update_marks_missing_critical_as_unconfirmed_missing()
    test_expected_placed_template_uses_tcp_place_center()
    print("scoped observation memory tests passed")
