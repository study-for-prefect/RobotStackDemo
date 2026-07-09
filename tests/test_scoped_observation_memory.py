#!/usr/bin/env python3

from robot_scene_pipeline.scene_memory import update_from_detections, update_from_scoped_detections
from tools.workflows.stack_demo.observation_scope import _scope_report, expected_placed_template
from tools.workflows.stack_demo.post_place_observation import (
    _planned_height_fallback,
    _restore_geometry_confirmed_placed_memory,
)


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


def test_planned_height_fallback_accepts_confirmed_placed_object_xy():
    held = {
        "id": 3,
        "label": "square green",
        "geometry_center_m": [0.1, 0.2, 0.015],
        "dimensions_m": [0.024, 0.024, 0.024],
    }
    place_step = {
        "tcp_place_xy_base_m": [0.30, 0.10],
        "object_offset_base_xy_m": [0.0, 0.0],
        "release_z_base_m": 0.036,
    }
    final_stack = {"top_z_base_m": 0.024, "stack_xy_base_m": [0.30, 0.10]}
    noisy_stack = {
        "valid": True,
        "top_z_base_m": 0.029,
        "placement_base_top_z_m": 0.029,
    }
    scoped_report = {
        "attempts": [
            {
                "observed_critical": [
                    {"template_id": 3, "template_label": "square green", "distance_m": 0.002}
                ]
            }
        ]
    }
    corrected = _planned_height_fallback(held, place_step, final_stack, noisy_stack, scoped_report)
    assert corrected["top_z_base_m"] == 0.048
    assert corrected["height_estimation_method"] == "planned_place_height_fallback"


def test_post_place_scope_rejects_xy_match_with_bad_height():
    template = {
        "id": 3,
        "label": "square green",
        "geometry_center_m": [0.3271, 0.209, 0.0342],
    }
    wrong_height = detection("square green", [0.3186, 0.2387, -0.0146])
    observed, missing = _scope_report(
        {"objects": [wrong_height]},
        [template],
        max_dist_m=0.07,
        max_z_delta_m=0.025,
    )
    assert observed == []
    assert missing[0]["template_id"] == 3


def test_post_place_scope_accepts_stack_detection_by_top_z():
    template = {
        "id": 1,
        "label": "square blue",
        "geometry_center_m": [0.2854, 0.1899, 0.05775],
    }
    stacked_detection = {
        "id": 1,
        "label": "square blue",
        "geometry_center_m": [0.286, 0.1916, 0.0184],
        "dimensions_m": [0.0237, 0.0236, 0.0752],
        "top_z_base_m": 0.0559,
    }
    observed, missing = _scope_report(
        {"objects": [stacked_detection]},
        [template],
        max_dist_m=0.07,
        max_z_delta_m=0.025,
    )
    assert missing == []
    assert observed[0]["template_id"] == 1


def test_geometry_confirmed_post_place_restores_memory_state():
    memory = {
        "objects": {
            "square_blue_1": {
                "label": "square blue",
                "state": "unconfirmed_missing",
                "visible": False,
                "pushable": False,
                "graspable": False,
                "observation_confidence": 0.15,
                "missing_observation_scope": "after_place",
                "unconfirmed_missing_count": 1,
            }
        }
    }
    report = {
        "status": "critical_missing",
        "critical_memory_ids": ["square_blue_1"],
    }
    restored = _restore_geometry_confirmed_placed_memory(memory, report)
    obj = restored["objects"]["square_blue_1"]
    assert obj["state"] == "placed"
    assert obj["post_place_confirmed_by"] == "stack_growth_geometry"
    assert obj["observation_confidence"] == 0.7
    assert "missing_observation_scope" not in obj


if __name__ == "__main__":
    test_scoped_update_keeps_missing_objects_low_confidence_not_operable()
    test_scoped_update_marks_missing_critical_as_unconfirmed_missing()
    test_expected_placed_template_uses_tcp_place_center()
    test_planned_height_fallback_accepts_confirmed_placed_object_xy()
    test_post_place_scope_rejects_xy_match_with_bad_height()
    test_post_place_scope_accepts_stack_detection_by_top_z()
    test_geometry_confirmed_post_place_restores_memory_state()
    print("scoped observation memory tests passed")
