#!/usr/bin/env python3

from tools.workflows.stack_demo.target_recovery import (
    missing_target_clearance_relations,
    state_with_missing_target,
)


def obj(object_id, center, size=(0.04, 0.04, 0.03), role="loose_movable", state="free"):
    return {
        "id": object_id,
        "label": str(object_id),
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "role": role,
        "state": state,
        "visible": True,
    }


def test_missing_target_state_injects_locked_template_first():
    target = obj("target", [0.40, 0.00, 0.015])
    conflicting = obj("target", [0.80, 0.00, 0.015])
    state, locked = state_with_missing_target({"objects": [conflicting]}, target)
    assert state["objects"][0]["geometry_center_m"] == target["geometry_center_m"]
    assert state["objects"][0]["target_detection_status"] == "missing_in_current_observation"
    assert locked["visible"] is False


def test_missing_target_clearance_prefers_nearby_high_loose_objects():
    target = obj("target", [0.40, 0.00, 0.015], size=(0.04, 0.04, 0.02))
    high_near = obj("high_near", [0.44, 0.00, 0.04], size=(0.04, 0.04, 0.06))
    low_near = obj("low_near", [0.43, 0.02, 0.012], size=(0.04, 0.04, 0.02))
    far_high = obj("far_high", [0.70, 0.00, 0.04], size=(0.04, 0.04, 0.06))
    protected = obj("base", [0.42, -0.02, 0.05], role="base", state="locked")
    state = {"objects": [target, high_near, low_near, far_high, protected]}
    relations = missing_target_clearance_relations(
        state,
        target,
        protected_object_ids=["base"],
        radius_m=0.10,
        min_top_z_delta_m=0.01,
    )
    assert [relation["subject"] for relation in relations] == ["high_near"]
    assert relations[0]["reason"] == "target_missing_nearby_high_loose_object"


if __name__ == "__main__":
    test_missing_target_state_injects_locked_template_first()
    test_missing_target_clearance_prefers_nearby_high_loose_objects()
    print("target recovery tests passed")
