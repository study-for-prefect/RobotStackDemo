#!/usr/bin/env python3

import builtins
import os
import tempfile
from types import SimpleNamespace

from tools.workflows.stack_demo import push_flow


def test_manual_clearance_reobserves_and_updates_memory():
    target = {
        "id": 2,
        "label": "square green",
        "geometry_center_m": [0.50, 0.10, 0.02],
        "dimensions_m": [0.04, 0.04, 0.03],
    }
    observed_state = {"objects": [target]}
    saved = {}
    originals = {
        "input": builtins.input,
        "capture": push_flow.capture_empty_observation,
        "update": push_flow.update_from_detections,
        "save": push_flow.save_memory,
        "reacquire": push_flow.reacquire_target,
        "relations": push_flow.build_geometry_relations,
    }
    try:
        builtins.input = lambda _prompt: ""
        push_flow.capture_empty_observation = (
            lambda _args, _output_dir, _held_object_id: observed_state
        )
        push_flow.update_from_detections = (
            lambda memory, objects: {"updated": True, "objects_seen": len(objects)}
        )
        push_flow.save_memory = lambda memory, path: saved.update(
            {"memory": memory, "path": path}
        )
        push_flow.reacquire_target = lambda state, _template: state["objects"][0]
        push_flow.build_geometry_relations = lambda _objects, target_id=None, **_kwargs: [
            {
                "type": "target_grasp_analysis",
                "object": target_id,
                "action": "pick",
                "grasp_feasible": True,
                "selected_grasp_yaw_deg": 0.0,
            }
        ]

        with tempfile.TemporaryDirectory() as cycle_dir:
            args = SimpleNamespace(memory_json=os.path.join(cycle_dir, "memory.json"))
            runtime = {"held_object_id": None, "current_stage": "test"}
            state, memory, held = push_flow._manual_clear_and_reobserve(
                args,
                cycle_dir,
                runtime,
                {"updated": False},
                target,
                1,
                None,
                "no_feasible_automatic_push_direction",
                {"candidate_results": []},
            )
            assert state is observed_state
            assert memory["updated"] is True
            assert held["id"] == 2
            assert runtime["current_stage"] == "observation_after_manual_clearing"
            assert saved["memory"]["objects_seen"] == 1
            assert os.path.exists(
                os.path.join(cycle_dir, "geometry_relations_after_manual_clearing.json")
            )
    finally:
        builtins.input = originals["input"]
        push_flow.capture_empty_observation = originals["capture"]
        push_flow.update_from_detections = originals["update"]
        push_flow.save_memory = originals["save"]
        push_flow.reacquire_target = originals["reacquire"]
        push_flow.build_geometry_relations = originals["relations"]


if __name__ == "__main__":
    test_manual_clearance_reobserves_and_updates_memory()
    print("manual push clearance tests passed")
