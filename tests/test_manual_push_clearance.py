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


def test_automatic_clearance_reobserves_and_exits_to_pick():
    target = {
        "id": "target",
        "label": "target",
        "geometry_center_m": [0.40, 0.00, 0.015],
        "dimensions_m": [0.04, 0.04, 0.03],
        "role": "loose_movable",
        "state": "free",
    }
    obstacle = {
        "id": "obstacle",
        "label": "obstacle",
        "geometry_center_m": [0.46, 0.00, 0.015],
        "dimensions_m": [0.04, 0.04, 0.03],
        "role": "loose_movable",
        "state": "free",
    }
    state_before = {"objects": [target, obstacle], "table_bounds": {"xmin": 0.1, "xmax": 0.9, "ymin": -0.5, "ymax": 0.5}}
    state_after = {"objects": [target], "table_bounds": state_before["table_bounds"]}
    blocking_relation = {
        "type": "blocking_grasp",
        "subject": "obstacle",
        "object": "target",
        "source": "test",
    }
    relation_calls = {"count": 0}
    saved = {}
    originals = {
        "relations": push_flow._relations_for_target,
        "frontier": push_flow.build_frontier_clearance_plan,
        "run": push_flow.run,
        "command": push_flow.push_clear_command,
        "close": push_flow.close_gripper_command,
        "observe": push_flow.observe_empty_with_scope,
        "reacquire": push_flow.reacquire_target,
        "mark": push_flow.mark_pushed,
        "save": push_flow.save_memory,
        "delta": push_flow.observed_push_delta_m,
    }
    try:
        def fake_relations(_state, _held, _base_id, _stack, _args, base_template=None):
            relation_calls["count"] += 1
            if relation_calls["count"] == 1:
                return [
                    {
                        "type": "target_grasp_analysis",
                        "object": "target",
                        "action": "push_clearing",
                        "grasp_feasible": False,
                    },
                    blocking_relation,
                ]
            return [
                {
                    "type": "target_grasp_analysis",
                    "object": "target",
                    "action": "pick",
                    "grasp_feasible": True,
                    "selected_grasp_yaw_deg": 0.0,
                }
            ]

        def fake_frontier(*_args, **_kwargs):
            candidate = {
                "candidate_id": "safe_1",
                "action": "nudge",
                "action_type": "nudge",
                "obstacle_id": "obstacle",
                "target_object_id": "target",
                "direction_base": [1.0, 0.0, 0.0],
                "distance_m": 0.025,
                "direction_source": "test",
                "score": 1.0,
                "utility_score": 1.0,
                "easiness_score": 0.5,
                "risk_score": 0.0,
                "reason": "push_reduces_current_blockers_and_preserves_future_tasks",
                "push_evaluation": {"direction_base": [1.0, 0.0, 0.0], "distance_m": 0.025},
                "blocks": ["target"],
            }
            return {
                "obstruction_graph": {"nodes": [], "edges": [], "frontier": []},
                "obstacle_frontier_candidates": [],
                "all_clearance_action_candidates": [candidate],
                "safe_clearance_candidates": [candidate],
                "selected_clearance_action": candidate,
            }

        push_flow._relations_for_target = fake_relations
        push_flow.build_frontier_clearance_plan = fake_frontier
        push_flow.run = lambda _command: None
        push_flow.push_clear_command = lambda _args, _path: ["push"]
        push_flow.close_gripper_command = lambda _args: ["close"]
        push_flow.observe_empty_with_scope = lambda *_args, **_kwargs: (state_after, {"action_history": []}, {"status": "critical_confirmed"})
        push_flow.reacquire_target = lambda _state, _template: target
        push_flow.mark_pushed = lambda memory, *_args, **_kwargs: memory
        push_flow.save_memory = lambda memory, path: saved.update({"memory": memory, "path": path})
        push_flow.observed_push_delta_m = lambda _obstacle, _state: 0.03

        with tempfile.TemporaryDirectory() as cycle_dir:
            args = SimpleNamespace(
                execute=True,
                execute_push_clearing=True,
                enable_llm_push_selection=False,
                push_clearing_distance_m=0.05,
                push_clearing_lift_m=0.05,
                push_clearing_contact_z_offset_m=0.015,
                grasp_gripper_outer_width_m=0.04,
                grasp_gripper_inner_width_m=0.02,
                grasp_approach_length_m=0.02,
                push_tool_width_m=0.035,
                push_tool_safety_margin_m=0.005,
                max_automatic_push_clearing_attempts=2,
                memory_json=os.path.join(cycle_dir, "memory.json"),
                instruction="stack target",
                model="test",
                ollama_url="http://127.0.0.1:11434/api/chat",
                timeout=1,
                num_predict=64,
            )
            runtime = {"held_object_id": None, "current_stage": "test"}
            state, memory, held = push_flow.handle_push_clearing_before_pick(
                args,
                cycle_dir,
                runtime,
                {"action_history": []},
                state_before,
                target,
                target,
                base_id="base",
                previous_locked_stack=None,
            )
            assert state is state_after
            assert held["grasp_feasible"] is True
            assert relation_calls["count"] == 2
            assert os.path.exists(os.path.join(cycle_dir, "clearance_step_01_result.json"))
            assert os.path.exists(os.path.join(cycle_dir, "multi_step_clearance_summary.json"))
    finally:
        push_flow._relations_for_target = originals["relations"]
        push_flow.build_frontier_clearance_plan = originals["frontier"]
        push_flow.run = originals["run"]
        push_flow.push_clear_command = originals["command"]
        push_flow.close_gripper_command = originals["close"]
        push_flow.observe_empty_with_scope = originals["observe"]
        push_flow.reacquire_target = originals["reacquire"]
        push_flow.mark_pushed = originals["mark"]
        push_flow.save_memory = originals["save"]
        push_flow.observed_push_delta_m = originals["delta"]


if __name__ == "__main__":
    test_manual_clearance_reobserves_and_updates_memory()
    test_automatic_clearance_reobserves_and_exits_to_pick()
    print("manual push clearance tests passed")
