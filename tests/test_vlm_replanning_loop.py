import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from robot_scene_pipeline.tool_swept_volume import check_tool_swept_volume
from robot_scene_pipeline.vlm_action_validation import validate_vlm_action_decision
from robot_scene_pipeline.vlm_replanning import (
    action_fingerprint,
    advance_scene_revision,
    duplicate_proposal_feedback,
    find_duplicate_failed_proposal,
)
from tools.workflows.stack_demo import vlm_action_loop
from tools.workflows.stack_demo import scene as stack_scene


class VlmReplanningLoopTests(unittest.TestCase):
    def test_stack_semantic_failure_is_fed_back_before_second_order(self):
        state = _stack_state()
        calls = []

        def fake_call(_args, policy_input):
            calls.append(policy_input)
            order = [0, 1, 3] if len(calls) == 1 else [0, 1, 2, 3]
            return {
                "call_status": "parsed",
                "decision": {
                    "task_type": "stack_blocks", "full_stack_order": order,
                    "structure_plan": {}, "object_bindings": _bindings(state, order),
                    "reason": "proposal", "confidence": 0.8,
                },
            }

        original = stack_scene.call_vlm_stack_policy
        stack_scene.call_vlm_stack_policy = fake_call
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                decision = stack_scene._call_initial_vlm_stack_decision(
                    SimpleNamespace(
                        instruction="以红色为底，再放绿色、蓝色、黄色",
                        max_vlm_stack_attempts=3,
                    ),
                    state,
                    output_dir,
                )
        finally:
            stack_scene.call_vlm_stack_policy = original

        self.assertEqual(decision["full_stack_order"], [0, 1, 2, 3])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["failure_history"][0]["validation_stage"], "stack_semantic_validation")

    def test_first_hard_failure_is_fed_back_and_second_direction_passes(self):
        calls = []

        def fake_evaluate(*_args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                decision = _nudge([1.0, 0.0, 0.0], "-x")
                return None, decision, {
                    "accepted": False,
                    "decision": decision,
                    "reason": "tool_swept_volume_rejected",
                    "failed_fields": ["tool_swept_volume_clear"],
                    "checks": {"tool_swept_volume_clear": {"ok": False, "detail": {}}},
                    "tool_swept_volume_report": {
                        "hard_collisions": [{"id": 4, "stage": "approach_to_contact", "entity_type": "protected_structure"}],
                    },
                }, {"failures": []}
            selected = {"action_type": "nudge", "object_id": 2, "direction_base": [0.0, 1.0, 0.0]}
            return selected, selected, {"accepted": True, "decision": _nudge([0.0, 1.0, 0.0], "-y")}, {}

        original = vlm_action_loop.evaluate_autonomous_vlm_action_attempt
        vlm_action_loop.evaluate_autonomous_vlm_action_attempt = fake_evaluate
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                selected, _report, safety, _preflight = vlm_action_loop.select_autonomous_vlm_action(
                    SimpleNamespace(max_vlm_action_attempts=5), output_dir, {}, {}, [], 4, {}, 1, scene_revision=12,
                )
                with open(os.path.join(output_dir, "autonomous_action_history.json"), "r", encoding="utf-8") as handle:
                    history = json.load(handle)
        finally:
            vlm_action_loop.evaluate_autonomous_vlm_action_attempt = original

        self.assertIsNotNone(selected)
        self.assertTrue(safety["accepted"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["failure_history"][0]["validation_stage"], "geometry_validation")
        self.assertEqual(len(history["attempts"]), 2)

    def test_max_attempts_stops_safe(self):
        def always_reject(*_args, **_kwargs):
            decision = _nudge([1.0, 0.0, 0.0], "-x")
            safety = {"accepted": False, "decision": decision, "reason": "unsafe", "failed_fields": [], "checks": {}}
            return None, decision, safety, {}

        original = vlm_action_loop.evaluate_autonomous_vlm_action_attempt
        vlm_action_loop.evaluate_autonomous_vlm_action_attempt = always_reject
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                selected, report, safety, _preflight = vlm_action_loop.select_autonomous_vlm_action(
                    SimpleNamespace(max_vlm_action_attempts=2), output_dir, {}, {}, [], 4, {}, 1, scene_revision=3,
                )
        finally:
            vlm_action_loop.evaluate_autonomous_vlm_action_attempt = original

        self.assertIsNone(selected)
        self.assertEqual(report["reason"], "no_valid_vlm_action_after_replanning")
        self.assertEqual(len(safety["failure_history"]), 2)

    def test_duplicate_detection_is_limited_to_same_scene_revision(self):
        proposal = _nudge([1.0, 0.0, 0.0], "-x")
        feedback = duplicate_proposal_feedback(proposal, 7, 1)

        self.assertIsNotNone(find_duplicate_failed_proposal(proposal, 7, [feedback]))
        self.assertIsNone(find_duplicate_failed_proposal(proposal, 8, [feedback]))
        self.assertEqual(action_fingerprint(proposal, 7)["distance_range_m"], [0.02, 0.03])

    def test_movable_object_contact_is_recoverable(self):
        selected, safety = validate_vlm_action_decision(
            _grounded_nudge(), _contact_scene(), protected_ids=[4],
        )

        self.assertIsNotNone(selected)
        self.assertTrue(safety["accepted"])
        self.assertEqual(safety["recoverable_contacts"][0]["contacted_object_id"], 3)

    def test_gf225_tool_contact_with_movable_object_is_hard_collision(self):
        plan = _push_plan()
        objects = _contact_scene()["objects"]
        objects[2] = {**objects[2], "geometry_center_m": [-0.04, 0.0, 0.02]}
        report = check_tool_swept_volume(
            plan,
            objects,
            ignore_object_ids=[2],
            gripper_outer_width_m=0.04,
            finger_length_m=0.08,
            tool_depth_m=0.03,
            fingertip_thickness_m=0.006,
            safety_margin_m=0.002,
            tcp_offset_tool_m=[0.0, 0.0, 0.15],
        )

        self.assertFalse(report["feasible"])
        self.assertTrue(any(item["id"] == 3 and item["collision_source"] == "gf225_tool" for item in report["hard_collisions"]))

    def test_new_observation_advances_scene_revision(self):
        runtime = {"scene_revision": 4}
        state = {}

        revision = advance_scene_revision(runtime, state)

        self.assertEqual(revision, 5)
        self.assertEqual(state["scene_revision"], 5)


def _nudge(direction, contact_side):
    return {
        "action_type": "nudge", "object_id": 2, "target_object_id": 1,
        "contact_side": contact_side, "direction_base": direction,
        "distance_m": 0.025, "gripper_yaw_rad": 0.0,
    }


def _grounded_nudge():
    return {
        **_nudge([1.0, 0.0, 0.0], "-x"),
        "object_label": "square yellow", "object_center_base_m": [0.0, 0.0, 0.02],
        "target_object_label": "square green", "target_object_center_base_m": [0.20, 0.20, 0.02],
        "reason": "clear green", "confidence": 0.8,
    }


def _contact_scene():
    return {
        "table_bounds": {"xmin": -0.3, "xmax": 0.3, "ymin": -0.3, "ymax": 0.3},
        "objects": [
            {"id": 1, "label": "square green", "geometry_center_m": [0.20, 0.20, 0.02], "dimensions_m": [0.03, 0.03, 0.04]},
            {"id": 2, "label": "square yellow", "geometry_center_m": [0.0, 0.0, 0.02], "dimensions_m": [0.03, 0.03, 0.04], "pushable": True},
            {"id": 3, "label": "square blue", "geometry_center_m": [0.04, 0.0, 0.02], "dimensions_m": [0.03, 0.03, 0.04], "pushable": True},
            {"id": 4, "label": "square red", "geometry_center_m": [0.20, -0.20, 0.02], "dimensions_m": [0.03, 0.03, 0.04], "state": "locked"},
        ],
    }


def _push_plan():
    return {
        "schema_version": "push_execution_plan_v1", "frame_id": "base_link",
        "obstacle": _contact_scene()["objects"][1],
        "direction_base": [1.0, 0.0, 0.0], "distance_m": 0.025,
        "lift_m": 0.05, "contact_z_offset_m": 0.015,
        "gripper_yaw_rad": 0.0, "target_yaw_deg": 0.0,
    }


def _stack_state():
    return {
        "objects": [
            {"id": 0, "label": "square red", "geometry_center_m": [0.0, 0.0, 0.02]},
            {"id": 1, "label": "square green", "geometry_center_m": [0.1, 0.0, 0.02]},
            {"id": 2, "label": "square blue", "geometry_center_m": [0.2, 0.0, 0.02]},
            {"id": 3, "label": "square yellow", "geometry_center_m": [0.3, 0.0, 0.02]},
        ]
    }


def _bindings(state, order):
    object_map = {item["id"]: item for item in state["objects"]}
    return [
        {
            "object_id": object_id,
            "observed_label": object_map[object_id]["label"],
            "geometry_center_base_m": object_map[object_id]["geometry_center_m"],
        }
        for object_id in order
    ]


if __name__ == "__main__":
    unittest.main()
