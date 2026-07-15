import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tools.workflows.stack_demo.clutter.edge_generation import generate_physical_edges
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.common.action_edges import ActionType
from tools.workflows.stack_demo.common.action_execution import execute_one_edge
from tools.workflows.stack_demo.common.action_validation import FinalSafetyGate
from tools.workflows.stack_demo.organize.completion import evaluate_organize_completion
from tools.workflows.stack_demo.organize.placement import build_color_target_regions, build_safe_slots
from tools.workflows.stack_demo.organize.state import build_organize_task_state
from tools.workflows.stack_demo.policy.edge_selector import QwenEdgeSelector
from tools.workflows.stack_demo.policy.target_selector import QwenTargetSelector

from tests.new_arch_fixtures import (
    MockExecutor,
    MockObserver,
    MockQwenClient,
    config,
    edge,
    placement,
    raw_object,
    scene,
)


class NewExecutionOrganizeTests(unittest.TestCase):
    def test_12_no_feasible_edge_is_not_task_complete(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        state = build_organize_task_state(current, config())
        result = evaluate_organize_completion(current, state)
        self.assertFalse(result["task_complete"])
        self.assertFalse(result["no_feasible_edge_is_completion_evidence"])

    def test_13_missing_expected_track_blocks_completion(self):
        current = scene([], revision=2, expected=["t1"])
        state = build_organize_task_state(current, config())
        self.assertFalse(evaluate_organize_completion(current, state)["task_complete"])
        self.assertEqual(state.missing_expected_tracks, ("t1",))

    def test_19_open_gripper_place_descent_collision_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        selected = edge()
        checks = dict(selected.precheck_results)
        checks["place_descent_safe"] = False
        selected = replace(selected, precheck_results=checks)
        gate = FinalSafetyGate(config(), lambda item: {"passed": True, "moveit_plan_only": False, "mode": "offline"}, require_real_moveit_plan=False)
        result = gate.validate(selected, current, lambda action, state: (True, "ok"))
        self.assertFalse(result.passed)
        self.assertIn("place_descent_safe", result.failure_reasons)

    def test_20_grasp_failure_skips_transport_and_place(self):
        before = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        unchanged = scene([raw_object(9, "t1", [0.5, 0.0, 0.02])], revision=2, expected=["t1"])
        executor = MockExecutor()
        result = execute_one_edge(edge(), before, executor, MockObserver([unchanged]), lambda action, state: (True, {}))
        self.assertEqual(result.status, "grasp_failed")
        self.assertNotIn("transport", executor.calls)
        self.assertNotIn("place_descent", executor.calls)

    def test_21_verified_grasp_allows_place(self):
        before = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        after_lift = scene([
            raw_object(6, "t1", [0.5, 0.0, 0.12]),
        ], revision=2, expected=["t1"])
        after_place = scene([raw_object(7, "t1", [0.28, 0.18, 0.02])], revision=3, expected=["t1"])
        executor = MockExecutor()
        result = execute_one_edge(edge(), before, executor, MockObserver([after_lift, after_place]), lambda action, state: (True, {"inside": True}))
        self.assertTrue(result.success)
        self.assertIn("transport", executor.calls)
        self.assertIn("release", executor.calls)

    def test_22_place_failure_never_marks_success(self):
        before = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        after_lift = scene([
            raw_object(6, "t1", [0.5, 0.0, 0.12]),
        ], revision=2, expected=["t1"])
        after_place = scene([raw_object(7, "t1", [0.45, 0.0, 0.02])], revision=3, expected=["t1"])
        result = execute_one_edge(edge(), before, MockExecutor(), MockObserver([after_lift, after_place]), lambda action, state: (False, {"inside": False}))
        self.assertFalse(result.success)
        self.assertEqual(result.status, "place_failed")

    def test_23_organize_color_region_completion(self):
        initial = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], color="red")])
        previous = build_organize_task_state(initial, config())
        placed = scene([raw_object(7, "t1", [0.28, 0.17, 0.02], color="red")], revision=2, expected=["t1"])
        state = build_organize_task_state(placed, config(), previous=previous)
        self.assertTrue(evaluate_organize_completion(placed, state)["task_complete"])

    def test_24_single_frame_miss_keeps_expected_count(self):
        first = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        previous = build_organize_task_state(first, config())
        missed = scene([], revision=2, expected=["t1"])
        state = build_organize_task_state(missed, config(), previous=previous)
        self.assertEqual(state.expected_tracks, ("t1",))
        self.assertEqual(state.missing_expected_tracks, ("t1",))

    def test_open_gripper_slots_avoid_existing_object(self):
        current = scene([
            raw_object(1, "red", [0.5, 0.0, 0.02], color="red"),
            raw_object(2, "blue", [0.265, 0.105, 0.02], label="square blue", color="blue"),
        ])
        regions = build_color_target_regions(current, config(), {"red": "red", "blue": "blue"})
        slots = build_safe_slots(current, config(), regions)
        self.assertNotIn(0.265, [round(item["position_m"][0], 3) for item in slots["red"]])

    def test_35_clearance_cannot_move_protected_structure(self):
        current = scene([
            raw_object(1, "target", [0.5, 0.0, 0.02]),
            raw_object(2, "blocker", [0.46, 0.0, 0.02]),
        ], protected=["blocker"])
        selected = edge(action_type=ActionType.NUDGE_BLOCKER, target="target", acted="blocker", candidate_id="edge_nudge")
        selected = replace(selected, decision_metadata={"object_ref": "scene_1:obj_2"})
        gate = FinalSafetyGate(config(), lambda item: {"passed": True, "moveit_plan_only": False}, require_real_moveit_plan=False)
        result = gate.validate(selected, current, lambda action, state: (True, "ok"))
        self.assertFalse(result.passed)
        self.assertIn("protected_tracks_safe", result.failure_reasons)

    def test_36_opposite_push_directions_have_distinct_fingerprints(self):
        positive = edge(action_type=ActionType.NUDGE_BLOCKER, direction=[1, 0, 0])
        negative = edge(action_type=ActionType.NUDGE_BLOCKER, direction=[-1, 0, 0])
        self.assertNotEqual(positive.failure_fingerprint, negative.failure_fingerprint)

    def test_37_single_target_option_source_is_truthful(self):
        current, options, _ = _one_option()
        result = QwenTargetSelector(MockQwenClient([])).select(current, {}, options)
        self.assertEqual(result.decision_source, "single_feasible_target_option")

    def test_38_single_edge_source_is_truthful(self):
        current, options, generated = _one_option()
        selected = QwenEdgeSelector(MockQwenClient([])).select(current, {}, options[0], generated.edges_by_target["t1"])
        self.assertEqual(selected.decision_source, "single_feasible_edge")

    def test_39_qwen_connection_error_is_not_completion(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])])
        generated = generate_physical_edges(current, ["t1", "t2"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        result = QwenTargetSelector(MockQwenClient([ConnectionError("offline")])).select(current, {}, options)
        self.assertIsNone(result.selected)
        self.assertEqual(result.decision_source, "policy_invalid_output")

    def test_40_current_no_action_is_not_completion(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        state = build_organize_task_state(current, config())
        self.assertTrue(state.unresolved_tracks)
        self.assertFalse(evaluate_organize_completion(current, state)["task_complete"])

    def test_nudge_requires_measured_clearance_gain(self):
        before = scene([
            raw_object(1, "target", [0.50, 0.00, 0.02]),
            raw_object(2, "blocker", [0.46, 0.00, 0.02]),
        ])
        unchanged = scene([
            raw_object(7, "target", [0.50, 0.00, 0.02]),
            raw_object(8, "blocker", [0.46, 0.00, 0.02]),
        ], revision=2, expected=["target", "blocker"])
        action = edge(
            action_type=ActionType.NUDGE_BLOCKER, target="target", acted="blocker",
            direction=[-1.0, 0.0, 0.0],
        )
        result = execute_one_edge(action, before, MockExecutor(), MockObserver([unchanged]), lambda item, state: (True, {}))
        self.assertFalse(result.success)
        self.assertEqual(result.status, "nudge_failed")

    def test_nudge_success_uses_fresh_track_motion(self):
        before = scene([
            raw_object(1, "target", [0.50, 0.00, 0.02]),
            raw_object(2, "blocker", [0.46, 0.00, 0.02]),
        ])
        moved = scene([
            raw_object(7, "target", [0.50, 0.00, 0.02]),
            raw_object(8, "blocker", [0.445, 0.00, 0.02]),
        ], revision=2, expected=["target", "blocker"])
        action = edge(
            action_type=ActionType.NUDGE_BLOCKER, target="target", acted="blocker",
            direction=[-1.0, 0.0, 0.0],
        )
        result = execute_one_edge(action, before, MockExecutor(), MockObserver([moved]), lambda item, state: (True, {}))
        self.assertTrue(result.success)
        self.assertEqual(result.status, "nudge_verified")


def _one_option():
    current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
    generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
    options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
    return current, options, generated


if __name__ == "__main__":
    unittest.main()
