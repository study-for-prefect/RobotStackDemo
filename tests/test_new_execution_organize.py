import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.perception_contract import PerceptionServerError
from tools.workflows.stack_demo.app import (
    _LiveObserver,
    _merge_expected_tracks,
    _reobserve_without_action,
)
from tools.workflows.stack_demo.clutter.edge_generation import (
    PlacementTarget,
    TargetSpec,
    generate_physical_edges,
)
from tools.workflows.stack_demo.clutter.extraction_planner import ClutterExtractionPlanner
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.common.action_edges import ActionType
from tools.workflows.stack_demo.common.action_execution import execute_one_edge
from tools.workflows.stack_demo.common.action_validation import FinalSafetyGate
from tools.workflows.stack_demo.common.moveit_adapter import MoveItEdgeAdapter
from tools.workflows.stack_demo.common.track_lifecycle import mark_track_after_place
from tools.workflows.stack_demo.commands import (
    _refresh_tf_with_retry,
    _perception_snapshot_with_retry,
    return_to_ready_observation,
)
from tools.workflows.stack_demo.common.cycle_logging import CycleLogger
from tools.workflows.stack_demo.organize.completion import evaluate_organize_completion
from tools.workflows.stack_demo.organize.placement import build_color_target_regions, build_safe_slots
from tools.workflows.stack_demo.organize.planner import OrganizePlanner
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
    def test_live_house_review_receives_incremented_scene_revision(self):
        before = scene([raw_object(1, "roof", [0.50, 0.0, 0.02])], revision=4)
        reviewed_revisions = []

        def review(_args, raw, _directory, _task_type):
            reviewed_revisions.append(raw["scene_revision"])
            return raw

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "tools.workflows.stack_demo.app.capture_empty_current_pose",
            return_value={"frame_id": "base_link", "objects": []},
        ), patch(
            "tools.workflows.stack_demo.app._review_live_observation",
            side_effect=review,
        ):
            observer = _LiveObserver(
                SimpleNamespace(), config(), "build_house", Path(output_dir), 1,
                before, (), [], set(), {}, edge(revision=4),
            )
            observed = observer.observe("post_grasp")

        self.assertEqual(reviewed_revisions, [5])
        self.assertEqual(observed.scene_revision, 5)

    def test_no_action_house_reobserve_reviews_the_new_revision(self):
        before = scene([raw_object(1, "roof", [0.50, 0.0, 0.02])], revision=6)
        reviewed_revisions = []

        def review(_args, raw, _directory, _task_type):
            reviewed_revisions.append(raw["scene_revision"])
            return raw

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "tools.workflows.stack_demo.app.capture_empty_current_pose",
            return_value={"frame_id": "base_link", "objects": []},
        ), patch(
            "tools.workflows.stack_demo.app._review_live_observation",
            side_effect=review,
        ):
            observed, raw, _ = _reobserve_without_action(
                SimpleNamespace(offline_scene_state=None), config(), "build_house",
                Path(output_dir), 1, 1, before, (), [], set(), {},
            )

        self.assertEqual(reviewed_revisions, [7])
        self.assertEqual(raw["scene_revision"], 7)
        self.assertEqual(observed.scene_revision, 7)

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

    def test_missing_target_plus_contact_hold_and_live_visual_context_verifies_grasp(self):
        before = scene([
            raw_object(1, "target", [0.50, 0.0, 0.02]),
            raw_object(2, "context", [0.42, 0.08, 0.02], color="blue"),
        ])
        after = scene([
            raw_object(
                9, "uncertain_context", [0.43, 0.07, 0.02], color="blue",
                track_match_confidence=0.0,
                tracking_ambiguous=True,
            ),
        ], revision=2, expected=["target", "context"])
        action = edge(target="target", acted="target")
        result = execute_one_edge(
            action,
            before,
            MockExecutor(),
            MockObserver([after, after]),
            lambda selected, state: (True, {}),
        )
        self.assertTrue(result.post_grasp_verification.success)
        self.assertIn(
            "uncertain_context",
            result.post_grasp_verification.evidence["visible_non_target_context_tracks"],
        )

    def test_contact_hold_survives_fully_occluded_post_grasp_camera_view(self):
        before = scene([raw_object(1, "target", [0.50, 0.0, 0.02])])
        occluded = scene([], revision=2, expected=["target"])
        placed = scene([
            raw_object(7, "target", [0.28, 0.18, 0.02]),
        ], revision=3, expected=["target"])
        result = execute_one_edge(
            edge(target="target", acted="target"),
            before,
            MockExecutor(),
            MockObserver([occluded, placed]),
            lambda selected, state: (True, {"inside": True}),
        )
        self.assertTrue(result.success)
        self.assertTrue(
            result.post_grasp_verification.evidence["camera_view_fully_occluded"],
        )
        self.assertTrue(
            result.post_grasp_verification.evidence["gripper_hint_used_as_sole_evidence"],
        )

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
        self.assertEqual(executor.calls[-2:], ["retreat", "return_observation"])

    def test_place_descent_failure_retreats_while_holding_and_never_releases(self):
        class FailedDescentExecutor(MockExecutor):
            def descend_place(self, selected):
                self.calls.append("place_descent")
                raise RuntimeError("final position error exceeded")

        before = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        after_lift = scene([
            raw_object(6, "t1", [0.5, 0.0, 0.12]),
        ], revision=2, expected=["t1"])
        executor = FailedDescentExecutor()

        with self.assertRaisesRegex(RuntimeError, "final position error exceeded"):
            execute_one_edge(
                edge(), before, executor, MockObserver([after_lift]),
                lambda action, state: (True, {"inside": True}),
            )

        self.assertEqual(executor.calls[-3:], ["transport", "place_descent", "retreat"])
        self.assertNotIn("release", executor.calls)

    def test_22_place_failure_never_marks_success(self):
        before = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        after_lift = scene([
            raw_object(6, "t1", [0.5, 0.0, 0.12]),
        ], revision=2, expected=["t1"])
        after_place = scene([raw_object(7, "t1", [0.45, 0.0, 0.02])], revision=3, expected=["t1"])
        executor = MockExecutor()
        result = execute_one_edge(edge(), before, executor, MockObserver([after_lift, after_place]), lambda action, state: (False, {"inside": False}))
        self.assertFalse(result.success)
        self.assertEqual(result.status, "place_failed")
        self.assertEqual(executor.calls[-2:], ["retreat", "return_observation"])
        self.assertEqual(result.stages[-1]["reason"], "place_not_verified")

    def test_lifted_same_geometry_new_track_verifies_grasp(self):
        before = scene([raw_object(
            1, "track_red_01", [0.4507, 0.1915, -0.0055],
            label="rectangle red", color="red", size=[0.0453, 0.0266, 0.0138],
        )])
        after_lift = scene([raw_object(
            2, "track_red_03", [0.4626, 0.1964, 0.0114],
            label="rectangle red", color="red", size=[0.0375, 0.0059, 0.0497],
        )], revision=2, expected=["track_red_01"])
        placed = scene([raw_object(
            3, "track_red_01", [0.40, 0.27, 0.08],
            label="rectangle red", color="red",
        )], revision=3, expected=["track_red_01"])
        result = execute_one_edge(
            edge(target="track_red_01", acted="track_red_01"),
            before, MockExecutor(), MockObserver([after_lift, placed]),
            lambda selected, state: (True, {}),
        )
        self.assertTrue(result.post_grasp_verification.success)
        self.assertEqual(
            result.post_grasp_verification.evidence["lifted_geometry_rebound_track_id"],
            "track_red_03",
        )

    def test_tf_lookup_retries_one_transient_failure(self):
        args = SimpleNamespace(
            ros_python="python3", tf_json="/tmp/test_scene_tf.json",
            base_frame="base_link", camera_frame="camera_color_optical_frame",
            tool_frame="tool0", tf_timeout=1.0,
        )
        transient = __import__("subprocess").CalledProcessError(1, ["tf_lookup"])
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "tools.workflows.stack_demo.commands.run_non_actuating",
            side_effect=[transient, None],
        ) as mocked:
            _refresh_tf_with_retry(args, output_dir)
            attempts = json.loads(
                (Path(output_dir) / "tf_lookup_attempts.json").read_text()
            )
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual([item["ok"] for item in attempts], [False, True])

    def test_23_organize_color_region_completion(self):
        initial = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], color="red")])
        previous = build_organize_task_state(initial, config())
        placed = scene([raw_object(7, "t1", [0.41, 0.27, 0.02], color="red")], revision=2, expected=["t1"])
        state = build_organize_task_state(placed, config(), previous=previous)
        self.assertTrue(evaluate_organize_completion(placed, state)["task_complete"])

    def test_24_single_frame_miss_keeps_expected_count(self):
        first = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        previous = build_organize_task_state(first, config())
        missed = scene([], revision=2, expected=["t1"])
        state = build_organize_task_state(missed, config(), previous=previous)
        self.assertEqual(state.expected_tracks, ("t1",))
        self.assertEqual(state.missing_expected_tracks, ("t1",))

    def test_stable_late_visible_track_is_added_to_expected_tracks(self):
        raw = {"objects": [{"track_id": "late_red"}]}
        memory = {"tracks": {"late_red": {"history": [{}, {}]}}}
        self.assertEqual(
            _merge_expected_tracks(("initial",), raw, memory),
            ("initial", "late_red"),
        )

    def test_task_state_unions_new_scene_expected_tracks(self):
        initial = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])], expected=["t1"])
        previous = build_organize_task_state(initial, config())
        expanded = scene([
            raw_object(1, "t1", [0.5, 0.0, 0.02]),
            raw_object(2, "t2", [0.45, 0.08, 0.02], color="blue"),
        ], revision=2, expected=["t1", "t2"])
        state = build_organize_task_state(expanded, config(), previous=previous)
        self.assertEqual(state.expected_tracks, ("t1", "t2"))

    def test_open_gripper_slots_avoid_existing_object(self):
        current = scene([
            raw_object(1, "red", [0.5, 0.0, 0.02], color="red"),
            raw_object(2, "blue", [0.265, 0.2475, 0.02], label="square blue", color="blue"),
        ])
        regions = build_color_target_regions(current, config(), {"red": "red", "blue": "blue"})
        slots = build_safe_slots(current, config(), regions)
        self.assertNotIn(0.265, [round(item["position_m"][0], 3) for item in slots["red"]])

    def test_color_slots_keep_yaw_independent_footprint_away_from_region_edges(self):
        current = scene([raw_object(
            1, "blue", [0.50, 0.0, 0.02], color="blue",
            size=[0.0237, 0.0229, 0.0254],
        )])
        regions = build_color_target_regions(current, config(), {"blue": "blue"})
        slots = build_safe_slots(current, config(), regions)["blue"]
        bounds = regions["blue"]["bounds_base_m"]
        radius = 0.5 * (0.0237 ** 2 + 0.0229 ** 2) ** 0.5
        edge_allowance = config().section("organize")["observation_region_tolerance_m"] / 3.0
        self.assertGreater(len(slots), 1)
        self.assertTrue(all(
            bounds["xmin"] - edge_allowance <= slot["position_m"][0] - radius
            and slot["position_m"][0] + radius <= bounds["xmax"] + edge_allowance
            and bounds["ymin"] - edge_allowance <= slot["position_m"][1] - radius
            and slot["position_m"][1] + radius <= bounds["ymax"] + edge_allowance
            for slot in slots
        ))

    def test_organize_direct_placement_keeps_all_clear_slots_and_contacts_table(self):
        current = scene([raw_object(
            1, "red", [0.50, 0.0, 0.03], color="red", size=[0.023, 0.023, 0.06],
        )])
        state = build_organize_task_state(current, config())
        targets = OrganizePlanner(config(), None)._placement(
            current, state, current.current_objects[0],
        )
        self.assertGreater(len(targets), 1)
        self.assertTrue(all(
            target.additional_physical_parameters.get("placement_slot_id")
            for target in targets
        ))
        self.assertTrue(all(
            abs(target.place_pose["position_m"][2] - 0.03) < 1e-9
            for target in targets
        ))

    def test_organize_placement_adds_yaw_only_clearance_alternatives(self):
        current = scene([raw_object(
            1, "red", [0.50, 0.0, 0.03], color="red", size=[0.03, 0.03, 0.06],
        )])
        state = build_organize_task_state(current, config())
        generated = generate_physical_edges(
            current, ["red"], "organize_blocks", config(),
            lambda obj, interval: OrganizePlanner(config(), None)._placement(
                current, state, obj, interval,
            ),
            lambda obj: None,
        )
        ordinary = [
            item for item in generated.edges_by_target["red"]
            if item.action_type == ActionType.PICK_PLACE
            and item.physical_parameters["placement_yaw_policy"]
            == "safe_height_yaw_only_for_clearance"
        ]
        self.assertTrue(ordinary)
        self.assertTrue(any(
            item.physical_parameters["transport_path"][-2]["motion_role"]
            == "ordinary_yaw_only_at_safe_height"
            for item in ordinary
        ))
        self.assertTrue(all(
            item.physical_parameters["orientation_mode"] == "downward_yaw_only"
            for item in ordinary
        ))

    def test_unverified_released_object_keeps_destination_track_anchor(self):
        memory = {"tracks": {"t1": {
            "track_id": "t1", "center_base_m": [0.5, 0.0, 0.02], "history": [],
        }}}
        mark_track_after_place(memory, edge(), 3, success=False)
        self.assertEqual(memory["tracks"]["t1"]["manipulation_state"], "placed_unverified")
        self.assertEqual(memory["tracks"]["t1"]["center_base_m"], [0.28, 0.18, 0.02])

    def test_released_staging_location_is_reserved_even_if_not_visible(self):
        current = scene([raw_object(1, "red", [0.50, 0.0, 0.02], color="red")])
        current = replace(current, recent_action_results=({
            "acted_object_track_id": "blue",
            "target_region_id": "organize_staging",
            "release_executed": True,
            "planned_place_position_m": [0.59, -0.04, 0.02],
            "acted_object_size_m": [0.03, 0.03, 0.04],
            "success": False,
        },))
        state = build_organize_task_state(current, config())
        target = OrganizePlanner(config(), None)._staging(
            current, state, current.current_objects[0],
        )
        self.assertIsNotNone(target)
        self.assertNotEqual(target.place_pose["position_m"][:2], [0.59, -0.04])

    def test_transient_rgb_depth_skew_retries_inside_same_observation(self):
        args = SimpleNamespace()
        transient = PerceptionServerError(
            "skew", status=503, error_code="rgb_depth_not_synchronized",
        )
        with patch(
            "tools.workflows.stack_demo.commands.perception_server_snapshot",
            side_effect=[transient, {"ok": True}],
        ) as snapshot, patch("tools.workflows.stack_demo.commands.time.sleep"):
            response, attempts = _perception_snapshot_with_retry(
                args, "/tmp/unused", capture_reason="post_grasp", scene_revision=2,
            )
        self.assertEqual(response, {"ok": True})
        self.assertEqual(snapshot.call_count, 2)
        self.assertEqual([item["ok"] for item in attempts], [False, True])

    def test_post_action_ready_return_enables_bounded_wrist_staging(self):
        args = SimpleNamespace(
            execute=True,
            ready_pose_json="config/rectangle_ready_pose.json",
            velocity=0.15,
            acceleration=0.15,
            base_frame="base_link",
            tool_frame="tool0",
            tf_timeout=8.0,
            yes=True,
            init_stable_wait_s=0.0,
            ros_python="/usr/bin/python3",
        )
        with patch("tools.workflows.stack_demo.commands.run") as run:
            return_to_ready_observation(args)
        command = run.call_args.args[0]
        self.assertIn("--ready-stage-wrist-3", command)
        self.assertEqual(command[command.index("--velocity") + 1], "0.3")
        self.assertEqual(command[command.index("--acceleration") + 1], "0.3")

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

    def test_self_nudge_is_clearance_not_direct_graspability(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        nudge = edge(action_type=ActionType.NUDGE_BLOCKER, target="t1", acted="t1")
        options = build_target_options(
            current,
            "organize_blocks",
            {"t1": [nudge]},
            {},
        )
        self.assertEqual(len(options), 1)
        self.assertFalse(options[0].direct_graspable)
        self.assertEqual(options[0].direct_edge_count, 0)
        self.assertEqual(options[0].physical_clearance_edge_count, 1)

    def test_38_multiple_independent_edges_use_qwen_selection(self):
        current, options, generated = _one_option()
        candidate_id = generated.edges_by_target["t1"][0].candidate_id
        selected = QwenEdgeSelector(MockQwenClient([{
            "selected_candidate_id": candidate_id,
            "backup_candidate_ids": [],
            "reason_codes": ["minimum_risk"],
        }])).select(current, {}, options[0], generated.edges_by_target["t1"])
        self.assertEqual(selected.decision_source, "qwen_edge_selection")

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
        executor = MockExecutor()
        result = execute_one_edge(action, before, executor, MockObserver([moved]), lambda item, state: (True, {}))
        self.assertTrue(result.success)
        self.assertEqual(result.status, "nudge_verified")
        self.assertEqual(executor.calls, ["nudge", "return_observation"])
        self.assertEqual(
            [item["stage"] for item in result.stages[:3]],
            ["horizontal_push", "return_to_observation_pose", "fresh_observation"],
        )

    def test_self_target_nudge_uses_directional_displacement_not_self_clearance(self):
        before = scene([raw_object(1, "red", [0.30, 0.00, 0.02])])
        moved = scene(
            [raw_object(7, "red", [0.348, 0.00, 0.02])],
            revision=2,
            expected=["red"],
        )
        action = edge(
            action_type=ActionType.NUDGE_BLOCKER,
            target="red",
            acted="red",
            direction=[1.0, 0.0, 0.0],
        )
        result = execute_one_edge(
            action,
            before,
            MockExecutor(),
            MockObserver([moved]),
            lambda item, state: (True, {}),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.status, "nudge_verified")
        self.assertEqual(
            result.post_nudge_verification.evidence["verification_mode"],
            "self_target_directional_displacement",
        )
        self.assertFalse(result.post_nudge_verification.evidence["clearance_gain_required"])

    def test_execute_yes_authorizes_selected_nudge_without_legacy_flag(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        generated = generate_physical_edges(
            current, ["t1"], "organize_blocks", config(), placement, lambda obj: None,
        )
        nudge = next(
            item for item in generated.edges_by_target["t1"]
            if item.action_type == ActionType.NUDGE_BLOCKER
        )
        args = SimpleNamespace(
            execute=True,
            yes=True,
            execute_push_clearing=False,
            ros_python="/usr/bin/python3",
            tool_frame="tool0",
            tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as run:
            MoveItEdgeAdapter(args, config(), directory).execute_nudge(nudge)
        command = run.call_args.args[0]
        self.assertIn("--execute", command)
        self.assertIn("--yes", command)

    def test_push_execution_reuses_edge_specific_contact_height(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        generated = generate_physical_edges(
            current, ["t1"], "organize_blocks", config(), placement, lambda obj: None,
        )
        nudge = next(
            item for item in generated.edges_by_target["t1"]
            if item.action_type == ActionType.NUDGE_BLOCKER
        )
        nudge = replace(nudge, physical_parameters={
            **nudge.physical_parameters,
            "contact_z_offset_m": 0.0102,
        })
        args = SimpleNamespace(
            execute=True, yes=True, execute_push_clearing=False,
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            plan_path = MoveItEdgeAdapter(args, config(), directory)._write_push_plan(nudge)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(plan["contact_z_offset_m"], 0.0102)

    def test_close_gripper_returns_machine_readable_holding_hint(self):
        args = SimpleNamespace(
            execute=True,
            yes=True,
            ros_python="/usr/bin/python3",
            gripper_port="/dev/ttyUSB0",
        )
        with tempfile.TemporaryDirectory() as directory:
            def write_result(command):
                path = Path(command[command.index("--gripper-result-json") + 1])
                path.write_text(json.dumps({
                    "command": "close",
                    "status": True,
                    "position": 259,
                    "accepted": True,
                    "holding_detected": True,
                }), encoding="utf-8")

            with patch(
                "tools.workflows.stack_demo.common.moveit_adapter.run",
                side_effect=write_result,
            ):
                holding = MoveItEdgeAdapter(args, config(), directory).close_gripper(edge())
        self.assertTrue(holding)

    def test_simple_organize_scene_plans_direct_grasp_without_eager_push_plans(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], color="red")])
        client = MockQwenClient([])
        calls = []

        def checker(selected):
            calls.append(selected.candidate_id)
            return {"passed": True, "moveit_plan_only": True, "mode": "moveit_plan_only"}

        gate = FinalSafetyGate(config(), checker, require_real_moveit_plan=True)
        planner = ClutterExtractionPlanner(
            config(), QwenTargetSelector(client), QwenEdgeSelector(client), gate, checker,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = planner.plan_cycle(
                scene=current,
                task_type="organize_blocks",
                task_state={},
                target_track_ids=["t1"],
                placement_provider=placement,
                staging_provider=lambda obj: None,
                task_precondition=lambda selected, state: (True, "ok"),
                logger=CycleLogger(Path(directory)),
            )
        self.assertIsNotNone(result.selected_edge)
        self.assertNotEqual(result.selected_edge.action_type, ActionType.NUDGE_BLOCKER)
        self.assertEqual(result.decision_source, "code_direct_grasp_priority")
        self.assertEqual(calls, [result.selected_edge.candidate_id])
        self.assertFalse(client.calls)

    def test_simple_house_scene_prioritizes_direct_support_without_qwen_push_reasoning(self):
        current = scene([raw_object(1, "support", [0.5, 0.0, 0.02])])
        client = MockQwenClient([])
        calls = []

        def checker(selected):
            calls.append(selected.candidate_id)
            return {"passed": True, "moveit_plan_only": True, "mode": "moveit_plan_only"}

        def house_placement(_obj, _interval, _role):
            return PlacementTarget(
                ActionType.PLACE_HOUSE_ROLE,
                None,
                "left_support_lower",
                {"frame_id": "base_link", "position_m": [0.3775, 0.27, 0.015], "yaw_deg": 0.0},
                1.0,
                ("complete_house_role:left_support_lower",),
                {
                    "transport_safe": True, "place_descent_safe": True,
                    "release_safe": True, "return_safe": True,
                    "protected_safe": True,
                },
            )

        gate = FinalSafetyGate(config(), checker, require_real_moveit_plan=True)
        planner = ClutterExtractionPlanner(
            config(), QwenTargetSelector(client), QwenEdgeSelector(client), gate, checker,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = planner.plan_cycle(
                scene=current,
                task_type="build_house",
                task_state={},
                target_track_ids=(),
                placement_provider=lambda _obj, _interval: None,
                staging_provider=lambda _obj: None,
                task_precondition=lambda _selected, _state: (True, "ok"),
                logger=CycleLogger(Path(directory)),
                target_specs=(TargetSpec(
                    "support__left_support_lower", "support", "left_support_lower",
                ),),
                variant_placement_provider=house_placement,
            )
        self.assertIsNotNone(result.selected_edge)
        self.assertEqual(result.selected_edge.action_type, ActionType.PLACE_HOUSE_ROLE)
        self.assertEqual(result.decision_source, "code_direct_grasp_priority")
        self.assertEqual(calls, [result.selected_edge.candidate_id])
        self.assertFalse(client.calls)

    def test_normal_block_pose_forces_downward_yaw_and_joint_wrist3(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], color="red")])
        selected = generate_physical_edges(
            current, ["t1"], "organize_blocks", config(), placement, lambda obj: None,
        ).edges_by_target["t1"][0]
        release = dict(selected.physical_parameters["release_pose"])
        release["orientation_xyzw"] = [0.0, 0.0, 0.0, 1.0]
        physical = {**selected.physical_parameters, "release_pose": release}
        selected = replace(selected, physical_parameters=physical)
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run_non_actuating"
        ) as run_plan:
            MoveItEdgeAdapter(args, config(), directory).check(selected)
        release_command = run_plan.call_args_list[-1].args[0]
        strategy_index = release_command.index("--pre-rotate-strategy")
        quaternion_index = release_command.index("--hover-orientation-xyzw")
        self.assertEqual(release_command[strategy_index + 1], "joint-wrist3")
        sign_index = release_command.index("--pre-rotate-wrist-yaw-sign")
        self.assertEqual(release_command[sign_index + 1], "negative")
        self.assertIn("--hover-disable-orientation-settle", release_command)
        quaternion = [float(value) for value in release_command[quaternion_index + 1:quaternion_index + 5]]
        self.assertAlmostEqual(quaternion[2], 0.0)
        self.assertAlmostEqual(quaternion[3], 0.0)

    def test_organize_objects_use_ordinary_contact_release_even_for_shape_labels(self):
        ordinary_scene = scene([raw_object(1, "ordinary", [0.5, 0.0, 0.02])])
        ordinary = generate_physical_edges(
            ordinary_scene, ["ordinary"], "organize_blocks", config(), placement,
            lambda obj: None,
        ).edges_by_target["ordinary"][0]
        self.assertEqual(ordinary.physical_parameters["release_height_extra_m"], 0.0)
        self.assertEqual(
            ordinary.physical_parameters["release_pose"]["position_m"][2],
            ordinary.physical_parameters["place_pose"]["position_m"][2],
        )

        special_scene = scene([raw_object(
            2, "triangle", [0.5, 0.0, 0.02], label="triangle red",
        )])
        special = generate_physical_edges(
            special_scene, ["triangle"], "organize_blocks", config(), placement,
            lambda obj: None,
        ).edges_by_target["triangle"][0]
        self.assertAlmostEqual(special.physical_parameters["release_height_extra_m"], 0.0)
        self.assertEqual(special.physical_parameters["orientation_mode"], "downward_yaw_only")

    def test_transport_skips_duplicate_fixed_yaw_destination_pose(self):
        selected = edge()
        duplicate = {
            "position_m": [0.28, 0.18, 0.12],
            "yaw_deg": 35.0,
        }
        physical = {
            **selected.physical_parameters,
            "orientation_mode": "downward_yaw_only",
            "transport_path": [
                {"position_m": [0.5, 0.0, 0.12], "yaw_deg": 35.0},
                duplicate,
                dict(duplicate),
            ],
        }
        selected = replace(selected, physical_parameters=physical)
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            MoveItEdgeAdapter,
            "_run_explicit_pose",
        ) as run_pose:
            MoveItEdgeAdapter(args, config(), directory).transport(selected)
        self.assertEqual(run_pose.call_count, 1)
        self.assertTrue(run_pose.call_args.kwargs["preserve_current_orientation"])

    def test_ordinary_safe_height_yaw_waypoint_does_not_lock_old_yaw(self):
        selected = edge()
        physical = {
            **selected.physical_parameters,
            "orientation_mode": "downward_yaw_only",
            "transport_path": [
                {"position_m": [0.5, 0.0, 0.12], "yaw_deg": 35.0},
                {"position_m": [0.28, 0.18, 0.12], "yaw_deg": 35.0},
                {"position_m": [0.28, 0.18, 0.12], "yaw_deg": 0.0},
            ],
        }
        selected = replace(selected, physical_parameters=physical)
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            MoveItEdgeAdapter, "_run_explicit_pose",
        ) as run_pose:
            MoveItEdgeAdapter(args, config(), directory).transport(selected)
        self.assertEqual(run_pose.call_count, 2)
        self.assertTrue(run_pose.call_args_list[0].kwargs["preserve_current_orientation"])
        self.assertFalse(run_pose.call_args_list[1].kwargs["preserve_current_orientation"])

    def test_normal_post_grasp_place_locks_current_3d_orientation(self):
        selected = edge()
        physical = {
            **selected.physical_parameters,
            "orientation_mode": "downward_yaw_only",
            "release_pose": {"position_m": [0.28, 0.18, 0.04], "yaw_deg": 35.0},
        }
        selected = replace(selected, physical_parameters=physical)
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as execute:
            MoveItEdgeAdapter(args, config(), directory).descend_place(selected)
        command = execute.call_args.args[0]
        self.assertIn("--hover-preserve-current-orientation", command)
        self.assertNotIn("--pre-rotate-before-translation", command)

    def test_normal_grasp_descent_and_lift_do_not_replan_3d_orientation(self):
        selected = edge()
        selected = replace(
            selected,
            physical_parameters={
                **selected.physical_parameters,
                "orientation_mode": "downward_yaw_only",
                "grasp_pose": {"position_m": [0.27, 0.23, 0.0], "yaw_deg": -25.0},
                "lift_pose": {"position_m": [0.27, 0.23, 0.1], "yaw_deg": -25.0},
            },
        )
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as execute:
            adapter = MoveItEdgeAdapter(args, config(), directory)
            adapter.descend_grasp(selected)
            adapter.lift_for_observation(selected)
        self.assertEqual(execute.call_count, 2)
        for call in execute.call_args_list:
            command = call.args[0]
            self.assertIn("--hover-preserve-current-orientation", command)
            self.assertNotIn("--pre-rotate-before-translation", command)

    def test_special_shape_keeps_separate_full_3d_motion_policy(self):
        selected = edge()
        selected = replace(
            selected,
            physical_parameters={
                **selected.physical_parameters,
                "acted_object_shape": "triangle",
                "orientation_mode": "fixed_grasp_tcp_3d_rotation",
                "grasp_pose": {
                    "position_m": [0.27, 0.23, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        )
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as execute:
            MoveItEdgeAdapter(args, config(), directory).descend_grasp(selected)
        command = execute.call_args.args[0]
        self.assertNotIn("--hover-preserve-current-orientation", command)
        self.assertIn("--pre-rotate-before-translation", command)
        self.assertNotIn("--hover-disable-orientation-settle", command)
        strategy_index = command.index("--pre-rotate-strategy")
        self.assertEqual(command[strategy_index + 1], "pose")

    def test_special_shape_place_descent_disables_remote_ik_settle(self):
        selected = edge()
        selected = replace(
            selected,
            physical_parameters={
                **selected.physical_parameters,
                "acted_object_shape": "triangle",
                "orientation_mode": "fixed_grasp_tcp_3d_rotation",
                "release_pose": {
                    "position_m": [0.4, 0.27, 0.069],
                    "orientation_xyzw": [0.9989, 0.0475, 0.0, 0.0],
                },
            },
        )
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as execute:
            MoveItEdgeAdapter(args, config(), directory).descend_place(selected)
        command = execute.call_args.args[0]
        self.assertIn("--hover-disable-orientation-settle", command)
        self.assertIn("--pre-rotate-before-translation", command)
        strategy_index = command.index("--pre-rotate-strategy")
        self.assertEqual(command[strategy_index + 1], "pose")

    def test_special_preflight_uses_one_continuous_full_3d_pose_sequence(self):
        selected = edge()
        quat0 = [0.70710678, 0.0, 0.0, 0.70710678]
        quat1 = [0.6830127, -0.1830127, 0.1830127, 0.6830127]
        selected = replace(selected, physical_parameters={
            **selected.physical_parameters,
            "orientation_mode": "fixed_grasp_tcp_3d_rotation",
            # Historical special edges keep the full grasp quaternion on lift;
            # approach/grasp must inherit it without falling back to yaw.
            "approach_pose": {"position_m": [0.50, 0.0, 0.10], "yaw_deg": 0.0},
            "grasp_pose": {"position_m": [0.50, 0.0, 0.02], "yaw_deg": 0.0},
            "lift_pose": {"position_m": [0.50, 0.0, 0.12], "orientation_xyzw": quat0},
            "transport_path": [
                {"position_m": [0.50, 0.0, 0.12], "orientation_xyzw": quat0},
                {"position_m": [0.50, 0.0, 0.12], "orientation_xyzw": quat1,
                 "motion_role": "fixed_tcp_rotation_1"},
                {"position_m": [0.28, 0.18, 0.12], "orientation_xyzw": quat1,
                 "motion_role": "roof_transport"},
            ],
            "release_pose": {"position_m": [0.28, 0.18, 0.05], "orientation_xyzw": quat1},
        })
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run_non_actuating"
        ) as plan:
            result = MoveItEdgeAdapter(args, config(), directory).check(selected)
            self.assertTrue(result["passed"])
            self.assertEqual(plan.call_count, 1)
            command = plan.call_args.args[0]
            self.assertIn("--pose-sequence-json", command)
            self.assertNotIn("--hover-only", command)
            sequence_path = Path(command[command.index("--pose-sequence-json") + 1])
            payload = json.loads(sequence_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "tcp_pose_sequence_v1")
        self.assertEqual(payload["waypoints"][0]["orientation_xyzw"], quat0)
        self.assertEqual(payload["waypoints"][1]["orientation_xyzw"], quat0)
        self.assertEqual(
            [waypoint["name"] for waypoint in payload["waypoints"]],
            ["approach_pose", "grasp_pose", "lift_pose", "fixed_tcp_rotation_1",
             "roof_transport", "release_pose", "vertical_retreat_final_orientation_held"],
        )
        self.assertEqual(payload["waypoints"][3]["orientation_xyzw"], quat1)
        self.assertEqual(payload["waypoints"][-1]["orientation_xyzw"], quat1)

    def test_special_transport_executes_as_one_continuous_sequence(self):
        selected = edge()
        quat0 = [0.70710678, 0.0, 0.0, 0.70710678]
        quat1 = [0.6830127, -0.1830127, 0.1830127, 0.6830127]
        selected = replace(selected, physical_parameters={
            **selected.physical_parameters,
            "orientation_mode": "fixed_grasp_tcp_3d_rotation",
            "transport_path": [
                {"position_m": [0.50, 0.0, 0.12], "orientation_xyzw": quat0},
                {"position_m": [0.50, 0.0, 0.12], "orientation_xyzw": quat1},
                {"position_m": [0.28, 0.18, 0.12], "orientation_xyzw": quat1},
            ],
        })
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as execute:
            MoveItEdgeAdapter(args, config(), directory).transport(selected)
            self.assertEqual(execute.call_count, 1)
            command = execute.call_args.args[0]
            self.assertIn("--pose-sequence-json", command)
            self.assertIn("--execute", command)

    def test_special_retreat_keeps_release_quaternion_above_roof(self):
        selected = edge()
        release_quaternion = [0.92, -0.02, 0.01, -0.39]
        selected = replace(selected, physical_parameters={
            **selected.physical_parameters,
            "orientation_mode": "fixed_grasp_tcp_3d_rotation",
            "release_pose": {
                "position_m": [0.4, 0.27, 0.06],
                "orientation_xyzw": release_quaternion,
            },
        })
        args = SimpleNamespace(
            ros_python="/usr/bin/python3", tool_frame="tool0", tf_timeout=8.0,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "tools.workflows.stack_demo.common.moveit_adapter.run"
        ) as execute:
            MoveItEdgeAdapter(args, config(), directory).retreat(selected)
        command = execute.call_args.args[0]
        quaternion_index = command.index("--hover-orientation-xyzw")
        actual = [float(value) for value in command[quaternion_index + 1:quaternion_index + 5]]
        self.assertEqual(actual, release_quaternion)
        self.assertNotEqual(actual, [1.0, 0.0, 0.0, 0.0])
        self.assertIn("--hover-disable-orientation-settle", command)

    def test_tiny_negative_quaternion_cli_value_never_uses_exponent_notation(self):
        value = MoveItEdgeAdapter._cli_float(-9.290394557142738e-05)
        self.assertEqual(value, "-0.00009290394557143")
        self.assertNotIn("e", value.lower())


def _one_option():
    current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
    generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
    options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
    return current, options, generated


if __name__ == "__main__":
    unittest.main()
