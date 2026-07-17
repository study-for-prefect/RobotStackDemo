import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.perception_contract import PerceptionServerError
from tools.workflows.stack_demo.app import _merge_expected_tracks
from tools.workflows.stack_demo.clutter.edge_generation import generate_physical_edges
from tools.workflows.stack_demo.clutter.extraction_planner import ClutterExtractionPlanner
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.common.action_edges import ActionType
from tools.workflows.stack_demo.common.action_execution import execute_one_edge
from tools.workflows.stack_demo.common.action_validation import FinalSafetyGate
from tools.workflows.stack_demo.common.moveit_adapter import MoveItEdgeAdapter
from tools.workflows.stack_demo.common.track_lifecycle import mark_track_after_place
from tools.workflows.stack_demo.commands import (
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
            item.physical_parameters["transport_path"][-1]["motion_role"]
            == "ordinary_yaw_only_at_safe_height"
            for item in ordinary
        ))
        self.assertTrue(all(
            item.physical_parameters["orientation_policy"] == "downward_yaw_only"
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
        self.assertIn("--ready-stage-wrist-3", run.call_args.args[0])

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

    def test_ordinary_release_contacts_surface_but_special_shape_keeps_gap(self):
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
        self.assertAlmostEqual(special.physical_parameters["release_height_extra_m"], 0.01)

    def test_transport_skips_duplicate_fixed_yaw_destination_pose(self):
        selected = edge()
        duplicate = {
            "position_m": [0.28, 0.18, 0.12],
            "yaw_deg": 35.0,
        }
        physical = {
            **selected.physical_parameters,
            "orientation_policy": "downward_yaw_only",
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
            "orientation_policy": "downward_yaw_only",
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
            "orientation_policy": "downward_yaw_only",
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
                "orientation_policy": "downward_yaw_only",
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
                "orientation_policy": "full_3d_allowed",
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


def _one_option():
    current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
    generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
    options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
    return current, options, generated


if __name__ == "__main__":
    unittest.main()
