import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.tool_swept_volume import check_transport_swept_volume
from tools.workflows.stack_demo.arguments import parse_args
from tools.workflows.stack_demo.clutter.edge_generation import (
    PlacementTarget,
    generate_physical_edges,
)
from tools.workflows.stack_demo.clutter.path_safety import placement_path_checks
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.commands import assert_non_actuating_command, init_ready_pose
from tools.workflows.stack_demo.common.action_edges import ActionType
from tools.workflows.stack_demo.common.action_edges import build_failure_fingerprint
from tools.workflows.stack_demo.common.action_execution import execute_one_edge
from tools.workflows.stack_demo.common.track_lifecycle import (
    mark_track_after_place,
    mark_track_held,
    predicted_track_centers,
)
from tools.workflows.stack_demo.house.placement import staging_orientation_target
from tools.workflows.stack_demo.live_integration import (
    _action_names_from_topics,
    validate_live_integration_args,
)

from tests.new_arch_fixtures import (
    MockExecutor,
    MockObserver,
    config,
    edge,
    placement,
    raw_object,
    scene,
)


class Retry4ArchitectureFixTests(unittest.TestCase):
    def test_push_wrist_yaw_changes_failure_fingerprint(self):
        common = dict(
            scene_revision=1,
            action_type=ActionType.NUDGE_BLOCKER,
            primary_target_track_id="target",
            acted_object_track_id="blocker",
            task_role=None,
            target_region_id=None,
        )
        first = build_failure_fingerprint(
            **common,
            physical_parameters={
                "push_direction_base": [-1.0, 0.0, 0.0],
                "push_distance_m": 0.04,
                "push_wrist_yaw_deg": 0.0,
            },
        )
        second = build_failure_fingerprint(
            **common,
            physical_parameters={
                "push_direction_base": [-1.0, 0.0, 0.0],
                "push_distance_m": 0.04,
                "push_wrist_yaw_deg": 135.0,
            },
        )
        self.assertNotEqual(first, second)

    def test_humble_action_names_are_discovered_from_topics(self):
        topics = {
            "/move_action/_action/status",
            "/move_action/_action/feedback",
            "/joint_states",
        }
        self.assertEqual(_action_names_from_topics(topics), {"/move_action"})

    def test_live_integration_rejects_every_actuation_opt_in(self):
        for name in ("execute", "yes", "execute_push_clearing"):
            args = _live_args()
            setattr(args, name, True)
            with self.assertRaisesRegex(ValueError, name.replace("_", "-")):
                validate_live_integration_args(args)

    def test_live_integration_forces_one_step_plan_only(self):
        args = _live_args()
        validate_live_integration_args(args)
        self.assertFalse(args.execute)
        self.assertTrue(args.moveit_plan_only)
        self.assertEqual(args.max_task_steps, 1)

    def test_non_actuating_guard_rejects_motion_and_gripper_flags(self):
        for flag in ("--execute", "--yes", "--enable-gripper", "--gripper-close-only"):
            with self.assertRaises(RuntimeError):
                assert_non_actuating_command(["python3", "tool.py", flag])
        assert_non_actuating_command(["python3", "tool.py", "--hover-only"])

    def test_plan_only_never_initializes_ready_pose(self):
        args = SimpleNamespace(execute=False)
        with patch("tools.workflows.stack_demo.commands.run") as run:
            init_ready_pose(args)
        run.assert_not_called()

    def test_extract_to_staging_uses_staging_verification(self):
        before = scene([raw_object(1, "t1", [0.50, 0.0, 0.02])])
        staged_edge = replace(edge(action_type=ActionType.EXTRACT_TO_STAGING), target_region_id="staging")
        after_lift = scene([raw_object(2, "t1", [0.50, 0.0, 0.12])], revision=2, expected=["t1"])
        after_place = scene([raw_object(3, "t1", [0.28, 0.18, 0.02])], revision=3, expected=["t1"])
        result = execute_one_edge(
            staged_edge, before, MockExecutor(), MockObserver([after_lift, after_place]),
            lambda action, state: (False, {"final_target": False}),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.post_place_verification.evidence["verification_kind"], "staging_place_success")

    def test_pick_away_requires_actual_clearance_gain(self):
        before = scene([
            raw_object(1, "target", [0.50, 0.0, 0.02]),
            raw_object(2, "blocker", [0.46, 0.0, 0.02]),
        ], expected=["target", "blocker"])
        action = edge(
            action_type=ActionType.PICK_AWAY_BLOCKER, target="target", acted="blocker",
        )
        action = replace(action, expected_clearance_gain_m=0.01)
        after_lift = scene([
            raw_object(3, "target", [0.50, 0.0, 0.02]),
            raw_object(4, "blocker", [0.46, 0.0, 0.12]),
        ], revision=2, expected=["target", "blocker"])
        after_place = scene([
            raw_object(5, "target", [0.50, 0.0, 0.02]),
            raw_object(6, "blocker", [0.28, 0.18, 0.02]),
        ], revision=3, expected=["target", "blocker"])
        result = execute_one_edge(
            action, before, MockExecutor(), MockObserver([after_lift, after_place]),
            lambda selected, state: (False, {}),
        )
        self.assertTrue(result.success)
        self.assertEqual(result.post_place_verification.evidence["verification_kind"], "blocker_removed_success")

    def test_release_clearance_uses_actual_gripper_yaw(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], yaw_deg=30.0)])
        obj = current.current_objects[0]
        physical = {
            "release_pose": {"position_m": [0.28, 0.18, 0.03], "yaw_deg": 47.0},
            "release_gripper_yaw_deg": 47.0,
            "transport_path": [
                {"position_m": [0.5, 0.0, 0.15], "yaw_deg": 35.0},
                {"position_m": [0.28, 0.18, 0.15], "yaw_deg": 47.0},
            ],
        }
        with patch(
            "tools.workflows.stack_demo.clutter.path_safety.evaluate_gripper_pose_clearance",
            return_value={"finger_safe": True, "palm_safe": True, "blocking_track_ids": []},
        ) as clearance, patch(
            "tools.workflows.stack_demo.clutter.path_safety.check_transport_swept_volume",
            return_value={"feasible": True, "checked_components": ["palm"], "collisions": [], "reason": "clear"},
        ):
            checks = placement_path_checks(
                current, obj,
                {"position_m": [0.28, 0.18, 0.02], "yaw_deg": 10.0},
                physical, config(),
            )
        self.assertEqual(clearance.call_args.args[2], 47.0)
        self.assertEqual(checks["release_gripper_yaw_deg"], 47.0)

    def test_transport_checks_all_gf225_components(self):
        result = check_transport_swept_volume(
            [
                {"position_m": [0.4, 0.0, 0.15], "yaw_deg": 25.0},
                {"position_m": [0.5, 0.1, 0.15], "yaw_deg": 55.0},
            ],
            [],
            held_object_id="held",
            held_size_m=[0.03, 0.03, 0.04],
            gripper_outer_width_m=0.112,
            fingertip_width_m=0.025,
            upper_finger_width_m=0.062,
            upper_finger_height_m=0.070,
            palm_width_m=0.112,
            palm_depth_m=0.040,
            palm_height_m=0.150,
            tcp_offset_tool_m=[0.0, 0.0, 0.150],
            safety_margin_m=0.006,
        )
        self.assertEqual(set(result["checked_components"]), {
            "held_object", "fingertips", "upper_fingers", "palm", "tcp_to_gripper_body",
        })

    def test_track_lifecycle_held_then_placed_and_place_anchor(self):
        memory = {"tracks": {"t1": {"track_id": "t1", "center_base_m": [0.5, 0.0, 0.02], "history": []}}}
        action = edge()
        mark_track_held(memory, action, 2, held_state_confidence=0.8)
        self.assertEqual(memory["tracks"]["t1"]["manipulation_state"], "held_by_gripper")
        self.assertEqual(predicted_track_centers(action, "post_place", memory)["t1"], [0.28, 0.18, 0.02])
        mark_track_after_place(memory, action, 3, success=True)
        self.assertEqual(memory["tracks"]["t1"]["manipulation_state"], "placed")
        self.assertEqual(memory["tracks"]["t1"]["center_base_m"], [0.28, 0.18, 0.02])

    def test_direct_edge_failure_still_generates_staging_edge(self):
        current = scene([raw_object(1, "t1", [0.50, 0.0, 0.02])])

        def staging(obj):
            return PlacementTarget(
                ActionType.EXTRACT_TO_STAGING, "staging", None,
                {"frame_id": "base_link", "position_m": [0.60, -0.04, 0.02], "yaw_deg": 0.0},
                0.0, ("staged",),
                {"transport_safe": True, "place_descent_safe": True, "release_safe": True, "return_safe": True, "protected_safe": True},
            )

        def checker(action):
            return {"passed": action.action_type == ActionType.EXTRACT_TO_STAGING, "moveit_plan_only": True}

        generated = generate_physical_edges(
            current, ["t1"], "organize_blocks", config(), placement, staging,
            plan_checker=checker,
        )
        self.assertTrue(any(edge.action_type == ActionType.EXTRACT_TO_STAGING for edge in generated.edges_by_target["t1"]))

    def test_target_option_exposes_independent_selection_features(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        option = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)[0]
        for name in (
            "blocker_count", "physical_clearance_edge_count", "expected_clearance_steps",
            "best_first_step_clearance_gain_m", "direct_edge_count", "feasible_transport_edge_count",
            "protected_structure_min_clearance_m", "released_track_count", "best_moveit_cost",
        ):
            self.assertTrue(hasattr(option, name))

    def test_house_regrasp_requires_verified_physical_orientation_transition(self):
        current = scene([raw_object(1, "roof", [0.5, 0.0, 0.02], label="concave_rectangle", at_staging=True)])
        self.assertIsNone(staging_orientation_target(current.current_objects[0], "roof", config()))
        source = dict(current.current_objects[0].source)
        source["orientation_transition"] = {
            "geometry_verified": True,
            "regrasp_pose": {"frame_id": "base_link", "position_m": [0.5, 0.0, 0.02], "yaw_deg": 90.0},
            "staging_place_pose": {"frame_id": "base_link", "position_m": [0.58, -0.04, 0.02], "yaw_deg": 0.0},
            "expected_orientation_after": {"groove_face_state": "opening_down", "face_up": True},
        }
        obj = replace(current.current_objects[0], source=source)
        target = staging_orientation_target(obj, "roof", config())
        self.assertEqual(target.action_type, ActionType.REGRASP_FOR_ORIENTATION)
        self.assertTrue(target.additional_physical_parameters["orientation_plan"]["face_change_geometry_verified"])

    def test_fallback_default_remains_disabled(self):
        self.assertFalse(parse_args([]).allow_snapshot_subprocess_fallback)


def _live_args():
    return SimpleNamespace(
        live_integration_check=True,
        execute=False,
        yes=False,
        execute_push_clearing=False,
        moveit_plan_only=False,
        max_task_steps=20,
    )


if __name__ == "__main__":
    unittest.main()
