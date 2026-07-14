import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.workflows.stack_demo import clearance_execution
from tools.workflows.stack_demo import vlm_action


class VlmNudgePreflightTests(unittest.TestCase):
    def test_successful_moveit_preflight_marks_nudge_executable_safe(self):
        args = _args()
        args.execute = True
        scene = _scene_with_blocked_contact_side()
        scene["objects"] = scene["objects"][:2]
        with patch.object(clearance_execution, "run", return_value=None), patch.object(
            clearance_execution, "check_tool_swept_volume",
            return_value={
                "feasible": True, "reason": "tool_swept_volume_clear",
                "contact_status": "clear", "controlled_contacts": [],
                "gripper_collision_profile": [],
            },
        ), patch.object(
            clearance_execution, "push_preflight_command", return_value=["true"],
        ):
            with tempfile.TemporaryDirectory() as output_dir:
                result = clearance_execution.preflight_nudge_action(
                    args, output_dir, scene,
                    {
                        "action_type": "nudge", "action_id": "nudge_2",
                        "object_id": 2, "obstacle_id": 2, "target_object_id": 1,
                        "direction_base": [1.0, 0.0, 0.0], "distance_m": 0.02,
                        "contact_side": "-x", "gripper_yaw_rad": 0.0,
                    },
                    1,
                )
        self.assertTrue(result["moveit_feasible"])
        self.assertTrue(result["executable_safe"])

    def test_geometry_only_dry_run_does_not_call_moveit(self):
        args = _args()
        args.execute = False
        args.moveit_plan_only = False
        scene = _scene_with_blocked_contact_side()
        scene["objects"] = scene["objects"][:2]
        original_run = clearance_execution.run
        clearance_execution.run = lambda _command: self.fail("plain dry-run must not contact MoveIt")
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                result = clearance_execution.preflight_nudge_action(
                    args,
                    output_dir,
                    scene,
                    {
                        "action_type": "nudge", "action_id": "nudge_2",
                        "object_id": 2, "obstacle_id": 2, "target_object_id": 1,
                        "direction_base": [1.0, 0.0, 0.0], "distance_m": 0.02,
                        "contact_side": "-x", "gripper_yaw_rad": 0.0,
                    },
                    1,
                )
        finally:
            clearance_execution.run = original_run

        self.assertEqual(
            result["moveit_preflight_skipped"],
            "execution_disabled_without_moveit_plan_only",
        )

    def test_neighbor_on_contact_side_rejects_before_moveit(self):
        original_run = clearance_execution.run
        clearance_execution.run = lambda _command: self.fail("MoveIt must not run after tool sweep rejection")
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                result = clearance_execution.preflight_nudge_action(
                    _args(),
                    output_dir,
                    _scene_with_blocked_contact_side(),
                    {
                        "action_type": "nudge",
                        "action_id": "nudge_2",
                        "object_id": 2,
                        "obstacle_id": 2,
                        "target_object_id": 1,
                        "direction_base": [1.0, 0.0, 0.0],
                        "distance_m": 0.02,
                        "contact_side": "-x",
                        "gripper_yaw_rad": 0.0,
                        "reason": "clear target",
                    },
                    1,
                )
        finally:
            clearance_execution.run = original_run

        self.assertFalse(result["moveit_feasible"])
        self.assertEqual(result["preflight_failure_stage"], "tool_swept_volume")
        collision_ids = {item["id"] for item in result["tool_swept_volume_report"]["collisions"]}
        self.assertIn(3, collision_ids)

        selected, safety = vlm_action._apply_preflight_result(
            result,
            {"object_id": 2},
            {"accepted": True, "moveit_feasible": None, "checks": {}, "failed_fields": []},
            {"failures": []},
        )
        self.assertIsNone(selected)
        self.assertEqual(safety["reason"], "tool_swept_volume_rejected")
        self.assertIn("tool_swept_volume_clear", safety["failed_fields"])

    def test_push_plan_uses_vlm_target_and_direction_aligned_yaw(self):
        plan = clearance_execution._build_nudge_execution_plan(
            _args(),
            _scene_with_blocked_contact_side(),
            {
                "action_type": "nudge",
                "action_id": "nudge_2",
                "object_id": 2,
                "obstacle_id": 2,
                "target_object_id": 1,
                "direction_base": [0.0, 1.0, 0.0],
                "distance_m": 0.02,
                "contact_side": "-y",
                "gripper_yaw_rad": 1.5707963267948966,
                "reason": "clear target",
            },
        )

        self.assertEqual(plan["target_object_id"], 1)
        self.assertEqual(plan["obstacle_object_id"], 2)
        self.assertEqual(plan["push_orientation_policy"], "align_to_target_yaw")
        self.assertAlmostEqual(plan["target_yaw_deg"], 90.0)


def _args():
    return SimpleNamespace(
        push_clearing_lift_m=0.05,
        push_clearing_contact_z_offset_m=0.015,
        push_tool_yaw_offset_deg=0.0,
        grasp_gripper_outer_width_m=0.112,
        push_tool_finger_length_m=0.12,
        push_tool_safety_margin_m=0.005,
    )


def _scene_with_blocked_contact_side():
    return {
        "objects": [
            {
                "id": 1,
                "label": "task block",
                "geometry_center_m": [0.20, 0.20, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            },
            {
                "id": 2,
                "label": "pushed block",
                "geometry_center_m": [0.0, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            },
            {
                "id": 3,
                "label": "contact-side neighbor",
                "geometry_center_m": [-0.04, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
