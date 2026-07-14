import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch

from tools.workflows.stack_demo.arguments import parse_args
from tools.workflows.stack_demo.commands import pose_command
from tools.workflows.stack_demo.pick import pick_approach_command


class StackDemoArgumentsTests(unittest.TestCase):
    def test_pre_rotate_wrist_yaw_sign_defaults_to_negative(self):
        with patch("sys.argv", ["stack_demo_pipeline.py"]):
            args = parse_args()

        self.assertEqual(args.pre_rotate_wrist_yaw_sign, "negative")

    def test_pre_rotate_wrist_yaw_sign_can_still_be_restricted(self):
        with patch("sys.argv", ["stack_demo_pipeline.py", "--pre-rotate-wrist-yaw-sign", "positive"]):
            args = parse_args()

        self.assertEqual(args.pre_rotate_wrist_yaw_sign, "positive")

    def test_pre_rotate_defaults_to_three_times_approach_motion_scales(self):
        with patch(
            "sys.argv",
            [
                "stack_demo_pipeline.py",
                "--velocity", "0.11",
                "--acceleration", "0.12",
            ],
        ):
            args = parse_args()

        ready = pose_command(args, "ready.json")
        approach = pick_approach_command(args, "pick.json")
        self.assertAlmostEqual(args.pre_rotate_velocity, 0.33)
        self.assertAlmostEqual(args.pre_rotate_acceleration, 0.36)
        self.assertEqual(ready[ready.index("--velocity") + 1], "0.11")
        self.assertEqual(approach[approach.index("--velocity") + 1], "0.11")
        self.assertAlmostEqual(
            float(approach[approach.index("--pre-rotate-velocity") + 1]), 0.33,
        )
        self.assertEqual(ready[ready.index("--acceleration") + 1], "0.12")
        self.assertAlmostEqual(
            float(approach[approach.index("--pre-rotate-acceleration") + 1]), 0.36,
        )

    def test_default_pre_rotate_motion_scales_are_clamped(self):
        with patch(
            "sys.argv",
            ["stack_demo_pipeline.py", "--velocity", "0.5", "--acceleration", "0.6"],
        ):
            args = parse_args()

        self.assertEqual(args.pre_rotate_velocity, 1.0)
        self.assertEqual(args.pre_rotate_acceleration, 1.0)

    def test_pre_rotate_motion_scales_can_be_explicitly_overridden(self):
        with patch(
            "sys.argv",
            [
                "stack_demo_pipeline.py",
                "--velocity", "0.11",
                "--acceleration", "0.12",
                "--pre-rotate-velocity", "0.15",
                "--pre-rotate-acceleration", "0.16",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.pre_rotate_velocity, 0.15)
        self.assertEqual(args.pre_rotate_acceleration, 0.16)

    def test_deprecated_vlm_action_policy_flag_is_removed(self):
        with patch("sys.argv", ["stack_demo_pipeline.py", "--use-vlm-action-policy"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_push_tool_geometry_can_be_calibrated(self):
        with patch(
            "sys.argv",
            [
                "stack_demo_pipeline.py",
                "--push-tool-yaw-offset-deg", "90",
                "--push-tool-finger-length-m", "0.11",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.push_tool_yaw_offset_deg, 90.0)
        self.assertEqual(args.push_tool_finger_length_m, 0.11)

    def test_old_vlm_clearance_policy_flag_is_removed(self):
        with patch("sys.argv", ["stack_demo_pipeline.py", "--use-vlm-clearance-policy"]):
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_qwen_reasoning_runtime_defaults_are_explicit(self):
        with patch("sys.argv", ["stack_demo_pipeline.py"]):
            args = parse_args()
        self.assertEqual(args.vlm_think_mode, "auto")
        self.assertEqual(args.vlm_read_timeout_sec, 1200.0)
        self.assertEqual(args.vlm_keep_alive, "1h")
        self.assertEqual(args.vlm_max_backend_retries, 3)
        self.assertEqual(args.vlm_max_budget_retries, 3)
        self.assertFalse(args.unload_model_after_task)
        self.assertTrue(args.unload_vlm_before_execution)


if __name__ == "__main__":
    unittest.main()
