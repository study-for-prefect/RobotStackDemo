import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch

from tools.workflows.stack_demo.arguments import parse_args


class StackDemoArgumentsTests(unittest.TestCase):
    def test_pre_rotate_wrist_yaw_sign_defaults_to_negative(self):
        with patch("sys.argv", ["stack_demo_pipeline.py"]):
            args = parse_args()

        self.assertEqual(args.pre_rotate_wrist_yaw_sign, "negative")

    def test_pre_rotate_wrist_yaw_sign_can_still_be_restricted(self):
        with patch("sys.argv", ["stack_demo_pipeline.py", "--pre-rotate-wrist-yaw-sign", "positive"]):
            args = parse_args()

        self.assertEqual(args.pre_rotate_wrist_yaw_sign, "positive")

    def test_vlm_action_policy_flag_is_available(self):
        with patch("sys.argv", ["stack_demo_pipeline.py", "--use-vlm-action-policy"]):
            args = parse_args()

        self.assertTrue(args.use_vlm_action_policy)

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


if __name__ == "__main__":
    unittest.main()
