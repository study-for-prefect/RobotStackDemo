import unittest
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


if __name__ == "__main__":
    unittest.main()
