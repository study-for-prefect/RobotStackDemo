import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path


def _load_trajectory_module():
    if "sensor_msgs.msg" not in sys.modules:
        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

        class JointState:
            def __init__(self):
                self.name = []
                self.position = []

        sensor_msgs_msg.JointState = JointState
        sensor_msgs.msg = sensor_msgs_msg
        sys.modules["sensor_msgs"] = sensor_msgs
        sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    module_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "robot"
        / "moveit_preview"
        / "trajectory.py"
    )
    spec = importlib.util.spec_from_file_location("moveit_preview_trajectory_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trajectory = _load_trajectory_module()


class _Point:
    def __init__(self, positions):
        self.positions = positions


class _Trajectory:
    def __init__(self, joint_names, positions):
        self.joint_names = joint_names
        self.points = [_Point(point) for point in positions]


class MoveItPreviewTrajectoryTests(unittest.TestCase):
    def test_max_joint_delta_uses_adjacent_trajectory_points(self):
        traj = _Trajectory(
            ["shoulder_pan_joint", "wrist_3_joint"],
            [
                [0.0, 0.0],
                [0.6, 0.1],
                [1.2, 0.2],
            ],
        )

        joint_name, delta = trajectory.max_joint_delta(traj)

        self.assertEqual(joint_name, "shoulder_pan_joint")
        self.assertAlmostEqual(delta, 0.6)

    def test_max_joint_delta_still_reports_discrete_joint_jump(self):
        traj = _Trajectory(
            ["shoulder_pan_joint", "wrist_3_joint"],
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [0.3, 1.7],
            ],
        )

        joint_name, delta = trajectory.max_joint_delta(traj)

        self.assertEqual(joint_name, "wrist_3_joint")
        self.assertAlmostEqual(delta, 1.6)

    def test_max_joint_start_goal_delta_reports_slow_branch_jump(self):
        traj = _Trajectory(
            ["shoulder_pan_joint", "wrist_3_joint"],
            [
                [0.0, 0.0],
                [0.4, 0.4],
                [0.8, 0.8],
                [1.2, 1.2],
            ],
        )

        joint_name, delta = trajectory.max_joint_start_goal_delta(traj)

        self.assertEqual(joint_name, "shoulder_pan_joint")
        self.assertAlmostEqual(delta, 1.2)

    def test_gripper_close_accepts_contact_position_when_status_is_zero(self):
        args = Namespace(gripper_open_position=1000, gripper_close_position=0)

        self.assertTrue(trajectory.gripper_command_accepted(args, "close", 0, 259))
        self.assertFalse(trajectory.gripper_command_accepted(args, "close", 0, 500))
        self.assertFalse(trajectory.gripper_command_accepted(args, "open", 0, 259))
        self.assertTrue(trajectory.gripper_command_accepted(args, "open", 2, 1000))


if __name__ == "__main__":
    unittest.main()
