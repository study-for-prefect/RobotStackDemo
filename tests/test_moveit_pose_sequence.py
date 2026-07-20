from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sensor_msgs.msg import JointState

from tools.robot.moveit_preview.pose_sequence import load_pose_sequence, run_pose_sequence


class _Logger:
    def error(self, _message):
        pass


class _Node:
    def __init__(self):
        self.latest_joint_state = JointState(name=["wrist_3_joint"], position=[0.0])

    def current_tool_transform(self, timeout):
        del timeout
        return object()

    def get_logger(self):
        return _Logger()


class MoveItPoseSequenceTests(unittest.TestCase):
    def test_plan_only_chains_each_trajectory_endpoint_into_next_request(self):
        initial = JointState(name=["wrist_3_joint"], position=[0.0])
        endpoint1 = JointState(name=["wrist_3_joint"], position=[0.4])
        endpoint2 = JointState(name=["wrist_3_joint"], position=[0.8])
        starts = []

        def plan(*_args, **kwargs):
            starts.append(kwargs["start_joint_state"])
            return endpoint1 if len(starts) == 1 else endpoint2

        waypoints = [
            {"name": "rotate_1", "position_m": [0.5, 0.0, 0.12],
             "orientation_xyzw": [0.70710678, 0.0, 0.0, 0.70710678]},
            {"name": "rotate_2", "position_m": [0.5, 0.0, 0.12],
             "orientation_xyzw": [0.6830127, -0.1830127, 0.1830127, 0.6830127]},
        ]
        with patch(
            "tools.robot.moveit_preview.pose_sequence.transform_position_quat",
            return_value=([0.4, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]),
        ), patch(
            "tools.robot.moveit_preview.pose_sequence.plan_and_maybe_execute_motion",
            side_effect=plan,
        ):
            ok = run_pose_sequence(
                _Node(), SimpleNamespace(tf_timeout=1.0, execute=False), initial,
                [0.0, 0.0, 0.16], waypoints,
            )
        self.assertTrue(ok)
        self.assertIs(starts[0], initial)
        self.assertIs(starts[1], endpoint1)

    def test_loader_refuses_yaw_only_waypoint(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
            json.dump({
                "schema_version": "tcp_pose_sequence_v1",
                "frame_id": "base_link",
                "waypoints": [{"position_m": [0.5, 0.0, 0.1], "yaw_deg": 30.0}],
            }, handle)
            handle.flush()
            with self.assertRaisesRegex(RuntimeError, "full orientation_xyzw"):
                load_pose_sequence(handle.name)


if __name__ == "__main__":
    unittest.main()
