import math
import unittest
from argparse import Namespace

from tools.calibration.yaw_rotation_probe_runtime.motion import (
    build_yaw_motion_command,
    hover_target_from_tool0_position,
)
from tools.calibration.yaw_rotation_probe_runtime.pose_math import (
    build_yaw_targets,
    quaternion_to_rpy_xyzw,
    rpy_to_quaternion_xyzw,
)


class YawRotationProbeTests(unittest.TestCase):
    def test_targets_keep_tool_position_and_roll_pitch(self):
        source_quat = rpy_to_quaternion_xyzw(math.radians(12.0), math.radians(-8.0), math.radians(31.0))
        payload = build_yaw_targets([0.4, -0.2, 0.35], source_quat)
        source_rpy = quaternion_to_rpy_xyzw(source_quat)

        for target in payload["targets"].values():
            self.assertEqual(target["tool0_position"], [0.4, -0.2, 0.35])
            roll, pitch, yaw = quaternion_to_rpy_xyzw(target["tool0_quat"])
            self.assertAlmostEqual(roll, source_rpy[0], places=9)
            self.assertAlmostEqual(pitch, source_rpy[1], places=9)
            self.assertAlmostEqual(math.degrees(yaw), target["yaw_deg"], places=9)

    def test_camera_position_changes_when_tool_has_camera_offset(self):
        source_quat = rpy_to_quaternion_xyzw(0.0, 0.0, 0.0)
        payload = build_yaw_targets(
            [1.0, 2.0, 0.3],
            source_quat,
            camera_position_m=[1.1, 2.0, 0.3],
            camera_quat_xyzw=source_quat,
            yaw_values_deg=(0, 90),
        )

        cam0 = payload["targets"]["yaw_p0deg"]["camera_link_position"]
        cam90 = payload["targets"]["yaw_p90deg"]["camera_link_position"]
        self.assertAlmostEqual(cam0[0], 1.1, places=9)
        self.assertAlmostEqual(cam0[1], 2.0, places=9)
        self.assertAlmostEqual(cam90[0], 1.0, places=9)
        self.assertAlmostEqual(cam90[1], 2.1, places=9)
        self.assertAlmostEqual(cam90[2], 0.3, places=9)

    def test_hover_target_reconstructs_same_tool0_position(self):
        tool0_position = [0.42, -0.18, 0.36]
        tool_offset_base = [-0.01, 0.02, 0.0]
        tool_z_offset = 0.15
        hover_target = hover_target_from_tool0_position(tool0_position, tool_z_offset, tool_offset_base)
        reconstructed_tool0 = [
            hover_target[0] + tool_offset_base[0],
            hover_target[1] + tool_offset_base[1],
            hover_target[2] + tool_z_offset + tool_offset_base[2],
        ]
        for actual, expected in zip(reconstructed_tool0, tool0_position):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_auto_motion_command_drives_hover_only_yaw_target(self):
        target = {
            "tool0_position": [0.4, -0.2, 0.35],
            "tool0_quat": rpy_to_quaternion_xyzw(0.1, -0.2, math.radians(45.0)),
        }
        args = Namespace(
            ros_python="/usr/bin/python3",
            motion_tool_z_offset=0.15,
            motion_tool_offset_base=[0.0, 0.0, 0.0],
            velocity=0.12,
            acceleration=0.12,
            pre_rotate_velocity=0.2,
            pre_rotate_acceleration=0.2,
            planning_time=5.0,
            tf_timeout=2.0,
            base_frame="base_link",
            tool_frame="tool0",
            pre_rotate_strategy="joint-wrist3",
            pre_rotate_wrist_yaw_sign="negative",
            pre_rotate_wrist_direction="auto",
            max_joint_delta=1.2,
            max_pre_rotate_joint_delta=math.pi,
            max_grasp_yaw_error_deg=3.0,
            orientation_settle_error_deg=0.2,
            execute=True,
            yes=True,
        )
        command = build_yaw_motion_command(args, target)
        self.assertIn("--hover-only", command)
        self.assertIn("--joint-space", command)
        self.assertIn("--execute", command)
        self.assertIn("--yes", command)
        hover_index = command.index("--hover-target-base") + 1
        hover_target = [float(value) for value in command[hover_index:hover_index + 3]]
        for actual, expected in zip(hover_target, [0.4, -0.2, 0.2]):
            self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    unittest.main()
