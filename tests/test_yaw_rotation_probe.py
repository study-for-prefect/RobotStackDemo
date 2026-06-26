import math
import re
import unittest
from argparse import Namespace

from tools.calibration.yaw_rotation_probe_runtime.motion import (
    build_yaw_motion_command,
    hover_target_from_tool0_position,
)
from tools.calibration.yaw_rotation_probe_runtime.arguments import parse_args
from tools.calibration.yaw_rotation_probe_runtime.record_modes import (
    build_missed_record,
    build_pose_failed_record,
    pose_check,
    yaw_motion_arg_variants,
)
from tools.calibration.yaw_rotation_probe_runtime.recording import (
    select_detection,
    selection_failure_reason,
)
from tools.calibration.yaw_rotation_probe_runtime.pose_math import (
    build_yaw_targets,
    quaternion_xyzw_to_matrix,
    quaternion_to_rpy_xyzw,
    rpy_to_quaternion_xyzw,
    tool_z_axis_base,
    vertical_down_quaternion_for_yaw,
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

    def test_vertical_down_targets_keep_tool_z_axis_down(self):
        payload = build_yaw_targets(
            [0.2, 0.1, 0.35],
            vertical_down_quaternion_for_yaw(0.0),
            yaw_values_deg=(0, 45, 90, 180),
        )
        for target in payload["targets"].values():
            z_axis = tool_z_axis_base(target["tool0_quat"])
            self.assertAlmostEqual(z_axis[0], 0.0, places=9)
            self.assertAlmostEqual(z_axis[1], 0.0, places=9)
            self.assertAlmostEqual(z_axis[2], -1.0, places=9)
            self.assertEqual(target["tool0_position"], [0.2, 0.1, 0.35])

    def test_hover_target_reconstructs_same_tool0_position(self):
        tool0_position = [0.42, -0.18, 0.36]
        tool0_quat = vertical_down_quaternion_for_yaw(30.0)
        tcp_offset_tool = [-0.015, 0.0, 0.15]
        hover_target = hover_target_from_tool0_position(tool0_position, tool0_quat, tcp_offset_tool)
        tcp_offset_base = quaternion_xyzw_to_matrix(tool0_quat).dot(tcp_offset_tool).astype(float).tolist()
        reconstructed_tool0 = [
            hover_target[i] - tcp_offset_base[i]
            for i in range(3)
        ]
        for actual, expected in zip(reconstructed_tool0, tool0_position):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_auto_motion_command_drives_hover_only_yaw_target(self):
        target = {
            "tool0_position": [0.4, -0.2, 0.35],
            "tool0_quat": [-0.999989495446819, -1.3845783497326288e-06, -0.004573548607806859, 0.00030273293761267294],
        }
        args = Namespace(
            ros_python="/usr/bin/python3",
            motion_tcp_offset_tool=[-0.015, 0.0, 0.15],
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
        self.assertNotIn("--joint-space", command)
        self.assertIn("--execute", command)
        self.assertIn("--yes", command)
        scientific_number = re.compile(r"^-?\d+(?:\.\d+)?e[+-]?\d+$", re.IGNORECASE)
        self.assertFalse(any(scientific_number.match(value) for value in command))
        hover_index = command.index("--hover-target-base") + 1
        hover_target = [float(value) for value in command[hover_index:hover_index + 3]]
        tcp_offset_base = quaternion_xyzw_to_matrix(target["tool0_quat"]).dot([-0.015, 0.0, 0.15])
        expected_hover = [
            float(target["tool0_position"][i]) + float(tcp_offset_base[i])
            for i in range(3)
        ]
        for actual, expected in zip(hover_target, expected_hover):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_missed_record_keeps_yaw_sequence_analysis_fields(self):
        args = Namespace(base_frame="base_link", tool_frame="tool0", camera_frame="camera_link")
        targets = {
            "targets": {
                "yaw_p90deg": {
                    "yaw_deg": 90.0,
                    "tool0_position": [0.2, 0.1, 0.35],
                    "tool0_quat": vertical_down_quaternion_for_yaw(90.0),
                }
            }
        }
        record = build_missed_record(
            args,
            90.0,
            "no selected detection with base_link point",
            ([0.2, 0.1, 0.35], vertical_down_quaternion_for_yaw(90.0)),
            ([0.21, 0.1, 0.35], [0.0, 0.0, 0.0, 1.0]),
            targets,
            attempts=5,
        )
        self.assertEqual(record["status"], "missed_detection")
        self.assertEqual(record["target_pose_key"], "yaw_p90deg")
        self.assertEqual(record["attempts"], 5)
        self.assertIsNone(record["point_camera_xyz"])
        self.assertIsNone(record["point_base_xyz"])
        self.assertEqual(record["tool0_position"], [0.2, 0.1, 0.35])
        self.assertIn("tool0_pose", record)
        self.assertIn("camera_link_pose", record)

    def test_pose_check_rejects_unreached_yaw_before_recording_detection(self):
        args = Namespace(
            pose_check_orientation_deg=2.0,
            pose_check_z_axis_deg=1.0,
            pose_check_position_m=0.005,
            base_frame="base_link",
            tool_frame="tool0",
            camera_frame="camera_link",
        )
        actual_tool_pose = (
            [0.20030135854089293, 0.11453881561230728, 0.34656066121268264],
            [0.9947524791217777, 0.10214189507605909, -0.005865867397520513, -0.0003607644985802385],
        )
        target_pose = {
            "tool0_position": [0.200662266267, 0.114755155944, 0.351099530149],
            "tool0_quat": vertical_down_quaternion_for_yaw(0.0),
        }
        check = pose_check(args, actual_tool_pose, target_pose)
        self.assertFalse(check["ok"])
        self.assertGreater(check["orientation_error_deg"], 11.0)
        targets = {"targets": {"yaw_p0deg": target_pose}}
        record = build_pose_failed_record(
            args,
            0.0,
            actual_tool_pose,
            ([0.278, 0.162, 0.344], [0.0, 0.0, 0.0, 1.0]),
            targets,
            check,
            attempts=1,
        )
        self.assertEqual(record["status"], "pose_failed")
        self.assertEqual(record["target_pose_key"], "yaw_p0deg")
        self.assertIn("pose_check", record)
        self.assertIn("tool0_pose", record)
        self.assertIn("camera_link_pose", record)

    def test_default_pose_threshold_accepts_small_yaw_error_sample(self):
        args = Namespace(
            pose_check_orientation_deg=5.0,
            pose_check_z_axis_deg=1.0,
            pose_check_position_m=0.005,
        )
        actual_tool_pose = (
            [0.2013341450464915, 0.11448046992861949, 0.35064109061891024],
            [0.9995013554717989, -0.03156292504364329, 0.00087828710952241, 0.000225354119979848],
        )
        target_pose = {
            "tool0_position": [0.200662266267, 0.114755155944, 0.351099530149],
            "tool0_quat": vertical_down_quaternion_for_yaw(0.0),
        }
        check = pose_check(args, actual_tool_pose, target_pose)
        self.assertTrue(check["ok"])

    def test_default_pose_orientation_threshold_is_five_degrees(self):
        args = parse_args(["--output-dir", "/tmp/yaw_probe_args_test"])
        self.assertEqual(args.pose_check_orientation_deg, 5.0)

    def test_select_detection_falls_back_to_single_base_valid_detection(self):
        detections = [
            {
                "id": 3,
                "label": "red_block",
                "confidence": 0.91,
                "area_px": 1200,
                "point_base_xyz": [0.2, 0.1, 0.03],
            }
        ]

        selected = select_detection(detections, "yellow")

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], 3)
        self.assertIn("selection_warning", selected)

    def test_selection_failure_reason_reports_label_filter_mismatch(self):
        detections = [
            {
                "id": 1,
                "label": "red_block",
                "point_base_xyz": [0.2, 0.1, 0.03],
            },
            {
                "id": 2,
                "label": "green_block",
                "point_base_xyz": [0.3, 0.1, 0.03],
            },
        ]

        reason = selection_failure_reason(detections, "yellow")

        self.assertIn("none matches target label filter 'yellow'", reason)

    def test_yaw_motion_variants_include_pose_fallback(self):
        args = Namespace(
            pre_rotate_strategy="joint-wrist3",
            pre_rotate_wrist_yaw_sign="negative",
            pre_rotate_wrist_direction="auto",
        )
        variants = yaw_motion_arg_variants(args)
        keys = [
            (
                variant.pre_rotate_strategy,
                variant.pre_rotate_wrist_yaw_sign,
                variant.pre_rotate_wrist_direction,
            )
            for variant in variants
        ]
        self.assertIn(("joint-wrist3", "auto", "auto"), keys)
        self.assertIn(("joint-wrist3", "positive", "auto"), keys)
        self.assertIn(("pose", "negative", "auto"), keys)


if __name__ == "__main__":
    unittest.main()
