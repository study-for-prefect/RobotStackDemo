#!/usr/bin/env python3

import copy
import math
import unittest

import numpy as np

from tools.calibration.xy_bias_diagnosis import (
    analyze_dataset,
    build_stack_calibration,
    fit_workspace_model,
    trial_key,
)


def yaw_dataset(base_error, local_error):
    samples = []
    for trial_index, yaw_deg in enumerate((0.0, 90.0, 180.0, -90.0)):
        angle = math.radians(yaw_deg)
        c = math.cos(angle)
        s = math.sin(angle)
        rotated = np.array(
            [
                c * local_error[0] - s * local_error[1],
                s * local_error[0] + c * local_error[1],
            ]
        )
        error_mm = 1000.0 * (np.asarray(base_error) + rotated)
        samples.append(
            {
                "trial_index": trial_index,
                "position_label": "fixed",
                "yaw_deg": yaw_deg,
                "repeat": 1,
                "visual_geometry_xy_m": [0.40, 0.05],
                "error_x_mm": error_mm[0],
                "error_y_mm": error_mm[1],
                "robot_goal_error_xy_mm": [0.1, -0.2],
            }
        )
    return {
        "phase": "yaw",
        "error_convention": "observed_gripper_center_minus_physical_object_center",
        "collection_offsets": {
            "tcp_offset_tool_m": [0.0, 0.0, 0.15],
        },
        "samples": samples + copy.deepcopy(samples),
    }


class XYBiasDiagnosisTest(unittest.TestCase):
    def test_yaw_model_separates_base_and_rotating_error(self):
        dataset = yaw_dataset([0.003, -0.002], [0.0012, 0.0007])
        report = analyze_dataset(dataset)
        model = report["yaw_model"]

        np.testing.assert_allclose(model["base_error_m"], [0.003, -0.002], atol=1e-12)
        np.testing.assert_allclose(
            model["yaw_local_error_m"],
            [0.0012, 0.0007],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model["suggested_base_offset_delta_m"],
            [-0.003, 0.002],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model["suggested_updated_tcp_offset_tool_m"],
            [-0.0012, -0.0007, 0.15],
            atol=1e-12,
        )
        self.assertIsNone(report["workspace_model"])

    def test_workspace_affine_has_correction_sign(self):
        xy = np.asarray(
            [
                [0.30, -0.10],
                [0.30, 0.10],
                [0.40, 0.00],
                [0.50, -0.10],
                [0.50, 0.10],
                [0.60, 0.00],
            ]
        )
        error_coefficients = np.asarray(
            [
                [0.01, -0.02, 0.003],
                [-0.012, 0.005, -0.002],
            ]
        )
        error = np.column_stack([xy, np.ones(len(xy))]).dot(error_coefficients.T)
        model = fit_workspace_model(xy, error)

        expected = np.eye(3)[:2] - error_coefficients
        np.testing.assert_allclose(model["suggested_affine_xy"], expected, atol=1e-12)

    def test_trial_key_is_stable_across_numeric_types(self):
        left = {"position_label": "fixed", "yaw_deg": 90, "repeat": 2}
        right = {"position_label": "fixed", "yaw_deg": 90.0, "repeat": 2}
        self.assertEqual(trial_key(left), trial_key(right))

    def test_build_stack_calibration_uses_unified_fields(self):
        report = analyze_dataset(yaw_dataset([0.003, -0.002], [0.0012, 0.0007]))
        calibration = build_stack_calibration(report)

        np.testing.assert_allclose(calibration["grasp_base_bias_m"], [-0.003, 0.002, 0.0], atol=1e-12)
        np.testing.assert_allclose(calibration["tcp_offset_tool_m"], [-0.0012, -0.0007, 0.15], atol=1e-12)
        self.assertEqual(calibration["place_base_bias_m"], [0.0, 0.0, 0.0])
        self.assertEqual(calibration["max_correction_m"], {"xy_m": 0.02, "z_m": 0.015})


if __name__ == "__main__":
    unittest.main()
