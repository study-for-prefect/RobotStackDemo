#!/usr/bin/env python3

import json
import os
import tempfile
import unittest

from tools.workflows.two_stage_pick.planning import build_tcp_error_corrected_plan


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def plan_with_geometry(center, target):
    return {
        "steps": [
            {
                "step": 1,
                "action": "pick",
                "status": "planned",
                "target_position_m": list(target),
                "approach_position_m": [target[0], target[1], target[2] + 0.10],
                "geometry_center_m": list(center),
                "geometry_frame": "base_link",
                "pointcloud_geometry_valid": True,
            }
        ]
    }


class TwoStageTcpCorrectionTest(unittest.TestCase):
    def test_second_snapshot_measures_tcp_target_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_plan = os.path.join(tmpdir, "first.json")
            second_plan = os.path.join(tmpdir, "second.json")
            corrected_plan = os.path.join(tmpdir, "corrected.json")
            report_path = os.path.join(tmpdir, "report.json")
            tf_json = os.path.join(tmpdir, "tf.json")

            write_json(first_plan, plan_with_geometry([0.400, 0.100, 0.020], [0.405, 0.098, 0.030]))
            write_json(second_plan, plan_with_geometry([0.402, 0.099, 0.020], [9.0, 9.0, 9.0]))
            write_json(
                tf_json,
                {
                    "tool_transform": {
                        "matrix_4x4": [
                            [1.0, 0.0, 0.0, 0.406],
                            [0.0, 1.0, 0.0, 0.096],
                            [0.0, 0.0, 1.0, -0.030],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    }
                },
            )

            build_tcp_error_corrected_plan(
                first_plan,
                second_plan,
                tf_json,
                corrected_plan,
                report_path,
                max_correction_m=0.006,
                max_grasp_offset_m=0.05,
                tcp_offset_tool=[0.0, 0.0, 0.15],
            )

            with open(corrected_plan, "r", encoding="utf-8") as f:
                corrected = json.load(f)["steps"][0]
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

        self.assertEqual(corrected["target_position_m"], [0.406, 0.099, 0.03])
        self.assertEqual(corrected["coordinate_source"], "locked_plan_plus_second_snapshot_tcp_target_error")
        self.assertFalse(report["used_second_absolute_target"])
        self.assertAlmostEqual(report["desired_tcp_to_target_xy_m"][0], 0.005)
        self.assertAlmostEqual(report["desired_tcp_to_target_xy_m"][1], -0.002)
        self.assertAlmostEqual(report["observed_tcp_to_target_xy_m"][0], 0.004)
        self.assertAlmostEqual(report["observed_tcp_to_target_xy_m"][1], -0.003)


if __name__ == "__main__":
    unittest.main()
