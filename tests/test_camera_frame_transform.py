import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from robot_scene_pipeline.ros_topic_capture import CameraIntrinsics
from robot_scene_pipeline.tabletop_geometry import deproject_depth_roi
from robot_scene_pipeline.instance_pointcloud import transform_points
from robot_scene_pipeline.tf_transform import apply_transform, attach_base_coordinates
from tools.monitoring.realtime_monitor.transforms import optical_to_camera_link


class ConstantDepthFrame:
    def get_width(self):
        return 2

    def get_height(self):
        return 2

    def get_distance(self, x, y):
        return 1.0


class CameraFrameTransformTests(unittest.TestCase):
    def test_optical_to_camera_link_axis_direction(self):
        point_optical = np.array([0.10, 0.20, 0.80])

        np.testing.assert_allclose(
            optical_to_camera_link(point_optical),
            [0.80, -0.10, -0.20],
            atol=1e-12,
        )

    def test_pointcloud_transform_applies_optical_conversion_once(self):
        points_optical = np.array([[0.10, 0.20, 0.80]], dtype=float)
        identity = np.eye(4, dtype=float)

        converted = transform_points(
            points_optical,
            matrix=identity,
            point_mode="optical-to-camera-link",
        )
        direct = transform_points(converted, matrix=identity, point_mode="direct")

        np.testing.assert_allclose(converted, [[0.80, -0.10, -0.20]], atol=1e-12)
        np.testing.assert_allclose(direct, converted, atol=1e-12)

    def test_pointcloud_transform_default_is_direct_optical_frame(self):
        points_optical = np.array([[0.10, 0.20, 0.80]], dtype=float)
        identity = np.eye(4, dtype=float)

        transformed = transform_points(points_optical, matrix=identity)

        np.testing.assert_allclose(transformed, points_optical, atol=1e-12)

    def test_direct_tf_uses_optical_frame_matrix_without_axis_conversion(self):
        matrix = np.eye(4, dtype=float)
        matrix[:3, 3] = [0.3, -0.2, 0.1]
        point_optical = [0.10, 0.20, 0.80]

        transformed = apply_transform(matrix, point_optical)

        np.testing.assert_allclose(transformed, [0.40, 0.0, 0.90], atol=1e-12)

    def test_tf_json_child_frame_must_match_requested_camera_frame(self):
        payload = {
            "parent_frame": "base_link",
            "child_frame": "camera_link",
            "matrix_4x4": np.eye(4, dtype=float).tolist(),
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
            json.dump(payload, tmp)
            tmp.flush()

            with self.assertRaisesRegex(RuntimeError, "child_frame mismatch"):
                attach_base_coordinates(
                    [{"center_3d_m": [0.10, 0.20, 0.80]}],
                    "base_link",
                    "camera_depth_optical_frame",
                    0.1,
                    tf_json=tmp.name,
                    point_mode="direct",
                )

    def test_direct_point_mode_rejects_camera_link_frame(self):
        with self.assertRaisesRegex(RuntimeError, "direct requires an optical camera frame"):
            attach_base_coordinates(
                [{"center_3d_m": [0.10, 0.20, 0.80]}],
                "base_link",
                "camera_link",
                0.1,
                tf_json="",
                point_mode="direct",
            )

    def test_ros_topic_intrinsics_do_not_call_realsense_sdk_deproject(self):
        class NativeIntrinsics:
            pass

        class FakeRealSense:
            intrinsics = NativeIntrinsics

            @staticmethod
            def rs2_deproject_pixel_to_point(intrinsics, pixel, depth):
                raise AssertionError("custom ROS intrinsics must use pinhole fallback")

        intrinsics = CameraIntrinsics(
            width=2,
            height=2,
            fx=2.0,
            fy=2.0,
            ppx=0.0,
            ppy=0.0,
            model="plumb_bob",
            coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        )

        with mock.patch.dict("sys.modules", {"pyrealsense2": FakeRealSense}):
            points = deproject_depth_roi(
                ConstantDepthFrame(),
                intrinsics,
                roi=(0, 0, 1, 1),
                stride=1,
                max_depth_m=2.0,
            )

        np.testing.assert_allclose(
            points,
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0],
                [0.0, 0.5, 1.0],
                [0.5, 0.5, 1.0],
            ],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
