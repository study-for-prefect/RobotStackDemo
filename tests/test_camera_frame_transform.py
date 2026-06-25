import unittest

import numpy as np

from robot_scene_pipeline.instance_pointcloud import transform_points
from tools.monitoring.realtime_monitor.transforms import optical_to_camera_link


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


if __name__ == "__main__":
    unittest.main()
