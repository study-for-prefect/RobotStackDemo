import unittest

from tools.robot.tf_lookup_json import tf_tree_diagnosis


STATIC_ONLY_FRAMES = """
camera_link:
  parent: 'wrist_3_link'
base_link:
  parent: 'world'
base:
  parent: 'base_link'
tool0:
  parent: 'flange'
flange:
  parent: 'wrist_3_link'
camera_color_frame:
  parent: 'camera_link'
camera_color_optical_frame:
  parent: 'camera_color_frame'
"""


class TfLookupDiagnosisTests(unittest.TestCase):
    def test_diagnosis_reports_missing_ur_dynamic_chain(self):
        message = tf_tree_diagnosis(
            STATIC_ONLY_FRAMES,
            base_frame="base_link",
            camera_frame="camera_color_optical_frame",
            tool_frame="tool0",
            require_tool=True,
        )

        self.assertIn("base component", message)
        self.assertIn("camera component", message)
        self.assertIn("tool component", message)
        self.assertIn("missing UR dynamic chain frames", message)
        self.assertIn("base_link to wrist_3_link is not connected", message)
        self.assertIn("ros2 topic echo /joint_states --once", message)

    def test_diagnosis_reports_missing_requested_frame(self):
        message = tf_tree_diagnosis(
            STATIC_ONLY_FRAMES,
            base_frame="map",
            camera_frame="camera_color_optical_frame",
            tool_frame="tool0",
            require_tool=True,
        )

        self.assertIn("Missing requested TF frame candidates", message)
        self.assertIn("base_frame=map", message)


if __name__ == "__main__":
    unittest.main()
