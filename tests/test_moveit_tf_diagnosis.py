import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_tf_node_module():
    if "rclpy" not in sys.modules:
        rclpy = types.ModuleType("rclpy")
        rclpy_duration = types.ModuleType("rclpy.duration")
        rclpy_node = types.ModuleType("rclpy.node")
        rclpy_time = types.ModuleType("rclpy.time")

        class Duration:
            def __init__(self, seconds=0.0):
                self.seconds = seconds

        class Node:
            pass

        class Time:
            pass

        rclpy_duration.Duration = Duration
        rclpy_node.Node = Node
        rclpy_time.Time = Time
        sys.modules["rclpy"] = rclpy
        sys.modules["rclpy.duration"] = rclpy_duration
        sys.modules["rclpy.node"] = rclpy_node
        sys.modules["rclpy.time"] = rclpy_time

    if "pymoveit2" not in sys.modules:
        pymoveit2 = types.ModuleType("pymoveit2")
        pymoveit2_robots = types.ModuleType("pymoveit2.robots")
        ur = types.SimpleNamespace()
        pymoveit2.MoveIt2 = object
        pymoveit2_robots.ur = ur
        sys.modules["pymoveit2"] = pymoveit2
        sys.modules["pymoveit2.robots"] = pymoveit2_robots

    if "sensor_msgs.msg" not in sys.modules:
        sensor_msgs = types.ModuleType("sensor_msgs")
        sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
        sensor_msgs_msg.JointState = object
        sensor_msgs.msg = sensor_msgs_msg
        sys.modules["sensor_msgs"] = sensor_msgs
        sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    if "tf2_ros" not in sys.modules:
        tf2_ros = types.ModuleType("tf2_ros")
        tf2_ros.Buffer = object
        tf2_ros.TransformListener = object
        sys.modules["tf2_ros"] = tf2_ros

    module_path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "robot"
        / "moveit_preview"
        / "tf_node.py"
    )
    spec = importlib.util.spec_from_file_location("moveit_tf_node_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tf_node = _load_tf_node_module()


class MoveItTfDiagnosisTests(unittest.TestCase):
    def test_parent_frames_and_nearby_frames_are_reported(self):
        frames_yaml = """
camera_link:
  parent: 'wrist_3_link'
camera_color_optical_frame:
  parent: 'camera_color_frame'
tool0_controller:
  parent: 'base'
"""
        frames = tf_node.tf_frame_names(frames_yaml)
        self.assertIn("base", frames)

        message = tf_node.tf_tree_diagnosis(frames_yaml, "base_link", "tool0")
        self.assertIn("Missing exact TF frame candidates", message)
        self.assertIn("nearby_base_frames=['base']", message)
        self.assertIn("nearby_tool_frames=['tool0_controller']", message)


if __name__ == "__main__":
    unittest.main()
