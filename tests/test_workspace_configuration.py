import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from tools.workflows.stack_demo.commands import plan_only_command
from tools.workflows.stack_demo.workspace import attach_configured_workspace, load_workspace_bounds


class WorkspaceConfigurationTests(unittest.TestCase):
    def test_loads_six_direction_base_link_meter_bounds(self):
        path = self._write({
            "frame_id": "base_link", "unit": "meter",
            "bounds": {
                "xmin": 0.2, "xmax": 0.5, "ymin": 0.0,
                "ymax": 0.3, "zmin": -0.03, "zmax": 0.25,
            },
        })
        bounds = load_workspace_bounds(path)
        self.assertEqual(bounds["xmin"], 0.2)
        self.assertEqual(bounds["zmax"], 0.25)

    def test_rejects_wrong_frame_or_missing_vertical_bounds(self):
        wrong_frame = self._write({
            "frame_id": "camera_link", "unit": "meter",
            "bounds": {key: value for key, value in (
                ("xmin", 0), ("xmax", 1), ("ymin", 0),
                ("ymax", 1), ("zmin", 0), ("zmax", 1),
            )},
        })
        with self.assertRaisesRegex(RuntimeError, "Workspace frame"):
            load_workspace_bounds(wrong_frame)
        missing_z = self._write({
            "frame_id": "base_link", "unit": "meter",
            "bounds": {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
        })
        with self.assertRaisesRegex(RuntimeError, "zmin"):
            load_workspace_bounds(missing_z)

    def test_existing_offline_bounds_are_not_overwritten(self):
        original = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
        state = {"table_bounds": original}
        result = attach_configured_workspace(
            state, SimpleNamespace(workspace_bounds_json="missing.json", base_frame="base_link"),
        )
        self.assertIs(result["table_bounds"], original)

    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        with handle:
            json.dump(payload, handle)
        return handle.name


class CommandSafetyTests(unittest.TestCase):
    def test_plan_only_command_removes_execution_and_gripper_options(self):
        command = [
            "python3", "moveit_plan_preview.py", "--plan-json", "pick.json",
            "--enable-gripper", "--gripper-port", "/dev/ttyUSB0",
            "--skip-gripper-init", "--close-gripper-for-push", "--execute", "--yes",
        ]
        result = plan_only_command(command)
        self.assertNotIn("--execute", result)
        self.assertNotIn("--enable-gripper", result)
        self.assertNotIn("--skip-gripper-init", result)
        self.assertNotIn("--close-gripper-for-push", result)
        self.assertNotIn("--gripper-port", result)
        self.assertNotIn("/dev/ttyUSB0", result)
        self.assertIn("--yes", result)


if __name__ == "__main__":
    unittest.main()
