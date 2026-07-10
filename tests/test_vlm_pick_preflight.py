import tempfile
import unittest
from types import SimpleNamespace

from tools.workflows.stack_demo import pick_preflight


class VlmPickPreflightTests(unittest.TestCase):
    def test_pick_moveit_preflight_failure_is_fail_safe(self):
        originals = (
            pick_preflight.build_offline_pick_plan,
            pick_preflight.pick_command,
            pick_preflight.run,
        )
        try:
            pick_preflight.build_offline_pick_plan = self._write_fake_plan
            pick_preflight.pick_command = lambda _args, path: ["moveit", "--plan-json", path, "--execute"]
            pick_preflight.run = lambda _command: self._raise_planning_failure()
            with tempfile.TemporaryDirectory() as output_dir:
                result = pick_preflight.preflight_pick_action(
                    SimpleNamespace(),
                    output_dir,
                    _scene(),
                    {"action_type": "pick", "object_id": 7, "selected_grasp_yaw_deg": 15.0},
                    1,
                )
        finally:
            (
                pick_preflight.build_offline_pick_plan,
                pick_preflight.pick_command,
                pick_preflight.run,
            ) = originals

        self.assertFalse(result["moveit_feasible"])
        self.assertFalse(result["executable_safe"])
        self.assertIn("planning failed", result["moveit_preflight_error"])

    @staticmethod
    def _write_fake_plan(_state, _obj, output_path, _args):
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        return {"steps": []}

    @staticmethod
    def _raise_planning_failure():
        raise RuntimeError("planning failed")


def _scene():
    return {
        "objects": [
            {
                "id": 7,
                "label": "green block",
                "geometry_frame": "base_link",
                "geometry_center_m": [0.2, 0.0, 0.03],
                "dimensions_m": [0.04, 0.04, 0.04],
            }
        ]
    }


if __name__ == "__main__":
    unittest.main()
