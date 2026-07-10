import os
import tempfile
import unittest

from tools.workflows.stack_demo.push_flow import _ensure_unique_scene_object_ids


class PushFlowSceneIdTests(unittest.TestCase):
    def test_duplicate_snapshot_ids_are_reassigned_before_vlm_action(self):
        state = {
            "objects": [
                {"id": 0, "label": "square red", "geometry_center_m": [0.29, 0.17, 0.0]},
                {"id": 1, "label": "triangle", "geometry_center_m": [0.40, 0.10, 0.0]},
                {"id": 3, "label": "square blue", "geometry_center_m": [0.35, 0.03, 0.0]},
                {"id": 4, "label": "square green", "geometry_center_m": [0.35, 0.14, 0.0]},
                {"id": 0, "label": "square yellow", "geometry_center_m": [0.35, 0.16, 0.0]},
            ],
        }
        held_target = {"id": 0, "label": "square yellow", "geometry_center_m": [0.35, 0.16, 0.0]}

        with tempfile.TemporaryDirectory() as tmpdir:
            unique_state, unique_target = _ensure_unique_scene_object_ids(state, held_target, tmpdir)

            ids = [obj["id"] for obj in unique_state["objects"]]
            self.assertEqual(len(ids), len(set(str(value) for value in ids)))
            self.assertNotEqual(unique_target["id"], 0)
            self.assertEqual(unique_target["label"], "square yellow")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "scene_state_unique_object_ids.json")))


if __name__ == "__main__":
    unittest.main()
