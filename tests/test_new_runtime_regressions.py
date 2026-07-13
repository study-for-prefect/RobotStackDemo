import glob
import json
import os
import unittest

from robot_scene_pipeline.object_tracking import update_scene_tracks
from robot_scene_pipeline.stack_binding import (
    deterministic_unique_stack_binding, missing_stack_colors,
    stack_binding_selection_is_ambiguous,
)
from robot_scene_pipeline.task_routing import route_task_type
from robot_scene_pipeline.organize_scope import organize_scope_objects
from robot_scene_pipeline.task_semantic_validation import infer_object_shape


ROOT = os.path.dirname(os.path.dirname(__file__))
RUNTIME = os.path.join(ROOT, "runtime")
STACK_INSTRUCTION = "红绿蓝黄依次向上堆叠"


class NewRuntimeRegressionTests(unittest.TestCase):
    def test_qwen3_linear_stack_replays_known_correct_binding(self):
        state = _state("linear_stack_qwen3_30b_20260712_184333", "initial_order")
        update_scene_tracks({}, state["objects"], 1)
        selected = deterministic_unique_stack_binding(state, STACK_INSTRUCTION)
        self.assertEqual(selected["full_stack_order"], [3, 1, 2, 0])

    def test_qwen3_house_empty_content_is_recorded_as_generation_failure(self):
        paths = glob.glob(os.path.join(RUNTIME, "build_house_qwen3_30b_20260712_184615", "task_contract_attempt_*_raw.json"))
        outputs = [_load(path) for path in paths]
        self.assertTrue(outputs)
        self.assertTrue(all(item.get("raw_content") == "" for item in outputs))
        self.assertEqual(route_task_type("搭房子"), "build_house")

    def test_qwen3_organize_is_routed_without_house_schema(self):
        self.assertTrue(os.path.isdir(os.path.join(RUNTIME, "organize_blocks_qwen3_30b_20260712_184818")))
        self.assertEqual(route_task_type("按颜色整理积木"), "organize_blocks")

    def test_qwen25_multi_instance_stack_requires_model_selection_not_random_guess(self):
        state = _state("linear_stack_20260712_202038", "initial_order")
        update_scene_tracks({}, state["objects"], 1)
        self.assertEqual(missing_stack_colors(state, STACK_INSTRUCTION), [])
        self.assertTrue(stack_binding_selection_is_ambiguous(state, STACK_INSTRUCTION))
        self.assertIsNone(deterministic_unique_stack_binding(state, STACK_INSTRUCTION))
        outputs = glob.glob(os.path.join(RUNTIME, "linear_stack_20260712_202038", "initial_order_vlm", "vlm_stack_attempt_*_output.json"))
        self.assertTrue(any(len((_load(path).get("decision") or {}).get("full_stack_order", [])) == 3 for path in outputs))

    def test_qwen25_house_concave_label_is_a_roof_candidate(self):
        state = _state("build_house_20260712_202457", "initial_task_scene")
        concave = next(obj for obj in state["objects"] if obj.get("label") == "concave")
        self.assertEqual(infer_object_shape(concave), "concave_rectangle")

    def test_qwen25_organize_old_house_contract_is_now_route_mismatch(self):
        path = os.path.join(RUNTIME, "organize_blocks_20260712_202701", "task_contract_attempt_01_raw.json")
        old = _load(path)["decision"]
        self.assertEqual(old["task_type"], "build_house")
        self.assertEqual(route_task_type("按颜色整理积木"), "organize_blocks")

    def test_latest_organize_excludes_edge_and_non_color_false_detections(self):
        state = _state("organize_blocks_qwen3_30b_20260713_095215", "initial_task_scene")
        state["workspace_bounds"] = {
            "xmin": 0.235, "xmax": 0.443, "ymin": 0.03,
            "ymax": 0.322, "zmin": -0.03, "zmax": 0.25,
        }
        selected = organize_scope_objects(state)
        self.assertEqual([obj["id"] for obj in selected], [0, 1, 2, 3, 4, 5])

    def test_latest_organize_deduplicates_same_color_same_geometry(self):
        state = _state("organize_blocks_fixed_20260713_102035", "initial_task_scene")
        state["workspace_bounds"] = {
            "xmin": 0.235, "xmax": 0.443, "ymin": 0.03,
            "ymax": 0.322, "zmin": -0.03, "zmax": 0.25,
        }
        selected = organize_scope_objects(state)
        self.assertNotIn(9, [obj["id"] for obj in selected])
        self.assertIn(8, [obj["id"] for obj in selected])


def _state(runtime_name, observation_dir):
    path = os.path.join(RUNTIME, runtime_name, observation_dir, "private_scene_state.json")
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    state["scene_revision"] = 1
    return state


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    unittest.main()
