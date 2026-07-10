import unittest

from robot_scene_pipeline.vlm_stack_policy import (
    build_vlm_stack_decision_input,
    validate_vlm_stack_decision,
)


class VlmStackPolicyTests(unittest.TestCase):
    def test_valid_stack_decision_passes(self):
        validated = validate_vlm_stack_decision(
            {
                "call_status": "parsed",
                "decision": {
                    "task_type": "stack_blocks",
                    "base_object_id": 1,
                    "full_stack_order": [1, 2],
                    "stack_order": [2],
                    "reason": "blue base red top",
                    "confidence": 0.9,
                },
            },
            _state(),
            "把红色积木放到蓝色积木上",
        )

        self.assertEqual(validated["base_object_id"], 1)
        self.assertEqual(validated["stack_order"], [2])
        self.assertEqual(validated["decision_source"], "vlm_stack_policy")

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "base_object_id": 1,
                        "full_stack_order": [1, 99],
                        "stack_order": [99],
                    },
                },
                _state(),
                "stack blocks",
            )

    def test_base_repeated_in_stack_order_rejected(self):
        with self.assertRaises(ValueError):
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "base_object_id": 1,
                        "full_stack_order": [1, 2],
                        "stack_order": [1, 2],
                    },
                },
                _state(),
                "stack blocks",
            )

    def test_missing_explicit_target_rejected(self):
        with self.assertRaises(ValueError):
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "base_object_id": 1,
                        "full_stack_order": [1],
                        "stack_order": [],
                        "reason": "forgot red target",
                    },
                },
                _state(),
                "把红色积木放到蓝色积木上",
            )

    def test_explicit_instruction_order_conflict_rejected(self):
        state = {
            "objects": [
                {"id": 0, "label": "square red", "geometry_center_m": [0, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
                {"id": 1, "label": "square green", "geometry_center_m": [0.1, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
                {"id": 2, "label": "square yellow", "geometry_center_m": [0.2, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
                {"id": 3, "label": "square blue", "geometry_center_m": [0.3, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
            ],
        }

        with self.assertRaises(ValueError):
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "base_object_id": 0,
                        "full_stack_order": [0, 1, 2, 3],
                        "stack_order": [1, 2, 3],
                        "reason": "text says green blue yellow but array is green yellow blue",
                    },
                },
                state,
                "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面，再把黄色积木放到蓝色上面",
            )

    def test_stack_input_does_not_include_camera_intrinsics(self):
        payload = build_vlm_stack_decision_input(_state(), "stack blocks")
        encoded = str(payload)

        self.assertNotIn("camera_profile", encoded)
        self.assertNotIn("fx", encoded)
        self.assertIn("geometry_center_base_m", encoded)

    def test_stack_input_lists_same_label_instances(self):
        state = _state()
        state["objects"].append(
            {
                "id": 3,
                "label": "red block",
                "geometry_center_m": [0.16, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            }
        )

        payload = build_vlm_stack_decision_input(state, "stack blocks")

        groups = payload["scene_integrity"]["same_label_instance_groups"]
        self.assertEqual(groups[0]["label"], "red block")
        self.assertEqual([item["id"] for item in groups[0]["instances"]], [2, 3])

    def test_stack_rejects_duplicate_scene_object_ids(self):
        state = _state()
        state["objects"][1]["id"] = 1

        with self.assertRaises(ValueError):
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "base_object_id": 1,
                        "full_stack_order": [1],
                        "stack_order": [],
                    },
                },
                state,
                "stack blocks",
            )


def _state():
    return {
        "snapshot_image": "/tmp/snapshot.jpg",
        "annotated_image": "/tmp/annotated.jpg",
        "objects": [
            {
                "id": 1,
                "label": "blue block",
                "confidence": 0.9,
                "geometry_frame": "base_link",
                "geometry_center_m": [0.0, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            },
            {
                "id": 2,
                "label": "red block",
                "confidence": 0.9,
                "geometry_frame": "base_link",
                "geometry_center_m": [0.08, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
