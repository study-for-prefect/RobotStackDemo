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
                    "structure_plan": {},
                    "object_bindings": _bindings(_state(), [1, 2]),
                    "reason": "blue base red top",
                    "confidence": 0.9,
                },
            },
            _state(),
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
            )

    def test_stack_binding_label_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "base_object_id": 1,
                        "full_stack_order": [1, 2],
                        "stack_order": [2],
                        "structure_plan": {},
                        "object_bindings": [
                            {"object_id": 1, "observed_label": "red block", "geometry_center_base_m": [0.0, 0.0, 0.02]},
                            {"object_id": 2, "observed_label": "red block", "geometry_center_base_m": [0.08, 0.0, 0.02]},
                        ],
                        "reason": "mismatched id and label",
                        "confidence": 0.5,
                    },
                },
                _state(),
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
            )

    def test_code_does_not_repair_vlm_plan_from_instruction_rules(self):
        validated = validate_vlm_stack_decision(
            {
                "call_status": "parsed",
                "decision": {
                    "task_type": "stack_blocks",
                    "base_object_id": 1,
                    "full_stack_order": [1],
                    "stack_order": [],
                    "structure_plan": {},
                    "object_bindings": _bindings(_state(), [1]),
                    "reason": "VLM chose no placement",
                    "confidence": 0.4,
                },
            },
            _state(),
        )

        self.assertEqual(validated["full_stack_order"], [1])
        self.assertNotIn("explicit_rule_repair", validated)

    def test_code_does_not_override_vlm_order_with_color_parser(self):
        state = {
            "objects": [
                {"id": 0, "label": "square red", "geometry_center_m": [0, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
                {"id": 1, "label": "square green", "geometry_center_m": [0.1, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
                {"id": 2, "label": "square yellow", "geometry_center_m": [0.2, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
                {"id": 3, "label": "square blue", "geometry_center_m": [0.3, 0, 0], "dimensions_m": [0.03, 0.03, 0.03]},
            ],
        }

        validated = validate_vlm_stack_decision(
            {
                "call_status": "parsed",
                "decision": {
                    "task_type": "stack_blocks",
                    "base_object_id": 0,
                    "full_stack_order": [0, 1, 2, 3],
                    "stack_order": [1, 2, 3],
                    "structure_plan": {},
                    "object_bindings": _bindings(state, [0, 1, 2, 3]),
                    "reason": "VLM owns task interpretation",
                    "confidence": 0.7,
                },
            },
            state,
        )

        self.assertEqual(validated["full_stack_order"], [0, 1, 2, 3])
        self.assertNotIn("explicit_rule_diagnostic", validated)

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


def _bindings(state, object_ids):
    object_map = {obj["id"]: obj for obj in state["objects"]}
    return [
        {
            "object_id": object_id,
            "observed_label": object_map[object_id]["label"],
            "geometry_center_base_m": object_map[object_id]["geometry_center_m"],
        }
        for object_id in object_ids
    ]


if __name__ == "__main__":
    unittest.main()
