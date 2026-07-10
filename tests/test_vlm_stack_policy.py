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
                    "full_stack_order": [1, 2],
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

    def test_conflicting_legacy_fields_are_ignored_and_derived(self):
        validated = validate_vlm_stack_decision(
            {
                "call_status": "parsed",
                "decision": {
                    "task_type": "stack_blocks",
                    "base_object_id": 99,
                    "full_stack_order": [1, 2],
                    "stack_order": [1, 99],
                    "structure_plan": {},
                    "object_bindings": _bindings(_state(), [1, 2]),
                    "reason": "full order is authoritative",
                    "confidence": 0.8,
                },
            },
            _state(),
        )

        self.assertEqual(validated["base_object_id"], 1)
        self.assertEqual(validated["stack_order"], [2])

    def test_missing_required_blue_returns_structured_feedback(self):
        state = {
            "objects": [
                {"id": 0, "label": "square red", "geometry_center_m": [0, 0, 0]},
                {"id": 1, "label": "square green", "geometry_center_m": [0.1, 0, 0]},
                {"id": 2, "label": "square yellow", "geometry_center_m": [0.2, 0, 0]},
                {"id": 3, "label": "square blue", "geometry_center_m": [0.3, 0, 0]},
                {"id": 4, "label": "square yellow", "geometry_center_m": [0.4, 0, 0]},
            ]
        }
        with self.assertRaises(ValueError) as raised:
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "full_stack_order": [0, 1, 2, 4],
                        "structure_plan": {},
                        "object_bindings": _bindings(state, [0, 1, 2, 4]),
                        "reason": "incorrect order",
                        "confidence": 0.7,
                    },
                },
                state,
                "以红色为底，再放绿色、蓝色、黄色",
            )

        feedback = raised.exception.feedback
        error_types = {item["type"] for item in feedback["errors"]}
        self.assertIn("missing_required_label", error_types)
        self.assertIn("duplicate_label", error_types)

    def test_repeated_base_in_authoritative_order_returns_feedback(self):
        with self.assertRaises(ValueError) as raised:
            validate_vlm_stack_decision(
                {
                    "call_status": "parsed",
                    "decision": {
                        "task_type": "stack_blocks",
                        "full_stack_order": [1, 2, 1],
                        "structure_plan": {},
                        "object_bindings": _bindings(_state(), [1, 2]),
                        "reason": "base repeated",
                        "confidence": 0.5,
                    },
                },
                _state(),
            )

        self.assertEqual(raised.exception.feedback["errors"][0]["type"], "duplicate_object_id")

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
        self.assertNotIn("base_object_id", payload["output_schema"])
        self.assertNotIn("stack_order", payload["output_schema"])

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
