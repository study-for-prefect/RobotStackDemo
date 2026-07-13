import tempfile
import unittest
from types import SimpleNamespace

from robot_scene_pipeline.object_tracking import update_scene_tracks
from robot_scene_pipeline.stack_binding import (
    deterministic_unique_stack_binding,
    order_fingerprint,
    stack_binding_selection_is_ambiguous,
    validate_stack_binding,
)
from robot_scene_pipeline.vlm_replanning import VlmReplanningExhausted
from tools.workflows.stack_demo import scene as stack_scene


INSTRUCTION = "红绿蓝黄依次向上堆叠"


class StackBindingProtocolTests(unittest.TestCase):
    def test_selected_by_color_requires_all_four_slots(self):
        state = _state()
        decision = _decision(state)
        del decision["selected_by_color"]["yellow"]
        with self.assertRaisesRegex(ValueError, "all four"):
            validate_stack_binding(decision, state, INSTRUCTION)

    def test_track_ids_must_be_unique(self):
        state = _state()
        decision = _decision(state)
        decision["selected_by_color"]["yellow"] = dict(decision["selected_by_color"]["red"])
        with self.assertRaises(ValueError):
            validate_stack_binding(decision, state, INSTRUCTION)

    def test_actual_color_must_match_slot(self):
        state = _state()
        decision = _decision(state)
        decision["selected_by_color"]["red"] = dict(decision["selected_by_color"]["green"])
        with self.assertRaisesRegex(ValueError, "actual label"):
            validate_stack_binding(decision, state, INSTRUCTION)

    def test_same_failed_order_is_not_fully_validated_twice(self):
        state = _state(two_yellow=True)
        invalid = _decision(state)
        invalid["selected_by_color"]["red"] = dict(invalid["selected_by_color"]["green"])
        calls = []

        def fake_call(*_args, **_kwargs):
            return {"call_status": "parsed", "decision": invalid}

        original_call = stack_scene.call_vlm_stack_policy
        original_validate = stack_scene.validate_vlm_stack_decision
        stack_scene.call_vlm_stack_policy = fake_call

        def counted_validate(*args, **kwargs):
            calls.append(1)
            return original_validate(*args, **kwargs)

        stack_scene.validate_vlm_stack_decision = counted_validate
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                with self.assertRaises(VlmReplanningExhausted):
                    stack_scene._call_initial_vlm_stack_decision(
                        SimpleNamespace(instruction=INSTRUCTION, max_vlm_stack_attempts=2),
                        state, output_dir,
                    )
        finally:
            stack_scene.call_vlm_stack_policy = original_call
            stack_scene.validate_vlm_stack_decision = original_validate
        self.assertEqual(len(calls), 1)

    def test_two_yellow_candidates_leave_instance_choice_to_model(self):
        state = _state(two_yellow=True)
        self.assertTrue(stack_binding_selection_is_ambiguous(state, INSTRUCTION))
        first = validate_stack_binding(_decision(state, yellow_index=3), state, INSTRUCTION)
        second = validate_stack_binding(_decision(state, yellow_index=4), state, INSTRUCTION)
        self.assertNotEqual(first["order_fingerprint"], second["order_fingerprint"])

    def test_unique_combination_uses_deterministic_mapping(self):
        state = _state()
        selected = deterministic_unique_stack_binding(state, INSTRUCTION)
        self.assertEqual(selected["full_stack_order"], [1, 2, 3, 4])
        self.assertEqual(selected["decision_source"], "stack_binding_deterministic_unique_fallback")

    def test_multiple_combinations_are_never_randomly_selected(self):
        state = _state(two_yellow=True)
        self.assertIsNone(deterministic_unique_stack_binding(state, INSTRUCTION))


def _state(two_yellow=False):
    objects = [
        _object(1, "square red"), _object(2, "square green"),
        _object(3, "square blue"), _object(4, "square yellow"),
    ]
    if two_yellow:
        objects.append(_object(5, "square yellow", x=0.2))
    state = {"scene_revision": 1, "objects": objects}
    update_scene_tracks({}, objects, 1)
    return state


def _decision(state, yellow_index=3):
    by_color = {
        "red": state["objects"][0], "green": state["objects"][1],
        "blue": state["objects"][2], "yellow": state["objects"][yellow_index],
    }
    return {
        "schema_version": "stack_binding_v2", "structure": "tower", "strategy": "vertical_stack",
        "selected_by_color": {
            color: {"object_ref": obj["object_ref"], "track_id": obj["track_id"]}
            for color, obj in by_color.items()
        },
        "reason": "select instances", "confidence": 0.9,
    }


def _object(object_id, label, x=0.1):
    return {
        "id": object_id, "label": label,
        "geometry_center_m": [x, object_id * 0.05, 0.02],
        "dimensions_m": [0.03, 0.03, 0.04],
    }


if __name__ == "__main__":
    unittest.main()
