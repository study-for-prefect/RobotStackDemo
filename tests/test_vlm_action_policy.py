import json
import unittest
from types import SimpleNamespace

from robot_scene_pipeline import vlm_action_policy


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self._content}}


class VlmActionPolicyTests(unittest.TestCase):
    def test_legal_nudge_decision_validates_before_moveit(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(),
            _scene(),
            _target(),
            protected_ids=[3],
        )

        self.assertIsNotNone(selected)
        self.assertTrue(safety["accepted"])
        self.assertEqual(selected["action_type"], "nudge")
        self.assertEqual(selected["obstacle_id"], 2)
        self.assertEqual(selected["direction_base"], [1.0, 0.0, 0.0])

    def test_legal_pick_decision_validates_for_current_target(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            {
                "action_type": "pick",
                "object_id": 1,
                "target_object_id": 1,
                "reason": "target is graspable",
                "confidence": 0.9,
                "raw_decision": {},
            },
            _scene(),
            _target(),
            protected_ids=[3],
            analysis={"grasp_feasible": True, "selected_grasp_yaw_deg": 0.0},
        )

        self.assertIsNotNone(selected)
        self.assertTrue(safety["accepted"])
        self.assertEqual(selected["action_type"], "pick")
        self.assertEqual(selected["object_id"], 1)

    def test_invalid_object_id_rejected(self):
        decision = _decision(object_id=99)

        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            decision,
            _scene(),
            _target(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertFalse(safety["accepted"])
        self.assertIn("object_id_exists", safety["failed_fields"])

    def test_target_object_id_must_be_current_target_not_base(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            {
                "action_type": "pick",
                "object_id": 1,
                "target_object_id": 3,
                "reason": "wrongly used placement base as target",
                "confidence": 0.9,
                "raw_decision": {},
            },
            _scene(),
            _target(),
            protected_ids=[3],
            analysis={"grasp_feasible": True, "selected_grasp_yaw_deg": 0.0},
        )

        detail = safety["checks"]["target_object_id_matches_current_target"]["detail"]
        self.assertIsNone(selected)
        self.assertIn("target_object_id_matches_current_target", safety["failed_fields"])
        self.assertEqual(detail["expected_current_target_object_id"], 1)
        self.assertEqual(detail["actual_target_object_id"], 3)

    def test_locked_or_protected_object_rejected(self):
        scene = _scene()
        scene["objects"][1]["state"] = "locked"

        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(),
            scene,
            _target(),
            protected_ids=[2, 3],
        )

        self.assertIsNone(selected)
        self.assertFalse(safety["accepted"])
        self.assertIn("object_not_base_placed_locked_protected", safety["failed_fields"])

    def test_push_distance_too_large_rejected(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(push_distance_m=0.08),
            _scene(),
            _target(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertIn("push_distance_in_range", safety["failed_fields"])

    def test_unnormalized_direction_rejected(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(push_direction_base=[2.0, 0.0, 0.0]),
            _scene(),
            _target(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertIn("push_direction_base_unit_xy_vector", safety["failed_fields"])

    def test_moveit_preflight_failure_rejects_validated_nudge(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(),
            _scene(),
            _target(),
            protected_ids=[3],
        )

        final = vlm_action_policy.mark_moveit_result(selected, safety, False, error="planning failed")

        self.assertFalse(final["accepted"])
        self.assertFalse(final["moveit_feasible"])
        self.assertIn("moveit_preflight_passed", final["failed_fields"])
        self.assertEqual(final["reason"], "moveit_preflight_failed")

    def test_legal_pick_away_decision_validates_safe_place(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            {
                "action_type": "pick_away",
                "object_id": 2,
                "target_object_id": 1,
                "safe_place_center_base_m": [0.16, -0.16, 0.02],
                "reason": "remove blocker",
                "confidence": 0.7,
                "raw_decision": {},
            },
            _scene(),
            _target(),
            protected_ids=[3],
        )

        self.assertIsNotNone(selected)
        self.assertTrue(safety["accepted"])
        self.assertEqual(selected["action_type"], "pick_away")
        self.assertEqual(selected["safe_place_center_m"], [0.16, -0.16, 0.02])

    def test_pick_away_place_colliding_with_protected_rejected(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            {
                "action_type": "pick_away",
                "object_id": 2,
                "target_object_id": 1,
                "safe_place_center_base_m": [0.25, 0.20, 0.02],
                "reason": "bad place",
                "confidence": 0.7,
                "raw_decision": {},
            },
            _scene(),
            _target(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertIn("safe_place_avoids_visible_objects", safety["failed_fields"])

    def test_illegal_json_fail_safe_stops(self):
        original_post = vlm_action_policy.requests.post
        try:
            vlm_action_policy.requests.post = lambda *_args, **_kwargs: _FakeResponse("not json")
            raw = vlm_action_policy.call_vlm_action_policy(
                SimpleNamespace(model="test", ollama_url="http://test", timeout=1, num_predict=64, no_image=True),
                {"objects": [], "target_object": {}},
            )
        finally:
            vlm_action_policy.requests.post = original_post

        self.assertEqual(raw["call_status"], "fail_safe_stop")
        self.assertEqual(raw["decision"]["action_type"], "stop")
        self.assertIn("vlm_json_or_call_failed", raw["decision"]["reason"])

    def test_action_input_disambiguates_target_id_from_stack_reference(self):
        payload = vlm_action_policy.build_vlm_action_decision_input(
            "/tmp/scene.png",
            "/tmp/overlay.png",
            _scene(),
            _target(),
            {"grasp_feasible": True, "selected_grasp_yaw_deg": 0.0},
            protected_ids=[3],
            base_id=3,
            memory={"structure": {"base": "base_block", "current_top": "base_block", "placed_order": ["base_block"]}},
            step_index=1,
        )
        prompt = vlm_action_policy.build_vlm_action_prompt(payload)

        self.assertEqual(payload["current_task_target_object_id"], 1)
        self.assertIn("not target_object_id", payload["stack_reference"]["note"])
        self.assertIn("target_object_id 永远表示当前循环的 target_object.id", prompt)


def _scene():
    return {
        "base_frame": "base_link",
        "objects": [
            _target(),
            {
                "id": 2,
                "label": "red_block",
                "geometry_center_m": [0.10, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
                "role": "loose_movable",
                "state": "free",
                "pushable": True,
            },
            {
                "id": 3,
                "label": "base_block",
                "geometry_center_m": [0.25, 0.20, 0.02],
                "dimensions_m": [0.04, 0.04, 0.04],
                "role": "base",
                "state": "locked",
                "pushable": False,
            },
        ],
    }


def _target():
    return {
        "id": 1,
        "label": "blue_block",
        "geometry_center_m": [0.0, 0.0, 0.02],
        "dimensions_m": [0.03, 0.03, 0.04],
        "role": "loose_movable",
        "state": "free",
        "pushable": True,
    }


def _decision(
    object_id=2,
    target_object_id=1,
    push_direction_base=None,
    push_distance_m=0.02,
):
    return {
        "action_type": "nudge",
        "object_id": object_id,
        "target_object_id": target_object_id,
        "push_direction_base": push_direction_base or [1.0, 0.0, 0.0],
        "push_distance_m": push_distance_m,
        "reason": "clear target grasp corridor",
        "confidence": 0.8,
        "raw_decision": {},
    }


if __name__ == "__main__":
    unittest.main()
