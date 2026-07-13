import json
import unittest
from types import SimpleNamespace

from robot_scene_pipeline import vlm_action_policy
from robot_scene_pipeline.ollama_policy_client import PolicyCallResult


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
            protected_ids=[3],
        )

        self.assertIsNotNone(selected)
        self.assertTrue(safety["accepted"])
        self.assertEqual(selected["action_type"], "nudge")
        self.assertEqual(selected["obstacle_id"], 2)
        self.assertEqual(selected["direction_base"], [1.0, 0.0, 0.0])

    def test_legal_pick_decision_runs_grasp_validation_after_vlm_choice(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            {
                "action_type": "pick",
                "object_id": 1,
                "object_label": "blue_block",
                "object_center_base_m": [0.0, 0.0, 0.02],
                "target_object_id": 1,
                "target_object_label": "blue_block",
                "target_object_center_base_m": [0.0, 0.0, 0.02],
                "reason": "target is graspable",
                "confidence": 0.9,
                "raw_decision": {},
            },
            _scene(),
            protected_ids=[3],
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
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertFalse(safety["accepted"])
        self.assertIn("object_id_exists", safety["failed_fields"])

    def test_target_object_id_is_vlm_chosen_and_only_checked_for_existence(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            {
                "action_type": "pick",
                "object_id": 1,
                "object_label": "blue_block",
                "object_center_base_m": [0.0, 0.0, 0.02],
                "target_object_id": 3,
                "target_object_label": "base_block",
                "target_object_center_base_m": [0.25, 0.20, 0.02],
                "reason": "VLM declares which task object the action advances",
                "confidence": 0.9,
                "raw_decision": {},
            },
            _scene(),
            protected_ids=[3],
        )

        self.assertIsNotNone(selected)
        self.assertTrue(safety["checks"]["target_object_id_exists"]["ok"])
        self.assertEqual(selected["target_object_id"], 3)

    def test_locked_or_protected_object_rejected(self):
        scene = _scene()
        scene["objects"][1]["state"] = "locked"

        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(),
            scene,
            protected_ids=[2, 3],
        )

        self.assertIsNone(selected)
        self.assertFalse(safety["accepted"])
        self.assertIn("object_not_base_placed_locked_protected", safety["failed_fields"])

    def test_duplicate_scene_object_ids_rejected(self):
        scene = _scene()
        scene["objects"][1]["id"] = 1

        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(),
            scene,
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertEqual(safety["reason"], "scene_object_ids_not_unique")
        self.assertIn("scene_object_ids_unique", safety["failed_fields"])

    def test_push_distance_too_large_rejected(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(push_distance_m=0.08),
            _scene(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertIn("push_distance_in_range", safety["failed_fields"])

    def test_unnormalized_direction_rejected(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(push_direction_base=[2.0, 0.0, 0.0]),
            _scene(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertIn("push_direction_base_unit_xy_vector", safety["failed_fields"])

    def test_moveit_preflight_failure_rejects_validated_nudge(self):
        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            _decision(),
            _scene(),
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
                "object_label": "red_block",
                "object_center_base_m": [0.10, 0.0, 0.02],
                "target_object_id": 1,
                "target_object_label": "blue_block",
                "target_object_center_base_m": [0.0, 0.0, 0.02],
                "safe_place_center_base_m": [0.16, -0.16, 0.02],
                "reason": "remove blocker",
                "confidence": 0.7,
                "raw_decision": {},
            },
            _scene(),
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
                "object_label": "red_block",
                "object_center_base_m": [0.10, 0.0, 0.02],
                "target_object_id": 1,
                "target_object_label": "blue_block",
                "target_object_center_base_m": [0.0, 0.0, 0.02],
                "safe_place_center_base_m": [0.25, 0.20, 0.02],
                "reason": "bad place",
                "confidence": 0.7,
                "raw_decision": {},
            },
            _scene(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertIn("safe_place_avoids_visible_objects", safety["failed_fields"])

    def test_illegal_json_is_generation_failure_not_stop(self):
        failure = PolicyCallResult(
            transport_status="ok", generation_status="finalization_failed",
            policy_kind="action_proposal", model="test",
            error_type="FINALIZATION_FAILED", error_message="not json",
        )
        original_call = vlm_action_policy.call_policy
        try:
            vlm_action_policy.call_policy = lambda *_args, **_kwargs: failure
            raw = vlm_action_policy.call_vlm_action_policy(
                SimpleNamespace(model="test", ollama_url="http://test", timeout=1, num_predict=64, no_image=True),
                {"objects": [], "target_object": {}},
            )
        finally:
            vlm_action_policy.call_policy = original_call

        self.assertEqual(raw["call_status"], "finalization_failed")
        self.assertIsNone(raw["decision"])
        self.assertEqual(raw["error_type"], "FINALIZATION_FAILED")

    def test_autonomous_action_parser_requires_problem_and_benefit(self):
        with self.assertRaises(ValueError):
            vlm_action_policy.parse_vlm_action_decision_text(
                json.dumps({
                    "action_type": "pick",
                    "object_id": 1,
                    "object_label": "blue_block",
                    "object_center_base_m": [0.0, 0.0, 0.02],
                    "target_object_id": 1,
                    "target_object_label": "blue_block",
                    "target_object_center_base_m": [0.0, 0.0, 0.02],
                    "reason": "pick it",
                    "confidence": 0.8,
                })
            )

    def test_autonomous_action_parser_rejects_out_of_range_confidence(self):
        with self.assertRaises(ValueError):
            vlm_action_policy.parse_vlm_action_decision_text(
                json.dumps({
                    "scene_problem": "clear scene",
                    "action_type": "pick",
                    "object_id": 1,
                    "object_label": "blue_block",
                    "object_center_base_m": [0.0, 0.0, 0.02],
                    "target_object_id": 1,
                    "target_object_label": "blue_block",
                    "target_object_center_base_m": [0.0, 0.0, 0.02],
                    "predicted_scene_benefit": "advance the stack",
                    "risk_assessment": "low",
                    "reason": "target is isolated",
                    "confidence": 1.4,
                })
            )

    def test_autonomous_action_parser_keeps_scene_diagnosis_and_prediction(self):
        decision = vlm_action_policy.parse_vlm_action_decision_text(
            json.dumps({
                "scene_problem": "object 2 blocks access to object 1",
                "action_type": "nudge",
                "object_id": 2,
                "object_label": "red_block",
                "object_center_base_m": [0.10, 0.0, 0.02],
                "target_object_id": 1,
                "target_object_label": "blue_block",
                "target_object_center_base_m": [0.0, 0.0, 0.02],
                "contact_side": "-x",
                "direction_base": [1.0, 0.0, 0.0],
                "distance_m": 0.02,
                "gripper_yaw_rad": 0.0,
                "predicted_scene_benefit": "open a grasp corridor around object 1",
                "risk_assessment": "object 2 may rotate",
                "reason": "minimal displacement",
                "confidence": 0.82,
            })
        )

        self.assertEqual(decision["scene_problem"], "object 2 blocks access to object 1")
        self.assertIn("grasp corridor", decision["predicted_scene_benefit"])

    def test_action_grounding_label_mismatch_is_rejected(self):
        decision = _decision()
        decision["object_label"] = "square green"

        selected, safety = vlm_action_policy.validate_vlm_action_decision(
            decision,
            _scene(),
            protected_ids=[3],
        )

        self.assertIsNone(selected)
        self.assertEqual(safety["reason"], "object_semantic_binding_mismatch")
        self.assertIn("object_label_matches_selected_id", safety["failed_fields"])

    def test_action_input_contains_objective_scene_and_advisory_task_focus(self):
        payload = vlm_action_policy.build_vlm_action_decision_input(
            "/tmp/scene.png",
            "/tmp/overlay.png",
            _scene(),
            _target(),
            protected_ids=[3],
            base_id=3,
            memory={"structure": {"base": "base_block", "current_top": "base_block", "placed_order": ["base_block"]}},
            step_index=1,
        )
        prompt = vlm_action_policy.build_vlm_action_prompt(payload)

        self.assertNotIn("id", payload["task_goal"]["current_plan_focus"])
        self.assertIn("object_ref", payload["task_goal"]["current_plan_focus"])
        self.assertTrue(payload["task_goal"]["current_plan_focus_is_advisory"])
        self.assertNotIn("target_grasp_state", payload)
        encoded = json.dumps(payload)
        self.assertNotIn("grasp_feasible", encoded)
        self.assertNotIn("blocking_objects", encoded)
        self.assertNotIn("action_candidates", encoded)
        self.assertNotIn("recommended_direction", encoded)
        self.assertIn("contact_rule", payload["manipulator_geometry"])
        self.assertIn("没有代码生成的候选动作", prompt)

    def test_blocked_grasp_feedback_becomes_high_priority_clearing_directive(self):
        payload = vlm_action_policy.build_vlm_action_decision_input(
            "/tmp/scene.png", "/tmp/overlay.png", _scene(), _target(),
            protected_ids=[3], base_id=3, memory={}, step_index=1,
            failure_history=[{
                "failed_checks": [{
                    "type": "selected_object_grasp_feasible",
                    "all_grasps_blocked": True,
                    "blocking_objects": [{
                        "id": 4, "label": "square yellow",
                        "blocker_category": "loose_movable",
                    }],
                }],
            }],
        )
        prompt = vlm_action_policy.build_vlm_action_prompt(payload)
        self.assertIn("禁止重复 pick 目标", prompt)
        self.assertIn("square yellow", prompt)
        self.assertIn("nudge 或 pick_away", prompt)

    def test_action_input_lists_same_label_instances_by_stable_reference(self):
        scene = _scene()
        scene["objects"].append(
            {
                "id": 4,
                "label": "red_block",
                "geometry_center_m": [0.2, 0.0, 0.02],
                "dimensions_m": [0.03, 0.03, 0.04],
            }
        )

        payload = vlm_action_policy.build_vlm_action_decision_input(
            "/tmp/scene.png",
            "/tmp/overlay.png",
            scene,
            _target(),
            protected_ids=[3],
            base_id=3,
            memory={},
            step_index=1,
        )

        groups = payload["scene_integrity"]["same_label_instance_groups"]
        self.assertFalse(payload["scene_integrity"]["duplicate_object_ids"])
        self.assertEqual(groups[0]["label"], "red_block")
        self.assertTrue(all("id" not in item for item in groups[0]["instances"]))
        self.assertTrue(all("object_ref" in item and "track_id" in item for item in groups[0]["instances"]))


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
    objects = {
        1: ("blue_block", [0.0, 0.0, 0.02]),
        2: ("red_block", [0.10, 0.0, 0.02]),
        3: ("base_block", [0.25, 0.20, 0.02]),
    }
    object_grounding = objects.get(object_id, (None, None))
    target_grounding = objects.get(target_object_id, (None, None))
    return {
        "scene_problem": "the intended task object is obstructed",
        "action_type": "nudge",
        "object_id": object_id,
        "object_label": object_grounding[0],
        "object_center_base_m": object_grounding[1],
        "target_object_id": target_object_id,
        "target_object_label": target_grounding[0],
        "target_object_center_base_m": target_grounding[1],
        "contact_side": "-x",
        "direction_base": push_direction_base or [1.0, 0.0, 0.0],
        "distance_m": push_distance_m,
        "gripper_yaw_rad": 0.0,
        "predicted_scene_benefit": "increase free space around the task object",
        "risk_assessment": "the obstacle may rotate",
        "reason": "clear target grasp corridor",
        "confidence": 0.8,
        "raw_decision": {},
    }


if __name__ == "__main__":
    unittest.main()
