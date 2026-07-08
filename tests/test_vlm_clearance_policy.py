#!/usr/bin/env python3

import json
import unittest
from types import SimpleNamespace

from robot_scene_pipeline import vlm_clearance_policy


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self._content}}


class VlmClearancePolicyTests(unittest.TestCase):
    def test_physical_candidates_do_not_expose_scores(self):
        payload = vlm_clearance_policy.build_physical_clearance_candidates(
            [
                {
                    "candidate_id": "push_001",
                    "action_type": "nudge",
                    "obstacle_id": 2,
                    "target_object_id": 5,
                    "direction_base": [1.0, 0.0, 0.0],
                    "distance_m": 0.025,
                    "score": 9.0,
                    "utility_score": 8.0,
                    "risk_score": 0.0,
                    "geometry_feasible": True,
                    "approach_path_safe": True,
                    "push_swept_safe": True,
                    "push_end_safe": True,
                    "protected_structure_safe": True,
                    "future_task_feasible": True,
                    "moveit_feasible": True,
                }
            ]
        )

        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("score", encoded)
        candidate = payload["candidates"][0]
        self.assertTrue(candidate["collision_free"])
        self.assertTrue(candidate["moveit_feasible"])
        self.assertEqual(candidate["object_id"], 2)

    def test_vlm_selects_only_known_candidate(self):
        original_post = vlm_clearance_policy.requests.post
        try:
            vlm_clearance_policy.requests.post = lambda *_args, **_kwargs: _FakeResponse(
                json.dumps(
                    {
                        "selected_candidate_id": "push_001",
                        "decision_type": "nudge",
                        "object_id": 2,
                        "target_object_id": 5,
                        "reason": "smallest disturbance",
                        "confidence": 0.8,
                        "need_reobserve_after_action": True,
                    }
                )
            )
            policy_input = _policy_input()
            report = vlm_clearance_policy.call_vlm_clearance_policy(
                SimpleNamespace(model="test", ollama_url="http://test", timeout=1, num_predict=64, no_image=True),
                policy_input,
            )
        finally:
            vlm_clearance_policy.requests.post = original_post

        self.assertEqual(report["selection_status"], "selected")
        self.assertEqual(report["selected_candidate_id"], "push_001")
        self.assertEqual(report["decision_type"], "nudge")

    def test_vlm_parse_failure_is_fail_safe_reobserve(self):
        original_post = vlm_clearance_policy.requests.post
        try:
            vlm_clearance_policy.requests.post = lambda *_args, **_kwargs: _FakeResponse("not json")
            report = vlm_clearance_policy.call_vlm_clearance_policy(
                SimpleNamespace(model="test", ollama_url="http://test", timeout=1, num_predict=64, no_image=True),
                _policy_input(),
            )
        finally:
            vlm_clearance_policy.requests.post = original_post

        self.assertEqual(report["selection_status"], "fail_safe")
        self.assertEqual(report["decision_type"], "reobserve")
        self.assertIsNone(report["selected_candidate_id"])

    def test_final_safety_gate_rejects_failed_physical_field(self):
        physical = _physical_candidates(moveit_feasible=False)
        policy_output = {
            "selected_candidate_id": "push_001",
            "decision_type": "nudge",
        }

        gate = vlm_clearance_policy.evaluate_final_safety_gate(policy_output, physical)

        self.assertFalse(gate["accepted"])
        self.assertTrue(gate["rejected_by_safety_gate"])
        self.assertEqual(gate["failed_fields"], ["moveit_feasible"])


def _physical_candidates(moveit_feasible=True):
    return {
        "schema_version": "physical_clearance_candidates_v1",
        "candidates": [
            {
                "candidate_id": "push_001",
                "action_type": "nudge",
                "object_id": 2,
                "obstacle_object_id": 2,
                "target_object_id": 5,
                "direction_base": [1.0, 0.0, 0.0],
                "distance_m": 0.025,
                "collision_free": True,
                "moveit_feasible": moveit_feasible,
                "sweep_collision_free": True,
                "workspace_feasible": True,
                "gripper_feasible": True,
                "brief_physical_description": "test",
            }
        ],
    }


def _policy_input():
    return vlm_clearance_policy.build_policy_input(
        None,
        None,
        {"objects": []},
        {"target_object": {"id": 5}},
        _physical_candidates(),
    )


if __name__ == "__main__":
    unittest.main()

