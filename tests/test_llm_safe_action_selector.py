#!/usr/bin/env python3

import json
from types import SimpleNamespace

from robot_scene_pipeline import llm_safe_action_selector


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self.content}}


def test_llm_selects_only_safe_candidate_id():
    original_post = llm_safe_action_selector.requests.post
    try:
        llm_safe_action_selector.requests.post = lambda *_args, **_kwargs: FakeResponse(
            json.dumps({"selected_candidate_id": "safe_b", "reason": "best progress"})
        )
        args = SimpleNamespace(enable_llm_push_selection=True, model="test", ollama_url="http://test", timeout=1, num_predict=64)
        report = llm_safe_action_selector.select_safe_push_candidate(
            args,
            "clear target",
            {"id": "target", "label": "target"},
            {
                "candidates": [
                    {"candidate_id": "safe_a", "feasible": True, "score": 0.5},
                    {"candidate_id": "safe_b", "feasible": True, "score": 0.4},
                    {"candidate_id": "unsafe", "feasible": False, "score": 9.0},
                ]
            },
            step_index=1,
        )
        assert report["selection_status"] == "selected"
        assert report["selection_source"] == "llm_safe_candidate_selector"
        assert report["selected_candidate_id"] == "safe_b"
    finally:
        llm_safe_action_selector.requests.post = original_post


def test_llm_invalid_candidate_falls_back_to_geometry():
    original_post = llm_safe_action_selector.requests.post
    try:
        llm_safe_action_selector.requests.post = lambda *_args, **_kwargs: FakeResponse(
            json.dumps({"selected_candidate_id": "unsafe", "reason": "not allowed"})
        )
        args = SimpleNamespace(enable_llm_push_selection=True, model="test", ollama_url="http://test", timeout=1, num_predict=64)
        report = llm_safe_action_selector.select_safe_push_candidate(
            args,
            "clear target",
            {"id": "target", "label": "target"},
            {"candidates": [{"candidate_id": "safe_a", "feasible": True, "score": 0.5}]},
            step_index=1,
        )
        assert report["selection_status"] == "fallback_geometry_score"
        assert report["selection_source"] == "geometry_score_fallback"
        assert report["selected_candidate_id"] is None
    finally:
        llm_safe_action_selector.requests.post = original_post


if __name__ == "__main__":
    test_llm_selects_only_safe_candidate_id()
    test_llm_invalid_candidate_falls_back_to_geometry()
    print("llm safe action selector tests passed")
