import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from robot_scene_pipeline.ollama_policy_client import (
    POLICY_GENERATION_CONFIG, _next_num_predict, call_policy, unload_model,
)


SCHEMA = {
    "type": "object",
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
    "additionalProperties": False,
}


class OllamaPolicyClientTests(unittest.TestCase):
    def test_official_thinking_alias_is_rejected_before_structured_backend_call(self):
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post") as post:
            result = call_policy(_args(model="qwen3-vl:8b"), "action_proposal", _messages(), SCHEMA)
        self.assertEqual(result.error_type, "THINKING_MODEL_UNSUITABLE_FOR_STRUCTURED_POLICY")
        self.assertIn("qwen3-vl:8b-instruct", result.error_message)
        post.assert_not_called()

    def test_default_budgets_are_bounded_and_retry_grows_gradually(self):
        self.assertEqual(POLICY_GENERATION_CONFIG["task_contract"]["num_predict"], 2048)
        self.assertEqual(POLICY_GENERATION_CONFIG["grounded_task_plan"]["num_predict"], 4096)
        self.assertEqual(POLICY_GENERATION_CONFIG["task_contract"]["num_ctx"], 12288)
        self.assertEqual(POLICY_GENERATION_CONFIG["grounded_task_plan"]["num_ctx"], 12288)
        self.assertEqual(POLICY_GENERATION_CONFIG["action_proposal"]["num_ctx"], 12288)
        self.assertEqual(POLICY_GENERATION_CONFIG["action_proposal"]["num_predict"], 2048)
        self.assertEqual(POLICY_GENERATION_CONFIG["action_replan"]["num_predict"], 2048)
        self.assertEqual(
            [_next_num_predict(value) for value in (2048, 4096, 6144, 8192)],
            [4096, 6144, 8192, 12288],
        )

    def test_oversized_text_is_rejected_without_expanding_context_or_calling_backend(self):
        messages = [{"role": "user", "content": "x" * 100000}]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post") as post:
            result = call_policy(_args(), "action_proposal", messages, SCHEMA)
        self.assertEqual(result.error_type, "INPUT_CONTEXT_BUDGET_EXCEEDED")
        post.assert_not_called()

    def test_thinking_and_valid_content_are_both_preserved(self):
        response = _response(_envelope(thinking="deep", content='{"ok": true}'))
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response):
            result = call_policy(_args(), "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.thinking, "deep")
        self.assertEqual(result.parsed_decision, {"ok": True})

    def test_auto_thinking_is_disabled_for_bounded_structured_generation(self):
        response = _response(_envelope(content='{"ok": true}', include_thinking=False))
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response) as post:
            result = call_policy(_args(), "grounded_task_plan", _messages(), SCHEMA)
        self.assertEqual(result.parsed_decision, {"ok": True})
        self.assertFalse(post.call_args.kwargs["json"]["think"])

    def test_explicit_thinking_remains_available_for_action_generation(self):
        response = _response(_envelope(thinking="short", content='{"ok": true}'))
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response) as post:
            call_policy(_args(vlm_think_mode="on"), "action_proposal", _messages(), SCHEMA)
        self.assertTrue(post.call_args.kwargs["json"]["think"])

    def test_cpu_diagnostic_mode_is_forwarded_to_ollama(self):
        response = _response(_envelope(content='{"ok": true}', include_thinking=False))
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response) as post:
            call_policy(_args(vlm_num_gpu=0), "action_proposal", _messages(), SCHEMA)
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_gpu"], 0)

    def test_empty_content_after_thinking_uses_finalizer(self):
        responses = [
            _response(_envelope(thinking="finished reasoning", content="")),
            _response(_envelope(content='{"ok": true}')),
        ]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses) as post:
            result = call_policy(_args(), "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.generation_status, "parsed_after_finalization")
        self.assertEqual(post.call_count, 2)
        self.assertFalse(post.call_args_list[1].kwargs["json"]["think"])

    def test_length_retries_reasoning_with_larger_budget(self):
        responses = [
            _response(_envelope(thinking="long", content='{"ok":', done_reason="length", eval_count=8192)),
            _response(_envelope(thinking="done", content='{"ok": true}')),
        ]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses) as post:
            result = call_policy(_args(), "task_contract", _messages(), SCHEMA)
        first = post.call_args_list[0].kwargs["json"]["options"]["num_predict"]
        second = post.call_args_list[1].kwargs["json"]["options"]["num_predict"]
        self.assertEqual(result.parsed_decision, {"ok": True})
        self.assertGreater(second, first)

    def test_empty_thinking_and_content_is_backend_failure_not_stop(self):
        with patch(
            "robot_scene_pipeline.ollama_policy_client.requests.post",
            side_effect=[_response(_envelope())] * 3,
        ):
            result = call_policy(_args(), "action_proposal", _messages(), SCHEMA)
        self.assertIsNone(result.parsed_decision)
        self.assertEqual(result.error_type, "EMPTY_MODEL_MESSAGE")

    def test_qwen25_response_without_thinking_is_valid(self):
        response = _response(_envelope(content='{"ok": true}', include_thinking=False))
        args = _args(model="qwen2.5vl:7b-q4_K_M")
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response) as post:
            result = call_policy(args, "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.parsed_decision, {"ok": True})
        self.assertNotIn("think", post.call_args.kwargs["json"])

    def test_http_500_preserves_body_and_fails_backend(self):
        response = _response({}, status=500, text="ollama crashed")
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response):
            result = call_policy(_args(), "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.error_type, "HTTP_ERROR")
        self.assertEqual(result.raw_response, "ollama crashed")

    def test_timeout_is_backend_failure(self):
        with patch(
            "robot_scene_pipeline.ollama_policy_client.requests.post",
            side_effect=requests.Timeout("read timeout"),
        ):
            result = call_policy(_args(), "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.error_type, "REQUEST_TIMEOUT")
        self.assertIsNone(result.parsed_decision)

    def test_finalizer_empty_content_fails_explicitly(self):
        args = _args(vlm_max_backend_retries=1)
        responses = [
            _response(_envelope(thinking="complete", content="")),
            _response(_envelope()),
            _response(_envelope()),
        ]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses):
            result = call_policy(args, "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.error_type, "FINALIZATION_FAILED")
        self.assertIsNone(result.parsed_decision)

    def test_unsupported_think_parameter_is_removed_once(self):
        args = _args(vlm_think_mode="on")
        responses = [
            _response({}, status=400, text="unknown unsupported field think"),
            _response(_envelope(content='{"ok": true}')),
        ]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses) as post:
            result = call_policy(args, "task_contract", _messages(), SCHEMA)
        self.assertEqual(result.parsed_decision, {"ok": True})
        self.assertIn("think", post.call_args_list[0].kwargs["json"])
        self.assertNotIn("think", post.call_args_list[1].kwargs["json"])

    def test_unsupported_assistant_thinking_replay_uses_internal_context(self):
        args = _args(vlm_max_backend_retries=1)
        responses = [
            _response(_envelope(thinking="finished reasoning", content="")),
            _response({}, status=400, text="unsupported unknown assistant thinking field"),
            _response(_envelope(content='{"ok": true}')),
        ]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses) as post:
            result = call_policy(args, "task_contract", _messages(), SCHEMA)
        second_messages = post.call_args_list[1].kwargs["json"]["messages"]
        third_messages = post.call_args_list[2].kwargs["json"]["messages"]
        self.assertEqual(result.parsed_decision, {"ok": True})
        self.assertTrue(any(item.get("role") == "assistant" and "thinking" in item for item in second_messages))
        self.assertFalse(any(item.get("role") == "assistant" and "thinking" in item for item in third_messages))
        self.assertTrue(any("内部推理上下文" in item.get("content", "") for item in third_messages))

    def test_diagnostics_files_include_thinking_content_and_token_fields(self):
        with tempfile.TemporaryDirectory() as output_dir:
            response = _response(_envelope(thinking="t", content='{"ok": true}'))
            with patch("robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response):
                call_policy(_args(output_dir=output_dir), "task_contract", _messages(), SCHEMA, artifact_dir=output_dir)
            call_dirs = os.listdir(os.path.join(output_dir, "ollama_calls"))
            call_dir = os.path.join(output_dir, "ollama_calls", call_dirs[0])
            with open(os.path.join(call_dir, "ollama_diagnostics.json"), encoding="utf-8") as handle:
                diagnostics = json.load(handle)
            self.assertEqual(diagnostics["thinking_length_chars"], 1)
            self.assertEqual(diagnostics["eval_count"], 20)
            self.assertTrue(os.path.isfile(os.path.join(call_dir, "ollama_response.json")))

    def test_unload_uses_generate_keep_alive_zero_without_inference_messages(self):
        response = _response({"done": True})
        with patch(
            "robot_scene_pipeline.ollama_policy_client.requests.post", return_value=response,
        ) as post:
            result = unload_model(_args())
        self.assertEqual(result.transport_status, "ok")
        self.assertEqual(result.generation_status, "model_unloaded")
        self.assertEqual(post.call_args.args[0], "http://local/api/generate")
        self.assertEqual(post.call_args.kwargs["json"], {
            "model": "qwen3-vl:30b-a3b-instruct", "keep_alive": 0,
        })


def _args(**overrides):
    values = {
        "model": "qwen3-vl:30b-a3b-instruct", "ollama_url": "http://local/api/chat",
        "vlm_think_mode": "auto", "vlm_read_timeout_sec": 1,
        "vlm_keep_alive": "1h", "vlm_max_backend_retries": 3,
        "vlm_max_budget_retries": 3, "vlm_num_ctx": 0,
        "vlm_num_predict": 0, "vlm_num_gpu": -1,
        "vlm_finalizer_num_predict": 4096,
        "output_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _messages():
    return [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]


def _envelope(thinking="", content="", done_reason="stop", eval_count=20, include_thinking=True):
    message = {"role": "assistant", "content": content}
    if include_thinking:
        message["thinking"] = thinking
    return {
        "message": message, "done": True, "done_reason": done_reason,
        "prompt_eval_count": 10, "eval_count": eval_count,
        "prompt_eval_duration": 1, "eval_duration": 2,
        "total_duration": 3, "load_duration": 4,
    }


class _response:
    def __init__(self, value, status=200, text=""):
        self._value = value
        self.status_code = status
        self.text = text

    def json(self):
        return self._value


if __name__ == "__main__":
    unittest.main()
