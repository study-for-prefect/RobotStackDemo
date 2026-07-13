"""Unified Ollama policy transport, reasoning, budget, and JSON finalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from .task_schemas import validate_against_schema


POLICY_GENERATION_CONFIG = {
    "stack_order": {"num_ctx": 12288, "num_predict": 4096},
    "task_contract": {"num_ctx": 8192, "num_predict": 2048},
    "grounded_task_plan": {"num_ctx": 16384, "num_predict": 4096},
    "action_proposal": {"num_ctx": 16384, "num_predict": 3072},
    "action_replan": {"num_ctx": 16384, "num_predict": 3072},
    "orientation_analysis": {"num_ctx": 24576, "num_predict": 8192},
    "final_json_generation": {"num_ctx": 12288, "num_predict": 2048},
}
_MODEL_RUNTIME: Dict[str, dict] = {}


@dataclass
class PolicyCallResult:
    transport_status: str
    generation_status: str
    policy_kind: str
    model: str
    thinking: str = ""
    content: str = ""
    message_role: str = ""
    parsed_decision: Optional[dict] = None
    schema_valid: bool = False
    http_status: Optional[int] = None
    done: Optional[bool] = None
    done_reason: Optional[str] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_duration_ns: Optional[int] = None
    total_duration_ns: Optional[int] = None
    load_duration_ns: Optional[int] = None
    backend_attempt: int = 0
    reasoning_attempt: int = 1
    finalization_attempt: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    raw_response: Any = None

    def to_dict(self) -> dict:
        return asdict(self)


def call_policy(
    args: Any,
    policy_kind: str,
    messages: List[dict],
    response_schema: dict,
    artifact_dir: Optional[str] = None,
    reasoning_attempt: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.85,
) -> PolicyCallResult:
    """Run one logical policy call with backend, budget, and finalization retries."""
    model = str(getattr(args, "model", "qwen3-vl:8b-instruct"))
    replacement = _structured_instruct_replacement(model)
    if replacement and policy_kind != "orientation_analysis":
        return PolicyCallResult(
            transport_status="model_rejected",
            generation_status="model_variant_unsuitable",
            policy_kind=policy_kind,
            model=model,
            error_type="THINKING_MODEL_UNSUITABLE_FOR_STRUCTURED_POLICY",
            error_message=(
                "{} is an Ollama thinking variant and may consume the entire JSON budget; use {}"
            ).format(model, replacement),
        )
    think_mode = str(getattr(args, "vlm_think_mode", "auto"))
    think_requested = _think_enabled(think_mode, model)
    if think_mode == "auto" and policy_kind != "orientation_analysis":
        think_requested = False
    num_ctx, num_predict = _generation_budget(args, policy_kind)
    estimated_input_tokens = _estimate_input_tokens(messages)
    if estimated_input_tokens + num_predict + 2048 > num_ctx:
        return PolicyCallResult(
            transport_status="input_rejected",
            generation_status="input_context_exceeded",
            policy_kind=policy_kind,
            model=model,
            error_type="INPUT_CONTEXT_BUDGET_EXCEEDED",
            error_message=(
                "estimated input {} + output {} + reserve 2048 exceeds fixed num_ctx {}"
            ).format(estimated_input_tokens, num_predict, num_ctx),
        )
    max_backend = max(1, int(getattr(args, "vlm_max_backend_retries", 3)))
    max_budget = max(1, int(getattr(args, "vlm_max_budget_retries", 3)))
    budget_attempt = 0
    while budget_attempt < max_budget:
        budget_attempt += 1
        result, think_requested = _backend_retry_call(
            args, policy_kind, messages, response_schema, artifact_dir,
            reasoning_attempt, 0, max_backend, think_requested,
            num_ctx, num_predict, temperature, top_p, budget_attempt,
        )
        if result.transport_status != "ok":
            return result
        if result.done_reason == "length" or _looks_truncated(result.content, result.eval_count, num_predict):
            if budget_attempt >= max_budget:
                result.generation_status = "budget_exhausted"
                result.error_type = "TOKEN_BUDGET_EXHAUSTED"
                return result
            old_predict = num_predict
            num_predict = _next_num_predict(num_predict)
            if estimated_input_tokens + num_predict + 2048 > num_ctx:
                result.generation_status = "budget_exhausted"
                result.error_type = "TOKEN_BUDGET_EXHAUSTED"
                result.error_message = "retry output budget would exceed fixed num_ctx {}".format(num_ctx)
                return result
            print("Budget Retry {}/{}\nold_num_predict={}\nnew_num_predict={}".format(
                budget_attempt, max_budget, old_predict, num_predict,
            ), flush=True)
            continue
        parsed, schema_errors = _parse_and_validate(result.content, response_schema)
        if parsed is not None and not schema_errors:
            result.parsed_decision = parsed
            result.schema_valid = True
            result.generation_status = "parsed"
            return result
        if result.done_reason in (None, "stop") and (result.thinking or result.content):
            return _finalize_policy(
                args, policy_kind, messages, response_schema, result,
                artifact_dir, reasoning_attempt, num_ctx, max_backend,
            )
        result.generation_status = "json_or_schema_invalid"
        result.error_type = "JSON_SCHEMA_INVALID" if parsed is not None else "CONTENT_JSON_INVALID"
        result.error_message = json.dumps(schema_errors, ensure_ascii=False) if schema_errors else "content is not strict JSON"
        return result
    raise AssertionError("unreachable budget loop")


def unload_model(args: Any, artifact_dir: Optional[str] = None) -> PolicyCallResult:
    """Explicitly release the configured model only when requested by the workflow."""
    message = [{"role": "user", "content": "release model"}]
    clone = _ArgsOverride(args, vlm_keep_alive="0", vlm_max_backend_retries=1, vlm_think_mode="off")
    return call_policy(clone, "final_json_generation", message, {"type": "object"}, artifact_dir)


def warm_model(args: Any, artifact_dir: Optional[str] = None) -> PolicyCallResult:
    """Load the model into Ollama before task planning and keep it resident."""
    return call_policy(
        _ArgsOverride(args, vlm_think_mode="off", vlm_max_backend_retries=1),
        "final_json_generation",
        [{"role": "user", "content": "只输出空JSON对象 {}，用于模型预热。"}],
        {"type": "object", "additionalProperties": False},
        artifact_dir=artifact_dir,
    )


def _backend_retry_call(
    args: Any, policy_kind: str, messages: List[dict], response_schema: dict,
    artifact_dir: Optional[str], reasoning_attempt: int, finalization_attempt: int,
    max_backend: int, think_requested: bool, num_ctx: int, num_predict: int,
    temperature: float, top_p: float, budget_attempt: int,
) -> Tuple[PolicyCallResult, bool]:
    model = str(getattr(args, "model", "qwen3-vl:8b-instruct"))
    omit_think_parameter = False
    for backend_attempt in range(1, max_backend + 1):
        print("Backend Attempt {}/{}\npolicy={}\nmodel={}\nthink={}\nnum_ctx={}\nnum_predict={}".format(
            backend_attempt, max_backend, policy_kind, model,
            str(think_requested).lower(), num_ctx, num_predict,
        ), flush=True)
        payload = _request_payload(
            args, messages, response_schema, think_requested,
            num_ctx, num_predict, temperature, top_p,
            include_think_parameter=not omit_think_parameter,
        )
        result = _single_http_call(
            args, policy_kind, payload, reasoning_attempt,
            finalization_attempt, backend_attempt,
        )
        _save_call_artifacts(
            artifact_dir, policy_kind, reasoning_attempt, finalization_attempt,
            backend_attempt, budget_attempt, payload, result, think_requested,
            num_ctx, num_predict,
        )
        _record_model_runtime(args, result, num_ctx, num_predict)
        if result.error_type == "THINK_PARAMETER_UNSUPPORTED" and "think" in payload:
            think_requested = False
            omit_think_parameter = True
            continue
        if result.transport_status == "ok":
            print("Reasoning result:\nthinking_chars={}\ncontent_chars={}\ndone_reason={}".format(
                len(result.thinking), len(result.content), result.done_reason or "unknown",
            ), flush=True)
            return result, think_requested
    return result, think_requested


def _single_http_call(
    args: Any, policy_kind: str, payload: dict, reasoning_attempt: int,
    finalization_attempt: int, backend_attempt: int,
) -> PolicyCallResult:
    model = str(payload["model"])
    result = PolicyCallResult(
        transport_status="backend_failed", generation_status="backend_failed",
        policy_kind=policy_kind, model=model, backend_attempt=backend_attempt,
        reasoning_attempt=reasoning_attempt, finalization_attempt=finalization_attempt,
    )
    response = None
    try:
        response = requests.post(
            getattr(args, "ollama_url", "http://127.0.0.1:11434/api/chat"),
            json=payload,
            timeout=float(getattr(args, "vlm_read_timeout_sec", 1200)),
        )
        status_value = getattr(response, "status_code", None)
        result.http_status = int(status_value) if isinstance(status_value, int) else None
        if result.http_status is not None and not 200 <= result.http_status < 300:
            body = getattr(response, "text", "")
            result.raw_response = body
            replay = any(
                message.get("role") == "assistant" and "thinking" in message
                for message in payload.get("messages", [])
            )
            if replay and _think_unsupported(body):
                result.error_type = "ASSISTANT_THINKING_REPLAY_UNSUPPORTED"
            else:
                result.error_type = "THINK_PARAMETER_UNSUPPORTED" if _think_unsupported(body) else "HTTP_ERROR"
            result.error_message = "HTTP {}: {}".format(result.http_status, body[:500])
            return result
        try:
            data = response.json()
        except Exception as exc:
            result.raw_response = getattr(response, "text", "")
            result.error_type = "INVALID_RESPONSE_JSON"
            result.error_message = str(exc)
            return result
        result.raw_response = data
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            result.error_type = "INVALID_RESPONSE_ENVELOPE"
            result.error_message = "Ollama response requires a message object"
            return result
        _populate_response_fields(result, data)
        if result.done is False:
            result.error_type = "INVALID_RESPONSE_ENVELOPE"
            result.error_message = "non-stream response returned done=false"
            return result
        if not result.thinking and not result.content:
            result.error_type = "EMPTY_MODEL_MESSAGE"
            result.error_message = "thinking and content are both empty"
            return result
        result.transport_status = "ok"
        result.generation_status = "generated"
        return result
    except requests.Timeout as exc:
        result.error_type = "REQUEST_TIMEOUT"
        result.error_message = str(exc)
    except requests.RequestException as exc:
        result.error_type = "CONNECTION_ERROR"
        result.error_message = str(exc)
    except Exception as exc:
        result.error_type = "OLLAMA_CLIENT_ERROR"
        result.error_message = str(exc)
    if response is not None and result.raw_response is None:
        result.raw_response = getattr(response, "text", "")
    return result


def _finalize_policy(
    args: Any, policy_kind: str, messages: List[dict], response_schema: dict,
    reasoning: PolicyCallResult, artifact_dir: Optional[str], reasoning_attempt: int,
    num_ctx: int, max_backend: int,
) -> PolicyCallResult:
    final_predict = int(getattr(args, "vlm_finalizer_num_predict", 4096))
    replay_supported = True
    for final_attempt in range(1, 3):
        print("Finalization Attempt {}/2".format(final_attempt), flush=True)
        final_messages = list(messages)
        if replay_supported:
            final_messages.append({
                "role": "assistant", "thinking": reasoning.thinking,
                "content": reasoning.content,
            })
        else:
            final_messages.append({
                "role": "user",
                "content": "内部推理上下文（不要重新分析图像）：\n{}".format(reasoning.thinking),
            })
        final_messages.append({
            "role": "user",
            "content": (
                "基于你刚才已经完成的推理，不要重新分析图像，不要输出解释。"
                "仅输出一个完全符合给定JSON Schema的JSON对象，不得使用Markdown代码围栏。"
            ),
        })
        result, _ = _backend_retry_call(
            args, "final_json_generation", final_messages, response_schema,
            artifact_dir, reasoning_attempt, final_attempt, max_backend, False,
            num_ctx, final_predict, 0.0, 0.8, 1,
        )
        if result.error_type == "ASSISTANT_THINKING_REPLAY_UNSUPPORTED" and replay_supported:
            replay_supported = False
            continue
        if result.transport_status != "ok":
            result.generation_status = "finalization_failed"
            result.error_type = result.error_type or "FINALIZATION_FAILED"
            continue
        parsed, schema_errors = _parse_and_validate(result.content, response_schema)
        valid = parsed is not None and not schema_errors
        print("final_content_json_valid={}".format(str(valid).lower()), flush=True)
        if valid:
            result.policy_kind = policy_kind
            result.parsed_decision = parsed
            result.schema_valid = True
            result.generation_status = "parsed_after_finalization"
            result.thinking = reasoning.thinking
            return result
    result.generation_status = "finalization_failed"
    result.error_type = "FINALIZATION_FAILED"
    result.error_message = "Finalizer did not return valid schema JSON"
    return result


def _request_payload(
    args: Any, messages: List[dict], response_schema: dict, think_requested: bool,
    num_ctx: int, num_predict: int, temperature: float, top_p: float,
    include_think_parameter: bool = True,
) -> dict:
    payload = {
        "model": getattr(args, "model", "qwen3-vl:8b-instruct"),
        "messages": messages,
        "stream": False,
        "format": response_schema,
        "keep_alive": str(getattr(args, "vlm_keep_alive", "1h")),
        "options": {
            "temperature": float(temperature), "top_p": float(top_p),
            "num_ctx": int(num_ctx), "num_predict": int(num_predict),
        },
    }
    num_gpu = int(getattr(args, "vlm_num_gpu", -1))
    if num_gpu >= 0:
        payload["options"]["num_gpu"] = num_gpu
    model_name = str(payload["model"]).lower()
    if "qwen3" in model_name and include_think_parameter:
        # Ollama's Qwen3 default may still reason when the field is omitted.
        payload["think"] = bool(think_requested)
    elif think_requested and include_think_parameter:
        payload["think"] = True
    return payload


def _generation_budget(args: Any, policy_kind: str) -> Tuple[int, int]:
    config = POLICY_GENERATION_CONFIG.get(policy_kind, POLICY_GENERATION_CONFIG["action_proposal"])
    num_ctx = int(getattr(args, "vlm_num_ctx", 0) or config["num_ctx"])
    num_predict = int(getattr(args, "vlm_num_predict", 0) or config["num_predict"])
    return num_ctx, num_predict


def _populate_response_fields(result: PolicyCallResult, data: dict) -> None:
    message = data.get("message") or {}
    result.message_role = str(message.get("role") or "")
    result.thinking = str(message.get("thinking") or "")
    result.content = str(message.get("content") or "")
    result.done = data.get("done")
    result.done_reason = data.get("done_reason")
    result.prompt_eval_count = data.get("prompt_eval_count")
    result.eval_count = data.get("eval_count")
    result.prompt_eval_duration_ns = data.get("prompt_eval_duration")
    result.eval_duration_ns = data.get("eval_duration")
    result.total_duration_ns = data.get("total_duration")
    result.load_duration_ns = data.get("load_duration")


def _parse_and_validate(content: str, schema: dict) -> Tuple[Optional[dict], list]:
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None, []
    if not isinstance(parsed, dict):
        return None, []
    return parsed, validate_against_schema(parsed, schema)


def _save_call_artifacts(
    artifact_dir: Optional[str], policy_kind: str, reasoning_attempt: int,
    finalization_attempt: int, backend_attempt: int, budget_attempt: int,
    payload: dict, result: PolicyCallResult, think_requested: bool,
    num_ctx: int, num_predict: int,
) -> None:
    if not artifact_dir:
        return
    label = "{}_r{:02d}_f{:02d}_budget{:02d}_backend{:02d}".format(
        policy_kind, reasoning_attempt, finalization_attempt, budget_attempt, backend_attempt,
    )
    directory = os.path.join(artifact_dir, "ollama_calls", label)
    os.makedirs(directory, exist_ok=True)
    _write_json(os.path.join(directory, "ollama_request.json"), payload)
    if isinstance(result.raw_response, (dict, list)):
        _write_json(os.path.join(directory, "ollama_response.json"), result.raw_response)
    else:
        _write_text(os.path.join(directory, "ollama_response.json"), str(result.raw_response or ""))
    _write_text(os.path.join(directory, "ollama_thinking.txt"), result.thinking)
    _write_text(os.path.join(directory, "ollama_content.txt"), result.content)
    diagnostics = result.to_dict()
    diagnostics.update({
        "think_requested": think_requested,
        "thinking_length_chars": len(result.thinking),
        "content_length_chars": len(result.content),
        "actual_num_ctx": int(num_ctx),
        "actual_num_predict": int(num_predict),
    })
    diagnostics.pop("raw_response", None)
    diagnostics.pop("parsed_decision", None)
    _write_json(os.path.join(directory, "ollama_diagnostics.json"), diagnostics)


def _looks_truncated(content: str, eval_count: Optional[int], num_predict: int) -> bool:
    stripped = str(content or "").strip()
    at_limit = eval_count is not None and int(eval_count) >= max(1, int(num_predict) - 8)
    return bool(stripped and not stripped.endswith("}") and at_limit)


def _next_num_predict(current: int) -> int:
    if current < 4096:
        return 4096
    if current < 6144:
        return 6144
    if current < 8192:
        return 8192
    return current + 4096


def _estimate_input_tokens(messages: List[dict]) -> int:
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    return max(1, chars // 3)


def _think_enabled(mode: str, model: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return "qwen3" in model.lower()


def _structured_instruct_replacement(model: str) -> Optional[str]:
    normalized = str(model or "").strip().lower()
    return {
        "qwen3-vl": "qwen3-vl:8b-instruct",
        "qwen3-vl:latest": "qwen3-vl:8b-instruct",
        "qwen3-vl:8b": "qwen3-vl:8b-instruct",
        "qwen3-vl:8b-thinking": "qwen3-vl:8b-instruct",
        "qwen3-vl:30b": "qwen3-vl:30b-a3b-instruct",
        "qwen3-vl:30b-a3b-thinking": "qwen3-vl:30b-a3b-instruct",
    }.get(normalized)


def _think_unsupported(body: str) -> bool:
    text = str(body or "").lower()
    return "think" in text and any(token in text for token in ("unsupported", "unknown", "not support", "invalid"))


def _write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _write_text(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(value)


def _record_model_runtime(args: Any, result: PolicyCallResult, num_ctx: int, num_predict: int) -> None:
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return
    model = result.model
    report = _MODEL_RUNTIME.setdefault(model, {
        "model": model,
        "keep_alive": str(getattr(args, "vlm_keep_alive", "1h")),
        "first_load_duration": None,
        "first_load_duration_ns": None,
        "subsequent_load_duration": [],
        "subsequent_load_duration_ns": [],
        "repeated_load_detected": False,
        "calls": [],
    })
    load_duration = result.load_duration_ns
    if report["first_load_duration_ns"] is None:
        report["first_load_duration_ns"] = load_duration
        report["first_load_duration"] = load_duration
    else:
        report["subsequent_load_duration_ns"].append(load_duration)
        report["subsequent_load_duration"].append(load_duration)
        first = int(report["first_load_duration_ns"] or 0)
        if load_duration is not None and int(load_duration) > max(1_000_000_000, first // 2):
            report["repeated_load_detected"] = True
    report["calls"].append({
        "policy_kind": result.policy_kind,
        "load_duration_ns": load_duration,
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "num_ctx": int(num_ctx),
        "num_predict": int(num_predict),
    })
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "model_runtime_diagnostics.json"), report)


class _ArgsOverride:
    def __init__(self, source: Any, **overrides: Any):
        self._source = source
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._source, name)
