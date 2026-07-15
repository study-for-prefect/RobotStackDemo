"""One-call, no-history Qwen transport for restricted selections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from robot_scene_pipeline.io_utils import image_to_base64
from robot_scene_pipeline.ollama_policy_client import call_policy


@dataclass(frozen=True)
class QwenCallResult:
    request: Mapping[str, Any]
    raw_output: str
    parsed_output: Mapping[str, Any] | None
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    format_repair_used: bool = False


class QwenSelectionClient(Protocol):
    def call(
        self,
        policy_kind: str,
        request: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        image_paths: Sequence[str],
        artifact_dir: str | None,
    ) -> QwenCallResult: ...


class StatelessQwenClient:
    """Issue a fresh two-message request; no prior assistant reply is retained."""

    def __init__(self, args: Any):
        self._args = args

    def call(
        self,
        policy_kind: str,
        request: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        image_paths: Sequence[str],
        artifact_dir: str | None,
    ) -> QwenCallResult:
        user_message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
        }
        images = [image_to_base64(path) for path in image_paths if path and Path(path).is_file()]
        if images:
            user_message["images"] = images
        messages = [
            {
                "role": "system",
                "content": (
                    "只比较输入中的受限ID并输出严格JSON。不得创建动作、对象、参数或完成声明；"
                    "严格按输入selection_priority从前到后比较；不得依赖数组顺序、图片阅读顺序或任何历史消息。"
                ),
            },
            user_message,
        ]
        result = call_policy(
            self._args,
            policy_kind,
            messages,
            dict(response_schema),
            artifact_dir=artifact_dir,
            temperature=0.0,
            top_p=0.85,
        )
        return QwenCallResult(
            request={"messages": messages, "response_schema": dict(response_schema)},
            raw_output=result.content,
            parsed_output=result.parsed_decision,
            success=bool(result.transport_status == "ok" and result.schema_valid),
            error_type=result.error_type,
            error_message=result.error_message,
            format_repair_used=result.generation_status == "parsed_after_finalization",
        )


class JsonFileQwenClient:
    """Offline mock policy used by tests and recorded dry-runs."""

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)

    def call(
        self,
        policy_kind: str,
        request: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        image_paths: Sequence[str],
        artifact_dir: str | None,
    ) -> QwenCallResult:
        path = self._directory / f"{policy_kind}.json"
        try:
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            return QwenCallResult(
                request={"messages": [
                    {"role": "system", "content": "restricted offline mock"},
                    {"role": "user", "content": dict(request)},
                ]},
                raw_output=raw,
                parsed_output=parsed if isinstance(parsed, dict) else None,
                success=isinstance(parsed, dict),
            )
        except Exception as exc:
            return QwenCallResult(
                request=dict(request), raw_output="", parsed_output=None,
                success=False, error_type="MOCK_POLICY_INVALID", error_message=str(exc),
            )
