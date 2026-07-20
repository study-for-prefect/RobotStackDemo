"""Validated HTTP contract shared by perception clients and the server."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import time
from typing import Any, Mapping

from .tf_transform import frame_matches, validate_point_mode_for_frame


# Bump whenever a persistent server must be restarted to load perception
# behavior changes.  The workflow checks this value before requesting a frame,
# so an old in-memory process cannot silently serve new runs.
PERCEPTION_PIPELINE_REVISION = "build-house-triangle-postplace-v3"


@dataclass(frozen=True)
class PerceptionSnapshotRequest:
    output_dir: str
    tf_json: str
    base_frame: str
    camera_frame: str
    tf_point_mode: str
    score_thresh: float
    request_id: str
    capture_reason: str
    scene_revision: int
    client_timeout_s: float
    tf_max_age_s: float

    def to_query(self) -> dict[str, str]:
        value = asdict(self)
        return {key: str(item) for key, item in value.items()}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerceptionServerHealth:
    payload: Mapping[str, Any]

    @property
    def ready(self) -> bool:
        return bool(self.payload.get("ok") and self.payload.get("ready"))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


class PerceptionServerError(RuntimeError):
    """Structured error that preserves server and transport diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_code: str = "internal_server_error",
        error_type: str | None = None,
        request_id: str | None = None,
        request_url: str | None = None,
        effective_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = str(error_code)
        self.error_type = error_type
        self.request_id = request_id
        self.request_url = request_url
        self.effective_config = dict(effective_config or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "error": str(self),
            "request_id": self.request_id,
            "request_url": self.request_url,
            "effective_config": self.effective_config,
        }

    def concise_message(self) -> str:
        return (
            "Perception server snapshot failed: "
            f"status={self.status} error_code={self.error_code} "
            f"error={self} request_id={self.request_id}"
        )


class PerceptionRequestError(ValueError):
    def __init__(self, error_code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.error_code = str(error_code)
        self.status = int(status)


class DetectorProcessingError(RuntimeError):
    pass


class SceneProcessingError(RuntimeError):
    pass


def parse_snapshot_request(query: Mapping[str, Any], *, default_tf_max_age_s: float) -> PerceptionSnapshotRequest:
    required = (
        "output_dir", "tf_json", "base_frame", "camera_frame", "tf_point_mode",
        "score_thresh", "request_id",
    )
    missing = [name for name in required if not str(query.get(name) or "").strip()]
    if missing:
        code = "tf_json_missing" if "tf_json" in missing else "invalid_request"
        raise PerceptionRequestError(code, f"Missing required snapshot parameters: {', '.join(missing)}")
    output_dir = os.path.abspath(str(query["output_dir"]))
    tf_json = os.path.abspath(str(query["tf_json"]))
    if not os.path.isabs(str(query["output_dir"])):
        raise PerceptionRequestError("invalid_request", "output_dir must be an absolute path")
    if not os.path.isabs(str(query["tf_json"])):
        raise PerceptionRequestError("invalid_request", "tf_json must be an absolute path")
    score_thresh = _finite_float(query["score_thresh"], "score_thresh")
    if not 0.01 <= score_thresh <= 1.0:
        raise PerceptionRequestError("invalid_request", "score_thresh must be in [0.01, 1.0]")
    client_timeout_s = _finite_float(query.get("client_timeout_s", 15.0), "client_timeout_s")
    tf_max_age_s = _finite_float(query.get("tf_max_age_s", default_tf_max_age_s), "tf_max_age_s")
    if client_timeout_s <= 0.0 or tf_max_age_s <= 0.0:
        raise PerceptionRequestError("invalid_request", "timeout values must be positive")
    try:
        scene_revision = int(query.get("scene_revision", 1))
    except (TypeError, ValueError) as exc:
        raise PerceptionRequestError("invalid_request", "scene_revision must be an integer") from exc
    request = PerceptionSnapshotRequest(
        output_dir=output_dir,
        tf_json=tf_json,
        base_frame=str(query["base_frame"]).lstrip("/"),
        camera_frame=str(query["camera_frame"]).lstrip("/"),
        tf_point_mode=str(query["tf_point_mode"]),
        score_thresh=score_thresh,
        request_id=str(query["request_id"]),
        capture_reason=str(query.get("capture_reason") or "unspecified"),
        scene_revision=scene_revision,
        client_timeout_s=client_timeout_s,
        tf_max_age_s=tf_max_age_s,
    )
    try:
        validate_point_mode_for_frame(request.camera_frame, request.tf_point_mode)
    except (RuntimeError, ValueError) as exc:
        raise PerceptionRequestError("invalid_request", str(exc)) from exc
    validate_tf_file(request)
    return request


def validate_tf_file(request: PerceptionSnapshotRequest) -> Mapping[str, Any]:
    if not request.tf_json:
        raise PerceptionRequestError("tf_json_missing", "TF JSON is required when use_tf=true")
    if not os.path.isfile(request.tf_json):
        raise PerceptionRequestError("tf_json_not_found", f"TF JSON does not exist: {request.tf_json}")
    try:
        with open(request.tf_json, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PerceptionRequestError("tf_json_invalid", f"TF JSON cannot be parsed: {exc}") from exc
    if not isinstance(payload, dict):
        raise PerceptionRequestError("tf_json_invalid", "TF JSON root must be an object")
    if not frame_matches(payload.get("parent_frame"), request.base_frame) or not frame_matches(
        payload.get("child_frame"), request.camera_frame
    ):
        raise PerceptionRequestError(
            "tf_frame_mismatch",
            "TF JSON frames do not match requested "
            f"{request.base_frame} <- {request.camera_frame}: "
            f"{payload.get('parent_frame')} <- {payload.get('child_frame')}",
        )
    matrix = payload.get("matrix_4x4")
    if not _finite_matrix4(matrix):
        raise PerceptionRequestError("tf_json_invalid", "TF JSON matrix_4x4 must contain 16 finite numbers")
    timestamp = _finite_float(payload.get("timestamp"), "TF timestamp", error_code="tf_json_invalid")
    age_s = time.time() - timestamp
    if age_s < -2.0 or age_s > request.tf_max_age_s:
        raise PerceptionRequestError(
            "tf_json_invalid",
            f"TF JSON is stale or future-dated: age_s={age_s:.3f}, max_age_s={request.tf_max_age_s:.3f}",
        )
    return payload


def config_fingerprint(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def structured_error_payload(
    error_code: str,
    exc: Exception,
    *,
    request_id: str | None,
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": str(error_code),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "request_id": request_id,
        "effective_config": dict(effective_config),
    }


def _finite_float(value: Any, name: str, *, error_code: str = "invalid_request") -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionRequestError(error_code, f"{name} must be numeric") from exc
    if not math.isfinite(output):
        raise PerceptionRequestError(error_code, f"{name} must be finite")
    return output


def _finite_matrix4(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        numbers = [float(item) for row in value for item in row]
    except (TypeError, ValueError):
        return False
    return all(isinstance(row, (list, tuple)) and len(row) == 4 for row in value) and len(numbers) == 16 and all(
        math.isfinite(item) for item in numbers
    )
