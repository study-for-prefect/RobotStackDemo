"""Validated HTTP client for the persistent RGB-D perception server."""

from __future__ import annotations

import json
import os
from typing import Any
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from robot_scene_pipeline.perception_contract import (
    PerceptionServerError,
    PerceptionServerHealth,
    PerceptionSnapshotRequest,
)

from .constants import PROJECT_ROOT


DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"


def perception_camera_frame(args: Any) -> str:
    frame = str(getattr(args, "camera_frame", DEFAULT_CAMERA_FRAME) or DEFAULT_CAMERA_FRAME).lstrip("/")
    return DEFAULT_CAMERA_FRAME if frame == "camera_link" else frame


def build_perception_snapshot_request(
    args: Any,
    output_dir: str,
    *,
    capture_reason: str = "unspecified",
    scene_revision: int = 1,
) -> PerceptionSnapshotRequest:
    return PerceptionSnapshotRequest(
        output_dir=os.path.abspath(output_dir),
        tf_json=os.path.abspath(str(args.tf_json)),
        base_frame=str(args.base_frame).lstrip("/"),
        camera_frame=perception_camera_frame(args),
        tf_point_mode=str(args.tf_point_mode),
        score_thresh=float(args.score_thresh),
        request_id=f"snapshot-{uuid.uuid4().hex[:16]}",
        capture_reason=str(capture_reason),
        scene_revision=int(scene_revision),
        client_timeout_s=float(args.perception_server_timeout_s),
        tf_max_age_s=float(args.tf_max_age_s),
    )


def perception_server_health(args: Any) -> PerceptionServerHealth:
    base_url = str(args.perception_server_url or "").rstrip("/")
    if not base_url:
        raise RuntimeError("no perception server URL configured")
    url = f"{base_url}/health"
    payload = _get_json(url, float(args.perception_server_timeout_s), request_id=None)
    health = PerceptionServerHealth(payload)
    if not health.ready:
        raise PerceptionServerError(
            "perception camera is not ready",
            status=503,
            error_code="camera_not_ready",
            error_type="CameraNotReady",
            request_url=url,
            effective_config=payload,
        )
    mismatch = perception_config_mismatches(args, payload)
    if mismatch:
        raise PerceptionServerError(
            f"perception server fixed configuration mismatch: {mismatch}",
            status=409,
            error_code="perception_server_config_mismatch",
            error_type="ConfigurationMismatch",
            request_url=url,
            effective_config={
                "expected": _expected_server_config(args),
                "actual": dict(payload),
                "mismatched_fields": mismatch,
            },
        )
    return health


def perception_server_snapshot(
    args: Any,
    output_dir: str,
    *,
    capture_reason: str = "unspecified",
    scene_revision: int = 1,
) -> dict[str, Any]:
    base_url = str(args.perception_server_url or "").rstrip("/")
    if not base_url:
        raise RuntimeError("no perception server URL configured")
    request = build_perception_snapshot_request(
        args, output_dir, capture_reason=capture_reason, scene_revision=scene_revision,
    )
    url = f"{base_url}/snapshot?{urlencode(request.to_query())}"
    payload = _get_json(url, float(args.perception_server_timeout_s), request_id=request.request_id)
    if not payload.get("ok"):
        raise PerceptionServerError(
            str(payload.get("error") or "perception server snapshot failed"),
            status=200,
            error_code=str(payload.get("error_code") or "internal_server_error"),
            error_type=payload.get("error_type"),
            request_id=str(payload.get("request_id") or request.request_id),
            request_url=url,
            effective_config=payload.get("effective_config"),
        )
    return payload


def _get_json(url: str, timeout_s: float, *, request_id: str | None) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=float(timeout_s)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = {}
        raise PerceptionServerError(
            str(error_payload.get("error") or body or str(exc)),
            status=int(exc.code),
            error_code=str(error_payload.get("error_code") or "http_error"),
            error_type=error_payload.get("error_type") or type(exc).__name__,
            request_id=str(error_payload.get("request_id") or request_id or "") or None,
            request_url=url,
            effective_config=error_payload.get("effective_config"),
        ) from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        error_code = "connection_timeout" if isinstance(reason, TimeoutError) else "connection_failed"
        raise PerceptionServerError(
            str(reason), error_code=error_code, error_type=type(reason).__name__,
            request_id=request_id, request_url=url,
        ) from exc
    except TimeoutError as exc:
        raise PerceptionServerError(
            str(exc), error_code="connection_timeout", error_type=type(exc).__name__,
            request_id=request_id, request_url=url,
        ) from exc
    except json.JSONDecodeError as exc:
        raise PerceptionServerError(
            f"server returned invalid JSON: {exc}", error_code="invalid_response",
            error_type=type(exc).__name__, request_id=request_id, request_url=url,
        ) from exc
    if not isinstance(payload, dict):
        raise PerceptionServerError(
            "server response must be a JSON object", error_code="invalid_response",
            request_id=request_id, request_url=url,
        )
    return payload


def perception_config_mismatches(args: Any, health: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = _expected_server_config(args)
    mismatch: dict[str, dict[str, Any]] = {}
    for field, expected_value in expected.items():
        actual = health.get(field)
        equal = (
            abs(float(actual) - float(expected_value)) <= 1e-9
            if isinstance(expected_value, float) and isinstance(actual, (int, float))
            else actual == expected_value
        )
        if not equal:
            mismatch[field] = {"expected": expected_value, "actual": actual}
    return mismatch


def _expected_server_config(args: Any) -> dict[str, Any]:
    weight = str(args.detector_weight)
    if not os.path.isabs(weight):
        weight = os.path.join(PROJECT_ROOT, weight)
    return {
        "semantic_review_candidate_pool": True,
        "color_topic": str(args.color_topic),
        "depth_topic": str(args.depth_topic),
        "camera_info_topic": str(args.camera_info_topic),
        "configured_camera_frame": perception_camera_frame(args),
        "base_frame": str(args.base_frame).lstrip("/"),
        "tf_point_mode": str(args.tf_point_mode),
        "detector_weight": os.path.abspath(weight),
        "detector_imgsz": int(args.detector_imgsz),
        "detector_iou": float(args.detector_iou),
        "candidate_score_thresh": float(args.candidate_score_thresh),
        "detector_device": str(args.detector_device),
    }
