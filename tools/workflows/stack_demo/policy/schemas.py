"""Strict ID-only Qwen response schemas and semantic parsers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from robot_scene_pipeline.schema_validation import validate_against_schema

from ..clutter.target_options import TargetOption
from ..common.action_edges import PhysicalActionEdge


TARGET_SELECTION_SCHEMA = {
    "type": "object",
    "required": ["selected_target_option_id", "reason_codes"],
    "properties": {
        "selected_target_option_id": {"type": "string"},
        "reason_codes": {
            "type": "array", "items": {"type": "string"},
            "minItems": 1, "maxItems": 6, "uniqueItems": True,
        },
    },
    "additionalProperties": False,
}

EDGE_SELECTION_SCHEMA = {
    "type": "object",
    "required": ["selected_candidate_id", "backup_candidate_ids", "reason_codes"],
    "properties": {
        "selected_candidate_id": {"type": "string"},
        "backup_candidate_ids": {
            "type": "array", "items": {"type": "string"},
            "maxItems": 3, "uniqueItems": True,
        },
        "reason_codes": {
            "type": "array", "items": {"type": "string"},
            "minItems": 1, "maxItems": 6, "uniqueItems": True,
        },
    },
    "additionalProperties": False,
}


class PolicyOutputError(ValueError):
    """A Qwen response is not a valid selection from the supplied revision."""


def parse_target_selection(
    raw: str | Mapping[str, Any],
    options: Sequence[TargetOption],
    scene_revision: int,
) -> dict[str, Any]:
    value = _object(raw)
    errors = validate_against_schema(value, TARGET_SELECTION_SCHEMA)
    if errors:
        raise PolicyOutputError(json.dumps(errors, ensure_ascii=False))
    selected = str(value["selected_target_option_id"])
    if not selected:
        raise PolicyOutputError("selected_target_option_id is empty")
    by_id = {item.target_option_id: item for item in options}
    if selected not in by_id:
        raise PolicyOutputError(f"unknown target_option_id: {selected}")
    if by_id[selected].scene_revision != int(scene_revision):
        raise PolicyOutputError("target option belongs to a stale scene_revision")
    return dict(value)


def parse_edge_selection(
    raw: str | Mapping[str, Any],
    edges: Sequence[PhysicalActionEdge],
    scene_revision: int,
) -> dict[str, Any]:
    value = _object(raw)
    errors = validate_against_schema(value, EDGE_SELECTION_SCHEMA)
    if errors:
        raise PolicyOutputError(json.dumps(errors, ensure_ascii=False))
    selected = str(value["selected_candidate_id"])
    backups = [str(item) for item in value["backup_candidate_ids"]]
    if not selected or any(not item for item in backups):
        raise PolicyOutputError("candidate ids must not be empty")
    if selected in backups or len(set(backups)) != len(backups):
        raise PolicyOutputError("candidate ids must not be duplicated")
    by_id = {item.candidate_id: item for item in edges}
    unknown = [item for item in [selected, *backups] if item not in by_id]
    if unknown:
        raise PolicyOutputError(f"unknown candidate ids: {unknown}")
    if any(by_id[item].scene_revision != int(scene_revision) for item in [selected, *backups]):
        raise PolicyOutputError("candidate belongs to a stale scene_revision")
    return dict(value)


def _object(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PolicyOutputError("policy output is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PolicyOutputError("policy output must be one JSON object")
    return value
