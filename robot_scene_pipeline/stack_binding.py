"""Four-color stack instance binding, order identity, and unique fallback."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .llm_stack_blocks import color_mentions, object_label_contains
from .object_tracking import parse_object_ref


STACK_COLORS = ("red", "green", "blue", "yellow")
STACK_BINDING_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version", "structure", "strategy", "selected_by_color",
        "reason", "confidence",
    ],
    "properties": {
        "schema_version": {"const": "stack_binding_v2"},
        "structure": {"const": "tower"},
        "strategy": {"const": "vertical_stack"},
        "selected_by_color": {
            "type": "object",
            "required": list(STACK_COLORS),
            "properties": {
                color: {
                    "type": "object",
                    "required": ["object_ref", "track_id"],
                    "properties": {
                        "object_ref": {"type": "string"},
                        "track_id": {"type": "string"},
                    },
                    "additionalProperties": False,
                }
                for color in STACK_COLORS
            },
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}


def required_color_order(instruction: str) -> List[str]:
    colors = color_mentions(instruction or "")
    return list(STACK_COLORS) if colors == list(STACK_COLORS) else colors


def order_fingerprint(decision: dict, colors: List[str]) -> str:
    selected = decision.get("selected_by_color") or {}
    tracks = [str((selected.get(color) or {}).get("track_id") or "missing:{}".format(color)) for color in colors]
    return json.dumps(tracks, ensure_ascii=False, separators=(",", ":"))


def validate_stack_binding(decision: dict, state: dict, instruction: str) -> dict:
    colors = required_color_order(instruction)
    if colors != list(STACK_COLORS):
        raise ValueError("stack_binding_v2 requires explicit red, green, blue, yellow order")
    selected = decision.get("selected_by_color")
    if not isinstance(selected, dict) or any(color not in selected for color in colors):
        raise ValueError("stack binding must include all four required colors")
    revision = int(state.get("scene_revision", 1))
    objects = [obj for obj in state.get("objects", []) if isinstance(obj, dict)]
    chosen = []
    tracks = []
    for color in colors:
        binding = selected.get(color) or {}
        obj = _resolve_binding(binding, objects, revision)
        if not object_label_contains(obj, color):
            raise ValueError("stack color slot {} references actual label {}".format(color, obj.get("label")))
        chosen.append(obj)
        tracks.append(str(obj.get("track_id")))
    if len(tracks) != len(set(tracks)):
        raise ValueError("stack binding track ids must be unique")
    full_order = [obj.get("id") for obj in chosen]
    return {
        "schema_version": "stack_binding_v2",
        "task_type": "stack_blocks",
        "structure_plan": {"structure_type": "tower", "roles": [], "assembly_steps": [], "limitations": []},
        "selected_by_color": selected,
        "full_stack_order": full_order,
        "base_object_id": full_order[0],
        "stack_order": full_order[1:],
        "object_bindings": [
            {
                "object_id": obj.get("id"), "object_ref": obj.get("object_ref"),
                "track_id": obj.get("track_id"), "observed_label": obj.get("label"),
                "geometry_center_base_m": obj.get("geometry_center_m") or obj.get("center_3d_base_m"),
            }
            for obj in chosen
        ],
        "order_fingerprint": order_fingerprint(decision, colors),
        "reason": decision.get("reason"),
        "confidence": float(decision.get("confidence", 0.0)),
        "decision_source": "vlm_stack_binding",
        "stack_order_semantics": "place_order_excludes_base",
    }


def deterministic_unique_stack_binding(state: dict, instruction: str) -> Optional[dict]:
    colors = required_color_order(instruction)
    if colors != list(STACK_COLORS):
        return None
    candidates = {
        color: [obj for obj in state.get("objects", []) if isinstance(obj, dict) and object_label_contains(obj, color)]
        for color in colors
    }
    if any(len(candidates[color]) != 1 for color in colors):
        return None
    decision = {
        "schema_version": "stack_binding_v2", "structure": "tower", "strategy": "vertical_stack",
        "selected_by_color": {
            color: {
                "object_ref": candidates[color][0].get("object_ref"),
                "track_id": candidates[color][0].get("track_id"),
            }
            for color in colors
        },
        "reason": "Each requested color has exactly one legal visible instance.",
        "confidence": 1.0,
    }
    validated = validate_stack_binding(decision, state, instruction)
    validated["decision_source"] = "stack_binding_deterministic_unique_fallback"
    return validated


def stack_binding_selection_is_ambiguous(state: dict, instruction: str) -> bool:
    colors = required_color_order(instruction)
    return any(
        sum(object_label_contains(obj, color) for obj in state.get("objects", []) if isinstance(obj, dict)) > 1
        for color in colors
    )


def missing_stack_colors(state: dict, instruction: str) -> List[str]:
    return [
        color for color in required_color_order(instruction)
        if not any(
            object_label_contains(obj, color)
            for obj in state.get("objects", []) if isinstance(obj, dict)
        )
    ]


def _resolve_binding(binding: dict, objects: List[dict], revision: int) -> dict:
    parsed = parse_object_ref(binding.get("object_ref"))
    if parsed is None or parsed[0] != revision:
        raise ValueError("stack binding contains stale object_ref")
    by_ref = next((obj for obj in objects if str(obj.get("object_ref")) == str(binding.get("object_ref"))), None)
    by_track = next((obj for obj in objects if str(obj.get("track_id")) == str(binding.get("track_id"))), None)
    if by_ref is None or by_track is None or by_ref is not by_track:
        raise ValueError("stack binding object_ref and track_id do not resolve to one visible object")
    return by_ref
