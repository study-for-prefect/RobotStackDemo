"""Parse Qwen-proposed clearing actions as non-authoritative candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _finite_direction(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x_value = float(value[0])
        y_value = float(value[1])
        z_value = float(value[2]) if len(value) > 2 else 0.0
    except (TypeError, ValueError):
        return None
    norm = math.hypot(x_value, y_value)
    if not all(math.isfinite(item) for item in (x_value, y_value, z_value)) or norm < 1e-9:
        return None
    return [x_value / norm, y_value / norm, z_value]


def parse_qwen_action_candidates(path: Union[str, Path], min_confidence: float = 0.4) -> Dict[str, Any]:
    """Load Qwen candidate actions, filtering malformed or low-confidence items.

    Qwen output is only used as candidate generation input. Downstream geometry,
    swept-volume, future-task, and MoveIt checks must still accept the action.
    """
    path = Path(path)
    if not path.exists():
        return {
            "parse_status": "missing",
            "scene_type": None,
            "candidate_actions": [],
            "forbidden_objects": [],
            "source_path": str(path),
        }
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {
            "parse_status": "invalid_json",
            "scene_type": None,
            "candidate_actions": [],
            "forbidden_objects": [],
            "source_path": str(path),
        }

    raw_actions = payload.get("candidate_actions", [])
    actions: List[Dict[str, Any]] = []
    if isinstance(raw_actions, list):
        for raw in raw_actions:
            if not isinstance(raw, dict):
                continue
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            direction = _finite_direction(raw.get("direction_base"))
            if confidence < float(min_confidence) or direction is None:
                continue
            obstacle_id = raw.get("obstacle_id")
            if obstacle_id is None:
                continue
            actions.append(
                {
                    "type": str(raw.get("type") or "push_away"),
                    "obstacle_id": obstacle_id,
                    "direction_base": direction,
                    "reason": str(raw.get("reason") or "qwen_candidate"),
                    "confidence": confidence,
                    "source": "qwen",
                }
            )

    forbidden = payload.get("forbidden_objects", [])
    if not isinstance(forbidden, list):
        forbidden = []
    return {
        "parse_status": "ok",
        "scene_type": payload.get("scene_type"),
        "candidate_actions": actions,
        "forbidden_objects": [item for item in forbidden if isinstance(item, dict)],
        "source_path": str(path),
    }
