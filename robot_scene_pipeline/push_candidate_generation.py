"""Generate and merge rule/Qwen push clearing candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .geometry_relations import get_center


ObjectDict = Dict[str, Any]
CandidateDict = Dict[str, Any]


def normalize_xy(direction: Iterable[float]) -> Optional[List[float]]:
    try:
        values = [float(value) for value in list(direction)[:2]]
    except (TypeError, ValueError):
        return None
    norm = math.hypot(values[0], values[1])
    if norm < 1e-9 or not all(math.isfinite(value) for value in values):
        return None
    return [values[0] / norm, values[1] / norm, 0.0]


def _same_direction(first: Iterable[float], second: Iterable[float]) -> bool:
    first_xy = normalize_xy(first)
    second_xy = normalize_xy(second)
    if first_xy is None or second_xy is None:
        return False
    return abs(first_xy[0] - second_xy[0]) < 1e-6 and abs(first_xy[1] - second_xy[1]) < 1e-6


def _find_object(objects: Iterable[ObjectDict], object_id: Any) -> Optional[ObjectDict]:
    for obj in objects:
        if str(obj.get("id")) == str(object_id):
            return obj
    return None


def _rule_directions(obstacle: ObjectDict, target: ObjectDict) -> List[CandidateDict]:
    raw = [
        ("base_positive_x", [1.0, 0.0, 0.0]),
        ("base_negative_x", [-1.0, 0.0, 0.0]),
        ("base_positive_y", [0.0, 1.0, 0.0]),
        ("base_negative_y", [0.0, -1.0, 0.0]),
    ]
    candidates: List[CandidateDict] = []
    for source, direction in raw:
        normalized = normalize_xy(direction)
        if normalized is None or any(_same_direction(normalized, item["direction_base"]) for item in candidates):
            continue
        candidates.append({"action": "push_away", "obstacle_id": obstacle.get("id"), "source": source, "direction_base": normalized})
    return candidates


def _qwen_candidates_for_obstacle(qwen_candidates: Iterable[dict], obstacle: ObjectDict) -> List[CandidateDict]:
    candidates: List[CandidateDict] = []
    for raw in qwen_candidates or []:
        if str(raw.get("obstacle_id")) != str(obstacle.get("id")):
            continue
        direction = normalize_xy(raw.get("direction_base"))
        if direction is None:
            continue
        candidates.append(
            {
                "action": str(raw.get("type") or "push_away"),
                "obstacle_id": obstacle.get("id"),
                "source": "qwen",
                "direction_base": direction,
                "qwen_confidence": float(raw.get("confidence", 0.0)),
                "qwen_reason": raw.get("reason"),
            }
        )
    return candidates


def _dedupe_candidates(candidates: Iterable[CandidateDict]) -> List[CandidateDict]:
    output: List[CandidateDict] = []
    for candidate in candidates:
        direction = candidate.get("direction_base")
        obstacle_id = str(candidate.get("obstacle_id"))
        if any(str(item.get("obstacle_id")) == obstacle_id and _same_direction(direction, item.get("direction_base", [])) for item in output):
            continue
        output.append(candidate)
    return output


def build_joint_push_candidates(
    objects: Iterable[ObjectDict],
    target: ObjectDict,
    obstacle_ids: Iterable[Any],
    qwen_candidates: Iterable[dict] = (),
) -> List[CandidateDict]:
    scene_objects = [obj for obj in objects if isinstance(obj, dict)]
    candidates: List[CandidateDict] = []
    for obstacle_id in obstacle_ids:
        obstacle = _find_object(scene_objects, obstacle_id)
        if obstacle is None:
            continue
        candidates.extend(_rule_directions(obstacle, target))
        candidates.extend(_qwen_candidates_for_obstacle(qwen_candidates, obstacle))
    return _dedupe_candidates(candidates)
