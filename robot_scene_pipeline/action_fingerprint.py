"""Stable action normalization and per-step failed-action ledger helpers."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, Optional, Sequence


def normalize_action_fingerprint(action: dict, state: dict, scene_revision: int) -> dict:
    """Build a track-based, tolerance-bucketed fingerprint independent of prose."""
    operated = _track_for(action, state, "object")
    target = _track_for(action, state, "target_object")
    value = {
        "strategy_id": str(action.get("strategy_id") or _default_strategy(action)),
        "action_type": str(action.get("action_type") or ""),
        "operated_track_id": operated,
        "target_track_id": target,
        "target_role": action.get("target_role") or action.get("role_id"),
        "direction_bin": direction_bin(action.get("direction_base") or action.get("push_direction_base")),
        "distance_bin_m": _quantize(action.get("distance_m", action.get("push_distance_m")), 0.005),
        "grasp_yaw_bin_deg": _yaw_bin(action.get("gripper_yaw_rad", action.get("grasp_yaw_rad"))),
        "destination_bin": _destination_bin(action),
    }
    value["fingerprint"] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    value["scene_revision"] = int(scene_revision)
    return value


def direction_bin(value: Any) -> Optional[str]:
    if not isinstance(value, (list, tuple)) or len(value) < 2: return None
    try: x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError): return None
    norm = math.hypot(x, y)
    if norm <= 1e-9: return "other"
    x, y = x / norm, y / norm
    if abs(x) >= 0.924: return "+X" if x > 0 else "-X"
    if abs(y) >= 0.924: return "+Y" if y > 0 else "-Y"
    if x * y > 0: return "diagonal_1"
    if x * y < 0: return "diagonal_2"
    return "other"


def build_replanning_context(ledger: Iterable[dict], attempt: int, max_attempts: int) -> dict:
    """Return deterministic escalation constraints for the next VLM request."""
    entries = list(ledger)
    counts: Dict[str, int] = {}
    for item in entries:
        action_type = str(item.get("action", {}).get("action_type") or "")
        counts[action_type] = counts.get(action_type, 0) + 1
    forbidden_types = sorted(key for key, count in counts.items() if key and count >= 2) if attempt >= 3 else []
    strategies = {item.get("strategy_id") for item in entries if item.get("strategy_id")}
    return {
        "attempt": int(attempt), "max_attempts": int(max_attempts),
        "failed_actions": [
            {"fingerprint": item.get("fingerprint"), "strategy_id": item.get("strategy_id"),
             "action_type": item.get("action", {}).get("action_type"), "operated_object": item.get("operated_track_id"),
             "reason_codes": item.get("reason_codes", [])}
            for item in entries
        ],
        "hard_constraints": {
            "forbidden_action_fingerprints": [item.get("fingerprint") for item in entries],
            "forbidden_action_types": forbidden_types,
            "required_strategy_change": attempt >= 4,
            "safe_stop_allowed": attempt >= max_attempts and len(strategies) >= 2 and len(entries) >= 2,
        },
    }


def ledger_entry(attempt: int, fingerprint: dict, action: dict, failure_stage: str, reason_codes: Sequence[str], scene_revision: int) -> dict:
    return {
        "attempt_index": int(attempt), "fingerprint": fingerprint["fingerprint"],
        "strategy_id": fingerprint["strategy_id"], "operated_track_id": fingerprint["operated_track_id"],
        "target_track_id": fingerprint["target_track_id"], "action": dict(action),
        "failure_stage": failure_stage, "reason_codes": list(reason_codes), "scene_revision": int(scene_revision),
    }


def _track_for(action: dict, state: dict, prefix: str) -> Optional[str]:
    explicit = action.get("{}_track_id".format(prefix))
    if prefix == "object":
        explicit = explicit or action.get("selected_track_id")
    if explicit: return str(explicit)
    object_id = action.get("{}_id".format(prefix))
    reference = action.get("{}_ref".format(prefix))
    if prefix == "object":
        object_id = object_id if object_id is not None else action.get("selected_object_id")
        reference = reference or action.get("selected_object_ref")
    for obj in state.get("objects", []):
        if reference and str(obj.get("object_ref")) == str(reference): return obj.get("track_id") or str(reference)
        if object_id is not None and str(obj.get("id")) == str(object_id): return obj.get("track_id") or "scene_{}:obj_{}".format(scene_revision_placeholder(state), object_id)
    return str(explicit or reference) if explicit or reference else None


def scene_revision_placeholder(state: dict) -> int:
    return int(state.get("scene_revision", 0))


def _quantize(value: Any, increment: float) -> Optional[float]:
    try: return round(round(float(value) / increment) * increment, 3)
    except (TypeError, ValueError): return None


def _yaw_bin(value: Any) -> Optional[float]:
    try: return round(round(math.degrees(float(value)) / 5.0) * 5.0, 1)
    except (TypeError, ValueError): return None


def _destination_bin(action: dict) -> Optional[str]:
    place = action.get("safe_place_center_base_m") or (action.get("target_pose_base") or {}).get("position_m")
    if isinstance(place, (list, tuple)) and len(place) >= 2:
        return "xy_{:.2f}_{:.2f}".format(round(float(place[0]), 2), round(float(place[1]), 2))
    return action.get("destination_role")


def _default_strategy(action: dict) -> str:
    return {"pick": "direct_pick_target", "pick_place": "direct_pick_target", "nudge": "clear_blocker_by_nudge", "pick_away": "clear_blocker_by_pick_away", "reobserve": "reobserve_scene", "stop": "safe_stop"}.get(str(action.get("action_type")), "unspecified")
