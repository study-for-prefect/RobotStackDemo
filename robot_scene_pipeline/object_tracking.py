"""Stable cross-frame object references and one-to-one detection rebinding."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .geometry_relations import get_center, get_size
from .task_semantic_validation import infer_object_shape


def object_ref(scene_revision: int, detector_object_id: Any) -> str:
    """Return the only frame-local object identifier exposed to a VLM."""
    return "scene_{}:obj_{}".format(int(scene_revision), detector_object_id)


def update_scene_tracks(
    memory: dict, detections: List[dict], scene_revision: int,
    max_movement_m: float = 0.15, predicted_displacements: Optional[Dict[str, List[float]]] = None,
    mark_unseen_invisible: bool = True,
) -> Tuple[dict, List[dict]]:
    """Assign previous tracks to current detections with a global one-to-one greedy cost ordering."""
    tracks = memory.setdefault("tracks", {})
    previous = [track for track in tracks.values() if track.get("visible", True)]
    if mark_unseen_invisible:
        for track in tracks.values():
            track["visible"] = False
    candidates = []
    for track in previous:
        for index, detection in enumerate(detections):
            features = _match_features(track, detection, predicted_displacements or {})
            if features["compatible"] and features["center_distance_m"] <= max_movement_m:
                candidates.append((features["cost"], str(track["track_id"]), index, features))
    matches = []
    for _cost, track_id, index, features in _minimum_cost_one_to_one(candidates):
        competing = [item[0] for item in candidates if item[2] == index and item[1] != track_id]
        ambiguous = bool(competing and abs(min(competing) - features["cost"]) < 0.08)
        matches.append((track_id, index, features, ambiguous))
    assigned_detections = {item[1] for item in matches}
    assignments = []
    for track_id, index, features, ambiguous in matches:
        assignments.append(_bind(tracks[track_id], detections[index], scene_revision, features, ambiguous))
    for index, detection in enumerate(detections):
        if index in assigned_detections: continue
        track_id = _new_track_id(tracks, detection)
        tracks[track_id] = {"track_id": track_id, "history": []}
        assignments.append(_bind(tracks[track_id], detection, scene_revision, {"new_track": True}, False))
    memory.setdefault("track_history", []).append({"scene_revision": int(scene_revision), "assignments": assignments})
    return memory, assignments


def _minimum_cost_one_to_one(candidates: List[tuple]) -> List[tuple]:
    """Solve the small detection cost matrix for max-cardinality, minimum cost."""
    track_ids = sorted({item[1] for item in candidates})
    by_track = {
        track_id: sorted((item for item in candidates if item[1] == track_id), key=lambda item: item[0])
        for track_id in track_ids
    }

    @lru_cache(maxsize=None)
    def solve(track_index: int, used_mask: int) -> Tuple[int, float, tuple]:
        if track_index >= len(track_ids):
            return 0, 0.0, ()
        best_count, best_cost, best_pairs = solve(track_index + 1, used_mask)
        for item in by_track[track_ids[track_index]]:
            detection_index = int(item[2])
            if used_mask & (1 << detection_index):
                continue
            count, cost, pairs = solve(track_index + 1, used_mask | (1 << detection_index))
            candidate = (count + 1, cost + float(item[0]), (item,) + pairs)
            if candidate[0] > best_count or (candidate[0] == best_count and candidate[1] < best_cost):
                best_count, best_cost, best_pairs = candidate
        return best_count, best_cost, best_pairs

    return list(solve(0, 0)[2])


def resolve_action_references(action: dict, state: dict, scene_revision: int) -> Tuple[Optional[dict], Optional[dict]]:
    """Resolve track/object refs to current detector ids and reject stale or ambiguous bindings."""
    objects = [obj for obj in state.get("objects", []) if isinstance(obj, dict)]
    output = dict(action)
    for prefix in ("object", "target_object"):
        reference = action.get("{}_ref".format(prefix))
        track_id = action.get("{}_track_id".format(prefix))
        legacy_id = action.get("{}_id".format(prefix))
        obj = None
        reference_object = None
        track_object = None
        if track_id:
            track_object = next((item for item in objects if str(item.get("track_id")) == str(track_id)), None)
            obj = track_object
        if reference:
            parsed = parse_object_ref(reference)
            if parsed is None or parsed[0] != int(scene_revision):
                return None, {"reason": "stale_or_invalid_object_ref", "field": "{}_ref".format(prefix)}
            reference_object = next((item for item in objects if str(item.get("id")) == parsed[1]), None)
            obj = reference_object
        if track_object is not None and reference_object is not None and track_object is not reference_object:
            return None, {"reason": "object_reference_track_mismatch", "field": prefix}
        if track_id and track_object is None:
            return None, {"reason": "selected_track_not_visible", "field": "{}_track_id".format(prefix)}
        if not track_id and not reference and legacy_id is not None:
            declared_revision = action.get("scene_revision", state.get("scene_revision"))
            if declared_revision is None:
                return None, {"reason": "legacy_object_id_requires_scene_revision", "field": "{}_id".format(prefix)}
            if int(declared_revision) != int(scene_revision):
                return None, {"reason": "stale_legacy_object_id", "field": "{}_id".format(prefix)}
            obj = next((item for item in objects if str(item.get("id")) == str(legacy_id)), None)
        if obj is None and prefix == "object":
            return None, {"reason": "selected_object_reference_not_found", "field": prefix}
        if obj is not None:
            if obj.get("tracking_ambiguous"):
                return None, {"reason": "ambiguous_track_requires_reobserve", "track_id": obj.get("track_id")}
            output["{}_id".format(prefix)] = obj.get("id")
            output["{}_ref".format(prefix)] = obj.get("object_ref") or object_ref(scene_revision, obj.get("id"))
            output["{}_track_id".format(prefix)] = obj.get("track_id")
    return output, None


def parse_object_ref(value: Any) -> Optional[Tuple[int, str]]:
    match = re.fullmatch(r"scene_(\d+):obj_(.+)", str(value or ""))
    return (int(match.group(1)), match.group(2)) if match else None


def _bind(track: dict, detection: dict, revision: int, features: dict, ambiguous: bool) -> dict:
    previous_ref = track.get("current_object_ref")
    current_ref = object_ref(revision, detection.get("id"))
    confidence = 0.45 if ambiguous else max(0.0, min(1.0, 1.0 - float(features.get("cost", 0.0))))
    detection.update({
        "object_ref": current_ref,
        "track_id": track["track_id"],
        "semantic_shape": infer_object_shape(detection),
        "tracking_ambiguous": ambiguous,
        "track_match_confidence": confidence,
    })
    track.update({
        "current_object_ref": current_ref, "detector_object_id": detection.get("id"), "label": detection.get("label"),
        "semantic_shape": infer_object_shape(detection), "color": _color(detection), "center_base_m": get_center(detection),
        "dimensions_m": get_size(detection), "bbox": detection.get("bbox_xyxy_px") or detection.get("bbox"),
        "role": detection.get("role", track.get("role")), "state": detection.get("state", track.get("state")),
        "visible": True, "tracking_ambiguous": ambiguous,
    })
    record = {"track_id": track["track_id"], "previous_object_ref": previous_ref, "current_object_ref": current_ref, "match_confidence": confidence, "ambiguous": ambiguous, "match_features": features}
    track.setdefault("history", []).append(record)
    return record


def _match_features(track: dict, detection: dict, predicted: Dict[str, List[float]]) -> dict:
    shape_ok = track.get("semantic_shape") in (None, "unknown", infer_object_shape(detection))
    color_ok = track.get("color") in (None, "unknown", _color(detection))
    old, current = track.get("center_base_m"), get_center(detection)
    prediction = predicted.get(str(track.get("track_id")), [0.0, 0.0, 0.0])
    distance = math.dist([old[i] + prediction[i] for i in range(3)], current[:3]) if old and current else math.inf
    old_size, size = track.get("dimensions_m"), get_size(detection)
    size_delta = sum(abs(float(a) - float(b)) for a, b in zip(old_size, size)) if old_size and size else 0.1
    old_box, box = track.get("bbox"), detection.get("bbox_xyxy_px") or detection.get("bbox")
    bbox_overlap = _bbox_iou(old_box, box)
    cost = distance / 0.15 + size_delta / 0.10 + (1.0 - bbox_overlap) * 0.15
    if track.get("role") in {"base", "structure", "protected"}: cost *= 0.75
    return {"compatible": bool(shape_ok and color_ok and current is not None), "shape_compatible": shape_ok, "color_compatible": color_ok, "center_distance_m": distance, "size_delta_m": size_delta, "bbox_iou": bbox_overlap, "cost": cost}


def _new_track_id(tracks: dict, detection: dict) -> str:
    token = _color(detection)
    if token == "unknown": token = infer_object_shape(detection)
    serial = 1
    while "track_{}_{:02d}".format(token, serial) in tracks: serial += 1
    return "track_{}_{:02d}".format(token, serial)


def _color(obj: dict) -> str:
    label = str(obj.get("label") or "").lower()
    return next((value for value in ("red", "green", "blue", "yellow", "orange", "purple", "cyan") if value in label), "unknown")


def _bbox_iou(first: Any, second: Any) -> float:
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)) or len(first) < 4 or len(second) < 4: return 0.0
    overlap_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0])); overlap_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = overlap_x * overlap_y
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1]); area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-9, area_a + area_b - intersection)
