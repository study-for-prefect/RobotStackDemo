"""Stable cross-frame object references and one-to-one detection rebinding."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .geometry_relations import get_center, get_size
from .object_semantics import infer_object_shape


def object_ref(scene_revision: int, detector_object_id: Any) -> str:
    """Return the only frame-local object identifier exposed to a VLM."""
    return "scene_{}:obj_{}".format(int(scene_revision), detector_object_id)


def update_scene_tracks(
    memory: dict, detections: List[dict], scene_revision: int,
    max_movement_m: float = 0.15, predicted_displacements: Optional[Dict[str, List[float]]] = None,
    predicted_centers: Optional[Dict[str, List[float]]] = None,
    mark_unseen_invisible: bool = True,
) -> Tuple[dict, List[dict]]:
    """Assign previous tracks to current detections with a global one-to-one greedy cost ordering."""
    # The detector can occasionally emit the same physical block twice with
    # identical metric geometry.  Suppress those duplicates before assignment;
    # otherwise a second persistent track is created and may steal the identity
    # from the original track when only one detection remains in the next frame.
    detections[:] = _deduplicate_detections(detections)
    tracks = memory.setdefault("tracks", {})
    # A wrist-camera observation can legitimately miss a grasped object for one
    # or two revisions.  Keep recently seen tracks eligible for explicit
    # rebinding instead of replacing their identity with a detector id.
    previous = [
        track for track in tracks.values()
        if (
            int(scene_revision) - int(track.get("last_seen_revision", scene_revision)) <= 3
            or track.get("manipulation_state") in {
                "held_by_gripper", "placed", "placed_unverified", "unresolved",
            }
        )
    ]
    if mark_unseen_invisible:
        for track in tracks.values():
            track["visible"] = False
    candidates = []
    for track in previous:
        for index, detection in enumerate(detections):
            features = _match_features(
                track, detection, predicted_displacements or {}, predicted_centers or {},
            )
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


def _deduplicate_detections(detections: List[dict]) -> List[dict]:
    """Keep one metric detection per same-color/shape 8 mm spatial cluster."""
    unique: List[dict] = []
    for detection in detections:
        center = get_center(detection)
        duplicate = False
        for accepted in unique:
            accepted_center = get_center(accepted)
            if (
                center is not None
                and accepted_center is not None
                and _color(detection) == _color(accepted)
                and infer_object_shape(detection) == infer_object_shape(accepted)
                and math.dist(center[:3], accepted_center[:3]) <= 0.008
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(detection)
    return unique


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


def _bind(track: dict, detection: dict, revision: int, features: dict, ambiguous: bool) -> dict:
    previous_ref = track.get("current_object_ref")
    current_ref = object_ref(revision, detection.get("id"))
    confidence = 0.45 if ambiguous else max(0.0, min(1.0, 1.0 - float(features.get("cost", 0.0))))
    measured_size = get_size(detection)
    previous_size = get_size({"dimensions_m": track.get("dimensions_m")})
    if measured_size is None and previous_size is not None:
        # A grasped object can remain detectable near the wrist camera while
        # its point cloud is clipped by the image boundary.  Preserve only the
        # already-verified size of the same rebound track; its current center
        # remains measured from this frame.
        measured_size = list(previous_size)
        detection["dimensions_m"] = list(previous_size)
        detection["dimensions_temporal_fallback"] = True
        detection["dimensions_source"] = "previous_valid_same_track"
        detection["dimensions_source_revision"] = track.get("last_seen_revision")
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
        "dimensions_m": measured_size, "bbox": detection.get("bbox_xyxy_px") or detection.get("bbox"),
        "role": detection.get("role", track.get("role")), "state": detection.get("state", track.get("state")),
        "visible": True, "tracking_ambiguous": ambiguous,
        "last_seen_revision": int(revision),
    })
    record = {"track_id": track["track_id"], "previous_object_ref": previous_ref, "current_object_ref": current_ref, "match_confidence": confidence, "ambiguous": ambiguous, "match_features": features}
    track.setdefault("history", []).append(record)
    return record


def _match_features(
    track: dict,
    detection: dict,
    predicted: Dict[str, List[float]],
    predicted_centers: Dict[str, List[float]],
) -> dict:
    shape_ok = track.get("semantic_shape") in (None, "unknown", infer_object_shape(detection))
    color_ok = track.get("color") in (None, "unknown", _color(detection))
    old, current = track.get("center_base_m"), get_center(detection)
    track_id = str(track.get("track_id"))
    prediction = predicted.get(track_id, [0.0, 0.0, 0.0])
    anchored = predicted_centers.get(track_id)
    expected = anchored if anchored is not None else [old[i] + prediction[i] for i in range(3)] if old else None
    distance = math.dist(expected[:3], current[:3]) if expected and current else math.inf
    action_anchor_compatible = bool(anchored is None or distance <= 0.06)
    old_size, size = track.get("dimensions_m"), get_size(detection)
    size_measurement_available = bool(size)
    if old_size and size:
        size_delta = sum(abs(float(a) - float(b)) for a, b in zip(old_size, size))
    elif old_size:
        # Missing current geometry is common near the wrist-camera boundary.
        # It is uncertainty, not evidence of a 10 cm size mismatch.
        size_delta = 0.015
    else:
        size_delta = 0.0
    old_box, box = track.get("bbox"), detection.get("bbox_xyxy_px") or detection.get("bbox")
    bbox_overlap = _bbox_iou(old_box, box)
    cost = distance / 0.15 + size_delta / 0.10 + (1.0 - bbox_overlap) * 0.15
    if track.get("role") in {"base", "structure", "protected"}: cost *= 0.75
    return {
        "compatible": bool(
            shape_ok and color_ok and current is not None and action_anchor_compatible
        ),
        "shape_compatible": shape_ok,
        "color_compatible": color_ok,
        "center_distance_m": distance,
        "action_anchor_compatible": action_anchor_compatible,
        "size_delta_m": size_delta,
        "size_measurement_available": size_measurement_available,
        "bbox_iou": bbox_overlap,
        "cost": cost,
        "prediction_source": "action_pose_anchor" if anchored is not None else "displacement_or_last_seen",
    }


def _new_track_id(tracks: dict, detection: dict) -> str:
    token = _color(detection)
    if token == "unknown": token = infer_object_shape(detection)
    serial = 1
    while "track_{}_{:02d}".format(token, serial) in tracks: serial += 1
    return "track_{}_{:02d}".format(token, serial)


def _color(obj: dict) -> str:
    visual_color = str(obj.get("visual_color") or "").lower()
    if visual_color:
        return visual_color
    label = str(obj.get("label") or "").lower()
    return next((value for value in ("red", "green", "blue", "yellow", "orange", "purple", "cyan") if value in label), "unknown")


def _bbox_iou(first: Any, second: Any) -> float:
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)) or len(first) < 4 or len(second) < 4: return 0.0
    overlap_x = max(0.0, min(first[2], second[2]) - max(first[0], second[0])); overlap_y = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = overlap_x * overlap_y
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1]); area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1e-9, area_a + area_b - intersection)
