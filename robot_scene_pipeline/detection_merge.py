"""3D duplicate detection merge helpers."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

ObjectDict = Dict[str, Any]


def _as_float_list(value: Any, length: int) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    try:
        return [float(item) for item in value[:length]]
    except (TypeError, ValueError):
        return None


def _bbox_iou(a: Any, b: Any) -> float:
    box_a = _as_float_list(a, 4)
    box_b = _as_float_list(b, 4)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _object_score(obj: ObjectDict, preferred_id: Optional[Any]) -> Tuple[int, float, int, int]:
    is_preferred = int(preferred_id is not None and str(obj.get("id")) == str(preferred_id))
    confidence = float(obj.get("confidence") or obj.get("score") or 0.0)
    has_geometry = int(_as_float_list(obj.get("geometry_center_m"), 3) is not None)
    has_size = int(_as_float_list(obj.get("dimensions_m"), 3) is not None)
    return is_preferred, confidence, has_geometry + has_size, len(str(obj.get("id")))


def _similar_3d(
    a: ObjectDict,
    b: ObjectDict,
    center_threshold_m: float,
    dimension_threshold_m: float,
    bbox_iou_threshold: float,
) -> bool:
    if str(a.get("label")) != str(b.get("label")):
        return False
    center_a = _as_float_list(a.get("geometry_center_m"), 3)
    center_b = _as_float_list(b.get("geometry_center_m"), 3)
    if center_a is None or center_b is None:
        return False
    center_distance = math.sqrt(sum((center_a[i] - center_b[i]) ** 2 for i in range(3)))
    if center_distance > float(center_threshold_m):
        return False
    dims_a = _as_float_list(a.get("dimensions_m"), 3)
    dims_b = _as_float_list(b.get("dimensions_m"), 3)
    dims_close = (
        dims_a is not None
        and dims_b is not None
        and max(abs(dims_a[i] - dims_b[i]) for i in range(3)) <= float(dimension_threshold_m)
    )
    height_close = (
        dims_a is not None
        and dims_b is not None
        and abs(dims_a[2] - dims_b[2]) <= float(dimension_threshold_m)
    )
    bbox_iou = max(
        _bbox_iou(a.get("bbox_xyxy"), b.get("bbox_xyxy")),
        _bbox_iou(a.get("bbox"), b.get("bbox")),
    )
    return bool(dims_close or (height_close and bbox_iou >= float(bbox_iou_threshold)))


def merge_duplicate_objects_3d(
    objects: Iterable[ObjectDict],
    *,
    preferred_id: Optional[Any] = None,
    center_threshold_m: float = 0.025,
    dimension_threshold_m: float = 0.020,
    bbox_iou_threshold: float = 0.50,
) -> List[ObjectDict]:
    """Merge same-label detections that describe the same 3D object."""
    merged: List[ObjectDict] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        match_index = None
        for index, existing in enumerate(merged):
            if _similar_3d(existing, obj, center_threshold_m, dimension_threshold_m, bbox_iou_threshold):
                match_index = index
                break
        if match_index is None:
            merged.append(copy.deepcopy(obj))
            continue
        existing = merged[match_index]
        preferred = max([existing, obj], key=lambda item: _object_score(item, preferred_id))
        duplicate = obj if preferred is existing else existing
        output = copy.deepcopy(preferred)
        duplicate_ids = list(existing.get("merged_duplicate_ids", []))
        for candidate_id in (existing.get("id"), obj.get("id"), duplicate.get("id")):
            if candidate_id is not None and str(candidate_id) != str(output.get("id")):
                if candidate_id not in duplicate_ids:
                    duplicate_ids.append(candidate_id)
        output["merged_duplicate_ids"] = duplicate_ids
        output["merge_reason"] = "same_label_near_3d_center_similar_dimensions"
        output["merge_confidence"] = max(
            float(existing.get("merge_confidence") or 0.0),
            float(existing.get("confidence") or existing.get("score") or 0.0),
            float(obj.get("confidence") or obj.get("score") or 0.0),
        )
        merged[match_index] = output
    return merged
