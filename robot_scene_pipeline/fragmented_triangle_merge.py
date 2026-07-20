"""Evidence-gated union of one green triangular prism split into two masks."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def merge_fragmented_green_triangle_detections(
    detections: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge only connected, depth-continuous, sloped green mask fragments.

    The output is still labelled as a square/triangle review candidate.  Shape
    classification happens later from the recomputed union point cloud; this
    function establishes only that two detector instances describe one mask.
    """
    remaining = [copy.deepcopy(item) for item in detections]
    merge_log: list[dict[str, Any]] = []
    consumed: set[int] = set()
    output: list[dict[str, Any]] = []
    for first_index, first in enumerate(remaining):
        if first_index in consumed:
            continue
        match = None
        for second_index in range(first_index + 1, len(remaining)):
            if second_index in consumed:
                continue
            evidence = _fragment_pair_evidence(first, remaining[second_index])
            if evidence is not None:
                match = (second_index, evidence)
                break
        if match is None:
            output.append(first)
            continue
        second_index, evidence = match
        second = remaining[second_index]
        consumed.add(second_index)
        merged = _merge_pair(first, second, evidence)
        output.append(merged)
        merge_log.append({
            "kept_candidate_id": merged.get("id"),
            "fragment_candidate_ids": evidence["fragment_candidate_ids"],
            "evidence": evidence,
        })
    return output, merge_log


def _fragment_pair_evidence(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not (_green_square_triangle(first) and _green_square_triangle(second)):
        return None
    first_mask = first.get("_mask_bool")
    second_mask = second.get("_mask_bool")
    if not (
        isinstance(first_mask, np.ndarray) and isinstance(second_mask, np.ndarray)
        and first_mask.shape == second_mask.shape and first_mask.ndim == 2
    ):
        return None
    first_mask = first_mask.astype(bool)
    second_mask = second_mask.astype(bool)
    first_area = int(np.count_nonzero(first_mask))
    second_area = int(np.count_nonzero(second_mask))
    if min(first_area, second_area) < 100:
        return None
    intersection_ratio = float(np.count_nonzero(first_mask & second_mask)) / min(
        first_area, second_area,
    )
    if intersection_ratio > 0.20 or not _adjacent_boxes(first, second):
        return None
    union = first_mask | second_mask
    connected = cv2.morphologyEx(
        union.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
    )
    component_count = cv2.connectedComponents(connected)[0] - 1
    if component_count != 1 or not _has_sloped_exterior(union):
        return None
    support_gap = _scalar_gap(first, second, "local_support_surface", "support_z_base_m")
    top_gap = _direct_scalar_gap(first, second, "top_z_base_m")
    if support_gap is None or top_gap is None or support_gap > 0.004 or top_gap > 0.004:
        return None
    if not (_cube_like_fragment(first) and _cube_like_fragment(second)):
        return None
    union_extents = _combined_point_extents(first, second)
    if union_extents is None:
        return None
    longest, middle, shortest = union_extents
    if not (
        0.030 <= longest <= 0.070
        and 0.012 <= shortest <= middle <= 0.035
        and 1.45 <= longest / max(middle, 1e-9) <= 2.8
        and middle / max(shortest, 1e-9) <= 1.6
    ):
        return None
    return {
        "fragment_candidate_ids": [first.get("id"), second.get("id")],
        "same_measured_color": True,
        "mask_union_connected": True,
        "sloped_union_exterior": True,
        "mask_intersection_over_smaller": round(intersection_ratio, 4),
        "support_surface_gap_m": round(support_gap, 6),
        "top_surface_gap_m": round(top_gap, 6),
        "combined_pointcloud_extents_m": [round(value, 6) for value in union_extents],
        "individual_fragments_cube_like": True,
        "shape_classification_deferred_to_union_geometry": True,
    }


def _green_square_triangle(item: Mapping[str, Any]) -> bool:
    label = str(item.get("label") or "").lower()
    return bool(
        str(item.get("visual_color") or "").lower() == "green"
        and any(token in label for token in ("square", "triangle"))
        and item.get("pointcloud_geometry_valid") is True
    )


def _adjacent_boxes(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    boxes = [list(item.get("bbox") or item.get("bbox_xyxy_px") or ()) for item in (first, second)]
    if any(len(box) < 4 for box in boxes):
        return False
    a, b = boxes
    x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    x_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    widths = [max(1.0, box[2] - box[0]) for box in boxes]
    heights = [max(1.0, box[3] - box[1]) for box in boxes]
    return bool(
        y_overlap / min(heights) >= 0.65 and x_gap <= 4.0
        or x_overlap / min(widths) >= 0.65 and y_gap <= 4.0
    )


def _has_sloped_exterior(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True).reshape(-1, 2)
    if len(polygon) < 3:
        return False
    _, _, width, height = cv2.boundingRect(contour)
    minimum_edge = 0.15 * math.hypot(width, height)
    for index, point in enumerate(polygon):
        delta = polygon[(index + 1) % len(polygon)] - point
        length = float(np.linalg.norm(delta))
        if length < minimum_edge:
            continue
        angle = abs(math.degrees(math.atan2(float(delta[1]), float(delta[0])))) % 90.0
        axis_error = min(angle, 90.0 - angle)
        if axis_error >= 12.0:
            return True
    return False


def _cube_like_fragment(item: Mapping[str, Any]) -> bool:
    dimensions = item.get("dimensions_m")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3:
        return False
    values = sorted(float(value) for value in dimensions[:3])
    return values[0] >= 0.012 and values[-1] / max(values[0], 1e-9) <= 1.35


def _combined_point_extents(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    arrays = [item.get("_object_points_base") for item in (first, second)]
    if not all(isinstance(value, np.ndarray) and value.ndim == 2 for value in arrays):
        return None
    points = np.concatenate(arrays, axis=0)
    if len(points) < 80:
        return None
    centered = points - np.median(points, axis=0)
    _, vectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    projections = centered.dot(vectors)
    extents = np.percentile(projections, 98.0, axis=0) - np.percentile(projections, 2.0, axis=0)
    return tuple(sorted((float(value) for value in extents), reverse=True))


def _scalar_gap(
    first: Mapping[str, Any], second: Mapping[str, Any], container: str, key: str,
) -> float | None:
    values = []
    for item in (first, second):
        nested = item.get(container)
        if not isinstance(nested, Mapping) or nested.get(key) is None:
            return None
        values.append(float(nested[key]))
    return abs(values[0] - values[1])


def _direct_scalar_gap(
    first: Mapping[str, Any], second: Mapping[str, Any], key: str,
) -> float | None:
    if first.get(key) is None or second.get(key) is None:
        return None
    return abs(float(first[key]) - float(second[key]))


def _merge_pair(
    first: Mapping[str, Any], second: Mapping[str, Any], evidence: Mapping[str, Any],
) -> dict[str, Any]:
    preferred = max((first, second), key=lambda item: float(item.get("confidence", 0.0)))
    merged = copy.deepcopy(preferred)
    union = first["_mask_bool"].astype(bool) | second["_mask_bool"].astype(bool)
    ys, xs = np.nonzero(union)
    merged.update({
        "_mask_bool": union,
        "bbox": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        "center_px": [float(np.median(xs)), float(np.median(ys))],
        "mask_area_px": int(np.count_nonzero(union)),
        "fragment_merge_candidate_ids": list(evidence["fragment_candidate_ids"]),
        "fragment_merge_evidence": dict(evidence),
        "fragment_union_requires_square_triangle_review": True,
    })
    return merged
