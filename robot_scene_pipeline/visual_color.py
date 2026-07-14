"""Conservative visual color evidence for tabletop block detections."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np


SUPPORTED_VISUAL_COLORS = ("red", "green", "blue", "yellow")


def _color_for_hue(hue: np.ndarray) -> np.ndarray:
    """Map OpenCV hue values to supported colors; -1 means unsupported."""
    result = np.full(hue.shape, -1, dtype=np.int8)
    result[(hue <= 10) | (hue >= 170)] = 0  # red wraps around 0
    result[(hue >= 39) & (hue <= 89)] = 1
    result[(hue >= 90) & (hue <= 135)] = 2
    result[(hue >= 15) & (hue <= 38)] = 3
    return result


def _bbox_mask(det: Dict[str, Any], image_shape: tuple[int, ...]) -> Optional[np.ndarray]:
    bbox = det.get("bbox") or det.get("bbox_xyxy_px")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    height, width = image_shape[:2]
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    # An inset reduces table/background contamination when segmentation is absent.
    inset_x = 0.12 * max(0.0, x2 - x1)
    inset_y = 0.12 * max(0.0, y2 - y1)
    left = max(0, min(width, int(round(x1 + inset_x))))
    right = max(0, min(width, int(round(x2 - inset_x))))
    top = max(0, min(height, int(round(y1 + inset_y))))
    bottom = max(0, min(height, int(round(y2 - inset_y))))
    if right <= left or bottom <= top:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _detection_mask(det: Dict[str, Any], image_shape: tuple[int, ...]) -> tuple[Optional[np.ndarray], str]:
    mask = det.get("_mask_bool")
    if isinstance(mask, np.ndarray) and mask.shape == image_shape[:2] and np.count_nonzero(mask):
        return mask.astype(bool, copy=False), "instance_mask_hsv"
    return _bbox_mask(det, image_shape), "bbox_inset_hsv"


def attach_visual_colors(
    detections: Iterable[Dict[str, Any]],
    frame_bgr: Any,
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    minimum_pixels: int = 40,
    minimum_confidence: float = 0.55,
) -> Iterable[Dict[str, Any]]:
    """Attach conservative HSV color evidence without changing detector labels."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        return detections
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    candidate_by_id = {
        item.get("id"): item for item in (candidates or []) if isinstance(item, dict)
    }
    for det in detections:
        if not isinstance(det, dict):
            continue
        for key in (
            "visual_color", "visual_color_confidence", "visual_color_source",
            "visual_color_pixel_count",
        ):
            det.pop(key, None)
        mask, source = _detection_mask(det, frame_bgr.shape)
        if mask is None:
            continue
        pixels = hsv[mask]
        if not pixels.size:
            continue
        chromatic = pixels[(pixels[:, 1] >= 70) & (pixels[:, 2] >= 30)]
        if len(chromatic) < int(minimum_pixels):
            continue
        mapped = _color_for_hue(chromatic[:, 0])
        counts = np.bincount(mapped[mapped >= 0], minlength=len(SUPPORTED_VISUAL_COLORS))
        if not counts.size or int(counts.max()) < int(minimum_pixels):
            continue
        index = int(np.argmax(counts))
        confidence = float(counts[index]) / float(len(chromatic))
        if confidence < float(minimum_confidence):
            continue
        fields = {
            "visual_color": SUPPORTED_VISUAL_COLORS[index],
            "visual_color_confidence": round(confidence, 4),
            "visual_color_source": source,
            "visual_color_pixel_count": int(counts[index]),
        }
        det.update(fields)
        candidate = candidate_by_id.get(det.get("id"))
        if candidate is not None:
            candidate.update(fields)
    return detections
