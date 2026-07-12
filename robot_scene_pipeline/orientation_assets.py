"""Prepare full-resolution RGB/depth crops for roof and triangle VLM observations."""

from __future__ import annotations

import copy
import os
from typing import Optional, Sequence


def prepare_orientation_candidate_assets(state: dict, output_dir: str) -> dict:
    """Attach per-object crop paths while preserving the original full-scene assets."""
    output = copy.deepcopy(state)
    os.makedirs(output_dir, exist_ok=True)
    rgb_path = state.get("snapshot_image") or state.get("rgb_image")
    depth_path = state.get("depth_colormap_image") or state.get("depth_image")
    for obj in output.get("objects", []):
        label = str(obj.get("label") or "").lower()
        if "rectangle" not in label and "triangle" not in label:
            continue
        bbox = obj.get("bbox_xyxy_px") or obj.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            obj["orientation_asset_status"] = "bbox_unavailable"
            continue
        object_id = str(obj.get("id")).replace(os.sep, "_")
        rgb_crop = _crop_image(rgb_path, bbox, os.path.join(output_dir, "object_{}_rgb.png".format(object_id)))
        depth_crop = _crop_image(depth_path, bbox, os.path.join(output_dir, "object_{}_depth.png".format(object_id)))
        if rgb_crop:
            obj["rgb_crop"] = rgb_crop
        if depth_crop:
            obj["depth_crop"] = depth_crop
        obj["orientation_asset_status"] = "ready" if rgb_crop else "rgb_crop_unavailable"
    return output


def _crop_image(source_path: Optional[str], bbox: Sequence[float], output_path: str) -> Optional[str]:
    if not source_path or not os.path.exists(source_path):
        return None
    try:
        import cv2

        image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
        x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or not cv2.imwrite(output_path, crop):
            return None
        return output_path
    except Exception:
        return None
