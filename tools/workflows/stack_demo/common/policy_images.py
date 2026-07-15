"""Create the latest clean RGB and a minimal short-ID overlay for Qwen."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


def build_policy_images(
    observation: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[str, ...]:
    """Never reuse old images and never draw geometry text on the overlay."""
    source = (
        observation.get("snapshot_image")
        or observation.get("image_path")
        or observation.get("color_image_path")
        or observation.get("rgb_image")
    )
    if not source or not Path(str(source)).is_file():
        return ()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    clean = directory / "policy_latest_clean_rgb.jpg"
    shutil.copyfile(str(source), clean)
    overlay = directory / "policy_latest_id_overlay.jpg"
    try:
        from PIL import Image, ImageDraw

        image = Image.open(clean).convert("RGB")
        draw = ImageDraw.Draw(image)
        for obj in observation.get("objects", []):
            if not isinstance(obj, Mapping):
                continue
            bbox = obj.get("bbox_xyxy_px") or obj.get("bbox") or obj.get("bbox_xyxy")
            if not isinstance(bbox, Sequence) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(float(value)) for value in bbox[:4]]
            short_id = str(obj.get("id", "?"))
            draw.rectangle((x1, y1, x2, y2), outline="yellow", width=3)
            draw.text((x1 + 2, y1 + 2), short_id, fill="yellow", stroke_width=2, stroke_fill="black")
        image.save(overlay, quality=92)
        return str(clean), str(overlay)
    except Exception:
        return (str(clean),)
