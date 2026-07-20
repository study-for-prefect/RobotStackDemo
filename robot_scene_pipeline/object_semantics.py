"""Small factual color/shape helpers shared by perception and task state."""

from __future__ import annotations

from typing import Optional


SUPPORTED_BLOCK_COLORS = ("red", "green", "blue", "yellow")


def infer_object_color(obj: dict) -> Optional[str]:
    """Prefer measured visual color over a detector label token."""
    visual = str(obj.get("visual_color") or obj.get("color") or "").strip().lower()
    if visual in SUPPORTED_BLOCK_COLORS:
        return visual
    tokens = set(
        str(obj.get("label") or obj.get("class_name") or "")
        .lower()
        .replace("_", " ")
        .split()
    )
    matches = [color for color in SUPPORTED_BLOCK_COLORS if color in tokens]
    return matches[0] if len(matches) == 1 else None


def infer_object_shape(obj: dict) -> str:
    explicit = str(
        obj.get("semantic_shape") or obj.get("fused_shape") or obj.get("shape") or ""
    ).strip().lower()
    return normalize_semantic_shape(explicit or str(obj.get("label") or ""))


def normalize_semantic_shape(label: str) -> str:
    value = str(label or "").strip().lower().replace("-", " ").replace("_", " ")
    if "concave" in value:
        return "concave_rectangle"
    if "triangle" in value:
        return "triangle"
    if "rectangle" in value:
        return "rectangle"
    if "square" in value:
        return "square"
    return "unknown"
