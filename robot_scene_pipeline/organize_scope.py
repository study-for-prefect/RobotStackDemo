"""Define the factual scene scope for color-organization tasks."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .task_geometry import object_inside_workspace


SUPPORTED_BLOCK_COLORS = ("red", "green", "blue", "yellow")


def color_value_from_label(label: object) -> Optional[str]:
    """Return one unambiguous detector color token."""
    tokens = set(str(label or "").lower().replace("_", " ").split())
    matches = [color for color in SUPPORTED_BLOCK_COLORS if color in tokens]
    return matches[0] if len(matches) == 1 else None


def color_value_from_object(obj: object) -> Optional[str]:
    """Prefer visual color evidence, with detector label as a compatibility fallback."""
    if not isinstance(obj, dict):
        return None
    visual_color = str(obj.get("visual_color") or "").lower()
    if visual_color in SUPPORTED_BLOCK_COLORS:
        return visual_color
    return color_value_from_label(obj.get("label"))


def organize_scope_objects(state: dict) -> List[dict]:
    """Keep color-labeled blocks whose full footprint is inside calibrated bounds."""
    workspace = state.get("table_bounds") or state.get("workspace_bounds")
    candidates = []
    for obj in state.get("objects", []):
        if not isinstance(obj, dict) or obj.get("is_workspace"):
            continue
        if color_value_from_object(obj) is None:
            continue
        if _detection_clipped_by_image_edge(obj, state) and not _edge_geometry_usable(obj):
            continue
        if workspace is not None and not object_inside_workspace(obj, workspace):
            continue
        candidates.append(obj)
    output = []
    for obj in sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        if any(_same_detection(obj, kept) for kept in output):
            continue
        output.append(obj)
    return sorted(output, key=lambda item: int(item.get("id", 0)))


def organize_excluded_objects(state: dict) -> Iterable[dict]:
    included = {id(obj) for obj in organize_scope_objects(state)}
    return [obj for obj in state.get("objects", []) if isinstance(obj, dict) and id(obj) not in included]


def _same_detection(first: dict, second: dict) -> bool:
    if str(first.get("label") or "").lower() != str(second.get("label") or "").lower():
        return False
    first_center = first.get("geometry_center_m") or first.get("center_3d_base_m")
    second_center = second.get("geometry_center_m") or second.get("center_3d_base_m")
    if not isinstance(first_center, list) or not isinstance(second_center, list):
        return False
    squared_distance = sum(
        (float(first_center[index]) - float(second_center[index])) ** 2 for index in range(3)
    )
    return squared_distance <= 0.003 ** 2


def _detection_clipped_by_image_edge(obj: dict, state: dict) -> bool:
    bbox = obj.get("bbox_xyxy_px") or obj.get("bbox")
    profile = state.get("camera_profile") or {}
    width = float(profile.get("color_width") or 640)
    height = float(profile.get("color_height") or 480)
    if not isinstance(bbox, list) or len(bbox) < 4:
        return False
    margin_px = 2.0
    return (
        float(bbox[0]) <= margin_px or float(bbox[1]) <= margin_px
        or float(bbox[2]) >= width - margin_px or float(bbox[3]) >= height - margin_px
    )


def _edge_geometry_usable(obj: dict) -> bool:
    """Keep a clipped block only when RGB-D still produced plausible grasp geometry."""
    center = obj.get("geometry_center_m") or obj.get("center_3d_base_m")
    dimensions = obj.get("dimensions_m")
    if not obj.get("pointcloud_geometry_valid") or not obj.get("depth_geometry_observable"):
        return False
    if not isinstance(center, list) or len(center) < 3:
        return False
    if not isinstance(dimensions, list) or len(dimensions) < 3:
        return False
    try:
        dx, dy, dz = [float(value) for value in dimensions[:3]]
    except (TypeError, ValueError):
        return False
    return 0.015 <= dx <= 0.060 and 0.015 <= dy <= 0.060 and 0.015 <= dz <= 0.060
