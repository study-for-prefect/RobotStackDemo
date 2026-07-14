"""Load and attach a calibrated base_link workspace to scene observations."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


REQUIRED_BOUNDS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def attach_configured_workspace(state: dict, args: Any) -> dict:
    """Attach configured bounds without replacing bounds supplied by an offline state."""
    path = str(getattr(args, "workspace_bounds_json", "") or "")
    if not path:
        return state
    if not os.path.isfile(path) and (
        state.get("table_bounds") is not None or state.get("workspace_bounds") is not None
    ):
        return state
    workspace = load_workspace_bounds(path, getattr(args, "base_frame", "base_link"))
    if state.get("table_bounds") is None and state.get("workspace_bounds") is None:
        state["workspace_bounds"] = workspace
    state["workspace_bounds_source"] = os.path.abspath(path)
    layout = load_organize_layout_bounds(path, workspace)
    if layout is not None:
        state["organize_layout_bounds"] = layout
    return state


def load_workspace_bounds(path: str, expected_frame: str = "base_link") -> Dict[str, float]:
    """Load a six-direction, SI-unit workspace calibration file."""
    if not os.path.isfile(path):
        raise RuntimeError("WORKSPACE_CONFIGURATION_MISSING: {}".format(path))
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    frame = str(payload.get("frame_id") or "")
    unit = str(payload.get("unit") or "")
    bounds = payload.get("bounds")
    if frame != str(expected_frame):
        raise RuntimeError("Workspace frame must be {}; got {}.".format(expected_frame, frame or "missing"))
    if unit != "meter":
        raise RuntimeError("Workspace unit must be meter; got {}.".format(unit or "missing"))
    if not isinstance(bounds, dict) or any(key not in bounds for key in REQUIRED_BOUNDS):
        raise RuntimeError("Workspace bounds require {}.".format(", ".join(REQUIRED_BOUNDS)))
    normalized = {key: float(bounds[key]) for key in REQUIRED_BOUNDS}
    for lower, upper in (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax")):
        if normalized[lower] >= normalized[upper]:
            raise RuntimeError("Workspace {} must be smaller than {}.".format(lower, upper))
    return normalized


def load_organize_layout_bounds(path: str, workspace: Dict[str, float]):
    """Load the standard-camera observable subset used only for organize destinations."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("organize_layout_bounds")
    if value is None:
        return None
    keys = ("xmin", "xmax", "ymin", "ymax")
    if not isinstance(value, dict) or any(key not in value for key in keys):
        raise RuntimeError("organize_layout_bounds requires xmin/xmax/ymin/ymax")
    normalized = {key: float(value[key]) for key in keys}
    if normalized["xmin"] >= normalized["xmax"] or normalized["ymin"] >= normalized["ymax"]:
        raise RuntimeError("organize_layout_bounds lower values must be smaller than upper values")
    if (
        normalized["xmin"] < workspace["xmin"] or normalized["xmax"] > workspace["xmax"]
        or normalized["ymin"] < workspace["ymin"] or normalized["ymax"] > workspace["ymax"]
    ):
        raise RuntimeError("organize_layout_bounds must be inside workspace bounds")
    return normalized
