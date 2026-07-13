"""Load and attach a calibrated base_link workspace to scene observations."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


REQUIRED_BOUNDS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def attach_configured_workspace(state: dict, args: Any) -> dict:
    """Attach configured bounds without replacing bounds supplied by an offline state."""
    if state.get("table_bounds") is not None or state.get("workspace_bounds") is not None:
        return state
    path = str(getattr(args, "workspace_bounds_json", "") or "")
    if not path:
        return state
    state["workspace_bounds"] = load_workspace_bounds(path, getattr(args, "base_frame", "base_link"))
    state["workspace_bounds_source"] = os.path.abspath(path)
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
