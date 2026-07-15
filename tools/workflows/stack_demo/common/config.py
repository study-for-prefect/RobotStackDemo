"""Validated single-source planner and workspace configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StackDemoConfig:
    """Planner configuration with one authoritative source per physical value."""

    values: Mapping[str, Any]
    workspace: Mapping[str, float]
    organize_layout: Mapping[str, float]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing planner configuration section: {name}")
        return value


def load_stack_demo_config(
    planner_path: str | Path,
    workspace_path: str | Path,
) -> StackDemoConfig:
    """Load and validate physical constants without copying workspace values."""
    planner = _load_object(planner_path)
    workspace_document = _load_object(workspace_path)
    if planner.get("schema_version") != "stack_demo_planner_v1":
        raise ValueError("planner config must use stack_demo_planner_v1")
    if planner.get("coordinate_frame") != "base_link":
        raise ValueError("planner coordinate_frame must be base_link")
    if workspace_document.get("frame_id") != "base_link":
        raise ValueError("workspace frame_id must be base_link")
    if workspace_document.get("unit") != "meter":
        raise ValueError("workspace unit must be meter")
    workspace = _bounds(workspace_document.get("bounds"), include_z=True)
    organize_layout = _bounds(
        workspace_document.get("organize_layout_bounds"), include_z=False,
    )
    _require_sections(planner)
    _validate_planner_values(planner)
    return StackDemoConfig(planner, workspace, organize_layout)


def _load_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def _bounds(value: Any, *, include_z: bool) -> dict[str, float]:
    keys = ["xmin", "xmax", "ymin", "ymax"]
    if include_z:
        keys.extend(["zmin", "zmax"])
    if not isinstance(value, dict) or any(key not in value for key in keys):
        raise ValueError(f"workspace bounds require {', '.join(keys)}")
    output = {key: float(value[key]) for key in keys}
    for low, high in (("xmin", "xmax"), ("ymin", "ymax")):
        if output[low] >= output[high]:
            raise ValueError(f"workspace {low} must be lower than {high}")
    if include_z and output["zmin"] >= output["zmax"]:
        raise ValueError("workspace zmin must be lower than zmax")
    return output


def _require_sections(planner: Mapping[str, Any]) -> None:
    required = ("gripper", "safety", "grasp", "clearing", "motion", "policy", "organize", "house")
    missing = [name for name in required if not isinstance(planner.get(name), dict)]
    if missing:
        raise ValueError(f"missing planner configuration sections: {missing}")


def _validate_planner_values(planner: Mapping[str, Any]) -> None:
    gripper = planner["gripper"]
    grasp = planner["grasp"]
    safety = planner["safety"]
    if float(gripper["open_inner_width_m"]) >= float(gripper["open_outer_width_m"]):
        raise ValueError("GF225 inner opening must be smaller than its outer outline")
    if float(grasp["minimum_continuous_safe_yaw_span_deg"]) < 10.0:
        raise ValueError("minimum continuous safe grasp yaw span must be at least 10 degrees")
    if abs(float(safety["release_height_extra_m"]) - 0.010) > 1e-9:
        raise ValueError("verified organize release safety gap must remain 10 mm")
    motion = planner["motion"]
    if float(motion["near_object_velocity"]) >= float(motion["high_clearance_rotation_velocity"]):
        raise ValueError("near-object velocity must remain below high-clearance rotation velocity")
