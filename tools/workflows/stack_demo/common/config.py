"""Validated single-source planner and workspace configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from tools.robot.tool_geometry import TOOL0_TO_TCP_OFFSET_TOOL_M


@dataclass(frozen=True)
class StackDemoConfig:
    """Planner configuration with one authoritative source per physical value."""

    values: Mapping[str, Any]
    workspace: Mapping[str, float]
    organize_layout: Mapping[str, float]
    default_initial_clutter: Mapping[str, float]
    robot_exclusion_geometry: tuple[Mapping[str, float], ...]

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
    default_initial_clutter = _bounds(
        workspace_document.get("default_initial_clutter_bounds"), include_z=False,
    )
    robot_exclusions = tuple(
        _bounds(item, include_z=False)
        for item in workspace_document.get("robot_exclusion_geometry", ())
        if isinstance(item, Mapping)
    )
    _require_sections(planner)
    _validate_planner_values(planner)
    _validate_organize_layout(planner, workspace, organize_layout, default_initial_clutter)
    return StackDemoConfig(
        planner,
        workspace,
        organize_layout,
        default_initial_clutter,
        robot_exclusions,
    )


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
    configured_tcp = tuple(float(value) for value in gripper["tcp_offset_tool_m"])
    if configured_tcp != TOOL0_TO_TCP_OFFSET_TOOL_M:
        raise ValueError(
            "planner tool0->TCP offset must match the measured shared tool geometry "
            f"{TOOL0_TO_TCP_OFFSET_TOOL_M}"
        )
    if float(grasp["minimum_continuous_safe_yaw_span_deg"]) < 10.0:
        raise ValueError("minimum continuous safe grasp yaw span must be at least 10 degrees")
    if abs(float(safety["release_height_extra_m"]) - 0.010) > 1e-9:
        raise ValueError("verified organize release safety gap must remain 10 mm")
    motion = planner["motion"]
    if float(motion["near_object_velocity"]) >= float(motion["high_clearance_rotation_velocity"]):
        raise ValueError("near-object velocity must remain below high-clearance rotation velocity")
    if abs(float(motion["max_wrist_3_start_goal_delta_rad"]) - 1.75) > 1e-9:
        raise ValueError("motion.max_wrist_3_start_goal_delta_rad must remain 1.75")
    if int(planner["policy"].get("max_consecutive_reobserve", 0)) < 1:
        raise ValueError("policy.max_consecutive_reobserve must be at least one")


def _validate_organize_layout(
    planner: Mapping[str, Any],
    workspace: Mapping[str, float],
    layout: Mapping[str, float],
    initial_clutter: Mapping[str, float],
) -> None:
    for bounds_name, bounds in (("organize layout", layout), ("initial clutter", initial_clutter)):
        if not (
            workspace["xmin"] <= bounds["xmin"] < bounds["xmax"] <= workspace["xmax"]
            and workspace["ymin"] <= bounds["ymin"] < bounds["ymax"] <= workspace["ymax"]
        ):
            raise ValueError(f"{bounds_name} bounds must remain inside the configured workspace")
    colors = tuple(str(value) for value in planner["organize"]["colors"])
    width = float(planner["organize"]["target_region_width_y_m"])
    available_y = float(layout["ymax"]) - float(layout["ymin"])
    if len(colors) < 2 or width <= 0.0 or available_y <= len(colors) * width:
        raise ValueError("organize layout must fit fixed-width color regions with positive channels")
    x_overlap = min(layout["xmax"], initial_clutter["xmax"]) > max(layout["xmin"], initial_clutter["xmin"])
    y_overlap = min(layout["ymax"], initial_clutter["ymax"]) > max(layout["ymin"], initial_clutter["ymin"])
    if x_overlap and y_overlap:
        raise ValueError("organize target layout must not overlap the default initial clutter bounds")
