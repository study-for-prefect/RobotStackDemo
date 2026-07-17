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
    _validate_house_staging(planner, workspace)
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
    if abs(float(safety["ordinary_release_height_extra_m"])) > 1e-9:
        raise ValueError("ordinary placement must release at the table-contact center height")
    if abs(float(safety["special_shape_release_height_extra_m"]) - 0.010) > 1e-9:
        raise ValueError("special-shape placement safety gap must remain 10 mm")
    motion = planner["motion"]
    if float(motion["near_object_velocity"]) >= float(motion["high_clearance_rotation_velocity"]):
        raise ValueError("near-object velocity must remain below high-clearance rotation velocity")
    if abs(float(motion["max_wrist_3_start_goal_delta_rad"]) - 1.75) > 1e-9:
        raise ValueError("motion.max_wrist_3_start_goal_delta_rad must remain 1.75")
    if int(planner["policy"].get("max_consecutive_reobserve", 0)) < 1:
        raise ValueError("policy.max_consecutive_reobserve must be at least one")
    if abs(float(planner["house"]["support_inner_gap_m"]) - 0.015) > 1e-9:
        raise ValueError("house.support_inner_gap_m must remain 0.015")


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
    size = float(planner["organize"]["target_region_size_m"])
    available_x = float(layout["xmax"]) - float(layout["xmin"])
    available_y = float(layout["ymax"]) - float(layout["ymin"])
    quadrants = planner["organize"].get("quadrant_by_color")
    required = {
        "image_top_left", "image_bottom_left",
        "image_top_right", "image_bottom_right",
    }
    if (
        len(colors) != 4 or size <= 0.0
        or available_x <= 2.0 * size or available_y <= 2.0 * size
        or not isinstance(quadrants, Mapping)
        or {str(quadrants.get(color)) for color in colors} != required
    ):
        raise ValueError(
            "organize layout must fit four square image quadrants with positive x/y channels"
        )
    # The calibrated 640x480 view cannot contain two 80 mm image columns only
    # on the far side of the default clutter bounds.  Physical slot generation
    # checks current objects, so a target anchor overlapping live clutter is
    # simply unavailable rather than treating the whole square as forbidden.


def _validate_house_staging(
    planner: Mapping[str, Any],
    workspace: Mapping[str, float],
) -> None:
    """Require several calibrated, camera-visible fallback staging centers."""
    house = planner["house"]
    camera_bounds = _bounds(
        house.get("orientation_staging_camera_bounds_base_m"), include_z=False,
    )
    if not (
        workspace["xmin"] <= camera_bounds["xmin"] < camera_bounds["xmax"] <= workspace["xmax"]
        and workspace["ymin"] <= camera_bounds["ymin"] < camera_bounds["ymax"] <= workspace["ymax"]
    ):
        raise ValueError("house orientation staging camera bounds must remain in workspace")
    candidates = house.get("orientation_staging_candidates_base_m")
    if not isinstance(candidates, list) or len(candidates) < 4:
        raise ValueError(
            "house.orientation_staging_candidates_base_m must contain at least four points"
        )
    inset = float(planner["safety"]["workspace_inset_m"])
    seen: set[tuple[float, float]] = set()
    for value in candidates:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("each house orientation staging point must contain x and y")
        point = (float(value[0]), float(value[1]))
        if point in seen:
            raise ValueError("house orientation staging points must be unique")
        seen.add(point)
        if not (
            workspace["xmin"] + inset <= point[0] <= workspace["xmax"] - inset
            and workspace["ymin"] + inset <= point[1] <= workspace["ymax"] - inset
            and camera_bounds["xmin"] <= point[0] <= camera_bounds["xmax"]
            and camera_bounds["ymin"] <= point[1] <= camera_bounds["ymax"]
        ):
            raise ValueError(
                "house orientation staging point must remain in the inset workspace and camera view"
            )
