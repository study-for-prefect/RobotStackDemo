"""Code-owned target poses and orientation gates for each house role."""

from __future__ import annotations

from typing import Any, Mapping

from ..clutter.edge_generation import PlacementTarget
from ..common.action_edges import ActionType
from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .state import HouseTaskState
from .structure import object_at_pose, role_observation_checks


def house_placement_target(
    scene: ClutterSceneState,
    state: HouseTaskState,
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
) -> PlacementTarget | None:
    pose = _role_pose(scene, state, obj, role, config)
    checks: dict[str, bool] = {
        "transport_safe": True,
        "place_descent_safe": True,
        "release_safe": True,
        "return_safe": True,
        "protected_safe": True,
    }
    action_type = ActionType.PLACE_HOUSE_ROLE
    planned = object_at_pose(obj, pose)
    planned_bindings = {**dict(state.role_bindings), role: obj.track_id}
    role_valid, role_checks = role_observation_checks(
        scene, planned_bindings, role, config, role_object=planned,
    )
    checks.update({f"structure_{key}": value for key, value in role_checks.items() if isinstance(value, bool)})
    if not role_valid:
        if role in {"roof", "triangle_top"}:
            return staging_orientation_target(obj, role, config)
        return None
    if state.current_repair_state and role in state.current_repair_state.get("repair_required_roles", []):
        action_type = ActionType.REPAIR_STRUCTURE
    return PlacementTarget(
        action_type=action_type,
        target_region_id=None,
        task_role=role,
        place_pose=pose,
        task_progress_gain=1.0,
        expected_effects=(f"complete_house_role:{role}", "protect_completed_structure"),
        precheck_results=checks,
    )


def staging_orientation_target(
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
) -> PlacementTarget:
    x = float(config.workspace["xmax"]) - 0.06
    y = float(config.workspace["ymin"]) + 0.06
    regrasp = bool(obj.source.get("at_staging") and obj.source.get("orientation_regrasp_feasible"))
    return PlacementTarget(
        action_type=ActionType.REGRASP_FOR_ORIENTATION if regrasp else ActionType.EXTRACT_TO_STAGING,
        target_region_id="house_orientation_staging",
        task_role=role,
        place_pose={"frame_id": "base_link", "position_m": [x, y, obj.center_xyz_m[2]], "yaw_deg": obj.yaw_deg},
        task_progress_gain=0.0,
        expected_effects=("stage_for_orientation_reobservation", "do_not_assume_airborne_flip"),
        precheck_results={
            "transport_safe": True, "place_descent_safe": True,
            "release_safe": True, "return_safe": True, "protected_safe": True,
        },
    )


def _role_pose(
    scene: ClutterSceneState,
    state: HouseTaskState,
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
) -> dict[str, Any]:
    house = config.section("house")
    origin_x, origin_y, origin_z = [float(value) for value in house["origin_center_base_m"]]
    half_spacing = 0.5 * float(house["left_right_spacing_m"])
    x = origin_x - half_spacing if role.startswith("left_") else origin_x + half_spacing if role.startswith("right_") else origin_x
    z = origin_z
    if role.endswith("_upper"):
        lower_role = role.replace("upper", "lower")
        lower = scene.object_by_track(state.role_bindings.get(lower_role, ""))
        if lower is None:
            raise ValueError(f"missing verified lower support for {role}")
        x, origin_y = lower.center_xyz_m[:2]
        z = lower.center_xyz_m[2] + 0.5 * (lower.size_xyz_m[2] + obj.size_xyz_m[2])
    elif role == "roof":
        supports = [scene.object_by_track(state.role_bindings.get(name, "")) for name in ("left_support_upper", "right_support_upper")]
        if any(item is None for item in supports):
            raise ValueError("roof requires both verified upper supports")
        x = 0.5 * (supports[0].center_xyz_m[0] + supports[1].center_xyz_m[0])
        origin_y = 0.5 * (supports[0].center_xyz_m[1] + supports[1].center_xyz_m[1])
        z = max(item.center_xyz_m[2] + 0.5 * item.size_xyz_m[2] for item in supports) + 0.5 * obj.size_xyz_m[2]
    elif role == "triangle_top":
        roof = scene.object_by_track(state.role_bindings.get("roof", ""))
        if roof is None:
            raise ValueError("triangle_top requires a verified roof")
        x, origin_y = roof.center_xyz_m[:2]
        z = roof.center_xyz_m[2] + 0.5 * (roof.size_xyz_m[2] + obj.size_xyz_m[2])
    return {"frame_id": "base_link", "position_m": [x, origin_y, z], "yaw_deg": 0.0}
