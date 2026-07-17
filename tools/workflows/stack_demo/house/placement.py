"""Code-owned target poses and orientation gates for each house role."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

from ..clutter.edge_generation import PlacementTarget
from ..clutter.grasp_edges import (
    GraspInterval,
    evaluate_gripper_pose_clearance,
    normalize_gripper_yaw_deg,
)
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
    *,
    interval: GraspInterval | None = None,
) -> PlacementTarget | tuple[PlacementTarget, ...] | None:
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
            return staging_orientation_targets(
                scene, obj, role, config, interval=interval,
            ) or None
        return None
    aligned = _edge_aligned_grasp(scene, obj, interval, config)
    if aligned is None:
        staging_targets = staging_orientation_targets(
            scene, obj, role, config, interval=interval,
        )
        if not staging_targets:
            return None
        return tuple(
            replace(
                staging,
                target_region_id="house_edge_alignment_staging",
                expected_effects=(
                    "stage_for_edge_aligned_regrasp",
                    "require_fresh_orientation_observation",
                ),
                additional_physical_parameters={
                    **dict(staging.additional_physical_parameters or {}),
                    "edge_aligned_final_grasp_required": True,
                },
            )
            for staging in staging_targets
        )
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
        additional_physical_parameters={
            "required_grasp_yaw_deg": aligned["yaw_deg"],
            "required_grasp_checks": aligned["checks"],
            "edge_aligned_final_grasp_required": True,
            "edge_alignment_error_deg": aligned["edge_alignment_error_deg"],
        },
    )


def staging_orientation_target(
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
    *,
    scene: ClutterSceneState | None = None,
    interval: GraspInterval | None = None,
) -> PlacementTarget | None:
    """Compatibility helper for callers that can consume only one target."""
    targets = staging_orientation_targets(
        scene, obj, role, config, interval=interval,
    )
    return targets[0] if targets else None


def staging_orientation_targets(
    scene: ClutterSceneState | None,
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
    *,
    interval: GraspInterval | None = None,
) -> tuple[PlacementTarget, ...]:
    """Return every currently safe camera-visible staging alternative."""
    at_staging = bool(obj.source.get("at_staging"))
    transition = obj.source.get("orientation_transition")
    if at_staging:
        if not isinstance(transition, Mapping) or not bool(transition.get("geometry_verified")):
            return ()
        regrasp_pose = transition.get("regrasp_pose")
        next_pose = transition.get("staging_place_pose")
        expected = transition.get("expected_orientation_after")
        if not all(isinstance(value, Mapping) for value in (regrasp_pose, next_pose, expected)):
            return ()
        orientation_plan = _orientation_plan(obj, role, next_pose, expected, transition)
        return (PlacementTarget(
            action_type=ActionType.REGRASP_FOR_ORIENTATION,
            target_region_id="house_orientation_staging",
            task_role=role,
            place_pose=dict(next_pose),
            task_progress_gain=0.0,
            expected_effects=("verified_orientation_regrasp", "require_fresh_orientation_observation"),
            precheck_results=_staging_checks(),
            additional_physical_parameters={
                "grasp_pose": dict(regrasp_pose),
                "orientation_plan": orientation_plan,
            },
        ),)

    release_yaw = normalize_gripper_yaw_deg(
        interval.selected_yaw_deg if interval is not None else obj.yaw_deg
    )
    safe_poses = _safe_staging_poses(scene, obj, release_yaw, config)
    targets = []
    for candidate_index, (staging_pose, staging_checks, clearance) in enumerate(
        safe_poses, start=1,
    ):
        common_parameters = {
            "staging_candidate_id": f"house_staging_{candidate_index}",
            "staging_candidate_count": len(safe_poses),
            "staging_selection_mode": "live_scene_ranked_multi_point",
            "camera_reobservable": True,
            "minimum_obstacle_clearance_m": clearance,
        }
        if role not in {"roof", "triangle_top"}:
            targets.append(PlacementTarget(
                action_type=ActionType.EXTRACT_TO_STAGING,
                target_region_id="house_blocker_staging",
                task_role=role,
                place_pose=staging_pose,
                task_progress_gain=0.0,
                expected_effects=("remove_blocker_to_staging", "require_fresh_observation"),
                precheck_results=staging_checks,
                additional_physical_parameters=common_parameters,
            ))
            continue
        expected = _current_orientation_evidence(obj, role)
        orientation_plan = _orientation_plan(obj, role, staging_pose, expected, {})
        targets.append(PlacementTarget(
            action_type=ActionType.EXTRACT_TO_STAGING,
            target_region_id="house_orientation_staging",
            task_role=role,
            place_pose=staging_pose,
            task_progress_gain=0.0,
            expected_effects=(
                "stage_for_orientation_reobservation",
                "do_not_assume_airborne_flip",
                "require_fresh_observation",
            ),
            precheck_results=staging_checks,
            additional_physical_parameters={
                **common_parameters,
                "orientation_plan": orientation_plan,
            },
        ))
    return tuple(targets)


def _safe_staging_poses(
    scene: ClutterSceneState | None,
    obj: SceneObjectState,
    release_yaw_deg: float,
    config: StackDemoConfig,
) -> tuple[tuple[dict[str, Any], dict[str, Any], float], ...]:
    house = config.section("house")
    safety = config.section("safety")
    candidates = house["orientation_staging_candidates_base_m"]
    obstacles = tuple(
        item for item in (scene.current_objects if scene is not None else ())
        if item.track_id != obj.track_id
    )
    table_z = (
        min(item.center_xyz_m[2] - 0.5 * item.size_xyz_m[2] for item in scene.current_objects)
        if scene is not None and scene.current_objects
        else obj.center_xyz_m[2] - 0.5 * obj.size_xyz_m[2]
    )
    center_z = table_z + 0.5 * obj.size_xyz_m[2]
    output = []
    for configured_index, value in enumerate(candidates, start=1):
        x, y = float(value[0]), float(value[1])
        if math.hypot(x - obj.center_xyz_m[0], y - obj.center_xyz_m[1]) < float(
            safety["minimum_progress_translation_m"]
        ):
            continue
        pose = {
            "frame_id": "base_link",
            "position_m": [x, y, center_z],
            "yaw_deg": float(obj.yaw_deg),
        }
        moved = object_at_pose(obj, pose)
        footprint_safe = _staging_footprint_safe(moved, obstacles, config)
        if not footprint_safe:
            continue
        gripper_checks = dict(evaluate_gripper_pose_clearance(
            moved, (moved, *obstacles), release_yaw_deg, config,
        ))
        gripper_safe = all(bool(gripper_checks.get(key)) for key in (
            "opening_ok", "center_offset_ok", "contact_length_ok",
            "finger_safe", "palm_safe", "descent_safe", "lift_safe",
        ))
        if not gripper_safe:
            continue
        clearance = _minimum_staging_clearance(moved, obstacles)
        checks = _staging_checks({
            "staging_workspace_safe": True,
            "staging_robot_exclusion_safe": True,
            "staging_footprint_clear": True,
            "staging_open_gripper_descent_safe": True,
            "camera_reobservable": True,
            "configured_staging_candidate_index": configured_index,
            "staging_release_yaw_deg": release_yaw_deg,
            "staging_gripper_blocking_track_ids": list(
                gripper_checks.get("blocking_track_ids", ())
            ),
        })
        output.append((pose, checks, clearance, configured_index))
    output.sort(key=lambda item: (-item[2], item[3]))
    return tuple((pose, checks, round(clearance, 6)) for pose, checks, clearance, _ in output)


def _staging_footprint_safe(
    moved: SceneObjectState,
    obstacles: tuple[SceneObjectState, ...],
    config: StackDemoConfig,
) -> bool:
    clearance = float(config.section("safety")["object_clearance_m"])
    inset = float(config.section("safety")["workspace_inset_m"])
    half_x, half_y = _rotated_half_extents(moved)
    bounds = (
        moved.center_xyz_m[0] - half_x,
        moved.center_xyz_m[0] + half_x,
        moved.center_xyz_m[1] - half_y,
        moved.center_xyz_m[1] + half_y,
    )
    workspace = config.workspace
    if not (
        bounds[0] >= float(workspace["xmin"]) + inset
        and bounds[1] <= float(workspace["xmax"]) - inset
        and bounds[2] >= float(workspace["ymin"]) + inset
        and bounds[3] <= float(workspace["ymax"]) - inset
    ):
        return False
    camera_bounds = config.section("house")["orientation_staging_camera_bounds_base_m"]
    if not (
        bounds[0] >= float(camera_bounds["xmin"])
        and bounds[1] <= float(camera_bounds["xmax"])
        and bounds[2] >= float(camera_bounds["ymin"])
        and bounds[3] <= float(camera_bounds["ymax"])
    ):
        return False
    for exclusion in config.robot_exclusion_geometry:
        if _rectangles_overlap(bounds, (
            float(exclusion["xmin"]), float(exclusion["xmax"]),
            float(exclusion["ymin"]), float(exclusion["ymax"]),
        ), clearance):
            return False
    for other in obstacles:
        other_half_x, other_half_y = _rotated_half_extents(other)
        other_bounds = (
            other.center_xyz_m[0] - other_half_x,
            other.center_xyz_m[0] + other_half_x,
            other.center_xyz_m[1] - other_half_y,
            other.center_xyz_m[1] + other_half_y,
        )
        if _rectangles_overlap(bounds, other_bounds, clearance):
            return False
    return True


def _rotated_half_extents(obj: SceneObjectState) -> tuple[float, float]:
    angle = math.radians(float(obj.yaw_deg))
    half_x, half_y = 0.5 * obj.size_xyz_m[0], 0.5 * obj.size_xyz_m[1]
    return (
        abs(half_x * math.cos(angle)) + abs(half_y * math.sin(angle)),
        abs(half_x * math.sin(angle)) + abs(half_y * math.cos(angle)),
    )


def _rectangles_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    clearance: float,
) -> bool:
    return not (
        first[1] + clearance <= second[0]
        or second[1] + clearance <= first[0]
        or first[3] + clearance <= second[2]
        or second[3] + clearance <= first[2]
    )


def _minimum_staging_clearance(
    moved: SceneObjectState,
    obstacles: tuple[SceneObjectState, ...],
) -> float:
    half_x, half_y = _rotated_half_extents(moved)
    object_radius = math.hypot(half_x, half_y)
    clearances = []
    for other in obstacles:
        other_half_x, other_half_y = _rotated_half_extents(other)
        center_distance = math.hypot(
            moved.center_xyz_m[0] - other.center_xyz_m[0],
            moved.center_xyz_m[1] - other.center_xyz_m[1],
        )
        clearances.append(
            center_distance - object_radius - math.hypot(other_half_x, other_half_y)
        )
    return min(clearances, default=1.0)


def _orientation_plan(
    obj: SceneObjectState,
    role: str,
    staging_pose: Mapping[str, Any],
    expected_after: Mapping[str, Any],
    transition: Mapping[str, Any],
) -> dict[str, Any]:
    current = _current_orientation_evidence(obj, role)
    if role == "roof":
        return {
            "role": role,
            "current_groove_face_state": current.get("groove_face_state", "unknown"),
            "target_groove_face_state": transition.get("target_groove_face_state", "opening_down"),
            "current_long_axis_yaw_deg": current.get("long_axis_yaw_deg"),
            "staging_pose": dict(staging_pose),
            "regrasp_pose": transition.get("regrasp_pose"),
            "final_roof_orientation": transition.get("final_roof_orientation"),
            "face_change_geometry_verified": bool(transition.get("geometry_verified")),
            "expected_orientation_after": dict(expected_after),
        }
    return {
        "role": role,
        "current_apex_direction": current.get("apex_direction", "unknown"),
        "current_base_edge_direction": current.get("base_edge_direction", "unknown"),
        "current_face_state": current.get("face_state", "unknown"),
        "staging_pose": dict(staging_pose),
        "regrasp_pose": transition.get("regrasp_pose"),
        "final_triangle_yaw_deg": transition.get("final_triangle_yaw_deg"),
        "face_change_geometry_verified": bool(transition.get("geometry_verified")),
        "expected_orientation_after": dict(expected_after),
    }


def _current_orientation_evidence(obj: SceneObjectState, role: str) -> dict[str, Any]:
    keys = (
        ("groove_face_state", "face_up", "long_axis_yaw_deg")
        if role == "roof"
        else ("apex_direction", "base_edge_direction", "face_state", "target_yaw_deg")
    )
    return {key: obj.source[key] for key in keys if key in obj.source}


def _staging_checks(additional: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "transport_safe": True,
        "place_descent_safe": True,
        "release_safe": True,
        "return_safe": True,
        "protected_safe": True,
        **dict(additional or {}),
    }


def _role_pose(
    scene: ClutterSceneState,
    state: HouseTaskState,
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
) -> dict[str, Any]:
    house = config.section("house")
    origin_x, origin_y, origin_z = [float(value) for value in house["origin_center_base_m"]]
    inner_gap = float(house["support_inner_gap_m"])
    nominal_half_spacing = 0.5 * (float(obj.size_xyz_m[0]) + inner_gap)
    x = origin_x - nominal_half_spacing if role.startswith("left_") else origin_x + nominal_half_spacing if role.startswith("right_") else origin_x
    z = origin_z
    if role.endswith("_upper"):
        lower_role = role.replace("upper", "lower")
        lower = scene.object_by_track(state.role_bindings.get(lower_role, ""))
        if lower is None:
            raise ValueError(f"missing verified lower support for {role}")
        x, origin_y = lower.center_xyz_m[:2]
        z = lower.center_xyz_m[2] + 0.5 * (lower.size_xyz_m[2] + obj.size_xyz_m[2])
    elif role.endswith("_lower"):
        opposite_role = (
            "right_support_lower" if role.startswith("left_")
            else "left_support_lower"
        )
        opposite = scene.object_by_track(state.role_bindings.get(opposite_role, ""))
        if opposite is not None:
            center_spacing = (
                0.5 * (float(obj.size_xyz_m[0]) + float(opposite.size_xyz_m[0]))
                + inner_gap
            )
            x = opposite.center_xyz_m[0] + (
                -center_spacing if role.startswith("left_") else center_spacing
            )
            origin_y = opposite.center_xyz_m[1]
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


def _edge_aligned_grasp(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    interval: GraspInterval | None,
    config: StackDemoConfig,
) -> dict[str, Any] | None:
    """Select a collision-free grasp parallel to one measured object edge."""
    if interval is None:
        return None
    edge_axis = float(obj.yaw_deg)
    candidates = (
        normalize_gripper_yaw_deg(edge_axis),
        normalize_gripper_yaw_deg(edge_axis + 90.0),
    )
    feasible = []
    for yaw in candidates:
        checks = dict(evaluate_gripper_pose_clearance(
            obj, scene.current_objects, yaw, config,
        ))
        safe = all(bool(checks.get(key)) for key in (
            "opening_ok", "center_offset_ok", "contact_length_ok",
            "finger_safe", "palm_safe", "descent_safe", "lift_safe",
        ))
        if not safe:
            continue
        delta = abs((yaw - float(interval.selected_yaw_deg) + 90.0) % 180.0 - 90.0)
        feasible.append((delta, yaw, checks))
    if not feasible:
        return None
    delta, yaw, checks = min(feasible, key=lambda item: (item[0], abs(item[1])))
    edge_error = min(
        abs((yaw - axis + 90.0) % 180.0 - 90.0)
        for axis in candidates
    )
    return {
        "yaw_deg": yaw,
        "checks": checks,
        "edge_alignment_error_deg": edge_error,
        "selected_interval_delta_deg": delta,
    }
