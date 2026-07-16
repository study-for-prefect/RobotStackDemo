"""Code-owned placement, transport, and push swept-volume prechecks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from robot_scene_pipeline.pushed_object_sweep import analyze_push_contact_chain
from robot_scene_pipeline.tool_swept_volume import check_tool_swept_volume, check_transport_swept_volume
from tools.robot.push_primitives import build_push_targets

from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .grasp_edges import evaluate_gripper_pose_clearance


def placement_path_checks(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    place_pose: Mapping[str, Any],
    physical: Mapping[str, Any],
    config: StackDemoConfig,
) -> dict[str, Any]:
    position = place_pose.get("position_m")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return {
            "transport_safe": False, "place_descent_safe": False,
            "release_safe": False, "return_safe": False,
        }
    moved = SceneObjectState(
        **{
            **{name: getattr(obj, name) for name in obj.__dataclass_fields__},
            "center_xyz_m": tuple(float(value) for value in position[:3]),
            "yaw_deg": float(place_pose.get("yaw_deg", obj.yaw_deg)),
        }
    )
    obstacles = [other for other in scene.current_objects if other.track_id != obj.track_id]
    release_pose = physical.get("release_pose") or {}
    release_yaw = float(release_pose.get("yaw_deg", physical.get("release_gripper_yaw_deg", obj.yaw_deg)))
    clearance = evaluate_gripper_pose_clearance(
        moved, [moved, *obstacles],
        release_yaw, config,
    )
    gripper = config.section("gripper")
    transport = check_transport_swept_volume(
        physical.get("transport_path", ()),
        [_geometry_object_dict(item) for item in obstacles],
        held_object_id=obj.track_id,
        held_size_m=obj.size_xyz_m,
        gripper_outer_width_m=float(gripper["open_outer_width_m"]),
        fingertip_width_m=float(gripper["fingertip_width_m"]),
        upper_finger_width_m=float(gripper["upper_finger_width_m"]),
        upper_finger_height_m=float(gripper["upper_finger_height_m"]),
        palm_width_m=float(gripper["palm_width_m"]),
        palm_depth_m=float(gripper["palm_depth_m"]),
        palm_height_m=float(gripper["palm_height_m"]),
        tcp_offset_tool_m=gripper["tcp_offset_tool_m"],
        safety_margin_m=float(config.section("safety")["object_clearance_m"]),
    )
    place_clear = bool(clearance.get("finger_safe") and clearance.get("palm_safe"))
    return {
        "transport_safe": bool(transport.get("feasible")),
        "place_descent_safe": place_clear,
        "release_safe": place_clear,
        "return_safe": place_clear,
        "place_finger_safe": bool(clearance.get("finger_safe")),
        "place_palm_safe": bool(clearance.get("palm_safe")),
        "place_blocking_track_ids": list(clearance.get("blocking_track_ids", [])),
        "release_gripper_yaw_deg": release_yaw,
        "transport_checked_components": transport.get("checked_components", []),
        "transport_blocking_track_ids": sorted({str(item.get("id")) for item in transport.get("collisions", [])}),
        "transport_swept_volume_reason": transport.get("reason"),
    }


def build_nudge_parameters(
    blocker: SceneObjectState,
    direction: Sequence[float],
    distance: float,
    wrist_yaw: float,
    config: StackDemoConfig,
) -> dict[str, Any]:
    plan = _push_plan(blocker, direction, distance, wrist_yaw, config)
    targets = build_push_targets(plan)
    return {
        "push_direction": _direction_label(direction),
        "push_direction_base": [float(value) for value in direction[:3]],
        "push_distance_m": float(distance),
        "push_wrist_yaw_deg": float(wrist_yaw),
        "push_start": {"frame_id": "base_link", "position_m": list(targets["contact"])},
        "push_end": {"frame_id": "base_link", "position_m": list(targets["push_end"])},
        "prepush_pose": {"frame_id": "base_link", "position_m": list(targets["pre_push"])},
        "precontact_pose": {"frame_id": "base_link", "position_m": list(targets["pre_contact"])},
        "contact_side": _opposite_side(direction),
        "contact_standoff_m": float(plan["contact_standoff_m"]),
        "contact_clearance_m": float(plan["contact_clearance_m"]),
        "prepush_clearance_m": float(plan["prepush_clearance_m"]),
        "object_contact_extent_m": float(plan["object_contact_extent_m"]),
        "tool_contact_extent_m": float(plan["tool_contact_extent_m"]),
    }


def nudge_sweep_checks(
    scene: ClutterSceneState,
    blocker: SceneObjectState,
    physical: Mapping[str, Any],
    config: StackDemoConfig,
) -> dict[str, Any]:
    plan = _push_plan(
        blocker,
        physical["push_direction_base"],
        float(physical["push_distance_m"]),
        float(physical["push_wrist_yaw_deg"]),
        config,
    )
    plan["workspace_bounds"] = dict(config.workspace)
    gripper = config.section("gripper")
    geometry_objects = [_geometry_object_dict(item) for item in scene.current_objects]
    entry_report = _check_push_tool_sweep(
        plan,
        geometry_objects,
        blocker.track_id,
        scene.protected_tracks,
        gripper,
        config,
        analyze_chain=False,
    )
    entry_hard = list(entry_report.get("hard_collisions") or [])
    contact_side_collisions = [
        item for item in entry_hard
        if item.get("collision_source") == "gf225_tool"
        and item.get("id") is not None
        and item.get("stage") in {
            "pre_push_pose", "descend_to_pre_contact", "pre_contact_pose",
            "approach_to_contact", "contact_pose",
        }
    ]
    if contact_side_collisions:
        report = entry_report
        hard = contact_side_collisions
        chain = {
            "chain_track_ids": [blocker.track_id],
            "chain_object_count": 1,
            "secondary_contact_expected": False,
            "estimated_displacements_m": {blocker.track_id: float(physical["push_distance_m"])},
            "estimation_method": "not_run_contact_side_blocked",
            "controlled_contacts": [],
            "hard_collisions": [],
        }
    else:
        chain = analyze_push_contact_chain(
            plan,
            geometry_objects,
            (blocker.track_id,),
            scene.protected_tracks,
            float(config.section("safety")["object_clearance_m"]),
        )
        report = _check_push_tool_sweep(
            plan,
            geometry_objects,
            blocker.track_id,
            scene.protected_tracks,
            gripper,
            config,
            analyze_chain=True,
        )
        hard = list(report.get("hard_collisions") or [])
    stages = {str(item.get("stage")) for item in hard}
    workspace_safe = _push_motion_inside_workspace(plan, config.workspace)
    rejection_reasons = _nudge_rejection_reasons(
        hard,
        workspace_safe,
        bool(contact_side_collisions),
    )
    blocking_track_ids = sorted({
        str(item.get("id"))
        for item in contact_side_collisions
        if item.get("id") is not None
    })
    return {
        "passed": bool(report.get("feasible")) and workspace_safe,
        "geometry_checks_passed": bool(report.get("feasible")) and workspace_safe,
        "prepush_reachable": "pre_push_pose" not in stages,
        "prepush_descent_safe": not stages.intersection({
            "descend_to_pre_contact", "pre_contact_pose", "approach_to_contact", "contact_pose",
        }),
        "contact_side_clear": not contact_side_collisions,
        "contact_side": physical.get("contact_side"),
        "blocking_track_ids": blocking_track_ids,
        "horizontal_sweep_safe": "horizontal_push" not in stages,
        "push_end_safe": "retreat" not in stages,
        "protected_safe": not any(
            item.get("entity_type") in {"protected_completed_object", "protected_structure"}
            for item in hard
        ),
        "ordinary_objects_safe": not any(
            item.get("entity_type") == "ordinary_object" for item in hard
        ),
        "gripper_swept_volume_safe": not any(
            item.get("collision_source") == "gf225_tool" for item in hard
        ),
        "pushed_object_swept_volume_safe": not any(
            item.get("collision_source") == "pushed_object" for item in hard
        ),
        "workspace_safe": workspace_safe,
        "loose_contact_only_horizontal": all(
            item.get("stage") == "horizontal_push" for item in report.get("controlled_contacts", [])
        ),
        "swept_volume_reason": report.get("reason"),
        "rejection_reasons": rejection_reasons,
        "collision_details": hard,
        "chain_track_ids": list(chain["chain_track_ids"]),
        "chain_object_count": int(chain["chain_object_count"]),
        "secondary_contact_expected": bool(chain["secondary_contact_expected"]),
        "estimated_displacements_m": dict(chain["estimated_displacements_m"]),
        "estimated_displacements_are_upper_bounds": True,
        "push_chain_estimation_method": chain["estimation_method"],
        "controlled_secondary_contacts": list(chain["controlled_contacts"]),
        "push_chain_failure_details": list(chain["hard_collisions"]),
    }


def _check_push_tool_sweep(
    plan: Mapping[str, Any],
    geometry_objects: Sequence[Mapping[str, Any]],
    blocker_track_id: str,
    protected_tracks: Sequence[str],
    gripper: Mapping[str, Any],
    config: StackDemoConfig,
    *,
    analyze_chain: bool,
) -> Mapping[str, Any]:
    return check_tool_swept_volume(
        dict(plan),
        geometry_objects,
        ignore_object_ids=(blocker_track_id,),
        protected_object_ids=protected_tracks,
        gripper_outer_width_m=float(gripper["open_outer_width_m"]),
        finger_length_m=float(gripper["finger_length_m"]),
        tool_depth_m=float(gripper["palm_depth_m"]),
        fingertip_thickness_m=float(gripper["fingertip_thickness_m"]),
        safety_margin_m=float(config.section("safety")["object_clearance_m"]),
        tcp_offset_tool_m=gripper["tcp_offset_tool_m"],
        controlled_contact={"enabled": True, "allow_loose_chain_contact": True},
        analyze_push_chain_contacts=analyze_chain,
    )


def _push_plan(
    blocker: SceneObjectState,
    direction: Sequence[float],
    distance: float,
    wrist_yaw: float,
    config: StackDemoConfig,
) -> dict[str, Any]:
    contact_geometry = _push_contact_geometry(blocker, direction, wrist_yaw, config)
    return {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "obstacle": {
            "id": blocker.track_id,
            "geometry_center_m": list(blocker.center_xyz_m),
            "dimensions_m": list(blocker.size_xyz_m),
        },
        "direction_base": [float(value) for value in direction[:3]],
        "distance_m": float(distance),
        "lift_m": float(config.section("safety")["approach_height_m"]),
        "retreat_lift_m": float(config.section("safety")["observation_height_m"]),
        "contact_z_offset_m": float(config.section("clearing")["push_contact_z_offset_m"]),
        **contact_geometry,
        "target_yaw_deg": float(wrist_yaw),
        "robot_exclusion_geometry": [
            dict(item) for item in config.robot_exclusion_geometry
        ],
    }


def _push_contact_geometry(
    blocker: SceneObjectState,
    direction: Sequence[float],
    wrist_yaw: float,
    config: StackDemoConfig,
) -> dict[str, float]:
    """Compute a collision-aware center-to-TCP standoff for the fingertip."""
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        raise ValueError("push direction XY norm must be non-zero")
    dx, dy = dx / norm, dy / norm

    object_yaw = math.radians(float(blocker.yaw_deg))
    object_u = (math.cos(object_yaw), math.sin(object_yaw))
    object_v = (-object_u[1], object_u[0])
    object_extent = (
        0.5 * float(blocker.size_xyz_m[0]) * abs(dx * object_u[0] + dy * object_u[1])
        + 0.5 * float(blocker.size_xyz_m[1]) * abs(dx * object_v[0] + dy * object_v[1])
    )

    tool_yaw = math.radians(float(wrist_yaw))
    tool_u = (math.cos(tool_yaw), math.sin(tool_yaw))
    tool_v = (-tool_u[1], tool_u[0])
    gripper = config.section("gripper")
    tool_extent = (
        0.5 * float(gripper["palm_depth_m"]) * abs(dx * tool_u[0] + dy * tool_u[1])
        + 0.5 * float(gripper["fingertip_width_m"]) * abs(dx * tool_v[0] + dy * tool_v[1])
    )
    contact_clearance = float(config.section("safety")["object_clearance_m"])
    prepush_clearance = float(config.section("clearing")["prepush_clearance_m"])
    return {
        "object_contact_extent_m": object_extent,
        "tool_contact_extent_m": tool_extent,
        "contact_clearance_m": contact_clearance,
        "contact_standoff_m": object_extent + tool_extent + contact_clearance,
        "prepush_clearance_m": prepush_clearance,
    }


def _geometry_object_dict(obj: SceneObjectState) -> dict[str, Any]:
    return {
        "id": obj.track_id,
        "track_id": obj.track_id,
        "label": obj.class_name,
        "geometry_center_m": list(obj.center_xyz_m),
        "dimensions_m": list(obj.size_xyz_m),
        "table_yaw_deg": obj.yaw_deg,
        "is_completed": obj.already_completed,
        "is_fixed": bool(obj.source.get("is_fixed", False)),
        "movable": obj.source.get("movable", True),
        "role": obj.source.get("role"),
        "protection_kind": (
            "completed_object" if obj.already_completed
            else "protected_structure" if obj.protected
            else None
        ),
    }


def _push_motion_inside_workspace(
    plan: Mapping[str, Any],
    workspace: Mapping[str, float],
) -> bool:
    obstacle = plan.get("obstacle") or {}
    center = obstacle.get("geometry_center_m")
    size = obstacle.get("dimensions_m")
    direction = plan.get("direction_base")
    if not all(isinstance(value, (list, tuple)) for value in (center, size, direction)):
        return False
    end = [
        float(center[0]) + float(direction[0]) * float(plan["distance_m"]),
        float(center[1]) + float(direction[1]) * float(plan["distance_m"]),
    ]
    centers = ([float(center[0]), float(center[1])], end)
    object_inside = all(
        float(workspace["xmin"]) <= point[0] - 0.5 * float(size[0])
        and point[0] + 0.5 * float(size[0]) <= float(workspace["xmax"])
        and float(workspace["ymin"]) <= point[1] - 0.5 * float(size[1])
        and point[1] + 0.5 * float(size[1]) <= float(workspace["ymax"])
        for point in centers
    )
    targets = build_push_targets(dict(plan))
    tool_inside = all(
        float(workspace["xmin"]) <= point[0] <= float(workspace["xmax"])
        and float(workspace["ymin"]) <= point[1] <= float(workspace["ymax"])
        for point in targets.values()
    )
    return object_inside and tool_inside


def _nudge_rejection_reasons(
    collisions: Sequence[Mapping[str, Any]],
    workspace_safe: bool,
    contact_side_blocked: bool,
) -> list[str]:
    reasons: list[str] = []
    if contact_side_blocked:
        reasons.append("pre_push_contact_side_blocked")
    if any(item.get("entity_type") == "protected_completed_object" for item in collisions):
        reasons.append("protected_completed_object_collision")
    if any(
        item.get("entity_type") in {
            "fixed_obstacle", "protected_structure", "table", "robot_exclusion_geometry",
        }
        for item in collisions
    ):
        reasons.append("fixed_obstacle_collision")
    if any(item.get("reason") == "push_chain_workspace_violation" for item in collisions):
        reasons.append("push_chain_workspace_violation")
    elif not workspace_safe:
        reasons.append("workspace_violation")
    if any(
        item.get("collision_source") == "gf225_tool"
        and item.get("entity_type") == "ordinary_object"
        and item.get("stage") not in {"pre_push_pose", "approach_to_contact", "contact_pose"}
        for item in collisions
    ):
        reasons.append("gripper_swept_volume_collision")
    return list(dict.fromkeys(reasons))


def _opposite_side(direction: Sequence[float]) -> str:
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "-x" if float(direction[0]) > 0 else "+x"
    return "-y" if float(direction[1]) > 0 else "+y"


def _direction_label(direction: Sequence[float]) -> str:
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "+x" if float(direction[0]) > 0 else "-x"
    return "+y" if float(direction[1]) > 0 else "-y"
