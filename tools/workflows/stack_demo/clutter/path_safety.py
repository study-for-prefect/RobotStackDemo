"""Code-owned placement, transport, and push swept-volume prechecks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from robot_scene_pipeline.geometry_relations import object_xy_aabb
from robot_scene_pipeline.tool_swept_volume import check_tool_swept_volume
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
    clearance = evaluate_gripper_pose_clearance(
        moved, [moved, *obstacles],
        float(place_pose.get("yaw_deg", obj.yaw_deg)), config,
    )
    transport_safe = _held_transport_path_safe(
        obj, physical.get("transport_path", ()), obstacles, config,
    )
    place_clear = bool(clearance.get("finger_safe") and clearance.get("palm_safe"))
    return {
        "transport_safe": transport_safe,
        "place_descent_safe": place_clear,
        "release_safe": place_clear,
        "return_safe": place_clear,
        "place_finger_safe": bool(clearance.get("finger_safe")),
        "place_palm_safe": bool(clearance.get("palm_safe")),
    }


def build_nudge_parameters(
    blocker: SceneObjectState,
    direction: Sequence[float],
    distance: float,
    wrist_yaw: float,
    config: StackDemoConfig,
) -> dict[str, Any]:
    targets = build_push_targets(_push_plan(blocker, direction, distance, wrist_yaw, config))
    return {
        "push_direction_base": [float(value) for value in direction[:3]],
        "push_distance_m": float(distance),
        "push_wrist_yaw_deg": float(wrist_yaw),
        "push_start": {"frame_id": "base_link", "position_m": list(targets["contact"])},
        "push_end": {"frame_id": "base_link", "position_m": list(targets["push_end"])},
        "prepush_pose": {"frame_id": "base_link", "position_m": list(targets["pre_push"])},
        "contact_side": _opposite_side(direction),
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
    report = check_tool_swept_volume(
        plan,
        [_geometry_object_dict(item) for item in scene.current_objects],
        ignore_object_ids=(blocker.track_id,),
        protected_object_ids=scene.protected_tracks,
        gripper_outer_width_m=float(gripper["open_outer_width_m"]),
        finger_length_m=float(gripper["finger_length_m"]),
        tool_depth_m=float(gripper["palm_depth_m"]),
        fingertip_thickness_m=float(gripper["fingertip_thickness_m"]),
        safety_margin_m=float(config.section("safety")["object_clearance_m"]),
        tcp_offset_tool_m=gripper["tcp_offset_tool_m"],
        controlled_contact={"enabled": True, "allow_loose_chain_contact": True},
    )
    hard = list(report.get("hard_collisions") or [])
    stages = {str(item.get("stage")) for item in hard}
    return {
        "passed": bool(report.get("feasible")),
        "geometry_checks_passed": bool(report.get("feasible")),
        "prepush_reachable": "pre_push_pose" not in stages,
        "prepush_descent_safe": not stages.intersection({"approach_to_contact", "contact_pose"}),
        "horizontal_sweep_safe": "horizontal_push" not in stages,
        "push_end_safe": "retreat" not in stages,
        "protected_safe": not any(item.get("entity_type") == "protected_structure" for item in hard),
        "loose_contact_only_horizontal": all(
            item.get("stage") == "horizontal_push" for item in report.get("controlled_contacts", [])
        ),
        "swept_volume_reason": report.get("reason"),
    }


def _held_transport_path_safe(
    held: SceneObjectState,
    path: Sequence[Mapping[str, Any]],
    obstacles: Sequence[SceneObjectState],
    config: StackDemoConfig,
) -> bool:
    positions = [pose.get("position_m") for pose in path if isinstance(pose, Mapping)]
    positions = [item for item in positions if isinstance(item, (list, tuple)) and len(item) >= 3]
    if len(positions) < 2:
        return False
    margin = float(config.section("safety")["object_clearance_m"])
    half = [0.5 * value + margin for value in held.size_xyz_m]
    for start, end in zip(positions, positions[1:]):
        swept = {
            "xmin": min(float(start[0]), float(end[0])) - half[0],
            "xmax": max(float(start[0]), float(end[0])) + half[0],
            "ymin": min(float(start[1]), float(end[1])) - half[1],
            "ymax": max(float(start[1]), float(end[1])) + half[1],
            "zmin": min(float(start[2]), float(end[2])) - half[2],
            "zmax": max(float(start[2]), float(end[2])) + half[2],
        }
        for obstacle in obstacles:
            bounds = object_xy_aabb(_geometry_object_dict(obstacle))
            if bounds and _aabb_overlap_3d(swept, bounds):
                return False
    return True


def _push_plan(
    blocker: SceneObjectState,
    direction: Sequence[float],
    distance: float,
    wrist_yaw: float,
    config: StackDemoConfig,
) -> dict[str, Any]:
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
        "contact_z_offset_m": float(config.section("clearing")["push_contact_z_offset_m"]),
        "target_yaw_deg": float(wrist_yaw),
    }


def _geometry_object_dict(obj: SceneObjectState) -> dict[str, Any]:
    return {
        "id": obj.track_id,
        "track_id": obj.track_id,
        "label": obj.class_name,
        "geometry_center_m": list(obj.center_xyz_m),
        "dimensions_m": list(obj.size_xyz_m),
        "table_yaw_deg": obj.yaw_deg,
    }


def _aabb_overlap_3d(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    return all(
        min(float(first[high]), float(second[high])) > max(float(first[low]), float(second[low]))
        for low, high in (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))
    )


def _opposite_side(direction: Sequence[float]) -> str:
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "-x" if float(direction[0]) > 0 else "+x"
    return "-y" if float(direction[1]) > 0 else "+y"
