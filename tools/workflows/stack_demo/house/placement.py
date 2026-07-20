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
from .orientation_trajectory import (
    FULL_3D_ORIENTATION_SHAPES,
    beam_axis_from_supports,
    build_airborne_orientation_plan,
    build_triangle_tabletop_step_pose,
    orientation_airborne_blocking_tracks,
    solve_target_object_orientations,
    semantic_face_ready_for_structural_tilt,
    special_shape_evidence_ready,
    triangle_tabletop_evidence_ready,
)


def house_placement_target(
    scene: ClutterSceneState,
    state: HouseTaskState,
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
    *,
    interval: GraspInterval | None = None,
) -> PlacementTarget | tuple[PlacementTarget, ...] | None:
    if role == "triangle_top" and obj.shape == "triangle":
        obj = _with_verified_triangle_tabletop_pose(scene, obj, config)
    pose = _role_pose(scene, state, obj, role, config)
    if obj.shape in FULL_3D_ORIENTATION_SHAPES and role in {"roof", "triangle_top"}:
        if special_shape_evidence_ready(obj, config):
            face_ready, face_checks = semantic_face_ready_for_structural_tilt(obj, config)
            if not face_ready:
                staged = staging_orientation_targets(
                    scene, obj, role, config, interval=interval,
                )
                return tuple(
                    replace(
                        target,
                        expected_effects=(
                            "tabletop_face_reorientation_before_final_place",
                            "require_fresh_orientation_observation",
                        ),
                        additional_physical_parameters={
                            **dict(target.additional_physical_parameters or {}),
                            **face_checks,
                            "staging_purpose": "tabletop_face_reorientation",
                            "failed_direct_orientation_reason": (
                                "semantic_face_requires_tabletop_reorientation"
                            ),
                            "airborne_face_change_forbidden": True,
                        },
                    )
                    for target in staged
                ) or None
        direct = _special_shape_targets(scene, state, obj, role, pose, config, interval)
        if direct:
            return direct
        if interval is not None and special_shape_evidence_ready(obj, config):
            blockers = orientation_airborne_blocking_tracks(scene, obj, config)
            if blockers:
                # Emit a non-executable diagnostic edge.  Shared edge
                # generation extracts these concrete blockers and offers
                # pick-away actions instead of shuttling the roof again.
                return (PlacementTarget(
                    action_type=ActionType.PLACE_HOUSE_ROLE,
                    target_region_id=None,
                    task_role=role,
                    place_pose=pose,
                    task_progress_gain=1.0,
                    expected_effects=("clear_orientation_space_before_full_3d_place",),
                    precheck_results={
                        "orientation_airborne_space_safe": False,
                        "orientation_space_blocking_track_ids": list(blockers),
                    },
                    additional_physical_parameters={
                        "required_grasp_yaw_deg": interval.selected_yaw_deg,
                        "required_grasp_checks": dict(interval.selected_check),
                        "failed_direct_orientation_reason": "orientation_airborne_space_blocked",
                    },
                ),)
        return staging_orientation_targets(scene, obj, role, config, interval=interval) or None
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
    aligned_grasps = _edge_aligned_grasps(scene, obj, interval, config)
    if not aligned_grasps:
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
    return tuple(
        PlacementTarget(
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
                "edge_aligned_grasp_option_index": option_index,
                "edge_aligned_grasp_option_count": len(aligned_grasps),
            },
        )
        for option_index, aligned in enumerate(aligned_grasps, start=1)
    )


def _with_verified_triangle_tabletop_pose(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    config: StackDemoConfig,
) -> SceneObjectState:
    """Recognize a stable apex-up 2:1:1 triangle from fresh metric geometry.

    A right triangular prism resting on its hypotenuse has that measured long
    edge parallel to the table and its right-angle altitude pointing upward.
    This fresh support fact is more reliable than choosing an arbitrary
    three-corner subset from the four-corner projected colour hull.
    """
    source = obj.source
    long_axis = source.get("long_axis_base")
    visible_normal = source.get("visible_face_normal_base")
    support = source.get("local_support_surface")
    support_z = support.get("support_z_base_m") if isinstance(support, Mapping) else None
    observed_bottom_z = obj.center_xyz_m[2] - 0.5 * obj.size_xyz_m[2]
    minimum = float(config.section("house_orientation")[
        "triangle_tabletop_minimum_orientation_confidence"
    ])
    maximum_long_axis_vertical = float(config.section("house_orientation").get(
        "triangle_tabletop_hypotenuse_max_vertical_component", 0.20,
    ))
    if not (
        source.get("metric_triangle_geometry_confirmed") is True
        and isinstance(long_axis, (list, tuple)) and len(long_axis) == 3
        and abs(float(long_axis[2])) <= maximum_long_axis_vertical
        and isinstance(visible_normal, (list, tuple)) and len(visible_normal) == 3
        and float(visible_normal[2]) >= float(config.section("house_orientation")[
            "triangle_minimum_upward_component"
        ])
        and support_z is not None
        and abs(observed_bottom_z - float(support_z))
        <= float(config.section("house")["support_height_tolerance_m"])
        and float(source.get("orientation_confidence", obj.orientation_confidence)) >= minimum
    ):
        return obj
    return replace(obj, source={
        **dict(source),
        "semantic_shape": "triangle",
        "semantic_shape_uncertain": False,
        "designated_right_angle_edge_base": [0.0, 0.0, 1.0],
        "tabletop_apex_up_geometry_verified": True,
        "triangle_right_angle_direction_source": (
            "fresh_metric_2_to_1_to_1_hypotenuse_support_geometry"
        ),
    })


def _special_shape_targets(
    scene: ClutterSceneState,
    state: HouseTaskState,
    obj: SceneObjectState,
    role: str,
    pose: Mapping[str, Any],
    config: StackDemoConfig,
    interval: GraspInterval | None,
    *,
    allow_staged_triangle_face_change: bool = False,
) -> tuple[PlacementTarget, ...]:
    """Create only real quaternion/fixed-TCP final-placement candidates."""
    if interval is None or not special_shape_evidence_ready(obj, config):
        return ()
    verified_roof_track = (
        state.role_bindings.get("roof")
        if state.role_completion.get("roof") is True
        else None
    )
    beam_axis = beam_axis_from_supports(
        scene, state.role_bindings, config,
        verified_roof_track_id=verified_roof_track,
    )
    if beam_axis is None:
        return ()
    face_ready, face_checks = semantic_face_ready_for_structural_tilt(obj, config)
    if not face_ready and not allow_staged_triangle_face_change:
        return ()
    targets = []
    for target_object_pose in solve_target_object_orientations(
        obj, pose["position_m"], beam_axis, config,
        allow_staged_triangle_face_change=allow_staged_triangle_face_change,
    ):
        orientation_config = config.section("house_orientation")
        is_tilted = float(target_object_pose["target_tilt_deg"]) > float(
            orientation_config["tilted_place_clearance_min_tilt_deg"]
        )
        staged_face_change = bool(allow_staged_triangle_face_change and not face_ready)
        tilted_clearance = (
            float(orientation_config["tilted_place_clearance_m"])
            if is_tilted or staged_face_change else 0.0
        )
        target_object_pose = {
            **target_object_pose,
            "tilted_place_clearance_m": tilted_clearance,
        }
        trajectory = build_airborne_orientation_plan(
            scene, obj, interval.selected_yaw_deg, target_object_pose, config,
        )
        if trajectory is None:
            continue
        targets.append(PlacementTarget(
            action_type=ActionType.PLACE_HOUSE_ROLE,
            target_region_id=None,
            task_role=role,
            place_pose=target_object_pose,
            task_progress_gain=1.0,
            expected_effects=(f"complete_house_role:{role}", "protect_completed_structure"),
            precheck_results={
                "transport_safe": True, "place_descent_safe": True,
                "release_safe": True, "return_safe": True, "protected_safe": True,
                "semantic_shape_certain": True, "measured_beam_axis_available": True,
                "beam_axis_from_current_frame_geometry": True,
                "fixed_tcp_orientation_sweep_safe": True,
                "semantic_face_evidence_available": True,
                **(
                    {"staged_triangle_face_change_authorized": True}
                    if staged_face_change
                    else {"semantic_face_ready_before_pick": True}
                ),
            },
            additional_physical_parameters={
                "orientation_trajectory": trajectory,
                "observed_object_pose": {
                    "frame_id": "base_link", "position_m": list(obj.center_xyz_m),
                    "orientation_xyzw": list(obj.source["orientation_xyzw"]),
                },
                "target_object_pose": target_object_pose,
                "grasp_tcp_object_transform": trajectory["grasp_tcp_object_transform"],
                "release_grasp_tcp_pose": trajectory["release_grasp_tcp_pose"],
                "airborne_adjustment_candidates": [trajectory["airborne_candidate_checks"]],
                "orientation_candidates": [target_object_pose],
                "orientation_sweep_checks": trajectory["orientation_sweep_checks"],
                "measured_beam_axis_base": list(beam_axis),
                "measured_beam_axis_source": (
                    "visible_upper_support_centers"
                    if _scene_object(scene, state.role_bindings.get("left_support_upper", "")) is not None
                    and _scene_object(scene, state.role_bindings.get("right_support_upper", "")) is not None
                    else "current_frame_verified_60_to_67mm_roof_long_axis"
                ),
                "required_3d_rotation_deg": trajectory["actual_rotation_angle_deg"],
                "held_object_rotation_radius_m": trajectory["held_object_rotation_radius_m"],
                "required_grasp_yaw_deg": interval.selected_yaw_deg,
                "required_grasp_checks": dict(interval.selected_check),
                **face_checks,
                "airborne_adjustment_kind": "minimum_required_target_orientation_alignment",
                "tilted_place_clearance_m": tilted_clearance,
                "tilted_place_clearance_applied": bool(is_tilted or staged_face_change),
            },
        ))
    return tuple(targets)


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
    if (
        role == "triangle_top"
        and scene is not None
        and interval is not None
        and obj.shape == "triangle"
        and triangle_tabletop_evidence_ready(obj, config)
    ):
        ready, _ = semantic_face_ready_for_structural_tilt(obj, config)
        if not ready:
            incremental = _triangle_tabletop_step_targets(
                scene, obj, role, config, interval,
            )
            if incremental:
                return incremental
    relevant_results = tuple(
        result for result in (scene.recent_action_results if scene is not None else ())
        if result.get("acted_object_track_id") == obj.track_id
        and result.get("task_role") == role
    )
    latest_failed_final = max(
        (
            index for index, result in enumerate(relevant_results)
            if result.get("action_type") == ActionType.PLACE_HOUSE_ROLE.value
            and result.get("success") is False
        ),
        default=-1,
    )
    successful_staging_after_final_failure = any(
        index > latest_failed_final
        and result.get("success") is True
        and result.get("action_type") in {
            ActionType.EXTRACT_TO_STAGING.value,
            ActionType.REGRASP_FOR_ORIENTATION.value,
        }
        for index, result in enumerate(relevant_results)
    )
    if role in {"roof", "triangle_top"} and successful_staging_after_final_failure:
        # One camera-visible staging attempt is allowed.  If the fresh frame
        # still cannot produce verified SE(3) evidence, moving among more XY
        # slots has no demonstrated orientation benefit and must not loop.  A
        # later failed final placement starts a new repair episode: the part
        # is on the structure again and must be returned to the table before
        # changing which triangular face/edge is presented.
        return ()
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
    roof_blocker_staging = (
        role == "blocker"
        and obj.shape in set(config.section("house")["roof_classes"])
    )
    if roof_blocker_staging:
        safe_poses = _far_house_staging_poses(safe_poses, config)
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
            "staging_purpose": (
                "change_orientation_observation" if role in {"roof", "triangle_top"}
                else "change_accessibility"
            ),
            "expected_next_grasp_family": "stable_edge_aligned",
            "expected_orientation_evidence": (
                ["long_axis_base", "broad_face_or_groove_or_right_angle_axis", "orientation_xyzw"]
                if role in {"roof", "triangle_top"} else []
            ),
            "requires_fresh_vlm_verification": role in {"roof", "triangle_top"},
            "failed_direct_orientation_reason": str(
                obj.source.get("failed_direct_orientation_reason")
                or "insufficient_semantic_3d_evidence_or_airborne_sweep"
            ),
        }
        if role not in {"roof", "triangle_top"}:
            targets.append(PlacementTarget(
                action_type=ActionType.EXTRACT_TO_STAGING,
                target_region_id=(
                    "house_roof_far_staging"
                    if roof_blocker_staging else "house_blocker_staging"
                ),
                task_role=role,
                place_pose=staging_pose,
                task_progress_gain=0.0,
                expected_effects=(
                    "remove_blocker_to_far_staging"
                    if roof_blocker_staging else "remove_blocker_to_staging",
                    "require_fresh_observation",
                ),
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


def _triangle_tabletop_step_targets(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    role: str,
    config: StackDemoConfig,
    interval: GraspInterval,
) -> tuple[PlacementTarget, ...]:
    """One 45-degree rigid rotation followed by release and fresh observation."""
    target_pose = build_triangle_tabletop_step_pose(
        obj, obj.center_xyz_m[:2], config,
    )
    if target_pose is None:
        return ()
    trajectory = build_airborne_orientation_plan(
        scene, obj, interval.selected_yaw_deg, target_pose, config,
    )
    if trajectory is None:
        return ()
    clearance = float(config.section("house_orientation")["tilted_place_clearance_m"])
    return (PlacementTarget(
        action_type=ActionType.EXTRACT_TO_STAGING,
        target_region_id="house_orientation_staging",
        task_role=role,
        place_pose=target_pose,
        task_progress_gain=0.0,
        expected_effects=(
            "rotate_triangle_one_45deg_step_on_table",
            "release_before_orientation_decision",
            "require_fresh_orientation_observation",
        ),
        precheck_results=_staging_checks({
            "incremental_45deg_tabletop_reorientation": True,
            "final_house_transport_forbidden_this_edge": True,
        }),
        additional_physical_parameters={
            "orientation_trajectory": trajectory,
            "observed_object_pose": {
                "frame_id": "base_link",
                "position_m": list(obj.center_xyz_m),
                "orientation_xyzw": list(obj.source["orientation_xyzw"]),
            },
            "target_object_pose": target_pose,
            "grasp_tcp_object_transform": trajectory["grasp_tcp_object_transform"],
            "release_grasp_tcp_pose": trajectory["release_grasp_tcp_pose"],
            "airborne_adjustment_candidates": [trajectory["airborne_candidate_checks"]],
            "orientation_candidates": [target_pose],
            "orientation_sweep_checks": trajectory["orientation_sweep_checks"],
            "required_3d_rotation_deg": trajectory["actual_rotation_angle_deg"],
            "held_object_rotation_radius_m": trajectory["held_object_rotation_radius_m"],
            "required_grasp_yaw_deg": interval.selected_yaw_deg,
            "required_grasp_checks": dict(interval.selected_check),
            "staging_purpose": "incremental_tabletop_apex_up_reorientation",
            "camera_reobservable": True,
            "requires_fresh_vlm_verification": True,
            "airborne_adjustment_kind": "one_45deg_step_then_table_release",
            "tilted_place_clearance_m": clearance,
            "tilted_place_clearance_applied": True,
            "final_house_transport_forbidden_this_edge": True,
        },
    ),)


def _far_house_staging_poses(
    safe_poses: tuple[tuple[dict[str, Any], dict[str, Any], float], ...],
    config: StackDemoConfig,
) -> tuple[tuple[dict[str, Any], dict[str, Any], float], ...]:
    """Keep roof-clearing staging away from the unfinished house footprint."""
    house = config.section("house")
    origin_x, origin_y = (float(value) for value in house["origin_center_base_m"][:2])
    minimum_distance = float(house["roof_blocker_staging_min_house_distance_m"])

    def distance(item: tuple[dict[str, Any], dict[str, Any], float]) -> float:
        position = item[0]["position_m"]
        return math.hypot(float(position[0]) - origin_x, float(position[1]) - origin_y)

    far = tuple(item for item in safe_poses if distance(item) >= minimum_distance)
    eligible = far or tuple(sorted(safe_poses, key=lambda item: -distance(item))[:1])
    return tuple(sorted(eligible, key=lambda item: -distance(item)))


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
        item for item in (
            (*scene.current_objects, *scene.collision_obstacles)
            if scene is not None else ()
        )
        if item.track_id != obj.track_id
    )
    # Never infer the table from only the currently visible objects: after the
    # house is built the sole movable item can be sitting on the roof, which
    # made the old code release it at roof height and let it fall/roll.  The
    # calibrated base_link table plane is stable across observations.
    table_z = float(house.get("table_surface_z_m", 0.0))
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
            "opening_ok", "axial_coverage_ok", "stable_opposed_contact_ok",
            "center_offset_ok", "contact_length_ok", "finger_safe", "palm_safe",
            "descent_safe", "lift_safe",
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
        lower = _scene_object(scene, state.role_bindings.get(lower_role, ""))
        if lower is None:
            raise ValueError(f"missing verified lower support for {role}")
        x, origin_y = lower.center_xyz_m[:2]
        z = lower.center_xyz_m[2] + 0.5 * (lower.size_xyz_m[2] + obj.size_xyz_m[2])
    elif role.endswith("_lower"):
        opposite_role = (
            "right_support_lower" if role.startswith("left_")
            else "left_support_lower"
        )
        opposite = _scene_object(scene, state.role_bindings.get(opposite_role, ""))
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
        supports = [
            _scene_object(scene, state.role_bindings.get(name, ""))
            for name in ("left_support_upper", "right_support_upper")
        ]
        if any(item is None for item in supports):
            raise ValueError("roof requires both verified upper supports")
        # The roof defines the configured house center.  Using the midpoint of
        # two noisy/partially merged support masks moved the live roof target
        # about 5 mm away from that center in the 20260719 runs.
        x, origin_y = origin_x, origin_y
        z = max(item.center_xyz_m[2] + 0.5 * item.size_xyz_m[2] for item in supports) + 0.5 * obj.size_xyz_m[2]
    elif role == "triangle_top":
        roof = _scene_object(scene, state.role_bindings.get("roof", ""))
        if roof is None:
            raise ValueError("triangle_top requires a verified roof")
        # Like the roof itself, the triangle belongs at the configured house
        # center.  The visible roof mask is commonly clipped by the supports
        # and previously displaced the live target by several millimetres.
        x, origin_y = origin_x, origin_y
        z = roof.center_xyz_m[2] + 0.5 * (roof.size_xyz_m[2] + obj.size_xyz_m[2])
    z += float(house.get(
        "roof_final_place_z_offset_m" if role == "roof" else "final_place_z_offset_m",
        0.0,
    ))
    return {"frame_id": "base_link", "position_m": [x, origin_y, z], "yaw_deg": 0.0}


def _scene_object(
    scene: ClutterSceneState,
    track_id: str,
) -> SceneObjectState | None:
    """Resolve a verified role from the live frame or remembered collision state."""
    return scene.object_by_track(track_id) or next(
        (obj for obj in scene.collision_obstacles if obj.track_id == track_id),
        None,
    )


def _edge_aligned_grasps(
    scene: ClutterSceneState,
    obj: SceneObjectState,
    interval: GraspInterval | None,
    config: StackDemoConfig,
) -> tuple[dict[str, Any], ...]:
    """Return both collision-free edge grasps for destination evaluation."""
    if interval is None:
        return ()
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
            "opening_ok", "axial_coverage_ok", "stable_opposed_contact_ok",
            "center_offset_ok", "contact_length_ok", "finger_safe", "palm_safe",
            "descent_safe", "lift_safe",
        ))
        if not safe:
            continue
        delta = abs((yaw - float(interval.selected_yaw_deg) + 90.0) % 180.0 - 90.0)
        feasible.append((delta, yaw, checks))
    output = []
    for delta, yaw, checks in sorted(feasible, key=lambda item: (item[0], abs(item[1]))):
        edge_error = min(
            abs((yaw - axis + 90.0) % 180.0 - 90.0)
            for axis in candidates
        )
        output.append({
            "yaw_deg": yaw,
            "checks": checks,
            "edge_alignment_error_deg": edge_error,
            "selected_interval_delta_deg": delta,
        })
    return tuple(output)
