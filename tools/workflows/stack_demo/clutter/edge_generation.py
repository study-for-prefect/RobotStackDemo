"""Generate executable first-step edges before any Qwen policy call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..common.action_edges import ActionType, PhysicalActionEdge, make_edge
from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .candidate_diagnostics import CandidateGenerationAudit
from .grasp_edges import (
    GraspInterval,
    GraspScanResult,
    normalize_gripper_yaw_deg,
    scan_grasp_yaws,
)
from .nudge_geometry import (
    blocked_intervals_for as _blocked_intervals_for,
    direction_name as _direction_name,
    nudge_region_effects as _nudge_region_effects,
    object_separation as _separation,
    opposite_side as _opposite_side,
    point_separation as _point_separation,
)
from .path_safety import build_nudge_parameters, nudge_sweep_checks, placement_path_checks
from .push_orientation_priority import axis_alignment_error_deg


@dataclass(frozen=True)
class PlacementTarget:
    action_type: ActionType
    target_region_id: str | None
    task_role: str | None
    place_pose: Mapping[str, Any]
    task_progress_gain: float
    expected_effects: tuple[str, ...]
    precheck_results: Mapping[str, Any] | None = None
    additional_physical_parameters: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EdgeGenerationResult:
    edges_by_target: Mapping[str, tuple[PhysicalActionEdge, ...]]
    grasp_scans: Mapping[str, GraspScanResult]
    audit: CandidateGenerationAudit


@dataclass(frozen=True)
class TargetSpec:
    """One legal task interpretation of a physical target track."""

    option_key: str
    track_id: str
    task_role: str | None = None


PlacementProvider = Callable[[SceneObjectState, GraspInterval], PlacementTarget | None]
VariantPlacementProvider = Callable[[SceneObjectState, GraspInterval, str | None], PlacementTarget | None]
StagingProvider = Callable[[SceneObjectState], PlacementTarget | None]
PlanChecker = Callable[[PhysicalActionEdge], Mapping[str, Any]]

FULL_3D_ORIENTATION_SHAPES = frozenset({"triangle", "concave_rectangle"})


def generate_physical_edges(
    scene: ClutterSceneState,
    target_track_ids: Sequence[str],
    task_type: str,
    config: StackDemoConfig,
    placement_provider: PlacementProvider,
    staging_provider: StagingProvider,
    *,
    plan_checker: PlanChecker | None = None,
    target_specs: Sequence[TargetSpec] | None = None,
    variant_placement_provider: VariantPlacementProvider | None = None,
) -> EdgeGenerationResult:
    """Generate grasp, blocker-removal, and side-push edges for legal targets."""
    checker = plan_checker or geometry_dry_run_plan_checker
    specs = tuple(target_specs or (TargetSpec(track_id, track_id) for track_id in target_track_ids))
    physical_tracks = tuple(dict.fromkeys(spec.track_id for spec in specs))
    scans = {
        track_id: scan_grasp_yaws(obj, scene.current_objects, config)
        for track_id in physical_tracks
        if (obj := scene.object_by_track(track_id)) is not None
    }
    audit = CandidateGenerationAudit(physical_tracks)
    for scan in scans.values():
        audit.record_grasp_scan(scan)
    output: dict[str, tuple[PhysicalActionEdge, ...]] = {}
    for spec in specs:
        target = scene.object_by_track(spec.track_id)
        scan = scans.get(spec.track_id)
        if (
            target is None or scan is None or target.protected or target.already_completed
            or bool(target.source.get("tracking_ambiguous"))
        ):
            output[spec.option_key] = ()
            continue
        provider = (
            (lambda obj, interval: variant_placement_provider(obj, interval, spec.task_role))
            if variant_placement_provider is not None else placement_provider
        )
        direct_edges = _direct_edges(
            scene, target, scan, task_type, config, provider, checker,
            candidate_suffix=_candidate_suffix(spec.option_key, spec.track_id),
        )
        for edge in direct_edges:
            audit.record_direct_edge(edge)
        edges = [edge for edge in direct_edges if edge.precheck_results.get("passed")]
        if task_type == "organize_blocks":
            if not edges and scan.safe_intervals:
                staging = staging_provider(target)
                if staging is not None:
                    edges.extend(
                        edge for edge in _staging_edges(
                            scene, target, scan, task_type, config, staging, checker,
                            task_role=spec.task_role,
                            candidate_suffix=_candidate_suffix(spec.option_key, spec.track_id),
                        )
                        if edge.precheck_results.get("passed")
                    )
            audit.begin_push_object(target.track_id)
            edges.extend(_nudge_edges(
                scene, target, target, task_type, config, checker, scan,
                task_role=spec.task_role,
                candidate_suffix=_candidate_suffix(spec.option_key, spec.track_id),
                audit=audit,
            ))
        elif not any(edge.precheck_results.get("passed") for edge in direct_edges):
            if scan.safe_intervals:
                staging = staging_provider(target)
                if staging is not None:
                    edges.extend(_staging_edges(
                        scene, target, scan, task_type, config, staging, checker,
                        task_role=spec.task_role,
                        candidate_suffix=_candidate_suffix(spec.option_key, spec.track_id),
                    ))
            direct_failure_blockers = _direct_failure_blockers(direct_edges)
            edges.extend(_clearance_edges(
                scene, target, scan, scans, task_type, config, staging_provider, checker,
                task_role=spec.task_role,
                candidate_suffix=_candidate_suffix(spec.option_key, spec.track_id),
                additional_blocker_track_ids=direct_failure_blockers,
            ))
        output[spec.option_key] = tuple(edge for edge in edges if edge.precheck_results.get("passed"))
    return EdgeGenerationResult(edges_by_target=output, grasp_scans=scans, audit=audit)


def geometry_dry_run_plan_checker(edge: PhysicalActionEdge) -> Mapping[str, Any]:
    """Record that offline dry-run did not claim a real MoveIt plan result."""
    return {
        "passed": True,
        "geometry_checks_passed": True,
        "moveit_plan_only": False,
        "mode": "offline_geometry_dry_run",
        "executable_without_final_moveit_gate": False,
    }


def _direct_edges(
    scene: ClutterSceneState,
    target: SceneObjectState,
    scan: GraspScanResult,
    task_type: str,
    config: StackDemoConfig,
    placement_provider: PlacementProvider,
    checker: PlanChecker,
    *,
    candidate_suffix: str = "",
) -> list[PhysicalActionEdge]:
    edges = []
    limit = int(config.section("grasp")["maximum_edges_per_target"])
    for index, interval in enumerate(scan.safe_intervals[:limit], start=1):
        placement = placement_provider(target, interval)
        if placement is None:
            continue
        physical = _pick_parameters(target, interval, placement, config)
        placement_checks = placement_path_checks(scene, target, placement.place_pose, physical, config)
        prechecks = {
            "finger_safe": True, "palm_safe": True,
            "descent_safe": True, "lift_safe": True,
            **dict(placement.precheck_results or {}),
            **placement_checks,
        }
        prechecks["passed"] = all(value for value in prechecks.values() if isinstance(value, bool))
        prechecks["geometry_checks_passed"] = prechecks["passed"]
        edge = make_edge(
            candidate_id=f"edge_r{scene.scene_revision}_{target.track_id}{candidate_suffix}_grasp_{index}",
            scene_revision=scene.scene_revision,
            action_type=placement.action_type,
            task_type=task_type,
            primary_target_track_id=target.track_id,
            acted_object_track_id=target.track_id,
            task_role=placement.task_role,
            target_region_id=placement.target_region_id,
            physical_parameters=physical,
            expected_effects=placement.expected_effects,
            expected_released_tracks=_released_neighbors(scene, target),
            task_progress_gain=placement.task_progress_gain,
            risk_score=_grasp_risk(interval),
            protected_tracks=scene.protected_tracks,
            precheck_results=prechecks,
            decision_metadata={"grasp_interval": interval.to_dict(), "object_ref": target.object_ref},
        )
        edges.append(_with_plan_check(edge, checker))
    return edges


def _clearance_edges(
    scene: ClutterSceneState,
    target: SceneObjectState,
    scan: GraspScanResult,
    scans: Mapping[str, GraspScanResult],
    task_type: str,
    config: StackDemoConfig,
    staging_provider: StagingProvider,
    checker: PlanChecker,
    *,
    task_role: str | None = None,
    candidate_suffix: str = "",
    additional_blocker_track_ids: Sequence[str] = (),
) -> list[PhysicalActionEdge]:
    edges: list[PhysicalActionEdge] = []
    blocker_ids = tuple(dict.fromkeys((*scan.blocker_track_ids, *additional_blocker_track_ids)))
    for blocker_id in blocker_ids:
        blocker = scene.object_by_track(blocker_id)
        if (
            blocker is None or blocker.protected or blocker.already_completed
            or bool(blocker.source.get("tracking_ambiguous"))
        ):
            continue
        blocker_scan = scans.get(blocker_id) or scan_grasp_yaws(blocker, scene.current_objects, config)
        if blocker_scan.graspable:
            staging = staging_provider(blocker)
            if staging is not None:
                interval = blocker_scan.safe_intervals[0]
                physical = _pick_parameters(blocker, interval, staging, config)
                placement_checks = placement_path_checks(scene, blocker, staging.place_pose, physical, config)
                prechecks = {
                    "finger_safe": True, "palm_safe": True,
                    "descent_safe": True, "lift_safe": True,
                    **dict(staging.precheck_results or {}),
                    **placement_checks,
                }
                prechecks["passed"] = all(value for value in prechecks.values() if isinstance(value, bool))
                prechecks["geometry_checks_passed"] = prechecks["passed"]
                edge = make_edge(
                    candidate_id=f"edge_r{scene.scene_revision}_{target.track_id}{candidate_suffix}_pick_blocker_{blocker.track_id}",
                    scene_revision=scene.scene_revision,
                    action_type=ActionType.PICK_AWAY_BLOCKER,
                    task_type=task_type,
                    primary_target_track_id=target.track_id,
                    acted_object_track_id=blocker.track_id,
                    task_role=task_role,
                    target_region_id=staging.target_region_id,
                    physical_parameters=physical,
                    expected_effects=("remove_direct_grasp_blocker",) + staging.expected_effects,
                    expected_clearance_gain_m=_separation(target, blocker),
                    expected_released_tracks=(target.track_id,),
                    task_progress_gain=staging.task_progress_gain,
                    risk_score=0.35 + _grasp_risk(interval),
                    protected_tracks=scene.protected_tracks,
                    precheck_results=prechecks,
                    decision_metadata={
                        "blocked_intervals": _blocked_intervals_for(scan, blocker_id),
                        "object_ref": blocker.object_ref,
                        "primary_target_object_ref": target.object_ref,
                        "acted_object_geometry": {
                            "geometry_center_m": list(blocker.center_xyz_m),
                            "dimensions_m": list(blocker.size_xyz_m),
                        },
                    },
                )
                checked = _with_plan_check(edge, checker)
                if checked.precheck_results.get("passed"):
                    edges.append(checked)
        edges.extend(_nudge_edges(
            scene, target, blocker, task_type, config, checker, scan,
            task_role=task_role, candidate_suffix=candidate_suffix,
        ))
    return edges


def _staging_edges(
    scene: ClutterSceneState,
    target: SceneObjectState,
    scan: GraspScanResult,
    task_type: str,
    config: StackDemoConfig,
    staging: PlacementTarget,
    checker: PlanChecker,
    *,
    task_role: str | None,
    candidate_suffix: str,
) -> list[PhysicalActionEdge]:
    edges: list[PhysicalActionEdge] = []
    if staging.action_type not in {ActionType.EXTRACT_TO_STAGING, ActionType.REGRASP_FOR_ORIENTATION}:
        return edges
    for index, interval in enumerate(scan.safe_intervals[:1], start=1):
        physical = _pick_parameters(target, interval, staging, config)
        checks = {
            "finger_safe": True,
            "palm_safe": True,
            "descent_safe": True,
            "lift_safe": True,
            **dict(staging.precheck_results or {}),
            **placement_path_checks(scene, target, staging.place_pose, physical, config),
        }
        checks["passed"] = all(value for value in checks.values() if isinstance(value, bool))
        checks["geometry_checks_passed"] = checks["passed"]
        edge = make_edge(
            candidate_id=f"edge_r{scene.scene_revision}_{target.track_id}{candidate_suffix}_staging_{index}",
            scene_revision=scene.scene_revision,
            action_type=staging.action_type,
            task_type=task_type,
            primary_target_track_id=target.track_id,
            acted_object_track_id=target.track_id,
            task_role=task_role,
            target_region_id=staging.target_region_id,
            physical_parameters=physical,
            expected_effects=staging.expected_effects,
            task_progress_gain=staging.task_progress_gain,
            risk_score=0.4 + _grasp_risk(interval),
            protected_tracks=scene.protected_tracks,
            precheck_results=checks,
            decision_metadata={
                "grasp_interval": interval.to_dict(),
                "object_ref": target.object_ref,
                "staging_position_tolerance_m": 0.025,
            },
        )
        edges.append(_with_plan_check(edge, checker))
    return edges


def _nudge_edges(
    scene: ClutterSceneState,
    target: SceneObjectState,
    blocker: SceneObjectState,
    task_type: str,
    config: StackDemoConfig,
    checker: PlanChecker,
    target_scan: GraspScanResult,
    *,
    task_role: str | None = None,
    candidate_suffix: str = "",
    audit: CandidateGenerationAudit | None = None,
) -> list[PhysicalActionEdge]:
    clearing = config.section("clearing")
    edges = []
    checked_edges = []
    old_separation = _separation(target, blocker)
    for direction in clearing["push_directions_base"]:
        contact_side = _opposite_side(direction)
        for distance in clearing["push_distances_m"]:
            moved_center = (
                blocker.center_xyz_m[0] + float(direction[0]) * float(distance),
                blocker.center_xyz_m[1] + float(direction[1]) * float(distance),
                blocker.center_xyz_m[2],
            )
            gain = _point_separation(target.center_xyz_m, moved_center) - old_separation
            if gain < float(clearing["minimum_expected_clearance_gain_m"]):
                continue
            region_effects = _nudge_region_effects(blocker, moved_center, scene)
            for wrist_yaw in clearing["safe_wrist_yaws_deg"]:
                physical = build_nudge_parameters(blocker, direction, distance, wrist_yaw, config)
                axis_error = axis_alignment_error_deg(float(wrist_yaw))
                physical["push_axis_alignment_error_deg"] = axis_error
                physical["axis_aligned_push_orientation"] = axis_error < 1e-6
                sweep = nudge_sweep_checks(scene, blocker, physical, config)
                physical = {
                    **physical,
                    "chain_track_ids": list(sweep.get("chain_track_ids", [blocker.track_id])),
                    "chain_object_count": int(sweep.get("chain_object_count", 1)),
                    "secondary_contact_expected": bool(sweep.get("secondary_contact_expected")),
                    "estimated_displacements_m": dict(sweep.get("estimated_displacements_m", {})),
                    "estimated_displacements_are_upper_bounds": True,
                    "push_chain_estimation_method": sweep.get("push_chain_estimation_method"),
                }
                suffix = f"{_direction_name(direction)}_{int(round(float(distance) * 1000))}_{int(wrist_yaw)}"
                edge = make_edge(
                    candidate_id=f"edge_r{scene.scene_revision}_{target.track_id}{candidate_suffix}_nudge_{blocker.track_id}_{suffix}",
                    scene_revision=scene.scene_revision,
                    action_type=ActionType.NUDGE_BLOCKER,
                    task_type=task_type,
                    primary_target_track_id=target.track_id,
                    acted_object_track_id=blocker.track_id,
                    task_role=task_role,
                    physical_parameters=physical,
                    expected_effects=(
                        (
                            "clear_or_organize_target"
                            if target.track_id == blocker.track_id
                            else "increase_target_grasp_clearance"
                        ),
                        "require_fresh_observation",
                        *(("push_chain_generated",) if physical["secondary_contact_expected"] else ()),
                        *region_effects["expected_effects"],
                    ),
                    expected_clearance_gain_m=gain,
                    expected_released_tracks=(
                        _released_neighbors(scene, blocker)
                        if target.track_id == blocker.track_id else (target.track_id,)
                    ),
                    task_progress_gain=float(region_effects["task_progress_gain"]),
                    risk_score=(
                        0.55 + 0.5 * float(distance)
                        + 0.12 * max(0, int(physical["chain_object_count"]) - 1)
                        + 0.20 * axis_error / 45.0
                        + float(region_effects["risk_delta"])
                    ),
                    protected_tracks=scene.protected_tracks,
                    precheck_results=sweep,
                    decision_metadata={
                        "direct_blocker_for_intervals": _blocked_intervals_for(target_scan, blocker.track_id),
                        "loose_contact_allowed_phase": "horizontal_push_only",
                        "object_ref": blocker.object_ref,
                        "primary_target_object_ref": target.object_ref,
                        "acted_object_geometry": {
                            "geometry_center_m": list(blocker.center_xyz_m),
                            "dimensions_m": list(blocker.size_xyz_m),
                        },
                        "destination_color_region": region_effects["destination_color_region"],
                        "destination_region_id": region_effects["destination_region_id"],
                        "destination_region_occupants": region_effects["destination_region_occupants"],
                        "color_region_is_collision_geometry": False,
                        "chain_track_ids": list(physical["chain_track_ids"]),
                        "chain_object_count": int(physical["chain_object_count"]),
                        "secondary_contact_expected": bool(physical["secondary_contact_expected"]),
                        "estimated_displacements_m": dict(physical["estimated_displacements_m"]),
                        "push_axis_alignment_error_deg": axis_error,
                        "orientation_priority": "axis_aligned_0_or_90_preferred_when_equivalent",
                    },
                )
                checked = _with_plan_check(edge, checker)
                checked_edges.append(checked)
    for checked in checked_edges:
        if audit is not None:
            audit.record_push_edge(checked)
        if checked.precheck_results.get("passed"):
            edges.append(checked)
    return edges


def _pick_parameters(
    target: SceneObjectState,
    interval: GraspInterval,
    placement: PlacementTarget,
    config: StackDemoConfig,
) -> dict[str, Any]:
    safety = config.section("safety")
    x, y, z = target.center_xyz_m
    approach_z = z + float(safety["approach_height_m"])
    lift_z = z + float(safety["observation_height_m"])
    place_pose = dict(placement.place_pose)
    grasp_yaw = normalize_gripper_yaw_deg(interval.selected_yaw_deg)
    current_object_yaw = float(target.yaw_deg)
    requested_object_yaw = float(place_pose.get("yaw_deg", current_object_yaw))
    object_relative_to_gripper_yaw = _axis_delta_deg(current_object_yaw, grasp_yaw)
    full_3d_orientation = target.shape in FULL_3D_ORIENTATION_SHAPES
    if full_3d_orientation:
        expected_object_yaw = requested_object_yaw
        release_gripper_yaw = _normalize_axis_yaw_deg(
            expected_object_yaw - object_relative_to_gripper_yaw
        )
        placement_yaw_policy = "special_shape_target_orientation"
    else:
        # Organize completion does not require final yaw.  Keep the grasp
        # orientation fixed through transport, descent, release, and retreat.
        expected_object_yaw = current_object_yaw
        release_gripper_yaw = grasp_yaw
        placement_yaw_policy = "preserve_grasp_yaw_until_release"
        place_pose["yaw_deg"] = expected_object_yaw
    release_pose = dict(place_pose)
    release_position = list(place_pose["position_m"])
    release_position[2] += float(safety["release_height_extra_m"])
    release_pose["position_m"] = release_position
    release_pose["yaw_deg"] = release_gripper_yaw
    orientation_policy = (
        "full_3d_allowed"
        if full_3d_orientation
        else "downward_yaw_only"
    )
    destination_at_lift = {
        "frame_id": "base_link",
        "position_m": list(place_pose["position_m"][:2]) + [lift_z],
        "yaw_deg": grasp_yaw,
        "motion_role": "translate_to_destination_preserving_grasp_yaw",
    }
    transport_path = [
        {
            "frame_id": "base_link",
            "position_m": [x, y, lift_z],
            "yaw_deg": grasp_yaw,
            "motion_role": "source_lift",
        },
        destination_at_lift,
    ]
    if full_3d_orientation:
        transport_path.append({
            "frame_id": "base_link",
            "position_m": list(place_pose["position_m"][:2]) + [lift_z],
            "yaw_deg": release_gripper_yaw,
            "motion_role": "special_shape_orientation_at_destination",
        })
    return {
        "orientation_policy": orientation_policy,
        "placement_yaw_policy": placement_yaw_policy,
        "acted_object_shape": target.shape,
        "grasp_yaw_deg": grasp_yaw,
        "grasp_pose": {"frame_id": "base_link", "position_m": [x, y, z], "yaw_deg": grasp_yaw},
        "approach_pose": {"frame_id": "base_link", "position_m": [x, y, approach_z], "yaw_deg": grasp_yaw},
        "lift_pose": {"frame_id": "base_link", "position_m": [x, y, lift_z], "yaw_deg": grasp_yaw},
        "place_pose": place_pose,
        "release_pose": release_pose,
        "grasp_object_relative_yaw_deg": object_relative_to_gripper_yaw,
        "requested_place_object_yaw_deg": requested_object_yaw,
        "expected_place_object_yaw_deg": expected_object_yaw,
        "release_gripper_yaw_deg": release_gripper_yaw,
        "release_height_extra_m": float(safety["release_height_extra_m"]),
        "transport_path": transport_path,
        "grasp_checks": dict(interval.selected_check),
        **dict(placement.additional_physical_parameters or {}),
    }


def _axis_delta_deg(first: float, second: float) -> float:
    return (float(first) - float(second) + 90.0) % 180.0 - 90.0


def _normalize_axis_yaw_deg(value: float) -> float:
    return normalize_gripper_yaw_deg(value)


def _with_plan_check(edge: PhysicalActionEdge, checker: PlanChecker) -> PhysicalActionEdge:
    result = (
        dict(checker(edge))
        if bool(edge.precheck_results.get("passed"))
        else {
            "passed": False,
            "moveit_plan_only": False,
            "mode": "skipped_geometry_precheck_failed",
        }
    )
    merged = {**dict(edge.precheck_results), **result}
    merged["passed"] = bool(edge.precheck_results.get("passed")) and bool(result.get("passed"))
    rejection_reasons = list(edge.precheck_results.get("rejection_reasons", ()))
    if bool(edge.precheck_results.get("passed")) and not bool(result.get("passed")):
        rejection_reasons.append("moveit_plan_failed")
    merged["rejection_reasons"] = list(dict.fromkeys(rejection_reasons))
    return make_edge(
        candidate_id=edge.candidate_id,
        scene_revision=edge.scene_revision,
        action_type=edge.action_type,
        task_type=edge.task_type,
        primary_target_track_id=edge.primary_target_track_id,
        acted_object_track_id=edge.acted_object_track_id,
        task_role=edge.task_role,
        target_region_id=edge.target_region_id,
        physical_parameters=edge.physical_parameters,
        expected_effects=edge.expected_effects,
        expected_clearance_gain_m=edge.expected_clearance_gain_m,
        expected_released_tracks=edge.expected_released_tracks,
        task_progress_gain=edge.task_progress_gain,
        risk_score=edge.risk_score,
        protected_tracks=edge.protected_tracks,
        precheck_results=merged,
        decision_metadata=edge.decision_metadata,
    )


def _direct_failure_blockers(edges: Sequence[PhysicalActionEdge]) -> tuple[str, ...]:
    return tuple(sorted({
        str(track_id)
        for edge in edges
        for key in ("transport_blocking_track_ids", "place_blocking_track_ids")
        for track_id in edge.precheck_results.get(key, [])
        if track_id
    }))


def _released_neighbors(scene: ClutterSceneState, target: SceneObjectState) -> tuple[str, ...]:
    released = []
    for other in scene.current_objects:
        if target.track_id in other.neighbors and len(other.blocked_sides) > 0:
            released.append(other.track_id)
    return tuple(sorted(released))


def _grasp_risk(interval: GraspInterval) -> float:
    clearance = float(interval.selected_check.get("fingertip_clearance_m", 0.0))
    return max(0.0, 0.5 - interval.span_deg / 180.0 - clearance)


def _candidate_suffix(option_key: str, track_id: str) -> str:
    if option_key == track_id:
        return ""
    safe = "".join(character if character.isalnum() else "_" for character in option_key)
    return f"_{safe}"
