"""Generate executable first-step edges before any Qwen policy call."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from ..common.action_edges import ActionType, PhysicalActionEdge, make_edge
from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .grasp_edges import (
    GraspInterval,
    GraspScanResult,
    scan_grasp_yaws,
)
from .path_safety import build_nudge_parameters, nudge_sweep_checks, placement_path_checks


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
        edges = list(direct_edges)
        if not any(edge.precheck_results.get("passed") for edge in direct_edges):
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
    return EdgeGenerationResult(edges_by_target=output, grasp_scans=scans)


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
) -> list[PhysicalActionEdge]:
    clearing = config.section("clearing")
    edges = []
    old_separation = _separation(target, blocker)
    for direction in clearing["push_directions_base"]:
        contact_side = _opposite_side(direction)
        if contact_side not in blocker.free_sides:
            continue
        for distance in clearing["push_distances_m"]:
            moved_center = (
                blocker.center_xyz_m[0] + float(direction[0]) * float(distance),
                blocker.center_xyz_m[1] + float(direction[1]) * float(distance),
                blocker.center_xyz_m[2],
            )
            gain = _point_separation(target.center_xyz_m, moved_center) - old_separation
            if gain < float(clearing["minimum_expected_clearance_gain_m"]):
                continue
            if not _object_center_inside_workspace(blocker, moved_center, scene):
                continue
            if _inside_target_region(moved_center, scene.target_regions):
                continue
            if _sweep_hits_protected(blocker, moved_center, scene):
                continue
            for wrist_yaw in clearing["safe_wrist_yaws_deg"]:
                physical = build_nudge_parameters(blocker, direction, distance, wrist_yaw, config)
                sweep = nudge_sweep_checks(scene, blocker, physical, config)
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
                    expected_effects=("increase_target_grasp_clearance", "require_fresh_observation"),
                    expected_clearance_gain_m=gain,
                    expected_released_tracks=(target.track_id,),
                    task_progress_gain=0.0,
                    risk_score=0.55 + 0.5 * float(distance),
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
                    },
                )
                checked = _with_plan_check(edge, checker)
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
    grasp_yaw = float(interval.selected_yaw_deg)
    current_object_yaw = float(target.yaw_deg)
    expected_object_yaw = float(place_pose.get("yaw_deg", current_object_yaw))
    object_relative_to_gripper_yaw = _axis_delta_deg(current_object_yaw, grasp_yaw)
    release_gripper_yaw = _normalize_axis_yaw_deg(
        expected_object_yaw - object_relative_to_gripper_yaw
    )
    release_pose = dict(place_pose)
    release_position = list(place_pose["position_m"])
    release_position[2] += float(safety["release_height_extra_m"])
    release_pose["position_m"] = release_position
    release_pose["yaw_deg"] = release_gripper_yaw
    return {
        "grasp_yaw_deg": grasp_yaw,
        "grasp_pose": {"frame_id": "base_link", "position_m": [x, y, z], "yaw_deg": grasp_yaw},
        "approach_pose": {"frame_id": "base_link", "position_m": [x, y, approach_z], "yaw_deg": grasp_yaw},
        "lift_pose": {"frame_id": "base_link", "position_m": [x, y, lift_z], "yaw_deg": grasp_yaw},
        "place_pose": place_pose,
        "release_pose": release_pose,
        "grasp_object_relative_yaw_deg": object_relative_to_gripper_yaw,
        "expected_place_object_yaw_deg": expected_object_yaw,
        "release_gripper_yaw_deg": release_gripper_yaw,
        "release_height_extra_m": float(safety["release_height_extra_m"]),
        "transport_path": [
            {"frame_id": "base_link", "position_m": [x, y, lift_z], "yaw_deg": grasp_yaw},
            {
                "frame_id": "base_link",
                "position_m": list(place_pose["position_m"][:2]) + [lift_z],
                "yaw_deg": release_gripper_yaw,
            },
        ],
        "grasp_checks": dict(interval.selected_check),
        **dict(placement.additional_physical_parameters or {}),
    }


def _axis_delta_deg(first: float, second: float) -> float:
    return (float(first) - float(second) + 90.0) % 180.0 - 90.0


def _normalize_axis_yaw_deg(value: float) -> float:
    return round((float(value) + 90.0) % 180.0 - 90.0, 6)


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


def _separation(first: SceneObjectState, second: SceneObjectState) -> float:
    return _point_separation(first.center_xyz_m, second.center_xyz_m)


def _point_separation(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _object_center_inside_workspace(
    obj: SceneObjectState,
    center: Sequence[float],
    scene: ClutterSceneState,
) -> bool:
    bounds = _workspace_from_edge_clearances(obj)
    x, y = float(center[0]), float(center[1])
    return bounds["xmin"] <= x - obj.size_xyz_m[0] / 2.0 and x + obj.size_xyz_m[0] / 2.0 <= bounds["xmax"] and bounds["ymin"] <= y - obj.size_xyz_m[1] / 2.0 and y + obj.size_xyz_m[1] / 2.0 <= bounds["ymax"]


def _workspace_from_edge_clearances(obj: SceneObjectState) -> dict[str, float]:
    x, y, _ = obj.center_xyz_m
    sx, sy, _ = obj.size_xyz_m
    return {
        "xmin": x - sx / 2.0 - float(obj.edge_clearances_m["-x"]),
        "xmax": x + sx / 2.0 + float(obj.edge_clearances_m["+x"]),
        "ymin": y - sy / 2.0 - float(obj.edge_clearances_m["-y"]),
        "ymax": y + sy / 2.0 + float(obj.edge_clearances_m["+y"]),
    }


def _inside_target_region(center: Sequence[float], regions: Sequence[Mapping[str, Any]]) -> bool:
    for region in regions:
        bounds = region.get("bounds_base_m") or region.get("bounds")
        if isinstance(bounds, Mapping) and float(bounds["xmin"]) <= center[0] <= float(bounds["xmax"]) and float(bounds["ymin"]) <= center[1] <= float(bounds["ymax"]):
            return True
    return False


def _sweep_hits_protected(
    blocker: SceneObjectState,
    moved_center: Sequence[float],
    scene: ClutterSceneState,
) -> bool:
    xmin = min(blocker.center_xyz_m[0], moved_center[0]) - blocker.size_xyz_m[0] / 2.0
    xmax = max(blocker.center_xyz_m[0], moved_center[0]) + blocker.size_xyz_m[0] / 2.0
    ymin = min(blocker.center_xyz_m[1], moved_center[1]) - blocker.size_xyz_m[1] / 2.0
    ymax = max(blocker.center_xyz_m[1], moved_center[1]) + blocker.size_xyz_m[1] / 2.0
    for track_id in scene.protected_tracks:
        obj = scene.object_by_track(track_id)
        if obj is None:
            continue
        ox, oy, _ = obj.center_xyz_m
        if xmin <= ox + obj.size_xyz_m[0] / 2.0 and xmax >= ox - obj.size_xyz_m[0] / 2.0 and ymin <= oy + obj.size_xyz_m[1] / 2.0 and ymax >= oy - obj.size_xyz_m[1] / 2.0:
            return True
    return False


def _direction_name(direction: Sequence[float]) -> str:
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "px" if float(direction[0]) > 0 else "nx"
    return "py" if float(direction[1]) > 0 else "ny"


def _opposite_side(direction: Sequence[float]) -> str:
    """Return the block contact side opposite a code-owned push direction."""
    if abs(float(direction[0])) >= abs(float(direction[1])):
        return "-x" if float(direction[0]) > 0 else "+x"
    return "-y" if float(direction[1]) > 0 else "+y"


def _blocked_intervals_for(scan: GraspScanResult, blocker_id: str) -> list[float]:
    return [float(sample["yaw_deg"]) for sample in scan.samples if blocker_id in sample.get("blocking_track_ids", [])]


def _candidate_suffix(option_key: str, track_id: str) -> str:
    if option_key == track_id:
        return ""
    safe = "".join(character if character.isalnum() else "_" for character in option_key)
    return f"_{safe}"
