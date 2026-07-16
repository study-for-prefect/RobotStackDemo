"""Multi-evidence verification using fresh post-action observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping

from .action_edges import ActionType, PhysicalActionEdge
from .scene_state import ClutterSceneState


@dataclass(frozen=True)
class ActionVerification:
    success: bool
    status: str
    evidence: Mapping[str, Any]
    scene_revision: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DestinationCheck = Callable[[PhysicalActionEdge, ClutterSceneState], tuple[bool, Mapping[str, Any]]]


def verify_grasp_result(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after_lift: ClutterSceneState,
    *,
    gripper_holding_hint: bool | None = None,
    minimum_displacement_m: float = 0.015,
) -> ActionVerification:
    """Require visual scene evidence; gripper state is auxiliary and never sufficient."""
    original = before.object_by_track(edge.acted_object_track_id)
    current = after_lift.object_by_track(edge.acted_object_track_id)
    if original is None:
        return ActionVerification(False, "grasp_failed", {"reason": "pre_grasp_track_missing"}, after_lift.scene_revision)
    displacement = None
    if current is not None:
        displacement = math.dist(original.center_xyz_m, current.center_xyz_m)
    track_missing = current is None
    moved = displacement is not None and displacement >= minimum_displacement_m
    reappeared_near_original = bool(
        current is not None and math.dist(original.center_xyz_m, current.center_xyz_m) < minimum_displacement_m
    )
    other_at_original_position = tuple(sorted(
        obj.track_id for obj in after_lift.current_objects
        if obj.track_id != edge.acted_object_track_id
        and math.dist(original.center_xyz_m, obj.center_xyz_m) < 0.5 * max(original.size_xyz_m[:2])
    ))
    original_position_clear = bool((track_missing or moved) and not reappeared_near_original and not other_at_original_position)
    left_table_workspace = track_missing or (
        current.center_xyz_m[2] > original.center_xyz_m[2] + max(0.5 * original.size_xyz_m[2], minimum_displacement_m)
    )
    other_before = set(before.visible_tracks) - {edge.acted_object_track_id}
    context_tracks_still_visible = tuple(sorted(other_before.intersection(after_lift.visible_tracks)))
    reliably_rebound_context = tuple(sorted(
        track_id for track_id in context_tracks_still_visible
        if (obj := after_lift.object_by_track(track_id)) is not None
        and not bool(obj.source.get("tracking_ambiguous"))
        and float(obj.source.get("track_match_confidence", 1.0)) >= 0.5
    ))
    visible_non_target_context = tuple(sorted(
        obj.track_id for obj in after_lift.current_objects
        if obj.track_id != edge.acted_object_track_id
    ))
    view_continuity = bool(visible_non_target_context)
    missing_supported = bool(
        track_missing
        and original_position_clear
        and (
            len(reliably_rebound_context) >= 2
            or (
                gripper_holding_hint is True
                and len(visible_non_target_context) >= 1
            )
        )
    )
    scene_change_supports_grasp = bool(
        moved
        or missing_supported
    )
    success = bool(original_position_clear and left_table_workspace and scene_change_supports_grasp)
    return ActionVerification(
        success=success,
        status="grasp_verified" if success else "grasp_failed",
        scene_revision=after_lift.scene_revision,
        evidence={
            "target_missing_from_table_view": track_missing,
            "target_displacement_m": displacement,
            "target_moved_from_original_position": moved,
            "target_reappeared_near_original": reappeared_near_original,
            "other_tracks_at_original_position": list(other_at_original_position),
            "original_position_clear": original_position_clear,
            "target_left_table_workspace": left_table_workspace,
            "scene_change_supports_grasp": scene_change_supports_grasp,
            "context_tracks_still_visible": list(context_tracks_still_visible),
            "reliably_rebound_context_tracks": list(reliably_rebound_context),
            "visible_non_target_context_tracks": list(visible_non_target_context),
            "view_continuity_supported": view_continuity,
            "gripper_holding_hint": gripper_holding_hint,
            "gripper_hint_used_as_sole_evidence": False,
            "camera_limit": (
                "wrist-mounted D435i may not see an object inside the raised GF225; "
                "verification therefore uses original-location clearance, track displacement, and scene change"
            ),
        },
    )


def verify_place_result(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after_place: ClutterSceneState,
    destination_check: DestinationCheck,
) -> ActionVerification:
    verifier = {
        ActionType.PICK_PLACE: final_task_place_success,
        ActionType.EXTRACT_THEN_PLACE: final_task_place_success,
        ActionType.PLACE_HOUSE_ROLE: house_role_place_success,
        ActionType.EXTRACT_TO_STAGING: staging_place_success,
        ActionType.PICK_AWAY_BLOCKER: blocker_removed_success,
        ActionType.REGRASP_FOR_ORIENTATION: orientation_staging_success,
        ActionType.REPAIR_STRUCTURE: repair_structure_success,
    }.get(edge.action_type)
    if verifier is None:
        destination_ok, details = False, {"reason": f"unsupported placement action {edge.action_type.value}"}
    elif verifier in {final_task_place_success, house_role_place_success, repair_structure_success}:
        destination_ok, details = verifier(edge, before, after_place, destination_check)
    else:
        destination_ok, details = verifier(edge, before, after_place)
    visible = after_place.object_by_track(edge.acted_object_track_id) is not None
    success = bool(destination_ok and visible)
    return ActionVerification(
        success=success,
        status="place_verified" if success else "place_failed",
        scene_revision=after_place.scene_revision,
        evidence={
            "track_visible": visible,
            "destination_predicate": bool(destination_ok),
            "verification_kind": verifier.__name__ if verifier is not None else "unsupported",
            **dict(details),
        },
    )


def final_task_place_success(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
    destination_check: DestinationCheck,
) -> tuple[bool, Mapping[str, Any]]:
    del before
    return destination_check(edge, after)


def house_role_place_success(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
    destination_check: DestinationCheck,
) -> tuple[bool, Mapping[str, Any]]:
    del before
    valid, details = destination_check(edge, after)
    return bool(valid and edge.task_role), {"house_role": edge.task_role, **dict(details)}


def repair_structure_success(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
    destination_check: DestinationCheck,
) -> tuple[bool, Mapping[str, Any]]:
    del before
    valid, details = destination_check(edge, after)
    return bool(valid), {"repair_predicate": bool(valid), **dict(details)}


def staging_place_success(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
) -> tuple[bool, Mapping[str, Any]]:
    del before
    current = after.object_by_track(edge.acted_object_track_id)
    pose = edge.physical_parameters.get("place_pose") or {}
    target = pose.get("position_m") if isinstance(pose, Mapping) else None
    distance = None
    if current is not None and isinstance(target, (list, tuple)) and len(target) >= 3:
        distance = math.dist(current.center_xyz_m, [float(value) for value in target[:3]])
    tolerance = float(edge.decision_metadata.get("staging_position_tolerance_m", 0.025))
    return bool(distance is not None and distance <= tolerance), {
        "staging_region_id": edge.target_region_id,
        "staging_position_error_m": distance,
        "staging_position_tolerance_m": tolerance,
    }


def blocker_removed_success(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
) -> tuple[bool, Mapping[str, Any]]:
    staged, staging_details = staging_place_success(edge, before, after)
    blocker_before = before.object_by_track(edge.acted_object_track_id)
    blocker_after = after.object_by_track(edge.acted_object_track_id)
    target_before = before.object_by_track(edge.primary_target_track_id)
    target_after = after.object_by_track(edge.primary_target_track_id)
    clearance_gain = None
    if all(item is not None for item in (blocker_before, blocker_after, target_before, target_after)):
        clearance_gain = _xy_distance(target_after, blocker_after) - _xy_distance(target_before, blocker_before)
    minimum = max(0.003, min(float(edge.expected_clearance_gain_m), 0.010))
    return bool(staged and clearance_gain is not None and clearance_gain >= minimum), {
        **dict(staging_details),
        "measured_target_clearance_gain_m": clearance_gain,
        "minimum_clearance_gain_m": minimum,
    }


def orientation_staging_success(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
) -> tuple[bool, Mapping[str, Any]]:
    staged, staging_details = staging_place_success(edge, before, after)
    current = after.object_by_track(edge.acted_object_track_id)
    plan = edge.physical_parameters.get("orientation_plan") or {}
    expected = plan.get("expected_orientation_after") if isinstance(plan, Mapping) else None
    observed = {} if current is None else {
        key: current.source.get(key)
        for key in (
            "groove_face_state", "face_up", "long_axis_yaw_deg", "apex_direction",
            "base_edge_direction", "face_state", "target_yaw_deg",
        ) if key in current.source
    }
    orientation_matches = bool(expected) and all(observed.get(key) == value for key, value in expected.items())
    return bool(staged and orientation_matches), {
        **dict(staging_details),
        "expected_orientation_after": expected,
        "observed_orientation": observed,
        "orientation_transition_verified": orientation_matches,
    }


def verify_nudge_result(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    after: ClutterSceneState,
    *,
    minimum_displacement_m: float = 0.010,
    minimum_clearance_gain_m: float = 0.003,
) -> ActionVerification:
    """Verify a push from measured track motion and protected-track stability."""
    blocker_before = before.object_by_track(edge.acted_object_track_id)
    blocker_after = after.object_by_track(edge.acted_object_track_id)
    target_before = before.object_by_track(edge.primary_target_track_id)
    target_after = after.object_by_track(edge.primary_target_track_id)
    direction = edge.physical_parameters.get("push_direction_base") or ()
    self_target_push = edge.primary_target_track_id == edge.acted_object_track_id
    required_visible = all(item is not None for item in (blocker_before, blocker_after, target_before, target_after))
    signed_displacement = None
    clearance_gain = None
    if required_visible and len(direction) >= 2:
        dx = blocker_after.center_xyz_m[0] - blocker_before.center_xyz_m[0]
        dy = blocker_after.center_xyz_m[1] - blocker_before.center_xyz_m[1]
        signed_displacement = dx * float(direction[0]) + dy * float(direction[1])
        if not self_target_push:
            clearance_gain = _xy_distance(target_after, blocker_after) - _xy_distance(target_before, blocker_before)
    protected_stable = True
    protected_motion: dict[str, float | None] = {}
    for track_id in before.protected_tracks:
        old, new = before.object_by_track(track_id), after.object_by_track(track_id)
        displacement = None if old is None or new is None else math.dist(old.center_xyz_m, new.center_xyz_m)
        protected_motion[track_id] = displacement
        if displacement is None or displacement > 0.005:
            protected_stable = False
    success = bool(
        required_visible
        and signed_displacement is not None and signed_displacement >= minimum_displacement_m
        and (
            self_target_push
            or (clearance_gain is not None and clearance_gain >= minimum_clearance_gain_m)
        )
        and protected_stable
    )
    return ActionVerification(
        success=success,
        status="nudge_verified" if success else "nudge_failed",
        scene_revision=after.scene_revision,
        evidence={
            "required_tracks_visible": required_visible,
            "verification_mode": (
                "self_target_directional_displacement"
                if self_target_push else "blocker_clearance_gain"
            ),
            "signed_push_direction_displacement_m": signed_displacement,
            "minimum_displacement_m": minimum_displacement_m,
            "measured_target_clearance_gain_m": clearance_gain,
            "clearance_gain_required": not self_target_push,
            "minimum_clearance_gain_m": (
                None if self_target_push else minimum_clearance_gain_m
            ),
            "protected_tracks_stable": protected_stable,
            "protected_track_displacements_m": protected_motion,
        },
    )


def _xy_distance(first, second) -> float:
    return math.hypot(
        first.center_xyz_m[0] - second.center_xyz_m[0],
        first.center_xyz_m[1] - second.center_xyz_m[1],
    )
