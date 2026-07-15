"""Multi-evidence verification using fresh post-action observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping

from .action_edges import PhysicalActionEdge
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
    original_position_clear = track_missing or moved
    left_table_workspace = track_missing or (
        current.center_xyz_m[2] > original.center_xyz_m[2] + max(0.5 * original.size_xyz_m[2], minimum_displacement_m)
    )
    other_before = set(before.visible_tracks) - {edge.acted_object_track_id}
    context_tracks_still_visible = tuple(sorted(other_before.intersection(after_lift.visible_tracks)))
    view_continuity = bool(context_tracks_still_visible)
    scene_change_supports_grasp = bool(
        moved
        or (track_missing and (view_continuity or gripper_holding_hint is True))
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
            "original_position_clear": original_position_clear,
            "target_left_table_workspace": left_table_workspace,
            "scene_change_supports_grasp": scene_change_supports_grasp,
            "context_tracks_still_visible": list(context_tracks_still_visible),
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
    after_place: ClutterSceneState,
    destination_check: DestinationCheck,
) -> ActionVerification:
    destination_ok, details = destination_check(edge, after_place)
    visible = after_place.object_by_track(edge.acted_object_track_id) is not None
    success = bool(destination_ok and visible)
    return ActionVerification(
        success=success,
        status="place_verified" if success else "place_failed",
        scene_revision=after_place.scene_revision,
        evidence={"track_visible": visible, "destination_predicate": bool(destination_ok), **dict(details)},
    )


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
    required_visible = all(item is not None for item in (blocker_before, blocker_after, target_before, target_after))
    signed_displacement = None
    clearance_gain = None
    if required_visible and len(direction) >= 2:
        dx = blocker_after.center_xyz_m[0] - blocker_before.center_xyz_m[0]
        dy = blocker_after.center_xyz_m[1] - blocker_before.center_xyz_m[1]
        signed_displacement = dx * float(direction[0]) + dy * float(direction[1])
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
        and clearance_gain is not None and clearance_gain >= minimum_clearance_gain_m
        and protected_stable
    )
    return ActionVerification(
        success=success,
        status="nudge_verified" if success else "nudge_failed",
        scene_revision=after.scene_revision,
        evidence={
            "required_tracks_visible": required_visible,
            "signed_push_direction_displacement_m": signed_displacement,
            "minimum_displacement_m": minimum_displacement_m,
            "measured_target_clearance_gain_m": clearance_gain,
            "minimum_clearance_gain_m": minimum_clearance_gain_m,
            "protected_tracks_stable": protected_stable,
            "protected_track_displacements_m": protected_motion,
        },
    )


def _xy_distance(first, second) -> float:
    return math.hypot(
        first.center_xyz_m[0] - second.center_xyz_m[0],
        first.center_xyz_m[1] - second.center_xyz_m[1],
    )
