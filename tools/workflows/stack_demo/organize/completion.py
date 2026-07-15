"""Latest-observation completion predicates for organization."""

from __future__ import annotations

from typing import Any, Mapping

from ..common.scene_state import ClutterSceneState
from .state import OrganizeTaskState


def evaluate_organize_completion(
    scene: ClutterSceneState,
    state: OrganizeTaskState,
) -> dict[str, Any]:
    expected_resolved = set(state.completed_tracks) == set(state.expected_tracks)
    no_missing = not state.missing_expected_tracks
    no_unresolved = not state.unresolved_tracks
    overlap_pairs = _illegal_overlap_pairs(scene, state)
    latest = scene.scene_revision == state.scene_revision
    complete = bool(expected_resolved and no_missing and no_unresolved and not overlap_pairs and latest)
    return {
        "task_type": "organize_blocks",
        "scene_revision": scene.scene_revision,
        "task_complete": complete,
        "expected_tracks_all_resolved": expected_resolved,
        "missing_expected_tracks": list(state.missing_expected_tracks),
        "unresolved_tracks": list(state.unresolved_tracks),
        "completed_tracks": list(state.completed_tracks),
        "illegal_overlap_pairs": overlap_pairs,
        "latest_observation_confirmed": latest,
        "no_feasible_edge_is_completion_evidence": False,
    }


def _illegal_overlap_pairs(
    scene: ClutterSceneState,
    state: OrganizeTaskState,
) -> list[list[str]]:
    completed = [scene.object_by_track(track_id) for track_id in state.completed_tracks]
    completed = [obj for obj in completed if obj is not None]
    pairs = []
    for index, first in enumerate(completed):
        for second in completed[index + 1:]:
            dx = abs(first.center_xyz_m[0] - second.center_xyz_m[0])
            dy = abs(first.center_xyz_m[1] - second.center_xyz_m[1])
            if dx < 0.5 * (first.size_xyz_m[0] + second.size_xyz_m[0]) and dy < 0.5 * (first.size_xyz_m[1] + second.size_xyz_m[1]):
                pairs.append([first.track_id, second.track_id])
    return pairs
