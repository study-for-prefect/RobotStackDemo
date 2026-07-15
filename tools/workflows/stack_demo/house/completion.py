"""Code-owned completion for the latest six-role house observation."""

from __future__ import annotations

from typing import Any

from ..common.scene_state import ClutterSceneState
from .roles import HOUSE_ROLES
from .state import HouseTaskState


def evaluate_house_completion(scene: ClutterSceneState, state: HouseTaskState) -> dict[str, Any]:
    roles_complete = all(bool(state.role_completion.get(role)) for role in HOUSE_ROLES)
    bindings_distinct = len(set(state.role_bindings.values())) == len(HOUSE_ROLES)
    no_missing = not state.missing_expected_tracks
    latest = state.scene_revision == scene.scene_revision
    complete = bool(roles_complete and bindings_distinct and no_missing and latest)
    return {
        "task_type": "build_house",
        "scene_revision": scene.scene_revision,
        "task_complete": complete,
        "role_completion": dict(state.role_completion),
        "role_bindings": dict(state.role_bindings),
        "six_distinct_role_tracks": bindings_distinct,
        "missing_expected_tracks": list(state.missing_expected_tracks),
        "protected_structure_tracks": list(state.protected_structure_tracks),
        "latest_observation_confirmed": latest,
        "no_feasible_edge_is_completion_evidence": False,
    }
