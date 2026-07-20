"""Code-owned completion for the latest six-role house observation."""

from __future__ import annotations

from typing import Any

from ..common.scene_state import ClutterSceneState
from ..common.config import StackDemoConfig
from .roles import HOUSE_ROLES
from .state import HouseTaskState


def evaluate_house_completion(
    scene: ClutterSceneState,
    state: HouseTaskState,
    config: StackDemoConfig | None = None,
) -> dict[str, Any]:
    roles_complete = all(bool(state.role_completion.get(role)) for role in HOUSE_ROLES)
    bindings_distinct = len(set(state.role_bindings.values())) == len(HOUSE_ROLES)
    no_missing = not state.unresolved_missing_tracks
    latest = state.scene_revision == scene.scene_revision
    expected_support_edges = {
        ("left_support_lower", "left_support_upper"),
        ("right_support_lower", "right_support_upper"),
        ("left_support_upper", "roof"), ("right_support_upper", "roof"),
        ("roof", "triangle_top"),
    }
    observed_edges = {
        (lower_role, upper_role)
        for lower_role, lower_track in state.role_bindings.items()
        for upper_role, upper_track in state.role_bindings.items()
        if any(
            relation.get("support") == lower_track
            and relation.get("supported") == upper_track
            and relation.get("verified") is True
            for relation in state.support_relations
        )
    }
    support_graph_connected = expected_support_edges.issubset(observed_edges)
    total_height, expected_minimum_height = _structure_height(scene, state, config)
    total_height_valid = bool(
        total_height is not None and expected_minimum_height is not None
        and total_height >= expected_minimum_height
    )
    complete = bool(
        roles_complete and bindings_distinct and no_missing and latest
        and support_graph_connected and total_height_valid
    )
    return {
        "task_type": "build_house",
        "scene_revision": scene.scene_revision,
        "task_complete": complete,
        "role_completion": dict(state.role_completion),
        "role_bindings": dict(state.role_bindings),
        "role_status": dict(state.role_status),
        "six_distinct_role_tracks": bindings_distinct,
        "missing_expected_tracks": list(state.missing_expected_tracks),
        "unresolved_missing_tracks": list(state.unresolved_missing_tracks),
        "verified_occluded_tracks": list(state.verified_occluded_tracks),
        "geometry_rebound_tracks": list(state.geometry_rebound_tracks),
        "inferred_hidden_tracks": list(state.inferred_hidden_tracks),
        "support_graph_connected": support_graph_connected,
        "support_graph_edges": [list(edge) for edge in sorted(observed_edges)],
        "structure_total_height_m": total_height,
        "minimum_required_total_height_m": expected_minimum_height,
        "structure_total_height_valid": total_height_valid,
        "protected_structure_tracks": list(state.protected_structure_tracks),
        "latest_observation_confirmed": latest,
        "no_feasible_edge_is_completion_evidence": False,
    }


def _structure_height(
    scene: ClutterSceneState,
    state: HouseTaskState,
    config: StackDemoConfig | None = None,
) -> tuple[float | None, float | None]:
    objects = {
        role: scene.object_by_track(track) or next(
            (item for item in scene.collision_obstacles if item.track_id == track), None,
        )
        for role, track in state.role_bindings.items()
    }
    visible = [item for item in objects.values() if item is not None]
    if not visible:
        return None, None
    observed = max(item.center_xyz_m[2] + 0.5 * item.size_xyz_m[2] for item in visible) - min(
        item.center_xyz_m[2] - 0.5 * item.size_xyz_m[2] for item in visible
    )
    merged_column_layer_sizes: dict[str, float] = {}
    for side in ("left", "right"):
        lower_role = f"{side}_support_lower"
        upper_role = f"{side}_support_upper"
        upper = objects.get(upper_role)
        if (
            upper is not None
            and state.role_bindings.get(lower_role) in state.inferred_hidden_tracks
        ):
            merged_column_layer_sizes[lower_role] = 0.5 * float(upper.size_xyz_m[2])
            merged_column_layer_sizes[upper_role] = 0.5 * float(upper.size_xyz_m[2])
    layer_sizes = []
    role_layers = (("left_support_lower", "right_support_lower"),
                   ("left_support_upper", "right_support_upper"),
                   ("roof",), ("triangle_top",))
    for roles in role_layers:
        values = []
        for role in roles:
            if role in merged_column_layer_sizes:
                values.append(merged_column_layer_sizes[role])
            elif (
                config is not None
                and state.role_bindings.get(role) in state.inferred_hidden_tracks
                and role.endswith(("_lower", "_upper"))
            ):
                values.append(float(config.section("special_shape_vlm")["square_edge_length_m"]))
            elif role == "roof" and config is not None:
                values.append(float(config.section("house")["roof_nominal_thickness_m"]))
            elif objects.get(role) is not None:
                values.append(float(objects[role].size_xyz_m[2]))
        if not values:
            return observed, None
        layer_sizes.append(sum(values) / len(values))
    return observed, max(0.0, sum(layer_sizes) - 0.020)
