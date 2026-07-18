"""Task-specific state for the six-role house."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from ..common.config import StackDemoConfig
from ..common.action_edges import ActionType
from ..common.scene_state import ClutterSceneState
from .orientation import roof_orientation_from_object, triangle_orientation_from_object
from .roles import HOUSE_ROLES, ROLE_DEPENDENCIES
from .structure import role_observation_checks


@dataclass(frozen=True)
class HouseTaskState:
    scene_revision: int
    expected_tracks: tuple[str, ...]
    role_requirements: Mapping[str, Mapping[str, Any]]
    role_bindings: Mapping[str, str]
    role_completion: Mapping[str, bool]
    role_candidate_tracks: Mapping[str, tuple[str, ...]]
    support_relations: tuple[Mapping[str, Any], ...]
    center_offsets: Mapping[str, float]
    layer_heights: Mapping[str, float]
    support_inner_gap: float
    protected_structure_tracks: tuple[str, ...]
    orientation_states: Mapping[str, Mapping[str, Any]]
    current_repair_state: Mapping[str, Any] | None
    recent_failures: tuple[Mapping[str, Any], ...]
    missing_expected_tracks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_house_task_state(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    *,
    previous: HouseTaskState | None = None,
) -> HouseTaskState:
    """Rebuild bindings/protection from the latest observation and verified role facts."""
    expected = tuple(previous.expected_tracks if previous else scene.expected_tracks)
    bindings = dict(previous.role_bindings if previous else {})
    remembered_tracks = {obj.track_id for obj in scene.collision_obstacles}
    for role, track_id in list(bindings.items()):
        previously_verified_and_remembered = bool(
            previous
            and previous.role_completion.get(role)
            and track_id in remembered_tracks
        )
        if track_id not in scene.visible_tracks and not previously_verified_and_remembered:
            bindings.pop(role)
    def bind_role(role: str, track_id: str) -> None:
        # One physical object cannot occupy two house roles.  In particular, a
        # verified repair may move a previously bound object to another role.
        for bound_role, bound_track in list(bindings.items()):
            if bound_track == track_id and bound_role != role:
                bindings.pop(bound_role)
        bindings[role] = track_id

    for obj in scene.current_objects:
        role = str(obj.source.get("house_role") or "")
        if role in HOUSE_ROLES:
            bind_role(role, obj.track_id)
    role_binding_actions = {
        ActionType.PLACE_HOUSE_ROLE.value,
        ActionType.REPAIR_STRUCTURE.value,
    }
    for result in scene.recent_action_results:
        role = str(result.get("task_role") or "")
        track = str(result.get("acted_object_track_id") or "")
        action_type = str(result.get("action_type") or "")
        if (
            role in HOUSE_ROLES
            and track
            and bool(result.get("success"))
            and action_type in role_binding_actions
        ):
            bind_role(role, track)
    inferred_hidden_supports = _restore_merged_column_bindings(
        scene, config, bindings, bind_role,
    )
    inferred_hidden_supports.update(_restore_stacked_support_bindings(
        scene, config, bindings, bind_role,
    ))
    _restore_visible_geometric_bindings(scene, config, bindings, bind_role)
    orientation_states: dict[str, Mapping[str, Any]] = {}
    for obj in scene.current_objects:
        if obj.shape in set(config.section("house")["roof_classes"]):
            orientation_states[obj.track_id] = roof_orientation_from_object(obj, config).to_dict()
        elif obj.shape in set(config.section("house")["triangle_classes"]):
            orientation_states[obj.track_id] = triangle_orientation_from_object(obj).to_dict()
    completion: dict[str, bool] = {}
    for role in HOUSE_ROLES:
        geometry_valid, _ = role_observation_checks(scene, bindings, role, config)
        track_id = bindings.get(role, "")
        verified_occluded = bool(
            previous
            and previous.role_completion.get(role)
            and previous.role_bindings.get(role) == track_id
            and track_id in remembered_tracks
            and track_id not in scene.visible_tracks
        )
        inferred_stacked_upper = bool(
            role.endswith("_upper")
            and bindings.get(role.replace("upper", "lower")) in inferred_hidden_supports
            and track_id in scene.visible_tracks
        )
        completion[role] = bool(
            (
                geometry_valid or verified_occluded or inferred_stacked_upper
                or track_id in inferred_hidden_supports
            )
            and all(completion.get(dependency, False) for dependency in ROLE_DEPENDENCIES[role])
        )
    protected = tuple(sorted(bindings[role] for role in HOUSE_ROLES if completion[role]))
    candidates = _role_candidates(scene, config, excluded_tracks=protected)
    return HouseTaskState(
        scene_revision=scene.scene_revision,
        expected_tracks=expected,
        role_requirements=_role_requirements(config),
        role_bindings=bindings,
        role_completion=completion,
        role_candidate_tracks=candidates,
        support_relations=_support_relations(scene, bindings),
        center_offsets=_center_offsets(scene, bindings),
        layer_heights=_layer_heights(scene, bindings),
        support_inner_gap=float(config.section("house")["support_inner_gap_m"]),
        protected_structure_tracks=protected,
        orientation_states=orientation_states,
        current_repair_state=_repair_state(completion, bindings),
        recent_failures=tuple(dict(item) for item in scene.recent_action_results if not item.get("success")),
        missing_expected_tracks=tuple(sorted(set(expected) - set(scene.visible_tracks))),
    )


def _restore_merged_column_bindings(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    bindings: dict[str, str],
    bind_role,
) -> set[str]:
    """Recover both roles when segmentation merges two touching cubes."""
    house = config.section("house")
    inferred: set[str] = set()
    for obj in scene.current_objects:
        if obj.shape not in set(house["support_classes"]):
            continue
        x_size, y_size, height = (float(value) for value in obj.size_xyz_m)
        footprint_min = min(x_size, y_size)
        footprint_max = max(x_size, y_size)
        if not (
            footprint_min >= 0.015
            and footprint_max / footprint_min <= 1.35
            and 1.65 * footprint_min <= height <= 2.35 * footprint_max
        ):
            continue
        support = obj.source.get("local_support_surface")
        support_z = support.get("support_z_base_m") if isinstance(support, Mapping) else None
        if support_z is None or abs(
            obj.center_xyz_m[2] - 0.5 * height - float(support_z)
        ) > float(house["support_height_tolerance_m"]):
            continue
        for side in ("left", "right"):
            lower_role = f"{side}_support_lower"
            upper_role = f"{side}_support_upper"
            if lower_role in bindings or upper_role in bindings:
                continue
            # Perspective/merged masks often stretch one footprint axis; the
            # normal placement predicate likewise uses the observed x extent.
            cube_width = footprint_max
            expected_x = float(house["origin_center_base_m"][0]) + (
                (-0.5 if side == "left" else 0.5)
                * (cube_width + float(house["support_inner_gap_m"]))
            )
            offset = math.hypot(
                obj.center_xyz_m[0] - expected_x,
                obj.center_xyz_m[1] - float(house["origin_center_base_m"][1]),
            )
            if offset > float(house["center_tolerance_m"]):
                continue
            hidden_track = f"inferred_hidden_{side}_support_below_{obj.track_id}"
            bindings[lower_role] = hidden_track
            bind_role(upper_role, obj.track_id)
            inferred.add(hidden_track)
            break
    return inferred


def _restore_stacked_support_bindings(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    bindings: dict[str, str],
    bind_role,
) -> set[str]:
    """Recover a hidden lower support from a visibly supported upper cube."""
    house = config.section("house")
    if not scene.current_objects:
        return set()
    table_z = min(
        obj.center_xyz_m[2] - 0.5 * obj.size_xyz_m[2]
        for obj in scene.current_objects
    )
    inferred: set[str] = set()
    for obj in scene.current_objects:
        support = obj.source.get("local_support_surface")
        support_z = support.get("support_z_base_m") if isinstance(support, Mapping) else None
        if support_z is None or not _cube_like_support(obj):
            continue
        support_z = float(support_z)
        if support_z - table_z < 0.012:
            continue
        for side in ("left", "right"):
            lower_role = f"{side}_support_lower"
            upper_role = f"{side}_support_upper"
            if lower_role in bindings or upper_role in bindings:
                continue
            half_spacing = 0.5 * (
                float(obj.size_xyz_m[0]) + float(house["support_inner_gap_m"])
            )
            expected_x = float(house["origin_center_base_m"][0]) + (
                -half_spacing if side == "left" else half_spacing
            )
            offset = math.hypot(
                obj.center_xyz_m[0] - expected_x,
                obj.center_xyz_m[1] - float(house["origin_center_base_m"][1]),
            )
            bottom_gap = abs(
                obj.center_xyz_m[2] - 0.5 * obj.size_xyz_m[2] - support_z
            )
            if (
                offset > float(house["center_tolerance_m"])
                or bottom_gap > float(house["support_height_tolerance_m"])
            ):
                continue
            hidden_track = f"inferred_hidden_{side}_support_below_{obj.track_id}"
            bindings[lower_role] = hidden_track
            bind_role(upper_role, obj.track_id)
            inferred.add(hidden_track)
            break
    return inferred


def _restore_visible_geometric_bindings(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    bindings: dict[str, str],
    bind_role,
) -> None:
    """Recover completed roles from a fresh run's current observation.

    Output directories do not carry task history across runs.  A visible,
    grounded object already satisfying a code-owned role must therefore be
    rebound from geometry before candidate generation, or it can be picked up
    again as an unfinished block.
    """
    house = config.section("house")
    classes_by_role = {
        role: set(
            house["roof_classes"] if role == "roof"
            else house["triangle_classes"] if role == "triangle_top"
            else house["support_classes"]
        )
        for role in HOUSE_ROLES
    }
    used_tracks = set(bindings.values())
    for role in HOUSE_ROLES:
        if role in bindings:
            continue
        matches = []
        for obj in scene.current_objects:
            semantic_match = obj.shape in classes_by_role[role]
            # A fresh process has no action anchor with which to recover a
            # detector label flip.  For an already-settled support only, exact
            # role geometry plus cube-like metric dimensions are sufficient
            # fallback evidence.  This does not make such objects available as
            # ordinary support candidates; it only restores structure already
            # occupying a code-owned house pose.
            metric_support_fallback = bool(
                role.endswith(("_lower", "_upper"))
                and _cube_like_support(obj)
            )
            if (
                obj.track_id in used_tracks
                or not (semantic_match or metric_support_fallback)
            ):
                continue
            valid, checks = role_observation_checks(
                scene, bindings, role, config, role_object=obj,
            )
            if valid:
                score = sum(
                    abs(float(value))
                    for key, value in checks.items()
                    if key.endswith(("_offset_m", "_gap_m"))
                    and isinstance(value, (int, float))
                )
                matches.append((score, obj.track_id))
        if matches:
            _, track_id = min(matches, key=lambda item: (item[0], item[1]))
            bind_role(role, track_id)
            used_tracks.add(track_id)


def _cube_like_support(obj) -> bool:
    dimensions = [float(value) for value in obj.size_xyz_m]
    smallest = min(dimensions)
    return bool(
        smallest >= 0.015
        and max(dimensions) / smallest <= 1.35
        and max(dimensions) <= 0.040
    )


def _role_candidates(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    *,
    excluded_tracks: tuple[str, ...] = (),
) -> dict[str, tuple[str, ...]]:
    house = config.section("house")
    excluded = set(excluded_tracks)
    squares = tuple(obj.track_id for obj in scene.current_objects if obj.shape in set(house["support_classes"]) and not obj.protected and obj.track_id not in excluded)
    roofs = tuple(obj.track_id for obj in scene.current_objects if obj.shape in set(house["roof_classes"]) and not obj.protected and obj.track_id not in excluded)
    triangles = tuple(obj.track_id for obj in scene.current_objects if obj.shape in set(house["triangle_classes"]) and not obj.protected and obj.track_id not in excluded)
    return {
        "left_support_lower": squares, "right_support_lower": squares,
        "left_support_upper": squares, "right_support_upper": squares,
        "roof": roofs, "triangle_top": triangles,
    }


def _role_requirements(config: StackDemoConfig) -> dict[str, Mapping[str, Any]]:
    house = config.section("house")
    return {
        role: {
            "dependencies": list(ROLE_DEPENDENCIES[role]),
            "classes": list(
                house["roof_classes"] if role == "roof"
                else house["triangle_classes"] if role == "triangle_top"
                else house["support_classes"]
            ),
        }
        for role in HOUSE_ROLES
    }


def _support_relations(scene: ClutterSceneState, bindings: Mapping[str, str]) -> tuple[Mapping[str, Any], ...]:
    relations = []
    for lower, upper in (("left_support_lower", "left_support_upper"), ("right_support_lower", "right_support_upper")):
        if lower in bindings and upper in bindings:
            relations.append({"support": bindings[lower], "supported": bindings[upper], "verified": True})
    if "roof" in bindings:
        for upper in ("left_support_upper", "right_support_upper"):
            if upper in bindings:
                relations.append({"support": bindings[upper], "supported": bindings["roof"], "verified": True})
    return tuple(relations)


def _center_offsets(scene: ClutterSceneState, bindings: Mapping[str, str]) -> dict[str, float]:
    output = {}
    for lower, upper, name in (("left_support_lower", "left_support_upper", "left_column"), ("right_support_lower", "right_support_upper", "right_column")):
        first, second = scene.object_by_track(bindings.get(lower, "")), scene.object_by_track(bindings.get(upper, ""))
        if first and second:
            output[name] = ((first.center_xyz_m[0] - second.center_xyz_m[0]) ** 2 + (first.center_xyz_m[1] - second.center_xyz_m[1]) ** 2) ** 0.5
    return output


def _layer_heights(scene: ClutterSceneState, bindings: Mapping[str, str]) -> dict[str, float]:
    return {
        role: obj.center_xyz_m[2]
        for role, track in bindings.items()
        if (obj := scene.object_by_track(track)) is not None
    }


def _repair_state(completion: Mapping[str, bool], bindings: Mapping[str, str]) -> Mapping[str, Any] | None:
    unstable = [role for role, track in bindings.items() if not completion.get(role)]
    return None if not unstable else {"repair_required_roles": unstable}
