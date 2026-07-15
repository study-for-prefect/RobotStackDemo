"""Task-specific state for the six-role house."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..common.config import StackDemoConfig
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
    left_right_spacing: float
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
    candidates = _role_candidates(scene, config)
    bindings = dict(previous.role_bindings if previous else {})
    for role, track_id in list(bindings.items()):
        if track_id not in scene.visible_tracks:
            bindings.pop(role)
    for obj in scene.current_objects:
        role = str(obj.source.get("house_role") or "")
        if role in HOUSE_ROLES:
            bindings[role] = obj.track_id
    for result in scene.recent_action_results:
        role = str(result.get("task_role") or "")
        track = str(result.get("acted_object_track_id") or "")
        if role in HOUSE_ROLES and track and bool(result.get("success")):
            bindings[role] = track
    orientation_states: dict[str, Mapping[str, Any]] = {}
    for obj in scene.current_objects:
        if obj.shape in set(config.section("house")["roof_classes"]):
            orientation_states[obj.track_id] = roof_orientation_from_object(obj, config).to_dict()
        elif obj.shape in set(config.section("house")["triangle_classes"]):
            orientation_states[obj.track_id] = triangle_orientation_from_object(obj).to_dict()
    completion: dict[str, bool] = {}
    for role in HOUSE_ROLES:
        geometry_valid, _ = role_observation_checks(scene, bindings, role, config)
        completion[role] = bool(
            geometry_valid
            and all(completion.get(dependency, False) for dependency in ROLE_DEPENDENCIES[role])
        )
    protected = tuple(sorted(bindings[role] for role in HOUSE_ROLES if completion[role]))
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
        left_right_spacing=float(config.section("house")["left_right_spacing_m"]),
        protected_structure_tracks=protected,
        orientation_states=orientation_states,
        current_repair_state=_repair_state(completion, bindings),
        recent_failures=tuple(dict(item) for item in scene.recent_action_results if not item.get("success")),
        missing_expected_tracks=tuple(sorted(set(expected) - set(scene.visible_tracks))),
    )


def _role_candidates(scene: ClutterSceneState, config: StackDemoConfig) -> dict[str, tuple[str, ...]]:
    house = config.section("house")
    squares = tuple(obj.track_id for obj in scene.current_objects if obj.shape in set(house["support_classes"]) and not obj.protected)
    roofs = tuple(obj.track_id for obj in scene.current_objects if obj.shape in set(house["roof_classes"]) and not obj.protected)
    triangles = tuple(obj.track_id for obj in scene.current_objects if obj.shape in set(house["triangle_classes"]) and not obj.protected)
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
