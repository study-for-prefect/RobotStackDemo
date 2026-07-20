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
    role_status: Mapping[str, str]
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
    verified_occluded_tracks: tuple[str, ...]
    geometry_rebound_tracks: tuple[str, ...]
    inferred_hidden_tracks: tuple[str, ...]
    unresolved_missing_tracks: tuple[str, ...]

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
    continuation_preserves_structure = bool(
        previous
        and previous.role_bindings.get("roof") in remembered_tracks
        and any(
            result.get("task_role") == "triangle_top"
            and result.get("continuation_preserve_verified_roof") is True
            and result.get("explicit_structure_invalidation") is not True
            for result in scene.recent_action_results
        )
    )
    for role, track_id in list(bindings.items()):
        paired_upper = role.replace("lower", "upper") if role.endswith("_lower") else ""
        verified_hidden_lower = bool(
            previous
            and track_id in previous.inferred_hidden_tracks
            and previous.role_completion.get(role)
            and previous.role_bindings.get(paired_upper) in remembered_tracks
        )
        previously_verified_and_remembered = bool(
            previous
            and previous.role_completion.get(role)
            and track_id in remembered_tracks
        )
        preserved_hidden_under_verified_roof = bool(
            continuation_preserves_structure
            and previous
            and previous.role_completion.get(role)
            and track_id in previous.inferred_hidden_tracks
        )
        if (
            track_id not in scene.visible_tracks
            and not previously_verified_and_remembered
            and not verified_hidden_lower
            and not preserved_hidden_under_verified_roof
        ):
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
    inferred_hidden_supports = {
        track_id for role, track_id in bindings.items()
        if role.endswith("_lower")
        and previous
        and track_id in previous.inferred_hidden_tracks
    }
    if continuation_preserves_structure and previous:
        inferred_hidden_supports.update(previous.inferred_hidden_tracks)
    inferred_hidden_supports.update(_restore_merged_column_bindings(
        scene, config, bindings, bind_role,
    ))
    inferred_hidden_supports.update(_restore_stacked_support_bindings(
        scene, config, bindings, bind_role,
    ))
    elevated_roof = _elevated_assembled_roof(scene, config)
    if elevated_roof is not None:
        inferred_hidden_supports.update(_restore_elevated_roof_bindings(
            elevated_roof, bindings, bind_role,
        ))
    _restore_visible_geometric_bindings(scene, config, bindings, bind_role)
    orientation_states: dict[str, Mapping[str, Any]] = {}
    for obj in scene.current_objects:
        if obj.shape in set(config.section("house")["roof_classes"]):
            orientation_states[obj.track_id] = roof_orientation_from_object(obj, config).to_dict()
        elif obj.shape in set(config.section("house")["triangle_classes"]):
            orientation_states[obj.track_id] = triangle_orientation_from_object(obj).to_dict()
    completion: dict[str, bool] = {}
    role_status: dict[str, str] = {}
    verified_occluded_tracks: set[str] = set()
    roof_preservation_results = tuple(
        result for result in scene.recent_action_results
        if result.get("task_role") == "triangle_top"
        and result.get("continuation_preserve_verified_roof") is True
    )
    for role in HOUSE_ROLES:
        geometry_valid, geometry_checks = role_observation_checks(
            scene, bindings, role, config,
        )
        track_id = bindings.get(role, "")
        verified_occluded = bool(
            previous
            and previous.role_completion.get(role)
            and previous.role_bindings.get(role) == track_id
            and track_id in remembered_tracks
            and track_id not in scene.visible_tracks
        )
        if verified_occluded:
            verified_occluded_tracks.add(track_id)
        stable_visible = bool(
            previous
            and previous.role_completion.get(role)
            and previous.role_bindings.get(role) == track_id
            and track_id in scene.visible_tracks
            and _stable_visible_role(role, geometry_checks, config)
        )
        verified_roof_under_continued_triangle = bool(
            role == "roof"
            and previous
            and previous.role_completion.get("roof")
            and previous.role_bindings.get("roof") == track_id
            and track_id in remembered_tracks.union(scene.visible_tracks)
            and roof_preservation_results
            and not any(
                result.get("explicit_structure_invalidation") is True
                for result in roof_preservation_results
            )
        )
        inferred_stacked_upper = bool(
            role.endswith("_upper")
            and bindings.get(role.replace("upper", "lower")) in inferred_hidden_supports
            and track_id in scene.visible_tracks
        )
        elevated_roof_verified = bool(
            role == "roof"
            and elevated_roof is not None
            and track_id == elevated_roof.track_id
            and all(
                bindings.get(support_role) in inferred_hidden_supports
                for support_role in (
                    "left_support_lower", "left_support_upper",
                    "right_support_lower", "right_support_upper",
                )
            )
        )
        completion[role] = bool(
            (
                geometry_valid or verified_occluded or stable_visible or inferred_stacked_upper
                or verified_roof_under_continued_triangle
                or elevated_roof_verified
                or track_id in inferred_hidden_supports
            )
            and all(completion.get(dependency, False) for dependency in ROLE_DEPENDENCIES[role])
        )
        explicitly_invalidated = bool(
            track_id and any(
                result.get("task_role") == role
                and result.get("explicit_structure_invalidation") is True
                for result in scene.recent_action_results
            )
        )
        if completion[role]:
            role_status[role] = "COMPLETED_OCCLUDED" if (
                verified_occluded or track_id in inferred_hidden_supports
            ) else "COMPLETED_VISIBLE"
        elif explicitly_invalidated:
            role_status[role] = "INVALIDATED"
        elif track_id:
            role_status[role] = "REPAIRABLE"
        else:
            role_status[role] = "UNPLACED"
    # A repairable role must remain movable.  Protecting it here removes the
    # bound track from role candidates and the planner then rejects the repair
    # for moving a protected object, producing a permanent zero-edge loop.
    protected = tuple(sorted({bindings[role] for role in HOUSE_ROLES
                              if role_status[role] in {"COMPLETED_VISIBLE", "COMPLETED_OCCLUDED"}}))
    geometry_rebound_tracks = {
        previous.role_bindings[role]
        for role in HOUSE_ROLES
        if previous and previous.role_bindings.get(role)
        and previous.role_bindings.get(role) != bindings.get(role)
        and bindings.get(role) in scene.visible_tracks
    }
    inferred_hidden_tracks = set(inferred_hidden_supports)
    unresolved_missing = (
        set(expected) - set(scene.visible_tracks) - verified_occluded_tracks
        - geometry_rebound_tracks - inferred_hidden_tracks
    )
    candidates = _role_candidates(scene, config, excluded_tracks=protected)
    return HouseTaskState(
        scene_revision=scene.scene_revision,
        expected_tracks=expected,
        role_requirements=_role_requirements(config),
        role_bindings=bindings,
        role_completion=completion,
        role_status=role_status,
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
        verified_occluded_tracks=tuple(sorted(verified_occluded_tracks)),
        geometry_rebound_tracks=tuple(sorted(geometry_rebound_tracks)),
        inferred_hidden_tracks=tuple(sorted(inferred_hidden_tracks)),
        unresolved_missing_tracks=tuple(sorted(unresolved_missing)),
    )


def _stable_visible_role(
    role: str,
    checks: Mapping[str, Any],
    config: StackDemoConfig,
) -> bool:
    """Apply tight temporal hysteresis to an already verified visible roof."""
    if role == "triangle_top":
        return bool(
            checks.get("role_object_visible") is True
            and checks.get("roof_visible") is True
            and checks.get("base_contact_valid") is True
            and checks.get("roof_center_offset_valid") is True
            and checks.get("long_edge_matches_roof_axis") is True
            and checks.get("apex_up") is True
            and checks.get("base_edge_down") is True
            and checks.get("face_state_valid") is True
        )
    if role != "roof":
        return False
    # No action has touched an already verified roof between these live
    # observations. Preserve its contact/face fact across noisy depth/PCA
    # frames, but still require the independent span, coverage and axis facts.
    required = (
        "both_upper_supports_visible", "long_axis_matches_support_span",
        "support_spacing_valid", "support_height_difference_valid",
        "covers_left_support", "covers_right_support",
        "roof_width_covers_supports",
    )
    if not all(checks.get(key) is True for key in required):
        return False
    return float(checks.get("roof_center_offset_m", math.inf)) <= (
        2.0 * float(config.section("house")["center_tolerance_m"])
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
            and footprint_max / footprint_min <= 1.45
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
            existing_upper = bindings.get(upper_role)
            # A fresh process may geometrically recover the visible top block
            # before recovering its occluded lower support.  That existing
            # same-track upper binding is evidence to complete, not a reason
            # to skip the merged-column recovery.
            if lower_role in bindings or (
                existing_upper is not None and existing_upper != obj.track_id
            ):
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
            if existing_upper is None:
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


def _restore_elevated_roof_bindings(
    roof: Any,
    bindings: dict[str, str],
    bind_role,
) -> set[str]:
    """Restore the fully occluded two-column structure below a measured roof."""
    inferred: set[str] = set()
    for role in (
        "left_support_lower", "left_support_upper",
        "right_support_lower", "right_support_upper",
    ):
        if role in bindings:
            continue
        track_id = f"inferred_hidden_{role}_under_{roof.track_id}"
        bind_role(role, track_id)
        inferred.add(track_id)
    if "roof" not in bindings:
        bind_role("roof", roof.track_id)
    return inferred


def _elevated_assembled_roof(
    scene: ClutterSceneState,
    config: StackDemoConfig,
):
    """Return a current-frame roof at house XY and 60-67 mm above its support.

    The independent top cluster and local support ring provide the two Z
    measurements.  No prior action result or theoretical layer count is used.
    """
    house = config.section("house")
    height_min = float(house["assembled_roof_height_above_table_min_m"])
    height_max = float(house["assembled_roof_height_above_table_max_m"])
    origin_x, origin_y = (float(value) for value in house["origin_center_base_m"][:2])
    matches = []
    for obj in scene.current_objects:
        if obj.shape not in set(house["roof_classes"]):
            continue
        source = obj.source
        top_z = source.get("top_z_base_m")
        support = source.get("local_support_surface")
        support_z = support.get("support_z_base_m") if isinstance(support, Mapping) else None
        if top_z is None or support_z is None:
            continue
        total_height = float(top_z) - float(support_z)
        center_offset = math.hypot(
            obj.center_xyz_m[0] - origin_x,
            obj.center_xyz_m[1] - origin_y,
        )
        if not (
            height_min <= total_height <= height_max
            and center_offset <= 2.0 * float(house["center_tolerance_m"])
        ):
            continue
        midpoint = 0.5 * (height_min + height_max)
        matches.append((center_offset, abs(total_height - midpoint), obj.track_id, obj))
    return min(matches, key=lambda item: item[:3])[-1] if matches else None


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
    if "roof" in bindings and "triangle_top" in bindings:
        relations.append({"support": bindings["roof"], "supported": bindings["triangle_top"], "verified": True})
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
