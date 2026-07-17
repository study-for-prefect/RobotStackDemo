"""Task-specific state for color organization only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..common.config import StackDemoConfig
from ..common.scene_state import ClutterSceneState
from .placement import build_color_target_regions, build_safe_slots, region_occupancy


@dataclass(frozen=True)
class OrganizeTaskState:
    scene_revision: int
    expected_tracks: tuple[str, ...]
    visible_tracks: tuple[str, ...]
    missing_expected_tracks: tuple[str, ...]
    completed_tracks: tuple[str, ...]
    unresolved_tracks: tuple[str, ...]
    color_target_regions: Mapping[str, Mapping[str, Any]]
    safe_slots_by_color: Mapping[str, tuple[Mapping[str, Any], ...]]
    region_occupancy: Mapping[str, tuple[str, ...]]
    track_colors: Mapping[str, str]
    current_action_result: Mapping[str, Any] | None
    recent_failures: tuple[Mapping[str, Any], ...]
    staging_reservations: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_organize_task_state(
    scene: ClutterSceneState,
    config: StackDemoConfig,
    *,
    previous: OrganizeTaskState | None = None,
    current_action_result: Mapping[str, Any] | None = None,
) -> OrganizeTaskState:
    """Preserve expected tracks and committed regions across detector misses."""
    # The live observer already carries expected identities forward and may
    # intentionally replace a stale ID after one-to-one identity handoff.
    # Unioning the previous state would resurrect that phantom missing ID.
    expected = tuple(sorted(set(scene.expected_tracks)))
    colors = dict(previous.track_colors if previous else {})
    colors.update({obj.track_id: obj.color for obj in scene.current_objects})
    regions = (
        dict(previous.color_target_regions)
        if previous else build_color_target_regions(scene, config, colors)
    )
    occupancy = region_occupancy(
        scene,
        regions,
        float(config.section("organize")["observation_region_tolerance_m"]),
    )
    completed = tuple(sorted(
        track_id for track_id in expected
        if track_id in scene.visible_tracks
        and _in_correct_region(track_id, colors, occupancy)
    ))
    unresolved = tuple(sorted(set(expected) - set(completed)))
    failures = tuple(
        dict(item) for item in scene.recent_action_results
        if not bool(item.get("success"))
    )
    staging_reservations = _staging_reservations(scene.recent_action_results)
    return OrganizeTaskState(
        scene_revision=scene.scene_revision,
        expected_tracks=expected,
        visible_tracks=scene.visible_tracks,
        missing_expected_tracks=tuple(sorted(set(expected) - set(scene.visible_tracks))),
        completed_tracks=completed,
        unresolved_tracks=unresolved,
        color_target_regions=regions,
        safe_slots_by_color=build_safe_slots(scene, config, regions),
        region_occupancy=occupancy,
        track_colors=colors,
        current_action_result=None if current_action_result is None else dict(current_action_result),
        recent_failures=failures,
        staging_reservations=staging_reservations,
    )


def _in_correct_region(
    track_id: str,
    colors: Mapping[str, str],
    occupancy: Mapping[str, tuple[str, ...]],
) -> bool:
    color = colors.get(track_id)
    return color is not None and track_id in occupancy.get(color, ())


def _staging_reservations(
    action_results: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    """Keep the latest released location for tracks left in organize staging."""
    latest_by_track: dict[str, Mapping[str, Any]] = {}
    for item in action_results:
        track_id = str(item.get("acted_object_track_id") or "")
        if track_id:
            latest_by_track[track_id] = item
    output = []
    for track_id, item in latest_by_track.items():
        if item.get("target_region_id") != "organize_staging" or not item.get("release_executed"):
            continue
        position = item.get("planned_place_position_m")
        size = item.get("acted_object_size_m")
        if not (
            isinstance(position, (list, tuple)) and len(position) >= 3
            and isinstance(size, (list, tuple)) and len(size) >= 3
        ):
            continue
        output.append({
            "track_id": track_id,
            "position_m": [float(value) for value in position[:3]],
            "size_m": [float(value) for value in size[:3]],
        })
    return tuple(sorted(output, key=lambda item: str(item["track_id"])))
