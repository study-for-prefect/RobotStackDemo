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
    expected = tuple(previous.expected_tracks if previous else scene.expected_tracks)
    colors = dict(previous.track_colors if previous else {})
    colors.update({obj.track_id: obj.color for obj in scene.current_objects})
    regions = (
        dict(previous.color_target_regions)
        if previous else build_color_target_regions(scene, config, colors)
    )
    occupancy = region_occupancy(scene, regions)
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
    )


def _in_correct_region(
    track_id: str,
    colors: Mapping[str, str],
    occupancy: Mapping[str, tuple[str, ...]],
) -> bool:
    color = colors.get(track_id)
    return color is not None and track_id in occupancy.get(color, ())
