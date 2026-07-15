"""Current-frame clutter state and stable expected-track accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping, Sequence

from robot_scene_pipeline.object_semantics import infer_object_color, infer_object_shape


SIDE_VECTORS = {
    "+x": (1.0, 0.0),
    "-x": (-1.0, 0.0),
    "+y": (0.0, 1.0),
    "-y": (0.0, -1.0),
}


@dataclass(frozen=True)
class SceneObjectState:
    object_ref: str
    detector_id: str
    track_id: str
    class_name: str
    color: str
    shape: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    yaw_deg: float
    orientation_confidence: float
    neighbors: tuple[str, ...] = ()
    edge_clearances_m: Mapping[str, float] = field(default_factory=dict)
    blocked_sides: tuple[str, ...] = ()
    free_sides: tuple[str, ...] = ()
    already_completed: bool = False
    protected: bool = False
    currently_visible: bool = True
    source: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source", None)
        return value


@dataclass(frozen=True)
class ClutterSceneState:
    scene_revision: int
    coordinate_frame: str
    observation_timestamp: str
    current_objects: tuple[SceneObjectState, ...]
    expected_tracks: tuple[str, ...]
    visible_tracks: tuple[str, ...]
    missing_expected_tracks: tuple[str, ...]
    completed_tracks: tuple[str, ...]
    protected_tracks: tuple[str, ...]
    free_regions: tuple[Mapping[str, Any], ...]
    occupied_regions: tuple[Mapping[str, Any], ...]
    target_regions: tuple[Mapping[str, Any], ...]
    recent_action_results: tuple[Mapping[str, Any], ...]
    forbidden_action_fingerprints: tuple[str, ...]

    def object_by_track(self, track_id: str) -> SceneObjectState | None:
        return next((obj for obj in self.current_objects if obj.track_id == track_id), None)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["current_objects"] = [obj.to_dict() for obj in self.current_objects]
        return value


def build_clutter_scene_state(
    observation: Mapping[str, Any],
    workspace: Mapping[str, float],
    *,
    expected_tracks: Iterable[str] | None = None,
    completed_tracks: Iterable[str] = (),
    protected_tracks: Iterable[str] = (),
    target_regions: Sequence[Mapping[str, Any]] = (),
    free_regions: Sequence[Mapping[str, Any]] = (),
    recent_action_results: Sequence[Mapping[str, Any]] = (),
    forbidden_action_fingerprints: Iterable[str] = (),
) -> ClutterSceneState:
    """Build an explicit base_link state without treating detector ids as identity."""
    frame = str(observation.get("frame_id") or observation.get("geometry_frame") or "base_link")
    if frame != "base_link":
        raise ValueError(f"scene coordinate frame must be base_link, got {frame}")
    revision = int(observation.get("scene_revision", 1))
    completed = frozenset(str(value) for value in completed_tracks)
    protected = frozenset(str(value) for value in protected_tracks)
    raw_objects = [item for item in observation.get("objects", []) if isinstance(item, dict)]
    detector_ids = [str(item.get("id", item.get("detector_id", ""))) for item in raw_objects]
    if len(detector_ids) != len(set(detector_ids)):
        raise ValueError("detector ids must be unique inside one scene_revision")
    base_objects = [_base_object(item, revision, completed, protected) for item in raw_objects]
    objects = tuple(_with_relations(obj, base_objects, workspace) for obj in base_objects)
    visible = tuple(sorted({obj.track_id for obj in objects}))
    expected = tuple(sorted(set(str(value) for value in (expected_tracks or visible))))
    missing = tuple(sorted(set(expected) - set(visible)))
    occupied = tuple(_occupied_region(obj) for obj in objects)
    timestamp = str(
        observation.get("observation_timestamp")
        or observation.get("timestamp")
        or datetime.now(timezone.utc).isoformat()
    )
    return ClutterSceneState(
        scene_revision=revision,
        coordinate_frame="base_link",
        observation_timestamp=timestamp,
        current_objects=objects,
        expected_tracks=expected,
        visible_tracks=visible,
        missing_expected_tracks=missing,
        completed_tracks=tuple(sorted(completed)),
        protected_tracks=tuple(sorted(protected)),
        free_regions=tuple(dict(item) for item in free_regions),
        occupied_regions=occupied,
        target_regions=tuple(dict(item) for item in target_regions),
        recent_action_results=tuple(dict(item) for item in recent_action_results),
        forbidden_action_fingerprints=tuple(sorted(set(forbidden_action_fingerprints))),
    )


def _base_object(
    raw: Mapping[str, Any],
    revision: int,
    completed: frozenset[str],
    protected: frozenset[str],
) -> SceneObjectState:
    detector_id = str(raw.get("id", raw.get("detector_id", "")))
    if not detector_id:
        raise ValueError("every visible object needs a current-frame detector id")
    track_id = str(raw.get("track_id") or "")
    if not track_id:
        raise ValueError("every object needs an explicit cross-frame track_id binding")
    center = _vector3(raw.get("geometry_center_m") or raw.get("center_3d_base_m"), "center")
    size = _vector3(raw.get("dimensions_m") or raw.get("size_xyz_m"), "size")
    label = str(raw.get("label") or raw.get("class_name") or "unknown").strip()
    color = str(infer_object_color(dict(raw)) or "unknown")
    shape = str(infer_object_shape(dict(raw)))
    yaw = raw.get("yaw_deg")
    if yaw is None and raw.get("yaw_rad") is not None:
        yaw = math.degrees(float(raw["yaw_rad"]))
    return SceneObjectState(
        object_ref=f"scene_{revision}:obj_{detector_id}",
        detector_id=detector_id,
        track_id=track_id,
        class_name=label,
        color=color,
        shape=shape,
        center_xyz_m=center,
        size_xyz_m=size,
        yaw_deg=float(yaw or 0.0),
        orientation_confidence=float(raw.get("orientation_confidence", 0.0)),
        already_completed=track_id in completed,
        protected=track_id in protected,
        currently_visible=True,
        source=dict(raw),
    )


def _with_relations(
    obj: SceneObjectState,
    objects: Sequence[SceneObjectState],
    workspace: Mapping[str, float],
) -> SceneObjectState:
    neighbors: list[str] = []
    blocked: set[str] = set()
    for other in objects:
        if other.track_id == obj.track_id:
            continue
        dx = other.center_xyz_m[0] - obj.center_xyz_m[0]
        dy = other.center_xyz_m[1] - obj.center_xyz_m[1]
        edge_gap = math.hypot(dx, dy) - 0.5 * max(obj.size_xyz_m[:2]) - 0.5 * max(other.size_xyz_m[:2])
        if edge_gap <= 0.06:
            neighbors.append(other.track_id)
        if edge_gap <= 0.012:
            blocked.add("+x" if abs(dx) >= abs(dy) and dx >= 0 else "-x" if abs(dx) >= abs(dy) else "+y" if dy >= 0 else "-y")
    cx, cy, _ = obj.center_xyz_m
    sx, sy, _ = obj.size_xyz_m
    clearances = {
        "+x": float(workspace["xmax"]) - (cx + sx / 2.0),
        "-x": (cx - sx / 2.0) - float(workspace["xmin"]),
        "+y": float(workspace["ymax"]) - (cy + sy / 2.0),
        "-y": (cy - sy / 2.0) - float(workspace["ymin"]),
    }
    blocked.update(side for side, value in clearances.items() if value < 0.015)
    free = tuple(side for side in SIDE_VECTORS if side not in blocked)
    return SceneObjectState(
        **{
            **{name: getattr(obj, name) for name in obj.__dataclass_fields__},
            "neighbors": tuple(sorted(neighbors)),
            "edge_clearances_m": {key: round(value, 6) for key, value in clearances.items()},
            "blocked_sides": tuple(side for side in SIDE_VECTORS if side in blocked),
            "free_sides": free,
        }
    )


def _occupied_region(obj: SceneObjectState) -> dict[str, Any]:
    cx, cy, cz = obj.center_xyz_m
    sx, sy, sz = obj.size_xyz_m
    return {
        "region_id": f"occupied_{obj.track_id}",
        "track_id": obj.track_id,
        "bounds_base_m": {
            "xmin": cx - sx / 2.0,
            "xmax": cx + sx / 2.0,
            "ymin": cy - sy / 2.0,
            "ymax": cy + sy / 2.0,
            "zmin": cz - sz / 2.0,
            "zmax": cz + sz / 2.0,
        },
    }


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"object {name} must contain three base_link meter values")
    output = tuple(float(item) for item in value[:3])
    if not all(math.isfinite(item) for item in output):
        raise ValueError(f"object {name} must be finite")
    return output


def _token(label: str, choices: Sequence[str], fallback: str) -> str:
    lower = label.lower().replace("-", "_")
    return next((choice for choice in choices if choice in lower), fallback)

