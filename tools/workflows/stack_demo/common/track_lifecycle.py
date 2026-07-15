"""Explicit held/placed track lifecycle and action-anchored rebinding hints."""

from __future__ import annotations

from typing import Any, Mapping

from .action_edges import PhysicalActionEdge


def mark_track_held(
    memory: dict[str, Any],
    edge: PhysicalActionEdge,
    grasp_scene_revision: int,
    *,
    held_state_confidence: float,
) -> None:
    track = memory.setdefault("tracks", {}).setdefault(
        edge.acted_object_track_id,
        {"track_id": edge.acted_object_track_id, "history": []},
    )
    track.update({
        "manipulation_state": "held_by_gripper",
        "grasp_scene_revision": int(grasp_scene_revision),
        "grasp_pose": dict(edge.physical_parameters.get("grasp_pose") or {}),
        "grasp_yaw_deg": float(edge.physical_parameters.get("grasp_yaw_deg", 0.0)),
        "relative_pose_to_tool": edge.physical_parameters.get("relative_pose_to_tool"),
        "last_known_table_pose": list(track.get("center_base_m") or ()),
        "held_state_confidence": max(0.0, min(1.0, float(held_state_confidence))),
        "visible": False,
    })
    track.setdefault("history", []).append({
        "scene_revision": int(grasp_scene_revision),
        "manipulation_state": "held_by_gripper",
        "candidate_id": edge.candidate_id,
    })


def mark_track_after_place(
    memory: dict[str, Any],
    edge: PhysicalActionEdge,
    scene_revision: int,
    *,
    success: bool,
) -> None:
    track = memory.setdefault("tracks", {}).setdefault(
        edge.acted_object_track_id,
        {"track_id": edge.acted_object_track_id, "history": []},
    )
    place_pose = edge.physical_parameters.get("place_pose") or {}
    position = place_pose.get("position_m") if isinstance(place_pose, Mapping) else None
    track.update({
        "manipulation_state": "placed" if success else "unresolved",
        "placed_scene_revision": int(scene_revision),
        "held_state_confidence": 0.0,
    })
    if success and isinstance(position, (list, tuple)) and len(position) >= 3:
        track["center_base_m"] = [float(value) for value in position[:3]]
    track.setdefault("history", []).append({
        "scene_revision": int(scene_revision),
        "manipulation_state": track["manipulation_state"],
        "candidate_id": edge.candidate_id,
    })


def predicted_track_centers(
    edge: PhysicalActionEdge,
    reason: str,
    memory: Mapping[str, Any],
) -> dict[str, list[float]]:
    """Anchor a manipulated track to an actual edge pose, never old+place-lift."""
    destination: Any = None
    if reason == "post_place":
        destination = edge.physical_parameters.get("place_pose", {}).get("position_m")
    elif reason == "post_grasp_at_safe_height":
        destination = edge.physical_parameters.get("lift_pose", {}).get("position_m")
    elif reason == "post_nudge":
        track = (memory.get("tracks") or {}).get(edge.acted_object_track_id, {})
        center = track.get("center_base_m")
        direction = edge.physical_parameters.get("push_direction_base", ())
        distance = float(edge.physical_parameters.get("push_distance_m", 0.0))
        if isinstance(center, (list, tuple)) and len(center) >= 3 and len(direction) >= 2:
            destination = [
                float(center[0]) + float(direction[0]) * distance,
                float(center[1]) + float(direction[1]) * distance,
                float(center[2]),
            ]
    if not isinstance(destination, (list, tuple)) or len(destination) < 3:
        return {}
    return {edge.acted_object_track_id: [float(value) for value in destination[:3]]}
