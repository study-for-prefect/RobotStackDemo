"""Simple 2.5D swept-volume checks for GF225-style push clearing."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from robot_scene_pipeline.geometry_relations import object_xy_aabb, xy_aabb_overlap
from tools.robot.push_primitives import build_push_targets


ObjectDict = Dict[str, Any]


def _segment_aabb(
    start: List[float],
    end: List[float],
    half_width_m: float,
    z_margin_m: float,
) -> Dict[str, float]:
    return {
        "xmin": min(start[0], end[0]) - half_width_m,
        "xmax": max(start[0], end[0]) + half_width_m,
        "ymin": min(start[1], end[1]) - half_width_m,
        "ymax": max(start[1], end[1]) + half_width_m,
        "zmin": min(start[2], end[2]) - z_margin_m,
        "zmax": max(start[2], end[2]) + z_margin_m,
    }


def _z_overlap(first: Dict[str, float], second: Dict[str, float]) -> bool:
    return min(first["zmax"], second["zmax"]) > max(first["zmin"], second["zmin"])


def _object_id(obj: ObjectDict) -> str:
    return str(obj.get("id"))


def push_tool_swept_aabbs(
    push_plan: Dict[str, Any],
    gripper_outer_width_m: float = 0.112,
    finger_length_m: float = 0.12,
    safety_margin_m: float = 0.01,
) -> List[Dict[str, Any]]:
    """Return conservative AABBs for pre/contact/push/retreat tool volume."""
    targets = build_push_targets(push_plan)
    half_width = 0.5 * float(gripper_outer_width_m) + float(safety_margin_m)
    z_margin = 0.5 * float(finger_length_m) + float(safety_margin_m)
    return [
        {
            "stage": "pre_push_pose",
            **_segment_aabb(targets["pre_push"], targets["pre_push"], half_width, z_margin),
        },
        {
            "stage": "vertical_approach",
            **_segment_aabb(targets["pre_push"], targets["contact"], half_width, z_margin),
        },
        {
            "stage": "contact_pose",
            **_segment_aabb(targets["contact"], targets["contact"], half_width, z_margin),
        },
        {
            "stage": "horizontal_push",
            **_segment_aabb(targets["contact"], targets["push_end"], half_width, z_margin),
        },
        {
            "stage": "retreat",
            **_segment_aabb(targets["push_end"], targets["retreat"], half_width, z_margin),
        },
    ]


def check_tool_swept_volume(
    push_plan: Dict[str, Any],
    objects: Iterable[ObjectDict],
    ignore_object_ids: Optional[Iterable[Any]] = None,
    gripper_outer_width_m: float = 0.112,
    finger_length_m: float = 0.12,
    safety_margin_m: float = 0.01,
) -> Dict[str, Any]:
    """Reject push plans whose gripper envelope intersects scene objects."""
    ignored = {str(value) for value in (ignore_object_ids or [])}
    try:
        swept = push_tool_swept_aabbs(
            push_plan,
            gripper_outer_width_m=gripper_outer_width_m,
            finger_length_m=finger_length_m,
            safety_margin_m=safety_margin_m,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {"feasible": False, "reason": "invalid_push_plan", "collisions": [], "error": str(exc)}

    collisions: List[Dict[str, Any]] = []
    for obj in objects or []:
        if not isinstance(obj, dict) or obj.get("visible") is False or _object_id(obj) in ignored:
            continue
        obj_aabb = object_xy_aabb(obj)
        if not obj_aabb:
            continue
        for swept_aabb in swept:
            if xy_aabb_overlap(swept_aabb, obj_aabb)[2] <= 0.0 or not _z_overlap(swept_aabb, obj_aabb):
                continue
            stage = str(swept_aabb["stage"])
            reason = "vertical_approach_collision" if stage == "vertical_approach" else "tool_swept_collision"
            collisions.append(
                {
                    "id": obj.get("id"),
                    "label": obj.get("label"),
                    "role": obj.get("role"),
                    "state": obj.get("state"),
                    "stage": stage,
                    "reason": reason,
                }
            )

    if collisions:
        return {
            "feasible": False,
            "reason": collisions[0]["reason"],
            "collisions": collisions,
            "swept_aabbs": swept,
        }
    return {"feasible": True, "reason": "tool_swept_volume_clear", "collisions": [], "swept_aabbs": swept}
