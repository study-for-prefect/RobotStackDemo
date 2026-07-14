"""Pose-aware conservative swept-volume checks for GF225 push clearing."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from robot_scene_pipeline.geometry_relations import object_xy_aabb
from tools.robot.push_primitives import build_push_targets


ObjectDict = Dict[str, Any]
DEFAULT_PUSH_PROFILE = [
    {"name": "tip", "z_from_tip_min_m": 0.000, "z_from_tip_max_m": 0.025, "width_m": 0.025},
    {"name": "upper_fingers", "z_from_tip_min_m": 0.025, "z_from_tip_max_m": 0.070, "width_m": 0.062},
    {"name": "gripper_body", "z_from_tip_min_m": 0.070, "z_from_tip_max_m": 0.150, "width_m": 0.112},
]


def push_tool_swept_obbs(
    push_plan: Dict[str, Any],
    gripper_outer_width_m: float = 0.112,
    finger_length_m: float = 0.12,
    tool_depth_m: float = 0.04,
    fingertip_thickness_m: float = 0.01,
    safety_margin_m: float = 0.01,
    tcp_offset_tool_m: Optional[Sequence[float]] = None,
    push_profile: Optional[Sequence[dict]] = None,
) -> List[Dict[str, Any]]:
    """Return yaw-oriented tool envelopes anchored above the TCP contact point."""
    targets = build_push_targets(push_plan)
    yaw_value = push_plan.get("gripper_yaw_rad")
    if yaw_value is None:
        yaw_value = math.radians(float(push_plan["target_yaw_deg"]))
    yaw = float(yaw_value)
    half_depth = 0.5 * float(tool_depth_m) + float(safety_margin_m)
    tip_below_tcp = 0.5 * float(fingertip_thickness_m) + float(safety_margin_m)
    tcp_offset_z = abs(float((tcp_offset_tool_m or [0.0, 0.0, 0.0])[2]))
    height_above_tcp = max(float(finger_length_m), tcp_offset_z) + float(safety_margin_m)
    stages = [
        ("pre_push_pose", targets["pre_push"], targets["pre_push"]),
        ("approach_to_contact", targets["pre_push"], targets["contact"]),
        ("contact_pose", targets["contact"], targets["contact"]),
        ("horizontal_push", targets["contact"], targets["push_end"]),
        ("retreat", targets["push_end"], targets["retreat"]),
    ]
    envelopes = []
    for stage, start, end in stages:
        for profile in push_profile or DEFAULT_PUSH_PROFILE:
            half_width = 0.5 * float(profile["width_m"]) + float(safety_margin_m)
            envelope = _segment_obb(stage, start, end, yaw, half_depth, half_width, tip_below_tcp, height_above_tcp)
            tip_z = min(float(start[2]), float(end[2])) - 0.5 * float(fingertip_thickness_m)
            envelope.update({
                "profile_name": profile["name"], "profile_width_m": float(profile["width_m"]),
                "zmin": tip_z + float(profile["z_from_tip_min_m"]),
                "zmax": max(float(start[2]), float(end[2])) - 0.5 * float(fingertip_thickness_m) + float(profile["z_from_tip_max_m"]),
            })
            envelope["diagnostic_aabb"] = _obb_aabb(envelope)
            envelopes.append(envelope)
    return envelopes


def push_tool_swept_aabbs(
    push_plan: Dict[str, Any],
    gripper_outer_width_m: float = 0.112,
    finger_length_m: float = 0.12,
    safety_margin_m: float = 0.01,
) -> List[Dict[str, Any]]:
    """Compatibility name; returned envelopes are OBBs with diagnostic AABBs."""
    return push_tool_swept_obbs(
        push_plan,
        gripper_outer_width_m=gripper_outer_width_m,
        finger_length_m=finger_length_m,
        safety_margin_m=safety_margin_m,
    )


def check_tool_swept_volume(
    push_plan: Dict[str, Any],
    objects: Iterable[ObjectDict],
    ignore_object_ids: Optional[Iterable[Any]] = None,
    protected_object_ids: Optional[Iterable[Any]] = None,
    gripper_outer_width_m: float = 0.112,
    finger_length_m: float = 0.12,
    tool_depth_m: float = 0.04,
    fingertip_thickness_m: float = 0.01,
    safety_margin_m: float = 0.01,
    tcp_offset_tool_m: Optional[Sequence[float]] = None,
    push_profile: Optional[Sequence[dict]] = None,
    controlled_contact: Optional[dict] = None,
) -> Dict[str, Any]:
    """Hard-reject tool/table/object collisions while reporting object contacts separately."""
    ignored = {str(value) for value in (ignore_object_ids or [])}
    protected = {str(value) for value in (protected_object_ids or [])}
    try:
        swept = push_tool_swept_obbs(
            push_plan,
            gripper_outer_width_m=gripper_outer_width_m,
            finger_length_m=finger_length_m,
            tool_depth_m=tool_depth_m,
            fingertip_thickness_m=fingertip_thickness_m,
            safety_margin_m=safety_margin_m,
            tcp_offset_tool_m=tcp_offset_tool_m,
            push_profile=push_profile,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return {"feasible": False, "reason": "invalid_push_plan", "hard_collisions": [], "error": str(exc)}

    hard_collisions: List[Dict[str, Any]] = []
    controlled_contacts: List[Dict[str, Any]] = []
    contact_config = controlled_contact or {}
    scene_objects = [obj for obj in objects or [] if isinstance(obj, dict)]
    for obj in scene_objects:
        if obj.get("visible") is False or str(obj.get("id")) in ignored:
            continue
        obj_aabb = object_xy_aabb(obj)
        if not obj_aabb:
            continue
        for envelope in swept:
            overlap = _obb_aabb_overlap(envelope, obj_aabb)
            if overlap is None:
                continue
            entity_type = "protected_structure" if str(obj.get("id")) in protected else "movable_object"
            collision = {
                "id": obj.get("id"),
                "label": obj.get("label"),
                "role": obj.get("role"),
                "state": obj.get("state"),
                "entity_type": entity_type,
                "collision_source": "gf225_tool",
                "stage": envelope["stage"],
                "reason": "tool_swept_volume_collision",
                "minimum_clearance_m": round(-overlap, 6),
                "profile_name": envelope.get("profile_name"), "profile_width_m": envelope.get("profile_width_m"),
            }
            controlled_ok, contact_checks = _controlled_contact_allowed(
                push_plan, obj, envelope, overlap, entity_type, contact_config,
                {item.get("id") for item in controlled_contacts},
            )
            collision["controlled_contact_checks"] = contact_checks
            if controlled_ok:
                collision.update({"classification": "controlled_contact", "expected_passive_displacement_m": min(overlap, float(contact_config.get("max_expected_passive_displacement_m", 0.015)))})
                controlled_contacts.append(collision)
            else:
                collision["classification"] = "hard_collision"
                hard_collisions.append(collision)

    table_collision = _table_collision(push_plan, swept)
    if table_collision:
        hard_collisions.append(table_collision)
    recoverable_contacts = _pushed_object_contacts(push_plan, scene_objects, ignored | protected)
    feasible = not hard_collisions
    return {
        "schema_version": "gf225_tool_swept_volume_v2",
        "feasible": feasible,
        "reason": ("controlled_contact" if feasible and controlled_contacts else "tool_swept_volume_clear") if feasible else hard_collisions[0]["reason"],
        "contact_status": "hard_collision" if hard_collisions else ("controlled_contact" if controlled_contacts else "clear"),
        "collision_policy": {
            "gf225_tool_contact": "segmented_strict_or_controlled_clearance_contact",
            "pushed_object_to_movable_object": "recoverable_contact",
        },
        "hard_collisions": hard_collisions,
        "collisions": hard_collisions,
        "recoverable_contacts": recoverable_contacts,
        "controlled_contacts": controlled_contacts,
        "requires_reobservation": bool(controlled_contacts or recoverable_contacts),
        "swept_obbs": swept,
        "gripper_collision_profile": list(push_profile or DEFAULT_PUSH_PROFILE),
    }


def _controlled_contact_allowed(
    push_plan: dict, obj: dict, envelope: dict, overlap: float, entity_type: str,
    config: dict, contacted_ids: set,
) -> Tuple[bool, dict]:
    expected = float(overlap)
    size = obj.get("dimensions_m") or obj.get("size_m") or []
    footprint_limit = 0.5 * min(float(value) for value in size[:2]) if len(size) >= 2 else 0.0
    loose_chain_contact = bool(config.get("allow_loose_chain_contact", False))
    entry_side_clear = envelope.get("stage") not in {
        "approach_to_contact", "contact_pose",
    }
    checks = {
        "enabled": bool(config.get("enabled", True)),
        "loose_unprotected_object": (
            entity_type == "movable_object"
            and str(obj.get("role") or "") not in {"base", "structure", "support"}
            and str(obj.get("state") or "") not in {"placed", "locked", "protected"}
        ),
        "side_intrusion_within_limit": (
            loose_chain_contact
            or expected <= float(config.get("max_side_intrusion_m", 0.005))
        ),
        "passive_displacement_within_limit": (
            loose_chain_contact
            or expected <= float(config.get("max_expected_passive_displacement_m", 0.015))
        ),
        "contacted_object_count_within_limit": (
            loose_chain_contact
            or obj.get("id") in contacted_ids
            or len(contacted_ids) < int(config.get("max_contacted_objects", 2))
        ),
        "withdrawal_clear": (
            loose_chain_contact or envelope.get("stage") != "retreat"
        ),
        # A side push must descend through an empty contact-side column.  Loose
        # chain contact is permitted only once horizontal pushing has begun;
        # it must never justify descending on top of a neighbouring block.
        "entry_contact_side_clear": entry_side_clear,
        "topple_risk_low": (
            loose_chain_contact
            or (footprint_limit > 0.0 and expected <= footprint_limit)
        ),
        "workspace_safe": _passive_motion_inside_workspace(push_plan, obj, expected),
    }
    return all(checks.values()), checks


def _passive_motion_inside_workspace(push_plan: dict, obj: dict, distance_m: float) -> bool:
    bounds = push_plan.get("workspace_bounds") or push_plan.get("table_bounds")
    center = obj.get("geometry_center_m") or obj.get("center_base_m")
    size = obj.get("dimensions_m") or obj.get("size_m")
    direction = push_plan.get("direction_base")
    if not isinstance(bounds, dict):
        return True
    if not all(isinstance(value, (list, tuple)) for value in (center, size, direction)):
        return False
    next_x = float(center[0]) + float(direction[0]) * distance_m
    next_y = float(center[1]) + float(direction[1]) * distance_m
    return (
        next_x - 0.5 * float(size[0]) >= float(bounds.get("xmin", -math.inf))
        and next_x + 0.5 * float(size[0]) <= float(bounds.get("xmax", math.inf))
        and next_y - 0.5 * float(size[1]) >= float(bounds.get("ymin", -math.inf))
        and next_y + 0.5 * float(size[1]) <= float(bounds.get("ymax", math.inf))
    )


def _segment_obb(
    stage: str,
    start: Sequence[float],
    end: Sequence[float],
    yaw: float,
    half_depth: float,
    half_width: float,
    tip_below_tcp: float,
    height_above_tcp: float,
) -> Dict[str, Any]:
    axis_u = [math.cos(yaw), math.sin(yaw)]
    axis_v = [-math.sin(yaw), math.cos(yaw)]
    start_u, start_v = _project_xy(start, axis_u, axis_v)
    end_u, end_v = _project_xy(end, axis_u, axis_v)
    umin, umax = min(start_u, end_u) - half_depth, max(start_u, end_u) + half_depth
    vmin, vmax = min(start_v, end_v) - half_width, max(start_v, end_v) + half_width
    center_u, center_v = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    center_xy = [
        center_u * axis_u[0] + center_v * axis_v[0],
        center_u * axis_u[1] + center_v * axis_v[1],
    ]
    envelope = {
        "stage": stage,
        "center_xy_m": center_xy,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "half_extent_u_m": 0.5 * (umax - umin),
        "half_extent_v_m": 0.5 * (vmax - vmin),
        "zmin": min(float(start[2]), float(end[2])) - tip_below_tcp,
        "zmax": max(float(start[2]), float(end[2])) + height_above_tcp,
        "tcp_anchor_semantics": "asymmetric_from_fingertip_contact_up_toward_tool0",
        "yaw_rad": yaw,
    }
    envelope["diagnostic_aabb"] = _obb_aabb(envelope)
    return envelope


def _project_xy(point: Sequence[float], axis_u: Sequence[float], axis_v: Sequence[float]) -> Tuple[float, float]:
    return (
        float(point[0]) * axis_u[0] + float(point[1]) * axis_u[1],
        float(point[0]) * axis_v[0] + float(point[1]) * axis_v[1],
    )


def _obb_corners(envelope: Dict[str, Any]) -> List[List[float]]:
    center = envelope["center_xy_m"]
    axis_u, axis_v = envelope["axis_u"], envelope["axis_v"]
    hu, hv = envelope["half_extent_u_m"], envelope["half_extent_v_m"]
    return [
        [center[0] + su * hu * axis_u[0] + sv * hv * axis_v[0],
         center[1] + su * hu * axis_u[1] + sv * hv * axis_v[1]]
        for su in (-1.0, 1.0) for sv in (-1.0, 1.0)
    ]


def _obb_aabb(envelope: Dict[str, Any]) -> Dict[str, float]:
    corners = _obb_corners(envelope)
    return {
        "xmin": min(point[0] for point in corners), "xmax": max(point[0] for point in corners),
        "ymin": min(point[1] for point in corners), "ymax": max(point[1] for point in corners),
        "zmin": envelope["zmin"], "zmax": envelope["zmax"],
    }


def _obb_aabb_overlap(envelope: Dict[str, Any], aabb: Dict[str, float]) -> Optional[float]:
    if min(envelope["zmax"], aabb["zmax"]) <= max(envelope["zmin"], aabb["zmin"]):
        return None
    obb_corners = _obb_corners(envelope)
    box_corners = [
        [aabb[x], aabb[y]]
        for x in ("xmin", "xmax") for y in ("ymin", "ymax")
    ]
    axes = ([1.0, 0.0], [0.0, 1.0], envelope["axis_u"], envelope["axis_v"])
    overlaps = []
    for axis in axes:
        first = [point[0] * axis[0] + point[1] * axis[1] for point in obb_corners]
        second = [point[0] * axis[0] + point[1] * axis[1] for point in box_corners]
        overlap = min(max(first), max(second)) - max(min(first), min(second))
        if overlap <= 0.0:
            return None
        overlaps.append(overlap)
    overlaps.append(min(envelope["zmax"], aabb["zmax"]) - max(envelope["zmin"], aabb["zmin"]))
    return min(overlaps)


def _table_collision(push_plan: Dict[str, Any], swept: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    obstacle = push_plan.get("obstacle") or {}
    center, size = obstacle.get("geometry_center_m"), obstacle.get("dimensions_m")
    if not isinstance(center, (list, tuple)) or not isinstance(size, (list, tuple)) or len(center) < 3 or len(size) < 3:
        return None
    table_z = float(center[2]) - 0.5 * float(size[2])
    lowest = min(float(envelope["zmin"]) for envelope in swept)
    if lowest >= table_z - 0.001:
        return None
    return {
        "id": None,
        "label": "table",
        "entity_type": "table",
        "collision_source": "gf225_tool",
        "stage": min(swept, key=lambda item: item["zmin"])["stage"],
        "reason": "tool_table_collision",
        "minimum_clearance_m": round(lowest - table_z, 6),
    }


def _pushed_object_contacts(
    push_plan: Dict[str, Any], objects: List[ObjectDict], excluded_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    obstacle = push_plan.get("obstacle") or {}
    obstacle_aabb = object_xy_aabb(obstacle)
    direction, distance = push_plan.get("direction_base"), push_plan.get("distance_m")
    if not obstacle_aabb or not isinstance(direction, (list, tuple)) or len(direction) < 2:
        return []
    dx, dy = float(direction[0]) * float(distance), float(direction[1]) * float(distance)
    swept = dict(obstacle_aabb)
    swept.update({
        "xmin": min(obstacle_aabb["xmin"], obstacle_aabb["xmin"] + dx),
        "xmax": max(obstacle_aabb["xmax"], obstacle_aabb["xmax"] + dx),
        "ymin": min(obstacle_aabb["ymin"], obstacle_aabb["ymin"] + dy),
        "ymax": max(obstacle_aabb["ymax"], obstacle_aabb["ymax"] + dy),
    })
    contacts = []
    excluded = {str(value) for value in excluded_ids}
    for obj in objects:
        if str(obj.get("id")) in excluded:
            continue
        other = object_xy_aabb(obj)
        if other and _aabb_xy_overlap(swept, other):
            contacts.append({
                "type": "pushed_object_contact_with_movable_object",
                "operated_object_id": obstacle.get("id"),
                "contacted_object_id": obj.get("id"),
                "contacted_object_label": obj.get("label"),
                "classification": "recoverable_contact",
            })
    return contacts


def _aabb_xy_overlap(first: Dict[str, float], second: Dict[str, float]) -> bool:
    return min(first["xmax"], second["xmax"]) > max(first["xmin"], second["xmin"]) and (
        min(first["ymax"], second["ymax"]) > max(first["ymin"], second["ymin"])
    )
