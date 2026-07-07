"""Scoped post-action observation and guarded memory updates."""

import copy
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from robot_scene_pipeline.scene_memory import update_from_scoped_detections
from tools.planning.decision_to_execution import write_json

from .commands import (
    capture_empty_current_pose,
    init_ready_pose,
    relative_translate_command,
    run,
)


ObjectDict = Dict[str, Any]


def _finite_xyz(obj: ObjectDict) -> Optional[List[float]]:
    for key in ("geometry_center_m", "center_base_m", "center_base", "last_pose_base", "target_position_m"):
        value = obj.get(key)
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            continue
        try:
            center = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(item) for item in center):
            return center
    return None


def _label(obj: ObjectDict) -> Optional[str]:
    value = obj.get("label") or obj.get("class_name") or obj.get("name")
    return None if value is None else str(value)


def _xy_distance(first: ObjectDict, second: ObjectDict) -> float:
    first_center = _finite_xyz(first)
    second_center = _finite_xyz(second)
    if first_center is None or second_center is None:
        return math.inf
    return math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])


def _matching_object(
    objects: Iterable[ObjectDict],
    template: ObjectDict,
    max_dist_m: float,
    max_z_delta_m: Optional[float] = None,
) -> Optional[ObjectDict]:
    label = _label(template)
    if label is None:
        return None
    template_center = _finite_xyz(template)
    candidates = [
        obj for obj in objects
        if isinstance(obj, dict) and _label(obj) == label and _finite_xyz(obj) is not None
    ]
    if max_z_delta_m is not None and template_center is not None:
        candidates = [
            obj for obj in candidates
            if abs(_finite_xyz(obj)[2] - template_center[2]) <= float(max_z_delta_m)
        ]
    candidates.sort(key=lambda obj: _xy_distance(obj, template))
    if candidates and _xy_distance(candidates[0], template) <= float(max_dist_m):
        return candidates[0]
    return None


def _memory_id_for_template(memory: dict, template: ObjectDict, max_dist_m: float) -> Optional[str]:
    label = _label(template)
    center = _finite_xyz(template)
    if label is None or center is None:
        return None
    best_id = None
    best_dist = math.inf
    for obj_id, obj in memory.get("objects", {}).items():
        if obj.get("label") != label:
            continue
        old_center = obj.get("last_pose_base")
        if not isinstance(old_center, list) or len(old_center) < 2:
            continue
        dist = math.hypot(float(old_center[0]) - center[0], float(old_center[1]) - center[1])
        if dist < best_dist:
            best_id = obj_id
            best_dist = dist
    return best_id if best_id is not None and best_dist <= float(max_dist_m) else None


def _parse_offsets(raw_value: Any) -> List[List[float]]:
    if raw_value is None:
        raw_value = "0,0,0;0.04,0,0;-0.04,0,0;0,0.04,0;0,-0.04,0"
    if isinstance(raw_value, str):
        offsets = []
        for item in raw_value.split(";"):
            if not item.strip():
                continue
            values = [float(value.strip()) for value in item.split(",")]
            if len(values) != 3:
                raise ValueError("Each scoped observation offset must contain exactly three values.")
            offsets.append(values)
        return offsets or [[0.0, 0.0, 0.0]]
    return [[float(value) for value in offset[:3]] for offset in raw_value]


def _merge_objects(states: Iterable[dict], merge_dist_m: float = 0.025) -> List[ObjectDict]:
    merged: List[ObjectDict] = []
    for state in states:
        for obj in state.get("objects", []):
            if not isinstance(obj, dict):
                continue
            center = _finite_xyz(obj)
            label = _label(obj)
            if center is None or label is None:
                continue
            existing = _matching_object(merged, obj, merge_dist_m)
            if existing is None:
                merged.append(copy.deepcopy(obj))
                continue
            if existing.get("pointcloud_geometry_valid") is not True and obj.get("pointcloud_geometry_valid") is True:
                existing.update(copy.deepcopy(obj))
    return merged


def _scope_report(
    fused_state: dict,
    critical_templates: Iterable[ObjectDict],
    max_dist_m: float,
    allow_label_fallback: bool = False,
    max_z_delta_m: Optional[float] = None,
) -> Tuple[List[dict], List[dict]]:
    observed = []
    missing = []
    objects = fused_state.get("objects", [])
    for template in critical_templates:
        match = _matching_object(objects, template, max_dist_m, max_z_delta_m=max_z_delta_m)
        match_policy = "nearest_label_within_distance_gate"
        if match is None and allow_label_fallback:
            match = _unique_label_fallback_match(objects, template)
            match_policy = "unique_label_outside_distance_gate"
        item = {
            "template_id": template.get("id"),
            "template_label": _label(template),
            "template_center_m": _finite_xyz(template),
        }
        if match is None:
            missing.append(item)
        else:
            item.update({
                "observed_id": match.get("id"),
                "observed_center_m": _finite_xyz(match),
                "distance_m": round(_xy_distance(match, template), 6),
                "z_delta_m": round(abs(_finite_xyz(match)[2] - _finite_xyz(template)[2]), 6)
                if _finite_xyz(match) is not None and _finite_xyz(template) is not None
                else None,
                "match_policy": match_policy,
            })
            observed.append(item)
    return observed, missing


def _unique_label_fallback_match(objects: Iterable[ObjectDict], template: ObjectDict) -> Optional[ObjectDict]:
    label = _label(template)
    if label is None:
        return None
    candidates = [
        obj for obj in objects
        if isinstance(obj, dict) and _label(obj) == label and _finite_xyz(obj) is not None
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def expected_placed_template(held_object: ObjectDict, place_step: dict) -> ObjectDict:
    template = copy.deepcopy(held_object)
    tcp_xy = place_step.get("tcp_place_xy_base_m")
    offset_xy = place_step.get("object_offset_base_xy_m")
    release_z = place_step.get("release_z_base_m")
    if (
        isinstance(tcp_xy, list)
        and len(tcp_xy) >= 2
        and isinstance(offset_xy, list)
        and len(offset_xy) >= 2
        and release_z is not None
    ):
        template["geometry_center_m"] = [
            float(tcp_xy[0]) + float(offset_xy[0]),
            float(tcp_xy[1]) + float(offset_xy[1]),
            float(release_z),
        ]
    else:
        target_position = place_step.get("target_position_m")
        if isinstance(target_position, list) and len(target_position) >= 3:
            template["geometry_center_m"] = [float(value) for value in target_position[:3]]
    template["reacquire_source"] = "expected_post_place_pose"
    return template


def observe_empty_with_scope(
    args: Any,
    output_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    critical_templates: Iterable[ObjectDict],
    noncritical_templates: Optional[Iterable[ObjectDict]] = None,
    scope_name: str = "post_action",
    description: str = "Post-action observation",
    allow_critical_label_fallback: bool = False,
    critical_match_z_tolerance_m: Optional[float] = None,
) -> Tuple[Optional[dict], dict, dict]:
    critical = [copy.deepcopy(obj) for obj in critical_templates if isinstance(obj, dict)]
    noncritical = [copy.deepcopy(obj) for obj in (noncritical_templates or []) if isinstance(obj, dict)]
    if runtime.get("held_object_id") is not None:
        raise RuntimeError("{} forbidden while holding object {}.".format(description, runtime["held_object_id"]))
    if getattr(args, "offline_scene_state", ""):
        return None, memory, {"scope_name": scope_name, "status": "offline_scene_state"}

    os.makedirs(output_dir, exist_ok=True)
    offsets = _parse_offsets(getattr(args, "scoped_observation_recovery_offsets_base", None))
    max_attempts = max(1, int(getattr(args, "scoped_observation_max_attempts", len(offsets))))
    max_dist_m = float(getattr(args, "scoped_observation_match_distance_m", 0.07))
    states = []
    attempts = []
    selected_state = None
    selected_missing: List[dict] = []

    for attempt_index, offset in enumerate(offsets[:max_attempts], start=1):
        attempt_dir = os.path.join(output_dir, "attempt_{:02d}".format(attempt_index))
        init_ready_pose(args)
        if getattr(args, "execute", False) and any(abs(value) > 1e-9 for value in offset):
            run(relative_translate_command(args, offset))
        try:
            state = capture_empty_current_pose(
                args,
                attempt_dir,
                runtime.get("held_object_id"),
                refresh_tf=attempt_index > 1,
            )
        except RuntimeError as exc:
            attempts.append({"attempt": attempt_index, "offset_base_m": offset, "error": str(exc)})
            continue
        if state is None:
            return None, memory, {"scope_name": scope_name, "status": "no_live_observation"}
        states.append(state)
        fused_state = copy.deepcopy(state)
        fused_state["objects"] = _merge_objects(states)
        observed, missing = _scope_report(
            fused_state,
            critical,
            max_dist_m,
            allow_label_fallback=allow_critical_label_fallback,
            max_z_delta_m=critical_match_z_tolerance_m,
        )
        attempts.append({
            "attempt": attempt_index,
            "offset_base_m": offset,
            "object_count": len(state.get("objects", [])),
            "fused_object_count": len(fused_state.get("objects", [])),
            "observed_critical": observed,
            "missing_critical": missing,
        })
        selected_state = fused_state
        selected_missing = missing
        if not missing:
            break
        print(
            "{} missing critical objects on attempt {}: {}.".format(
                description,
                attempt_index,
                [item.get("template_label") for item in missing],
            ),
            flush=True,
        )

    if selected_state is None:
        raise RuntimeError("{} failed: no observation attempts produced a scene state.".format(description))

    critical_memory_ids = [
        obj_id for obj_id in (
            _memory_id_for_template(memory, template, max_dist_m=max_dist_m)
            for template in critical
        )
        if obj_id is not None
    ]
    noncritical_memory_ids = [
        obj_id for obj_id in (
            _memory_id_for_template(memory, template, max_dist_m=max_dist_m)
            for template in noncritical
        )
        if obj_id is not None
    ]
    scoped_ids = sorted(set(critical_memory_ids + noncritical_memory_ids))
    memory = update_from_scoped_detections(
        memory,
        selected_state.get("objects", []),
        scoped_object_ids=scoped_ids,
        critical_object_ids=critical_memory_ids,
        observation_scope=scope_name,
    )
    report = {
        "schema_version": "scoped_observation_v1",
        "scope_name": scope_name,
        "description": description,
        "status": "critical_confirmed" if not selected_missing else "critical_missing",
        "critical_memory_ids": critical_memory_ids,
        "noncritical_memory_ids": noncritical_memory_ids,
        "attempts": attempts,
        "fused_object_count": len(selected_state.get("objects", [])),
        "missing_critical": selected_missing,
    }
    write_json(os.path.join(output_dir, "scoped_observation_report.json"), report)
    if selected_missing:
        raise RuntimeError(
            "{} refused to continue: critical objects still missing after scoped recovery: {}".format(
                description,
                [item.get("template_label") for item in selected_missing],
            )
        )
    return selected_state, memory, report
