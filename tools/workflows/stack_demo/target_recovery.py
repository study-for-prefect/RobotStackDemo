"""Target recovery and missing-target clearance relation helpers."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from robot_scene_pipeline.geometry_relations import get_center, get_size, xy_distance
from robot_scene_pipeline.scene_memory import save_memory
from tools.planning.decision_to_execution import write_json

from .observation_scope import observe_empty_with_scope
from .scene import reacquire_target


ObjectDict = Dict[str, Any]


def _top_z(obj: ObjectDict) -> Optional[float]:
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return None
    return float(center[2]) + 0.5 * float(size[2])


def _is_clearance_object(obj: ObjectDict, protected_ids: Iterable[Any]) -> bool:
    protected = {str(value) for value in protected_ids or []}
    if str(obj.get("id")) in protected:
        return False
    if obj.get("pushable") is False:
        return False
    if obj.get("role") in ("base", "structure"):
        return False
    if obj.get("state") in ("locked", "placed"):
        return False
    if obj.get("visible") is False:
        return False
    return get_center(obj) is not None and get_size(obj) is not None


def state_with_missing_target(current_state: dict, held_template: ObjectDict) -> Tuple[dict, ObjectDict]:
    target = copy.deepcopy(held_template)
    target["reacquire_source"] = "locked_missing_target_template"
    target["target_detection_status"] = "missing_in_current_observation"
    target["visible"] = False
    state = copy.deepcopy(current_state)
    target_id = str(target.get("id"))
    objects = [
        obj for obj in state.get("objects", [])
        if str(obj.get("id")) != target_id
    ]
    state["objects"] = [target] + objects
    state["missing_target_template_id"] = target.get("id")
    return state, target


def missing_target_clearance_relations(
    current_state: dict,
    target: ObjectDict,
    protected_object_ids: Iterable[Any],
    radius_m: float,
    min_top_z_delta_m: float,
) -> List[dict]:
    target_top = _top_z(target)
    target_center = get_center(target)
    if target_center is None:
        return []
    relations = []
    for obj in current_state.get("objects", []):
        if not isinstance(obj, dict) or str(obj.get("id")) == str(target.get("id")):
            continue
        if not _is_clearance_object(obj, protected_object_ids):
            continue
        distance = xy_distance(obj, target)
        if not math.isfinite(distance) or distance > float(radius_m):
            continue
        obj_top = _top_z(obj)
        if obj_top is None:
            continue
        top_delta = None if target_top is None else obj_top - target_top
        if top_delta is not None and top_delta < float(min_top_z_delta_m):
            continue
        relations.append(
            {
                "type": "should_push_away",
                "subject": obj.get("id"),
                "object": target.get("id"),
                "source": "missing_target_high_obstacle_recovery",
                "reason": "target_missing_nearby_high_loose_object",
                "distance_to_locked_target_m": round(distance, 6),
                "obstacle_top_z_base_m": round(obj_top, 6),
                "target_top_z_base_m": None if target_top is None else round(target_top, 6),
            }
        )
    relations.sort(
        key=lambda item: (
            -float(item.get("obstacle_top_z_base_m") or 0.0),
            float(item.get("distance_to_locked_target_m") or 0.0),
        )
    )
    return relations


def recover_or_lock_missing_target(
    args: Any,
    cycle_dir: str,
    runtime: Dict[str, Any],
    memory: dict,
    current_state: dict,
    held_template: ObjectDict,
    protected_templates: Optional[Iterable[ObjectDict]] = None,
) -> Tuple[dict, dict, ObjectDict, dict]:
    report_path = os.path.join(cycle_dir, "target_recovery_report.json")
    initial_error = None
    try:
        target = copy.deepcopy(reacquire_target(current_state, held_template))
        report = {
            "schema_version": "target_recovery_report_v1",
            "status": "already_visible",
            "target_object_id": target.get("id"),
            "target_label": target.get("label"),
        }
        write_json(report_path, report)
        return current_state, memory, target, report
    except RuntimeError as first_error:
        initial_error = str(first_error)

    if getattr(args, "offline_scene_state", ""):
        locked_state, locked_target = state_with_missing_target(current_state, held_template)
        report = {
            "schema_version": "target_recovery_report_v1",
            "status": "locked_template_used_offline",
            "target_object_id": locked_target.get("id"),
            "target_label": locked_target.get("label"),
            "initial_error": initial_error,
        }
        write_json(report_path, report)
        return locked_state, memory, locked_target, report

    try:
        observed_state, memory, scoped_report = observe_empty_with_scope(
            args,
            os.path.join(cycle_dir, "target_recovery_observation"),
            runtime,
            memory,
            critical_templates=[held_template],
            noncritical_templates=protected_templates or [],
            scope_name="target_recovery_before_clearance",
            description="Target recovery observation",
            allow_critical_label_fallback=True,
        )
        if observed_state is not None:
            target = copy.deepcopy(reacquire_target(observed_state, held_template))
            report = {
                "schema_version": "target_recovery_report_v1",
                "status": "recovered_by_scoped_observation",
                "target_object_id": target.get("id"),
                "target_label": target.get("label"),
                "scoped_observation": scoped_report,
            }
            write_json(report_path, report)
            save_memory(memory, args.memory_json)
            return observed_state, memory, target, report
    except RuntimeError as recovery_error:
        locked_state, locked_target = state_with_missing_target(current_state, held_template)
        report = {
            "schema_version": "target_recovery_report_v1",
            "status": "locked_missing_target_template",
            "target_object_id": locked_target.get("id"),
            "target_label": locked_target.get("label"),
            "initial_error": initial_error,
            "recovery_error": str(recovery_error),
        }
        write_json(report_path, report)
        return locked_state, memory, locked_target, report

    locked_state, locked_target = state_with_missing_target(current_state, held_template)
    report = {
        "schema_version": "target_recovery_report_v1",
        "status": "locked_missing_target_template",
        "target_object_id": locked_target.get("id"),
        "target_label": locked_target.get("label"),
        "initial_error": initial_error,
    }
    write_json(report_path, report)
    return locked_state, memory, locked_target, report
