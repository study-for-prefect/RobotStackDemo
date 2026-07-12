# robot_scene_pipeline/scene_memory.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from .object_tracking import update_scene_tracks


Memory = Dict[str, Any]
Obj = Dict[str, Any]


def default_memory(task: str = "build_blocks") -> Memory:
    return {
        "version": 1,
        "task": task,
        "step_index": 0,
        "objects": {},
        "structure": {
            "base": None,
            "placed_order": [],
            "current_top": None,
            "top_center_base": None,
            "top_z": None,
        },
        "action_history": [],
        "tracks": {},
        "track_history": [],
        "role_binding_history": [],
    }


def load_memory(path: str | Path, task: str = "build_blocks") -> Memory:
    path = Path(path)
    if not path.exists():
        return default_memory(task=task)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory: Memory, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def reset_memory(path: str | Path, task: str = "build_blocks") -> Memory:
    memory = default_memory(task=task)
    save_memory(memory, path)
    return memory


def _dist_xy(a: List[float], b: List[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _get_center_base(det: Obj) -> Optional[List[float]]:
    for key in [
        "center_base",
        "center_base_m",
        "geometry_center_m",
        "base_center_m",
    ]:
        v = det.get(key)
        if isinstance(v, list) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
    return None


def _get_size_m(det: Obj) -> Optional[List[float]]:
    for key in [
        "size_m",
        "dimensions_m",
        "bbox_size_m",
    ]:
        v = det.get(key)
        if isinstance(v, list) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
    return None


def _get_label(det: Obj) -> Optional[str]:
    label = det.get("label") or det.get("class_name") or det.get("name")
    if label is None:
        return None
    return str(label)


def _get_yaw(det: Obj) -> Optional[float]:
    for key in ["yaw_rad", "yaw", "theta_rad"]:
        if key in det and det[key] is not None:
            return float(det[key])
    return None


def _get_top_z(det: Obj, center: List[float], size: Optional[List[float]]) -> Optional[float]:
    for key in ["top_z", "top_z_m", "geometry_top_z_m"]:
        if key in det and det[key] is not None:
            return float(det[key])

    if size is not None:
        return float(center[2] + size[2] * 0.5)

    return None


def _make_object_id(label: str, objects: Dict[str, Obj]) -> str:
    prefix = label.replace(" ", "_")
    index = 1

    while f"{prefix}_{index}" in objects:
        index += 1

    return f"{prefix}_{index}"


def _find_match(
    memory: Memory,
    label: str,
    center: List[float],
    max_dist_m: float = 0.05,
    excluded_ids: Optional[Set[str]] = None,
) -> Optional[str]:
    best_id = None
    best_dist = float("inf")

    for obj_id, obj in memory["objects"].items():
        if str(obj_id) in (excluded_ids or set()):
            continue
        if obj.get("label") != label:
            continue

        old_center = obj.get("last_pose_base")
        if not old_center:
            continue

        d = _dist_xy(center, old_center)

        if d < best_dist and d <= max_dist_m:
            best_dist = d
            best_id = obj_id

    return best_id


def _default_operability(role: str, state: str) -> Tuple[bool, bool]:
    locked = role in ["base", "structure"] or state in ["placed", "locked"]

    if locked:
        return False, False

    return True, True


def update_from_detections(
    memory: Memory,
    detections: List[Obj],
    match_dist_m: float = 0.05,
    scene_revision: Optional[int] = None,
) -> Memory:
    """
    用当前检测结果更新场景记忆。

    detections 可以来自 snapshot_pipeline 的 objects。
    只要求每个物体至少包含：
    - label / class_name / name
    - geometry_center_m / center_base / center_base_m
    - dimensions_m / size_m
    """

    memory["step_index"] = int(memory.get("step_index", 0)) + 1
    step_index = memory["step_index"]
    memory, assignments = update_scene_tracks(memory, detections, scene_revision or step_index)
    memory["last_track_assignment"] = assignments

    for obj in memory["objects"].values():
        obj["visible"] = False

    matched_memory_ids: Set[str] = set()
    for det in detections:
        label = _get_label(det)
        center = _get_center_base(det)
        size = _get_size_m(det)

        if label is None or center is None:
            continue

        obj_id = _find_match(memory, label, center, max_dist_m=match_dist_m, excluded_ids=matched_memory_ids)

        if obj_id is None:
            obj_id = _make_object_id(label, memory["objects"])
            memory["objects"][obj_id] = {
                "id": obj_id,
                "label": label,
                "role": "unknown",
                "state": "free",
            }

        mem_obj = memory["objects"][obj_id]
        matched_memory_ids.add(str(obj_id))

        role = mem_obj.get("role", "unknown")
        state = mem_obj.get("state", "free")

        yaw = _get_yaw(det)
        top_z = _get_top_z(det, center, size)

        graspable, pushable = _default_operability(role, state)

        mem_obj.update({
            "id": obj_id,
            "label": label,
            "last_pose_base": center,
            "size_m": size,
            "yaw_rad": yaw,
            "top_z": top_z,
            "visible": True,
            "graspable": graspable,
            "pushable": pushable,
            "last_seen_step": step_index,
            "track_id": det.get("track_id"),
            "object_ref": det.get("object_ref"),
            "tracking_ambiguous": det.get("tracking_ambiguous", False),
        })

    for obj in memory["objects"].values():
        if not obj.get("visible", False):
            obj["state"] = "missing" if obj.get("state") == "free" else obj.get("state")

    return memory


def update_from_scoped_detections(
    memory: Memory,
    detections: List[Obj],
    match_dist_m: float = 0.05,
    scoped_object_ids: Optional[Iterable[str]] = None,
    critical_object_ids: Optional[Iterable[str]] = None,
    observation_scope: str = "post_action",
    scene_revision: Optional[int] = None,
) -> Memory:
    """
    Update memory from a scoped observation after pick / push / place.

    Unlike update_from_detections(), this function does not turn every unseen
    free object into missing. Objects inside the requested scope that are not
    detected are kept as low-confidence, non-operable memory until a later
    observation confirms them again.
    """

    memory["step_index"] = int(memory.get("step_index", 0)) + 1
    step_index = memory["step_index"]
    memory, assignments = update_scene_tracks(
        memory,
        detections,
        scene_revision or step_index,
        mark_unseen_invisible=False,
    )
    memory["last_track_assignment"] = assignments
    scoped_ids: Set[str] = {str(value) for value in scoped_object_ids or []}
    critical_ids: Set[str] = {str(value) for value in critical_object_ids or []}
    seen_ids: Set[str] = set()
    matched_memory_ids: Set[str] = set()

    for obj_id in scoped_ids:
        obj = memory.get("objects", {}).get(obj_id)
        if obj is None:
            continue
        obj["visible"] = False
        obj["last_observation_scope"] = observation_scope

    for det in detections:
        label = _get_label(det)
        center = _get_center_base(det)
        size = _get_size_m(det)

        if label is None or center is None:
            continue

        obj_id = _find_match(
            memory, label, center, max_dist_m=match_dist_m,
            excluded_ids=matched_memory_ids,
        )

        if obj_id is None:
            obj_id = _make_object_id(label, memory["objects"])
            memory["objects"][obj_id] = {
                "id": obj_id,
                "label": label,
                "role": "unknown",
                "state": "free",
            }

        mem_obj = memory["objects"][obj_id]
        matched_memory_ids.add(str(obj_id))
        role = mem_obj.get("role", "unknown")
        state = mem_obj.get("state", "free")
        yaw = _get_yaw(det)
        top_z = _get_top_z(det, center, size)
        graspable, pushable = _default_operability(role, state)

        mem_obj.update({
            "id": obj_id,
            "label": label,
            "last_pose_base": center,
            "size_m": size,
            "yaw_rad": yaw,
            "top_z": top_z,
            "visible": True,
            "graspable": graspable,
            "pushable": pushable,
            "last_seen_step": step_index,
            "observation_confidence": 1.0,
            "last_observation_scope": observation_scope,
            "track_id": det.get("track_id"),
            "object_ref": det.get("object_ref"),
            "tracking_ambiguous": det.get("tracking_ambiguous", False),
        })
        mem_obj.pop("missing_observation_scope", None)
        seen_ids.add(str(obj_id))

    missing_scoped_ids = scoped_ids - seen_ids
    for obj_id in missing_scoped_ids:
        obj = memory.get("objects", {}).get(obj_id)
        if obj is None:
            continue
        previous_count = int(obj.get("unconfirmed_missing_count", 0))
        obj["visible"] = False
        obj["last_missing_step"] = step_index
        obj["unconfirmed_missing_count"] = previous_count + 1
        obj["missing_observation_scope"] = observation_scope
        obj["observation_confidence"] = 0.15 if obj_id in critical_ids else 0.35
        obj["graspable"] = False
        obj["pushable"] = False
        if obj_id in critical_ids:
            obj["state"] = "unconfirmed_missing"
        elif obj.get("state") == "missing":
            obj["state"] = "free"

    return memory


def set_role(
    memory: Memory,
    obj_id: str,
    role: str,
    state: Optional[str] = None,
) -> Memory:
    obj = memory["objects"][obj_id]
    obj["role"] = role

    if state is not None:
        obj["state"] = state

    _sync_track_role(memory, obj)
    _record_role_binding(memory, obj, role)

    if role in ["base", "structure"] or obj.get("state") in ["placed", "locked"]:
        obj["graspable"] = False
        obj["pushable"] = False

    return memory


def lock_object(memory: Memory, obj_id: str) -> Memory:
    obj = memory["objects"][obj_id]
    obj["state"] = "locked"
    obj["graspable"] = False
    obj["pushable"] = False
    _sync_track_role(memory, obj)
    return memory


def set_base(memory: Memory, obj_id: str) -> Memory:
    obj = memory["objects"][obj_id]

    obj["role"] = "base"
    obj["state"] = "locked"
    obj["graspable"] = False
    obj["pushable"] = False
    _sync_track_role(memory, obj)
    _record_role_binding(memory, obj, "base")

    memory["structure"]["base"] = obj_id
    memory["structure"]["current_top"] = obj_id
    memory["structure"]["top_center_base"] = obj.get("last_pose_base")
    memory["structure"]["top_z"] = obj.get("top_z")

    if obj_id not in memory["structure"]["placed_order"]:
        memory["structure"]["placed_order"].append(obj_id)

    return memory


def mark_placed(
    memory: Memory,
    obj_id: str,
    target_id: Optional[str] = None,
    result: str = "success",
) -> Memory:
    obj = memory["objects"][obj_id]

    obj["role"] = "structure"
    obj["state"] = "placed"
    obj["graspable"] = False
    obj["pushable"] = False
    _sync_track_role(memory, obj)
    _record_role_binding(memory, obj, "structure")

    if obj_id not in memory["structure"]["placed_order"]:
        memory["structure"]["placed_order"].append(obj_id)

    memory["structure"]["current_top"] = obj_id
    memory["structure"]["top_center_base"] = obj.get("last_pose_base")
    memory["structure"]["top_z"] = obj.get("top_z")

    memory["action_history"].append({
        "step_index": memory.get("step_index", 0),
        "action": "place",
        "object": obj_id,
        "target": target_id,
        "result": result,
    })

    return memory


def _record_role_binding(memory: Memory, obj: Obj, role: str) -> None:
    track_id = obj.get("track_id")
    if not track_id:
        return
    history = memory.setdefault("role_binding_history", [])
    previous = next((item for item in reversed(history) if item.get("role") == role), None)
    event = "role_rebound" if previous and previous.get("track_id") != track_id else "role_bound"
    history.append({
        "event": event,
        "role": role,
        "track_id": track_id,
        "object_ref": obj.get("object_ref"),
        "step_index": int(memory.get("step_index", 0)),
    })


def _sync_track_role(memory: Memory, obj: Obj) -> None:
    track = memory.get("tracks", {}).get(str(obj.get("track_id")))
    if track is not None:
        track["role"] = obj.get("role")
        track["state"] = obj.get("state")


def mark_pushed(
    memory: Memory,
    obj_id: str,
    direction_base: List[float],
    distance_m: float,
    reason: str = "clear_obstacle",
    result: str = "executed",
    observed_delta_m: Optional[float] = None,
    unblocked_target: Optional[bool] = None,
    affected_future_targets: Optional[List[str]] = None,
    affected_place_regions: Optional[List[str]] = None,
) -> Memory:
    obj = memory["objects"][obj_id]
    obj["role"] = "obstacle"
    obj["state"] = "free"
    expected_delta_m = float(distance_m)
    final_result = result
    if observed_delta_m is not None and observed_delta_m < 0.5 * expected_delta_m:
        final_result = "failure"

    memory["action_history"].append({
        "step_index": memory.get("step_index", 0),
        "action": "push",
        "object": obj_id,
        "direction_base": direction_base,
        "distance_m": distance_m,
        "expected_delta_m": expected_delta_m,
        "observed_delta_m": observed_delta_m,
        "reason": reason,
        "result": final_result,
        "unblocked_target": unblocked_target,
        "affected_future_targets": affected_future_targets or [],
        "affected_place_regions": affected_place_regions or [],
    })

    return memory


def get_current_top(memory: Memory) -> Optional[Obj]:
    obj_id = memory["structure"].get("current_top")
    if obj_id is None:
        return None
    return memory["objects"].get(obj_id)


def get_locked_object_ids(memory: Memory) -> List[str]:
    locked = []

    for obj_id, obj in memory["objects"].items():
        if obj.get("state") in ["locked", "placed"]:
            locked.append(obj_id)
        elif obj.get("role") in ["base", "structure"]:
            locked.append(obj_id)

    return locked


def is_locked(memory: Memory, obj_id: str) -> bool:
    return obj_id in get_locked_object_ids(memory)


def find_objects_by_label(memory: Memory, label: str, visible_only: bool = True) -> List[Obj]:
    out = []

    for obj in memory["objects"].values():
        if obj.get("label") != label:
            continue

        if visible_only and not obj.get("visible", False):
            continue

        out.append(obj)

    return out
