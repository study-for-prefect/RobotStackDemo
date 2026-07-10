"""Small context helpers for stack-demo push clearing."""

import copy
import math
from typing import Any, Iterable, List, Optional


def protected_objects(
    current_state: dict,
    base_id: Any,
    previous_locked_stack: Optional[dict],
    protected_object_ids: Optional[Iterable[Any]] = None,
) -> List[dict]:
    protected_ids = {str(value) for value in protected_object_ids or []}
    if not protected_ids:
        protected_ids = {str(base_id)}
        if previous_locked_stack:
            for obj in previous_locked_stack.get("stack_objects", []):
                if obj.get("id") is not None:
                    protected_ids.add(str(obj["id"]))
    output = []
    for obj in current_state.get("objects", []):
        if str(obj.get("id")) in protected_ids:
            output.append(obj)
        elif obj.get("role") in ("base", "structure") or obj.get("state") in ("locked", "placed"):
            output.append(obj)
    return output


def protected_stack_templates(
    current_state: dict,
    base_id: Any,
    previous_locked_stack: Optional[dict],
    base_template: Optional[dict] = None,
) -> List[dict]:
    templates = []
    if isinstance(base_template, dict):
        templates.append(copy.deepcopy(base_template))
    elif base_id is not None:
        for obj in current_state.get("objects", []):
            if str(obj.get("id")) == str(base_id):
                templates.append(copy.deepcopy(obj))
                break
    if previous_locked_stack:
        templates.extend(copy.deepcopy(previous_locked_stack.get("stack_objects", [])))
    return templates


def observed_push_delta_m(before_obstacle: dict, after_state: dict) -> Optional[float]:
    before_center = before_obstacle.get("geometry_center_m")
    if not isinstance(before_center, list) or len(before_center) < 2:
        return None
    after = None
    for obj in after_state.get("objects", []):
        if str(obj.get("id")) == str(before_obstacle.get("id")):
            after = obj
            break
    if after is None:
        return None
    after_center = after.get("geometry_center_m")
    if not isinstance(after_center, list) or len(after_center) < 2:
        return None
    return math.hypot(float(after_center[0]) - float(before_center[0]), float(after_center[1]) - float(before_center[1]))
