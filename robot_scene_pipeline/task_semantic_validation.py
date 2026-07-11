"""Schema and scene-binding validation for house and organization tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .geometry_relations import get_center, get_size
from .llm_stack_blocks import object_label_contains
from .vlm_task_policy import TASK_TYPES


class TaskSemanticValidationError(ValueError):
    """A VLM task contract or temporary binding requires replanning feedback."""

    def __init__(self, message: str, feedback: dict):
        super().__init__(message)
        self.feedback = feedback


def validate_task_contract(contract: dict, semantics_config: dict) -> dict:
    """Validate immutable task meaning without selecting scene object ids."""
    if not isinstance(contract, dict):
        _raise("unknown", [{"type": "contract_not_object"}])
    task_type = str(contract.get("task_type") or "")
    if task_type not in TASK_TYPES:
        _raise(task_type or "unknown", [{"type": "unsupported_task_type", "allowed": sorted(TASK_TYPES)}])
    if _contains_permanent_object_id(contract.get("goal_spec")):
        _raise(task_type, [{"type": "contract_must_not_bind_detection_object_id"}])
    goal_spec = contract.get("goal_spec")
    if not isinstance(goal_spec, dict):
        _raise(task_type, [{"type": "missing_goal_spec"}])
    if task_type == "build_house":
        errors = _validate_house_contract(goal_spec)
    else:
        errors = _validate_organize_contract(goal_spec, semantics_config)
    if errors:
        _raise(task_type, errors)
    validated = dict(contract)
    validated["schema_version"] = "task_contract_v1"
    validated["task_type"] = task_type
    validated["goal_spec"] = goal_spec
    validated["confidence"] = _confidence(contract.get("confidence"))
    if not str(contract.get("reason") or "").strip():
        _raise(task_type, [{"type": "missing_reason"}])
    return validated


def validate_grounded_task_plan(
    plan: dict, contract: dict, state: dict, scene_revision: int, semantics_config: dict,
) -> dict:
    """Validate temporary current-scene bindings; never repair VLM choices."""
    task_type = contract["task_type"]
    if not isinstance(plan, dict) or str(plan.get("task_type")) != task_type:
        _raise(task_type, [{"type": "grounded_plan_task_type_mismatch"}])
    if int(plan.get("scene_revision", -1)) != int(scene_revision):
        _raise(task_type, [{"type": "stale_scene_revision", "expected": scene_revision, "actual": plan.get("scene_revision")}])
    if task_type == "build_house":
        errors, normalized = _validate_house_plan(plan, contract, state, semantics_config)
    else:
        errors, normalized = _validate_organize_plan(plan, contract, state, semantics_config)
    if errors:
        _raise(task_type, errors)
    normalized.update({"schema_version": "grounded_task_plan_v1", "task_type": task_type, "scene_revision": int(scene_revision)})
    return normalized


def object_matches_role(obj: dict, role: dict) -> bool:
    """Check factual label/category/size requirements without assigning an object."""
    requirements = role.get("requirements") or {}
    category = str(requirements.get("category") or "").lower()
    label = str(obj.get("label") or "").lower()
    category_tokens = [token for token in category.replace("_", " ").split() if token and token != "block"]
    if category and not all(token in label for token in category_tokens):
        return False
    size_range = requirements.get("size_range_m") or {}
    size = get_size(obj)
    if size_range and size is None:
        return False
    for index, axis in enumerate(("x", "y", "z")):
        bounds = size_range.get(axis)
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or not bounds[0] <= size[index] <= bounds[1]):
            return False
    return True


def temporary_binding_feedback(
    role: dict, previous_object_id: Any, state: dict, scene_revision: int,
) -> dict:
    """Provide eligible facts for a VLM replacement choice without selecting one."""
    matches = [
        {"object_id": obj.get("id"), "label": obj.get("label")}
        for obj in _objects(state) if object_matches_role(obj, role)
    ]
    return {
        "validation_stage": "object_binding_validation", "passed": False,
        "reason": "temporary_role_assignment_invalid", "role_id": role.get("role_id"),
        "previous_object_id": previous_object_id, "scene_revision": int(scene_revision),
        "available_matching_objects": matches, "required_response": "select_a_current_matching_object",
    }


def _validate_house_contract(goal: dict) -> List[dict]:
    roles = goal.get("roles") or goal.get("required_roles")
    if not isinstance(roles, list):
        return [{"type": "missing_roles"}]
    role_ids = [item.get("role_id") for item in roles if isinstance(item, dict)]
    required = {"left_support", "right_support", "roof"}
    errors = [{"type": "missing_required_role", "role_id": role} for role in sorted(required - set(role_ids))]
    if len(role_ids) != len(set(role_ids)):
        errors.append({"type": "duplicate_role_id"})
    relations = goal.get("required_relations")
    if not isinstance(relations, list):
        errors.append({"type": "missing_required_relations"})
    return errors


def _validate_organize_contract(goal: dict, config: dict) -> List[dict]:
    defaults = config.get("organize_defaults", {})
    for key, value in defaults.items():
        goal.setdefault(key, value)
    errors = []
    if goal.get("grouping_key") not in {"color", "shape", "category"}:
        errors.append({"type": "unsupported_grouping_key"})
    if goal.get("layout_type") not in {"rows", "columns", "grid"}:
        errors.append({"type": "unsupported_layout_type"})
    if bool(goal.get("allow_stacking")):
        errors.append({"type": "organize_task_must_not_require_stacking"})
    return errors


def _validate_house_plan(plan: dict, contract: dict, state: dict, _config: dict) -> Tuple[List[dict], dict]:
    roles = _house_roles(contract)
    bindings = plan.get("role_assignments") or plan.get("role_bindings")
    if not isinstance(bindings, list):
        return [{"type": "missing_role_assignments"}], {}
    by_role, object_ids, errors = {}, set(), []
    object_map = {str(obj.get("id")): obj for obj in _objects(state)}
    for binding in bindings:
        role_id = binding.get("role_id") if isinstance(binding, dict) else None
        object_id = binding.get("selected_object_id", binding.get("object_id")) if isinstance(binding, dict) else None
        role = roles.get(role_id)
        obj = object_map.get(str(object_id))
        if role is None:
            errors.append({"type": "unknown_role", "role_id": role_id}); continue
        if obj is None:
            errors.append(temporary_binding_feedback(role, object_id, state, plan.get("scene_revision", 0))); continue
        if object_id in object_ids:
            errors.append({"type": "object_assigned_to_multiple_roles", "object_id": object_id}); continue
        if not object_matches_role(obj, role):
            errors.append({"type": "selected_object_no_longer_matches_role", "role_id": role_id, "object_id": object_id}); continue
        if not _binding_matches_object(binding, obj):
            errors.append({"type": "object_binding_grounding_mismatch", "role_id": role_id, "object_id": object_id}); continue
        object_ids.add(object_id)
        by_role[role_id] = _normalized_binding(binding, obj, role)
    for role_id in roles:
        if role_id not in by_role:
            errors.append({"type": "missing_required_role", "role_id": role_id})
    relations = contract["goal_spec"].get("required_relations", [])
    errors.extend(_validate_house_relations(relations, roles))
    steps = plan.get("assembly_steps", [])
    errors.extend(_validate_assembly_steps(steps))
    return errors, {"role_assignments": list(by_role.values()), "assembly_steps": steps}


def _validate_organize_plan(plan: dict, contract: dict, state: dict, _config: dict) -> Tuple[List[dict], dict]:
    groups, regions = plan.get("groups"), plan.get("target_regions")
    if not isinstance(groups, list) or not isinstance(regions, list):
        return [{"type": "missing_groups_or_target_regions"}], {}
    objects = _objects(state); object_map = {str(obj.get("id")): obj for obj in objects}; assigned = set(); errors = []
    region_map = {item.get("region_id"): item for item in regions if isinstance(item, dict)}
    for group in groups:
        region = region_map.get(group.get("target_region_id"))
        if region is None:
            errors.append({"type": "unknown_target_region", "group_id": group.get("group_id")}); continue
        for object_id in group.get("object_ids", []):
            obj = object_map.get(str(object_id))
            if obj is None:
                errors.append({"type": "unknown_object_id", "object_id": object_id}); continue
            if object_id in assigned:
                errors.append({"type": "object_assigned_to_multiple_groups", "object_id": object_id}); continue
            if not object_label_contains(obj, str(group.get("group_value") or "")):
                errors.append({"type": "group_value_does_not_match_label", "object_id": object_id}); continue
            assigned.add(object_id)
    required_ids = {obj.get("id") for obj in objects}
    for object_id in sorted(required_ids - assigned):
        errors.append({"type": "object_missing_from_groups", "object_id": object_id})
    errors.extend(_validate_regions(regions, objects, contract["goal_spec"].get("minimum_spacing_m", 0.015), state))
    spacing = float(contract["goal_spec"].get("minimum_spacing_m", 0.015))
    for group in groups:
        region = region_map.get(group.get("target_region_id")) or {}
        box = region.get("bounds_base_m") or {}
        capacity = (float(box.get("xmax", 0)) - float(box.get("xmin", 0))) * (float(box.get("ymax", 0)) - float(box.get("ymin", 0)))
        needed = 0.0
        for object_id in group.get("object_ids", []):
            obj = object_map.get(str(object_id)); size = get_size(obj) if obj else None
            if size: needed += (max(size[0], size[1]) + spacing) ** 2
        if needed > capacity:
            errors.append({"type": "target_region_capacity_insufficient", "region_id": group.get("target_region_id")})
    return errors, {"groups": groups, "target_regions": regions}


def _house_roles(contract: dict) -> Dict[str, dict]:
    roles = contract["goal_spec"].get("roles") or contract["goal_spec"].get("required_roles") or []
    return {item["role_id"]: item for item in roles if isinstance(item, dict) and item.get("role_id")}


def _validate_house_relations(relations: list, roles: dict) -> List[dict]:
    errors, support_edges = [], []
    for relation in relations:
        subject, obj = relation.get("subject_role"), relation.get("object_role")
        if subject and subject not in roles: errors.append({"type": "relation_references_unknown_role", "role_id": subject})
        if obj and obj not in roles: errors.append({"type": "relation_references_unknown_role", "role_id": obj})
        if relation.get("type") == "supports":
            if subject == obj: errors.append({"type": "self_support_relation", "role_id": subject})
            support_edges.append((subject, obj))
    if _has_cycle(support_edges): errors.append({"type": "support_relation_cycle"})
    roof_supports = [edge[0] for edge in support_edges if edge[1] == "roof"]
    if len(set(roof_supports)) < 2: errors.append({"type": "roof_requires_two_distinct_supports"})
    return errors


def _validate_assembly_steps(steps: list) -> List[dict]:
    by_id = {item.get("step_id"): item for item in steps if isinstance(item, dict)}
    edges = [(dependency, step_id) for step_id, item in by_id.items() for dependency in item.get("prerequisites", [])]
    errors = [{"type": "invalid_assembly_dependency", "step_id": step, "missing_prerequisite": dep} for dep, step in edges if dep not in by_id]
    if _has_cycle(edges): errors.append({"type": "assembly_dependency_cycle"})
    roof = next((item for item in by_id.values() if item.get("role_id") == "roof"), None)
    support_steps = {
        step_id for step_id, item in by_id.items()
        if item.get("role_id") in {"left_support", "right_support"}
    }
    if roof and not support_steps.issubset(set(roof.get("prerequisites", []))):
        errors.append({"type": "roof_step_missing_support_prerequisites"})
    return errors


def _validate_regions(regions: list, objects: list, spacing: float, state: dict) -> List[dict]:
    errors = []; bounds = state.get("table_bounds") or state.get("workspace_bounds")
    for index, region in enumerate(regions):
        box = region.get("bounds_base_m") or {}
        if not _valid_box(box) or (bounds and not _contains(bounds, box)):
            errors.append({"type": "invalid_or_outside_workspace_region", "region_id": region.get("region_id")})
        for other in regions[index + 1:]:
            if _boxes_overlap(box, other.get("bounds_base_m") or {}): errors.append({"type": "target_regions_overlap", "region_id": region.get("region_id")})
        capacity = (box.get("xmax", 0) - box.get("xmin", 0)) * (box.get("ymax", 0) - box.get("ymin", 0))
        if capacity <= 0: continue
        required_areas = [(max(get_size(obj)[:2]) + spacing) ** 2 for obj in objects if get_size(obj)]
        if required_areas and capacity < min(required_areas):
            errors.append({"type": "target_region_capacity_insufficient", "region_id": region.get("region_id")})
    return errors


def _binding_matches_object(binding: dict, obj: dict) -> bool:
    if str(binding.get("observed_label") or "").lower() != str(obj.get("label") or "").lower(): return False
    expected, actual = binding.get("geometry_center_base_m"), get_center(obj)
    return isinstance(expected, list) and actual is not None and len(expected) >= 3 and math.dist(expected[:3], actual[:3]) <= 0.005


def _normalized_binding(binding: dict, obj: dict, role: dict) -> dict:
    return {"role_id": role["role_id"], "selected_object_id": obj.get("id"), "observed_label": obj.get("label"), "geometry_center_base_m": get_center(obj), "assignment_status": "temporary", "replaceable": bool(role.get("replaceable", True))}


def _objects(state: dict) -> List[dict]:
    return [obj for obj in state.get("objects", []) if isinstance(obj, dict) and not obj.get("is_workspace")]


def _contains(outer: dict, inner: dict) -> bool:
    return (
        float(outer["xmin"]) <= float(inner["xmin"])
        and float(outer["xmax"]) >= float(inner["xmax"])
        and float(outer["ymin"]) <= float(inner["ymin"])
        and float(outer["ymax"]) >= float(inner["ymax"])
    )


def _valid_box(box: dict) -> bool:
    required = ("xmin", "xmax", "ymin", "ymax")
    return (
        all(key in box for key in required)
        and float(box["xmin"]) < float(box["xmax"])
        and float(box["ymin"]) < float(box["ymax"])
    )


def _boxes_overlap(first: dict, second: dict) -> bool:
    if not _valid_box(first) or not _valid_box(second):
        return False
    return (
        min(first["xmax"], second["xmax"]) > max(first["xmin"], second["xmin"])
        and min(first["ymax"], second["ymax"]) > max(first["ymin"], second["ymin"])
    )


def _has_cycle(edges: Iterable[Tuple[Any, Any]]) -> bool:
    graph = {}
    for parent, child in edges:
        if parent and child:
            graph.setdefault(parent, set()).add(child)
    visited, active = set(), set()

    def visit(node):
        if node in active:
            return True
        if node in visited:
            return False
        visited.add(node)
        active.add(node)
        result = any(visit(child) for child in graph.get(node, ()))
        active.remove(node)
        return result

    return any(visit(node) for node in graph)


def _contains_permanent_object_id(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in {"object_id", "selected_object_id"} or _contains_permanent_object_id(item)
            for key, item in value.items()
        )
    return any(_contains_permanent_object_id(item) for item in value) if isinstance(value, list) else False


def _confidence(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        _raise("unknown", [{"type": "invalid_confidence"}])
    if not 0 <= value <= 1:
        _raise("unknown", [{"type": "invalid_confidence"}])
    return value


def _raise(task_type: str, errors: List[dict]) -> None:
    raise TaskSemanticValidationError(
        "Task semantic validation failed.",
        {"validation_stage": "task_semantic_validation", "passed": False, "task_type": task_type, "errors": errors},
    )
