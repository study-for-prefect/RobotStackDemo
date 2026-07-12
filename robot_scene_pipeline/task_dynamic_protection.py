"""Rebuild protected task structure from each current scene revision."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .geometry_relations import get_center, get_size


ROLE_REQUIRED_PREDICATES = {
    "left_support_lower": {"left_support_lower.on_table"},
    "right_support_lower": {"right_support_lower.on_table"},
    "left_support_upper": {
        "left_support_lower.on_table",
        "left_support_lower.supports.left_support_upper",
        "left_column.vertical_aligned",
    },
    "right_support_upper": {
        "right_support_lower.on_table",
        "right_support_lower.supports.right_support_upper",
        "right_column.vertical_aligned",
    },
    "roof": {
        "left_support_lower.supports.left_support_upper",
        "right_support_lower.supports.right_support_upper",
        "columns.height_aligned",
        "left_support_upper.supports.roof",
        "right_support_upper.supports.roof",
        "roof.bridges.upper_supports",
        "roof.correct_face_up",
        "roof.opening_down",
        "roof.straight_edge_up",
        "roof.orientation_correct",
    },
    "triangle_top": {
        "roof.orientation_correct",
        "roof.supports.triangle_top",
        "triangle_top.correct_face",
        "triangle_top.apex_up",
        "triangle_top.not_side_lying",
        "triangle_top.centered_on_roof",
    },
}


def derive_dynamic_protection(
    scene: dict,
    task_contract: dict,
    task_goal_progress: dict,
    role_assignment: Optional[dict],
) -> dict:
    """Protect only roles whose current geometry predicates remain satisfied."""
    output: Dict[str, Any] = {
        "schema_version": "dynamic_task_protection_v1",
        "scene_revision": scene.get("scene_revision"),
        "protected_object_ids": [],
        "protected_regions": [],
        "protected_relations": [],
        "protected_roles": [],
    }
    if task_contract.get("task_type") != "build_house":
        return output
    satisfied = set(task_goal_progress.get("satisfied_predicates") or [])
    observations = task_goal_progress.get("role_observations") or {}
    assignment = role_assignment or {}
    for role_id, required in ROLE_REQUIRED_PREDICATES.items():
        if not required.issubset(satisfied):
            continue
        observation = observations.get(role_id) or {}
        object_id = observation.get("observed_object_id")
        if object_id is None:
            object_id = (assignment.get(role_id) or {}).get("current_object_id")
        obj = _object_by_id(scene, object_id)
        region = _protected_region(obj, observation)
        role_record = {
            "role_id": role_id,
            "current_object_id": object_id,
            "protected_region": region,
        }
        output["protected_roles"].append(role_record)
        if object_id is not None:
            output["protected_object_ids"].append(object_id)
        if region is not None:
            output["protected_regions"].append({"role_id": role_id, **region})
    protected_role_ids = {item["role_id"] for item in output["protected_roles"]}
    for relation in task_contract.get("goal_spec", {}).get("required_relations", []):
        roles = _relation_roles(relation)
        if roles and roles.issubset(protected_role_ids):
            output["protected_relations"].append(dict(relation))
    return output


def _object_by_id(scene: dict, object_id: Any) -> Optional[dict]:
    return next(
        (obj for obj in scene.get("objects", []) if str(obj.get("id")) == str(object_id)),
        None,
    )


def _protected_region(obj: Optional[dict], observation: dict) -> Optional[dict]:
    center = get_center(obj) if obj else observation.get("center_base_m")
    size = get_size(obj) if obj else None
    if center is None:
        return None
    region: Dict[str, Any] = {
        "center_base_m": list(center[:3]),
        "position_tolerance_m": 0.015,
    }
    if size is not None:
        region["dimensions_m"] = list(size[:3])
    return region


def _relation_roles(relation: dict) -> set:
    roles = {
        relation.get("subject_role"),
        relation.get("object_role"),
        *(relation.get("object_roles") or []),
    }
    return {str(role) for role in roles if role}
