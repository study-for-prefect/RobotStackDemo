"""Resolve concise house role references into factual internal task bindings."""

from __future__ import annotations

from typing import Any, List

from .geometry_relations import get_center
from .house_task_definition import canonical_house_assembly_steps


def expand_house_grounded_plan(plan: dict, state: dict) -> dict:
    """Resolve VLM refs and inject observed facts plus canonical assembly steps."""
    objects = [
        obj for obj in state.get("objects", [])
        if isinstance(obj, dict) and not obj.get("is_workspace")
    ]
    assignments = []
    role_object_ids = {}
    for binding in plan.get("role_bindings", []):
        obj = _object_for_ref_and_track(
            objects, binding.get("object_ref"), binding.get("track_id"),
        )
        object_id = None if obj is None else obj.get("id")
        role_object_ids[binding.get("role_id")] = object_id
        assignments.append({
            "role_id": binding.get("role_id"),
            "selected_object_id": object_id,
            "observed_label": None if obj is None else obj.get("label"),
            "geometry_center_base_m": None if obj is None else get_center(obj),
            "assignment_status": "temporary",
            "replaceable": True,
            "object_ref": binding.get("object_ref"),
            "track_id": binding.get("track_id"),
            "confidence": binding.get("confidence"),
        })
    observations = []
    for observation in plan.get("orientation_observations", []):
        normalized = dict(observation)
        normalized["selected_object_id"] = role_object_ids.get(observation.get("role_id"))
        observations.append(normalized)
    return {
        "schema_version": "grounded_task_plan_v1",
        "task_type": "build_house",
        "scene_revision": plan.get("scene_revision"),
        "role_assignments": assignments,
        "assembly_steps": canonical_house_assembly_steps(),
        "orientation_observations": observations,
        "reason": plan.get("reason"),
        "confidence": plan.get("confidence"),
    }


def _object_for_ref_and_track(objects: List[dict], object_ref: Any, track_id: Any) -> Any:
    by_ref = next((obj for obj in objects if str(obj.get("object_ref")) == str(object_ref)), None)
    by_track = next((obj for obj in objects if str(obj.get("track_id")) == str(track_id)), None)
    return by_ref if by_ref is not None and by_ref is by_track else None
