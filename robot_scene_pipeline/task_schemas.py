"""Authoritative JSON Schemas shared by VLM formatting and task validators."""

from __future__ import annotations

from typing import Dict

from .house_task_definition import HOUSE_ROLE_IDS, HOUSE_STRUCTURE_VARIANT


ROLE_ID_SCHEMA = {"type": "string", "enum": list(HOUSE_ROLE_IDS)}
VECTOR3_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
QUATERNION_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}

ORIENTATION_OBSERVATION_SCHEMA = {
    "type": "object",
    "required": ["role_id", "selected_object_id", "confidence", "flip_required", "yaw_adjustment_required", "reason"],
    "properties": {
        "role_id": {"type": "string", "enum": ["roof", "triangle_top"]},
        "selected_object_id": {"type": ["integer", "string"]},
        "shape": {"type": "string", "enum": ["concave_rectangle", "rectangle", "triangle"]},
        "visible_face": {"type": "string", "enum": ["front", "back", "side", "unknown"]},
        "opening_direction_image": {"type": "string", "enum": ["up", "down", "left", "right", "diagonal", "unknown"]},
        "straight_edge_direction_image": {"type": "string", "enum": ["up", "down", "left", "right", "diagonal", "unknown"]},
        "support_legs_visible": {"type": "boolean"},
        "pose_state": {"type": "string", "enum": ["upright", "upside_down", "side_lying", "flat", "unknown"]},
        "apex_direction_image": {"type": "string", "enum": ["up", "down", "left", "right", "diagonal", "unknown"]},
        "apex_ambiguous": {"type": "boolean"},
        "right_angle_vertex": {"type": ["integer", "null"]},
        "apex_vertex": {"type": ["integer", "null"]},
        "upward_vertex": {"type": ["integer", "null"]},
        "flip_required": {"type": "boolean"},
        "yaw_adjustment_required": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "additionalProperties": True,
}

TASK_CONTRACT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "task_type", "goal_spec", "reason", "confidence"],
    "properties": {
        "schema_version": {"const": "task_contract_v1"},
        "task_type": {"type": "string", "enum": ["build_house", "organize_blocks"]},
        "goal_spec": {
            "type": "object",
            "properties": {
                "structure_variant": {"const": HOUSE_STRUCTURE_VARIANT},
                "roles": {
                    "type": "array", "minItems": 6, "maxItems": 6,
                    "items": {
                        "type": "object", "required": ["role_id", "requirements"],
                        "properties": {"role_id": ROLE_ID_SCHEMA, "requirements": {"type": "object"}},
                        "additionalProperties": True,
                    },
                },
                "required_relations": {"type": "array"},
            },
            "additionalProperties": True,
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

ROLE_ASSIGNMENT_SCHEMA = {
    "type": "object",
    "required": ["role_id", "selected_object_id", "observed_label", "geometry_center_base_m"],
    "properties": {
        "role_id": ROLE_ID_SCHEMA,
        "selected_object_id": {"type": ["integer", "string"]},
        "observed_label": {"type": "string"},
        "geometry_center_base_m": VECTOR3_SCHEMA,
        "assignment_status": {"type": "string", "enum": ["temporary"]},
        "replaceable": {"type": "boolean"},
    },
    "additionalProperties": True,
}

GROUNDED_HOUSE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "task_type", "scene_revision", "role_assignments", "assembly_steps", "orientation_observations"],
    "properties": {
        "schema_version": {"const": "grounded_task_plan_v1"},
        "task_type": {"const": "build_house"},
        "scene_revision": {"type": "integer"},
        "role_assignments": {"type": "array", "items": ROLE_ASSIGNMENT_SCHEMA, "minItems": 6, "maxItems": 6},
        "assembly_steps": {"type": "array", "minItems": 6, "maxItems": 6},
        "orientation_observations": {"type": "array", "items": ORIENTATION_OBSERVATION_SCHEMA, "minItems": 2},
    },
    "additionalProperties": True,
}

GROUNDED_ORGANIZE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "task_type", "scene_revision", "groups", "target_regions"],
    "properties": {
        "schema_version": {"const": "grounded_task_plan_v1"},
        "task_type": {"const": "organize_blocks"},
        "scene_revision": {"type": "integer"},
        "groups": {"type": "array"},
        "target_regions": {"type": "array"},
    },
    "additionalProperties": True,
}

TASK_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type"],
    "properties": {
        "strategy_id": {"type": "string"},
        "action_type": {"type": "string", "enum": ["pick_place", "pick_reorient_place", "nudge", "pick_away", "reobserve", "stop"]},
        "selected_object_ref": {"type": "string"},
        "selected_track_id": {"type": "string"},
        "selected_object_id": {"type": ["integer", "string"]},
        "role_id": ROLE_ID_SCHEMA,
        "group_id": {"type": "string"},
        "target_region_id": {"type": "string"},
        "object_label": {"type": "string"},
        "object_center_base_m": VECTOR3_SCHEMA,
        "scene_revision": {"type": "integer"},
        "target_pose_base": {
            "type": "object",
            "properties": {
                "position_m": VECTOR3_SCHEMA,
                "position": VECTOR3_SCHEMA,
                "yaw_rad": {"type": "number"},
                "orientation_xyzw": QUATERNION_SCHEMA,
            },
            "additionalProperties": False,
        },
        "source_pose_base": {
            "type": "object",
            "properties": {"position": VECTOR3_SCHEMA, "orientation_xyzw": QUATERNION_SCHEMA},
            "additionalProperties": False,
        },
        "reorientation_plan": {
            "type": "object",
            "properties": {
                "required": {"type": "boolean"},
                "rotation_type": {"type": "string", "enum": ["flip", "yaw_only"]},
                "rotation_axis_tool": VECTOR3_SCHEMA,
                "rotation_angle_deg": {"type": "number"},
                "intermediate_pose_count": {"type": "integer"},
                "perform_above_safe_height": {"type": "boolean"},
            },
            "additionalProperties": True,
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": True,
}


def schema_for_policy(policy_kind: str, task_type: str = "") -> Dict[str, object]:
    """Return the exact schema supplied to Ollama for one task-policy call."""
    if policy_kind == "task_contract":
        return TASK_CONTRACT_SCHEMA
    if policy_kind == "grounded_task_plan":
        return GROUNDED_HOUSE_PLAN_SCHEMA if task_type == "build_house" else GROUNDED_ORGANIZE_PLAN_SCHEMA
    return TASK_ACTION_SCHEMA


def validate_against_schema(value: object, schema: dict, path: str = "$") -> list:
    """Validate the schema subset used by task messages without an extra dependency."""
    errors = []
    expected_type = schema.get("type")
    allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
    if expected_type and not any(_matches_type(value, item) for item in allowed_types):
        return [{"type": "json_schema_type_mismatch", "path": path, "expected": expected_type}]
    if "const" in schema and value != schema["const"]:
        errors.append({"type": "json_schema_const_mismatch", "path": path, "expected": schema["const"]})
    if "enum" in schema and value not in schema["enum"]:
        errors.append({"type": "json_schema_enum_mismatch", "path": path, "allowed": schema["enum"]})
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append({"type": "json_schema_missing_required", "path": "{}.{}".format(path, key)})
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_against_schema(item, properties[key], "{}.{}".format(path, key)))
            elif schema.get("additionalProperties") is False:
                errors.append({"type": "json_schema_additional_property", "path": "{}.{}".format(path, key)})
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            errors.append({"type": "json_schema_too_few_items", "path": path})
        if schema.get("maxItems") is not None and len(value) > int(schema["maxItems"]):
            errors.append({"type": "json_schema_too_many_items", "path": path})
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(validate_against_schema(item, item_schema, "{}[{}]".format(path, index)))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if schema.get("minimum") is not None and value < schema["minimum"]:
            errors.append({"type": "json_schema_below_minimum", "path": path})
        if schema.get("maximum") is not None and value > schema["maximum"]:
            errors.append({"type": "json_schema_above_maximum", "path": path})
    return errors


def _matches_type(value: object, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)
