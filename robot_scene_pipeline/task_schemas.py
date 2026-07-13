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

BUILD_HOUSE_CONTRACT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "task_type", "goal_spec", "reason", "confidence"],
    "properties": {
        "schema_version": {"const": "task_contract_v1"},
        "task_type": {"const": "build_house"},
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

ORGANIZE_BLOCKS_CONTRACT_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "task_type", "goal_spec", "reason", "confidence"],
    "properties": {
        "schema_version": {"const": "task_contract_v1"},
        "task_type": {"const": "organize_blocks"},
        "goal_spec": {
            "type": "object",
            "required": ["grouping_key", "layout_type", "include_scope", "allow_stacking"],
            "properties": {
                "grouping_key": {"const": "color"},
                "layout_type": {"type": "string", "enum": ["rows", "columns", "regions"]},
                "include_scope": {"const": "all_detected_blocks"},
                "allow_stacking": {"const": False},
                "minimum_spacing_m": {"type": "number"},
                "alignment_tolerance_m": {"type": "number"},
                "row_color_order": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["red", "green", "blue", "yellow"]},
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

TASK_CONTRACT_SCHEMAS = {
    "build_house": BUILD_HOUSE_CONTRACT_SCHEMA,
    "organize_blocks": ORGANIZE_BLOCKS_CONTRACT_SCHEMA,
}
# Compatibility alias for code that validates after task_type is already known.
TASK_CONTRACT_SCHEMA = BUILD_HOUSE_CONTRACT_SCHEMA

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

HOUSE_ROLE_BINDING_SCHEMA = {
    "type": "object",
    "required": ["role_id", "object_ref", "track_id", "confidence"],
    "properties": {
        "role_id": ROLE_ID_SCHEMA,
        "object_ref": {"type": "string"},
        "track_id": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

HOUSE_ORIENTATION_BINDING_SCHEMA = {
    "type": "object",
    "required": ["role_id", "object_ref", "track_id", "confidence"],
    "properties": {
        **ORIENTATION_OBSERVATION_SCHEMA["properties"],
        "object_ref": {"type": "string"},
        "track_id": {"type": "string"},
    },
    "additionalProperties": True,
}

GROUNDED_HOUSE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "scene_revision", "role_bindings", "orientation_observations", "reason", "confidence"],
    "properties": {
        "schema_version": {"const": "grounded_house_plan_v1"},
        "scene_revision": {"type": "integer"},
        "role_bindings": {"type": "array", "items": HOUSE_ROLE_BINDING_SCHEMA, "minItems": 6, "maxItems": 6},
        "orientation_observations": {"type": "array", "items": HOUSE_ORIENTATION_BINDING_SCHEMA},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

LEGACY_GROUNDED_HOUSE_PLAN_SCHEMA = {
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

ORGANIZE_GROUP_SCHEMA = {
    "type": "object",
    "required": ["group_id", "group_value", "object_ids", "target_region_id"],
    "properties": {
        "group_id": {"type": "string"},
        "group_value": {"type": "string"},
        "object_ids": {"type": "array", "items": {"type": ["integer", "string"]}},
        "target_region_id": {"type": "string"},
        "minimum_required_count": {"type": "integer", "minimum": 1},
    },
    "additionalProperties": False,
}

ORGANIZE_REGION_SCHEMA = {
    "type": "object",
    "required": ["region_id", "bounds_base_m"],
    "properties": {
        "region_id": {"type": "string"},
        "bounds_base_m": {
            "type": "object",
            "required": ["xmin", "xmax", "ymin", "ymax"],
            "properties": {
                "xmin": {"type": "number"}, "xmax": {"type": "number"},
                "ymin": {"type": "number"}, "ymax": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

GROUNDED_ORGANIZE_PLAN_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "task_type", "scene_revision", "groups", "target_regions"],
    "properties": {
        "schema_version": {"const": "grounded_task_plan_v1"},
        "task_type": {"const": "organize_blocks"},
        "scene_revision": {"type": "integer"},
        "groups": {"type": "array", "items": ORGANIZE_GROUP_SCHEMA, "minItems": 1},
        "target_regions": {"type": "array", "items": ORGANIZE_REGION_SCHEMA, "minItems": 1},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

TASK_ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type"],
    "oneOf": [
        {
            "properties": {
                "action_type": {"type": "string", "enum": ["pick_place", "pick_reorient_place"]},
            },
            "required": [
                "selected_object_id", "object_label", "object_center_base_m",
                "scene_revision", "target_pose_base",
            ],
        },
        {
            "properties": {
                "action_type": {"type": "string", "enum": ["nudge", "pick_away"]},
            },
        },
        {
            "properties": {
                "action_type": {"type": "string", "enum": ["reobserve", "stop"]},
            },
        },
    ],
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

# Ollama/llama.cpp currently does not reliably enforce required fields nested
# below oneOf.  Keep the semantic schema above for code-side branch validation,
# but give constrained generation a flat set of required keys.  Fields that do
# not apply to reobserve/stop or to the other task family are explicitly null.
TASK_ACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "strategy_id", "action_type", "selected_object_ref", "selected_track_id",
        "selected_object_id", "role_id", "group_id", "target_region_id",
        "object_label", "object_center_base_m", "scene_revision",
        "target_pose_base", "reason", "confidence",
    ],
    "properties": {
        "strategy_id": {"type": "string"},
        "action_type": TASK_ACTION_SCHEMA["properties"]["action_type"],
        "selected_object_ref": {"type": ["string", "null"]},
        "selected_track_id": {"type": ["string", "null"]},
        "selected_object_id": {"type": ["integer", "string", "null"]},
        "role_id": {"type": ["string", "null"], "enum": list(HOUSE_ROLE_IDS) + [None]},
        "group_id": {"type": ["string", "null"]},
        "target_region_id": {"type": ["string", "null"]},
        "object_label": {"type": ["string", "null"]},
        "object_center_base_m": {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 3, "maxItems": 3,
        },
        "scene_revision": {"type": "integer"},
        "target_pose_base": {
            "type": ["object", "null"],
            "properties": TASK_ACTION_SCHEMA["properties"]["target_pose_base"]["properties"],
            "required": ["position_m"],
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

VLM_ACTION_SCHEMA = {
    "type": "object",
    "required": [
        "action_type", "scene_problem", "predicted_scene_benefit",
        "risk_assessment", "reason", "confidence",
    ],
    "properties": {
        "strategy_id": {"type": "string"},
        "action_type": {"type": "string", "enum": ["pick", "nudge", "pick_away", "reobserve", "stop"]},
        "scene_problem": {"type": "string"},
        "object_ref": {"type": ["string", "null"]},
        "object_track_id": {"type": ["string", "null"]},
        "object_id": {"type": ["integer", "string", "null"]},
        "object_label": {"type": ["string", "null"]},
        "object_center_base_m": {"type": ["array", "null"], "items": {"type": "number"}},
        "target_object_ref": {"type": ["string", "null"]},
        "target_object_track_id": {"type": ["string", "null"]},
        "target_object_id": {"type": ["integer", "string", "null"]},
        "target_object_label": {"type": ["string", "null"]},
        "target_object_center_base_m": {"type": ["array", "null"], "items": {"type": "number"}},
        "contact_side": {"type": ["string", "null"]},
        "direction_base": {"type": ["array", "null"], "items": {"type": "number"}},
        "distance_m": {"type": ["number", "null"]},
        "gripper_yaw_rad": {"type": ["number", "null"]},
        "safe_place_center_base_m": {"type": ["array", "null"], "items": {"type": "number"}},
        "predicted_scene_benefit": {"type": "string"},
        "risk_assessment": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "alternative_actions": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": True,
}

# As with TASK_ACTION_OUTPUT_SCHEMA, constrained generation needs top-level
# required keys because required fields inside conditional branches are not
# reliably honored by the local Ollama/llama.cpp build.  Non-applicable motion
# fields are null for pick/reobserve/stop and are checked semantically afterward.
VLM_ACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "strategy_id", "action_type", "scene_problem",
        "object_ref", "object_track_id", "object_label", "object_center_base_m",
        "target_object_ref", "target_object_track_id", "target_object_label",
        "target_object_center_base_m", "contact_side", "direction_base",
        "distance_m", "gripper_yaw_rad", "safe_place_center_base_m",
        "predicted_scene_benefit", "risk_assessment", "reason", "confidence",
        "alternative_actions",
    ],
    "properties": {
        "strategy_id": {"type": "string"},
        "action_type": VLM_ACTION_SCHEMA["properties"]["action_type"],
        "scene_problem": {"type": "string"},
        "object_ref": {"type": ["string", "null"]},
        "object_track_id": {"type": ["string", "null"]},
        "object_label": {"type": ["string", "null"]},
        "object_center_base_m": {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 3, "maxItems": 3,
        },
        "target_object_ref": {"type": ["string", "null"]},
        "target_object_track_id": {"type": ["string", "null"]},
        "target_object_label": {"type": ["string", "null"]},
        "target_object_center_base_m": {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 3, "maxItems": 3,
        },
        "contact_side": {"type": ["string", "null"]},
        "direction_base": {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 3, "maxItems": 3,
        },
        "distance_m": {"type": ["number", "null"]},
        "gripper_yaw_rad": {"type": ["number", "null"]},
        "safe_place_center_base_m": {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 3, "maxItems": 3,
        },
        "predicted_scene_benefit": {"type": "string"},
        "risk_assessment": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "alternative_actions": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": False,
}


def schema_for_policy(policy_kind: str, task_type: str = "") -> Dict[str, object]:
    """Return the exact schema supplied to Ollama for one task-policy call."""
    if policy_kind == "task_contract":
        return TASK_CONTRACT_SCHEMAS.get(task_type, BUILD_HOUSE_CONTRACT_SCHEMA)
    if policy_kind == "grounded_task_plan":
        return GROUNDED_HOUSE_PLAN_SCHEMA if task_type == "build_house" else GROUNDED_ORGANIZE_PLAN_SCHEMA
    return TASK_ACTION_OUTPUT_SCHEMA


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
    if "oneOf" in schema:
        branch_errors = [validate_against_schema(value, branch, path) for branch in schema["oneOf"]]
        matching = [index for index, item in enumerate(branch_errors) if not item]
        if len(matching) != 1:
            errors.append({
                "type": "json_schema_one_of_mismatch", "path": path,
                "matching_branches": matching, "branch_errors": branch_errors,
            })
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
