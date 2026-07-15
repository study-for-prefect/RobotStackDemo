"""Small dependency-free validator for the JSON Schema subset used by Ollama."""

from __future__ import annotations

from typing import Any, Mapping


def validate_against_schema(value: object, schema: Mapping[str, Any], path: str = "$") -> list[dict[str, Any]]:
    """Return structured validation errors for the supported schema subset."""
    errors: list[dict[str, Any]] = []
    expected_type = schema.get("type")
    allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
    if expected_type and not any(_matches_type(value, str(item)) for item in allowed_types):
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
                "type": "json_schema_one_of_mismatch",
                "path": path,
                "matching_branches": matching,
                "branch_errors": branch_errors,
            })
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append({"type": "json_schema_missing_required", "path": f"{path}.{key}"})
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_against_schema(item, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append({"type": "json_schema_additional_property", "path": f"{path}.{key}"})
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            errors.append({"type": "json_schema_too_few_items", "path": path})
        if schema.get("maxItems") is not None and len(value) > int(schema["maxItems"]):
            errors.append({"type": "json_schema_too_many_items", "path": path})
        if schema.get("uniqueItems") and len({repr(item) for item in value}) != len(value):
            errors.append({"type": "json_schema_items_not_unique", "path": path})
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{index}]"))
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
