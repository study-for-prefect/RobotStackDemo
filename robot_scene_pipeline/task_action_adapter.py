"""Single field-name adapter between task actions and legacy motion validators."""

from __future__ import annotations

from typing import Any, Dict


class TaskActionSchemaError(ValueError):
    """A task action has conflicting or missing current-scene object identity."""

    def __init__(self, reason: str, action: Dict[str, Any]):
        super().__init__(reason)
        self.feedback = {
            "validation_stage": "action_schema_validation",
            "passed": False,
            "reason": reason,
            "selected_object_id": action.get("selected_object_id"),
        }


def adapt_task_action_to_legacy_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Copy selected_object_id to object_id without changing any action decision."""
    selected_object_id = action.get("selected_object_id")
    legacy_object_id = action.get("object_id")
    if selected_object_id is not None and legacy_object_id is not None:
        if str(selected_object_id) != str(legacy_object_id):
            raise TaskActionSchemaError("conflicting_object_id_fields", action)
        raise TaskActionSchemaError("legacy_object_id_forbidden", action)
    if selected_object_id is None:
        raise TaskActionSchemaError("missing_selected_object_id", action)
    adapted = dict(action)
    adapted["object_id"] = selected_object_id
    return adapted
