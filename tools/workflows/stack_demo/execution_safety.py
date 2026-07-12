"""Top-level execution-source safety gates shared by every stack workflow."""

from __future__ import annotations

from typing import Any


OFFLINE_SOURCE_FIELDS = (
    "offline_scene_state",
    "offline_snapshot",
    "mock_perception",
    "recorded_perception",
)


def validate_execution_source(args: Any) -> None:
    """Forbid robot motion when perception is supplied by an offline source."""
    if not bool(getattr(args, "execute", False)):
        return
    enabled = [field for field in OFFLINE_SOURCE_FIELDS if getattr(args, field, None)]
    if enabled:
        flags = ", ".join("--{}".format(field.replace("_", "-")) for field in enabled)
        raise RuntimeError(
            "Real robot execution is forbidden when an offline perception source is enabled: {}.".format(flags)
        )
