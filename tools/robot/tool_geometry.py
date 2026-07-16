"""Authoritative measured transform from UR tool0 to the GF225 grasp TCP."""

from __future__ import annotations


TOOL0_TO_TCP_OFFSET_TOOL_M: tuple[float, float, float] = (0.0, 0.0, 0.16)


def default_tcp_offset_tool_m() -> list[float]:
    """Return a mutable CLI value without duplicating the measured constant."""
    return list(TOOL0_TO_TCP_OFFSET_TOOL_M)
