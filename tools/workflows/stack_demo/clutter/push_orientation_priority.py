"""Code-owned safety priority for equivalent push-tool orientations."""

from __future__ import annotations

from typing import Sequence

from ..common.action_edges import PhysicalActionEdge


def axis_alignment_error_deg(yaw_deg: float) -> float:
    """Return distance to the nearest 0/90-degree gripper axis."""
    return abs((float(yaw_deg) + 45.0) % 90.0 - 45.0)


def prefer_axis_aligned_pushes(
    edges: Sequence[PhysicalActionEdge],
) -> tuple[PhysicalActionEdge, ...]:
    """Filter a diagonal selection only when an equivalent axis edge passed.

    Callers group edges by object, direction, and distance, so this does not
    compare different physical effects.  Generation and audit still retain all
    feasible physical edges; this filter is applied at the policy boundary. A
    diagonal yaw remains available when every 0/90-degree alternative fails.
    """
    groups: dict[tuple[str, str, float], list[PhysicalActionEdge]] = {}
    for edge in edges:
        physical = edge.physical_parameters
        key = (
            edge.acted_object_track_id,
            str(physical.get("push_direction") or ""),
            round(float(physical.get("push_distance_m", 0.0)), 6),
        )
        groups.setdefault(key, []).append(edge)

    prioritized = []
    for group in groups.values():
        has_axis_aligned_pass = any(
            bool(edge.precheck_results.get("passed"))
            and axis_alignment_error_deg(
                float(edge.physical_parameters.get("push_wrist_yaw_deg", 0.0))
            ) < 1e-6
            for edge in group
        )
        for edge in group:
            yaw = float(edge.physical_parameters.get("push_wrist_yaw_deg", 0.0))
            if (
                not has_axis_aligned_pass
                or not edge.precheck_results.get("passed")
                or axis_alignment_error_deg(yaw) < 1e-6
            ):
                prioritized.append(edge)
    return tuple(prioritized)
