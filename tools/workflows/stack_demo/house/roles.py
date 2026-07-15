"""Fixed six-role house ontology and hard assembly prerequisites."""

from __future__ import annotations

from typing import Mapping


HOUSE_ROLES = (
    "left_support_lower",
    "right_support_lower",
    "left_support_upper",
    "right_support_upper",
    "roof",
    "triangle_top",
)

ROLE_DEPENDENCIES = {
    "left_support_lower": (),
    "right_support_lower": (),
    "left_support_upper": ("left_support_lower",),
    "right_support_upper": ("right_support_lower",),
    "roof": ("left_support_upper", "right_support_upper"),
    "triangle_top": ("roof",),
}


def legal_incomplete_roles(role_completion: Mapping[str, bool]) -> tuple[str, ...]:
    """Return roles whose verified structural prerequisites are satisfied."""
    return tuple(
        role for role in HOUSE_ROLES
        if not bool(role_completion.get(role))
        and all(bool(role_completion.get(dependency)) for dependency in ROLE_DEPENDENCIES[role])
    )


def role_precondition_satisfied(role: str, role_completion: Mapping[str, bool]) -> bool:
    if role not in ROLE_DEPENDENCIES or bool(role_completion.get(role)):
        return False
    return all(bool(role_completion.get(dependency)) for dependency in ROLE_DEPENDENCIES[role])
