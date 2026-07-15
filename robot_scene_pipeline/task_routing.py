"""Deterministic routing for the two supported final tasks."""

from __future__ import annotations


def route_task_type(instruction: str) -> str:
    text = str(instruction or "").lower()
    if any(token in text for token in ("整理", "按颜色", "分类", "分组", "归类")):
        return "organize_blocks"
    if any(token in text for token in ("搭房子", "建房子", "房屋")):
        return "build_house"
    return "unsupported"
