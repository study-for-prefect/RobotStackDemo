"""Fixed per-cycle artifact set for auditable decisions and execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PLANNING_ARTIFACTS = (
    "scene_state.json",
    "task_state.json",
    "target_options.json",
    "qwen_target_request.json",
    "qwen_target_response.json",
    "selected_target.json",
    "physical_action_edges.json",
    "qwen_edge_request.json",
    "qwen_edge_response.json",
    "selected_edge.json",
    "final_safety_gate.json",
)

EXECUTION_ARTIFACTS = (
    "execution_result.json",
    "post_grasp_verification.json",
    "post_place_verification.json",
    "action_history.json",
)


class CycleLogger:
    def __init__(self, cycle_dir: str | Path):
        self.directory = Path(cycle_dir)
        self.directory.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, value: Any) -> None:
        path = self.directory / filename
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def ensure_planning_artifacts(self, reason: str) -> None:
        self._ensure(PLANNING_ARTIFACTS, reason)

    def ensure_execution_artifacts(self, reason: str) -> None:
        self._ensure(EXECUTION_ARTIFACTS, reason)

    def _ensure(self, names: tuple[str, ...], reason: str) -> None:
        for name in names:
            path = self.directory / name
            if not path.exists():
                self.write(name, {"skipped_reason": reason})
