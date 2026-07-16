"""Fixtures for the code-generated-edge/Qwen-selection architecture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.workflows.stack_demo.clutter.edge_generation import PlacementTarget
from tools.workflows.stack_demo.common.action_edges import ActionType, PhysicalActionEdge, make_edge
from tools.workflows.stack_demo.common.config import StackDemoConfig, load_stack_demo_config
from tools.workflows.stack_demo.common.scene_state import ClutterSceneState, build_clutter_scene_state
from tools.workflows.stack_demo.policy.qwen_client import QwenCallResult


ROOT = Path(__file__).resolve().parents[1]


def config() -> StackDemoConfig:
    return load_stack_demo_config(
        ROOT / "config" / "stack_demo_planner.json",
        ROOT / "config" / "workspace_bounds.json",
    )


def raw_object(
    detector_id: int,
    track_id: str,
    center: Sequence[float],
    *,
    label: str = "square red",
    size: Sequence[float] = (0.03, 0.03, 0.04),
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": detector_id,
        "track_id": track_id,
        "label": label,
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "yaw_deg": 0.0,
        **extra,
    }


def scene(
    objects: Sequence[Mapping[str, Any]],
    *,
    revision: int = 1,
    expected: Sequence[str] | None = None,
    completed: Sequence[str] = (),
    protected: Sequence[str] = (),
) -> ClutterSceneState:
    return build_clutter_scene_state(
        {"scene_revision": revision, "frame_id": "base_link", "objects": list(objects)},
        config().workspace,
        expected_tracks=expected,
        completed_tracks=completed,
        protected_tracks=protected,
    )


def placement(obj, interval=None) -> PlacementTarget:
    return PlacementTarget(
        ActionType.PICK_PLACE,
        "organize_red",
        None,
        {"frame_id": "base_link", "position_m": [0.28, 0.18, 0.02], "yaw_deg": 0.0},
        1.0,
        ("placed",),
        {
            "transport_safe": True,
            "place_descent_safe": True,
            "release_safe": True,
            "return_safe": True,
            "protected_safe": True,
        },
    )


def edge(
    *,
    revision: int = 1,
    candidate_id: str = "edge_1",
    target: str = "t1",
    acted: str = "t1",
    action_type: ActionType = ActionType.PICK_PLACE,
    direction: Sequence[float] | None = None,
    task_role: str | None = None,
) -> PhysicalActionEdge:
    physical: dict[str, Any] = {
        "grasp_yaw_deg": 35.0,
        "grasp_pose": {"frame_id": "base_link", "position_m": [0.5, 0.0, 0.02], "yaw_deg": 35.0},
        "approach_pose": {"frame_id": "base_link", "position_m": [0.5, 0.0, 0.07], "yaw_deg": 35.0},
        "lift_pose": {"frame_id": "base_link", "position_m": [0.5, 0.0, 0.12], "yaw_deg": 35.0},
        "place_pose": {"frame_id": "base_link", "position_m": [0.28, 0.18, 0.02], "yaw_deg": 0.0},
        "transport_path": [{"frame_id": "base_link", "position_m": [0.5, 0.0, 0.12]}],
    }
    if action_type == ActionType.NUDGE_BLOCKER:
        physical = {
            "push_direction_base": list(direction or [1.0, 0.0, 0.0]),
            "push_distance_m": 0.04,
            "push_start": {"frame_id": "base_link", "position_m": [0.45, 0.0, 0.02]},
            "push_end": {"frame_id": "base_link", "position_m": [0.49, 0.0, 0.02]},
            "prepush_pose": {"frame_id": "base_link", "position_m": [0.45, 0.0, 0.07]},
        }
    return make_edge(
        candidate_id=candidate_id,
        scene_revision=revision,
        action_type=action_type,
        task_type="build_house" if task_role else "organize_blocks",
        primary_target_track_id=target,
        acted_object_track_id=acted,
        task_role=task_role,
        target_region_id="organize_red" if not task_role else None,
        physical_parameters=physical,
        precheck_results={
            "passed": True, "finger_safe": True, "palm_safe": True,
            "descent_safe": True, "lift_safe": True, "transport_safe": True,
            "place_descent_safe": True, "release_safe": True, "return_safe": True,
            "prepush_reachable": True, "prepush_descent_safe": True,
            "horizontal_sweep_safe": True, "push_end_safe": True,
            "protected_safe": True,
        },
        decision_metadata={"object_ref": f"scene_{revision}:obj_1"},
    )


class MockQwenClient:
    def __init__(self, outputs: Sequence[Mapping[str, Any] | Exception]):
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def call(self, policy_kind, request, response_schema, image_paths, artifact_dir):
        self.calls.append({
            "policy_kind": policy_kind,
            "request": request,
            "response_schema": response_schema,
            "image_paths": list(image_paths),
        })
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            return QwenCallResult(request, "", None, False, type(output).__name__, str(output))
        return QwenCallResult(request, str(output), dict(output), True)


class MockExecutor:
    def __init__(self):
        self.calls: list[str] = []

    def approach_grasp(self, edge): self.calls.append("approach")
    def descend_grasp(self, edge): self.calls.append("descend")
    def close_gripper(self, edge): self.calls.append("close"); return True
    def lift_for_observation(self, edge): self.calls.append("lift")
    def transport(self, edge): self.calls.append("transport")
    def descend_place(self, edge): self.calls.append("place_descent")
    def release(self, edge): self.calls.append("release")
    def retreat(self, edge): self.calls.append("retreat")
    def execute_nudge(self, edge): self.calls.append("nudge")
    def return_to_observation_pose(self, edge): self.calls.append("return_observation")


class MockObserver:
    def __init__(self, scenes: Sequence[ClutterSceneState]):
        self.scenes = list(scenes)
        self.reasons: list[str] = []

    def observe(self, reason: str) -> ClutterSceneState:
        self.reasons.append(reason)
        return self.scenes.pop(0)
