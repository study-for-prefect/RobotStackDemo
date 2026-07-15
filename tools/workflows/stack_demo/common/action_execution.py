"""Execute exactly one immutable edge with mandatory post-grasp gating."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from .action_edges import ActionType, PhysicalActionEdge
from .result_verification import (
    ActionVerification,
    DestinationCheck,
    verify_grasp_result,
    verify_nudge_result,
    verify_place_result,
)
from .scene_state import ClutterSceneState


class EdgeExecutor(Protocol):
    def approach_grasp(self, edge: PhysicalActionEdge) -> None: ...
    def descend_grasp(self, edge: PhysicalActionEdge) -> None: ...
    def close_gripper(self, edge: PhysicalActionEdge) -> bool | None: ...
    def lift_for_observation(self, edge: PhysicalActionEdge) -> None: ...
    def transport(self, edge: PhysicalActionEdge) -> None: ...
    def descend_place(self, edge: PhysicalActionEdge) -> None: ...
    def release(self, edge: PhysicalActionEdge) -> None: ...
    def retreat(self, edge: PhysicalActionEdge) -> None: ...
    def execute_nudge(self, edge: PhysicalActionEdge) -> None: ...


class SceneObserver(Protocol):
    def observe(self, reason: str) -> ClutterSceneState: ...


@dataclass(frozen=True)
class EdgeExecutionResult:
    status: str
    candidate_id: str
    stages: tuple[Mapping[str, Any], ...]
    post_grasp_verification: ActionVerification | None
    post_place_verification: ActionVerification | None
    final_scene: ClutterSceneState
    post_nudge_verification: ActionVerification | None = None

    @property
    def success(self) -> bool:
        if self.post_place_verification is not None:
            return self.post_place_verification.success
        if self.post_nudge_verification is not None:
            return self.post_nudge_verification.success
        return False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["success"] = self.success
        value["final_scene"] = self.final_scene.to_dict()
        return value


def execute_one_edge(
    edge: PhysicalActionEdge,
    before: ClutterSceneState,
    executor: EdgeExecutor,
    observer: SceneObserver,
    destination_check: DestinationCheck,
) -> EdgeExecutionResult:
    """Never transport a grasped object until fresh visual evidence verifies pickup."""
    stages: list[Mapping[str, Any]] = []
    if edge.action_type == ActionType.NUDGE_BLOCKER:
        executor.execute_nudge(edge)
        stages.append({"stage": "horizontal_push", "status": "executed"})
        observed = observer.observe("post_nudge")
        stages.append({"stage": "fresh_observation", "scene_revision": observed.scene_revision})
        verification = verify_nudge_result(edge, before, observed)
        return EdgeExecutionResult(
            verification.status, edge.candidate_id, tuple(stages), None, None,
            observed, verification,
        )

    executor.approach_grasp(edge)
    stages.append({"stage": "approach_target", "status": "executed"})
    executor.descend_grasp(edge)
    stages.append({"stage": "descend", "status": "executed"})
    holding_hint = executor.close_gripper(edge)
    stages.append({"stage": "close_gripper", "status": "executed", "holding_hint": holding_hint})
    executor.lift_for_observation(edge)
    stages.append({"stage": "lift_to_observation_height", "status": "executed"})
    after_lift = observer.observe("post_grasp_at_safe_height")
    stages.append({"stage": "fresh_post_grasp_observation", "scene_revision": after_lift.scene_revision})
    grasp = verify_grasp_result(edge, before, after_lift, gripper_holding_hint=holding_hint)
    if not grasp.success:
        stages.append({"stage": "transport", "status": "skipped", "reason": "grasp_failed"})
        stages.append({"stage": "place", "status": "skipped", "reason": "grasp_failed"})
        return EdgeExecutionResult("grasp_failed", edge.candidate_id, tuple(stages), grasp, None, after_lift)

    executor.transport(edge)
    stages.append({"stage": "transport", "status": "executed_after_grasp_verification"})
    executor.descend_place(edge)
    stages.append({"stage": "place_descent", "status": "executed"})
    executor.release(edge)
    stages.append({"stage": "release", "status": "executed"})
    executor.retreat(edge)
    stages.append({"stage": "retreat", "status": "executed"})
    after_place = observer.observe("post_place")
    stages.append({"stage": "fresh_post_place_observation", "scene_revision": after_place.scene_revision})
    place = verify_place_result(edge, after_place, destination_check)
    return EdgeExecutionResult(
        "place_verified" if place.success else "place_failed",
        edge.candidate_id,
        tuple(stages),
        grasp,
        place,
        after_place,
    )
