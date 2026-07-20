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
    def return_to_observation_pose(self, edge: PhysicalActionEdge) -> None: ...


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
        executor.return_to_observation_pose(edge)
        stages.append({"stage": "return_to_observation_pose", "status": "executed"})
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

    mark_held = getattr(observer, "mark_held", None)
    if callable(mark_held):
        mark_held(edge, grasp)
        stages.append({"stage": "track_state", "status": "held_by_gripper"})

    executor.transport(edge)
    stages.append({"stage": "transport", "status": "executed_after_grasp_verification"})
    trajectory = edge.physical_parameters.get("orientation_trajectory")
    if isinstance(trajectory, Mapping):
        stages.extend([
            {"stage": "safe_orientation_adjustment_approach_pose", "status": "executed_far_from_house"},
            {"stage": "safe_orientation_adjustment_start_pose", "status": "executed"},
            {"stage": "fixed_tcp_orientation_waypoints", "status": "executed",
             "waypoint_count": len(trajectory.get("waypoints", ())),
             "maximum_tcp_position_error_m": trajectory.get("orientation_sweep_checks", {}).get("maximum_tcp_position_error_m")},
            {"stage": "orientation_adjustment_complete_pose", "status": "final_3d_orientation_reached"},
            {"stage": "final_pre_place_pose", "status": "translated_with_final_orientation_held"},
        ])
    elif edge.physical_parameters.get("ordinary_yaw_adjustment"):
        stages.append({
            "stage": "safe_height_yaw_adjustment",
            "status": "completed_before_final_transport",
            **dict(edge.physical_parameters["ordinary_yaw_adjustment"]),
        })
    try:
        executor.descend_place(edge)
    except Exception as descent_error:
        stages.append({
            "stage": "place_descent",
            "status": "failed",
            "reason": str(descent_error),
        })
        try:
            executor.retreat(edge)
            stages.append({
                "stage": "retreat_while_holding",
                "status": "executed_after_place_descent_failure",
            })
        except Exception as retreat_error:
            raise RuntimeError(
                "place descent failed and safe retreat while holding also failed: "
                f"descent={descent_error}; retreat={retreat_error}"
            ) from descent_error
        raise
    stages.append({"stage": "place_descent", "status": "executed"})
    executor.release(edge)
    stages.append({"stage": "release", "status": "executed"})
    executor.retreat(edge)
    stages.append({
        "stage": "retreat",
        "status": "executed_to_safe_height_before_place_verification",
    })
    # A vertical retreat still leaves the wrist-mounted D435i above the house.
    # Restore the calibrated overview before judging the released placement.
    executor.return_to_observation_pose(edge)
    stages.append({
        "stage": "return_to_observation_pose",
        "status": "executed_before_place_verification",
    })
    after_place = observer.observe("post_place")
    stages.append({
        "stage": "fresh_post_place_observation_at_safe_height",
        "scene_revision": after_place.scene_revision,
    })
    place = verify_place_result(edge, before, after_place, destination_check)
    mark_after_place = getattr(observer, "mark_after_place", None)
    if callable(mark_after_place):
        mark_after_place(edge, place)
        stages.append({
            "stage": "track_state",
            "status": "placed" if place.success else "unresolved",
        })
    if not place.success:
        stages.append({
            "stage": "place_verification",
            "status": "skipped",
            "reason": "place_not_verified",
        })
    return EdgeExecutionResult(
        "place_verified" if place.success else "place_failed",
        edge.candidate_id,
        tuple(stages),
        grasp,
        place,
        after_place,
    )
