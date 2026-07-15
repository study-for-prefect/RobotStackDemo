"""Final immutable safety gate applied after Qwen selects an edge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from ..clutter.edge_generation import PlanChecker
from .action_edges import ActionType, PhysicalActionEdge
from .config import StackDemoConfig
from .scene_state import ClutterSceneState


TaskPrecondition = Callable[[PhysicalActionEdge, ClutterSceneState], tuple[bool, str]]


@dataclass(frozen=True)
class FinalSafetyGateResult:
    passed: bool
    scene_revision: int
    candidate_id: str
    checks: Mapping[str, Mapping[str, Any]]
    failure_reasons: tuple[str, ...]
    moveit_result: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinalSafetyGate:
    """Recheck the selected edge without changing any object or parameter."""

    def __init__(
        self,
        config: StackDemoConfig,
        final_plan_checker: PlanChecker,
        *,
        require_real_moveit_plan: bool,
    ):
        self._config = config
        self._final_plan_checker = final_plan_checker
        self._require_real_moveit_plan = bool(require_real_moveit_plan)

    def validate(
        self,
        edge: PhysicalActionEdge,
        scene: ClutterSceneState,
        task_precondition: TaskPrecondition,
    ) -> FinalSafetyGateResult:
        checks: dict[str, dict[str, Any]] = {}
        _record(checks, "scene_revision_current", edge.scene_revision == scene.scene_revision)
        acted = scene.object_by_track(edge.acted_object_track_id)
        target = scene.object_by_track(edge.primary_target_track_id)
        _record(checks, "acted_track_visible_or_rebound", acted is not None)
        _record(checks, "primary_target_visible_or_rebound", target is not None)
        _record(checks, "acted_track_binding_unambiguous", acted is not None and not bool(acted.source.get("tracking_ambiguous")))
        _record(checks, "target_track_binding_unambiguous", target is not None and not bool(target.source.get("tracking_ambiguous")))
        _record(checks, "target_not_completed", target is not None and not target.already_completed)
        expected_ref = edge.decision_metadata.get("object_ref")
        _record(checks, "object_ref_current_revision", acted is not None and expected_ref == acted.object_ref)
        precondition_ok, precondition_reason = task_precondition(edge, scene)
        _record(checks, "task_precondition", precondition_ok, precondition_reason)
        _record(checks, "finger_safe", bool(edge.precheck_results.get("finger_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "palm_safe", bool(edge.precheck_results.get("palm_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "descent_sweep_safe", bool(edge.precheck_results.get("descent_safe", edge.precheck_results.get("prepush_descent_safe", False))))
        _record(checks, "lift_sweep_safe", bool(edge.precheck_results.get("lift_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "transport_sweep_safe", bool(edge.precheck_results.get("transport_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "place_descent_safe", bool(edge.precheck_results.get("place_descent_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "release_safe", bool(edge.precheck_results.get("release_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "return_path_safe", bool(edge.precheck_results.get("return_safe", edge.action_type == ActionType.NUDGE_BLOCKER)))
        _record(checks, "push_preapproach_safe", bool(edge.precheck_results.get("prepush_reachable", edge.action_type != ActionType.NUDGE_BLOCKER)))
        _record(checks, "push_horizontal_sweep_safe", bool(edge.precheck_results.get("horizontal_sweep_safe", edge.action_type != ActionType.NUDGE_BLOCKER)))
        _record(checks, "push_end_safe", bool(edge.precheck_results.get("push_end_safe", edge.action_type != ActionType.NUDGE_BLOCKER)))
        _record(checks, "protected_tracks_safe", acted is not None and not acted.protected and bool(edge.precheck_results.get("protected_safe", True)))
        _record(checks, "workspace_legal", _workspace_legal(edge, self._config.workspace))
        _record(checks, "fingerprint_not_forbidden", edge.failure_fingerprint not in scene.forbidden_action_fingerprints)
        moveit = dict(self._final_plan_checker(edge))
        moveit_ok = bool(moveit.get("passed"))
        if self._require_real_moveit_plan:
            moveit_ok = moveit_ok and bool(moveit.get("moveit_plan_only"))
        _record(checks, "moveit_plan_only_success", moveit_ok, str(moveit.get("mode", "")))
        failures = tuple(name for name, value in checks.items() if not value["passed"])
        return FinalSafetyGateResult(
            passed=not failures,
            scene_revision=scene.scene_revision,
            candidate_id=edge.candidate_id,
            checks=checks,
            failure_reasons=failures,
            moveit_result=moveit,
        )


def _record(checks: dict[str, dict[str, Any]], name: str, passed: bool, detail: str = "") -> None:
    checks[name] = {"passed": bool(passed), "detail": detail}


def _workspace_legal(edge: PhysicalActionEdge, bounds: Mapping[str, float]) -> bool:
    positions = []
    for key in ("grasp_pose", "approach_pose", "lift_pose", "place_pose", "release_pose", "push_start", "push_end", "prepush_pose"):
        pose = edge.physical_parameters.get(key)
        if isinstance(pose, Mapping) and isinstance(pose.get("position_m"), (list, tuple)):
            positions.append(pose["position_m"])
    for pose in edge.physical_parameters.get("transport_path", []):
        if isinstance(pose, Mapping) and isinstance(pose.get("position_m"), (list, tuple)):
            positions.append(pose["position_m"])
    return bool(positions) and all(
        len(position) >= 3
        and float(bounds["xmin"]) <= float(position[0]) <= float(bounds["xmax"])
        and float(bounds["ymin"]) <= float(position[1]) <= float(bounds["ymax"])
        and float(bounds["zmin"]) <= float(position[2]) <= float(bounds["zmax"])
        for position in positions
    )
