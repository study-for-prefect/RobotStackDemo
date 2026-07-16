"""Shared one-edge clutter planner for organize and house tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..common.action_edges import ActionType, PhysicalActionEdge
from ..common.action_validation import FinalSafetyGate, FinalSafetyGateResult, TaskPrecondition
from ..common.config import StackDemoConfig
from ..common.cycle_logging import CycleLogger
from ..common.scene_state import ClutterSceneState
from ..policy.edge_selector import EdgeSelectionOutcome, QwenEdgeSelector
from ..policy.target_selector import QwenTargetSelector, TargetSelectionOutcome
from .edge_generation import (
    PlacementProvider,
    PlanChecker,
    StagingProvider,
    TargetSpec,
    VariantPlacementProvider,
    generate_physical_edges,
    geometry_dry_run_plan_checker,
)
from .target_options import TargetOption, build_target_options


@dataclass(frozen=True)
class PlanningCycleResult:
    selected_target: TargetOption | None
    selected_edge: PhysicalActionEdge | None
    target_selection: TargetSelectionOutcome
    edge_selection: EdgeSelectionOutcome | None
    safety_gate: FinalSafetyGateResult | None
    decision_source: str
    reobserve_required: bool
    task_complete: bool = False


class ClutterExtractionPlanner:
    """Generate all first steps, ask Qwen only to select IDs, and run a final gate."""

    def __init__(
        self,
        config: StackDemoConfig,
        target_selector: QwenTargetSelector,
        edge_selector: QwenEdgeSelector,
        final_safety_gate: FinalSafetyGate,
        plan_checker: PlanChecker,
    ):
        self._config = config
        self._target_selector = target_selector
        self._edge_selector = edge_selector
        self._final_safety_gate = final_safety_gate
        self._plan_checker = plan_checker

    def plan_cycle(
        self,
        scene: ClutterSceneState,
        task_type: str,
        task_state: Mapping[str, Any],
        target_track_ids: Sequence[str],
        placement_provider: PlacementProvider,
        staging_provider: StagingProvider,
        task_precondition: TaskPrecondition,
        logger: CycleLogger,
        *,
        task_roles: Mapping[str, str] | None = None,
        target_specs: Sequence[TargetSpec] | None = None,
        variant_placement_provider: VariantPlacementProvider | None = None,
        image_paths: Sequence[str] = (),
    ) -> PlanningCycleResult:
        # Candidate generation owns geometry and must remain complete/auditable.
        # MoveIt is intentionally deferred: planning every push combination
        # before choosing an action dominated simple scenes by tens of seconds.
        generated = generate_physical_edges(
            scene, target_track_ids, task_type, self._config,
            placement_provider, staging_provider,
            plan_checker=geometry_dry_run_plan_checker,
            target_specs=target_specs,
            variant_placement_provider=variant_placement_provider,
        )
        all_edges = [edge for edges in generated.edges_by_target.values() for edge in edges]
        options = build_target_options(
            scene, task_type, generated.edges_by_target, generated.grasp_scans,
            task_roles=task_roles,
        )
        logger.write("scene_state.json", scene.to_dict())
        logger.write("task_state.json", dict(task_state))
        logger.write("target_options.json", [item.to_dict() for item in options])
        logger.write("physical_action_edges.json", [item.to_dict() for item in all_edges])
        generation_summary = generated.audit.summary(
            physical_action_edge_count=len(all_edges),
            target_option_count=len(options),
        )
        logger.write("candidate_generation_summary.json", generation_summary)
        logger.write("candidate_rejections.json", generated.audit.rejections)
        if generation_summary["candidate_generation_internal_error"]:
            raise RuntimeError(
                "candidate_generation_internal_error: unresolved objects produced no raw candidates"
            )

        failed_direct_ids: set[str] = set()
        if task_type == "organize_blocks":
            direct_result = self._first_moveit_feasible_direct_grasp(
                scene,
                task_precondition,
                options,
                all_edges,
                logger,
            )
            if direct_result is not None:
                return direct_result
            failed_direct_ids = {
                edge.candidate_id for edge in all_edges if _is_direct_grasp(edge)
            }
        if failed_direct_ids:
            selectable_by_target = {
                key: tuple(
                    edge for edge in edges
                    if edge.candidate_id not in failed_direct_ids
                )
                for key, edges in generated.edges_by_target.items()
            }
            options = build_target_options(
                scene, task_type, selectable_by_target, generated.grasp_scans,
                task_roles=task_roles,
            )
            logger.write("target_options.json", [item.to_dict() for item in options])
            logger.write("selection_eligibility.json", {
                "moveit_failed_direct_grasp_candidate_ids": sorted(failed_direct_ids),
                "fallback_selection_scope": "non_direct_candidates",
            })
        target_result = self._target_selector.select(scene, task_state, options, image_paths, str(logger.directory))
        logger.write("qwen_target_request.json", target_result.request)
        logger.write("qwen_target_response.json", target_result.response)
        logger.write("selected_target.json", _target_log(target_result))
        if target_result.selected is None:
            logger.ensure_planning_artifacts(target_result.decision_source)
            return PlanningCycleResult(None, None, target_result, None, None, target_result.decision_source, True)

        selected_edge_ids = set(target_result.selected.feasible_first_step_edge_ids)
        selected_edges = [edge for edge in all_edges if edge.candidate_id in selected_edge_ids]
        edge_result = self._edge_selector.select(
            scene, task_state, target_result.selected, selected_edges, image_paths, str(logger.directory),
        )
        logger.write("qwen_edge_request.json", edge_result.request)
        logger.write("qwen_edge_response.json", edge_result.response)
        logger.write("selected_edge.json", _edge_log(edge_result))
        if edge_result.selected is None:
            logger.ensure_planning_artifacts(edge_result.decision_source)
            return PlanningCycleResult(target_result.selected, None, target_result, edge_result, None, edge_result.decision_source, True)

        gate = self._final_safety_gate.validate(edge_result.selected, scene, task_precondition)
        logger.write("final_safety_gate.json", gate.to_dict())
        logger.ensure_planning_artifacts("not_applicable")
        if not gate.passed:
            return PlanningCycleResult(
                target_result.selected, None, target_result, edge_result, gate,
                "reobserve", True,
            )
        return PlanningCycleResult(
            target_result.selected, edge_result.selected, target_result, edge_result, gate,
            edge_result.decision_source, False,
        )

    def _first_moveit_feasible_direct_grasp(
        self,
        scene: ClutterSceneState,
        task_precondition: TaskPrecondition,
        options: Sequence[TargetOption],
        all_edges: Sequence[PhysicalActionEdge],
        logger: CycleLogger,
    ) -> PlanningCycleResult | None:
        direct_edges = sorted(
            (edge for edge in all_edges if _is_direct_grasp(edge)),
            key=lambda edge: (-edge.task_progress_gain, edge.risk_score, edge.candidate_id),
        )
        attempts = []
        for rank, edge in enumerate(direct_edges, start=1):
            gate = self._final_safety_gate.validate(edge, scene, task_precondition)
            attempts.append({"priority_rank": rank, **gate.to_dict()})
            if not gate.passed:
                continue
            target = next(
                option for option in options
                if edge.candidate_id in option.feasible_first_step_edge_ids
            )
            skipped = {
                "skipped_reason": "moveit_feasible_direct_grasp_code_priority",
                "planned_direct_candidates_until_success": rank,
            }
            target_result = TargetSelectionOutcome(
                target,
                "code_direct_grasp_priority",
                ("direct_grasp_available", "push_reasoning_skipped"),
                skipped,
                skipped,
            )
            edge_result = EdgeSelectionOutcome(
                edge,
                (),
                "code_direct_grasp_priority",
                ("geometry_safe", "moveit_plan_only_passed", "direct_grasp_preferred"),
                skipped,
                skipped,
            )
            logger.write("direct_grasp_plan_attempts.json", attempts)
            logger.write("qwen_target_request.json", target_result.request)
            logger.write("qwen_target_response.json", target_result.response)
            logger.write("selected_target.json", _target_log(target_result))
            logger.write("qwen_edge_request.json", edge_result.request)
            logger.write("qwen_edge_response.json", edge_result.response)
            logger.write("selected_edge.json", _edge_log(edge_result))
            logger.write("final_safety_gate.json", gate.to_dict())
            logger.ensure_planning_artifacts("not_applicable")
            return PlanningCycleResult(
                target,
                edge,
                target_result,
                edge_result,
                gate,
                edge_result.decision_source,
                False,
            )
        if attempts:
            logger.write("direct_grasp_plan_attempts.json", attempts)
        return None


def _is_direct_grasp(edge: PhysicalActionEdge) -> bool:
    return edge.action_type in {ActionType.PICK_PLACE, ActionType.EXTRACT_THEN_PLACE}


def _target_log(result: TargetSelectionOutcome) -> dict[str, Any]:
    return {
        "decision_source": result.decision_source,
        "selected_target_option_id": None if result.selected is None else result.selected.target_option_id,
        "reason_codes": list(result.reason_codes),
        "error": result.error,
        "silent_code_reselection": False,
    }


def _edge_log(result: EdgeSelectionOutcome) -> dict[str, Any]:
    return {
        "decision_source": result.decision_source,
        "selected_candidate_id": None if result.selected is None else result.selected.candidate_id,
        "backup_candidate_ids": list(result.backup_candidate_ids),
        "reason_codes": list(result.reason_codes),
        "error": result.error,
        "silent_code_reselection": False,
    }
