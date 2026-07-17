"""Six-role house planner using the shared clutter extraction layer."""

from __future__ import annotations

from ..clutter.edge_generation import TargetSpec
from ..clutter.extraction_planner import ClutterExtractionPlanner, PlanningCycleResult
from ..common.action_edges import PhysicalActionEdge
from ..common.config import StackDemoConfig
from ..common.cycle_logging import CycleLogger
from ..common.scene_state import ClutterSceneState
from .placement import house_placement_target, staging_orientation_targets
from .roles import legal_incomplete_roles, role_precondition_satisfied
from .state import HouseTaskState


class HousePlanner:
    def __init__(self, config: StackDemoConfig, extraction_planner: ClutterExtractionPlanner):
        self._config = config
        self._extraction = extraction_planner

    def plan_cycle(
        self,
        scene: ClutterSceneState,
        state: HouseTaskState,
        logger: CycleLogger,
        *,
        image_paths: tuple[str, ...] = (),
    ) -> PlanningCycleResult:
        roles = legal_incomplete_roles(state.role_completion)
        target_specs = _candidate_role_specs(state, roles)
        return self._extraction.plan_cycle(
            scene=scene,
            task_type="build_house",
            task_state=state.to_dict(),
            target_track_ids=(),
            placement_provider=lambda obj, interval: None,
            staging_provider=lambda obj: staging_orientation_targets(
                scene, obj, "blocker", self._config,
            ),
            task_precondition=lambda edge, current: self._precondition(state, edge, current),
            logger=logger,
            target_specs=target_specs,
            variant_placement_provider=lambda obj, interval, role: house_placement_target(
                scene, state, obj, str(role), self._config, interval=interval,
            ),
            image_paths=image_paths,
        )

    @staticmethod
    def _precondition(
        state: HouseTaskState,
        edge: PhysicalActionEdge,
        scene: ClutterSceneState,
    ) -> tuple[bool, str]:
        role = edge.task_role
        if role in state.role_completion and not role_precondition_satisfied(role, state.role_completion):
            return False, f"house role prerequisite is not satisfied: {role}"
        if edge.acted_object_track_id in state.protected_structure_tracks:
            return False, "selected edge would move a protected completed structure"
        if edge.primary_target_track_id not in scene.visible_tracks:
            return False, "house target is not visible in current revision"
        return True, "house role and protected-structure preconditions remain valid"


def _candidate_role_specs(
    state: HouseTaskState,
    legal_roles: tuple[str, ...],
) -> tuple[TargetSpec, ...]:
    """Expose every legal role/track pair without detector-order binding."""
    specs = []
    for role in legal_roles:
        for track in state.role_candidate_tracks.get(role, ()):
            specs.append(TargetSpec(
                option_key=f"{track}__{role}",
                track_id=track,
                task_role=role,
            ))
    return tuple(specs)
