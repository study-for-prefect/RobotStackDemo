"""Color-organization planner built on the shared clutter extractor."""

from __future__ import annotations

from typing import Any, Mapping

from ..clutter.edge_generation import PlacementTarget
from ..clutter.grasp_edges import GraspInterval, normalize_gripper_yaw_deg
from ..clutter.extraction_planner import ClutterExtractionPlanner, PlanningCycleResult
from ..common.action_edges import ActionType, PhysicalActionEdge
from ..common.config import StackDemoConfig
from ..common.cycle_logging import CycleLogger
from ..common.scene_state import ClutterSceneState, SceneObjectState
from .placement import rank_slots_by_clearance, select_staging_position, table_contact_center_z
from .state import OrganizeTaskState


class OrganizePlanner:
    def __init__(self, config: StackDemoConfig, extraction_planner: ClutterExtractionPlanner):
        self._config = config
        self._extraction = extraction_planner

    def plan_cycle(
        self,
        scene: ClutterSceneState,
        state: OrganizeTaskState,
        logger: CycleLogger,
        *,
        image_paths: tuple[str, ...] = (),
    ) -> PlanningCycleResult:
        legal_targets = [
            track_id for track_id in state.unresolved_tracks
            if track_id in scene.visible_tracks
        ]
        return self._extraction.plan_cycle(
            scene=scene,
            task_type="organize_blocks",
            task_state=state.to_dict(),
            target_track_ids=legal_targets,
            placement_provider=lambda obj, interval: self._placement(scene, state, obj, interval),
            # Organize staging is outside the wrist-camera observation area and
            # has zero task progress.  If no color-row placement is currently
            # safe, retain the generated pushes and reobserve instead.
            staging_provider=lambda obj: None,
            task_precondition=lambda edge, current: self._precondition(state, edge, current),
            logger=logger,
            image_paths=image_paths,
        )

    def _placement(
        self,
        scene: ClutterSceneState,
        state: OrganizeTaskState,
        obj: SceneObjectState,
        interval: GraspInterval | None = None,
    ) -> tuple[PlacementTarget, ...]:
        slots = rank_slots_by_clearance(scene, state.safe_slots_by_color.get(obj.color, ()))
        targets = []
        grasp_yaw = None if interval is None else normalize_gripper_yaw_deg(interval.selected_yaw_deg)
        release_yaws = _ordinary_release_yaws(grasp_yaw, obj.yaw_deg)
        for slot in slots:
            for preference_rank, release_yaw in enumerate(release_yaws):
                position = list(slot["position_m"])
                position[2] = table_contact_center_z(scene, obj)
                additional = {
                    "placement_slot_id": str(slot["slot_id"]),
                    "placement_yaw_preference_rank": preference_rank,
                }
                if release_yaw is not None:
                    additional["ordinary_release_gripper_yaw_deg"] = release_yaw
                targets.append(PlacementTarget(
                    action_type=(
                        ActionType.EXTRACT_THEN_PLACE if len(obj.neighbors) >= 3
                        else ActionType.PICK_PLACE
                    ),
                    target_region_id=str(slot["region_id"]),
                    task_role=None,
                    place_pose={
                        "frame_id": "base_link",
                        "position_m": position,
                        "yaw_deg": float(slot["yaw_deg"]),
                    },
                    task_progress_gain=1.0,
                    expected_effects=("enter_correct_color_region", "remove_from_clutter"),
                    precheck_results={
                        "transport_safe": True,
                        "place_descent_safe": bool(slot["open_gripper_descent_safe"]),
                        "release_safe": True,
                        "return_safe": True,
                        "protected_safe": True,
                    },
                    additional_physical_parameters=additional,
                ))
        return tuple(targets)

    def _staging(
        self,
        scene: ClutterSceneState,
        state: OrganizeTaskState,
        obj: SceneObjectState,
    ) -> PlacementTarget | None:
        position = select_staging_position(
            scene, self._config, obj, state.staging_reservations,
        )
        if position is None:
            return None
        return PlacementTarget(
            action_type=ActionType.EXTRACT_TO_STAGING,
            target_region_id="organize_staging",
            task_role=None,
            place_pose={
                "frame_id": "base_link",
                "position_m": position,
                "yaw_deg": 0.0,
            },
            task_progress_gain=0.0,
            expected_effects=("temporary_staging",),
            precheck_results={
                "transport_safe": True, "place_descent_safe": True,
                "release_safe": True, "return_safe": True, "protected_safe": True,
            },
        )

    @staticmethod
    def _precondition(
        state: OrganizeTaskState,
        edge: PhysicalActionEdge,
        scene: ClutterSceneState,
    ) -> tuple[bool, str]:
        if edge.primary_target_track_id not in state.unresolved_tracks:
            return False, "target already completed or outside expected tracks"
        if edge.primary_target_track_id not in scene.visible_tracks:
            return False, "target is not visible in current revision"
        return True, "organize target remains unresolved"


def _ordinary_release_yaws(
    grasp_yaw: float | None,
    object_yaw: float,
) -> tuple[float | None, ...]:
    """Prefer a tidy object edge axis, retaining no-rotation and clearance fallbacks."""
    if grasp_yaw is None:
        return (None,)
    relative = (float(object_yaw) - grasp_yaw + 90.0) % 180.0 - 90.0
    tidy = [
        normalize_gripper_yaw_deg(-relative),
        normalize_gripper_yaw_deg(-90.0 - relative),
    ]
    object_axis_error = min(
        abs((float(object_yaw) - axis + 90.0) % 180.0 - 90.0)
        for axis in (0.0, -90.0)
    )
    values = ([grasp_yaw, *tidy] if object_axis_error <= 8.0 else [*tidy, grasp_yaw])
    for value in (0.0, -90.0):
        if all(abs(((value - old + 90.0) % 180.0) - 90.0) > 1e-6 for old in values):
            values.append(value)
    return tuple(values)
