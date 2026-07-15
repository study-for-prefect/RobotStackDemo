"""Color-organization planner built on the shared clutter extractor."""

from __future__ import annotations

from typing import Any, Mapping

from ..clutter.edge_generation import PlacementTarget
from ..clutter.extraction_planner import ClutterExtractionPlanner, PlanningCycleResult
from ..common.action_edges import ActionType, PhysicalActionEdge
from ..common.config import StackDemoConfig
from ..common.cycle_logging import CycleLogger
from ..common.scene_state import ClutterSceneState, SceneObjectState
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
            placement_provider=lambda obj, interval: self._placement(state, obj),
            staging_provider=lambda obj: self._staging(state, obj),
            task_precondition=lambda edge, current: self._precondition(state, edge, current),
            logger=logger,
            image_paths=image_paths,
        )

    def _placement(self, state: OrganizeTaskState, obj: SceneObjectState) -> PlacementTarget | None:
        slots = state.safe_slots_by_color.get(obj.color, ())
        occupied = {track for tracks in state.region_occupancy.values() for track in tracks}
        slot = next((item for item in slots if item.get("occupied_by") not in occupied), None)
        if slot is None:
            return None
        return PlacementTarget(
            action_type=ActionType.EXTRACT_THEN_PLACE if len(obj.neighbors) >= 3 else ActionType.PICK_PLACE,
            target_region_id=str(slot["region_id"]),
            task_role=None,
            place_pose={
                "frame_id": "base_link",
                "position_m": list(slot["position_m"]),
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
        )

    def _staging(self, state: OrganizeTaskState, obj: SceneObjectState) -> PlacementTarget | None:
        direct = self._placement(state, obj)
        if direct is not None:
            return direct
        x = float(self._config.workspace["xmax"]) - 0.06
        y = float(self._config.workspace["ymin"]) + 0.06
        return PlacementTarget(
            action_type=ActionType.EXTRACT_TO_STAGING,
            target_region_id="organize_staging",
            task_role=None,
            place_pose={"frame_id": "base_link", "position_m": [x, y, obj.center_xyz_m[2]], "yaw_deg": 0.0},
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
