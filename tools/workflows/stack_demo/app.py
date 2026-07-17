"""Thin top-level lifecycle for the two code-state/Qwen-selection workflows."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from robot_scene_pipeline.ollama_policy_client import unload_model
from robot_scene_pipeline.object_tracking import update_scene_tracks
from robot_scene_pipeline.task_routing import route_task_type

from .arguments import parse_args
from .clutter.edge_generation import geometry_dry_run_plan_checker
from .clutter.extraction_planner import ClutterExtractionPlanner
from .commands import capture_empty_current_pose, capture_empty_observation, load_json
from .common.action_execution import EdgeExecutionResult, SceneObserver, execute_one_edge
from .common.action_validation import FinalSafetyGate
from .common.config import StackDemoConfig, load_stack_demo_config
from .common.cycle_logging import CycleLogger
from .common.moveit_adapter import MoveItEdgeAdapter
from .common.policy_images import build_policy_images
from .common.scene_state import ClutterSceneState, SceneObjectState, build_clutter_scene_state
from .common.track_lifecycle import mark_track_after_place, mark_track_held, predicted_track_centers
from .execution_safety import validate_execution_source
from .house.completion import evaluate_house_completion
from .house.planner import HousePlanner
from .house.state import HouseTaskState, build_house_task_state
from .house.structure import role_observation_checks
from .live_integration import (
    LiveIntegrationReport,
    capture_after_joint_state,
    capture_ros_preflight,
    validate_live_snapshot,
    validate_live_integration_args,
)
from .organize.completion import evaluate_organize_completion
from .organize.planner import OrganizePlanner
from .organize.state import OrganizeTaskState, build_organize_task_state
from .policy.edge_selector import QwenEdgeSelector
from .policy.qwen_client import JsonFileQwenClient, StatelessQwenClient
from .policy.target_selector import QwenTargetSelector


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_live_integration_args(args)
    validate_execution_source(args)
    if args.execute and not args.yes:
        raise RuntimeError("real execution requires both --execute and --yes")
    config = load_stack_demo_config(args.planner_config, args.workspace_bounds_json)
    _apply_tcp_offset_config(args, config)
    _apply_motion_config(args, config)
    task_type = args.task_type if args.task_type != "auto" else route_task_type(args.instruction)
    if task_type not in {"organize_blocks", "build_house"}:
        raise RuntimeError("stack_demo_pipeline now supports only organize_blocks and build_house")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    integration = LiveIntegrationReport(output) if args.live_integration_check else None
    preflight_complete = False
    try:
        if integration is not None:
            integration.write()
            preflight = capture_ros_preflight(args, output)
            preflight_complete = True
            integration.update(camera_topics_ok=bool(preflight.get("camera_topics_ok")))
            integration.artifact("ros_preflight", output / "ros_preflight.json")
            integration.artifact("before_joint_state", output / "before_joint_state.json")
        return _run(args, config, task_type, output, integration=integration)
    except Exception as exc:
        (output / "failure_state.json").write_text(json.dumps({
            "task_type": task_type,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "task_complete": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if integration is not None:
            integration.fail(integration.values.get("failure_stage") or "pipeline", exc)
        raise
    finally:
        if integration is not None and preflight_complete:
            try:
                delta = capture_after_joint_state(args, output)
                integration.update(joint_state_delta=delta)
                integration.artifact("after_joint_state", output / "after_joint_state.json")
                integration.artifact("joint_state_delta", output / "joint_state_delta.json")
            except Exception as exc:
                integration.fail("after_joint_state", exc)
        if args.unload_model_after_task and not args.mock_policy_response_dir:
            unload_model(args, str(output))


def _run(
    args: Any,
    config: StackDemoConfig,
    task_type: str,
    output: Path,
    *,
    integration: LiveIntegrationReport | None = None,
) -> int:
    raw = _initial_observation(args, output)
    if integration is not None:
        initial = output / "initial_observation"
        snapshot_validation = validate_live_snapshot(initial, raw)
        integration.update(
            tf_ok=(initial / "tf_status.json").is_file(),
            perception_health_ok=(initial / "perception_server_health.json").is_file(),
            snapshot_ok=(initial / "private_scene_state.json").is_file(),
            parameter_contract_ok=(initial / "perception_snapshot_response.json").is_file(),
            snapshot_validation=snapshot_validation,
        )
        for name in (
            "snapshot.jpg", "annotated_detector.jpg", "private_scene_state.json",
            "detector_objects_3d.json", "tabletop_geometry.json", "tf_status.json",
            "perception_server_health.json", "perception_snapshot_request.json",
        ):
            path = initial / name
            if path.is_file():
                integration.artifact(name, path)
    track_memory: dict[str, Any] = {}
    _bind_observation_tracks(raw, track_memory, trust_recorded_ids=bool(args.offline_scene_state))
    scene = build_clutter_scene_state(raw, config.workspace)
    expected_tracks = scene.expected_tracks
    policy_client = (
        JsonFileQwenClient(args.mock_policy_response_dir)
        if args.mock_policy_response_dir else StatelessQwenClient(args)
    )
    failure_limit = int(config.section("policy")["recent_failure_limit"])
    target_selector = QwenTargetSelector(policy_client, failure_limit)
    edge_selector = QwenEdgeSelector(policy_client, failure_limit)
    action_history: list[dict[str, Any]] = []
    forbidden: set[str] = set()
    organize_state: OrganizeTaskState | None = None
    house_state: HouseTaskState | None = None
    consecutive_reobserve = 0
    max_consecutive_reobserve = int(
        config.section("policy")["max_consecutive_reobserve"]
    )

    for cycle_index in range(1, max(1, args.max_task_steps) + 1):
        cycle_dir = output / f"cycle_{cycle_index:03d}_revision_{scene.scene_revision}"
        logger = CycleLogger(cycle_dir)
        scene, task_state, completion = _decorate_task_scene(
            scene, task_type, config, expected_tracks, organize_state, house_state,
            action_history, forbidden,
        )
        if task_type == "organize_blocks":
            organize_state = task_state
        else:
            house_state = task_state
        logger.write("task_completion.json", completion)
        if completion["task_complete"]:
            logger.ensure_planning_artifacts("task_complete_from_latest_observation")
            logger.ensure_execution_artifacts("task_complete_from_latest_observation")
            return 0

        adapter = MoveItEdgeAdapter(args, config, cycle_dir)
        plan_checker = adapter.check if (args.moveit_plan_only or args.execute) else geometry_dry_run_plan_checker
        final_gate = FinalSafetyGate(
            config, plan_checker,
            require_real_moveit_plan=bool(args.moveit_plan_only or args.execute),
        )
        extraction = ClutterExtractionPlanner(config, target_selector, edge_selector, final_gate, plan_checker)
        images = () if args.no_image else build_policy_images(raw, cycle_dir)
        if task_type == "organize_blocks":
            cycle = OrganizePlanner(config, extraction).plan_cycle(scene, organize_state, logger, image_paths=images)
        else:
            cycle = HousePlanner(config, extraction).plan_cycle(scene, house_state, logger, image_paths=images)
        if cycle.selected_edge is None:
            logger.ensure_execution_artifacts(cycle.decision_source)
            # Every empty cycle below is followed by a fresh observation and
            # full candidate regeneration in this same task.  Detector jitter
            # changes geometry by millimetres, so revision/pose equality must
            # not reset the bounded consecutive-empty retry counter.
            consecutive_reobserve += 1
            logger.write("reobserve_attempt.json", {
                "decision_source": cycle.decision_source,
                "consecutive_unchanged_empty_scans": consecutive_reobserve,
                "max_consecutive_reobserve": max_consecutive_reobserve,
                "will_reobserve": consecutive_reobserve < max_consecutive_reobserve,
                "candidate_summary_path": "candidate_generation_summary.json",
                "candidate_rejections_path": "candidate_rejections.json",
            })
            if integration is not None:
                edges = _load_json_list(cycle_dir / "physical_action_edges.json")
                integration.update(
                    physical_edges_generated=bool(edges),
                    qwen_ok=False,
                    failure_stage=(
                        "no_selected_physical_edge"
                        if consecutive_reobserve >= max_consecutive_reobserve else None
                    ),
                )
            if consecutive_reobserve >= max_consecutive_reobserve:
                return 2
            scene, raw, expected_tracks = _reobserve_without_action(
                args,
                config,
                output,
                cycle_index,
                consecutive_reobserve,
                scene,
                expected_tracks,
                action_history,
                forbidden,
                track_memory,
            )
            continue
        consecutive_reobserve = 0
        if not args.execute:
            logger.ensure_execution_artifacts("moveit_plan_only" if args.moveit_plan_only else "dry_run_no_motion")
            if integration is not None:
                edges = _load_json_list(cycle_dir / "physical_action_edges.json")
                qwen_ok = bool(
                    cycle.target_selection.decision_source == "qwen_target_selection"
                    and cycle.edge_selection is not None
                    and cycle.edge_selection.decision_source == "qwen_edge_selection"
                )
                moveit_ok = bool(cycle.safety_gate and cycle.safety_gate.passed)
                integration.update(
                    physical_edges_generated=bool(edges),
                    qwen_ok=qwen_ok,
                    selected_edge=cycle.selected_edge.candidate_id,
                    moveit_plan_only_attempted=True,
                    moveit_plan_only_ok=moveit_ok,
                    failure_stage=None if qwen_ok and moveit_ok else (
                        "qwen_selection_skipped_or_failed" if not qwen_ok else "moveit_plan_only_failed"
                    ),
                )
                integration.artifact("selected_edge", cycle_dir / "selected_edge.json")
                integration.artifact("final_safety_gate", cycle_dir / "final_safety_gate.json")
            return 0

        observer = _LiveObserver(
            args, config, output, cycle_index, scene, expected_tracks,
            action_history, forbidden, track_memory, cycle.selected_edge,
        )
        result = execute_one_edge(
            cycle.selected_edge,
            scene,
            adapter,
            observer,
            _destination_check(task_type, config, organize_state, house_state),
        )
        entry = {
            "candidate_id": cycle.selected_edge.candidate_id,
            "action_type": cycle.selected_edge.action_type.value,
            "primary_target_track_id": cycle.selected_edge.primary_target_track_id,
            "acted_object_track_id": cycle.selected_edge.acted_object_track_id,
            "task_role": cycle.selected_edge.task_role,
            "target_region_id": cycle.selected_edge.target_region_id,
            "failure_fingerprint": cycle.selected_edge.failure_fingerprint,
            "success": result.success,
            "status": result.status,
            "scene_revision": result.final_scene.scene_revision,
            "planned_place_position_m": list(
                (cycle.selected_edge.physical_parameters.get("place_pose") or {}).get(
                    "position_m", ()
                )
            ),
            "acted_object_size_m": list(
                scene.object_by_track(cycle.selected_edge.acted_object_track_id).size_xyz_m
            ) if scene.object_by_track(cycle.selected_edge.acted_object_track_id) else [],
            "release_executed": any(
                stage.get("stage") == "release" and stage.get("status") == "executed"
                for stage in result.stages
            ),
        }
        action_history.append(entry)
        _log_execution(logger, result, action_history)
        if not result.success:
            forbidden.add(cycle.selected_edge.failure_fingerprint)
        scene = result.final_scene
        raw = observer.latest_raw
        expected_tracks = observer.expected_tracks
    return 2


def _reobserve_without_action(
    args: Any,
    config: StackDemoConfig,
    output: Path,
    cycle_index: int,
    attempt: int,
    before: ClutterSceneState,
    expected_tracks: tuple[str, ...],
    history: list[dict[str, Any]],
    forbidden: set[str],
    track_memory: dict[str, Any],
) -> tuple[ClutterSceneState, dict[str, Any], tuple[str, ...]]:
    """Capture and rebuild a fresh scene after an empty/invalid planning cycle."""
    if args.offline_scene_state:
        raw = load_json(args.offline_scene_state)
        raw["scene_revision"] = before.scene_revision + 1
        _bind_observation_tracks(raw, track_memory, trust_recorded_ids=True)
    else:
        directory = output / f"observation_cycle_{cycle_index:03d}_reobserve_{attempt}"
        raw = capture_empty_current_pose(
            args,
            str(directory),
            None,
            allow_holding=False,
            refresh_tf=False,
        )
        if raw is None:
            raise RuntimeError("fresh reobserve did not produce a scene")
        raw["scene_revision"] = before.scene_revision + 1
        _bind_observation_tracks(raw, track_memory)
    merged_expected = _merge_expected_tracks(expected_tracks, raw, track_memory)
    scene = build_clutter_scene_state(
        raw,
        config.workspace,
        expected_tracks=merged_expected,
        recent_action_results=history[-5:],
        forbidden_action_fingerprints=forbidden,
    )
    return scene, raw, merged_expected


def _scene_geometry_signature(scene: ClutterSceneState) -> tuple[Any, ...]:
    return tuple(sorted(
        (
            obj.track_id,
            *(round(float(value), 4) for value in obj.center_xyz_m),
            *(round(float(value), 4) for value in obj.size_xyz_m),
        )
        for obj in scene.current_objects
    ))


def _load_json_list(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _decorate_task_scene(
    scene: ClutterSceneState,
    task_type: str,
    config: StackDemoConfig,
    expected_tracks: tuple[str, ...],
    organize_previous: OrganizeTaskState | None,
    house_previous: HouseTaskState | None,
    history: list[dict[str, Any]],
    forbidden: set[str],
) -> tuple[ClutterSceneState, OrganizeTaskState | HouseTaskState, dict[str, Any]]:
    scene = replace(
        scene,
        expected_tracks=expected_tracks,
        missing_expected_tracks=tuple(sorted(set(expected_tracks) - set(scene.visible_tracks))),
        recent_action_results=tuple(history[-5:]),
        forbidden_action_fingerprints=tuple(sorted(forbidden)),
    )
    if task_type == "organize_blocks":
        latest_result = history[-1] if history else None
        state = build_organize_task_state(
            scene, config, previous=organize_previous,
            current_action_result=latest_result,
        )
        scene = _apply_task_marks(
            scene,
            state.completed_tracks,
            state.completed_tracks,
            tuple(state.color_target_regions.values()),
        )
        state = build_organize_task_state(
            scene, config, previous=state,
            current_action_result=latest_result,
        )
        return scene, state, evaluate_organize_completion(scene, state)
    state = build_house_task_state(scene, config, previous=house_previous)
    scene = _apply_task_marks(scene, (), state.protected_structure_tracks, ())
    state = build_house_task_state(scene, config, previous=state)
    return scene, state, evaluate_house_completion(scene, state)


def _apply_task_marks(
    scene: ClutterSceneState,
    completed: tuple[str, ...],
    protected: tuple[str, ...],
    target_regions: tuple[Mapping[str, Any], ...],
) -> ClutterSceneState:
    completed_set, protected_set = set(completed), set(protected)
    objects = tuple(replace(
        obj,
        already_completed=obj.track_id in completed_set,
        protected=obj.track_id in protected_set,
    ) for obj in scene.current_objects)
    return replace(
        scene,
        current_objects=objects,
        completed_tracks=tuple(sorted(completed_set)),
        protected_tracks=tuple(sorted(protected_set)),
        target_regions=target_regions,
    )


class _LiveObserver(SceneObserver):
    def __init__(
        self,
        args: Any,
        config: StackDemoConfig,
        output: Path,
        cycle_index: int,
        before: ClutterSceneState,
        expected_tracks: tuple[str, ...],
        history: list[dict[str, Any]],
        forbidden: set[str],
        track_memory: dict[str, Any],
        selected_edge,
    ):
        self._args, self._config, self._output = args, config, output
        self._cycle_index, self._revision = cycle_index, before.scene_revision
        self._expected, self._history, self._forbidden = expected_tracks, history, forbidden
        self._track_memory, self._selected_edge = track_memory, selected_edge
        self.latest_raw: dict[str, Any] = {}

    def observe(self, reason: str) -> ClutterSceneState:
        directory = self._output / f"observation_cycle_{self._cycle_index:03d}_{reason}"
        raw = capture_empty_current_pose(
            self._args, str(directory), None,
            allow_holding=True, refresh_tf=False,
        )
        if raw is None:
            raise RuntimeError(f"fresh observation missing after {reason}")
        self._revision += 1
        raw["scene_revision"] = self._revision
        _bind_observation_tracks(
            raw,
            self._track_memory,
            predicted_centers=predicted_track_centers(
                self._selected_edge, reason, self._track_memory,
            ),
        )
        self._expected = _merge_expected_tracks(
            self._expected, raw, self._track_memory,
        )
        self.latest_raw = raw
        return build_clutter_scene_state(
            raw, self._config.workspace,
            expected_tracks=self._expected,
            recent_action_results=self._history[-5:],
            forbidden_action_fingerprints=self._forbidden,
        )

    @property
    def expected_tracks(self) -> tuple[str, ...]:
        return self._expected

    def mark_held(self, edge, verification) -> None:
        evidence = verification.evidence
        confidence = 0.9 if evidence.get("target_moved_from_original_position") else 0.75
        mark_track_held(
            self._track_memory,
            edge,
            verification.scene_revision,
            held_state_confidence=confidence,
        )

    def mark_after_place(self, edge, verification) -> None:
        mark_track_after_place(
            self._track_memory,
            edge,
            verification.scene_revision,
            success=verification.success,
        )


def _destination_check(
    task_type: str,
    config: StackDemoConfig,
    organize_state: OrganizeTaskState | None,
    house_state: HouseTaskState | None,
):
    def check(edge, scene):
        if task_type == "organize_blocks":
            state = build_organize_task_state(scene, config, previous=organize_state)
            return edge.acted_object_track_id in state.completed_tracks, {
                "completed_tracks": list(state.completed_tracks),
                "target_region_id": edge.target_region_id,
            }
        role = edge.task_role
        bindings = dict(house_state.role_bindings if house_state else {})
        if role:
            bindings[role] = edge.acted_object_track_id
        valid, role_checks = role_observation_checks(scene, bindings, str(role or ""), config)
        return bool(role and valid), {
            "task_role": role,
            "role_observation_checks": role_checks,
        }
    return check


def _log_execution(logger: CycleLogger, result: EdgeExecutionResult, history: list[dict[str, Any]]) -> None:
    logger.write("execution_result.json", result.to_dict())
    logger.write(
        "post_grasp_verification.json",
        {"skipped_reason": "not_a_grasp_edge"} if result.post_grasp_verification is None else result.post_grasp_verification.to_dict(),
    )
    logger.write(
        "post_place_verification.json",
        {"skipped_reason": result.status} if result.post_place_verification is None else result.post_place_verification.to_dict(),
    )
    logger.write("action_history.json", history)


def _initial_observation(args: Any, output: Path) -> dict[str, Any]:
    if args.offline_scene_state:
        raw = load_json(args.offline_scene_state)
    else:
        raw = capture_empty_observation(args, str(output / "initial_observation"), None)
        if raw is None:
            raise RuntimeError("initial live observation was not produced")
    raw.setdefault("scene_revision", 1)
    raw.setdefault("frame_id", "base_link")
    return raw


def _apply_motion_config(args: Any, config: StackDemoConfig) -> None:
    motion = config.section("motion")
    args.velocity = float(motion["translation_velocity"])
    args.acceleration = float(motion["translation_acceleration"])
    args.pre_rotate_velocity = float(motion["high_clearance_rotation_velocity"])
    args.pre_rotate_acceleration = float(motion["high_clearance_rotation_acceleration"])
    args.place_velocity = float(motion["near_object_velocity"])
    args.place_acceleration = float(motion["near_object_acceleration"])
    policy = config.section("policy")
    if not args.model:
        args.model = str(policy["model"])
    if int(args.vlm_num_ctx) <= 0:
        args.vlm_num_ctx = int(policy["num_ctx"])
    if int(args.vlm_num_predict) <= 0:
        args.vlm_num_predict = int(policy["num_predict"])


def _apply_tcp_offset_config(args: Any, config: StackDemoConfig) -> None:
    configured = [float(value) for value in config.section("gripper")["tcp_offset_tool_m"]]
    requested = [float(value) for value in args.tcp_offset_tool]
    if any(abs(left - right) > 1e-9 for left, right in zip(requested, configured)):
        raise ValueError(
            f"--tcp-offset-tool {requested} must match planner config {configured}; "
            "the transform may only have one runtime source"
        )
    args.tcp_offset_tool = configured
    print(f"tool_frame={args.tool_frame}", flush=True)
    print(f"tcp_offset_tool={args.tcp_offset_tool}", flush=True)


def _bind_observation_tracks(
    raw: dict[str, Any],
    memory: dict[str, Any],
    *,
    trust_recorded_ids: bool = False,
    predicted_displacements: Mapping[str, list[float]] | None = None,
    predicted_centers: Mapping[str, list[float]] | None = None,
) -> None:
    objects = [item for item in raw.get("objects", []) if isinstance(item, dict)]
    revision = int(raw.get("scene_revision", 1))
    if trust_recorded_ids:
        if any(not item.get("track_id") for item in objects):
            raise ValueError("offline scene objects must provide explicit track_id values")
        if len({str(item["track_id"]) for item in objects}) != len(objects):
            raise ValueError("offline scene track_id values must be unique")
        for item in objects:
            item["object_ref"] = f"scene_{revision}:obj_{item.get('id')}"
        raw["track_rebinding_results"] = {"source": "trusted_recorded_scene", "assignments": []}
        return
    _, assignments = update_scene_tracks(
        memory,
        objects,
        revision,
        predicted_displacements=dict(predicted_displacements or {}),
        predicted_centers=dict(predicted_centers or {}),
    )
    raw["objects"] = objects
    raw["track_rebinding_results"] = {
        "source": "explicit_one_to_one_rebinding",
        "assignments": assignments,
    }


def _merge_expected_tracks(
    expected_tracks: tuple[str, ...],
    raw: Mapping[str, Any],
    memory: Mapping[str, Any],
) -> tuple[str, ...]:
    """Add stable new tracks and replace stale IDs rebound to the same object."""
    merged = set(expected_tracks)
    tracks = memory.get("tracks", {}) if isinstance(memory, Mapping) else {}
    stable_visible: dict[str, Mapping[str, Any]] = {}
    for item in raw.get("objects", ()):
        if not isinstance(item, Mapping):
            continue
        track_id = str(item.get("track_id") or "")
        track = tracks.get(track_id, {}) if isinstance(tracks, Mapping) else {}
        history = track.get("history", ()) if isinstance(track, Mapping) else ()
        if track_id and len(history) >= 2:
            merged.add(track_id)
            stable_visible[track_id] = track

    # A full-scene observation after a wrist-camera occlusion can create a new
    # track ID for an existing block. Rotation may also change the detector's
    # shape label, so use color, metric size, and a tight position gate for the
    # identity handoff. Ambiguous matches remain unresolved.
    visible_ids = {
        str(item.get("track_id"))
        for item in raw.get("objects", ())
        if isinstance(item, Mapping) and item.get("track_id")
    }
    replacement_candidates: list[tuple[float, str, str]] = []
    for stale_id in sorted(merged - visible_ids):
        stale = tracks.get(stale_id, {}) if isinstance(tracks, Mapping) else {}
        stale_center = stale.get("center_base_m") if isinstance(stale, Mapping) else None
        stale_size = stale.get("dimensions_m") if isinstance(stale, Mapping) else None
        stale_color = str(stale.get("color") or "unknown") if isinstance(stale, Mapping) else "unknown"
        if not _finite_xyz(stale_center):
            continue
        matches: list[tuple[float, str]] = []
        for current_id, current in stable_visible.items():
            if current_id == stale_id:
                continue
            current_color = str(current.get("color") or "unknown")
            if stale_color == "unknown" or current_color != stale_color:
                continue
            current_center = current.get("center_base_m")
            if not _finite_xyz(current_center):
                continue
            distance = math.dist(
                [float(value) for value in stale_center[:3]],
                [float(value) for value in current_center[:3]],
            )
            if distance > 0.012:
                continue
            current_size = current.get("dimensions_m")
            if _finite_xyz(stale_size) and _finite_xyz(current_size):
                size_delta = sum(
                    abs(float(stale_size[index]) - float(current_size[index]))
                    for index in range(3)
                )
                if size_delta > 0.020:
                    continue
            matches.append((distance, current_id))
        if len(matches) == 1:
            replacement_candidates.append((matches[0][0], stale_id, matches[0][1]))

    used_current: set[str] = set()
    replacements: dict[str, str] = {}
    for _distance, stale_id, current_id in sorted(replacement_candidates):
        if current_id in used_current:
            continue
        merged.discard(stale_id)
        merged.add(current_id)
        used_current.add(current_id)
        replacements[stale_id] = current_id
    if replacements and isinstance(memory, dict):
        memory.setdefault("track_aliases", {}).update(replacements)
        for stale_id, current_id in replacements.items():
            stale = memory.get("tracks", {}).get(stale_id)
            if isinstance(stale, dict):
                stale["superseded_by_track_id"] = current_id
    if replacements and isinstance(raw, dict):
        raw["expected_track_replacements"] = dict(sorted(replacements.items()))
    return tuple(sorted(merged))


def _finite_xyz(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return False
    try:
        return all(math.isfinite(float(value[index])) for index in range(3))
    except (TypeError, ValueError):
        return False
