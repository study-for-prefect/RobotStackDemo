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
from .perception_semantic_review import review_build_house_scene


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
    raw = _initial_observation(args, output, task_type)
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
    continuation = _load_live_continuation(args, task_type)
    track_memory: dict[str, Any] = dict(continuation.get("track_memory") or {})
    _bind_observation_tracks(raw, track_memory, trust_recorded_ids=bool(args.offline_scene_state))
    scene = build_clutter_scene_state(raw, config.workspace)
    expected_tracks = tuple(continuation.get("expected_tracks") or scene.expected_tracks)
    policy_client = (
        JsonFileQwenClient(args.mock_policy_response_dir)
        if args.mock_policy_response_dir else StatelessQwenClient(args)
    )
    failure_limit = int(config.section("policy")["recent_failure_limit"])
    target_selector = QwenTargetSelector(policy_client, failure_limit)
    edge_selector = QwenEdgeSelector(policy_client, failure_limit)
    action_history: list[dict[str, Any]] = list(continuation.get("action_history") or [])
    forbidden: set[str] = set()
    organize_state: OrganizeTaskState | None = None
    house_state: HouseTaskState | None = continuation.get("house_state")
    if task_type == "build_house" and house_state is not None:
        house_state, expected_tracks = _rebind_released_tabletop_triangle(
            scene, house_state, expected_tracks, action_history,
        )
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
        if task_type == "build_house":
            logger.write("house_completion_state.json", completion)
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
                task_type,
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
            args, config, task_type, output, cycle_index, scene, expected_tracks,
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
            "staging_purpose": cycle.selected_edge.physical_parameters.get("staging_purpose"),
            "final_house_transport_forbidden_this_edge": bool(
                cycle.selected_edge.physical_parameters.get(
                    "final_house_transport_forbidden_this_edge", False,
                )
            ),
        }
        action_history.append(entry)
        _log_execution(logger, result, action_history)
        if not result.success:
            forbidden.add(cycle.selected_edge.failure_fingerprint)
        scene = result.final_scene
        raw = observer.latest_raw
        expected_tracks = observer.expected_tracks
        if result.status == "place_failed":
            print(
                "Placement was not visually verified; keeping the robot at the "
                "safe retreat pose and stopping this run.",
                flush=True,
            )
            return 2
    return 2


def _rebind_released_tabletop_triangle(
    scene: ClutterSceneState,
    state: HouseTaskState,
    expected_tracks: tuple[str, ...],
    history: list[dict[str, Any]],
) -> tuple[HouseTaskState, tuple[str, ...]]:
    """Hand a released triangle identity to its unique fresh 3-D detection."""
    if not history:
        return state, expected_tracks
    latest = history[-1]
    if not (
        latest.get("status") == "released_tabletop_step_requires_fresh_geometry"
        and latest.get("action_type") == "extract_to_staging"
        and latest.get("task_role") == "triangle_top"
        and latest.get("release_executed") is True
    ):
        return state, expected_tracks
    old_track = str(state.role_bindings.get("triangle_top") or "")
    if old_track and scene.object_by_track(old_track) is not None:
        return state, expected_tracks
    matches = [
        item for item in scene.current_objects
        if item.shape == "triangle"
        and item.color == "green"
        and item.source.get("metric_triangle_geometry_confirmed") is True
    ]
    if len(matches) != 1:
        return state, expected_tracks
    new_track = matches[0].track_id
    bindings = {**dict(state.role_bindings), "triangle_top": new_track}
    rebound_expected = tuple(
        new_track if str(track) == old_track else str(track)
        for track in expected_tracks
    )
    if new_track not in rebound_expected:
        rebound_expected = (*rebound_expected, new_track)
    latest.update({
        "acted_object_track_id": new_track,
        "geometry_rebound_track_id": new_track,
        "fresh_metric_triangle_rebound": True,
    })
    return replace(
        state,
        expected_tracks=rebound_expected,
        role_bindings=bindings,
        role_completion={**dict(state.role_completion), "triangle_top": False},
        role_status={**dict(state.role_status), "triangle_top": "REPAIRABLE"},
        geometry_rebound_tracks=tuple(sorted({*state.geometry_rebound_tracks, new_track})),
    ), rebound_expected


def _load_live_continuation(args: Any, task_type: str) -> dict[str, Any]:
    """Load continuity facts, never an observation, from a released live edge.

    The next run still starts with the camera and revalidates every role.  The
    prior cycle supplies only stable track identities and already verified
    occluded support bindings that cannot be rediscovered beneath an opaque
    roof.
    """
    value = str(getattr(args, "continue_from_cycle", "") or "")
    if not value:
        return {}
    if task_type != "build_house" or args.offline_scene_state:
        raise ValueError("--continue-from-cycle requires a live build_house run")
    cycle = Path(value)
    execution = load_json(str(cycle / "execution_result.json"))
    task_state_raw = load_json(str(cycle / "task_state.json"))
    history = _load_json_list(cycle / "action_history.json")
    stages = execution.get("stages") or []
    final_scene = execution.get("final_scene") or {}
    visible_final = tuple(final_scene.get("current_objects", ()))
    verification = execution.get("post_place_verification") or {}
    role_checks = (verification.get("evidence") or {}).get("role_observation_checks") or {}
    other_failed_checks = sorted(
        key for key, passed in role_checks.items()
        if isinstance(passed, bool) and not passed and key != "vertical_contact_valid"
    )
    released_roof = bool(
        execution.get("status") == "place_failed"
        and (verification.get("evidence") or {}).get("house_role") == "roof"
        and any(item.get("stage") == "release" and item.get("status") == "executed" for item in stages)
        and role_checks.get("vertical_contact_valid") is False
        and not other_failed_checks
    )
    released_triangle = bool(
        execution.get("status") == "place_failed"
        and (verification.get("evidence") or {}).get("house_role") == "triangle_top"
        and any(item.get("stage") == "release" and item.get("status") == "executed" for item in stages)
        and role_checks.get("role_object_visible") is True
        and role_checks.get("roof_visible") is True
        and role_checks.get("base_contact_valid") is True
    )
    released_triangle_table_candidates = tuple(
        item for item in visible_final
        if str(item.get("shape") or "") == "triangle"
        and str(item.get("color") or "") == "green"
    )
    released_triangle_to_table = bool(
        execution.get("status") == "place_failed"
        and history
        and history[-1].get("action_type") == "place_house_role"
        and history[-1].get("task_role") == "triangle_top"
        and any(item.get("stage") == "release" and item.get("status") == "executed" for item in stages)
        and len(released_triangle_table_candidates) == 1
    )
    staged_orientation = bool(
        execution.get("status") == "place_verified"
        and history
        and history[-1].get("action_type") == "extract_to_staging"
        and history[-1].get("success") is True
        and any(item.get("stage") == "release" and item.get("status") == "executed" for item in stages)
    )
    released_staging_candidates = tuple(
        item for item in visible_final
        if str(item.get("shape") or "") in {"triangle", "rectangle", "concave_rectangle"}
        and str(item.get("color") or "") == "green"
    )
    released_staging = bool(
        execution.get("status") == "place_failed"
        and history
        and history[-1].get("action_type") == "extract_to_staging"
        and history[-1].get("task_role") == "triangle_top"
        and any(item.get("stage") == "release" and item.get("status") == "executed" for item in stages)
        and len(released_staging_candidates) == 1
    )
    released_tabletop_step_requires_fresh_geometry = bool(
        execution.get("status") == "place_failed"
        and history
        and history[-1].get("action_type") == "extract_to_staging"
        and history[-1].get("task_role") == "triangle_top"
        and "tabletop_face_reorientation" in str(
            history[-1].get("failure_fingerprint") or ""
        )
        and any(
            item.get("stage") == "release" and item.get("status") == "executed"
            for item in stages
        )
        and not released_staging_candidates
    )
    if not (
        released_roof or released_triangle or released_triangle_to_table
        or staged_orientation or released_staging
        or released_tabletop_step_requires_fresh_geometry
    ):
        raise ValueError(
            "continuation cycle must be a released structure visual rejection or verified staging"
        )
    if not task_state_raw or not history:
        raise ValueError("continuation cycle is missing task_state/action_history")
    task_state_raw = dict(task_state_raw)
    role_bindings = dict(task_state_raw.get("role_bindings") or {})
    if released_roof:
        continued_roof_track = str(history[-1].get("acted_object_track_id") or "")
        if not continued_roof_track:
            raise ValueError("continuation action history has no acted roof track")
        role_bindings["roof"] = continued_roof_track
        task_state_raw["role_completion"] = {
            **dict(task_state_raw.get("role_completion") or {}),
            "roof": True,
        }
        task_state_raw["role_status"] = {
            **dict(task_state_raw.get("role_status") or {}),
            "roof": "COMPLETED_VISIBLE",
        }
    elif not str(role_bindings.get("roof") or ""):
        raise ValueError("verified staging continuation has no preserved roof binding")
    if released_triangle or released_triangle_to_table:
        continued_triangle_track = str(
            released_triangle_table_candidates[0].get("track_id")
            if released_triangle_to_table
            else history[-1].get("acted_object_track_id")
            or ""
        )
        if not continued_triangle_track:
            raise ValueError("continuation action history has no acted triangle track")
        role_bindings["triangle_top"] = continued_triangle_track
        if released_triangle_to_table:
            old_triangle_track = str(history[-1].get("acted_object_track_id") or "")
            task_state_raw["expected_tracks"] = [
                continued_triangle_track if str(track) == old_triangle_track else str(track)
                for track in task_state_raw.get("expected_tracks", ())
            ]
        task_state_raw["role_completion"] = {
            **dict(task_state_raw.get("role_completion") or {}),
            "triangle_top": False,
        }
        task_state_raw["role_status"] = {
            **dict(task_state_raw.get("role_status") or {}),
            "triangle_top": "REPAIRABLE",
        }
    if released_staging:
        old_triangle_track = str(history[-1].get("acted_object_track_id") or "")
        continued_triangle_track = str(released_staging_candidates[0].get("track_id") or "")
        if not continued_triangle_track:
            raise ValueError("released staging continuation has no rebound triangle track")
        role_bindings["triangle_top"] = continued_triangle_track
        task_state_raw["expected_tracks"] = [
            continued_triangle_track if str(track) == old_triangle_track else str(track)
            for track in task_state_raw.get("expected_tracks", ())
        ]
    elif released_tabletop_step_requires_fresh_geometry:
        # The tabletop rotation and release are physically complete even when
        # the immediately following visual review drops the changed side view.
        # Preserve the acted identity only as a provisional binding; the new
        # run has already captured a fresh observation and must remeasure the
        # full pose before either another 45-degree step or roof transport.
        continued_triangle_track = str(history[-1].get("acted_object_track_id") or "")
        if not continued_triangle_track:
            raise ValueError("released tabletop step has no acted triangle track")
        role_bindings["triangle_top"] = continued_triangle_track
        task_state_raw["role_completion"] = {
            **dict(task_state_raw.get("role_completion") or {}),
            "triangle_top": False,
        }
        task_state_raw["role_status"] = {
            **dict(task_state_raw.get("role_status") or {}),
            "triangle_top": "REPAIRABLE",
        }
    task_state_raw["role_bindings"] = role_bindings
    corrected_history = [dict(item) for item in history]
    if released_roof:
        corrected_history[-1].update({
            "success": True,
            "status": "place_requires_fresh_continuation_confirmation",
            "continuation_requires_fresh_geometry": True,
        })
    elif released_triangle or released_triangle_to_table:
        corrected_history[-1].update({
            "acted_object_track_id": continued_triangle_track,
            "success": False,
            "status": "released_triangle_on_table_requires_incremental_orientation_repair",
            "continuation_preserve_verified_roof": True,
        })
    elif released_staging:
        corrected_history[-1].update({
            "acted_object_track_id": continued_triangle_track,
            "success": True,
            "status": "place_verified_from_unique_staging_geometry_rebound",
            "geometry_rebound_track_id": continued_triangle_track,
            "continuation_preserve_verified_roof": True,
        })
    elif released_tabletop_step_requires_fresh_geometry:
        corrected_history[-1].update({
            "success": True,
            "status": "released_tabletop_step_requires_fresh_geometry",
            "continuation_requires_fresh_geometry": True,
            "continuation_preserve_verified_roof": True,
        })
    tracks: dict[str, Any] = {}
    continuity_track_ids = {
        str(track) for track in task_state_raw.get("expected_tracks", ()) if track
    }.union(
        str(track) for track in role_bindings.values() if track
    ).union(
        str(item.get("acted_object_track_id"))
        for item in history if item.get("acted_object_track_id")
    )
    continuity_visible_final = tuple(
        item for item in visible_final
        if str(item.get("track_id") or "") in continuity_track_ids
    )
    remembered_final = tuple(
        item for item in final_scene.get("collision_obstacles", ())
        if str(item.get("track_id") or "") in continuity_track_ids
        if not any(
            item.get("color") == visible.get("color")
            and len(item.get("center_xyz_m") or ()) >= 3
            and len(visible.get("center_xyz_m") or ()) >= 3
            and math.dist(item["center_xyz_m"], visible["center_xyz_m"]) <= 0.012
            for visible in visible_final
        )
    )
    for item in (*continuity_visible_final, *remembered_final):
        track_id = str(item.get("track_id") or "")
        if not track_id:
            continue
        tracks[track_id] = {
            "track_id": track_id,
            "label": item.get("class_name"),
            "semantic_shape": item.get("shape"),
            "color": item.get("color"),
            "center_base_m": list(item.get("center_xyz_m") or ()),
            "dimensions_m": list(item.get("size_xyz_m") or ()),
            "table_yaw_deg": float(item.get("yaw_deg") or 0.0),
            "last_seen_revision": 1,
            "visible": False,
            "history": [{"source": "validated_live_continuation"}] * 2,
        }
    return {
        "track_memory": {"tracks": tracks, "track_history": []},
        "expected_tracks": tuple(task_state_raw.get("expected_tracks") or tracks),
        "action_history": corrected_history,
        "house_state": HouseTaskState(**task_state_raw),
    }


def _reobserve_without_action(
    args: Any,
    config: StackDemoConfig,
    task_type: str,
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
        raw = _review_live_observation(args, raw, directory, task_type)
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
    return scene, state, evaluate_house_completion(scene, state, config)


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
        task_type: str,
        output: Path,
        cycle_index: int,
        before: ClutterSceneState,
        expected_tracks: tuple[str, ...],
        history: list[dict[str, Any]],
        forbidden: set[str],
        track_memory: dict[str, Any],
        selected_edge,
    ):
        self._args, self._config, self._task_type, self._output = (
            args, config, task_type, output,
        )
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
        raw = _review_live_observation(
            self._args, raw, directory, self._task_type,
        )
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
        if role == "triangle_top":
            valid, role_checks = _released_triangle_destination_check(
                edge, scene, bindings, config, role_checks,
            )
        rebound_track_id = None
        if role and not valid and scene.object_by_track(edge.acted_object_track_id) is None:
            expected_pose = edge.physical_parameters.get("place_pose", {})
            expected_position = expected_pose.get("position_m") if isinstance(expected_pose, Mapping) else None
            candidates = sorted(scene.current_objects, key=lambda item: (
                math.dist(item.center_xyz_m, expected_position)
                if isinstance(expected_position, (list, tuple)) and len(expected_position) >= 3
                else float("inf")
            ))
            for candidate in candidates:
                candidate_bindings = {**bindings, role: candidate.track_id}
                candidate_valid, candidate_checks = role_observation_checks(
                    scene, candidate_bindings, role, config,
                )
                if candidate_valid:
                    valid, role_checks = True, candidate_checks
                    rebound_track_id = candidate.track_id
                    break
        return bool(role and valid), {
            "task_role": role,
            "role_observation_checks": role_checks,
            "geometry_rebound_track_id": rebound_track_id,
            "verified_structure_occlusion": bool(
                valid and scene.object_by_track(edge.acted_object_track_id) is None
                and rebound_track_id is None
            ),
        }
    return check


def _released_triangle_destination_check(
    edge: Any,
    scene: ClutterSceneState,
    bindings: Mapping[str, str],
    config: StackDemoConfig,
    original_checks: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Fuse fresh placement geometry with the executed rigid SE(3) target."""
    checks = dict(original_checks)
    triangle = scene.object_by_track(edge.acted_object_track_id)
    roof = _scene_object_for_verification(scene, str(bindings.get("roof") or ""))
    physical = edge.physical_parameters
    expected = physical.get("target_object_pose") or physical.get("place_pose") or {}
    expected_position = expected.get("position_m") if isinstance(expected, Mapping) else None
    target_edge = expected.get("designated_right_angle_edge_world") if isinstance(expected, Mapping) else None
    beam = expected.get("beam_axis_world") if isinstance(expected, Mapping) else None
    if (
        triangle is None or roof is None
        or not isinstance(expected_position, (list, tuple)) or len(expected_position) < 3
    ):
        checks["executed_target_pose_available"] = False
        return False, checks
    house = config.section("house")
    xy_error = math.dist(triangle.center_xyz_m[:2], expected_position[:2])
    commanded_target_face_up = bool(
        isinstance(target_edge, (list, tuple)) and len(target_edge) == 3
        and float(target_edge[2]) >= float(config.section("house_orientation")["minimum_upward_normal_component"])
    )
    beam_yaw = (
        math.degrees(math.atan2(float(beam[1]), float(beam[0])))
        if isinstance(beam, (list, tuple)) and len(beam) == 3 else 0.0
    )
    yaw_error = abs((float(triangle.yaw_deg) - beam_yaw + 90.0) % 180.0 - 90.0)
    roof_support_half = 0.5 * min(float(value) for value in roof.size_xyz_m[:2])
    support_margin = roof_support_half - xy_error
    uncertainty = 0.002
    checks.update({
        "long_edge_matches_roof_axis": yaw_error <= float(house["orientation_tolerance_deg"]),
        "long_edge_alignment_error_deg": yaw_error,
        "center_of_mass_supported": xy_error <= roof_support_half,
        "support_margin_valid": (
            support_margin + uncertainty >= float(house["minimum_support_margin_m"])
        ),
        "roof_center_offset_valid": xy_error <= float(house["center_tolerance_m"]),
        "triangle_center_offset_m": xy_error,
        "support_margin_m": support_margin,
        "support_margin_uncertainty_m": uncertainty,
        "executed_target_pose_available": True,
        "executed_rigid_target_face_up": commanded_target_face_up,
        "fresh_observation_confirms_apex_up": checks.get("apex_up") is True,
        "verification_center_reference": "executed_live_roof_target_pose",
        "verification_axis_source": "fresh_semantic_apex_and_detector_axis",
    })
    return all(value for value in checks.values() if isinstance(value, bool)), checks


def _scene_object_for_verification(
    scene: ClutterSceneState,
    track_id: str,
) -> Any | None:
    """Resolve a visible object or retained geometry occluded by the placement."""
    return scene.object_by_track(track_id) or next(
        (item for item in scene.collision_obstacles if item.track_id == track_id),
        None,
    )


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
    verification = result.post_place_verification
    logger.write(
        "post_place_3d_verification.json",
        verification.to_dict() if verification is not None else {"skipped_reason": result.status},
    )


def _initial_observation(args: Any, output: Path, task_type: str) -> dict[str, Any]:
    if args.offline_scene_state:
        raw = load_json(args.offline_scene_state)
    else:
        raw = capture_empty_observation(args, str(output / "initial_observation"), None)
        if raw is None:
            raise RuntimeError("initial live observation was not produced")
    raw.setdefault("scene_revision", 1)
    raw.setdefault("frame_id", "base_link")
    if not args.offline_scene_state:
        raw = _review_live_observation(
            args, raw, output / "initial_observation", task_type,
        )
    return raw


def _review_live_observation(
    args: Any,
    raw: dict[str, Any],
    output_dir: Path,
    task_type: str,
) -> dict[str, Any]:
    """Apply the house visual review uniformly to every live observation."""
    if task_type != "build_house":
        return raw
    return review_build_house_scene(args, raw, output_dir)


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
    if int(args.vlm_num_ctx) != 32768:
        raise ValueError(f"stack_demo requires vlm num_ctx=32768, got {args.vlm_num_ctx}")
    if int(args.vlm_num_predict) <= 0:
        args.vlm_num_predict = int(policy["num_predict"])
    print(f"effective_vlm_num_ctx={args.vlm_num_ctx}", flush=True)
    print(f"effective_vlm_num_predict={args.vlm_num_predict}", flush=True)
    print(f"model={args.model}", flush=True)


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
    visible_ids = {str(item.get("track_id")) for item in objects if item.get("track_id")}
    remembered = []
    for track_id, track in sorted((memory.get("tracks") or {}).items()):
        # A confirmed identity handoff means the stale ID and its successor
        # refer to one physical block. Keeping the stale ID as a remembered
        # obstacle would create coincident geometry and reject every grasp of
        # the visible successor as a finger collision.
        if (
            track_id in visible_ids
            or track.get("manipulation_state") == "held_by_gripper"
            or track.get("superseded_by_track_id")
        ):
            continue
        center = track.get("center_base_m")
        size = track.get("dimensions_m")
        if not (_finite_xyz(center) and _finite_xyz(size)):
            continue
        remembered.append({
            "id": f"remembered_{track_id}",
            "track_id": str(track_id),
            "label": str(track.get("label") or track.get("semantic_shape") or "remembered object"),
            "visual_color": str(track.get("color") or "unknown"),
            "geometry_center_m": [float(value) for value in center[:3]],
            "dimensions_m": [float(value) for value in size[:3]],
            "table_yaw_deg": float(track.get("table_yaw_deg") or 0.0),
            "orientation_confidence": 0.0,
            "collision_memory_only": True,
            "last_seen_revision": int(track.get("last_seen_revision", revision)),
        })
    raw["remembered_collision_obstacles"] = remembered


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
