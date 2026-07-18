import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.object_tracking import update_scene_tracks
from robot_scene_pipeline.ollama_policy_client import call_policy
from tools.workflows.stack_demo.app import (
    _LiveObserver,
    _bind_observation_tracks,
    _review_live_observation,
    main,
)
from tools.workflows.stack_demo.clutter.edge_generation import (
    _opposite_side,
    _staging_clearance_gain,
    generate_physical_edges,
)
from tools.workflows.stack_demo.clutter.extraction_planner import _final_gate_candidates
from tools.workflows.stack_demo.clutter.grasp_edges import (
    evaluate_gripper_pose_clearance,
    scan_grasp_yaws,
)
from tools.workflows.stack_demo.clutter.path_safety import placement_path_checks
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.policy.qwen_client import StatelessQwenClient
from tools.workflows.stack_demo.policy.edge_selector import (
    EdgeSelectionOutcome,
    _repair_redundant_action_aliases,
)
from tools.workflows.stack_demo.policy.schemas import (
    PolicyOutputError,
    parse_edge_selection,
    parse_target_selection,
)
from tools.workflows.stack_demo.policy.target_selector import QwenTargetSelector
from tools.workflows.stack_demo.perception_semantic_review import _apply_decisions, _candidate_view
from tools.workflows.stack_demo.common.action_edges import ActionType
from tools.workflows.stack_demo.common.scene_state import build_clutter_scene_state

from tests.new_arch_fixtures import MockQwenClient, config, edge, placement, raw_object, scene


class NewSceneEdgePolicyTests(unittest.TestCase):
    def test_every_live_house_observation_uses_semantic_review(self):
        args = SimpleNamespace()
        raw = {"objects": []}
        reviewed = {"objects": [], "reviewed": True}
        with patch(
            "tools.workflows.stack_demo.app.review_build_house_scene",
            return_value=reviewed,
        ) as review:
            self.assertIs(
                _review_live_observation(
                    args, raw, Path("/tmp/house_observation"), "build_house",
                ),
                reviewed,
            )
            self.assertIs(
                _review_live_observation(
                    args, raw, Path("/tmp/organize_observation"), "organize_blocks",
                ),
                raw,
            )
        review.assert_called_once_with(
            args, raw, Path("/tmp/house_observation"),
        )

    def test_post_action_observer_reviews_before_track_binding(self):
        args = SimpleNamespace()
        before = SimpleNamespace(scene_revision=7)
        selected_edge = SimpleNamespace()
        observer = _LiveObserver(
            args, config(), "build_house", Path("/tmp/run"), 3, before,
            (), [], set(), {}, selected_edge,
        )
        raw = {"objects": []}
        reviewed = {"objects": [], "reviewed": True}
        expected_scene = object()
        with (
            patch(
                "tools.workflows.stack_demo.app.capture_empty_current_pose",
                return_value=raw,
            ),
            patch(
                "tools.workflows.stack_demo.app._review_live_observation",
                return_value=reviewed,
            ) as review,
            patch("tools.workflows.stack_demo.app._bind_observation_tracks") as bind,
            patch("tools.workflows.stack_demo.app.predicted_track_centers", return_value={}),
            patch("tools.workflows.stack_demo.app._merge_expected_tracks", return_value=()),
            patch(
                "tools.workflows.stack_demo.app.build_clutter_scene_state",
                return_value=expected_scene,
            ),
        ):
            self.assertIs(observer.observe("post_place"), expected_scene)
        review.assert_called_once_with(
            args,
            raw,
            Path("/tmp/run/observation_cycle_003_post_place"),
            "build_house",
        )
        self.assertIs(bind.call_args.args[0], reviewed)

    def test_vlm_semantic_review_corrects_primary_and_promotes_only_measured_candidate(self):
        primary = raw_object(
            1, "ignored", [0.36, 0.02, 0.0],
            label="square green", visual_color="green",
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        low_measured = raw_object(
            2, "ignored", [0.31, 0.25, 0.0],
            label="rectangle", visual_color="red",
            primary_detector_passed=False, pointcloud_geometry_valid=True,
        )
        low_without_depth = raw_object(
            3, "ignored", [0.40, 0.10, 0.0],
            label="square blue", visual_color="blue",
            primary_detector_passed=False, pointcloud_geometry_valid=False,
        )
        reviewed = _apply_decisions(
            {"objects": [primary], "semantic_review_candidates": [primary, low_measured, low_without_depth]},
            [primary, low_measured, low_without_depth],
            {"decisions": [
                {"candidate_id": 1, "accept": True, "corrected_shape": "triangle", "confidence": "high"},
                {"candidate_id": 2, "accept": True, "corrected_shape": "concave_rectangle", "confidence": "medium"},
                {"candidate_id": 3, "accept": True, "corrected_shape": "square", "confidence": "high"},
            ]},
        )
        self.assertEqual([item["id"] for item in reviewed["objects"]], [1, 2])
        self.assertEqual(reviewed["objects"][0]["label"], "triangle")
        self.assertEqual(reviewed["objects"][1]["label"], "concave rectangle")
        self.assertTrue(reviewed["objects"][1]["vlm_promoted_low_confidence_candidate"])

    def test_vlm_semantic_review_never_deletes_primary_detection(self):
        primary = raw_object(
            1, "ignored", [0.36, 0.02, 0.0],
            label="triangle", visual_color="green",
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        reviewed = _apply_decisions(
            {"objects": [primary]}, [primary],
            {"decisions": [{
                "candidate_id": 1, "accept": False,
                "corrected_shape": "square", "confidence": "high",
            }]},
        )
        self.assertEqual(reviewed["objects"], [primary])

    def test_vlm_semantic_review_suppresses_duplicate_low_score_proposals(self):
        first = raw_object(
            1, "ignored", [0.26, 0.17, 0.0], label="semi circle",
            bbox=[305.6, 365.7, 358.6, 426.8], confidence=0.42,
            primary_detector_passed=False, pointcloud_geometry_valid=True,
        )
        duplicate = raw_object(
            2, "ignored", [0.26, 0.17, 0.0], label="semi circle",
            bbox=[305.5, 365.6, 358.7, 426.7], confidence=0.24,
            primary_detector_passed=False, pointcloud_geometry_valid=True,
        )
        reviewed = _apply_decisions(
            {"objects": [], "semantic_review_candidates": [first, duplicate]},
            [first, duplicate],
            {"decisions": [
                {"candidate_id": 1, "accept": True, "corrected_shape": "square", "confidence": "high"},
                {"candidate_id": 2, "accept": True, "corrected_shape": "square", "confidence": "high"},
            ]},
        )
        self.assertEqual([item["id"] for item in reviewed["objects"]], [1])
        self.assertEqual(reviewed["semantic_review_suppressed_duplicate_ids"], [2])

    def test_vlm_semantic_review_suppresses_duplicate_primary_proposals(self):
        first = raw_object(
            1, "ignored", [0.388, 0.271, 0.023], label="square yellow",
            bbox=[100.0, 100.0, 170.0, 170.0], confidence=0.95,
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        duplicate = raw_object(
            2, "ignored", [0.388, 0.271, 0.023], label="semi circle",
            bbox=[100.5, 100.5, 169.5, 169.5], confidence=0.90,
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        reviewed = _apply_decisions(
            {"objects": [first, duplicate]}, [first, duplicate],
            {"decisions": [
                {"candidate_id": 1, "accept": True, "corrected_shape": "square", "confidence": "high"},
                {"candidate_id": 2, "accept": False, "corrected_shape": "semi_circle", "confidence": "low"},
            ]},
        )
        self.assertEqual([item["id"] for item in reviewed["objects"]], [1])
        self.assertEqual(reviewed["semantic_review_suppressed_duplicate_ids"], [2])

    def test_low_score_cube_visually_classified_square_recovers_after_placement(self):
        placed = raw_object(
            5, "ignored", [0.384, 0.271, 0.0],
            label="semi circle", visual_color="yellow",
            size=(0.0253, 0.0234, 0.0225),
            primary_detector_passed=False, pointcloud_geometry_valid=True,
            bbox_xyxy_px=[109.5, 143.2, 174.0, 202.3],
        )
        reviewed = _apply_decisions(
            {"objects": [], "semantic_review_candidates": [placed]}, [placed],
            {"decisions": [{
                "candidate_id": 5, "accept": False,
                "corrected_shape": "square", "confidence": "low",
            }]},
        )
        self.assertEqual([item["id"] for item in reviewed["objects"]], [5])
        self.assertEqual(reviewed["objects"][0]["label"], "square yellow")
        self.assertTrue(
            reviewed["objects"][0]["low_confidence_square_geometry_recovery"]
        )
        self.assertEqual(reviewed["semantic_review_recovered_candidate_ids"], [5])

    def test_green_square_is_forced_through_square_triangle_confusion_review(self):
        candidate = raw_object(
            1, "ignored", [0.363, 0.022, 0.0],
            label="square green", visual_color="green",
            bbox=[570.7, 154.2, 638.4, 238.6], mask_area_px=4943,
            dimensions_m=[0.0351, 0.0255, 0.0188],
            footprint_aspect_ratio=1.38,
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        view = _candidate_view(candidate)
        self.assertTrue(view["mandatory_square_triangle_review"])
        self.assertEqual(view["shape_choice_constraint"], ["square", "triangle"])
        self.assertAlmostEqual(view["mask_bbox_fill_ratio"], 0.865, places=3)

    def test_clipped_green_square_verdict_cannot_become_house_support(self):
        candidate = raw_object(
            3, "ignored", [0.303, 0.338, 0.0],
            label="square green", visual_color="green",
            bbox=[0.47, 292.56, 47.74, 356.36], mask_shape=[736, 960],
            dimensions_m=[0.0305, 0.0131, 0.0224],
            footprint_aspect_ratio=2.335,
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        reviewed = _apply_decisions(
            {"objects": [candidate]}, [candidate],
            {"decisions": [{
                "candidate_id": 3, "accept": True,
                "corrected_shape": "square", "confidence": "high",
            }]},
        )
        self.assertEqual(reviewed["objects"], [])
        self.assertEqual(reviewed["semantic_review_safety_rejections"], [{
            "candidate_id": 3,
            "reason": "green_square_silhouette_clipped_at_image_boundary",
        }])
        self.assertTrue(_candidate_view(candidate)["silhouette_touches_image_boundary"])

    def test_unclipped_visually_confirmed_green_square_remains_support_candidate(self):
        candidate = raw_object(
            3, "ignored", [0.36, 0.18, 0.0],
            label="square green", visual_color="green",
            bbox=[300.0, 200.0, 360.0, 260.0], mask_shape=[736, 960],
            dimensions_m=[0.024, 0.023, 0.024],
            footprint_aspect_ratio=1.04,
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        reviewed = _apply_decisions(
            {"objects": [candidate]}, [candidate],
            {"decisions": [{
                "candidate_id": 3, "accept": True,
                "corrected_shape": "square", "confidence": "high",
            }]},
        )
        self.assertEqual([item["id"] for item in reviewed["objects"]], [3])
        self.assertEqual(reviewed["semantic_review_safety_rejections"], [])

    def test_unreviewed_green_yolo_square_is_not_used_as_support(self):
        candidate = raw_object(
            3, "ignored", [0.36, 0.18, 0.0],
            label="square green", visual_color="green",
            bbox=[300.0, 200.0, 360.0, 260.0], mask_shape=[736, 960],
            primary_detector_passed=True, pointcloud_geometry_valid=True,
        )
        reviewed = _apply_decisions(
            {"objects": [candidate]}, [candidate], None,
        )
        self.assertEqual(reviewed["objects"], [])
        self.assertEqual(
            reviewed["semantic_review_safety_rejections"][0]["reason"],
            "green_square_missing_reliable_square_triangle_review",
        )

    def test_unique_redundant_extract_to_staging_id_is_resolved_exactly(self):
        staging = edge(
            candidate_id="edge_r1_track_blue_02__left_support_lower_staging_1_grasp_1",
            action_type=ActionType.EXTRACT_TO_STAGING,
        )
        parsed, repairs = _repair_redundant_action_aliases({
            "selected_candidate_id": (
                "edge_r1_track_blue_02__left_support_lower_"
                "extract_to_staging_1_grasp_1"
            ),
            "backup_candidate_ids": [],
            "reason_codes": ["all_physical_prechecks_passed"],
        }, (staging,))

        self.assertEqual(parsed["selected_candidate_id"], staging.candidate_id)
        self.assertEqual(repairs, [{
            "received": (
                "edge_r1_track_blue_02__left_support_lower_"
                "extract_to_staging_1_grasp_1"
            ),
            "resolved": staging.candidate_id,
        }])

    def test_unknown_candidate_id_is_not_fuzzily_repaired(self):
        staging = edge(
            candidate_id="edge_r1_target_staging_1_grasp_1",
            action_type=ActionType.EXTRACT_TO_STAGING,
        )
        parsed, repairs = _repair_redundant_action_aliases({
            "selected_candidate_id": "edge_r1_target_staging_9_grasp_1",
            "backup_candidate_ids": [],
            "reason_codes": ["x"],
        }, (staging,))

        self.assertEqual(
            parsed["selected_candidate_id"], "edge_r1_target_staging_9_grasp_1",
        )
        self.assertEqual(repairs, [])

    def test_missing_last_known_block_remains_collision_only_obstacle(self):
        memory = {"tracks": {"track_blue_02": {
            "track_id": "track_blue_02",
            "label": "square blue",
            "semantic_shape": "square",
            "color": "blue",
            "center_base_m": [0.2699, 0.0926, -0.0020],
            "dimensions_m": [0.0236, 0.0232, 0.0235],
            "table_yaw_deg": 0.0,
            "last_seen_revision": 1,
            "history": [{"scene_revision": 1}],
        }}}
        raw = {
            "scene_revision": 7,
            "frame_id": "base_link",
            "objects": [raw_object(
                1, "track_blue_01", [0.3994, 0.2812, -0.0011],
                label="square blue", size=[0.0246, 0.0240, 0.0254], yaw_deg=-71.88,
            )],
        }

        _bind_observation_tracks(raw, memory)
        state = build_clutter_scene_state(raw, config().workspace, expected_tracks=(
            "track_blue_01", "track_blue_02",
        ))
        target = state.current_objects[0]
        place_pose = {"position_m": [0.3, 0.08, -0.00145], "yaw_deg": -71.88}
        checks = placement_path_checks(
            state,
            target,
            place_pose,
            {
                "release_pose": {"position_m": [0.3, 0.08, -0.00145], "yaw_deg": -85.0},
                "transport_path": [],
            },
            config(),
        )

        self.assertEqual(state.visible_tracks, ("track_blue_01",))
        self.assertEqual(state.missing_expected_tracks, ("track_blue_02",))
        self.assertEqual(
            [item.track_id for item in state.collision_obstacles],
            ["track_blue_02"],
        )
        self.assertFalse(checks["place_descent_safe"])
        self.assertIn("track_blue_02", checks["place_blocking_track_ids"])

    def test_superseded_track_is_not_a_coincident_collision_obstacle(self):
        memory = {"tracks": {
            "track_yellow_01": {
                "track_id": "track_yellow_01",
                "label": "square yellow",
                "semantic_shape": "square",
                "color": "yellow",
                "center_base_m": [0.3011, 0.0806, -0.0019],
                "dimensions_m": [0.0237, 0.0236, 0.0223],
                "last_seen_revision": 3,
                "history": [{"scene_revision": 1}, {"scene_revision": 3}],
                "superseded_by_track_id": "track_yellow_03",
            },
            "track_yellow_03": {
                "track_id": "track_yellow_03",
                "label": "square yellow",
                "semantic_shape": "square",
                "color": "yellow",
                "center_base_m": [0.3021, 0.0810, -0.0026],
                "dimensions_m": [0.0237, 0.0227, 0.0249],
                "last_seen_revision": 7,
                "history": [{"scene_revision": 6}, {"scene_revision": 7}],
            },
        }}
        raw = {
            "scene_revision": 8,
            "frame_id": "base_link",
            "objects": [raw_object(
                1, "track_yellow_03", [0.3021, 0.0810, -0.0026],
                label="square yellow", size=[0.0237, 0.0227, 0.0249], yaw_deg=4.2,
            )],
        }

        _bind_observation_tracks(raw, memory)

        self.assertEqual(raw["remembered_collision_obstacles"], [])

    def test_final_gate_tries_policy_backup_then_equivalent_staging_points(self):
        selected = edge(
            candidate_id="staging_selected",
            action_type=ActionType.EXTRACT_TO_STAGING,
            task_role="roof",
        )
        equivalent = edge(
            candidate_id="staging_equivalent",
            action_type=ActionType.EXTRACT_TO_STAGING,
            task_role="roof",
        )
        different_role = edge(
            candidate_id="staging_other_role",
            action_type=ActionType.EXTRACT_TO_STAGING,
            task_role="triangle_top",
        )
        policy_backup = edge(
            candidate_id="policy_backup",
            action_type=ActionType.NUDGE_BLOCKER,
            task_role="roof",
        )
        outcome = EdgeSelectionOutcome(
            selected,
            (policy_backup.candidate_id,),
            "qwen_edge_selection",
            ("ranked",),
            {},
            {},
        )
        ordered = _final_gate_candidates(
            outcome,
            (selected, equivalent, different_role, policy_backup),
        )
        self.assertEqual(
            [item.candidate_id for item in ordered],
            ["staging_selected", "policy_backup", "staging_equivalent"],
        )

    def test_code_owned_push_direction_has_opposite_contact_side(self):
        self.assertEqual(_opposite_side([1.0, 0.0, 0.0]), "-x")
        self.assertEqual(_opposite_side([-1.0, 0.0, 0.0]), "+x")
        self.assertEqual(_opposite_side([0.0, 1.0, 0.0]), "-y")
        self.assertEqual(_opposite_side([0.0, -1.0, 0.0]), "+y")

    def test_direct_grasp_uses_physical_fingertip_length_near_adjacent_block(self):
        state = scene([
            raw_object(
                1, "track_red_01", [0.3593, 0.1527, -0.0012],
                size=[0.0227, 0.0209, 0.0258], yaw_deg=13.47,
            ),
            raw_object(
                2, "track_blue_02", [0.3857, 0.1544, -0.0014],
                label="square blue", size=[0.0239, 0.0226, 0.0248], yaw_deg=3.26,
            ),
        ])
        target = next(obj for obj in state.current_objects if obj.track_id == "track_red_01")

        scan = scan_grasp_yaws(target, state.current_objects, config())

        self.assertTrue(scan.graspable)
        safe_yaws = [float(item["yaw_deg"]) for item in scan.samples if item["safe"]]
        self.assertIn(5.0, safe_yaws)
        self.assertIn(10.0, safe_yaws)
        selected = scan.safe_intervals[0].selected_check
        self.assertLessEqual(selected["fingertip_axial_overhang_m"], 0.001)

    def test_taller_neighbor_in_open_upper_finger_descent_envelope_blocks_yaw(self):
        state = scene([
            raw_object(
                1, "target", [0.3257, 0.1544, -0.0016],
                size=[0.0215, 0.0207, 0.0258], yaw_deg=89.39,
            ),
            raw_object(
                2, "near_rectangle", [0.3908, 0.2148, -0.0051],
                label="rectangle", size=[0.0588, 0.0282, 0.0127], yaw_deg=64.53,
            ),
        ])
        target = state.current_objects[0]

        check = evaluate_gripper_pose_clearance(
            target, state.current_objects, 89.39, config(),
        )

        self.assertTrue(check["finger_safe"])
        self.assertFalse(check["upper_finger_safe"])
        self.assertFalse(check["descent_safe"])
        self.assertEqual(
            check["upper_finger_blocking_track_ids"], ["near_rectangle"],
        )

    def test_placement_rejects_neighbor_hit_by_one_upper_gripper_side(self):
        state = scene([
            raw_object(
                1, "placed_blue", [0.3257, 0.1544, -0.0016],
                label="square blue", size=[0.0215, 0.0207, 0.0258], yaw_deg=89.39,
            ),
            raw_object(
                2, "other_blue", [0.3908, 0.2148, -0.0051],
                label="square blue", size=[0.0588, 0.0282, 0.0127], yaw_deg=64.53,
            ),
        ])
        target = state.current_objects[0]
        place_pose = {
            "frame_id": "base_link",
            "position_m": list(target.center_xyz_m),
            "yaw_deg": target.yaw_deg,
        }

        checks = placement_path_checks(
            state,
            target,
            place_pose,
            {
                "release_pose": {**place_pose, "yaw_deg": 89.39},
                "transport_path": [],
            },
            config(),
        )

        # The narrow fingertips clear, but one wide upper finger does not.
        self.assertTrue(checks["place_finger_safe"])
        self.assertFalse(checks["place_upper_finger_safe"])
        self.assertFalse(checks["place_gripper_descent_safe"])
        self.assertFalse(checks["place_descent_safe"])
        self.assertFalse(checks["release_safe"])
        self.assertIn("other_blue", checks["place_blocking_track_ids"])
        self.assertEqual(
            checks["place_upper_finger_blocking_track_ids"], ["other_blue"],
        )

    def test_blocker_staging_gain_uses_destination_not_initial_separation(self):
        state = scene([
            raw_object(1, "target", [0.3269, 0.2020, 0.0]),
            raw_object(2, "blocker", [0.3572, 0.2891, 0.0]),
        ])

        gain = _staging_clearance_gain(
            state.current_objects[0],
            state.current_objects[1],
            {"position_m": [0.3000, 0.2800, 0.0]},
        )

        self.assertLess(gain, 0.0)
        self.assertAlmostEqual(gain, -0.0097, places=4)

    def test_old_manipulated_track_remains_eligible_for_destination_rebinding(self):
        memory = {"tracks": {"track_red_01": {
            "track_id": "track_red_01",
            "label": "square red",
            "color": "red",
            "semantic_shape": "square",
            "center_base_m": [0.28, 0.2475, 0.02],
            "dimensions_m": [0.025, 0.023, 0.024],
            "last_seen_revision": 1,
            "manipulation_state": "placed_unverified",
            "history": [],
        }}}
        detection = raw_object(
            8, "temporary", [0.282, 0.248, 0.02], color="red",
            size=[0.025, 0.023, 0.024],
        )
        detection.pop("track_id", None)
        update_scene_tracks(memory, [detection], 8)
        self.assertEqual(detection["track_id"], "track_red_01")

    def test_01_main_entry_loads_new_planner(self):
        fixture = Path(__file__).parent / "fixtures" / "new_arch_single_red_scene.json"
        with tempfile.TemporaryDirectory() as output:
            mock = Path(output) / "mock_policy"
            mock.mkdir()
            (mock / "edge_selection.json").write_text(json.dumps({
                "selected_candidate_id": "edge_r1_track_red_01_grasp_1",
                "backup_candidate_ids": [],
                "reason_codes": ["direct_task_progress"],
            }))
            result = main([
                "--task-type", "organize_blocks",
                "--offline-scene-state", str(fixture),
                "--mock-policy-response-dir", str(mock),
                "--output-dir", output,
                "--no-image",
            ])
            self.assertEqual(result, 0)
            self.assertTrue(list(Path(output).glob("cycle_*/physical_action_edges.json")))

    def test_02_target_options_require_feasible_first_step(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", {"t1": generated.edges_by_target["t1"], "t2": ()}, generated.grasp_scans)
        self.assertEqual([item.target_track_id for item in options], ["t1"])

    def test_03_infeasible_object_never_reaches_qwen(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        client = MockQwenClient([])
        outcome = QwenTargetSelector(client).select(current, {}, [], ())
        self.assertIsNone(outcome.selected)
        self.assertFalse(client.calls)

    def test_04_target_option_contains_required_geometry_features(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.56, 0.0, 0.02])])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        value = options[0].to_dict()
        for key in ("neighbor_count", "edge_clearances_m", "blocked_sides", "safe_grasp_interval_count", "estimated_clearance_cost", "releases_track_ids"):
            self.assertIn(key, value)

    def test_05_qwen_initial_request_has_no_assistant_history(self):
        args = _ollama_args()
        response = SimpleNamespace(
            transport_status="ok", schema_valid=True, content='{"selected_target_option_id":"x","reason_codes":["a"]}',
            parsed_decision={"selected_target_option_id": "x", "reason_codes": ["a"]},
            error_type=None, error_message=None, generation_status="parsed",
        )
        with patch("tools.workflows.stack_demo.policy.qwen_client.call_policy", return_value=response) as call:
            StatelessQwenClient(args).call("target_selection", {"scene_revision": 4}, {}, (), None)
        messages = call.call_args.args[2]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])

    def test_06_request_contains_only_latest_scene_revision(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])], revision=9)
        generated = generate_physical_edges(current, ["t1", "t2"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        client = MockQwenClient([{"selected_target_option_id": options[0].target_option_id, "reason_codes": ["low_clearance_cost"]}])
        outcome = QwenTargetSelector(client).select(current, {}, options)
        self.assertEqual(outcome.request["scene_revision"], 9)
        self.assertTrue(all(item["scene_revision"] == 9 for item in outcome.request["target_options"]))

    def test_07_unknown_target_option_id_is_rejected(self):
        with self.assertRaises(PolicyOutputError):
            parse_target_selection({"selected_target_option_id": "missing", "reason_codes": ["x"]}, [], 1)

    def test_08_unknown_candidate_id_is_rejected(self):
        with self.assertRaises(PolicyOutputError):
            parse_edge_selection({"selected_candidate_id": "missing", "backup_candidate_ids": [], "reason_codes": ["x"]}, [edge()], 1)

    def test_09_stale_scene_revision_is_rejected(self):
        with self.assertRaises(PolicyOutputError):
            parse_edge_selection({"selected_candidate_id": "edge_1", "backup_candidate_ids": [], "reason_codes": ["x"]}, [edge(revision=1)], 2)

    def test_10_invalid_json_gets_only_one_format_repair(self):
        args = _ollama_args(vlm_max_backend_retries=1)
        responses = [_response("not json"), _response("still not json")]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses) as post:
            result = call_policy(args, "target_selection", [{"role": "user", "content": "x"}], {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}, "additionalProperties": False})
        self.assertFalse(result.schema_valid)
        self.assertEqual(post.call_count, 2)

    def test_11_second_invalid_output_never_triggers_code_reselection(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])])
        generated = generate_physical_edges(current, ["t1", "t2"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        client = MockQwenClient([{"selected_target_option_id": "unknown", "reason_codes": ["x"]}])
        outcome = QwenTargetSelector(client).select(current, {}, options)
        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.decision_source, "policy_invalid_output")

    def test_14_safe_diagonal_grasp_is_accepted(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        result = scan_grasp_yaws(current.current_objects[0], current.current_objects, config())
        self.assertTrue(result.graspable)
        self.assertNotIn(result.safe_intervals[0].selected_yaw_deg, (0.0, 90.0))

    def test_15_non_edge_aligned_yaw_generates_edge(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], yaw_deg=0.0)])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        yaw = generated.edges_by_target["t1"][0].physical_parameters["grasp_yaw_deg"]
        self.assertNotIn(yaw, (0.0, 90.0))

    def test_normal_block_preserves_grasp_yaw_through_release(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], yaw_deg=30.0)])

        def target(obj, interval):
            value = placement(obj, interval)
            return value.__class__(
                **{**value.__dict__, "place_pose": {
                    "frame_id": "base_link",
                    "position_m": [0.28, 0.18, 0.02],
                    "yaw_deg": 10.0,
                }}
            )

        generated = generate_physical_edges(
            current, ["t1"], "organize_blocks", config(), target, lambda obj: None,
        )
        physical = generated.edges_by_target["t1"][0].physical_parameters
        self.assertEqual(
            physical["release_gripper_yaw_deg"],
            physical["grasp_yaw_deg"],
        )
        self.assertEqual(physical["requested_place_object_yaw_deg"], 10.0)
        self.assertEqual(physical["expected_place_object_yaw_deg"], 30.0)
        self.assertEqual(
            physical["placement_yaw_policy"],
            "preserve_grasp_yaw_until_release",
        )
        self.assertEqual(physical["orientation_policy"], "downward_yaw_only")
        self.assertEqual(len(physical["transport_path"]), 2)
        self.assertEqual(
            physical["transport_path"][1]["yaw_deg"],
            physical["grasp_yaw_deg"],
        )

    def test_16_narrow_yaw_interval_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        def checker(target, objects, yaw):
            safe = 20 <= yaw < 25
            return _geometry_result(safe)
        result = scan_grasp_yaws(current.current_objects[0], current.current_objects, config(), geometry_check=checker)
        self.assertFalse(result.graspable)
        self.assertEqual(result.rejected_narrow_intervals_deg, ((20.0, 25.0),))

    def test_17_single_point_corner_contact_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], effective_contact_length_m=0.001)])
        self.assertFalse(scan_grasp_yaws(current.current_objects[0], current.current_objects, config()).graspable)

    def test_18_gf225_palm_collision_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        def checker(target, objects, yaw):
            value = _geometry_result(True); value["palm_safe"] = False; return value
        self.assertFalse(scan_grasp_yaws(current.current_objects[0], current.current_objects, config(), geometry_check=checker).graspable)

    def test_detector_ids_are_rebound_to_tracks_per_revision(self):
        memory = {}
        first = [
            raw_object(10, "ignored", [0.40, 0.05, 0.02], label="square red"),
            raw_object(20, "ignored", [0.50, 0.05, 0.02], label="square blue"),
        ]
        for item in first:
            item.pop("track_id")
        update_scene_tracks(memory, first, 1)
        red_track = first[0]["track_id"]
        second = [
            raw_object(99, "ignored", [0.402, 0.05, 0.02], label="square red"),
            raw_object(77, "ignored", [0.502, 0.05, 0.02], label="square blue"),
        ]
        for item in second:
            item.pop("track_id")
        update_scene_tracks(memory, second, 2)
        self.assertEqual(second[0]["track_id"], red_track)
        self.assertEqual(second[0]["object_ref"], "scene_2:obj_99")

    def test_duplicate_metric_detections_do_not_create_competing_tracks(self):
        memory = {}
        detections = [
            raw_object(1, "ignored", [0.260, 0.168, 0.02], label="square yellow"),
            raw_object(2, "ignored", [0.261, 0.168, 0.02], label="square yellow"),
        ]
        for item in detections:
            item.pop("track_id")
        update_scene_tracks(memory, detections, 1)
        self.assertEqual(len(detections), 1)
        self.assertEqual(len(memory["tracks"]), 1)
        original_track = detections[0]["track_id"]

        next_frame = [
            raw_object(9, "ignored", [0.260, 0.169, 0.02], label="square yellow"),
        ]
        next_frame[0].pop("track_id")
        update_scene_tracks(memory, next_frame, 2)
        self.assertEqual(next_frame[0]["track_id"], original_track)

    def test_scene_state_uses_measured_color_for_shape_only_detector_label(self):
        obj = raw_object(1, "t1", [0.5, 0.0, 0.02], label="semi square")
        obj["visual_color"] = "yellow"
        state = build_clutter_scene_state(
            {"scene_revision": 1, "objects": [obj]},
            config().workspace,
        )
        self.assertEqual(state.current_objects[0].color, "yellow")

    def test_rebound_held_object_reuses_only_its_previous_valid_size(self):
        memory = {}
        first = raw_object(
            1, "ignored", [0.30, 0.18, 0.00],
            label="square yellow", color="yellow", size=[0.024, 0.023, 0.025],
        )
        first.pop("track_id")
        update_scene_tracks(memory, [first], 1)
        track_id = first["track_id"]
        lifted = raw_object(
            9, "ignored", [0.297, 0.190, 0.094],
            label="square yellow", color="yellow", size=[0.024, 0.023, 0.025],
        )
        lifted.pop("track_id")
        lifted["center_3d_base_m"] = lifted["geometry_center_m"]
        lifted["geometry_center_m"] = None
        lifted["dimensions_m"] = None
        lifted["pointcloud_geometry_valid"] = False
        update_scene_tracks(
            memory,
            [lifted],
            2,
            predicted_centers={track_id: [0.30, 0.18, 0.10]},
        )
        self.assertEqual(lifted["track_id"], track_id)
        self.assertEqual(lifted["dimensions_m"], [0.024, 0.023, 0.025])
        self.assertTrue(lifted["dimensions_temporal_fallback"])
        self.assertEqual(lifted["dimensions_source"], "previous_valid_same_track")
        self.assertGreaterEqual(lifted["track_match_confidence"], 0.5)
        state = build_clutter_scene_state(
            {"scene_revision": 2, "objects": [lifted]},
            config().workspace,
            expected_tracks=[track_id],
        )
        self.assertEqual(state.current_objects[0].size_xyz_m, (0.024, 0.023, 0.025))
        self.assertAlmostEqual(state.current_objects[0].center_xyz_m[2], 0.094)

    def test_post_action_anchor_survives_tight_same_color_shape_misclassification(self):
        memory = {}
        original = raw_object(
            1, "ignored", [0.4459, 0.1392, -0.001],
            label="square yellow", visual_color="yellow", size=[0.0212, 0.0205, 0.0237],
        )
        original.pop("track_id")
        update_scene_tracks(memory, [original], 1)
        track_id = original["track_id"]
        memory["tracks"][track_id]["manipulation_state"] = "placed_unverified"

        released = raw_object(
            9, "ignored", [0.3755, 0.1781, -0.0006],
            label="semi circle", visual_color="yellow", size=[0.0213, 0.0209, 0.0226],
        )
        released.pop("track_id")
        _, assignments = update_scene_tracks(
            memory,
            [released],
            2,
            predicted_centers={track_id: [0.3776, 0.1782, 0.015]},
        )
        self.assertEqual(released["track_id"], track_id)
        features = assignments[0]["match_features"]
        self.assertFalse(features["shape_compatible"])
        self.assertTrue(features["action_anchor_shape_override"])

    def test_post_action_anchor_does_not_override_shape_outside_tight_gate(self):
        memory = {}
        original = raw_object(
            1, "ignored", [0.4459, 0.1392, -0.001],
            label="square yellow", visual_color="yellow", size=[0.0212, 0.0205, 0.0237],
        )
        original.pop("track_id")
        update_scene_tracks(memory, [original], 1)
        track_id = original["track_id"]
        other = raw_object(
            9, "ignored", [0.3970, 0.1782, 0.015],
            label="semi circle", visual_color="yellow", size=[0.0213, 0.0209, 0.0226],
        )
        other.pop("track_id")
        update_scene_tracks(
            memory,
            [other],
            2,
            predicted_centers={track_id: [0.3776, 0.1782, 0.015]},
        )
        self.assertNotEqual(other["track_id"], track_id)

    def test_new_detection_without_size_is_ignored_without_losing_valid_scene(self):
        invalid = raw_object(1, "ignored", [0.4, 0.0, 0.02])
        invalid.pop("track_id")
        invalid["dimensions_m"] = None
        update_scene_tracks({}, [invalid], 1)
        valid = raw_object(2, "valid", [0.5, 0.0, 0.02])
        observation = {"scene_revision": 1, "objects": [valid, invalid]}
        state = build_clutter_scene_state(observation, config().workspace)
        self.assertEqual([obj.track_id for obj in state.current_objects], ["valid"])
        self.assertEqual(
            observation["ignored_objects_without_metric_geometry"][0]["track_id"],
            invalid["track_id"],
        )


def _geometry_result(safe):
    return {
        "opening_ok": safe, "center_offset_ok": safe, "contact_length_ok": safe,
        "finger_safe": safe, "palm_safe": safe, "descent_safe": safe, "lift_safe": safe,
        "fingertip_clearance_m": 0.02, "palm_clearance_m": 0.02,
        "effective_contact_length_m": 0.02, "vertical_lift_clearance_m": 0.1,
        "blocking_track_ids": [],
    }


def _ollama_args(**changes):
    values = dict(model="qwen3-vl:8b-instruct", ollama_url="http://local/api/chat", vlm_think_mode="off", vlm_num_ctx=0, vlm_num_predict=0, vlm_num_gpu=-1, vlm_read_timeout_sec=1, vlm_keep_alive="1h", vlm_max_backend_retries=1, vlm_max_budget_retries=1, vlm_finalizer_num_predict=768, output_dir=None)
    values.update(changes)
    return SimpleNamespace(**values)


class _response:
    status_code = 200
    text = ""
    def __init__(self, content): self.content = content
    def json(self):
        return {"message": {"role": "assistant", "content": self.content}, "done": True, "done_reason": "stop"}


if __name__ == "__main__":
    unittest.main()
