import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robot_scene_pipeline.ollama_policy_client import (
    DEFAULT_VLM_NUM_CTX,
    POLICY_GENERATION_CONFIG,
    call_policy,
)
from tests.new_arch_fixtures import config, raw_object, scene
from tools.workflows.stack_demo.clutter.grasp_edges import (
    evaluate_gripper_pose_clearance,
    scan_grasp_yaws,
)
from tools.workflows.stack_demo.clutter.edge_generation import (
    TargetSpec,
    generate_physical_edges,
)
from tools.workflows.stack_demo.common.pose3d import (
    axis_alignment_error_deg,
    rotation_error_deg,
    transform_vector,
)
from tools.workflows.stack_demo.house.orientation_trajectory import (
    FULL_3D_ORIENTATION_SHAPES,
    beam_axis_from_supports,
    build_airborne_orientation_plan,
    build_triangle_tabletop_step_pose,
    orientation_airborne_blocking_tracks,
    solve_target_object_orientations,
    semantic_face_ready_for_structural_tilt,
    special_shape_evidence_ready,
    triangle_tabletop_evidence_ready,
)
from tools.workflows.stack_demo.house.placement import (
    _with_verified_triangle_tabletop_pose,
    house_placement_target,
)
from tools.workflows.stack_demo.house.state import build_house_task_state
from tools.workflows.stack_demo.house.structure import role_observation_checks
from robot_scene_pipeline.fragmented_triangle_merge import (
    merge_fragmented_green_triangle_detections,
)
from robot_scene_pipeline.tabletop_geometry import estimate_triangle_vertical_profile
from tools.workflows.stack_demo.perception_semantic_review import (
    _apply_decisions,
    _attach_shape_semantic_axis,
    _camera_direction_in_base,
    _fuse_special_shape_evidence,
    _green_triangle_metric_evidence,
    _triangle_right_angle_direction_from_silhouette,
    _validate_special_response,
)
from tools.workflows.stack_demo.app import _released_triangle_destination_check


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["ok"], "properties": {"ok": {"type": "boolean"}},
}


class BuildHouse3DFixTests(unittest.TestCase):
    def test_unobstructed_triangle_silhouette_recovers_right_angle_direction(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV silhouette fallback is not installed")
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((100, 120, 3), dtype=np.uint8)
            cv2.fillConvexPoly(
                image,
                np.array([[15, 65], [80, 20], [80, 65]], dtype=np.int32),
                (0, 255, 0),
            )
            snapshot = Path(directory) / "triangle.jpg"
            self.assertTrue(cv2.imwrite(str(snapshot), image))
            direction = _triangle_right_angle_direction_from_silhouette(
                snapshot, {
                    "bbox_xyxy_px": [15, 20, 81, 66],
                    "visual_color": "green",
                },
            )
            self.assertEqual(direction, "right")

    def test_four_corner_prism_projection_does_not_invent_triangle_direction(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV silhouette fallback is not installed")
        with tempfile.TemporaryDirectory() as directory:
            image = np.zeros((120, 140, 3), dtype=np.uint8)
            cv2.fillConvexPoly(
                image,
                np.array([[120, 55], [38, 89], [21, 60], [63, 24]], dtype=np.int32),
                (0, 255, 0),
            )
            snapshot = Path(directory) / "triangle_prism.jpg"
            self.assertTrue(cv2.imwrite(str(snapshot), image))
            self.assertIsNone(
                _triangle_right_angle_direction_from_silhouette(
                    snapshot, {
                        "bbox_xyxy_px": [21, 24, 121, 90],
                        "visual_color": "green",
                    },
                ),
            )

    def test_triangle_image_direction_is_transformed_by_camera_rotation(self):
        item = {
            "long_axis_base": [1.0, 0.0, 0.0],
            "visible_face_normal_base": [0.0, 0.0, 1.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        decision = {
            "shape_label": "triangle",
            "orientation_confidence": 0.95,
            "triangle_right_angle_edge_direction_image": "right",
        }
        self.assertTrue(_attach_shape_semantic_axis(
            item, decision,
            camera_rotation_base=((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        ))
        self.assertAlmostEqual(item["designated_right_angle_edge_base"][0], 0.0)
        self.assertLess(item["designated_right_angle_edge_base"][1], -0.99)

    def test_triangle_metric_model_is_two_by_one_by_one_square_edges(self):
        evidence = _green_triangle_metric_evidence({
            "visual_color": "green",
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.048, 0.024, 0.024],
        }, {
            "square_edge_length_m": 0.024,
            "triangle_hypotenuse_to_square_edge_ratio": 2.0,
            "triangle_face_altitude_to_square_edge_ratio": 1.0,
            "triangle_prism_height_to_square_edge_ratio": 1.0,
            "triangle_dimension_ratio_tolerance": 0.30,
        })
        self.assertTrue(evidence["triangle_geometry_confirmed"])
        self.assertFalse(evidence["square_geometry_confirmed"])

        live_evidence = _green_triangle_metric_evidence({
            "visual_color": "green",
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0323, 0.0187, 0.0245],
        }, {
            "square_edge_length_m": 0.024,
            "triangle_dimension_ratio_tolerance": 0.35,
        })
        self.assertTrue(live_evidence["triangle_geometry_confirmed"])

    def test_metric_triangle_is_not_suppressed_when_side_view_looks_square(self):
        candidate = {
            "id": 1,
            "label": "square green",
            "visual_color": "green",
            "confidence": 0.201,
            "primary_detector_passed": False,
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0400, 0.0258, 0.0186],
            "bbox_xyxy_px": [360, 100, 455, 165],
            "mask_shape": [480, 640],
        }
        reviewed = _apply_decisions(
            {"objects": [], "semantic_review_candidates": [candidate]},
            [candidate],
            {"decisions": [{
                "candidate_id": 1,
                "accept": True,
                "corrected_shape": "square",
                "confidence": "medium",
            }]},
            geometry_model={
                "square_edge_length_m": 0.024,
                "triangle_dimension_ratio_tolerance": 0.35,
            },
        )
        self.assertEqual(len(reviewed["objects"]), 1)
        item = reviewed["objects"][0]
        self.assertEqual(item["label"], "triangle")
        self.assertEqual(item["visual_color"], "green")
        self.assertTrue(item["metric_triangle_shape_override"])
        self.assertEqual(reviewed["semantic_review_metric_triangle_override_ids"], [1])
        self.assertEqual(reviewed["semantic_review_safety_rejections"], [])

    def test_metric_triangle_survives_low_confidence_visual_rejection(self):
        candidate = {
            "id": 2, "label": "triangle green", "visual_color": "green",
            "confidence": 0.359, "primary_detector_passed": False,
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0367, 0.0222, 0.0186],
            "bbox_xyxy_px": [281, 291, 366, 344], "mask_shape": [736, 960],
        }
        for wrong_visual_shape in ("square", "rectangle"):
            with self.subTest(wrong_visual_shape=wrong_visual_shape):
                reviewed = _apply_decisions(
                    {"objects": []}, [candidate], {"decisions": [{
                        "candidate_id": 2, "accept": False,
                        "corrected_shape": wrong_visual_shape, "confidence": "low",
                    }]},
                    geometry_model={
                        "square_edge_length_m": 0.024,
                        "triangle_dimension_ratio_tolerance": 0.35,
                    },
                )
                self.assertEqual([item["id"] for item in reviewed["objects"]], [2])
                self.assertEqual(reviewed["objects"][0]["label"], "triangle")
                self.assertTrue(reviewed["objects"][0]["metric_triangle_shape_override"])

    def test_physically_impossible_blue_sliver_cannot_enter_collision_scene(self):
        candidate = {
            "id": 1, "label": "square blue", "visual_color": "blue",
            "confidence": 0.549, "primary_detector_passed": True,
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0204, 0.0023, 0.0209],
            "bbox_xyxy_px": [320, 200, 350, 250], "mask_shape": [736, 960],
        }
        reviewed = _apply_decisions(
            {"objects": []}, [candidate], {"decisions": [{
                "candidate_id": 1, "accept": True,
                "corrected_shape": "rectangle", "confidence": "high",
            }]},
        )
        self.assertEqual(reviewed["objects"], [])
        self.assertEqual(reviewed["semantic_review_safety_rejections"], [{
            "candidate_id": 1,
            "reason": "physically_implausible_thin_block_geometry",
        }])

    def test_complete_metric_triangle_wins_over_overlapping_partial_square_box(self):
        partial = {
            "id": 1, "label": "square green", "visual_color": "green",
            "confidence": 0.9255, "primary_detector_passed": True,
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0224, 0.0215, 0.0073],
            "bbox_xyxy_px": [364, 103, 421, 154], "mask_shape": [736, 960],
        }
        complete = {
            "id": 2, "label": "square green", "visual_color": "green",
            "confidence": 0.3727, "primary_detector_passed": False,
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0398, 0.0231, 0.0176],
            "bbox_xyxy_px": [365, 103, 450, 161], "mask_shape": [736, 960],
        }
        reviewed = _apply_decisions(
            {"objects": []}, [partial, complete], {"decisions": [
                {"candidate_id": 1, "accept": True, "corrected_shape": "square", "confidence": "high"},
            ]},
            geometry_model={
                "square_edge_length_m": 0.024,
                "triangle_dimension_ratio_tolerance": 0.35,
            },
        )
        self.assertEqual([item["id"] for item in reviewed["objects"]], [2])
        self.assertEqual(reviewed["objects"][0]["label"], "triangle")
        self.assertEqual(reviewed["semantic_review_metric_triangle_override_ids"], [2])
        self.assertEqual(reviewed["semantic_review_suppressed_duplicate_ids"], [])
        self.assertEqual(reviewed["semantic_review_safety_rejections"], [{
            "candidate_id": 1,
            "reason": "physically_implausible_thin_block_geometry",
        }])

    def test_metric_triangle_evidence_supplements_borderline_vlm_confidence(self):
        raw = _special_raw("triangle")
        raw.update({
            "id": 1,
            "visual_color": "green",
            "pointcloud_geometry_valid": True,
            "dimensions_m": [0.0382, 0.0197, 0.0251],
            "visible_face_normal_base": [0.0, 0.0, 1.0],
        })
        decision = _special_decision("1")
        decision.update({
            "shape_label": "triangle",
            "shape_confidence": 0.7078,
            "broad_face_visible": True,
            "triangle_right_angle_edge_direction_image": "right",
            "orientation_confidence": 0.8,
        })
        fused, _ = _fuse_special_shape_evidence(
            {"scene_revision": 1, "objects": [raw]}, [decision], {
                "minimum_shape_confidence": 0.75,
                "square_edge_length_m": 0.024,
                "triangle_dimension_ratio_tolerance": 0.35,
            },
            camera_rotation_base=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        )
        item = fused["objects"][0]
        self.assertTrue(item["metric_triangle_geometry_confirmed"])
        self.assertFalse(item["semantic_shape_uncertain"])
        self.assertEqual(item["designated_right_angle_edge_base"], [1.0, 0.0, 0.0])

    def test_fresh_3d_vertical_profile_recovers_apex_up_when_crop_is_quadrilateral(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is required for point-cloud profile verification")
        points = []
        for height_fraction in np.linspace(0.0, 1.0, 21):
            half_length = 0.024 * (1.0 - height_fraction)
            for x_value in np.linspace(-half_length, half_length, 21):
                for y_value in (-0.012, 0.012):
                    points.append([x_value, y_value, 0.050 + 0.024 * height_fraction])
        profile = estimate_triangle_vertical_profile(
            np.asarray(points), [1.0, 0.0, 0.0], 0.050, 0.074,
        )
        self.assertTrue(profile["apex_up_confirmed"])
        self.assertFalse(profile["apex_down_confirmed"])
        inverted_points = np.asarray([
            [x_value, y_value, 0.050 + 0.024 * height_fraction]
            for height_fraction in np.linspace(0.0, 1.0, 21)
            for x_value in np.linspace(
                -0.024 * height_fraction, 0.024 * height_fraction, 21,
            )
            for y_value in (-0.012, 0.012)
        ])
        inverted = estimate_triangle_vertical_profile(
            inverted_points, [1.0, 0.0, 0.0], 0.050, 0.074,
        )
        self.assertFalse(inverted["apex_up_confirmed"])
        self.assertTrue(inverted["apex_down_confirmed"])

        raw = _special_raw("triangle")
        raw.update({
            "visual_color": "green",
            "dimensions_m": [0.0423, 0.0253, 0.0202],
            "pointcloud_geometry_valid": True,
            "triangle_vertical_profile": profile,
        })
        decision = _special_decision("1")
        decision.update({
            "shape_label": "triangle", "shape_confidence": 0.3384,
            "broad_face_visible": True, "broad_face_confidence": 0.99,
            "triangle_right_angle_edge_direction_image": "unknown",
            "orientation_confidence": 0.99, "occlusion": "none",
        })
        fused, _ = _fuse_special_shape_evidence(
            {"scene_revision": 3, "objects": [raw]}, [decision], {
                "minimum_shape_confidence": 0.75,
                "square_edge_length_m": 0.024,
                "triangle_dimension_ratio_tolerance": 0.35,
            },
        )
        item = fused["objects"][0]
        self.assertFalse(item["semantic_shape_uncertain"])
        self.assertEqual(item["designated_right_angle_edge_base"], [0.0, 0.0, 1.0])
        self.assertTrue(item["fresh_3d_apex_up_confirmed"])
        self.assertEqual(
            item["triangle_right_angle_direction_source"],
            "fresh_mask_depth_vertical_profile_apex_up",
        )

    def test_commanded_triangle_quaternion_cannot_override_wrong_observed_apex(self):
        current = scene([
            raw_object(
                1, "triangle", [0.400, 0.270, 0.071], label="triangle",
                size=(0.048, 0.024, 0.024), yaw_deg=0.0,
            ),
            raw_object(
                2, "roof", [0.400, 0.270, 0.052], label="rectangle",
                size=(0.060, 0.030, 0.014), yaw_deg=0.0,
            ),
        ])
        edge = SimpleNamespace(
            acted_object_track_id="triangle",
            physical_parameters={"target_object_pose": {
                "position_m": [0.400, 0.270, 0.071],
                "designated_right_angle_edge_world": [0.0, 0.0, 1.0],
                "beam_axis_world": [1.0, 0.0, 0.0],
            }},
        )
        valid, checks = _released_triangle_destination_check(
            edge, current, {"roof": "roof"}, config(), {
                "role_object_visible": True,
                "roof_visible": True,
                "apex_up": False,
                "base_edge_down": False,
                "face_state_valid": False,
                "base_contact_valid": True,
            },
        )
        self.assertFalse(valid)
        self.assertFalse(checks["fresh_observation_confirms_apex_up"])
        self.assertTrue(checks["executed_rigid_target_face_up"])

    def test_post_place_verification_uses_retained_roof_when_triangle_occludes_it(self):
        roof = scene([raw_object(
            1, "roof", [0.400, 0.270, 0.018], label="rectangle",
            size=(0.060, 0.030, 0.064),
        )]).current_objects[0]
        current = scene([raw_object(
            2, "triangle", [0.397, 0.270, 0.060], label="triangle",
            size=(0.042, 0.025, 0.020), yaw_deg=0.0,
        )])
        current = replace(current, collision_obstacles=(roof,))
        edge = SimpleNamespace(
            acted_object_track_id="triangle",
            physical_parameters={"target_object_pose": {
                "position_m": [0.400, 0.270, 0.060],
                "designated_right_angle_edge_world": [0.0, 0.0, 1.0],
                "beam_axis_world": [1.0, 0.0, 0.0],
            }},
        )
        original = {
            "role_object_visible": True, "roof_visible": True,
            "apex_up": True, "base_edge_down": True, "face_state_valid": True,
            "long_edge_matches_roof_axis": True, "base_contact_valid": True,
            "center_of_mass_supported": True, "support_margin_valid": True,
            "roof_center_offset_valid": True,
        }
        valid, checks = _released_triangle_destination_check(
            edge, current, {"roof": "roof"}, config(), original,
        )
        self.assertTrue(valid)
        self.assertTrue(checks["executed_target_pose_available"])
        self.assertTrue(checks["fresh_observation_confirms_apex_up"])

    def test_093822_fresh_profile_and_retained_roof_pass_triangle_role_checks(self):
        roof = scene([raw_object(
            1, "track_red_01", [0.413858, 0.266017, 0.017613],
            label="rectangle", size=(0.057084, 0.022316, 0.064954),
            long_axis_base=[0.9985, -0.054, 0.0],
        )]).current_objects[0]
        triangle = raw_object(
            2, "track_green_01", [0.396500, 0.269623, 0.059796],
            label="triangle", size=(0.042274, 0.025296, 0.020238),
            yaw_deg=-3.732,
            long_axis_base=[0.997730, -0.067249, -0.003602],
            designated_right_angle_edge_base=[0.0, 0.0, 1.0],
            fresh_3d_apex_up_confirmed=True,
            triangle_vertical_profile={
                "available": True, "lower_to_upper_extent_ratio": 1.752,
                "apex_up_confirmed": True, "apex_down_confirmed": False,
            },
        )
        current = scene([triangle])
        current = replace(current, collision_obstacles=(roof,))
        valid, checks = role_observation_checks(
            current,
            {"roof": "track_red_01", "triangle_top": "track_green_01"},
            "triangle_top",
            config(),
        )
        self.assertTrue(valid)
        self.assertTrue(checks["apex_up"])
        self.assertTrue(checks["base_edge_down"])
        self.assertTrue(checks["center_of_mass_supported"])
        self.assertAlmostEqual(checks["long_edge_alignment_error_deg"], 0.764, delta=0.2)

        previous_roof = raw_object(
            1, "track_red_01", [0.400, 0.270, 0.018],
            label="rectangle", size=(0.060, 0.030, 0.064),
            top_z_base_m=0.050,
            local_support_surface={"support_z_base_m": -0.014},
            long_axis_base=[1.0, 0.0, 0.0],
        )
        previous = build_house_task_state(scene([previous_roof]), config())
        self.assertTrue(previous.role_completion["roof"])
        continued_scene = replace(
            current,
            recent_action_results=({
                "task_role": "triangle_top",
                "action_type": "place_house_role",
                "acted_object_track_id": "track_green_01",
                "success": False,
                "continuation_preserve_verified_roof": True,
            },),
        )
        continued = build_house_task_state(
            continued_scene, config(), previous=previous,
        )
        self.assertTrue(all(continued.role_completion.values()))
        self.assertEqual(continued.role_status["roof"], "COMPLETED_OCCLUDED")
        self.assertEqual(continued.role_status["triangle_top"], "COMPLETED_VISIBLE")

    def test_all_policy_kinds_send_one_32768_context(self):
        self.assertEqual(DEFAULT_VLM_NUM_CTX, 32768)
        self.assertNotIn("num_ctx", json.dumps(POLICY_GENERATION_CONFIG))
        for kind in POLICY_GENERATION_CONFIG:
            with self.subTest(kind=kind), patch(
                "robot_scene_pipeline.ollama_policy_client.requests.post",
                return_value=_Response(),
            ) as post:
                result = call_policy(_args(), kind, [{"role": "user", "content": "{}"}], SCHEMA)
                self.assertTrue(result.schema_valid)
                self.assertEqual(post.call_args.kwargs["json"]["options"]["num_ctx"], 32768)

    def test_stale_context_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires vlm num_ctx=32768"):
            call_policy(_args(vlm_num_ctx=16384), "edge_selection", [], SCHEMA)

    def test_three_shapes_use_full_se3_family(self):
        self.assertEqual(
            FULL_3D_ORIENTATION_SHAPES,
            frozenset({"rectangle", "concave_rectangle", "triangle"}),
        )

    def test_non_world_x_beam_axis_drives_all_target_quaternions(self):
        beam = (math.sqrt(0.5), math.sqrt(0.5), 0.0)
        for shape in FULL_3D_ORIENTATION_SHAPES:
            obj = _special_object(shape)
            targets = solve_target_object_orientations(obj, [0.4, 0.27, 0.12], beam, config())
            self.assertEqual(len(targets), 1)
            for target in targets:
                long_axis = transform_vector(target["orientation_xyzw"], (1.0, 0.0, 0.0))
                self.assertLess(axis_alignment_error_deg(long_axis, beam), 1e-6)

    def test_rectangle_broad_face_up_and_concave_opening_down(self):
        rectangle = solve_target_object_orientations(
            _special_object("rectangle"), [0.4, 0.27, 0.12], [0.0, 1.0, 0.0], config(),
        )
        concave = solve_target_object_orientations(
            _special_object("concave_rectangle"), [0.4, 0.27, 0.12], [0.0, 1.0, 0.0], config(),
        )
        self.assertTrue(all(item["broad_face_normal_world"][2] > 0.45 for item in rectangle))
        self.assertTrue(all(item["groove_opening_normal_world"][2] < -0.45 for item in concave))

    def test_face_selection_is_separate_from_structural_tilt(self):
        correct = _special_object("rectangle")
        ready, checks = semantic_face_ready_for_structural_tilt(correct, config())
        self.assertTrue(ready)
        self.assertFalse(checks["tabletop_face_reorientation_required"])

        wrong_raw = _special_raw("rectangle")
        wrong_raw["broad_face_normal_base"] = [0.0, 0.0, -1.0]
        wrong = scene([wrong_raw]).current_objects[0]
        ready, checks = semantic_face_ready_for_structural_tilt(wrong, config())
        self.assertFalse(ready)
        self.assertTrue(checks["tabletop_face_reorientation_required"])

    def test_triangle_designated_edge_is_up(self):
        targets = solve_target_object_orientations(
            _special_object("triangle"), [0.4, 0.27, 0.12], [1.0, 0.2, 0.0], config(),
        )
        self.assertTrue(all(item["designated_right_angle_edge_world"][2] > 0.45 for item in targets))

    def test_live_table_triangle_cannot_skip_incremental_tabletop_reorientation(self):
        raw = _special_raw("triangle")
        raw.update({
            "orientation_xyzw": [0.00934, -0.05273, -0.63215, 0.77299],
            "long_axis_base": [0.1952, -0.9783, 0.0697],
            "designated_right_angle_edge_base": [0.99556, -0.00715, 0.09383],
            "visible_face_normal_base": [-0.0933, 0.0522, 0.9943],
        })
        obj = scene([raw]).current_objects[0]
        targets = solve_target_object_orientations(
            obj, [0.4, 0.27, 0.065], [1.0, 0.0, 0.0], config(),
            allow_staged_triangle_face_change=True,
        )
        self.assertEqual(targets, ())

    def test_triangle_tabletop_repair_is_exactly_one_45_degree_step_toward_apex_up(self):
        raw = _special_raw("triangle")
        raw.update({
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "long_axis_base": [1.0, 0.0, 0.0],
            "designated_right_angle_edge_base": [0.0, 1.0, 0.0],
            "visible_face_normal_base": [0.0, 0.0, 1.0],
        })
        obj = scene([raw]).current_objects[0]
        target = build_triangle_tabletop_step_pose(obj, obj.center_xyz_m[:2], config())
        self.assertIsNotNone(target)
        self.assertAlmostEqual(
            rotation_error_deg(raw["orientation_xyzw"], target["orientation_xyzw"]),
            45.0,
        )
        self.assertGreater(target["designated_right_angle_edge_world"][2], 0.70)
        self.assertEqual(target["position_m"][:2], list(obj.center_xyz_m[:2]))
        self.assertEqual(
            target["target_orientation_source"],
            "incremental_tabletop_apex_up_step",
        )

    def test_fresh_supported_metric_triangle_already_apex_up_skips_another_step(self):
        raw = _special_raw("triangle")
        raw.update({
            "geometry_center_m": [0.4064, 0.1282, -0.0043],
            "dimensions_m": [0.0411, 0.0238, 0.0186],
            "orientation_confidence": 0.5873,
            "metric_triangle_geometry_confirmed": True,
            "long_axis_base": [0.125645, -0.983276, 0.131843],
            "visible_face_normal_base": [-0.200786, 0.104942, 0.973998],
            "designated_right_angle_edge_base": [0.99, 0.0, 0.10],
            "local_support_surface": {"support_z_base_m": -0.013599},
        })
        current = scene([raw])
        updated = _with_verified_triangle_tabletop_pose(
            current, current.current_objects[0], config(),
        )
        self.assertTrue(updated.source["tabletop_apex_up_geometry_verified"])
        self.assertEqual(updated.source["designated_right_angle_edge_base"], [0.0, 0.0, 1.0])
        self.assertEqual(current.recent_action_results, ())

    def test_fresh_metric_triangle_without_table_contact_is_not_apex_up_verified(self):
        raw = _special_raw("triangle")
        raw.update({
            "metric_triangle_geometry_confirmed": True,
            "long_axis_base": [1.0, 0.0, 0.1],
            "visible_face_normal_base": [0.0, 0.0, 1.0],
            "designated_right_angle_edge_base": [1.0, 0.0, 0.0],
        })
        current = scene([raw])
        updated = _with_verified_triangle_tabletop_pose(
            current, current.current_objects[0], config(),
        )
        self.assertIs(updated, current.current_objects[0])

    def test_triangle_grasp_rejects_apex_clamp_and_selects_side_contact(self):
        current = scene([raw_object(
            1, "triangle", [0.3975, 0.1205, -0.0048], label="triangle",
            size=(0.0416, 0.0224, 0.0181), yaw_deg=89.75,
        )])
        scan = scan_grasp_yaws(
            current.current_objects[0], current.current_objects, config(),
        )
        apex_clamp = next(item for item in scan.samples if item["yaw_deg"] == 0.0)
        self.assertTrue(apex_clamp["opening_ok"])
        self.assertFalse(apex_clamp["stable_opposed_contact_ok"])
        self.assertFalse(apex_clamp["safe"])
        self.assertTrue(scan.safe_intervals)
        self.assertAlmostEqual(scan.safe_intervals[0].selected_yaw_deg, -90.0)
        self.assertTrue(scan.safe_intervals[0].selected_check["stable_opposed_contact_ok"])

    def test_split_sloped_green_mask_is_merged_before_triangle_review(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV fragment merge dependencies are unavailable")
        full = np.zeros((100, 130), dtype=np.uint8)
        cv2.fillConvexPoly(full, np.array([[20, 72], [36, 28], [108, 28], [108, 72]]), 1)
        left = full.astype(bool)
        left[:, 66:] = False
        right = full.astype(bool)
        right[:, :66] = False
        x_values = np.linspace(-0.024, 0.024, 12)
        y_values = np.linspace(-0.012, 0.012, 7)
        z_values = np.linspace(-0.012, 0.012, 7)
        cloud = np.array([(x, y, z) for x in x_values for y in y_values for z in z_values])

        def fragment(candidate_id, mask, points):
            ys, xs = np.nonzero(mask)
            return {
                "id": candidate_id, "label": "square green", "visual_color": "green",
                "confidence": 0.7, "primary_detector_passed": True,
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                "_mask_bool": mask, "pointcloud_geometry_valid": True,
                "dimensions_m": [0.024, 0.023, 0.022],
                "top_z_base_m": 0.008,
                "local_support_surface": {"support_z_base_m": -0.014},
                "_object_points_base": points,
            }

        merged, log = merge_fragmented_green_triangle_detections([
            fragment(1, left, cloud[cloud[:, 0] <= 0.0]),
            fragment(2, right, cloud[cloud[:, 0] >= 0.0]),
        ])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["fragment_merge_candidate_ids"], [1, 2])
        self.assertTrue(log[0]["evidence"]["shape_classification_deferred_to_union_geometry"])

    def test_two_adjacent_green_square_masks_are_not_merged_without_sloped_union(self):
        import numpy as np
        first_mask = np.zeros((80, 100), dtype=bool)
        second_mask = np.zeros((80, 100), dtype=bool)
        first_mask[20:60, 10:50] = True
        second_mask[20:60, 50:90] = True
        cloud = np.array([
            [x, y, z]
            for x in np.linspace(-0.024, 0.024, 10)
            for y in np.linspace(-0.012, 0.012, 6)
            for z in np.linspace(-0.012, 0.012, 6)
        ])
        objects = []
        for candidate_id, mask, points in (
            (1, first_mask, cloud[cloud[:, 0] <= 0.0]),
            (2, second_mask, cloud[cloud[:, 0] >= 0.0]),
        ):
            ys, xs = np.nonzero(mask)
            objects.append({
                "id": candidate_id, "label": "square green", "visual_color": "green",
                "confidence": 0.7, "pointcloud_geometry_valid": True,
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                "_mask_bool": mask, "dimensions_m": [0.024, 0.023, 0.022],
                "top_z_base_m": 0.008,
                "local_support_surface": {"support_z_base_m": -0.014},
                "_object_points_base": points,
            })
        merged, log = merge_fragmented_green_triangle_detections(objects)
        self.assertEqual(len(merged), 2)
        self.assertEqual(log, [])

    def test_current_frame_verified_roof_axis_allows_direct_triangle_place(self):
        roof = raw_object(
            1, "roof", [0.4080, 0.2722, 0.0186], label="rectangle",
            size=(0.0612, 0.0294, 0.0634), yaw_deg=-0.68,
            top_z_base_m=0.0503,
            local_support_surface={"support_z_base_m": -0.0131},
            long_axis_base=[0.9993, -0.0199, 0.0316],
        )
        triangle = raw_object(
            2, "triangle", [0.3030, 0.0857, -0.0070], label="triangle",
            size=(0.0345, 0.0237, 0.0163), yaw_deg=89.14,
            semantic_shape="triangle", semantic_shape_confidence=0.618,
            semantic_shape_uncertain=True, metric_triangle_geometry_confirmed=True,
            orientation_confidence=0.5385,
            orientation_xyzw=[-0.0444, 0.0444, 0.6927, 0.7185],
            long_axis_base=[0.0365, 0.9914, -0.1253],
            visible_face_normal_base=[0.0023, 0.1253, 0.9921],
            designated_right_angle_edge_base=[1.0, 0.0, 0.0],
            local_support_surface={"support_z_base_m": -0.01512},
            evidence_scene_revision=1,
        )
        current = scene([roof, triangle])
        state = build_house_task_state(current, config())
        self.assertTrue(state.role_completion["roof"])
        axis = beam_axis_from_supports(
            current, state.role_bindings, config(),
            verified_roof_track_id=state.role_bindings["roof"],
        )
        self.assertIsNotNone(axis)
        interval = scan_grasp_yaws(
            current.object_by_track("triangle"), current.current_objects, config(),
        ).safe_intervals[0]
        targets = house_placement_target(
            current, state, current.object_by_track("triangle"),
            "triangle_top", config(), interval=interval,
        )
        self.assertTrue(targets)
        self.assertTrue(all(
            target.action_type.value == "place_house_role" for target in targets
        ))
        self.assertTrue(all(
            target.additional_physical_parameters["measured_beam_axis_source"]
            == "current_frame_verified_60_to_67mm_roof_long_axis"
            for target in targets
        ))

        protected_scene = replace(current, protected_tracks=("roof",))
        generated = generate_physical_edges(
            protected_scene, (), "build_house", config(),
            lambda _obj, _interval: None,
            lambda _obj: None,
            target_specs=(TargetSpec(
                "triangle__triangle_top", "triangle", "triangle_top",
            ),),
            variant_placement_provider=lambda obj, candidate_interval, role: (
                house_placement_target(
                    protected_scene, state, obj, str(role), config(),
                    interval=candidate_interval,
                )
            ),
        )
        edges = generated.edges_by_target["triangle__triangle_top"]
        self.assertTrue(edges)
        self.assertTrue(all(edge.precheck_results["passed"] for edge in edges))
        self.assertTrue(all(
            not edge.precheck_results["place_blocking_track_ids"] for edge in edges
        ))

    def test_fixed_tcp_waypoints_and_tool0_offset_compensation(self):
        current = scene([_special_raw("rectangle")])
        obj = current.current_objects[0]
        target = solve_target_object_orientations(
            obj, [0.4, 0.27, 0.12], [0.0, 1.0, 0.0], config(),
        )[0]
        plan = build_airborne_orientation_plan(current, obj, 0.0, target, config())
        self.assertIsNotNone(plan)
        fixed = plan["fixed_grasp_tcp_position_m"]
        self.assertTrue(all(math.dist(item["position_m"], fixed) <= 0.001 for item in plan["waypoints"]))
        tool_positions = [tuple(item["tool0_pose"]["position_m"]) for item in plan["waypoints"]]
        self.assertGreater(len(set(tool_positions)), 1)
        self.assertEqual(plan["adjustment_location"], "safe_airborne_zone")
        self.assertEqual(
            plan["adjustment_kind"],
            "minimum_required_target_orientation_alignment",
        )
        self.assertFalse(plan["semantic_face_reorientation_required"])

    def test_actual_rotation_is_current_to_target_not_fixed_45(self):
        current = scene([_special_raw("rectangle")])
        obj = current.current_objects[0]
        target = solve_target_object_orientations(obj, [0.4, 0.27, 0.12], [0.0, 1.0, 0.0], config())[0]
        plan = build_airborne_orientation_plan(current, obj, 0.0, target, config())
        expected = rotation_error_deg(plan["start_orientation_xyzw"], plan["target_orientation_xyzw"])
        self.assertAlmostEqual(plan["actual_rotation_angle_deg"], expected)
        self.assertNotAlmostEqual(plan["actual_rotation_angle_deg"], 45.0)

    def test_correct_face_and_axis_produce_no_flip(self):
        current = scene([_special_raw("rectangle")])
        obj = current.current_objects[0]
        target = solve_target_object_orientations(
            obj, [0.4, 0.27, 0.12], [1.0, 0.0, 0.0], config(),
        )[0]
        self.assertFalse(target["forced_nominal_flip"])
        self.assertAlmostEqual(target["target_tilt_deg"], 0.0)
        plan = build_airborne_orientation_plan(current, obj, 0.0, target, config())
        self.assertAlmostEqual(plan["actual_rotation_angle_deg"], 0.0)

    def test_airborne_rotation_is_rejected_when_every_staging_zone_is_blocked(self):
        raw = _special_raw("rectangle")
        blockers = [
            raw_object(index + 10, f"blocker_{index}", [x, y, 0.12], size=(0.08, 0.08, 0.08))
            for index, (x, y) in enumerate(((0.4, 0.08), (0.4, 0.28), (0.3, 0.08),
                                            (0.3, 0.28), (0.4, 0.18), (0.3, 0.18)))
        ]
        current = scene([raw, *blockers])
        obj = current.object_by_track("rectangle")
        target = solve_target_object_orientations(
            obj, [0.4, 0.27, 0.12], [0.0, 1.0, 0.0], config(),
        )[0]
        self.assertIsNone(build_airborne_orientation_plan(current, obj, 0.0, target, config()))

    def test_missing_or_conflicting_semantics_blocks_direct_special_pose(self):
        raw = _special_raw("rectangle")
        raw["semantic_shape_uncertain"] = True
        obj = scene([raw]).current_objects[0]
        self.assertFalse(special_shape_evidence_ready(obj, config()))

    def test_qwen_cannot_create_candidate_id(self):
        value = {"scene_revision": 7, "objects": [{"candidate_id": "not_supplied"}]}
        result = _validate_special_response(value, ["1", "2"], 7)
        self.assertEqual(result["objects"], [])
        self.assertIn("candidate_id_not_allowed", result["rejected_objects"][0]["errors"])

    def test_qwen_duplicate_candidate_id_is_rejected(self):
        value = {"scene_revision": 7, "objects": [
            {"candidate_id": "1"}, {"candidate_id": "1"},
        ]}
        result = _validate_special_response(value, ["1", "2"], 7)
        self.assertEqual(result["objects"], [])
        self.assertTrue(all(
            "duplicate_candidate_id" in item["errors"]
            for item in result["rejected_objects"]
        ))

    def test_one_out_of_range_confidence_does_not_discard_valid_peer(self):
        valid = _special_decision("1")
        invalid = {**_special_decision("2"), "orientation_confidence": 1.0456}
        result = _validate_special_response(
            {"scene_revision": 7, "objects": [valid, invalid]}, ["1", "2"], 7,
        )
        self.assertEqual([item["candidate_id"] for item in result["objects"]], ["1"])
        self.assertEqual(result["rejected_objects"][0]["candidate_id"], "2")
        self.assertIn(
            "orientation_confidence_outside_0_1",
            result["rejected_objects"][0]["errors"],
        )

    def test_camera_left_is_transformed_and_projected_into_visible_face(self):
        direction = _camera_direction_in_base(
            "left",
            ((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
            [0.0, 0.0, 1.0],
        )
        self.assertEqual(direction, (0.0, 1.0, 0.0))

    def test_172451_orientation_space_reports_concrete_blockers(self):
        roof = raw_object(
            1, "roof", [0.38495, 0.07539, 0.00086], label="concave rectangle",
            size=(0.05614, 0.02581, 0.03028),
        )
        current = scene([
            roof,
            raw_object(2, "green", [0.29805, 0.07366, -0.00168], size=(0.02596, 0.01796, 0.02332)),
            raw_object(3, "red_02", [0.38861, 0.16892, -0.00634], label="rectangle", size=(0.05763, 0.02666, 0.01514)),
        ])
        self.assertEqual(
            orientation_airborne_blocking_tracks(current, current.object_by_track("roof"), config()),
            ("green", "red_02"),
        )

    def test_config_bounds_semantic_batches_to_six(self):
        special = config().section("special_shape_vlm")
        self.assertEqual(special["maximum_candidates_per_request"], 6)
        self.assertLessEqual(special["hard_maximum_candidates_per_request"], 8)

    def test_long_rectangle_exceeding_finger_axial_coverage_fails(self):
        current = scene([raw_object(
            1, "long", [0.5, 0.0, 0.02], label="rectangle", size=(0.14, 0.02, 0.02),
        )])
        check = evaluate_gripper_pose_clearance(current.current_objects[0], current.current_objects, 0.0, config())
        self.assertFalse(check["axial_coverage_ok"])
        self.assertGreater(check["left_axial_overhang_m"], config().section("grasp")["allowed_axial_overhang_m"])

    def test_edge_grasp_sorting_precedes_corner_grasps(self):
        current = scene([raw_object(
            1, "rectangle", [0.5, 0.0, 0.02], label="rectangle", size=(0.06, 0.02, 0.02),
        )])
        scan = scan_grasp_yaws(current.current_objects[0], current.current_objects, config())
        self.assertTrue(scan.safe_intervals)
        self.assertIn(scan.safe_intervals[0].selected_check["grasp_class"], {0, 1})


def _special_raw(shape):
    semantic_axis = {
        "rectangle": {"broad_face_normal_base": [0.0, 0.0, 1.0]},
        "concave_rectangle": {"groove_opening_normal_base": [0.0, 0.0, -1.0]},
        "triangle": {"designated_right_angle_edge_base": [0.0, 0.0, 1.0]},
    }[shape]
    return raw_object(
        1, shape, [0.52, -0.02, 0.025], label=shape, size=(0.07, 0.03, 0.02),
        semantic_shape=shape, semantic_shape_confidence=0.95,
        semantic_shape_uncertain=False, orientation_confidence=0.95,
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0], long_axis_base=[1.0, 0.0, 0.0],
        evidence_scene_revision=1, **semantic_axis,
    )


def _special_object(shape):
    return scene([_special_raw(shape)]).current_objects[0]


def _special_decision(candidate_id):
    return {
        "candidate_id": candidate_id,
        "shape_label": "rectangle",
        "shape_confidence": 0.9,
        "long_axis_image_deg": 0.0,
        "broad_face_visible": True,
        "broad_face_confidence": 0.9,
        "groove_visible": False,
        "groove_opening_direction_camera": "unknown",
        "triangle_right_angle_edge_direction_image": "unknown",
        "occlusion": "none",
        "orientation_confidence": 0.9,
        "evidence": [],
    }


def _args(**changes):
    values = dict(
        model="qwen3-vl:8b-instruct", ollama_url="http://local/api/chat",
        vlm_think_mode="off", vlm_num_ctx=0, vlm_num_predict=0, vlm_num_gpu=-1,
        vlm_read_timeout_sec=1, vlm_keep_alive="1h", vlm_max_backend_retries=1,
        vlm_max_budget_retries=1, vlm_finalizer_num_predict=768, output_dir=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class _Response:
    status_code = 200
    text = ""

    @staticmethod
    def json():
        return {"message": {"role": "assistant", "content": '{"ok": true}'},
                "done": True, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}


if __name__ == "__main__":
    unittest.main()
