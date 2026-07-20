from dataclasses import replace
import unittest
from unittest.mock import patch

from tools.workflows.stack_demo.clutter.edge_generation import (
    PlacementTarget,
    TargetSpec,
    generate_physical_edges,
)
from tools.workflows.stack_demo.clutter.grasp_edges import scan_grasp_yaws
from tools.workflows.stack_demo.common.action_edges import ActionType
from tools.workflows.stack_demo.house.completion import evaluate_house_completion
from tools.workflows.stack_demo.house.placement import (
    _role_pose,
    house_placement_target,
    staging_orientation_targets,
)
from tools.workflows.stack_demo.house.planner import _candidate_role_specs
from tools.workflows.stack_demo.house.orientation import (
    RoofOrientationState,
    TriangleOrientationState,
    roof_direct_place_valid,
    triangle_direct_place_valid,
)
from tools.workflows.stack_demo.house.roles import legal_incomplete_roles
from tools.workflows.stack_demo.house.state import build_house_task_state
from tools.workflows.stack_demo.house.structure import role_observation_checks

from tests.new_arch_fixtures import config, raw_object, scene


class NewHousePlannerTests(unittest.TestCase):
    def test_roof_target_uses_configured_house_center_without_negative_z_offset(self):
        left = raw_object(
            1, "left_upper", [0.383, 0.266, 0.010],
            size=[0.024, 0.024, 0.050], label="square yellow",
        )
        right = raw_object(
            2, "right_upper", [0.419, 0.264, 0.011],
            size=[0.025, 0.025, 0.050], label="square blue",
        )
        roof = raw_object(
            3, "roof", [0.45, 0.18, 0.0],
            size=[0.056, 0.027, 0.014], label="rectangle red",
        )
        current = scene([left, right, roof])
        state = replace(
            build_house_task_state(current, config()),
            role_bindings={
                "left_support_upper": "left_upper",
                "right_support_upper": "right_upper",
            },
        )
        pose = _role_pose(current, state, current.object_by_track("roof"), "roof", config())
        self.assertEqual(pose["position_m"][:2], [0.4, 0.27])
        self.assertAlmostEqual(pose["position_m"][2], 0.043, places=6)

    def test_triangle_target_uses_configured_center_not_noisy_roof_mask_center(self):
        roof = raw_object(
            1, "roof", [0.405, 0.268, 0.052],
            size=[0.060, 0.030, 0.014], label="rectangle red",
        )
        triangle = raw_object(
            2, "triangle", [0.416, 0.139, 0.0],
            size=[0.048, 0.024, 0.024], label="triangle green",
        )
        current = scene([roof, triangle])
        state = replace(
            build_house_task_state(current, config()),
            role_bindings={"roof": "roof"},
        )
        pose = _role_pose(
            current, state, current.object_by_track("triangle"), "triangle_top", config(),
        )
        self.assertEqual(pose["position_m"][:2], [0.4, 0.27])

    def test_successful_orientation_staging_is_not_repeated_without_evidence_gain(self):
        current = scene([raw_object(
            1, "roof", [0.40, 0.08, 0.0],
            label="rectangle red", size=[0.056, 0.027, 0.014],
        )])
        current = replace(current, recent_action_results=({
            "success": True,
            "action_type": "extract_to_staging",
            "acted_object_track_id": "roof",
            "task_role": "roof",
        },))
        interval = scan_grasp_yaws(
            current.object_by_track("roof"), current.current_objects, config(),
        ).safe_intervals[0]
        self.assertEqual(
            staging_orientation_targets(
                current, current.object_by_track("roof"), "roof", config(), interval=interval,
            ),
            (),
        )

    def test_failed_final_triangle_placement_allows_new_tabletop_reorientation(self):
        current = scene([raw_object(
            1, "triangle", [0.40, 0.27, 0.06],
            label="triangle", size=[0.048, 0.024, 0.024],
        )])
        current = replace(current, recent_action_results=(
            {
                "success": True,
                "action_type": "extract_to_staging",
                "acted_object_track_id": "triangle",
                "task_role": "triangle_top",
            },
            {
                "success": False,
                "action_type": "place_house_role",
                "acted_object_track_id": "triangle",
                "task_role": "triangle_top",
            },
        ))
        triangle = current.object_by_track("triangle")
        interval = scan_grasp_yaws(triangle, current.current_objects, config()).safe_intervals[0]
        targets = staging_orientation_targets(
            current, triangle, "triangle_top", config(), interval=interval,
        )
        self.assertTrue(targets)
        self.assertTrue(all(
            target.additional_physical_parameters["staging_purpose"]
            == "change_orientation_observation"
            for target in targets
        ))

    def test_shared_blocker_fallback_cannot_restaging_successful_roof(self):
        current = scene([raw_object(
            1, "roof", [0.30, 0.18, 0.0],
            label="rectangle red", size=[0.056, 0.027, 0.014],
        )])
        current = replace(current, recent_action_results=({
            "success": True,
            "action_type": "extract_to_staging",
            "acted_object_track_id": "roof",
            "task_role": "roof",
        },))
        generated = generate_physical_edges(
            current, (), "build_house", config(),
            lambda _obj, _interval: None,
            lambda obj: staging_orientation_targets(
                current, obj, "blocker", config(),
            ),
            target_specs=(TargetSpec("roof__roof", "roof", "roof"),),
            variant_placement_provider=lambda _obj, _interval, _role: None,
        )
        self.assertEqual(generated.edges_by_target["roof__roof"], ())

    def test_merged_two_cube_column_verifies_upper_contact(self):
        lower = raw_object(
            1, "lower", [0.430, 0.270, 0.0],
            size=[0.025, 0.025, 0.025], label="square blue",
        )
        merged = raw_object(
            2, "upper", [0.430, 0.270, 0.012],
            size=[0.023, 0.023, 0.049], label="square blue",
        )
        current = scene([lower, merged])
        valid, checks = role_observation_checks(
            current,
            {"right_support_lower": "lower", "right_support_upper": "upper"},
            "right_support_upper",
            config(),
        )
        self.assertTrue(valid)
        self.assertEqual(
            checks["vertical_contact_verification_mode"], "merged_two_support_column",
        )

    def test_fresh_merged_column_restores_both_support_roles(self):
        merged = raw_object(
            1, "column", [0.420, 0.270, 0.011],
            size=[0.023, 0.023, 0.049], label="square blue",
            local_support_surface={"support_z_base_m": -0.0135},
        )
        state = build_house_task_state(scene([merged]), config())
        self.assertTrue(state.role_completion["right_support_lower"])
        self.assertTrue(state.role_completion["right_support_upper"])
        self.assertEqual(state.role_bindings["right_support_upper"], "column")
        self.assertTrue(
            state.role_bindings["right_support_lower"].startswith("inferred_hidden_")
        )

    def test_prebound_merged_upper_still_restores_hidden_lower(self):
        merged = raw_object(
            1, "column", [0.420, 0.263, 0.0114],
            size=[0.0256, 0.0241, 0.0478], label="square blue",
            local_support_surface={"support_z_base_m": -0.0125},
        )
        current = scene([merged])
        seed = build_house_task_state(current, config())
        previous = replace(
            seed,
            role_bindings={"right_support_upper": "column"},
            role_completion=_completion(),
        )
        state = build_house_task_state(current, config(), previous=previous)
        self.assertTrue(state.role_completion["right_support_lower"])
        self.assertTrue(state.role_completion["right_support_upper"])
        self.assertEqual(state.role_status["right_support_lower"], "COMPLETED_OCCLUDED")
        self.assertEqual(state.role_status["right_support_upper"], "COMPLETED_VISIBLE")
        self.assertEqual(state.role_bindings["right_support_upper"], "column")
        self.assertIn(state.role_bindings["right_support_lower"], state.inferred_hidden_tracks)

    def test_perspective_stretched_164624_blue_column_is_still_complete(self):
        merged = raw_object(
            1, "track_blue_01", [0.41684, 0.26434, 0.01142],
            size=[0.03428, 0.02482, 0.04905], label="square blue",
            local_support_surface={"support_z_base_m": -0.01311},
        )
        current = scene([merged])
        seed = build_house_task_state(current, config())
        previous = replace(
            seed,
            role_bindings={"right_support_upper": "track_blue_01"},
            role_completion=_completion(),
        )
        state = build_house_task_state(current, config(), previous=previous)
        self.assertEqual(state.role_status["right_support_lower"], "COMPLETED_OCCLUDED")
        self.assertEqual(state.role_status["right_support_upper"], "COMPLETED_VISIBLE")
        self.assertIn("track_blue_01", state.protected_structure_tracks)

    def test_repairable_bound_track_remains_a_movable_candidate(self):
        current = scene([raw_object(
            1, "misplaced", [0.50, 0.05, 0.02], label="square blue",
        )])
        seed = build_house_task_state(current, config())
        previous = replace(
            seed,
            role_bindings={"left_support_lower": "misplaced"},
            role_completion=_completion(),
        )
        state = build_house_task_state(current, config(), previous=previous)
        self.assertEqual(state.role_status["left_support_lower"], "REPAIRABLE")
        self.assertNotIn("misplaced", state.protected_structure_tracks)
        self.assertIn("misplaced", state.role_candidate_tracks["left_support_lower"])

    def test_tall_rectangle_is_not_a_merged_support_column(self):
        lower = raw_object(
            1, "lower", [0.430, 0.270, 0.0],
            size=[0.025, 0.025, 0.025], label="square blue",
        )
        rectangle = raw_object(
            2, "roof", [0.430, 0.270, 0.012],
            size=[0.023, 0.023, 0.049], label="rectangle red",
        )
        current = scene([lower, rectangle])
        valid, checks = role_observation_checks(
            current,
            {"right_support_lower": "lower", "right_support_upper": "roof"},
            "right_support_upper",
            config(),
        )
        self.assertFalse(valid)
        self.assertEqual(
            checks["vertical_contact_verification_mode"], "separate_support_surfaces",
        )

    def test_25_left_upper_requires_left_lower(self):
        completion = _completion()
        self.assertNotIn("left_support_upper", legal_incomplete_roles(completion))
        completion["left_support_lower"] = True
        self.assertIn("left_support_upper", legal_incomplete_roles(completion))

    def test_26_right_upper_requires_right_lower(self):
        completion = _completion()
        self.assertNotIn("right_support_upper", legal_incomplete_roles(completion))
        completion["right_support_lower"] = True
        self.assertIn("right_support_upper", legal_incomplete_roles(completion))

    def test_27_roof_requires_both_upper_supports(self):
        completion = _completion()
        completion.update(left_support_lower=True, right_support_lower=True, left_support_upper=True)
        self.assertNotIn("roof", legal_incomplete_roles(completion))
        completion["right_support_upper"] = True
        self.assertIn("roof", legal_incomplete_roles(completion))

    def test_28_triangle_requires_roof(self):
        completion = _completion()
        self.assertNotIn("triangle_top", legal_incomplete_roles(completion))
        completion["roof"] = True
        self.assertIn("triangle_top", legal_incomplete_roles(completion))

    def test_29_wrong_roof_groove_face_cannot_place(self):
        roof = _valid_roof(groove_face_state="opening_up", face_up=False)
        valid, checks = roof_direct_place_valid(roof, 0.0, config())
        self.assertFalse(valid)
        self.assertFalse(checks["groove_face_correct"])

    def test_30_wrong_roof_long_axis_cannot_place(self):
        roof = _valid_roof(long_axis_yaw_deg=90.0)
        valid, checks = roof_direct_place_valid(roof, 0.0, config())
        self.assertFalse(valid)
        self.assertFalse(checks["long_axis_matches_support_span"])

    def test_31_insufficient_roof_support_coverage_is_rejected(self):
        roof = _valid_roof(left_support_coverage_m=0.001)
        valid, checks = roof_direct_place_valid(roof, 0.0, config())
        self.assertFalse(valid)
        self.assertFalse(checks["covers_left_support"])

    def test_32_triangle_apex_direction_must_be_up(self):
        triangle = _valid_triangle(apex_direction="down")
        valid, checks = triangle_direct_place_valid(triangle, config())
        self.assertFalse(valid)
        self.assertFalse(checks["apex_up"])

    def test_33_triangle_base_contact_must_be_valid(self):
        triangle = _valid_triangle(base_contact=False)
        valid, checks = triangle_direct_place_valid(triangle, config())
        self.assertFalse(valid)
        self.assertFalse(checks["base_contact_valid"])

    def test_34_completed_house_tracks_become_protected(self):
        current = scene(_completed_house_objects(), expected=[f"t{index}" for index in range(1, 7)])
        state = build_house_task_state(current, config())
        self.assertEqual(set(state.protected_structure_tracks), {f"t{index}" for index in range(1, 7)})
        self.assertTrue(evaluate_house_completion(current, state)["task_complete"])

    def test_completion_height_uses_merged_columns_and_nominal_roof_thickness(self):
        current = scene([
            raw_object(3, "t3", [0.382, 0.270, 0.012], size=(0.024, 0.024, 0.050)),
            raw_object(4, "t4", [0.425, 0.270, 0.012], size=(0.024, 0.024, 0.050)),
            raw_object(5, "t5", [0.400, 0.270, 0.019], label="rectangle",
                       size=(0.062, 0.030, 0.063)),
            raw_object(6, "t6", [0.407, 0.268, 0.063], label="triangle",
                       size=(0.035, 0.018, 0.025)),
        ])
        base = build_house_task_state(
            scene(_completed_house_objects(), expected=[f"t{index}" for index in range(1, 7)]),
            config(),
        )
        bindings = {
            "left_support_lower": "hidden_left", "left_support_upper": "t3",
            "right_support_lower": "hidden_right", "right_support_upper": "t4",
            "roof": "t5", "triangle_top": "t6",
        }
        support_edges = (
            ("hidden_left", "t3"), ("hidden_right", "t4"),
            ("t3", "t5"), ("t4", "t5"), ("t5", "t6"),
        )
        state = replace(
            base,
            scene_revision=current.scene_revision,
            role_bindings=bindings,
            role_completion={role: True for role in bindings},
            unresolved_missing_tracks=(),
            inferred_hidden_tracks=("hidden_left", "hidden_right"),
            support_relations=tuple(
                {"support": lower, "supported": upper, "verified": True}
                for lower, upper in support_edges
            ),
        )
        result = evaluate_house_completion(current, state, config())
        self.assertTrue(result["structure_total_height_valid"])
        self.assertIsNotNone(result["minimum_required_total_height_m"])
        self.assertLess(result["minimum_required_total_height_m"], 0.080)
        self.assertTrue(result["task_complete"])

    def test_all_legal_role_track_pairs_are_exposed_without_order_binding(self):
        current = scene([
            raw_object(1, "a", [0.40, 0.00, 0.02], label="square red"),
            raw_object(2, "b", [0.50, 0.00, 0.02], label="square blue"),
        ])
        state = build_house_task_state(current, config())
        specs = _candidate_role_specs(
            state, ("left_support_lower", "right_support_lower"),
        )
        self.assertEqual(
            {(item.track_id, item.task_role) for item in specs},
            {
                ("a", "left_support_lower"), ("a", "right_support_lower"),
                ("b", "left_support_lower"), ("b", "right_support_lower"),
            },
        )

    def test_final_house_place_requires_edge_aligned_grasp(self):
        current = scene([
            raw_object(1, "support", [0.50, 0.0, 0.02], yaw_deg=23.0),
        ])
        state = build_house_task_state(current, config())
        obj = current.current_objects[0]
        interval = scan_grasp_yaws(obj, current.current_objects, config()).safe_intervals[0]
        targets = house_placement_target(
            current, state, obj, "left_support_lower", config(), interval=interval,
        )
        self.assertIsInstance(targets, tuple)
        self.assertEqual(
            {target.additional_physical_parameters["required_grasp_yaw_deg"] for target in targets},
            {23.0, -67.0},
        )
        for target in targets:
            self.assertEqual(target.action_type, ActionType.PLACE_HOUSE_ROLE)
            self.assertAlmostEqual(
                target.additional_physical_parameters["edge_alignment_error_deg"], 0.0,
            )
            self.assertAlmostEqual(
                target.place_pose["position_m"][2],
                config().section("house")["origin_center_base_m"][2] - 0.005,
            )

    def test_blocked_edge_grasp_stages_for_fresh_edge_aligned_regrasp(self):
        current = scene([
            raw_object(1, "support", [0.50, 0.0, 0.02], yaw_deg=23.0),
            raw_object(2, "occupied", [0.40, 0.08, 0.02]),
        ])
        state = build_house_task_state(current, config())
        obj = current.current_objects[0]
        interval = scan_grasp_yaws(obj, current.current_objects, config()).safe_intervals[0]
        with patch(
            "tools.workflows.stack_demo.house.placement._edge_aligned_grasps",
            return_value=(),
        ):
            targets = house_placement_target(
                current, state, obj, "left_support_lower", config(), interval=interval,
            )
        self.assertIsInstance(targets, tuple)
        self.assertGreater(len(targets), 1)
        self.assertTrue(all(
            target.action_type == ActionType.EXTRACT_TO_STAGING
            and target.target_region_id == "house_edge_alignment_staging"
            for target in targets
        ))
        self.assertNotIn(
            [0.40, 0.08],
            [target.place_pose["position_m"][:2] for target in targets],
        )
        self.assertTrue(all(
            target.precheck_results["staging_open_gripper_descent_safe"]
            and target.additional_physical_parameters["camera_reobservable"]
            for target in targets
        ))

    def test_second_layer_keeps_safe_zero_degree_direct_option(self):
        current = scene([raw_object(1, "upper", [0.50, 0.0, 0.02])])
        obj = current.object_by_track("upper")

        grasp_order = [0.0, -90.0]

        def placement(_obj, _interval, _role):
            common = dict(
                action_type=ActionType.PLACE_HOUSE_ROLE,
                target_region_id=None,
                task_role="left_support_upper",
                place_pose={
                    "frame_id": "base_link",
                    "position_m": [0.3775, 0.27, 0.055],
                    "yaw_deg": 0.0,
                },
                task_progress_gain=1.0,
                expected_effects=("complete_house_role:left_support_upper",),
                precheck_results={
                    "transport_safe": True, "place_descent_safe": True,
                    "release_safe": True, "return_safe": True,
                    "protected_safe": True,
                },
            )
            return tuple(PlacementTarget(
                **common,
                additional_physical_parameters={"required_grasp_yaw_deg": yaw},
            ) for yaw in grasp_order)

        def path_check(_scene, _obj, _pose, physical, _config):
            safe = abs(float(physical["release_gripper_yaw_deg"])) < 1e-6
            return {
                "transport_safe": True, "place_descent_safe": safe,
                "release_safe": safe, "return_safe": safe,
                "place_blocking_track_ids": ([] if safe else ["left_neighbor"]),
            }

        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.placement_path_checks",
            side_effect=path_check,
        ) as mocked_path_check:
            generated = generate_physical_edges(
                current, (), "build_house", config(),
                lambda _obj, _interval: None,
                lambda _obj: self.fail("safe direct alternative must avoid staging"),
                target_specs=(TargetSpec(
                    "upper__left_support_upper", "upper", "left_support_upper",
                ),),
                variant_placement_provider=placement,
            )
        edges = generated.edges_by_target["upper__left_support_upper"]
        self.assertTrue(edges)
        self.assertEqual(
            {edge.physical_parameters["release_gripper_yaw_deg"] for edge in edges},
            {0.0},
        )
        self.assertTrue(all(
            edge.action_type == ActionType.PLACE_HOUSE_ROLE for edge in edges
        ))
        self.assertEqual(mocked_path_check.call_count, 2)

        grasp_order[:] = [-90.0, 0.0]
        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.placement_path_checks",
            side_effect=path_check,
        ) as mocked_path_check:
            generated = generate_physical_edges(
                current, (), "build_house", config(),
                lambda _obj, _interval: None,
                lambda _obj: self.fail("orthogonal direct alternative must avoid staging"),
                target_specs=(TargetSpec(
                    "upper__left_support_upper", "upper", "left_support_upper",
                ),),
                variant_placement_provider=placement,
            )
        edges = generated.edges_by_target["upper__left_support_upper"]
        self.assertEqual(
            {edge.physical_parameters["release_gripper_yaw_deg"] for edge in edges},
            {0.0},
        )
        self.assertEqual(mocked_path_check.call_count, 2)

    def test_unused_roof_blocker_stages_far_from_house(self):
        current = scene([raw_object(
            1, "roof", [0.48, 0.18, 0.02],
            label="concave rectangle red", size=(0.06, 0.03, 0.02),
        )])
        roof = current.object_by_track("roof")
        targets = staging_orientation_targets(
            current, roof, "blocker", config(),
        )
        self.assertTrue(targets)
        origin = config().section("house")["origin_center_base_m"]
        minimum = config().section("house")["roof_blocker_staging_min_house_distance_m"]
        self.assertTrue(all(
            target.target_region_id == "house_roof_far_staging"
            and ((target.place_pose["position_m"][0] - origin[0]) ** 2
                 + (target.place_pose["position_m"][1] - origin[1]) ** 2) ** 0.5 >= minimum
            for target in targets
        ))

    def test_all_occupied_staging_points_produce_no_unsafe_target(self):
        candidates = config().section("house")["orientation_staging_candidates_base_m"]
        objects = [raw_object(1, "support", [0.50, 0.0, 0.02], yaw_deg=23.0)]
        objects.extend(
            raw_object(index + 2, f"occupied_{index}", [x, y, 0.02])
            for index, (x, y) in enumerate(candidates)
        )
        current = scene(objects)
        obj = current.object_by_track("support")
        interval = scan_grasp_yaws(obj, current.current_objects, config()).safe_intervals[0]
        self.assertEqual(
            staging_orientation_targets(
                current, obj, "left_support_lower", config(), interval=interval,
            ),
            (),
        )

    def test_staging_candidates_are_ranked_and_keep_table_contact_height(self):
        current = scene([
            raw_object(1, "support", [0.50, 0.0, 0.035], size=(0.03, 0.03, 0.05)),
            raw_object(2, "near_first", [0.41, 0.14, 0.01], size=(0.02, 0.02, 0.02)),
        ])
        obj = current.object_by_track("support")
        interval = scan_grasp_yaws(obj, current.current_objects, config()).safe_intervals[0]
        targets = staging_orientation_targets(
            current, obj, "left_support_lower", config(), interval=interval,
        )
        self.assertGreater(len(targets), 1)
        clearances = [
            target.additional_physical_parameters["minimum_obstacle_clearance_m"]
            for target in targets
        ]
        self.assertEqual(clearances, sorted(clearances, reverse=True))
        expected_center_z = config().section("house")["table_surface_z_m"] + 0.025
        self.assertTrue(all(
            abs(target.place_pose["position_m"][2] - expected_center_z) < 1e-9
            for target in targets
        ))

    def test_moveit_failure_at_one_staging_side_keeps_other_options(self):
        current = scene([
            raw_object(1, "support", [0.50, 0.0, 0.02]),
        ])

        def placement(obj, interval):
            return staging_orientation_targets(
                current, obj, "left_support_lower", config(), interval=interval,
            )

        def checker(edge):
            x = edge.physical_parameters["place_pose"]["position_m"][0]
            return {"passed": x < 0.35, "moveit_plan_only": True}

        generated = generate_physical_edges(
            current,
            ("support",),
            "build_house",
            config(),
            placement,
            lambda obj: (),
            plan_checker=checker,
        )
        edges = generated.edges_by_target["support"]
        self.assertTrue(edges)
        self.assertTrue(all(
            edge.physical_parameters["place_pose"]["position_m"][0] == 0.30
            for edge in edges
        ))

    def test_known_final_place_blocker_suppresses_target_staging_loop(self):
        current = scene([
            raw_object(1, "target", [0.40, 0.20, 0.02]),
            raw_object(2, "blocker", [0.40, 0.27, 0.02]),
        ])

        def placement(obj, interval):
            return PlacementTarget(
                action_type=ActionType.PLACE_HOUSE_ROLE,
                target_region_id=None,
                task_role="left_support_lower",
                place_pose={
                    "frame_id": "base_link",
                    "position_m": [0.40, 0.27, 0.02],
                    "yaw_deg": 0.0,
                },
                task_progress_gain=1.0,
                expected_effects=("complete_house_role:left_support_lower",),
                precheck_results={
                    "transport_safe": True,
                    "place_descent_safe": True,
                    "release_safe": True,
                    "return_safe": True,
                    "protected_safe": True,
                },
            )

        generated = generate_physical_edges(
            current,
            ("target",),
            "build_house",
            config(),
            placement,
            lambda obj: staging_orientation_targets(
                current, obj, "left_support_lower", config(),
            ),
        )
        edges = generated.edges_by_target["target"]
        self.assertTrue(edges)
        self.assertFalse(any(
            edge.action_type == ActionType.EXTRACT_TO_STAGING
            and edge.acted_object_track_id == "target"
            for edge in edges
        ))
        self.assertTrue(any(
            edge.acted_object_track_id == "blocker"
            and edge.action_type in {
                ActionType.PICK_AWAY_BLOCKER, ActionType.NUDGE_BLOCKER,
            }
            for edge in edges
        ))

    def test_172451_wrong_groove_face_uses_tabletop_staging_not_airborne_flip(self):
        roof = raw_object(
            1, "roof", [0.38495, 0.07539, 0.00086],
            label="concave rectangle red", size=(0.05614, 0.02581, 0.03028),
            semantic_shape="concave_rectangle", semantic_shape_confidence=0.95,
            semantic_shape_uncertain=False, orientation_confidence=0.95,
            orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
            long_axis_base=[1.0, 0.0, 0.0],
            groove_opening_normal_base=[0.0, 0.0, 1.0],
            evidence_scene_revision=1,
        )
        objects = [
            roof,
            raw_object(
                2, "green", [0.29805, 0.07366, -0.00168],
                label="square green", size=(0.02596, 0.01796, 0.02332),
            ),
            raw_object(
                3, "red_02", [0.38861, 0.16892, -0.00634],
                label="rectangle red", size=(0.05763, 0.02666, 0.01514),
            ),
            raw_object(
                4, "left_upper", [0.3775, 0.270, 0.043],
                label="square yellow", size=(0.025, 0.025, 0.025),
            ),
            raw_object(
                5, "right_upper", [0.4225, 0.270, 0.043],
                label="square blue", size=(0.025, 0.025, 0.025),
            ),
        ]
        current = scene(objects, protected=("left_upper", "right_upper"))
        completion = _completion()
        completion.update(
            left_support_lower=True, right_support_lower=True,
            left_support_upper=True, right_support_upper=True,
        )
        state = replace(
            build_house_task_state(current, config()),
            role_bindings={
                "left_support_upper": "left_upper",
                "right_support_upper": "right_upper",
            },
            role_completion=completion,
        )

        generated = generate_physical_edges(
            current, (), "build_house", config(),
            lambda _obj, _interval: None,
            lambda obj: staging_orientation_targets(
                current, obj, "blocker", config(),
            ),
            target_specs=(TargetSpec("roof__roof", "roof", "roof"),),
            variant_placement_provider=lambda obj, interval, role: house_placement_target(
                current, state, obj, role, config(), interval=interval,
            ),
        )
        edges = generated.edges_by_target["roof__roof"]
        self.assertTrue(edges)
        self.assertFalse(any(
            edge.acted_object_track_id == "roof"
            and edge.action_type == ActionType.PLACE_HOUSE_ROLE
            for edge in edges
        ))
        self.assertTrue(any(
            edge.acted_object_track_id == "roof"
            and edge.action_type == ActionType.EXTRACT_TO_STAGING
            and edge.physical_parameters["staging_purpose"]
            == "tabletop_face_reorientation"
            and edge.physical_parameters["airborne_face_change_forbidden"]
            for edge in edges
        ))

    def test_correct_roof_face_is_preserved_without_forced_flip(self):
        objects = [
            raw_object(
                1, "roof", [0.52, -0.02, 0.015],
                label="rectangle red", size=(0.060, 0.026, 0.018),
                semantic_shape="rectangle", semantic_shape_confidence=0.95,
                semantic_shape_uncertain=False, orientation_confidence=0.95,
                orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                long_axis_base=[1.0, 0.0, 0.0],
                broad_face_normal_base=[0.0, 0.0, 1.0],
                evidence_scene_revision=1,
            ),
            raw_object(
                2, "left_upper", [0.3775, 0.270, 0.043],
                label="square yellow", size=(0.025, 0.025, 0.025),
            ),
            raw_object(
                3, "right_upper", [0.4225, 0.270, 0.043],
                label="square blue", size=(0.025, 0.025, 0.025),
            ),
        ]
        current = scene(objects, protected=("left_upper", "right_upper"))
        completion = _completion()
        completion.update(
            left_support_lower=True, right_support_lower=True,
            left_support_upper=True, right_support_upper=True,
        )
        state = replace(
            build_house_task_state(current, config()),
            role_bindings={
                "left_support_upper": "left_upper",
                "right_support_upper": "right_upper",
            },
            role_completion=completion,
        )
        generated = generate_physical_edges(
            current, (), "build_house", config(),
            lambda _obj, _interval: None,
            lambda _obj: (),
            target_specs=(TargetSpec("roof__roof", "roof", "roof"),),
            variant_placement_provider=lambda obj, interval, role: house_placement_target(
                current, state, obj, role, config(), interval=interval,
            ),
        )
        target_poses = [
            edge.physical_parameters["target_object_pose"]
            for edge in generated.edges_by_target["roof__roof"]
            if edge.action_type == ActionType.PLACE_HOUSE_ROLE
        ]
        self.assertEqual(len(target_poses), 1)
        self.assertFalse(target_poses[0]["forced_nominal_flip"])
        self.assertAlmostEqual(target_poses[0]["target_tilt_deg"], 0.0)
        roof_edges = [
            edge for edge in generated.edges_by_target["roof__roof"]
            if edge.action_type == ActionType.PLACE_HOUSE_ROLE
        ]
        self.assertTrue(all(
            edge.physical_parameters["airborne_adjustment_kind"]
            == "minimum_required_target_orientation_alignment"
            and not edge.physical_parameters["tabletop_face_reorientation_required"]
            and abs(edge.physical_parameters["tilted_place_clearance_m"]) < 1e-9
            and not edge.physical_parameters["tilted_place_clearance_applied"]
            for edge in roof_edges
        ))

    def test_wrong_rectangle_face_requires_tabletop_staging_before_roof(self):
        objects = [
            raw_object(
                1, "roof", [0.52, -0.02, 0.015],
                label="rectangle red", size=(0.060, 0.026, 0.018),
                semantic_shape="rectangle", semantic_shape_confidence=0.95,
                semantic_shape_uncertain=False, orientation_confidence=0.95,
                orientation_xyzw=[1.0, 0.0, 0.0, 0.0],
                long_axis_base=[1.0, 0.0, 0.0],
                broad_face_normal_base=[0.0, 0.0, -1.0],
                evidence_scene_revision=1,
            ),
            raw_object(2, "left_upper", [0.3775, 0.270, 0.043]),
            raw_object(3, "right_upper", [0.4225, 0.270, 0.043]),
        ]
        current = scene(objects, protected=("left_upper", "right_upper"))
        completion = _completion()
        completion.update(
            left_support_lower=True, right_support_lower=True,
            left_support_upper=True, right_support_upper=True,
        )
        state = replace(
            build_house_task_state(current, config()),
            role_bindings={
                "left_support_upper": "left_upper",
                "right_support_upper": "right_upper",
            },
            role_completion=completion,
        )
        roof = current.object_by_track("roof")
        interval = scan_grasp_yaws(
            roof, current.current_objects, config(),
        ).safe_intervals[0]
        targets = house_placement_target(
            current, state, roof, "roof", config(), interval=interval,
        )
        self.assertTrue(targets)
        self.assertTrue(all(
            target.action_type == ActionType.EXTRACT_TO_STAGING
            and target.additional_physical_parameters["staging_purpose"]
            == "tabletop_face_reorientation"
            and target.additional_physical_parameters["airborne_face_change_forbidden"]
            for target in targets
        ))

    def test_roof_blocking_support_is_picked_far_away_and_never_pushed(self):
        current = scene([
            raw_object(1, "target", [0.50, 0.05, 0.02]),
            raw_object(
                2, "roof_blocker", [0.40, 0.27, 0.02],
                label="rectangle", size=(0.045, 0.025, 0.02),
            ),
        ])

        def placement(obj, interval):
            return PlacementTarget(
                action_type=ActionType.PLACE_HOUSE_ROLE,
                target_region_id=None,
                task_role="left_support_lower",
                place_pose={
                    "frame_id": "base_link",
                    "position_m": [0.40, 0.27, 0.02],
                    "yaw_deg": 0.0,
                },
                task_progress_gain=1.0,
                expected_effects=("complete_house_role:left_support_lower",),
                precheck_results={
                    "transport_safe": True, "place_descent_safe": True,
                    "release_safe": True, "return_safe": True,
                    "protected_safe": True,
                },
            )

        generated = generate_physical_edges(
            current, ("target",), "build_house", config(), placement,
            lambda obj: staging_orientation_targets(
                current, obj, "blocker", config(),
            ),
        )
        edges = generated.edges_by_target["target"]
        self.assertTrue(edges)
        self.assertTrue(any(
            edge.action_type == ActionType.PICK_AWAY_BLOCKER
            and edge.acted_object_track_id == "roof_blocker"
            and edge.target_region_id == "house_roof_far_staging"
            for edge in edges
        ))
        self.assertFalse(any(
            edge.action_type == ActionType.NUDGE_BLOCKER
            and edge.acted_object_track_id == "roof_blocker"
            for edge in edges
        ))

    def test_house_supports_use_fifteen_millimetre_inner_gap(self):
        current = scene(_completed_house_objects(), expected=[f"t{index}" for index in range(1, 7)])
        state = build_house_task_state(current, config())
        left = current.object_by_track("t3")
        right = current.object_by_track("t4")
        center_spacing = right.center_xyz_m[0] - left.center_xyz_m[0]
        inner_gap = center_spacing - 0.5 * left.size_xyz_m[0] - 0.5 * right.size_xyz_m[0]
        self.assertAlmostEqual(state.support_inner_gap, 0.015)
        self.assertAlmostEqual(inner_gap, 0.015)

    def test_settled_lower_support_uses_local_table_contact_for_height(self):
        current = scene([raw_object(
            1,
            "placed",
            [0.3775, 0.27, -0.0007],
            size=(0.03, 0.03, 0.0239),
            local_support_surface={"support_z_base_m": -0.01265},
        )])
        valid, checks = role_observation_checks(
            current, {"left_support_lower": "placed"},
            "left_support_lower", config(),
        )
        self.assertTrue(valid)
        self.assertTrue(checks["layer_height_valid"])
        self.assertEqual(
            checks["layer_height_verification_mode"],
            "observed_local_support_contact",
        )

    def test_staging_and_nudge_success_do_not_bind_house_role(self):
        current = scene([raw_object(1, "target", [0.50, 0.0, 0.02])])
        current = current.__class__(**{
            **current.__dict__,
            "recent_action_results": (
                {
                    "action_type": ActionType.EXTRACT_TO_STAGING.value,
                    "acted_object_track_id": "staged",
                    "task_role": "left_support_lower",
                    "success": True,
                },
                {
                    "action_type": ActionType.NUDGE_BLOCKER.value,
                    "acted_object_track_id": "blocker",
                    "task_role": "left_support_lower",
                    "success": True,
                },
                {
                    "action_type": ActionType.PLACE_HOUSE_ROLE.value,
                    "acted_object_track_id": "target",
                    "task_role": "left_support_lower",
                    "success": True,
                },
            ),
        })
        state = build_house_task_state(current, config())
        self.assertEqual(state.role_bindings, {"left_support_lower": "target"})

    def test_fresh_run_recovers_grounded_lower_role_from_geometry(self):
        current = scene([raw_object(
            1,
            "already_placed",
            [0.381, 0.27, -0.0007],
            size=(0.023, 0.023, 0.0239),
            local_support_surface={"support_z_base_m": -0.01265},
        )])
        state = build_house_task_state(current, config())
        self.assertEqual(
            state.role_bindings["left_support_lower"], "already_placed",
        )
        self.assertTrue(state.role_completion["left_support_lower"])
        self.assertIn("already_placed", state.protected_structure_tracks)
        self.assertNotIn(
            "already_placed", state.role_candidate_tracks["left_support_lower"],
        )

    def test_fresh_run_recovers_cube_support_after_shape_label_flip(self):
        current = scene([raw_object(
            1,
            "relabelled_support",
            [0.41815, 0.27, -0.0006],
            label="semi circle",
            size=(0.0213, 0.0209, 0.0226),
            local_support_surface={"support_z_base_m": -0.0119},
        )])
        state = build_house_task_state(current, config())
        self.assertEqual(
            state.role_bindings["right_support_lower"],
            "relabelled_support",
        )
        self.assertTrue(state.role_completion["right_support_lower"])
        self.assertIn("relabelled_support", state.protected_structure_tracks)

    def test_fresh_run_recovers_visible_upper_and_hidden_lower_from_support_surface(self):
        current = scene([
            raw_object(
                1, "left_upper", [0.3879, 0.2707, 0.0227],
                size=(0.0261, 0.0226, 0.0246),
                local_support_surface={"support_z_base_m": 0.01033},
            ),
            raw_object(
                2, "right_lower", [0.4221, 0.2731, -0.0017],
                size=(0.0257, 0.0238, 0.0241),
                local_support_surface={"support_z_base_m": -0.01375},
            ),
            raw_object(3, "loose", [0.4556, 0.1225, -0.0013]),
        ])
        state = build_house_task_state(current, config())
        hidden = state.role_bindings["left_support_lower"]
        self.assertTrue(hidden.startswith("inferred_hidden_left_support_below_"))
        self.assertEqual(state.role_bindings["left_support_upper"], "left_upper")
        self.assertEqual(state.role_bindings["right_support_lower"], "right_lower")
        self.assertTrue(state.role_completion["left_support_lower"])
        self.assertTrue(state.role_completion["left_support_upper"])
        self.assertTrue(state.role_completion["right_support_lower"])
        self.assertFalse(state.role_completion["right_support_upper"])
        self.assertEqual(legal_incomplete_roles(state.role_completion), ("right_support_upper",))
        self.assertIn("left_upper", state.protected_structure_tracks)

    def test_fresh_run_recovers_four_hidden_supports_from_elevated_roof(self):
        roof = raw_object(
            1, "assembled_roof", [0.4014, 0.2678, 0.0187],
            label="rectangle", size=(0.0597, 0.0295, 0.0639),
            local_support_surface={"support_z_base_m": -0.01327},
            top_z_base_m=0.0506,
            height_surface_cluster={"coverage_ratio": 0.66},
            yaw_deg=2.65,
        )
        triangle = raw_object(
            2, "triangle", [0.4025, 0.1304, -0.0052],
            label="triangle", size=(0.0410, 0.0237, 0.0186),
        )
        current = scene([roof, triangle])

        state = build_house_task_state(current, config())

        for role in (
            "left_support_lower", "left_support_upper",
            "right_support_lower", "right_support_upper", "roof",
        ):
            self.assertTrue(state.role_completion[role], role)
        self.assertFalse(state.role_completion["triangle_top"])
        self.assertEqual(state.role_bindings["roof"], "assembled_roof")
        self.assertEqual(len(state.inferred_hidden_tracks), 4)
        self.assertEqual(legal_incomplete_roles(state.role_completion), ("triangle_top",))

    def test_table_rectangle_cannot_infer_hidden_house_supports(self):
        roof = raw_object(
            1, "loose_roof", [0.400, 0.270, -0.006],
            label="rectangle", size=(0.060, 0.030, 0.014),
            local_support_surface={"support_z_base_m": -0.013},
            top_z_base_m=0.001,
            height_surface_cluster={"coverage_ratio": 0.80},
        )

        state = build_house_task_state(scene([roof]), config())

        self.assertFalse(state.inferred_hidden_tracks)
        self.assertFalse(state.role_completion["roof"])

    def test_verified_lower_support_remains_complete_while_occluded_by_upper(self):
        visible = scene([raw_object(
            1,
            "lower",
            [0.381, 0.27, -0.0007],
            size=(0.023, 0.023, 0.0239),
            local_support_surface={"support_z_base_m": -0.01265},
        )])
        previous = build_house_task_state(visible, config())
        self.assertTrue(previous.role_completion["left_support_lower"])
        remembered_lower = replace(
            visible.object_by_track("lower"),
            currently_visible=False,
            protected=False,
        )
        occluded = replace(
            scene([raw_object(2, "upper_candidate", [0.50, 0.05, 0.02])]),
            collision_obstacles=(remembered_lower,),
        )
        current = build_house_task_state(occluded, config(), previous=previous)
        self.assertEqual(current.role_bindings["left_support_lower"], "lower")
        self.assertTrue(current.role_completion["left_support_lower"])
        self.assertIn("lower", current.protected_structure_tracks)
        upper = occluded.object_by_track("upper_candidate")
        interval = scan_grasp_yaws(
            upper, (*occluded.current_objects, *occluded.collision_obstacles), config(),
        ).safe_intervals[0]
        targets = house_placement_target(
            occluded, current, upper, "left_support_upper", config(), interval=interval,
        )
        self.assertTrue(targets)
        self.assertAlmostEqual(
            targets[0].place_pose["position_m"][0],
            remembered_lower.center_xyz_m[0],
        )


def _completion():
    return {
        "left_support_lower": False,
        "right_support_lower": False,
        "left_support_upper": False,
        "right_support_upper": False,
        "roof": False,
        "triangle_top": False,
    }


def _valid_roof(**changes):
    values = dict(
        track_id="roof",
        groove_face_state="opening_down",
        face_up=True,
        long_axis_yaw_deg=0.0,
        current_grasp_pose=None,
        satisfies_roof_orientation=True,
        left_support_coverage_m=0.01,
        right_support_coverage_m=0.01,
        center_offset_m=0.0,
        left_support_margin_m=0.01,
        right_support_margin_m=0.01,
        placement_stable=True,
    )
    values.update(changes)
    return RoofOrientationState(**values)


def _valid_triangle(**changes):
    values = dict(
        track_id="triangle",
        apex_direction="up",
        base_edge_direction="roof_aligned",
        face_state="upright",
        target_yaw_deg=0.0,
        base_contact=True,
        center_of_mass_projection_m=0.001,
        support_margin_m=0.01,
        roof_relative_pose={"centered": True},
    )
    values.update(changes)
    return TriangleOrientationState(**values)


def _completed_house_objects():
    roles = (
        "left_support_lower", "right_support_lower", "left_support_upper",
        "right_support_upper", "roof", "triangle_top",
    )
    labels = (
        "square red", "square blue", "square green", "square yellow",
        "concave_rectangle orange", "triangle purple",
    )
    centers = (
        [0.3775, 0.27, 0.02], [0.4225, 0.27, 0.02],
        [0.3775, 0.27, 0.06], [0.4225, 0.27, 0.06],
        [0.40, 0.27, 0.09], [0.40, 0.27, 0.12],
    )
    sizes = (
        [0.03, 0.03, 0.04], [0.03, 0.03, 0.04],
        [0.03, 0.03, 0.04], [0.03, 0.03, 0.04],
        [0.14, 0.03, 0.02], [0.03, 0.03, 0.04],
    )
    extras = ({}, {}, {}, {}, {
        "groove_face_state": "opening_down", "face_up": True,
        "satisfies_roof_orientation": True, "long_axis_yaw_deg": 0.0,
    }, {
        "apex_direction": "up", "base_edge_direction": "roof_aligned",
        "face_state": "upright", "center_of_mass_projection_m": 0.0,
    })
    objects = []
    for index, (role, label, center, size, extra) in enumerate(
        zip(roles, labels, centers, sizes, extras), start=1,
    ):
        objects.append(raw_object(
            index, f"t{index}", center, label=label, size=size,
            house_role=role, **extra,
        ))
    return objects


if __name__ == "__main__":
    unittest.main()
