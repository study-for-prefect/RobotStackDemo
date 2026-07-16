import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.pushed_object_sweep import analyze_push_contact_chain
from tools.robot.moveit_preview.orientation import object_yaw_candidate_values
from tools.workflows.stack_demo.app import main
from tools.workflows.stack_demo.clutter.edge_generation import generate_physical_edges
from tools.workflows.stack_demo.clutter.path_safety import build_nudge_parameters
from tools.workflows.stack_demo.clutter.push_orientation_priority import prefer_axis_aligned_pushes
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.clutter.grasp_edges import (
    normalize_gripper_yaw_deg,
    scan_grasp_yaws,
)
from tools.workflows.stack_demo.policy.edge_selector import _edge_request

from tests.new_arch_fixtures import config, placement, raw_object, scene


class OrganizeCandidateGenerationFixTests(unittest.TestCase):
    def test_145_degree_grasp_axis_normalizes_to_negative_35(self):
        self.assertEqual(normalize_gripper_yaw_deg(145.0), -35.0)

    def test_all_scanned_grasp_axes_use_unique_half_turn_interval(self):
        current = scene([raw_object(1, "a", [0.50, 0.0, 0.02])])
        scan = scan_grasp_yaws(current.current_objects[0], current.current_objects, config())
        self.assertEqual(len(scan.samples), 36)
        self.assertTrue(all(-90.0 <= item["yaw_deg"] < 90.0 for item in scan.samples))

    def test_release_planning_exposes_both_180_degree_equivalent_yaws(self):
        values = object_yaw_candidate_values({
            "target_yaw_deg": 145.0,
            "target_yaw_valid": True,
            "parallel_gripper_axis_equivalent": True,
        }, SimpleNamespace())
        self.assertEqual(set(values), {-35.0, 145.0})

    def test_each_unfinished_object_gets_direct_scan_and_all_push_combinations(self):
        current = scene([
            raw_object(1, "a", [0.46, -0.02, 0.02], color="red"),
            raw_object(2, "b", [0.54, 0.08, 0.02], color="blue"),
        ])
        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.nudge_sweep_checks",
            side_effect=_passing_push_checks,
        ):
            generated = generate_physical_edges(
                current,
                ["a", "b"],
                "organize_blocks",
                config(),
                placement,
                lambda obj: None,
            )
        edge_count = sum(len(edges) for edges in generated.edges_by_target.values())
        summary = generated.audit.summary(
            physical_action_edge_count=edge_count,
            target_option_count=2,
        )
        for track_id in ("a", "b"):
            item = summary["objects"][track_id]
            self.assertEqual(item["direct_grasp"]["raw_generated_count"], 36)
            directions = item["push_directions"]
            self.assertEqual(set(directions), {"+x", "-x", "+y", "-y"})
            self.assertEqual(
                {key: value["contact_side"] for key, value in directions.items()},
                {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"},
            )
            self.assertTrue(all(value["generated_count"] == 9 for value in directions.values()))

    def test_failed_direct_grasps_do_not_suppress_pushes(self):
        current = scene([raw_object(1, "a", [0.50, 0.0, 0.02])])
        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.scan_grasp_yaws",
            side_effect=lambda obj, objects, cfg: _all_failed_scan(obj, objects, cfg),
        ), patch(
            "tools.workflows.stack_demo.clutter.edge_generation.nudge_sweep_checks",
            side_effect=_passing_push_checks,
        ):
            generated = generate_physical_edges(
                current, ["a"], "organize_blocks", config(), placement, lambda obj: None,
            )
        self.assertTrue(generated.edges_by_target["a"])
        self.assertTrue(all(edge.action_type.value == "nudge_blocker" for edge in generated.edges_by_target["a"]))

    def test_equivalent_diagonal_push_is_removed_when_axis_aligned_push_passes(self):
        current = scene([raw_object(1, "a", [0.50, 0.0, 0.02])])
        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.nudge_sweep_checks",
            side_effect=_passing_push_checks,
        ):
            generated = generate_physical_edges(
                current, ["a"], "organize_blocks", config(), placement, lambda obj: None,
            )
        pushes = [
            item for item in generated.edges_by_target["a"]
            if item.action_type.value == "nudge_blocker"
        ]
        self.assertTrue(pushes)
        self.assertEqual(
            {item.physical_parameters["push_wrist_yaw_deg"] for item in pushes},
            {0.0, 90.0, 135.0},
        )
        selectable = prefer_axis_aligned_pushes(pushes)
        self.assertEqual(
            {item.physical_parameters["push_wrist_yaw_deg"] for item in selectable},
            {0.0, 90.0},
        )

    def test_prepush_descends_outside_contact_then_approaches_horizontally(self):
        current = scene([raw_object(1, "a", [0.50, 0.0, 0.02])])
        obj = current.current_objects[0]
        physical = build_nudge_parameters(
            obj, [1.0, 0.0, 0.0], 0.05, 0.0, config(),
        )
        prepush = physical["prepush_pose"]["position_m"]
        precontact = physical["precontact_pose"]["position_m"]
        contact = physical["push_start"]["position_m"]
        push_end = physical["push_end"]["position_m"]
        plan = {
            "schema_version": "push_execution_plan_v1",
            "frame_id": "base_link",
            "obstacle": {
                "geometry_center_m": list(obj.center_xyz_m),
                "dimensions_m": list(obj.size_xyz_m),
            },
            "direction_base": [1.0, 0.0, 0.0],
            "distance_m": 0.05,
            "lift_m": 0.05,
            "retreat_lift_m": 0.10,
            "contact_z_offset_m": 0.015,
            "contact_standoff_m": physical["contact_standoff_m"],
            "contact_clearance_m": physical["contact_clearance_m"],
            "prepush_clearance_m": physical["prepush_clearance_m"],
        }
        from tools.robot.push_primitives import build_push_targets
        targets = build_push_targets(plan)
        self.assertAlmostEqual(prepush[0], precontact[0])
        self.assertAlmostEqual(prepush[1], precontact[1])
        self.assertGreater(prepush[2], precontact[2])
        self.assertAlmostEqual(contact[0] - precontact[0], 0.015)
        self.assertAlmostEqual(
            physical["contact_standoff_m"],
            physical["object_contact_extent_m"]
            + physical["tool_contact_extent_m"]
            + physical["contact_clearance_m"],
        )
        self.assertAlmostEqual(push_end[0] - contact[0], 0.056)
        self.assertAlmostEqual(targets["pre_push"][2] - targets["contact"][2], 0.05)
        self.assertAlmostEqual(targets["retreat"][2] - targets["push_end"][2], 0.10)

    def test_blocked_contact_side_rejects_only_that_direction(self):
        current = scene([raw_object(1, "a", [0.50, 0.0, 0.02])])

        def checks(scene_state, blocker, physical, planner_config):
            if physical["contact_side"] == "+x":
                return _passing_push_checks(
                    scene_state,
                    blocker,
                    physical,
                    planner_config,
                    passed=False,
                    reason="pre_push_contact_side_blocked",
                    blocking=["neighbor"],
                )
            return _passing_push_checks(scene_state, blocker, physical, planner_config)

        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.nudge_sweep_checks",
            side_effect=checks,
        ):
            generated = generate_physical_edges(
                current, ["a"], "organize_blocks", config(), placement, lambda obj: None,
            )
        summary = generated.audit.summary(
            physical_action_edge_count=len(generated.edges_by_target["a"]),
            target_option_count=1,
        )
        directions = summary["objects"]["a"]["push_directions"]
        self.assertEqual(directions["-x"]["accepted_count"], 0)
        self.assertEqual(directions["-x"]["blocking_track_ids"], ["neighbor"])
        self.assertGreater(directions["+x"]["accepted_count"], 0)
        self.assertGreater(directions["+y"]["accepted_count"], 0)
        self.assertGreater(directions["-y"]["accepted_count"], 0)

    def test_second_recorded_scene_no_longer_collapses_to_empty_graph(self):
        current = _recorded_scene("organize_execute_20260716_100336")
        generated = generate_physical_edges(
            current,
            current.expected_tracks,
            "organize_blocks",
            config(),
            placement,
            lambda obj: None,
        )
        edges = [edge for values in generated.edges_by_target.values() for edge in values]
        summary = generated.audit.summary(
            physical_action_edge_count=len(edges),
            target_option_count=sum(bool(values) for values in generated.edges_by_target.values()),
        )
        self.assertGreater(len(edges), 0)
        self.assertEqual(summary["direct_grasp_objects_scanned"], 7)
        self.assertEqual(summary["push_directions_scanned"], 28)
        self.assertGreater(summary["push_chain_generated_count"], 0)

    def test_four_latest_recorded_scenes_keep_many_physical_candidates(self):
        minimum_edges = {
            "organize_execute_20260716_115426": 30,
            "organize_execute_20260716_115532": 30,
            "organize_execute_20260716_115554": 30,
            "organize_execute_20260716_120021": 30,
        }
        for run_name, minimum in minimum_edges.items():
            with self.subTest(run_name=run_name):
                current = _recorded_scene(run_name)
                generated = generate_physical_edges(
                    current,
                    current.expected_tracks,
                    "organize_blocks",
                    config(),
                    placement,
                    lambda obj: None,
                )
                edges = [
                    edge
                    for values in generated.edges_by_target.values()
                    for edge in values
                ]
                options = build_target_options(
                    current,
                    "organize_blocks",
                    generated.edges_by_target,
                    generated.grasp_scans,
                )
                self.assertGreaterEqual(len(edges), minimum)
                self.assertGreaterEqual(len(options), 4)

    def test_large_edge_selection_request_is_compact(self):
        run_name = "organize_execute_20260716_115426"
        current = _recorded_scene(run_name)
        generated = generate_physical_edges(
            current,
            current.expected_tracks,
            "organize_blocks",
            config(),
            placement,
            lambda obj: None,
        )
        options = build_target_options(
            current,
            "organize_blocks",
            generated.edges_by_target,
            generated.grasp_scans,
        )
        selected = next(item for item in options if item.target_track_id == "track_yellow_02")
        eligible = [
            edge
            for values in generated.edges_by_target.values()
            for edge in values
            if edge.candidate_id in selected.feasible_first_step_edge_ids
        ]
        task_path = (
            Path(__file__).resolve().parents[1]
            / "runtime" / run_name / "cycle_001_revision_1" / "task_state.json"
        )
        request = _edge_request(
            current,
            json.loads(task_path.read_text()),
            selected,
            eligible,
            5,
        )
        serialized = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        self.assertLess(len(serialized), 16_000)
        # The explicit 15 mm pre-contact descent can safely reject a few entry
        # columns that the old contact-XY descent treated as clear; the policy
        # must still receive a broad set rather than a singleton fallback.
        self.assertGreaterEqual(len(request["physical_edges"]), 18)
        self.assertNotIn("physical_parameters", request["physical_edges"][0])
        self.assertNotIn("precheck_results", request["physical_edges"][0])
        self.assertNotIn("failure_fingerprint", request["physical_edges"][0])

    def test_missing_unresolved_track_reports_internal_generation_error(self):
        current = scene([])
        generated = generate_physical_edges(
            current, ["missing"], "organize_blocks", config(), placement, lambda obj: None,
        )
        summary = generated.audit.summary(
            physical_action_edge_count=0,
            target_option_count=0,
        )
        self.assertTrue(summary["candidate_generation_internal_error"])
        self.assertEqual(
            generated.audit.rejections[-1]["rejection_reason"],
            "candidate_generation_internal_error",
        )

    def test_empty_cycle_reobserves_three_times_before_exit(self):
        raw = {
            "scene_revision": 1,
            "frame_id": "base_link",
            "objects": [{
                **raw_object(1, "ambiguous", [0.50, 0.0, 0.02]),
                "tracking_ambiguous": True,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene.json"
            source.write_text(json.dumps(raw), encoding="utf-8")
            result = main([
                "--task-type", "organize_blocks",
                "--offline-scene-state", str(source),
                "--output-dir", str(root / "output"),
                "--no-image",
            ])
            summaries = sorted((root / "output").glob("cycle_*/candidate_generation_summary.json"))
            attempts = sorted((root / "output").glob("cycle_*/reobserve_attempt.json"))
        self.assertEqual(result, 2)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(len(attempts), 3)


class PushChainAnalysisTests(unittest.TestCase):
    def test_a_pushes_b_then_c_as_controlled_chain(self):
        objects = [
            _geometry("a", 0.30),
            _geometry("b", 0.335),
            _geometry("c", 0.370),
        ]
        result = analyze_push_contact_chain(
            _push_plan(objects[0], 0.09), objects, ("a",), (), 0.002,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["chain_track_ids"], ["a", "b", "c"])
        self.assertEqual(len(result["controlled_contacts"]), 2)

    def test_chain_size_is_not_a_hard_rejection(self):
        objects = [_geometry(chr(97 + index), 0.30 + 0.032 * index) for index in range(5)]
        result = analyze_push_contact_chain(
            _push_plan(objects[0], 0.15), objects, ("a",), (), 0.001,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["chain_object_count"], 5)

    def test_chain_rejects_actual_completed_object(self):
        objects = [_geometry("a", 0.30), _geometry("done", 0.34, completed=True)]
        result = analyze_push_contact_chain(
            _push_plan(objects[0], 0.06), objects, ("a",), ("done",), 0.002,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["hard_collisions"][0]["reason"], "protected_completed_object_collision")

    def test_chain_rejects_fixed_obstacle(self):
        objects = [_geometry("a", 0.30), {**_geometry("wall", 0.34), "is_fixed": True}]
        result = analyze_push_contact_chain(
            _push_plan(objects[0], 0.06), objects, ("a",), (), 0.002,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["hard_collisions"][0]["reason"], "fixed_obstacle_collision")

    def test_chain_rejects_workspace_boundary(self):
        primary = _geometry("a", 0.62)
        plan = _push_plan(primary, 0.06)
        plan["workspace_bounds"] = {"xmin": 0.235, "xmax": 0.65, "ymin": -0.1, "ymax": 0.4}
        result = analyze_push_contact_chain(plan, [primary], ("a",), (), 0.002)
        self.assertFalse(result["passed"])
        self.assertEqual(result["hard_collisions"][0]["reason"], "push_chain_workspace_violation")


def _passing_push_checks(
    scene_state,
    blocker,
    physical,
    planner_config,
    *,
    passed=True,
    reason=None,
    blocking=(),
):
    return {
        "passed": passed,
        "geometry_checks_passed": passed,
        "prepush_reachable": passed,
        "prepush_descent_safe": passed,
        "horizontal_sweep_safe": passed,
        "push_end_safe": passed,
        "protected_safe": passed,
        "contact_side_clear": passed,
        "blocking_track_ids": list(blocking),
        "rejection_reasons": [] if reason is None else [reason],
        "chain_track_ids": [blocker.track_id],
        "chain_object_count": 1,
        "secondary_contact_expected": False,
        "estimated_displacements_m": {blocker.track_id: physical["push_distance_m"]},
        "push_chain_estimation_method": "test",
    }


def _all_failed_scan(obj, objects, planner_config):
    return scan_grasp_yaws(
        obj,
        objects,
        planner_config,
        geometry_check=lambda target, scene_objects, yaw: {
            "opening_ok": True,
            "center_offset_ok": True,
            "contact_length_ok": True,
            "finger_safe": False,
            "palm_safe": True,
            "descent_safe": False,
            "lift_safe": False,
            "blocking_track_ids": [],
        },
    )


def _recorded_scene(run_name):
    path = (
        Path(__file__).resolve().parents[1]
        / "runtime" / run_name / "cycle_001_revision_1" / "scene_state.json"
    )
    saved = json.loads(path.read_text())
    raw = {
        "scene_revision": saved["scene_revision"],
        "frame_id": "base_link",
        "objects": [
            {
                "id": item["detector_id"],
                "track_id": item["track_id"],
                "label": item["class_name"],
                "visual_color": item["color"],
                "geometry_center_m": item["center_xyz_m"],
                "dimensions_m": item["size_xyz_m"],
                "yaw_deg": item["yaw_deg"],
                "orientation_confidence": item["orientation_confidence"],
            }
            for item in saved["current_objects"]
        ],
    }
    return scene(
        raw["objects"],
        revision=saved["scene_revision"],
        expected=saved["expected_tracks"],
    )


def _geometry(track_id, x, *, completed=False):
    return {
        "id": track_id,
        "geometry_center_m": [x, 0.0, 0.02],
        "dimensions_m": [0.03, 0.03, 0.04],
        "table_yaw_deg": 0.0,
        "is_completed": completed,
        "protection_kind": "completed_object" if completed else None,
    }


def _push_plan(primary, distance):
    return {
        "schema_version": "push_execution_plan_v1",
        "frame_id": "base_link",
        "obstacle": primary,
        "direction_base": [1.0, 0.0, 0.0],
        "distance_m": distance,
    }


if __name__ == "__main__":
    unittest.main()
