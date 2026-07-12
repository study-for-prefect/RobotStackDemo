import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.task_action_adapter import (
    TaskActionSchemaError,
    adapt_task_action_to_legacy_action,
)
from robot_scene_pipeline.task_action_validation import validate_task_action
from robot_scene_pipeline.task_dynamic_protection import derive_dynamic_protection
from robot_scene_pipeline.task_geometry import (
    footprint_boundary_distance,
    footprint_inside_region,
    footprint_overlap,
    object_footprint_polygon,
)
from robot_scene_pipeline.task_goal_evaluator import (
    evaluate_columns_layout,
    evaluate_grid_layout,
    evaluate_rows_layout,
    evaluate_task_goal_progress,
)
from robot_scene_pipeline.vlm_action_validation import validate_vlm_action_decision
from tools.workflows.stack_demo.execution_safety import validate_execution_source
from tools.workflows.stack_demo.task_workflow import (
    _select_task_action,
    _state_with_dynamic_protection,
    run_semantic_task_workflow,
)


CONFIG = {
    "organize_defaults": {
        "grouping_key": "color", "layout_type": "rows", "include_scope": "all_detected_blocks",
        "allow_stacking": False, "minimum_spacing_m": 0.015, "alignment_tolerance_m": 0.012,
    },
    "house_semantics": {
        "support_height_tolerance_m": 0.008, "vertical_contact_tolerance_m": 0.006,
        "minimum_support_overlap_ratio": 0.2, "minimum_support_separation_m": 0.025,
        "maximum_structure_tilt_deg": 10.0,
    },
}


class ActionSchemaTests(unittest.TestCase):
    def test_adapter_copies_selected_id_without_changing_action(self):
        action = {"action_type": "nudge", "selected_object_id": 7, "distance_m": 0.02}
        adapted = adapt_task_action_to_legacy_action(action)
        self.assertEqual(adapted["object_id"], 7)
        self.assertEqual(adapted["distance_m"], 0.02)
        self.assertNotIn("object_id", action)

    def test_pick_away_adapter_preserves_safe_place(self):
        action = {"action_type": "pick_away", "selected_object_id": 3, "safe_place_center_base_m": [0.2, 0.1, 0.02]}
        adapted = adapt_task_action_to_legacy_action(action)
        self.assertEqual(adapted["object_id"], 3)
        self.assertEqual(adapted["safe_place_center_base_m"], action["safe_place_center_base_m"])

    def test_conflicting_ids_are_rejected(self):
        with self.assertRaises(TaskActionSchemaError) as raised:
            adapt_task_action_to_legacy_action({"selected_object_id": 1, "object_id": 2})
        self.assertEqual(raised.exception.feedback["reason"], "conflicting_object_id_fields")

    def test_even_matching_legacy_id_is_forbidden_in_vlm_action(self):
        with self.assertRaises(TaskActionSchemaError) as raised:
            adapt_task_action_to_legacy_action({"selected_object_id": 1, "object_id": 1})
        self.assertEqual(raised.exception.feedback["reason"], "legacy_object_id_forbidden")

    def test_missing_selected_id_is_rejected(self):
        with self.assertRaises(TaskActionSchemaError):
            adapt_task_action_to_legacy_action({"action_type": "nudge", "object_id": 1})

    def test_adapted_nudge_is_read_by_legacy_validator(self):
        state = _clearance_state()
        decision = {
            "action_type": "nudge", "selected_object_id": 1, "object_label": "square red",
            "object_center_base_m": [0.25, 0.0, 0.02], "target_object_id": 2,
            "target_object_label": "square blue", "target_object_center_base_m": [0.40, 0.0, 0.02],
            "direction_base": [0.0, 1.0], "distance_m": 0.02, "contact_side": "negative_y",
            "gripper_yaw_rad": 0.0,
        }
        _action, report = validate_vlm_action_decision(
            adapt_task_action_to_legacy_action(decision), state, protected_ids=[],
        )
        self.assertTrue(report["checks"]["object_id_exists"]["ok"])


class OrganizeActionTests(unittest.TestCase):
    def test_valid_organize_pick_place_needs_no_role_id(self):
        action, report = validate_task_action(
            _organize_action(), _organize_state(), _organize_contract(), _organize_plan(), {},
        )
        self.assertTrue(report["accepted"])
        self.assertNotIn("role_id", action)
        self.assertNotIn("object_id", action)

    def test_unknown_group_is_rejected(self):
        proposal = _organize_action(); proposal["group_id"] = "missing"
        _action, report = validate_task_action(proposal, _organize_state(), _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "unknown_group_id")

    def test_object_not_matching_group_is_rejected(self):
        proposal = _organize_action(); proposal.update({"selected_object_id": 2, "object_label": "square blue", "object_center_base_m": [0.45, 0.0, 0.02]})
        _action, report = validate_task_action(proposal, _organize_state(), _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "selected_object_does_not_match_group")

    def test_wrong_target_region_is_rejected(self):
        proposal = _organize_action(); proposal["target_region_id"] = "blue"
        _action, report = validate_task_action(proposal, _organize_state(), _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "target_region_does_not_match_group")

    def test_center_inside_but_footprint_outside_region_is_rejected(self):
        proposal = _organize_action(); proposal["target_pose_base"]["position_m"][0] = 0.295
        _action, report = validate_task_action(proposal, _organize_state(), _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "target_pose_outside_group_region")

    def test_layout_violation_is_rejected_without_pose_correction(self):
        state = _organize_state(); state["objects"].append(_object(3, "square red", [0.18, 0.0, 0.02]))
        proposal = _organize_action(); proposal["target_pose_base"]["position_m"] = [0.25, 0.03, 0.02]
        _action, report = validate_task_action(proposal, state, _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "target_pose_violates_group_layout")
        self.assertEqual(proposal["target_pose_base"]["position_m"], [0.25, 0.03, 0.02])

    def test_disappeared_temporary_object_requests_reselection(self):
        state = _organize_state(); state["objects"] = [state["objects"][1]]
        _action, report = validate_task_action(
            _organize_action(), state, _organize_contract(), _organize_plan(), {},
        )
        self.assertEqual(report["reason"], "selected_object_no_longer_exists")


class GeometryAndCompletionTests(unittest.TestCase):
    def test_rotated_footprint_can_cross_region_with_center_inside(self):
        obj = _object(1, "rectangle red", [0.29, 0.0, 0.02], [0.08, 0.02, 0.04]); obj["yaw_rad"] = 0.785
        region = {"xmin": 0.2, "xmax": 0.3, "ymin": -0.05, "ymax": 0.05}
        self.assertFalse(footprint_inside_region(object_footprint_polygon(obj), region))

    def test_footprints_overlap_when_centers_look_separated(self):
        first = _object(1, "rectangle red", [0.20, 0.0, 0.02], [0.08, 0.03, 0.04])
        second = _object(2, "rectangle red", [0.26, 0.0, 0.02], [0.08, 0.03, 0.04])
        self.assertTrue(footprint_overlap(object_footprint_polygon(first), object_footprint_polygon(second)))
        self.assertEqual(footprint_boundary_distance(object_footprint_polygon(first), object_footprint_polygon(second)), 0.0)

    def test_empty_required_group_is_not_complete_and_is_temporarily_unobserved(self):
        plan = _organize_plan(); plan["groups"].append({"group_id": "green_group", "group_value": "green", "object_ids": [], "target_region_id": "green"})
        plan["target_regions"].append({"region_id": "green", "bounds_base_m": {"xmin": 0.31, "xmax": 0.34, "ymin": -0.05, "ymax": 0.05}})
        progress = evaluate_task_goal_progress(_organize_state(), _organize_contract(), plan, CONFIG)
        self.assertFalse(progress["task_complete"])
        diagnostic = next(item for item in progress["group_diagnostics"] if item["group_id"] == "green_group")
        self.assertTrue(diagnostic["temporarily_unobserved"])

    def test_boundary_spacing_too_small_is_not_complete(self):
        state = _organize_state(); state["objects"] = [_object(1, "square red", [0.20, 0.0, 0.02]), _object(3, "square red", [0.243, 0.0, 0.02]), state["objects"][1]]
        progress = evaluate_task_goal_progress(state, _organize_contract(), _organize_plan(), CONFIG)
        self.assertFalse(progress["task_complete"])
        self.assertTrue(next(item for item in progress["group_diagnostics"] if item["group_id"] == "red_group")["spacing_violations"])

    def test_rows_columns_and_grid_have_distinct_evaluators(self):
        row = [_object(1, "x", [0.2, 0.0, 0.02]), _object(2, "x", [0.25, 0.005, 0.02])]
        column = [_object(1, "x", [0.2, 0.0, 0.02]), _object(2, "x", [0.205, 0.05, 0.02])]
        grid = [_object(i + 1, "x", [x, y, 0.02]) for i, (x, y) in enumerate(((0.2, 0.0), (0.25, 0.0), (0.2, 0.05), (0.25, 0.05)))]
        self.assertTrue(evaluate_rows_layout(row, 0.012))
        self.assertFalse(evaluate_columns_layout(row, 0.012))
        self.assertTrue(evaluate_columns_layout(column, 0.012))
        self.assertTrue(evaluate_grid_layout(grid, 0.012))

    def test_incomplete_grid_is_rejected(self):
        diagonal = [_object(1, "x", [0.2, 0.0, 0.02]), _object(2, "x", [0.25, 0.05, 0.02])]
        self.assertFalse(evaluate_grid_layout(diagonal, 0.012))

    def test_object_in_other_groups_region_keeps_organize_incomplete(self):
        state = _organize_state(); state["objects"][0]["geometry_center_m"][0] = 0.40
        state["objects"][1]["geometry_center_m"][0] = 0.50
        progress = evaluate_task_goal_progress(state, _organize_contract(), _organize_plan(), CONFIG)
        self.assertFalse(progress["task_complete"])
        self.assertIn("red_group.all_inside", progress["unsatisfied_predicates"])


class HouseCombinationTests(unittest.TestCase):
    def test_later_legal_role_combination_completes_house(self):
        state = _house_state()
        state["objects"].insert(0, _object(9, "square green", [0.20, 0.0, 0.02]))
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertTrue(progress["task_complete"])
        self.assertGreater(len(progress["role_assignment_candidates"]), 1)

    def test_same_object_cannot_fill_both_support_roles(self):
        state = _house_state(); state["objects"] = [state["objects"][0], state["objects"][2]]
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertFalse(progress["task_complete"])
        self.assertIn({"type": "insufficient_role_resources"}, progress["scene_events"])

    def test_id_change_rebinds_to_current_geometry(self):
        state = _house_state()
        for obj in state["objects"]: obj["id"] += 20
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertTrue(progress["task_complete"])
        self.assertEqual(progress["selected_role_assignment"]["roof"]["current_object_id"], 23)

    def test_support_height_difference_breaks_completion(self):
        state = _house_state(); state["objects"][1]["geometry_center_m"][2] += 0.012
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertIn("supports.height_aligned", progress["unsatisfied_predicates"])

    def test_support_boundary_separation_too_small_breaks_completion(self):
        state = _house_state(); state["objects"][1]["geometry_center_m"][0] = 0.34
        state["objects"][2]["geometry_center_m"][0] = 0.32
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertIn("supports.separated", progress["unsatisfied_predicates"])

    def test_roof_tilt_breaks_completion(self):
        state = _house_state(); state["objects"][2]["tilt_deg"] = 15.0
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertIn("roof.tilt_within_limit", progress["unsatisfied_predicates"])

    def test_full_house_footprint_outside_workspace_is_rejected(self):
        state = _house_state(); state["table_bounds"]["xmax"] = 0.40
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertIn("house.inside_workspace", progress["unsatisfied_predicates"])

    def test_broken_structure_enters_repair_and_reports_lost_predicate(self):
        before = evaluate_task_goal_progress(_house_state(), _house_contract(), _house_plan(), CONFIG)
        state = _house_state(); state["objects"][2]["geometry_center_m"][0] = 0.30
        after = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG, before)
        self.assertTrue(after["repair_required"])
        self.assertTrue(any(event["type"] == "goal_predicate_lost" for event in after["scene_events"]))

    def test_large_motion_that_preserves_geometry_is_recoverable(self):
        before = evaluate_task_goal_progress(_house_state(), _house_contract(), _house_plan(), CONFIG)
        state = _house_state()
        for obj in state["objects"]: obj["geometry_center_m"][0] += 0.10
        after = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG, before)
        self.assertTrue(after["task_complete"])
        self.assertTrue(any(event["type"] == "object_moved_far" for event in after["scene_events"]))

    def test_current_id_change_is_reported_without_task_failure(self):
        before = evaluate_task_goal_progress(_house_state(), _house_contract(), _house_plan(), CONFIG)
        state = _house_state()
        for obj in state["objects"]: obj["id"] += 10
        after = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG, before)
        self.assertTrue(after["task_complete"])
        self.assertTrue(any(event["type"] == "object_id_changed" for event in after["scene_events"]))

    def test_preferred_complete_combination_wins_when_multiple_are_legal(self):
        state = _house_state()
        state["objects"].extend([
            _object(4, "square green", [0.30, 0.0, 0.02]),
            _object(5, "square orange", [0.40, 0.0, 0.02]),
        ])
        plan = {"role_assignments": [
            {"role_id": "left_support", "selected_object_id": 4},
            {"role_id": "right_support", "selected_object_id": 5},
            {"role_id": "roof", "selected_object_id": 3},
        ]}
        progress = evaluate_task_goal_progress(state, _house_contract(), plan, CONFIG)
        self.assertEqual(progress["selected_role_assignment"]["left_support"]["current_object_id"], 4)
        self.assertEqual(progress["selected_role_assignment"]["right_support"]["current_object_id"], 5)

    def test_outside_objects_do_not_block_inside_same_class_replacements(self):
        state = _house_state()
        state["objects"][0]["geometry_center_m"][0] = 0.02
        state["objects"][1]["geometry_center_m"][0] = 0.68
        state["objects"].extend([
            _object(4, "square green", [0.30, 0.0, 0.02]),
            _object(5, "square orange", [0.40, 0.0, 0.02]),
        ])
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        self.assertTrue(progress["task_complete"])
        selected_ids = {item["current_object_id"] for item in progress["selected_role_assignment"].values()}
        self.assertTrue({4, 5}.issubset(selected_ids))

    def test_vlm_completion_claim_does_not_override_geometry(self):
        state = _house_state(); state["objects"][2]["geometry_center_m"][0] = 0.55
        plan = _house_plan(); plan["task_complete"] = True; plan["done"] = True
        progress = evaluate_task_goal_progress(state, _house_contract(), plan, CONFIG)
        self.assertFalse(progress["task_complete"])


class DynamicProtectionAndSafetyTests(unittest.TestCase):
    def test_satisfied_current_roles_become_protected(self):
        state = _house_state(); progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        protection = derive_dynamic_protection(state, _house_contract(), progress, progress["selected_role_assignment"])
        self.assertEqual(set(protection["protected_object_ids"]), {1, 2, 3})
        self.assertEqual(len(protection["protected_regions"]), 3)

    def test_protection_tracks_changed_detection_ids(self):
        state = _house_state()
        for obj in state["objects"]: obj["id"] += 10
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        protection = derive_dynamic_protection(state, _house_contract(), progress, progress["selected_role_assignment"])
        self.assertEqual(set(protection["protected_object_ids"]), {11, 12, 13})

    def test_failed_role_predicate_removes_that_role_protection(self):
        state = _house_state(); state["objects"][0]["geometry_center_m"][2] = 0.06
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)
        protection = derive_dynamic_protection(state, _house_contract(), progress, progress["selected_role_assignment"])
        roles = {item["role_id"] for item in protection["protected_roles"]}
        self.assertNotIn("left_support", roles)

    def test_execute_with_offline_scene_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            validate_execution_source(SimpleNamespace(execute=True, offline_scene_state="scene.json"))

    def test_dry_run_with_offline_scene_is_allowed(self):
        validate_execution_source(SimpleNamespace(execute=False, offline_scene_state="scene.json"))

    def test_mock_perception_execution_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            validate_execution_source(SimpleNamespace(execute=True, mock_perception=True))

    def test_semantic_workflow_cannot_bypass_offline_execution_guard(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(execute=True, offline_scene_state="scene.json", output_dir=output_dir)
            with self.assertRaises(RuntimeError):
                run_semantic_task_workflow(args)

    def test_protected_regions_are_forwarded_as_physical_objects(self):
        protection = {
            "protected_object_ids": [1],
            "protected_regions": [{"center_base_m": [0.3, 0.0, 0.02], "dimensions_m": [0.03, 0.03, 0.04]}],
        }
        state, protected_ids = _state_with_dynamic_protection(_house_state(), protection)
        self.assertIn("dynamic_protected_region_0", protected_ids)
        self.assertEqual(state["objects"][-1]["state"], "protected")

    def test_moveit_failure_is_fed_back_before_later_action_passes(self):
        proposal = {
            "action_type": "pick_place", "role_id": "left_support", "selected_object_id": 1,
            "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.02],
            "scene_revision": 4, "target_pose_base": {"position_m": [0.25, -0.08, 0.02], "yaw_rad": 0.0},
        }
        args = SimpleNamespace(max_vlm_action_attempts=2, execute=True)
        preflight_results = [
            {**adapt_task_action_to_legacy_action(proposal), "moveit_feasible": False, "moveit_preflight_error": "no IK"},
            {**adapt_task_action_to_legacy_action(proposal), "moveit_feasible": True},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            with patch("tools.workflows.stack_demo.task_workflow.call_vlm_task_policy", return_value={"decision": proposal}), patch(
                "tools.workflows.stack_demo.task_workflow._preflight_task_action", side_effect=preflight_results,
            ) as preflight:
                action, _report = _select_task_action(
                    args, _house_state(), _house_contract(), _house_plan(), {}, {}, CONFIG,
                    output_dir, 1,
                )
        self.assertTrue(action["moveit_feasible"])
        self.assertEqual(preflight.call_count, 2)


def _object(object_id, label, center, size=None):
    return {"id": object_id, "label": label, "geometry_center_m": list(center), "dimensions_m": size or [0.03, 0.03, 0.04], "geometry_frame": "base_link"}


def _house_contract():
    return {"task_type": "build_house", "goal_spec": {"roles": [
        {"role_id": "left_support", "requirements": {"category": "square_block"}},
        {"role_id": "right_support", "requirements": {"category": "square_block"}},
        {"role_id": "roof", "requirements": {"category": "rectangle_block"}},
    ], "required_relations": [
        {"type": "on_table", "subject_role": "left_support"},
        {"type": "on_table", "subject_role": "right_support"},
        {"type": "supports", "subject_role": "left_support", "object_role": "roof"},
        {"type": "supports", "subject_role": "right_support", "object_role": "roof"},
    ]}}


def _house_state():
    return {"scene_revision": 4, "table_bounds": {"xmin": 0.1, "xmax": 0.6, "ymin": -0.3, "ymax": 0.3}, "objects": [
        _object(1, "square red", [0.30, 0.0, 0.02]),
        _object(2, "square blue", [0.40, 0.0, 0.02]),
        _object(3, "rectangle yellow", [0.35, 0.0, 0.07], [0.14, 0.03, 0.06]),
    ]}


def _house_plan():
    return {"role_assignments": [
        {"role_id": "left_support", "selected_object_id": 1},
        {"role_id": "right_support", "selected_object_id": 2},
        {"role_id": "roof", "selected_object_id": 3},
    ]}


def _organize_contract():
    return {"task_type": "organize_blocks", "goal_spec": dict(CONFIG["organize_defaults"])}


def _organize_state():
    return {"scene_revision": 2, "table_bounds": {"xmin": 0.1, "xmax": 0.6, "ymin": -0.3, "ymax": 0.3}, "objects": [
        _object(1, "square red", [0.20, 0.0, 0.02]), _object(2, "square blue", [0.45, 0.0, 0.02]),
    ]}


def _organize_plan():
    return {"groups": [
        {"group_id": "red_group", "group_value": "red", "object_ids": [1], "target_region_id": "red"},
        {"group_id": "blue_group", "group_value": "blue", "object_ids": [2], "target_region_id": "blue"},
    ], "target_regions": [
        {"region_id": "red", "bounds_base_m": {"xmin": 0.15, "xmax": 0.30, "ymin": -0.05, "ymax": 0.05}},
        {"region_id": "blue", "bounds_base_m": {"xmin": 0.35, "xmax": 0.55, "ymin": -0.05, "ymax": 0.05}},
    ]}


def _organize_action():
    return {
        "action_type": "pick_place", "group_id": "red_group", "target_region_id": "red",
        "selected_object_id": 1, "object_label": "square red", "object_center_base_m": [0.20, 0.0, 0.02],
        "scene_revision": 2, "target_pose_base": {"position_m": [0.25, 0.0, 0.02], "yaw_rad": 0.0},
    }


def _clearance_state():
    state = _organize_state()
    for obj in state["objects"]: obj["pushable"] = True
    return state


if __name__ == "__main__":
    unittest.main()
