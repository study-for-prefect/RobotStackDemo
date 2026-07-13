import copy
import unittest

from robot_scene_pipeline.task_action_validation import validate_task_action
from robot_scene_pipeline.task_goal_evaluator import evaluate_task_goal_progress
from robot_scene_pipeline.task_semantic_validation import (
    TaskSemanticValidationError,
    temporary_binding_feedback,
    validate_grounded_task_plan,
    validate_task_contract,
)
from tests.house_task_fixtures import HOUSE_CONFIG, house_contract, house_plan, house_state


CONFIG = HOUSE_CONFIG


class TaskSemanticTests(unittest.TestCase):
    def test_house_contract_never_accepts_permanent_detection_id(self):
        contract = _house_contract()
        contract["goal_spec"]["roles"][0]["object_id"] = 1

        with self.assertRaises(TaskSemanticValidationError):
            validate_task_contract(contract, CONFIG)

    def test_valid_six_role_house_plan_passes(self):
        validated = validate_grounded_task_plan(_house_plan(), _house_contract(), _house_state(), 4, CONFIG)

        self.assertEqual(validated["scene_revision"], 4)
        self.assertEqual(len(validated["role_assignments"]), 6)
        self.assertTrue(all(item["assignment_status"] == "temporary" for item in validated["role_assignments"]))

    def test_duplicate_support_object_is_rejected(self):
        plan = _house_plan(); plan["role_assignments"][1]["selected_object_id"] = 1
        plan["role_assignments"][1]["observed_label"] = "square red"; plan["role_assignments"][1]["geometry_center_base_m"] = [0.30, 0.0, 0.02]

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, _house_contract(), _house_state(), 4, CONFIG)

        self.assertIn("object_assigned_to_multiple_roles", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_missing_house_role_and_cyclic_supports_are_rejected(self):
        contract = _house_contract(); contract["goal_spec"]["roles"] = contract["goal_spec"]["roles"][:2]
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_task_contract(contract, CONFIG)
        self.assertIn("missing_required_role", {item["type"] for item in raised.exception.feedback["errors"]})

        contract = _house_contract(); contract["goal_spec"]["required_relations"].append({"type": "supports", "subject_role": "triangle_top", "object_role": "roof"})
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(_house_plan(), contract, _house_state(), 4, CONFIG)
        self.assertIn("support_relation_cycle", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_model_assembly_steps_are_replaced_by_canonical_steps(self):
        plan = _house_plan(); plan["assembly_steps"][4]["prerequisites"] = ["step_03_left_upper"]
        validated = validate_grounded_task_plan(plan, _house_contract(), _house_state(), 4, CONFIG)
        self.assertEqual(validated["assembly_steps"][4]["prerequisites"], ["left_support_upper", "right_support_upper"])

    def test_roof_without_two_support_contacts_is_not_complete(self):
        from robot_scene_pipeline.orientation_fusion import fuse_house_orientation_observations
        state = _house_state(); state["objects"][4]["geometry_center_m"] = [0.30, 0.0, 0.09]
        plan = fuse_house_orientation_observations(_house_plan(), state, CONFIG)
        progress = evaluate_task_goal_progress(state, _house_contract(), plan, CONFIG)

        self.assertFalse(progress["task_complete"])
        self.assertIn("right_support_upper.supports.roof", progress["unsatisfied_predicates"])

    def test_house_id_change_and_large_motion_keep_satisfied_role_geometry(self):
        state = _house_state()
        for index, obj in enumerate(state["objects"]): obj["id"] = index + 20
        from robot_scene_pipeline.orientation_fusion import fuse_house_orientation_observations
        plan = _house_plan()
        for index, binding in enumerate(plan["role_assignments"]): binding["selected_object_id"] = index + 20
        for index, observation in enumerate(plan["orientation_observations"]): observation["selected_object_id"] = index + 24
        progress = evaluate_task_goal_progress(state, _house_contract(), fuse_house_orientation_observations(plan, state, CONFIG), CONFIG)

        self.assertTrue(progress["task_complete"])
        self.assertEqual(progress["role_observations"]["roof"]["observed_object_id"], 24)

    def test_missing_temporary_binding_returns_reselection_feedback(self):
        feedback = temporary_binding_feedback(_house_contract()["goal_spec"]["roles"][0], 1, _house_state(), 5)

        self.assertEqual(feedback["reason"], "temporary_role_assignment_invalid")
        self.assertEqual(feedback["required_response"], "select_a_current_matching_object")

    def test_organize_groups_require_unique_full_coverage(self):
        contract = _organize_contract(); plan = _organize_plan()
        validate_grounded_task_plan(plan, contract, _organize_state(), 2, CONFIG)
        bad = copy.deepcopy(plan); bad["groups"][1]["object_ids"].append(1)

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(bad, contract, _organize_state(), 2, CONFIG)

        self.assertIn("object_assigned_to_multiple_groups", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_organize_rejects_missing_object_and_overlapping_regions(self):
        plan = _organize_plan(); plan["groups"][1]["object_ids"] = []
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, _organize_contract(), _organize_state(), 2, CONFIG)
        self.assertIn("object_missing_from_groups", {item["type"] for item in raised.exception.feedback["errors"]})

        plan = _organize_plan(); plan["target_regions"][1]["bounds_base_m"] = {"xmin": 0.25, "xmax": 0.40, "ymin": -0.05, "ymax": 0.05}
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, _organize_contract(), _organize_state(), 2, CONFIG)
        self.assertIn("target_regions_overlap", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_organized_rows_complete_from_current_geometry_not_initial_ids(self):
        progress = evaluate_task_goal_progress(_organize_state(), _organize_contract(), _organize_plan(), CONFIG)

        self.assertTrue(progress["task_complete"])

    def test_organize_id_change_and_new_same_color_object_are_recomputed(self):
        state = _organize_state(); state["objects"][0]["id"] = 8; state["objects"].append(_object(9, "square red", [0.33, 0.0, 0.02]))
        progress = evaluate_task_goal_progress(state, _organize_contract(), _organize_plan(), CONFIG)

        self.assertFalse(progress["task_complete"])
        self.assertIn("red_group.all_inside", progress["unsatisfied_predicates"])

    def test_stale_pick_place_action_is_rejected_for_replanning(self):
        plan = validate_grounded_task_plan(_house_plan(), _house_contract(), _house_state(), 4, CONFIG)
        proposal = {"action_type": "pick_place", "role_id": "left_support_lower", "selected_object_id": 1, "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.02], "scene_revision": 3, "target_pose_base": {"position_m": [0.30, 0.0, 0.02], "yaw_rad": 0.0}}
        selected, report = validate_task_action(proposal, _house_state(), _house_contract(), plan, {})

        self.assertIsNone(selected)
        self.assertEqual(report["reason"], "stale_scene_revision")

    def test_current_pick_place_preserves_vlm_selected_object_and_pose(self):
        plan = validate_grounded_task_plan(_house_plan(), _house_contract(), _house_state(), 4, CONFIG)
        proposal = {"action_type": "pick_place", "role_id": "left_support_lower", "selected_object_id": 1, "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.02], "scene_revision": 4, "target_pose_base": {"position_m": [0.28, -0.08, 0.02], "yaw_rad": 0.0}, "expected_goal_predicates": ["left_support_lower.on_table"]}
        selected, report = validate_task_action(proposal, _house_state(), _house_contract(), plan, {})

        self.assertTrue(report["accepted"])
        self.assertEqual(selected["selected_object_id"], 1)
        self.assertNotIn("object_id", selected)
        self.assertEqual(selected["target_pose_base"], proposal["target_pose_base"])


def _house_contract():
    return house_contract()


def _house_state():
    return house_state()


def _house_plan():
    return house_plan()


def _organize_contract():
    return validate_task_contract({"schema_version": "task_contract_v1", "task_type": "organize_blocks", "goal_spec": {"grouping_key": "color", "layout_type": "rows", "include_scope": "all_detected_blocks", "allow_stacking": False, "minimum_spacing_m": 0.015, "alignment_tolerance_m": 0.012}, "reason": "group by color", "confidence": 0.8}, CONFIG)


def _organize_state(): return {"scene_revision": 2, "table_bounds": {"xmin": 0.1, "xmax": 0.6, "ymin": -0.3, "ymax": 0.3}, "objects": [_object(1, "square red", [0.20, 0.0, 0.02]), _object(2, "square blue", [0.45, 0.0, 0.02])]}
def _organize_plan(): return {"schema_version": "grounded_task_plan_v1", "task_type": "organize_blocks", "scene_revision": 2, "groups": [{"group_id": "red_group", "group_value": "red", "object_ids": [1], "target_region_id": "red"}, {"group_id": "blue_group", "group_value": "blue", "object_ids": [2], "target_region_id": "blue"}], "target_regions": [{"region_id": "red", "bounds_base_m": {"xmin": 0.15, "xmax": 0.30, "ymin": -0.05, "ymax": 0.05}}, {"region_id": "blue", "bounds_base_m": {"xmin": 0.35, "xmax": 0.55, "ymin": -0.05, "ymax": 0.05}}]}
def _object(object_id, label, center, size=None): return {"id": object_id, "label": label, "geometry_center_m": center, "dimensions_m": size or [0.03, 0.03, 0.04], "geometry_frame": "base_link"}


if __name__ == "__main__": unittest.main()
