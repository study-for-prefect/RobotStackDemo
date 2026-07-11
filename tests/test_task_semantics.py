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


CONFIG = {
    "organize_defaults": {"grouping_key": "color", "layout_type": "rows", "include_scope": "all_detected_blocks", "allow_stacking": False, "minimum_spacing_m": 0.015, "alignment_tolerance_m": 0.012},
    "house_semantics": {"vertical_contact_tolerance_m": 0.006, "minimum_support_overlap_ratio": 0.20, "minimum_support_separation_m": 0.025},
}


class TaskSemanticTests(unittest.TestCase):
    def test_house_contract_never_accepts_permanent_detection_id(self):
        contract = _house_contract()
        contract["goal_spec"]["roles"][0]["object_id"] = 1

        with self.assertRaises(TaskSemanticValidationError):
            validate_task_contract(contract, CONFIG)

    def test_valid_house_plan_with_two_supports_and_roof_passes(self):
        validated = validate_grounded_task_plan(_house_plan(), _house_contract(), _house_state(), 4, CONFIG)

        self.assertEqual(validated["scene_revision"], 4)
        self.assertEqual(len(validated["role_assignments"]), 3)
        self.assertTrue(all(item["assignment_status"] == "temporary" for item in validated["role_assignments"]))

    def test_duplicate_support_object_is_rejected(self):
        plan = _house_plan(); plan["role_assignments"][1]["selected_object_id"] = 1; plan["role_assignments"][1]["object_id"] = 1
        plan["role_assignments"][1]["observed_label"] = "square red"; plan["role_assignments"][1]["geometry_center_base_m"] = [0.30, 0.0, 0.02]

        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, _house_contract(), _house_state(), 4, CONFIG)

        self.assertIn("object_assigned_to_multiple_roles", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_missing_house_role_and_cyclic_supports_are_rejected(self):
        contract = _house_contract(); contract["goal_spec"]["roles"] = contract["goal_spec"]["roles"][:2]
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_task_contract(contract, CONFIG)
        self.assertIn("missing_required_role", {item["type"] for item in raised.exception.feedback["errors"]})

        contract = _house_contract(); contract["goal_spec"]["required_relations"].append({"type": "supports", "subject_role": "roof", "object_role": "left_support"})
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(_house_plan(), contract, _house_state(), 4, CONFIG)
        self.assertIn("support_relation_cycle", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_roof_step_must_depend_on_both_support_steps(self):
        plan = _house_plan(); plan["assembly_steps"][2]["prerequisites"] = ["left"]
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, _house_contract(), _house_state(), 4, CONFIG)
        self.assertIn("roof_step_missing_support_prerequisites", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_roof_without_two_support_contacts_is_not_complete(self):
        state = _house_state(); state["objects"][2]["geometry_center_m"] = [0.30, 0.0, 0.07]
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)

        self.assertFalse(progress["task_complete"])
        self.assertIn("right_support.supports.roof", progress["unsatisfied_predicates"])

    def test_house_id_change_and_large_motion_keep_satisfied_role_geometry(self):
        state = _house_state()
        for index, obj in enumerate(state["objects"]): obj["id"] = index + 20
        progress = evaluate_task_goal_progress(state, _house_contract(), _house_plan(), CONFIG)

        self.assertTrue(progress["task_complete"])
        self.assertEqual(progress["role_observations"]["roof"]["observed_object_id"], 22)

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
        proposal = {"action_type": "pick_place", "role_id": "left_support", "selected_object_id": 1, "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.02], "scene_revision": 3, "target_pose_base": {"position_m": [0.30, 0.0, 0.02], "yaw_rad": 0.0}}
        selected, report = validate_task_action(proposal, _house_state(), _house_contract(), plan, {})

        self.assertIsNone(selected)
        self.assertEqual(report["reason"], "stale_scene_revision")

    def test_current_pick_place_preserves_vlm_selected_object_and_pose(self):
        plan = validate_grounded_task_plan(_house_plan(), _house_contract(), _house_state(), 4, CONFIG)
        proposal = {"action_type": "pick_place", "role_id": "left_support", "selected_object_id": 1, "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.02], "scene_revision": 4, "target_pose_base": {"position_m": [0.28, -0.08, 0.02], "yaw_rad": 0.0}, "expected_goal_predicates": ["left_support.on_table"]}
        selected, report = validate_task_action(proposal, _house_state(), _house_contract(), plan, {})

        self.assertTrue(report["accepted"])
        self.assertEqual(selected["object_id"], 1)
        self.assertEqual(selected["target_pose_base"], proposal["target_pose_base"])


def _house_contract():
    return {"schema_version": "task_contract_v1", "task_type": "build_house", "reason": "two supports and a roof", "confidence": 0.9, "goal_spec": {"roles": [{"role_id": "left_support", "requirements": {"category": "square_block"}, "replaceable": True}, {"role_id": "right_support", "requirements": {"category": "square_block"}, "replaceable": True}, {"role_id": "roof", "requirements": {"category": "rectangle_block"}, "replaceable": True}], "required_relations": [{"type": "on_table", "subject_role": "left_support"}, {"type": "on_table", "subject_role": "right_support"}, {"type": "left_of", "subject_role": "left_support", "object_role": "right_support"}, {"type": "supports", "subject_role": "left_support", "object_role": "roof"}, {"type": "supports", "subject_role": "right_support", "object_role": "roof"}, {"type": "bridges", "subject_role": "roof", "object_roles": ["left_support", "right_support"]}]}}


def _house_state():
    return {"scene_revision": 4, "table_bounds": {"xmin": 0.1, "xmax": 0.6, "ymin": -0.3, "ymax": 0.3}, "objects": [_object(1, "square red", [0.30, 0.0, 0.02]), _object(2, "square blue", [0.40, 0.0, 0.02]), _object(3, "rectangle yellow", [0.35, 0.0, 0.07], [0.14, 0.03, 0.06])]}


def _house_plan():
    state = _house_state(); bindings = []
    for role, obj in zip(("left_support", "right_support", "roof"), state["objects"]): bindings.append({"role_id": role, "selected_object_id": obj["id"], "object_id": obj["id"], "observed_label": obj["label"], "geometry_center_base_m": obj["geometry_center_m"], "assignment_status": "temporary", "replaceable": True})
    return {"schema_version": "grounded_task_plan_v1", "task_type": "build_house", "scene_revision": 4, "role_assignments": bindings, "assembly_steps": [{"step_id": "left", "role_id": "left_support", "prerequisites": []}, {"step_id": "right", "role_id": "right_support", "prerequisites": []}, {"step_id": "roof", "role_id": "roof", "prerequisites": ["left", "right"]}]}


def _organize_contract():
    return validate_task_contract({"schema_version": "task_contract_v1", "task_type": "organize_blocks", "goal_spec": {"grouping_key": "color", "layout_type": "rows", "include_scope": "all_detected_blocks", "allow_stacking": False, "minimum_spacing_m": 0.015, "alignment_tolerance_m": 0.012}, "reason": "group by color", "confidence": 0.8}, CONFIG)


def _organize_state(): return {"scene_revision": 2, "table_bounds": {"xmin": 0.1, "xmax": 0.6, "ymin": -0.3, "ymax": 0.3}, "objects": [_object(1, "square red", [0.20, 0.0, 0.02]), _object(2, "square blue", [0.45, 0.0, 0.02])]}
def _organize_plan(): return {"schema_version": "grounded_task_plan_v1", "task_type": "organize_blocks", "scene_revision": 2, "groups": [{"group_id": "red_group", "group_value": "red", "object_ids": [1], "target_region_id": "red"}, {"group_id": "blue_group", "group_value": "blue", "object_ids": [2], "target_region_id": "blue"}], "target_regions": [{"region_id": "red", "bounds_base_m": {"xmin": 0.15, "xmax": 0.30, "ymin": -0.05, "ymax": 0.05}}, {"region_id": "blue", "bounds_base_m": {"xmin": 0.35, "xmax": 0.55, "ymin": -0.05, "ymax": 0.05}}]}
def _object(object_id, label, center, size=None): return {"id": object_id, "label": label, "geometry_center_m": center, "dimensions_m": size or [0.03, 0.03, 0.04], "geometry_frame": "base_link"}


if __name__ == "__main__": unittest.main()
