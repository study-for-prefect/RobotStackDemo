import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robot_scene_pipeline.house_task_definition import HOUSE_ROLE_IDS
from robot_scene_pipeline.orientation_fusion import (
    analyze_triangle_contour,
    fuse_house_orientation_observations,
    triangle_inner_angles,
)
from robot_scene_pipeline.reorientation_planner import plan_reorientation
from robot_scene_pipeline.task_action_validation import validate_task_action
from robot_scene_pipeline.task_dynamic_protection import derive_dynamic_protection
from robot_scene_pipeline.task_goal_evaluator import evaluate_task_goal_progress
from robot_scene_pipeline.task_semantic_validation import (
    TaskSemanticValidationError,
    validate_grounded_task_plan,
    validate_task_contract,
)
from robot_scene_pipeline.task_schemas import GROUNDED_HOUSE_PLAN_SCHEMA
from robot_scene_pipeline.ollama_policy_client import PolicyCallResult
from robot_scene_pipeline.vlm_task_policy import _task_prompt, build_task_contract_input, call_vlm_task_policy
from tools.workflows.stack_demo.task_execution import (
    execute_pick_place_and_reobserve,
    preflight_pick_place_action,
)

from tests.house_task_fixtures import HOUSE_CONFIG, house_contract, house_plan, house_state, task_object


class SixRoleContractTests(unittest.TestCase):
    def test_contract_requires_exact_six_roles(self):
        validated = validate_task_contract(house_contract(), HOUSE_CONFIG)
        self.assertEqual([item["role_id"] for item in validated["goal_spec"]["roles"]], list(HOUSE_ROLE_IDS))

    def test_old_three_role_house_is_rejected(self):
        contract = house_contract(); contract["goal_spec"]["roles"] = contract["goal_spec"]["roles"][:3]
        with self.assertRaises(TaskSemanticValidationError):
            validate_task_contract(contract, HOUSE_CONFIG)

    def test_wall_or_door_role_is_rejected(self):
        contract = house_contract(); contract["goal_spec"]["roles"][0]["role_id"] = "wall"
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_task_contract(contract, HOUSE_CONFIG)
        self.assertIn("unknown_house_role", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_support_role_must_require_square(self):
        contract = house_contract(); contract["goal_spec"]["roles"][0]["requirements"] = {"shape": "rectangle"}
        with self.assertRaises(TaskSemanticValidationError):
            validate_task_contract(contract, HOUSE_CONFIG)

    def test_prompt_contains_full_structure_and_all_roles(self):
        policy_input = build_task_contract_input(house_state(), "搭房子", HOUSE_CONFIG)
        prompt = _task_prompt(policy_input, "task_contract")
        self.assertIn("两个正方形作为第一层左右支撑", prompt)
        self.assertIn("triangle_top", prompt)
        self.assertIn("仅 yaw 无效", prompt)
        self.assertGreaterEqual(prompt.count("wall、door"), 1)

    def test_ollama_receives_same_grounded_house_json_schema(self):
        policy_input = {"task_contract": {"task_type": "build_house"}}
        args = Mock(no_image=True, ollama_url="http://local", model="test", num_predict=100, timeout=1)
        result = PolicyCallResult(
            transport_status="ok", generation_status="parsed", policy_kind="grounded_task_plan",
            model="test", content="{}", parsed_decision={}, schema_valid=True,
        )
        with patch("robot_scene_pipeline.vlm_task_policy.call_policy", return_value=result) as call:
            call_vlm_task_policy(args, policy_input, "grounded_task_plan")
        self.assertEqual(call.call_args.args[3], GROUNDED_HOUSE_PLAN_SCHEMA)
        self.assertEqual(call.call_args.args[2][0]["role"], "system")
        self.assertIn("triangle_top", call.call_args.args[2][0]["content"])


class GroundedHouseBindingTests(unittest.TestCase):
    def test_valid_plan_has_six_distinct_bindings(self):
        validated = validate_grounded_task_plan(house_plan(), house_contract(), house_state(), 4, HOUSE_CONFIG)
        self.assertEqual(len(validated["role_assignments"]), 6)
        self.assertEqual(len({item["selected_object_id"] for item in validated["role_assignments"]}), 6)

    def test_duplicate_id_is_rejected(self):
        plan = house_plan(); plan["role_assignments"][1]["selected_object_id"] = 1
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, house_contract(), house_state(), 4, HOUSE_CONFIG)
        self.assertIn("object_assigned_to_multiple_roles", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_triangle_must_bind_triangle(self):
        plan = house_plan(); plan["role_assignments"][-1].update({"selected_object_id": 5, "observed_label": "concave_rectangle yellow", "geometry_center_base_m": [0.35, 0.0, 0.09]})
        with self.assertRaises(TaskSemanticValidationError):
            validate_grounded_task_plan(plan, house_contract(), house_state(), 4, HOUSE_CONFIG)

    def test_concave_roof_is_preferred_when_available(self):
        state = house_state(); state["objects"].append(task_object(7, "rectangle cyan", [0.5, 0.1, 0.02], [0.14, 0.03, 0.02]))
        plan = house_plan(); plan["role_assignments"][4].update({"selected_object_id": 7, "observed_label": "rectangle cyan", "geometry_center_base_m": [0.5, 0.1, 0.02]})
        plan["orientation_observations"][0]["selected_object_id"] = 7
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(plan, house_contract(), state, 4, HOUSE_CONFIG)
        self.assertIn("concave_rectangle_must_be_preferred_for_roof", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_rectangle_roof_is_allowed_when_no_concave_exists(self):
        state = house_state(); state["objects"][4]["label"] = "rectangle yellow"
        plan = house_plan(); plan["role_assignments"][4]["observed_label"] = "rectangle yellow"; plan["orientation_observations"][0]["shape"] = "rectangle"
        validated = validate_grounded_task_plan(plan, house_contract(), state, 4, HOUSE_CONFIG)
        self.assertEqual(validated["role_assignments"][4]["selected_object_id"], 5)

    def test_resource_shortage_rejects_grounded_execution_plan(self):
        state = house_state(); state["objects"] = state["objects"][:-1]
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_grounded_task_plan(house_plan(), house_contract(), state, 4, HOUSE_CONFIG)
        self.assertIn("insufficient_role_resources", {item["type"] for item in raised.exception.feedback["errors"]})


class OrientationFusionTests(unittest.TestCase):
    def test_rectangle_upgrades_to_concave_when_vlm_and_depth_agree(self):
        state, plan = house_state(), house_plan(); state["objects"][4]["label"] = "rectangle yellow"
        fused = fuse_house_orientation_observations(plan, state, HOUSE_CONFIG)
        self.assertEqual(fused["fused_orientation_results"][0]["fused_shape"], "concave_rectangle")

    def test_front_face_correct_does_not_require_flip(self):
        fused = fuse_house_orientation_observations(house_plan(), house_state(), HOUSE_CONFIG)
        self.assertFalse(fused["fused_orientation_results"][0]["flip_required"])

    def test_back_face_requires_flip(self):
        plan = house_plan(); plan["orientation_observations"][0].update({"visible_face": "back", "flip_required": True})
        fused = fuse_house_orientation_observations(plan, house_state(), HOUSE_CONFIG)
        self.assertTrue(fused["fused_orientation_results"][0]["flip_required"])

    def test_low_confidence_requires_reobserve(self):
        plan = house_plan(); plan["orientation_observations"][0]["confidence"] = 0.2
        fused = fuse_house_orientation_observations(plan, house_state(), HOUSE_CONFIG)
        self.assertTrue(fused["fused_orientation_results"][0]["reobserve_required"])

    def test_right_angle_vertex_is_not_apex(self):
        contour = analyze_triangle_contour(house_state()["objects"][5])
        self.assertEqual(contour["right_angle_vertex"], 2)
        plan = house_plan(); plan["orientation_observations"][1].update({"apex_vertex": 2, "upward_vertex": 2})
        fused = fuse_house_orientation_observations(plan, house_state(), HOUSE_CONFIG)
        triangle = fused["fused_orientation_results"][1]
        self.assertFalse(triangle["apex_up"])
        self.assertTrue(triangle["reobserve_required"])

    def test_triangle_angles_are_computed_from_all_vertices(self):
        angles = triangle_inner_angles([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(max(angles), 90.0, places=5)
        self.assertAlmostEqual(sum(angles), 180.0, places=5)

    def test_acute_apex_up_passes(self):
        fused = fuse_house_orientation_observations(house_plan(), house_state(), HOUSE_CONFIG)
        self.assertTrue(fused["fused_orientation_results"][1]["apex_up"])

    def test_vlm_contour_conflict_requires_reobserve(self):
        plan = house_plan(); plan["orientation_observations"][1]["right_angle_vertex"] = 0
        fused = fuse_house_orientation_observations(plan, house_state(), HOUSE_CONFIG)
        self.assertIn("vlm_and_contour_right_angle_conflict", fused["fused_orientation_results"][1]["reobserve_reasons"])


class ReorientationPlannerTests(unittest.TestCase):
    def test_angle_is_computed_and_not_fixed_to_45(self):
        target = [math.sin(math.radians(30)), 0.0, 0.0, math.cos(math.radians(30))]
        plan = plan_reorientation([0, 0, 0, 1], target, [0, 0, 0, 1], [0.3, 0, 0.02], [0.35, 0, 0.09], HOUSE_CONFIG)
        self.assertAlmostEqual(plan["rotation_angle_deg"], 60.0, places=4)
        self.assertNotEqual(plan["rotation_angle_deg"], 45.0)

    def test_flip_has_roll_pitch_and_safe_height(self):
        target = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]
        plan = plan_reorientation([0, 0, 0, 1], target, [0, 0, 0, 1], [0.3, 0, 0.02], [0.35, 0, 0.09], HOUSE_CONFIG)
        self.assertTrue(plan["roll_pitch_component"])
        self.assertTrue(all(item["position_base_m"][2] >= 0.21 for item in plan["waypoints"]))

    def test_all_intermediate_poses_call_collision_checker(self):
        checker = Mock(return_value=True)
        target = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]
        plan = plan_reorientation([0, 0, 0, 1], target, [0, 0, 0, 1], [0.3, 0, 0.02], [0.35, 0, 0.09], HOUSE_CONFIG, collision_check=checker)
        self.assertEqual(checker.call_count, len(plan["waypoints"]))
        self.assertTrue(plan["all_waypoints_collision_checked"])

    def test_slerp_generates_normalized_quaternions(self):
        target = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]
        plan = plan_reorientation([0, 0, 0, 1], target, [0, 0, 0, 1], [0.3, 0, 0.02], [0.35, 0, 0.09], HOUSE_CONFIG)
        for quaternion in plan["intermediate_orientations"]:
            self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0, places=6)


class HouseActionValidationTests(unittest.TestCase):
    def test_roof_cannot_execute_before_upper_supports_complete(self):
        state, plan = house_state(), fuse_house_orientation_observations(house_plan(), house_state(), HOUSE_CONFIG)
        action, report = validate_task_action(_roof_action("pick_place"), state, house_contract(), plan, {}, {}, HOUSE_CONFIG)
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "role_prerequisites_not_satisfied")

    def test_triangle_cannot_execute_before_roof_complete(self):
        state, plan = house_state(), fuse_house_orientation_observations(house_plan(), house_state(), HOUSE_CONFIG)
        action, report = validate_task_action(_triangle_action(), state, house_contract(), plan, {}, {}, HOUSE_CONFIG)
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "role_prerequisites_not_satisfied")

    def test_back_face_requires_pick_reorient_place(self):
        state, plan, progress = _back_face_context()
        action, report = validate_task_action(_roof_action("pick_place"), state, house_contract(), plan, progress, {}, HOUSE_CONFIG)
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "flip_requires_pick_reorient_place")

    def test_back_face_reorient_action_contains_computed_roll_pitch_plan(self):
        state, plan, progress = _back_face_context()
        action, report = validate_task_action(_roof_action("pick_reorient_place"), state, house_contract(), plan, progress, {}, HOUSE_CONFIG)
        self.assertTrue(report["accepted"])
        self.assertTrue(action["reorientation_plan"]["roll_pitch_component"])
        self.assertAlmostEqual(action["reorientation_plan"]["rotation_angle_deg"], 45.0, places=4)
        self.assertTrue(action["reorientation_plan"]["perform_above_safe_height"])

    def test_yaw_only_plan_cannot_satisfy_flip(self):
        state, plan, progress = _back_face_context()
        half = math.radians(30) / 2.0
        state["objects"][4]["orientation_xyzw"] = [0.0, 0.0, math.sin(half), math.cos(half)]
        action, report = validate_task_action(_roof_action("pick_reorient_place"), state, house_contract(), plan, progress, {}, HOUSE_CONFIG)
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "flip_plan_must_include_roll_or_pitch")

    def test_correct_face_forbids_unnecessary_flip(self):
        state = house_state(); plan = fuse_house_orientation_observations(house_plan(), state, HOUSE_CONFIG)
        progress = evaluate_task_goal_progress(state, house_contract(), plan, HOUSE_CONFIG)
        action, report = validate_task_action(_roof_action("pick_reorient_place"), state, house_contract(), plan, progress, {}, HOUSE_CONFIG)
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "unnecessary_three_dimensional_reorientation")

    def test_triangle_target_orientation_is_computed_from_house_frame(self):
        state = house_state(); plan = fuse_house_orientation_observations(house_plan(), state, HOUSE_CONFIG)
        progress = evaluate_task_goal_progress(state, house_contract(), plan, HOUSE_CONFIG)
        action, report = validate_task_action(_triangle_action(), state, house_contract(), plan, progress, {}, HOUSE_CONFIG)
        self.assertTrue(report["accepted"])
        self.assertEqual(action["target_pose_base"]["orientation_source"], "house_frame_code_geometry")
        self.assertAlmostEqual(action["target_pose_base"]["orientation_xyzw"][2], math.sqrt(0.5), places=6)


class ReorientationExecutionTests(unittest.TestCase):
    def test_every_reorientation_waypoint_is_moveit_preflighted(self):
        action = _validated_reorientation_action()
        args = SimpleNamespace(execute=True)
        with patch("tools.workflows.stack_demo.task_execution.build_pick_place_plans", return_value=("pick.json", "place.json")), patch(
            "tools.workflows.stack_demo.task_execution._pick_command", return_value=["pick"],
        ), patch("tools.workflows.stack_demo.task_execution._place_command", return_value=["place"]), patch(
            "tools.workflows.stack_demo.task_execution._reorientation_waypoint_command", return_value=["waypoint"],
        ), patch("tools.workflows.stack_demo.task_execution.run") as run:
            checked = preflight_pick_place_action(args, "/tmp", house_state(), action, 1)
        self.assertEqual(run.call_count, len(action["reorientation_plan"]["waypoints"]) + 2)
        self.assertTrue(checked["reorientation_plan"]["all_waypoints_collision_checked"])

    def test_reorientation_execution_requires_fresh_post_place_observation(self):
        action = _validated_reorientation_action()
        args = SimpleNamespace(execute=True)
        observed = {"scene_revision": 5, "objects": []}
        with patch("tools.workflows.stack_demo.task_execution.build_pick_place_plans", return_value=("pick.json", "place.json")), patch(
            "tools.workflows.stack_demo.task_execution._pick_command", return_value=["pick"],
        ), patch("tools.workflows.stack_demo.task_execution._place_command", return_value=["place"]), patch(
            "tools.workflows.stack_demo.task_execution._reorientation_waypoint_command", return_value=["waypoint"],
        ), patch("tools.workflows.stack_demo.task_execution.run"), patch(
            "tools.workflows.stack_demo.task_execution.capture_empty_observation", return_value=observed,
        ) as capture:
            next_state, result = execute_pick_place_and_reobserve(args, "/tmp", {}, house_state(), action, 1)
        self.assertIs(next_state, observed)
        capture.assert_called_once()
        self.assertEqual(result["post_place_orientation_result"], "pending_fresh_observation")


class SixRoleGoalTests(unittest.TestCase):
    def test_complete_six_role_house_passes_all_predicates(self):
        plan = fuse_house_orientation_observations(house_plan(), house_state(), HOUSE_CONFIG)
        progress = evaluate_task_goal_progress(house_state(), house_contract(), plan, HOUSE_CONFIG)
        self.assertTrue(progress["task_complete"])

    def test_missing_triangle_is_insufficient_resources(self):
        state = house_state(); state["objects"] = state["objects"][:-1]
        plan = fuse_house_orientation_observations(house_plan(), state, HOUSE_CONFIG)
        progress = evaluate_task_goal_progress(state, house_contract(), plan, HOUSE_CONFIG)
        self.assertFalse(progress["task_complete"])
        self.assertIn({"type": "insufficient_role_resources"}, progress["scene_events"])

    def test_triangle_right_angle_up_prevents_completion(self):
        plan = house_plan(); plan["orientation_observations"][1].update({"apex_vertex": 2, "upward_vertex": 2})
        fused = fuse_house_orientation_observations(plan, house_state(), HOUSE_CONFIG)
        progress = evaluate_task_goal_progress(house_state(), house_contract(), fused, HOUSE_CONFIG)
        self.assertIn("triangle_top.apex_up", progress["unsatisfied_predicates"])

    def test_wrong_roof_face_unprotects_roof_and_triangle(self):
        plan = house_plan(); plan["orientation_observations"][0].update({"visible_face": "back", "flip_required": True})
        fused = fuse_house_orientation_observations(plan, house_state(), HOUSE_CONFIG)
        progress = evaluate_task_goal_progress(house_state(), house_contract(), fused, HOUSE_CONFIG)
        protection = derive_dynamic_protection(house_state(), house_contract(), progress, progress["selected_role_assignment"])
        protected_roles = {item["role_id"] for item in protection["protected_roles"]}
        self.assertNotIn("roof", protected_roles)
        self.assertNotIn("triangle_top", protected_roles)


def _roof_action(action_type):
    return {
        "action_type": action_type, "role_id": "roof", "selected_object_id": 5,
        "object_label": "concave_rectangle yellow", "object_center_base_m": [0.35, 0.0, 0.09],
        "scene_revision": 4, "target_pose_base": {"position_m": [0.35, 0.0, 0.09]}, "reason": "place roof",
    }


def _triangle_action():
    return {
        "action_type": "pick_place", "role_id": "triangle_top", "selected_object_id": 6,
        "object_label": "triangle purple", "object_center_base_m": [0.35, 0.0, 0.12],
        "scene_revision": 4, "target_pose_base": {"position_m": [0.35, 0.0, 0.12]}, "reason": "place triangle",
    }


def _back_face_context():
    state, raw_plan = house_state(), house_plan()
    angle = math.radians(45.0) / 2.0
    state["objects"][4]["orientation_xyzw"] = [math.sin(angle), 0.0, 0.0, math.cos(angle)]
    raw_plan["orientation_observations"][0].update({"visible_face": "back", "flip_required": True})
    plan = fuse_house_orientation_observations(raw_plan, state, HOUSE_CONFIG)
    progress = evaluate_task_goal_progress(state, house_contract(), plan, HOUSE_CONFIG)
    progress["satisfied_predicates"] = list(set(progress["satisfied_predicates"]) | {
        "left_support_lower.supports.left_support_upper", "right_support_lower.supports.right_support_upper",
        "left_column.vertical_aligned", "right_column.vertical_aligned", "columns.height_aligned",
    })
    return state, plan, progress


def _validated_reorientation_action():
    state, plan, progress = _back_face_context()
    action, report = validate_task_action(
        _roof_action("pick_reorient_place"), state, house_contract(), plan, progress, {}, HOUSE_CONFIG,
    )
    if not report.get("accepted"):
        raise AssertionError(report)
    return action


if __name__ == "__main__":
    unittest.main()
