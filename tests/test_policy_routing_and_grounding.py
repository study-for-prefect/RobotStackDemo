import unittest

from robot_scene_pipeline.house_task_definition import canonical_house_assembly_steps
from robot_scene_pipeline.object_tracking import update_scene_tracks
from robot_scene_pipeline.task_routing import route_task_type
from robot_scene_pipeline.task_schemas import (
    BUILD_HOUSE_CONTRACT_SCHEMA,
    GROUNDED_HOUSE_PLAN_SCHEMA,
    GROUNDED_ORGANIZE_PLAN_SCHEMA,
    ORGANIZE_BLOCKS_CONTRACT_SCHEMA,
    TASK_ACTION_OUTPUT_SCHEMA,
    schema_for_policy,
    validate_against_schema,
)
from robot_scene_pipeline.task_semantic_validation import (
    TaskSemanticValidationError,
    infer_object_shape,
    normalize_semantic_shape,
    object_matches_role,
    validate_grounded_task_plan,
    validate_task_contract,
)
from robot_scene_pipeline.vlm_task_policy import (
    _task_prompt, build_grounded_task_plan_input, build_task_action_input,
    build_task_contract_input, compact_goal_progress,
)
from tests.test_task_semantics import CONFIG, _house_contract, _house_state


class PolicyRoutingAndGroundingTests(unittest.TestCase):
    def test_house_action_input_lists_only_roles_whose_prerequisites_are_ready(self):
        progress = compact_goal_progress({
            "task_type": "build_house",
            "satisfied_predicates": [
                "left_support_lower.on_table", "right_support_lower.on_table",
                "columns.height_aligned",
            ],
            "unsatisfied_predicates": [
                "left_support_lower.supports.left_support_upper",
                "right_support_lower.supports.right_support_upper",
                "left_column.vertical_aligned", "right_column.vertical_aligned",
                "roof.orientation_correct", "roof.supports.triangle_top",
            ],
        })
        self.assertEqual(
            progress["eligible_role_ids_for_next_action"],
            ["left_support_upper", "right_support_upper"],
        )

    def test_action_prompt_omits_combinatorial_role_assignment_audit_log(self):
        progress = {
            "task_complete": False,
            "unsatisfied_predicates": ["roof.supports.triangle_top"],
            "selected_role_assignment": {"roof": {"id": 1}},
            "role_assignment_candidates": [{"large": "x" * 100000}],
        }
        payload = build_task_action_input(
            {"objects": [], "scene_revision": 1},
            _house_contract(),
            {"task_type": "build_house"},
            progress,
            1,
        )
        self.assertNotIn("role_assignment_candidates", payload["current_goal_progress"])
        self.assertIn("selected_role_assignment", payload["current_goal_progress"])
        self.assertNotIn("orientation_results", payload["current_goal_progress"])
        self.assertNotIn("x" * 1000, _task_prompt(payload, "task_action"))

    def test_action_prompt_removes_grounded_plan_fields_duplicated_by_fused_results(self):
        payload = build_task_action_input(
            {"objects": [], "scene_revision": 1}, _house_contract(), {
                "task_type": "build_house",
                "role_assignments": [],
                "assembly_steps": [{"step_id": "fixed"}],
                "orientation_observations": [{"reason": "duplicate"}],
                "fused_orientation_results": [{"role_id": "roof"}],
                "reason": "verbose",
            }, {"unsatisfied_predicates": []}, 1,
        )
        compact = payload["grounded_task_plan"]
        self.assertNotIn("assembly_steps", compact)
        self.assertNotIn("orientation_observations", compact)
        self.assertNotIn("reason", compact)
        self.assertIn("fused_orientation_results", compact)

    def test_instruction_routes_to_one_task_family(self):
        self.assertEqual(route_task_type("按颜色整理积木"), "organize_blocks")
        self.assertEqual(route_task_type("请搭房子"), "build_house")
        self.assertEqual(route_task_type("红绿蓝黄依次向上堆叠"), "stack_blocks")
        self.assertEqual(route_task_type(
            "以红色积木为底，把绿色积木放到红色上面，再把蓝色积木放到绿色上面，"
            "再把黄色积木放到蓝色上面"
        ), "stack_blocks")

    def test_organize_contract_prompt_has_no_house_ontology(self):
        payload = build_task_contract_input({}, "按颜色整理积木", CONFIG)
        prompt = _task_prompt(payload, "task_contract")
        forbidden = (
            "HOUSE_DEFINITION_PROMPT", "two_column_two_level_roof_triangle",
            "left_support_lower", "right_support_lower", "triangle_top",
        )
        self.assertEqual(payload["expected_task_type"], "organize_blocks")
        self.assertTrue(all(token not in prompt for token in forbidden))
        self.assertNotIn("objects", payload)
        self.assertNotIn("scene_rgb", payload)

    def test_organize_grounded_prompt_has_no_house_semantics(self):
        contract = {
            "task_type": "organize_blocks", "goal_spec": {
                "grouping_key": "color", "layout_type": "rows",
                "include_scope": "all_detected_blocks", "allow_stacking": False,
            },
        }
        payload = build_grounded_task_plan_input(
            {"objects": [], "scene_revision": 1}, contract, 1, CONFIG,
        )
        prompt = _task_prompt(payload, "grounded_task_plan")
        self.assertNotIn("house_semantics", prompt)
        self.assertNotIn("left_support_lower", prompt)
        self.assertNotIn("triangle_top", prompt)
        self.assertIn("organize_plan_rules", payload)
        self.assertIn("水平带", prompt)

    def test_organize_grounded_schema_defines_groups_and_regions(self):
        group = GROUNDED_ORGANIZE_PLAN_SCHEMA["properties"]["groups"]["items"]
        region = GROUNDED_ORGANIZE_PLAN_SCHEMA["properties"]["target_regions"]["items"]
        self.assertEqual(
            set(group["required"]),
            {"group_id", "group_value", "object_ids", "target_region_id"},
        )
        self.assertEqual(
            set(region["properties"]["bounds_base_m"]["required"]),
            {"xmin", "xmax", "ymin", "ymax"},
        )

    def test_organize_prompt_offers_future_destination_row_slots(self):
        contract = {
            "task_type": "organize_blocks", "goal_spec": {
                "grouping_key": "color", "layout_type": "rows",
                "include_scope": "all_detected_blocks", "allow_stacking": False,
            },
        }
        state = {
            "workspace_bounds": {"xmin": 0.2, "xmax": 0.5, "ymin": 0.0, "ymax": 0.4},
            "objects": [
                {"id": 1, "label": "square red", "geometry_center_m": [0.3, 0.1, 0.0], "dimensions_m": [0.02, 0.02, 0.02]},
                {"id": 2, "label": "square blue", "geometry_center_m": [0.4, 0.2, 0.0], "dimensions_m": [0.02, 0.02, 0.02]},
            ],
        }
        payload = build_grounded_task_plan_input(state, contract, 1, CONFIG)
        self.assertEqual(len(payload["layout_slot_candidates"]), 2)
        self.assertGreater(
            payload["layout_slot_candidates"][0]["bounds_base_m"]["ymin"], 0.2,
        )
        self.assertEqual(
            payload["layout_slot_candidates"][-1]["bounds_base_m"]["ymax"], 0.4,
        )
        self.assertEqual(
            [slot["assigned_color"] for slot in payload["layout_slot_candidates"]],
            ["red", "blue"],
        )
        self.assertIn("未来放置区", _task_prompt(payload, "grounded_task_plan"))

    def test_organize_slots_use_observable_layout_not_full_workspace(self):
        contract = {
            "task_type": "organize_blocks", "goal_spec": {
                "grouping_key": "color", "layout_type": "rows",
                "include_scope": "all_detected_blocks", "allow_stacking": False,
            },
        }
        state = {
            "workspace_bounds": {
                "xmin": 0.235, "xmax": 0.65, "ymin": -0.10, "ymax": 0.40,
            },
            "organize_layout_bounds": {
                "xmin": 0.25, "xmax": 0.42, "ymin": 0.04, "ymax": 0.30,
            },
            "objects": [
                {"id": 1, "label": "square red", "geometry_center_m": [0.3, 0.1, 0.0], "dimensions_m": [0.02, 0.02, 0.02]},
                {"id": 2, "label": "square blue", "geometry_center_m": [0.4, 0.2, 0.0], "dimensions_m": [0.02, 0.02, 0.02]},
            ],
        }
        slots = build_grounded_task_plan_input(state, contract, 1, CONFIG)[
            "layout_slot_candidates"
        ]
        self.assertTrue(slots)
        self.assertTrue(all(slot["bounds_base_m"]["xmin"] == 0.25 for slot in slots))
        self.assertTrue(all(slot["bounds_base_m"]["xmax"] == 0.42 for slot in slots))
        self.assertGreaterEqual(slots[0]["bounds_base_m"]["ymin"], 0.20)
        self.assertEqual(slots[-1]["bounds_base_m"]["ymax"], 0.30)

    def test_task_action_output_schema_exposes_complete_clearance_parameters(self):
        required = set(TASK_ACTION_OUTPUT_SCHEMA["required"])
        self.assertTrue({
            "target_object_ref", "target_object_track_id", "target_object_label",
            "target_object_center_base_m", "contact_side", "direction_base",
            "distance_m", "gripper_yaw_rad", "safe_place_center_base_m",
        }.issubset(required))
        self.assertIn(
            "yaw_rad",
            TASK_ACTION_OUTPUT_SCHEMA["properties"]["target_pose_base"]["required"],
        )

    def test_prompt_does_not_duplicate_structured_output_schema(self):
        payload = {
            "task_contract": {"task_type": "organize_blocks"},
            "failure_history": [],
            "output_schema": {
                "type": "object",
                "properties": {"unique_schema_marker": {"type": "string"}},
            },
        }
        prompt = _task_prompt(payload, "task_action")
        self.assertNotIn("unique_schema_marker", prompt)
        self.assertNotIn('"output_schema"', prompt)

    def test_blocked_organize_grasp_prompt_requires_complete_clearance_action(self):
        contract = {
            "task_type": "organize_blocks", "goal_spec": {
                "grouping_key": "color", "layout_type": "rows",
                "include_scope": "all_detected_blocks", "allow_stacking": False,
            },
        }
        state = {
            "objects": [
                {"id": 1, "object_ref": "scene_1:obj_1", "track_id": "track_red_01", "label": "square red", "geometry_center_m": [0.30, 0.0, 0.0], "dimensions_m": [0.02, 0.02, 0.02]},
                {"id": 2, "object_ref": "scene_1:obj_2", "track_id": "track_blue_01", "label": "square blue", "geometry_center_m": [0.34, 0.0, 0.0], "dimensions_m": [0.02, 0.02, 0.02]},
            ],
        }
        failure = {
            "rejected_action": {
                "selected_object_ref": "scene_1:obj_1", "selected_track_id": "track_red_01",
                "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.0],
            },
            "failed_checks": [{
                "type": "selected_object_grasp_feasible", "grasp_feasible": False,
                "all_grasps_blocked": True,
                "blocking_objects": [{"id": 2, "label": "square blue", "blocker_category": "loose_movable"}],
            }],
        }
        payload = build_task_action_input(
            state, contract, {"task_type": "organize_blocks"}, {}, 1,
            failure_history=[failure],
        )
        prompt = _task_prompt(payload, "task_action")
        self.assertIn("禁止再次对它输出 pick_place", prompt)
        self.assertIn("target_object_*", prompt)
        self.assertIn("direction_base 为三维单位 XY 向量", prompt)
        self.assertIn("scene_1:obj_1", prompt)

    def test_bad_nudge_direction_prompt_demands_unit_vector_and_opposite_contact(self):
        contract = {"task_type": "organize_blocks", "goal_spec": {}}
        payload = build_task_action_input(
            {"objects": []}, contract, {"task_type": "organize_blocks"}, {}, 1,
            failure_history=[{
                "failed_checks": [
                    {"type": "push_direction_base_unit_xy_vector"},
                    {"type": "contact_side_matches_push_direction"},
                ],
            }],
        )
        prompt = _task_prompt(payload, "task_action")
        self.assertIn("[0,1,0]", prompt)
        self.assertIn("+Y方向用-y", prompt)

    def test_contract_schemas_are_task_specific(self):
        self.assertIs(schema_for_policy("task_contract", "build_house"), BUILD_HOUSE_CONTRACT_SCHEMA)
        self.assertIs(schema_for_policy("task_contract", "organize_blocks"), ORGANIZE_BLOCKS_CONTRACT_SCHEMA)
        self.assertEqual(BUILD_HOUSE_CONTRACT_SCHEMA["properties"]["task_type"]["const"], "build_house")
        self.assertEqual(ORGANIZE_BLOCKS_CONTRACT_SCHEMA["properties"]["task_type"]["const"], "organize_blocks")

    def test_contract_task_type_must_match_route(self):
        with self.assertRaises(TaskSemanticValidationError) as raised:
            validate_task_contract(_house_contract(), CONFIG, expected_task_type="organize_blocks")
        self.assertIn("task_type_instruction_mismatch", {item["type"] for item in raised.exception.feedback["errors"]})

    def test_validated_organize_contract_can_be_loaded_and_validated_again(self):
        raw = {
            "schema_version": "task_contract_v1", "task_type": "organize_blocks",
            "goal_spec": {
                "grouping_key": "color", "layout_type": "rows",
                "include_scope": "all_detected_blocks", "allow_stacking": False,
            },
            "reason": "group colors", "confidence": 0.9,
        }
        config = {
            **CONFIG,
            "organize_defaults": {
                **CONFIG["organize_defaults"],
                "row_color_order": ["red", "green", "blue", "yellow"],
            },
        }
        first = validate_task_contract(raw, config, expected_task_type="organize_blocks")
        second = validate_task_contract(first, config, expected_task_type="organize_blocks")
        self.assertEqual(second["goal_spec"]["row_color_order"], ["red", "green", "blue", "yellow"])

    def test_concave_aliases_normalize_and_match_roof(self):
        role = {"requirements": {"shape_any": ["concave_rectangle", "rectangle"]}}
        for label in ("concave", "concave rectangle", "concave_rectangle"):
            self.assertEqual(normalize_semantic_shape(label), "concave_rectangle")
            self.assertEqual(infer_object_shape({"label": label}), "concave_rectangle")
            self.assertTrue(object_matches_role({"label": label}, role))

    def test_concise_house_plan_resolves_refs_and_injects_fixed_steps(self):
        state = _tracked_house_state()
        plan = _concise_house_plan(state)

        validated = validate_grounded_task_plan(plan, _house_contract(), state, 4, CONFIG)

        self.assertEqual(len(validated["role_assignments"]), 6)
        self.assertEqual(validated["assembly_steps"], canonical_house_assembly_steps())
        self.assertEqual(validated["role_assignments"][4]["observed_label"], "concave")

    def test_model_cannot_declare_house_completion_or_assembly_steps(self):
        state = _tracked_house_state()
        plan = _concise_house_plan(state)
        plan["assembly_status"] = "completed"
        plan["assembly_steps"] = []
        errors = validate_against_schema(plan, GROUNDED_HOUSE_PLAN_SCHEMA)
        paths = {item["path"] for item in errors}
        self.assertIn("$.assembly_status", paths)
        self.assertIn("$.assembly_steps", paths)


def _tracked_house_state():
    state = _house_state()
    state["scene_revision"] = 4
    state["objects"][4]["label"] = "concave"
    update_scene_tracks({}, state["objects"], 4)
    return state


def _concise_house_plan(state):
    roles = (
        "left_support_lower", "right_support_lower", "left_support_upper",
        "right_support_upper", "roof", "triangle_top",
    )
    bindings = []
    for role, obj in zip(roles, state["objects"]):
        bindings.append({
            "role_id": role, "object_ref": obj["object_ref"],
            "track_id": obj["track_id"], "confidence": 0.95,
        })
    orientations = []
    for role in ("roof", "triangle_top"):
        binding = next(item for item in bindings if item["role_id"] == role)
        orientations.append({**binding, "confidence": 0.9})
    return {
        "schema_version": "grounded_house_plan_v1", "scene_revision": 4,
        "role_bindings": bindings, "orientation_observations": orientations,
        "reason": "bind current instances", "confidence": 0.9,
    }


if __name__ == "__main__":
    unittest.main()
