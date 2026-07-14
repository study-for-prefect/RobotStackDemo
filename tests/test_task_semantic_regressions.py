import copy
import json
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
from robot_scene_pipeline.grasp_yaw_search import select_best_grasp
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
from robot_scene_pipeline.orientation_fusion import fuse_house_orientation_observations
from robot_scene_pipeline.vlm_action_validation import validate_vlm_action_decision
from robot_scene_pipeline.vlm_task_policy import _task_action_recovery_directive
from robot_scene_pipeline.vlm_replanning import action_validation_feedback
from tools.workflows.stack_demo.execution_safety import validate_execution_source
from tools.workflows.stack_demo.task_execution import (
    _validate_pick_grasp_geometry,
    preflight_pick_place_action,
)
from tools.workflows.stack_demo.task_workflow import (
    _execute_task_action,
    _reobserve,
    _resolve_task_action_reference,
    _select_task_action,
    _state_with_dynamic_protection,
    _task_nudge_semantic_error,
    _deduplicate_perception_state,
    _expected_state_after_action,
    _recover_trusted_executed_object,
    _normalize_task_action_protocol,
    _retry_geometry_checked_organize_target,
    _normalize_organize_selected_group,
    _normalize_organize_grounded_plan,
    _normalize_organize_grasp_first_proposal,
    _task_action_priority_error,
    _repeats_physically_failed_strategy,
    run_semantic_task_workflow,
)
from tests.house_task_fixtures import HOUSE_CONFIG, house_contract, house_plan, house_state


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
CONFIG["house_semantics"] = copy.deepcopy(HOUSE_CONFIG["house_semantics"])


class ActionSchemaTests(unittest.TestCase):
    def test_grasp_feedback_omits_large_candidate_diagnostics(self):
        report = {
            "reason": "selected_pick_not_grasp_feasible",
            "failed_fields": ["selected_object_grasp_feasible"],
            "checks": {"selected_object_grasp_feasible": {"detail": {
                "grasp_feasible": False,
                "all_grasps_blocked": True,
                "blocking_objects": [{"id": 2, "label": "square blue"}],
                "candidate_results": [{"yaw_deg": value} for value in range(180)],
                "blocking_objects_by_interval": [{"interval_deg": [0, 180]}],
                "required_next_action": "choose clearance",
            }}},
        }
        feedback = action_validation_feedback(
            {"selected_object_id": 1, "action_type": "pick_place"},
            report, 1, 1,
        )
        detail = feedback["failed_checks"][0]
        self.assertNotIn("candidate_results", detail)
        self.assertNotIn("blocking_objects_by_interval", detail)
        self.assertEqual(detail["blocking_objects"][0]["id"], 2)

    def test_physical_strategy_failure_allows_new_object_or_clearance(self):
        replanning = {
            "hard_constraints": {"required_strategy_change": True},
            "failed_actions": [{
                "strategy_id": "organize_blocks",
                "action_type": "pick_place",
                "operated_object": "track_red_01",
                "reason_codes": ["selected_pick_not_grasp_feasible"],
            }],
        }
        same_pick = {
            "strategy_id": "organize_blocks", "action_type": "pick_place",
        }
        self.assertTrue(_repeats_physically_failed_strategy(
            same_pick, {"operated_track_id": "track_red_01"}, replanning,
        ))
        self.assertFalse(_repeats_physically_failed_strategy(
            same_pick, {"operated_track_id": "track_blue_01"}, replanning,
        ))
        self.assertFalse(_repeats_physically_failed_strategy(
            {"strategy_id": "organize_blocks", "action_type": "nudge"},
            {"operated_track_id": "track_red_01"}, replanning,
        ))

    def test_physical_nudge_failure_allows_opposite_push_direction(self):
        failed_fingerprint = {
            "strategy_id": "clear_blocker_by_nudge",
            "action_type": "nudge",
            "operated_track_id": "track_yellow_01",
            "direction_bin": "+Y",
            "distance_bin_m": 0.04,
            "grasp_yaw_bin_deg": 0.0,
        }
        replanning = {
            "hard_constraints": {"required_strategy_change": True},
            "failed_actions": [{
                "fingerprint": json.dumps(failed_fingerprint),
                "strategy_id": "clear_blocker_by_nudge",
                "action_type": "nudge",
                "operated_object": "track_yellow_01",
                "reason_codes": ["tool_swept_volume_rejected"],
            }],
        }
        proposal = {
            "strategy_id": "clear_blocker_by_nudge", "action_type": "nudge",
        }

        self.assertTrue(_repeats_physically_failed_strategy(
            proposal,
            dict(failed_fingerprint, fingerprint=json.dumps(failed_fingerprint)),
            replanning,
        ))
        opposite = dict(
            failed_fingerprint,
            direction_bin="-Y",
            fingerprint="opposite-direction-fingerprint",
        )
        self.assertFalse(_repeats_physically_failed_strategy(
            proposal, opposite, replanning,
        ))

    def test_organize_grounded_normalization_removes_absent_color_and_duplicate_assignment(self):
        policy_input = {
            "objects": [
                {"detector_object_id": 0, "label": "square blue", "visual_color": "blue"},
                {"detector_object_id": 1, "label": "semi circle", "visual_color": "yellow"},
                {"detector_object_id": 6, "label": "square red", "visual_color": "red"},
            ],
            "layout_slot_candidates": [
                {"assigned_color": "red", "bounds_base_m": {"xmin": .25, "xmax": .42, "ymin": .20, "ymax": .23}},
                {"assigned_color": "blue", "bounds_base_m": {"xmin": .25, "xmax": .42, "ymin": .23, "ymax": .26}},
                {"assigned_color": "yellow", "bounds_base_m": {"xmin": .25, "xmax": .42, "ymin": .26, "ymax": .29}},
            ],
        }
        hallucinated = {
            "schema_version": "grounded_task_plan_v1", "task_type": "organize_blocks",
            "scene_revision": 99,
            "groups": [
                {"group_id": "group_green", "group_value": "green", "object_ids": [1], "target_region_id": "green"},
                {"group_id": "group_yellow", "group_value": "yellow", "object_ids": [1], "target_region_id": "yellow"},
            ],
            "target_regions": [],
        }

        normalized, report = _normalize_organize_grounded_plan(
            hallucinated, policy_input, 3,
        )

        self.assertIsNotNone(report)
        self.assertEqual(normalized["scene_revision"], 3)
        self.assertEqual(
            [group["group_value"] for group in normalized["groups"]],
            ["red", "blue", "yellow"],
        )
        self.assertEqual(
            [object_id for group in normalized["groups"] for object_id in group["object_ids"]],
            [6, 0, 1],
        )
        self.assertEqual(len({region["region_id"] for region in normalized["target_regions"]}), 3)

    def test_organize_grounded_normalization_preserves_committed_row_bounds(self):
        policy_input = {
            "objects": [{
                "detector_object_id": 7, "label": "square red", "visual_color": "red",
            }],
            "layout_slot_candidates": [{
                "assigned_color": "red",
                "bounds_base_m": {"xmin": .25, "xmax": .42, "ymin": .22, "ymax": .25},
            }],
        }
        previous = {
            "task_type": "organize_blocks",
            "groups": [{
                "group_id": "group_red", "group_value": "red",
                "target_region_id": "target_region_red", "object_ids": [2],
            }],
            "target_regions": [{
                "region_id": "target_region_red",
                "bounds_base_m": {"xmin": .25, "xmax": .42, "ymin": .17, "ymax": .20},
            }],
        }

        normalized, report = _normalize_organize_grounded_plan(
            {}, policy_input, 4, previous,
        )

        self.assertTrue(report["preserved_previous_target_regions"])
        self.assertEqual(
            normalized["target_regions"][0]["bounds_base_m"],
            previous["target_regions"][0]["bounds_base_m"],
        )

    def test_organize_bad_model_choice_is_rebound_to_feasible_grasp(self):
        state = {
            "scene_revision": 3,
            "objects": [
                {
                    "id": 1, "object_ref": "scene_3:obj_1", "track_id": "track_blue_01",
                    "label": "square blue", "geometry_center_m": [.30, .10, 0.0],
                },
                {
                    "id": 3, "object_ref": "scene_3:obj_3", "track_id": "track_red_01",
                    "label": "square red", "geometry_center_m": [.38, .18, 0.0],
                },
            ],
        }
        physical = {
            "direct_grasp_available": True,
            "candidates": [
                {"selected_object_id": 1, "grasp_feasible": False},
                {
                    "selected_object_id": 3, "selected_object_ref": "scene_3:obj_3",
                    "selected_track_id": "track_red_01", "grasp_feasible": True,
                },
            ],
        }

        normalized, changed = _normalize_organize_grasp_first_proposal(
            {"action_type": "stop", "reason": "model gave up"},
            "organize_blocks", state, physical,
        )

        self.assertEqual(normalized["action_type"], "pick_place")
        self.assertEqual(normalized["selected_object_id"], 3)
        self.assertEqual(normalized["target_pose_base"]["position_m"], [.38, .18, 0.0])
        self.assertIn("selected_object_id", changed)

    def test_organize_graspable_object_forbids_clearance_and_blocked_pick(self):
        physical = {
            "candidates": [
                {"selected_object_id": 1, "selected_object_ref": "scene_1:obj_1", "grasp_feasible": False},
                {"selected_object_id": 3, "selected_object_ref": "scene_1:obj_3", "grasp_feasible": True},
            ],
        }
        self.assertEqual(
            _task_action_priority_error(
                {"action_type": "nudge", "selected_object_id": 1},
                "organize_blocks", physical,
            ),
            "direct_grasp_available_before_clearance",
        )
        self.assertEqual(
            _task_action_priority_error(
                {"action_type": "pick_place", "selected_object_id": 1},
                "organize_blocks", physical,
            ),
            "selected_object_not_grasp_feasible_while_alternative_available",
        )
        self.assertIsNone(_task_action_priority_error(
            {"action_type": "pick_place", "selected_object_ref": "scene_1:obj_3"},
            "organize_blocks", physical,
        ))

    def test_all_current_picks_blocked_authorizes_clearance_without_failed_pick_roundtrip(self):
        proposal = {
            "action_type": "nudge",
            "object_center_base_m": [0.30, 0.10, 0.0],
            "target_object_center_base_m": [0.34, 0.10, 0.0],
            "direction_base": [-1.0, 0.0, 0.0],
            "distance_m": 0.04,
        }
        physical = {
            "direct_grasp_available": False,
            "candidates": [{"selected_object_id": 1, "grasp_feasible": False}],
        }
        self.assertIsNone(_task_nudge_semantic_error(
            proposal,
            {"task_type": "organize_blocks"},
            [],
            {"target_regions": []},
            physical,
        ))

    def test_organize_pick_place_drops_house_and_clearance_union_fields(self):
        proposal = {
            "strategy_id": "pick_place", "action_type": "pick_place",
            "selected_object_ref": "scene_2:obj_6", "role_id": "roof",
            "scene_revision": 3,
            "target_object_ref": "scene_2:obj_3", "target_object_id": 3,
            "contact_side": "+y", "direction_base": [0.0, 0.0, 0.0],
            "distance_m": 0.01, "gripper_yaw_rad": 0.0,
            "safe_place_center_base_m": [0.39, 0.09, 0.0],
            "target_pose_base": {
                "position_m": [0.263, 0.091, 0.0], "yaw_rad": 0.0,
            },
        }
        normalized, changed = _normalize_task_action_protocol(
            proposal, "organize_blocks", 2,
        )
        self.assertEqual(normalized["strategy_id"], "organize_blocks")
        self.assertEqual(normalized["scene_revision"], 2)
        self.assertIn("scene_revision", changed)
        self.assertEqual(
            normalized["target_pose_base"]["position_m"], [0.263, 0.091, 0.0],
        )
        for key in (
            "role_id", "target_object_ref", "target_object_id", "contact_side",
            "direction_base", "distance_m", "gripper_yaw_rad",
            "safe_place_center_base_m",
        ):
            self.assertNotIn(key, normalized)
            self.assertIn(key, changed)

    def test_protocol_normalization_does_not_rebind_stale_object_ref(self):
        normalized, changed = _normalize_task_action_protocol({
            "strategy_id": "organize_blocks", "action_type": "pick_place",
            "selected_object_ref": "scene_1:obj_6", "scene_revision": 1,
        }, "organize_blocks", 2)
        self.assertEqual(normalized["scene_revision"], 1)
        self.assertNotIn("scene_revision", changed)

    def test_geometry_checked_organize_suggestion_is_revalidated_immediately(self):
        proposal = {
            "action_type": "pick_place",
            "target_pose_base": {"position_m": [0.39, 0.09, 0.0], "yaw_rad": 0.0},
        }
        report = {"checks": {"target_pose_base": {"detail": {
            "suggested_collision_free_position_m": [0.2635, 0.0907, 0.0],
        }}}}
        accepted = {"action_type": "pick_place", "moveit_feasible": False}
        with patch(
            "tools.workflows.stack_demo.task_workflow.validate_task_action",
            return_value=(accepted, {"accepted": True}),
        ) as validate:
            action, corrected, corrected_report = _retry_geometry_checked_organize_target(
                proposal, report, {}, {"task_type": "organize_blocks"}, {}, {}, {}, {},
            )
        self.assertIs(action, accepted)
        self.assertEqual(
            corrected["target_pose_base"]["position_m"], [0.2635, 0.0907, 0.0],
        )
        self.assertTrue(corrected_report["automatic_target_correction"]["revalidated"])
        self.assertEqual(
            validate.call_args.args[0]["target_pose_base"]["position_m"],
            [0.2635, 0.0907, 0.0],
        )

    def test_organize_region_midpoint_suggestion_is_revalidated_immediately(self):
        proposal = {
            "action_type": "pick_place",
            "target_pose_base": {"position_m": [0.335, 0.17, 0.0], "yaw_rad": 0.0},
        }
        report = {"checks": {"target_pose_base": {"detail": {
            "suggested_interval_midpoint_position_m": [0.335, 0.2025, 0.0],
        }}}}
        accepted = {"action_type": "pick_place"}
        with patch(
            "tools.workflows.stack_demo.task_workflow.validate_task_action",
            return_value=(accepted, {"accepted": True}),
        ) as validate:
            action, corrected, corrected_report = _retry_geometry_checked_organize_target(
                proposal, report, {}, {"task_type": "organize_blocks"}, {}, {}, {}, {},
            )
        self.assertIs(action, accepted)
        self.assertEqual(
            corrected["target_pose_base"]["position_m"], [0.335, 0.2025, 0.0],
        )
        self.assertEqual(
            corrected_report["automatic_target_correction"]["source"],
            "validator_suggested_interval_midpoint_position_m",
        )
        self.assertEqual(validate.call_count, 1)

    def test_selected_color_corrects_stale_organize_group_and_region(self):
        state = {"objects": [{
            "id": 1, "label": "square blue", "visual_color": "blue",
        }]}
        plan = {"groups": [
            {"group_id": "group_red", "group_value": "red", "target_region_id": "red"},
            {"group_id": "group_blue", "group_value": "blue", "target_region_id": "blue"},
        ]}
        normalized, changed = _normalize_organize_selected_group({
            "action_type": "pick_place", "selected_object_id": 1,
            "group_id": "group_red", "target_region_id": "red",
        }, "organize_blocks", state, plan)
        self.assertEqual(normalized["group_id"], "group_blue")
        self.assertEqual(normalized["target_region_id"], "blue")
        self.assertEqual(changed, ["group_id", "target_region_id"])

    def test_expected_post_action_state_moves_only_selected_object(self):
        state = {"objects": [
            _object(1, "square red", [0.30, 0.14, -0.002]),
            _object(2, "square blue", [0.38, 0.10, -0.002]),
        ]}
        expected = _expected_state_after_action(state, {
            "action_type": "pick_place", "selected_object_id": 1,
            "target_pose_base": {"position_m": [0.31, 0.02, -0.002]},
        })
        self.assertEqual(expected["objects"][0]["geometry_center_m"], [0.31, 0.02, -0.002])
        self.assertEqual(expected["objects"][1]["geometry_center_m"], [0.38, 0.10, -0.002])
        self.assertEqual(state["objects"][0]["geometry_center_m"], [0.30, 0.14, -0.002])

    def test_trusted_resume_restores_only_executed_selected_object(self):
        reference = {"objects": [
            _object(4, "square red", [0.28, 0.10, -0.002]),
            _object(5, "square green", [0.40, 0.20, -0.002]),
        ]}
        action = {
            "action_type": "pick_place", "selected_object_id": 4,
            "target_pose_base": {"position_m": [0.44, -0.04, -0.002]},
        }
        expected = _expected_state_after_action(reference, action)
        current = {"objects": [_object(1, "square blue", [0.34, 0.17, -0.002])]}
        recovered, report = _recover_trusted_executed_object(
            reference, expected, action, {"status": "executed_and_reobserved"}, current,
        )
        self.assertEqual(len(recovered["objects"]), 2)
        restored = recovered["objects"][-1]
        self.assertEqual(restored["geometry_center_m"], [0.44, -0.04, -0.002])
        self.assertEqual(restored["id"], 2)
        self.assertTrue(restored["trusted_executed_recovery"])
        self.assertFalse(restored["visible"])
        self.assertNotIn("object_ref", restored)
        self.assertEqual(len(report["recovered"]), 1)
        self.assertFalse(any(obj.get("label") == "square green" for obj in recovered["objects"]))

    def test_trusted_resume_refuses_when_selected_color_remains_at_source(self):
        reference = {"objects": [_object(4, "square red", [0.28, 0.10, -0.002])]}
        action = {
            "action_type": "pick_place", "selected_object_id": 4,
            "target_pose_base": {"position_m": [0.44, -0.04, -0.002]},
        }
        expected = _expected_state_after_action(reference, action)
        current = {"objects": [_object(9, "square red", [0.281, 0.101, -0.002])]}
        with self.assertRaisesRegex(
            RuntimeError, "TRUSTED_RESUME_ACTION_NOT_REFLECTED_AT_SOURCE",
        ):
            _recover_trusted_executed_object(
                reference, expected, action,
                {"status": "executed_and_reobserved"}, current,
            )

    def test_premature_organize_clearance_feedback_requires_pick_place(self):
        directive = _task_action_recovery_directive({"failure_history": [{
            "failed_checks": [{
                "type": "organize_clearance_requires_physical_grasp_blockage",
            }],
        }]})
        self.assertIn("必须改为 pick_place", directive)
        self.assertIn("禁止继续 nudge", directive)

    def test_blocked_organize_pick_immediately_directs_clearance(self):
        directive = _task_action_recovery_directive({
            "current_goal_progress": {
                "group_diagnostics": [{"outside_region": [1, 2]}],
            },
            "objects": [
                {"detector_object_id": 1, "object_ref": "scene_1:obj_1", "visual_color": "red"},
                {"detector_object_id": 2, "object_ref": "scene_1:obj_2", "visual_color": "blue"},
            ],
            "failure_history": [{
                "rejected_action": {
                    "selected_object_id": 1,
                    "selected_object_ref": "scene_1:obj_1",
                },
                "failed_checks": [{
                    "type": "selected_object_grasp_feasible", "all_grasps_blocked": True,
                    "blocking_objects": [{"id": 3}],
                }],
            }],
        })
        self.assertIn("立即清障", directive)
        self.assertIn("0.03,0.05", directive)
        self.assertIn("recoverable contact", directive)

    def test_task_state_removes_only_near_identical_3d_duplicates(self):
        duplicate = _object(2, "semi circle", [0.3415, 0.0869, -0.0022])
        duplicate.update({"visual_color": "yellow", "confidence": 0.75})
        preferred = _object(1, "semi circle", [0.3415, 0.0869, -0.0022])
        preferred.update({"visual_color": "yellow", "confidence": 0.84})
        adjacent = _object(3, "semi circle", [0.3655, 0.0869, -0.0022])
        adjacent.update({"visual_color": "yellow", "confidence": 0.90})
        state = _deduplicate_perception_state({"objects": [duplicate, preferred, adjacent]})
        self.assertEqual(len(state["objects"]), 2)
        kept = next(item for item in state["objects"] if item.get("merged_duplicate_ids"))
        self.assertEqual(kept["id"], 1)
        self.assertIn(2, kept["merged_duplicate_ids"])

    def test_clearance_target_reference_is_resolved_with_operated_reference(self):
        state = {
            "scene_revision": 2,
            "objects": [
                {"id": 1, "object_ref": "scene_2:obj_1", "track_id": "track_blue_01"},
                {"id": 2, "object_ref": "scene_2:obj_2", "track_id": "track_red_01"},
            ],
        }
        resolved = _resolve_task_action_reference({
            "action_type": "nudge", "scene_revision": 2,
            "selected_object_ref": "scene_2:obj_1", "selected_track_id": "track_blue_01",
            "target_object_ref": "scene_2:obj_2", "target_object_track_id": "track_red_01",
        }, state)
        self.assertEqual(resolved["selected_object_id"], 1)
        self.assertEqual(resolved["target_object_id"], 2)
        self.assertEqual(resolved["target_object_ref"], "scene_2:obj_2")

    def test_nudge_contact_side_is_derived_opposite_to_unit_direction(self):
        state = {
            "scene_revision": 2,
            "objects": [
                {"id": 1, "object_ref": "scene_2:obj_1", "track_id": "track_blue_01"},
                {"id": 2, "object_ref": "scene_2:obj_2", "track_id": "track_red_01"},
            ],
        }
        resolved = _resolve_task_action_reference({
            "action_type": "nudge", "scene_revision": 2,
            "selected_object_ref": "scene_2:obj_1", "target_object_ref": "scene_2:obj_2",
            "direction_base": [0.0, -1.0, 0.0], "contact_side": "-y",
        }, state)
        self.assertEqual(resolved["contact_side"], "+y")
        self.assertEqual(resolved["model_reported_contact_side"], "-y")

    def test_nudge_nonunit_xy_direction_is_normalized_without_changing_sign(self):
        state = {
            "scene_revision": 2,
            "objects": [
                {"id": 1, "object_ref": "scene_2:obj_1", "track_id": "track_blue_01"},
                {"id": 2, "object_ref": "scene_2:obj_2", "track_id": "track_red_01"},
            ],
        }
        resolved = _resolve_task_action_reference({
            "action_type": "nudge", "scene_revision": 2,
            "selected_object_ref": "scene_2:obj_1", "target_object_ref": "scene_2:obj_2",
            "direction_base": [0.0, -0.015, 0.0], "contact_side": "-y",
        }, state)
        self.assertEqual(resolved["direction_base"], [0.0, -1.0, 0.0])
        self.assertEqual(resolved["contact_side"], "+y")
        self.assertEqual(resolved["model_reported_direction_base"], [0.0, -0.015, 0.0])

    def test_organize_clearance_requires_reported_blocker_and_blocked_target(self):
        progress = {"task_type": "organize_blocks"}
        proposal = {
            "action_type": "nudge", "selected_object_id": 2,
            "target_object_ref": "scene_2:obj_1",
        }
        self.assertEqual(
            _task_nudge_semantic_error(proposal, progress, []),
            "organize_clearance_requires_physical_grasp_blockage",
        )
        history = [{
            "rejected_action": {"selected_object_ref": "scene_2:obj_1"},
            "failed_checks": [{
                "type": "selected_object_grasp_feasible", "all_grasps_blocked": True,
                "blocking_objects": [{"id": 2, "blocker_category": "loose_movable"}],
            }],
        }]
        self.assertIsNone(_task_nudge_semantic_error(proposal, progress, history))
        self.assertIsNone(
            _task_nudge_semantic_error({**proposal, "selected_object_id": 3}, progress, history),
        )

    def test_organize_allows_clearance_after_one_blocked_grasp(self):
        progress = {
            "task_type": "organize_blocks",
            "group_diagnostics": [{"outside_region": [1, 2]}],
        }
        history = [{
            "rejected_action": {
                "selected_object_id": 1,
                "selected_object_ref": "scene_1:obj_1",
            },
            "failed_checks": [{
                "type": "selected_object_grasp_feasible", "all_grasps_blocked": True,
                "blocking_objects": [{"id": 3, "blocker_category": "loose_movable"}],
            }],
        }]
        proposal = {
            "action_type": "nudge", "selected_object_id": 3,
            "target_object_ref": "scene_1:obj_1",
        }
        self.assertIsNone(_task_nudge_semantic_error(proposal, progress, history))

    def test_organize_nudge_end_avoids_final_target_regions(self):
        progress = {"task_type": "organize_blocks"}
        history = [{
            "rejected_action": {"selected_object_ref": "scene_1:obj_1"},
            "failed_checks": [{
                "type": "selected_object_grasp_feasible", "all_grasps_blocked": True,
                "blocking_objects": [{"id": 3}],
            }],
        }]
        proposal = {
            "action_type": "nudge", "selected_object_id": 3,
            "target_object_ref": "scene_1:obj_1",
            "object_center_base_m": [0.30, 0.10, 0.0],
            "target_object_center_base_m": [0.35, 0.10, 0.0],
            "direction_base": [-1.0, 0.0, 0.0], "distance_m": 0.04,
        }
        plan = {"target_regions": [{
            "bounds_base_m": {"xmin": 0.25, "xmax": 0.28, "ymin": 0.08, "ymax": 0.12},
        }]}
        self.assertEqual(
            _task_nudge_semantic_error(proposal, progress, history, plan),
            "organize_clearance_avoid_target_regions",
        )

    def test_organize_all_outside_blocked_allows_reported_clearance(self):
        progress = {
            "task_type": "organize_blocks",
            "group_diagnostics": [{"outside_region": [1, 2]}],
        }
        history = [
            {
                "rejected_action": {
                    "selected_object_id": object_id,
                    "selected_object_ref": "scene_1:obj_{}".format(object_id),
                },
                "failed_checks": [{
                    "type": "selected_object_grasp_feasible", "all_grasps_blocked": True,
                    "blocking_objects": ([{"id": 3, "blocker_category": "loose_movable"}] if object_id == 2 else []),
                }],
            }
            for object_id in (1, 2)
        ]
        proposal = {
            "action_type": "pick_away", "selected_object_id": 3,
            "target_object_ref": "scene_1:obj_2",
        }
        self.assertIsNone(_task_nudge_semantic_error(proposal, progress, history))

    def test_organize_nudge_must_move_blocker_away_from_blocked_target(self):
        progress = {"task_type": "organize_blocks"}
        history = [{
            "rejected_action": {"selected_object_ref": "scene_1:obj_3"},
            "failed_checks": [{
                "type": "selected_object_grasp_feasible", "all_grasps_blocked": True,
                "blocking_objects": [{"id": 4, "blocker_category": "loose_movable"}],
            }],
        }]
        proposal = {
            "action_type": "nudge", "selected_object_id": 4,
            "target_object_ref": "scene_1:obj_3",
            "object_center_base_m": [0.3644, 0.1245, -0.001],
            "target_object_center_base_m": [0.3811, 0.0913, -0.0015],
            "direction_base": [0.0, -1.0, 0.0], "distance_m": 0.03,
        }
        self.assertEqual(
            _task_nudge_semantic_error(proposal, progress, history),
            "organize_clearance_must_increase_target_separation",
        )
        self.assertIsNone(_task_nudge_semantic_error(
            {**proposal, "direction_base": [0.0, 1.0, 0.0]}, progress, history,
        ))

    def test_pick_action_schema_requires_target_pose(self):
        proposal = _organize_action()
        proposal.pop("target_pose_base")
        _action, report = validate_task_action(
            proposal, _organize_state(), _organize_contract(), _organize_plan(), {},
        )
        self.assertEqual(report["reason"], "task_action_json_schema_invalid")
        self.assertTrue(any(
            error["type"] == "json_schema_one_of_mismatch"
            for error in report["schema_errors"]
        ))

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
    def test_real_scene_narrow_grasp_interval_requires_clearance(self):
        objects = [
            _object(0, "square yellow", [.3059, .1254, -.0016], [.0235, .0230, .0231]),
            _object(2, "square red", [.3376, .2193, -.0002], [.0233, .0209, .0210]),
            _object(3, "square blue", [.3393, .1405, -.0011], [.0225, .0217, .0249]),
            _object(4, "square blue", [.2993, .0952, -.0014], [.0237, .0220, .0239]),
            _object(5, "square red", [.3086, .0496, -.0021], [.0237, .0226, .0256]),
        ]
        for obj, yaw in zip(objects, (-56.93, 89.83, -0.49, 39.85, 0.0)):
            obj["table_yaw_deg"] = yaw

        permissive = select_best_grasp(
            objects[2], objects, gripper_inner_width_m=.049,
            min_feasible_yaw_span_deg=0.0,
        )
        robust = select_best_grasp(
            objects[2], objects, gripper_inner_width_m=.049,
            min_feasible_yaw_span_deg=10.0,
        )

        self.assertTrue(permissive["grasp_feasible"])
        self.assertEqual(permissive["feasible_yaw_intervals_deg"], [[23.0, 27.0]])
        self.assertFalse(robust["grasp_feasible"])
        self.assertEqual(robust["robust_feasible_yaw_intervals_deg"], [])
        self.assertEqual(
            robust["rejected_narrow_feasible_yaw_intervals_deg"], [[23.0, 27.0]],
        )

    def test_place_gripper_collision_with_prior_row_gets_safe_same_region_target(self):
        state = {
            "scene_revision": 2,
            "table_bounds": {
                "xmin": .235, "xmax": .65, "ymin": -.1, "ymax": .4,
            },
            "objects": [
                _object(2, "square red", [.3376, .2193, -.0002], [.0233, .0209, .0210]),
                _object(3, "square blue", [.3393, .1405, -.0011], [.0225, .0217, .0249]),
            ],
        }
        contract = _organize_contract()
        plan = {
            "groups": [{
                "group_id": "blue_group", "group_value": "blue",
                "object_ids": [3], "target_region_id": "blue",
            }],
            "target_regions": [{
                "region_id": "blue",
                "bounds_base_m": {"xmin": .25, "xmax": .42, "ymin": .2362, "ymax": .2686},
            }],
        }
        proposal = {
            "action_type": "pick_place", "group_id": "blue_group",
            "target_region_id": "blue", "selected_object_id": 3,
            "object_label": "square blue",
            "object_center_base_m": [.3393, .1405, -.0011],
            "scene_revision": 2,
            "target_pose_base": {
                "position_m": [.335, .2524, -.0011], "yaw_rad": 0.0,
            },
        }

        action, report = validate_task_action(proposal, state, contract, plan, {})

        self.assertIsNone(action)
        self.assertEqual(report["reason"], "target_pose_gripper_clearance_blocked")
        self.assertEqual(
            report["checks"]["target_pose_base"]["detail"]["blocking_objects"][0]["id"],
            2,
        )
        suggested = report["checks"]["target_pose_base"]["detail"][
            "suggested_collision_free_position_m"
        ]
        corrected = copy.deepcopy(proposal)
        corrected["target_pose_base"]["position_m"] = suggested
        accepted, corrected_report = validate_task_action(
            corrected, state, contract, plan, {},
        )
        self.assertIsNotNone(accepted)
        self.assertTrue(corrected_report["accepted"])

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

    def test_organize_rejects_inside_member_while_same_group_has_outside_member(self):
        state = _organize_state()
        state["objects"].append(_object(3, "square red", [0.50, 0.20, 0.02]))
        plan = _organize_plan()
        plan["groups"][0]["object_ids"] = [1, 3]
        progress = {"group_diagnostics": [{
            "group_id": "red_group", "inside_region": [1], "outside_region": [3],
        }]}
        action, report = validate_task_action(
            _organize_action(), state, _organize_contract(), plan, progress,
        )
        self.assertIsNone(action)
        self.assertEqual(
            report["reason"],
            "selected_object_already_inside_while_group_has_outside_members",
        )

    def test_organize_rejects_noop_pick_place(self):
        proposal = _organize_action()
        proposal["target_pose_base"]["position_m"] = [0.20, 0.0, 0.02]
        action, report = validate_task_action(
            proposal, _organize_state(), _organize_contract(), _organize_plan(), {},
        )
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "target_pose_too_close_to_source")
        suggested = report["checks"]["target_pose_base"]["detail"][
            "suggested_collision_free_position_m"
        ]
        self.assertGreater(
            ((suggested[0] - 0.20) ** 2 + (suggested[1] - 0.0) ** 2) ** 0.5,
            0.006,
        )
        self.assertNotEqual(
            (round(suggested[0], 2), round(suggested[1], 2)),
            (round(0.20, 2), round(0.0, 2)),
        )

    def test_center_inside_but_footprint_outside_region_is_rejected(self):
        proposal = _organize_action(); proposal["target_pose_base"]["position_m"][0] = 0.295
        _action, report = validate_task_action(proposal, _organize_state(), _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "target_pose_outside_group_region")
        detail = report["checks"]["target_pose_base"]["detail"]
        self.assertEqual(detail["target_region_bounds_base_m"], {
            "xmin": 0.15, "xmax": 0.30, "ymin": -0.05, "ymax": 0.05,
        })
        self.assertAlmostEqual(detail["allowed_center_x_m"][1], 0.285)
        self.assertEqual(detail["suggested_interval_midpoint_position_m"], [0.225, 0.0, 0.02])

    def test_layout_violation_is_rejected_without_pose_correction(self):
        state = _organize_state(); state["objects"].append(_object(3, "square red", [0.18, 0.0, 0.02]))
        proposal = _organize_action(); proposal["target_pose_base"]["position_m"] = [0.25, 0.03, 0.02]
        _action, report = validate_task_action(proposal, state, _organize_contract(), _organize_plan(), {})
        self.assertEqual(report["reason"], "target_pose_violates_group_layout")
        self.assertEqual(proposal["target_pose_base"]["position_m"], [0.25, 0.03, 0.02])

    def test_overlap_feedback_supplies_geometry_checked_recovery_position(self):
        state = _organize_state()
        state["objects"].append(_object(3, "square red", [0.50, 0.20, 0.02]))
        plan = _organize_plan()
        plan["groups"][0]["object_ids"] = [1, 3]
        proposal = _organize_action()
        proposal.update({
            "selected_object_id": 3,
            "object_center_base_m": [0.50, 0.20, 0.02],
        })
        proposal["target_pose_base"]["position_m"] = [0.20, 0.0, 0.02]
        action, report = validate_task_action(
            proposal, state, _organize_contract(), plan,
            {"group_diagnostics": [{"group_id": "red_group", "outside_region": [3]}]},
        )
        self.assertIsNone(action)
        self.assertEqual(report["reason"], "target_pose_overlaps_planned_object")
        suggested = report["checks"]["target_pose_base"]["detail"][
            "suggested_collision_free_position_m"
        ]
        self.assertNotEqual(suggested[:2], [0.20, 0.0])
        corrected = dict(proposal)
        corrected["target_pose_base"] = {"position_m": suggested, "yaw_rad": 0.0}
        accepted, corrected_report = validate_task_action(
            corrected, state, _organize_contract(), plan,
            {"group_diagnostics": [{"group_id": "red_group", "outside_region": [3]}]},
        )
        self.assertIsNotNone(accepted)
        self.assertTrue(corrected_report["accepted"])

    def test_first_member_can_enter_empty_target_row_progressively(self):
        state = _organize_state()
        state["objects"].append(_object(3, "square red", [0.50, 0.20, 0.02]))
        proposal = _organize_action()
        proposal["target_pose_base"]["position_m"] = [0.25, 0.03, 0.02]
        action, report = validate_task_action(
            proposal, state, _organize_contract(), _organize_plan(), {},
        )
        self.assertTrue(report["accepted"])
        self.assertEqual(action["selected_object_id"], 1)

    def test_disappeared_temporary_object_requests_reselection(self):
        state = _organize_state(); state["objects"] = [state["objects"][1]]
        _action, report = validate_task_action(
            _organize_action(), state, _organize_contract(), _organize_plan(), {},
        )
        self.assertEqual(report["reason"], "selected_object_no_longer_exists")


class GeometryAndCompletionTests(unittest.TestCase):
    def test_goal_progress_tolerates_small_observed_row_boundary_jitter(self):
        state = _organize_state()
        # The 30 mm square crosses xmax=0.30 by only 1 mm after noisy observed
        # yaw/center reconstruction; commanded targets are validated strictly.
        state["objects"][0]["geometry_center_m"] = [0.286, 0.0, 0.02]

        progress = evaluate_task_goal_progress(
            state, _organize_contract(), _organize_plan(), CONFIG,
        )

        red = next(
            item for item in progress["group_diagnostics"]
            if item["group_id"] == "red_group"
        )
        self.assertEqual(red["inside_region"], [1])
        self.assertEqual(red["outside_region"], [])

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


class DynamicProtectionAndSafetyTests(unittest.TestCase):
    def test_runtime_red_collision_yaw_is_rejected_as_blocked(self):
        target = _object(6, "square red", [0.2778, 0.0465, -0.0021], [0.0242, 0.0222, 0.0229])
        target["table_yaw_deg"] = 66.25
        objects = [
            target,
            _object(3, "square yellow", [0.2949, 0.0907, -0.0026], [0.0250, 0.0235, 0.0249]),
            _object(5, "square green", [0.2715, 0.1183, -0.0020], [0.0257, 0.0217, 0.0236]),
        ]
        action = {"action_type": "pick_place", "object_id": 6}
        selected, report = _validate_pick_grasp_geometry(SimpleNamespace(), {"objects": objects}, action)
        self.assertFalse(report["grasp_feasible"])
        self.assertNotIn("selected_grasp_yaw_deg", selected)
        self.assertTrue(any(start <= 156.25 < end for start, end in report["blocked_yaw_intervals_deg"]))

    def test_organize_pick_blocked_at_all_yaws_stops_before_moveit(self):
        target = _object(1, "square red", [0.40, 0.0, 0.02], [0.024, 0.024, 0.024])
        objects = [target]
        for index, (x, y) in enumerate(((0.45, 0.0), (0.35, 0.0), (0.40, 0.05), (0.40, -0.05)), 2):
            objects.append(_object(index, "square blue", [x, y, 0.02], [0.03, 0.03, 0.03]))
        args = SimpleNamespace(execute=True, moveit_plan_only=False)
        action = {"action_type": "pick_place", "object_id": 1}
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "tools.workflows.stack_demo.task_execution.build_pick_place_plans"
        ) as build_plans:
            checked = preflight_pick_place_action(args, output_dir, {"objects": objects}, action, 1)
        build_plans.assert_not_called()
        self.assertFalse(checked["moveit_feasible"])
        self.assertEqual(checked["preflight_failure_type"], "selected_pick_not_grasp_feasible")
        self.assertTrue(checked["physical_grasp_validation"]["all_grasps_blocked"])

    def test_collision_checked_yaw_is_forwarded_to_pick_plan_builder(self):
        target = _object(1, "square red", [0.3289, 0.0888, -0.002], [0.0226, 0.0222, 0.025])
        target["table_yaw_deg"] = -0.8
        obstacle = _object(2, "square blue", [0.3304, 0.1338, 0.0], [0.0225, 0.0216, 0.0257])
        obstacle["table_yaw_deg"] = 0.12
        args = SimpleNamespace(execute=False, moveit_plan_only=False)
        action = {"action_type": "pick_place", "object_id": 1}
        captured = {}

        def fake_build(_args, _cycle, _state, checked_action, _step):
            captured.update(checked_action)
            return "pick.json", "place.json"

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "tools.workflows.stack_demo.task_execution.build_pick_place_plans",
            side_effect=fake_build,
        ):
            preflight_pick_place_action(args, output_dir, {"objects": [target, obstacle]}, action, 1)
        self.assertAlmostEqual(captured["selected_grasp_yaw_deg"], 89.2, delta=2.0)
        self.assertEqual(captured["grasp_yaw_source"], "target_principal_axis")

    def test_clearance_execution_requires_separate_authorization(self):
        args = SimpleNamespace(execute=True, execute_push_clearing=False)
        with self.assertRaisesRegex(RuntimeError, "CLEARANCE_EXECUTION_NOT_AUTHORIZED"):
            _execute_task_action(
                args, "unused", {}, {}, {}, {"action_type": "nudge"}, 1,
            )

    def test_offline_reobserve_fails_without_calling_live_perception(self):
        args = SimpleNamespace(offline_scene_state="fixed_scene.json")
        with patch("tools.workflows.stack_demo.task_workflow.capture_scene_observation") as capture:
            with self.assertRaisesRegex(RuntimeError, "OFFLINE_REOBSERVE_UNAVAILABLE"):
                _reobserve(args, "/tmp/cycle", {})
        capture.assert_not_called()

    def test_satisfied_current_roles_become_protected(self):
        state = _house_state(); progress = _house_progress(state)
        protection = derive_dynamic_protection(state, _house_contract(), progress, progress["selected_role_assignment"])
        self.assertEqual(set(protection["protected_object_ids"]), {1, 2, 3, 4, 5, 6})
        self.assertEqual(len(protection["protected_regions"]), 6)

    def test_protection_tracks_changed_detection_ids(self):
        state = _house_state()
        for obj in state["objects"]: obj["id"] += 10
        plan = _house_plan()
        for index, binding in enumerate(plan["role_assignments"]): binding["selected_object_id"] = index + 11
        plan["orientation_observations"][0]["selected_object_id"] = 15
        plan["orientation_observations"][1]["selected_object_id"] = 16
        progress = evaluate_task_goal_progress(state, _house_contract(), fuse_house_orientation_observations(plan, state, CONFIG), CONFIG)
        protection = derive_dynamic_protection(state, _house_contract(), progress, progress["selected_role_assignment"])
        self.assertEqual(set(protection["protected_object_ids"]), {11, 12, 13, 14, 15, 16})

    def test_failed_role_predicate_removes_that_role_protection(self):
        state = _house_state()
        for obj in state["objects"][:4]: obj["geometry_center_m"][2] += 0.04
        progress = _house_progress(state)
        protection = derive_dynamic_protection(state, _house_contract(), progress, progress["selected_role_assignment"])
        roles = {item["role_id"] for item in protection["protected_roles"]}
        self.assertNotIn("left_support_lower", roles)

    def test_execute_with_offline_scene_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            validate_execution_source(SimpleNamespace(execute=True, offline_scene_state="scene.json"))

    def test_dry_run_with_offline_scene_is_allowed(self):
        validate_execution_source(SimpleNamespace(execute=False, offline_scene_state="scene.json"))

    def test_mock_perception_execution_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            validate_execution_source(SimpleNamespace(execute=True, mock_perception=True))

    def test_execute_and_moveit_plan_only_are_mutually_exclusive(self):
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            validate_execution_source(SimpleNamespace(execute=True, moveit_plan_only=True))

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
            "strategy_id": "place_from_left", "action_type": "pick_place", "role_id": "left_support_lower", "selected_object_id": 1,
            "object_label": "square red", "object_center_base_m": [0.30, 0.0, 0.02],
            "scene_revision": 4, "target_pose_base": {"position_m": [0.25, -0.08, 0.02], "yaw_rad": 0.0},
        }
        second_proposal = {**proposal, "strategy_id": "place_from_right"}
        args = SimpleNamespace(max_vlm_action_attempts=2, execute=True)
        preflight_results = [
            {**adapt_task_action_to_legacy_action(proposal), "moveit_feasible": False, "moveit_preflight_error": "no IK"},
            {**adapt_task_action_to_legacy_action(proposal), "moveit_feasible": True},
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            with patch("tools.workflows.stack_demo.task_workflow.call_vlm_task_policy", side_effect=[{"call_status": "parsed", "decision": proposal}, {"call_status": "parsed", "decision": second_proposal}]), patch(
                "tools.workflows.stack_demo.task_workflow._preflight_task_action", side_effect=preflight_results,
            ) as preflight:
                action, _report = _select_task_action(
                    args, _house_state(), _house_contract(), _house_plan(), {}, {}, CONFIG,
                    output_dir, 1,
                )
        self.assertTrue(action["moveit_feasible"])
        self.assertEqual(preflight.call_count, 2)

    def test_correctable_nudge_parameters_do_not_forbid_same_strategy_retry(self):
        proposal = {
            "strategy_id": "clear_blocker_by_nudge", "action_type": "nudge",
            "selected_object_id": 1, "target_object_id": 2,
            "object_label": "square red", "object_center_base_m": [0.20, 0.0, 0.02],
            "target_object_label": "square blue", "target_object_center_base_m": [0.45, 0.0, 0.02],
            "scene_revision": 2, "direction_base": [0.0, 1.0, 0.0],
            "distance_m": 0.03, "contact_side": "+y", "gripper_yaw_rad": 0.0,
        }
        corrected = {**proposal, "contact_side": "-y"}
        args = SimpleNamespace(max_vlm_action_attempts=2, execute=True, moveit_plan_only=False)
        accepted_action = {
            "action_type": "nudge", "object_id": 1, "target_object_id": 2,
            "direction_base": [0.0, 1.0, 0.0], "distance_m": 0.03,
            "contact_side": "-y", "gripper_yaw_rad": 0.0,
        }
        invalid = {
            "accepted": False, "reason": "push_parameters_invalid",
            "failed_fields": ["contact_side_matches_push_direction"],
            "checks": {"contact_side_matches_push_direction": {"ok": False, "detail": {"contact_side": "+y"}}},
        }
        with tempfile.TemporaryDirectory() as output_dir, patch(
            "tools.workflows.stack_demo.task_workflow.call_vlm_task_policy",
            side_effect=[
                {"call_status": "parsed", "decision": proposal},
                {"call_status": "parsed", "decision": corrected},
            ],
        ), patch(
            "tools.workflows.stack_demo.task_workflow.validate_vlm_action_decision",
            side_effect=[(None, invalid), (accepted_action, {"accepted": True})],
        ), patch(
            "tools.workflows.stack_demo.task_workflow._preflight_task_action",
            return_value={**accepted_action, "moveit_feasible": True},
        ):
            action, _report = _select_task_action(
                args, _clearance_state(), _organize_contract(), _organize_plan(), {}, {}, CONFIG,
                output_dir, 1,
            )
        self.assertEqual(action["contact_side"], "-y")

    def test_grounding_metadata_correction_is_not_blocked_as_duplicate_action(self):
        proposal = {
            "strategy_id": "place_lower", "action_type": "pick_place",
            "role_id": "left_support_lower", "selected_object_id": 1,
            "object_label": "square red", "object_center_base_m": [0.45, 0.20, 0.02],
            "scene_revision": 4,
            "target_pose_base": {"position_m": [0.25, -0.08, 0.02], "yaw_rad": 0.0},
        }
        corrected = copy.deepcopy(proposal)
        corrected["object_center_base_m"] = [0.30, 0.0, 0.02]
        args = SimpleNamespace(max_vlm_action_attempts=2, execute=False, moveit_plan_only=False)
        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "tools.workflows.stack_demo.task_workflow.call_vlm_task_policy",
                side_effect=[
                    {"call_status": "parsed", "decision": proposal},
                    {"call_status": "parsed", "decision": corrected},
                ],
            ), patch(
                "tools.workflows.stack_demo.task_workflow._preflight_task_action",
                side_effect=lambda _args, _cycle, _state, action, _step: action,
            ):
                action, _report = _select_task_action(
                    args, _house_state(), _house_contract(), _house_plan(), {}, {}, CONFIG,
                    output_dir, 1,
                )
        self.assertEqual(action["object_center_base_m"], [0.30, 0.0, 0.02])


def _object(object_id, label, center, size=None):
    return {"id": object_id, "label": label, "geometry_center_m": list(center), "dimensions_m": size or [0.03, 0.03, 0.04], "geometry_frame": "base_link"}


def _house_contract():
    return house_contract()


def _house_state():
    return house_state()


def _house_plan():
    return house_plan()


def _house_progress(state):
    plan = fuse_house_orientation_observations(_house_plan(), state, CONFIG)
    return evaluate_task_goal_progress(state, _house_contract(), plan, CONFIG)


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
