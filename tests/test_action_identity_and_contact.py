import glob
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from robot_scene_pipeline.action_fingerprint import (
    build_replanning_context,
    ledger_entry,
    normalize_action_fingerprint,
)
from robot_scene_pipeline.grasp_yaw_search import _gripper_finger_boxes
from robot_scene_pipeline.object_tracking import resolve_action_references, update_scene_tracks
from robot_scene_pipeline.tool_swept_volume import check_tool_swept_volume, push_tool_swept_obbs
from tools.workflows.stack_demo import vlm_action
from tools.workflows.stack_demo.vlm_action_loop import _take_untried_vlm_fallback


ROOT = os.path.dirname(os.path.dirname(__file__))


class ActionIdentityAndContactTests(unittest.TestCase):
    def test_reason_and_small_distance_tail_do_not_change_fingerprint(self):
        state = _tracked_state()
        first = normalize_action_fingerprint(_nudge(reason="first", distance=0.0251), state, 12)
        second = normalize_action_fingerprint(_nudge(reason="different", distance=0.0249), state, 12)

        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["direction_bin"], "+X")
        self.assertEqual(first["distance_bin_m"], 0.025)

    def test_replanning_escalates_action_type_strategy_and_safe_stop(self):
        state = _tracked_state()
        first_fp = normalize_action_fingerprint(_nudge(strategy="clear_a"), state, 12)
        second_action = _nudge(strategy="clear_b", direction=[0.0, 1.0, 0.0])
        second_fp = normalize_action_fingerprint(second_action, state, 12)
        ledger = [
            ledger_entry(1, first_fp, _nudge(strategy="clear_a"), "collision", ["blocked"], 12),
            ledger_entry(2, second_fp, second_action, "collision", ["blocked"], 12),
        ]

        third = build_replanning_context(ledger, 3, 5)["hard_constraints"]
        fourth = build_replanning_context(ledger, 4, 5)["hard_constraints"]
        fifth = build_replanning_context(ledger, 5, 5)["hard_constraints"]
        self.assertEqual(third["forbidden_action_types"], ["nudge"])
        self.assertTrue(fourth["required_strategy_change"])
        self.assertTrue(fifth["safe_stop_allowed"])

    def test_pick_place_pose_corrections_do_not_forbid_the_required_action_type(self):
        state = _tracked_state()
        action = {
            "strategy_id": "place_upper", "action_type": "pick_place",
            "selected_object_id": 1, "role_id": "left_support_upper",
            "target_pose_base": {"position_m": [0.3, 0.0, 0.05]},
        }
        fingerprint = normalize_action_fingerprint(action, state, 12)
        ledger = [
            ledger_entry(index, fingerprint, action, "action_semantic_validation", ["bad_pose"], 12)
            for index in (1, 2)
        ]
        constraints = build_replanning_context(ledger, 3, 5)["hard_constraints"]
        self.assertNotIn("pick_place", constraints["forbidden_action_types"])

    def test_track_survives_detector_id_and_order_changes(self):
        memory = {}
        first = [_object(1, "square green", 0.0), _object(2, "square green", 0.10)]
        update_scene_tracks(memory, first, 1)
        left_track, right_track = first[0]["track_id"], first[1]["track_id"]
        second = [_object(9, "square green", 0.101), _object(8, "square green", 0.002)]

        _memory, assignments = update_scene_tracks(memory, second, 2)

        self.assertEqual(second[0]["track_id"], right_track)
        self.assertEqual(second[1]["track_id"], left_track)
        self.assertEqual(len({item["track_id"] for item in assignments}), 2)
        self.assertEqual(second[1]["object_ref"], "scene_2:obj_8")

    def test_ambiguous_track_requires_reobserve(self):
        memory = {}
        first = [_object(1, "square green", -0.01), _object(2, "square green", 0.01)]
        update_scene_tracks(memory, first, 1)
        current = [_object(7, "square green", 0.0)]
        update_scene_tracks(memory, current, 2)

        _resolved, error = resolve_action_references(
            {"action_type": "pick", "object_track_id": current[0]["track_id"]},
            {"scene_revision": 2, "objects": current},
            2,
        )
        self.assertTrue(current[0]["tracking_ambiguous"])
        self.assertEqual(error["reason"], "ambiguous_track_requires_reobserve")

    def test_stale_object_ref_and_mismatched_track_are_rejected(self):
        state = _tracked_state()
        _resolved, stale = resolve_action_references(
            {"action_ref": "unused", "object_ref": "scene_11:obj_1"}, state, 12,
        )
        self.assertEqual(stale["reason"], "stale_or_invalid_object_ref")
        _resolved, mismatch = resolve_action_references(
            {"object_ref": "scene_12:obj_1", "object_track_id": "track_red_01"}, state, 12,
        )
        self.assertEqual(mismatch["reason"], "object_reference_track_mismatch")

    def test_duplicate_blacklist_returns_before_geometry_validation(self):
        state = _tracked_state()
        proposal = _nudge()
        proposal.update({
            "object_ref": "scene_12:obj_1", "object_track_id": "track_green_01",
            "target_object_ref": "scene_12:obj_2", "target_object_track_id": "track_red_01",
            "object_label": "square green", "target_object_label": "square red",
            "object_center_base_m": [0.0, 0.0, 0.02],
            "target_object_center_base_m": [0.1, 0.0, 0.02],
        })
        fingerprint = normalize_action_fingerprint(proposal, state, 12)["fingerprint"]
        original_call = vlm_action.call_vlm_action_policy
        original_validate = vlm_action.validate_vlm_action_decision
        vlm_action.call_vlm_action_policy = lambda *_args, **_kwargs: {"call_status": "parsed", "decision": proposal}
        vlm_action.validate_vlm_action_decision = lambda *_args, **_kwargs: self.fail("geometry validation ran")
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                selected, _report, safety, preflight = vlm_action.evaluate_autonomous_vlm_action_attempt(
                    SimpleNamespace(no_image=True), output_dir, state, state["objects"][0], [], None,
                    {}, 1, scene_revision=12,
                    replanning_context={"hard_constraints": {"forbidden_action_fingerprints": [fingerprint]}},
                )
        finally:
            vlm_action.call_vlm_action_policy = original_call
            vlm_action.validate_vlm_action_decision = original_validate
        self.assertIsNone(selected)
        self.assertEqual(safety["reason"], "duplicate_failed_action")
        self.assertFalse(preflight["run_moveit_preflight"])

    def test_backend_failure_never_reaches_reference_or_fingerprint(self):
        original_call = vlm_action.call_vlm_action_policy
        original_resolve = vlm_action.resolve_action_references
        original_fingerprint = vlm_action.normalize_action_fingerprint
        vlm_action.call_vlm_action_policy = lambda *_args, **_kwargs: {
            "call_status": "backend_failed", "error_type": "REQUEST_TIMEOUT", "decision": None,
        }
        vlm_action.resolve_action_references = lambda *_args, **_kwargs: self.fail("reference validation ran")
        vlm_action.normalize_action_fingerprint = lambda *_args, **_kwargs: self.fail("fingerprint generation ran")
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                selected, report, safety, preflight = vlm_action.evaluate_autonomous_vlm_action_attempt(
                    SimpleNamespace(no_image=True), output_dir, _tracked_state(),
                    _tracked_state()["objects"][0], [], None, {}, 1, scene_revision=12,
                )
        finally:
            vlm_action.call_vlm_action_policy = original_call
            vlm_action.resolve_action_references = original_resolve
            vlm_action.normalize_action_fingerprint = original_fingerprint
        self.assertIsNone(selected)
        self.assertIsNone(report["decision"])
        self.assertTrue(safety["backend_failure"])
        self.assertTrue(preflight["backend_failure"])

    def test_stop_does_not_require_object_reference(self):
        stop = {
            "action_type": "stop", "strategy_id": "safe_stop", "reason": "unsafe",
            "scene_problem": "unsafe", "predicted_scene_benefit": "none",
            "risk_assessment": "collision", "confidence": 0.9,
        }
        original_call = vlm_action.call_vlm_action_policy
        original_resolve = vlm_action.resolve_action_references
        vlm_action.call_vlm_action_policy = lambda *_args, **_kwargs: {"call_status": "parsed", "decision": stop}
        vlm_action.resolve_action_references = lambda *_args, **_kwargs: self.fail("stop reference validation ran")
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                selected, report, safety, preflight = vlm_action.evaluate_autonomous_vlm_action_attempt(
                    SimpleNamespace(no_image=True), output_dir, _tracked_state(),
                    _tracked_state()["objects"][0], [], None, {}, 1, scene_revision=12,
                )
        finally:
            vlm_action.call_vlm_action_policy = original_call
            vlm_action.resolve_action_references = original_resolve
        self.assertIsNone(selected)
        self.assertEqual(report["action_type"], "stop")
        self.assertEqual(safety["reason"], "policy_requested_stop")
        self.assertEqual(preflight["control_action"], "stop")

    def test_fallback_uses_only_model_generated_untried_candidate(self):
        state = _tracked_state()
        failed = _nudge()
        failed.update({"object_track_id": "track_green_01", "target_object_track_id": "track_red_01"})
        context = build_replanning_context([
            ledger_entry(1, normalize_action_fingerprint(failed, state, 12), failed, "collision", ["blocked"], 12)
        ], 2, 5)
        repeated = dict(failed)
        alternative = {
            **failed, "strategy_id": "clear_blocker_by_pick_away", "action_type": "pick_away",
            "safe_place_center_base_m": [0.2, 0.2, 0.02], "confidence": 0.8,
        }

        selected = _take_untried_vlm_fallback(
            [repeated, alternative], state, state["objects"][0], 12, context,
        )

        self.assertEqual(selected["action_type"], "pick_away")
        self.assertEqual(selected["strategy_id"], "clear_blocker_by_pick_away")

    def test_segmented_push_profile_uses_measured_widths(self):
        swept = push_tool_swept_obbs(_push_plan(), safety_margin_m=0.0)
        contact = {item["profile_name"]: item for item in swept if item["stage"] == "contact_pose"}

        self.assertEqual(contact["tip"]["profile_width_m"], 0.025)
        self.assertEqual(contact["upper_fingers"]["profile_width_m"], 0.062)
        self.assertEqual(contact["gripper_body"]["profile_width_m"], 0.112)
        self.assertAlmostEqual(contact["tip"]["zmax"] - contact["tip"]["zmin"], 0.025)

    def test_open_gripper_gap_is_not_a_collision_box(self):
        boxes = _gripper_finger_boxes(0.02, 0.049 / 2.0, 0.112 / 2.0)

        self.assertEqual(len(boxes), 2)
        self.assertTrue(all(box["vmax"] <= -0.049 / 2.0 or box["vmin"] >= 0.049 / 2.0 for box in boxes))

    def test_light_loose_contact_is_controlled_but_protected_is_hard(self):
        loose = _object(3, "square blue", -0.037, y=0.0, size=[0.006, 0.01, 0.01])
        loose["geometry_center_m"][2] = 0.017
        loose["pushable"] = True
        report = check_tool_swept_volume(
            _push_plan(), [loose], ignore_object_ids=[2], safety_margin_m=0.0,
            tool_depth_m=0.01, fingertip_thickness_m=0.006,
            controlled_contact={"enabled": True, "max_side_intrusion_m": 0.005},
        )
        protected = check_tool_swept_volume(
            _push_plan(), [loose], ignore_object_ids=[2], protected_object_ids=[3],
            safety_margin_m=0.0, tool_depth_m=0.01, fingertip_thickness_m=0.006,
            controlled_contact={"enabled": True, "max_side_intrusion_m": 0.005},
        )

        self.assertTrue(report["feasible"])
        self.assertEqual(report["contact_status"], "controlled_contact")
        self.assertTrue(report["requires_reobservation"])
        self.assertFalse(protected["feasible"])
        self.assertEqual(protected["contact_status"], "hard_collision")

    def test_linear_stack_runtime_repeated_nudge_has_one_fingerprint_and_is_semantically_invalid(self):
        cycle = os.path.join(ROOT, "runtime", "linear_stack_20260712_164342", "cycle_01_object_1")
        if not os.path.isdir(cycle):
            self.skipTest("optional recorded runtime fixture is not present")
        with open(os.path.join(cycle, "scene_state_before_action.json"), encoding="utf-8") as handle:
            state = json.load(handle)
        state["scene_revision"] = 1
        for obj in state["objects"]:
            color = obj["label"].split()[-1]
            obj["track_id"] = "track_{}_01".format(color)
            obj["object_ref"] = "scene_1:obj_{}".format(obj["id"])
        fingerprints = set()
        for path in sorted(glob.glob(os.path.join(cycle, "vlm_action_attempt_*_output.json"))):
            with open(path, encoding="utf-8") as handle:
                decision = json.load(handle)["decision"]
            fingerprints.add(normalize_action_fingerprint(decision, state, 1)["fingerprint"])
            self.assertEqual(
                vlm_action._stack_action_semantic_error(decision, state["objects"][1]),
                "nudge_cannot_satisfy_vertical_stack_relation",
            )
        self.assertEqual(len(fingerprints), 1)

    def test_four_recorded_runtime_scenes_load_and_receive_unique_tracks(self):
        relative_states = [
            ("linear_stack_20260712_164224", "initial_order/private_scene_state.json"),
            ("linear_stack_20260712_164342", "initial_order/private_scene_state.json"),
            ("organize_blocks_20260712_164734", "initial_task_scene/private_scene_state.json"),
            ("build_house_20260712_165331", "initial_task_scene/private_scene_state.json"),
        ]
        checked = 0
        for runtime_name, state_path in relative_states:
            with self.subTest(runtime=runtime_name):
                path = os.path.join(ROOT, "runtime", runtime_name, state_path)
                # runtime/ is intentionally not a portable source fixture.  Replay
                # whichever recordings are present without failing a clean clone.
                if not os.path.isfile(path):
                    continue
                with open(path, encoding="utf-8") as handle:
                    state = json.load(handle)
                objects = state.get("objects", [])
                memory, assignments = update_scene_tracks({}, objects, 1)
                track_ids = [obj["track_id"] for obj in objects]
                self.assertTrue(objects)
                self.assertEqual(len(track_ids), len(set(track_ids)))
                self.assertEqual(len(assignments), len(objects))
                self.assertEqual(len(memory["track_history"]), 1)
                checked += 1
        self.assertGreater(checked, 0, "no recorded runtime scene fixture is available")


def _object(object_id, label, x, y=0.0, size=None):
    return {
        "id": object_id, "label": label, "geometry_center_m": [x, y, 0.02],
        "dimensions_m": size or [0.03, 0.03, 0.04],
    }


def _tracked_state():
    return {
        "scene_revision": 12,
        "objects": [
            {**_object(1, "square green", 0.0), "object_ref": "scene_12:obj_1", "track_id": "track_green_01"},
            {**_object(2, "square red", 0.1), "object_ref": "scene_12:obj_2", "track_id": "track_red_01"},
        ],
    }


def _nudge(reason="clear", distance=0.025, strategy="clear_blocker_by_nudge", direction=None):
    return {
        "strategy_id": strategy, "action_type": "nudge", "object_id": 1, "target_object_id": 2,
        "direction_base": direction or [1.0, 0.0, 0.0], "distance_m": distance,
        "gripper_yaw_rad": 0.0, "reason": reason,
    }


def _push_plan():
    return {
        "schema_version": "push_execution_plan_v1", "frame_id": "base_link",
        "obstacle": _object(2, "square yellow", 0.0),
        "direction_base": [1.0, 0.0, 0.0], "distance_m": 0.025,
        "lift_m": 0.05, "contact_z_offset_m": 0.015,
        "gripper_yaw_rad": 0.0, "target_yaw_deg": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
