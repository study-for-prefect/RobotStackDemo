import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.object_tracking import update_scene_tracks
from robot_scene_pipeline.ollama_policy_client import call_policy
from tools.workflows.stack_demo.app import main
from tools.workflows.stack_demo.clutter.edge_generation import _opposite_side, generate_physical_edges
from tools.workflows.stack_demo.clutter.grasp_edges import scan_grasp_yaws
from tools.workflows.stack_demo.clutter.target_options import build_target_options
from tools.workflows.stack_demo.policy.qwen_client import StatelessQwenClient
from tools.workflows.stack_demo.policy.schemas import (
    PolicyOutputError,
    parse_edge_selection,
    parse_target_selection,
)
from tools.workflows.stack_demo.policy.target_selector import QwenTargetSelector
from tools.workflows.stack_demo.common.scene_state import build_clutter_scene_state

from tests.new_arch_fixtures import MockQwenClient, config, edge, placement, raw_object, scene


class NewSceneEdgePolicyTests(unittest.TestCase):
    def test_code_owned_push_direction_has_opposite_contact_side(self):
        self.assertEqual(_opposite_side([1.0, 0.0, 0.0]), "-x")
        self.assertEqual(_opposite_side([-1.0, 0.0, 0.0]), "+x")
        self.assertEqual(_opposite_side([0.0, 1.0, 0.0]), "-y")
        self.assertEqual(_opposite_side([0.0, -1.0, 0.0]), "+y")

    def test_01_main_entry_loads_new_planner(self):
        fixture = Path(__file__).parent / "fixtures" / "new_arch_single_red_scene.json"
        with tempfile.TemporaryDirectory() as output:
            mock = Path(output) / "mock_policy"
            mock.mkdir()
            (mock / "edge_selection.json").write_text(json.dumps({
                "selected_candidate_id": "edge_r1_track_red_01_grasp_1",
                "backup_candidate_ids": [],
                "reason_codes": ["direct_task_progress"],
            }))
            result = main([
                "--task-type", "organize_blocks",
                "--offline-scene-state", str(fixture),
                "--mock-policy-response-dir", str(mock),
                "--output-dir", output,
                "--no-image",
            ])
            self.assertEqual(result, 0)
            self.assertTrue(list(Path(output).glob("cycle_*/physical_action_edges.json")))

    def test_02_target_options_require_feasible_first_step(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", {"t1": generated.edges_by_target["t1"], "t2": ()}, generated.grasp_scans)
        self.assertEqual([item.target_track_id for item in options], ["t1"])

    def test_03_infeasible_object_never_reaches_qwen(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        client = MockQwenClient([])
        outcome = QwenTargetSelector(client).select(current, {}, [], ())
        self.assertIsNone(outcome.selected)
        self.assertFalse(client.calls)

    def test_04_target_option_contains_required_geometry_features(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.56, 0.0, 0.02])])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        value = options[0].to_dict()
        for key in ("neighbor_count", "edge_clearances_m", "blocked_sides", "safe_grasp_interval_count", "estimated_clearance_cost", "releases_track_ids"):
            self.assertIn(key, value)

    def test_05_qwen_initial_request_has_no_assistant_history(self):
        args = _ollama_args()
        response = SimpleNamespace(
            transport_status="ok", schema_valid=True, content='{"selected_target_option_id":"x","reason_codes":["a"]}',
            parsed_decision={"selected_target_option_id": "x", "reason_codes": ["a"]},
            error_type=None, error_message=None, generation_status="parsed",
        )
        with patch("tools.workflows.stack_demo.policy.qwen_client.call_policy", return_value=response) as call:
            StatelessQwenClient(args).call("target_selection", {"scene_revision": 4}, {}, (), None)
        messages = call.call_args.args[2]
        self.assertEqual([item["role"] for item in messages], ["system", "user"])

    def test_06_request_contains_only_latest_scene_revision(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])], revision=9)
        generated = generate_physical_edges(current, ["t1", "t2"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        client = MockQwenClient([{"selected_target_option_id": options[0].target_option_id, "reason_codes": ["low_clearance_cost"]}])
        outcome = QwenTargetSelector(client).select(current, {}, options)
        self.assertEqual(outcome.request["scene_revision"], 9)
        self.assertTrue(all(item["scene_revision"] == 9 for item in outcome.request["target_options"]))

    def test_07_unknown_target_option_id_is_rejected(self):
        with self.assertRaises(PolicyOutputError):
            parse_target_selection({"selected_target_option_id": "missing", "reason_codes": ["x"]}, [], 1)

    def test_08_unknown_candidate_id_is_rejected(self):
        with self.assertRaises(PolicyOutputError):
            parse_edge_selection({"selected_candidate_id": "missing", "backup_candidate_ids": [], "reason_codes": ["x"]}, [edge()], 1)

    def test_09_stale_scene_revision_is_rejected(self):
        with self.assertRaises(PolicyOutputError):
            parse_edge_selection({"selected_candidate_id": "edge_1", "backup_candidate_ids": [], "reason_codes": ["x"]}, [edge(revision=1)], 2)

    def test_10_invalid_json_gets_only_one_format_repair(self):
        args = _ollama_args(vlm_max_backend_retries=1)
        responses = [_response("not json"), _response("still not json")]
        with patch("robot_scene_pipeline.ollama_policy_client.requests.post", side_effect=responses) as post:
            result = call_policy(args, "target_selection", [{"role": "user", "content": "x"}], {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}, "additionalProperties": False})
        self.assertFalse(result.schema_valid)
        self.assertEqual(post.call_count, 2)

    def test_11_second_invalid_output_never_triggers_code_reselection(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02]), raw_object(2, "t2", [0.55, 0.2, 0.02])])
        generated = generate_physical_edges(current, ["t1", "t2"], "organize_blocks", config(), placement, lambda obj: None)
        options = build_target_options(current, "organize_blocks", generated.edges_by_target, generated.grasp_scans)
        client = MockQwenClient([{"selected_target_option_id": "unknown", "reason_codes": ["x"]}])
        outcome = QwenTargetSelector(client).select(current, {}, options)
        self.assertIsNone(outcome.selected)
        self.assertEqual(outcome.decision_source, "policy_invalid_output")

    def test_14_safe_diagonal_grasp_is_accepted(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        result = scan_grasp_yaws(current.current_objects[0], current.current_objects, config())
        self.assertTrue(result.graspable)
        self.assertNotIn(result.safe_intervals[0].selected_yaw_deg, (0.0, 90.0))

    def test_15_non_edge_aligned_yaw_generates_edge(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], yaw_deg=0.0)])
        generated = generate_physical_edges(current, ["t1"], "organize_blocks", config(), placement, lambda obj: None)
        yaw = generated.edges_by_target["t1"][0].physical_parameters["grasp_yaw_deg"]
        self.assertNotIn(yaw, (0.0, 90.0))

    def test_normal_block_preserves_grasp_yaw_through_release(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], yaw_deg=30.0)])

        def target(obj, interval):
            value = placement(obj, interval)
            return value.__class__(
                **{**value.__dict__, "place_pose": {
                    "frame_id": "base_link",
                    "position_m": [0.28, 0.18, 0.02],
                    "yaw_deg": 10.0,
                }}
            )

        generated = generate_physical_edges(
            current, ["t1"], "organize_blocks", config(), target, lambda obj: None,
        )
        physical = generated.edges_by_target["t1"][0].physical_parameters
        self.assertEqual(
            physical["release_gripper_yaw_deg"],
            physical["grasp_yaw_deg"],
        )
        self.assertEqual(physical["requested_place_object_yaw_deg"], 10.0)
        self.assertEqual(physical["expected_place_object_yaw_deg"], 30.0)
        self.assertEqual(
            physical["placement_yaw_policy"],
            "preserve_grasp_yaw_until_release",
        )
        self.assertEqual(physical["orientation_policy"], "downward_yaw_only")
        self.assertEqual(len(physical["transport_path"]), 2)
        self.assertEqual(
            physical["transport_path"][1]["yaw_deg"],
            physical["grasp_yaw_deg"],
        )

    def test_16_narrow_yaw_interval_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        def checker(target, objects, yaw):
            safe = 20 <= yaw < 25
            return _geometry_result(safe)
        result = scan_grasp_yaws(current.current_objects[0], current.current_objects, config(), geometry_check=checker)
        self.assertFalse(result.graspable)
        self.assertEqual(result.rejected_narrow_intervals_deg, ((20.0, 25.0),))

    def test_17_single_point_corner_contact_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02], effective_contact_length_m=0.001)])
        self.assertFalse(scan_grasp_yaws(current.current_objects[0], current.current_objects, config()).graspable)

    def test_18_gf225_palm_collision_is_rejected(self):
        current = scene([raw_object(1, "t1", [0.5, 0.0, 0.02])])
        def checker(target, objects, yaw):
            value = _geometry_result(True); value["palm_safe"] = False; return value
        self.assertFalse(scan_grasp_yaws(current.current_objects[0], current.current_objects, config(), geometry_check=checker).graspable)

    def test_detector_ids_are_rebound_to_tracks_per_revision(self):
        memory = {}
        first = [
            raw_object(10, "ignored", [0.40, 0.05, 0.02], label="square red"),
            raw_object(20, "ignored", [0.50, 0.05, 0.02], label="square blue"),
        ]
        for item in first:
            item.pop("track_id")
        update_scene_tracks(memory, first, 1)
        red_track = first[0]["track_id"]
        second = [
            raw_object(99, "ignored", [0.402, 0.05, 0.02], label="square red"),
            raw_object(77, "ignored", [0.502, 0.05, 0.02], label="square blue"),
        ]
        for item in second:
            item.pop("track_id")
        update_scene_tracks(memory, second, 2)
        self.assertEqual(second[0]["track_id"], red_track)
        self.assertEqual(second[0]["object_ref"], "scene_2:obj_99")

    def test_scene_state_uses_measured_color_for_shape_only_detector_label(self):
        obj = raw_object(1, "t1", [0.5, 0.0, 0.02], label="semi square")
        obj["visual_color"] = "yellow"
        state = build_clutter_scene_state(
            {"scene_revision": 1, "objects": [obj]},
            config().workspace,
        )
        self.assertEqual(state.current_objects[0].color, "yellow")

    def test_rebound_held_object_reuses_only_its_previous_valid_size(self):
        memory = {}
        first = raw_object(
            1, "ignored", [0.30, 0.18, 0.00],
            label="square yellow", color="yellow", size=[0.024, 0.023, 0.025],
        )
        first.pop("track_id")
        update_scene_tracks(memory, [first], 1)
        track_id = first["track_id"]
        lifted = raw_object(
            9, "ignored", [0.297, 0.190, 0.094],
            label="square yellow", color="yellow", size=[0.024, 0.023, 0.025],
        )
        lifted.pop("track_id")
        lifted["center_3d_base_m"] = lifted["geometry_center_m"]
        lifted["geometry_center_m"] = None
        lifted["dimensions_m"] = None
        lifted["pointcloud_geometry_valid"] = False
        update_scene_tracks(
            memory,
            [lifted],
            2,
            predicted_centers={track_id: [0.30, 0.18, 0.10]},
        )
        self.assertEqual(lifted["track_id"], track_id)
        self.assertEqual(lifted["dimensions_m"], [0.024, 0.023, 0.025])
        self.assertTrue(lifted["dimensions_temporal_fallback"])
        self.assertEqual(lifted["dimensions_source"], "previous_valid_same_track")
        self.assertGreaterEqual(lifted["track_match_confidence"], 0.5)
        state = build_clutter_scene_state(
            {"scene_revision": 2, "objects": [lifted]},
            config().workspace,
            expected_tracks=[track_id],
        )
        self.assertEqual(state.current_objects[0].size_xyz_m, (0.024, 0.023, 0.025))
        self.assertAlmostEqual(state.current_objects[0].center_xyz_m[2], 0.094)

    def test_new_detection_without_size_still_fails_closed(self):
        invalid = raw_object(1, "ignored", [0.4, 0.0, 0.02])
        invalid.pop("track_id")
        invalid["dimensions_m"] = None
        update_scene_tracks({}, [invalid], 1)
        with self.assertRaisesRegex(ValueError, "object size"):
            build_clutter_scene_state(
                {"scene_revision": 1, "objects": [invalid]},
                config().workspace,
            )


def _geometry_result(safe):
    return {
        "opening_ok": safe, "center_offset_ok": safe, "contact_length_ok": safe,
        "finger_safe": safe, "palm_safe": safe, "descent_safe": safe, "lift_safe": safe,
        "fingertip_clearance_m": 0.02, "palm_clearance_m": 0.02,
        "effective_contact_length_m": 0.02, "vertical_lift_clearance_m": 0.1,
        "blocking_track_ids": [],
    }


def _ollama_args(**changes):
    values = dict(model="qwen3-vl:8b-instruct", ollama_url="http://local/api/chat", vlm_think_mode="off", vlm_num_ctx=0, vlm_num_predict=0, vlm_num_gpu=-1, vlm_read_timeout_sec=1, vlm_keep_alive="1h", vlm_max_backend_retries=1, vlm_max_budget_retries=1, vlm_finalizer_num_predict=768, output_dir=None)
    values.update(changes)
    return SimpleNamespace(**values)


class _response:
    status_code = 200
    text = ""
    def __init__(self, content): self.content = content
    def json(self):
        return {"message": {"role": "assistant", "content": self.content}, "done": True, "done_reason": "stop"}


if __name__ == "__main__":
    unittest.main()
