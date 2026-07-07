#!/usr/bin/env python3

import json
import os
import zipfile
from types import SimpleNamespace

from robot_scene_pipeline.llm_stack_blocks import rule_stack_blocks_decision
from tools.workflows.stack_demo.obstruction_frontier import build_frontier_clearance_plan
from tools.workflows.stack_demo.scene import estimate_current_stack


ZIP_213559 = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute_20260706_213559.zip"
)
ZIP_213757 = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute_20260706_213757.zip"
)
ZIP_77_1 = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute77-1.zip"
)
ZIP_77_2 = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute77-2.zip"
)
ZIP_77_6_LATEST = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute77-6(1).zip"
)
ZIP_77_11 = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute77-11.zip"
)
ZIP_77_12 = (
    "/Users/wujl/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/"
    "wxid_at6wr1ixd8dv22_7973/temp/drag/stack_push_execute77-12.zip"
)


def args():
    return SimpleNamespace(
        grasp_gripper_outer_width_m=0.112,
        grasp_gripper_inner_width_m=0.048,
        grasp_approach_length_m=0.02,
        obstruction_graph_max_depth=3,
        clearance_nudge_distance_m=0.025,
        clearance_frontier_top_k=6,
        clearance_candidate_top_n_per_obstacle=8,
        push_clearing_lift_m=0.05,
        push_clearing_contact_z_offset_m=0.015,
        push_tool_width_m=0.035,
        push_tool_safety_margin_m=0.005,
        search_radius_m=0.06,
    )


def _load_zip_json(zip_path, relative_path):
    with zipfile.ZipFile(zip_path) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        return json.loads(archive.read("{}/{}".format(prefix, relative_path)))


def _cycle_name(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        return sorted({name.split("/")[1] for name in archive.namelist() if "/cycle_" in name})[0]


def test_213757_color_rule_avoids_low_confidence_green_8():
    if not os.path.exists(ZIP_213757):
        print("skip: zip fixture not present")
        return
    state = _load_zip_json(ZIP_213757, "initial_order/private_scene_state.json")
    decision = rule_stack_blocks_decision(state["instruction"], state.get("objects", []))
    assert decision["base_object_id"] == 4
    assert decision["stack_order"][0] == 1
    assert 8 not in decision["full_stack_order"]


def test_213559_replay_generates_enabling_preflight_candidates():
    if not os.path.exists(ZIP_213559):
        print("skip: zip fixture not present")
        return
    cycle = _cycle_name(ZIP_213559)
    state = _load_zip_json(ZIP_213559, "{}/scene_state_before_action.json".format(cycle))
    failure = _load_zip_json(ZIP_213559, "{}/failure_state.json".format(cycle))
    target = next(obj for obj in state["objects"] if str(obj.get("id")) == str(failure["target_object_id"]))
    plan = build_frontier_clearance_plan(state, target, protected_ids=[], args=args())
    assert any(
        candidate.get("enabling_clearance_candidate")
        and candidate.get("approach_path_safe")
        and candidate.get("push_swept_safe")
        for candidate in plan["preflight_clearance_candidates"]
    )


def _replay_preflight_candidates(zip_path):
    cycle = _cycle_name(zip_path)
    state = _load_zip_json(zip_path, "{}/scene_state_before_action.json".format(cycle))
    failure = _load_zip_json(zip_path, "{}/failure_state.json".format(cycle))
    target = next(obj for obj in state["objects"] if str(obj.get("id")) == str(failure["target_object_id"]))
    return build_frontier_clearance_plan(state, target, protected_ids=[], args=args())["preflight_clearance_candidates"]


def test_77_replays_generate_preflight_candidates_for_simple_clearance():
    for zip_path in (ZIP_77_1, ZIP_77_2, ZIP_77_6_LATEST):
        if not os.path.exists(zip_path):
            print("skip: zip fixture not present {}".format(zip_path))
            return
        candidates = _replay_preflight_candidates(zip_path)
        assert candidates
        assert all(candidate.get("clearance_preflight_allowed") for candidate in candidates)
        nudge_candidates = [
            candidate for candidate in candidates
            if candidate.get("action_type") == "nudge"
        ]
        assert all(candidate.get("push_end_safe") for candidate in nudge_candidates)
        assert all(candidate.get("protected_structure_safe") for candidate in nudge_candidates)


def test_77_11_replay_excludes_soft_collision_nudges_from_preflight():
    if not os.path.exists(ZIP_77_11):
        print("skip: zip fixture not present {}".format(ZIP_77_11))
        return
    cycle = _cycle_name(ZIP_77_11)
    state = _load_zip_json(ZIP_77_11, "{}/scene_state_before_action.json".format(cycle))
    selected = _load_zip_json(ZIP_77_11, "{}/selected_action.json".format(cycle))
    target_id = selected["selected_clearance_action"]["target_object_id"]
    target = next(obj for obj in state["objects"] if str(obj.get("id")) == str(target_id))
    candidates = build_frontier_clearance_plan(state, target, protected_ids=[], args=args())["preflight_clearance_candidates"]
    assert candidates
    assert not any(
        candidate.get("action_type") == "nudge"
        and (not candidate.get("approach_path_safe") or not candidate.get("push_swept_safe"))
        for candidate in candidates
    )


def test_77_12_first_stack_estimate_uses_requested_red_base_only():
    if not os.path.exists(ZIP_77_12):
        print("skip: zip fixture not present {}".format(ZIP_77_12))
        return
    state = _load_zip_json(ZIP_77_12, "cycle_01_object_2/scene_state_before_action.json")
    decision = _load_zip_json(ZIP_77_12, "stack_blocks_decision.json")
    base = next(obj for obj in state["objects"] if str(obj.get("id")) == str(decision["base_object_id"]))
    _base, stack = estimate_current_stack(state, base, None, args())
    assert stack["placement_base_object_id"] == decision["base_object_id"]
    assert stack["stack_height_object_count"] == 1


if __name__ == "__main__":
    test_213757_color_rule_avoids_low_confidence_green_8()
    test_213559_replay_generates_enabling_preflight_candidates()
    test_77_replays_generate_preflight_candidates_for_simple_clearance()
    test_77_11_replay_excludes_soft_collision_nudges_from_preflight()
    test_77_12_first_stack_estimate_uses_requested_red_base_only()
    print("stack zip replay tests passed")
