#!/usr/bin/env python3

import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from robot_scene_pipeline.geometry_relations import get_center, xy_distance
from robot_scene_pipeline.push_grasp_joint_evaluator import predict_scene_after_push
from robot_scene_pipeline.vlm_clearance_policy import build_physical_clearance_candidates
from tools.robot.push_primitives import build_push_targets
from tools.workflows.stack_demo.obstruction_frontier import build_frontier_clearance_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = PROJECT_ROOT / "runtime"


class RuntimeStackPushReplayTests(unittest.TestCase):
    def test_stack_push_execute_replays_have_consistent_push_distances(self):
        run_dirs = sorted(RUNTIME_DIR.glob("stack_push_execute*"))
        if not run_dirs:
            self.skipTest("runtime/stack_push_execute* fixtures are not present")

        checked_nudges = 0
        checked_runs = []
        for run_dir in run_dirs:
            for cycle_dir in sorted(run_dir.glob("cycle_*_object_*")):
                state_path = cycle_dir / "scene_state_before_action.json"
                if not state_path.exists():
                    continue
                state = _load_json(state_path)
                target = _target_for_cycle(state, cycle_dir)
                if target is None:
                    continue
                plan = build_frontier_clearance_plan(state, target, protected_ids=[], args=_args())
                physical = build_physical_clearance_candidates(plan.get("all_clearance_action_candidates", []))
                self.assertNotIn("score", json.dumps(physical, ensure_ascii=False))

                for candidate in plan.get("all_clearance_action_candidates", []):
                    if candidate.get("action_type") != "nudge":
                        continue
                    if candidate.get("direction_base") is None or candidate.get("distance_m") is None:
                        continue
                    obstacle = _object_by_id(state, candidate.get("obstacle_id"))
                    if obstacle is None:
                        continue
                    self._assert_predicted_push_distance_matches_candidate(state, target, obstacle, candidate)
                    checked_nudges += 1
                    if str(run_dir.name) not in checked_runs:
                        checked_runs.append(str(run_dir.name))

        self.assertGreater(checked_nudges, 0, "no nudge candidates were replayed")
        self.assertGreaterEqual(len(checked_runs), 3)

    def _assert_predicted_push_distance_matches_candidate(self, state, target, obstacle, candidate):
        before_center = get_center(obstacle)
        self.assertIsNotNone(before_center)
        predicted = predict_scene_after_push(
            state,
            obstacle.get("id"),
            candidate["direction_base"],
            candidate["distance_m"],
        )
        predicted_obstacle = _object_by_id(predicted, obstacle.get("id"))
        after_center = get_center(predicted_obstacle)
        self.assertIsNotNone(after_center)

        moved_m = math.hypot(after_center[0] - before_center[0], after_center[1] - before_center[1])
        self.assertAlmostEqual(moved_m, float(candidate["distance_m"]), places=6)

        reference_id = candidate.get("distance_reference_object_id", target.get("id"))
        reference_after = _object_by_id(predicted, reference_id) or target
        predicted_distance = xy_distance(predicted_obstacle, reference_after)
        if candidate.get("target_distance_after_m") is not None:
            self.assertAlmostEqual(
                predicted_distance,
                float(candidate["target_distance_after_m"]),
                places=5,
            )

        push_plan = {
            "schema_version": "push_execution_plan_v1",
            "frame_id": "base_link",
            "direction_base": candidate["direction_base"],
            "distance_m": candidate["distance_m"],
            "lift_m": _args().push_clearing_lift_m,
            "contact_z_offset_m": _args().push_clearing_contact_z_offset_m,
            "obstacle": obstacle,
        }
        if candidate.get("feasible"):
            build_push_targets(push_plan)


def _args():
    return SimpleNamespace(
        grasp_gripper_outer_width_m=0.112,
        grasp_gripper_inner_width_m=0.048,
        grasp_gripper_side_clearance_m=0.006,
        grasp_approach_length_m=0.02,
        obstruction_graph_max_depth=3,
        clearance_nudge_distance_m=0.025,
        clearance_frontier_top_k=6,
        clearance_candidate_top_n_per_obstacle=8,
        clearance_safe_place_max_distance_m=0.14,
        clearance_safe_place_avoid_all_visible_objects=True,
        push_clearing_lift_m=0.05,
        push_clearing_contact_z_offset_m=0.015,
        push_tool_width_m=0.035,
        push_tool_safety_margin_m=0.005,
        search_radius_m=0.06,
    )


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _object_by_id(state, object_id):
    for obj in state.get("objects", []):
        if str(obj.get("id")) == str(object_id):
            return obj
    return None


def _target_for_cycle(state, cycle_dir):
    for name in ("selected_clearance_action.json", "failure_state.json", "selected_action.json"):
        path = cycle_dir / name
        if not path.exists():
            continue
        payload = _load_json(path)
        target_id = payload.get("target_object_id")
        if target_id is None:
            selected = payload.get("selected_clearance_action") or {}
            target_id = selected.get("target_object_id")
        target = _object_by_id(state, target_id)
        if target is not None:
            return target
    try:
        target_id = str(cycle_dir.name).split("_object_", 1)[1]
    except IndexError:
        return None
    return _object_by_id(state, target_id)


if __name__ == "__main__":
    unittest.main()
