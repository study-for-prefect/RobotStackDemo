from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from tools.robot.moveit_preview.arguments import parse_args as parse_moveit_args
from tools.robot.moveit_preview.orientation import tool0_goal_from_tcp
from tools.robot.tool_geometry import TOOL0_TO_TCP_OFFSET_TOOL_M
from tools.workflows.stack_demo.app import _apply_task_marks
from tools.workflows.stack_demo.arguments import parse_args as parse_stack_args
from tools.workflows.stack_demo.clutter.edge_generation import (
    _nudge_edges,
    generate_physical_edges,
)
from tools.workflows.stack_demo.clutter.grasp_edges import GraspScanResult
from tools.workflows.stack_demo.clutter.path_safety import (
    build_nudge_parameters,
    nudge_sweep_checks,
)
from tools.workflows.stack_demo.common.moveit_adapter import MoveItEdgeAdapter
from tools.workflows.stack_demo.organize.placement import build_color_target_regions
from tools.workflows.stack_demo.organize.state import build_organize_task_state

from tests.new_arch_fixtures import config, placement, raw_object, scene


class ColorRegionTests(unittest.TestCase):
    def test_color_regions_have_fixed_width_and_clearing_channels(self):
        current = scene([raw_object(1, "green", [0.35, 0.12, 0.02], color="green")])
        regions = build_color_target_regions(current, config(), {"green": "green"})
        ordered = [regions[color]["bounds_base_m"] for color in ("red", "green", "blue", "yellow")]
        for bounds in ordered:
            self.assertAlmostEqual(bounds["ymax"] - bounds["ymin"], 0.035)
        for first, second in zip(ordered, ordered[1:]):
            self.assertAlmostEqual(second["ymin"] - first["ymax"], 0.010)

    def test_default_clutter_is_not_completed_by_old_large_region(self):
        current = scene([
            raw_object(1, "green", [0.3677, 0.1182, -0.001], color="green", size=[0.0238, 0.0197, 0.0226]),
        ])
        state = build_organize_task_state(current, config())
        self.assertEqual(state.completed_tracks, ())
        self.assertEqual(state.unresolved_tracks, ("green",))

    def test_center_inside_without_footprint_inside_is_not_complete(self):
        current = scene([
            raw_object(1, "red", [0.30, 0.232, 0.02], color="red", size=[0.03, 0.03, 0.04]),
        ])
        state = build_organize_task_state(current, config())
        self.assertEqual(state.completed_tracks, ())

    def test_completed_organize_tracks_become_actual_protected_objects(self):
        current = scene([
            raw_object(1, "red", [0.30, 0.2475, 0.02], color="red", size=[0.02, 0.02, 0.04]),
        ])
        state = build_organize_task_state(current, config())
        marked = _apply_task_marks(current, state.completed_tracks, state.completed_tracks, ())
        self.assertEqual(marked.protected_tracks, ("red",))
        self.assertTrue(marked.current_objects[0].protected)

    def test_empty_color_regions_allow_same_and_other_color_nudge_candidates(self):
        same = self._nudge_candidates("red", "red", 0.219, 0.23, 0.265)
        self.assertTrue(any(edge.task_progress_gain == 1.0 for edge in same))
        other = self._nudge_candidates("red", "green", 0.246, 0.275, 0.31)
        self.assertTrue(other)
        self.assertTrue(any(
            edge.decision_metadata["destination_color_region"] == "green" for edge in other
        ))

    def test_completed_object_swept_collision_is_rejected_by_geometry(self):
        current = scene([
            raw_object(1, "blocker", [0.35, 0.25, 0.02], color="red"),
            raw_object(2, "done", [0.39, 0.25, 0.02], color="blue"),
        ], completed=["done"], protected=["done"])
        blocker = current.object_by_track("blocker")
        physical = build_nudge_parameters(blocker, [1.0, 0.0, 0.0], 0.04, 0.0, config())
        checks = nudge_sweep_checks(current, blocker, physical, config())
        self.assertFalse(checks["passed"])
        self.assertIn("protected_completed_object_collision", checks["rejection_reasons"])

    def _nudge_candidates(
        self,
        blocker_color: str,
        region_color: str,
        blocker_y: float,
        region_ymin: float,
        region_ymax: float,
    ):
        current = scene([
            raw_object(1, "target", [0.35, blocker_y, 0.02], color="blue"),
            raw_object(2, "blocker", [0.30, blocker_y, 0.02], color=blocker_color),
        ])
        current = replace(current, target_regions=({
            "region_id": f"organize_{region_color}",
            "color": region_color,
            "bounds_base_m": {
                "xmin": 0.25, "xmax": 0.42,
                "ymin": region_ymin, "ymax": region_ymax,
            },
        },))
        target = current.object_by_track("target")
        blocker = current.object_by_track("blocker")
        scan = GraspScanResult("target", (), (), (), ("blocker",))
        passing = {
            "passed": True,
            "geometry_checks_passed": True,
            "prepush_reachable": True,
            "prepush_descent_safe": True,
            "horizontal_sweep_safe": True,
            "push_end_safe": True,
            "protected_safe": True,
            "rejection_reasons": [],
        }
        with patch(
            "tools.workflows.stack_demo.clutter.edge_generation.nudge_sweep_checks",
            return_value=passing,
        ):
            return _nudge_edges(
                current, target, blocker, "organize_blocks", config(),
                lambda edge: {"passed": True, "moveit_plan_only": True}, scan,
            )


class TcpOffsetTests(unittest.TestCase):
    def test_cli_config_and_moveit_defaults_are_016(self):
        self.assertEqual(parse_stack_args([]).tcp_offset_tool, [0.0, 0.0, 0.16])
        self.assertEqual(config().section("gripper")["tcp_offset_tool_m"], [0.0, 0.0, 0.16])
        with patch.object(sys, "argv", ["moveit_plan_preview.py"]):
            moveit_args = parse_moveit_args()
            self.assertEqual(moveit_args.tcp_offset_tool, [0.0, 0.0, 0.16])
            self.assertEqual(moveit_args.max_wrist_3_start_goal_delta, 1.75)
        self.assertEqual(TOOL0_TO_TCP_OFFSET_TOOL_M, (0.0, 0.0, 0.16))

    def test_workflow_passes_one_tcp_offset_to_moveit(self):
        args = type("Args", (), {
            "ros_python": "/usr/bin/python3", "tool_frame": "tool0", "tf_timeout": 8.0,
        })()
        adapter = MoveItEdgeAdapter(args, config(), Path("/tmp/tcp_offset_test"))
        command = adapter._push_command(Path("push.json"), execute=False)
        self.assertEqual(command.count("--tcp-offset-tool"), 1)
        index = command.index("--tcp-offset-tool")
        self.assertEqual([float(value) for value in command[index + 1:index + 4]], [0.0, 0.0, 0.16])

    def test_tool0_goal_applies_offset_once(self):
        tcp_target = [0.50, 0.10, 0.30]
        goal = tool0_goal_from_tcp(tcp_target, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.16])
        for actual, expected in zip(goal, [0.50, 0.10, 0.14]):
            self.assertAlmostEqual(actual, expected)
        self.assertNotEqual(goal[2], -0.02)

    def test_grasp_height_semantics_do_not_include_tcp_offset(self):
        current = scene([raw_object(1, "target", [0.50, 0.0, 0.02])])
        generated = generate_physical_edges(
            current, ["target"], "organize_blocks", config(), placement, lambda obj: None,
        )
        physical = generated.edges_by_target["target"][0].physical_parameters
        self.assertAlmostEqual(physical["grasp_pose"]["position_m"][2], 0.02)
        self.assertAlmostEqual(physical["approach_pose"]["position_m"][2], 0.07)
        self.assertAlmostEqual(physical["lift_pose"]["position_m"][2], 0.12)


if __name__ == "__main__":
    unittest.main()
