"""Stable perception/geometry tests retained independently of the deleted VLM planner."""

import unittest

from robot_scene_pipeline.geometry_relations import (
    blocks_grasp,
    build_geometry_relations,
    is_near,
    is_on,
    is_supporting,
    safe_to_push,
)
from robot_scene_pipeline.grasp_yaw_search import normalize_yaw_signed_180


def obj(object_id, center, size=(0.04, 0.04, 0.03), **extra):
    value = {
        "id": object_id,
        "label": str(object_id),
        "geometry_center_m": list(center),
        "dimensions_m": list(size),
        "table_yaw_deg": 0.0,
        "visible": True,
    }
    value.update(extra)
    return value


def analysis(relations):
    return next(item for item in relations if item.get("type") == "target_grasp_analysis")


class GeometryRelationsTests(unittest.TestCase):
    def test_near_relation(self):
        self.assertTrue(is_near(obj("a", (0.4, 0.0, 0.015)), obj("b", (0.43, 0.0, 0.015))))

    def test_on_and_supporting_relations(self):
        lower = obj("lower", (0.4, 0.0, 0.015))
        upper = obj("upper", (0.4, 0.0, 0.045))
        self.assertTrue(is_on(upper, lower))
        self.assertTrue(is_supporting(lower, upper))

    def test_blocking_grasp(self):
        target = obj("target", (0.4, 0.0, 0.015))
        blocker = obj("blocker", (0.47, 0.0, 0.015), (0.06, 0.03, 0.03))
        self.assertTrue(blocks_grasp(blocker, target))

    def test_locked_object_is_not_safe_to_push(self):
        locked = obj("locked", (0.47, 0.0, 0.015), role="base", state="locked", pushable=True)
        self.assertFalse(safe_to_push(locked, [locked], [1.0, 0.0, 0.0]))

    def test_free_object_can_have_safe_push_space(self):
        target = obj("target", (0.4, 0.0, 0.015))
        blocker = obj("blocker", (0.45, 0.0, 0.015), pushable=True)
        self.assertTrue(safe_to_push(
            blocker, [target, blocker], [1.0, 0.0, 0.0], distance_m=0.03,
            table_bounds={"xmin": 0.2, "xmax": 0.65, "ymin": -0.1, "ymax": 0.4},
        ))

    def test_side_block_can_leave_continuous_grasp_yaw(self):
        target = obj("target", (0.4, 0.0, 0.015), (0.03, 0.03, 0.03))
        neighbor = obj("neighbor", (0.44, 0.0, 0.015), (0.03, 0.03, 0.03))
        result = analysis(build_geometry_relations([target, neighbor], target_id="target"))
        self.assertIn("grasp_feasible", result)

    def test_object_on_top_requires_removal(self):
        target = obj("target", (0.4, 0.0, 0.015))
        top = obj("top", (0.4, 0.0, 0.045))
        result = analysis(build_geometry_relations([target, top], target_id="target"))
        self.assertEqual(result["action"], "remove_top_object")

    def test_protected_blocker_does_not_generate_safe_push(self):
        target = obj("target", (0.4, 0.0, 0.015))
        blocker = obj("blocker", (0.44, 0.0, 0.015), role="placed", state="locked")
        relations = build_geometry_relations(
            [target, blocker], target_id="target",
            table_bounds={"xmin": 0.2, "xmax": 0.65, "ymin": -0.1, "ymax": 0.4},
        )
        self.assertFalse(any(item.get("type") == "safe_to_push" for item in relations))

    def test_signed_yaw_normalization(self):
        self.assertAlmostEqual(normalize_yaw_signed_180(179.0), -1.0)


if __name__ == "__main__":
    unittest.main()
