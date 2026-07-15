import unittest

from tools.workflows.stack_demo.house.completion import evaluate_house_completion
from tools.workflows.stack_demo.house.planner import _candidate_role_specs
from tools.workflows.stack_demo.house.orientation import (
    RoofOrientationState,
    TriangleOrientationState,
    roof_direct_place_valid,
    triangle_direct_place_valid,
)
from tools.workflows.stack_demo.house.roles import legal_incomplete_roles
from tools.workflows.stack_demo.house.state import build_house_task_state

from tests.new_arch_fixtures import config, raw_object, scene


class NewHousePlannerTests(unittest.TestCase):
    def test_25_left_upper_requires_left_lower(self):
        completion = _completion()
        self.assertNotIn("left_support_upper", legal_incomplete_roles(completion))
        completion["left_support_lower"] = True
        self.assertIn("left_support_upper", legal_incomplete_roles(completion))

    def test_26_right_upper_requires_right_lower(self):
        completion = _completion()
        self.assertNotIn("right_support_upper", legal_incomplete_roles(completion))
        completion["right_support_lower"] = True
        self.assertIn("right_support_upper", legal_incomplete_roles(completion))

    def test_27_roof_requires_both_upper_supports(self):
        completion = _completion()
        completion.update(left_support_lower=True, right_support_lower=True, left_support_upper=True)
        self.assertNotIn("roof", legal_incomplete_roles(completion))
        completion["right_support_upper"] = True
        self.assertIn("roof", legal_incomplete_roles(completion))

    def test_28_triangle_requires_roof(self):
        completion = _completion()
        self.assertNotIn("triangle_top", legal_incomplete_roles(completion))
        completion["roof"] = True
        self.assertIn("triangle_top", legal_incomplete_roles(completion))

    def test_29_wrong_roof_groove_face_cannot_place(self):
        roof = _valid_roof(groove_face_state="opening_up", face_up=False)
        valid, checks = roof_direct_place_valid(roof, 0.0, config())
        self.assertFalse(valid)
        self.assertFalse(checks["groove_face_correct"])

    def test_30_wrong_roof_long_axis_cannot_place(self):
        roof = _valid_roof(long_axis_yaw_deg=90.0)
        valid, checks = roof_direct_place_valid(roof, 0.0, config())
        self.assertFalse(valid)
        self.assertFalse(checks["long_axis_matches_support_span"])

    def test_31_insufficient_roof_support_coverage_is_rejected(self):
        roof = _valid_roof(left_support_coverage_m=0.001)
        valid, checks = roof_direct_place_valid(roof, 0.0, config())
        self.assertFalse(valid)
        self.assertFalse(checks["covers_left_support"])

    def test_32_triangle_apex_direction_must_be_up(self):
        triangle = _valid_triangle(apex_direction="down")
        valid, checks = triangle_direct_place_valid(triangle, config())
        self.assertFalse(valid)
        self.assertFalse(checks["apex_up"])

    def test_33_triangle_base_contact_must_be_valid(self):
        triangle = _valid_triangle(base_contact=False)
        valid, checks = triangle_direct_place_valid(triangle, config())
        self.assertFalse(valid)
        self.assertFalse(checks["base_contact_valid"])

    def test_34_completed_house_tracks_become_protected(self):
        current = scene(_completed_house_objects(), expected=[f"t{index}" for index in range(1, 7)])
        state = build_house_task_state(current, config())
        self.assertEqual(set(state.protected_structure_tracks), {f"t{index}" for index in range(1, 7)})
        self.assertTrue(evaluate_house_completion(current, state)["task_complete"])

    def test_all_legal_role_track_pairs_are_exposed_without_order_binding(self):
        current = scene([
            raw_object(1, "a", [0.40, 0.00, 0.02], label="square red"),
            raw_object(2, "b", [0.50, 0.00, 0.02], label="square blue"),
        ])
        state = build_house_task_state(current, config())
        specs = _candidate_role_specs(
            state, ("left_support_lower", "right_support_lower"),
        )
        self.assertEqual(
            {(item.track_id, item.task_role) for item in specs},
            {
                ("a", "left_support_lower"), ("a", "right_support_lower"),
                ("b", "left_support_lower"), ("b", "right_support_lower"),
            },
        )


def _completion():
    return {
        "left_support_lower": False,
        "right_support_lower": False,
        "left_support_upper": False,
        "right_support_upper": False,
        "roof": False,
        "triangle_top": False,
    }


def _valid_roof(**changes):
    values = dict(
        track_id="roof",
        groove_face_state="opening_down",
        face_up=True,
        long_axis_yaw_deg=0.0,
        current_grasp_pose=None,
        satisfies_roof_orientation=True,
        left_support_coverage_m=0.01,
        right_support_coverage_m=0.01,
        center_offset_m=0.0,
        left_support_margin_m=0.01,
        right_support_margin_m=0.01,
        placement_stable=True,
    )
    values.update(changes)
    return RoofOrientationState(**values)


def _valid_triangle(**changes):
    values = dict(
        track_id="triangle",
        apex_direction="up",
        base_edge_direction="roof_aligned",
        face_state="upright",
        target_yaw_deg=0.0,
        base_contact=True,
        center_of_mass_projection_m=0.001,
        support_margin_m=0.01,
        roof_relative_pose={"centered": True},
    )
    values.update(changes)
    return TriangleOrientationState(**values)


def _completed_house_objects():
    roles = (
        "left_support_lower", "right_support_lower", "left_support_upper",
        "right_support_upper", "roof", "triangle_top",
    )
    labels = (
        "square red", "square blue", "square green", "square yellow",
        "concave_rectangle orange", "triangle purple",
    )
    centers = (
        [0.31, 0.18, 0.02], [0.41, 0.18, 0.02],
        [0.31, 0.18, 0.06], [0.41, 0.18, 0.06],
        [0.36, 0.18, 0.09], [0.36, 0.18, 0.12],
    )
    sizes = (
        [0.03, 0.03, 0.04], [0.03, 0.03, 0.04],
        [0.03, 0.03, 0.04], [0.03, 0.03, 0.04],
        [0.14, 0.03, 0.02], [0.03, 0.03, 0.04],
    )
    extras = ({}, {}, {}, {}, {
        "groove_face_state": "opening_down", "face_up": True,
        "satisfies_roof_orientation": True, "long_axis_yaw_deg": 0.0,
    }, {
        "apex_direction": "up", "base_edge_direction": "roof_aligned",
        "face_state": "upright", "center_of_mass_projection_m": 0.0,
    })
    objects = []
    for index, (role, label, center, size, extra) in enumerate(
        zip(roles, labels, centers, sizes, extras), start=1,
    ):
        objects.append(raw_object(
            index, f"t{index}", center, label=label, size=size,
            house_role=role, **extra,
        ))
    return objects


if __name__ == "__main__":
    unittest.main()
