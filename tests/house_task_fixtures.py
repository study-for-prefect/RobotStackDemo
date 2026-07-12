"""Shared deterministic six-role house fixtures for offline unit tests."""

from robot_scene_pipeline.house_task_definition import (
    canonical_house_assembly_steps,
    canonical_house_goal_spec,
)


HOUSE_CONFIG = {
    "organize_defaults": {
        "grouping_key": "color", "layout_type": "rows", "include_scope": "all_detected_blocks",
        "allow_stacking": False, "minimum_spacing_m": 0.015, "alignment_tolerance_m": 0.012,
    },
    "house_semantics": {
        "structure_variant": "two_column_two_level_roof_triangle",
        "house_up_direction_base_xy": [0.0, 1.0],
        "column_center_tolerance_m": 0.008,
        "column_height_tolerance_m": 0.008,
        "vertical_contact_tolerance_m": 0.006,
        "minimum_support_overlap_ratio": 0.20,
        "orientation_confidence_threshold": 0.75,
        "triangle_center_tolerance_m": 0.010,
        "safe_reorientation_height_m": 0.12,
        "maximum_single_reorientation_step_deg": 20.0,
        "maximum_total_reorientation_deg": 180.0,
    },
}


def house_contract():
    return {
        "schema_version": "task_contract_v1", "task_type": "build_house",
        "goal_spec": canonical_house_goal_spec(), "reason": "fixed six-role house", "confidence": 0.95,
    }


def house_state():
    image_axes = {"image_up_direction_base_xy": [0.0, 1.0], "image_right_direction_base_xy": [1.0, 0.0]}
    return {
        "scene_revision": 4,
        "table_plane": {"z_base_m": 0.0, "normal_base": [0.0, 0.0, 1.0]},
        "table_bounds": {"xmin": 0.1, "xmax": 0.6, "ymin": -0.3, "ymax": 0.3},
        "objects": [
            task_object(1, "square red", [0.30, 0.0, 0.02]),
            task_object(2, "square blue", [0.40, 0.0, 0.02]),
            task_object(3, "square green", [0.30, 0.0, 0.06]),
            task_object(4, "square orange", [0.40, 0.0, 0.06]),
            task_object(5, "concave_rectangle yellow", [0.35, 0.0, 0.09], [0.14, 0.03, 0.02], **image_axes,
                        center_region_height_m=0.010, side_region_heights_m=[0.020, 0.020], orientation_xyzw=[0.0, 0.0, 0.0, 1.0]),
            task_object(6, "triangle purple", [0.35, 0.0, 0.12], [0.03, 0.02, 0.04], **image_axes,
                        contour_inner_angles_deg=[45.0, 45.0, 90.0],
                        orientation_xyzw=[0.0, 0.0, 0.7071067811865475, 0.7071067811865476]),
        ],
    }


def house_plan():
    state = house_state()
    roles = (
        "left_support_lower", "right_support_lower", "left_support_upper",
        "right_support_upper", "roof", "triangle_top",
    )
    assignments = [
        {
            "role_id": role_id, "selected_object_id": obj["id"], "observed_label": obj["label"],
            "geometry_center_base_m": obj["geometry_center_m"], "assignment_status": "temporary", "replaceable": True,
        }
        for role_id, obj in zip(roles, state["objects"])
    ]
    return {
        "schema_version": "grounded_task_plan_v1", "task_type": "build_house", "scene_revision": 4,
        "role_assignments": assignments, "assembly_steps": canonical_house_assembly_steps(),
        "orientation_observations": [
            {
                "role_id": "roof", "selected_object_id": 5, "shape": "concave_rectangle",
                "visible_face": "front", "opening_direction_image": "down",
                "straight_edge_direction_image": "up", "support_legs_visible": True,
                "flip_required": False, "yaw_adjustment_required": False,
                "confidence": 0.95, "reason": "groove and two legs visible",
            },
            {
                "role_id": "triangle_top", "selected_object_id": 6, "shape": "triangle",
                "pose_state": "upright", "apex_direction_image": "up", "apex_ambiguous": False,
                "right_angle_vertex": 2, "apex_vertex": 0, "upward_vertex": 0,
                "flip_required": False, "yaw_adjustment_required": False,
                "confidence": 0.95, "reason": "acute apex is upright",
            },
        ],
    }


def task_object(object_id, label, center, size=None, **extra):
    return {
        "id": object_id, "label": label, "geometry_center_m": list(center),
        "dimensions_m": size or [0.03, 0.03, 0.04], "geometry_frame": "base_link", **extra,
    }
