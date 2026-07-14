import unittest

import cv2
import numpy as np

from robot_scene_pipeline.llm_stack_blocks import object_label_contains
from robot_scene_pipeline.organize_scope import color_value_from_object, organize_scope_objects
from robot_scene_pipeline.visual_color import attach_visual_colors


class VisualColorTests(unittest.TestCase):
    def test_supported_colors_are_recovered_from_masks(self):
        image = np.zeros((100, 400, 3), dtype=np.uint8)
        colors = {
            "red": (0, 0, 180),
            "green": (0, 140, 0),
            "blue": (180, 80, 0),
            "yellow": (0, 190, 220),
        }
        detections = []
        for index, (name, bgr) in enumerate(colors.items()):
            left, right = index * 100 + 10, index * 100 + 90
            cv2.rectangle(image, (left, 10), (right, 90), bgr, -1)
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[10:91, left:right + 1] = True
            detections.append({
                "id": index, "label": "uncolored shape", "bbox": [left, 10, right, 90],
                "_mask_bool": mask,
            })
        attach_visual_colors(detections, image)
        self.assertEqual([item.get("visual_color") for item in detections], list(colors))
        self.assertTrue(all(item.get("visual_color_source") == "instance_mask_hsv" for item in detections))

    def test_bbox_fallback_and_gray_rejection(self):
        image = np.full((80, 160, 3), 128, dtype=np.uint8)
        image[10:70, 10:70] = (0, 180, 220)
        detections = [
            {"id": 0, "label": "semi circle", "bbox": [5, 5, 75, 75]},
            {"id": 1, "label": "semi square", "bbox": [85, 5, 155, 75]},
        ]
        attach_visual_colors(detections, image)
        self.assertEqual(detections[0].get("visual_color"), "yellow")
        self.assertNotIn("visual_color", detections[1])

    def test_visual_color_is_authoritative_for_color_matching(self):
        obj = {"label": "square blue", "visual_color": "red"}
        self.assertTrue(object_label_contains(obj, "red"))
        self.assertFalse(object_label_contains(obj, "blue"))
        self.assertEqual(color_value_from_object(obj), "red")

    def test_organize_scope_accepts_shape_only_label_with_visual_color(self):
        obj = {
            "id": 7, "label": "semi square", "visual_color": "red", "confidence": 0.9,
            "bbox_xyxy_px": [20, 20, 80, 80],
            "geometry_center_m": [0.4, 0.1, 0.02], "dimensions_m": [0.04, 0.04, 0.04],
        }
        state = {
            "objects": [obj], "camera_profile": {"color_width": 640, "color_height": 480},
            "workspace_bounds": {"xmin": 0.2, "xmax": 0.65, "ymin": -0.1, "ymax": 0.4},
        }
        self.assertEqual([item["id"] for item in organize_scope_objects(state)], [7])

    def test_edge_detection_is_kept_when_rgbd_geometry_is_usable(self):
        obj = {
            "id": 8, "label": "square yellow", "visual_color": "yellow",
            "confidence": 0.9, "bbox_xyxy_px": [0, 20, 42, 78],
            "geometry_center_m": [0.445, 0.335, -0.001],
            "dimensions_m": [0.023, 0.020, 0.025],
            "pointcloud_geometry_valid": True,
            "depth_geometry_observable": True,
        }
        state = {
            "objects": [obj], "camera_profile": {"color_width": 640, "color_height": 480},
            "workspace_bounds": {
                "xmin": 0.235, "xmax": 0.65, "ymin": -0.1, "ymax": 0.4,
            },
        }
        self.assertEqual([item["id"] for item in organize_scope_objects(state)], [8])

    def test_edge_detection_is_rejected_when_rgbd_geometry_is_invalid(self):
        obj = {
            "id": 9, "label": "square yellow", "visual_color": "yellow",
            "confidence": 0.9, "bbox_xyxy_px": [0, 20, 42, 78],
            "geometry_center_m": [0.445, 0.335, -0.001],
            "dimensions_m": [0.023, 0.020, 0.025],
            "pointcloud_geometry_valid": False,
            "depth_geometry_observable": True,
        }
        state = {
            "objects": [obj], "camera_profile": {"color_width": 640, "color_height": 480},
            "workspace_bounds": {
                "xmin": 0.235, "xmax": 0.65, "ymin": -0.1, "ymax": 0.4,
            },
        }
        self.assertEqual(organize_scope_objects(state), [])


if __name__ == "__main__":
    unittest.main()
