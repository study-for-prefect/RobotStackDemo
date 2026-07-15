import unittest

import cv2
import numpy as np

from robot_scene_pipeline.object_semantics import infer_object_color
from robot_scene_pipeline.visual_color import attach_visual_colors, object_label_contains


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
        for index, (_, bgr) in enumerate(colors.items()):
            left, right = index * 100 + 10, index * 100 + 90
            cv2.rectangle(image, (left, 10), (right, 90), bgr, -1)
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[10:91, left:right + 1] = True
            detections.append({
                "id": index,
                "label": "uncolored shape",
                "bbox": [left, 10, right, 90],
                "_mask_bool": mask,
            })
        attach_visual_colors(detections, image)
        self.assertEqual([item.get("visual_color") for item in detections], list(colors))
        self.assertTrue(
            all(item.get("visual_color_source") == "instance_mask_hsv" for item in detections)
        )

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
        self.assertEqual(infer_object_color(obj), "red")

    def test_shape_only_label_uses_measured_visual_color(self):
        self.assertEqual(
            infer_object_color({"label": "semi square", "visual_color": "yellow"}),
            "yellow",
        )


if __name__ == "__main__":
    unittest.main()
