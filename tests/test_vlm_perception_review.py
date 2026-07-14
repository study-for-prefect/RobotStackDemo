import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from robot_scene_pipeline.vlm_perception_review import (
    review_and_recover_expected_misses,
    review_and_recover_low_confidence_candidates,
    review_and_recover_static_misses,
)


class _Result:
    generation_status = "parsed"
    error_message = None
    parsed_decision = {
        "scene_unchanged": True,
        "missing_objects": [{
            "track_id": "track_green_01", "still_visible": True,
            "visual_color": "green", "confidence": 0.96,
        }],
        "confidence": 0.95,
    }


class _ExpectedResult:
    generation_status = "parsed"
    error_message = None
    parsed_decision = {
        "expected_scene_consistent": True,
        "missing_objects": [{
            "track_id": "track_green_01", "still_visible": True,
            "visual_color": "green", "confidence": 0.97,
        }],
        "confidence": 0.96,
    }


class _LowConfidenceResult:
    generation_status = "parsed"
    error_message = None
    parsed_decision = {
        "candidates": [{
            "candidate_id": "track_green_01", "is_real_block": True,
            "visual_color": "green", "confidence": 0.97,
        }],
        "confidence": 0.96,
    }


class VlmPerceptionReviewTests(unittest.TestCase):
    def test_static_missing_detection_is_restored_only_after_vlm_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            image = directory + "/scene.jpg"
            with open(image, "wb") as handle:
                handle.write(b"jpeg")
            previous = _state(image, include_green=True)
            current = _state(image, include_green=False)
            with patch("robot_scene_pipeline.vlm_perception_review.call_policy", return_value=_Result()):
                recovered, report = review_and_recover_static_misses(
                    SimpleNamespace(), previous, current, directory,
                )
        self.assertTrue(report["recovered"])
        green = next(item for item in recovered["objects"] if item.get("visual_color") == "green")
        self.assertTrue(green["vlm_perception_recovered"])
        self.assertEqual(green["geometry_source"], "previous_static_observation_vlm_confirmed")

    def test_consistent_yolo_scene_skips_vlm(self):
        state = _state("unused", include_green=True)
        with patch("robot_scene_pipeline.vlm_perception_review.call_policy") as call:
            recovered, report = review_and_recover_static_misses(
                SimpleNamespace(), state, state, "/tmp",
            )
        call.assert_not_called()
        self.assertFalse(report["triggered"])
        self.assertIs(recovered, state)

    def test_expected_post_action_miss_uses_expected_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            image = directory + "/scene.jpg"
            with open(image, "wb") as handle:
                handle.write(b"jpeg")
            expected = _state(image, include_green=True)
            current = _state(image, include_green=False)
            with patch("robot_scene_pipeline.vlm_perception_review.call_policy", return_value=_ExpectedResult()):
                recovered, report = review_and_recover_expected_misses(
                    SimpleNamespace(), expected, current, directory, "known pick_place",
                )
        self.assertTrue(report["recovered"])
        green = next(item for item in recovered["objects"] if item.get("visual_color") == "green")
        self.assertEqual(green["geometry_center_m"], [0.35, 0.15, 0.0])
        self.assertEqual(green["geometry_source"], "expected_post_action_geometry_vlm_confirmed")

    def test_low_confidence_extra_uses_its_rgbd_geometry_after_vlm_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            image = directory + "/scene.jpg"
            with open(image, "wb") as handle:
                handle.write(b"jpeg")
            primary = _state(image, include_green=False)
            low = _state(image, include_green=True)
            with patch("robot_scene_pipeline.vlm_perception_review.call_policy", return_value=_LowConfidenceResult()):
                recovered, report = review_and_recover_low_confidence_candidates(
                    SimpleNamespace(), primary, low, directory,
                )
        self.assertTrue(report["recovered"])
        green = next(item for item in recovered["objects"] if item.get("visual_color") == "green")
        self.assertEqual(green["geometry_center_m"], [0.35, 0.15, 0.0])
        self.assertEqual(green["geometry_source"], "low_confidence_rgbd_candidate_vlm_confirmed")

    def test_low_confidence_duplicate_location_skips_vlm(self):
        primary = _state("unused", include_green=True)
        low = _state("unused", include_green=True)
        low["objects"].append({
            **low["objects"][-1], "id": 7, "label": "square blue", "confidence": 0.2,
        })
        with patch("robot_scene_pipeline.vlm_perception_review.call_policy") as call:
            recovered, report = review_and_recover_low_confidence_candidates(
                SimpleNamespace(), primary, low, "/tmp",
            )
        call.assert_not_called()
        self.assertFalse(report["triggered"])
        self.assertIs(recovered, primary)

    def test_low_confidence_static_rgbd_jitter_is_not_a_new_object(self):
        primary = _state("unused", include_green=False)
        low = _state("unused", include_green=False)
        low["objects"][0]["geometry_center_m"] = [0.319, 0.137, 0.0]
        with patch("robot_scene_pipeline.vlm_perception_review.call_policy") as call:
            recovered, report = review_and_recover_low_confidence_candidates(
                SimpleNamespace(), primary, low, "/tmp",
            )
        call.assert_not_called()
        self.assertFalse(report["triggered"])
        self.assertIs(recovered, primary)


def _state(image, include_green):
    objects = [{
        "id": 0, "track_id": "track_blue_01", "label": "square blue",
        "visual_color": "blue", "geometry_center_m": [0.31, 0.13, 0.0],
        "dimensions_m": [0.025, 0.025, 0.025], "bbox_xyxy_px": [10, 10, 30, 30],
    }]
    if include_green:
        objects.append({
            "id": 6, "track_id": "track_green_01", "label": "square green",
            "visual_color": "green", "geometry_center_m": [0.35, 0.15, 0.0],
            "dimensions_m": [0.025, 0.025, 0.025], "bbox_xyxy_px": [40, 10, 60, 30],
        })
    return {"objects": objects, "snapshot_image": image, "annotated_image": image}


if __name__ == "__main__":
    unittest.main()
