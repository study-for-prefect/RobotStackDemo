import io
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, patch

from robot_scene_pipeline.depth_geometry import build_private_state
from robot_scene_pipeline.perception_contract import (
    PerceptionRequestError,
    PerceptionServerError,
    parse_snapshot_request,
)
from robot_scene_pipeline.perception_server import server_error_response
from tools.workflows.stack_demo.common.scene_state import build_clutter_scene_state
from tools.workflows.stack_demo.perception_client import (
    build_perception_snapshot_request,
    perception_config_mismatches,
    perception_server_snapshot,
)


class PerceptionServerContractTests(unittest.TestCase):
    def test_private_scene_state_does_not_require_task_instruction(self):
        args = SimpleNamespace(
            camera_frame="camera_color_optical_frame",
            base_frame="base_link",
            use_tf=True,
        )
        state = build_private_state(
            args,
            [],
            "/tmp/snapshot.jpg",
            "/tmp/annotated.jpg",
            {"source": "test"},
        )
        self.assertEqual(state["schema_version"], "private_scene_state_v1")
        self.assertTrue(state["scene_id"].startswith("snapshot_"))
        self.assertEqual(state["frame_id"], "base_link")
        self.assertEqual(state["coordinate_frame"], "base_link")
        self.assertNotIn("instruction", state)

    def test_scene_builder_accepts_legacy_snapshot_id_when_base_frame_is_explicit(self):
        scene = build_clutter_scene_state(
            {
                "frame_id": "snapshot_1784103370491",
                "base_frame": "base_link",
                "objects": [],
            },
            {
                "xmin": 0.235,
                "xmax": 0.65,
                "ymin": -0.10,
                "ymax": 0.40,
                "zmin": -0.03,
                "zmax": 0.25,
            },
        )
        self.assertEqual(scene.coordinate_frame, "base_link")

    def test_http_500_preserves_server_json_error_detail(self):
        args = _client_args()
        error = HTTPError(
            "http://127.0.0.1:8765/snapshot",
            500,
            "Internal Server Error",
            {},
            io.BytesIO(b'{"ok":false,"error_code":"scene_processing_failed","error_type":"RuntimeError","error":"RGB-D detail","request_id":"r1"}'),
        )
        with patch("tools.workflows.stack_demo.perception_client.urlopen", side_effect=error):
            with self.assertRaises(PerceptionServerError) as raised:
                perception_server_snapshot(args, "/tmp/output")
        self.assertEqual(raised.exception.status, 500)
        self.assertEqual(raised.exception.error_code, "scene_processing_failed")
        self.assertIn("RGB-D detail", raised.exception.concise_message())

    def test_non_json_http_error_body_is_preserved(self):
        error = HTTPError("http://local/snapshot", 500, "Internal", {}, io.BytesIO(b"raw server failure"))
        with patch("tools.workflows.stack_demo.perception_client.urlopen", side_effect=error):
            with self.assertRaises(PerceptionServerError) as raised:
                perception_server_snapshot(_client_args(), "/tmp/output")
        self.assertIn("raw server failure", str(raised.exception))

    def test_client_request_sends_absolute_tf_and_all_dynamic_frames(self):
        args = _client_args()
        request = build_perception_snapshot_request(args, "relative/output", capture_reason="initial", scene_revision=7)
        self.assertTrue(os.path.isabs(request.output_dir))
        self.assertTrue(os.path.isabs(request.tf_json))
        self.assertEqual(request.base_frame, "base_link")
        self.assertEqual(request.camera_frame, "camera_color_optical_frame")
        self.assertEqual(request.tf_point_mode, "direct")
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"ok":true}'
        with patch(
            "tools.workflows.stack_demo.perception_client.urlopen", return_value=response,
        ) as opened:
            perception_server_snapshot(args, "relative/output", capture_reason="initial", scene_revision=7)
        query = parse_qs(urlparse(opened.call_args.args[0]).query)
        for name in ("output_dir", "tf_json", "base_frame", "camera_frame", "tf_point_mode", "request_id", "capture_reason", "scene_revision"):
            self.assertIn(name, query)
        self.assertTrue(os.path.isabs(query["output_dir"][0]))

    def test_tf_json_missing_and_invalid_have_distinct_errors(self):
        with self.assertRaises(PerceptionRequestError) as missing:
            parse_snapshot_request({
                "output_dir": "/tmp/out", "base_frame": "base_link",
                "camera_frame": "camera_color_optical_frame", "tf_point_mode": "direct",
                "score_thresh": "0.5", "request_id": "r1",
            }, default_tf_max_age_s=30.0)
        self.assertEqual(missing.exception.error_code, "tf_json_missing")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tf.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(PerceptionRequestError) as invalid:
                parse_snapshot_request(_server_query(str(path)), default_tf_max_age_s=30.0)
        self.assertEqual(invalid.exception.error_code, "tf_json_invalid")

    def test_camera_not_ready_maps_to_503(self):
        status, payload = server_error_response(
            TimeoutError("no frame"), request_id="r2", effective_config={"camera": "test"},
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error_code"], "camera_not_ready")
        self.assertEqual(payload["request_id"], "r2")

    def test_detector_mismatch_is_reported(self):
        args = _client_args()
        health = {
            "color_topic": args.color_topic,
            "depth_topic": args.depth_topic,
            "camera_info_topic": args.camera_info_topic,
            "configured_camera_frame": args.camera_frame,
            "base_frame": args.base_frame,
            "tf_point_mode": args.tf_point_mode,
            "detector_weight": str((Path.cwd() / args.detector_weight).resolve()),
            "detector_imgsz": 640,
            "detector_iou": args.detector_iou,
            "detector_device": args.detector_device,
        }
        mismatch = perception_config_mismatches(args, health)
        self.assertEqual(mismatch["detector_imgsz"], {"expected": 960, "actual": 640})


def _client_args():
    return SimpleNamespace(
        perception_server_url="http://127.0.0.1:8765",
        perception_server_timeout_s=15.0,
        tf_max_age_s=30.0,
        tf_json="/tmp/scene_tf_base_color_optical.json",
        base_frame="base_link",
        camera_frame="camera_color_optical_frame",
        tf_point_mode="direct",
        score_thresh=0.5,
        detector_weight="models/yolo/weights/best.pt",
        detector_imgsz=960,
        detector_iou=0.45,
        detector_device="cuda:0",
        color_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/aligned_depth_to_color/image_raw",
        camera_info_topic="/camera/camera/color/camera_info",
    )


def _server_query(tf_path: str):
    return {
        "output_dir": "/tmp/out",
        "tf_json": tf_path,
        "base_frame": "base_link",
        "camera_frame": "camera_color_optical_frame",
        "tf_point_mode": "direct",
        "score_thresh": "0.5",
        "request_id": "r1",
        "client_timeout_s": "15",
        "tf_max_age_s": "30",
    }


if __name__ == "__main__":
    unittest.main()
