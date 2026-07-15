#!/usr/bin/env python3
"""Persistent RGB-D perception server for stack demo workflows."""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .depth_geometry import add_depth_args
from .detector_runtime import DetectorModel, add_detector_args
from .io_utils import project_path
from .perception_runtime import process_rgbd_scene
from .ros_topic_capture import RosRgbdSubscriber, add_ros_topic_args
from .tabletop_geometry import add_tabletop_args
from .tf_transform import DEFAULT_TF_POINT_MODE, add_tf_args, frame_matches


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent YOLO + RGB-D perception server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--request-timeout-s", type=float, default=3.0)
    parser.add_argument("--detector-config", default="config/yolo_detector.json")
    add_ros_topic_args(parser)
    add_detector_args(parser)
    add_depth_args(parser)
    add_tf_args(parser)
    add_tabletop_args(parser)
    parser.set_defaults(use_tf=True, estimate_tabletop=True, tf_point_mode=DEFAULT_TF_POINT_MODE, tf_json="")
    return parser.parse_args()


def _cli_flag_present(flag: str) -> bool:
    import sys

    return flag in set(sys.argv[1:])


def apply_detector_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = project_path(getattr(args, "detector_config", ""))
    if not config_path or not os.path.exists(config_path):
        return args
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not _cli_flag_present("--detector-weight") and config.get("weights"):
        args.detector_weight = config["weights"]
    if not _cli_flag_present("--score-thresh") and config.get("conf") is not None:
        args.score_thresh = float(config["conf"])
    if not _cli_flag_present("--detector-iou") and config.get("iou") is not None:
        args.detector_iou = float(config["iou"])
    if not _cli_flag_present("--detector-imgsz") and config.get("imgsz") is not None:
        args.detector_imgsz = int(config["imgsz"])
    if not _cli_flag_present("--detector-device") and config.get("device"):
        args.detector_device = config["device"]
    return args


class PerceptionServerState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.detector = DetectorModel(args)
        self.subscriber = RosRgbdSubscriber(
            args.color_topic,
            args.depth_topic,
            args.camera_info_topic,
            depth_scale_m=args.ros_depth_scale_m,
            node_name="robot_scene_perception_server",
        )
        self._stop = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._request_lock = threading.Lock()

    def start(self) -> None:
        self._spin_thread.start()

    def close(self) -> None:
        self._stop.set()
        self._spin_thread.join(timeout=2.0)
        self.subscriber.close()

    def _spin_loop(self) -> None:
        while not self._stop.is_set():
            self.subscriber._rclpy.spin_once(self.subscriber.node, timeout_sec=0.05)

    def wait_latest_frame(self, timeout_s: float) -> Any:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self.subscriber.ready(require_depth=True):
                return self.subscriber.latest_frame()
            time.sleep(0.02)
        raise TimeoutError("No latest RGB-D frame is ready after {:.2f}s.".format(float(timeout_s)))

    def snapshot(self, query: Dict[str, Any]) -> Dict[str, Any]:
        output_dir = query.get("output_dir") or query.get("output-dir")
        if not output_dir:
            raise ValueError("Missing output_dir query parameter.")
        output_dir = project_path(str(output_dir))
        request_args = SimpleNamespace(**vars(self.args))
        request_args.output_dir = output_dir
        if query.get("tf_json"):
            request_args.tf_json = project_path(str(query["tf_json"]))
        if query.get("camera_frame"):
            request_args.camera_frame = str(query["camera_frame"]).lstrip("/")
        if query.get("score_thresh") is not None:
            requested_threshold = float(query["score_thresh"])
            if not 0.01 <= requested_threshold <= 1.0:
                raise ValueError("score_thresh must be in [0.01, 1.0]")
            request_args.score_thresh = requested_threshold
        frame = self.wait_latest_frame(float(self.args.request_timeout_s))
        source_frame = str((frame.profile or {}).get("coordinate_frame") or "")
        if source_frame and request_args.use_tf and request_args.tf_point_mode == DEFAULT_TF_POINT_MODE:
            if not frame_matches(source_frame, request_args.camera_frame):
                request_args.camera_frame = source_frame
        with self._request_lock:
            private_state, paths = process_rgbd_scene(
                request_args,
                self.detector,
                frame.frame_bgr,
                frame.depth_frame,
                frame.intrinsics,
                frame.profile,
                output_dir,
            )
        return {
            "ok": True,
            "output_dir": output_dir,
            "paths": paths,
            "frame_color_seq": frame.color_seq,
            "object_count": len(private_state.get("objects", [])),
            "private_state": private_state,
        }


def make_handler(state: PerceptionServerState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(200, {"ok": True, "ready": state.subscriber.ready(require_depth=True)})
                return
            if parsed.path != "/snapshot":
                self._send_json(404, {"ok": False, "error": "Unknown path: {}".format(parsed.path)})
                return
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            try:
                self._send_json(200, state.snapshot(query))
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *values: Any) -> None:
            print("[perception_server] " + fmt % values, flush=True)

    return Handler


def main() -> int:
    args = apply_detector_config(parse_args())
    args.detector_weight = project_path(args.detector_weight)
    args.tf_json = project_path(args.tf_json)
    state = PerceptionServerState(args)
    state.start()
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    print(
        "perception_server listening on http://{}:{}  topics=({}, {})".format(
            args.host,
            args.port,
            args.color_topic,
            args.depth_topic,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
