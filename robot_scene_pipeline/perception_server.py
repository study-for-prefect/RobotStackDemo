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
from .perception_contract import (
    DetectorProcessingError,
    PerceptionRequestError,
    SceneProcessingError,
    config_fingerprint,
    parse_snapshot_request,
    structured_error_payload,
)
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
    parser.add_argument("--max-tf-age-s", type=float, default=30.0)
    parser.add_argument("--max-rgb-depth-skew-s", type=float, default=0.25)
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
    if not _cli_flag_present("--candidate-score-thresh") and config.get("candidate_conf") is not None:
        args.candidate_score_thresh = float(config["candidate_conf"])
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

    def effective_config(self) -> Dict[str, Any]:
        return {
            "semantic_review_candidate_pool": True,
            "color_topic": str(self.args.color_topic),
            "depth_topic": str(self.args.depth_topic),
            "camera_info_topic": str(self.args.camera_info_topic),
            "configured_camera_frame": str(self.args.camera_frame).lstrip("/"),
            "base_frame": str(self.args.base_frame).lstrip("/"),
            "tf_point_mode": str(self.args.tf_point_mode),
            "detector_weight": os.path.abspath(str(self.args.detector_weight)),
            "detector_imgsz": int(self.args.detector_imgsz),
            "detector_iou": float(self.args.detector_iou),
            "candidate_score_thresh": float(self.args.candidate_score_thresh),
            "detector_device": str(self.args.detector_device),
            "request_timeout_s": float(self.args.request_timeout_s),
            "max_tf_age_s": float(self.args.max_tf_age_s),
            "max_rgb_depth_skew_s": float(self.args.max_rgb_depth_skew_s),
        }

    def health(self) -> Dict[str, Any]:
        status = self.subscriber.status()
        config = self.effective_config()
        ready = bool(status["rgb_ready"] and status["depth_ready"] and status["camera_info_ready"])
        return {
            "ok": True,
            "ready": ready,
            **status,
            "color_topic": config["color_topic"],
            "depth_topic": config["depth_topic"],
            "camera_info_topic": config["camera_info_topic"],
            "source_camera_frame": (
                status.get("camera_info_frame_id")
                or status.get("depth_frame_id")
                or status.get("color_frame_id")
            ),
            **{key: value for key, value in config.items() if key not in {"color_topic", "depth_topic", "camera_info_topic"}},
            "server_pid": os.getpid(),
            "config_fingerprint": config_fingerprint(config),
        }

    def snapshot(self, query: Dict[str, Any]) -> Dict[str, Any]:
        request = parse_snapshot_request(query, default_tf_max_age_s=float(self.args.max_tf_age_s))
        output_dir = request.output_dir
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            raise PerceptionRequestError("invalid_request", f"output_dir cannot be created: {exc}") from exc
        request_args = SimpleNamespace(**vars(self.args))
        request_args.output_dir = output_dir
        request_args.tf_json = request.tf_json
        request_args.base_frame = request.base_frame
        request_args.camera_frame = request.camera_frame
        request_args.tf_point_mode = request.tf_point_mode
        request_args.score_thresh = request.score_thresh
        frame = self.wait_latest_frame(float(self.args.request_timeout_s))
        self._validate_frame(frame, request.camera_frame)
        with self._request_lock:
            try:
                private_state, paths = process_rgbd_scene(
                    request_args,
                    self.detector,
                    frame.frame_bgr,
                    frame.depth_frame,
                    frame.intrinsics,
                    frame.profile,
                    output_dir,
                )
            except (DetectorProcessingError, SceneProcessingError):
                raise
            except Exception as exc:
                raise SceneProcessingError(str(exc)) from exc
        return {
            "ok": True,
            "output_dir": output_dir,
            "paths": paths,
            "frame_color_seq": frame.color_seq,
            "object_count": len(private_state.get("objects", [])),
            "private_state": private_state,
            "request_id": request.request_id,
            "scene_revision": request.scene_revision,
            "capture_reason": request.capture_reason,
            "effective_config": self.effective_config(),
        }

    def _validate_frame(self, frame: Any, expected_camera_frame: str) -> None:
        profile = frame.profile or {}
        source_frame = str(profile.get("coordinate_frame") or "")
        frames = [
            str(profile.get("color_frame_id") or ""),
            str(profile.get("depth_frame_id") or ""),
            str(profile.get("camera_info_frame_id") or ""),
        ]
        present = [value for value in frames if value]
        if source_frame and not frame_matches(source_frame, expected_camera_frame):
            raise PerceptionRequestError(
                "tf_frame_mismatch",
                f"RGB-D source frame {source_frame} does not match requested {expected_camera_frame}",
            )
        if any(not frame_matches(value, expected_camera_frame) for value in present):
            raise PerceptionRequestError(
                "rgb_depth_not_synchronized",
                f"RGB/depth/camera_info frames do not agree: {frames}",
                status=503,
            )
        color_stamp, depth_stamp = profile.get("color_stamp"), profile.get("depth_stamp")
        if color_stamp is None or depth_stamp is None:
            raise PerceptionRequestError(
                "camera_not_ready", "RGB/depth timestamps are unavailable", status=503,
            )
        skew = abs(float(color_stamp) - float(depth_stamp))
        if skew > float(self.args.max_rgb_depth_skew_s):
            raise PerceptionRequestError(
                "rgb_depth_not_synchronized",
                f"RGB/depth timestamp skew {skew:.6f}s exceeds {self.args.max_rgb_depth_skew_s:.6f}s",
                status=503,
            )


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
                self._send_json(200, state.health())
                return
            if parsed.path != "/snapshot":
                self._send_json(404, {"ok": False, "error": "Unknown path: {}".format(parsed.path)})
                return
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            try:
                self._send_json(200, state.snapshot(query))
            except Exception as exc:
                status, payload = server_error_response(
                    exc,
                    request_id=query.get("request_id"),
                    effective_config=state.effective_config(),
                )
                self._send_json(status, payload)

        def log_message(self, fmt: str, *values: Any) -> None:
            print("[perception_server] " + fmt % values, flush=True)

    return Handler


def server_error_response(
    exc: Exception,
    *,
    request_id: str | None,
    effective_config: Dict[str, Any],
) -> tuple[int, Dict[str, Any]]:
    if isinstance(exc, PerceptionRequestError):
        status, error_code = exc.status, exc.error_code
    elif isinstance(exc, TimeoutError):
        status, error_code = 503, "camera_not_ready"
    elif isinstance(exc, DetectorProcessingError):
        status, error_code = 500, "detector_failed"
    elif isinstance(exc, SceneProcessingError):
        status, error_code = 500, "scene_processing_failed"
    else:
        status, error_code = 500, "internal_server_error"
    return status, structured_error_payload(
        error_code, exc, request_id=request_id, effective_config=effective_config,
    )


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
