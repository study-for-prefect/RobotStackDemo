"""Command-line arguments for hover-only calibration."""

import argparse
import sys

from .constants import DEFAULT_OUTPUT_DIR

DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_TF_JSON = "/tmp/scene_tf_base_color_optical.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Move to a hover-only calibration pose above a detected block.")
    parser.add_argument("--label", required=True, help="Target label substring, e.g. green, red, yellow.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hover-height", type=float, default=0.08)
    parser.add_argument("--use-tf", action="store_true")
    parser.add_argument("--tf-json", default=DEFAULT_TF_JSON)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--tcp-offset-tool",
        nargs=3,
        type=float,
        default=[-0.015, 0.0, 0.15],
        metavar=("X", "Y", "Z"),
        help="tool0->TCP/gripper-center translation in tool0 coordinates.",
    )
    parser.add_argument(
        "--hover-orientation-mode",
        choices=("grasp_downward", "current", "fixed-yaw"),
        default="grasp_downward",
    )
    parser.add_argument("--fixed-hover-yaw", type=float, default=0.0)
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--safe-pre-rotate-height", type=float, default=0.12)
    parser.add_argument("--conda-env", default="yolo")
    parser.add_argument("--ros-python", default=sys.executable)
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--detector-config", default="config/yolo_detector.json")
    parser.add_argument("--detector-weight", default="")
    parser.add_argument("--score-thresh", type=float, default=None)
    parser.add_argument("--detector-imgsz", type=int, default=None)
    parser.add_argument("--detector-iou", type=float, default=None)
    parser.add_argument("--detector-device", default="")
    parser.add_argument("--object-mask-erode-px", type=int, default=2)
    parser.add_argument("--object-mask-dilate-fallback-px", type=int, default=4)
    parser.add_argument("--object-min-points", type=int, default=40)
    parser.add_argument(
        "--known-object-height-m",
        type=float,
        default=0.0,
        help="Optional measured height prior. Keep 0 for unknown objects; use e.g. 0.0235 for this known square block.",
    )
    parser.add_argument("--velocity", type=float, default=0.08)
    parser.add_argument("--acceleration", type=float, default=0.08)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.20)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.20)
    parser.add_argument(
        "--pre-rotate-strategy",
        choices=("joint-wrist3", "pose"),
        default="joint-wrist3",
    )
    parser.add_argument(
        "--pre-rotate-wrist-yaw-sign",
        choices=("auto", "positive", "negative"),
        default="negative",
        help="UR setup yaw-to-wrist mapping used for safe hover orientation staging.",
    )
    parser.add_argument("--orientation-settle-error-deg", type=float, default=0.2)
    parser.add_argument("--orientation-settle-attempts", type=int, default=2)
    parser.add_argument("--max-grasp-orientation-error-deg", type=float, default=0.5)
    return parser.parse_args()
