"""Command-line arguments for the yaw-only rotation probe."""

import argparse
import math
from pathlib import Path

from robot_scene_pipeline.ros_topic_capture import add_ros_topic_args
from tools.monitoring.realtime_monitor.constants import PROJECT_ROOT

from .pose_math import DEFAULT_YAWS_DEG


def parse_args() -> argparse.Namespace:
    root = Path(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description="Generate yaw-only tool0 target poses and record code-driven detection/TF checks."
    )
    parser.add_argument("--output-dir", default="/tmp/yaw_rotation_probe")
    parser.add_argument("--target-poses-json", default="")
    parser.add_argument(
        "--record-mode",
        choices=("auto", "manual"),
        default="auto",
        help="auto drives each yaw target with MoveIt. manual keeps the old typed-yaw recording mode.",
    )
    parser.add_argument("--yaw-values-deg", nargs="+", type=float, default=list(DEFAULT_YAWS_DEG))

    add_ros_topic_args(parser)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--camera-frame", default="camera_link")
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    parser.add_argument("--tool0-position", nargs=3, type=float, default=None)
    parser.add_argument("--tool0-quat", nargs=4, type=float, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually drive the robot through yaw targets.")
    parser.add_argument("--yes", action="store_true", help="Do not ask MoveIt confirmation for each executed yaw.")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--motion-tool-z-offset", type=float, default=0.15)
    parser.add_argument("--motion-tool-offset-base", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--motion-settle-s", type=float, default=0.6)
    parser.add_argument("--velocity", type=float, default=0.12)
    parser.add_argument("--acceleration", type=float, default=0.12)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.20)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.20)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--pre-rotate-strategy", choices=("joint-wrist3", "pose"), default="joint-wrist3")
    parser.add_argument("--pre-rotate-wrist-yaw-sign", choices=("auto", "positive", "negative"), default="negative")
    parser.add_argument("--pre-rotate-wrist-direction", choices=("auto", "positive", "negative"), default="auto")
    parser.add_argument("--max-joint-delta", type=float, default=1.2)
    parser.add_argument("--max-pre-rotate-joint-delta", type=float, default=math.pi)
    parser.add_argument("--max-grasp-yaw-error-deg", type=float, default=3.0)
    parser.add_argument("--orientation-settle-error-deg", type=float, default=0.2)

    parser.add_argument("--weight", default=str(root / "runs" / "segment" / "building_block5" / "weights" / "best.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument("--max-area", type=int, default=180000)
    parser.add_argument("--target-label-contains", default="")
    parser.add_argument("--ignore-zone", type=int, nargs=4, default=None)
    parser.add_argument("--depth-sample-radius", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--record-attempts", type=int, default=5)
    parser.add_argument("--preview-ms", type=int, default=800)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument(
        "--tf-point-mode",
        choices=("optical-to-camera-link", "direct"),
        default="optical-to-camera-link",
        help="Depth deprojection returns optical XYZ. Use optical-to-camera-link with base<-camera_link TF.",
    )
    parser.add_argument("--show-depth", action="store_true")
    return parser.parse_args()
