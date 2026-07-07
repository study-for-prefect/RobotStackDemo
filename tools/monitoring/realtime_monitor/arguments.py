"""Command-line arguments for realtime scene monitoring."""

import argparse
from pathlib import Path

from robot_scene_pipeline.ros_topic_capture import add_ros_topic_args

from .constants import PROJECT_ROOT

DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_TF_POINT_MODE = "direct"
DEFAULT_TF_JSON = "/tmp/scene_tf_base_color_optical.json"

def parse_args():
    root = Path(PROJECT_ROOT)
    parser = argparse.ArgumentParser("Realtime YOLO-seg monitor for RealSense with base_link 3D coordinates")
    parser.add_argument("--weight", default=str(root / "runs" / "segment" / "building_block5" / "weights" / "best.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=960)
    add_ros_topic_args(parser)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--frame-timeout-ms", type=int, default=5000)
    parser.add_argument("--rs-exposure", type=float, default=-1.0)
    parser.add_argument("--rs-gain", type=float, default=-1.0)
    parser.add_argument(
        "--rs-depth-preset",
        choices=("none", "default", "high_accuracy", "high_density", "medium_density"),
        default="high_accuracy",
    )
    parser.add_argument("--rs-emitter-enabled", type=int, default=1)
    parser.add_argument("--rs-laser-power", type=float, default=-1.0)
    parser.add_argument("--require-usb3", action="store_true", dest="require_usb3", default=True)
    parser.add_argument("--allow-usb2", action="store_false", dest="require_usb3")
    parser.add_argument("--allow-low-fps-fallback", action="store_true", default=False)

    # 只过滤夹爪区域。默认按当前分辨率自动取右下角区域。
    parser.add_argument("--ignore-zone", type=int, nargs=4, default=None)

    parser.add_argument("--min-area", type=int, default=2000)
    parser.add_argument("--max-area", type=int, default=180000)
    parser.add_argument("--show-depth", action="store_true")

    # TF bridge output:
    # python3 tools/robot/tf_lookup_json.py --base-frame base_link --camera-frame camera_color_optical_frame --output /tmp/scene_tf_base_color_optical.json
    parser.add_argument("--tf-json", default=DEFAULT_TF_JSON)
    parser.add_argument("--tf-reload-s", type=float, default=0.2)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument(
        "--tf-point-mode",
        choices=("optical-to-camera-link", "direct"),
        default=DEFAULT_TF_POINT_MODE,
        help=(
            "RealSense deprojection is optical-frame XYZ. Use direct when "
            "tf-json is base_link<-the optical frame reported by camera_info; "
            "optical-to-camera-link is legacy for base_link<-camera_link."
        ),
    )
    parser.add_argument("--depth-sample-radius", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--monitor-detail", choices=("compact", "verbose"), default="compact")
    parser.add_argument("--hide-camera-coord", action="store_true")
    parser.add_argument("--estimate-tabletop", dest="estimate_tabletop", action="store_true", default=True)
    parser.add_argument("--no-estimate-tabletop", dest="estimate_tabletop", action="store_false")
    parser.add_argument(
        "--known-block-height-m",
        type=float,
        default=0.0,
        help="Optional physical-height prior. Default 0 disables inferred height/topZ for unknown objects.",
    )
    parser.add_argument("--plane-point-stride", type=int, default=8)
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.006)
    parser.add_argument("--plane-ransac-iterations", type=int, default=120)
    parser.add_argument("--plane-min-inliers", type=int, default=300)
    parser.add_argument("--plane-max-depth-m", type=float, default=2.0)
    parser.add_argument("--plane-min-up-alignment", type=float, default=0.70)
    parser.add_argument("--object-point-stride", type=int, default=2)
    parser.add_argument("--object-min-height-m", type=float, default=0.004)
    parser.add_argument("--object-max-height-m", type=float, default=0.20)
    parser.add_argument("--object-min-points", type=int, default=25)
    parser.add_argument("--object-mask-erode-px", type=int, default=2)
    parser.add_argument("--object-mask-dilate-fallback-px", type=int, default=4)
    parser.add_argument("--object-outlier-percentile", type=float, default=2.0)
    parser.add_argument("--yaw-min-aspect-ratio", type=float, default=1.20)
    return parser.parse_args()
