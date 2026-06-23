"""Command-line arguments for perception stability diagnostics."""

import argparse

from robot_scene_pipeline.depth_geometry import add_depth_args
from robot_scene_pipeline.detector_runtime import add_detector_args
from robot_scene_pipeline.realsense_capture import add_realsense_args
from robot_scene_pipeline.tabletop_geometry import add_tabletop_args
from robot_scene_pipeline.tf_transform import add_tf_args

from .constants import DEFAULT_OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture a fixed scene repeatedly and summarize perception stability."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--instruction", default="perception stability probe")
    parser.add_argument("--detector-config", default="config/yolo_detector.json")
    parser.add_argument("--target-object-id", type=int, default=None)
    parser.add_argument(
        "--target-label-contains",
        default="",
        help="Optional case-insensitive label substring filter for recorded objects.",
    )
    parser.add_argument("--pre-capture-delay", type=float, default=0.0)
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    parser.add_argument("--xy-std-threshold-m", type=float, default=0.003)
    parser.add_argument("--top-z-std-threshold-m", type=float, default=0.003)
    parser.add_argument("--dimension-range-threshold-m", type=float, default=0.004)
    parser.add_argument("--yaw-large-jitter-deg", type=float, default=5.0)
    parser.add_argument(
        "--spatial-cluster-radius-m",
        type=float,
        default=0.02,
        help="XY radius used to summarize records as physical object clusters.",
    )
    add_realsense_args(parser)
    add_detector_args(parser)
    add_depth_args(parser)
    add_tf_args(parser)
    add_tabletop_args(parser)
    parser.add_argument("--no-use-tf", action="store_false", dest="use_tf")
    parser.add_argument("--no-estimate-tabletop", action="store_false", dest="estimate_tabletop")
    parser.set_defaults(use_tf=True, estimate_tabletop=True)
    return parser.parse_args()
