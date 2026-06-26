"""Command-line arguments for XY bias collection and analysis."""

import argparse
import sys

from .constants import DEFAULT_OUTPUT_DIR

DEFAULT_TF_JSON = "/tmp/scene_tf_base_color_optical.json"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose base-fixed, yaw-local TCP, and workspace-dependent XY bias."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Run hover trials and record measured signed XY errors.")
    collect.add_argument("--phase", choices=("yaw", "workspace"), required=True)
    collect.add_argument("--label", required=True)
    collect.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    collect.add_argument("--repeats", type=int, default=3)
    collect.add_argument("--yaws", nargs="+", type=float, default=[0.0, 90.0, 180.0, -90.0])
    collect.add_argument(
        "--positions",
        nargs="+",
        default=["center", "left", "right", "front", "back"],
        help="Manual workspace position labels used by --phase workspace.",
    )
    collect.add_argument("--workspace-yaw", type=float, default=0.0)
    collect.add_argument("--hover-height", type=float, default=0.08)
    collect.add_argument("--known-object-height-m", type=float, default=0.0)
    collect.add_argument("--tf-json", default=DEFAULT_TF_JSON)
    collect.add_argument("--detector-config", default="config/yolo_detector.json")
    collect.add_argument("--conda-env", default="yolo")
    collect.add_argument("--ros-python", default=sys.executable)
    collect.add_argument("--tcp-offset-tool", nargs=3, type=float, default=[-0.015, 0.0, 0.15])
    collect.add_argument(
        "--pre-rotate-wrist-yaw-sign",
        choices=("positive", "negative"),
        default="negative",
    )
    collect.add_argument("--execute", action="store_true")
    collect.add_argument("--yes", action="store_true")
    collect.add_argument(
        "--no-position-prompt",
        action="store_true",
        help="Do not pause before each workspace position (mainly for automated runs).",
    )
    collect.add_argument(
        "--measurements-json",
        default="",
        help="Optional non-interactive list of {'error_x_mm','error_y_mm'} entries in trial order.",
    )

    analyze = subparsers.add_parser("analyze", help="Fit diagnostic models from a samples JSON file.")
    analyze.add_argument("samples_json")
    analyze.add_argument("--output", default="")
    return parser.parse_args()
