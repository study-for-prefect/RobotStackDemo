"""Command-line arguments for the two-stage visual pick workflow."""

import argparse

DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_TF_JSON = "/tmp/scene_tf_base_color_optical.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage visual pick with pure base-link XY correction.")
    parser.add_argument("--object-label", default="rectangle")
    parser.add_argument("--object-id", type=int, default=None)
    parser.add_argument("--nearest-base-xy", nargs=2, type=float, default=None)
    parser.add_argument("--first-dir", default="/tmp/current_scene")
    parser.add_argument("--second-dir", default="/tmp/current_scene_second")
    parser.add_argument("--tf-json", default=DEFAULT_TF_JSON)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--ready-pose-json", default="config/rectangle_ready_pose.json")
    parser.add_argument("--conda-env", default="scene_graph_benchmark")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--detector-weight", default="models/yolo/weights/best.pt")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--approach-height-m", type=float, default=0.05)
    parser.add_argument("--pick-target-lift-m", type=float, default=0.010)
    parser.add_argument("--tcp-offset-tool", nargs=3, type=float, default=[0.0, 0.0, 0.15])
    parser.add_argument("--grasp-axis", choices=("long", "short"), default="long")
    parser.add_argument("--yaw-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--force-yaw-labels",
        default="square",
        help="Comma-separated label substrings that should use detected yaw even if aspect-ratio yaw validity is false.",
    )
    parser.add_argument("--pre-rotate-wrist-direction", choices=("positive", "negative"), default="negative")
    parser.add_argument("--max-joint-delta", type=float, default=2.0)
    parser.add_argument("--max-pre-rotate-joint-delta", type=float, default=3.1416)
    parser.add_argument("--max-grasp-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--max-correction-m", type=float, default=0.05)
    parser.add_argument("--max-grasp-offset-m", type=float, default=0.05)
    parser.add_argument("--first-snapshot-retry-count", type=int, default=1)
    parser.add_argument("--first-snapshot-stable-wait-s", type=float, default=0.5)
    parser.add_argument(
        "--pick-observation-offset-camera",
        nargs=3,
        type=float,
        default=[0.0, 0.04, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Optical-camera offset used only when the close pick observation misses the target.",
    )
    parser.add_argument("--stack-demo-mode", action="store_true")
    parser.add_argument("--fixed-square-yaw-deg", type=float, default=0.0)
    parser.add_argument("--stack-square-yaw-mode", choices=("detected", "fixed"), default="detected")
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--xy-correction-json", default="")
    parser.add_argument("--velocity", type=float, default=0.1)
    parser.add_argument("--acceleration", type=float, default=0.1)
    parser.add_argument("--moveit-tf-timeout", type=float, default=10.0)
    parser.add_argument("--second-snapshot-retry-count", type=int, default=2)
    parser.add_argument(
        "--second-snapshot-retry-offset-camera",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Override the retry offset for a failed second snapshot. Defaults to --pick-observation-offset-camera.",
    )
    parser.add_argument(
        "--disable-gripper",
        action="store_false",
        dest="enable_gripper",
        default=True,
        help="Run the final descent without opening/closing the gripper.",
    )
    parser.add_argument(
        "--no-release-after-pick",
        action="store_false",
        dest="release_after_pick",
        default=True,
        help="Keep holding the object after pick instead of putting it back down and opening.",
    )
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--post-close-wait", type=float, default=1.0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to move the robot. Without this flag, only print the workflow commands.",
    )
    args = parser.parse_args()
    if args.second_snapshot_retry_offset_camera is None:
        args.second_snapshot_retry_offset_camera = list(args.pick_observation_offset_camera)
    return args
