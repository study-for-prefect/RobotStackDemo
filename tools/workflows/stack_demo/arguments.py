"""Command-line arguments for the stack demo workflow."""

import argparse
import os

from .constants import PROJECT_ROOT

DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_TF_JSON = "/tmp/scene_tf_base_color_optical.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a per-block closed-loop stack demo.")
    parser.add_argument("--instruction", default="把积木按指定顺序叠起来")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runtime"))
    parser.add_argument("--stack-decision-json", default="")
    parser.add_argument(
        "--force-llm-decision",
        action="store_true",
        help=(
            "Always call the VLM/LLM to parse the initial stack order. "
            "Detected ids are still repaired from explicit color instructions before execution."
        ),
    )
    parser.add_argument("--offline-scene-state", default="")
    parser.add_argument("--base-object-id", type=int, default=None)
    parser.add_argument("--stack-order", nargs="+", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--execute-push-clearing",
        action="store_true",
        help="Execute geometry-based obstacle push clearing before pick when should_push_away is detected.",
    )
    parser.add_argument(
        "--push-clearing-distance-m",
        type=float,
        default=0.05,
        help="Default push distance for obstacle clearing.",
    )
    parser.add_argument(
        "--push-clearing-lift-m",
        type=float,
        default=0.05,
        help="Lift height before and after push clearing.",
    )
    parser.add_argument(
        "--push-clearing-contact-z-offset-m",
        type=float,
        default=0.015,
        help="Contact z offset above table for push clearing.",
    )
    parser.add_argument(
        "--push-clearing-min-contact-z-offset-m",
        type=float,
        default=0.020,
        help=(
            "Minimum closed-gripper TCP contact z offset above the estimated table for push clearing. "
            "This safety floor is only for nudge/push clearing, not normal pick height."
        ),
    )
    parser.add_argument("--push-tool-width-m", type=float, default=0.035)
    parser.add_argument("--push-tool-safety-margin-m", type=float, default=0.005)
    parser.add_argument("--grasp-gripper-side-clearance-m", type=float, default=0.006)
    parser.add_argument("--max-automatic-push-clearing-attempts", type=int, default=4)
    parser.set_defaults(enable_llm_push_selection=True)
    parser.add_argument("--enable-llm-push-selection", dest="enable_llm_push_selection", action="store_true")
    parser.add_argument("--disable-llm-push-selection", dest="enable_llm_push_selection", action="store_false")
    parser.add_argument("--missing-target-clearance-radius-m", type=float, default=0.10)
    parser.add_argument("--high-block-min-top-z-delta-m", type=float, default=0.01)
    parser.add_argument("--obstruction-graph-max-depth", type=int, default=3)
    parser.add_argument("--clearance-nudge-distance-m", type=float, default=0.025)
    parser.add_argument("--clearance-frontier-top-k", type=int, default=6)
    parser.add_argument("--clearance-candidate-top-n-per-obstacle", type=int, default=8)
    parser.add_argument(
        "--debug-dump-full-candidates",
        action="store_true",
        help="Write full clearance candidate debug JSON with nested evaluator details.",
    )
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--conda-env", default="yolo")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--model", default="qwen2.5vl:7b-q4_K_M")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-predict", type=int, default=1024)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--detector-weight", default="models/yolo/weights/best.pt")
    parser.add_argument(
        "--perception-server-url",
        default=os.environ.get("ROBOT_SCENE_PERCEPTION_URL", "http://127.0.0.1:8765"),
        help="Persistent perception server base URL. Use empty string only with --allow-snapshot-subprocess-fallback.",
    )
    parser.add_argument("--perception-server-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--allow-snapshot-subprocess-fallback",
        action="store_true",
        help="Allow legacy snapshot_pipeline subprocess if the persistent perception server is unavailable.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--tf-json", default=DEFAULT_TF_JSON)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--ready-pose-json", default="config/rectangle_ready_pose.json")
    parser.add_argument("--init-stable-wait-s", type=float, default=1.0)
    parser.add_argument("--initial-observation-retry-count", type=int, default=1)
    parser.add_argument("--initial-observation-stable-wait-s", type=float, default=0.5)
    parser.add_argument(
        "--initial-observation-recovery-offsets-base",
        default="0,0,-0.04;0.03,0,-0.02;-0.03,0,-0.02;0,0.03,-0.02;0,-0.03,-0.02",
        help="Semicolon-separated base_link XYZ offsets used between initial observation retries.",
    )
    parser.add_argument("--xy-correction-json", default="")
    parser.add_argument(
        "--calibration-json",
        default="",
        help=(
            "Unified calibration JSON. Supports affine_xy, grasp_base_bias_m, "
            "place_base_bias_m, tcp_offset_tool_m, and max_correction_m. "
            "When set, it supersedes --xy-correction-json for stack pick/place correction."
        ),
    )
    parser.add_argument("--search-radius-m", type=float, default=0.06)
    parser.add_argument("--close-stack-search-radius-m", type=float, default=0.04)
    parser.add_argument("--target-exclusion-radius-m", type=float, default=0.035)
    parser.add_argument("--min-pick-place-xy-distance-m", type=float, default=0.04)
    parser.add_argument("--max-place-second-snapshot-correction-m", type=float, default=0.015)
    parser.add_argument("--held-base-top-z-tolerance-m", type=float, default=0.015)
    parser.add_argument("--approach-height-m", type=float, default=0.05)
    parser.add_argument(
        "--pick-observation-offset-base",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Deprecated compatibility option; close-observation retry now uses optical camera offsets only.",
    )
    parser.add_argument(
        "--pick-observation-offset-camera",
        nargs=3,
        type=float,
        default=[0.0, 0.04, 0.0],
        help="Optical-camera offset used only when the close target observation misses the object.",
    )
    parser.add_argument(
        "--place-observation-offset-camera",
        nargs=3,
        type=float,
        default=[0.0, 0.04, 0.0],
        help="Optical-camera offset used only when the held-object close base observation misses the base.",
    )
    parser.add_argument("--grasp-bias-base", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--grasp-bias-camera", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--pick-target-lift-m", type=float, default=0.010)
    parser.add_argument("--pick-target-z-margin-m", type=float, default=0.002)
    parser.add_argument("--release-gap-m", type=float, default=0.010)
    parser.add_argument("--place-top-z-bias-m", type=float, default=0.0)
    parser.add_argument("--fixed-square-yaw-deg", type=float, default=0.0)
    parser.add_argument("--stack-square-yaw-mode", choices=("detected", "fixed"), default="detected")
    parser.add_argument("--square-yaw-snap-tolerance-deg", type=float, default=0.0)
    parser.add_argument("--grasp-axis", choices=("long", "short"), default="long")
    parser.add_argument("--gripper-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--grasp-gripper-outer-width-m", type=float, default=0.112)
    parser.add_argument("--grasp-gripper-inner-width-m", type=float, default=0.048)
    parser.add_argument(
        "--grasp-approach-length-m",
        type=float,
        default=0.02,
        help="Planar top-grasp envelope extension along the gripper axis; keep small for vertical tabletop grasps.",
    )
    parser.add_argument("--max-grasp-yaw-error-deg", type=float, default=5.0)
    parser.add_argument("--max-grasp-orientation-error-deg", type=float, default=0.5)
    parser.add_argument("--pre-rotate-wrist-yaw-sign", choices=("positive", "negative"), default="negative")
    parser.add_argument("--max-pre-rotate-joint-delta", type=float, default=3.1416)
    parser.add_argument("--max-second-snapshot-correction-m", type=float, default=0.006)
    parser.add_argument("--second-snapshot-max-z-error-m", type=float, default=0.06)
    parser.add_argument("--second-snapshot-hover-above-object-m", type=float, default=0.10)
    parser.add_argument("--max-grasp-offset-m", type=float, default=0.05)
    parser.add_argument(
        "--enable-second-pick-snapshot",
        action="store_true",
        help=(
            "After the first pick approach, capture a close target snapshot and use it only for XY correction. "
            "Default is off so execution picks from the locked first observation."
        ),
    )
    parser.add_argument("--second-snapshot-stable-wait-s", type=float, default=0.5)
    parser.add_argument("--second-snapshot-retry-offset-camera", nargs=3, type=float, default=[0.0, 0.04, 0.0])
    parser.add_argument("--close-observation-retry-count", type=int, default=1)
    parser.add_argument("--close-observation-retry-base-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--scoped-observation-recovery-offsets-base",
        default="0,0,0;0.04,0,0;-0.04,0,0;0,0.04,0;0,-0.04,0",
        help=(
            "Semicolon-separated base_link XYZ offsets for scoped recovery observations, "
            "for example '0,0,0;0.04,0,0;0,0.04,0'."
        ),
    )
    parser.add_argument("--scoped-observation-max-attempts", type=int, default=5)
    parser.add_argument("--scoped-observation-match-distance-m", type=float, default=0.07)
    parser.add_argument("--post-place-match-z-tolerance-m", type=float, default=0.025)
    parser.add_argument(
        "--clearance-safe-place-max-distance-m",
        type=float,
        default=0.14,
        help="Maximum XY distance from the current target for temporary pick-away placement.",
    )
    parser.add_argument("--place-yaw-strategy", choices=("stack", "base", "held"), default="base")
    parser.add_argument("--place-center-strategy", choices=("top", "base"), default="top")
    parser.add_argument("--max-stack-top-center-offset-m", type=float, default=0.015)
    parser.add_argument("--object-offset-tool", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--place-bias-base", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--max-place-bias-base-m", type=float, default=0.01)
    parser.add_argument("--tcp-offset-tool", nargs=3, type=float, default=[0.0, 0.0, 0.15])
    parser.add_argument("--velocity", type=float, default=0.08)
    parser.add_argument("--acceleration", type=float, default=0.08)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.20)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.20)
    parser.add_argument("--ready-max-joint-delta", type=float, default=1.30)
    parser.add_argument("--ready-joint-tolerance", type=float, default=0.15)
    parser.add_argument("--place-velocity", type=float, default=0.03)
    parser.add_argument("--place-acceleration", type=float, default=0.03)
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument(
        "--memory-json",
        default=None,
        help="Path to scene memory json. Default: <output-dir>/scene_memory.json",
    )

    parser.add_argument(
        "--resume-memory",
        action="store_true",
        help="Resume existing scene memory instead of starting from current run.",
    )
    return parser.parse_args()
