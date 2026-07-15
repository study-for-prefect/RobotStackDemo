"""Command-line arguments for the two supported closed-loop tasks."""

from __future__ import annotations

import argparse
import os

from .constants import PROJECT_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan organize_blocks or build_house using code-generated physical edges.",
    )
    parser.add_argument("--instruction", default="按颜色整理积木")
    parser.add_argument("--task-type", choices=("auto", "organize_blocks", "build_house"), default="auto")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runtime", "stack_demo"))
    parser.add_argument("--planner-config", default="config/stack_demo_planner.json")
    parser.add_argument("--workspace-bounds-json", default="config/workspace_bounds.json")
    parser.add_argument("--offline-scene-state", default="")
    parser.add_argument("--mock-policy-response-dir", default="")
    parser.add_argument("--max-task-steps", type=int, default=20)

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--moveit-plan-only", action="store_true")
    parser.add_argument(
        "--execute-push-clearing", action="store_true",
        help="Additional opt-in required before executing a selected nudge edge.",
    )

    parser.add_argument("--model", default="", help="Override planner-config policy.model.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--vlm-think-mode", choices=("off",), default="off")
    parser.add_argument("--vlm-num-ctx", type=int, default=0)
    parser.add_argument("--vlm-num-predict", type=int, default=0)
    parser.add_argument("--vlm-num-gpu", type=int, default=-1)
    parser.add_argument("--vlm-finalizer-num-predict", type=int, default=768)
    parser.add_argument("--vlm-read-timeout-sec", type=float, default=120.0)
    parser.add_argument("--vlm-keep-alive", default="1h")
    parser.add_argument("--vlm-max-backend-retries", type=int, default=2)
    parser.add_argument("--vlm-max-budget-retries", type=int, default=2)
    parser.add_argument("--unload-model-after-task", action="store_true")
    parser.add_argument("--no-image", action="store_true")

    parser.add_argument("--perception-server-url", default=os.environ.get("ROBOT_SCENE_PERCEPTION_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--perception-server-timeout-s", type=float, default=15.0)
    parser.add_argument("--allow-snapshot-subprocess-fallback", action="store_true")
    parser.add_argument("--conda-env", default="yolo")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--detector-weight", default="models/yolo/weights/best.pt")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    parser.add_argument("--detector-iou", type=float, default=0.45)
    parser.add_argument("--detector-device", default="cuda:0")
    parser.add_argument("--tf-json", default="/tmp/scene_tf_base_color_optical.json")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--ready-pose-json", default="config/rectangle_ready_pose.json")
    parser.add_argument("--init-stable-wait-s", type=float, default=1.0)
    parser.add_argument("--second-snapshot-stable-wait-s", type=float, default=0.5)
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    return parser.parse_args(argv)
