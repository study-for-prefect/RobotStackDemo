"""Command-line arguments for MoveIt plan preview and execution."""

import argparse
import math

from pymoveit2.robots import ur

from .constants import (
    DEFAULT_ORIENTATION_COMMAND_CORRECTION_XYZW,
    DEFAULT_PLAN,
    DEFAULT_QUAT_XYZW,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Plan or execute MoveIt motions from robot_execution_plan.json.")
    parser.add_argument("--plan-json", default=DEFAULT_PLAN)
    parser.add_argument(
        "--push-plan-json",
        default="",
        help="Run one four-stage tabletop push from push_execution_plan_v1 JSON.",
    )
    parser.add_argument(
        "--close-gripper-for-push",
        action="store_true",
        help="In --push-plan-json --execute mode, close the gripper as a rigid paddle, preflight, execute, then reopen.",
    )
    parser.add_argument(
        "--ready-only",
        action="store_true",
        help="Only plan/execute --ready-joint-pose-json, then exit without loading or running a pick plan.",
    )
    parser.add_argument(
        "--gripper-open-only",
        action="store_true",
        help="Open the gripper with no arm motion or plan loading, then exit.",
    )
    parser.add_argument(
        "--gripper-close-only",
        action="store_true",
        help="Close the gripper with no arm motion or plan loading, then exit.",
    )
    parser.add_argument(
        "--relative-tool-translation-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Move tool0 by this relative base_link translation with current orientation, then exit.",
    )
    parser.add_argument(
        "--hover-only",
        action="store_true",
        help="Move tool0 to a hover target only. No gripper, descent, lift, pick, or place actions are run.",
    )
    parser.add_argument(
        "--hover-target-base",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Hover TCP target point in base_link. --tcp-offset-tool is used to produce the tool0 goal.",
    )
    parser.add_argument(
        "--hover-orientation-xyzw",
        nargs=4,
        type=float,
        default=None,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Hover tool0 target orientation in xyzw order.",
    )
    parser.add_argument("--step", type=int, default=1, help="Plan one step number. Ignored by --all-approaches.")
    parser.add_argument("--all-approaches", action="store_true", help="Plan all steps that have approach_position_m.")
    parser.add_argument(
        "--path-mode",
        choices=("approach", "full"),
        default="approach",
        help="approach: only move above targets. full: approach -> target -> retreat, with optional gripper actions.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually execute selected motions.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for interactive confirmation before execution.")
    parser.add_argument(
        "--allow-partial-plan",
        action="store_true",
        help="Allow execution when some plan steps are not planned. Default refuses partial execution.",
    )
    parser.add_argument("--cartesian", dest="cartesian", action="store_true", default=True, help="Use Cartesian planning. This is the default.")
    parser.add_argument("--joint-space", dest="cartesian", action="store_false", help="Use normal MoveIt joint-space pose planning.")
    parser.add_argument("--cartesian-max-step", type=float, default=0.005)
    parser.add_argument("--cartesian-fraction-threshold", type=float, default=0.90)
    parser.add_argument(
        "--min-trajectory-duration",
        type=float,
        default=0.0,
        help="Stretch planned trajectory timestamps to be at least this many seconds before execution.",
    )
    parser.add_argument(
        "--min-point-dt",
        type=float,
        default=0.0,
        help="Stretch trajectory timestamps so consecutive points are at least this many seconds apart.",
    )
    parser.add_argument(
        "--tcp-offset-tool",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.15],
        metavar=("X", "Y", "Z"),
        help="tool0->TCP/gripper-center translation in tool0 coordinates.",
    )
    parser.add_argument(
        "--tcp-calibration-json",
        default="",
        help="Use calibrated tool0->TCP translation. This overrides --tcp-offset-tool.",
    )
    parser.add_argument(
        "--tcp-target-offset-base",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Add a fixed base_link offset to the requested TCP target before applying calibrated TCP conversion.",
    )
    parser.add_argument("--min-z", type=float, default=0.05, help="Reject tool0 goals below this base_link z.")
    parser.add_argument("--max-z", type=float, default=0.80, help="Reject tool0 goals above this base_link z.")
    parser.add_argument("--max-radius", type=float, default=0.90, help="Reject xy radius beyond this value.")
    parser.add_argument("--quat-xyzw", nargs=4, type=float, default=DEFAULT_QUAT_XYZW)
    parser.add_argument(
        "--orientation-command-correction-xyzw",
        nargs=4,
        type=float,
        default=DEFAULT_ORIENTATION_COMMAND_CORRECTION_XYZW,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Base-frame quaternion pre-compensation applied only to MoveIt pose commands.",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("current", "fixed", "object-yaw"),
        default="current",
        help="Use current orientation, fixed --quat-xyzw, or valid object yaw from the execution plan.",
    )
    parser.add_argument(
        "--grasp-axis",
        choices=("long", "short"),
        default="long",
        help="With object-yaw, align the tool yaw reference with the object's long or short footprint axis.",
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="Fixed calibration offset added after object yaw and grasp-axis selection.",
    )
    parser.add_argument(
        "--yaw-sign",
        choices=("positive", "negative"),
        default="positive",
        help="Use object yaw as reported or invert its sign before applying the grasp offset.",
    )
    parser.add_argument(
        "--invalid-yaw-fallback",
        choices=("fixed", "current", "error"),
        default="fixed",
        help="Behavior for square/round objects whose object yaw is not reliable.",
    )
    parser.add_argument(
        "--diagnostic-yaw-deg",
        nargs="+",
        type=float,
        default=None,
        help="Repeat each selected approach-only step at explicit base-link yaw values for offset diagnosis.",
    )
    parser.add_argument(
        "--pre-rotate-before-translation",
        action="store_true",
        help=(
            "Before the first Cartesian/pose translation of each step, first plan the selected "
            "orientation at the current tool0 position. This stages yaw change before XY/Z motion."
        ),
    )
    parser.add_argument(
        "--pre-rotate-strategy",
        choices=("joint-wrist3", "pose"),
        default="joint-wrist3",
        help="joint-wrist3 changes only wrist_3_joint for yaw staging; pose uses MoveIt pose IK candidates.",
    )
    parser.add_argument(
        "--pre-rotate-wrist-yaw-sign",
        choices=("auto", "positive", "negative"),
        default="auto",
        help="Mapping from base yaw delta to wrist_3_joint delta for --pre-rotate-strategy joint-wrist3.",
    )
    parser.add_argument(
        "--pre-rotate-wrist-direction",
        choices=("auto", "positive", "negative"),
        default="auto",
        help="Restrict the actual wrist_3_joint pre-rotation direction. Use this to avoid cable wrap.",
    )
    parser.add_argument("--tf-timeout", type=float, default=8.0)
    parser.add_argument("--group-name", default=ur.MOVE_GROUP_ARM)
    parser.add_argument("--base-link", default=ur.base_link_name())
    parser.add_argument("--end-effector", default=ur.end_effector_name())
    parser.add_argument(
        "--no-auto-resolve-tf-frames",
        action="store_false",
        dest="auto_resolve_tf_frames",
        default=True,
        help="Disable suffix-based recovery for prefixed base/tool frame names.",
    )
    parser.add_argument("--velocity", type=float, default=0.2)
    parser.add_argument("--acceleration", type=float, default=0.2)
    parser.add_argument("--pre-rotate-velocity", type=float, default=0.4)
    parser.add_argument("--pre-rotate-acceleration", type=float, default=0.4)
    parser.add_argument(
        "--safe-pre-rotate-height",
        type=float,
        default=0.12,
        help="Minimum base_link z used before hover-only orientation pre-rotation.",
    )
    parser.add_argument("--place-on-top-velocity", type=float, default=0.03)
    parser.add_argument("--place-on-top-acceleration", type=float, default=0.03)
    parser.add_argument("--place-on-top-release-wait", type=float, default=1.0)
    parser.add_argument(
        "--max-grasp-yaw-error-deg",
        type=float,
        default=5.0,
        help="Refuse the final pick descent when current TCP yaw differs from grasp_yaw by more than this.",
    )
    parser.add_argument(
        "--max-grasp-orientation-error-deg",
        type=float,
        default=0.5,
        help="Refuse final pick descent when the full tool orientation differs from the target by more than this.",
    )
    parser.add_argument(
        "--orientation-settle-error-deg",
        type=float,
        default=0.2,
        help="At approach height, retry an in-place orientation correction above this full orientation error.",
    )
    parser.add_argument("--orientation-settle-attempts", type=int, default=2)
    parser.add_argument("--orientation-settle-wait-s", type=float, default=0.25)
    parser.add_argument(
        "--pre-rotate-yaw-repair-attempts",
        type=int,
        default=1,
        help="Retry an in-place pose pre-rotate when joint-wrist3 pre-rotate leaves too much yaw error.",
    )
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--max-joint-delta", type=float, default=1.2, help="Warn if any joint changes more than this radian value.")
    parser.add_argument(
        "--max-pre-rotate-joint-delta",
        type=float,
        default=math.pi,
        help="Separate joint-delta limit for the intentional wrist_3 pre-rotation stage.",
    )
    parser.add_argument(
        "--ready-joint-pose-json",
        default="",
        help="Optional fixed joint pose JSON to plan/execute before selected steps.",
    )
    parser.add_argument(
        "--ready-joint-tolerance",
        type=float,
        default=0.03,
        help="Skip ready-pose planning when every joint is already within this many radians.",
    )
    parser.add_argument("--enable-gripper", action="store_true", help="Enable DH PGC gripper actions in --path-mode full.")
    parser.add_argument("--gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-slave-id", type=int, default=1)
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-modbus-retries", type=int, default=3)
    parser.add_argument("--gripper-modbus-retry-wait-s", type=float, default=0.08)
    parser.add_argument("--skip-gripper-init", action="store_true")
    parser.add_argument("--gripper-full-calibration", action="store_true")
    parser.add_argument("--gripper-force", type=int, default=50)
    parser.add_argument("--gripper-speed", type=int, default=30)
    parser.add_argument("--gripper-open-position", type=int, default=1000)
    parser.add_argument("--gripper-close-position", type=int, default=0)
    parser.add_argument("--gripper-wait", type=float, default=2.0)
    parser.add_argument(
        "--open-gripper-at-start",
        action="store_true",
        help="Open the gripper after initialization before any motion/plan steps.",
    )
    parser.add_argument(
        "--post-close-wait",
        type=float,
        default=1.0,
        help="Extra pause after a pick close command before retreating.",
    )
    parser.add_argument(
        "--release-after-pick",
        action="store_true",
        help="After a pick retreat, descend back to target, open gripper, then retreat again.",
    )
    return parser.parse_args()
