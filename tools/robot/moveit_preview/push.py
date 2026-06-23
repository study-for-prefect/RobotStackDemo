"""MoveIt-backed execution of a validated four-stage push-clearing plan."""

import json
from typing import Any, Dict, List, Tuple

from sensor_msgs.msg import JointState

from robot_scene_pipeline.grasp_orientation import normalize_quaternion_xyzw
from tools.robot.push_primitives import build_push_targets

from .execution import plan_and_maybe_execute_motion, plan_motion_trajectory
from .orientation import tool0_goal_from_approach, transform_position_quat
from .trajectory import joint_state_from_trajectory, max_joint_delta


def load_push_plan(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _push_stage_goals(push_plan: dict, args: Any) -> List[Tuple[str, List[float], bool]]:
    targets = build_push_targets(push_plan)
    tool_offset = [float(value) for value in args.tool_offset_base]
    stages = [
        ("pre_push", targets["pre_push"], False),
        ("contact", targets["contact"], True),
        ("push_end", targets["push_end"], True),
        ("retreat", targets["retreat"], True),
    ]
    return [
        (
            name,
            tool0_goal_from_approach(target, args.tool_z_offset, tool_offset),
            cartesian,
        )
        for name, target, cartesian in stages
    ]


def run_push_plan(node: Any, args: Any, planning_start_state: Any) -> bool:
    """Preflight all four MoveIt trajectories, then execute them once in order."""
    push_plan = load_push_plan(args.push_plan_json)
    stage_goals = _push_stage_goals(push_plan, args)
    push_quat = normalize_quaternion_xyzw(args.quat_xyzw)
    current_tool = node.current_tool_transform(timeout=args.tf_timeout)
    current_pos, _current_quat = transform_position_quat(current_tool)

    preflight = []
    start_state = planning_start_state
    previous_position = current_pos
    for stage_name, tool_goal, cartesian in stage_goals:
        trajectory = plan_motion_trajectory(
            node,
            args,
            tool_goal,
            push_quat,
            cartesian=cartesian,
            start_joint_state=start_state,
        )
        if trajectory is None:
            raise RuntimeError(
                "Push clearing preflight failed to plan stage '{}'.".format(stage_name)
            )
        delta = max_joint_delta(trajectory)
        if delta and delta[1] > float(args.max_joint_delta):
            raise RuntimeError(
                "Push clearing preflight refused stage '{}': {} delta {:.3f} rad "
                "exceeds {:.3f}.".format(
                    stage_name,
                    delta[0],
                    delta[1],
                    float(args.max_joint_delta),
                )
            )
        preflight.append(
            (stage_name, tool_goal, cartesian, trajectory, start_state, previous_position)
        )
        start_state = joint_state_from_trajectory(trajectory) or start_state
        previous_position = tool_goal

    node.get_logger().info(
        "Push clearing preflight passed for stages: {}".format(
            ", ".join(item[0] for item in preflight)
        )
    )
    for stage_name, tool_goal, cartesian, trajectory, start_state, previous_position in preflight:
        result = plan_and_maybe_execute_motion(
            node,
            args,
            {
                "step": "push",
                "action": "push_clearing",
                "object_id": push_plan.get("obstacle_object_id"),
            },
            stage_name,
            tool_goal,
            push_quat,
            previous_position,
            "Push clearing stage {} -> tool0_goal={}".format(
                stage_name,
                [round(value, 4) for value in tool_goal],
            ),
            cartesian=cartesian,
            trajectory=trajectory,
            start_joint_state=start_state,
        )
        if not result:
            return False
        if isinstance(result, JointState):
            planning_start_state = result
    return True
