"""MoveIt plan-only checker and staged UR5/GF225 executor for immutable edges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from robot_scene_pipeline.grasp_orientation import downward_quaternion_for_yaw

from ..commands import (
    close_gripper_command,
    open_gripper_command,
    return_to_ready_observation,
    run,
    run_non_actuating,
)
from .action_edges import ActionType, PhysicalActionEdge
from .config import StackDemoConfig


class MoveItEdgeAdapter:
    """Use existing MoveIt/GF225 commands without changing a selected edge."""

    def __init__(self, args: Any, config: StackDemoConfig, artifact_dir: str | Path):
        self._args = args
        self._config = config
        self._artifact_dir = Path(artifact_dir)
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def check(self, edge: PhysicalActionEdge) -> Mapping[str, Any]:
        try:
            if edge.action_type == ActionType.NUDGE_BLOCKER:
                path = self._write_push_plan(edge)
                run_non_actuating(self._push_command(path, execute=False))
            else:
                for name in ("approach_pose", "grasp_pose", "lift_pose"):
                    self._run_pose(edge, name, execute=False)
                for pose in edge.physical_parameters.get("transport_path", []):
                    self._run_explicit_pose(
                        pose,
                        float(pose.get("yaw_deg", self._yaw(edge))),
                        execute=False,
                        orientation_policy=self._orientation_policy(edge),
                    )
                self._run_pose(edge, "release_pose", execute=False)
            return {"passed": True, "moveit_plan_only": True, "mode": "moveit_plan_only"}
        except Exception as exc:
            return {
                "passed": False,
                "moveit_plan_only": True,
                "mode": "moveit_plan_only",
                "error": str(exc),
            }

    def approach_grasp(self, edge: PhysicalActionEdge) -> None:
        self._run_pose(edge, "approach_pose", execute=True)

    def descend_grasp(self, edge: PhysicalActionEdge) -> None:
        self._run_pose(
            edge,
            "grasp_pose",
            execute=True,
            preserve_current_orientation=self._orientation_policy(edge) == "downward_yaw_only",
        )

    def close_gripper(self, edge: PhysicalActionEdge) -> bool | None:
        result_path = self._artifact_dir / f"{edge.candidate_id}_gripper_close_result.json"
        result_path.unlink(missing_ok=True)
        run(close_gripper_command(self._args, str(result_path)))
        if not result_path.is_file():
            return None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not bool(result.get("accepted")):
            raise RuntimeError("gripper close command did not reach an accepted state")
        return bool(result.get("holding_detected"))

    def lift_for_observation(self, edge: PhysicalActionEdge) -> None:
        self._run_pose(
            edge,
            "lift_pose",
            execute=True,
            preserve_current_orientation=self._orientation_policy(edge) == "downward_yaw_only",
        )

    def transport(self, edge: PhysicalActionEdge) -> None:
        path = list(edge.physical_parameters.get("transport_path", []))
        preserve_post_grasp_posture = self._orientation_policy(edge) == "downward_yaw_only"
        previous = path[0] if path else None
        for pose in path[1:]:
            if (
                self._orientation_policy(edge) == "downward_yaw_only"
                and isinstance(previous, Mapping)
                and self._same_fixed_yaw_pose(previous, pose)
            ):
                previous = pose
                continue
            self._run_explicit_pose(
                pose,
                float(pose.get("yaw_deg", self._yaw(edge))),
                execute=True,
                orientation_policy=self._orientation_policy(edge),
                # Ordinary translations lock the measured roll/pitch/yaw.
                # A distinct safe-height yaw waypoint deliberately invokes the
                # wrist-only pre-rotate; roll and pitch remain downward-fixed.
                preserve_current_orientation=(
                    preserve_post_grasp_posture
                    and isinstance(previous, Mapping)
                    and self._same_axis_yaw(previous, pose)
                ),
            )
            previous = pose

    def descend_place(self, edge: PhysicalActionEdge) -> None:
        self._run_pose(
            edge,
            "release_pose",
            execute=True,
            preserve_current_orientation=self._orientation_policy(edge) == "downward_yaw_only",
        )

    def release(self, edge: PhysicalActionEdge) -> None:
        run(open_gripper_command(self._args))

    def retreat(self, edge: PhysicalActionEdge) -> None:
        release = edge.physical_parameters["release_pose"]
        position = list(release["position_m"])
        position[2] += float(self._config.section("safety")["observation_height_m"])
        self._run_explicit_pose(
            {"frame_id": "base_link", "position_m": position},
            float(release.get("yaw_deg", self._yaw(edge))),
            execute=True,
            orientation_policy=self._orientation_policy(edge),
            preserve_current_orientation=self._orientation_policy(edge) == "downward_yaw_only",
        )

    def execute_nudge(self, edge: PhysicalActionEdge) -> None:
        explicitly_authorized = bool(
            getattr(self._args, "execute", False)
            and getattr(self._args, "yes", False)
        )
        if not explicitly_authorized and not bool(
            getattr(self._args, "execute_push_clearing", False)
        ):
            raise RuntimeError("nudge execution requires explicit --execute --yes authorization")
        path = self._write_push_plan(edge)
        run(self._push_command(path, execute=True))

    def return_to_observation_pose(self, edge: PhysicalActionEdge) -> None:
        del edge
        return_to_ready_observation(self._args)

    def _run_pose(
        self,
        edge: PhysicalActionEdge,
        name: str,
        *,
        execute: bool,
        preserve_current_orientation: bool = False,
    ) -> None:
        pose = edge.physical_parameters.get(name)
        if not isinstance(pose, Mapping):
            raise ValueError(f"edge is missing {name}")
        self._run_explicit_pose(
            pose,
            float(pose.get("yaw_deg", self._yaw(edge))),
            execute=execute,
            orientation_policy=self._orientation_policy(edge),
            preserve_current_orientation=preserve_current_orientation,
        )

    def _run_explicit_pose(
        self,
        pose: Mapping[str, Any],
        yaw_deg: float,
        *,
        execute: bool,
        orientation_policy: str = "downward_yaw_only",
        preserve_current_orientation: bool = False,
    ) -> None:
        position = pose.get("position_m")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            raise ValueError("edge pose must contain position_m")
        explicit_quaternion = pose.get("orientation_xyzw")
        full_3d = bool(
            orientation_policy == "full_3d_allowed"
            and isinstance(explicit_quaternion, (list, tuple))
            and len(explicit_quaternion) == 4
        )
        quaternion = (
            explicit_quaternion
            if full_3d
            else downward_quaternion_for_yaw([1.0, 0.0, 0.0, 0.0], yaw_deg)
        )
        motion = self._config.section("motion")
        command = [
            self._args.ros_python,
            "tools/robot/moveit_plan_preview.py",
            "--hover-only",
            "--hover-target-base", *[str(float(value)) for value in position[:3]],
            "--hover-orientation-xyzw", *[str(float(value)) for value in quaternion],
            "--tcp-offset-tool", *[str(value) for value in self._config.section("gripper")["tcp_offset_tool_m"]],
            "--velocity", str(motion["near_object_velocity"]),
            "--acceleration", str(motion["near_object_acceleration"]),
            "--max-wrist-3-start-goal-delta",
            str(motion["max_wrist_3_start_goal_delta_rad"]),
            "--base-link", "base_link",
            "--end-effector", self._args.tool_frame,
            "--tf-timeout", str(self._args.tf_timeout),
        ]
        if preserve_current_orientation:
            command.append("--hover-preserve-current-orientation")
        else:
            command.extend([
                "--pre-rotate-before-translation",
                "--pre-rotate-strategy", "pose" if full_3d else "joint-wrist3",
                # On this UR3/GF225 installation, increasing wrist_3 produces
                # decreasing base-frame tool yaw.  Auto cannot distinguish the
                # two equally cheap joint plans before execution and has chosen
                # the wrong sign in real runs (requested -25 deg, reached +30).
                "--pre-rotate-wrist-yaw-sign", "negative",
                "--pre-rotate-velocity", str(motion["high_clearance_rotation_velocity"]),
                "--pre-rotate-acceleration", str(motion["high_clearance_rotation_acceleration"]),
                "--safe-pre-rotate-height", str(motion["safe_pre_rotate_height_m"]),
            ])
            if not full_3d:
                # The safe-height wrist yaw stage already establishes the
                # ordinary block orientation.  A second full-pose correction
                # has produced remote IK branches for sub-degree residuals.
                command.append("--hover-disable-orientation-settle")
        if execute:
            command.extend(["--execute", "--yes"])
        if execute:
            run(command)
        else:
            run_non_actuating(command)

    def _write_push_plan(self, edge: PhysicalActionEdge) -> Path:
        geometry = edge.decision_metadata.get("acted_object_geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("nudge edge is missing acted object geometry")
        physical = edge.physical_parameters
        plan = {
            "schema_version": "push_execution_plan_v1",
            "frame_id": "base_link",
            "action_type": "nudge",
            "obstacle_object_id": edge.acted_object_track_id,
            "obstacle": dict(geometry),
            "direction_base": list(physical["push_direction_base"]),
            "distance_m": float(physical["push_distance_m"]),
            "lift_m": float(self._config.section("safety")["approach_height_m"]),
            "retreat_lift_m": float(self._config.section("safety")["observation_height_m"]),
            "contact_z_offset_m": float(self._config.section("clearing")["push_contact_z_offset_m"]),
            "contact_standoff_m": float(physical["contact_standoff_m"]),
            "contact_clearance_m": float(physical["contact_clearance_m"]),
            "prepush_clearance_m": float(physical["prepush_clearance_m"]),
            "object_contact_extent_m": float(physical["object_contact_extent_m"]),
            "tool_contact_extent_m": float(physical["tool_contact_extent_m"]),
            "push_orientation_policy": "align_to_target_yaw",
            "target_yaw_deg": float(physical["push_wrist_yaw_deg"]),
            "target_yaw_source": "code_generated_physical_edge",
        }
        path = self._artifact_dir / f"{edge.candidate_id}_push_plan.json"
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _push_command(self, path: Path, *, execute: bool) -> list[str]:
        motion = self._config.section("motion")
        command = [
            self._args.ros_python,
            "tools/robot/moveit_plan_preview.py",
            "--push-plan-json", str(path),
            "--tcp-offset-tool", *[str(value) for value in self._config.section("gripper")["tcp_offset_tool_m"]],
            "--velocity", str(motion["near_object_velocity"]),
            "--acceleration", str(motion["near_object_acceleration"]),
            "--base-link", "base_link",
            "--end-effector", self._args.tool_frame,
            "--tf-timeout", str(self._args.tf_timeout),
            "--max-wrist-3-start-goal-delta",
            str(motion["max_wrist_3_start_goal_delta_rad"]),
        ]
        if execute:
            command.extend(["--enable-gripper", "--close-gripper-for-push", "--execute", "--yes"])
        return command

    @staticmethod
    def _yaw(edge: PhysicalActionEdge) -> float:
        return float(edge.physical_parameters.get("grasp_yaw_deg", 0.0))

    @staticmethod
    def _orientation_policy(edge: PhysicalActionEdge) -> str:
        value = str(edge.physical_parameters.get("orientation_policy") or "downward_yaw_only")
        return value if value == "full_3d_allowed" else "downward_yaw_only"

    @staticmethod
    def _same_fixed_yaw_pose(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        first_position = first.get("position_m")
        second_position = second.get("position_m")
        if not all(
            isinstance(value, (list, tuple)) and len(value) >= 3
            for value in (first_position, second_position)
        ):
            return False
        same_position = all(
            abs(float(left) - float(right)) <= 1e-6
            for left, right in zip(first_position[:3], second_position[:3])
        )
        first_yaw = float(first.get("yaw_deg", 0.0))
        second_yaw = float(second.get("yaw_deg", 0.0))
        same_axis_yaw = abs(((first_yaw - second_yaw + 90.0) % 180.0) - 90.0) <= 1e-6
        return bool(same_position and same_axis_yaw)

    @staticmethod
    def _same_axis_yaw(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        first_yaw = float(first.get("yaw_deg", 0.0))
        second_yaw = float(second.get("yaw_deg", 0.0))
        return abs(((first_yaw - second_yaw + 90.0) % 180.0) - 90.0) <= 1e-6
