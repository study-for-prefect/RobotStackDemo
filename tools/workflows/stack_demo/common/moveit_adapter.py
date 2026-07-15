"""MoveIt plan-only checker and staged UR5/GF225 executor for immutable edges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from robot_scene_pipeline.grasp_orientation import downward_quaternion_for_yaw

from ..commands import close_gripper_command, open_gripper_command, run
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
                run(self._push_command(path, execute=False))
            else:
                for name in ("approach_pose", "grasp_pose", "lift_pose"):
                    self._run_pose(edge, name, execute=False)
                for pose in edge.physical_parameters.get("transport_path", []):
                    self._run_explicit_pose(
                        pose, float(pose.get("yaw_deg", self._yaw(edge))), execute=False,
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
        self._run_pose(edge, "grasp_pose", execute=True)

    def close_gripper(self, edge: PhysicalActionEdge) -> bool | None:
        run(close_gripper_command(self._args))
        return None

    def lift_for_observation(self, edge: PhysicalActionEdge) -> None:
        self._run_pose(edge, "lift_pose", execute=True)

    def transport(self, edge: PhysicalActionEdge) -> None:
        for pose in edge.physical_parameters.get("transport_path", [])[1:]:
            self._run_explicit_pose(
                pose, float(pose.get("yaw_deg", self._yaw(edge))), execute=True,
            )

    def descend_place(self, edge: PhysicalActionEdge) -> None:
        self._run_pose(edge, "release_pose", execute=True)

    def release(self, edge: PhysicalActionEdge) -> None:
        run(open_gripper_command(self._args))

    def retreat(self, edge: PhysicalActionEdge) -> None:
        release = edge.physical_parameters["release_pose"]
        position = list(release["position_m"])
        position[2] += float(self._config.section("safety")["approach_height_m"])
        self._run_explicit_pose(
            {"frame_id": "base_link", "position_m": position},
            float(release.get("yaw_deg", self._yaw(edge))),
            execute=True,
        )

    def execute_nudge(self, edge: PhysicalActionEdge) -> None:
        if not bool(getattr(self._args, "execute_push_clearing", False)):
            raise RuntimeError("nudge execution requires --execute-push-clearing")
        path = self._write_push_plan(edge)
        run(self._push_command(path, execute=True))

    def _run_pose(self, edge: PhysicalActionEdge, name: str, *, execute: bool) -> None:
        pose = edge.physical_parameters.get(name)
        if not isinstance(pose, Mapping):
            raise ValueError(f"edge is missing {name}")
        self._run_explicit_pose(pose, float(pose.get("yaw_deg", self._yaw(edge))), execute=execute)

    def _run_explicit_pose(self, pose: Mapping[str, Any], yaw_deg: float, *, execute: bool) -> None:
        position = pose.get("position_m")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            raise ValueError("edge pose must contain position_m")
        quaternion = pose.get("orientation_xyzw") or downward_quaternion_for_yaw([1.0, 0.0, 0.0, 0.0], yaw_deg)
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
            "--pre-rotate-before-translation",
            "--pre-rotate-velocity", str(motion["high_clearance_rotation_velocity"]),
            "--pre-rotate-acceleration", str(motion["high_clearance_rotation_acceleration"]),
            "--safe-pre-rotate-height", str(motion["safe_pre_rotate_height_m"]),
            "--base-link", "base_link",
            "--end-effector", self._args.tool_frame,
            "--tf-timeout", str(self._args.tf_timeout),
        ]
        if execute:
            command.extend(["--execute", "--yes"])
        run(command)

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
            "contact_z_offset_m": float(self._config.section("clearing")["push_contact_z_offset_m"]),
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
        ]
        if execute:
            command.extend(["--enable-gripper", "--close-gripper-for-push", "--execute", "--yes"])
        return command

    @staticmethod
    def _yaw(edge: PhysicalActionEdge) -> float:
        return float(edge.physical_parameters.get("grasp_yaw_deg", 0.0))
