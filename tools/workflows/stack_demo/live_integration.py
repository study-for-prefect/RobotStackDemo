"""Non-actuating live integration preflight and auditable result report."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping


REQUIRED_CAMERA_TOPICS = (
    "/camera/camera/color/image_raw",
    "/camera/camera/aligned_depth_to_color/image_raw",
    "/camera/camera/color/camera_info",
    "/joint_states",
)


@dataclass
class LiveIntegrationReport:
    output_dir: Path
    values: dict[str, Any] = field(default_factory=lambda: {
        "safe_mode": True,
        "robot_motion_executed": False,
        "gripper_command_executed": False,
        "camera_topics_ok": False,
        "tf_ok": False,
        "perception_health_ok": False,
        "snapshot_ok": False,
        "parameter_contract_ok": False,
        "qwen_ok": False,
        "physical_edges_generated": False,
        "selected_edge": None,
        "moveit_plan_only_attempted": False,
        "moveit_plan_only_ok": False,
        "failure_stage": None,
        "artifacts": {},
    })

    def update(self, **values: Any) -> None:
        self.values.update(values)
        self.write()

    def artifact(self, name: str, path: str | Path) -> None:
        self.values.setdefault("artifacts", {})[name] = str(Path(path).resolve())
        self.write()

    def fail(self, stage: str, exc: Exception | str) -> None:
        self.values["failure_stage"] = str(stage)
        self.values["error"] = str(exc)
        self.write()

    def write(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "integration_report.json"
        path.write_text(json.dumps(self.values, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_live_integration_args(args: Any) -> None:
    if not bool(getattr(args, "live_integration_check", False)):
        return
    conflicts = [
        flag for flag, enabled in (
            ("--execute", args.execute),
            ("--yes", args.yes),
            ("--execute-push-clearing", args.execute_push_clearing),
        ) if enabled
    ]
    if conflicts:
        raise ValueError(f"--live-integration-check is mutually exclusive with {', '.join(conflicts)}")
    args.execute = False
    args.yes = False
    args.execute_push_clearing = False
    args.moveit_plan_only = True
    args.max_task_steps = 1


def capture_ros_preflight(args: Any, output_dir: Path) -> Mapping[str, Any]:
    """Read one message from every required topic and inspect MoveIt graph endpoints."""
    rclpy, sensor_msgs, qos = _ros_imports()
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=None)
    node = rclpy.create_node("stack_demo_live_integration_preflight")
    received: dict[str, Any] = {}

    def remember(name: str):
        def callback(message: Any) -> None:
            received.setdefault(name, message)
        return callback

    subscriptions = [
        node.create_subscription(sensor_msgs.Image, args.color_topic, remember(args.color_topic), qos),
        node.create_subscription(sensor_msgs.Image, args.depth_topic, remember(args.depth_topic), qos),
        node.create_subscription(sensor_msgs.CameraInfo, args.camera_info_topic, remember(args.camera_info_topic), qos),
        node.create_subscription(sensor_msgs.JointState, "/joint_states", remember("/joint_states"), qos),
    ]
    del subscriptions
    deadline = time.time() + max(1.0, float(args.tf_timeout))
    try:
        while time.time() < deadline and len(received) < 4:
            rclpy.spin_once(node, timeout_sec=0.1)
        topic_names = {name for name, _types in node.get_topic_names_and_types()}
        service_names = {name for name, _types in node.get_service_names_and_types()}
        # ROS 2 Humble's rclpy.Node has no get_action_names_and_types(). Action
        # endpoints remain visible through their standard transport topics;
        # inspecting those names is read-only and never sends a goal.
        action_names = _action_names_from_topics(topic_names)
        missing = [name for name in (args.color_topic, args.depth_topic, args.camera_info_topic, "/joint_states") if name not in received]
        moveit_available = bool(
            any("move_action" in name for name in action_names)
            or any(token in name for name in service_names for token in ("compute_cartesian_path", "plan_kinematic_path"))
        )
        if missing:
            raise RuntimeError(f"required ROS topics did not produce a frame: {missing}")
        if not moveit_available:
            raise RuntimeError("MoveIt planning service/action is not visible; existing MoveIt must be started externally")
        before = _joint_state_payload(received["/joint_states"])
        _write_json(output_dir / "before_joint_state.json", before)
        report = {
            "required_topics": [args.color_topic, args.depth_topic, args.camera_info_topic, "/joint_states"],
            "visible_required_topics": sorted(topic_names.intersection({args.color_topic, args.depth_topic, args.camera_info_topic, "/joint_states"})),
            "camera_topics_ok": True,
            "moveit_graph_ok": True,
            "moveit_actions": sorted(name for name in action_names if "move" in name or "execute" in name),
            "moveit_services": sorted(name for name in service_names if "plan" in name or "cartesian" in name),
        }
        _write_json(output_dir / "ros_preflight.json", report)
        return report
    finally:
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def capture_after_joint_state(args: Any, output_dir: Path) -> Mapping[str, Any]:
    rclpy, sensor_msgs, qos = _ros_imports()
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=None)
    node = rclpy.create_node("stack_demo_live_integration_after_joint")
    received: list[Any] = []
    subscription = node.create_subscription(
        sensor_msgs.JointState, "/joint_states", lambda message: received.append(message), qos,
    )
    del subscription
    deadline = time.time() + max(1.0, float(args.tf_timeout))
    try:
        while time.time() < deadline and not received:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not received:
            raise RuntimeError("no /joint_states frame received after live integration check")
        after = _joint_state_payload(received[-1])
        _write_json(output_dir / "after_joint_state.json", after)
        before = json.loads((output_dir / "before_joint_state.json").read_text(encoding="utf-8"))
        before_map = dict(zip(before["name"], before["position"]))
        after_map = dict(zip(after["name"], after["position"]))
        common = sorted(set(before_map).intersection(after_map))
        deltas = {name: float(after_map[name]) - float(before_map[name]) for name in common}
        result = {
            "joint_position_delta_rad": deltas,
            "maximum_absolute_delta_rad": max((abs(value) for value in deltas.values()), default=0.0),
            "motion_command_issued_by_workflow": False,
        }
        _write_json(output_dir / "joint_state_delta.json", result)
        return result
    finally:
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def validate_live_snapshot(output_dir: Path, private_state: Mapping[str, Any]) -> Mapping[str, Any]:
    required = (
        "snapshot.jpg", "annotated_detector.jpg", "private_scene_state.json",
        "detector_objects_3d.json", "tabletop_geometry.json", "tf_status.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"live snapshot artifacts are missing: {missing}")
    profile = private_state.get("camera_profile") or {}
    if not bool(profile.get("depth_available")):
        raise RuntimeError("live snapshot depth_available is false")
    if str(private_state.get("coordinate_frame") or private_state.get("base_frame")) != "base_link":
        raise RuntimeError("live snapshot coordinate_frame is not base_link")
    tabletop = json.loads((output_dir / "tabletop_geometry.json").read_text(encoding="utf-8"))
    plane = tabletop.get("table_plane") or {}
    if tabletop.get("status") != "ok" or plane.get("frame") != "base_link":
        raise RuntimeError("live tabletop geometry is not valid in base_link")
    objects = [item for item in private_state.get("objects", []) if isinstance(item, dict)]
    for obj in objects:
        if not bool(obj.get("base_coordinate_valid")):
            raise RuntimeError(f"object {obj.get('id')} has invalid base coordinates")
        _assert_finite_tree(obj, f"object[{obj.get('id')}]")
    tf_payload = json.loads((output_dir / "tf_status.json").read_text(encoding="utf-8"))
    _assert_finite_tree(tf_payload, "tf_status")
    return {
        "depth_available": True,
        "coordinate_frame": "base_link",
        "object_count": len(objects),
        "tabletop_frame": "base_link",
        "finite_values": True,
    }


def _ros_imports() -> tuple[Any, Any, Any]:
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs import msg as sensor_msgs

    return rclpy, sensor_msgs, qos_profile_sensor_data


def _action_names_from_topics(topic_names: set[str]) -> set[str]:
    """Return ROS action prefixes exposed through standard action topics."""
    suffixes = ("/_action/status", "/_action/feedback")
    return {
        topic[: -len(suffix)]
        for topic in topic_names
        for suffix in suffixes
        if topic.endswith(suffix)
    }


def _joint_state_payload(message: Any) -> dict[str, Any]:
    positions = [float(value) for value in message.position]
    if not all(math.isfinite(value) for value in positions):
        raise RuntimeError("joint_states contains NaN or Inf")
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    timestamp = None if stamp is None else float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return {
        "timestamp": timestamp,
        "name": [str(value) for value in message.name],
        "position": positions,
        "velocity": [float(value) for value in message.velocity],
        "effort": [float(value) for value in message.effort],
    }


def _assert_finite_tree(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{path} contains NaN or Inf")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
