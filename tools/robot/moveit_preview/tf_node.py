"""ROS 2 node and TF frame resolution for MoveIt preview."""

import time
from typing import Dict, List

import rclpy
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

try:
    import yaml
except Exception:  # PyYAML is optional for unit tests and minimal ROS envs.
    yaml = None


def simple_tf_frame_payload(frames_yaml: str) -> Dict[str, dict]:
    payload = {}
    current = None
    for raw_line in str(frames_yaml or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith((" ", "\t")) and line.endswith(":"):
            current = line[:-1].strip().strip("'\"").lstrip("/")
            if current:
                payload.setdefault(current, {})
            continue
        if current and line.strip().startswith("parent:"):
            parent = line.split(":", 1)[1].strip().strip("'\"").lstrip("/")
            payload.setdefault(current, {})["parent"] = parent
    return payload

def tf_frame_names(frames_yaml):
    try:
        payload = yaml.safe_load(frames_yaml) if yaml is not None else simple_tf_frame_payload(frames_yaml)
        payload = payload or {}
    except Exception:
        payload = simple_tf_frame_payload(frames_yaml)
    names = set()
    for child, entry in payload.items():
        if child:
            names.add(str(child).lstrip("/"))
        if isinstance(entry, dict) and entry.get("parent"):
            names.add(str(entry["parent"]).lstrip("/"))
    return sorted(names)


def tf_name_matches(frame, requested):
    frame = str(frame).lstrip("/")
    requested = str(requested).lstrip("/")
    return frame == requested or frame.endswith("/" + requested) or frame.endswith("_" + requested)


def tf_candidate_pairs(frames, requested_base, requested_tool):
    bases = [frame for frame in frames if tf_name_matches(frame, requested_base)]
    tools = [frame for frame in frames if tf_name_matches(frame, requested_tool)]
    pairs = []
    for base_frame in bases:
        for tool_frame in tools:
            pair = (base_frame, tool_frame)
            if pair != (requested_base, requested_tool) and pair not in pairs:
                pairs.append(pair)
    return pairs


def nearby_frame_candidates(frames: List[str], requested: str) -> List[str]:
    requested = str(requested).lstrip("/")
    output = []
    for frame in frames:
        leaf = str(frame).split("/")[-1]
        if requested in leaf or leaf in requested:
            output.append(frame)
    return output[:8]


def tf_tree_diagnosis(frames_yaml, requested_base, requested_tool):
    frames = tf_frame_names(frames_yaml)
    base_matches = [frame for frame in frames if tf_name_matches(frame, requested_base)]
    tool_matches = [frame for frame in frames if tf_name_matches(frame, requested_tool)]
    if not frames:
        return (
            "No TF frames were received. Start robot_state_publisher and the UR driver, "
            "and verify ROS_DOMAIN_ID/RMW settings."
        )
    if not base_matches or not tool_matches:
        return (
            "Missing exact TF frame candidates: base_matches={} tool_matches={}. "
            "nearby_base_frames={} nearby_tool_frames={}.".format(
                base_matches,
                tool_matches,
                nearby_frame_candidates(frames, requested_base),
                nearby_frame_candidates(frames, requested_tool),
            )
        )
    return (
        "Base and tool frames exist but are disconnected. This usually means multiple robot_state_publisher "
        "instances, inconsistent UR prefixes, or a missing fixed joint between the robot and tool. "
        "base_matches={} tool_matches={}.".format(base_matches, tool_matches)
    )


class MoveItPreviewNode(Node):
    def __init__(self, args):
        super().__init__("moveit_plan_preview")
        self.joint_state_seen = False
        self.latest_joint_state = None
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True,
        )  # noqa F841
        self.base_frame = str(args.base_link).lstrip("/")
        self.tool_frame = str(args.end_effector).lstrip("/")
        if not args.gripper_open_only:
            self.base_frame, self.tool_frame = self.resolve_tool_frames(
                self.base_frame,
                self.tool_frame,
                timeout=args.tf_timeout,
                auto_resolve=args.auto_resolve_tf_frames,
            )
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur.joint_names(),
            base_link_name=self.base_frame,
            end_effector_name=self.tool_frame,
            group_name=args.group_name,
            use_move_group_action=True,
        )
        self.moveit2.max_velocity = args.velocity
        self.moveit2.max_acceleration = args.acceleration
        self.moveit2.allowed_planning_time = args.planning_time

    def joint_state_cb(self, msg):
        self.joint_state_seen = True
        self.latest_joint_state = msg

    def wait_for_joint_state(self, timeout=5.0):
        deadline = time.time() + timeout
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_state_seen:
                return True
        return False

    def current_tool_quat_xyzw(self, timeout=3.0):
        transform = self.current_tool_transform(timeout=timeout)
        q = transform.transform.rotation
        return [float(q.x), float(q.y), float(q.z), float(q.w)]

    def resolve_tool_frames(self, requested_base, requested_tool, timeout=8.0, auto_resolve=True):
        deadline = time.time() + float(timeout)
        last_error = "not attempted"
        known_frames = []
        while rclpy.ok() and time.time() < deadline:
            try:
                if self.tf_buffer.can_transform(
                    requested_base,
                    requested_tool,
                    Time(),
                    timeout=Duration(seconds=0.2),
                ):
                    return requested_base, requested_tool
                last_error = "can_transform returned False"
            except Exception as exc:
                last_error = repr(exc)
            if auto_resolve:
                known_frames = tf_frame_names(self.tf_buffer.all_frames_as_yaml())
                for base_frame, tool_frame in tf_candidate_pairs(known_frames, requested_base, requested_tool):
                    try:
                        if self.tf_buffer.can_transform(
                            base_frame,
                            tool_frame,
                            Time(),
                            timeout=Duration(seconds=0.05),
                        ):
                            self.get_logger().warning(
                                "Resolved configured TF frames {} -> {} to connected frames {} -> {}.".format(
                                    requested_base, requested_tool, base_frame, tool_frame
                                )
                            )
                            return base_frame, tool_frame
                    except Exception as exc:
                        last_error = repr(exc)
            time.sleep(0.1)
        frames_yaml = self.tf_buffer.all_frames_as_yaml()
        raise RuntimeError(
            "Robot TF preflight failed: cannot connect configured base/tool frames {} -> {} after {:.1f}s: {}\n"
            "{}\nKnown TF frames:\n{}".format(
                requested_base,
                requested_tool,
                float(timeout),
                last_error,
                tf_tree_diagnosis(frames_yaml, requested_base, requested_tool),
                frames_yaml,
            )
        )

    def current_tool_transform(self, timeout=8.0):
        deadline = time.time() + float(timeout)

        target_frame = self.base_frame
        source_frame = self.tool_frame
        last_error = "not attempted"

        while rclpy.ok() and time.time() < deadline:
            try:
                ok = self.tf_buffer.can_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )

                if ok:
                    return self.tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        Time(),
                        timeout=Duration(seconds=0.2),
                    )

                last_error = "can_transform returned False"

            except Exception as exc:
                last_error = repr(exc)

            time.sleep(0.1)

        try:
            frames = self.tf_buffer.all_frames_as_yaml()
        except Exception:
            frames = "<failed to dump TF frames>"

        raise RuntimeError(
            "Failed to read current {}->{} transform after {:.1f}s: {}\n"
            "Known TF frames:\n{}".format(
                target_frame,
                source_frame,
                float(timeout),
                last_error,
                frames,
            )
        )
