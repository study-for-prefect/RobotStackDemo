"""Trajectory timing, joint-state, confirmation, and gripper helpers."""

import math

from sensor_msgs.msg import JointState

def trajectory_points(trajectory):
    if trajectory is None:
        return None
    joint_traj = trajectory.joint_trajectory if hasattr(trajectory, "joint_trajectory") else trajectory
    return joint_traj.joint_names, joint_traj.points


def time_from_start_seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec / 1e9


def set_time_from_start_seconds(point, seconds):
    seconds = max(0.0, float(seconds))
    point.time_from_start.sec = int(math.floor(seconds))
    point.time_from_start.nanosec = int(round((seconds - point.time_from_start.sec) * 1e9))
    if point.time_from_start.nanosec >= 1000000000:
        point.time_from_start.sec += 1
        point.time_from_start.nanosec -= 1000000000


def stretch_trajectory_timing(trajectory, min_duration=0.0, min_point_dt=0.0):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return
    _, points = parsed
    if not points:
        return
    original_times = [time_from_start_seconds(point) for point in points]
    total = original_times[-1]
    scale = 1.0
    if min_duration > 0.0 and total > 0.0 and total < min_duration:
        scale = max(scale, float(min_duration) / total)
    scaled_times = [value * scale for value in original_times]
    if min_point_dt > 0.0:
        for index in range(1, len(scaled_times)):
            scaled_times[index] = max(scaled_times[index], scaled_times[index - 1] + float(min_point_dt))
    effective_scale = 1.0
    if total > 0.0 and scaled_times[-1] > 0.0:
        effective_scale = scaled_times[-1] / total
    for point, value in zip(points, scaled_times):
        set_time_from_start_seconds(point, value)
        if effective_scale > 1.0 and point.velocities:
            point.velocities = [float(v) / effective_scale for v in point.velocities]
        if effective_scale > 1.0 and point.accelerations:
            point.accelerations = [float(a) / (effective_scale * effective_scale) for a in point.accelerations]


def print_trajectory_summary(node, trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        node.get_logger().error("No trajectory returned.")
        return
    joint_names, points = parsed
    node.get_logger().info("Trajectory points: {}".format(len(points)))
    if not points:
        return
    for idx in sorted(set([0, len(points) - 1])):
        point = points[idx]
        t = time_from_start_seconds(point)
        values = ", ".join(
            "{}={:.4f}".format(name, pos)
            for name, pos in zip(joint_names, point.positions)
        )
        node.get_logger().info("  point {} t={:.3f}: {}".format(idx, t, values))


def max_joint_delta(trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return None
    joint_names, points = parsed
    if len(points) < 2:
        return None
    max_name = None
    max_delta = None
    previous_positions = points[0].positions
    for point in points[1:]:
        for name, previous_value, value in zip(joint_names, previous_positions, point.positions):
            delta = abs(float(value) - float(previous_value))
            if max_delta is None or delta > max_delta:
                max_name = name
                max_delta = delta
        previous_positions = point.positions
    if max_delta is None:
        return None
    return max_name, max_delta


def max_joint_start_goal_delta(trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return None
    joint_names, points = parsed
    if len(points) < 2:
        return None
    start_positions = points[0].positions
    goal_positions = points[-1].positions
    max_name = None
    max_delta = None
    for name, start_value, goal_value in zip(joint_names, start_positions, goal_positions):
        delta = abs(float(goal_value) - float(start_value))
        if max_delta is None or delta > max_delta:
            max_name = name
            max_delta = delta
    if max_delta is None:
        return None
    return max_name, max_delta


def setup_gripper(args):
    if not args.enable_gripper:
        return None
    try:
        from tools.robot.dh_gripper_runtime import DHPGCGripper
    except ImportError:
        from dh_gripper_runtime import DHPGCGripper

    gripper = DHPGCGripper(
        port=args.gripper_port,
        slave_id=args.gripper_slave_id,
        baudrate=args.gripper_baudrate,
        retries=args.gripper_modbus_retries,
        retry_wait_s=args.gripper_modbus_retry_wait_s,
    )
    print("Connected DH gripper: port={}, slave_id={}".format(args.gripper_port, args.gripper_slave_id))
    if not args.skip_gripper_init:
        if not gripper.init_gripper(full_calibration=args.gripper_full_calibration):
            gripper.close()
            raise RuntimeError("Gripper initialization failed.")
        print("Gripper initialization OK.")
    else:
        print("Gripper initialization skipped; preserving the current grip state.")
    gripper.set_force(args.gripper_force)
    gripper.set_speed(args.gripper_speed)
    print("Gripper force={}, speed={}".format(args.gripper_force, args.gripper_speed))
    return gripper


def gripper_position_for_command(args, name):
    if name == "open":
        return int(args.gripper_open_position)
    if name == "close":
        return int(args.gripper_close_position)
    raise ValueError("Unsupported gripper command: {}".format(name))


def gripper_command_accepted(args: object, command: str, status: object, current_position: object) -> bool:
    if status:
        return True
    if command != "close" or current_position is None:
        return False
    close_position = int(args.gripper_close_position)
    open_position = int(args.gripper_open_position)
    span = max(1, abs(open_position - close_position))
    tolerance = getattr(args, "gripper_close_contact_tolerance", None)
    if tolerance is None:
        tolerance = 0.35 * span
    return abs(int(current_position) - close_position) <= float(tolerance)


def maybe_confirm(args, prompt):
    if args.yes:
        return True
    confirm = input("{} Type yes: ".format(prompt))
    return confirm.strip().lower() == "yes"


def latest_joint_position_map(node):
    msg = node.latest_joint_state
    if msg is None:
        return {}
    return {name: float(position) for name, position in zip(msg.name, msg.position)}


def joint_position_map_from_state(joint_state):
    if joint_state is None:
        return {}
    return {name: float(position) for name, position in zip(joint_state.name, joint_state.position)}


def joint_position_map(node, start_joint_state=None):
    if start_joint_state is not None:
        return joint_position_map_from_state(start_joint_state)
    return latest_joint_position_map(node)


def joint_state_from_trajectory(trajectory):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return None
    joint_names, points = parsed
    if not points:
        return None
    state = JointState()
    state.name = list(joint_names)
    state.position = list(points[-1].positions)
    return state


def max_joint_error_to_goal(current_positions, joint_names, joint_positions):
    errors = []
    for name, target in zip(joint_names, joint_positions):
        if name not in current_positions:
            return None
        errors.append(abs(closest_angle_equivalent(target, current_positions[name]) - current_positions[name]))
    return max(errors) if errors else 0.0


def closest_angle_equivalent(value, reference):
    value = float(value)
    reference = float(reference)
    while value - reference > math.pi:
        value -= 2.0 * math.pi
    while value - reference < -math.pi:
        value += 2.0 * math.pi
    return value


def unwrap_continuous_joint_trajectory(trajectory, reference_positions):
    parsed = trajectory_points(trajectory)
    if parsed is None:
        return
    joint_names, points = parsed
    previous = [float(reference_positions.get(name, points[0].positions[index])) for index, name in enumerate(joint_names)]
    for point in points:
        positions = list(point.positions)
        for index, _name in enumerate(joint_names):
            positions[index] = closest_angle_equivalent(positions[index], previous[index])
        point.positions = positions
        previous = positions
