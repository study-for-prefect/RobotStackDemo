"""Top-level hover-only calibration workflow."""

import os
import subprocess
import sys
import time

from robot_scene_pipeline.grasp_orientation import quaternion_distance_rad
from robot_scene_pipeline.io_utils import project_path, write_json

from .arguments import parse_args
from .commands import moveit_hover_command, run
from .pose import hover_orientation, pose_payload, tool0_goal_from_tcp
from .scene import choose_target, load_json, matching_objects, snapshot_command
from .tf_lookup import lookup_tool_pose

def main() -> int:
    args = parse_args()
    if not args.use_tf:
        raise RuntimeError("Hover calibration requires --use-tf so geometry_center_m is in base_link.")
    args.output_dir = project_path(args.output_dir)
    args.tf_json = project_path(args.tf_json)
    args.detector_config = project_path(args.detector_config)
    if args.detector_weight:
        args.detector_weight = project_path(args.detector_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    snapshot_dir = os.path.join(args.output_dir, "snapshot")
    run(snapshot_command(args, snapshot_dir))
    state_path = os.path.join(snapshot_dir, "private_scene_state.json")
    state = load_json(state_path)
    target = choose_target(matching_objects(state, args.label), state=state, label_filter=args.label)
    if target.get("geometry_estimation_method") == "known_height_prior":
        print(
            "WARNING: target geometry uses the explicit {:.4f} m height prior; object height was not observable in depth.".format(
                float(args.known_object_height_m)
            ),
            flush=True,
        )

    current_tool0_pose_before = lookup_tool_pose(args.base_frame, args.tool_frame, args.tf_timeout)
    hover_quat, yaw_used, yaw_source = hover_orientation(args, target, current_tool0_pose_before)
    center = [float(value) for value in target["geometry_center_m"]]
    top_z = float(target["top_z_base_m"])
    hover_position = [center[0], center[1], top_z + float(args.hover_height)]
    tcp_offset_tool = [float(value) for value in args.tcp_offset_tool]
    tool0_goal_position = tool0_goal_from_tcp(hover_position, hover_quat, tcp_offset_tool)

    record = {
        "label": target.get("label"),
        "requested_label": args.label,
        "object_id": target.get("id"),
        "target_geometry_center_base": center,
        "target_top_z_base_m": top_z,
        "target_yaw_used": yaw_used,
        "target_yaw_source": yaw_source,
        "target_table_yaw_deg": target.get("table_yaw_deg"),
        "target_table_yaw_valid": target.get("table_yaw_valid"),
        "target_pointcloud_source": target.get("pointcloud_source"),
        "target_geometry_estimation_method": target.get("geometry_estimation_method"),
        "target_depth_geometry_observable": target.get("depth_geometry_observable"),
        "target_pointcloud_point_count": target.get("pointcloud_point_count"),
        "hover_target_position_base": hover_position,
        "hover_target_orientation_xyzw": hover_quat,
        "hover_target_pose": pose_payload(hover_position, hover_quat),
        "tool0_goal_position_base": tool0_goal_position,
        "current_tool0_pose_before": current_tool0_pose_before,
        "actual_tool0_pose_after": None,
        "tcp_offset_tool_m": tcp_offset_tool,
        "hover_height": float(args.hover_height),
        "hover_orientation_mode": args.hover_orientation_mode,
        "safe_pre_rotate_height": float(args.safe_pre_rotate_height),
        "snapshot_path": state.get("snapshot_image") or os.path.join(snapshot_dir, "snapshot.jpg"),
        "annotated_image_path": state.get("annotated_image") or os.path.join(snapshot_dir, "annotated_detector.jpg"),
        "private_scene_state_path": state_path,
        "timestamp": time.time(),
        "execute": bool(args.execute),
    }
    write_json(os.path.join(args.output_dir, "hover_record_before_move.json"), record)

    command = moveit_hover_command(args, hover_position, hover_quat)
    record["moveit_command"] = command
    try:
        run(command)
        record["moveit_status"] = "ok"
    except subprocess.CalledProcessError as exc:
        record["moveit_status"] = "failed"
        record["moveit_returncode"] = int(exc.returncode)
        try:
            record["actual_tool0_pose_after"] = lookup_tool_pose(args.base_frame, args.tool_frame, args.tf_timeout)
        except Exception as tf_exc:
            record["actual_tool0_pose_after_error"] = str(tf_exc)
        summary = {
            "schema_version": "hover_tool_offset_calibration_v1",
            "output_dir": args.output_dir,
            "hover_records": [record],
        }
        write_json(os.path.join(args.output_dir, "summary.json"), summary)
        print(
            "Hover motion failed safely; summary saved to {}".format(
                os.path.join(args.output_dir, "summary.json")
            ),
            file=sys.stderr,
            flush=True,
        )
        return int(exc.returncode or 1)

    record["actual_tool0_pose_after"] = lookup_tool_pose(args.base_frame, args.tool_frame, args.tf_timeout)
    record["timestamp_after"] = time.time()

    summary = {
        "schema_version": "hover_tool_offset_calibration_v1",
        "output_dir": args.output_dir,
        "hover_records": [record],
    }
    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("Saved hover calibration summary: {}".format(os.path.join(args.output_dir, "summary.json")), flush=True)
    print(
        "Target {} hover_position={} orientation={} yaw_source={}".format(
            target.get("label"),
            [round(value, 4) for value in hover_position],
            [round(value, 5) for value in hover_quat],
            yaw_source,
        ),
        flush=True,
    )
    return 0
