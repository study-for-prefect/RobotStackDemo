"""Hover-trial collection for XY bias diagnosis."""

import json
import math
import os
import time

import numpy as np

from robot_scene_pipeline.io_utils import project_path, write_json

from .constants import PROJECT_ROOT
from .io import load_json, run

def hover_command(args, output_dir, yaw_deg):
    command = [
        args.ros_python,
        "tools/calibration/hover_tool_offset_calibration.py",
        "--label",
        args.label,
        "--output-dir",
        output_dir,
        "--use-tf",
        "--tf-json",
        args.tf_json,
        "--detector-config",
        args.detector_config,
        "--conda-env",
        args.conda_env,
        "--hover-height",
        str(args.hover_height),
        "--hover-orientation-mode",
        "fixed-yaw",
        "--fixed-hover-yaw",
        str(yaw_deg),
        "--known-object-height-m",
        str(args.known_object_height_m),
        "--tcp-offset-tool",
        *[str(value) for value in args.tcp_offset_tool],
        "--pre-rotate-strategy",
        "joint-wrist3",
        "--pre-rotate-wrist-yaw-sign",
        args.pre_rotate_wrist_yaw_sign,
    ]
    if args.execute:
        command.append("--execute")
    if args.yes:
        command.append("--yes")
    return command


def measurement_prompt():
    while True:
        value = input(
            "输入有符号误差 error_x_mm error_y_mm（夹爪中心 - 积木中心），"
            "或输入 skip/quit: "
        ).strip()
        if value.lower() in ("quit", "q"):
            return "quit", None
        if value.lower() in ("skip", "s"):
            return "skip", None
        fields = value.replace(",", " ").split()
        if len(fields) != 2:
            print("需要两个数，例如: 3.0 3.2", flush=True)
            continue
        try:
            measured = [float(fields[0]), float(fields[1])]
            if not all(math.isfinite(value) for value in measured):
                raise ValueError
            return "ok", measured
        except ValueError:
            print("输入必须是两个有限数值。", flush=True)


def summary_record(summary_path):
    payload = load_json(summary_path)
    records = payload.get("hover_records", [])
    if not records:
        raise RuntimeError("No hover record in {}".format(summary_path))
    return records[-1]


def robot_goal_error_mm(record):
    if not record.get("execute"):
        return None
    goal = record.get("tool0_goal_position_base")
    actual = (record.get("actual_tool0_pose_after") or {}).get("position")
    if not isinstance(goal, list) or not isinstance(actual, list):
        return None
    return [1000.0 * (float(actual[i]) - float(goal[i])) for i in range(2)]


def perception_metadata(record):
    metadata = {
        "pointcloud_source": record.get("target_pointcloud_source"),
        "geometry_estimation_method": record.get("target_geometry_estimation_method"),
        "depth_geometry_observable": record.get("target_depth_geometry_observable"),
        "pointcloud_point_count": record.get("target_pointcloud_point_count"),
    }
    if metadata["pointcloud_source"] is not None:
        return metadata
    state_path = record.get("private_scene_state_path")
    if not state_path or not os.path.exists(state_path):
        return metadata
    state = load_json(state_path)
    object_id = record.get("object_id")
    target = next(
        (item for item in state.get("objects", []) if item.get("id") == object_id),
        None,
    )
    if target is None:
        return metadata
    return {
        "pointcloud_source": target.get("pointcloud_source"),
        "geometry_estimation_method": target.get("geometry_estimation_method"),
        "depth_geometry_observable": target.get("depth_geometry_observable"),
        "pointcloud_point_count": target.get("pointcloud_point_count"),
    }


def new_dataset(args):
    return {
        "schema_version": "xy_bias_diagnosis_samples_v1",
        "phase": args.phase,
        "label": args.label,
        "error_convention": "observed_gripper_center_minus_physical_object_center",
        "units": {"xy": "meter", "measured_error": "millimeter", "yaw": "degree"},
        "collection_offsets": {
            "tcp_offset_tool_m": [float(value) for value in args.tcp_offset_tool],
        },
        "created_at": time.time(),
        "samples": [],
    }


def trial_specs(args):
    if args.phase == "yaw":
        for yaw in args.yaws:
            for repeat in range(1, max(1, int(args.repeats)) + 1):
                yield {"position_label": "fixed", "yaw_deg": float(yaw), "repeat": repeat}
    else:
        for position in args.positions:
            for repeat in range(1, max(1, int(args.repeats)) + 1):
                yield {
                    "position_label": str(position),
                    "yaw_deg": float(args.workspace_yaw),
                    "repeat": repeat,
                }


def trial_key(spec):
    return (
        str(spec["position_label"]),
        round(float(spec["yaw_deg"]), 6),
        int(spec["repeat"]),
    )


def validate_dataset_compatibility(dataset, args):
    expected = {
        "phase": args.phase,
        "label": args.label,
    }
    for key, value in expected.items():
        if dataset.get(key) != value:
            raise ValueError(
                "Existing samples.json has {}={!r}, but this run requested {!r}.".format(
                    key,
                    dataset.get(key),
                    value,
                )
            )
    existing_offsets = dataset.get("collection_offsets") or {}
    requested_offsets = {
        "tcp_offset_tool_m": [float(value) for value in args.tcp_offset_tool],
    }
    if existing_offsets != requested_offsets:
        raise ValueError(
            "Existing samples were collected with different offsets. "
            "Use the recorded offsets or choose a new --output-dir."
        )


def collect(args):
    args.output_dir = project_path(args.output_dir)
    args.tf_json = project_path(args.tf_json)
    args.detector_config = project_path(args.detector_config)
    os.makedirs(args.output_dir, exist_ok=True)
    samples_path = os.path.join(args.output_dir, "samples.json")
    dataset = load_json(samples_path) if os.path.exists(samples_path) else new_dataset(args)
    validate_dataset_compatibility(dataset, args)
    measurements = load_json(project_path(args.measurements_json)) if args.measurements_json else []
    if measurements and not isinstance(measurements, list):
        raise ValueError("--measurements-json must contain a JSON list.")
    completed = {
        trial_key(sample)
        for sample in dataset.get("samples", [])
    }
    previous_position = None

    for trial_index, spec in enumerate(trial_specs(args)):
        if trial_key(spec) in completed:
            print(
                "Skipping completed trial: position={} yaw={:.1f} repeat={}".format(
                    spec["position_label"],
                    spec["yaw_deg"],
                    spec["repeat"],
                ),
                flush=True,
            )
            continue
        if spec["position_label"] != previous_position:
            previous_position = spec["position_label"]
            if args.phase == "workspace" and not args.no_position_prompt:
                input(
                    "\n将积木移动到工作区位置 '{}'，保持姿态不变，然后按 Enter。".format(
                        previous_position
                    )
                )

        trial_name = "{}_yaw_{:+06.1f}_r{:02d}".format(
            spec["position_label"],
            spec["yaw_deg"],
            spec["repeat"],
        ).replace("+", "p").replace("-", "m").replace(".", "_")
        trial_dir = os.path.join(args.output_dir, "trials", trial_name)
        print(
            "\nTrial: phase={} position={} yaw={:.1f} repeat={}".format(
                args.phase,
                spec["position_label"],
                spec["yaw_deg"],
                spec["repeat"],
            ),
            flush=True,
        )
        try:
            run(hover_command(args, trial_dir, spec["yaw_deg"]))
        except subprocess.CalledProcessError as exc:
            failure = {
                **spec,
                "trial_index": trial_index,
                "trial_dir": trial_dir,
                "hover_summary": os.path.join(trial_dir, "summary.json"),
                "returncode": int(exc.returncode),
                "timestamp": time.time(),
            }
            dataset.setdefault("failures", []).append(failure)
            write_json(samples_path, dataset)
            print(
                "\nHover trial failed safely and was not marked complete. "
                "Failure saved in {}. Re-run the same command after resolving "
                "the motion issue; completed trials will be skipped.".format(samples_path),
                file=sys.stderr,
                flush=True,
            )
            return int(exc.returncode or 1)
        record = summary_record(os.path.join(trial_dir, "summary.json"))

        if trial_index < len(measurements):
            item = measurements[trial_index]
            status = "ok"
            measured = [float(item["error_x_mm"]), float(item["error_y_mm"])]
            if not all(math.isfinite(value) for value in measured):
                raise ValueError(
                    "Measurement {} contains a non-finite value.".format(trial_index)
                )
            print("Using measurement {} mm".format(measured), flush=True)
        elif measurements:
            raise ValueError(
                "--measurements-json has no entry for trial index {}.".format(trial_index)
            )
        else:
            status, measured = measurement_prompt()
        if status == "quit":
            break
        if status == "skip":
            continue

        perception = perception_metadata(record)
        sample = {
            **spec,
            "trial_index": trial_index,
            "error_x_mm": measured[0],
            "error_y_mm": measured[1],
            "visual_geometry_xy_m": [float(value) for value in record["target_geometry_center_base"][:2]],
            "target_yaw_used_deg": record.get("target_yaw_used"),
            "target_yaw_source": record.get("target_yaw_source"),
            "robot_goal_error_xy_mm": robot_goal_error_mm(record),
            **perception,
            "trial_dir": trial_dir,
            "hover_summary": os.path.join(trial_dir, "summary.json"),
            "timestamp": time.time(),
        }
        dataset["samples"].append(sample)
        completed.add(trial_key(spec))
        write_json(samples_path, dataset)
        print("Saved sample: {}".format(samples_path), flush=True)

    if dataset["samples"]:
        report = analyze_dataset(dataset)
        write_json(os.path.join(args.output_dir, "diagnosis_report.json"), report)
        print_report(report)
    return 0
