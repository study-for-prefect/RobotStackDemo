#!/usr/bin/env python3
"""Collect and analyze systematic XY grasp-center bias.

Measured error convention:
    error_xy = observed gripper center - physical object center

The fitted corrections are incremental changes relative to the offsets already
used while collecting the samples.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from robot_scene_pipeline.io_utils import project_path, write_json


DEFAULT_OUTPUT_DIR = os.path.join(
    "runtime",
    "xy_bias_diagnosis",
    "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose base-fixed, yaw-local TCP, and workspace-dependent XY bias."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Run hover trials and record measured signed XY errors.")
    collect.add_argument("--phase", choices=("yaw", "workspace"), required=True)
    collect.add_argument("--label", required=True)
    collect.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    collect.add_argument("--repeats", type=int, default=3)
    collect.add_argument("--yaws", nargs="+", type=float, default=[0.0, 90.0, 180.0, -90.0])
    collect.add_argument(
        "--positions",
        nargs="+",
        default=["center", "left", "right", "front", "back"],
        help="Manual workspace position labels used by --phase workspace.",
    )
    collect.add_argument("--workspace-yaw", type=float, default=0.0)
    collect.add_argument("--hover-height", type=float, default=0.08)
    collect.add_argument("--known-object-height-m", type=float, default=0.0)
    collect.add_argument("--tf-json", default="/tmp/scene_tf_base_camera.json")
    collect.add_argument("--detector-config", default="config/yolo_detector.json")
    collect.add_argument("--conda-env", default="yolo")
    collect.add_argument("--ros-python", default=sys.executable)
    collect.add_argument("--tool-z-offset", type=float, default=0.15)
    collect.add_argument("--tool-offset-base", nargs=3, type=float, default=[-0.015, 0.0, 0.0])
    collect.add_argument("--tool-offset-yaw-local", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    collect.add_argument(
        "--pre-rotate-wrist-yaw-sign",
        choices=("positive", "negative"),
        default="negative",
    )
    collect.add_argument("--execute", action="store_true")
    collect.add_argument("--yes", action="store_true")
    collect.add_argument(
        "--no-position-prompt",
        action="store_true",
        help="Do not pause before each workspace position (mainly for automated runs).",
    )
    collect.add_argument(
        "--measurements-json",
        default="",
        help="Optional non-interactive list of {'error_x_mm','error_y_mm'} entries in trial order.",
    )

    analyze = subparsers.add_parser("analyze", help="Fit diagnostic models from a samples JSON file.")
    analyze.add_argument("samples_json")
    analyze.add_argument("--output", default="")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(command):
    print("$ {}".format(" ".join(str(value) for value in command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


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
        "--tool-z-offset",
        str(args.tool_z_offset),
        "--tool-offset-base",
        *[str(value) for value in args.tool_offset_base],
        "--tool-offset-yaw-local",
        *[str(value) for value in args.tool_offset_yaw_local],
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
            "tool_offset_base": [float(value) for value in args.tool_offset_base],
            "tool_offset_yaw_local": [float(value) for value in args.tool_offset_yaw_local],
            "tool_z_offset": float(args.tool_z_offset),
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
        "tool_offset_base": [float(value) for value in args.tool_offset_base],
        "tool_offset_yaw_local": [float(value) for value in args.tool_offset_yaw_local],
        "tool_z_offset": float(args.tool_z_offset),
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


def sample_arrays(dataset):
    samples = dataset.get("samples", [])
    if not samples:
        raise ValueError("No samples available.")
    xy = np.asarray([sample["visual_geometry_xy_m"] for sample in samples], dtype=float)
    yaw = np.radians(
        np.asarray(
            [
                sample.get("target_yaw_used_deg")
                if sample.get("target_yaw_used_deg") is not None
                else sample["yaw_deg"]
                for sample in samples
            ],
            dtype=float,
        )
    )
    error = np.asarray(
        [[sample["error_x_mm"], sample["error_y_mm"]] for sample in samples],
        dtype=float,
    ) / 1000.0
    robot_error = [
        sample.get("robot_goal_error_xy_mm")
        for sample in samples
        if isinstance(sample.get("robot_goal_error_xy_mm"), list)
    ]
    return samples, xy, yaw, error, robot_error


def fit_yaw_model(yaw, error):
    rows = []
    values = []
    for angle, measured in zip(yaw, error):
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        rows.append([1.0, 0.0, c, -s])
        values.append(float(measured[0]))
        rows.append([0.0, 1.0, s, c])
        values.append(float(measured[1]))
    matrix = np.asarray(rows, dtype=float)
    values = np.asarray(values, dtype=float)
    params, _, rank, singular = np.linalg.lstsq(matrix, values, rcond=None)
    predicted = matrix.dot(params).reshape((-1, 2))
    residual = error - predicted
    return {
        "rank": int(rank),
        "condition_number": float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None,
        "base_error_m": params[:2].tolist(),
        "yaw_local_error_m": params[2:].tolist(),
        "suggested_base_offset_delta_m": (-params[:2]).tolist(),
        "suggested_tool_offset_yaw_local_delta_m": (-params[2:]).tolist(),
        "rms_residual_mm": float(np.sqrt(np.mean(residual ** 2)) * 1000.0),
        "max_residual_mm": float(np.max(np.linalg.norm(residual, axis=1)) * 1000.0),
    }


def fit_workspace_model(xy, error):
    design = np.column_stack([xy, np.ones(len(xy), dtype=float)])
    coefficients = []
    predicted_columns = []
    ranks = []
    for axis in range(2):
        params, _, rank, _ = np.linalg.lstsq(design, error[:, axis], rcond=None)
        coefficients.append(params)
        predicted_columns.append(design.dot(params))
        ranks.append(rank)
    predicted = np.column_stack(predicted_columns)
    residual = error - predicted
    correction_affine = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    ) - np.asarray(coefficients)
    return {
        "rank": int(min(ranks)),
        "model_definition": "error_xy = error_affine_xy @ [visual_x, visual_y, 1]",
        "error_affine_xy": np.asarray(coefficients).tolist(),
        "suggested_affine_xy": correction_affine.tolist(),
        "rms_residual_mm": float(np.sqrt(np.mean(residual ** 2)) * 1000.0),
        "max_residual_mm": float(np.max(np.linalg.norm(residual, axis=1)) * 1000.0),
    }


def classify(yaw_fit, workspace_fit, robot_mean_mm, perception_source_counts):
    findings = []
    if robot_mean_mm is not None and np.linalg.norm(robot_mean_mm) > 1.0:
        findings.append("robot_execution_or_tf_error_is_non_negligible")
    if yaw_fit is not None:
        base_mm = 1000.0 * np.linalg.norm(yaw_fit["base_error_m"])
        local_mm = 1000.0 * np.linalg.norm(yaw_fit["yaw_local_error_m"])
        if local_mm >= 1.0:
            findings.append("yaw_rotating_component_suggests_tcp_or_gripper_center_offset")
        if base_mm >= 1.0:
            findings.append("base_fixed_component_suggests_handeye_or_fixed_visual_bias")
    if workspace_fit is not None:
        error_affine = np.asarray(workspace_fit["error_affine_xy"], dtype=float)
        slope = np.linalg.norm(error_affine[:, :2])
        if slope >= 0.005:
            findings.append("workspace_dependent_component_suggests_handeye_rotation_or_scale_error")
    if perception_source_counts.get("mask_known_height_fallback", 0):
        findings.append("known_height_projection_fallback_present_in_samples")
    if not findings:
        findings.append("no_dominant_systematic_component_detected")
    return findings


def analyze_dataset(dataset):
    samples, xy, yaw, error, robot_errors = sample_arrays(dataset)
    unique_yaws = np.unique(np.round(np.degrees(yaw), 3))
    unique_positions = np.unique(np.round(xy, 4), axis=0)
    phase = dataset.get("phase")
    yaw_fit = (
        fit_yaw_model(yaw, error)
        if phase == "yaw" and len(unique_yaws) >= 3 and len(samples) >= 6
        else None
    )
    workspace_fit = (
        fit_workspace_model(xy, error)
        if phase == "workspace" and len(unique_positions) >= 3 and len(samples) >= 6
        else None
    )
    robot_mean = (
        np.mean(np.asarray(robot_errors, dtype=float), axis=0)
        if robot_errors
        else None
    )
    measured_mean = np.mean(error, axis=0)
    measured_std = np.std(error, axis=0)
    perception_source_counts = {}
    for sample in samples:
        source = sample.get("pointcloud_source") or "unknown"
        perception_source_counts[source] = perception_source_counts.get(source, 0) + 1
    collection_offsets = dataset.get("collection_offsets") or {}
    if yaw_fit is not None:
        base = np.asarray(collection_offsets.get("tool_offset_base", [0.0, 0.0, 0.0]), dtype=float)
        local = np.asarray(
            collection_offsets.get("tool_offset_yaw_local", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        base[:2] += np.asarray(yaw_fit["suggested_base_offset_delta_m"], dtype=float)
        local[:2] += np.asarray(
            yaw_fit["suggested_tool_offset_yaw_local_delta_m"],
            dtype=float,
        )
        yaw_fit["suggested_updated_tool_offset_base_m"] = base.tolist()
        yaw_fit["suggested_updated_tool_offset_yaw_local_m"] = local.tolist()
    return {
        "schema_version": "xy_bias_diagnosis_report_v1",
        "sample_count": len(samples),
        "error_convention": dataset.get("error_convention"),
        "mean_measured_error_mm": (measured_mean * 1000.0).tolist(),
        "std_measured_error_mm": (measured_std * 1000.0).tolist(),
        "mean_robot_goal_error_mm": None if robot_mean is None else robot_mean.tolist(),
        "yaw_model": yaw_fit,
        "workspace_model": workspace_fit,
        "perception_source_counts": perception_source_counts,
        "likely_causes": classify(
            yaw_fit,
            workspace_fit,
            robot_mean,
            perception_source_counts,
        ),
        "application_note": (
            "Suggested offsets are incremental corrections relative to collection_offsets. "
            "Validate on held-out yaw and workspace positions before applying to grasp execution."
        ),
        "collection_offsets": collection_offsets,
    }


def print_report(report):
    print("\nDiagnosis report", flush=True)
    print("samples={}".format(report["sample_count"]), flush=True)
    print(
        "mean measured error mm={} std mm={}".format(
            [round(value, 3) for value in report["mean_measured_error_mm"]],
            [round(value, 3) for value in report["std_measured_error_mm"]],
        ),
        flush=True,
    )
    print(
        "perception sources={}".format(report["perception_source_counts"]),
        flush=True,
    )
    if report["yaw_model"]:
        model = report["yaw_model"]
        print(
            "yaw model: base_error_mm={} local_error_mm={} residual_rms_mm={:.3f}".format(
                [round(1000.0 * value, 3) for value in model["base_error_m"]],
                [round(1000.0 * value, 3) for value in model["yaw_local_error_m"]],
                model["rms_residual_mm"],
            ),
            flush=True,
        )
        print(
            "suggested incremental base delta mm={} local delta mm={}".format(
                [round(1000.0 * value, 3) for value in model["suggested_base_offset_delta_m"]],
                [
                    round(1000.0 * value, 3)
                    for value in model["suggested_tool_offset_yaw_local_delta_m"]
                ],
            ),
            flush=True,
        )
        print(
            "suggested updated tool offsets: base={} yaw_local={}".format(
                model["suggested_updated_tool_offset_base_m"],
                model["suggested_updated_tool_offset_yaw_local_m"],
            ),
            flush=True,
        )
    if report["workspace_model"]:
        print(
            "workspace suggested affine_xy={}".format(report["workspace_model"]["suggested_affine_xy"]),
            flush=True,
        )
    print("likely causes={}".format(report["likely_causes"]), flush=True)


def analyze(args):
    samples_path = project_path(args.samples_json)
    dataset = load_json(samples_path)
    report = analyze_dataset(dataset)
    output = project_path(args.output) if args.output else os.path.join(
        os.path.dirname(samples_path),
        "diagnosis_report.json",
    )
    write_json(output, report)
    print_report(report)
    print("Saved report: {}".format(output), flush=True)
    return 0


def main():
    args = parse_args()
    if args.command == "collect":
        return collect(args)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
