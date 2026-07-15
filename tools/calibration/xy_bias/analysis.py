"""XY bias model fitting, classification, and reports."""

import math
import os

import numpy as np

from robot_scene_pipeline.io_utils import project_path, write_json

from .io import load_json

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
        "suggested_tcp_offset_tool_delta_xy_m": (-params[2:]).tolist(),
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
        tcp_offset = np.asarray(collection_offsets.get("tcp_offset_tool_m", [0.0, 0.0, 0.0]), dtype=float)
        tcp_offset[:2] += np.asarray(yaw_fit["suggested_tcp_offset_tool_delta_xy_m"], dtype=float)
        yaw_fit["suggested_updated_tcp_offset_tool_m"] = tcp_offset.tolist()
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


def build_stack_calibration(report):
    """Build a legacy review artifact; current stack_demo does not load it directly."""
    calibration = {
        "schema_version": "stack_demo_calibration_v1",
        "source": "xy_bias_diagnosis.analyze",
        "max_correction_m": {
            "xy_m": 0.02,
            "z_m": 0.015,
        },
    }
    yaw_model = report.get("yaw_model")
    if yaw_model:
        base_delta = yaw_model.get("suggested_base_offset_delta_m") or [0.0, 0.0]
        calibration["grasp_base_bias_m"] = [float(base_delta[0]), float(base_delta[1]), 0.0]
        calibration["tcp_offset_tool_m"] = [
            float(value) for value in yaw_model["suggested_updated_tcp_offset_tool_m"][:3]
        ]
    workspace_model = report.get("workspace_model")
    if workspace_model:
        calibration["affine_xy"] = workspace_model["suggested_affine_xy"]
    calibration.setdefault("grasp_base_bias_m", [0.0, 0.0, 0.0])
    calibration.setdefault("place_base_bias_m", [0.0, 0.0, 0.0])
    return calibration


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
            "suggested incremental fixed-bias delta mm={} tcp-tool delta xy mm={}".format(
                [round(1000.0 * value, 3) for value in model["suggested_base_offset_delta_m"]],
                [
                    round(1000.0 * value, 3)
                    for value in model["suggested_tcp_offset_tool_delta_xy_m"]
                ],
            ),
            flush=True,
        )
        print(
            "suggested updated tcp_offset_tool={}".format(
                model["suggested_updated_tcp_offset_tool_m"],
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
    if args.output_calibration_json:
        calibration = build_stack_calibration(report)
        calibration_output = project_path(args.output_calibration_json)
        write_json(calibration_output, calibration)
        print("Saved stack demo calibration: {}".format(calibration_output), flush=True)
    print_report(report)
    print("Saved report: {}".format(output), flush=True)
    return 0
