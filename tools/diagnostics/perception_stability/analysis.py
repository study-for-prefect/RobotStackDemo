"""Statistical summaries for repeated perception samples."""

import csv
import math
from collections import defaultdict

import numpy as np

def finite_array(values, width=None):
    output = []
    for value in values:
        if value is None:
            continue
        if width is None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                output.append(number)
        else:
            if not isinstance(value, (list, tuple)) or len(value) < width:
                continue
            row = []
            ok = True
            for item in value[:width]:
                try:
                    number = float(item)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not math.isfinite(number):
                    ok = False
                    break
                row.append(number)
            if ok:
                output.append(row)
    if not output:
        return np.empty((0, width), dtype=float) if width is not None else np.empty((0,), dtype=float)
    return np.asarray(output, dtype=float)


def sample_std(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    return float(np.std(values, ddof=1))


def axial_yaw_std_deg(yaws_deg):
    yaws = finite_array(yaws_deg)
    if yaws.size < 2:
        return None
    doubled = np.deg2rad(yaws * 2.0)
    resultant = np.abs(np.mean(np.exp(1j * doubled)))
    resultant = max(resultant, 1e-12)
    circular_std = math.sqrt(max(0.0, -2.0 * math.log(resultant)))
    return math.degrees(circular_std) / 2.0


def summarize_group(args, key, rows):
    centers = finite_array([row.get("geometry_center_m") for row in rows], width=3)
    top_z = finite_array([row.get("top_z_base_m") for row in rows])
    dimensions = finite_array([row.get("dimensions_m") for row in rows], width=3)
    point_counts = finite_array([row.get("pointcloud_point_count") for row in rows])
    yaw_values = [row.get("table_yaw_deg") for row in rows]

    xy_std = [sample_std(centers[:, axis]) for axis in (0, 1)] if len(centers) >= 2 else [None, None]
    xy_std_max = None if any(value is None for value in xy_std) else max(xy_std)
    top_z_std = sample_std(top_z)
    dim_std = [sample_std(dimensions[:, axis]) for axis in range(3)] if len(dimensions) >= 2 else [None, None, None]
    dim_range = (
        (np.max(dimensions, axis=0) - np.min(dimensions, axis=0)).astype(float).tolist()
        if len(dimensions) >= 2
        else [None, None, None]
    )
    dim_range_max = None if any(value is None for value in dim_range) else float(max(dim_range))
    point_count_std = sample_std(point_counts)
    point_count_range = (
        float(np.max(point_counts) - np.min(point_counts)) if len(point_counts) >= 2 else None
    )
    point_count_cv = None
    if point_count_std is not None and len(point_counts) and float(np.mean(point_counts)) > 0:
        point_count_cv = float(point_count_std / float(np.mean(point_counts)))
    yaw_std = axial_yaw_std_deg(yaw_values)
    label = key[1]

    checks = {
        "xy_std_pass": xy_std_max is not None and xy_std_max < args.xy_std_threshold_m,
        "top_z_std_pass": top_z_std is not None and top_z_std < args.top_z_std_threshold_m,
        "dimension_range_pass": dim_range_max is not None and dim_range_max < args.dimension_range_threshold_m,
    }
    notes = []
    if xy_std_max is not None and xy_std_max > args.xy_std_threshold_m:
        notes.append("center XY std > {:.3f} m: perception is unstable".format(args.xy_std_threshold_m))
    if top_z_std is not None and top_z_std > args.top_z_std_threshold_m:
        notes.append("top_z std > {:.3f} m: depth/table estimation is unstable".format(args.top_z_std_threshold_m))
    if dim_range_max is not None and dim_range_max > args.dimension_range_threshold_m:
        notes.append("dimension range > {:.3f} m: size estimate is unstable".format(args.dimension_range_threshold_m))
    if yaw_std is not None and yaw_std > args.yaw_large_jitter_deg:
        notes.append("yaw jitter is large")
    if "square" in str(label).lower():
        notes.append("square-like label: do not rely on yaw for compensation")
    if point_count_cv is not None and point_count_cv > 0.20:
        notes.append("point_count coefficient of variation > 20%: check mask/depth alignment or depth noise")

    return {
        "object_id": key[0],
        "label": label,
        "sample_count": len(rows),
        "valid_geometry_count": int(sum(bool(row.get("pointcloud_geometry_valid")) for row in rows)),
        "center_xy_std_m": xy_std,
        "center_xy_std_max_m": xy_std_max,
        "top_z_std_m": top_z_std,
        "dimensions_std_m": dim_std,
        "dimensions_range_m": dim_range,
        "dimensions_range_max_m": dim_range_max,
        "yaw_axial_std_deg": yaw_std,
        "pointcloud_point_count_std": point_count_std,
        "pointcloud_point_count_range": point_count_range,
        "pointcloud_point_count_cv": point_count_cv,
        "checks": checks,
        "pass_acceptance": all(checks.values()),
        "notes": notes,
    }


def summarize_spatial_clusters(args, rows):
    clusters = []
    radius_m = float(getattr(args, "spatial_cluster_radius_m", 0.02))
    for row in rows:
        center = row.get("geometry_center_m")
        if not isinstance(center, list) or len(center) < 2:
            continue
        center_np = np.asarray(center[:3], dtype=float)
        best = None
        best_distance = float("inf")
        for cluster in clusters:
            distance = float(np.linalg.norm(center_np[:2] - cluster["mean_center_m"][:2]))
            if distance < best_distance:
                best = cluster
                best_distance = distance
        if best is not None and best_distance <= radius_m:
            best["rows"].append(row)
            best["mean_center_m"] = np.mean(
                [np.asarray(item["geometry_center_m"][:3], dtype=float) for item in best["rows"]],
                axis=0,
            )
        else:
            clusters.append({"rows": [row], "mean_center_m": center_np})

    summaries = []
    for index, cluster in enumerate(sorted(clusters, key=lambda item: (item["mean_center_m"][0], item["mean_center_m"][1])), 1):
        label_counts = defaultdict(int)
        object_id_counts = defaultdict(int)
        sample_indices = set()
        for row in cluster["rows"]:
            label_counts[str(row.get("label"))] += 1
            object_id_counts[str(row.get("object_id"))] += 1
            sample_indices.add(int(row.get("sample_index")))
        summary = summarize_group(
            args,
            (index, max(label_counts.items(), key=lambda item: item[1])[0]),
            cluster["rows"],
        )
        summary.update(
            {
                "cluster_id": index,
                "mean_geometry_center_m": cluster["mean_center_m"].astype(float).tolist(),
                "unique_sample_count": len(sample_indices),
                "record_count": len(cluster["rows"]),
                "label_counts": dict(label_counts),
                "object_id_counts": dict(object_id_counts),
                "duplicate_records_detected": len(cluster["rows"]) > len(sample_indices),
                "missed_samples": max(0, int(args.samples) - len(sample_indices)),
            }
        )
        summaries.append(summary)
    return summaries


def write_records_csv(path, rows):
    fieldnames = [
        "sample_index",
        "timestamp",
        "object_id",
        "label",
        "pointcloud_geometry_valid",
        "geometry_center_m",
        "top_z_base_m",
        "dimensions_m",
        "table_yaw_deg",
        "table_yaw_valid",
        "pointcloud_point_count",
        "sample_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("geometry_center_m", "dimensions_m"):
                encoded[key] = json.dumps(encoded.get(key), ensure_ascii=False)
            writer.writerow(encoded)
