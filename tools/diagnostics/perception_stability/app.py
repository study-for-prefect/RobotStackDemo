"""Top-level perception stability diagnostic workflow."""

import os
import time
from collections import defaultdict

from robot_scene_pipeline.detector_runtime import DetectorModel
from robot_scene_pipeline.io_utils import project_path, write_json
from robot_scene_pipeline.snapshot_pipeline import (
    apply_detector_config,
    drop_transient_detection_fields,
)

from .analysis import summarize_group, summarize_spatial_clusters, write_records_csv
from .arguments import parse_args
from .capture import jsonable, start_realsense_session, stop_realsense_session
from .sampling import run_one_sample

def main() -> None:
    args = parse_args()
    args = apply_detector_config(args)
    args.output_dir = project_path(args.output_dir)
    args.image_in = project_path(args.image_in) if args.image_in else ""
    args.detector_weight = project_path(args.detector_weight)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pre_capture_delay > 0:
        print("Waiting {:.2f}s before first capture...".format(args.pre_capture_delay), flush=True)
        time.sleep(args.pre_capture_delay)

    detector = DetectorModel(args)
    session = None
    all_records = []
    sample_errors = []
    try:
        if not args.image_in:
            session = start_realsense_session(args)
        for sample_index in range(1, int(args.samples) + 1):
            print("Capturing sample {}/{}...".format(sample_index, args.samples), flush=True)
            try:
                records = run_one_sample(args, detector, sample_index, session)
            except Exception as exc:
                sample_dir = os.path.join(args.output_dir, "sample_{:03d}".format(sample_index))
                os.makedirs(sample_dir, exist_ok=True)
                error = {
                    "sample_index": sample_index,
                    "timestamp": time.time(),
                    "error": str(exc),
                    "sample_dir": sample_dir,
                }
                sample_errors.append(error)
                write_json(os.path.join(sample_dir, "sample_error.json"), error)
                print("  sample failed: {}".format(exc), flush=True)
                if sample_index < int(args.samples) and args.sleep_s > 0:
                    time.sleep(float(args.sleep_s))
                continue
            all_records.extend(records)
            print("  recorded {} object(s)".format(len(records)), flush=True)
            if sample_index < int(args.samples) and args.sleep_s > 0:
                time.sleep(float(args.sleep_s))
    finally:
        stop_realsense_session(session)

    records_json_path = os.path.join(args.output_dir, "stability_records.json")
    records_csv_path = os.path.join(args.output_dir, "stability_records.csv")
    summary_path = os.path.join(args.output_dir, "stability_summary.json")
    write_json(records_json_path, jsonable({"records": all_records}))
    write_records_csv(records_csv_path, all_records)

    groups = defaultdict(list)
    for row in all_records:
        groups[(row.get("object_id"), row.get("label"))].append(row)
    summaries = [summarize_group(args, key, rows) for key, rows in sorted(groups.items(), key=lambda item: str(item[0]))]
    payload = {
        "schema_version": "perception_stability_probe_v1",
        "output_dir": args.output_dir,
        "sample_count_requested": int(args.samples),
        "sample_error_count": len(sample_errors),
        "record_count": len(all_records),
        "thresholds": {
            "xy_std_m": args.xy_std_threshold_m,
            "top_z_std_m": args.top_z_std_threshold_m,
            "dimension_range_m": args.dimension_range_threshold_m,
            "yaw_large_jitter_deg": args.yaw_large_jitter_deg,
        },
        "summaries": summaries,
        "spatial_cluster_radius_m": float(getattr(args, "spatial_cluster_radius_m", 0.02)),
        "spatial_cluster_summaries": summarize_spatial_clusters(args, all_records),
        "records_json": records_json_path,
        "records_csv": records_csv_path,
        "sample_errors": sample_errors,
        "recommendation": (
            "Do not tune robot compensation until the target object passes XY, top_z, and dimension stability."
        ),
    }
    write_json(summary_path, jsonable(payload))

    print("\nSaved records: {}".format(records_csv_path), flush=True)
    print("Saved summary: {}".format(summary_path), flush=True)
    for item in summaries:
        print(
            "object_id={} label='{}' pass={} xy_std_max={} top_z_std={} dim_range_max={}".format(
                item["object_id"],
                item["label"],
                item["pass_acceptance"],
                item["center_xy_std_max_m"],
                item["top_z_std_m"],
                item["dimensions_range_max_m"],
            ),
            flush=True,
        )
