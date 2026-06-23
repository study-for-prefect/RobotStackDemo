"""One-sample perception evaluation."""

import os

import cv2

from robot_scene_pipeline.depth_geometry import (
    attach_3d,
    build_private_state,
    coordinate_convention,
    draw_annotated,
)
from robot_scene_pipeline.tabletop_geometry import attach_tabletop_geometry
from robot_scene_pipeline.tf_transform import (
    attach_base_coordinates,
    resolved_transform_matrix,
    transform_summary,
)

from .capture import (
    capture_from_session,
    detector_metadata,
    jsonable,
    should_record_object,
)

def run_one_sample(args, detector, sample_index, session):
    sample_dir = os.path.join(args.output_dir, "sample_{:03d}".format(sample_index))
    os.makedirs(sample_dir, exist_ok=True)

    frame_bgr, depth_frame, intrinsics, used_profile = capture_from_session(args, session, sample_index)
    snapshot_path = os.path.join(sample_dir, "snapshot.jpg")
    annotated_path = os.path.join(sample_dir, "annotated_detector.jpg")
    objects_path = os.path.join(sample_dir, "detector_objects_3d.json")
    candidates_path = os.path.join(sample_dir, "detector_candidates.json")
    private_state_path = os.path.join(sample_dir, "private_scene_state.json")
    tf_status_path = os.path.join(sample_dir, "tf_status.json")
    tabletop_path = os.path.join(sample_dir, "tabletop_geometry.json")

    if args.save_images:
        cv2.imwrite(snapshot_path, frame_bgr)

    detections, candidates = detector.predict(
        frame_bgr,
        args.score_thresh,
        args.detector_scale,
        args.max_detections,
    )
    write_json(candidates_path, {**detector_metadata(args), "candidates": candidates[: args.debug_topk]})

    detections = attach_3d(detections, depth_frame, intrinsics, args.depth_window)
    tf_payload = {
        "enabled": bool(args.use_tf),
        "base_frame": args.base_frame,
        "camera_frame": args.camera_frame,
        "status": "not_requested",
    }
    transform_matrix = None
    if args.use_tf:
        detections, transform = attach_base_coordinates(
            detections,
            args.base_frame,
            args.camera_frame,
            args.tf_timeout,
            getattr(args, "tf_json", ""),
            getattr(args, "tf_point_mode", "optical-to-camera-link"),
        )
        transform_matrix = resolved_transform_matrix(transform)
        tf_payload.update({"status": "ok", "transform": transform_summary(transform)})
    write_json(tf_status_path, tf_payload)

    table_plane = None
    tabletop_payload = {"enabled": bool(args.estimate_tabletop), "status": "not_requested"}
    if args.estimate_tabletop:
        detections, table_plane = attach_tabletop_geometry(
            detections,
            depth_frame,
            intrinsics,
            args,
            transform_matrix=transform_matrix,
        )
        tabletop_payload.update(
            {
                "status": "ok",
                "table_plane": table_plane,
                "valid_object_count": sum(bool(det.get("pointcloud_geometry_valid")) for det in detections),
            }
        )
    write_json(tabletop_path, jsonable(tabletop_payload))

    clean_detections = drop_transient_detection_fields(detections)
    objects_payload = {
        **detector_metadata(args),
        "max_detections": args.max_detections,
        "coordinate_convention": coordinate_convention(args.camera_frame),
        "tf": tf_payload,
        "tabletop": tabletop_payload,
        "objects": clean_detections,
    }
    write_json(objects_path, jsonable(objects_payload))

    if args.save_images:
        annotated = draw_annotated(frame_bgr, clean_detections)
        cv2.imwrite(annotated_path, annotated)

    private_state = build_private_state(
        args,
        clean_detections,
        snapshot_path if args.save_images else None,
        annotated_path if args.save_images else None,
        used_profile,
        table_plane=table_plane,
    )
    write_json(private_state_path, jsonable(private_state))

    records = []
    for obj in private_state.get("objects", []):
        if not should_record_object(args, obj):
            continue
        records.append(
            {
                "sample_index": sample_index,
                "timestamp": private_state.get("timestamp"),
                "object_id": obj.get("id"),
                "label": obj.get("label"),
                "pointcloud_geometry_valid": obj.get("pointcloud_geometry_valid"),
                "geometry_center_m": obj.get("geometry_center_m"),
                "top_z_base_m": obj.get("top_z_base_m"),
                "dimensions_m": obj.get("dimensions_m"),
                "table_yaw_deg": obj.get("table_yaw_deg"),
                "table_yaw_valid": obj.get("table_yaw_valid"),
                "pointcloud_point_count": obj.get("pointcloud_point_count"),
                "sample_dir": sample_dir,
            }
        )
    return records
