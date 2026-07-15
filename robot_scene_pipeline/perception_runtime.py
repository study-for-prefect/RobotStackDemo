"""Shared single-frame perception runtime for snapshot and server paths."""

import os
from typing import Any, Dict, Tuple

import cv2

from .depth_geometry import attach_3d, build_private_state, coordinate_convention, draw_annotated
from .io_utils import write_json
from .snapshot_pipeline import drop_transient_detection_fields
from .tabletop_geometry import attach_tabletop_geometry
from .tf_transform import (
    DEFAULT_TF_POINT_MODE,
    attach_base_coordinates,
    resolved_transform_matrix,
    transform_summary,
)
from .visual_color import attach_visual_colors


def detector_metadata(args: Any) -> Dict[str, Any]:
    return {
        "weight": args.detector_weight,
        "score_threshold": args.score_thresh,
        "iou": args.detector_iou,
        "imgsz": args.detector_imgsz,
        "device": args.detector_device,
        "detector_scale": args.detector_scale,
    }


def scene_output_paths(output_dir: str) -> Dict[str, str]:
    return {
        "snapshot": os.path.join(output_dir, "snapshot.jpg"),
        "annotated": os.path.join(output_dir, "annotated_detector.jpg"),
        "objects": os.path.join(output_dir, "detector_objects_3d.json"),
        "candidates": os.path.join(output_dir, "detector_candidates.json"),
        "private_state": os.path.join(output_dir, "private_scene_state.json"),
        "tf_status": os.path.join(output_dir, "tf_status.json"),
        "tabletop": os.path.join(output_dir, "tabletop_geometry.json"),
    }


def process_rgbd_scene(
    args: Any,
    detector: Any,
    frame_bgr: Any,
    depth_frame: Any,
    intrinsics: Any,
    used_profile: Dict[str, Any],
    output_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Run detection, depth geometry, TF, tabletop geometry, and write snapshot outputs."""
    os.makedirs(output_dir, exist_ok=True)
    paths = scene_output_paths(output_dir)
    cv2.imwrite(paths["snapshot"], frame_bgr)

    detections, candidates = detector.predict(
        frame_bgr,
        args.score_thresh,
        args.detector_scale,
        args.max_detections,
    )
    detections = attach_visual_colors(detections, frame_bgr, candidates)
    metadata = detector_metadata(args)
    write_json(paths["candidates"], {**metadata, "candidates": candidates[: args.debug_topk]})

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
            getattr(args, "tf_point_mode", DEFAULT_TF_POINT_MODE),
        )
        transform_matrix = resolved_transform_matrix(transform)
        tf_payload.update({"status": "ok", "transform": transform_summary(transform)})
    write_json(paths["tf_status"], tf_payload)

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
    write_json(paths["tabletop"], tabletop_payload)

    detections = drop_transient_detection_fields(detections)
    write_json(
        paths["objects"],
        {
            **metadata,
            "max_detections": args.max_detections,
            "coordinate_convention": coordinate_convention(args.camera_frame),
            "tf": tf_payload,
            "tabletop": tabletop_payload,
            "objects": detections,
        },
    )

    annotated = draw_annotated(frame_bgr, detections)
    cv2.imwrite(paths["annotated"], annotated)
    private_state = build_private_state(
        args,
        detections,
        paths["snapshot"],
        paths["annotated"],
        used_profile,
        table_plane=table_plane,
    )
    private_state["perception_source"] = "perception_server"
    write_json(paths["private_state"], private_state)
    return private_state, paths
