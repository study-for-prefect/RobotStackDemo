"""Constrained VLM review of detector semantics before track creation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from robot_scene_pipeline.io_utils import image_to_base64
from robot_scene_pipeline.ollama_policy_client import call_policy


SHAPES = ("square", "triangle", "rectangle", "concave_rectangle", "semi_circle", "circle")
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decisions", "visible_object_count"],
    "properties": {
        "visible_object_count": {"type": "integer", "minimum": 0},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "accept", "corrected_shape", "confidence"],
                "properties": {
                    "candidate_id": {"type": "integer"},
                    "accept": {"type": "boolean"},
                    "corrected_shape": {"type": "string", "enum": list(SHAPES)},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
    },
}

SPECIAL_SHAPES = ("rectangle", "concave_rectangle", "triangle", "square", "other", "uncertain")
SPECIAL_SHAPE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["scene_revision", "objects"],
    "properties": {
        "scene_revision": {"type": "integer"},
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": [
                "candidate_id", "shape_label", "shape_confidence", "long_axis_image_deg",
                "broad_face_visible", "broad_face_confidence", "groove_visible",
                "groove_opening_direction_camera", "triangle_right_angle_edge_direction_image",
                "occlusion", "orientation_confidence", "evidence",
            ],
            "properties": {
                "candidate_id": {"type": "string"},
                "shape_label": {"type": "string", "enum": list(SPECIAL_SHAPES)},
                "shape_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "long_axis_image_deg": {"type": ["number", "null"], "minimum": -180.0, "maximum": 180.0},
                "broad_face_visible": {"type": "boolean"},
                "broad_face_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "groove_visible": {"type": "boolean"},
                "groove_opening_direction_camera": {"type": "string", "enum": ["up", "down", "left", "right", "toward_camera", "away_from_camera", "unknown"]},
                "triangle_right_angle_edge_direction_image": {"type": "string", "enum": ["up", "down", "left", "right", "unknown"]},
                "occlusion": {"type": "string", "enum": ["none", "partial", "severe"]},
                "orientation_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            },
        }},
    },
}


def review_build_house_scene(
    args: Any,
    raw_scene: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Review one house observation and promote measured low-score candidates."""
    artifact = output_dir / "semantic_detection_review.json"
    candidate_source = raw_scene.get("semantic_review_candidates") or raw_scene.get("objects", ())
    candidates = [
        dict(item) for item in candidate_source
        if isinstance(item, Mapping) and item.get("id") is not None
    ]
    if bool(getattr(args, "no_vlm_semantic_review", False)) or bool(getattr(args, "no_image", False)):
        _write(artifact, {"status": "skipped", "reason": "disabled_or_no_image"})
        return raw_scene
    snapshot = Path(str(raw_scene.get("snapshot_image") or ""))
    if not candidates or not snapshot.is_file():
        _write(artifact, {"status": "skipped", "reason": "no_review_candidates_or_image"})
        return raw_scene

    geometry_model = _special_vlm_config(args)
    candidate_views = [_candidate_view(item, geometry_model) for item in candidates]
    request = {
        "protocol": "semantic_detection_review_v1",
        "task": "build_house",
        "allowed_shapes": list(SHAPES),
        "candidates": candidate_views,
        "rules": [
            "the first image is the clean RGB image and the second image, when present, is the detector overlay",
            "visually inspect every supplied candidate_id and correct its shape from the visible outline",
            "metric dimensions and mask statistics are context only and must never replace visual inspection",
            (
                "for green candidates marked mandatory_square_triangle_review, ignore YOLO confidence and "
                "choose square versus triangle from the silhouette: an apex and sloped sides mean triangle; "
                "parallel opposite sides and four corners mean square"
            ),
            "green color alone is never evidence that an object is a triangle",
            "accept a below-threshold candidate only when a real distinct block is visible inside its bbox",
            "do not invent ids, boxes, masks, positions, or objects without a supplied candidate",
            "overlapping duplicate boxes for one physical block must not both be accepted",
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "你是视觉积木实例语义复核器，必须实际查看输入图片。"
                "第一张是干净RGB图，若有第二张则是带检测框的叠加图。"
                "只复核给定候选ID；不得创造物体或坐标。形状依据图片轮廓判断，"
                "颜色、尺寸和统计特征不能代替视觉判断。输出严格JSON。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            "images": _review_images(raw_scene, snapshot),
        },
    ]
    result = call_policy(
        args,
        "semantic_detection_review",
        messages,
        REVIEW_SCHEMA,
        artifact_dir=str(output_dir),
        temperature=0.0,
        top_p=0.8,
    )
    reviewed = _apply_decisions(
        raw_scene, candidates, result.parsed_decision, geometry_model=geometry_model,
    )
    _write(artifact, {
        "status": "applied" if result.schema_valid else "fallback_primary_detector_only",
        "request": request,
        "result": result.to_dict(),
        "accepted_object_ids": [item.get("id") for item in reviewed.get("objects", ())],
        "suppressed_duplicate_candidate_ids": list(
            reviewed.get("semantic_review_suppressed_duplicate_ids", ())
        ),
        "safety_rejected_candidate_ids": list(
            reviewed.get("semantic_review_safety_rejections", ())
        ),
        "recovered_low_confidence_candidate_ids": list(
            reviewed.get("semantic_review_recovered_candidate_ids", ())
        ),
        "metric_triangle_override_candidate_ids": list(
            reviewed.get("semantic_review_metric_triangle_override_ids", ())
        ),
        "safety_limit": "objects without a supplied mask/depth candidate cannot be promoted",
        "low_confidence_candidate_pool_available": bool(raw_scene.get("semantic_review_candidates")),
        "decision_authority": "qwen_vl_visual_review_of_rgb_and_detector_overlay",
    })
    fused = _review_special_shape_semantics(args, reviewed, output_dir, snapshot)
    _write_semantic_detector_overlay(snapshot, fused, output_dir)
    return fused


def _write_semantic_detector_overlay(
    snapshot: Path,
    scene: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Make the default detector overlay show fused labels, preserving raw YOLO."""
    if not snapshot.is_file():
        return
    annotated = output_dir / "annotated_detector.jpg"
    raw_annotated = output_dir / "annotated_detector_raw.jpg"
    if annotated.is_file() and not raw_annotated.exists():
        raw_annotated.write_bytes(annotated.read_bytes())
    with Image.open(snapshot) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in scene.get("objects", ()):
        if not isinstance(item, Mapping):
            continue
        bbox = list(item.get("bbox_xyxy_px") or item.get("bbox") or ())
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox[:4])
        label = str(item.get("label") or item.get("semantic_shape") or "object")
        candidate_id = item.get("id")
        caption = f"id={candidate_id} {label}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        text_box = draw.textbbox((x1, y1), caption)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x1, y1), caption, fill=(0, 255, 0))
    image.save(annotated, quality=95)


# Compatibility for callers outside the workflow that used the former,
# initial-observation-only public name.
review_initial_build_house_scene = review_build_house_scene


def _candidate_view(
    item: Mapping[str, Any],
    geometry_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bbox = list(item.get("bbox_xyxy_px") or item.get("bbox") or ())
    mask_area = item.get("mask_area_px")
    bbox_area = None
    if len(bbox) >= 4:
        bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
            0.0, float(bbox[3]) - float(bbox[1])
        )
    mask_fill = (
        float(mask_area) / bbox_area
        if mask_area is not None and bbox_area and bbox_area > 0.0
        else None
    )
    label = str(item.get("label") or "unknown")
    color = str(item.get("visual_color") or "unknown")
    square_triangle_review = bool(
        color == "green" and ("square" in label or "triangle" in label)
    )
    triangle_geometry = _green_triangle_metric_evidence(item, geometry_model)
    return {
        "candidate_id": int(item["id"]),
        "yolo_label": label,
        "yolo_confidence": round(float(item.get("confidence", item.get("score", 0.0))), 4),
        "primary_detector_passed": bool(item.get("primary_detector_passed", True)),
        "visual_color": color,
        "bbox_xyxy_px": bbox,
        "mask_bbox_fill_ratio": None if mask_fill is None else round(mask_fill, 4),
        "dimensions_m": list(item.get("dimensions_m") or ()),
        "footprint_aspect_ratio": item.get("footprint_aspect_ratio"),
        "metric_geometry_valid": bool(item.get("pointcloud_geometry_valid")),
        "mandatory_square_triangle_review": square_triangle_review,
        "green_square_triangle_metric_evidence": triangle_geometry,
        "silhouette_touches_image_boundary": _silhouette_touches_image_boundary(item),
        "shape_choice_constraint": ["square", "triangle"] if square_triangle_review else list(SHAPES),
    }


def _apply_decisions(
    raw_scene: dict[str, Any],
    candidates: list[dict[str, Any]],
    decision: Mapping[str, Any] | None,
    *,
    geometry_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    by_id = {int(item["id"]): item for item in candidates}
    decisions = {
        int(item["candidate_id"]): item
        for item in (decision or {}).get("decisions", ())
        if isinstance(item, Mapping) and item.get("candidate_id") in by_id
    }
    accepted = []
    suppressed_duplicates = []
    safety_rejections = []
    recovered_low_confidence = []
    geometry_square_overrides = []
    metric_triangle_overrides = []
    metric_triangle_override_ids = {
        candidate_id
        for candidate_id, item in by_id.items()
        if _metric_triangle_review_override(
            item,
            decisions.get(candidate_id),
            geometry_model=geometry_model,
            overlapping_visual_confirmation=any(
                other_id != candidate_id
                and _overlapping_visible_green_proposal(item, other)
                and str(other.get("visual_color") or "").lower() == "green"
                and (decisions.get(other_id) or {}).get("accept") is True
                and (decisions.get(other_id) or {}).get("confidence") in {"high", "medium"}
                for other_id, other in by_id.items()
            ),
        )
    }
    ordered = sorted(
        by_id.items(),
        key=lambda pair: (
            pair[0] not in metric_triangle_override_ids,
            not bool(pair[1].get("primary_detector_passed", True)),
            -float(pair[1].get("confidence", pair[1].get("score", 0.0))),
            pair[0],
        ),
    )
    for candidate_id, item in ordered:
        primary = bool(item.get("primary_detector_passed", True))
        review = decisions.get(candidate_id)
        metric_triangle_override = _metric_triangle_review_override(
            item,
            review,
            geometry_model=geometry_model,
            overlapping_visual_confirmation=candidate_id in metric_triangle_override_ids,
        )
        if metric_triangle_override:
            review = {
                **dict(review or {}),
                "corrected_shape": "triangle",
                "visual_corrected_shape": str((review or {}).get("corrected_shape") or ""),
                "metric_triangle_shape_override": True,
            }
            metric_triangle_overrides.append(candidate_id)
        physical_rejection = _physical_candidate_rejection(item, geometry_model)
        safety_rejection = physical_rejection or (
            None if metric_triangle_override else _square_triangle_safety_rejection(
                item, review, geometry_model=geometry_model,
            )
        )
        if safety_rejection is not None:
            safety_rejections.append({
                "candidate_id": candidate_id,
                "reason": safety_rejection,
            })
            continue
        promote = bool(
            review
            and review.get("accept") is True
            and review.get("confidence") in {"high", "medium"}
            and item.get("pointcloud_geometry_valid")
        ) or metric_triangle_override
        recover_square = _recover_measured_low_confidence_square(item, review)
        if not (primary or promote or recover_square):
            continue
        # Duplicate instance proposals can both clear the primary YOLO score
        # (the live yellow support was emitted once as square and again as
        # semi-circle with virtually identical masks).  Keep the stronger
        # proposal from the confidence-sorted order regardless of which score
        # threshold it crossed.  The IoU>=0.8 test remains deliberately too
        # strict to merge adjacent touching blocks.
        if any(
            _duplicate_candidate(item, kept)
            or (
                _overlapping_visible_green_proposal(item, kept)
                and (
                    metric_triangle_override
                    or bool(kept.get("metric_triangle_shape_override"))
                )
            )
            for kept in accepted
        ):
            suppressed_duplicates.append(candidate_id)
            continue
        output = dict(item)
        geometry_square = _recover_high_fill_yellow_cube(item, review)
        if review and (review.get("accept") is True or recover_square or metric_triangle_override):
            shape = str(review.get("corrected_shape"))
            output["yolo_original_label"] = output.get("label")
            output["label"] = _label_with_color(shape, output.get("visual_color"))
            output["vlm_semantic_review"] = dict(review)
            output["vlm_promoted_low_confidence_candidate"] = bool(not primary)
            output["low_confidence_square_geometry_recovery"] = recover_square
            if metric_triangle_override:
                output["metric_triangle_shape_override"] = True
                output["shape_override_reason"] = (
                    "measured_2_to_1_to_1_triangle_geometry_conflicts_with_visual_square_verdict"
                )
        if geometry_square:
            # A semicircle can fill at most about pi/4 of its tight axis-aligned
            # bounding box.  A >0.88 filled, cube-like yellow instance is the
            # repeatedly observed support-label confusion, not a semicircle.
            output["yolo_original_label"] = item.get("label")
            output["label"] = _label_with_color("square", item.get("visual_color"))
            output["high_fill_cube_shape_override"] = True
            geometry_square_overrides.append(candidate_id)
        if recover_square:
            recovered_low_confidence.append(candidate_id)
        accepted.append(output)
    return {
        **raw_scene,
        "objects": accepted,
        "semantic_review_suppressed_duplicate_ids": suppressed_duplicates,
        "semantic_review_safety_rejections": safety_rejections,
        "semantic_review_recovered_candidate_ids": recovered_low_confidence,
        "semantic_review_geometry_square_override_ids": geometry_square_overrides,
        "semantic_review_metric_triangle_override_ids": metric_triangle_overrides,
    }


def _metric_triangle_review_override(
    item: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    *,
    geometry_model: Mapping[str, Any] | None = None,
    overlapping_visual_confirmation: bool = False,
) -> bool:
    """Keep a measured triangle when its current projection looks rectangular.

    After a tabletop 3-D reorientation the triangular prism can expose a side
    face whose RGB outline has four corners.  A supplied mask with valid depth,
    the calibrated 2s-by-s-by-s point-cloud model, and an unclipped silhouette
    are sufficient physical evidence even when the visual reviewer rejects the
    projected shape.  This never creates a candidate absent from the detector.
    """
    direct_visual_confirmation = bool(
        review
        and review.get("accept") is True
        and review.get("confidence") in {"high", "medium"}
    )
    reviewed_shape = str((review or {}).get("corrected_shape") or "")
    detector_label = str(item.get("label") or "").lower()
    supplied_square_triangle_candidate = any(
        token in detector_label for token in ("square", "triangle")
    )
    if not (
        item.get("pointcloud_geometry_valid")
        and supplied_square_triangle_candidate
        and not _silhouette_touches_image_boundary(item)
        and (
            not direct_visual_confirmation
            or reviewed_shape in {"", "square", "triangle"}
            or overlapping_visual_confirmation
        )
    ):
        return False
    metric = _green_triangle_metric_evidence(item, geometry_model)
    return bool(
        metric
        and metric.get("triangle_geometry_confirmed") is True
        and metric.get("square_geometry_confirmed") is False
    )


def _physical_candidate_rejection(
    item: Mapping[str, Any], geometry_model: Mapping[str, Any] | None = None,
) -> str | None:
    """Reject depth slivers that cannot be a physical block in this kit.

    Partial edges can receive a high detector/VLM score, but a 2--3 mm oriented
    footprint extent is far below the thinnest 14 mm roof component.  Such a
    proposal must not become either a task object or a MoveIt collision body.
    """
    if not item.get("pointcloud_geometry_valid"):
        return None
    dimensions = item.get("dimensions_m")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3:
        return None
    measured = [abs(float(value)) for value in dimensions[:3]]
    minimum_extent = float(
        (geometry_model or {}).get("minimum_physical_block_extent_m", 0.008)
    )
    if min(measured) < minimum_extent and max(measured) >= 0.015:
        return "physically_implausible_thin_block_geometry"
    return None


def _recover_measured_low_confidence_square(
    item: Mapping[str, Any],
    review: Mapping[str, Any] | None,
) -> bool:
    """Recover a distinct cube candidate the VLM visually classified as square.

    A low detector score often occurs after a support is placed beside another
    block.  The VLM can still identify its outline while declining the generic
    ``accept`` field at low confidence.  Recovery remains bounded to supplied
    mask/depth candidates with cube-like metric dimensions; duplicate boxes
    are filtered by the caller.
    """
    if (
        not review
        or review.get("confidence") != "low"
        or str(review.get("corrected_shape")) != "square"
        or not item.get("pointcloud_geometry_valid")
    ):
        return False
    color = str(item.get("visual_color") or "").lower()
    if color == "green":
        return False
    dimensions = item.get("dimensions_m")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3:
        return False
    measured = [float(value) for value in dimensions[:3]]
    smallest = min(measured)
    return bool(
        smallest >= 0.015
        and max(measured) <= 0.040
        and max(measured) / smallest <= 1.35
    )


def _recover_high_fill_yellow_cube(
    item: Mapping[str, Any],
    review: Mapping[str, Any] | None,
) -> bool:
    """Correct the live yellow cube repeatedly labelled as a semicircle."""
    detector_shape = str(item.get("label") or "").lower().replace(" ", "_")
    reviewed_shape = str((review or {}).get("corrected_shape") or "").lower()
    if (
        str(item.get("visual_color") or "").lower() != "yellow"
        or detector_shape != "semi_circle"
        or reviewed_shape not in {"", "semi_circle"}
        or not item.get("pointcloud_geometry_valid")
        or _silhouette_touches_image_boundary(item)
    ):
        return False
    view = _candidate_view(item)
    fill = view.get("mask_bbox_fill_ratio")
    aspect = item.get("footprint_aspect_ratio")
    dimensions = item.get("dimensions_m")
    if (
        fill is None or float(fill) < 0.88
        or aspect is None or float(aspect) > 1.20
        or not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3
    ):
        return False
    measured = [float(value) for value in dimensions[:3]]
    return bool(
        min(measured) >= 0.015
        and max(measured) <= 0.040
        and max(measured) / min(measured) <= 1.35
    )


def _square_triangle_safety_rejection(
    item: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    *,
    geometry_model: Mapping[str, Any] | None = None,
) -> str | None:
    """Never promote an unverified green YOLO square to a house support."""
    label = str(item.get("label") or "").lower()
    color = str(item.get("visual_color") or "").lower()
    if color != "green" or "square" not in label:
        return None
    reliable = bool(
        review
        and review.get("accept") is True
        and review.get("confidence") in {"high", "medium"}
    )
    if not reliable:
        return "green_square_missing_reliable_square_triangle_review"
    if str(review.get("corrected_shape")) != "square":
        return None
    metric = _green_triangle_metric_evidence(item, geometry_model)
    if metric.get("triangle_geometry_confirmed") is True:
        return "green_square_verdict_conflicts_with_triangle_metric_geometry"
    if _silhouette_touches_image_boundary(item):
        return "green_square_silhouette_clipped_at_image_boundary"
    aspect = item.get("footprint_aspect_ratio")
    if aspect is not None and float(aspect) > 1.6:
        return "green_square_geometry_inconsistent_with_support_footprint"
    return None


def _green_triangle_metric_evidence(
    item: Mapping[str, Any],
    geometry_model: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compare the measured 3-D prism with the known 2s-by-s-by-s triangle.

    ``dimensions_m`` are the oriented tabletop footprint length/width plus
    height.  For this kit the triangle's long footprint edge is its
    hypotenuse (2s), while both the triangular-face altitude and prism height
    equal the square-block edge s.  This is independent evidence alongside
    the RGB silhouette; it is not inferred from the object's green colour.
    """
    if str(item.get("visual_color") or "").lower() != "green":
        return None
    dimensions = item.get("dimensions_m")
    if (
        not item.get("pointcloud_geometry_valid")
        or not isinstance(dimensions, (list, tuple)) or len(dimensions) < 3
    ):
        return {"available": False}
    model = geometry_model or {}
    edge = float(model.get("square_edge_length_m", 0.024))
    tolerance = float(model.get("triangle_dimension_ratio_tolerance", 0.30))
    expected = (
        float(model.get("triangle_hypotenuse_to_square_edge_ratio", 2.0)),
        float(model.get("triangle_face_altitude_to_square_edge_ratio", 1.0)),
        float(model.get("triangle_prism_height_to_square_edge_ratio", 1.0)),
    )
    measured = tuple(float(value) for value in dimensions[:3])
    ratios = tuple(value / edge for value in measured)
    matches = tuple(
        abs(actual - target) <= tolerance * target
        for actual, target in zip(ratios, expected)
    )
    square_matches = tuple(abs(value - 1.0) <= tolerance for value in ratios)
    return {
        "available": True,
        "square_edge_reference_m": edge,
        "measured_hypotenuse_face_altitude_prism_height_m": [round(value, 4) for value in measured],
        "ratios_to_square_edge": [round(value, 3) for value in ratios],
        "expected_ratios": list(expected),
        "ratio_tolerance_fraction": tolerance,
        "triangle_geometry_confirmed": all(matches),
        "square_geometry_confirmed": all(square_matches),
    }


def _silhouette_touches_image_boundary(item: Mapping[str, Any]) -> bool:
    bbox = list(item.get("bbox_xyxy_px") or item.get("bbox") or ())
    image_shape = list(item.get("mask_shape") or ())
    if len(bbox) < 4 or len(image_shape) < 2:
        return False
    height, width = float(image_shape[0]), float(image_shape[1])
    margin_px = 2.0
    return bool(
        float(bbox[0]) <= margin_px
        or float(bbox[1]) <= margin_px
        or float(bbox[2]) >= width - margin_px
        or float(bbox[3]) >= height - margin_px
    )


def _duplicate_candidate(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Match duplicate low-score proposals without merging adjacent blocks."""
    first_box = list(first.get("bbox_xyxy_px") or first.get("bbox") or ())
    second_box = list(second.get("bbox_xyxy_px") or second.get("bbox") or ())
    if len(first_box) < 4 or len(second_box) < 4:
        return False
    intersection_x = max(0.0, min(float(first_box[2]), float(second_box[2])) - max(
        float(first_box[0]), float(second_box[0])
    ))
    intersection_y = max(0.0, min(float(first_box[3]), float(second_box[3])) - max(
        float(first_box[1]), float(second_box[1])
    ))
    intersection = intersection_x * intersection_y
    first_area = max(0.0, float(first_box[2]) - float(first_box[0])) * max(
        0.0, float(first_box[3]) - float(first_box[1])
    )
    second_area = max(0.0, float(second_box[2]) - float(second_box[0])) * max(
        0.0, float(second_box[3]) - float(second_box[1])
    )
    union = first_area + second_area - intersection
    return bool(union > 0.0 and intersection / union >= 0.8)


def _overlapping_visible_green_proposal(
    first: Mapping[str, Any], second: Mapping[str, Any],
) -> bool:
    """Match a nested partial green box to its full-object proposal."""
    if not (
        str(first.get("visual_color") or "").lower() == "green"
        and str(second.get("visual_color") or "").lower() == "green"
    ):
        return False
    first_box = list(first.get("bbox_xyxy_px") or first.get("bbox") or ())
    second_box = list(second.get("bbox_xyxy_px") or second.get("bbox") or ())
    if len(first_box) < 4 or len(second_box) < 4:
        return False
    intersection_x = max(0.0, min(float(first_box[2]), float(second_box[2])) - max(
        float(first_box[0]), float(second_box[0])
    ))
    intersection_y = max(0.0, min(float(first_box[3]), float(second_box[3])) - max(
        float(first_box[1]), float(second_box[1])
    ))
    intersection = intersection_x * intersection_y
    areas = [
        max(0.0, float(box[2]) - float(box[0]))
        * max(0.0, float(box[3]) - float(box[1]))
        for box in (first_box, second_box)
    ]
    return bool(min(areas) > 0.0 and intersection / min(areas) >= 0.85)


def _label_with_color(shape: str, color: Any) -> str:
    token = str(color or "unknown")
    display = shape.replace("_", " ")
    return f"{display} {token}" if shape == "square" and token != "unknown" else display


def _review_images(raw_scene: Mapping[str, Any], snapshot: Path) -> list[str]:
    paths = [snapshot]
    annotated = Path(str(raw_scene.get("annotated_image") or ""))
    if annotated.is_file() and annotated.resolve() != snapshot.resolve():
        paths.append(annotated)
    return [image_to_base64(str(path)) for path in paths]


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _review_special_shape_semantics(
    args: Any,
    raw_scene: dict[str, Any],
    output_dir: Path,
    snapshot: Path,
) -> dict[str, Any]:
    """Send one annotated overview plus bounded candidate crops per stateless batch."""
    config = _special_vlm_config(args)
    if not bool(config.get("enabled", True)):
        return raw_scene
    candidates = [dict(item) for item in raw_scene.get("objects", ()) if isinstance(item, Mapping)]
    maximum = int(config.get("maximum_candidates_per_request", 6))
    hard_maximum = int(config.get("hard_maximum_candidates_per_request", 8))
    if not 1 <= maximum <= hard_maximum <= 8:
        raise ValueError("invalid special-shape semantic batch limits")
    crop_dir = output_dir / "special_shape_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    overview = Path(str(raw_scene.get("annotated_image") or snapshot))
    if overview.is_file():
        (output_dir / "special_shape_overview.jpg").write_bytes(overview.read_bytes())
    crop_paths = {str(item["id"]): _write_crop(
        snapshot, item, crop_dir, config,
    ) for item in candidates}
    all_results = []
    rejected_results = []
    requests = []
    raw_outputs = []
    camera_rotation = _camera_rotation_base(output_dir)
    for batch_index in range(0, len(candidates), maximum):
        batch = candidates[batch_index:batch_index + maximum]
        ids = [str(item["id"]) for item in batch]
        image_paths = [str(overview)] + [str(crop_paths[item]) for item in ids if crop_paths[item]]
        text_estimate = max(
            1, len(json.dumps([_candidate_view(item, config) for item in batch])) // 3,
        )
        image_estimate = 2048 + 1536 * len(batch)
        num_predict, reserve = 1536, 2048
        if text_estimate + image_estimate + num_predict + reserve > 32768:
            raise ValueError("special-shape visual request exceeds the fixed 32768-token budget")
        request = {
            "protocol": "special_shape_semantic_review_v1",
            "scene_revision": int(raw_scene.get("scene_revision", 1)),
            "allowed_candidate_ids": ids,
            "allowed_shapes": list(SPECIAL_SHAPES),
            "candidates": [_candidate_view(item, config) for item in batch],
            "image_order": ["annotated_scene_overview", *[f"candidate_crop:{item}" for item in ids]],
            "forbidden_outputs": ["robot_coordinates", "euler_angles", "quaternions", "joints", "actions", "trajectories", "completion_claims"],
            "triangle_direction_rule": (
                "For an unobstructed triangle crop, locate the 90-degree corner and report "
                "the dominant image direction from the triangle centroid toward that corner. "
                "Use unknown only when the corner is clipped or genuinely occluded."
            ),
            "budget": {
                "text_token_estimate": text_estimate,
                "conservative_image_token_estimate": image_estimate,
                "image_count": len(image_paths),
                "image_resolution_summary": _image_summaries(image_paths),
                "effective_num_ctx": 32768,
                "candidate_batch_size": len(batch),
                "num_predict": num_predict,
                "reserve": reserve,
            },
        }
        messages = [{"role": "system", "content": (
            "只输出给定候选的受限视觉语义证据JSON。不得输出坐标、欧拉角、四元数、"
            "机械臂动作、轨迹或完成声明。三角形无遮挡时必须找到90度角，并按该角相对"
            "轮廓中心的主图像方向填写triangle_right_angle_edge_direction_image；只有裁切或"
            "真实遮挡时才允许unknown。"
        )}, {"role": "user", "content": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
             "images": [image_to_base64(path) for path in image_paths]}]
        result = call_policy(args, "special_shape_semantic_review", messages, SPECIAL_SHAPE_SCHEMA,
                             artifact_dir=str(output_dir), temperature=0.0, top_p=0.8)
        parsed = result.parsed_decision or _parse_json_object(result.content)
        validated = _validate_special_response(parsed, ids, request["scene_revision"])
        requests.append(request)
        raw_outputs.append(result.content)
        if validated is not None:
            all_results.extend(validated["objects"])
            rejected_results.extend(validated["rejected_objects"])
    (output_dir / "vlm_shape_request.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "vlm_shape_response_raw.txt").write_text("\n\n".join(raw_outputs), encoding="utf-8")
    _write(output_dir / "vlm_shape_response_validated.json", {
        "scene_revision": raw_scene.get("scene_revision"),
        "objects": all_results,
        "rejected_objects": rejected_results,
    })
    fused, fusion_log = _fuse_special_shape_evidence(
        raw_scene, all_results, config, snapshot=snapshot,
        camera_rotation_base=camera_rotation,
    )
    _write(output_dir / "shape_fusion.json", fusion_log)
    _write(output_dir / "object_orientation_evidence.json", {
        "scene_revision": raw_scene.get("scene_revision"),
        "objects": [{key: item.get(key) for key in (
            "id", "semantic_shape", "semantic_shape_confidence", "semantic_shape_uncertain",
            "long_axis_base", "visible_face_normal_base", "broad_face_normal_base",
            "groove_opening_normal_base", "designated_right_angle_edge_base",
            "triangle_vertical_profile", "fresh_3d_apex_up_confirmed",
            "orientation_xyzw", "orientation_confidence", "evidence_scene_revision",
        )} for item in fused.get("objects", ())],
    })
    return fused


def _special_vlm_config(args: Any) -> Mapping[str, Any]:
    path = Path(str(getattr(args, "planner_config", "config/stack_demo_planner.json")))
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return json.loads(path.read_text(encoding="utf-8"))["special_shape_vlm"]
    except Exception:
        return {"enabled": True, "maximum_candidates_per_request": 6,
                "hard_maximum_candidates_per_request": 8, "crop_padding_ratio": 0.15,
                "crop_long_edge_px": 384, "minimum_shape_confidence": 0.75,
                "minimum_orientation_confidence": 0.70}


def _write_crop(snapshot: Path, item: Mapping[str, Any], crop_dir: Path, config: Mapping[str, Any]) -> Path | None:
    bbox = list(item.get("bbox_xyxy_px") or item.get("bbox") or ())
    if not snapshot.is_file() or len(bbox) < 4:
        return None
    with Image.open(snapshot) as image:
        width, height = image.size
        padding = float(config.get("crop_padding_ratio", 0.15)) * max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        box = (max(0, int(bbox[0] - padding)), max(0, int(bbox[1] - padding)),
               min(width, int(bbox[2] + padding)), min(height, int(bbox[3] + padding)))
        crop = image.crop(box)
        maximum = int(config.get("crop_long_edge_px", 384))
        resampling = getattr(Image, "Resampling", Image)
        crop.thumbnail((maximum, maximum), resampling.LANCZOS)
        path = crop_dir / f"candidate_{item['id']}.jpg"
        crop.convert("RGB").save(path, quality=92)
        return path


def _validate_special_response(value: Mapping[str, Any] | None, allowed_ids: list[str], revision: int) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        response_revision = int(value.get("scene_revision", -1))
    except (TypeError, ValueError):
        return None
    if response_revision != revision:
        return None
    objects = value.get("objects")
    if not isinstance(objects, list):
        return None
    ids = [str(item.get("candidate_id")) for item in objects if isinstance(item, Mapping)]
    duplicates = {item for item in ids if ids.count(item) > 1}
    accepted, rejected = [], []
    for item in objects:
        errors = _special_object_errors(item, allowed_ids, duplicates)
        if errors:
            rejected.append({
                "candidate_id": item.get("candidate_id") if isinstance(item, Mapping) else None,
                "errors": errors,
            })
        else:
            accepted.append(dict(item))
    return {
        "scene_revision": revision,
        "objects": accepted,
        "rejected_objects": rejected,
    }


def _fuse_special_shape_evidence(
    raw_scene: dict[str, Any],
    decisions: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    snapshot: Path | None = None,
    camera_rotation_base: tuple[tuple[float, float, float], ...] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_id = {str(item["candidate_id"]): item for item in decisions}
    output, log = [], []
    for original in raw_scene.get("objects", ()):
        item = dict(original)
        decision = by_id.get(str(item.get("id")))
        yolo_shape = _shape_from_label(str(item.get("label") or ""))
        uncertain = True
        if decision is not None:
            decision = dict(decision)
            metric_triangle = _green_triangle_metric_evidence(item, config)
            metric_triangle_confirmed = bool(
                str(decision.get("shape_label")) == "triangle"
                and metric_triangle
                and metric_triangle.get("triangle_geometry_confirmed") is True
            )
            triangle_direction_source = "vlm"
            if (
                decision.get("shape_label") == "triangle"
                and decision.get("triangle_right_angle_edge_direction_image") == "unknown"
                and decision.get("occlusion") in {"none", "partial"}
                and decision.get("broad_face_visible") is True
                and float(decision.get("broad_face_confidence", 0.0)) >= 0.90
                and float(decision.get("orientation_confidence", 0.0)) >= 0.80
            ):
                silhouette_direction = _triangle_right_angle_direction_from_silhouette(
                    snapshot, item,
                )
                if silhouette_direction is not None:
                    decision["triangle_right_angle_edge_direction_image"] = silhouette_direction
                    triangle_direction_source = "mask_silhouette_right_angle"
            qwen_shape = str(decision["shape_label"])
            confidence = float(decision["shape_confidence"])
            uncertain = bool(
                qwen_shape in {"uncertain", "other"}
                or qwen_shape != yolo_shape
                or (
                    confidence < float(config.get("minimum_shape_confidence", 0.75))
                    and not metric_triangle_confirmed
                )
                or decision["occlusion"] == "severe"
            )
            item.update({
                "shape_hypotheses": {"yolo": yolo_shape, "qwen": qwen_shape},
                "semantic_shape": qwen_shape if not uncertain else yolo_shape,
                "semantic_shape_confidence": confidence,
                "orientation_evidence": dict(decision),
                "orientation_confidence": min(float(item.get("orientation_confidence", 0.0)), float(decision["orientation_confidence"])),
                "evidence_scene_revision": int(raw_scene.get("scene_revision", 1)),
                "evidence_image_ids": ["special_shape_overview", f"candidate_{item.get('id')}_crop"],
                "green_square_triangle_metric_evidence": metric_triangle,
                "metric_triangle_geometry_confirmed": metric_triangle_confirmed,
            })
            if qwen_shape == "triangle":
                item["triangle_right_angle_direction_source"] = triangle_direction_source
            semantic_axis_attached = _attach_shape_semantic_axis(
                item, decision, camera_rotation_base=camera_rotation_base,
            )
            vertical_profile = item.get("triangle_vertical_profile")
            if (
                not semantic_axis_attached
                and metric_triangle_confirmed
                and isinstance(vertical_profile, Mapping)
                and vertical_profile.get("apex_up_confirmed") is True
            ):
                item["designated_right_angle_edge_base"] = [0.0, 0.0, 1.0]
                item["triangle_right_angle_direction_source"] = (
                    "fresh_mask_depth_vertical_profile_apex_up"
                )
                item["fresh_3d_apex_up_confirmed"] = True
                semantic_axis_attached = True
            uncertain = uncertain or not semantic_axis_attached
        item["semantic_shape_uncertain"] = uncertain
        output.append(item)
        log.append({"candidate_id": item.get("id"), "yolo_shape": yolo_shape,
                    "qwen_shape": None if decision is None else decision["shape_label"],
                    "fused_shape": item.get("semantic_shape", yolo_shape), "uncertain": uncertain,
                    "triangle_right_angle_direction_source": item.get("triangle_right_angle_direction_source")})
    return {**raw_scene, "objects": output}, {"scene_revision": raw_scene.get("scene_revision"), "objects": log}


def _triangle_right_angle_direction_from_silhouette(
    snapshot: Path | None,
    item: Mapping[str, Any],
) -> str | None:
    """Resolve an unobstructed triangle's right-angle corner from its color silhouette."""
    bbox = list(item.get("bbox_xyxy_px") or item.get("bbox") or ())
    if snapshot is None or not snapshot.is_file() or len(bbox) < 4:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        image = cv2.imread(str(snapshot), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox[:4])
        if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
            return None
        padding = max(4, int(round(0.12 * max(x2 - x1, y2 - y1))))
        left, top = max(0, x1 - padding), max(0, y1 - padding)
        right, bottom = min(width, x2 + padding), min(height, y2 + padding)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color = str(item.get("visual_color") or "").lower()
        color_ranges = {
            "green": (((30, 45, 25), (95, 255, 255)),),
            "blue": (((90, 45, 25), (135, 255, 255)),),
            "yellow": (((15, 45, 25), (38, 255, 255)),),
            "red": (((0, 45, 25), (12, 255, 255)), ((165, 45, 25), (179, 255, 255))),
        }
        ranges = color_ranges.get(color)
        if not ranges:
            return None
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8),
                ),
            )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 0.15 * float(crop.shape[0] * crop.shape[1]):
            return None
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.045 * perimeter, True).reshape(-1, 2)
        if len(polygon) != 3:
            # Four visible corners are a projected prism/side face. Choosing
            # any three of them fabricates a right-angle direction and caused
            # an already-correct tabletop pose to be flipped again.
            return None
        triangle = polygon
        angles = []
        for index, vertex in enumerate(triangle):
            first = triangle[(index - 1) % 3].astype(float) - vertex.astype(float)
            second = triangle[(index + 1) % 3].astype(float) - vertex.astype(float)
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator <= 1e-9:
                return None
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            angles.append(math.degrees(math.acos(cosine)))
        right_index = min(range(3), key=lambda index: abs(angles[index] - 90.0))
        if abs(angles[right_index] - 90.0) > 15.0:
            return None
        centroid = triangle.astype(float).mean(axis=0)
        delta = triangle[right_index].astype(float) - centroid
        if abs(float(delta[0])) >= abs(float(delta[1])):
            return "right" if delta[0] > 0.0 else "left"
        return "down" if delta[1] > 0.0 else "up"
    except (OSError, TypeError, ValueError, cv2.error):
        return None


def _attach_shape_semantic_axis(
    item: dict[str, Any],
    decision: Mapping[str, Any],
    *,
    camera_rotation_base: tuple[tuple[float, float, float], ...] | None = None,
) -> bool:
    long_axis = item.get("long_axis_base")
    normal = item.get("visible_face_normal_base")
    if not (_vector3(long_axis) and _vector3(normal) and item.get("orientation_xyzw")):
        return False
    shape = str(decision["shape_label"])
    minimum = 0.70
    if float(decision["orientation_confidence"]) < minimum:
        return False
    if shape == "rectangle":
        if not decision["broad_face_visible"] or float(decision["broad_face_confidence"]) < minimum:
            return False
        item["broad_face_normal_base"] = list(normal)
        return True
    if shape == "concave_rectangle":
        if not decision["groove_visible"] or decision["groove_opening_direction_camera"] == "unknown":
            return False
        direction = _camera_direction_in_base(
            str(decision["groove_opening_direction_camera"]),
            camera_rotation_base,
            normal,
        )
        if direction is None:
            return False
        item["groove_opening_normal_base"] = list(direction)
        return True
    if shape == "triangle":
        direction = decision["triangle_right_angle_edge_direction_image"]
        if direction == "unknown":
            return False
        designated = _camera_direction_in_base(
            str(direction), camera_rotation_base, normal,
        )
        if designated is None:
            return False
        item["designated_right_angle_edge_base"] = list(designated)
        return True
    return shape == "square"


def _parse_json_object(content: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _special_object_errors(
    value: Any,
    allowed_ids: list[str],
    duplicates: set[str],
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["object_not_mapping"]
    errors = []
    candidate_id = str(value.get("candidate_id"))
    if candidate_id not in allowed_ids:
        errors.append("candidate_id_not_allowed")
    if candidate_id in duplicates:
        errors.append("duplicate_candidate_id")
    if value.get("shape_label") not in SPECIAL_SHAPES:
        errors.append("invalid_shape_label")
    for key in ("shape_confidence", "broad_face_confidence", "orientation_confidence"):
        number = value.get(key)
        if not isinstance(number, (int, float)) or not math.isfinite(float(number)) or not 0.0 <= float(number) <= 1.0:
            errors.append(f"{key}_outside_0_1")
    if not isinstance(value.get("broad_face_visible"), bool):
        errors.append("broad_face_visible_not_boolean")
    if not isinstance(value.get("groove_visible"), bool):
        errors.append("groove_visible_not_boolean")
    if value.get("groove_opening_direction_camera") not in {
        "up", "down", "left", "right", "toward_camera", "away_from_camera", "unknown",
    }:
        errors.append("invalid_groove_direction")
    if value.get("triangle_right_angle_edge_direction_image") not in {
        "up", "down", "left", "right", "unknown",
    }:
        errors.append("invalid_triangle_direction")
    if value.get("occlusion") not in {"none", "partial", "severe"}:
        errors.append("invalid_occlusion")
    return errors


def _camera_rotation_base(
    output_dir: Path,
) -> tuple[tuple[float, float, float], ...] | None:
    path = output_dir / "tf_status.json"
    try:
        matrix = json.loads(path.read_text(encoding="utf-8"))["transform"]["matrix_4x4"]
        rotation = tuple(tuple(float(matrix[row][column]) for column in range(3)) for row in range(3))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return rotation if all(math.isfinite(value) for row in rotation for value in row) else None


def _camera_direction_in_base(
    direction: str,
    rotation: tuple[tuple[float, float, float], ...] | None,
    visible_normal_base: Any,
) -> tuple[float, float, float] | None:
    if rotation is None or not _vector3(visible_normal_base):
        return None
    camera_vectors = {
        "left": (-1.0, 0.0, 0.0), "right": (1.0, 0.0, 0.0),
        "up": (0.0, -1.0, 0.0), "down": (0.0, 1.0, 0.0),
        "toward_camera": (0.0, 0.0, -1.0), "away_from_camera": (0.0, 0.0, 1.0),
    }
    vector = camera_vectors.get(direction)
    if vector is None:
        return None
    transformed = tuple(sum(rotation[row][column] * vector[column] for column in range(3)) for row in range(3))
    if direction in {"left", "right", "up", "down"}:
        normal = _normalize3(visible_normal_base)
        dot = sum(transformed[index] * normal[index] for index in range(3))
        transformed = tuple(transformed[index] - dot * normal[index] for index in range(3))
    if math.sqrt(sum(value * value for value in transformed)) <= 1e-6:
        return None
    return _normalize3(transformed)


def _shape_from_label(label: str) -> str:
    normalized = label.lower().replace(" ", "_")
    return next((shape for shape in ("concave_rectangle", "rectangle", "triangle", "square") if shape in normalized), "other")


def _vector3(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(math.isfinite(float(item)) for item in value)


def _cross3(left: Any, right: Any) -> tuple[float, float, float]:
    return (left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0])


def _normalize3(value: Any) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    return tuple(float(item) / norm for item in value)


def _image_summaries(paths: list[str]) -> list[dict[str, int]]:
    output = []
    for path in paths:
        with Image.open(path) as image:
            output.append({"width_px": image.width, "height_px": image.height})
    return output
