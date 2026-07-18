"""Constrained VLM review of detector semantics before track creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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

    candidate_views = [_candidate_view(item) for item in candidates]
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
    reviewed = _apply_decisions(raw_scene, candidates, result.parsed_decision)
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
        "safety_limit": "objects without a supplied mask/depth candidate cannot be promoted",
        "low_confidence_candidate_pool_available": bool(raw_scene.get("semantic_review_candidates")),
        "decision_authority": "qwen_vl_visual_review_of_rgb_and_detector_overlay",
    })
    return reviewed


# Compatibility for callers outside the workflow that used the former,
# initial-observation-only public name.
review_initial_build_house_scene = review_build_house_scene


def _candidate_view(item: Mapping[str, Any]) -> dict[str, Any]:
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
        "silhouette_touches_image_boundary": _silhouette_touches_image_boundary(item),
        "shape_choice_constraint": ["square", "triangle"] if square_triangle_review else list(SHAPES),
    }


def _apply_decisions(
    raw_scene: dict[str, Any],
    candidates: list[dict[str, Any]],
    decision: Mapping[str, Any] | None,
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
    ordered = sorted(
        by_id.items(),
        key=lambda pair: (
            not bool(pair[1].get("primary_detector_passed", True)),
            -float(pair[1].get("confidence", pair[1].get("score", 0.0))),
            pair[0],
        ),
    )
    for candidate_id, item in ordered:
        primary = bool(item.get("primary_detector_passed", True))
        review = decisions.get(candidate_id)
        safety_rejection = _square_triangle_safety_rejection(item, review)
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
        )
        recover_square = _recover_measured_low_confidence_square(item, review)
        if not (primary or promote or recover_square):
            continue
        # Duplicate instance proposals can both clear the primary YOLO score
        # (the live yellow support was emitted once as square and again as
        # semi-circle with virtually identical masks).  Keep the stronger
        # proposal from the confidence-sorted order regardless of which score
        # threshold it crossed.  The IoU>=0.8 test remains deliberately too
        # strict to merge adjacent touching blocks.
        if any(_duplicate_candidate(item, kept) for kept in accepted):
            suppressed_duplicates.append(candidate_id)
            continue
        output = dict(item)
        if review and (review.get("accept") is True or recover_square):
            shape = str(review.get("corrected_shape"))
            output["yolo_original_label"] = output.get("label")
            output["label"] = _label_with_color(shape, output.get("visual_color"))
            output["vlm_semantic_review"] = dict(review)
            output["vlm_promoted_low_confidence_candidate"] = bool(not primary)
            output["low_confidence_square_geometry_recovery"] = recover_square
        if recover_square:
            recovered_low_confidence.append(candidate_id)
        accepted.append(output)
    return {
        **raw_scene,
        "objects": accepted,
        "semantic_review_suppressed_duplicate_ids": suppressed_duplicates,
        "semantic_review_safety_rejections": safety_rejections,
        "semantic_review_recovered_candidate_ids": recovered_low_confidence,
    }


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


def _square_triangle_safety_rejection(
    item: Mapping[str, Any],
    review: Mapping[str, Any] | None,
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
    if _silhouette_touches_image_boundary(item):
        return "green_square_silhouette_clipped_at_image_boundary"
    aspect = item.get("footprint_aspect_ratio")
    if aspect is not None and float(aspect) > 1.6:
        return "green_square_geometry_inconsistent_with_support_footprint"
    return None


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
