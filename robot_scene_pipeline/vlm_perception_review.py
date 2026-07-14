"""VLM review for an intermittent missed detection in a static scene."""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Dict, List, Tuple

from .io_utils import image_to_base64
from .ollama_policy_client import call_policy
from .organize_scope import color_value_from_object, organize_scope_objects


STATIC_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_unchanged", "missing_objects", "confidence"],
    "properties": {
        "scene_unchanged": {"type": "boolean"},
        "missing_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["track_id", "still_visible", "visual_color", "confidence"],
                "properties": {
                    "track_id": {"type": "string"},
                    "still_visible": {"type": "boolean"},
                    "visual_color": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}

EXPECTED_REVIEW_SCHEMA = copy.deepcopy(STATIC_REVIEW_SCHEMA)
EXPECTED_REVIEW_SCHEMA["required"][0] = "expected_scene_consistent"
EXPECTED_REVIEW_SCHEMA["properties"]["expected_scene_consistent"] = (
    EXPECTED_REVIEW_SCHEMA["properties"].pop("scene_unchanged")
)

LOW_CONFIDENCE_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates", "confidence"],
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "is_real_block", "visual_color", "confidence"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "is_real_block": {"type": "boolean"},
                    "visual_color": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


def review_and_recover_low_confidence_candidates(
    args: Any,
    primary_state: dict,
    low_confidence_state: dict,
    artifact_dir: str,
    match_distance_m: float = 0.015,
) -> Tuple[dict, dict]:
    """Use VLM only to confirm extra low-threshold detections; retain RGB-D geometry."""
    primary = organize_scope_objects(primary_state)
    low = organize_scope_objects(low_confidence_state)
    extras = _extra_low_confidence_clusters(primary, low, match_distance_m)
    report: Dict[str, Any] = {
        "triggered": bool(extras),
        "primary_count": len(primary),
        "low_confidence_count": len(low),
        "extra_candidate_ids": [_review_key(item) for item in extras],
        "recovered": False,
    }
    if not extras:
        report["reason"] = "no_new_low_confidence_geometry"
        return primary_state, report

    images = []
    for path in (
        primary_state.get("snapshot_image"),
        primary_state.get("annotated_image"),
        low_confidence_state.get("annotated_image"),
    ):
        if path and os.path.isfile(path):
            images.append(image_to_base64(path))
    if not images:
        report["reason"] = "review_images_unavailable"
        return primary_state, report

    prompt = (
        "你只复核低置信YOLO候选是不是桌面上的真实积木，不规划动作、不生成或修改坐标。"
        "图像依次为当前原图、正常阈值标注图、低阈值标注图。extra_candidates 中每项已有RGB-D三维几何；"
        "只看候选框覆盖的实物像素，忽略标注文字和检测器置信度。逐字复制 candidate_id。真实积木才 "
        "is_real_block=true，visual_color 填实际可见颜色。输出 confidence 是你看图确认实物的把握，不能抄YOLO分数。"
        "框若是已有积木的重复框、背景或无法确认，必须为 false。只输出严格JSON。\n"
        + json.dumps({
            "normal_yolo_detections": [_compact_expected(item) for item in primary],
            "extra_candidates": [_compact_candidate(item) for item in extras],
            "output_schema": LOW_CONFIDENCE_REVIEW_SCHEMA,
        }, ensure_ascii=False, indent=2)
    )
    result = call_policy(
        args,
        "action_proposal",
        [
            {"role": "system", "content": "你是保守的机器人低置信目标视觉复核器。"},
            {"role": "user", "content": prompt, "images": images},
        ],
        LOW_CONFIDENCE_REVIEW_SCHEMA,
        artifact_dir=artifact_dir,
        temperature=0.0,
        top_p=0.8,
    )
    decision = result.parsed_decision
    report.update({
        "vlm_status": result.generation_status,
        "vlm_error": result.error_message,
        "vlm_decision": decision,
    })
    if not isinstance(decision, dict) or float(decision.get("confidence", 0.0)) < 0.8:
        report["reason"] = "vlm_review_unavailable_or_uncertain"
        return primary_state, report

    verdicts = {str(item.get("candidate_id")): item for item in decision.get("candidates", [])}
    confirmed = []
    for item in extras:
        verdict = verdicts.get(_review_key(item))
        if (
            verdict is not None
            and verdict.get("is_real_block")
            and float(verdict.get("confidence", 0.0)) >= 0.8
            and str(verdict.get("visual_color") or "").lower() == color_value_from_object(item)
        ):
            confirmed.append(item)
    if not confirmed:
        report["reason"] = "vlm_confirmed_no_extra_blocks"
        return primary_state, report

    output, recovered_ids = _restore_low_confidence_objects(primary_state, confirmed)
    report.update({
        "recovered": True,
        "reason": "low_confidence_rgbd_candidate_vlm_confirmed",
        "recovered_object_ids": recovered_ids,
    })
    output["initial_low_confidence_perception_recovery"] = copy.deepcopy(report)
    return output, report


def review_and_recover_static_misses(
    args: Any,
    previous_state: dict,
    current_state: dict,
    artifact_dir: str,
    match_distance_m: float = 0.02,
) -> Tuple[dict, dict]:
    """Restore VLM-confirmed YOLO misses using prior geometry after no motion."""
    previous = organize_scope_objects(previous_state)
    current = organize_scope_objects(current_state)
    matches, missing, unmatched_current = _static_matches(previous, current, match_distance_m)
    report: Dict[str, Any] = {
        "triggered": bool(missing),
        "matched_count": len(matches),
        "previous_count": len(previous),
        "current_count": len(current),
        "missing_track_ids": [_review_key(item) for item in missing],
        "unmatched_current_ids": [item.get("id") for item in unmatched_current],
        "recovered": False,
    }
    if not missing:
        report["reason"] = "no_static_detection_loss"
        return current_state, report
    if unmatched_current:
        report["reason"] = "scene_not_static_or_new_detection_present"
        return current_state, report

    images = []
    for path in (
        previous_state.get("snapshot_image"),
        current_state.get("snapshot_image"),
        current_state.get("annotated_image"),
    ):
        if path and os.path.isfile(path):
            images.append(image_to_base64(path))
    if len(images) < 2:
        report["reason"] = "comparison_images_unavailable"
        return current_state, report

    expected = [_compact_expected(item) for item in previous]
    missing_expected = [_compact_expected(item) for item in missing]
    detected = [_compact_expected(item) for item in current]
    prompt = (
        "你只做静止桌面积木感知复核，不规划动作，也不生成坐标。图像顺序为：上一帧原图、当前原图、"
        "可选的当前YOLO标注图。机械臂和积木在两帧之间没有运动。判断当前YOLO漏掉的 track 是否仍在当前原图中。"
        "颜色以可见像素为准；不要把YOLO标注文字当作实物。只有清楚看见才 still_visible=true。"
        "track_id 必须逐字复制 missing_expected；visual_color 填实际可见颜色。只输出严格JSON。\n"
        + json.dumps({
            "previous_expected": expected,
            "current_yolo_detections": detected,
            "missing_expected": missing_expected,
            "output_schema": STATIC_REVIEW_SCHEMA,
        }, ensure_ascii=False, indent=2)
    )
    result = call_policy(
        args,
        "action_proposal",
        [
            {"role": "system", "content": "你是保守的机器人视觉漏检复核器。"},
            {"role": "user", "content": prompt, "images": images},
        ],
        STATIC_REVIEW_SCHEMA,
        artifact_dir=artifact_dir,
        temperature=0.0,
        top_p=0.8,
    )
    report["vlm_status"] = result.generation_status
    report["vlm_error"] = result.error_message
    decision = result.parsed_decision
    report["vlm_decision"] = decision
    if not isinstance(decision, dict):
        report["reason"] = "vlm_review_unavailable"
        return current_state, report
    if not decision.get("scene_unchanged") or float(decision.get("confidence", 0.0)) < 0.8:
        report["reason"] = "vlm_did_not_confirm_static_scene"
        return current_state, report

    reviewed = {str(item.get("track_id")): item for item in decision.get("missing_objects", [])}
    for item in missing:
        track_id = _review_key(item)
        verdict = reviewed.get(track_id)
        if (
            verdict is None
            or not verdict.get("still_visible")
            or float(verdict.get("confidence", 0.0)) < 0.8
            or str(verdict.get("visual_color") or "").lower() != color_value_from_object(item)
        ):
            report["reason"] = "vlm_did_not_confirm_all_missing_objects"
            return current_state, report

    output = copy.deepcopy(current_state)
    used_ids = {str(item.get("id")) for item in output.get("objects", [])}
    next_numeric = max(
        [int(item.get("id")) for item in output.get("objects", []) if str(item.get("id")).isdigit()] + [-1]
    ) + 1
    recovered_ids = []
    for item in missing:
        restored = copy.deepcopy(item)
        if str(restored.get("id")) in used_ids:
            while str(next_numeric) in used_ids:
                next_numeric += 1
            restored["id"] = next_numeric
            next_numeric += 1
        used_ids.add(str(restored.get("id")))
        restored.pop("object_ref", None)
        restored["vlm_perception_recovered"] = True
        restored["geometry_source"] = "previous_static_observation_vlm_confirmed"
        output.setdefault("objects", []).append(restored)
        recovered_ids.append(restored.get("id"))
    report.update({
        "recovered": True,
        "reason": "static_yolo_miss_vlm_confirmed",
        "recovered_object_ids": recovered_ids,
    })
    output["perception_recovery"] = copy.deepcopy(report)
    return output, report


def review_and_recover_expected_misses(
    args: Any,
    expected_state: dict,
    current_state: dict,
    artifact_dir: str,
    change_description: str,
    match_distance_m: float = 0.03,
) -> Tuple[dict, dict]:
    """Recover misses after one known action, never by inventing geometry."""
    expected = organize_scope_objects(expected_state)
    current = organize_scope_objects(current_state)
    matches, missing, unmatched_current = _static_matches(expected, current, match_distance_m)
    report: Dict[str, Any] = {
        "triggered": bool(missing),
        "matched_count": len(matches),
        "expected_count": len(expected),
        "current_count": len(current),
        "missing_track_ids": [_review_key(item) for item in missing],
        "unmatched_current_ids": [item.get("id") for item in unmatched_current],
        "recovered": False,
        "known_change": str(change_description),
    }
    if not missing:
        report["reason"] = "no_expected_detection_loss"
        return current_state, report
    if unmatched_current:
        report["reason"] = "unexpected_current_object_present"
        return current_state, report

    images = []
    for path in (expected_state.get("snapshot_image"), current_state.get("snapshot_image"), current_state.get("annotated_image")):
        if path and os.path.isfile(path):
            images.append(image_to_base64(path))
    if not images:
        report["reason"] = "comparison_images_unavailable"
        return current_state, report
    prompt = (
        "你只做已知机器人动作后的积木漏检复核，不规划动作、不生成坐标。输入给出的 expected_after_action"
        "几何来自执行前点云和已执行目标位姿，不需要你修改。当前YOLO可能漏检，但也可能真的抓取失败。"
        "请查看当前原图（最后一张可能是YOLO标注图），逐个判断 missing_expected 是否仍清楚可见。"
        "track_id 必须逐字复制；颜色以实物像素为准，不看标注文字。只有当前图像与 known_change 一致且"
        "所有未检目标都看得清楚时 expected_scene_consistent=true。只输出严格JSON。\n"
        + json.dumps({
            "known_change": change_description,
            "expected_after_action": [_compact_expected(item) for item in expected],
            "current_yolo_detections": [_compact_expected(item) for item in current],
            "missing_expected": [_compact_expected(item) for item in missing],
            "output_schema": EXPECTED_REVIEW_SCHEMA,
        }, ensure_ascii=False, indent=2)
    )
    result = call_policy(
        args, "action_proposal",
        [
            {"role": "system", "content": "你是保守的机器人执行后视觉漏检复核器。"},
            {"role": "user", "content": prompt, "images": images},
        ],
        EXPECTED_REVIEW_SCHEMA,
        artifact_dir=artifact_dir,
        temperature=0.0,
        top_p=0.8,
    )
    decision = result.parsed_decision
    report.update({
        "vlm_status": result.generation_status,
        "vlm_error": result.error_message,
        "vlm_decision": decision,
    })
    if not isinstance(decision, dict):
        report["reason"] = "vlm_review_unavailable"
        return current_state, report
    if not decision.get("expected_scene_consistent") or float(decision.get("confidence", 0.0)) < 0.8:
        report["reason"] = "vlm_did_not_confirm_expected_scene"
        return current_state, report
    reviewed = {str(item.get("track_id")): item for item in decision.get("missing_objects", [])}
    for item in missing:
        key = _review_key(item)
        verdict = reviewed.get(key)
        if (
            verdict is None or not verdict.get("still_visible")
            or float(verdict.get("confidence", 0.0)) < 0.8
            or str(verdict.get("visual_color") or "").lower() != color_value_from_object(item)
        ):
            report["reason"] = "vlm_did_not_confirm_all_missing_objects"
            return current_state, report
    output, recovered_ids = _restore_missing_objects(current_state, missing)
    report.update({
        "recovered": True,
        "reason": "expected_post_action_scene_vlm_confirmed",
        "recovered_object_ids": recovered_ids,
    })
    output["perception_recovery"] = copy.deepcopy(report)
    return output, report


def _static_matches(previous: List[dict], current: List[dict], limit: float):
    candidates = []
    for pi, prior in enumerate(previous):
        prior_center = prior.get("geometry_center_m") or prior.get("center_3d_base_m")
        prior_color = color_value_from_object(prior)
        for ci, now in enumerate(current):
            now_center = now.get("geometry_center_m") or now.get("center_3d_base_m")
            if not prior_center or not now_center or color_value_from_object(now) != prior_color:
                continue
            distance = math.dist([float(v) for v in prior_center[:2]], [float(v) for v in now_center[:2]])
            if distance <= float(limit):
                candidates.append((distance, pi, ci))
    matches = []
    used_previous, used_current = set(), set()
    for distance, pi, ci in sorted(candidates):
        if pi in used_previous or ci in used_current:
            continue
        used_previous.add(pi); used_current.add(ci); matches.append((pi, ci, distance))
    missing = [item for index, item in enumerate(previous) if index not in used_previous]
    unmatched_current = [item for index, item in enumerate(current) if index not in used_current]
    return matches, missing, unmatched_current


def _compact_expected(obj: dict) -> dict:
    return {
        "track_id": _review_key(obj),
        "detector_object_id": obj.get("id"),
        "label": obj.get("label"),
        "visual_color": color_value_from_object(obj),
        "bbox_xyxy_px": obj.get("bbox_xyxy_px") or obj.get("bbox"),
    }


def _compact_candidate(obj: dict) -> dict:
    return {
        "candidate_id": _review_key(obj),
        "detector_object_id": obj.get("id"),
        "label": obj.get("label"),
        "visual_color": color_value_from_object(obj),
        "bbox_xyxy_px": obj.get("bbox_xyxy_px") or obj.get("bbox"),
    }


def _extra_low_confidence_clusters(primary: List[dict], low: List[dict], limit: float) -> List[dict]:
    """Return one representative for each low-threshold 3D location absent from primary."""
    extras: List[dict] = []
    for candidate in low:
        center = candidate.get("geometry_center_m") or candidate.get("center_3d_base_m")
        if not center:
            continue
        if any(_planar_distance(candidate, existing) <= float(limit) for existing in primary):
            continue
        cluster_index = next(
            (index for index, existing in enumerate(extras)
             if _planar_distance(candidate, existing) <= float(limit)),
            None,
        )
        if cluster_index is None:
            extras.append(copy.deepcopy(candidate))
            continue
        if _low_candidate_rank(candidate) > _low_candidate_rank(extras[cluster_index]):
            extras[cluster_index] = copy.deepcopy(candidate)
    return extras


def _planar_distance(a: dict, b: dict) -> float:
    center_a = a.get("geometry_center_m") or a.get("center_3d_base_m")
    center_b = b.get("geometry_center_m") or b.get("center_3d_base_m")
    if not center_a or not center_b:
        return float("inf")
    return math.dist([float(v) for v in center_a[:2]], [float(v) for v in center_b[:2]])


def _low_candidate_rank(obj: dict) -> Tuple[int, float]:
    color = color_value_from_object(obj)
    label_matches_color = int(bool(color and color in str(obj.get("label") or "").lower()))
    return label_matches_color, float(obj.get("confidence") or obj.get("score") or 0.0)


def _restore_low_confidence_objects(current_state: dict, confirmed: List[dict]) -> Tuple[dict, List[Any]]:
    output = copy.deepcopy(current_state)
    used_ids = {str(item.get("id")) for item in output.get("objects", [])}
    next_numeric = max(
        [int(item.get("id")) for item in output.get("objects", []) if str(item.get("id")).isdigit()] + [-1]
    ) + 1
    recovered_ids = []
    for item in confirmed:
        restored = copy.deepcopy(item)
        while str(restored.get("id")) in used_ids:
            restored["id"] = next_numeric
            next_numeric += 1
        used_ids.add(str(restored.get("id")))
        restored.pop("object_ref", None)
        restored["vlm_perception_recovered"] = True
        restored["geometry_source"] = "low_confidence_rgbd_candidate_vlm_confirmed"
        output.setdefault("objects", []).append(restored)
        recovered_ids.append(restored.get("id"))
    return output, recovered_ids


def _review_key(obj: dict) -> str:
    return str(obj.get("track_id") or "expected_obj_{}".format(obj.get("id")))


def _restore_missing_objects(current_state: dict, missing: List[dict]) -> Tuple[dict, List[Any]]:
    output = copy.deepcopy(current_state)
    used_ids = {str(item.get("id")) for item in output.get("objects", [])}
    next_numeric = max(
        [int(item.get("id")) for item in output.get("objects", []) if str(item.get("id")).isdigit()] + [-1]
    ) + 1
    recovered_ids = []
    for item in missing:
        restored = copy.deepcopy(item)
        if str(restored.get("id")) in used_ids:
            while str(next_numeric) in used_ids:
                next_numeric += 1
            restored["id"] = next_numeric
            next_numeric += 1
        used_ids.add(str(restored.get("id")))
        restored.pop("object_ref", None)
        restored.pop("bbox_xyxy_px", None)
        restored.pop("bbox", None)
        restored["vlm_perception_recovered"] = True
        restored["geometry_source"] = "expected_post_action_geometry_vlm_confirmed"
        output.setdefault("objects", []).append(restored)
        recovered_ids.append(restored.get("id"))
    return output, recovered_ids
