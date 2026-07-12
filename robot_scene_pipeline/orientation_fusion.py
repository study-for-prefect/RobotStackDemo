"""Fuse geometry and VLM semantics for roof and triangle orientation decisions."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geometry_relations import get_center
from .task_semantic_validation import infer_object_shape


def build_house_frame(role_objects: Dict[str, dict], config: dict) -> dict:
    """Build house axes in base_link without treating image-up as a robot direction."""
    left = _column_center(role_objects, "left")
    right = _column_center(role_objects, "right")
    configured_up = config.get("house_semantics", {}).get("house_up_direction_base_xy", [0.0, 1.0])
    if left is None or right is None:
        house_x = _normalize_xy([-float(configured_up[1]), float(configured_up[0])])
        origin = None
    else:
        house_x = _normalize_xy([right[0] - left[0], right[1] - left[1]])
        origin = [(left[index] + right[index]) * 0.5 for index in range(3)]
    candidate_y = [-house_x[1], house_x[0]]
    if _dot_xy(candidate_y, configured_up) < 0.0:
        candidate_y = [-candidate_y[0], -candidate_y[1]]
    return {
        "frame_id": "house_frame",
        "origin_base_m": origin,
        "house_x_base_xy": house_x,
        "house_y_base_xy": candidate_y,
        "house_z_base": [0.0, 0.0, 1.0],
        "house_up_direction_base_xy": [float(value) for value in configured_up],
        "source": "column_centers_and_configured_house_up",
    }


def fuse_house_orientation_observations(
    plan: dict,
    state: dict,
    config: dict,
    previous_results: Optional[Sequence[dict]] = None,
) -> dict:
    """Attach deterministic fused results; uncertain/conflicting results require reobservation."""
    output = copy.deepcopy(plan)
    objects = {str(obj.get("id")): obj for obj in state.get("objects", []) if isinstance(obj, dict)}
    bindings = {item.get("role_id"): item for item in plan.get("role_assignments", [])}
    role_objects = {
        role_id: objects.get(str(binding.get("selected_object_id")))
        for role_id, binding in bindings.items()
    }
    house_frame = build_house_frame(role_objects, config)
    previous = {item.get("role_id"): item for item in previous_results or []}
    results = []
    for observation in plan.get("orientation_observations", []):
        role_id = observation.get("role_id")
        obj = role_objects.get(role_id)
        if role_id == "roof":
            result = fuse_roof_orientation(obj or {}, observation, config, house_frame)
        else:
            result = fuse_triangle_orientation(obj or {}, observation, config, house_frame)
        prior = previous.get(role_id)
        if prior and prior.get("orientation_state") != result.get("orientation_state"):
            result["reobserve_required"] = True
            result.setdefault("reobserve_reasons", []).append("consecutive_frames_inconsistent")
        result["candidate_target_orientations"] = target_orientation_candidates(role_id, house_frame)
        results.append(result)
    output["house_frame"] = house_frame
    output["fused_orientation_results"] = results
    return output


def fuse_roof_orientation(obj: dict, vlm: dict, config: dict, house_frame: Optional[dict] = None) -> dict:
    """Fuse concavity depth evidence with face/opening semantics."""
    threshold = float(config.get("house_semantics", {}).get("orientation_confidence_threshold", 0.75))
    depth_concave, geometry_confidence = _roof_depth_concavity(obj)
    vlm_shape = str(vlm.get("shape") or infer_object_shape(obj))
    detected_shape = infer_object_shape(obj)
    fused_shape = (
        "concave_rectangle"
        if depth_concave and vlm_shape == "concave_rectangle"
        else detected_shape
    )
    visible_face = str(vlm.get("visible_face") or "unknown")
    geometry_face = str(obj.get("geometry_visible_face") or "unknown")
    opening = str(vlm.get("opening_direction_image") or "unknown")
    opening_down = _image_direction_matches_house(opening, obj, house_frame, expected_sign=-1)
    straight_up = _image_direction_matches_house(
        str(vlm.get("straight_edge_direction_image") or "unknown"), obj, house_frame, expected_sign=1,
    )
    ambiguous = opening == "unknown" and fused_shape == "concave_rectangle"
    flip_required = bool(vlm.get("flip_required")) or visible_face in {"back", "side"}
    correct_face = (
        (fused_shape == "rectangle" and visible_face not in {"back", "side"} and not flip_required)
        or (fused_shape == "concave_rectangle" and visible_face == "front" and not flip_required)
    )
    orientation_state = "correct_face" if correct_face else "wrong_face"
    confidence = _fused_confidence(vlm.get("confidence"), geometry_confidence)
    conflicts = []
    if fused_shape == "concave_rectangle" and visible_face == "front" and bool(vlm.get("flip_required")):
        conflicts.append("vlm_face_and_flip_conflict")
    if detected_shape == "rectangle" and vlm_shape == "concave_rectangle" and not depth_concave:
        conflicts.append("vlm_concavity_not_supported_by_depth")
    if geometry_face != "unknown" and visible_face != "unknown" and geometry_face != visible_face:
        conflicts.append("geometry_and_vlm_face_conflict")
    reasons = []
    if confidence < threshold:
        reasons.append("orientation_confidence_below_threshold")
    if ambiguous:
        reasons.append("roof_opening_direction_unknown")
    if fused_shape == "concave_rectangle" and (opening_down is None or straight_up is None):
        reasons.append("image_direction_to_house_frame_unavailable")
    if obj.get("orientation_occluded"):
        reasons.append("roof_orientation_occluded")
    reasons.extend(conflicts)
    return {
        "role_id": "roof",
        "selected_object_id": vlm.get("selected_object_id"),
        "detected_shape": detected_shape,
        "fused_shape": fused_shape,
        "orientation_state": orientation_state,
        "orientation_confidence": confidence,
        "flip_required": flip_required,
        "yaw_required": bool(vlm.get("yaw_adjustment_required")),
        "correct_face_up": correct_face,
        "opening_down": bool(opening_down) or fused_shape == "rectangle",
        "straight_edge_up": bool(straight_up) or fused_shape == "rectangle",
        "geometry_orientation_result": {
            "depth_concave": depth_concave, "visible_face": geometry_face,
            "local_plane_normal_base": obj.get("local_plane_normal_base"),
            "pca_axes_base": obj.get("pca_axes_base") or obj.get("pca_main_axes"),
            "confidence": geometry_confidence,
        },
        "vlm_orientation_result": dict(vlm),
        "reobserve_required": bool(reasons),
        "reobserve_reasons": reasons,
    }


def fuse_triangle_orientation(obj: dict, vlm: dict, config: dict, house_frame: Optional[dict] = None) -> dict:
    """Reject right-angle-as-apex and fuse contour/VLM vertex semantics."""
    threshold = float(config.get("house_semantics", {}).get("orientation_confidence_threshold", 0.75))
    contour = analyze_triangle_contour(obj)
    pose_state = str(vlm.get("pose_state") or "unknown")
    geometry_pose_state = str(obj.get("geometry_pose_state") or "unknown")
    apex = vlm.get("apex_vertex")
    upward = vlm.get("upward_vertex")
    right_angle = contour.get("right_angle_vertex")
    conflicts = []
    if apex is not None and right_angle is not None and int(apex) == int(right_angle):
        conflicts.append("right_angle_vertex_cannot_be_target_apex")
    if vlm.get("right_angle_vertex") is not None and right_angle is not None and int(vlm["right_angle_vertex"]) != int(right_angle):
        conflicts.append("vlm_and_contour_right_angle_conflict")
    if geometry_pose_state != "unknown" and pose_state != "unknown" and geometry_pose_state != pose_state:
        conflicts.append("geometry_and_vlm_pose_conflict")
    apex_direction_up = _image_direction_matches_house(
        str(vlm.get("apex_direction_image") or "unknown"), obj, house_frame, expected_sign=1,
    )
    apex_up = apex is not None and upward is not None and int(apex) == int(upward) and apex_direction_up is True and not conflicts
    confidence = _fused_confidence(vlm.get("confidence"), contour.get("confidence", 0.0))
    reasons = []
    if confidence < threshold:
        reasons.append("orientation_confidence_below_threshold")
    if bool(vlm.get("apex_ambiguous")) or apex is None:
        reasons.append("triangle_apex_ambiguous")
    if apex_direction_up is None:
        reasons.append("image_direction_to_house_frame_unavailable")
    if obj.get("orientation_occluded"):
        reasons.append("triangle_orientation_occluded")
    reasons.extend(conflicts)
    correct_face = pose_state == "upright"
    not_side_lying = pose_state not in {"side_lying", "flat", "unknown"}
    return {
        "role_id": "triangle_top",
        "selected_object_id": vlm.get("selected_object_id"),
        "fused_shape": "triangle",
        "orientation_state": "upright_apex_up" if correct_face and apex_up else pose_state,
        "orientation_confidence": confidence,
        "flip_required": bool(vlm.get("flip_required")) or pose_state in {"upside_down", "side_lying", "flat"},
        "yaw_required": bool(vlm.get("yaw_adjustment_required")),
        "correct_face": correct_face,
        "apex_up": apex_up,
        "not_side_lying": not_side_lying,
        "right_angle_vertex": right_angle,
        "apex_vertex": apex,
        "upward_vertex": upward,
        "geometry_orientation_result": {
            **contour, "pose_state": geometry_pose_state,
            "local_plane_normal_base": obj.get("local_plane_normal_base"),
            "pca_axes_base": obj.get("pca_axes_base") or obj.get("pca_main_axes"),
        },
        "vlm_orientation_result": dict(vlm),
        "reobserve_required": bool(reasons),
        "reobserve_reasons": reasons,
    }


def analyze_triangle_contour(obj: dict) -> dict:
    """Identify a right-angle vertex from three contour angles, never from image height alone."""
    angles = obj.get("contour_inner_angles_deg") or []
    if (not isinstance(angles, (list, tuple)) or len(angles) != 3) and obj.get("contour_vertices_px"):
        angles = triangle_inner_angles(obj.get("contour_vertices_px"))
    if not isinstance(angles, (list, tuple)) or len(angles) != 3:
        return {"valid": False, "right_angle_vertex": None, "acute_vertices": [], "confidence": 0.0}
    values = [float(value) for value in angles]
    right_index = min(range(3), key=lambda index: abs(values[index] - 90.0))
    right_index = right_index if abs(values[right_index] - 90.0) <= 15.0 else None
    acute = [index for index, value in enumerate(values) if value < 75.0]
    return {
        "valid": abs(sum(values) - 180.0) <= 12.0,
        "inner_angles_deg": values,
        "right_angle_vertex": right_index,
        "acute_vertices": acute,
        "confidence": 0.9 if right_index is not None and len(acute) == 2 else 0.6,
    }


def triangle_inner_angles(vertices: Sequence[Sequence[float]]) -> List[float]:
    """Compute all three contour angles; image height is intentionally unused."""
    if not isinstance(vertices, (list, tuple)) or len(vertices) != 3:
        return []
    output = []
    for index, vertex in enumerate(vertices):
        first = vertices[(index - 1) % 3]
        second = vertices[(index + 1) % 3]
        a = [float(first[0]) - float(vertex[0]), float(first[1]) - float(vertex[1])]
        b = [float(second[0]) - float(vertex[0]), float(second[1]) - float(vertex[1])]
        denominator = math.hypot(*a) * math.hypot(*b)
        if denominator <= 1e-9:
            return []
        cosine = max(-1.0, min(1.0, _dot_xy(a, b) / denominator))
        output.append(math.degrees(math.acos(cosine)))
    return output


def target_orientation_candidates(role_id: str, house_frame: dict) -> List[List[float]]:
    """Return code-defined object-frame target quaternions aligned with house axes."""
    house_x = house_frame["house_x_base_xy"]
    yaw = math.atan2(house_x[1], house_x[0])
    if role_id == "roof":
        return [_rpy_to_quaternion(0.0, 0.0, yaw), _rpy_to_quaternion(0.0, 0.0, yaw + math.pi)]
    house_y = house_frame["house_y_base_xy"]
    triangle_yaw = math.atan2(house_y[1], house_y[0])
    return [_rpy_to_quaternion(0.0, 0.0, triangle_yaw)]


def _roof_depth_concavity(obj: dict) -> Tuple[bool, float]:
    center = obj.get("center_region_height_m")
    sides = obj.get("side_region_heights_m")
    if center is None or not isinstance(sides, (list, tuple)) or len(sides) < 2:
        return bool(obj.get("depth_concavity_detected")), float(obj.get("depth_orientation_confidence", 0.0))
    difference = min(float(sides[0]), float(sides[1])) - float(center)
    return difference >= 0.004, min(1.0, max(0.0, difference / 0.01))


def _column_center(objects: Dict[str, dict], side: str) -> Optional[List[float]]:
    centers = [
        get_center(objects.get("{}_support_{}".format(side, level)) or {})
        for level in ("lower", "upper")
    ]
    centers = [center for center in centers if center is not None]
    if not centers:
        return None
    return [sum(center[index] for center in centers) / len(centers) for index in range(3)]


def _normalize_xy(vector: Sequence[float]) -> List[float]:
    norm = math.hypot(float(vector[0]), float(vector[1]))
    if norm <= 1e-9:
        raise ValueError("house frame direction cannot be zero")
    return [float(vector[0]) / norm, float(vector[1]) / norm]


def _dot_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return float(first[0]) * float(second[0]) + float(first[1]) * float(second[1])


def _fused_confidence(vlm_confidence: Any, geometry_confidence: float) -> float:
    try:
        vlm_value = float(vlm_confidence)
    except (TypeError, ValueError):
        vlm_value = 0.0
    if geometry_confidence <= 0.0:
        return round(vlm_value, 4)
    return round(0.6 * vlm_value + 0.4 * float(geometry_confidence), 4)


def _image_direction_matches_house(
    direction: str, obj: dict, house_frame: Optional[dict], expected_sign: int,
) -> Optional[bool]:
    if direction == "unknown" or not house_frame:
        return None
    image_up = obj.get("image_up_direction_base_xy")
    image_right = obj.get("image_right_direction_base_xy")
    if not isinstance(image_up, (list, tuple)) or not isinstance(image_right, (list, tuple)):
        return None
    vectors = {
        "up": image_up,
        "down": [-float(image_up[0]), -float(image_up[1])],
        "right": image_right,
        "left": [-float(image_right[0]), -float(image_right[1])],
    }
    vector = vectors.get(direction)
    if vector is None:
        return None
    projection = _dot_xy(vector, house_frame["house_y_base_xy"])
    return projection * float(expected_sign) >= 0.7


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> List[float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]
