"""Estimate the current tabletop block stack from one private scene state."""

import math


def _finite_point(value, length=3):
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    try:
        point = [float(value[index]) for index in range(length)]
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def _geometry_object(obj):
    center = _finite_point(obj.get("geometry_center_m"))
    dimensions = _finite_point(obj.get("dimensions_m"))
    top_z = obj.get("top_z_base_m")
    try:
        top_z = None if top_z is None else float(top_z)
    except (TypeError, ValueError):
        top_z = None
    if (
        center is None
        or dimensions is None
        or obj.get("geometry_frame") != "base_link"
        or not obj.get("pointcloud_geometry_valid", True)
        or dimensions[2] <= 0.0
        or (top_z is not None and not math.isfinite(top_z))
    ):
        return None
    if top_z is None:
        top_z = center[2] + dimensions[2] / 2.0
    return {
        "id": int(obj["id"]),
        "label": obj.get("label"),
        "geometry_center_m": center,
        "dimensions_m": dimensions,
        "bottom_z_base_m": center[2] - dimensions[2] / 2.0,
        "top_z_base_m": top_z,
        "top_z_source": obj.get("top_z_source") or "geometry_center_plus_height",
        "top_surface_center_m": _finite_point(obj.get("top_surface_center_m")),
        "height_estimation_method": obj.get("height_estimation_method"),
        "table_yaw_deg": obj.get("table_yaw_deg"),
        "table_yaw_source": obj.get("table_yaw_source"),
    }


def _distance_xy(point, xy):
    return math.hypot(point[0] - xy[0], point[1] - xy[1])


def estimate_stack_state(
    scene_state,
    base_object_id=None,
    previous_stack_xy=None,
    search_radius_m=0.06,
    excluded_xy=None,
    exclusion_radius_m=0.0,
    excluded_object_ids=None,
    expected_top_z_base_m=None,
    top_z_tolerance_m=0.0,
):
    """Return the stack cluster and dynamic top Z from one current observation."""
    objects = []
    for obj in scene_state.get("objects", []):
        if obj.get("is_workspace"):
            continue
        parsed = _geometry_object(obj)
        if parsed is not None:
            objects.append(parsed)

    previous_xy = _finite_point(previous_stack_xy, length=2)
    excluded_xy = _finite_point(excluded_xy, length=2)
    exclusion_radius = float(exclusion_radius_m)
    excluded_ids = set()
    for value in excluded_object_ids or []:
        try:
            excluded_ids.add(int(value))
        except (TypeError, ValueError):
            pass
    stack_candidates = [obj for obj in objects if obj["id"] not in excluded_ids]
    expected_top_z = None if expected_top_z_base_m is None else float(expected_top_z_base_m)
    top_z_tolerance = float(top_z_tolerance_m)
    base = None
    base_source = None
    if base_object_id is not None:
        base = next((obj for obj in stack_candidates if obj["id"] == int(base_object_id)), None)
        if base is not None:
            base_source = "current_requested_base_object"
    if base is None and previous_xy is not None and stack_candidates:
        base = min(stack_candidates, key=lambda obj: _distance_xy(obj["geometry_center_m"], previous_xy))
        base_source = "nearest_current_object_to_previous_stack_xy"
    if base is None:
        return {
            "schema_version": "stack_state_v1",
            "valid": False,
            "reason": "No valid base_link geometry object matched the requested stack base.",
            "requested_base_object_id": base_object_id,
            "base_object_id": base_object_id,
            "stack_xy_base_m": previous_xy,
            "top_z_base_m": None,
            "excluded_object_ids": sorted(excluded_ids),
            "stack_objects": [],
        }

    if base_source == "current_requested_base_object":
        anchor_xy = base["geometry_center_m"][:2]
        anchor_source = "current_requested_base_object.geometry_center_m"
    elif previous_xy is not None:
        anchor_xy = previous_xy
        anchor_source = "previous_stack_xy_fallback"
    else:
        anchor_xy = base["geometry_center_m"][:2]
        anchor_source = "current_nearest_object.geometry_center_m"
    radius = float(search_radius_m)
    stack_objects = [
        obj
        for obj in stack_candidates
        if _distance_xy(obj["geometry_center_m"], anchor_xy) <= radius
        and (
            excluded_xy is None
            or _distance_xy(obj["geometry_center_m"], excluded_xy) > exclusion_radius
        )
        and (
            expected_top_z is None
            or abs(obj["top_z_base_m"] - expected_top_z) <= top_z_tolerance
        )
    ]
    if not stack_objects:
        return {
            "schema_version": "stack_state_v1",
            "valid": False,
            "reason": "No valid geometry object was found within the stack search radius.",
            "requested_base_object_id": base_object_id,
            "base_object_id": base["id"],
            "stack_xy_base_m": anchor_xy,
            "top_z_base_m": None,
            "search_radius_m": radius,
            "stack_anchor_source": anchor_source,
            "base_selection_source": base_source,
            "excluded_object_ids": sorted(excluded_ids),
            "stack_objects": [],
        }
    stack_objects.sort(key=lambda obj: (obj["top_z_base_m"], obj["id"]))
    weights = [max(obj["dimensions_m"][2], 1e-6) for obj in stack_objects]
    weight_sum = sum(weights)
    stack_xy = [
        sum(obj["geometry_center_m"][axis] * weight for obj, weight in zip(stack_objects, weights)) / weight_sum
        for axis in range(2)
    ]
    top_object = max(stack_objects, key=lambda obj: obj["top_z_base_m"])
    placement_xy = top_object["geometry_center_m"][:2]
    return {
        "schema_version": "stack_state_v1",
        "valid": True,
        "reason": "Estimated from current base_link point-cloud geometry.",
        "requested_base_object_id": base_object_id,
        "base_object_id": base["id"],
        "base_selection_source": base_source,
        "stack_anchor_xy_base_m": [round(value, 6) for value in anchor_xy],
        "stack_anchor_source": anchor_source,
        "stack_xy_base_m": [round(value, 6) for value in stack_xy],
        "top_z_base_m": round(top_object["top_z_base_m"], 6),
        "top_object_id": top_object["id"],
        "placement_base_object_id": top_object["id"],
        "placement_base_center_xy_m": [round(value, 6) for value in placement_xy],
        "placement_base_top_z_m": round(top_object["top_z_base_m"], 6),
        "placement_base_top_z_source": top_object.get("top_z_source"),
        "placement_base_top_surface_center_m": (
            None
            if top_object.get("top_surface_center_m") is None
            else [round(value, 6) for value in top_object["top_surface_center_m"]]
        ),
        "placement_base_height_estimation_method": top_object.get("height_estimation_method"),
        "placement_coordinate_source": "close_observation.top_object.geometry_center_and_top_z",
        "stack_yaw_deg": top_object.get("table_yaw_deg"),
        "stack_yaw_source": top_object.get("table_yaw_source"),
        "search_radius_m": radius,
        "excluded_xy_base_m": excluded_xy,
        "exclusion_radius_m": exclusion_radius,
        "excluded_object_ids": sorted(excluded_ids),
        "expected_top_z_base_m": expected_top_z,
        "top_z_tolerance_m": top_z_tolerance,
        "stack_height_object_count": len(stack_objects),
        "stack_objects": stack_objects,
    }


def verify_stack_growth(previous_stack_state, current_stack_state, min_top_increase_m=0.005):
    """Verify that a completed placement increased the observed stack top."""
    previous_top = previous_stack_state.get("top_z_base_m")
    current_top = current_stack_state.get("top_z_base_m")
    previous_count = int(previous_stack_state.get("stack_height_object_count") or 0)
    current_count = int(current_stack_state.get("stack_height_object_count") or 0)
    valid_observations = bool(previous_stack_state.get("valid") and current_stack_state.get("valid"))
    increase = None
    height_valid = False
    count_valid = False
    if valid_observations and previous_top is not None and current_top is not None:
        increase = float(current_top) - float(previous_top)
        height_valid = increase >= float(min_top_increase_m)
        count_valid = current_count > previous_count and increase >= -float(min_top_increase_m)
    valid = valid_observations and (height_valid or count_valid)
    if height_valid:
        reason = "Stack top increased after placement."
    elif count_valid:
        reason = "Stack object count increased; accepting noisy top-Z measurement."
    elif not valid_observations:
        reason = "Previous or current stack observation is invalid."
    else:
        reason = "Stack top did not increase enough after placement."
    return {
        "schema_version": "stack_verification_v1",
        "valid": valid,
        "previous_top_z_base_m": previous_top,
        "current_top_z_base_m": current_top,
        "top_increase_m": increase,
        "min_top_increase_m": float(min_top_increase_m),
        "previous_stack_height_object_count": previous_count,
        "current_stack_height_object_count": current_count,
        "height_growth_valid": height_valid,
        "object_count_growth_valid": count_valid,
        "reason": reason,
    }
