"""Pure-Python geometric relations for tabletop scene objects.

The helpers in this module only inspect object dictionaries. They do not call
robot execution, MoveIt, perception models, or language models.
"""

import math


def _finite_vector(value, minimum_length):
    if not isinstance(value, (list, tuple)) or len(value) < minimum_length:
        return None
    try:
        vector = [float(value[index]) for index in range(minimum_length)]
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _pose_position(value):
    direct = _finite_vector(value, 3)
    if direct is not None:
        return direct
    if not isinstance(value, dict):
        return None
    for key in ("position", "translation", "center"):
        direct = _finite_vector(value.get(key), 3)
        if direct is not None:
            return direct
    try:
        direct = [float(value["x"]), float(value["y"]), float(value["z"])]
    except (KeyError, TypeError, ValueError):
        return None
    return direct if all(math.isfinite(item) for item in direct) else None


def _same_object(a, b):
    if a is b:
        return True
    a_id = a.get("id") if isinstance(a, dict) else None
    b_id = b.get("id") if isinstance(b, dict) else None
    return a_id is not None and b_id is not None and a_id == b_id


def _object_name(obj):
    object_id = obj.get("id")
    if object_id is not None:
        return object_id
    label = obj.get("label")
    return label if label is not None else "unknown"


def _expanded_aabb(aabb, margin_m):
    if not aabb:
        return {}
    margin = max(0.0, float(margin_m))
    expanded = dict(aabb)
    expanded["xmin"] -= margin
    expanded["xmax"] += margin
    expanded["ymin"] -= margin
    expanded["ymax"] += margin
    return expanded


def _translated_aabb(aabb, dx, dy):
    if not aabb:
        return {}
    translated = dict(aabb)
    translated["xmin"] += dx
    translated["xmax"] += dx
    translated["ymin"] += dy
    translated["ymax"] += dy
    return translated


def _z_gap(aabb_a, aabb_b):
    if not aabb_a or not aabb_b:
        return math.inf
    if aabb_a["zmax"] < aabb_b["zmin"]:
        return aabb_b["zmin"] - aabb_a["zmax"]
    if aabb_b["zmax"] < aabb_a["zmin"]:
        return aabb_a["zmin"] - aabb_b["zmax"]
    return 0.0


def get_center(obj):
    """Return an object's base-frame XYZ center, or None when unavailable."""
    if not isinstance(obj, dict):
        return None
    for key in ("geometry_center_m", "last_pose_base", "center_base_m"):
        center = _pose_position(obj.get(key))
        if center is not None:
            return center
    return None


def get_size(obj):
    """Return positive XYZ dimensions, or None when unavailable."""
    if not isinstance(obj, dict):
        return None
    for key in ("dimensions_m", "size_m"):
        size = _finite_vector(obj.get(key), 3)
        if size is not None:
            size = [abs(value) for value in size]
            return size
    return None


def get_yaw_rad(obj):
    """Return object yaw in radians, defaulting to zero."""
    if not isinstance(obj, dict):
        return 0.0
    yaw_rad = obj.get("yaw_rad")
    if yaw_rad is not None:
        try:
            yaw = float(yaw_rad)
            if math.isfinite(yaw):
                return yaw
        except (TypeError, ValueError):
            pass
    yaw_deg = obj.get("table_yaw_deg")
    if yaw_deg is not None:
        try:
            yaw = math.radians(float(yaw_deg))
            if math.isfinite(yaw):
                return yaw
        except (TypeError, ValueError):
            pass
    return 0.0


def object_xy_aabb(obj, margin_m=0.0):
    """Return a yaw-aware axis-aligned XYZ bounding box in base_link."""
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return {}

    yaw = get_yaw_rad(obj)
    cos_yaw = abs(math.cos(yaw))
    sin_yaw = abs(math.sin(yaw))
    half_x = 0.5 * (cos_yaw * size[0] + sin_yaw * size[1])
    half_y = 0.5 * (sin_yaw * size[0] + cos_yaw * size[1])
    margin = max(0.0, float(margin_m))
    half_z = 0.5 * size[2]
    return {
        "xmin": center[0] - half_x - margin,
        "xmax": center[0] + half_x + margin,
        "ymin": center[1] - half_y - margin,
        "ymax": center[1] + half_y + margin,
        "zmin": center[2] - half_z,
        "zmax": center[2] + half_z,
    }


def xy_distance(a, b):
    """Return XY center distance between two objects, or infinity."""
    center_a = get_center(a)
    center_b = get_center(b)
    if center_a is None or center_b is None:
        return math.inf
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def xy_aabb_overlap(aabb_a, aabb_b):
    """Return overlap along X, overlap along Y, and overlap area."""
    if not aabb_a or not aabb_b:
        return 0.0, 0.0, 0.0
    overlap_x = max(
        0.0,
        min(aabb_a["xmax"], aabb_b["xmax"])
        - max(aabb_a["xmin"], aabb_b["xmin"]),
    )
    overlap_y = max(
        0.0,
        min(aabb_a["ymax"], aabb_b["ymax"])
        - max(aabb_a["ymin"], aabb_b["ymin"]),
    )
    return overlap_x, overlap_y, overlap_x * overlap_y


def is_near(a, b, threshold_m=0.06):
    return xy_distance(a, b) <= float(threshold_m)


def is_on(top, bottom, xy_overlap_ratio_threshold=0.25, z_gap_threshold_m=0.025):
    """Return True when top is vertically above and supported by bottom."""
    top_center = get_center(top)
    bottom_center = get_center(bottom)
    top_aabb = object_xy_aabb(top)
    bottom_aabb = object_xy_aabb(bottom)
    if (
        top_center is None
        or bottom_center is None
        or not top_aabb
        or not bottom_aabb
        or top_center[2] <= bottom_center[2]
    ):
        return False

    _, _, overlap_area = xy_aabb_overlap(top_aabb, bottom_aabb)
    top_area = (top_aabb["xmax"] - top_aabb["xmin"]) * (
        top_aabb["ymax"] - top_aabb["ymin"]
    )
    bottom_area = (bottom_aabb["xmax"] - bottom_aabb["xmin"]) * (
        bottom_aabb["ymax"] - bottom_aabb["ymin"]
    )
    reference_area = min(top_area, bottom_area)
    if reference_area <= 0.0:
        return False

    overlap_ratio = overlap_area / reference_area
    surface_gap = abs(top_aabb["zmin"] - bottom_aabb["zmax"])
    return (
        overlap_ratio >= float(xy_overlap_ratio_threshold)
        and surface_gap <= float(z_gap_threshold_m)
    )


def is_supporting(bottom, top):
    return is_on(top, bottom)


def is_locked(obj):
    if not isinstance(obj, dict):
        return False
    return (
        obj.get("role") in ("base", "structure")
        or obj.get("state") in ("locked", "placed")
    )


def grasp_corridor_aabb(
    target,
    grasp_yaw_rad=None,
    approach_length_m=0.12,
    corridor_width_m=0.09,
    margin_m=0.01,
):
    """Build a simplified axis-aligned grasp approach corridor."""
    target_aabb = object_xy_aabb(target)
    if not target_aabb:
        return {}

    yaw = get_yaw_rad(target) if grasp_yaw_rad is None else float(grasp_yaw_rad)
    if not math.isfinite(yaw):
        yaw = 0.0
    approach = max(0.0, float(approach_length_m))
    corridor_width = max(0.0, float(corridor_width_m))
    margin = max(0.0, float(margin_m))
    center = get_center(target)
    target_half_x = 0.5 * (target_aabb["xmax"] - target_aabb["xmin"])
    target_half_y = 0.5 * (target_aabb["ymax"] - target_aabb["ymin"])

    if abs(math.cos(yaw)) >= abs(math.sin(yaw)):
        half_x = target_half_x + approach
        half_y = max(target_half_y, 0.5 * corridor_width)
        axis = "x"
    else:
        half_x = max(target_half_x, 0.5 * corridor_width)
        half_y = target_half_y + approach
        axis = "y"

    return {
        "xmin": center[0] - half_x - margin,
        "xmax": center[0] + half_x + margin,
        "ymin": center[1] - half_y - margin,
        "ymax": center[1] + half_y + margin,
        "zmin": target_aabb["zmin"],
        "zmax": target_aabb["zmax"],
        "axis": axis,
    }


def blocks_grasp(obstacle, target, grasp_yaw_rad=None, ignore_locked=False):
    if not isinstance(obstacle, dict) or not isinstance(target, dict):
        return False
    if _same_object(obstacle, target):
        return False
    if obstacle.get("visible") is False:
        return False
    if ignore_locked and is_locked(obstacle):
        return False

    obstacle_aabb = object_xy_aabb(obstacle)
    target_aabb = object_xy_aabb(target)
    corridor = grasp_corridor_aabb(target, grasp_yaw_rad=grasp_yaw_rad)
    if not obstacle_aabb or not target_aabb or not corridor:
        return False
    if xy_aabb_overlap(obstacle_aabb, corridor)[2] <= 0.0:
        return False
    return _z_gap(obstacle_aabb, target_aabb) < 0.04


def choose_push_direction(obstacle, target):
    obstacle_center = get_center(obstacle)
    target_center = get_center(target)
    if obstacle_center is None or target_center is None:
        return [1.0, 0.0, 0.0]
    dx = obstacle_center[0] - target_center[0]
    dy = obstacle_center[1] - target_center[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-9:
        return [1.0, 0.0, 0.0]
    return [dx / distance, dy / distance, 0.0]


def has_free_push_space(
    obstacle,
    objects,
    direction_base,
    distance_m=0.05,
    table_bounds=None,
    clearance_m=0.01,
):
    obstacle_aabb = object_xy_aabb(obstacle)
    direction = _finite_vector(direction_base, 2)
    if not obstacle_aabb or direction is None:
        return False

    norm = math.hypot(direction[0], direction[1])
    if norm < 1e-9:
        return False
    distance = max(0.0, float(distance_m))
    dx = direction[0] / norm * distance
    dy = direction[1] / norm * distance
    moved_aabb = _translated_aabb(obstacle_aabb, dx, dy)
    clearance_aabb = _expanded_aabb(moved_aabb, clearance_m)

    if table_bounds is not None:
        try:
            if (
                clearance_aabb["xmin"] < float(table_bounds["xmin"])
                or clearance_aabb["xmax"] > float(table_bounds["xmax"])
                or clearance_aabb["ymin"] < float(table_bounds["ymin"])
                or clearance_aabb["ymax"] > float(table_bounds["ymax"])
            ):
                return False
        except (KeyError, TypeError, ValueError):
            return False

    for other in objects or []:
        if not isinstance(other, dict) or _same_object(obstacle, other):
            continue
        if not is_locked(other):
            continue
        other_aabb = object_xy_aabb(other)
        if not other_aabb:
            continue
        if xy_aabb_overlap(clearance_aabb, other_aabb)[2] > 0.0:
            return False
    return True


def safe_to_push(
    obstacle,
    objects,
    direction_base,
    distance_m=0.05,
    table_bounds=None,
):
    if is_locked(obstacle):
        return False
    if not isinstance(obstacle, dict) or obstacle.get("pushable") is False:
        return False
    return has_free_push_space(
        obstacle,
        objects,
        direction_base,
        distance_m=distance_m,
        table_bounds=table_bounds,
    )


def build_geometry_relations(objects, target_id=None, table_bounds=None):
    """Build deterministic pairwise and target-specific geometry relations."""
    scene_objects = [obj for obj in (objects or []) if isinstance(obj, dict)]
    relations = []

    for index, first in enumerate(scene_objects):
        for second in scene_objects[index + 1 :]:
            distance = xy_distance(first, second)
            if distance <= 0.06:
                relations.append(
                    {
                        "type": "near",
                        "subject": _object_name(first),
                        "object": _object_name(second),
                        "source": "geometry",
                        "distance_m": round(distance, 6),
                    }
                )

    for top in scene_objects:
        for bottom in scene_objects:
            if _same_object(top, bottom) or not is_on(top, bottom):
                continue
            relations.append(
                {
                    "type": "on",
                    "subject": _object_name(top),
                    "object": _object_name(bottom),
                    "source": "geometry",
                }
            )
            relations.append(
                {
                    "type": "supporting",
                    "subject": _object_name(bottom),
                    "object": _object_name(top),
                    "source": "geometry",
                }
            )

    target = next(
        (obj for obj in scene_objects if obj.get("id") == target_id),
        None,
    )
    if target is None:
        return relations

    for obstacle in scene_objects:
        if not blocks_grasp(obstacle, target):
            continue
        obstacle_name = _object_name(obstacle)
        target_name = _object_name(target)
        relations.append(
            {
                "type": "blocking_grasp",
                "subject": obstacle_name,
                "object": target_name,
                "source": "geometry",
                "reason": "obstacle_aabb_overlaps_grasp_corridor",
            }
        )

        direction = choose_push_direction(obstacle, target)
        if not safe_to_push(
            obstacle,
            scene_objects,
            direction,
            distance_m=0.05,
            table_bounds=table_bounds,
        ):
            continue

        rounded_direction = [round(value, 6) for value in direction]
        relations.append(
            {
                "type": "safe_to_push",
                "subject": obstacle_name,
                "object": target_name,
                "source": "geometry",
                "direction_base": rounded_direction,
                "distance_m": 0.05,
                "reason": "push_path_has_free_space",
            }
        )
        relations.append(
            {
                "type": "should_push_away",
                "subject": obstacle_name,
                "object": target_name,
                "source": "geometry",
                "direction_base": rounded_direction,
                "distance_m": 0.05,
                "reason": "blocking_grasp_and_safe_to_push",
            }
        )

    return relations
