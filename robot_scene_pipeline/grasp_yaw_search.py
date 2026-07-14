"""Adaptive continuous yaw search for parallel-gripper tabletop grasps."""

import math
from typing import Dict, Iterable, List, Optional, Tuple


ObjectDict = Dict[str, object]


def normalize_yaw_180(yaw_deg: float) -> float:
    return float(yaw_deg) % 180.0


def normalize_yaw_signed_180(yaw_deg: float) -> float:
    """Return the 180-degree-equivalent yaw closest to zero."""
    return ((float(yaw_deg) + 90.0) % 180.0) - 90.0


def equivalent_yaw_delta_deg(first_yaw_deg: float, second_yaw_deg: float) -> float:
    return abs(normalize_yaw_signed_180(float(first_yaw_deg) - float(second_yaw_deg)))


def _finite_vector(value: object, minimum_length: int) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum_length:
        return None
    try:
        output = [float(value[index]) for index in range(minimum_length)]
    except (TypeError, ValueError):
        return None
    return output if all(math.isfinite(item) for item in output) else None


def get_center(obj: ObjectDict) -> Optional[List[float]]:
    for key in ("geometry_center_m", "last_pose_base", "center_base_m", "center_3d_base_m"):
        center = _finite_vector(obj.get(key), 3)
        if center is not None:
            return center
    return None


def get_size(obj: ObjectDict) -> Optional[List[float]]:
    for key in ("dimensions_m", "size_m"):
        size = _finite_vector(obj.get(key), 3)
        if size is not None:
            return [abs(value) for value in size]
    return None


def get_yaw_deg(obj: ObjectDict) -> float:
    for key, scale in (("table_yaw_deg", 1.0), ("yaw_deg", 1.0), ("yaw_rad", 180.0 / math.pi)):
        value = obj.get(key)
        if value is None:
            continue
        try:
            yaw = float(value) * scale
        except (TypeError, ValueError):
            continue
        if math.isfinite(yaw):
            return normalize_yaw_180(yaw)
    return 0.0


def object_summary(obj: ObjectDict) -> dict:
    return {
        "id": obj.get("id"),
        "label": obj.get("label"),
        "role": obj.get("role"),
        "state": obj.get("state"),
        "pushable": obj.get("pushable", True),
    }


def is_same_object(first: ObjectDict, second: ObjectDict) -> bool:
    if first is second:
        return True
    first_id = first.get("id")
    second_id = second.get("id")
    if first_id is None or second_id is None or first_id != second_id:
        return False
    first_label = first.get("label")
    second_label = second.get("label")
    if first_label is not None and second_label is not None and first_label != second_label:
        return False
    first_center = get_center(first)
    second_center = get_center(second)
    if first_center is not None and second_center is not None:
        distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
        return distance <= 0.025
    return True


def _object_corners_xy(obj: ObjectDict) -> List[Tuple[float, float]]:
    center = get_center(obj)
    size = get_size(obj)
    if center is None or size is None:
        return []
    yaw = math.radians(get_yaw_deg(obj))
    ux = (math.cos(yaw), math.sin(yaw))
    uy = (-math.sin(yaw), math.cos(yaw))
    hx = 0.5 * size[0]
    hy = 0.5 * size[1]
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            corners.append(
                (
                    center[0] + sx * hx * ux[0] + sy * hy * uy[0],
                    center[1] + sx * hx * ux[1] + sy * hy * uy[1],
                )
            )
    return corners


def _project_object(obj: ObjectDict, origin_xy: Iterable[float], yaw_deg: float) -> Optional[dict]:
    corners = _object_corners_xy(obj)
    if not corners:
        return None
    ox, oy = [float(value) for value in list(origin_xy)[:2]]
    yaw = math.radians(float(yaw_deg))
    ux = (math.cos(yaw), math.sin(yaw))
    uy = (-math.sin(yaw), math.cos(yaw))
    us = []
    vs = []
    for x, y in corners:
        dx = x - ox
        dy = y - oy
        us.append(dx * ux[0] + dy * ux[1])
        vs.append(dx * uy[0] + dy * uy[1])
    return {
        "umin": min(us),
        "umax": max(us),
        "vmin": min(vs),
        "vmax": max(vs),
    }


def _z_ranges_relevant(obstacle: ObjectDict, target: ObjectDict, tolerance_m: float) -> bool:
    obstacle_center = get_center(obstacle)
    target_center = get_center(target)
    obstacle_size = get_size(obstacle)
    target_size = get_size(target)
    if obstacle_center is None or target_center is None or obstacle_size is None or target_size is None:
        return False
    obstacle_zmin = obstacle_center[2] - 0.5 * obstacle_size[2]
    obstacle_zmax = obstacle_center[2] + 0.5 * obstacle_size[2]
    target_zmin = target_center[2] - 0.5 * target_size[2]
    target_zmax = target_center[2] + 0.5 * target_size[2]
    if obstacle_zmax <= target_zmin + 0.002:
        return False
    if obstacle_zmin > target_zmax + float(tolerance_m):
        return False
    return True


def _aabb_overlap(first: dict, second: dict) -> bool:
    return (
        min(first["umax"], second["umax"]) > max(first["umin"], second["umin"])
        and min(first["vmax"], second["vmax"]) > max(first["vmin"], second["vmin"])
    )


def _aabb_gap(first: dict, second: dict) -> float:
    du = max(0.0, max(first["umin"] - second["umax"], second["umin"] - first["umax"]))
    dv = max(0.0, max(first["vmin"] - second["vmax"], second["vmin"] - first["vmax"]))
    return math.hypot(du, dv)


def _gripper_finger_boxes(half_u: float, inner_half_v: float, outer_half_v: float) -> List[dict]:
    if outer_half_v <= inner_half_v:
        return [
            {
                "umin": -half_u,
                "umax": half_u,
                "vmin": -outer_half_v,
                "vmax": outer_half_v,
            }
        ]
    return [
        {
            "umin": -half_u,
            "umax": half_u,
            "vmin": inner_half_v,
            "vmax": outer_half_v,
        },
        {
            "umin": -half_u,
            "umax": half_u,
            "vmin": -outer_half_v,
            "vmax": -inner_half_v,
        },
    ]


def _gripper_side_clearance_boxes(
    half_u: float,
    target_half_v: float,
    inner_half_v: float,
    side_clearance_m: float,
) -> List[dict]:
    clearance = max(0.0, float(side_clearance_m))
    side_limit = max(float(inner_half_v), float(target_half_v) + clearance)
    if side_limit <= float(target_half_v) + 1e-9:
        return []
    return [
        {
            "umin": -half_u,
            "umax": half_u,
            "vmin": target_half_v,
            "vmax": side_limit,
        },
        {
            "umin": -half_u,
            "umax": half_u,
            "vmin": -side_limit,
            "vmax": -target_half_v,
        },
    ]


def blocker_category(obj: ObjectDict) -> str:
    role = str(obj.get("role") or "").lower()
    state = str(obj.get("state") or "").lower()
    if role == "base" or obj.get("is_base"):
        return "base"
    if state == "placed":
        return "placed_structure"
    if role == "structure" or state == "locked":
        return "locked_structure"
    return "loose_movable"


def _candidate_yaws(
    target: ObjectDict,
    obstacles: Iterable[ObjectDict],
    yaw_step_deg: float,
    local_refine_step_deg: float,
    current_wrist_yaw_deg: Optional[float],
) -> List[dict]:
    anchors: List[dict] = []

    def add(yaw: float, source: str) -> None:
        yaw = normalize_yaw_180(yaw)
        if not any(abs(normalize_yaw_180(yaw - item["yaw_deg"])) < 1e-6 for item in anchors):
            anchors.append({"yaw_deg": yaw, "source": source})

    target_yaw = get_yaw_deg(target)
    add(target_yaw, "target_principal_axis")
    add(target_yaw + 90.0, "target_principal_axis")

    target_center = get_center(target)
    if target_center is not None:
        for obstacle in obstacles:
            obstacle_center = get_center(obstacle)
            if obstacle_center is None:
                continue
            angle = math.degrees(
                math.atan2(
                    obstacle_center[1] - target_center[1],
                    obstacle_center[0] - target_center[0],
                )
            )
            add(angle + 90.0, "obstacle_aware_perpendicular")
            add(angle + 90.0 - float(local_refine_step_deg), "obstacle_aware_refine")
            add(angle + 90.0 + float(local_refine_step_deg), "obstacle_aware_refine")

    if current_wrist_yaw_deg is not None:
        add(float(current_wrist_yaw_deg), "current_wrist_yaw_bias")

    step = max(1e-6, float(yaw_step_deg))
    count = max(1, int(math.ceil(180.0 / step)))
    for index in range(count):
        add(index * step, "uniform_fallback")

    refined = []
    refine = max(1e-6, float(local_refine_step_deg))
    radius = max(step, refine)
    for anchor in anchors:
        offset = -radius
        while offset <= radius + 1e-9:
            refined.append({"yaw_deg": normalize_yaw_180(anchor["yaw_deg"] + offset), "source": anchor["source"]})
            offset += refine

    output = []
    for item in anchors + refined:
        yaw = normalize_yaw_180(item["yaw_deg"])
        if any(abs(normalize_yaw_180(yaw - existing["yaw_deg"])) < 1e-6 for existing in output):
            continue
        output.append({"yaw_deg": round(yaw, 6), "source": item["source"]})
    return output


def _target_axis_alignment_delta_deg(target: ObjectDict, yaw_deg: float) -> float:
    target_yaw = get_yaw_deg(target)
    axes = (target_yaw, target_yaw + 90.0)
    return min(equivalent_yaw_delta_deg(yaw_deg, axis) for axis in axes)


def _evaluate_yaw(
    target: ObjectDict,
    obstacles: Iterable[ObjectDict],
    yaw_deg: float,
    gripper_outer_width_m: float,
    gripper_inner_width_m: float,
    approach_length_m: float,
    z_tolerance_m: float,
    side_clearance_m: float,
) -> dict:
    target_center = get_center(target)
    if target_center is None:
        return {"yaw_deg": float(yaw_deg), "feasible": False, "blocking_objects": [], "reason": "missing_target_center"}
    target_projection = _project_object(target, target_center[:2], yaw_deg)
    if target_projection is None:
        return {"yaw_deg": float(yaw_deg), "feasible": False, "blocking_objects": [], "reason": "missing_target_size"}

    target_half_u = max(abs(target_projection["umin"]), abs(target_projection["umax"]))
    target_half_v = max(abs(target_projection["vmin"]), abs(target_projection["vmax"]))
    target_width_v = target_projection["vmax"] - target_projection["vmin"]
    if target_width_v > float(gripper_inner_width_m) + 0.004:
        return {
            "yaw_deg": float(yaw_deg),
            "feasible": False,
            "blocking_objects": [],
            "reason": "target_exceeds_gripper_inner_width",
            "clearance_m": 0.0,
        }
    half_u = target_half_u + max(0.0, float(approach_length_m))
    outer_half_v = max(0.5 * float(gripper_outer_width_m), target_half_v)
    inner_half_v = 0.5 * float(gripper_inner_width_m)
    if float(gripper_outer_width_m) <= float(gripper_inner_width_m):
        inner_half_v = outer_half_v
    finger_boxes = _gripper_finger_boxes(half_u, inner_half_v, outer_half_v)
    side_clearance_boxes = _gripper_side_clearance_boxes(
        half_u,
        target_half_v,
        inner_half_v,
        side_clearance_m,
    )

    blockers = []
    clearances = []
    seen_blockers = set()
    for obstacle in obstacles:
        if obstacle.get("visible") is False or is_same_object(obstacle, target):
            continue
        if not _z_ranges_relevant(obstacle, target, z_tolerance_m):
            continue
        projection = _project_object(obstacle, target_center[:2], yaw_deg)
        if projection is None:
            continue
        overlaps_finger = any(_aabb_overlap(projection, box) for box in finger_boxes)
        overlaps_side_clearance = any(_aabb_overlap(projection, box) for box in side_clearance_boxes)
        if overlaps_finger or overlaps_side_clearance:
            key = (str(obstacle.get("id")), str(obstacle.get("label")), tuple(round(value, 4) for value in (get_center(obstacle) or [])[:2]))
            if key in seen_blockers:
                continue
            seen_blockers.add(key)
            summary = object_summary(obstacle)
            summary["blocker_category"] = blocker_category(obstacle)
            if overlaps_side_clearance and not overlaps_finger:
                summary["blocker_reason"] = "insufficient_gripper_side_clearance"
            blockers.append(summary)
        else:
            boxes = finger_boxes + side_clearance_boxes
            clearances.append(min(_aabb_gap(projection, box) for box in boxes))

    feasible = not blockers
    clearance = min(clearances) if clearances else 1.0
    return {
        "yaw_deg": round(float(yaw_deg), 6),
        "feasible": feasible,
        "blocking_objects": blockers,
        "clearance_m": round(float(clearance), 6),
    }


def evaluate_grasp_yaw(
    target: ObjectDict,
    objects: Iterable[ObjectDict],
    yaw_deg: float,
    gripper_outer_width_m: float = 0.112,
    gripper_inner_width_m: float = 0.049,
    approach_length_m: float = 0.02,
    z_tolerance_m: float = 0.04,
    side_clearance_m: float = 0.006,
) -> dict:
    """Evaluate one exact top-down gripper yaw against visible obstacles."""
    return _evaluate_yaw(
        target,
        [obj for obj in objects if isinstance(obj, dict)],
        yaw_deg,
        gripper_outer_width_m,
        gripper_inner_width_m,
        approach_length_m,
        z_tolerance_m,
        side_clearance_m,
    )


def _intervals(samples: List[dict], feasible: bool) -> Tuple[List[List[float]], List[dict]]:
    intervals: List[List[float]] = []
    blocker_intervals: List[dict] = []
    start = None
    blockers: Dict[str, dict] = {}
    previous_yaw = 0.0
    step = 1.0
    for index, sample in enumerate(samples):
        yaw = float(sample["yaw_deg"])
        current = bool(sample["feasible"]) == feasible
        if index > 0:
            step = yaw - previous_yaw
        if current and start is None:
            start = yaw
            blockers = {}
        if current and not feasible:
            for blocker in sample.get("blocking_objects", []):
                blockers[str(blocker.get("id"))] = blocker
        next_current = False
        if index + 1 < len(samples):
            next_current = bool(samples[index + 1]["feasible"]) == feasible
        if current and (not next_current or index + 1 >= len(samples)):
            end = min(180.0, yaw + max(step, 1e-6))
            intervals.append([round(float(start), 3), round(float(end), 3)])
            if not feasible:
                blocker_intervals.append(
                    {
                        "interval_deg": intervals[-1],
                        "blocking_objects": list(blockers.values()),
                    }
                )
            start = None
            blockers = {}
        previous_yaw = yaw
    return intervals, blocker_intervals


def select_best_grasp(
    target: ObjectDict,
    objects: Iterable[ObjectDict],
    yaw_step_deg: float = 15.0,
    local_refine_step_deg: float = 1.0,
    gripper_outer_width_m: float = 0.112,
    gripper_inner_width_m: float = 0.048,
    current_wrist_yaw_deg: Optional[float] = None,
    approach_length_m: float = 0.02,
    z_tolerance_m: float = 0.04,
    side_clearance_m: float = 0.006,
    min_feasible_yaw_span_deg: float = 10.0,
) -> dict:
    obstacles = [obj for obj in objects if isinstance(obj, dict) and not is_same_object(obj, target)]
    candidates = _candidate_yaws(target, obstacles, yaw_step_deg, local_refine_step_deg, current_wrist_yaw_deg)
    candidate_results = []
    for candidate in candidates:
        result = _evaluate_yaw(
            target,
            obstacles,
            candidate["yaw_deg"],
            gripper_outer_width_m,
            gripper_inner_width_m,
            approach_length_m,
            z_tolerance_m,
            side_clearance_m,
        )
        result["source"] = candidate["source"]
        candidate_results.append(result)

    local_step = max(1e-6, float(local_refine_step_deg))
    local_count = max(1, int(math.ceil(180.0 / local_step)))
    local_samples = [
        _evaluate_yaw(
            target,
            obstacles,
            min(index * local_step, 180.0 - 1e-6),
            gripper_outer_width_m,
            gripper_inner_width_m,
            approach_length_m,
            z_tolerance_m,
            side_clearance_m,
        )
        for index in range(local_count)
    ]
    feasible_intervals, _ = _intervals(local_samples, True)
    blocked_intervals, blockers_by_interval = _intervals(local_samples, False)

    minimum_span = max(0.0, float(min_feasible_yaw_span_deg))
    robust_intervals = [
        interval for interval in feasible_intervals
        if float(interval[1]) - float(interval[0]) >= minimum_span
    ]

    def inside_robust_interval(yaw_deg: float) -> bool:
        yaw = normalize_yaw_180(yaw_deg)
        for start, end in robust_intervals:
            span = float(end) - float(start)
            inset = min(2.0, 0.25 * span)
            if float(start) + inset <= yaw <= float(end) - inset:
                return True
        return False

    feasible_candidates = [
        item for item in candidate_results
        if item["feasible"] and inside_robust_interval(float(item["yaw_deg"]))
    ]
    if not feasible_candidates and robust_intervals:
        for start, end in robust_intervals:
            mid = normalize_yaw_180((float(start) + float(end)) / 2.0)
            result = _evaluate_yaw(
                target,
                obstacles,
                mid,
                gripper_outer_width_m,
                gripper_inner_width_m,
                approach_length_m,
                z_tolerance_m,
                side_clearance_m,
            )
            result["source"] = "feasible_interval_midpoint"
            if result["feasible"]:
                feasible_candidates.append(result)

    def score(item: dict) -> tuple:
        source_priority = {
            "current_wrist_yaw_bias": 4,
            "target_principal_axis": 3,
            "obstacle_aware_perpendicular": 2,
            "obstacle_aware_refine": 1,
            "feasible_interval_midpoint": 0,
            "uniform_fallback": -1,
        }.get(str(item.get("source")), 0)
        wrist_penalty = 0.0
        if current_wrist_yaw_deg is not None:
            delta = abs((float(item["yaw_deg"]) - float(current_wrist_yaw_deg) + 90.0) % 180.0 - 90.0)
            wrist_penalty = -delta / 180.0
        axis_delta = _target_axis_alignment_delta_deg(target, float(item["yaw_deg"]))
        return (-axis_delta, source_priority, float(item.get("clearance_m", 0.0)), wrist_penalty, -float(item["yaw_deg"]))

    feasible_candidates.sort(key=score, reverse=True)
    selected = feasible_candidates[0] if feasible_candidates else None
    selected_yaw = None if selected is None else normalize_yaw_signed_180(float(selected["yaw_deg"]))
    selected_axis_delta = None if selected is None else _target_axis_alignment_delta_deg(target, float(selected["yaw_deg"]))
    all_blockers: Dict[str, dict] = {}
    for interval in blockers_by_interval:
        for blocker in interval.get("blocking_objects", []):
            all_blockers[str(blocker.get("id"))] = blocker
    categories = {str(item.get("blocker_category")) for item in all_blockers.values()}
    all_grasps_blocked = not robust_intervals
    return {
        "selected_grasp_yaw_deg": None if selected is None else round(float(selected_yaw), 3),
        "selected_grasp_axis_delta_deg": None if selected is None else round(float(selected_axis_delta), 3),
        "selected_grasp_source": None if selected is None else selected.get("source"),
        "grasp_feasible": selected is not None,
        "feasible_yaw_intervals_deg": feasible_intervals,
        "robust_feasible_yaw_intervals_deg": robust_intervals,
        "rejected_narrow_feasible_yaw_intervals_deg": [
            interval for interval in feasible_intervals if interval not in robust_intervals
        ],
        "blocked_yaw_intervals_deg": blocked_intervals,
        "all_grasps_blocked": bool(all_grasps_blocked),
        "blocking_objects_by_interval": blockers_by_interval,
        "blocking_objects": list(all_blockers.values()),
        "blocked_by_base": "base" in categories,
        "blocked_by_locked_structure": "locked_structure" in categories,
        "blocked_by_placed_structure": "placed_structure" in categories,
        "candidate_results": candidate_results,
        "parameters": {
            "yaw_step_deg": float(yaw_step_deg),
            "local_refine_step_deg": float(local_refine_step_deg),
            "gripper_outer_width_m": float(gripper_outer_width_m),
            "gripper_inner_width_m": float(gripper_inner_width_m),
            "approach_length_m": float(approach_length_m),
            "side_clearance_m": float(side_clearance_m),
            "min_feasible_yaw_span_deg": minimum_span,
        },
    }
