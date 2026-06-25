"""Decision fields derived from adaptive grasp-yaw search results."""

from typing import Dict, Iterable, List

from .grasp_yaw_search import object_summary


def all_blocking_objects(grasp_result: Dict[str, object]) -> List[dict]:
    blockers = {}
    for blocker in grasp_result.get("blocking_objects", []):
        blockers[str(blocker.get("id"))] = blocker
    for interval in grasp_result.get("blocking_objects_by_interval", []):
        for blocker in interval.get("blocking_objects", []):
            blockers[str(blocker.get("id"))] = blocker
    return list(blockers.values())


def is_loose_movable_blocker(blocker: dict) -> bool:
    return (
        blocker.get("blocker_category") == "loose_movable"
        and blocker.get("pushable", True) is not False
    )


def make_grasp_analysis_relation(
    target_name: object,
    top_objects: Iterable[dict],
    grasp_result: Dict[str, object] = None,
) -> dict:
    top_objects = list(top_objects or [])
    if top_objects:
        above = [object_summary(obj) for obj in top_objects]
        return {
            "type": "target_grasp_analysis",
            "object": target_name,
            "source": "geometry",
            "action": "remove_top_object",
            "target_under_other_object": True,
            "object_above_target": above[0],
            "objects_above_target": above,
            "grasp_feasible": False,
            "selected_grasp_yaw_deg": None,
            "feasible_yaw_intervals_deg": [],
            "blocked_yaw_intervals_deg": [],
            "all_grasps_blocked": False,
            "blocking_objects_by_interval": [],
            "blocked_by_base": False,
            "blocked_by_locked_structure": False,
            "blocked_by_placed_structure": False,
            "replan_required": False,
            "reason": "target_under_other_object",
        }

    grasp_result = grasp_result or {}
    blockers = all_blocking_objects(grasp_result)
    all_grasps_blocked = bool(grasp_result.get("all_grasps_blocked"))
    if grasp_result.get("grasp_feasible"):
        action = "pick"
        replan_required = False
    elif all_grasps_blocked and blockers and all(is_loose_movable_blocker(blocker) for blocker in blockers):
        action = "push_clearing"
        replan_required = False
    else:
        action = "replan_required"
        replan_required = True

    return {
        "type": "target_grasp_analysis",
        "object": target_name,
        "source": "geometry",
        "action": action,
        "target_under_other_object": False,
        "object_above_target": None,
        "objects_above_target": [],
        "grasp_feasible": bool(grasp_result.get("grasp_feasible")),
        "selected_grasp_yaw_deg": grasp_result.get("selected_grasp_yaw_deg"),
        "selected_grasp_source": grasp_result.get("selected_grasp_source"),
        "feasible_yaw_intervals_deg": grasp_result.get("feasible_yaw_intervals_deg", []),
        "blocked_yaw_intervals_deg": grasp_result.get("blocked_yaw_intervals_deg", []),
        "all_grasps_blocked": all_grasps_blocked,
        "blocking_objects_by_interval": grasp_result.get("blocking_objects_by_interval", []),
        "blocking_objects": blockers,
        "blocked_by_base": bool(grasp_result.get("blocked_by_base")),
        "blocked_by_locked_structure": bool(grasp_result.get("blocked_by_locked_structure")),
        "blocked_by_placed_structure": bool(grasp_result.get("blocked_by_placed_structure")),
        "replan_required": replan_required,
        "reason": "adaptive_grasp_yaw_search",
        "grasp_search_parameters": grasp_result.get("parameters", {}),
    }
