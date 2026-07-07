"""Safety policy helpers for stack-demo clearance candidates."""

from __future__ import annotations


def refresh_executable_safe(candidate: dict) -> dict:
    preflight_allowed = candidate.get("clearance_preflight_allowed")
    if preflight_allowed is None:
        preflight_allowed = bool(
            candidate.get("feasible")
            and candidate.get("geometry_feasible")
            and candidate.get("approach_path_safe")
            and candidate.get("push_swept_safe")
            and candidate.get("push_end_safe")
            and candidate.get("future_task_feasible", True)
            and candidate.get("task_effective")
            and candidate.get("protected_structure_safe")
            and candidate.get("automatic_execution_allowed", False)
        )
    candidate["executable_safe"] = bool(
        preflight_allowed
        and candidate.get("moveit_feasible")
    )
    return candidate


def verified_clearance_progress(candidate: dict) -> bool:
    if bool(candidate.get("direct_clearance_candidate")):
        return True
    if bool(candidate.get("direct_progress_candidate")):
        return True
    if bool(candidate.get("enabling_clearance_candidate")):
        return True
    if max(0, int(candidate.get("blocker_count_reduction") or 0)) > 0:
        return True
    if max(0.0, float(candidate.get("current_grasp_gain") or 0.0)) > 0.0:
        return True
    return bool(candidate.get("post_push_grasp_feasible"))


def annotate_nudge_preflight_policy(candidate: dict, nudge_max_m: float) -> dict:
    if candidate.get("action_type") != "nudge":
        candidate["clearance_preflight_allowed"] = bool(
            candidate.get("feasible")
            and candidate.get("geometry_feasible")
            and candidate.get("protected_structure_safe")
            and candidate.get("task_effective")
            and candidate.get("automatic_execution_allowed")
        )
        return refresh_executable_safe(candidate)

    verified_progress = verified_clearance_progress(candidate)
    soft_geometry = _soft_clearance_geometry_allowed(candidate, nudge_max_m)
    strict_geometry = bool(
        candidate.get("feasible")
        and candidate.get("geometry_feasible")
        and candidate.get("approach_path_safe")
        and candidate.get("push_swept_safe")
        and candidate.get("push_end_safe")
    )
    geometry_ok = strict_geometry
    future_ok = bool(candidate.get("future_task_feasible", True))
    decision_ok = bool(candidate.get("automatic_execution_allowed") or verified_progress)
    candidate["verified_clearance_progress"] = verified_progress
    candidate["soft_clearance_geometry_allowed"] = soft_geometry
    candidate["future_task_clearance_override"] = False
    candidate["clearance_preflight_allowed"] = bool(
        _short_nudge(candidate, nudge_max_m)
        and geometry_ok
        and future_ok
        and decision_ok
        and candidate.get("protected_structure_safe")
        and candidate.get("task_effective")
    )
    if candidate["clearance_preflight_allowed"] and candidate.get("automatic_execution_allowed") is False:
        candidate["automatic_execution_allowed"] = True
        candidate["automatic_execution_reason"] = "verified_clearance_progress"
    return refresh_executable_safe(candidate)


def _short_nudge(candidate: dict, nudge_max_m: float) -> bool:
    if candidate.get("action_type") != "nudge":
        return False
    try:
        distance_m = float(candidate.get("distance_m") or 0.0)
    except (TypeError, ValueError):
        return False
    return distance_m <= float(nudge_max_m) + 1e-9


def _soft_clearance_geometry_allowed(candidate: dict, nudge_max_m: float) -> bool:
    progress_reasons = {
        "push_reduces_current_blockers_and_preserves_future_tasks",
        "push_enables_current_grasp_and_preserves_future_tasks",
        "push_increases_target_distance_and_preserves_future_tasks",
    }
    if candidate.get("reason") not in progress_reasons:
        return False
    if not _short_nudge(candidate, nudge_max_m):
        return False
    if not candidate.get("feasible"):
        return False
    if not candidate.get("push_end_safe"):
        return False
    if not candidate.get("protected_structure_safe"):
        return False
    if not verified_clearance_progress(candidate):
        return False
    return bool(not candidate.get("approach_path_safe") or not candidate.get("push_swept_safe"))
