"""Auditable counters and rejection records for physical candidate generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..common.action_edges import PhysicalActionEdge
from .grasp_edges import GraspScanResult


PUSH_DIRECTIONS = ("+x", "-x", "+y", "-y")


@dataclass
class CandidateGenerationAudit:
    unresolved_track_ids: Sequence[str]
    rejections: list[dict[str, Any]] = field(default_factory=list)
    chain_outcomes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._totals: Counter[str] = Counter()
        self._rejections_by_reason: Counter[str] = Counter()
        self._objects = {
            str(track_id): {
                "direct_grasp": {
                    "raw_generated_count": 0,
                    "geometry_passed_count": 0,
                    "accepted_count": 0,
                    "normalized_yaws_deg": [],
                },
                "push_directions": {
                    direction: {
                        "contact_side": _opposite_side(direction),
                        "generated_count": 0,
                        "geometry_passed_count": 0,
                        "accepted_count": 0,
                        "rejections_by_reason": {},
                        "blocking_track_ids": [],
                    }
                    for direction in PUSH_DIRECTIONS
                },
            }
            for track_id in self.unresolved_track_ids
        }

    def record_grasp_scan(self, scan: GraspScanResult) -> None:
        track = self._objects.setdefault(scan.target_track_id, _empty_object_summary())
        direct = track["direct_grasp"]
        direct["raw_generated_count"] += len(scan.samples)
        direct["geometry_passed_count"] += sum(bool(item.get("safe")) for item in scan.samples)
        direct["normalized_yaws_deg"] = [float(item["yaw_deg"]) for item in scan.samples]
        self._totals["direct_grasp_objects_scanned"] += 1
        self._totals["raw_direct_grasp_candidates_generated"] += len(scan.samples)
        self._totals["direct_grasp_candidates_geometry_passed"] += sum(
            bool(item.get("safe")) for item in scan.samples
        )
        for sample in scan.samples:
            if bool(sample.get("safe")):
                continue
            reason = _grasp_rejection_reason(sample)
            self._reject({
                "candidate_id": _grasp_sample_id(scan.target_track_id, sample["yaw_deg"]),
                "action_type": "direct_grasp",
                "target_track_id": scan.target_track_id,
                "grasp_yaw_deg": float(sample["yaw_deg"]),
                "push_direction": None,
                "contact_side": None,
                "push_distance_m": None,
                "chain_track_ids": [],
                "rejection_stage": "grasp_geometry",
                "rejection_reason": reason,
                "blocking_track_ids": list(sample.get("blocking_track_ids", [])),
                "wrist_3_start_goal_delta_rad": None,
                "wrist_3_limit_rad": 1.75,
                "moveit_error": None,
            })

    def record_direct_edge(self, edge: PhysicalActionEdge) -> None:
        passed = bool(edge.precheck_results.get("passed"))
        if passed:
            self._objects[edge.primary_target_track_id]["direct_grasp"]["accepted_count"] += 1
            self._totals["direct_grasp_candidates_accepted"] += 1
            return
        reason = _edge_rejection_reason(edge.precheck_results)
        self._reject(_edge_rejection(edge, reason, _rejection_stage(reason)))

    def record_push_edge(self, edge: PhysicalActionEdge) -> None:
        physical = edge.physical_parameters
        checks = edge.precheck_results
        direction = str(physical.get("push_direction") or "")
        direction_stats = self._objects[edge.primary_target_track_id]["push_directions"][direction]
        direction_stats["generated_count"] += 1
        self._totals["raw_push_candidates_generated"] += 1
        if bool(checks.get("geometry_checks_passed")):
            direction_stats["geometry_passed_count"] += 1
            self._totals["push_candidates_geometry_passed"] += 1

        chain_ids = list(checks.get("chain_track_ids") or physical.get("chain_track_ids") or [])
        secondary = bool(checks.get("secondary_contact_expected"))
        if secondary:
            self._totals["push_chain_generated_count"] += 1
            self._totals["controlled_secondary_contact_count"] += 1
        passed = bool(checks.get("passed"))
        if passed:
            direction_stats["accepted_count"] += 1
            self._totals["push_candidates_accepted"] += 1
        else:
            reason = _edge_rejection_reason(checks)
            direction_stats["rejections_by_reason"][reason] = (
                int(direction_stats["rejections_by_reason"].get(reason, 0)) + 1
            )
            blockers = sorted(set(direction_stats["blocking_track_ids"]).union(
                str(value) for value in checks.get("blocking_track_ids", ())
            ))
            direction_stats["blocking_track_ids"] = blockers
            if reason == "pre_push_contact_side_blocked":
                self._totals["contact_side_blocked_count"] += 1
            self._reject(_edge_rejection(edge, reason, _rejection_stage(reason)))
        if secondary:
            self.chain_outcomes.append({
                "candidate_id": edge.candidate_id,
                "target_track_id": edge.primary_target_track_id,
                "push_direction": direction,
                "contact_side": physical.get("contact_side"),
                "chain_track_ids": chain_ids,
                "accepted": passed,
                "reason": "push_chain_accepted" if passed else _edge_rejection_reason(checks),
            })

    def summary(
        self,
        *,
        physical_action_edge_count: int,
        target_option_count: int,
    ) -> dict[str, Any]:
        unresolved_count = len(tuple(self.unresolved_track_ids))
        internal_error = bool(
            unresolved_count > 0
            and self._totals["raw_direct_grasp_candidates_generated"] == 0
            and self._totals["raw_push_candidates_generated"] == 0
        )
        if internal_error and not self._rejections_by_reason["candidate_generation_internal_error"]:
            self._reject({
                "candidate_id": "candidate_generation_internal_error",
                "action_type": "candidate_generation",
                "target_track_id": "",
                "grasp_yaw_deg": None,
                "push_direction": None,
                "contact_side": None,
                "push_distance_m": None,
                "chain_track_ids": [],
                "rejection_stage": "generation",
                "rejection_reason": "candidate_generation_internal_error",
                "blocking_track_ids": [],
                "wrist_3_start_goal_delta_rad": None,
                "wrist_3_limit_rad": 1.75,
                "moveit_error": None,
            })
        return {
            "unresolved_object_count": unresolved_count,
            "direct_grasp_objects_scanned": self._totals["direct_grasp_objects_scanned"],
            "raw_direct_grasp_candidates_generated": self._totals["raw_direct_grasp_candidates_generated"],
            "direct_grasp_candidates_geometry_passed": self._totals["direct_grasp_candidates_geometry_passed"],
            "direct_grasp_candidates_accepted": self._totals["direct_grasp_candidates_accepted"],
            "push_objects_scanned": self._totals["push_objects_scanned"],
            "push_directions_scanned": self._totals["push_directions_scanned"],
            "raw_push_candidates_generated": self._totals["raw_push_candidates_generated"],
            "push_candidates_geometry_passed": self._totals["push_candidates_geometry_passed"],
            "push_candidates_accepted": self._totals["push_candidates_accepted"],
            "contact_side_blocked_count": self._totals["contact_side_blocked_count"],
            "push_chain_generated_count": self._totals["push_chain_generated_count"],
            "controlled_secondary_contact_count": self._totals["controlled_secondary_contact_count"],
            "physical_action_edge_count": int(physical_action_edge_count),
            "target_option_count": int(target_option_count),
            "candidate_generation_internal_error": internal_error,
            "rejections_by_reason": dict(sorted(self._rejections_by_reason.items())),
            "objects": self._objects,
            "push_chain_outcomes": self.chain_outcomes,
        }

    def begin_push_object(self, track_id: str) -> None:
        self._objects.setdefault(track_id, _empty_object_summary())
        self._totals["push_objects_scanned"] += 1
        self._totals["push_directions_scanned"] += len(PUSH_DIRECTIONS)

    def _reject(self, record: dict[str, Any]) -> None:
        self.rejections.append(record)
        self._rejections_by_reason[str(record["rejection_reason"])] += 1


def _empty_object_summary() -> dict[str, Any]:
    return {
        "direct_grasp": {
            "raw_generated_count": 0,
            "geometry_passed_count": 0,
            "accepted_count": 0,
            "normalized_yaws_deg": [],
        },
        "push_directions": {
            direction: {
                "contact_side": _opposite_side(direction),
                "generated_count": 0,
                "geometry_passed_count": 0,
                "accepted_count": 0,
                "rejections_by_reason": {},
                "blocking_track_ids": [],
            }
            for direction in PUSH_DIRECTIONS
        },
    }


def _opposite_side(direction: str) -> str:
    return {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}[direction]


def _grasp_sample_id(track_id: str, yaw_deg: Any) -> str:
    return f"grasp:{track_id}:{float(yaw_deg):+.1f}"


def _grasp_rejection_reason(sample: Mapping[str, Any]) -> str:
    if not bool(sample.get("opening_ok")):
        return "gripper_opening_insufficient"
    if not bool(sample.get("center_offset_ok")):
        return "grasp_center_offset_exceeded"
    if not bool(sample.get("contact_length_ok")):
        return "contact_length_insufficient"
    if not bool(sample.get("finger_safe")):
        return "finger_collision"
    if not bool(sample.get("palm_safe")):
        return "palm_collision"
    if not bool(sample.get("descent_safe")):
        return "pre_grasp_descent_collision"
    if not bool(sample.get("lift_safe")):
        return "lift_collision"
    return "grasp_geometry_rejected"


def _edge_rejection_reason(checks: Mapping[str, Any]) -> str:
    reasons = [str(value) for value in checks.get("rejection_reasons", ()) if value]
    if reasons:
        return reasons[0]
    if checks.get("error"):
        return "moveit_plan_failed"
    for key, reason in (
        ("transport_safe", "transport_collision"),
        ("place_descent_safe", "place_descent_collision"),
        ("release_safe", "release_collision"),
        ("return_safe", "return_collision"),
    ):
        if checks.get(key) is False:
            return reason
    return "geometry_rejected"


def _rejection_stage(reason: str) -> str:
    if reason == "pre_push_contact_side_blocked":
        return "contact_side_validation"
    if reason.startswith("push_chain") or reason in {
        "protected_completed_object_collision", "fixed_obstacle_collision",
    }:
        return "push_chain_validation"
    if reason in {"moveit_plan_failed", "ik_failed", "joint_delta_exceeded"}:
        return "moveit"
    return "geometry"


def _edge_rejection(
    edge: PhysicalActionEdge,
    reason: str,
    stage: str,
) -> dict[str, Any]:
    physical = edge.physical_parameters
    checks = edge.precheck_results
    return {
        "candidate_id": edge.candidate_id,
        "action_type": edge.action_type.value,
        "target_track_id": edge.primary_target_track_id,
        "grasp_yaw_deg": physical.get("grasp_yaw_deg"),
        "push_direction": physical.get("push_direction"),
        "contact_side": physical.get("contact_side"),
        "push_distance_m": physical.get("push_distance_m"),
        "chain_track_ids": list(checks.get("chain_track_ids") or physical.get("chain_track_ids") or []),
        "rejection_stage": stage,
        "rejection_reason": reason,
        "blocking_track_ids": list(checks.get("blocking_track_ids", [])),
        "wrist_3_start_goal_delta_rad": checks.get("wrist_3_start_goal_delta_rad"),
        "wrist_3_limit_rad": 1.75,
        "moveit_error": checks.get("error"),
    }
