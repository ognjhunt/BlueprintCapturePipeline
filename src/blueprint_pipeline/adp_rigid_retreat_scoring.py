"""Authored rigid withdrawal clearance from measured native grasp/scene poses.

Clearance is the signed separation from the subject's oriented collision bounds
along the destination's live qualified withdrawal axis, throughout final settle.
Commanded robot targets and caller-provided retreat verdicts are never consumed.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any


def _vector(value: Any, size: int) -> list[float] | None:
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        return None
    return [float(v) for v in value]


def _rotate(v: Sequence[float], q: Sequence[float]) -> list[float]:
    x, y, z, w = q
    a, b, c = v
    tx, ty, tz = 2*(y*c-z*b), 2*(z*a-x*c), 2*(x*b-y*a)
    return [a+w*tx+y*tz-z*ty, b+w*ty+z*tx-x*tz, c+w*tz+x*ty-y*tx]


def validate_retreat_criterion(value: Any) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != {
        "mode", "minimum_clearance_m", "withdrawal_unit_destination_frame"
    }:
        return ["rigid_task_success_contract_retreat_invalid"]
    direction = _vector(value.get("withdrawal_unit_destination_frame"), 3)
    clearance = value.get("minimum_clearance_m")
    if (value.get("mode") != "required" or direction is None
            or not math.isclose(sum(v*v for v in direction), 1.0, abs_tol=1e-6)
            or isinstance(clearance, bool) or not isinstance(clearance, (int, float))
            or not math.isfinite(clearance) or clearance <= 0):
        return ["rigid_task_success_contract_retreat_invalid"]
    return []



def _derive_retreat_criterion(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Internal compiler derivation; no artifact publication or stand-alone lane."""
    pose = _vector(task_spec.get("destination_pose_world"), 7)
    direction = _vector((task_spec.get("interaction_affordance") or {}).get(
        "insertion_withdrawal_unit_world"), 3)
    if pose is None or direction is None:
        raise ValueError("rigid_task_retreat_destination_qualification_missing")
    criterion = {"mode": "required", "minimum_clearance_m": task_spec.get("retreat_clearance_m"),
                 "withdrawal_unit_destination_frame": _rotate(
                     direction, [-pose[3], -pose[4], -pose[5], pose[6]])}
    errors = validate_retreat_criterion(criterion)
    if errors:
        raise ValueError("rigid_task_retreat_explicit_clearance_or_direction_missing")
    return criterion

def validate_retreat_binding(task_spec: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    """Join the owner criterion to destination-qualified geometry and clearance."""
    criterion = contract.get("criteria", {}).get("retreat")
    errors = validate_retreat_criterion(criterion)
    if errors:
        return errors
    criteria = contract["criteria"]
    pose = _vector(task_spec.get("destination_pose_world"), 7)
    affordance = task_spec.get("interaction_affordance") or {}
    qualified = _vector(affordance.get("insertion_withdrawal_unit_world"), 3)
    if (criteria.get("settling", {}).get("mode") != "required"
            or criteria.get("gripper_state", {}).get("mode") != "released"
            or criteria.get("terminal_task_contact", {}).get("mode") != "cleared"
            or task_spec.get("destination_relation") not in {"inside", "on"}
            or pose is None or qualified is None
            or task_spec.get("retreat_clearance_m") != criterion["minimum_clearance_m"]
            or not isinstance(task_spec.get("subject_collision_bounds_scoring_frame_m"), Mapping)
            or math.dist(_rotate(criterion["withdrawal_unit_destination_frame"], pose[3:]),
                         qualified) > 1e-6):
        errors.append("rigid_task_success_contract_retreat_binding_mismatch")
    return errors


def score_retreat(*, criterion: Mapping[str, Any], task_spec: Mapping[str, Any],
                  samples: Sequence[Mapping[str, Any]], window_samples: int,
                  release_width_m: float) -> dict[str, Any]:
    bounds = task_spec["subject_collision_bounds_scoring_frame_m"]
    clearances: list[float] = []
    gaps: list[int] = []
    released = True
    for row in samples[-window_samples:]:
        grasp = _vector(row.get("grasp_frame_position_world_m"), 3)
        subject = _vector(row.get("task_object_pose_world"), 7)
        destination = _vector(row.get("destination_pose_world"), 7)
        width = row.get("gripper_width_m")
        if (grasp is None or subject is None or destination is None
                or isinstance(width, bool) or not isinstance(width, (int, float))
                or not math.isfinite(width) or not isinstance(row.get("task_contact_active"), bool)):
            gaps.append(row["step_index"])
            continue
        direction = _rotate(criterion["withdrawal_unit_destination_frame"], destination[3:])
        corners = [_rotate(corner, subject[3:]) for corner in product(
            *zip(bounds["minimum"], bounds["maximum"], strict=True))]
        clearance = sum((grasp[i]-subject[i])*direction[i] for i in range(3)) - max(
            sum(corner[i]*direction[i] for i in range(3)) for corner in corners)
        clearances.append(clearance)
        released = released and row["task_contact_active"] is False and width >= release_width_m
    complete = len(samples) >= window_samples and not gaps
    minimum = min(clearances) if clearances else None
    return {
        "satisfied": complete and released and minimum is not None
        and minimum >= criterion["minimum_clearance_m"],
        "readback_complete": complete,
        "minimum_observed_clearance_m": minimum,
        "required_clearance_m": criterion["minimum_clearance_m"],
        "readback_gap_steps": gaps,
        "measurement_source": "native_grasp_frame_and_oriented_subject_bounds_in_live_destination_frame",
    }
