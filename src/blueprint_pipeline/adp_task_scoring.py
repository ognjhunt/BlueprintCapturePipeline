"""Task-neutral deterministic scoring for ADP rigid and articulated tasks.

The original ADP-009D scorer remains the authority for its sealed rigid
pick/place fixture.  This module adds a stable discriminator and an articulated
joint-state scorer without copying or weakening that legacy path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from adp009d_task_scoring import TaskScoringError, score_task_episode
except ModuleNotFoundError:  # repository package
    from .adp009d_task_scoring import TaskScoringError, score_task_episode
try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


TASK_SPEC_SCHEMA_VERSION = "adp_task_spec.v1"
ARTICULATED_REPORT_SCHEMA_VERSION = "adp_articulated_task_scoring.v1"
TASK_KIND_RIGID_PICK_PLACE = "rigid_pick_place"
TASK_KIND_ARTICULATED_OPEN_CLOSE = "articulated_open_close"

OUTCOME_NEVER_MOVED = "never_moved"
OUTCOME_MOVED_BELOW_THRESHOLD = "moved_below_threshold"
OUTCOME_OPENED_THEN_REBOUNDED = "opened_then_rebounded"
OUTCOME_NON_TASK_JOINT_MOVED = "non_task_joint_moved"
OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION = "joint_limit_or_containment_violation"
OUTCOME_COLLISION_FAILURE = "robot_or_scene_collision_failure"
OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE = "release_or_retreat_incomplete"
OUTCOME_OPENED_AND_SETTLED = "opened_and_settled"

_ARTICULATED_OUTCOME_RANK = {
    OUTCOME_NEVER_MOVED: 0,
    OUTCOME_NON_TASK_JOINT_MOVED: 0,
    OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION: 0,
    OUTCOME_COLLISION_FAILURE: 0,
    OUTCOME_MOVED_BELOW_THRESHOLD: 1,
    OUTCOME_OPENED_THEN_REBOUNDED: 2,
    OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE: 3,
    OUTCOME_OPENED_AND_SETTLED: 4,
}


class TaskNeutralScoringError(ValueError):
    """Stable, sorted task-neutral scoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION:
        errors.append("task_spec_schema_invalid")
    if spec.get("task_kind") != TASK_KIND_ARTICULATED_OPEN_CLOSE:
        errors.append("articulated_task_kind_invalid")
    target = str(spec.get("target_joint_id") or "")
    if not target:
        errors.append("articulated_target_joint_missing")
    joint_resets = spec.get("joint_reset_positions_rad")
    if not isinstance(joint_resets, Mapping) or target not in joint_resets:
        errors.append("articulated_joint_resets_invalid")
        joint_resets = {}
    normalized_resets: dict[str, float] = {}
    for joint_id, raw in joint_resets.items():
        value = _finite(raw)
        if not str(joint_id) or value is None:
            errors.append("articulated_joint_resets_invalid")
        else:
            normalized_resets[str(joint_id)] = value
    if not 1 <= len(normalized_resets) <= 4:
        errors.append("articulated_joint_count_outside_frozen_scope")
    interval = spec.get("target_success_interval_rad")
    if (
        not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or _finite(interval[0]) is None
        or _finite(interval[1]) is None
        or float(interval[0]) >= float(interval[1])
    ):
        errors.append("articulated_success_interval_invalid")
        normalized_interval = [0.0, 0.0]
    else:
        normalized_interval = [float(interval[0]), float(interval[1])]
    hard_limits = spec.get("joint_hard_limits_rad")
    normalized_limits: dict[str, list[float]] = {}
    if not isinstance(hard_limits, Mapping) or set(hard_limits) != set(normalized_resets):
        errors.append("articulated_joint_limits_invalid")
    else:
        for joint_id, raw in hard_limits.items():
            if (
                not isinstance(raw, Sequence)
                or isinstance(raw, (str, bytes))
                or len(raw) != 2
                or _finite(raw[0]) is None
                or _finite(raw[1]) is None
                or float(raw[0]) >= float(raw[1])
            ):
                errors.append("articulated_joint_limits_invalid")
            else:
                normalized_limits[str(joint_id)] = [float(raw[0]), float(raw[1])]
    fields = {
        "settle_window_samples": spec.get("settle_window_samples"),
        "maximum_settled_target_speed_rad_s": spec.get(
            "maximum_settled_target_speed_rad_s"
        ),
        "non_task_joint_motion_tolerance_rad": spec.get(
            "non_task_joint_motion_tolerance_rad"
        ),
        "movement_epsilon_rad": spec.get("movement_epsilon_rad"),
        "reset_tolerance_rad": spec.get("reset_tolerance_rad"),
    }
    normalized_fields: dict[str, float | int] = {}
    for field, raw in fields.items():
        value = _finite(raw)
        if value is None or value <= 0:
            errors.append(f"articulated_{field}_invalid")
        elif field == "settle_window_samples" and (
            isinstance(raw, bool) or not isinstance(raw, int)
        ):
            errors.append("articulated_settle_window_samples_invalid")
        else:
            normalized_fields[field] = int(raw) if field == "settle_window_samples" else value
    if target in normalized_resets and normalized_interval[0] <= normalized_resets[target] <= normalized_interval[1]:
        errors.append("articulated_reset_inside_success_interval")
    if errors:
        raise TaskNeutralScoringError(errors)
    return {
        "target_joint_id": target,
        "joint_reset_positions_rad": normalized_resets,
        "target_success_interval_rad": normalized_interval,
        "joint_hard_limits_rad": normalized_limits,
        **normalized_fields,
    }


def validate_articulated_task_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Public fail-closed validator for a frozen articulated scorer contract."""

    return _normalize_articulated_spec(spec)


def _normalize_articulated_samples(
    samples: Sequence[Mapping[str, Any]], *, joint_ids: set[str]
) -> list[dict[str, Any]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise TaskNeutralScoringError(["articulated_samples_invalid"])
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    previous_step: int | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append(f"articulated_sample_{index}_not_mapping")
            continue
        raw_step = sample.get("step_index")
        if isinstance(raw_step, bool) or not isinstance(raw_step, int):
            errors.append(f"articulated_sample_{index}_step_invalid")
            step = index
        else:
            step = raw_step
            if previous_step is not None and step <= previous_step:
                errors.append(f"articulated_sample_{index}_step_not_increasing")
            previous_step = step
        positions = sample.get("joint_positions_rad")
        velocities = sample.get("joint_velocities_rad_s")
        if not isinstance(positions, Mapping) or set(positions) != joint_ids:
            errors.append(f"articulated_sample_{index}_joint_positions_invalid")
            continue
        if not isinstance(velocities, Mapping) or set(velocities) != joint_ids:
            errors.append(f"articulated_sample_{index}_joint_velocities_invalid")
            continue
        normalized_positions: dict[str, float] = {}
        normalized_velocities: dict[str, float] = {}
        for joint_id in sorted(joint_ids):
            position = _finite(positions[joint_id])
            velocity = _finite(velocities[joint_id])
            if position is None:
                errors.append(f"articulated_sample_{index}_position_nonfinite:{joint_id}")
            else:
                normalized_positions[joint_id] = position
            if velocity is None:
                errors.append(f"articulated_sample_{index}_velocity_nonfinite:{joint_id}")
            else:
                normalized_velocities[joint_id] = velocity
        boolean_fields = (
            "task_contact_active",
            "joint_limit_violation",
            "containment_violation",
            "robot_collision_failure",
            "scene_collision_failure",
            "retreat_completed",
        )
        booleans: dict[str, bool] = {}
        for field in boolean_fields:
            raw = sample.get(field)
            if not isinstance(raw, bool):
                errors.append(f"articulated_sample_{index}_{field}_invalid")
            else:
                booleans[field] = raw
        normalized.append(
            {
                "step_index": step,
                "joint_positions_rad": normalized_positions,
                "joint_velocities_rad_s": normalized_velocities,
                **booleans,
            }
        )
    if errors:
        raise TaskNeutralScoringError(errors)
    return normalized


def score_articulated_task_episode(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score an open/close episode only from native deterministic state."""

    spec = _normalize_articulated_spec(task_spec)
    resets = spec["joint_reset_positions_rad"]
    target = spec["target_joint_id"]
    normalized = _normalize_articulated_samples(samples, joint_ids=set(resets))
    reset_tolerance = float(spec["reset_tolerance_rad"])
    reset_errors = {
        joint_id: abs(normalized[0]["joint_positions_rad"][joint_id] - reset)
        for joint_id, reset in resets.items()
    }
    offending = {
        joint_id: value
        for joint_id, value in reset_errors.items()
        if value > reset_tolerance
    }
    if offending:
        # The deviations are already computed here; dropping them turns a
        # one-line fix into another launch. Name the joint and the amount.
        raise TaskNeutralScoringError(
            [
                "articulated_episode_reset_readback_mismatch:"
                + ",".join(
                    f"{joint_id}={value:.5g}rad"
                    for joint_id, value in sorted(offending.items())
                )
                + f":tolerance={reset_tolerance:.5g}rad"
            ]
        )

    lower, upper = spec["target_success_interval_rad"]
    target_positions = [sample["joint_positions_rad"][target] for sample in normalized]
    target_reset = resets[target]
    target_displacements = [abs(value - target_reset) for value in target_positions]
    maximum_displacement = max(target_displacements)
    reached_success_interval = any(lower <= value <= upper for value in target_positions)
    window_count = int(spec["settle_window_samples"])
    settle_available = len(normalized) >= window_count
    settle = normalized[-window_count:] if settle_available else normalized
    settle_target_positions = [sample["joint_positions_rad"][target] for sample in settle]
    settle_target_velocities = [sample["joint_velocities_rad_s"][target] for sample in settle]
    settle_in_interval = settle_available and all(
        lower <= value <= upper for value in settle_target_positions
    )
    settle_speed_ok = settle_available and all(
        abs(value) <= float(spec["maximum_settled_target_speed_rad_s"])
        for value in settle_target_velocities
    )
    non_target = sorted(set(resets) - {target})
    non_target_max_delta = {
        joint_id: max(
            abs(sample["joint_positions_rad"][joint_id] - resets[joint_id])
            for sample in normalized
        )
        for joint_id in non_target
    }
    non_task_locked = all(
        value <= float(spec["non_task_joint_motion_tolerance_rad"])
        for value in non_target_max_delta.values()
    )
    # The same tolerance the sample-level check carries. rt59 failed both
    # controls on a hinge reading of -8.8e-11 rad - numerical dust, not an
    # excursion - because this inline check compared against the limit
    # exactly while every other limit check in the lane tolerates noise.
    limit_tolerance = float(spec.get("reset_tolerance_rad") or 0.0)
    hard_limit_violation = any(
        sample["joint_limit_violation"]
        or any(
            not (
                spec["joint_hard_limits_rad"][joint_id][0] - limit_tolerance
                <= position
                <= spec["joint_hard_limits_rad"][joint_id][1] + limit_tolerance
            )
            for joint_id, position in sample["joint_positions_rad"].items()
        )
        for sample in normalized
    )
    containment_violation = any(sample["containment_violation"] for sample in normalized)
    collision_failure = any(
        sample["robot_collision_failure"] or sample["scene_collision_failure"]
        for sample in normalized
    )
    released_in_settle = settle_available and all(
        not sample["task_contact_active"] for sample in settle
    )
    retreat_completed = bool(normalized[-1]["retreat_completed"])
    task_succeeded = bool(
        settle_in_interval
        and settle_speed_ok
        and non_task_locked
        and not hard_limit_violation
        and not containment_violation
        and not collision_failure
        and released_in_settle
        and retreat_completed
    )
    if task_succeeded:
        outcome = OUTCOME_OPENED_AND_SETTLED
    elif hard_limit_violation or containment_violation:
        outcome = OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION
    elif collision_failure:
        outcome = OUTCOME_COLLISION_FAILURE
    elif not non_task_locked:
        outcome = OUTCOME_NON_TASK_JOINT_MOVED
    elif settle_in_interval and settle_speed_ok and (
        not released_in_settle or not retreat_completed
    ):
        # Reaching and stably holding the requested angle is materially farther
        # than a rebound, but the task contract still requires release and
        # retreat.  Keep that failure rung distinct so neither a policy nor a
        # human reviewer can promote an assisted/unfinished open to success.
        outcome = OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE
    elif reached_success_interval:
        outcome = OUTCOME_OPENED_THEN_REBOUNDED
    elif maximum_displacement > float(spec["movement_epsilon_rad"]):
        outcome = OUTCOME_MOVED_BELOW_THRESHOLD
    else:
        outcome = OUTCOME_NEVER_MOVED

    report: dict[str, Any] = {
        "schema_version": ARTICULATED_REPORT_SCHEMA_VERSION,
        "status": "scored" if settle_available else "undetermined",
        "task_kind": TASK_KIND_ARTICULATED_OPEN_CLOSE,
        "task_succeeded": task_succeeded,
        "outcome": outcome,
        "outcome_rank": _ARTICULATED_OUTCOME_RANK[outcome],
        "measurements": {
            "sample_count": len(normalized),
            "first_step_index": normalized[0]["step_index"],
            "final_step_index": normalized[-1]["step_index"],
            "reset_readback_error_rad": reset_errors,
            "target_start_position_rad": target_positions[0],
            "target_final_position_rad": target_positions[-1],
            "target_maximum_displacement_rad": maximum_displacement,
            "target_reached_success_interval": reached_success_interval,
            "settle_window_available": settle_available,
            "settle_target_min_position_rad": min(settle_target_positions),
            "settle_target_max_position_rad": max(settle_target_positions),
            "settle_target_max_abs_velocity_rad_s": max(
                abs(value) for value in settle_target_velocities
            ),
            "non_target_max_delta_rad": non_target_max_delta,
            "released_in_settle": released_in_settle,
            "retreat_completed": retreat_completed,
        },
        "predicates": {
            "settle_in_success_interval": settle_in_interval,
            "settle_speed_within_limit": settle_speed_ok,
            "non_task_joints_locked": non_task_locked,
            "joint_hard_limits_respected": not hard_limit_violation,
            "containment_respected": not containment_violation,
            "collision_failure_absent": not collision_failure,
            "task_contact_released": released_in_settle,
            "retreat_completed": retreat_completed,
        },
        "thresholds": {
            "target_success_interval_rad": [lower, upper],
            "maximum_settled_target_speed_rad_s": spec[
                "maximum_settled_target_speed_rad_s"
            ],
            "non_task_joint_motion_tolerance_rad": spec[
                "non_task_joint_motion_tolerance_rad"
            ],
            "movement_epsilon_rad": spec["movement_epsilon_rad"],
            "reset_tolerance_rad": reset_tolerance,
            "settle_window_samples": window_count,
            "joint_hard_limits_rad": spec["joint_hard_limits_rad"],
        },
        "judgement_source": "deterministic_native_simulator_joint_state",
        "rendered_image_consulted": False,
        "learned_judge_consulted": False,
        "candidate_policy_queried_by_scorer": False,
        "caller_asserted_success_accepted": False,
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def score_task_episode_from_spec(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Dispatch a frozen task spec without scene- or object-name conditionals."""

    kind = task_spec.get("task_kind")
    if kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        return score_articulated_task_episode(task_spec=task_spec, samples=samples)
    if kind == TASK_KIND_RIGID_PICK_PLACE:
        if task_spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION:
            raise TaskNeutralScoringError(["task_spec_schema_invalid"])
        try:
            return score_task_episode(
                samples=samples,
                destination_position_world_m=task_spec["destination_position_world_m"],
                support_plane_z_m=float(task_spec["support_plane_z_m"]),
                settle_window_samples=int(task_spec["settle_window_samples"]),
                require_sealed_start_pose=bool(task_spec.get("require_sealed_start_pose", True)),
            )
        except (KeyError, TypeError, ValueError, TaskScoringError) as exc:
            if isinstance(exc, TaskScoringError):
                raise TaskNeutralScoringError(exc.errors) from exc
            raise TaskNeutralScoringError(["rigid_task_spec_invalid"]) from exc
    raise TaskNeutralScoringError(["task_kind_unsupported"])


__all__ = [
    "ARTICULATED_REPORT_SCHEMA_VERSION",
    "OUTCOME_COLLISION_FAILURE",
    "OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION",
    "OUTCOME_MOVED_BELOW_THRESHOLD",
    "OUTCOME_NEVER_MOVED",
    "OUTCOME_NON_TASK_JOINT_MOVED",
    "OUTCOME_OPENED_AND_SETTLED",
    "OUTCOME_OPENED_THEN_REBOUNDED",
    "OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE",
    "TASK_KIND_ARTICULATED_OPEN_CLOSE",
    "TASK_KIND_RIGID_PICK_PLACE",
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskNeutralScoringError",
    "score_articulated_task_episode",
    "score_task_episode_from_spec",
    "validate_articulated_task_spec",
]
