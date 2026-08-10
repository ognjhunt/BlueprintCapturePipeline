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
try:  # flat provider-bundle layout
    from articulation_graph_contract import (
        ArticulationGraphContractError,
        validate_articulation_graph,
    )
except ModuleNotFoundError:  # repository package
    from .articulation_graph_contract import (
        ArticulationGraphContractError,
        validate_articulation_graph,
    )


TASK_SPEC_SCHEMA_VERSION = "adp_task_spec.v1"
TASK_SPEC_GRAPH_SCHEMA_VERSION = "adp_task_spec.v2"
ARTICULATED_REPORT_SCHEMA_VERSION = "adp_articulated_task_scoring.v1"
RIGID_REPORT_SCHEMA_VERSION = "adp_rigid_task_scoring.v2"
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


def _normalize_legacy_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
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
    if not normalized_resets:
        errors.append("articulated_joint_resets_invalid")
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
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "target_joint_id": target,
        "target_joint_ids": [target],
        "joint_reset_positions_rad": normalized_resets,
        "joint_reset_positions": normalized_resets,
        "target_success_interval_rad": normalized_interval,
        "target_success_intervals": {target: normalized_interval},
        "joint_hard_limits_rad": normalized_limits,
        "joint_hard_limits": normalized_limits,
        "joint_roles": {
            joint_id: "target" if joint_id == target else "locked"
            for joint_id in normalized_resets
        },
        "dependent_joints": {},
        **normalized_fields,
    }


def _normalize_graph_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != TASK_SPEC_GRAPH_SCHEMA_VERSION:
        errors.append("task_spec_schema_invalid")
    if spec.get("task_kind") != TASK_KIND_ARTICULATED_OPEN_CLOSE:
        errors.append("articulated_task_kind_invalid")
    graph = spec.get("articulation_graph")
    if not isinstance(graph, Mapping):
        errors.append("articulated_graph_missing")
        normalized_graph: dict[str, Any] = {}
    else:
        try:
            normalized_graph = validate_articulation_graph(graph)
        except ArticulationGraphContractError as exc:
            errors.extend(exc.errors)
            normalized_graph = {}
    fields = {
        "settle_window_samples": spec.get("settle_window_samples"),
        "maximum_settled_target_speed": spec.get(
            "maximum_settled_target_speed"
        ),
        "locked_joint_motion_tolerance": spec.get(
            "locked_joint_motion_tolerance"
        ),
        "movement_epsilon": spec.get("movement_epsilon"),
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
            normalized_fields[field] = (
                int(raw) if field == "settle_window_samples" else value
            )
    joints = normalized_graph.get("joints") or []
    resets = {str(row["joint_id"]): float(row["reset_position"]) for row in joints}
    limits = {str(row["joint_id"]): list(row["limits"]) for row in joints}
    roles = {str(row["joint_id"]): str(row["role"]) for row in joints}
    reset_tolerances = {
        str(row["joint_id"]): float(row["reset_tolerance"]) for row in joints
    }
    dependent = {
        str(row["joint_id"]): dict(row["dependency"])
        for row in joints
        if row["role"] == "dependent"
    }
    success = (
        normalized_graph.get("success_predicate", {}).get("joint_intervals") or {}
    )
    targets = sorted(success)
    if errors:
        raise TaskNeutralScoringError(errors)
    return {
        "schema_version": TASK_SPEC_GRAPH_SCHEMA_VERSION,
        "articulation_graph": normalized_graph,
        "target_joint_id": targets[0],
        "target_joint_ids": targets,
        "joint_reset_positions": resets,
        "joint_reset_positions_rad": resets,
        "joint_reset_tolerances": reset_tolerances,
        "reset_tolerance_rad": max(reset_tolerances.values()),
        "target_success_intervals": {
            str(joint_id): list(interval) for joint_id, interval in success.items()
        },
        "target_success_interval_rad": list(success[targets[0]]),
        "joint_hard_limits": limits,
        "joint_hard_limits_rad": limits,
        "joint_roles": roles,
        "dependent_joints": dependent,
        "maximum_settled_target_speed_rad_s": normalized_fields[
            "maximum_settled_target_speed"
        ],
        "non_task_joint_motion_tolerance_rad": normalized_fields[
            "locked_joint_motion_tolerance"
        ],
        "movement_epsilon_rad": normalized_fields["movement_epsilon"],
        **normalized_fields,
    }


def _normalize_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") == TASK_SPEC_GRAPH_SCHEMA_VERSION:
        return _normalize_graph_articulated_spec(spec)
    return _normalize_legacy_articulated_spec(spec)


def validate_articulated_task_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Public fail-closed validator for a frozen articulated scorer contract."""

    return _normalize_articulated_spec(spec)


def _normalize_articulated_samples(
    samples: Sequence[Mapping[str, Any]], *, joint_ids: set[str], generic_units: bool = False
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
        position_field = "joint_positions" if generic_units else "joint_positions_rad"
        velocity_field = (
            "joint_velocities_per_s" if generic_units else "joint_velocities_rad_s"
        )
        positions = sample.get(position_field)
        velocities = sample.get(velocity_field)
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
    targets = list(spec["target_joint_ids"])
    normalized = _normalize_articulated_samples(
        samples,
        joint_ids=set(resets),
        generic_units=spec["schema_version"] == TASK_SPEC_GRAPH_SCHEMA_VERSION,
    )
    reset_tolerance = float(spec["reset_tolerance_rad"])
    reset_tolerances = spec.get("joint_reset_tolerances") or {
        joint_id: reset_tolerance for joint_id in resets
    }
    reset_errors = {
        joint_id: abs(normalized[0]["joint_positions_rad"][joint_id] - reset)
        for joint_id, reset in resets.items()
    }
    if any(
        value > float(reset_tolerances[joint_id])
        for joint_id, value in reset_errors.items()
    ):
        raise TaskNeutralScoringError(["articulated_episode_reset_readback_mismatch"])

    success_intervals = spec["target_success_intervals"]
    lower, upper = success_intervals[target]
    target_positions_by_joint = {
        joint_id: [
            sample["joint_positions_rad"][joint_id] for sample in normalized
        ]
        for joint_id in targets
    }
    target_positions = target_positions_by_joint[target]
    target_displacements_by_joint = {
        joint_id: [
            abs(value - resets[joint_id])
            for value in target_positions_by_joint[joint_id]
        ]
        for joint_id in targets
    }
    maximum_displacement_by_joint = {
        joint_id: max(values)
        for joint_id, values in target_displacements_by_joint.items()
    }
    maximum_displacement = max(maximum_displacement_by_joint.values())
    reached_success_interval = any(
        all(
            success_intervals[joint_id][0]
            <= sample["joint_positions_rad"][joint_id]
            <= success_intervals[joint_id][1]
            for joint_id in targets
        )
        for sample in normalized
    )
    window_count = int(spec["settle_window_samples"])
    settle_available = len(normalized) >= window_count
    settle = normalized[-window_count:] if settle_available else normalized
    settle_target_positions = [sample["joint_positions_rad"][target] for sample in settle]
    settle_target_velocities = [sample["joint_velocities_rad_s"][target] for sample in settle]
    settle_in_interval = settle_available and all(
        all(
            success_intervals[joint_id][0]
            <= sample["joint_positions_rad"][joint_id]
            <= success_intervals[joint_id][1]
            for joint_id in targets
        )
        for sample in settle
    )
    settle_speed_ok = settle_available and all(
        abs(sample["joint_velocities_rad_s"][joint_id])
        <= float(spec["maximum_settled_target_speed_rad_s"])
        for sample in settle
        for joint_id in targets
    )
    roles = spec["joint_roles"]
    locked_joint_ids = sorted(
        joint_id for joint_id, role in roles.items() if role == "locked"
    )
    locked_max_delta = {
        joint_id: max(
            abs(sample["joint_positions_rad"][joint_id] - resets[joint_id])
            for sample in normalized
        )
        for joint_id in locked_joint_ids
    }
    locked_joints_stable = all(
        value <= float(spec["non_task_joint_motion_tolerance_rad"])
        for value in locked_max_delta.values()
    )
    dependent_max_error: dict[str, float] = {}
    for joint_id, dependency in spec["dependent_joints"].items():
        driver = dependency["driver_joint_id"]
        multiplier = float(dependency["multiplier"])
        offset = float(dependency["offset"])
        dependent_max_error[joint_id] = max(
            abs(
                sample["joint_positions_rad"][joint_id]
                - (
                    multiplier * sample["joint_positions_rad"][driver]
                    + offset
                )
            )
            for sample in normalized
        )
    dependent_joints_consistent = all(
        dependent_max_error[joint_id]
        <= float(spec["dependent_joints"][joint_id]["tolerance"])
        for joint_id in dependent_max_error
    )
    non_task_locked = locked_joints_stable and dependent_joints_consistent
    hard_limit_violation = any(
        sample["joint_limit_violation"]
        or any(
            not (spec["joint_hard_limits_rad"][joint_id][0] <= position <= spec["joint_hard_limits_rad"][joint_id][1])
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
            "target_positions_by_joint": {
                joint_id: {
                    "start": values[0],
                    "final": values[-1],
                    "maximum_displacement": maximum_displacement_by_joint[joint_id],
                }
                for joint_id, values in target_positions_by_joint.items()
            },
            "target_reached_success_interval": reached_success_interval,
            "settle_window_available": settle_available,
            "settle_target_min_position_rad": min(settle_target_positions),
            "settle_target_max_position_rad": max(settle_target_positions),
            "settle_target_max_abs_velocity_rad_s": max(
                abs(value) for value in settle_target_velocities
            ),
            "non_target_max_delta_rad": locked_max_delta,
            "locked_joint_max_delta": locked_max_delta,
            "dependent_joint_max_error": dependent_max_error,
            "released_in_settle": released_in_settle,
            "retreat_completed": retreat_completed,
        },
        "predicates": {
            "settle_in_success_interval": settle_in_interval,
            "settle_speed_within_limit": settle_speed_ok,
            "non_task_joints_locked": non_task_locked,
            "locked_joints_stable": locked_joints_stable,
            "dependent_joints_consistent": dependent_joints_consistent,
            "joint_hard_limits_respected": not hard_limit_violation,
            "containment_respected": not containment_violation,
            "collision_failure_absent": not collision_failure,
            "task_contact_released": released_in_settle,
            "retreat_completed": retreat_completed,
        },
        "thresholds": {
            "target_success_interval_rad": [lower, upper],
            "target_success_intervals": success_intervals,
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


def _vector(value: Any, length: int, *, error: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskNeutralScoringError([error])
    result = [_finite(item) for item in value]
    if any(item is None for item in result):
        raise TaskNeutralScoringError([error])
    return [float(item) for item in result]


def _quaternion_angle(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _vector(a, 4, error="rigid_task_quaternion_invalid")
    qb = _vector(b, 4, error="rigid_task_quaternion_invalid")
    na = math.sqrt(sum(item * item for item in qa))
    nb = math.sqrt(sum(item * item for item in qb))
    if min(na, nb) <= 0.0:
        raise TaskNeutralScoringError(["rigid_task_quaternion_invalid"])
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)) / (na * nb))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _normalize_rigid_task_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if (
        spec.get("schema_version") != TASK_SPEC_GRAPH_SCHEMA_VERSION
        or spec.get("task_kind") != TASK_KIND_RIGID_PICK_PLACE
    ):
        errors.append("rigid_task_spec_schema_invalid")
    subject = str(spec.get("subject_asset_id") or "")
    if not subject:
        errors.append("rigid_task_subject_asset_id_missing")
    try:
        start_pose = _vector(
            spec.get("start_pose_world"), 7, error="rigid_task_start_pose_invalid"
        )
        orientation_reference = _vector(
            spec.get("destination_orientation_xyzw"),
            4,
            error="rigid_task_destination_orientation_invalid",
        )
        raw_bounds = spec["destination_position_bounds_world_m"]
        lower = _vector(raw_bounds["minimum"], 3, error="rigid_task_destination_invalid")
        upper = _vector(raw_bounds["maximum"], 3, error="rigid_task_destination_invalid")
        support_interval = _vector(
            spec["support_height_interval_m"],
            2,
            error="rigid_task_support_interval_invalid",
        )
    except (KeyError, TypeError, TaskNeutralScoringError) as exc:
        if isinstance(exc, TaskNeutralScoringError):
            errors.extend(exc.errors)
        else:
            errors.append("rigid_task_spec_invalid")
        start_pose = [0.0] * 7
        orientation_reference = [0.0, 0.0, 0.0, 1.0]
        lower = [0.0] * 3
        upper = [0.0] * 3
        support_interval = [0.0, 0.0]
    numeric_fields = (
        "destination_orientation_tolerance_rad",
        "minimum_translation_m",
        "minimum_lift_m",
        "movement_epsilon_m",
        "reset_translation_tolerance_m",
        "reset_orientation_tolerance_rad",
        "settle_position_tolerance_m",
        "settle_orientation_tolerance_rad",
        "release_gripper_width_min_m",
    )
    numbers: dict[str, float] = {}
    for field in numeric_fields:
        value = _finite(spec.get(field))
        if value is None or value < 0.0 or (
            value == 0.0 and field not in {"minimum_lift_m"}
        ):
            errors.append(f"rigid_task_{field}_invalid")
        else:
            numbers[field] = value
    settle = spec.get("settle_window_samples")
    if isinstance(settle, bool) or not isinstance(settle, int) or settle <= 0:
        errors.append("rigid_task_settle_window_samples_invalid")
    if spec.get("release_required") is not True:
        errors.append("rigid_task_release_contract_invalid")
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        errors.append("rigid_task_destination_invalid")
    if support_interval[0] >= support_interval[1]:
        errors.append("rigid_task_support_interval_invalid")
    for quaternion, error in (
        (start_pose[3:], "rigid_task_start_pose_invalid"),
        (orientation_reference, "rigid_task_destination_orientation_invalid"),
    ):
        if abs(sum(item * item for item in quaternion) - 1.0) > 1.0e-6:
            errors.append(error)
    if errors:
        raise TaskNeutralScoringError(errors)
    return {
        "subject_asset_id": subject,
        "start_pose_world": start_pose,
        "destination_position_bounds_world_m": {"minimum": lower, "maximum": upper},
        "destination_orientation_xyzw": orientation_reference,
        "support_height_interval_m": support_interval,
        "settle_window_samples": settle,
        "release_required": True,
        **numbers,
    }


def score_rigid_task_episode(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score a scene-neutral rigid relocation from deterministic native state."""

    spec = _normalize_rigid_task_spec(task_spec)
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        raise TaskNeutralScoringError(["rigid_task_samples_invalid"])
    normalized: list[dict[str, Any]] = []
    previous_step: int | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TaskNeutralScoringError([f"rigid_task_sample_invalid:{index}"])
        step = sample.get("step_index")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or (previous_step is not None and step <= previous_step)
        ):
            raise TaskNeutralScoringError([f"rigid_task_sample_step_invalid:{index}"])
        previous_step = step
        pose = _vector(
            sample.get("task_object_pose_world"),
            7,
            error=f"rigid_task_sample_pose_invalid:{index}",
        )
        if abs(sum(item * item for item in pose[3:]) - 1.0) > 1.0e-3:
            raise TaskNeutralScoringError([f"rigid_task_sample_pose_invalid:{index}"])
        normalized.append(
            {
                "step_index": step,
                "pose": pose,
                "gripper_width_m": _finite(sample.get("gripper_width_m")),
                "task_contact_active": sample.get("task_contact_active"),
                "robot_collision_failure": sample.get("robot_collision_failure"),
                "scene_collision_failure": sample.get("scene_collision_failure"),
                "containment_violation": sample.get("containment_violation"),
            }
        )

    start = normalized[0]["pose"]
    reset_translation_error = math.dist(start[:3], spec["start_pose_world"][:3])
    reset_orientation_error = _quaternion_angle(
        start[3:], spec["start_pose_world"][3:]
    )
    if (
        reset_translation_error > spec["reset_translation_tolerance_m"]
        or reset_orientation_error > spec["reset_orientation_tolerance_rad"]
    ):
        raise TaskNeutralScoringError(["rigid_task_reset_readback_mismatch"])
    positions = [row["pose"][:3] for row in normalized]
    translation = [math.dist(position[:2], start[:2]) for position in positions]
    lift = [position[2] - start[2] for position in positions]
    settle = normalized[-spec["settle_window_samples"] :]
    settle_available = len(normalized) >= spec["settle_window_samples"]
    lower = spec["destination_position_bounds_world_m"]["minimum"]
    upper = spec["destination_position_bounds_world_m"]["maximum"]
    destination_inside = settle_available and all(
        all(low <= value <= high for low, value, high in zip(lower, row["pose"][:3], upper, strict=True))
        for row in settle
    )
    orientation_errors = [
        _quaternion_angle(row["pose"][3:], spec["destination_orientation_xyzw"])
        for row in settle
    ]
    orientation_ok = settle_available and all(
        error <= spec["destination_orientation_tolerance_rad"]
        for error in orientation_errors
    )
    support_ok = settle_available and all(
        spec["support_height_interval_m"][0]
        <= row["pose"][2]
        <= spec["support_height_interval_m"][1]
        for row in settle
    )
    anchor = settle[-1]["pose"] if settle else start
    settled = settle_available and all(
        math.dist(row["pose"][:3], anchor[:3])
        <= spec["settle_position_tolerance_m"]
        and _quaternion_angle(row["pose"][3:], anchor[3:])
        <= spec["settle_orientation_tolerance_rad"]
        for row in settle
    )
    released = settle_available and all(
        row["gripper_width_m"] is not None
        and row["gripper_width_m"] >= spec["release_gripper_width_min_m"]
        and row["task_contact_active"] is False
        for row in settle
    )
    safety_fields = (
        "robot_collision_failure",
        "scene_collision_failure",
        "containment_violation",
    )
    safety_complete = all(
        isinstance(row[field], bool) for row in normalized for field in safety_fields
    )
    safety_ok = safety_complete and not any(
        row[field] for row in normalized for field in safety_fields
    )
    moved = max(translation) > spec["movement_epsilon_m"]
    translated = max(translation) >= spec["minimum_translation_m"]
    lifted = max(lift) >= spec["minimum_lift_m"]
    succeeded = (
        destination_inside
        and orientation_ok
        and support_ok
        and settled
        and released
        and safety_ok
        and translated
        and lifted
    )
    if not safety_complete:
        status = "undetermined"
        outcome = "native_safety_readback_missing"
    elif not safety_ok:
        status = "scored"
        outcome = "collision_or_containment_failure"
    elif succeeded:
        status = "scored"
        outcome = "placed_and_settled"
    elif not moved:
        status = "scored"
        outcome = OUTCOME_NEVER_MOVED
    elif destination_inside and not released:
        status = "scored"
        outcome = "release_incomplete"
    else:
        status = "scored" if settle_available else "undetermined"
        outcome = "moved_below_success_contract"
    report: dict[str, Any] = {
        "schema_version": RIGID_REPORT_SCHEMA_VERSION,
        "status": status,
        "task_kind": TASK_KIND_RIGID_PICK_PLACE,
        "subject_asset_id": spec["subject_asset_id"],
        "task_succeeded": succeeded,
        "outcome": outcome,
        "measurements": {
            "sample_count": len(normalized),
            "reset_translation_error_m": reset_translation_error,
            "reset_orientation_error_rad": reset_orientation_error,
            "maximum_translation_m": max(translation),
            "maximum_lift_m": max(lift),
            "settle_window_available": settle_available,
            "settle_destination_inside": destination_inside,
            "settle_orientation_ok": orientation_ok,
            "settle_support_height_ok": support_ok,
            "settled": settled,
            "released": released,
            "native_safety_readback_complete": safety_complete,
            "native_safety_ok": safety_ok,
        },
        "learned_judge_consulted": False,
        "candidate_policy_queried_by_scorer": False,
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
        if task_spec.get("schema_version") == TASK_SPEC_GRAPH_SCHEMA_VERSION:
            return score_rigid_task_episode(task_spec=task_spec, samples=samples)
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
    "RIGID_REPORT_SCHEMA_VERSION",
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
    "TASK_SPEC_GRAPH_SCHEMA_VERSION",
    "TaskNeutralScoringError",
    "score_articulated_task_episode",
    "score_rigid_task_episode",
    "score_task_episode_from_spec",
    "validate_articulated_task_spec",
]
