"""Deterministic scoring for rigid pick/place and planar-push episodes.

This module is split from :mod:`adp_task_scoring` so the articulated and rigid
scorers remain independently reviewable while preserving the public API.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .adp_task_scoring import (
    OUTCOME_NEVER_MOVED,
    OUTCOME_PUSHED_AND_SETTLED,
    RIGID_MANIPULATION_STRATEGIES,
    RIGID_REPORT_SCHEMA_VERSION,
    TASK_KIND_RIGID_PICK_PLACE,
    TASK_SPEC_GRAPH_SCHEMA_VERSION,
    TaskNeutralScoringError,
    _compatibility_rigid_success_criteria,
    _default_rigid_task_success_contract,
    _finite,
    _vector,
    validate_rigid_task_success_contract,
)
from .decision_evidence_contracts import canonical_digest
from .adp_rigid_retreat_scoring import score_retreat, validate_retreat_binding

def _quaternion_angle(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _vector(a, 4, error="rigid_task_quaternion_invalid")
    qb = _vector(b, 4, error="rigid_task_quaternion_invalid")
    na = math.sqrt(sum(item * item for item in qa))
    nb = math.sqrt(sum(item * item for item in qb))
    if min(na, nb) <= 0.0:
        raise TaskNeutralScoringError(["rigid_task_quaternion_invalid"])
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)) / (na * nb))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _rotate_xyzw(
    vector: Sequence[float], quaternion: Sequence[float]
) -> list[float]:
    x, y, z, w = quaternion
    qx, qy, qz, qw = (x, y, z, w)
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _rotate_inverse_xyzw(
    vector: Sequence[float], quaternion: Sequence[float]
) -> list[float]:
    return _rotate_xyzw(
        vector,
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
    )


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
    manipulation_strategy = str(
        spec.get("manipulation_strategy") or "pick_and_place"
    )
    if manipulation_strategy not in RIGID_MANIPULATION_STRATEGIES:
        errors.append("rigid_task_manipulation_strategy_invalid")
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
        "task_contact_minimum_force_n",
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
    normalized: dict[str, Any] = {
        "subject_asset_id": subject,
        "manipulation_strategy": manipulation_strategy,
        "start_pose_world": start_pose,
        "destination_position_bounds_world_m": {"minimum": lower, "maximum": upper},
        "destination_orientation_xyzw": orientation_reference,
        "support_height_interval_m": support_interval,
        "settle_window_samples": settle,
        "release_required": True,
        **numbers,
    }
    destination_relation = spec.get("destination_relation")
    if destination_relation is not None:
        if destination_relation not in {"inside", "on"}:
            raise TaskNeutralScoringError(
                ["rigid_task_destination_relation_invalid"]
            )
        destination_pose = _vector(
            spec.get("destination_pose_world"),
            7,
            error="rigid_task_destination_pose_invalid",
        )
        local_bounds = spec.get(
            "destination_position_bounds_destination_frame_m"
        )
        if not isinstance(local_bounds, Mapping):
            raise TaskNeutralScoringError(
                ["rigid_task_destination_local_bounds_invalid"]
            )
        local_lower = _vector(
            local_bounds.get("minimum"),
            3,
            error="rigid_task_destination_local_bounds_invalid",
        )
        local_upper = _vector(
            local_bounds.get("maximum"),
            3,
            error="rigid_task_destination_local_bounds_invalid",
        )
        subject_bounds = spec.get("subject_collision_bounds_scoring_frame_m")
        interior_bounds = spec.get("destination_interior_bounds_body_frame_m")
        if not isinstance(subject_bounds, Mapping) or not isinstance(
            interior_bounds, Mapping
        ):
            raise TaskNeutralScoringError(
                ["rigid_task_destination_collision_geometry_invalid"]
            )
        subject_lower = _vector(
            subject_bounds.get("minimum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        subject_upper = _vector(
            subject_bounds.get("maximum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        interior_lower = _vector(
            interior_bounds.get("minimum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        interior_upper = _vector(
            interior_bounds.get("maximum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        translation_tolerance = _finite(
            spec.get("destination_reset_translation_tolerance_m")
        )
        rotation_tolerance = _finite(
            spec.get("destination_reset_rotation_tolerance_rad")
        )
        if (
            abs(sum(item * item for item in destination_pose[3:]) - 1.0)
            > 1.0e-6
            or any(
                low >= high
                for low, high in zip(local_lower, local_upper, strict=True)
            )
            or any(
                low >= high
                for low, high in zip(subject_lower, subject_upper, strict=True)
            )
            or any(
                low >= high
                for low, high in zip(interior_lower, interior_upper, strict=True)
            )
            or translation_tolerance is None
            or translation_tolerance <= 0.0
            or rotation_tolerance is None
            or rotation_tolerance <= 0.0
        ):
            raise TaskNeutralScoringError(
                ["rigid_task_destination_local_contract_invalid"]
            )
        normalized.update(
            destination_relation=destination_relation,
            destination_pose_world=destination_pose,
            destination_position_bounds_destination_frame_m={
                "minimum": local_lower,
                "maximum": local_upper,
            },
            subject_collision_bounds_scoring_frame_m={
                "minimum": subject_lower,
                "maximum": subject_upper,
            },
            destination_interior_bounds_body_frame_m={
                "minimum": interior_lower,
                "maximum": interior_upper,
            },
            destination_reset_translation_tolerance_m=translation_tolerance,
            destination_reset_rotation_tolerance_rad=rotation_tolerance,
        )
    control_frequency_hz = _finite(spec.get("control_frequency_hz"))
    normalized["control_frequency_hz"] = (
        control_frequency_hz
        if control_frequency_hz is not None and control_frequency_hz > 0.0
        else None
    )
    raw_success_contract = spec.get("task_success_contract")
    if raw_success_contract is None:
        normalized["task_success_contract"] = _default_rigid_task_success_contract(
            normalized
        )
    else:
        validated_success_contract = validate_rigid_task_success_contract(
            raw_success_contract,
            expected_site_id=(
                str(spec["site_id"]) if spec.get("site_id") is not None else None
            ),
            expected_task_id=(
                str(spec["task_id"]) if spec.get("task_id") is not None else None
            ),
        )
        if (
            validated_success_contract["provenance"]["author_source"]
            == "compatibility_default"
            and validated_success_contract["criteria"]
            != _compatibility_rigid_success_criteria(normalized)
        ):
            raise TaskNeutralScoringError(
                ["rigid_task_success_contract_default_criteria_mismatch"]
            )
        if "retreat" in validated_success_contract["criteria"]:
            retreat_errors = validate_retreat_binding(spec, validated_success_contract)
            if retreat_errors:
                raise TaskNeutralScoringError(retreat_errors)
        normalized["task_success_contract"] = validated_success_contract
    return normalized


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
        destination_pose = None
        if spec.get("destination_relation") is not None:
            raw_destination_pose = sample.get("destination_pose_world")
            if raw_destination_pose is not None:
                destination_pose = _vector(
                    raw_destination_pose,
                    7,
                    error=f"rigid_task_sample_destination_pose_invalid:{index}",
                )
                if (
                    abs(sum(item * item for item in destination_pose[3:]) - 1.0)
                    > 1.0e-3
                ):
                    raise TaskNeutralScoringError(
                        [f"rigid_task_sample_destination_pose_invalid:{index}"]
                    )
        native_readback = sample.get("native_readback")
        if not isinstance(native_readback, Mapping):
            native_readback = {}
        task_contact_force_n = _finite(
            sample.get("task_robot_contact_peak_force_n")
        )
        task_contact_force_source = (
            "task_robot_contact_peak_force_n"
            if task_contact_force_n is not None
            else None
        )
        if task_contact_force_n is None:
            task_contact_force_n = _finite(sample.get("task_contact_force_n"))
            if task_contact_force_n is not None:
                task_contact_force_source = "task_contact_force_n"
        if task_contact_force_n is None:
            task_contact_force_n = _finite(
                native_readback.get("task_robot_contact_peak_force_n")
            )
            if task_contact_force_n is not None:
                task_contact_force_source = (
                    "native_readback.task_robot_contact_peak_force_n"
                )
        if task_contact_force_n is not None and task_contact_force_n < 0.0:
            task_contact_force_n = None
            task_contact_force_source = None
        normalized.append(
            {
                "step_index": step,
                "pose": pose,
                "destination_pose": destination_pose,
                "gripper_width_m": _finite(sample.get("gripper_width_m")),
                "task_contact_active": sample.get("task_contact_active"),
                "support_contact_active": sample.get("support_contact_active"),
                "robot_collision_failure": sample.get("robot_collision_failure"),
                "scene_collision_failure": sample.get("scene_collision_failure"),
                "containment_violation": sample.get("containment_violation"),
                "forbidden_robot_task_collision_failure": sample.get(
                    "forbidden_robot_task_collision_failure"
                ),
                "locked_joint_containment_violation": sample.get(
                    "locked_joint_containment_violation"
                ),
                "robot_task_forbidden_collision_peak_force_n": _finite(
                    sample.get("robot_task_forbidden_collision_peak_force_n")
                    if sample.get("robot_task_forbidden_collision_peak_force_n")
                    is not None
                    else native_readback.get(
                        "robot_task_forbidden_collision_peak_force_n"
                    )
                ),
                "robot_scene_contact_peak_force_n": _finite(
                    sample.get("robot_scene_contact_peak_force_n")
                    if sample.get("robot_scene_contact_peak_force_n") is not None
                    else native_readback.get("robot_scene_contact_peak_force_n")
                ),
                "task_scene_collision_peak_force_n": _finite(
                    sample.get("task_scene_collision_peak_force_n")
                    if sample.get("task_scene_collision_peak_force_n") is not None
                    else native_readback.get("task_scene_collision_peak_force_n")
                ),
                "collision_failure_minimum_force_n": _finite(
                    sample.get("collision_failure_minimum_force_n")
                ),
                "robot_task_forbidden_contact_pairs": sample.get(
                    "robot_task_forbidden_contact_pairs"
                ),
                "workspace_excursion": sample.get("workspace_excursion"),
                "task_contact_force_n": task_contact_force_n,
                "task_contact_force_source": task_contact_force_source,
                "contact_classes_active": sample.get("contact_classes_active"),
                "retry_count": sample.get("retry_count"),
                "regrasp_count": sample.get("regrasp_count"),
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
    success_contract = spec["task_success_contract"]
    criteria = success_contract["criteria"]
    settling_criteria = criteria["settling"]
    settling_required = settling_criteria["mode"] == "required"
    settle_window_samples = int(settling_criteria["window_samples"])
    settle_available = (
        len(normalized) >= settle_window_samples if settling_required else True
    )
    settle = (
        normalized[-settle_window_samples:]
        if settling_required
        else normalized[-1:]
    )
    destination_criteria = criteria["destination_containment"]
    destination_pose_readback_complete = (
        spec.get("destination_relation") is None
        or all(row["destination_pose"] is not None for row in normalized)
    )
    destination_pose_stable = destination_pose_readback_complete
    if spec.get("destination_relation") is not None:
        expected_destination_pose = spec["destination_pose_world"]
        destination_pose_stable = destination_pose_readback_complete and all(
            math.dist(row["destination_pose"][:3], expected_destination_pose[:3])
            <= spec["destination_reset_translation_tolerance_m"]
            and _quaternion_angle(
                row["destination_pose"][3:], expected_destination_pose[3:]
            )
            <= spec["destination_reset_rotation_tolerance_rad"]
            for row in normalized
        )
        subject_bounds = spec["subject_collision_bounds_scoring_frame_m"]
        subject_lower = subject_bounds["minimum"]
        subject_upper = subject_bounds["maximum"]
        interior_bounds = spec["destination_interior_bounds_body_frame_m"]
        lower = interior_bounds["minimum"]
        upper = interior_bounds["maximum"]

        def subject_inside_destination(row: Mapping[str, Any]) -> bool:
            destination_pose = row["destination_pose"]
            subject_corners_world = [
                [
                    row["pose"][axis] + rotated[axis]
                    for axis in range(3)
                ]
                for rotated in (
                    _rotate_xyzw([x, y, z], row["pose"][3:])
                    for x in (subject_lower[0], subject_upper[0])
                    for y in (subject_lower[1], subject_upper[1])
                    for z in (subject_lower[2], subject_upper[2])
                )
            ]
            corners_destination = [
                _rotate_inverse_xyzw(
                    [
                        point[axis] - destination_pose[axis]
                        for axis in range(3)
                    ],
                    destination_pose[3:],
                )
                for point in subject_corners_world
            ]
            return all(
                low <= value <= high
                for point in corners_destination
                for low, value, high in zip(lower, point, upper, strict=True)
            )

        destination_inside = (
            settle_available
            and destination_pose_stable
            and all(subject_inside_destination(row) for row in settle)
        )
    else:
        lower = destination_criteria["position_bounds_world_m"]["minimum"]
        upper = destination_criteria["position_bounds_world_m"]["maximum"]
        destination_inside = settle_available and all(
            all(
                low <= value <= high
                for low, value, high in zip(
                    lower, row["pose"][:3], upper, strict=True
                )
            )
            for row in settle
        )

        def subject_inside_destination(row: Mapping[str, Any]) -> bool:
            return all(
                low <= value <= high
                for low, value, high in zip(
                    lower, row["pose"][:3], upper, strict=True
                )
            )
    orientation_criteria = criteria["orientation"]
    orientation_errors = [
        _quaternion_angle(row["pose"][3:], orientation_criteria["reference_xyzw"])
        for row in settle
    ]
    orientation_ok = settle_available and all(
        error <= orientation_criteria["tolerance_rad"]
        for error in orientation_errors
    )
    support_criteria = criteria["support"]
    support_ok = settle_available and all(
        support_criteria["height_interval_m"][0]
        <= row["pose"][2]
        <= support_criteria["height_interval_m"][1]
        for row in settle
    )
    support_contact_complete = settle_available and all(
        isinstance(row["support_contact_active"], bool) for row in settle
    )
    support_contact_ok = support_contact_complete and all(
        row["support_contact_active"] is True for row in settle
    )
    anchor = settle[-1]["pose"] if settle else start
    settled = settle_available and all(
        math.dist(row["pose"][:3], anchor[:3])
        <= settling_criteria["position_tolerance_m"]
        and _quaternion_angle(row["pose"][3:], anchor[3:])
        <= settling_criteria["orientation_tolerance_rad"]
        for row in settle
    )
    gripper_criteria = criteria["gripper_state"]
    gripper_threshold = gripper_criteria["threshold_m"]
    released = settle_available and all(
        row["gripper_width_m"] is not None
        and row["gripper_width_m"] >= gripper_threshold
        and row["task_contact_active"] is False
        for row in settle
    ) if gripper_criteria["mode"] == "released" else False
    gripper_closed = settle_available and all(
        row["gripper_width_m"] is not None
        and row["gripper_width_m"] <= gripper_threshold
        for row in settle
    ) if gripper_criteria["mode"] == "closed_at_most" else False
    task_contact_complete = settle_available and all(
        isinstance(row["task_contact_active"], bool) for row in settle
    )
    task_contact_cleared = task_contact_complete and all(
        row["task_contact_active"] is False for row in settle
    )
    task_contact_maintained = task_contact_complete and all(
        row["task_contact_active"] is True for row in settle
    )
    safety_fields = (
        "robot_collision_failure",
        "scene_collision_failure",
        "containment_violation",
        "forbidden_robot_task_collision_failure",
        "locked_joint_containment_violation",
    )
    safety_complete = all(
        isinstance(row[field], bool) for row in normalized for field in safety_fields
    )
    safety_ok = safety_complete and not any(
        row[field] for row in normalized for field in safety_fields
    )
    safety_events: list[dict[str, Any]] = []
    for row in normalized:
        event_type = None
        measured_force_n = None
        if row["forbidden_robot_task_collision_failure"] is True:
            event_type = "forbidden_robot_object_contact_force_exceeded"
            measured_force_n = row[
                "robot_task_forbidden_collision_peak_force_n"
            ]
        elif row["robot_collision_failure"] is True:
            event_type = "robot_scene_contact_force_exceeded"
            measured_force_n = row["robot_scene_contact_peak_force_n"]
        elif row["scene_collision_failure"] is True:
            event_type = "task_scene_contact_force_exceeded"
            measured_force_n = row["task_scene_collision_peak_force_n"]
        elif row["containment_violation"] is True:
            event_type = "workspace_containment_excursion"
        elif row["locked_joint_containment_violation"] is True:
            event_type = "locked_joint_containment_excursion"
        if event_type is None:
            continue
        raw_pairs = row["robot_task_forbidden_contact_pairs"]
        contact_pair_identities = (
            [
                json.loads(json.dumps(item, allow_nan=False))
                for item in raw_pairs
                if isinstance(item, (str, Mapping))
            ]
            if event_type == "forbidden_robot_object_contact_force_exceeded"
            and isinstance(raw_pairs, Sequence)
            and not isinstance(raw_pairs, (str, bytes))
            else []
        )
        event = {
            "event_type": event_type,
            "step_index": row["step_index"],
            "simulation_time_seconds": (
                row["step_index"] / spec["control_frequency_hz"]
                if spec["control_frequency_hz"] is not None
                else None
            ),
            "measured_force_n": measured_force_n,
            "threshold_n": row["collision_failure_minimum_force_n"],
            "contact_pair_identities": contact_pair_identities,
            "contact_pair_identity_status": (
                "observed"
                if contact_pair_identities
                else "contact_pair_identity_missing"
            ),
        }
        safety_events.append(event)
    temporal_criteria = criteria["temporal_invariants"]
    no_drop_required = temporal_criteria["no_drop"]["mode"] == "required"
    contact_trace_complete = all(
        isinstance(row["task_contact_active"], bool) for row in normalized
    )
    support_trace_complete = all(
        isinstance(row["support_contact_active"], bool) for row in normalized
    )
    drop_events: list[dict[str, Any]] = []
    if contact_trace_complete and support_trace_complete:
        loss: dict[str, Any] | None = None

        def retain_drop_event(
            candidate: Mapping[str, Any],
            *,
            support_recontact: Mapping[str, Any] | None,
            task_contact_recovered_step: int | None = None,
        ) -> None:
            fall_m = max(
                0.0,
                float(candidate["reference_height_m"])
                - float(candidate["minimum_height_m"]),
            )
            if (
                candidate.get("unsupported_started_step") is None
                or fall_m < temporal_criteria["no_drop"]["minimum_fall_m"]
            ):
                return
            destination_inside_at_recontact = None
            support_recontact_step = None
            if support_recontact is not None:
                support_recontact_step = support_recontact["step_index"]
                destination_inside_at_recontact = (
                    support_recontact["destination_pose"] is not None
                    or spec.get("destination_relation") is None
                ) and subject_inside_destination(support_recontact)
            drop_events.append(
                {
                    "contact_lost_step": candidate["contact_lost_step"],
                    "unsupported_started_step": candidate[
                        "unsupported_started_step"
                    ],
                    "reference_height_m": candidate["reference_height_m"],
                    "minimum_height_m": candidate["minimum_height_m"],
                    "minimum_height_step": candidate["minimum_height_step"],
                    "support_recontact_step": support_recontact_step,
                    "task_contact_recovered_step": task_contact_recovered_step,
                    "fall_m": fall_m,
                    "destination_inside_at_recontact": (
                        destination_inside_at_recontact
                    ),
                }
            )

        for previous, row in zip(normalized, normalized[1:], strict=False):
            if previous["task_contact_active"] is True and row["task_contact_active"] is False:
                loss = {
                    "contact_lost_step": row["step_index"],
                    "reference_height_m": previous["pose"][2],
                    "minimum_height_m": previous["pose"][2],
                    "minimum_height_step": previous["step_index"],
                    "unsupported_started_step": None,
                }
            if loss is not None:
                if row["support_contact_active"] is False:
                    if loss["unsupported_started_step"] is None:
                        loss["unsupported_started_step"] = row["step_index"]
                    if row["pose"][2] < loss["minimum_height_m"]:
                        loss["minimum_height_m"] = row["pose"][2]
                        loss["minimum_height_step"] = row["step_index"]
                elif loss["unsupported_started_step"] is not None:
                    if row["pose"][2] < loss["minimum_height_m"]:
                        loss["minimum_height_m"] = row["pose"][2]
                        loss["minimum_height_step"] = row["step_index"]
                    retain_drop_event(loss, support_recontact=row)
                    loss = None
                    continue
                if loss is not None and row["task_contact_active"] is True:
                    retain_drop_event(
                        loss,
                        support_recontact=None,
                        task_contact_recovered_step=row["step_index"],
                    )
                    loss = None
        if loss is not None:
            retain_drop_event(loss, support_recontact=None)
    maximum_force_limit = temporal_criteria["maximum_task_contact_force_n"]
    force_trace_complete = all(
        row["task_contact_force_n"] is not None for row in normalized
    )
    peak_task_contact_force_n = (
        max(float(row["task_contact_force_n"]) for row in normalized)
        if force_trace_complete
        else None
    )
    task_contact_force_sources = sorted(
        {
            str(row["task_contact_force_source"])
            for row in normalized
            if row["task_contact_force_source"] is not None
        }
    )
    contact_classes_complete = all(
        isinstance(row["contact_classes_active"], Sequence)
        and not isinstance(row["contact_classes_active"], (str, bytes))
        and all(isinstance(item, str) for item in row["contact_classes_active"])
        for row in normalized
    )
    observed_contact_classes = sorted(
        {
            item
            for row in normalized
            for item in (
                row["contact_classes_active"] if contact_classes_complete else []
            )
        }
    )
    forbidden_contact_classes = set(temporal_criteria["forbidden_contact_classes"])
    observed_forbidden_contact_classes = sorted(
        forbidden_contact_classes.intersection(observed_contact_classes)
    )
    containment_trace_complete = all(
        isinstance(row["containment_violation"], bool) for row in normalized
    )
    containment_excursion_steps = [
        row["step_index"]
        for row in normalized
        if row["containment_violation"] is True
    ]
    workspace_trace_complete = all(
        isinstance(row["workspace_excursion"], bool) for row in normalized
    )
    workspace_excursion_steps = [
        row["step_index"]
        for row in normalized
        if row["workspace_excursion"] is True
    ]
    retry_counts = [row["retry_count"] for row in normalized]
    regrasp_counts = [row["regrasp_count"] for row in normalized]
    retry_trace_complete = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in retry_counts
    ) and all(
        previous <= current
        for previous, current in zip(retry_counts, retry_counts[1:], strict=False)
    )
    regrasp_trace_complete = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in regrasp_counts
    ) and all(
        previous <= current
        for previous, current in zip(regrasp_counts, regrasp_counts[1:], strict=False)
    )
    maximum_retries_observed = (
        max(row["retry_count"] for row in normalized) if retry_trace_complete else None
    )
    maximum_regrasps_observed = (
        max(row["regrasp_count"] for row in normalized)
        if regrasp_trace_complete
        else None
    )
    temporal_readback_gaps: list[str] = []
    if no_drop_required and not (contact_trace_complete and support_trace_complete):
        temporal_readback_gaps.append("no_drop_contact_support_trace")
    if maximum_force_limit is not None and not force_trace_complete:
        temporal_readback_gaps.append("task_contact_force_trace")
    if forbidden_contact_classes and not contact_classes_complete:
        temporal_readback_gaps.append("contact_class_trace")
    if (
        temporal_criteria["containment_excursions"] == "forbidden"
        and not containment_trace_complete
    ):
        temporal_readback_gaps.append("containment_excursion_trace")
    if (
        temporal_criteria["workspace_excursions"] == "forbidden"
        and not workspace_trace_complete
    ):
        temporal_readback_gaps.append("workspace_excursion_trace")
    if temporal_criteria["maximum_retries"] is not None and not retry_trace_complete:
        temporal_readback_gaps.append("retry_event_ledger")
    if temporal_criteria["maximum_regrasps"] is not None and not regrasp_trace_complete:
        temporal_readback_gaps.append("regrasp_event_ledger")
    motion_criteria = criteria["motion"]
    moved = max(translation) > motion_criteria["movement_epsilon_m"]
    # Simulator state is floating-point readback. Treat a value that differs
    # from an exact preregistered boundary only by round-off as on the
    # boundary; this is not a task tolerance and must remain far below any
    # physical/scoring tolerance in the task contract.
    minimum_translation = motion_criteria["minimum_translation_m"]
    translated = minimum_translation is None or (
        max(translation) >= minimum_translation
        or math.isclose(
            max(translation), minimum_translation, rel_tol=0.0, abs_tol=1.0e-12
        )
    )
    minimum_lift = motion_criteria["minimum_lift_m"]
    lifted = minimum_lift is None or (
        max(lift) >= minimum_lift
        or math.isclose(
            max(lift), minimum_lift, rel_tol=0.0, abs_tol=1.0e-12
        )
    )
    task_contact_mode = criteria["terminal_task_contact"]["mode"]
    task_contact_satisfied = (
        True
        if task_contact_mode == "ignored"
        else task_contact_cleared
        if task_contact_mode == "cleared"
        else task_contact_maintained
    )
    gripper_mode = gripper_criteria["mode"]
    gripper_satisfied = (
        True
        if gripper_mode == "ignored"
        else released
        if gripper_mode == "released"
        else gripper_closed
    )
    criterion_results = {
        "destination_containment": (
            destination_criteria["mode"] == "ignored" or destination_inside
        ),
        **(
            {"destination_pose_stability": destination_pose_stable}
            if spec.get("destination_relation") is not None
            else {}
        ),
        "orientation": (
            orientation_criteria["mode"] == "ignored" or orientation_ok
        ),
        "support_height": (
            support_criteria["height_mode"] == "ignored" or support_ok
        ),
        "support_contact": (
            support_criteria["contact_mode"] == "ignored" or support_contact_ok
        ),
        "terminal_task_contact": task_contact_satisfied,
        "gripper_state": gripper_satisfied,
        "settling": not settling_required or settled,
        "safety": safety_ok,
        "minimum_translation": translated,
        "minimum_lift": lifted,
        "no_drop": not no_drop_required
        or (
            contact_trace_complete
            and support_trace_complete
            and not drop_events
        ),
        "maximum_task_contact_force": (
            maximum_force_limit is None
            or (
                peak_task_contact_force_n is not None
                and peak_task_contact_force_n <= maximum_force_limit
            )
        ),
        "forbidden_contact_classes": (
            not forbidden_contact_classes
            or (
                contact_classes_complete
                and not observed_forbidden_contact_classes
            )
        ),
        "containment_excursions": (
            temporal_criteria["containment_excursions"] == "ignored"
            or (
                containment_trace_complete and not containment_excursion_steps
            )
        ),
        "workspace_excursions": (
            temporal_criteria["workspace_excursions"] == "ignored"
            or (workspace_trace_complete and not workspace_excursion_steps)
        ),
        "maximum_retries": (
            temporal_criteria["maximum_retries"] is None
            or (
                maximum_retries_observed is not None
                and maximum_retries_observed <= temporal_criteria["maximum_retries"]
            )
        ),
        "maximum_regrasps": (
            temporal_criteria["maximum_regrasps"] is None
            or (
                maximum_regrasps_observed is not None
                and maximum_regrasps_observed <= temporal_criteria["maximum_regrasps"]
            )
        ),
    }
    retreat = None
    if "retreat" in criteria:
        retreat = score_retreat(
            criterion=criteria["retreat"], task_spec=spec, samples=samples,
            window_samples=settle_window_samples,
            release_width_m=spec["release_gripper_width_min_m"],
        )
        criterion_results["retreat"] = retreat["satisfied"]
    succeeded = all(criterion_results.values())
    planar_push = spec["manipulation_strategy"] == "planar_push"
    required_task_contact_readback = task_contact_mode != "ignored"
    required_support_contact_readback = support_criteria["contact_mode"] == "required"
    if (
        not destination_pose_readback_complete
        or not safety_complete
        or (required_support_contact_readback and not support_contact_complete)
        or (required_task_contact_readback and not task_contact_complete)
        or temporal_readback_gaps
        or (retreat is not None and not retreat["readback_complete"])
    ):
        status = "undetermined"
        if not destination_pose_readback_complete:
            outcome = "native_destination_pose_readback_missing"
        elif not safety_complete:
            outcome = "native_safety_readback_missing"
        elif required_support_contact_readback and not support_contact_complete:
            outcome = "native_support_contact_readback_missing"
        elif required_task_contact_readback and not task_contact_complete:
            outcome = "native_task_contact_readback_missing"
        elif retreat is not None and not retreat["readback_complete"]:
            outcome = "native_retreat_readback_missing"
        else:
            outcome = "native_temporal_event_readback_missing"
    elif not safety_ok:
        status = "scored"
        outcome = "collision_or_containment_failure"
    elif succeeded:
        status = "scored"
        outcome = (
            OUTCOME_PUSHED_AND_SETTLED if planar_push else "placed_and_settled"
        )
    elif not moved:
        status = "scored"
        outcome = OUTCOME_NEVER_MOVED
    elif (
        planar_push
        and all(
            value
            for key, value in criterion_results.items()
            if key != "terminal_task_contact"
        )
        and not task_contact_satisfied
    ):
        status = "scored"
        outcome = "push_contact_not_cleared"
    elif (
        destination_inside
        and gripper_mode == "released"
        and not gripper_satisfied
    ):
        status = "scored"
        outcome = "release_incomplete"
    else:
        status = "scored" if settle_available else "undetermined"
        outcome = "moved_below_success_contract"
    failed_criteria = [
        name for name, satisfied in criterion_results.items() if not satisfied
    ]
    plain_reasons = {
        "retreat": "The released gripper did not maintain the authored withdrawal clearance from the object.",
        "destination_containment": "The object did not remain inside the authored destination.",
        "destination_pose_stability": "The destination moved beyond its qualified pose tolerance.",
        "orientation": "The object did not finish in the required orientation.",
        "support_height": "The object did not finish at the required support height.",
        "support_contact": "The object was not supported at the end of the episode.",
        "terminal_task_contact": "Robot contact with the task object did not match the required terminal state.",
        "gripper_state": "The gripper did not match the required terminal state.",
        "settling": "The object did not remain still for the required settling window.",
        "safety": "A collision or containment safety check failed or was unavailable.",
        "minimum_translation": "The object did not move the required minimum distance.",
        "minimum_lift": "The object was not lifted by the required minimum height.",
        "no_drop": "The object was dropped during the episode, even though it may later have reached the destination.",
        "maximum_task_contact_force": "Task contact exceeded the authored maximum force.",
        "forbidden_contact_classes": "A forbidden contact occurred during the episode.",
        "containment_excursions": "The object left its allowed containment region during the episode.",
        "workspace_excursions": "The object or robot left the authored workspace during the episode.",
        "maximum_retries": "The episode exceeded the authored retry limit.",
        "maximum_regrasps": "The episode exceeded the authored regrasp limit.",
    }
    specific_failure_reason = None
    if safety_events:
        first_safety_event = safety_events[0]
        if (
            first_safety_event["event_type"]
            == "forbidden_robot_object_contact_force_exceeded"
        ):
            force = first_safety_event["measured_force_n"]
            threshold = first_safety_event["threshold_n"]
            force_text = f"{force:g} N" if force is not None else "an unknown force"
            threshold_text = (
                f"{threshold:g} N" if threshold is not None else "the safety threshold"
            )
            specific_failure_reason = (
                "Forbidden robot-object contact reached "
                f"{force_text}, exceeding {threshold_text} at step "
                f"{first_safety_event['step_index']}."
            )
    report: dict[str, Any] = {
        "schema_version": RIGID_REPORT_SCHEMA_VERSION,
        "status": status,
        "task_kind": TASK_KIND_RIGID_PICK_PLACE,
        "subject_asset_id": spec["subject_asset_id"],
        "manipulation_strategy": spec["manipulation_strategy"],
        "task_success_contract": success_contract,
        "task_success_contract_digest": success_contract["contract_digest"],
        "task_succeeded": succeeded,
        "outcome": outcome,
        "criteria_satisfied": criterion_results,
        "failed_criteria": failed_criteria,
        "failure_reason_plain_english": (
            None
            if succeeded
            else specific_failure_reason
            if specific_failure_reason is not None
            else plain_reasons[failed_criteria[0]]
            if failed_criteria
            else "Required deterministic readback was unavailable."
        ),
        "event_ledger": {
            "schema_version": "rigid_task_event_ledger.v1",
            "safety_events": safety_events,
            "drop_events": drop_events,
            "peak_task_contact_force_n": peak_task_contact_force_n,
            "task_contact_force_sources": task_contact_force_sources,
            "observed_contact_classes": observed_contact_classes,
            "observed_forbidden_contact_classes": observed_forbidden_contact_classes,
            "containment_excursion_steps": containment_excursion_steps,
            "workspace_excursion_steps": workspace_excursion_steps,
            "maximum_retries_observed": maximum_retries_observed,
            "maximum_regrasps_observed": maximum_regrasps_observed,
            "required_readback_gaps": temporal_readback_gaps,
            "derived_only_from_episode_samples": True,
        },
        "measurements": {
            **({"retreat": retreat} if retreat is not None else {}),
            "sample_count": len(normalized),
            "reset_translation_error_m": reset_translation_error,
            "reset_orientation_error_rad": reset_orientation_error,
            "maximum_translation_m": max(translation),
            "maximum_lift_m": max(lift),
            "settle_window_available": settle_available,
            "settle_destination_inside": destination_inside,
            "destination_pose_readback_complete": destination_pose_readback_complete,
            "destination_pose_stable": destination_pose_stable,
            "settle_orientation_ok": orientation_ok,
            "settle_support_height_ok": support_ok,
            "settle_support_contact_readback_complete": support_contact_complete,
            "settle_support_contact_ok": support_contact_ok,
            "settled": settled,
            "released": released,
            "gripper_closed": gripper_closed,
            "settle_task_contact_readback_complete": task_contact_complete,
            "settle_task_contact_cleared": task_contact_cleared,
            "settle_task_contact_maintained": task_contact_maintained,
            "native_safety_readback_complete": safety_complete,
            "native_safety_ok": safety_ok,
        },
        "learned_judge_consulted": False,
        "candidate_policy_queried_by_scorer": False,
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


