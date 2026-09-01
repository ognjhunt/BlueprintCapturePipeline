"""Judge rigid construction gates solely from retained native readbacks.

Extracted from ``native_task_construction_plan`` so the plan compiler stays
inside its governed module budget; the evaluation contract, schema, and
digest behaviour are unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_construction_validation import (
    RIGID_SCHEMA_VERSION,
    NativeTaskConstructionPlanError,
    finite_vector as _finite_vector,
)
from .rigid_frame_transforms import rotate_vector_xyzw


def _quaternion(value: Any, *, error: str) -> list[float]:
    result = _finite_vector(value, length=4, error=error)
    if abs(sum(item * item for item in result) - 1.0) > 1.0e-6:
        raise NativeTaskConstructionPlanError([error])
    return result


def _quaternion_angle_xyzw(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b, strict=True)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def evaluate_rigid_construction_gates(
    *,
    phase_plan: Mapping[str, Any],
    phase_results: Sequence[Mapping[str, Any]],
    reset_replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate rigid construction solely from retained native readbacks."""

    if (
        phase_plan.get("schema_version") != RIGID_SCHEMA_VERSION
        or phase_plan.get("plan_digest")
        != canonical_digest(dict(phase_plan), digest_field="plan_digest")
    ):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_phase_plan_invalid"]
        )
    expected_ids = [row["phase_id"] for row in phase_plan["phases"]]
    observed = {
        str(row.get("phase_id") or ""): dict(row)
        for row in phase_results
        if isinstance(row, Mapping)
    }
    if set(observed) != set(expected_ids) or len(observed) != len(expected_ids):
        raise NativeTaskConstructionPlanError(
            ["native_rigid_construction_phase_results_invalid"]
        )

    def samples(phase_id: str) -> list[dict[str, Any]]:
        rows = list(observed[phase_id].get("task_samples") or [])
        terminal = observed[phase_id].get("task_sample")
        if isinstance(terminal, Mapping):
            rows.append(dict(terminal))
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise NativeTaskConstructionPlanError(
                [f"native_rigid_construction_readback_missing:{phase_id}"]
            )
        return [dict(row) for row in rows]

    def pose(sample: Mapping[str, Any]) -> list[float]:
        value = _finite_vector(
            sample.get("task_scoring_pose_world"),
            length=7,
            error="native_rigid_construction_scoring_pose_readback_invalid",
        )
        _quaternion(
            value[3:],
            error="native_rigid_construction_scoring_pose_readback_invalid",
        )
        return value

    def position(sample: Mapping[str, Any]) -> list[float]:
        return pose(sample)[:3]

    thresholds = phase_plan["thresholds"]
    contact_threshold = float(thresholds["task_contact_minimum_force_n"])
    collision_threshold = float(thresholds["collision_failure_minimum_force_n"])
    start = list(phase_plan["start_position_world_m"])
    bounds = phase_plan["destination_position_bounds_world_m"]
    support = list(phase_plan["support_height_interval_m"])
    settle_count = int(phase_plan["settle_window_samples"])
    settle_tolerance = float(thresholds["settle_position_tolerance_m"])
    settle_orientation_tolerance = float(
        thresholds["settle_orientation_tolerance_rad"]
    )
    destination_orientation_tolerance = float(
        thresholds["destination_orientation_tolerance_rad"]
    )

    all_samples = [sample for phase_id in expected_ids for sample in samples(phase_id)]
    collision_clear = all(
        max(
            float(sample.get("robot_scene_contact_peak_force_n", float("inf"))),
            float(
                sample.get(
                    "robot_task_forbidden_collision_peak_force_n", float("inf")
                )
            ),
            float(sample.get("task_scene_collision_peak_force_n", float("inf"))),
        )
        < collision_threshold
        and sample.get("locked_joint_containment_violation") is False
        for sample in all_samples
    )
    strategy = str(phase_plan.get("manipulation_strategy") or "pick_and_place")
    push = strategy == "planar_push"
    contact_phase_id = "push_contact" if push else "grasp_contact"
    contact_rows = samples(contact_phase_id)
    initial_contact = max(
        float(row.get("task_robot_contact_peak_force_n", 0.0))
        for row in contact_rows
    ) >= contact_threshold
    # Contact must be established where the plan authored it.  Object motion
    # during push_contact beyond the commanded interference bound means the
    # fingertips met the object before the commanded frame arrived -- an
    # understated ``closed_fingertip_forward_offset_m`` -- which is exactly
    # how attempt 001 punted the object 93 mm.  The force gate above brackets
    # the opposite error: an overstated offset never develops contact at all.
    push_contact_standoff = True
    if push and "push_contact_standoff" in phase_plan["gate_contract"]:
        push_contact_standoff = all(
            math.dist(position(sample), start)
            <= float(thresholds["push_contact_max_displacement_m"])
            for sample in contact_rows
        )
    if push:
        support_clearance = True
        relocation_ids = [
            phase_id for phase_id in expected_ids if phase_id.startswith("push_")
            and phase_id not in {"push_contact", "push_detach", "push_release"}
        ]
    else:
        lift_position = position(samples("lift_clearance")[-1])
        lift_delta = [lift_position[index] - start[index] for index in range(3)]
        support_clearance = sum(
            lift_delta[index]
            * float(phase_plan["interaction_affordance"]["lift_unit_world"][index])
            for index in range(3)
        ) + 1.0e-9 >= float(thresholds["minimum_lift_m"])
        relocation_ids = [
            phase_id
            for phase_id in expected_ids
            if phase_id.startswith("relocate_")
        ]
    relocation_terminal_samples = [samples(phase_id)[-1] for phase_id in relocation_ids]
    phase_by_id = {row["phase_id"]: row for row in phase_plan["phases"]}
    relocation_tracking = all(
        math.dist(
            position(sample),
            phase_by_id[phase_id]["expected_scoring_position_world_m"],
        )
        <= float(thresholds["relocation_tracking_tolerance_m"])
        and _quaternion_angle_xyzw(
            pose(sample)[3:],
            phase_by_id[phase_id]["expected_scoring_orientation_world_xyzw"],
        )
        <= destination_orientation_tolerance
        for phase_id, sample in zip(
            relocation_ids, relocation_terminal_samples, strict=True
        )
    )
    relocation_progress = (
        bool(relocation_terminal_samples)
        and math.dist(start, position(relocation_terminal_samples[-1]))
        >= float(thresholds["minimum_translation_m"])
    )
    relocation_path = relocation_tracking and relocation_progress
    closed_motion_ids = (
        [contact_phase_id, *relocation_ids]
        if push
        else ["lift_clearance", *relocation_ids, "place"]
    )
    closed_motion_samples = [
        sample for phase_id in closed_motion_ids for sample in samples(phase_id)
    ]
    contact_local = phase_plan["interaction_affordance"][
        "contact_point_scoring_frame_m"
    ]
    if push:
        contact_maintained = relocation_path and all(
            float(sample.get("task_robot_contact_peak_force_n", 0.0))
            >= contact_threshold
            and float(sample.get("task_support_contact_peak_force_n", 0.0))
            >= contact_threshold
            for sample in closed_motion_samples
        )
    else:
        contact_maintained = relocation_path and all(
            math.dist(
                [
                    pose(sample)[index]
                    + rotate_vector_xyzw(pose(sample)[3:], contact_local)[index]
                    for index in range(3)
                ],
                _finite_vector(
                    sample.get("grasp_frame_position_world_m"),
                    length=3,
                    error="native_rigid_construction_grasp_frame_readback_invalid",
                ),
            )
            <= float(thresholds["relocation_tracking_tolerance_m"])
            and float(sample.get("task_robot_contact_peak_force_n", 0.0))
            >= contact_threshold
            for sample in closed_motion_samples
        )
    release_phase_id = "push_release" if push else "release"
    release_rows = samples(release_phase_id)
    release = (
        observed[release_phase_id].get("gripper_state") == "open"
        and release_rows[-1].get("finger_separation_m") is not None
        and float(release_rows[-1]["finger_separation_m"])
        > float(contact_rows[-1].get("finger_separation_m", float("inf")))
        and float(release_rows[-1].get("task_robot_contact_peak_force_n", float("inf")))
        < contact_threshold
    )
    settle_rows = samples("settle_observe")[-settle_count:]
    settle_positions = [position(row) for row in settle_rows]
    settle_poses = [pose(row) for row in settle_rows]
    final_position = settle_positions[-1]
    destination_containment = all(
        low <= value <= high
        for low, value, high in zip(
            bounds["minimum"], final_position, bounds["maximum"], strict=True
        )
    )
    destination_orientation = all(
        _quaternion_angle_xyzw(
            row[3:], phase_plan["destination_orientation_xyzw"]
        )
        <= destination_orientation_tolerance
        for row in settle_poses
    )
    support_contact = (
        len(settle_rows) >= settle_count
        and all(support[0] <= row[2] <= support[1] for row in settle_poses)
        and all(
            float(row.get("task_support_contact_peak_force_n", 0.0))
            >= contact_threshold
            for row in settle_rows
        )
    )
    support_stability = len(settle_rows) >= settle_count and all(
        math.dist(final_position, observed_position) <= settle_tolerance
        and _quaternion_angle_xyzw(settle_poses[-1][3:], observed_pose[3:])
        <= settle_orientation_tolerance
        for observed_position, observed_pose in zip(
            settle_positions, settle_poses, strict=True
        )
    )
    workspace = phase_plan["workspace_position_bounds_world_m"]
    workspace_containment = all(
        all(
            low <= value <= high
            for low, value, high in zip(
                workspace["minimum"],
                position(sample),
                workspace["maximum"],
                strict=True,
            )
        )
        for sample in all_samples
    )
    reachability = all(
        observed[phase_id].get("target_reached") is True
        for phase_id in expected_ids
    )
    gate_values = {
        "base_collision_clearance": collision_clear,
        "release": release,
        "retreat": observed["retreat"].get("target_reached") is True,
        "support_contact": support_contact,
        "support_stability": support_stability,
        "destination_containment": (
            destination_containment and destination_orientation
        ),
        "workspace_containment": workspace_containment,
        "recovery": observed["recovery"].get("target_reached") is True,
        "reset_readback": reset_replay.get("passed") is True,
        **(
            {
                "precontact_reachability": observed["precontact"].get(
                    "target_reached"
                )
                is True,
                "push_contact": initial_contact,
                "push_contact_standoff": push_contact_standoff,
                "push_contact_maintained": contact_maintained,
                "push_path": relocation_path,
            }
            if push
            else {
                "pregrasp_reachability": observed["pregrasp"].get(
                    "target_reached"
                )
                is True,
                "grasp_contact": initial_contact,
                "grasp_retention": support_clearance and contact_maintained,
                "support_clearance": support_clearance,
                "relocation_path": relocation_path,
            }
        ),
    }
    gate_rows = [
        {
            "gate_id": gate_id,
            "measurement_authority": phase_plan["gate_contract"][gate_id],
            "passed": bool(gate_values[gate_id]),
        }
        for gate_id in phase_plan["required_gate_ids"]
    ]
    blockers = [
        f"native_rigid_construction_gate_failed:{row['gate_id']}"
        for row in gate_rows
        if not row["passed"]
    ]
    result = {
        "schema_version": "native_rigid_construction_gate_evaluation.v1",
        "phase_plan_digest": phase_plan["plan_digest"],
        "all_phase_targets_reached": reachability,
        "gates": gate_rows,
        "passed": not blockers and reachability,
        "blockers": sorted(blockers),
        "evaluation_digest": "",
    }
    result["evaluation_digest"] = canonical_digest(
        result, digest_field="evaluation_digest"
    )
    return result


__all__ = ["evaluate_rigid_construction_gates"]
