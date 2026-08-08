"""Retain the per-step episode record the loop already measures.

``run_policy_episode`` observes the arm after every environment step, holds
every commanded action row, and samples deterministic object state per step --
then historically discarded all of it in favour of summaries.  That made
"was it smooth?" unanswerable from the receipt and forced re-running paid
episodes to recover information the harness had already computed.

This module turns those in-memory collections into one retained, digest-bound
trace with a row per control step at the DROID 15 Hz rate:

* ``observation_joint_position_rad`` / ``observed_after_rad`` -- the state a
  lab's ``observation.state`` column is built from, before/after semantics
  stated rather than implied.
* ``action_droid`` -- the clipped DROID-space row actually executed (seven
  joint velocities plus absolute gripper), the lab-facing ``action`` column.
* ``object_sample`` -- can pose, gripper width, and grasp-frame position per
  step, so task progress is a curve rather than a terminal label.

The trace is also the replay contract: joint positions per step, gripper
width, and object pose are exactly the state a kinematic replay renderer needs
to re-render an episode at full frame rate without physics or a policy in the
loop.

Everything here is arithmetic over already-collected values; it never touches
a simulator and is hermetically testable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:  # flat provider-bundle layout
    from adp009d_droid_action_execution import ARM_JOINT_COUNT
except ModuleNotFoundError:  # repository package
    from .adp009d_droid_action_execution import ARM_JOINT_COUNT
try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest

STEP_TRACE_SCHEMA_VERSION = "adp009d_episode_step_trace.v1"
MOTION_QUALITY_SCHEMA_VERSION = "adp009d_episode_motion_quality.v1"

# DROID gripper convention: scalar in [0, 1], above 0.5 means closed.
DROID_GRIPPER_CLOSED_THRESHOLD = 0.5

BLOCKER_TRACE_LENGTHS = "step_trace_collection_lengths_inconsistent"
BLOCKER_TRACE_BEFORE_MISMATCH = "step_trace_observed_before_mismatch"
BLOCKER_TRACE_PARTIAL_CHUNK = "step_trace_policy_rows_not_horizon_aligned"
BLOCKER_TRACE_VALUES = "step_trace_values_invalid"
BLOCKER_TRACE_SAMPLE_INDEX = "step_trace_object_sample_index_not_contiguous"


class StepTraceError(ValueError):
    """Fail-closed step-trace contract errors."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _finite_vector(values: Any, *, width: int, error: str) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise StepTraceError([error]) from exc
    if len(vector) != width or not all(math.isfinite(value) for value in vector):
        raise StepTraceError([f"{error}:{len(vector)}"])
    return vector


def build_step_trace(
    *,
    joint_trace: Sequence[Sequence[float]],
    commanded_actions: Sequence[Mapping[str, Any]],
    object_samples: Sequence[Mapping[str, Any]],
    settle_isaac_action: Sequence[float],
    open_loop_horizon: int,
    control_hz: int,
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Assemble the retained per-step trace from the loop's collections.

    ``joint_trace`` holds the observed arm joints at reset plus after every
    environment step; ``object_samples`` matches it one-to-one.  Policy rows
    must fill whole chunks: a partial chunk means the loop and this trace
    disagree about what executed, which is a harness fault rather than data.
    """

    if int(open_loop_horizon) < 1 or int(control_hz) < 1:
        raise StepTraceError([BLOCKER_TRACE_VALUES])
    total_steps = len(joint_trace) - 1
    if total_steps < 1 or len(object_samples) != len(joint_trace):
        raise StepTraceError(
            [
                f"{BLOCKER_TRACE_LENGTHS}:joint_trace={len(joint_trace)}"
                f":object_samples={len(object_samples)}"
            ]
        )
    policy_steps = len(commanded_actions)
    settle_steps = total_steps - policy_steps
    if settle_steps < 0:
        raise StepTraceError(
            [f"{BLOCKER_TRACE_LENGTHS}:commanded={policy_steps}:total={total_steps}"]
        )
    if policy_steps % int(open_loop_horizon) != 0:
        raise StepTraceError(
            [f"{BLOCKER_TRACE_PARTIAL_CHUNK}:{policy_steps}%{open_loop_horizon}"]
        )
    limits = [
        _finite_vector(row, width=2, error=BLOCKER_TRACE_VALUES) for row in joint_limits
    ]
    if len(limits) != ARM_JOINT_COUNT:
        raise StepTraceError([f"{BLOCKER_TRACE_VALUES}:joint_limits={len(limits)}"])
    settle_action = _finite_vector(
        settle_isaac_action, width=ARM_JOINT_COUNT + 1, error=BLOCKER_TRACE_VALUES
    )

    observed = [
        _finite_vector(row, width=ARM_JOINT_COUNT, error=BLOCKER_TRACE_VALUES)
        for row in joint_trace
    ]

    rows: list[dict[str, Any]] = []
    for step in range(total_steps):
        sample = dict(object_samples[step + 1])
        if sample.get("step_index") != step + 1:
            raise StepTraceError(
                [f"{BLOCKER_TRACE_SAMPLE_INDEX}:{sample.get('step_index')}!={step + 1}"]
            )
        row: dict[str, Any] = {
            "step_index": step,
            "sim_time_s": step / float(control_hz),
            "observation_joint_position_rad": observed[step],
            "observed_after_rad": observed[step + 1],
            "object_sample": sample,
        }
        if step < policy_steps:
            action = commanded_actions[step]
            before = _finite_vector(
                action["observed_before_rad"],
                width=ARM_JOINT_COUNT,
                error=BLOCKER_TRACE_VALUES,
            )
            if before != observed[step]:
                raise StepTraceError([f"{BLOCKER_TRACE_BEFORE_MISMATCH}:{step}"])
            row.update(
                {
                    "phase": "policy",
                    "query_index": step // int(open_loop_horizon),
                    "chunk_row_index": step % int(open_loop_horizon),
                    "action_droid": _finite_vector(
                        action["clipped_droid_action"],
                        width=ARM_JOINT_COUNT + 1,
                        error=BLOCKER_TRACE_VALUES,
                    ),
                    "joint_velocity_command_rad_s": [
                        float(value)
                        for value in action.get("joint_velocity_command_rad_s") or []
                    ],
                    "joint_position_target_rad": _finite_vector(
                        action["joint_position_target_rad"],
                        width=ARM_JOINT_COUNT,
                        error=BLOCKER_TRACE_VALUES,
                    ),
                    "isaac_action": _finite_vector(
                        action["isaac_action"],
                        width=ARM_JOINT_COUNT + 1,
                        error=BLOCKER_TRACE_VALUES,
                    ),
                    "source_action_space": str(action["source_action_space"]),
                }
            )
        else:
            # The settle window applies the held release action: zero commanded
            # velocity in DROID space with the gripper explicitly open.
            row.update(
                {
                    "phase": "settle",
                    "query_index": None,
                    "chunk_row_index": None,
                    "action_droid": [0.0] * ARM_JOINT_COUNT + [0.0],
                    "joint_velocity_command_rad_s": [0.0] * ARM_JOINT_COUNT,
                    "joint_position_target_rad": settle_action[:ARM_JOINT_COUNT],
                    "isaac_action": settle_action,
                    "source_action_space": "settle_hold_release",
                }
            )
        rows.append(row)

    trace: dict[str, Any] = {
        "schema_version": STEP_TRACE_SCHEMA_VERSION,
        "control_hz": int(control_hz),
        "open_loop_horizon": int(open_loop_horizon),
        "total_steps": total_steps,
        "policy_steps": policy_steps,
        "settle_steps": settle_steps,
        "initial_object_sample": dict(object_samples[0]),
        "joint_limits_rad": limits,
        "state_semantics": (
            "observation_joint_position_rad_is_the_pre_step_observed_state"
        ),
        "action_semantics": (
            "action_droid_is_the_clipped_executed_droid_row_seven_joint_velocities"
            "_plus_absolute_gripper"
        ),
        "replay_sufficiency": (
            "joint_positions_gripper_width_and_can_pose_per_step_support_"
            "kinematic_replay_rendering_without_physics"
        ),
        "rows": rows,
    }
    trace["step_trace_digest"] = canonical_digest(
        trace, digest_field="step_trace_digest"
    )
    return trace


def _per_step_observed_velocity(rows: Sequence[Mapping[str, Any]], hz: float) -> list[list[float]]:
    return [
        [
            (row["observed_after_rad"][joint] - row["observation_joint_position_rad"][joint]) * hz
            for joint in range(ARM_JOINT_COUNT)
        ]
        for row in rows
    ]


def _diff_rate(series: Sequence[Sequence[float]], hz: float) -> list[list[float]]:
    return [
        [
            (series[index + 1][joint] - series[index][joint]) * hz
            for joint in range(ARM_JOINT_COUNT)
        ]
        for index in range(len(series) - 1)
    ]


def _max_abs(series: Sequence[Sequence[float]]) -> float:
    return max((abs(value) for row in series for value in row), default=0.0)


def _rms(series: Sequence[Sequence[float]]) -> float:
    values = [value for row in series for value in row]
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def derive_motion_quality(
    step_trace: Mapping[str, Any],
    *,
    joint_limits: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Derive smoothness and safety metrics from the retained trace.

    All quantities are finite differences of the 15 Hz observed joint stream,
    so they are diagnostics of executed motion, not of the policy's internal
    plan.  ``interior_*`` metrics exclude the first two steps, where starting
    from rest makes acceleration and jerk real rather than a smoothness fault.
    """

    rows = list(step_trace["rows"])
    hz = float(step_trace["control_hz"])
    if not rows:
        raise StepTraceError([BLOCKER_TRACE_LENGTHS])

    velocity = _per_step_observed_velocity(rows, hz)
    acceleration = _diff_rate(velocity, hz)
    jerk = _diff_rate(acceleration, hz)

    limits = [
        _finite_vector(row, width=2, error=BLOCKER_TRACE_VALUES) for row in joint_limits
    ]
    margins = [
        min(
            row["observed_after_rad"][joint] - limits[joint][0],
            limits[joint][1] - row["observed_after_rad"][joint],
        )
        for row in rows
        for joint in range(ARM_JOINT_COUNT)
    ]

    horizon = int(step_trace["open_loop_horizon"])
    policy_steps = int(step_trace["policy_steps"])
    boundaries = [
        index
        for index in range(horizon - 1, policy_steps - 1, horizon)
        if index + 1 < policy_steps
    ]
    commanded_jumps = []
    observed_jumps = []
    for index in boundaries:
        last_cmd = rows[index]["action_droid"][:ARM_JOINT_COUNT]
        next_cmd = rows[index + 1]["action_droid"][:ARM_JOINT_COUNT]
        commanded_jumps.append(
            max(abs(next_cmd[j] - last_cmd[j]) for j in range(ARM_JOINT_COUNT))
        )
        observed_jumps.append(
            max(
                abs(velocity[index + 1][j] - velocity[index][j])
                for j in range(ARM_JOINT_COUNT)
            )
        )

    gripper_states = [
        row["action_droid"][ARM_JOINT_COUNT] > DROID_GRIPPER_CLOSED_THRESHOLD
        for row in rows
        if row["phase"] == "policy"
    ]
    gripper_transitions = sum(
        1
        for index in range(1, len(gripper_states))
        if gripper_states[index] != gripper_states[index - 1]
    )

    initial_point = dict(step_trace.get("initial_object_sample") or {}).get(
        "grasp_frame_position_world_m"
    )
    path = [initial_point] + [
        row["object_sample"].get("grasp_frame_position_world_m") for row in rows
    ]
    path = [point for point in path if point is not None]
    path_length = sum(
        math.dist(path[index], path[index + 1]) for index in range(len(path) - 1)
    )
    net_displacement = math.dist(path[0], path[-1]) if len(path) > 1 else 0.0

    quality = {
        "schema_version": MOTION_QUALITY_SCHEMA_VERSION,
        "claim_scope": "executed_motion_diagnostics_from_15hz_observed_state",
        "observed_joint_velocity_max_abs_rad_s": _max_abs(velocity),
        "observed_joint_velocity_rms_rad_s": _rms(velocity),
        "observed_joint_acceleration_max_abs_rad_s2": _max_abs(acceleration),
        "interior_joint_acceleration_max_abs_rad_s2": _max_abs(acceleration[1:]),
        "observed_joint_jerk_max_abs_rad_s3": _max_abs(jerk),
        "interior_joint_jerk_max_abs_rad_s3": _max_abs(jerk[2:]),
        "observed_joint_jerk_rms_rad_s3": _rms(jerk),
        "joint_limit_min_margin_rad": min(margins) if margins else None,
        "chunk_boundary_count": len(boundaries),
        "chunk_boundary_commanded_velocity_jump_max_abs_rad_s": max(
            commanded_jumps, default=0.0
        ),
        "chunk_boundary_observed_velocity_jump_max_abs_rad_s": max(
            observed_jumps, default=0.0
        ),
        "gripper_droid_transition_count": gripper_transitions,
        "end_effector_path_length_m": path_length,
        "end_effector_net_displacement_m": net_displacement,
    }
    quality["motion_quality_digest"] = canonical_digest(
        quality, digest_field="motion_quality_digest"
    )
    return quality


__all__ = [
    "MOTION_QUALITY_SCHEMA_VERSION",
    "STEP_TRACE_SCHEMA_VERSION",
    "StepTraceError",
    "build_step_trace",
    "derive_motion_quality",
]
