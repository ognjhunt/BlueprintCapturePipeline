"""Contract tests for the per-step episode trace and derived motion quality.

The episode loop already observes joints after every environment step, holds
every commanded action row, and samples deterministic object state per step.
These tests pin the module that turns those in-memory collections into a
retained, digest-bound, per-step trace at the DROID 15 Hz control rate -- the
record labs expect and the input a kinematic replay renderer needs.
"""

from __future__ import annotations

import math

import pytest

from blueprint_pipeline.adp009d_episode_step_trace import (
    STEP_TRACE_SCHEMA_VERSION,
    StepTraceError,
    build_step_trace,
    derive_motion_quality,
)

CONTROL_HZ = 15
HORIZON = 4
JOINT_LIMITS = [[-2.9, 2.9]] * 7


def _commanded_action(velocity: float, *, gripper: float = 0.0) -> dict:
    return {
        "joint_position_target_rad": [velocity / CONTROL_HZ] * 7,
        "joint_velocity_command_rad_s": [velocity] * 7,
        "source_arm_command": [velocity] * 7,
        "source_action_space": "joint_velocity",
        "clipped_droid_action": [velocity] * 7 + [gripper],
        "observed_before_rad": [0.0] * 7,
        "isaac_action": [velocity / CONTROL_HZ] * 7 + [gripper],
    }


def _synthetic_episode(
    *,
    policy_steps: int = 8,
    settle_steps: int = 2,
    velocity: float = 0.1,
) -> dict:
    total = policy_steps + settle_steps
    joint_trace = [[0.0] * 7]
    commanded = []
    for step in range(total):
        before = list(joint_trace[-1])
        if step < policy_steps:
            action = _commanded_action(velocity)
            action["observed_before_rad"] = before
            commanded.append(action)
            after = [value + velocity / CONTROL_HZ for value in before]
        else:
            after = list(before)
        joint_trace.append(after)
    object_samples = [
        {
            "step_index": index,
            "can_pose_world": [1.0, 0.0, 0.5 + 0.001 * index, 1.0, 0.0, 0.0, 0.0],
            "gripper_width_m": 0.08,
            "grasp_frame_position_world_m": [0.4, 0.0, 0.6 + 0.001 * index],
        }
        for index in range(total + 1)
    ]
    return {
        "joint_trace": joint_trace,
        "commanded_actions": commanded,
        "object_samples": object_samples,
        "settle_isaac_action": [0.0] * 7 + [1.0],
        "open_loop_horizon": HORIZON,
        "control_hz": CONTROL_HZ,
        "joint_limits": JOINT_LIMITS,
    }


def test_step_trace_rows_carry_state_action_time_and_phase() -> None:
    trace = build_step_trace(**_synthetic_episode())

    assert trace["schema_version"] == STEP_TRACE_SCHEMA_VERSION
    assert trace["control_hz"] == CONTROL_HZ
    rows = trace["rows"]
    assert len(rows) == 10
    first = rows[0]
    assert first["step_index"] == 0
    assert first["sim_time_s"] == 0.0
    assert first["phase"] == "policy"
    assert first["query_index"] == 0
    assert first["chunk_row_index"] == 0
    assert first["observation_joint_position_rad"] == [0.0] * 7
    assert first["action_droid"] == [0.1] * 7 + [0.0]
    assert first["observed_after_rad"] == pytest.approx(
        [0.1 / CONTROL_HZ] * 7
    )
    assert first["object_sample"]["gripper_width_m"] == 0.08

    fifth = rows[5]
    assert fifth["query_index"] == 1
    assert fifth["chunk_row_index"] == 1
    assert fifth["sim_time_s"] == pytest.approx(5 / CONTROL_HZ)

    settle = rows[-1]
    assert settle["phase"] == "settle"
    assert settle["query_index"] is None
    # DROID space: open gripper is 0.0 regardless of the embodiment's own
    # open-command sign, which travels separately in isaac_action.
    assert settle["action_droid"] == [0.0] * 7 + [0.0]
    assert settle["isaac_action"] == [0.0] * 7 + [1.0]

    assert trace["policy_steps"] == 8
    assert trace["settle_steps"] == 2
    assert trace["step_trace_digest"].startswith("sha256:")


def test_step_trace_digest_is_stable_and_content_bound() -> None:
    first = build_step_trace(**_synthetic_episode())
    second = build_step_trace(**_synthetic_episode())
    changed_inputs = _synthetic_episode()
    changed_inputs["joint_trace"][3] = [0.5] * 7

    assert first["step_trace_digest"] == second["step_trace_digest"]
    with pytest.raises(StepTraceError):
        # A joint trace edited after the fact no longer matches the commanded
        # rows' observed-before record and must fail closed, not re-digest.
        build_step_trace(**changed_inputs)


def test_step_trace_rejects_length_and_consistency_violations() -> None:
    short_samples = _synthetic_episode()
    short_samples["object_samples"] = short_samples["object_samples"][:-1]
    with pytest.raises(StepTraceError):
        build_step_trace(**short_samples)

    ragged = _synthetic_episode()
    ragged["joint_trace"] = ragged["joint_trace"][:-1]
    with pytest.raises(StepTraceError):
        build_step_trace(**ragged)

    partial_chunk = _synthetic_episode(policy_steps=7)
    with pytest.raises(StepTraceError):
        build_step_trace(**partial_chunk)


def test_motion_quality_constant_velocity_has_zero_accel_and_jerk() -> None:
    trace = build_step_trace(**_synthetic_episode(policy_steps=8, settle_steps=0, velocity=0.2))
    quality = derive_motion_quality(
        trace, joint_limits=JOINT_LIMITS
    )

    assert quality["observed_joint_velocity_max_abs_rad_s"] == pytest.approx(0.2)
    # Startup from rest is one real acceleration event; after it the profile is
    # constant velocity, so interior acceleration and jerk must be exactly zero.
    assert quality["interior_joint_acceleration_max_abs_rad_s2"] == pytest.approx(0.0, abs=1e-9)
    assert quality["interior_joint_jerk_max_abs_rad_s3"] == pytest.approx(0.0, abs=1e-9)
    assert quality["joint_limit_min_margin_rad"] > 0.0
    assert quality["schema_version"]


def test_motion_quality_flags_chunk_boundary_discontinuity() -> None:
    inputs = _synthetic_episode(policy_steps=8, settle_steps=0, velocity=0.1)
    # Second chunk commands the opposite velocity: a replanning discontinuity
    # of exactly 0.2 rad/s at the boundary between rows 3 and 4.
    joint_trace = [[0.0] * 7]
    commanded = []
    for step in range(8):
        velocity = 0.1 if step < HORIZON else -0.1
        before = list(joint_trace[-1])
        action = _commanded_action(velocity)
        action["observed_before_rad"] = before
        action["joint_position_target_rad"] = [
            value + velocity / CONTROL_HZ for value in before
        ]
        action["isaac_action"] = action["joint_position_target_rad"] + [0.0]
        commanded.append(action)
        joint_trace.append([value + velocity / CONTROL_HZ for value in before])
    inputs["joint_trace"] = joint_trace
    inputs["commanded_actions"] = commanded

    trace = build_step_trace(**inputs)
    quality = derive_motion_quality(trace, joint_limits=JOINT_LIMITS)

    assert quality["chunk_boundary_count"] == 1
    assert quality["chunk_boundary_commanded_velocity_jump_max_abs_rad_s"] == pytest.approx(0.2)
    assert quality["chunk_boundary_observed_velocity_jump_max_abs_rad_s"] == pytest.approx(0.2)
    assert quality["gripper_droid_transition_count"] == 0


def test_motion_quality_reports_end_effector_path() -> None:
    trace = build_step_trace(**_synthetic_episode(policy_steps=8, settle_steps=2))
    quality = derive_motion_quality(trace, joint_limits=JOINT_LIMITS)

    # grasp frame climbs 1 mm per step across 10 steps.
    assert quality["end_effector_path_length_m"] == pytest.approx(0.010, abs=1e-9)
    assert quality["end_effector_net_displacement_m"] == pytest.approx(0.010, abs=1e-9)
    assert math.isfinite(quality["observed_joint_jerk_rms_rad_s3"])
