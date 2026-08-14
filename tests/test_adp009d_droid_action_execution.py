from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_action_execution import (
    ACTION_SPACE_JOINT_POSITION,
    BLOCKER_GRIPPER_CONVENTION_UNMEASURED,
    DROID_OPEN_LOOP_HORIZON,
    DroidActionExecutionError,
    GripperConvention,
    build_gripper_convention_probe_request,
    droid_row_to_isaac_action,
    isaac_steps_per_droid_action,
    plan_chunk_execution,
    validate_action_chunk,
)

# Franka limits, wide enough that only deliberate tests clamp.
_LIMITS = [[-2.9, 2.9]] * 7
_ZERO_JOINTS = [0.0] * 7
_MEASURED = GripperConvention(closed_command=1.0, open_command=0.0, measured_by_probe=True)


def _chunk(rows: int = 10, gripper: float = 0.0) -> np.ndarray:
    chunk = np.zeros((rows, 8), dtype=float)
    for index in range(rows):
        chunk[index, :7] = 0.01 * index
        chunk[index, 7] = gripper
    return chunk


def test_isaac_env_already_runs_at_droid_control_rate() -> None:
    """sim.dt 1/120 with decimation 8 is exactly 1/15 s, so one row is one step."""

    assert isaac_steps_per_droid_action() == 1


def test_a_fractional_control_ratio_fails_closed() -> None:
    """A non-integer ratio drifts the policy timeline against the simulator."""

    with pytest.raises(DroidActionExecutionError) as excinfo:
        isaac_steps_per_droid_action(sim_dt_seconds=1.0 / 120.0, decimation=7)
    assert any("control_hz" in e for e in excinfo.value.errors)

    # An exact integer multiple is fine: half-rate stepping is two steps per action.
    assert isaac_steps_per_droid_action(sim_dt_seconds=1.0 / 240.0, decimation=8) == 2


def test_gripper_convention_must_be_measured_never_defaulted() -> None:
    """An inverted convention turns every commanded grasp into a release."""

    unmeasured = GripperConvention(closed_command=1.0, open_command=0.0)
    with pytest.raises(DroidActionExecutionError) as excinfo:
        droid_row_to_isaac_action(
            _chunk()[0],
            current_joint_position=_ZERO_JOINTS,
            joint_limits=_LIMITS,
            gripper=unmeasured,
        )
    assert BLOCKER_GRIPPER_CONVENTION_UNMEASURED in excinfo.value.errors


def test_droid_gripper_scalar_maps_through_the_measured_convention() -> None:
    """DROID: scalar in [0,1], above 0.5 is closed."""

    closed = droid_row_to_isaac_action(
        _chunk(gripper=0.9)[0],
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
    )
    assert closed["gripper_closed"] is True
    assert closed["isaac_action"][7] == 1.0

    opened = droid_row_to_isaac_action(
        _chunk(gripper=0.1)[0],
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
    )
    assert opened["gripper_closed"] is False
    assert opened["isaac_action"][7] == 0.0

    # An inverted convention must produce inverted commands, not be silently fixed.
    inverted = GripperConvention(
        closed_command=0.0, open_command=1.0, measured_by_probe=True
    )
    flipped = droid_row_to_isaac_action(
        _chunk(gripper=0.9)[0],
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=inverted,
    )
    assert flipped["gripper_closed"] is True
    assert flipped["isaac_action"][7] == 0.0


def test_joint_targets_are_clamped_to_limits_and_the_clamp_is_reported() -> None:
    row = np.zeros(8)
    row[:7] = [5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current = [2.8, -2.8, 0.0, 0.0, 0.0, 0.0, 0.0]

    result = droid_row_to_isaac_action(
        row,
        current_joint_position=current,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
    )

    assert result["joint_position_target_rad"][0] == pytest.approx(2.9)
    assert result["joint_position_target_rad"][1] == pytest.approx(-2.9)
    assert result["joint_limit_clamped"] is True

    within = droid_row_to_isaac_action(
        _chunk()[0],
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
    )
    assert within["joint_limit_clamped"] is False


def test_groot_decoded_absolute_joints_are_not_integrated_as_velocities() -> None:
    """NVIDIA decodes RELATIVE model values back to raw absolute actions."""

    row = np.asarray([0.7, -0.8, 0.3, -1.2, 0.4, 1.1, -0.2, 1.0])
    result = droid_row_to_isaac_action(
        row,
        current_joint_position=[0.1] * 7,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
        action_space=ACTION_SPACE_JOINT_POSITION,
    )

    assert result["joint_position_target_rad"] == pytest.approx(row[:7])
    assert result["joint_velocity_command_rad_s"] == []
    assert result["source_action_space"] == (
        "groot_decoded_absolute_joint_position_plus_absolute_gripper"
    )
    assert result["position_adapter"] == (
        "decoded_absolute_joint_position_direct_with_limit_clamp"
    )
    assert result["position_adapter_max_joint_delta_rad"] is None

    plan = plan_chunk_execution(
        np.repeat(row[None, :], 40, axis=0),
        action_space=ACTION_SPACE_JOINT_POSITION,
    )
    assert plan["source_action_space"] == result["source_action_space"]
    assert plan["position_adapter_max_joint_delta_rad"] is None


def test_only_the_open_loop_horizon_executes_and_the_tail_is_reported() -> None:
    """Executing the tail runs the arm on predictions the policy expected to supersede."""

    plan = plan_chunk_execution(_chunk(rows=10))

    assert plan["executed_rows"] == DROID_OPEN_LOOP_HORIZON == 8
    assert plan["discarded_rows"] == 2
    assert len(plan["actions"]) == 8
    assert plan["isaac_steps_per_action"] == 1
    assert plan["control_hz"] == 15
    assert plan["candidate_policy_queried"] is True

    # Cosmos returns 32x8; the horizon does not change with chunk length.
    wide = plan_chunk_execution(_chunk(rows=32))
    assert wide["executed_rows"] == 8
    assert wide["discarded_rows"] == 24


def test_malformed_chunks_never_reach_the_simulator() -> None:
    for bad in (
        np.zeros((10, 7)),
        np.zeros((10, 9)),
        np.zeros(8),
        np.zeros((10, 8, 2)),
    ):
        with pytest.raises(DroidActionExecutionError):
            validate_action_chunk(bad)

    nonfinite = _chunk()
    nonfinite[3, 2] = np.nan
    with pytest.raises(DroidActionExecutionError):
        validate_action_chunk(nonfinite)

    # A chunk shorter than the horizon cannot be executed open-loop.
    with pytest.raises(DroidActionExecutionError) as excinfo:
        validate_action_chunk(_chunk(rows=4))
    assert any("horizon" in e for e in excinfo.value.errors)


def test_probe_request_refuses_to_default_on_an_ambiguous_result() -> None:
    request = build_gripper_convention_probe_request()

    assert request["candidate_commands"] == [0.0, 1.0]
    assert "finger_joint" in request["observed_joint_names"]
    assert "fails closed" in request["decision_rule"]


def test_plan_matches_the_repository_droid_velocity_action_semantics() -> None:
    """Velocity integration and the 0.5 gripper threshold agree with the bridge."""

    from blueprint_pipeline.droid_policy_bridge import (
        droid_action_to_mujoco_targets,
    )

    row = np.zeros(8)
    row[:7] = [5.0, -5.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    row[7] = 0.9

    mine = droid_row_to_isaac_action(
        row,
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
    )
    theirs = droid_action_to_mujoco_targets(
        row,
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
    )

    assert mine["joint_position_target_rad"] == theirs["joint_position_target_rad"]
    assert mine["joint_limit_clamped"] == theirs["joint_limit_clamped"]
    assert mine["source_action_space"] == "droid_joint_velocity_plus_absolute_gripper"
    # Both read "closed" from the same scalar, even though the command differs.
    assert mine["gripper_closed"] is True
    assert theirs["gripper_position_target_m"] == 0.0


def test_runtime_measures_the_gripper_convention_rather_than_assuming_it() -> None:
    """The executor refuses to run without this, so the runtime must supply it."""

    import inspect
    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime
    from blueprint_pipeline.adp009d_newton_gripper_drive import (
        measure_gripper_convention_and_newton_drive,
    )

    runtime_source = Path(runtime.__file__).read_text(encoding="utf-8")
    measurement_source = inspect.getsource(measure_gripper_convention_and_newton_drive)

    # The runtime must invoke the sealed measurement helper rather than carry a
    # second, drifting implementation of the convention probe.
    assert "measure_gripper_convention_and_newton_drive(" in runtime_source
    # Both candidate commands are actually applied, not inferred from one.
    assert "for command in (0.0, 1.0):" in measurement_source
    # The decision comes from measured finger separation, not a constant.
    assert "finger_separation_m" in measurement_source
    assert (
        'closes_at = 1.0 if separations["1.0"] < separations["0.0"] else 0.0'
        in measurement_source
    )
    # An indistinguishable result stays unmeasured rather than guessed.
    assert 'status="ambiguous"' in measurement_source
    assert "gripper_convention_travel_below_floor" in measurement_source
    # The helper resets for both measurements, and the runtime resets once more
    # afterward so the probe cannot leave the canonical hold state altered.
    assert "env.reset(seed=20260806)" in measurement_source
    probe = runtime_source[runtime_source.index("--- gripper convention probe") :]
    probe = probe[: probe.index('timings_seconds["gripper_convention_probe"]')]
    assert probe.count("env.reset(seed=20260806)") >= 1
    assert "gripper_convention_probe" in runtime_source


def test_runtime_phase_timings_are_closed_before_the_next_phase_starts() -> None:
    """A reused timer made camera and approach durations copy later work."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    camera_start = source.index("camera_retention_started = time.monotonic()")
    camera_end = source.index('timings_seconds["camera_retention"]', camera_start)
    gripper_start = source.index('_phase("gripper_convention_probe")', camera_start)
    assert camera_start < camera_end < gripper_start

    approach_start = source.index("wrist_approach_started = time.monotonic()")
    approach_end = source.index('timings_seconds["wrist_approach"]', approach_start)
    policy_start = source.index('_phase("policy_episode")', approach_start)
    assert approach_start < approach_end < policy_start


def test_a_measured_probe_result_constructs_a_usable_convention() -> None:
    """The probe's output shape must feed GripperConvention directly."""

    probe = {
        "status": "measured",
        "closed_command": 1.0,
        "open_command": 0.0,
        "separation_travel_m": 0.0412,
    }
    convention = GripperConvention(
        closed_command=probe["closed_command"],
        open_command=probe["open_command"],
        measured_by_probe=probe["status"] == "measured",
    )

    closed = droid_row_to_isaac_action(
        _chunk(gripper=0.9)[0],
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=convention,
    )
    assert closed["isaac_action"][7] == 1.0

    # An ambiguous probe must not yield a usable convention.
    ambiguous = GripperConvention(
        closed_command=1.0,
        open_command=0.0,
        measured_by_probe=False,
    )
    with pytest.raises(DroidActionExecutionError):
        droid_row_to_isaac_action(
            _chunk()[0],
            current_joint_position=_ZERO_JOINTS,
            joint_limits=_LIMITS,
            gripper=ambiguous,
        )
