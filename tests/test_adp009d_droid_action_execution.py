from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_action_execution import (
    ACTION_SPACE_JOINT_POSITION,
    ACTION_SPACE_JOINT_VELOCITY,
    BLOCKER_GRIPPER_BOUNDS,
    BLOCKER_GRIPPER_CONVENTION_UNMEASURED,
    BLOCKER_JOINT_POSITION_BOUNDS,
    BLOCKER_JOINT_VELOCITY_BOUNDS,
    DROID_OPEN_LOOP_HORIZON,
    DroidActionExecutionError,
    GripperConvention,
    SOURCE_GROOT_POSITION,
    SOURCE_PI05_POSITION,
    build_gripper_convention_probe_request,
    droid_row_to_isaac_action,
    isaac_steps_per_droid_action,
    plan_chunk_execution,
    validate_action_chunk,
    validate_candidate_action_bounds,
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
    row[:7] = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
    assert result["source_action_space"] == SOURCE_GROOT_POSITION
    assert result["position_adapter"] == (
        "decoded_absolute_joint_position_direct_within_limits"
    )
    assert result["position_adapter_max_joint_delta_rad"] is None

    plan = plan_chunk_execution(
        np.repeat(row[None, :], 40, axis=0),
        action_space=ACTION_SPACE_JOINT_POSITION,
    )
    assert plan["source_action_space"] == result["source_action_space"]
    assert plan["position_adapter_max_joint_delta_rad"] is None


def test_pi05_absolute_joints_retain_the_openpi_candidate_representation() -> None:
    row = np.asarray([0.7, -0.8, 0.3, -1.2, 0.4, 1.1, -0.2, 1.0])

    result = droid_row_to_isaac_action(
        row,
        current_joint_position=[0.1] * 7,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
        action_space=ACTION_SPACE_JOINT_POSITION,
        candidate_id="pi05_droid",
    )
    plan = plan_chunk_execution(
        np.repeat(row[None, :], 10, axis=0),
        action_space=ACTION_SPACE_JOINT_POSITION,
        candidate_id="pi05_droid",
    )

    assert result["joint_position_target_rad"] == pytest.approx(row[:7])
    assert result["source_action_space"] == SOURCE_PI05_POSITION
    assert plan["source_action_space"] == SOURCE_PI05_POSITION


def test_joint_position_utility_default_remains_groot_compatible() -> None:
    row = np.asarray([0.7, -0.8, 0.3, -1.2, 0.4, 1.1, -0.2, 1.0])

    result = droid_row_to_isaac_action(
        row,
        current_joint_position=[0.1] * 7,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
        action_space=ACTION_SPACE_JOINT_POSITION,
    )

    assert result["source_action_space"] == SOURCE_GROOT_POSITION


@pytest.mark.parametrize(
    ("candidate_id", "action_space"),
    [
        ("unknown_candidate", ACTION_SPACE_JOINT_POSITION),
        ("groot_n17_droid", ACTION_SPACE_JOINT_VELOCITY),
    ],
)
def test_explicit_candidate_action_space_mismatch_fails_closed(
    candidate_id: str, action_space: str
) -> None:
    with pytest.raises(
        DroidActionExecutionError,
        match="candidate_action_space_unsupported",
    ):
        plan_chunk_execution(
            _chunk(rows=10),
            action_space=action_space,
            candidate_id=candidate_id,
        )


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
    row[:7] = [1.0, -1.0, 0.1, 0.2, 0.3, 0.4, 0.5]
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


@pytest.mark.parametrize(
    ("action_space", "bad_dimension", "bad_value", "expected_blocker"),
    [
        (
            ACTION_SPACE_JOINT_VELOCITY,
            0,
            100.0,
            BLOCKER_JOINT_VELOCITY_BOUNDS,
        ),
        (
            ACTION_SPACE_JOINT_POSITION,
            0,
            100.0,
            BLOCKER_JOINT_POSITION_BOUNDS,
        ),
        (
            ACTION_SPACE_JOINT_VELOCITY,
            7,
            1.30,
            BLOCKER_GRIPPER_BOUNDS,
        ),
        (
            ACTION_SPACE_JOINT_POSITION,
            7,
            -0.30,
            BLOCKER_GRIPPER_BOUNDS,
        ),
    ],
)
def test_raw_candidate_bounds_refuse_instead_of_clipping(
    action_space: str,
    bad_dimension: int,
    bad_value: float,
    expected_blocker: str,
) -> None:
    chunk = np.zeros((10, 8), dtype=float)
    chunk[0, bad_dimension] = bad_value

    with pytest.raises(DroidActionExecutionError) as excinfo:
        validate_candidate_action_bounds(
            chunk,
            action_space=action_space,
            joint_limits=_LIMITS,
        )
    assert any(error.startswith(expected_blocker) for error in excinfo.value.errors)

    with pytest.raises(DroidActionExecutionError) as row_excinfo:
        droid_row_to_isaac_action(
            chunk[0],
            current_joint_position=_ZERO_JOINTS,
            joint_limits=_LIMITS,
            gripper=_MEASURED,
            action_space=action_space,
        )
    assert any(
        error.startswith(expected_blocker) for error in row_excinfo.value.errors
    )


def test_live_pi05_gripper_overshoot_validates_and_thresholds_closed() -> None:
    """Live run 20260825T125800Z: pi05 returned gripper 1.0253 on all 15 rows.

    The native DROID adapter clips this scalar to [0, 1] and binarizes at 0.5
    (``droid_policy_bridge.droid_action_to_mujoco_targets``), so the validator
    must accept the overshoot and the executed command must read "closed".
    Refusing it was a harness fault, not a policy result -- no action was
    applied and a rented L40's episode never ran.
    """

    from blueprint_pipeline.adp009d_droid_action_execution import (
        validate_candidate_action_bounds as validate,
    )

    overshoot = 1.0253319786572457
    chunk = np.zeros((15, 8), dtype=float)
    chunk[:, 7] = overshoot

    receipt = validate(
        chunk, action_space=ACTION_SPACE_JOINT_VELOCITY, joint_limits=_LIMITS
    )
    contract = receipt["gripper_contract"]
    assert contract["rows_outside_command_interval"] == 15
    assert contract["max_command_interval_overshoot"] == pytest.approx(
        overshoot - 1.0
    )
    assert contract["raw_accepted_bounds"] == [-0.25, 1.25]
    assert contract["command_interval"] == [0.0, 1.0]

    # GR00T's decoded joint-position space shares the same gripper channel.
    validate(chunk, action_space=ACTION_SPACE_JOINT_POSITION, joint_limits=_LIMITS)

    executed = droid_row_to_isaac_action(
        chunk[0],
        current_joint_position=_ZERO_JOINTS,
        joint_limits=_LIMITS,
        gripper=_MEASURED,
        action_space=ACTION_SPACE_JOINT_VELOCITY,
    )
    assert executed["gripper_closed"] is True
    assert executed["droid_gripper_scalar"] == pytest.approx(overshoot)
    assert executed["raw_candidate_bounds_validated"] is True


def test_gripper_values_inside_command_interval_report_zero_overshoot() -> None:
    from blueprint_pipeline.adp009d_droid_action_execution import (
        validate_candidate_action_bounds as validate,
    )

    receipt = validate(
        _chunk(gripper=0.9),
        action_space=ACTION_SPACE_JOINT_VELOCITY,
        joint_limits=_LIMITS,
    )
    contract = receipt["gripper_contract"]
    assert contract["rows_outside_command_interval"] == 0
    assert contract["max_command_interval_overshoot"] == 0.0


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


# --- declared per-channel contracts (company-supplied policy generalization) --
#
# With channel_contracts=None the validator is the historical DROID contract,
# proven by every test above running unmodified.  With declared contracts the
# per-channel envelope becomes data: refusal applies each channel's declared
# raw envelope, and command-interval overshoot is reported, never policed.

_TWO_CHANNEL_CONTRACTS = [
    {
        "name": "elevation_velocity",
        "kind": "bounded_continuous",
        "command_interval": [-1.0, 1.0],
        "raw_accepted_bounds": [-1.5, 1.5],
        "executed_semantics": "clipped to command interval before integration",
    },
    {
        "name": "gripper",
        "kind": "threshold_scalar",
        "command_interval": [0.0, 1.0],
        "raw_accepted_bounds": [-0.25, 1.25],
        "executed_semantics": "clip_to_command_interval_then_threshold_at_0.5",
    },
]


def test_declared_channel_contracts_validate_and_report_per_channel() -> None:
    chunk = np.zeros((6, 2), dtype=float)
    chunk[:, 0] = 1.2  # inside raw [-1.5, 1.5], outside command [-1, 1]
    chunk[:3, 1] = 1.0253  # the live pi05 overshoot, now as declared data

    receipt = validate_candidate_action_bounds(
        chunk,
        action_space="acme_two_channel_v1",
        channel_contracts=_TWO_CHANNEL_CONTRACTS,
    )

    assert receipt["action_space"] == "acme_two_channel_v1"
    assert receipt["validated_rows"] == 6
    assert receipt["raw_candidate_clipping_permitted"] is False
    applied = receipt["channel_contracts_applied"]
    assert [channel["name"] for channel in applied] == [
        "elevation_velocity",
        "gripper",
    ]
    assert applied[0]["rows_outside_command_interval"] == 6
    assert applied[0]["max_command_interval_overshoot"] == pytest.approx(0.2)
    assert applied[1]["rows_outside_command_interval"] == 3
    assert applied[1]["max_command_interval_overshoot"] == pytest.approx(0.0253)
    assert applied[1]["raw_accepted_bounds"] == [-0.25, 1.25]


def test_declared_channel_out_of_envelope_refuses_naming_the_channel() -> None:
    chunk = np.zeros((6, 2), dtype=float)
    chunk[2, 1] = 1.30  # beyond the gripper's declared raw envelope

    from blueprint_pipeline.adp009d_droid_action_execution import (
        BLOCKER_CHANNEL_BOUNDS,
    )

    with pytest.raises(DroidActionExecutionError) as excinfo:
        validate_candidate_action_bounds(
            chunk,
            action_space="acme_two_channel_v1",
            channel_contracts=_TWO_CHANNEL_CONTRACTS,
        )
    assert any(
        error.startswith(f"{BLOCKER_CHANNEL_BOUNDS}:gripper:")
        for error in excinfo.value.errors
    ), excinfo.value.errors


def test_declared_channel_width_mismatch_refuses() -> None:
    from blueprint_pipeline.adp009d_droid_action_execution import (
        BLOCKER_CHANNEL_WIDTH,
    )

    with pytest.raises(DroidActionExecutionError) as excinfo:
        validate_candidate_action_bounds(
            np.zeros((6, 3), dtype=float),
            action_space="acme_two_channel_v1",
            channel_contracts=_TWO_CHANNEL_CONTRACTS,
        )
    assert any(
        error.startswith(f"{BLOCKER_CHANNEL_WIDTH}:declared=2:chunk=3")
        for error in excinfo.value.errors
    )


def test_declared_contracts_are_not_trusted_blindly() -> None:
    """A self-contradictory envelope is a harness fault, not a policy result."""

    from blueprint_pipeline.adp009d_droid_action_execution import (
        BLOCKER_CHANNEL_CONTRACT_INVALID,
    )

    contradictory = [dict(_TWO_CHANNEL_CONTRACTS[0])]
    contradictory[0]["raw_accepted_bounds"] = [-0.5, 0.5]  # narrower than command
    with pytest.raises(DroidActionExecutionError) as excinfo:
        validate_candidate_action_bounds(
            np.zeros((6, 1), dtype=float),
            action_space="acme_one_channel_v1",
            channel_contracts=contradictory,
        )
    assert any(
        error.startswith(BLOCKER_CHANNEL_CONTRACT_INVALID)
        for error in excinfo.value.errors
    )

    with pytest.raises(DroidActionExecutionError):
        validate_candidate_action_bounds(
            np.zeros((6, 0), dtype=float),
            action_space="acme_empty_v1",
            channel_contracts=[],
        )
