from __future__ import annotations

import pytest

from blueprint_pipeline.adp009d_hold_trace import (
    ARM_JOINT_NAMES,
    HoldTraceError,
    classify_arm_hold_trace,
    extract_arm_effort_limits,
    extract_arm_sample,
)


class _FakeData:
    def __init__(self, **fields: object) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class _FakeRobot:
    def __init__(self, **fields: object) -> None:
        self.data = _FakeData(**fields)


HOLD_TARGET = (0.0, -0.6283185482025146, 0.0, -2.5132741928100586, 0.0, 1.884955644607544, 0.0)


def _sample(step_index: int, joint4: float, *, torque: list[float] | None = None) -> dict:
    """One trace row that only moves panda_joint4 away from the hold target."""

    positions = list(HOLD_TARGET)
    positions[3] = joint4
    row: dict = {"step_index": step_index, "joint_positions_rad": positions}
    if torque is not None:
        row["applied_torque_nm"] = torque
    return row


def test_reports_final_error_and_worst_joint() -> None:
    samples = [
        _sample(0, -2.5132741928100586),
        _sample(1, -2.0),
        _sample(2, -1.6602405309677124),
    ]

    summary = classify_arm_hold_trace(
        samples,
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
    )

    assert summary["sample_count"] == 3
    assert summary["worst_joint_name"] == "panda_joint4"
    assert summary["worst_joint_index"] == 3
    assert summary["final_maximum_error_rad"] == pytest.approx(0.8530336618, abs=1e-9)
    assert summary["maximum_error_rad_by_step"][0] == pytest.approx(0.0, abs=1e-12)


def test_a_still_growing_error_is_an_unconverged_transient() -> None:
    """The arm is still falling: the last third is meaningfully worse than the first."""

    samples = [_sample(index, -2.5132741928100586 + 0.05 * index) for index in range(12)]

    summary = classify_arm_hold_trace(
        samples,
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
    )

    assert summary["convergence"] == "diverging"
    assert summary["hold_failure_mode"] == "unconverged_transient"


def test_a_flat_out_of_tolerance_error_is_a_settled_offset() -> None:
    """The arm reached a stable wrong pose: a real steady-state drive deficit."""

    samples = [_sample(index, -2.4) for index in range(12)]

    summary = classify_arm_hold_trace(
        samples,
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
    )

    assert summary["convergence"] == "settled"
    assert summary["hold_failure_mode"] == "settled_offset"


def test_a_trace_inside_tolerance_is_not_a_failure() -> None:
    samples = [_sample(index, -2.5132741928100586) for index in range(12)]

    summary = classify_arm_hold_trace(
        samples,
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
    )

    assert summary["hold_failure_mode"] == "within_tolerance"


FRANKA_EFFORT_LIMITS_NM = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)


def _falling_trace(torque_joint4: float | None) -> list[dict]:
    """Twelve steps of panda_joint4 drifting, optionally with a torque column."""

    rows = []
    for index in range(12):
        torque = None
        if torque_joint4 is not None:
            torque = [0.0, 0.0, 0.0, torque_joint4, 0.0, 0.0, 0.0]
        rows.append(_sample(index, -2.5132741928100586 + 0.05 * index, torque=torque))
    return rows


def test_a_joint_pinned_at_its_effort_limit_is_reported_as_saturated() -> None:
    """Torque at the limit is the signature the actuator-limit fix targets."""

    summary = classify_arm_hold_trace(
        _falling_trace(87.0),
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
        effort_limits_nm=FRANKA_EFFORT_LIMITS_NM,
    )

    assert summary["hold_failure_mode"] == "effort_saturated"
    assert summary["torque"]["available"] is True
    assert summary["torque"]["saturated_joint_names"] == ["panda_joint4"]
    assert summary["torque"]["final_utilization_fraction"][3] == pytest.approx(1.0)


def test_torque_well_under_the_limit_is_not_saturation() -> None:
    """A weak-but-unsaturated drive is a different defect and must not be conflated."""

    summary = classify_arm_hold_trace(
        _falling_trace(10.0),
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
        effort_limits_nm=FRANKA_EFFORT_LIMITS_NM,
    )

    assert summary["hold_failure_mode"] == "unconverged_transient"
    assert summary["torque"]["saturated_joint_names"] == []
    assert summary["torque"]["final_utilization_fraction"][3] == pytest.approx(10.0 / 87.0)


def test_a_trace_without_torque_says_so_instead_of_guessing() -> None:
    """Absent evidence must read as absent, never as 'not saturated'."""

    summary = classify_arm_hold_trace(
        _falling_trace(None),
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
        effort_limits_nm=FRANKA_EFFORT_LIMITS_NM,
    )

    assert summary["torque"]["available"] is False
    assert summary["torque"]["unavailable_reason"] == "applied_torque_not_retained"
    assert summary["torque"]["saturated_joint_names"] == []
    assert summary["hold_failure_mode"] == "unconverged_transient"


def test_torque_without_effort_limits_cannot_claim_saturation() -> None:
    summary = classify_arm_hold_trace(
        _falling_trace(87.0),
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
    )

    assert summary["torque"]["available"] is True
    assert summary["torque"]["unavailable_reason"] == "effort_limits_not_supplied"
    assert summary["torque"]["final_utilization_fraction"] is None
    assert summary["hold_failure_mode"] == "unconverged_transient"


def test_saturation_outranks_the_kinematic_mode_even_when_settled() -> None:
    """A settled pose held against a saturated actuator is still a limit problem."""

    rows = []
    for index in range(12):
        rows.append(_sample(index, -2.4, torque=[0.0, 0.0, 0.0, 87.0, 0.0, 0.0, 0.0]))

    summary = classify_arm_hold_trace(
        rows,
        requested_joint_positions_rad=HOLD_TARGET,
        tolerance_rad=0.01,
        effort_limits_nm=FRANKA_EFFORT_LIMITS_NM,
    )

    assert summary["convergence"] == "settled"
    assert summary["hold_failure_mode"] == "effort_saturated"


def test_joint_names_cover_the_seven_arm_joints() -> None:
    assert ARM_JOINT_NAMES == (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )


def test_an_empty_trace_fails_closed() -> None:
    with pytest.raises(HoldTraceError, match="adp009d_hold_trace_empty"):
        classify_arm_hold_trace(
            [],
            requested_joint_positions_rad=HOLD_TARGET,
            tolerance_rad=0.01,
        )


def test_a_wrong_width_sample_fails_closed_rather_than_silently_truncating() -> None:
    with pytest.raises(HoldTraceError, match="adp009d_hold_trace_joint_width_invalid"):
        classify_arm_hold_trace(
            [{"step_index": 0, "joint_positions_rad": [0.0, 0.0, 0.0]}],
            requested_joint_positions_rad=HOLD_TARGET,
            tolerance_rad=0.01,
        )


def test_a_nonfinite_sample_is_a_typed_trace_gap() -> None:
    with pytest.raises(
        HoldTraceError, match="adp009d_hold_trace_joint_width_invalid_nonfinite"
    ):
        classify_arm_hold_trace(
            [_sample(0, float("nan"))],
            requested_joint_positions_rad=HOLD_TARGET,
            tolerance_rad=0.01,
        )


def test_an_invalid_saturation_threshold_fails_closed() -> None:
    with pytest.raises(
        HoldTraceError, match="adp009d_hold_trace_saturation_threshold_invalid"
    ):
        classify_arm_hold_trace(
            [_sample(0, -2.5132741928100586)],
            requested_joint_positions_rad=HOLD_TARGET,
            tolerance_rad=0.01,
            saturation_fraction=1.1,
        )


NINE_DOF_POSITIONS = [0.0, -0.63, 0.0, -2.51, 0.0, 1.88, 0.0, 0.04, 0.04]


def test_extract_arm_sample_reads_positions_and_torque() -> None:
    robot = _FakeRobot(
        joint_pos=[NINE_DOF_POSITIONS],
        applied_torque=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0]],
    )

    sample = extract_arm_sample(robot, step_index=7, to_list=list)

    assert sample["step_index"] == 7
    assert sample["joint_positions_rad"] == NINE_DOF_POSITIONS[:7]
    assert sample["applied_torque_nm"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_extract_arm_sample_falls_back_to_computed_torque() -> None:
    robot = _FakeRobot(
        joint_pos=[NINE_DOF_POSITIONS],
        computed_torque=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0]],
    )

    sample = extract_arm_sample(robot, step_index=0, to_list=list)

    assert sample["applied_torque_nm"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_extract_arm_sample_omits_torque_when_the_backend_exposes_none() -> None:
    """A backend without torque readback must still yield a usable position trace."""

    robot = _FakeRobot(joint_pos=[NINE_DOF_POSITIONS])

    sample = extract_arm_sample(robot, step_index=0, to_list=list)

    assert sample["joint_positions_rad"] == NINE_DOF_POSITIONS[:7]
    assert sample["applied_torque_nm"] is None


def test_extract_arm_sample_never_raises_when_readback_explodes() -> None:
    """Instrumentation must not be able to fail a run that would otherwise pass."""

    def _boom(_value: object) -> list[float]:
        raise RuntimeError("backend readback exploded")

    robot = _FakeRobot(joint_pos=[NINE_DOF_POSITIONS])

    assert extract_arm_sample(robot, step_index=3, to_list=_boom) is None


def test_extract_arm_effort_limits_prefers_the_simulated_limit_field() -> None:
    robot = _FakeRobot(
        joint_effort_limits_sim=[[87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 20.0, 20.0]],
        joint_effort_limits=[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
    )

    assert extract_arm_effort_limits(robot, to_list=list) == [
        87.0,
        87.0,
        87.0,
        87.0,
        12.0,
        12.0,
        12.0,
    ]


def test_extract_arm_effort_limits_returns_none_when_absent() -> None:
    assert extract_arm_effort_limits(_FakeRobot(), to_list=list) is None
