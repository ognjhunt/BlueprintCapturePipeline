"""Pin the velocity feedforward that stops damping braking the commanded motion.

Isaac Lab's implicit actuator computes
``stiffness * (pos_target - pos) + damping * (vel_target - vel)``.  A
position-only command leaves ``vel_target`` at zero, so the damping term brakes
in proportion to the motion we just asked for, and the joint settles where the
two terms cancel -- ``(stiffness / damping) * error``, which on this arm is
5 rad/s per rad of lag, reached while using two to three percent of the
available torque.  r21 measured exactly that: ~0.2-2.5 N*m applied against 87
and 12 N*m limits, with the commanded setpoint advancing 0.1 rad per sample and
the measured joint following at 0.04-0.05.

Declaring the intended velocity cancels that braking while tracking and leaves
the joint damped at rest, which is the standard feedforward remedy and keeps the
damping ratio intact -- this task ends in contact with a hinged door.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from blueprint_pipeline import native_franka_pose_servo as servo_module
from blueprint_pipeline.native_franka_action_math import (
    NativeFrankaActionMathError,
    implicit_pd_torque_terms,
    joint_velocity_feedforward_rad_s,
)
from blueprint_pipeline.native_task_construction_plan import (
    VELOCITY_FEEDFORWARD_SCALE,
    NativeTaskConstructionPlanError,
    joint_command_limits,
)


def test_feedforward_is_the_commanded_setpoint_advance_rate() -> None:
    assert joint_velocity_feedforward_rad_s(
        commanded_joint_positions_rad=[0.30, -0.20],
        previous_commanded_joint_positions_rad=[0.20, -0.10],
        control_period_seconds=0.05,
        scale=1.0,
    ) == pytest.approx([2.0, -2.0])


def test_feedforward_is_zero_when_the_setpoint_holds() -> None:
    """It must still damp at rest, so a held setpoint declares no velocity."""

    assert joint_velocity_feedforward_rad_s(
        commanded_joint_positions_rad=[0.30, -0.20],
        previous_commanded_joint_positions_rad=[0.30, -0.20],
        control_period_seconds=0.05,
    ) == pytest.approx([0.0, 0.0])


def test_feedforward_scale_zero_restores_position_only_commanding() -> None:
    assert joint_velocity_feedforward_rad_s(
        commanded_joint_positions_rad=[0.30],
        previous_commanded_joint_positions_rad=[0.20],
        control_period_seconds=0.05,
        scale=0.0,
    ) == pytest.approx([0.0])


@pytest.mark.parametrize(
    "period,scale",
    [(0.0, 1.0), (-0.05, 1.0), (0.05, -0.1), (0.05, 1.5), (float("nan"), 1.0)],
)
def test_feedforward_rejects_an_unusable_contract(period: float, scale: float) -> None:
    with pytest.raises(NativeFrankaActionMathError):
        joint_velocity_feedforward_rad_s(
            commanded_joint_positions_rad=[0.1],
            previous_commanded_joint_positions_rad=[0.0],
            control_period_seconds=period,
            scale=scale,
        )


def test_feedforward_cancels_the_damping_term_while_tracking() -> None:
    """The whole point: with the arm already at the commanded rate, the damping
    term contributes nothing and the stiffness term is free to close the lag."""

    terms = implicit_pd_torque_terms(
        commanded_joint_positions_rad=[0.2],
        measured_joint_positions_rad=[0.022],
        commanded_joint_velocities_rad_s=[0.88],
        measured_joint_velocities_rad_s=[0.88],
        joint_stiffness=[400.0],
        joint_damping=[80.0],
    )

    assert terms["damping_term_n_m"] == pytest.approx([0.0])
    assert terms["stiffness_term_n_m"] == pytest.approx([71.2])
    assert terms["predicted_torque_n_m"] == pytest.approx([71.2])


def test_r21_position_only_reading_is_reproduced_by_the_two_terms() -> None:
    """r21's ~0.2-2.5 N*m was not a broken buffer and not unapplied gains.

    Position-only commanding at the observed lag and speed puts the two terms
    within about 1 N*m of each other, which is the whole reading.  Pinning this
    keeps a correctly configured actuator from being misread as an unconfigured
    one again.
    """

    terms = implicit_pd_torque_terms(
        commanded_joint_positions_rad=[0.2],
        measured_joint_positions_rad=[0.022],
        commanded_joint_velocities_rad_s=[0.0],
        measured_joint_velocities_rad_s=[0.8775],
        joint_stiffness=[400.0],
        joint_damping=[80.0],
    )

    assert terms["stiffness_term_n_m"] == pytest.approx([71.2])
    assert terms["damping_term_n_m"] == pytest.approx([-70.2])
    assert abs(terms["predicted_torque_n_m"][0]) < 2.5


def test_torque_terms_reject_ragged_input() -> None:
    with pytest.raises(NativeFrankaActionMathError):
        implicit_pd_torque_terms(
            commanded_joint_positions_rad=[0.2, 0.1],
            measured_joint_positions_rad=[0.022],
            commanded_joint_velocities_rad_s=[0.0],
            measured_joint_velocities_rad_s=[0.5],
            joint_stiffness=[400.0],
            joint_damping=[80.0],
        )


def test_plan_seals_the_feedforward_scale_with_the_command_bounds() -> None:
    limits = joint_command_limits(
        max_joint_delta_rad=0.10,
        max_joint_setpoint_lead_rad=1.00,
        error="limits_invalid",
    )

    assert limits["velocity_feedforward_scale"] == pytest.approx(
        VELOCITY_FEEDFORWARD_SCALE
    )
    assert VELOCITY_FEEDFORWARD_SCALE == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [-0.1, 1.5, None, float("nan")])
def test_plan_rejects_an_out_of_range_feedforward_scale(scale) -> None:
    with pytest.raises(NativeTaskConstructionPlanError):
        joint_command_limits(
            max_joint_delta_rad=0.10,
            max_joint_setpoint_lead_rad=1.00,
            error="limits_invalid",
            velocity_feedforward_scale=scale,
        )


def test_plan_allows_disabling_the_feedforward_for_an_a_b_run() -> None:
    assert joint_command_limits(
        max_joint_delta_rad=0.10,
        max_joint_setpoint_lead_rad=1.00,
        error="limits_invalid",
        velocity_feedforward_scale=0.0,
    )["velocity_feedforward_scale"] == pytest.approx(0.0)


def _servo_source() -> str:
    return Path(inspect.getsourcefile(servo_module)).read_text(encoding="utf-8")


def test_servo_uses_the_non_deprecated_indexed_velocity_target_api() -> None:
    """``set_joint_velocity_target`` is deprecated in the pinned revision; the
    indexed form must be preferred so the lane does not warn or break."""

    source = _servo_source()
    indexed = source.index('"set_joint_velocity_target_index"')
    plain = source.index('"set_joint_velocity_target"')
    assert indexed < plain


def test_servo_declares_the_feedforward_before_returning_an_action() -> None:
    """The velocity target has to be written inside the same tick the position
    command is produced, because the caller steps the env immediately after."""

    tree = ast.parse(_servo_source())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "action_for_grasp_target"
    )
    called = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_write_joint_velocity_target" in called

    parameters = inspect.signature(
        servo_module.NativeFrankaDifferentialIkServo.action_for_grasp_target
    ).parameters
    assert "velocity_feedforward_scale" in parameters


def test_servo_clears_the_feedforward_on_reset() -> None:
    """A stale velocity target would keep driving the arm through a reset."""

    tree = ast.parse(_servo_source())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "reset_command_state"
    )
    called = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_write_joint_velocity_target" in called


def test_worker_and_controls_both_forward_the_sealed_scale() -> None:
    """Controls replay the dynamics construction qualified, so a feedforward on
    one side only would silently change the system between the two lanes."""

    for module_name in (
        "blueprint_pipeline.native_task_arena_construction_worker",
        "blueprint_pipeline.native_task_episode_environment",
    ):
        module = __import__(module_name, fromlist=["x"])
        source = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
        assert "velocity_feedforward_scale" in source, module_name
