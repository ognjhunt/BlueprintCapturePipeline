"""Pin the joint-command bounds the native construction lane actually executes.

Three production GPU runs of the arena construction lane reported byte-identical
summed arm travel while the sealed affordance's ``max_joint_delta_rad`` and
``max_joint_setpoint_lead_rad`` differed between them.  The cause was plumbing,
not physics: the phase-plan compiler validated both bounds and then dropped them
from ``execution_parameters``, and the worker called the pose servo without
them, so every run silently executed the servo's own defaults.  Identical travel
under a changed configuration is the signature of a configuration that never
arrived, so those runs cannot be read as evidence about the actuator.

These tests pin the whole path: the sealed value reaches the plan, the plan
reaches the servo call, an unexecutable pair is rejected off-GPU, and controls
replay the same bounds construction qualified.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from blueprint_pipeline import native_task_arena_construction_worker as worker
from blueprint_pipeline.native_franka_action_math import (
    NativeFrankaActionMathError,
    bounded_absolute_joint_setpoint,
)
from blueprint_pipeline.native_task_construction_plan import (
    MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD,
    NativeTaskConstructionPlanError,
    joint_command_limits,
)
from blueprint_pipeline.native_task_control_plan import (
    MAX_JOINT_DELTA_RAD as CONTROL_MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD as CONTROL_MAX_JOINT_SETPOINT_LEAD_RAD,
)
from blueprint_pipeline.native_articulated_control_plan import (
    MAX_JOINT_DELTA_RAD as ARTICULATED_MAX_JOINT_DELTA_RAD,
    MAX_JOINT_SETPOINT_LEAD_RAD as ARTICULATED_MAX_JOINT_SETPOINT_LEAD_RAD,
)


def test_construction_and_controls_share_one_command_limit_source() -> None:
    """Controls replay the duration construction qualified, so the bounds that
    shaped that motion have to be the same object, not two equal literals that
    can drift apart in a later edit."""

    assert CONTROL_MAX_JOINT_DELTA_RAD is MAX_JOINT_DELTA_RAD
    assert CONTROL_MAX_JOINT_SETPOINT_LEAD_RAD is MAX_JOINT_SETPOINT_LEAD_RAD
    assert ARTICULATED_MAX_JOINT_DELTA_RAD is MAX_JOINT_DELTA_RAD
    assert ARTICULATED_MAX_JOINT_SETPOINT_LEAD_RAD is MAX_JOINT_SETPOINT_LEAD_RAD


def test_compiler_rejects_a_lead_below_the_slew_before_any_paid_run() -> None:
    """``bounded_absolute_joint_setpoint`` rejects this pair mid-episode; the
    compiler must reject it while the run is still free."""

    with pytest.raises(NativeFrankaActionMathError):
        bounded_absolute_joint_setpoint(
            measured_joint_positions_rad=[0.0],
            desired_joint_positions_rad=[1.0],
            previous_commanded_joint_positions_rad=[0.0],
            max_command_slew_per_step_rad=0.10,
            max_setpoint_lead_rad=0.05,
        )

    with pytest.raises(NativeTaskConstructionPlanError):
        joint_command_limits(
            max_joint_delta_rad=0.10,
            max_joint_setpoint_lead_rad=0.05,
            error="limits_invalid",
        )


@pytest.mark.parametrize(
    "delta,lead",
    [(0.0, 0.20), (-0.03, 0.20), (0.03, 0.0), (0.03, None), (float("nan"), 0.20)],
)
def test_compiler_rejects_an_unusable_command_limit_pair(delta, lead) -> None:
    with pytest.raises(NativeTaskConstructionPlanError):
        joint_command_limits(
            max_joint_delta_rad=delta,
            max_joint_setpoint_lead_rad=lead,
            error="limits_invalid",
        )


def test_compiler_preserves_a_raised_pair_verbatim() -> None:
    assert joint_command_limits(
        max_joint_delta_rad=0.10,
        max_joint_setpoint_lead_rad=1.00,
        error="limits_invalid",
    ) == {"max_joint_delta_rad": 0.10, "max_joint_setpoint_lead_rad": 1.00}


@pytest.mark.parametrize(
    "execution",
    [
        {},
        {"max_joint_delta_rad": 0.03},
        {"max_joint_setpoint_lead_rad": 0.20},
        {"max_joint_delta_rad": 0.0, "max_joint_setpoint_lead_rad": 0.20},
        {"max_joint_delta_rad": 0.03, "max_joint_setpoint_lead_rad": 0.01},
        {"max_joint_delta_rad": "0.03", "max_joint_setpoint_lead_rad": None},
    ],
)
def test_worker_fails_closed_instead_of_defaulting_the_command_limits(
    execution: dict,
) -> None:
    with pytest.raises(RuntimeError, match="servo_command_limit"):
        worker._servo_command_limits(execution)


def test_worker_resolves_both_planned_limits() -> None:
    assert worker._servo_command_limits(
        {
            "max_joint_delta_rad": 0.10,
            "max_joint_setpoint_lead_rad": 1.00,
            "stable_samples": 2,
        }
    ) == {"max_joint_delta_rad": 0.10, "max_joint_setpoint_lead_rad": 1.00}


def _servo_call_keywords() -> set[str]:
    source = Path(inspect.getsourcefile(worker)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "action_for_grasp_target"
    ]
    assert len(calls) == 1, "expected exactly one native pose-servo call site"
    return {keyword.arg for keyword in calls[0].keywords if keyword.arg}


def test_servo_call_site_passes_the_planned_limits_not_the_defaults() -> None:
    """The defect was a call site that omitted two keyword arguments, which no
    behavioural test could see because the omitted values equalled the servo
    defaults.  Pin the call site itself."""

    assert {
        "max_joint_delta_rad",
        "max_joint_setpoint_lead_rad",
    } <= _servo_call_keywords()


def test_servo_still_exposes_the_limits_as_explicit_parameters() -> None:
    parameters = inspect.signature(
        __import__(
            "blueprint_pipeline.native_franka_pose_servo", fromlist=["x"]
        ).NativeFrankaDifferentialIkServo.action_for_grasp_target
    ).parameters
    assert "max_joint_delta_rad" in parameters
    assert "max_joint_setpoint_lead_rad" in parameters


class _FakeData:
    def __init__(self, **values: object) -> None:
        for name, value in values.items():
            setattr(self, name, value)


class _FakeActuator:
    def __init__(self, joint_names: list[str]) -> None:
        self.joint_names = joint_names


class _FakeRobot:
    def __init__(self, *, data: _FakeData, actuators: dict, joint_names: list[str]):
        self.data = data
        self.actuators = actuators
        self.joint_names = joint_names
        self.is_fixed_base = True


def _fake_robot(**overrides: object) -> _FakeRobot:
    arm = [f"panda_joint{index}" for index in range(1, 8)]
    joint_names = [*arm, "finger_joint", "left_inner_knuckle_joint"]
    data = _FakeData(
        joint_stiffness=[[80.0] * 7 + [0.0, 0.0]],
        joint_damping=[[4.0] * 7 + [0.0, 0.0]],
        joint_effort_limits_sim=[[87.0] * 4 + [12.0] * 3 + [0.0, 0.0]],
        applied_torque=[[1.5] * 7 + [0.0, 0.0]],
        **overrides,
    )
    return _FakeRobot(
        data=data,
        actuators={"panda_arm": _FakeActuator(arm)},
        joint_names=joint_names,
    )


def test_actuator_readback_reports_arm_gains_limits_and_torque() -> None:
    readback = worker.read_native_arm_actuator_readback(
        _fake_robot(), joint_ids=list(range(7))
    )

    assert readback["joint_stiffness"] == [80.0] * 7
    assert readback["joint_damping"] == [4.0] * 7
    assert readback["joint_effort_limit_n_m"] == [87.0] * 4 + [12.0] * 3
    assert readback["joint_effort_limit_n_m_source_attribute"] == (
        "joint_effort_limits_sim"
    )
    assert readback["applied_torque_n_m"] == [1.5] * 7


def test_actuator_readback_retains_absence_instead_of_failing_a_paid_run() -> None:
    robot = _fake_robot()
    del robot.data.joint_stiffness

    readback = worker.read_native_arm_actuator_readback(
        robot, joint_ids=list(range(7))
    )

    assert readback["joint_stiffness"]["available"] is False
    assert readback["joint_damping"] == [4.0] * 7


def test_actuator_readback_names_arm_joints_left_without_an_actuator() -> None:
    """The 'Not all actuators are configured' warning is only benign when the
    uncovered joints are the passive gripper linkage."""

    benign = worker.read_native_arm_actuator_readback(
        _fake_robot(), joint_ids=list(range(7))
    )
    assert benign["arm_joint_names_without_actuator_group"] == []
    assert benign["unactuated_joint_names"] == [
        "finger_joint",
        "left_inner_knuckle_joint",
    ]

    robot = _fake_robot()
    robot.actuators = {
        "panda_arm": _FakeActuator([f"panda_joint{index}" for index in range(1, 5)])
    }
    undriven = worker.read_native_arm_actuator_readback(
        robot, joint_ids=list(range(7))
    )
    assert undriven["arm_joint_names_without_actuator_group"] == [
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]


def test_applied_arm_torque_is_none_when_the_buffer_is_absent() -> None:
    robot = _fake_robot()
    assert worker._applied_arm_torque(robot, joint_ids=list(range(7))) == [1.5] * 7
    del robot.data.applied_torque
    assert worker._applied_arm_torque(robot, joint_ids=list(range(7))) is None


def test_realized_joint_target_is_read_back_for_action_pipeline_attribution() -> None:
    """Everything between our action and the drive is Arena's.  Retaining the
    realized target separates "the command was reshaped on the way in" from
    "the command arrived and the joint could not follow it"."""

    robot = _fake_robot(joint_pos_target=[[0.25] * 7 + [0.0, 0.0]])
    assert worker._commanded_arm_joint_target(
        robot, joint_ids=list(range(7))
    ) == [0.25] * 7

    assert (
        worker._commanded_arm_joint_target(_fake_robot(), joint_ids=list(range(7)))
        is None
    )


def test_lead_clamp_pins_the_command_once_the_measured_joint_stalls() -> None:
    """Why the lead bound is load-bearing, and why a stall is self-sustaining.

    A position-controlled joint develops ``stiffness * (command - measured)``.
    The clamp caps that error at the lead, so once the joint stops moving the
    command can no longer advance past ``measured + lead`` -- the loop settles at
    a fixed, bounded torque and stays there for the rest of the phase while the
    unbounded IK solution keeps running away.  Raising the lead is the only way
    the command side can ask for more torque.
    """

    stalled_measured = [0.0]
    already_leading = [0.20]
    far_target = [10.0]

    pinned = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=stalled_measured,
        desired_joint_positions_rad=far_target,
        previous_commanded_joint_positions_rad=already_leading,
        max_command_slew_per_step_rad=0.20,
        max_setpoint_lead_rad=0.20,
    )
    released = bounded_absolute_joint_setpoint(
        measured_joint_positions_rad=stalled_measured,
        desired_joint_positions_rad=far_target,
        previous_commanded_joint_positions_rad=already_leading,
        max_command_slew_per_step_rad=0.20,
        max_setpoint_lead_rad=1.00,
    )

    assert pinned == already_leading
    assert released == [0.40]
