from __future__ import annotations

import math
import sys
import types

import pytest

from blueprint_pipeline.adp009d_newton_gripper_drive import (
    assess_newton_gripper_drive_trace,
    build_newton_gripper_drive_candidate,
    configure_newton_gripper_drive_candidate,
    validate_newton_gripper_drive_candidate,
)


def test_candidate_is_derived_from_sealed_speed_effort_and_timestep() -> None:
    candidate = build_newton_gripper_drive_candidate()

    assert validate_newton_gripper_drive_candidate(candidate) == []
    assert candidate["comparison_eligible"] is False
    assert candidate["candidate_drive"]["armature_kg_m2"] == pytest.approx(0.1375)
    assert candidate["candidate_drive"]["maximum_target_step_rad"] == pytest.approx(1 / 15)
    assert candidate["derivation"]["explicit_stability_ratio"] < 2.0
    assert 0.020 <= candidate["derivation"]["rated_fingertip_speed_m_s"] <= 0.150


def test_candidate_tamper_is_rejected() -> None:
    candidate = build_newton_gripper_drive_candidate()
    candidate["candidate_drive"]["armature_kg_m2"] = 0.0

    assert validate_newton_gripper_drive_candidate(candidate) == [
        "adp009d_newton_gripper_drive_candidate_invalid"
    ]


def test_native_trace_requires_finite_speed_bounded_settled_motion() -> None:
    passed = assess_newton_gripper_drive_trace(
        positions_rad=[0.0, 0.05, 0.10], velocities_rad_s=[0.0, 0.8, 0.01]
    )
    assert passed["status"] == "passed"

    blocked = assess_newton_gripper_drive_trace(
        positions_rad=[0.0, math.nan], velocities_rad_s=[0.0, 1.2]
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [
        "adp009d_newton_gripper_drive_nonfinite",
        "adp009d_newton_gripper_drive_velocity_exceeded",
        "adp009d_newton_gripper_drive_not_settled",
    ]


def test_runtime_configuration_is_newton_only_and_rate_limits_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SourceAction:
        def process_actions(self, _actions):
            self._processed_actions = _FakeTensor(1.0)

    class _FakeTensor:
        def __init__(self, value: float):
            self.value = value

        def __getitem__(self, _key):
            return self

        def __sub__(self, value: float):
            return self.value - value

        def __add__(self, value: float):
            return self.value + value

        def clamp(self, *, min: float, max: float):
            return max if self.value > max else min if self.value < min else self.value

    actions_module = types.ModuleType("isaaclab_arena.embodiments.droid.actions")
    actions_module.BinaryJointPositionZeroToOneAction = SourceAction
    monkeypatch.setitem(
        sys.modules, "isaaclab_arena.embodiments.droid.actions", actions_module
    )
    actuator = types.SimpleNamespace(
        stiffness=None, damping=None, armature=None, velocity_limit=1.0
    )
    action_cfg = types.SimpleNamespace(class_type=SourceAction)
    embodiment = types.SimpleNamespace(
        scene_config=types.SimpleNamespace(
            robot=types.SimpleNamespace(actuators={"gripper": actuator})
        ),
        action_config=types.SimpleNamespace(gripper_action=action_cfg),
    )

    receipt = configure_newton_gripper_drive_candidate(
        embodiment, expected_contract=build_newton_gripper_drive_candidate()
    )
    assert receipt["status"] == "applied_for_native_identification"
    assert actuator.armature == pytest.approx(0.1375)
    action = action_cfg.class_type()
    action._asset = types.SimpleNamespace(
        data=types.SimpleNamespace(joint_pos=_FakeTensor(0.2))
    )
    action._joint_ids = [0]
    action.process_actions(None)
    assert action._processed_actions == pytest.approx(0.2 + 1 / 15)
