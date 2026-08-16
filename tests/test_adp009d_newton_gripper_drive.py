from __future__ import annotations

import math
import sys
import types

import pytest

from blueprint_pipeline.adp009d_newton_gripper_drive import (
    assess_newton_gripper_drive_trace,
    build_newton_gripper_drive_candidate,
    configure_newton_gripper_drive_candidate,
    measure_gripper_convention_and_newton_drive,
    validate_newton_gripper_drive_candidate,
)


def test_candidate_is_derived_from_sealed_speed_effort_and_timestep() -> None:
    candidate = build_newton_gripper_drive_candidate()

    assert validate_newton_gripper_drive_candidate(candidate) == []
    assert candidate["comparison_eligible"] is False
    assert candidate["candidate_drive"]["armature_kg_m2"] == pytest.approx(0.1375)
    target_rate = candidate["candidate_drive"]["target_rate_limit_rad_s"]
    assert target_rate == pytest.approx(0.654681991)
    assert candidate["candidate_drive"]["maximum_target_step_rad"] == pytest.approx(
        target_rate / 15
    )
    assert candidate["derivation"]["explicit_stability_ratio"] < 2.0
    assert candidate["native_acceptance"]["maximum_abs_joint_velocity_rad_s"] == 1.05
    assert candidate["derivation"]["retained_native_transient_readback_rad_s"] == (
        pytest.approx(1.4434489011764526)
    )
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
    assert passed["positions_rad"] == [0.0, 0.05, 0.10]
    assert passed["velocities_rad_s"] == [0.0, 0.8, 0.01]

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
    converted_values: list[object] = []

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
            if self.value > max:
                return max
            if self.value < min:
                return min
            return self.value

    class _FakeWarpArray:
        def __getitem__(self, key):
            if isinstance(key, tuple) and any(isinstance(item, list) for item in key):
                raise TypeError("'<' not supported between instances of 'list' and 'int'")
            return _FakeTensor(0.2)

    actions_module = types.ModuleType("isaaclab_arena.embodiments.droid.actions")
    actions_module.BinaryJointPositionZeroToOneAction = SourceAction
    monkeypatch.setitem(
        sys.modules, "isaaclab_arena.embodiments.droid.actions", actions_module
    )
    array_module = types.ModuleType("isaaclab.utils.array")

    def convert_to_torch(value: object) -> _FakeTensor:
        converted_values.append(value)
        return _FakeTensor(0.2)

    array_module.convert_to_torch = convert_to_torch
    monkeypatch.setitem(sys.modules, "isaaclab.utils.array", array_module)
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
    joint_pos = _FakeWarpArray()
    action._asset = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=joint_pos))
    action._joint_ids = [0]
    action.process_actions(None)
    assert converted_values == [joint_pos]
    target_rate = build_newton_gripper_drive_candidate()["candidate_drive"][
        "target_rate_limit_rad_s"
    ]
    assert action._processed_actions == pytest.approx(0.2 + target_rate / 15)


def test_native_probe_reads_body_names_from_pinned_articulation_data_api() -> None:
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(body_names=[]),
    )

    probe = measure_gripper_convention_and_newton_drive(
        env=object(),
        action=object(),
        robot=robot,
        torch=object(),
        to_torch=object(),
        backend="newton",
    )

    assert probe["status"] == "blocked"
    assert probe["blockers"] == ["gripper_convention_finger_bodies_missing"]
