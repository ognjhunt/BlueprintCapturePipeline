"""Contract tests for the compiler's task-aware reset derivation."""

from __future__ import annotations

import json
import math

import pytest

from blueprint_pipeline import task_evaluation_diagnostic_native_arena_compiler as mod


PUSH_TARGET = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
YAW_180 = [0.0, 0.0, 1.0, 0.0]
NOMINAL = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.6283185307179586,
    "panda_joint3": 0.0,
    "panda_joint4": -2.5132741228718345,
    "panda_joint5": 0.0,
    "panda_joint6": 1.8849555921538759,
    "panda_joint7": 0.0,
    "finger_joint": 0.0,
}


def _stub(monkeypatch, tmp_path, *, phases):
    def fake_packet(*, request, evidence_root, output_dir):
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "native_task_arena_scene_plan.v1.json").write_text(
            json.dumps({"stub": True}), encoding="utf-8"
        )
        return {}

    monkeypatch.setattr(mod, "materialize_native_task_arena_packet", fake_packet)
    monkeypatch.setattr(
        mod,
        "materialize_native_task_construction_phase_plan",
        lambda plan: {"phases": phases},
    )


def _call(tmp_path, **overrides):
    kwargs = {
        "packet_request": {"robot_joint_reset_positions_rad": dict(NOMINAL)},
        "evidence_root": tmp_path,
        "derivation_root": tmp_path / "reset-derivation",
        "robot_id": "franka_panda",
        "base_pose": {"orientation_xyzw": YAW_180},
    }
    kwargs.update(overrides)
    return mod._derive_task_aware_reset(**kwargs)


def test_derivation_fires_and_solves_the_authored_orientation(monkeypatch, tmp_path):
    _stub(
        monkeypatch,
        tmp_path,
        phases=[{"phase_id": "precontact", "orientation_world_xyzw": PUSH_TARGET}],
    )
    result = _call(tmp_path)
    assert result is not None, "derivation must fire, not silently no-op"
    assert result["nominal_slew_rad"] == pytest.approx(math.pi, abs=1e-6)
    assert result["residual_slew_rad"] < math.radians(2.0)
    updated = result["joint_reset_positions_rad"]
    assert updated["finger_joint"] == 0.0
    assert updated["panda_joint1"] == pytest.approx(0.0, abs=1e-9)
    assert updated["panda_joint5"] != NOMINAL["panda_joint5"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"robot_id": ""},
        {"robot_id": "no_such_robot"},
        {"packet_request": {"robot_joint_reset_positions_rad": {}}},
        {"packet_request": {}},
    ],
)
def test_derivation_declines_rather_than_inventing(monkeypatch, tmp_path, overrides):
    _stub(
        monkeypatch,
        tmp_path,
        phases=[{"phase_id": "precontact", "orientation_world_xyzw": PUSH_TARGET}],
    )
    assert _call(tmp_path, **overrides) is None


def test_derivation_declines_without_a_phase_to_face(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, phases=[])
    assert _call(tmp_path) is None


def test_derivation_declines_when_the_phase_has_no_orientation(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, phases=[{"phase_id": "precontact"}])
    assert _call(tmp_path) is None


def test_derivation_declines_when_the_arm_joint_is_absent(monkeypatch, tmp_path):
    _stub(
        monkeypatch,
        tmp_path,
        phases=[{"phase_id": "precontact", "orientation_world_xyzw": PUSH_TARGET}],
    )
    partial = {k: v for k, v in NOMINAL.items() if k != "panda_joint4"}
    assert (
        _call(tmp_path, packet_request={"robot_joint_reset_positions_rad": partial})
        is None
    )
