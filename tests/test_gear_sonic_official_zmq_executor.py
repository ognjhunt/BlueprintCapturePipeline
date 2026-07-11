from __future__ import annotations

import hashlib
import json

import pytest

from blueprint_pipeline import gear_sonic_official_zmq_executor as executor


def _request(action: dict) -> dict:
    return {
        "step_index": 3,
        "action": action,
        "source_action_sha256": hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def test_executor_sends_78d_sonic_action_to_official_protocol_and_uses_fk(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "wbc"
    model = root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setenv(executor.ROOT_ENV, str(root))
    monkeypatch.setenv(executor.MODEL_ENV, str(model))
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}
    calls = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {
            "token_state": kwargs["motion_token"],
            "body_q_target": [0.1] * 29,
            "body_q_measured": [0.0] * 29,
            "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
            "ros_timestamp": 123,
        }

    def fk_solver(**kwargs):
        assert kwargs["model_path"] == model
        assert kwargs["body_positions"] == [0.1] * 29
        return ["joint"] * 43, [0.1] * 43, [
            {"name": "right_wrist", "x": 0.1, "y": 0.2, "z": 1.0}
        ]

    result = executor.execute(
        _request(action), transport=transport, fk_solver=fk_solver
    )

    assert len(calls) == 1
    assert len(calls[0]["motion_token"]) == 64
    assert calls[0]["left_hand"] == action["sonic_action_chunk"][64:71]
    assert calls[0]["right_hand"] == action["sonic_action_chunk"][71:78]
    assert result["status"] == "completed"
    assert result["proprioceptive_state"]["official_controller_protocol"] == 4


def test_executor_rejects_shape_only_or_nonfinite_action(tmp_path, monkeypatch) -> None:
    root = tmp_path / "wbc"
    model = root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setenv(executor.ROOT_ENV, str(root))
    monkeypatch.setenv(executor.MODEL_ENV, str(model))
    with pytest.raises(ValueError, match="dimension_or_value_invalid"):
        executor.execute(_request({"action_chunk": [0.0] * 77}))
    with pytest.raises(ValueError, match="dimension_or_value_invalid"):
        executor.execute(_request({"action_chunk": [0.0] * 77 + [float("nan")]}))
