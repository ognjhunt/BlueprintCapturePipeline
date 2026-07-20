from __future__ import annotations

import hashlib
import json

import pytest

from blueprint_pipeline.groot_sonic_policy_endpoint import (
    ENDPOINT_LABEL,
    make_groot_sonic_zmq_policy_endpoint,
    project_chunk_to_root_delta,
)
from blueprint_pipeline.gear_sonic_official_zmq_executor import _validated_action_frames


def _fake_run_command_factory(chunks):
    calls = []

    def fake(*, payload, policy_server_url, groot_root=None, timeout_ms=0):
        calls.append(payload)
        index = min(len(calls) - 1, len(chunks) - 1)
        return (
            {
                "status": "completed",
                "runtime_result_id": f"runtime-{index}",
                "action": {
                    "action_chunk": chunks[index],
                    "action_units": ["latent"] * len(chunks[index]),
                    "action_timing": {"control_hz": 50.0},
                },
            },
            0,
        )

    return fake, calls


def test_endpoint_projects_real_chunk_and_varies_with_observation() -> None:
    chunks = [[0.5] * 78, [-0.5] * 78]
    fake, calls = _fake_run_command_factory(chunks)
    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        run_command=fake,
    )
    first = endpoint({"camera_frame_path": "/a.jpg"}, [], 1)
    second = endpoint({"camera_frame_path": "/b.jpg"}, [first], 2)
    assert first["status"] == "completed"
    assert first["policy_action"] == "UNITREE_G1_SONIC"
    assert first["endpoint"] == ENDPOINT_LABEL
    assert first["out_of_distribution_action_projection"] is False
    assert first["not_a_learned_robot_policy_action"] is False
    assert first["claim_boundary"]["task_success_proven"] is False
    assert first["root_position"] != second["root_position"]
    assert calls[0]["observation"]["camera_frame_path"] == "/a.jpg"
    assert calls[1]["observation"]["camera_frame_path"] == "/b.jpg"
    assert second["sonic_action_chunk_dim"] == 78


def test_endpoint_preserves_selected_frame_and_hashed_horizon_metadata() -> None:
    selected = [float(index) for index in range(78)]
    units = ["latent"] * 64 + ["rad"] * 14
    timing = {
        "control_hz": 50.0,
        "sample_period_seconds": 0.02,
        "selected_horizon_frame_index": 0,
        "source_horizon_frame_count": 40,
    }
    horizon = {
        "schema_version": "unitree_g1_sonic_action_horizon.v1",
        "frame_count": 40,
        "frame_dimension": 78,
        "full_dimension": 3120,
        "source_fieldwise_horizon_sha256": "a" * 64,
        "selected_frame_sha256": "b" * 64,
        "selected_frame_index": 0,
        "selection_mode": "fresh_receding_horizon_first_frame",
    }

    def fake(**_kwargs):
        return (
            {
                "status": "completed",
                "runtime_result_id": "runtime-horizon-1",
                "action": {
                    "action_chunk": selected,
                    "action_units": units,
                    "action_timing": timing,
                    "action_horizon": horizon,
                },
            },
            0,
        )

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        run_command=fake,
    )

    action = endpoint({"camera_frame_path": "/horizon.jpg"}, [], 0)

    assert action["sonic_action_chunk"] == selected
    assert action["sonic_action_chunk_dim"] == 78
    assert action["action_units"] == units
    assert action["action_timing"] == timing
    assert action["action_horizon"] == horizon
    assert action["controller_action"]["execution_frame_count"] == 1
    assert action["controller_action"]["frames"] == [selected]


def test_endpoint_explicitly_executes_full_hash_bound_model_horizon() -> None:
    frames = [
        [float(frame_index * 1000 + value_index) for value_index in range(78)]
        for frame_index in range(40)
    ]
    frames_sha256 = hashlib.sha256(
        json.dumps(frames, separators=(",", ":")).encode()
    ).hexdigest()

    def fake(**_kwargs):
        return (
            {
                "status": "completed",
                "runtime_result_id": "runtime-full-horizon-1",
                "action": {
                    "action_chunk": frames[0],
                    "action_units": ["latent"] * 64 + ["rad"] * 14,
                    "action_timing": {
                        "control_hz": 50.0,
                        "sample_period_seconds": 0.02,
                    },
                    "sonic_action_sequence": {
                        "schema_version": "unitree_g1_sonic_action_sequence.v1",
                        "frame_count": 40,
                        "frame_dimension": 78,
                        "control_hz": 50.0,
                        "sample_period_seconds": 0.02,
                        "frames": frames,
                        "frames_sha256": frames_sha256,
                        "source_fieldwise_horizon_sha256": "a" * 64,
                    },
                },
            },
            0,
        )

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        execution_frame_count=40,
        run_command=fake,
    )

    action = endpoint({"camera_frame_path": "/horizon.jpg"}, [], 0)

    controller = action["controller_action"]
    assert controller["schema_version"] == "gear_sonic_controller_action_sequence.v1"
    assert controller["execution_mode"] == "bounded_model_horizon_prefix"
    assert controller["execution_frame_count"] == 40
    assert controller["source_horizon_frame_count"] == 40
    assert controller["frames"] == frames
    assert controller["frames_sha256"] == frames_sha256
    assert controller["source_frames_sha256"] == frames_sha256
    assert controller["execution_duration_seconds"] == pytest.approx(0.8)
    assert action["sonic_action_execution_frame_count"] == 40
    assert action["sonic_action_execution_frames_sha256"] == frames_sha256


def test_endpoint_preserves_non_round_float32_frame_zero_for_official_executor() -> None:
    frame_zero = [-0.13916015625] + [float(index) / 1024.0 for index in range(1, 78)]
    frames = [frame_zero, [value + 0.0009765625 for value in frame_zero]]
    frames_sha256 = hashlib.sha256(
        json.dumps(frames, separators=(",", ":")).encode()
    ).hexdigest()

    def fake(**_kwargs):
        return (
            {
                "status": "completed",
                "runtime_result_id": "runtime-float32-horizon-1",
                "action": {
                    "action_chunk": frame_zero,
                    "action_units": ["latent"] * 64 + ["rad"] * 14,
                    "action_timing": {"control_hz": 50.0, "sample_period_seconds": 0.02},
                    "sonic_action_sequence": {
                        "schema_version": "unitree_g1_sonic_action_sequence.v1",
                        "frame_count": 2,
                        "frame_dimension": 78,
                        "control_hz": 50.0,
                        "sample_period_seconds": 0.02,
                        "frames": frames,
                        "frames_sha256": frames_sha256,
                        "source_fieldwise_horizon_sha256": "a" * 64,
                    },
                },
            },
            0,
        )

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        execution_frame_count=2,
        run_command=fake,
    )

    action = endpoint({"camera_frame_path": "/horizon.jpg"}, [], 0)
    validated_frames, contract = _validated_action_frames(action)

    assert action["sonic_action_chunk"][0] == -0.13916015625
    assert action["sonic_action_chunk"] == action["controller_action"]["frames"][0]
    assert validated_frames == frames
    assert contract["frames_sha256"] == frames_sha256


def test_endpoint_full_horizon_fails_closed_on_sequence_hash_mismatch() -> None:
    frames = [[0.0] * 78, [1.0] * 78]

    def fake(**_kwargs):
        return (
            {
                "status": "completed",
                "action": {
                    "action_chunk": frames[0],
                    "sonic_action_sequence": {
                        "schema_version": "unitree_g1_sonic_action_sequence.v1",
                        "frame_count": 2,
                        "frame_dimension": 78,
                        "control_hz": 50.0,
                        "frames": frames,
                        "frames_sha256": "0" * 64,
                    },
                },
            },
            0,
        )

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        execution_frame_count=2,
        run_command=fake,
    )
    with pytest.raises(RuntimeError, match="sonic_action_sequence_sha256_mismatch"):
        endpoint({"camera_frame_path": "/horizon.jpg"}, [], 0)


def test_endpoint_fails_closed_on_blocked_server() -> None:
    def fake(**_kwargs):
        return ({"status": "blocked", "blockers": ["x"]}, 1)

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        run_command=fake,
    )
    with pytest.raises(RuntimeError, match="groot_sonic_requery_blocked"):
        endpoint({}, [], 0)


def test_endpoint_fails_closed_on_empty_completed_action_chunk() -> None:
    def fake(**_kwargs):
        return ({"status": "completed", "action": {"action_chunk": []}}, 0)

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"measured": [1.0]},
        run_command=fake,
    )
    with pytest.raises(RuntimeError, match="blocked_empty_sonic_action_chunk"):
        endpoint({"camera_frame_path": "/a.jpg"}, [], 1)


def test_endpoint_requires_attempt_bound_initial_state_and_carries_it() -> None:
    captured = {}

    def fake(*, payload, **_kwargs):
        captured.update(payload["observation"])
        return (
            {
                "status": "completed",
                "runtime_result_id": "runtime-1",
                "action": {"action_chunk": [0.1] * 78},
            },
            0,
        )

    endpoint = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550",
        sonic_state={"left_arm": [0.0] * 7, "projected_gravity": [0.0, 0.0, -1.0]},
        run_command=fake,
    )
    endpoint({"camera_frame_path": "/f.jpg"}, [], 0)
    state = captured["unitree_g1_sonic_state"]
    assert len(state["left_arm"]) == 7 and len(state["projected_gravity"]) == 3
    assert captured["unitree_g1_sonic_state_source"] == (
        "attempt_bound_initial_simulator_proprioception"
    )
    assert captured["unitree_g1_sonic_state_metadata"]["surrogate"] is False

    missing = make_groot_sonic_zmq_policy_endpoint(
        policy_server_url="tcp://127.0.0.1:5550", run_command=fake
    )
    with pytest.raises(RuntimeError, match="proprio_missing"):
        missing({"camera_frame_path": "/f.jpg"}, [], 0)


def test_projection_is_deterministic_and_bounded() -> None:
    dx, dy, dyaw = project_chunk_to_root_delta([1.0] * 78)
    assert (dx, dy, dyaw) == project_chunk_to_root_delta([1.0] * 78)
    assert abs(dx) <= 0.06 and abs(dy) <= 0.06 and abs(dyaw) <= 0.15
