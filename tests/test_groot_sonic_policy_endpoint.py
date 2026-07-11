from __future__ import annotations

import pytest

from blueprint_pipeline.groot_sonic_policy_endpoint import (
    ENDPOINT_LABEL,
    make_groot_sonic_zmq_policy_endpoint,
    project_chunk_to_root_delta,
)


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
