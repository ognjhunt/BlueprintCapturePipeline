from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline import adp009d_policy_server_worker as worker


def test_readiness_requires_a_completed_round_trip_not_a_listening_port(
    monkeypatch,
) -> None:
    """One shipped server declares itself ready before it can serve at all."""

    attempts = {"n": 0}

    def _flaky(host, port):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionRefusedError("not up yet")
        return {"action_chunk_rows": 10, "action_chunk_width": 8}

    monkeypatch.setattr(worker, "attempt_round_trip", _flaky)
    monkeypatch.setattr(worker, "READINESS_POLL_SECONDS", 0.0)

    result = worker.wait_for_round_trip(
        host="127.0.0.1", port=8000, timeout_seconds=30.0, process=None
    )

    assert result["readiness_attempts"] == 3
    assert result["action_chunk_width"] == 8


def test_a_server_that_exits_is_reported_immediately_not_waited_out(
    monkeypatch,
) -> None:
    """Waiting fifteen minutes for a dead process wastes the whole run."""

    class _Dead:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(
        worker, "attempt_round_trip", lambda h, p: pytest.fail("should not be called")
    )

    with pytest.raises(RuntimeError) as excinfo:
        worker.wait_for_round_trip(
            host="127.0.0.1", port=8000, timeout_seconds=30.0, process=_Dead()
        )
    assert "exited_before_ready" in str(excinfo.value)


def test_a_malformed_chunk_is_not_readiness(monkeypatch) -> None:
    """A server answering with the wrong shape is not ready; it is broken."""

    import sys
    import types

    def _install_client(actions):
        class _Client:
            def __init__(self, **kwargs):
                pass

            def infer(self, observation):
                return {"actions": actions}

        transport = types.ModuleType("openpi_client.websocket_client_policy")
        transport.WebsocketClientPolicy = _Client
        package = types.ModuleType("openpi_client")
        package.websocket_client_policy = transport
        monkeypatch.setitem(sys.modules, "openpi_client", package)
        monkeypatch.setitem(
            sys.modules, "openpi_client.websocket_client_policy", transport
        )

    # Wrong width.
    _install_client(np.zeros((10, 7)))
    with pytest.raises(RuntimeError) as excinfo:
        worker.attempt_round_trip("127.0.0.1", 8000)
    assert "chunk_shape_invalid" in str(excinfo.value)

    # Too few rows for the open-loop horizon.
    _install_client(np.zeros((4, 8)))
    with pytest.raises(RuntimeError) as excinfo:
        worker.attempt_round_trip("127.0.0.1", 8000)
    assert "chunk_too_short" in str(excinfo.value)

    # Non-finite values.
    bad = np.zeros((10, 8))
    bad[0, 0] = np.nan
    _install_client(bad)
    with pytest.raises(RuntimeError) as excinfo:
        worker.attempt_round_trip("127.0.0.1", 8000)
    assert "nonfinite" in str(excinfo.value)

    # A well-formed chunk is accepted.
    _install_client(np.zeros((10, 8)))
    result = worker.attempt_round_trip("127.0.0.1", 8000)
    assert result["action_chunk_rows"] == 10
    assert result["action_chunk_width"] == 8


def test_the_probe_observation_matches_what_the_episode_will_send() -> None:
    """A shape mismatch must surface at startup, not mid-episode."""

    from blueprint_pipeline.adp009d_droid_observation import (
        DROID_EXTERIOR_VIEW_1,
        DROID_WRIST_VIEW,
    )
    from blueprint_pipeline.droid_policy_bridge import validate_droid_observation

    observation = worker._probe_observation()

    assert DROID_EXTERIOR_VIEW_1 in observation
    assert DROID_WRIST_VIEW in observation
    assert observation[DROID_EXTERIOR_VIEW_1].shape == (224, 224, 3)
    # It must satisfy the repository's own DROID contract.
    assert validate_droid_observation(observation) == []


def test_the_server_is_loopback_only() -> None:
    import inspect

    source = inspect.getsource(worker)
    assert "policy_server_must_be_loopback_only" in source
    assert worker.DEFAULT_HOST == "127.0.0.1"


def test_provisioning_starts_the_server_and_the_bundle_ships_the_episode() -> None:
    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT
    from blueprint_pipeline.adp009d_policy_provisioning import (
        build_provisioning_script,
    )

    script = build_provisioning_script("pi05_droid")
    assert "adp009d_policy_server_worker.py" in script
    assert "--host 127.0.0.1" in script
    # Started after the checkpoint exists, never before.
    assert script.index("adp009d_checkpoint_fetch_worker.py") < script.index(
        "adp009d_policy_server_worker.py"
    )
    # The runtime learns which candidate is bound.
    assert "BLUEPRINT_ADP009D_POLICY_CANDIDATE" in ENTRYPOINT


def test_the_runtime_runs_an_episode_only_with_a_measured_gripper() -> None:
    """An assumed convention would invert every grasp; refuse rather than guess."""

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert 'gripper_probe.get("status") == "measured"' in source
    assert "run_policy_episode(" in source
    assert "measured_by_probe=True" in source
    # Recorded, never fatal: the micro-check's evidence must survive it.
    episode = source[source.index("--- learned policy episode") :]
    assert "policy_episode_error" in episode
    assert "noqa: BLE001" in episode
