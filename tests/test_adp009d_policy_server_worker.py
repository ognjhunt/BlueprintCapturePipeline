from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from blueprint_pipeline import adp009d_policy_server_worker as worker


def test_readiness_requires_a_completed_round_trip_not_a_listening_port(
    monkeypatch,
) -> None:
    """One shipped server declares itself ready before it can serve at all."""

    attempts = {"n": 0}

    def _flaky(host, port, transport=None):
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


def test_one_blocked_attempt_cannot_starve_the_readiness_deadline(
    monkeypatch,
) -> None:
    """The GR00T client constructor hung past the worker's whole deadline."""

    never = threading.Event()

    def _blocked(host, port, transport=None):
        never.wait(60.0)
        raise AssertionError("unreachable")

    monkeypatch.setattr(worker, "attempt_round_trip", _blocked)
    monkeypatch.setattr(worker, "ROUND_TRIP_ATTEMPT_TIMEOUT_SECONDS", 0.02)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        worker.wait_for_round_trip(
            host="127.0.0.1",
            port=5555,
            timeout_seconds=1.0,
            process=None,
            transport=worker.TRANSPORT_GROOT_ZMQ,
        )

    assert time.monotonic() - started < 0.5
    assert "round_trip_attempt_timed_out" in str(excinfo.value)


def test_attempt_bound_has_measured_pi05_cold_start_headroom() -> None:
    """The bound must not reject the observed 53-second pi05 first inference."""

    assert worker.ROUND_TRIP_ATTEMPT_TIMEOUT_SECONDS >= 2 * 53.0
    assert worker.ROUND_TRIP_ATTEMPT_TIMEOUT_SECONDS < worker.READINESS_TIMEOUT_SECONDS


def test_failed_server_log_is_digest_bound_and_embedded_in_receipt(tmp_path) -> None:
    log = tmp_path / "adp009d_policy_server.groot_n17_droid.log"
    log.write_text("import ok\nconstructor blocked\n", encoding="utf-8")

    summary = worker._server_log_summary(log)

    assert summary["present"] is True
    assert summary["size_bytes"] == log.stat().st_size
    assert summary["sha256"].startswith("sha256:")
    assert "constructor blocked" in summary["tail"]


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
    # A batch per candidate now, so two policies can be ranked in one run.
    assert "run_episode_batch(" in source
    assert "measured_by_probe=True" in source
    # Recorded, never fatal: the micro-check's evidence must survive it.
    episode = source[source.index("--- learned policy episode") :]
    assert "policy_episode_error" in episode
    assert "noqa: BLE001" in episode


def test_the_serve_command_matches_openpis_pinned_cli() -> None:
    """Verified against scripts/serve_policy.py at the frozen source revision.

    That file uses tyro.cli(Args) where Args.policy is Checkpoint | Default and
    Checkpoint carries config and dir, so the union subcommand form is
    policy:checkpoint --policy.config=... --policy.dir=...  Its own
    DEFAULT_CHECKPOINT maps EnvMode.DROID to config="pi05_droid" against
    gs://openpi-assets/checkpoints/pi05_droid, which is the checkpoint we fetch.
    """

    command = worker.build_serve_command(
        candidate_id="pi05_droid",
        python="/venv/bin/python",
        source_root="/source/pi05_droid",
        checkpoint_root="/checkpoints/pi05_droid",
        port=8000,
    )

    assert "policy:checkpoint" in command
    assert "--policy.config=pi05_droid" in command
    assert "--policy.dir=/checkpoints/pi05_droid" in command
    # Port is a top-level Args field, not nested under policy.
    assert command[command.index("--port") + 1] == "8000"
    assert command.index("--port") < command.index("policy:checkpoint")


def test_a_skipped_episode_says_why_rather_than_vanishing() -> None:
    """A live run produced no episode and no error; nothing said the guard skipped.

    The entrypoint exported a passthrough of an unset variable, so the bound
    candidate read as empty and the runtime silently declined to run.  A run
    that produced no episode must be distinguishable from a policy that scored
    zero.
    """

    from pathlib import Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime
    from blueprint_pipeline.adp009d_native_microcheck_bundle import ENTRYPOINT

    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "policy_episode_skipped_reason" in source
    assert '"no_policy_candidate_bound"' in source
    assert '"policy_candidate_bound"' in source

    # The candidate is baked in, never a passthrough of an unset variable.
    assert "@@POLICY_CANDIDATE@@" in ENTRYPOINT
    assert "${BLUEPRINT_ADP009D_POLICY_CANDIDATE:-}" not in ENTRYPOINT


def test_each_candidate_gets_its_own_transport_and_launch() -> None:
    """openpi and GR00T share neither a transport nor a launch command.

    openpi serves a websocket via scripts/serve_policy.py; GR00T's client is
    gr00t.policy.server_client.PolicyClient over ZMQ.  A single hardcoded form
    can only ever serve one of them, which is what the first version did.
    """

    assert worker.transport_for("pi05_droid") == worker.TRANSPORT_OPENPI_WEBSOCKET
    assert worker.transport_for("groot_n17_droid") == worker.TRANSPORT_GROOT_ZMQ

    groot = worker.build_serve_command(
        candidate_id="groot_n17_droid",
        python="/venv/bin/python",
        source_root="/source/groot_n17_droid",
        checkpoint_root="/checkpoints/groot_n17_droid",
        port=5555,
    )
    assert "gr00t.policy.server" in groot
    assert "serve_policy.py" not in " ".join(groot)
    assert "--model-path" in groot

    # Default ports differ, and neither is guessed at the call site.
    assert worker.CANDIDATE_DEFAULT_PORTS[worker.TRANSPORT_OPENPI_WEBSOCKET] == 8000
    assert worker.CANDIDATE_DEFAULT_PORTS[worker.TRANSPORT_GROOT_ZMQ] == 5555

    with pytest.raises(RuntimeError):
        worker.transport_for("some_other_policy")


def test_the_episode_connects_to_the_port_that_actually_started() -> None:
    """Two transports mean two ports; a default would connect to the wrong one."""

    from pathlib import Path as _Path

    from blueprint_pipeline import adp009d_isaac_runtime as runtime

    source = _Path(runtime.__file__).read_text(encoding="utf-8")
    # Per candidate, or two candidates overwrite each other's receipt.
    assert 'f"adp009d_policy_server_receipt.{bound_candidate}.json"' in source
    assert 'server_receipt.get("status") != "ready"' in source
    # Read from the receipt the worker wrote, not a default: the episode
    # must connect to the port that actually started.
    assert 'int(receipt["port"])' in source
    # And it speaks GR00T's own client rather than assuming a websocket.
    assert "_GrootEpisodeClient" in source
    assert "get_action(observation)" in source
