from __future__ import annotations

import threading
import time
import json

import pytest

from blueprint_pipeline import adp009d_policy_server_worker as worker


def test_readiness_requires_a_completed_handshake_not_a_listening_port(
    monkeypatch,
) -> None:
    """One shipped server declares itself ready before it can serve at all."""

    attempts = {"n": 0}

    def _flaky(*args, **kwargs):
        del args, kwargs
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionRefusedError("not up yet")
        return {"handshake_completed": True, "candidate_policy_queried": False}

    monkeypatch.setattr(worker, "attempt_handshake", _flaky)
    monkeypatch.setattr(worker, "READINESS_POLL_SECONDS", 0.0)

    result = worker.wait_for_handshake(
        host="127.0.0.1", port=8000, timeout_seconds=30.0, process=None
    )

    assert result["readiness_attempts"] == 3
    assert result["candidate_policy_queried"] is False


def test_a_server_that_exits_is_reported_immediately_not_waited_out(
    monkeypatch,
) -> None:
    """Waiting fifteen minutes for a dead process wastes the whole run."""

    class _Dead:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(
        worker, "attempt_handshake", lambda *a, **k: pytest.fail("should not be called")
    )

    with pytest.raises(RuntimeError) as excinfo:
        worker.wait_for_handshake(
            host="127.0.0.1", port=8000, timeout_seconds=30.0, process=_Dead()
        )
    assert "exited_before_ready" in str(excinfo.value)


def test_one_blocked_attempt_cannot_starve_the_readiness_deadline(
    monkeypatch,
) -> None:
    """The GR00T client constructor hung past the worker's whole deadline."""

    never = threading.Event()

    def _blocked(*args, **kwargs):
        del args, kwargs
        never.wait(60.0)
        raise AssertionError("unreachable")

    monkeypatch.setattr(worker, "attempt_handshake", _blocked)
    monkeypatch.setattr(worker, "HANDSHAKE_ATTEMPT_TIMEOUT_SECONDS", 0.02)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as excinfo:
        worker.wait_for_handshake(
            host="127.0.0.1",
            port=5555,
            timeout_seconds=1.0,
            process=None,
            transport=worker.TRANSPORT_GROOT_ZMQ,
        )

    assert time.monotonic() - started < 0.5
    assert "handshake_attempt_timed_out" in str(excinfo.value)


def test_handshake_attempt_is_bounded_below_the_overall_readiness_deadline() -> None:
    assert 0 < worker.HANDSHAKE_ATTEMPT_TIMEOUT_SECONDS
    assert worker.HANDSHAKE_ATTEMPT_TIMEOUT_SECONDS < worker.READINESS_TIMEOUT_SECONDS


def test_failed_server_log_is_digest_bound_and_embedded_in_receipt(tmp_path) -> None:
    log = tmp_path / "adp009d_policy_server.groot_n17_droid.log"
    log.write_text("import ok\nconstructor blocked\n", encoding="utf-8")

    summary = worker._server_log_summary(log)

    assert summary["present"] is True
    assert summary["size_bytes"] == log.stat().st_size
    assert summary["sha256"].startswith("sha256:")
    assert "constructor blocked" in summary["tail"]


def test_failed_readiness_stops_the_server_before_isaac_can_start() -> None:
    """A failed JAX server must not retain the GPU beside Isaac."""

    class _RunningServer:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == worker.FAILED_SERVER_TERMINATE_TIMEOUT_SECONDS
            assert self.terminated is True
            return -15

    process = _RunningServer()
    result = worker._stop_failed_server(process)

    assert result == {"status": "terminated", "exit_code": -15}
    assert process.terminated is True


def test_openpi_readiness_validates_identity_without_inference(monkeypatch) -> None:
    observed = {"infer_calls": 0, "close_calls": 0}

    class _Spec:
        def __init__(self, **kwargs):
            observed["spec"] = kwargs

    class _Client:
        candidate_policy_queried = False

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def infer(self, observation):
            del observation
            observed["infer_calls"] += 1
            pytest.fail("readiness must not query the candidate")

        def evidence_summary(self):
            return {
                "identity_verified": True,
                "server_metadata": {"policy_id": "pi05_droid_jointpos_polaris"},
                "last_inference_evidence": None,
            }

        def close(self):
            observed["close_calls"] += 1

    monkeypatch.setattr(
        worker,
        "_openpi_adapter_types",
        lambda: (_Client, _Spec, lambda **kwargs: observed.update(kwargs)),
    )
    result = worker.attempt_handshake(
        "127.0.0.1",
        8000,
        policy_spec={"policy_id": "pi05_droid_jointpos_polaris"},
        candidate_id="pi05_droid",
    )

    assert result["handshake_completed"] is True
    assert result["candidate_policy_queried"] is False
    assert result["policy_state_advanced"] is False
    assert observed["infer_calls"] == 0
    assert observed["close_calls"] == 1


def test_groot_readiness_uses_the_identity_bound_nested_droid_adapter(monkeypatch) -> None:
    observed = {}

    class _Spec:
        pass

    class _Client:
        candidate_policy_queried = False

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def infer(self, observation):
            del observation
            pytest.fail("readiness must not query the candidate")

        def evidence_summary(self):
            return {
                "identity_verified": True,
                "transport": "groot",
                "last_inference_evidence": None,
            }

        def close(self):
            observed["close_calls"] = observed.get("close_calls", 0) + 1

    monkeypatch.setattr(
        worker,
        "_groot_adapter_types",
        lambda: (_Client, _Spec, lambda receipt, expected: dict(receipt)),
    )
    identity_receipt = {"status": "verified", "checkpoint_files_sha256": "a" * 64}

    result = worker.attempt_handshake(
        "127.0.0.1",
        5555,
        worker.TRANSPORT_GROOT_ZMQ,
        identity_receipt,
    )

    assert observed["worker_identity_receipt"] is identity_receipt
    assert result["candidate_inference_performed"] is False
    assert result["policy_adapter_evidence"]["identity_verified"] is True
    assert observed["close_calls"] == 1


def test_groot_readiness_closes_client_when_handshake_evidence_is_invalid(monkeypatch) -> None:
    observed = {"close_calls": 0}

    class _Spec:
        pass

    class _Client:
        candidate_policy_queried = False

        def __init__(self, **kwargs):
            del kwargs

        def evidence_summary(self):
            return {"identity_verified": False, "last_inference_evidence": None}

        def close(self):
            observed["close_calls"] += 1

    monkeypatch.setattr(
        worker,
        "_groot_adapter_types",
        lambda: (_Client, _Spec, lambda receipt, expected: dict(receipt)),
    )

    with pytest.raises(RuntimeError, match="identity_not_verified"):
        worker.attempt_handshake(
            "127.0.0.1",
            5555,
            worker.TRANSPORT_GROOT_ZMQ,
            {"status": "verified"},
        )

    assert observed["close_calls"] == 1


def test_invalid_groot_identity_never_launches_a_server(tmp_path, monkeypatch) -> None:
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    receipt_path = tmp_path / "server-receipt.json"

    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("invalid identity must not launch"),
    )

    exit_code = worker.main(
        [
            "--candidate-id",
            "groot_n17_droid",
            "--source-root",
            str(tmp_path / "source"),
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--python",
            "/venv/bin/python",
            "--log",
            str(tmp_path / "server.log"),
            "--receipt",
            str(receipt_path),
            "--worker-identity-receipt",
            str(identity_path),
        ]
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert receipt["status"] == "blocked"
    assert "worker_receipt_not_verified" in receipt["error"]
    assert receipt["server_pid"] is None


def test_ready_receipt_proves_zero_inference_handshake(tmp_path, monkeypatch) -> None:
    class _Process:
        pid = 1234

        def poll(self):
            return None

    policy_spec = tmp_path / "execution-spec.json"
    policy_spec.write_text(
        json.dumps({"policy_spec": {"policy_id": "pi05_droid_jointpos_polaris"}}),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "server-receipt.json"
    monkeypatch.setattr(worker, "build_serve_command", lambda **kwargs: ["server"])
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: _Process())
    monkeypatch.setattr(
        worker,
        "wait_for_handshake",
        lambda **kwargs: {
            "handshake_completed": True,
            "candidate_policy_queried": False,
            "candidate_inference_performed": False,
            "policy_state_advanced": False,
            "readiness_method": (
                "identity_bound_transport_handshake_without_inference"
            ),
        },
    )

    exit_code = worker.main(
        [
            "--candidate-id",
            "pi05_droid",
            "--source-root",
            str(tmp_path / "source"),
            "--checkpoint-root",
            str(tmp_path / "checkpoint"),
            "--python",
            "/venv/bin/python",
            "--log",
            str(tmp_path / "server.log"),
            "--receipt",
            str(receipt_path),
            "--policy-spec",
            str(policy_spec),
            "--checkpoint-inventory",
            str(tmp_path / "inventory.json"),
        ]
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert receipt["schema_version"] == "adp009d_policy_server_worker.v2"
    assert receipt["status"] == "ready"
    assert receipt["handshake_completed"] is True
    assert receipt["candidate_policy_queried"] is False
    assert receipt["candidate_inference_performed"] is False
    assert receipt["policy_state_advanced"] is False


def test_readiness_source_contains_no_synthetic_observation_or_inference() -> None:
    import inspect

    source = inspect.getsource(worker.attempt_handshake)
    assert "observation" not in source
    assert ".infer(" not in source
    assert "candidate_policy_queried" in source


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


def test_the_openpi_lane_serves_identity_not_openpis_stock_server() -> None:
    """The stock server can never satisfy the episode, so it is not launchable.

    `scripts/serve_policy.py` publishes empty websocket metadata by design,
    while `OpenPIWebsocketDroidPolicyClient` validates fourteen identity fields.
    Launching it meant paying for a full Isaac boot and scene build before the
    episode refused the server. This lane now launches Blueprint's
    identity-bound wrapper around the same pinned upstream server -- the
    pattern the cosmos lane already proved.
    """

    command = worker.build_serve_command(
        candidate_id="pi05_droid",
        python="/venv/bin/python",
        source_root="/source/pi05_droid",
        checkpoint_root="/checkpoints/pi05_droid",
        port=8000,
        policy_spec_path="/runtime/adp009d_policy_execution_spec.json",
        checkpoint_inventory_path="/runtime/adp009d_openpi_checkpoint_inventory.json",
        runtime_dir="/runtime",
    )

    assert command[:2] == [
        "/venv/bin/python",
        "/runtime/openpi_droid_policy_runtime.py",
    ]
    # The stock entrypoint and its tyro CLI must be gone, not merely unused.
    assert not any("serve_policy.py" in part for part in command)
    assert not any(part.startswith("policy:checkpoint") for part in command)
    assert command[command.index("--policy-spec") + 1] == (
        "/runtime/adp009d_policy_execution_spec.json"
    )
    assert command[command.index("--checkpoint-inventory") + 1] == (
        "/runtime/adp009d_openpi_checkpoint_inventory.json"
    )
    assert command[command.index("--checkpoint-dir") + 1] == "/checkpoints/pi05_droid"
    assert command[command.index("--port") + 1] == "8000"
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_openpi_lane_refuses_to_launch_without_its_identity_inputs() -> None:
    """Absent identity inputs must stop the launch, not fall back to stock.

    A fallback here is worth nothing: the only other command that could be
    produced is a server the episode is guaranteed to refuse.
    """

    import pytest as _pytest

    with _pytest.raises(RuntimeError) as excinfo:
        worker.build_serve_command(
            candidate_id="pi05_droid",
            python="/venv/bin/python",
            source_root="/source/pi05_droid",
            checkpoint_root="/checkpoints/pi05_droid",
            port=8000,
        )

    assert "policy_server_identity_inputs_missing" in str(excinfo.value)
    assert "checkpoint_inventory" in str(excinfo.value)
    assert "policy_spec" in str(excinfo.value)


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
    assert groot[:2] == [
        "/venv/bin/python",
        "/source/groot_n17_droid/gr00t/eval/run_gr00t_server.py",
    ]
    assert "gr00t.policy.server" not in groot
    assert groot[groot.index("--embodiment-tag") + 1] == (
        "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    )
    assert groot[groot.index("--host") + 1] == "127.0.0.1"
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
    # And it uses the identity-bound DROID modality adapter rather than sending
    # Blueprint's flat observation directly to GR00T's nested API.
    assert "GrootN17DroidPolicyClient" in source
    assert "worker_identity_receipt" in source


def test_readiness_refuses_a_server_the_episode_would_refuse(monkeypatch) -> None:
    """A stock server passes a bare probe and is then refused by the episode.

    `build_serve_command` launches OpenPI's stock `serve_policy.py`, which
    publishes no identity metadata. The arena policy worker builds
    `OpenPIWebsocketDroidPolicyClient`, whose __init__ validates 14 identity
    fields plus local checkpoint verification. Readiness previously used a raw
    client with no identity check, so it passed -- and the run then died at
    `inputs_verified` after a full Isaac boot and scene build, with zero policy
    queries. Fail at the cheap end instead.
    """

    class _Spec:
        def __init__(self, **kwargs):
            del kwargs

    class _RefusingClient:
        def __init__(self, **kwargs):
            del kwargs
            raise ValueError("policy_server_metadata_mismatch")

    monkeypatch.setattr(
        worker,
        "_openpi_adapter_types",
        lambda: (_RefusingClient, _Spec, lambda **kwargs: None),
    )
    with pytest.raises(ValueError, match="policy_server_metadata_mismatch"):
        worker.attempt_handshake(
            host="127.0.0.1",
            port=8000,
            policy_spec={"policy_id": "pi05_droid_jointpos_polaris"},
            candidate_id="pi05_droid",
        )


def test_serve_command_uses_the_shared_arena_device_constant() -> None:
    """Server and Isaac co-reside; they must name one device, not two literals.

    This was the only bare "cuda:0" left in the policy path, and it is the one
    place where the policy server and the simulator must agree about which card
    they share. A device disagreement is the failure that consumed r6-r11 in the
    construction link.
    """

    import inspect

    from blueprint_pipeline import adp009d_policy_server_worker
    from blueprint_pipeline.native_task_isaaclab_launch import (
        NATIVE_TASK_ARENA_DEVICE,
    )

    source = inspect.getsource(adp009d_policy_server_worker)
    assert '"cuda:0"' not in source
    assert "'cuda:0'" not in source
    assert "NATIVE_TASK_ARENA_DEVICE" in source
    assert NATIVE_TASK_ARENA_DEVICE == "cuda:0"


def test_readiness_accepts_a_server_that_publishes_identity(monkeypatch) -> None:
    """The counterpart to the stock-server refusal: identity must let it through.

    A gate that only ever refuses is indistinguishable from a broken lane, so
    the passing direction is pinned too.
    """

    class _Spec:
        def __init__(self, **kwargs):
            del kwargs

    class _IdentityBoundClient:
        candidate_policy_queried = False

        def __init__(self, **kwargs):
            del kwargs

        def evidence_summary(self):
            return {
                "identity_verified": True,
                "server_metadata": {"policy_id": "pi05_droid_jointpos_polaris"},
                "last_inference_evidence": None,
            }

        def close(self):
            pass

    monkeypatch.setattr(
        worker,
        "_openpi_adapter_types",
        lambda: (_IdentityBoundClient, _Spec, lambda **kwargs: None),
    )
    result = worker.attempt_handshake(
        host="127.0.0.1",
        port=8000,
        policy_spec={"policy_id": "pi05_droid_jointpos_polaris"},
        candidate_id="pi05_droid",
    )

    assert result["handshake_completed"] is True
    assert result["server_metadata"]["policy_id"] == "pi05_droid_jointpos_polaris"


def test_provisioning_passes_the_staged_identity_inputs_to_the_server() -> None:
    """A fix that never reaches the runtime is worth nothing.

    The wiring is only real if the generated worker script actually carries the
    staged paths, so assert against the emitted script rather than the builder.
    """

    from blueprint_pipeline.adp009d_policy_provisioning import (
        CHECKPOINT_INVENTORY_STAGED_NAME,
        POLICY_EXECUTION_SPEC_STAGED_NAME,
        build_provisioning_script,
    )

    script = build_provisioning_script("pi05_droid")

    assert f'--policy-spec "$RUNTIME_DIR/{POLICY_EXECUTION_SPEC_STAGED_NAME}"' in script
    assert (
        f'--checkpoint-inventory "$RUNTIME_DIR/{CHECKPOINT_INVENTORY_STAGED_NAME}"'
        in script
    )
    # The stock server must not survive anywhere in the emitted lane.
    assert "serve_policy.py" not in script

    # GR00T keeps its own launch and must not acquire openpi's identity flags.
    groot = build_provisioning_script("groot_n17_droid")
    assert "--policy-spec" not in groot
    assert "--worker-identity-receipt" in groot
