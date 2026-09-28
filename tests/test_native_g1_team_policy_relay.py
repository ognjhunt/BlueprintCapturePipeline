"""Actual Unix/JSONL transport; no provider or simulator qualification."""

import json
from pathlib import Path
import socket
import tempfile
import threading

import numpy as np
import pytest

from blueprint_pipeline import native_g1_team_policy_relay as relay
from blueprint_pipeline.native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_team_policy_jsonl_client import _process, _close


SECRET = "e" * 64
BINDING = relay.RelayBinding("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                             "sha256:" + "c" * 64, "container")


class Session:
    def __init__(self):
        action = [0.0] * 40
        action[3:9] = [1, 0, 0, 1, 0, 0]
        self.process = _process("{'ok': True, 'action_chunk': " + repr([action]) + "}")
        self.client = NativeG1TeamPolicyJsonlClient(self.process, timeout_seconds=1, profile_digest=BINDING.profile_digest)
        self.profile_digest = BINDING.profile_digest
        self.delivery_mode = BINDING.delivery_mode
        self.conformance = {"status": "synthetic_wire_compatible", "profile_digest": self.profile_digest,
                            "source_setup_digest": BINDING.setup_digest,
                            "schema_version": "native_g1_team_policy_synthetic_conformance.v1",
                            "delivery_mode": self.delivery_mode, "synthetic_policy_query_count": 1,
                            "site_observation_sent": False, "robot_preset_id": "unitree_g1_dex3_sonic_v1",
                            "returned_action_count": 1, "site_policy_query_count": 0,
                            "task_scored": False, "runtime_identity_verified": False, "rights_authorized": False,
                            "paid_launch_authorized": False, "public_redistribution_authorized": False,
                            "claim_ceiling": "planning_only"}
        self.conformance["receipt_digest"] = canonical_digest(self.conformance)
        self.closed = 0

    def close(self):
        if not self.closed:
            _close(self.process)
            self.closed += 1
        value = {"schema_version": "native_g1_team_runtime_session.v1", "status": "closed",
                "profile_digest": self.profile_digest, "delivery_mode": self.delivery_mode,
                "synthetic_conformance_digest": self.conformance["receipt_digest"],
                "child_teardown_digest": "sha256:" + "d" * 64, "child_teardown_required": True,
                "linked_scored_episode_result_digest": None, "linked_episode_media_verified_by_session": False,
                "claim_ceiling": "planning_only", "provider_teardown_verified": False}
        value["receipt_digest"] = canonical_digest(value)
        return value


@pytest.fixture
def running():
    # Unix pathname limit is shorter than pytest's normal temporary path.
    with tempfile.TemporaryDirectory(prefix="g1-relay-", dir=str(Path("/tmp").resolve())) as root:
        server = relay.G1PolicyRelayServer(path=Path(root) / "wire", binding=BINDING, secret=SECRET)
        sessions, results = [], []
        def factory():
            session = Session()
            sessions.append(session)
            return session
        def serve():
            results.append(server.serve_one(session_factory=factory, timeout_seconds=1))
        thread = threading.Thread(target=serve)
        thread.start()
        yield server, sessions, results, thread
        server.close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        for session in sessions:
            session.close()


@pytest.mark.slow
def test_actual_socket_to_jsonl_policy_round_trip_and_owned_close(running):
    server, sessions, results, thread = running
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    client.reset(seed=3)
    actions = client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                                 observation_state=[0.0] * 64, task="Place the book.")
    assert len(actions) == 1 and len(actions[0]) == 40
    assert client.candidate_policy_queried is True
    closed = client.close()
    assert closed["status"] == "closed"
    assert sessions[0].closed == 1
    assert sessions[0].process.poll() is not None
    assert SECRET not in json.dumps(closed)
    assert "Place the book" not in json.dumps(closed)
    assert closed["provider_teardown_verified"] is False
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert results[0]["policy_session_closed"] is True
    assert results[0]["forwarded_request_count"] == 2
    assert results[0]["synthetic_conformance_digest"] == sessions[0].conformance["receipt_digest"]
    assert results[0]["policy_session_close_digest"] == sessions[0].close()["receipt_digest"]
    assert results[0]["receipt_digest"] == canonical_digest(results[0], digest_field="receipt_digest")
    assert SECRET not in json.dumps(results)


@pytest.mark.parametrize("fault", ["secret", "packet", "profile", "setup", "mode"])
def test_wrong_relay_authority_never_opens_policy(running, fault):
    server, sessions, _, thread = running
    binding, secret = BINDING, SECRET
    if fault == "secret":
        secret = "b" * 64
    else:
        fields = dict(packet_digest=binding.packet_digest, profile_digest=binding.profile_digest,
                      setup_digest=binding.setup_digest, delivery_mode=binding.delivery_mode)
        fields[{"packet": "packet_digest", "profile": "profile_digest", "setup": "setup_digest", "mode": "delivery_mode"}[fault]] = (
            "noncontainer_artifact" if fault == "mode" else "sha256:" + "d" * 64)
        binding = relay.RelayBinding(**fields)
    with pytest.raises((ValueError, EOFError)):
        relay.G1PolicyRelayClient(path=server.path, binding=binding, secret=secret, timeout_seconds=1)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions == []


@pytest.mark.slow
def test_disconnect_closes_owned_policy_and_retains_secret_free_failure(running):
    server, sessions, results, thread = running
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    client._wire.close()
    # Server must reap its child before fixture cleanup can hide a leak.
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(sessions) == 1 and sessions[0].closed == 1
    assert all(session.process.poll() is not None for session in sessions)
    assert results[0]["failure_type"] == "EOFError"
    assert results[0]["policy_session_closed"] is True
    assert SECRET not in json.dumps(results)


@pytest.mark.slow
def test_idle_timeout_closes_owned_child(running):
    server, sessions, results, thread = running
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions[0].closed == 1 and sessions[0].process.poll() is not None
    assert results[0]["failure_type"] == "TimeoutError"
    assert results[0]["policy_session_closed"] is True
    client._wire.close()


@pytest.mark.slow
@pytest.mark.parametrize("connected", [False, True])
def test_host_stop_wakes_long_accept_or_read_and_reaps_owned_policy(connected):
    with tempfile.TemporaryDirectory(prefix="g1-relay-stop-", dir=str(Path("/tmp").resolve())) as root:
        server = relay.G1PolicyRelayServer(path=Path(root) / "wire", binding=BINDING, secret=SECRET)
        sessions, results = [], []
        def factory():
            session = Session()
            sessions.append(session)
            return session
        thread = threading.Thread(target=lambda: results.append(server.serve_one(
            session_factory=factory, timeout_seconds=600)))
        client = None
        try:
            thread.start()
            if connected:
                client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING,
                                                  secret=SECRET, timeout_seconds=1)
            server.close()
            thread.join(timeout=3)
            assert not thread.is_alive(), "host stop waited for the 600-second wire deadline"
            assert not server.path.exists()
            assert len(results) == 1 and results[0]["failure_type"] is not None
            assert results[0]["forwarded_request_count"] == 0
            assert len(sessions) == int(connected)
            assert all(s.closed == 1 and s.process.poll() is not None for s in sessions)
            assert results[0]["policy_session_closed"] is connected
        finally:
            server.close()
            if client is not None:
                client._wire.close()
            thread.join(timeout=3)
            for session in sessions:
                session.close()


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["profile", "setup", "status", "mode", "sealed"])
def test_factory_conformance_mismatch_never_forwards_site_input(running, monkeypatch, fault):
    server, sessions, results, thread = running
    original = Session.__init__
    def changed(self):
        original(self)
        if fault == "profile":
            self.conformance["profile_digest"] = "sha256:" + "f" * 64
        elif fault == "setup":
            self.conformance["source_setup_digest"] = "sha256:" + "f" * 64
        elif fault == "status":
            self.conformance["status"] = "unproven"
        elif fault == "mode":
            self.delivery_mode = "noncontainer_artifact"
        else:
            self.conformance["receipt_digest"] = "sha256:" + "f" * 64
    monkeypatch.setattr(Session, "__init__", changed)
    with pytest.raises(EOFError):
        relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions[0].client._request_index == 0
    assert sessions[0].closed == 1 and sessions[0].process.poll() is not None
    assert results[0]["forwarded_request_count"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["replay", "command", "extra", "payload"])
def test_untrusted_commands_and_replays_never_reach_policy(running, fault):
    server, sessions, results, thread = running
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    client.reset(seed=3)
    request = {"protocol": relay.PROTOCOL, "request_id": 1,
               "kind": "reset", "payload": {"seed": 4}}
    if fault == "replay":
        request["request_id"] = 0
    elif fault == "command":
        request["kind"] = "launch"
    elif fault == "extra":
        request["command"] = "SECRET COMMAND DO NOT RETAIN"
    else:
        request["payload"]["command"] = "SECRET COMMAND DO NOT RETAIN"
    client._wire.write(request)
    with pytest.raises(EOFError):
        client._wire.read()
    client._wire.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions[0].client._request_index == 1
    assert sessions[0].closed == 1 and sessions[0].process.poll() is not None
    assert results[0]["forwarded_request_count"] == 1
    assert "SECRET COMMAND" not in json.dumps(results)


@pytest.mark.slow
def test_invalid_action_does_not_count_as_valid_policy_inference(running, monkeypatch):
    server, sessions, results, thread = running
    original = Session.__init__
    def changed(self):
        original(self)
        _close(self.process)
        self.process = _process("{'ok': True, 'action_chunk': [[0.0] * 40]}")
        self.client = NativeG1TeamPolicyJsonlClient(self.process, timeout_seconds=1,
                                                  profile_digest=BINDING.profile_digest)
    monkeypatch.setattr(Session, "__init__", changed)
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    with pytest.raises(ValueError, match="rotation_invalid"):
        client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                           observation_state=[0.0] * 64, task="Place the book.")
    assert client.candidate_policy_queried is False
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions[0].closed == 1
    assert results[0]["policy_session_closed"] is True
    with pytest.raises(ValueError, match="failed"):
        client.close()


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["stall", "reset_ack"])
def test_failed_policy_exchange_is_not_retried_and_reaps_child(running, monkeypatch, fault):
    import subprocess
    import sys
    server, sessions, results, thread = running
    original = Session.__init__
    def changed(self):
        original(self)
        _close(self.process)
        if fault == "stall":
            self.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL)
        else:
            self.process = _process("{'ok': False}")
        self.client = NativeG1TeamPolicyJsonlClient(self.process, timeout_seconds=0.1,
                                                  profile_digest=BINDING.profile_digest)
    monkeypatch.setattr(Session, "__init__", changed)
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    with pytest.raises(EOFError):
        client.reset(seed=2)
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions[0].client._request_index == 1
    assert sessions[0].process.poll() is not None and sessions[0].closed == 1
    assert results[0]["forwarded_request_count"] == 0
    with pytest.raises(ValueError, match="failed"):
        client.reset(seed=2)


@pytest.mark.slow
def test_actual_policy_receives_exact_observation_bytes(running, monkeypatch):
    from blueprint_pipeline.native_g1_humanoidarena_policy_client import build_semantic_v3_infer_request
    server, sessions, _, _ = running
    original = Session.__init__
    def changed(self):
        original(self)
        _close(self.process)
        self.process = _process("{'ok': True, 'echo': {k: v for k, v in request.items() if k not in "
                                "{'protocol', 'request_id', 'kind', 'profile_digest'}}}")
        self.client = NativeG1TeamPolicyJsonlClient(self.process, timeout_seconds=1,
                                                  profile_digest=BINDING.profile_digest)
    monkeypatch.setattr(Session, "__init__", changed)
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    payload = build_semantic_v3_infer_request(front_rgb=image, observation_state=list(range(64)),
                                            task="Place the same sealed book.")
    assert client._exchange("infer", payload)["echo"] == payload
    client.close()
    assert sessions[0].closed == 1


@pytest.mark.slow
def test_failed_session_close_cannot_acknowledge_cleanup(running, monkeypatch):
    server, sessions, results, thread = running
    original = Session.close
    def changed(self):
        value = original(self)
        return {**value, "status": "teardown_blocked"}
    monkeypatch.setattr(Session, "close", changed)
    client = relay.G1PolicyRelayClient(path=server.path, binding=BINDING, secret=SECRET, timeout_seconds=1)
    with pytest.raises(EOFError):
        client.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions[0].process.poll() is not None
    assert results[0]["status"] == "failed"
    assert results[0]["policy_session_closed"] is False
    assert results[0]["provider_teardown_verified"] is False


def test_replayed_challenge_proof_never_opens_policy(running):
    server, sessions, _, thread = running
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    peer.connect(str(server.path))
    wire = relay.JsonlSocket(peer, timeout_seconds=1)
    challenge = wire.read()["challenge"]
    stale = "a" * 64 if challenge != "a" * 64 else "b" * 64
    from dataclasses import asdict
    wire.write({"protocol": relay.PROTOCOL, "binding": asdict(BINDING),
                "proof": relay._proof(SECRET, stale, BINDING)})
    with pytest.raises(EOFError):
        wire.read()
    wire.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert sessions == []


@pytest.mark.slow
def test_relay_host_import_requires_only_python_standard_library():
    import subprocess
    import sys
    script = "\n".join([
        "import importlib.util, sys",
        "spec = importlib.util.spec_from_file_location('g1_relay_host', sys.argv[1])",
        "module = importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name] = module",
        "spec.loader.exec_module(module)",
        "module.RelayBinding('sha256:' + 'a' * 64, 'sha256:' + 'b' * 64, 'sha256:' + 'c' * 64, 'container')",
        "assert not any(name.startswith(('numpy', 'torch', 'isaac', 'blueprint_pipeline')) for name in sys.modules)",
    ])
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, relay.__file__],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_refuses_existing_socket_and_preserves_foreign_replacement(tmp_path):
    with tempfile.TemporaryDirectory(prefix="g1-relay-", dir=str(Path("/tmp").resolve())) as root:
        path = Path(root) / "wire"
        path.write_text("keep")
        with pytest.raises(ValueError):
            relay.G1PolicyRelayServer(path=path, binding=BINDING, secret=SECRET)
        assert path.read_text() == "keep"
        path.unlink()
        server = relay.G1PolicyRelayServer(path=path, binding=BINDING, secret=SECRET)
        path.unlink()
        path.write_text("foreign")
        server.close()
        assert path.read_text() == "foreign"


def test_refuses_aliased_or_nonprivate_socket_parent():
    with tempfile.TemporaryDirectory(prefix="g1-relay-", dir=str(Path("/tmp").resolve())) as root:
        parent = Path(root)
        alias = parent / "alias"
        private = parent / "private"
        private.mkdir(mode=0o700)
        alias.symlink_to(private, target_is_directory=True)
        with pytest.raises(ValueError, match="path_invalid"):
            relay.G1PolicyRelayServer(path=alias / "wire", binding=BINDING, secret=SECRET)
        private.chmod(0o755)
        with pytest.raises(ValueError, match="parent_not_private"):
            relay.G1PolicyRelayServer(path=private / "wire", binding=BINDING, secret=SECRET)


@pytest.mark.slow
def test_relay_binds_actual_conformance_and_runtime_session_close(tmp_path):
    from blueprint_pipeline.native_g1_team_policy_conformance import run_g1_team_policy_synthetic_conformance
    from blueprint_pipeline.native_g1_team_runtime_session import NativeG1TeamRuntimeSession
    from tests.test_team_policy_delivery_profile import OWNER, _setup, _profile
    setup = _setup()
    profile = _profile(setup, {"mode": "noncontainer_artifact", "artifact_uri": "https://files.example.org/policy.tar.gz",
                              "artifact_sha256": "sha256:" + "b" * 64, "entrypoint": "policy/run.py",
                              "protocol": "jsonl_observation_action_v1"})
    binding = relay.RelayBinding(BINDING.packet_digest, profile["profile_digest"], setup["setup_digest"],
                                 "noncontainer_artifact")
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    process = _process("{'ok': True, 'action_chunk': " + repr([action]) + "}")
    policy = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=1, profile_digest=binding.profile_digest)
    conformance = run_g1_team_policy_synthetic_conformance(profile=profile, trusted_setup=setup,
                                                          authenticated_owner=OWNER, policy_client=policy)
    class Lease:
        def close(self):
            _close(process)
            value = {"status": "process_exited", "exit_code": process.poll()}
            value["receipt_digest"] = canonical_digest(value)
            return value
    session = NativeG1TeamRuntimeSession(client=policy, conformance=conformance, profile_digest=binding.profile_digest,
                                        delivery_mode=binding.delivery_mode, output_dir=tmp_path, lease=Lease())
    with tempfile.TemporaryDirectory(prefix="g1-relay-", dir=str(Path("/tmp").resolve())) as root:
        server = relay.G1PolicyRelayServer(path=Path(root) / "wire", binding=binding, secret=SECRET)
        results = []
        thread = threading.Thread(target=lambda: results.append(server.serve_one(session_factory=lambda: session,
                                                                                 timeout_seconds=1)))
        thread.start()
        try:
            client = relay.G1PolicyRelayClient(path=server.path, binding=binding, secret=SECRET, timeout_seconds=1)
            client.reset(seed=11)
            assert client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                                      observation_state=[0.0] * 64, task="Place the book.") == [action]
            client.close()
            thread.join(timeout=3)
            assert not thread.is_alive()
            closed = json.loads((tmp_path / "native_g1_team_runtime_session.v1.json").read_text())
            assert closed["child_teardown_required"] is True
            assert process.poll() is not None
            assert results[0]["policy_session_close_digest"] == closed["receipt_digest"]
            assert results[0]["synthetic_conformance_digest"] == conformance["receipt_digest"]
            assert results[0]["provider_teardown_verified"] is False
        finally:
            server.close()
            thread.join(timeout=3)
            session.close()


@pytest.mark.parametrize("body", [b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":1e999}\n',
                                  b'[]\n', b'{"partial":', b'x' * (relay.MAX_FRAME_BYTES + 1) + b'\n'])
def test_frame_parser_rejects_untrusted_json_and_oversized_input(body):
    left, right = socket.socketpair()
    def write():
        try:
            left.sendall(body)
        except OSError:
            pass
        finally:
            left.close()
    thread = threading.Thread(target=write)
    thread.start()
    try:
        with pytest.raises(ValueError):
            relay.JsonlSocket(right, timeout_seconds=1).read()
    finally:
        right.close()
        thread.join(timeout=2)
