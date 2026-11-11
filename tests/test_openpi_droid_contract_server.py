"""The contract server answers exactly what the real ``openpi_droid_https.v1`` client sends and expects.

Hermetic: a deterministic stand-in policy behind the real server on 127.0.0.1, reached by the real
client over the real bounded loopback transport. No checkpoint, GPU or network.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

from blueprint_pipeline.droid_https_policy_client import encode_observation
from blueprint_pipeline.openpi_droid_contract_server import ContractPolicyService, build_server, decode_observation
from blueprint_pipeline.session_policy_client import LoopbackDroidPolicySpec, session_policy_client

TOKEN = "a" * 64
DIGEST = "sha256:" + "4" * 64


class StandInPolicy:
    """Returns a chunk computed from every input it is given, so any lost bit changes the answer."""

    def __init__(self):
        self.seen = []

    def infer(self, observation):
        self.seen.append(observation)
        base = (float(observation["observation/exterior_image_1_left"].astype(np.int64).sum() % 997)
                + float(observation["observation/wrist_image_left"].astype(np.int64).sum() % 991)
                + float(np.sum(observation["observation/joint_position"])) + float(observation["observation/gripper_position"][0])
                + len(observation["prompt"]))
        actions = (np.arange(15 * 8, dtype=np.float32).reshape(15, 8) / 7.0 + np.float32(base)).astype(np.float32)
        return {"actions": actions, "policy_timing": {"infer_ms": 1.25}}


def _observation(seed=0):
    rng = np.random.default_rng(seed)
    return {"observation/exterior_image_1_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/joint_position": rng.normal(size=7), "observation/gripper_position": np.array([0.25]),
            "prompt": "pick up the tote and place it on the shelf"}


def _spec(port):
    origin = f"http://127.0.0.1:{port}"
    return {"policy_spec": {"policy_id": "pi05_droid_contract", "endpoint_url": origin + "/v1/infer",
                            "metadata_url": origin + "/v1/metadata", "allowed_origin": origin,
                            "model_identity_digest": DIGEST, "action_chunk_rows": 15, "open_loop_horizon": 8}}


@pytest.fixture()
def served():
    policy = StandInPolicy()
    server = build_server(ContractPolicyService(policy=policy, policy_spec=_spec(1)["policy_spec"], bearer_token=TOKEN),
                          port=0)
    port = server.server_address[1]
    service = server.RequestHandlerClass  # noqa: F841 - keep the handler bound for the thread's life
    # The served paths come from the spec; the port is the one the OS gave the listener.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield policy, server, _spec(port)
    server.shutdown()
    server.server_close()


def test_round_trip_through_the_real_client_gives_the_policy_the_native_arrays(served):
    policy, _server, spec = served
    client = session_policy_client(spec, TOKEN)
    readiness = client.preflight_readiness()
    assert readiness["identity_verified"] is True and readiness["candidate_inference_performed"] is False
    assert policy.seen == []  # Readiness never advances the policy.
    observation = _observation()
    actions = client.infer(observation)
    assert len(policy.seen) == 1
    seen = policy.seen[0]
    for key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
        assert seen[key].dtype == np.uint8 and np.array_equal(seen[key], observation[key])
    for key in ("observation/joint_position", "observation/gripper_position"):
        assert seen[key].dtype == np.float64 and np.array_equal(seen[key], observation[key])
    assert seen["prompt"] == observation["prompt"]
    # The policy's float32 chunk arrives bit-exact as float64 values.
    expected = StandInPolicy().infer(observation)["actions"]
    assert np.array_equal(np.asarray(actions, dtype=np.float32), expected)
    assert np.array_equal(np.asarray(actions, dtype=float), expected.astype(float))
    evidence = client.last_inference_evidence()
    assert evidence["actions_extracted"] is True and evidence["wire_response_keys"] == [
        "actions", "policy_timing", "server_timing"]
    assert client.candidate_policy_queried is True


def test_decode_is_the_exact_inverse_of_the_client_encoding():
    observation = _observation(3)
    decoded = decode_observation(json.loads(json.dumps(encode_observation(observation))))
    assert all(np.array_equal(decoded[key], observation[key]) for key in observation if key != "prompt")


def test_wrong_token_and_wrong_path_are_answers_the_policy_never_sees(served):
    policy, _server, spec = served
    with pytest.raises(ValueError, match="droid_https_policy_response_status_invalid:401"):
        session_policy_client(spec, "b" * 64).infer(_observation())
    assert policy.seen == []


def test_non_finite_actions_reach_the_client_as_a_policy_failure(served):
    policy, _server, spec = served
    policy.infer = lambda observation: {"actions": np.full((15, 8), np.nan, dtype=np.float32)}
    with pytest.raises(ValueError, match="droid_https_policy_response_nonfinite"):
        session_policy_client(spec, TOKEN).infer(_observation())


def test_policy_exception_is_the_policys_failure_not_the_servers(served):
    policy, _server, spec = served

    def broken(observation):
        raise RuntimeError("model exploded")
    policy.infer = broken
    with pytest.raises(ValueError, match="droid_https_policy_response_status_invalid:500"):
        session_policy_client(spec, TOKEN).infer(_observation())


def test_a_server_that_died_is_session_infrastructure_and_scores_void(served, tmp_path):
    from blueprint_pipeline.policy_canary_worker_evidence import _write_episode_failure_gap
    from blueprint_pipeline.team_policy_episode_outcome import episode_outcome

    _policy, server, spec = served
    client = session_policy_client(spec, TOKEN)
    client.preflight_readiness()
    server.shutdown()
    server.server_close()
    with pytest.raises(ValueError) as raised:
        client.infer(_observation())
    assert str(raised.value).startswith("session_policy_server_unavailable:")
    path = _write_episode_failure_gap(output_root=tmp_path, run_id="run",
        context={"cell_id": "cell", "candidate_id": "pi05_droid_contract"}, failure=raised.value,
        progress={"first_observation_retained": True, "candidate_policy_query_attempted": True,
                  "policy_inference_failed": True})
    evidence = json.loads(path.read_text())
    assert evidence["episode_failure_cause"] == "infrastructure"
    assert episode_outcome(evidence)[0] == "void"


def test_loopback_spec_refuses_any_non_loopback_endpoint():
    spec = _spec(8765)["policy_spec"]
    for change in ({"allowed_origin": "https://policy.example.com"},
                   {"endpoint_url": "http://10.0.0.5:8765/v1/infer"},
                   {"metadata_url": "http://127.0.0.1:8765/v1/metadata?x=1"}):
        with pytest.raises(ValueError, match="session_policy_endpoint_not_loopback"):
            LoopbackDroidPolicySpec(**{**spec, **change}).validate()
    with pytest.raises(ValueError, match="contract_server_must_be_loopback_only"):
        build_server(ContractPolicyService(policy=StandInPolicy(), policy_spec=spec, bearer_token=TOKEN),
                     port=0, host="0.0.0.0")


def test_canary_client_factory_routes_a_session_served_spec_to_the_loopback_client(served, tmp_path, monkeypatch):
    from blueprint_pipeline.droid_https_policy_client import DroidHttpsTeamPolicyClient
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client

    _policy, _server, spec = served
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    monkeypatch.setenv("BLUEPRINT_SESSION_POLICY_TOKEN_FILE", str(token))
    client = canary_policy_client({**spec, "candidate_id": "pi05_droid_contract",
                                   "policy_interface_id": "openpi_droid_https.v1",
                                   "policy_endpoint": {"credential_env": "BLUEPRINT_SESSION_POLICY_TOKEN_FILE"},
                                   "session_policy_server": {"server": "openpi_droid_contract_server.v1"}})
    assert isinstance(client, DroidHttpsTeamPolicyClient)
    assert client.preflight_readiness()["identity_verified"] is True


def test_server_module_makes_no_outbound_call():
    """The contract server's only socket is its loopback listener: it imports no client transport or provider SDK."""
    import ast

    source = Path(__file__).parents[1] / "src/blueprint_pipeline/openpi_droid_contract_server.py"
    imported = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
            imported.update(alias.name for alias in node.names)
    allowed = {"__future__", "annotations", "argparse", "base64", "binascii", "hmac", "io", "json", "math", "time",
               "threading", "collections.abc", "Callable", "Mapping", "http.server", "BaseHTTPRequestHandler",
               "ThreadingHTTPServer", "pathlib", "Path", "typing", "Any", "urllib.parse", "urlsplit", "numpy", "PIL",
               "Image", ".droid_https_policy_client", "IMAGE_KEYS", "INTERFACE_ID", "METADATA_SCHEMA",
               "REQUEST_SCHEMA", "STATE_KEYS", ".openpi_droid_policy_runtime", "OpenPIDroidPolicySpec",
               "load_identity_bound_policy"}
    assert imported <= allowed, sorted(imported - allowed)


# --- start-up on the GPU: bounded readiness, infrastructure failure, teardown receipt ---------------------------

class FakeProcess:
    def __init__(self, exit_after_polls=None, returncode=1):
        self.pid = os.getpid()
        self._polls, self._exit_after, self.returncode_value = 0, exit_after_polls, returncode
        self.terminated = False

    def poll(self):
        self._polls += 1
        if self.terminated or (self._exit_after is not None and self._polls > self._exit_after):
            return self.returncode_value
        return None

    @property
    def returncode(self):
        return self.poll()

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.returncode_value

    def kill(self):
        self.terminated = True


def _served_spec():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.session_served_policy import ENTRY_SCHEMA, session_served_spec

    native_policy = {"policy_id": "pi05_droid_jointpos_polaris", "config_name": "pi05_droid_jointpos_polaris",
                     "checkpoint_uri": "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris",
                     "checkpoint_object_manifest_sha256": "1" * 64, "checkpoint_generation_manifest_sha256": "2" * 64,
                     "checkpoint_inventory_sha256": "3" * 64, "checkpoint_object_count": 10,
                     "checkpoint_size_bytes": 1000, "action_space": "joint_position", "action_chunk_rows": 15,
                     "open_loop_horizon": 8}
    native = {"candidate_id": "pi05_droid", "policy_interface_id": "openpi_droid.v1", "policy_spec": native_policy,
              "checkpoint_digest": DIGEST, "runtime_identity_digest": "sha256:" + "5" * 64, "open_loop_horizon": 8,
              "policy_endpoint": {"host": "127.0.0.1", "port": 8000, "credential_env": "UNUSED"}}
    native["execution_spec_digest"] = canonical_digest(native, digest_field="execution_spec_digest")
    return session_served_spec(native, candidate_id="pi05_droid_contract",
        entry={"schema_version": ENTRY_SCHEMA, "server": "openpi_droid_contract_server.v1",
               "native_candidate_id": "pi05_droid", "startup_timeout_seconds": 60})


def _start(tmp_path, *, process, client_factory, clock):
    from blueprint_pipeline.session_policy_server_start import start

    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    spec = _served_spec()
    return start(spec=spec, spec_path=tmp_path / "spec.json", python="/opt/venv/bin/python", runtime=tmp_path,
                 output=tmp_path / "out", checkpoint_dir="/opt/ckpt", inventory="/opt/inv.json",
                 env={"BLUEPRINT_SESSION_POLICY_TOKEN_FILE": str(token)}, popen=lambda *a, **k: process,
                 client_factory=client_factory, monotonic=clock, sleep=lambda seconds: None)


def test_start_waits_for_the_identity_handshake_and_leaves_a_teardown_receipt(tmp_path):
    from blueprint_pipeline.adp009d_policy_server_worker import terminate_ready_server_from_receipt

    answers = iter([ValueError("session_policy_server_unavailable:ConnectionRefusedError")] * 3 + [None])

    class Client:
        def preflight_readiness(self):
            answer = next(answers)
            if answer is not None:
                raise answer
            return {"server_metadata": {"policy_id": "pi05_droid_contract"}}
    process = FakeProcess()
    receipt = _start(tmp_path, process=process, client_factory=lambda spec, token: Client(),
                     clock=iter(range(0, 1000)).__next__)
    assert receipt["status"] == "ready" and receipt["readiness_attempts"] == 4
    assert receipt["command"][:3] == ["/opt/venv/bin/python", "-m", "blueprint_pipeline.openpi_droid_contract_server"]
    assert TOKEN not in json.dumps(receipt)
    path = tmp_path / "out" / "adp009d_policy_server_receipt.pi05_droid_contract.json"
    teardown = terminate_ready_server_from_receipt(path, command_reader=lambda pid: None)
    assert teardown["policy_server_process_terminated"] is True


@pytest.mark.parametrize("failure", ["exits", "never_answers"])
def test_start_failure_is_an_infrastructure_failure_of_the_attempt(tmp_path, failure):
    class Silent:
        def preflight_readiness(self):
            raise ValueError("session_policy_server_unavailable:ConnectionRefusedError")
    process = FakeProcess(exit_after_polls=2 if failure == "exits" else None)
    with pytest.raises(RuntimeError) as raised:
        _start(tmp_path, process=process, client_factory=lambda spec, token: Silent(),
               clock=iter(range(0, 100_000, 7)).__next__)
    expected = ("session_policy_server_exited_before_ready" if failure == "exits"
                else "session_policy_server_not_ready_within_bound")
    assert str(raised.value).startswith(expected)
    receipt = json.loads((tmp_path / "out" / "adp009d_policy_server_receipt.pi05_droid_contract.json").read_text())
    assert receipt["status"] == "blocked" and receipt["blocker"] == expected and receipt["process_terminated"] is True


def test_start_cli_failure_exits_nonzero_so_the_entrypoint_records_the_provisioning_stage(tmp_path):
    # The entrypoint turns any non-zero provisioning exit into ``policy_canary_policy_provisioning_failed``.
    from blueprint_pipeline.beta_policy_screening_failure_scope import POLICY_PROVISIONING_STAGE_CODE
    from blueprint_pipeline.session_served_policy_bundle import provisioning_script

    script = provisioning_script(_served_spec())
    assert script.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in script
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_served_spec()))
    completed = subprocess.run(
        [sys.executable, "-m", "blueprint_pipeline.session_policy_server_start", "--python", "/nonexistent/python",
         "--runtime", str(tmp_path), "--output", str(tmp_path / "out"), "--execution-spec", str(spec_path),
         "--checkpoint-dir", "/nonexistent", "--checkpoint-inventory", "/nonexistent"],
        capture_output=True, text=True, env={**os.environ, "BLUEPRINT_SESSION_POLICY_TOKEN_FILE": ""},
        cwd=Path(__file__).parents[1], check=False)
    assert completed.returncode == 2
    assert "session_policy_token_unavailable" in completed.stderr
    assert POLICY_PROVISIONING_STAGE_CODE == "policy_canary_policy_provisioning_failed"


def test_serve_loads_with_the_native_loader_and_records_its_identity_without_the_token(tmp_path, monkeypatch):
    from blueprint_pipeline import openpi_droid_contract_server as server_module
    from blueprint_pipeline import openpi_droid_policy_runtime as runtime

    loaded = []

    def native_loader(*, spec, checkpoint_dir, checkpoint_inventory_path):
        loaded.append((spec.policy_id, checkpoint_dir, checkpoint_inventory_path))
        return StandInPolicy(), {"local_checkpoint_verified": True}
    monkeypatch.setattr(runtime, "load_identity_bound_policy", native_loader)
    monkeypatch.setattr(server_module.ContractHTTPServer, "serve_forever", lambda self: self.server_close())
    spec = _served_spec()
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    with pytest.raises(ValueError, match="contract_server_port_mismatch"):
        server_module.serve(execution_spec_path=tmp_path / "spec.json", checkpoint_dir="/ckpt",
                            checkpoint_inventory_path="/inv", token_file=token, port=9999,
                            identity_out=tmp_path / "identity.json")
    server_module.serve(execution_spec_path=tmp_path / "spec.json", checkpoint_dir="/ckpt",
                        checkpoint_inventory_path="/inv", token_file=token, port=8765,
                        identity_out=tmp_path / "identity.json")
    assert loaded == [("pi05_droid_jointpos_polaris", "/ckpt", "/inv")]
    identity = json.loads((tmp_path / "identity.json").read_text())
    assert identity["outbound_calls"] == "none" and identity["host"] == "127.0.0.1"
    assert identity["contract_metadata"]["policy_id"] == "pi05_droid_contract"
    assert TOKEN not in json.dumps(identity)
    token.chmod(0o644)
    with pytest.raises(ValueError, match="contract_server_token_file_not_private"):
        server_module.read_token(token)
