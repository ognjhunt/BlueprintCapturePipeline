"""The simulator must not send observations to an unqualified bridge."""
from __future__ import annotations

import json
import http.client
import shutil
import ssl
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline.company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from blueprint_pipeline.controlled_policy_configuration import canonical_request_digest
from blueprint_pipeline.controlled_policy_remote_sandbox import RemoteQualifiedSandboxFactory
from blueprint_pipeline.controlled_native_isaac import validate_native_controller_interface
from blueprint_pipeline.controlled_policy_bridge_server import QualifiedPolicyBridge
from tests.test_company_policy_container_contract_v2 import _contract


def _fixture():
    contract = validate_company_policy_container_contract_v2(_contract())
    request = {"job_id": "job-bridge-1", "policy_package": {"sim_controller_plugin": {
        "image_ref": contract["container"]["image"], "transport": "isolated_container_http_json_v1",
    }}}
    bridge = {"schema_version": "blueprint.qualified_policy_bridge.v1",
        "job_id": request["job_id"], "canonical_request_digest": canonical_request_digest(request),
        "contract_digest": contract["contract_digest"], "image_ref": contract["container"]["image"],
        "endpoint_url": "https://policy-worker.example/v1/controlled-policy",
        "bearer_token": "test-bridge-token-with-at-least-32-bytes",
        "qualification_receipt_digest": "sha256:" + "f" * 64,
        "expires_at_iso": (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat(),
        "model_artifact_sha256": None, "tls_certificate_pem": None}
    return contract, request, bridge


def test_bridge_requires_qualification_then_actions_and_cleanup():
    contract, request, bridge = _fixture()
    calls = []
    terminal = 0

    def send(*, route, body, timeout, max_bytes):
        nonlocal terminal
        calls.append(route)
        if route == "/ready":
            return json.dumps({**body, "status": "qualified_before_first_observation",
                "qualification_receipt_digest": bridge["qualification_receipt_digest"]}).encode()
        if route == "/actions":
            assert body["observation"]["state"] == {"joint_position": [0.2]}
            return b'{"actions":[[0.2,0.5]]}'
        if route == "/finish":
            assert body["policy_calls"] == 1
            return b'{"status":"cleanup_pending"}'
        terminal += 1
        if terminal == 1:
            return b'{"status":"terminal_pending"}'
        return json.dumps({**body, "status": "controlled_session_completed",
            "cleanup_complete": True, "policy_calls": 1}).encode()

    factory = RemoteQualifiedSandboxFactory(bridge, request=send)
    result = factory(contract=contract, job_request=request,
        qualified_session=lambda transport: {
            "actions": json.loads(transport(json.dumps({"state": {"joint_position": [0.2]}}).encode(), 1)),
            "executed_motor_steps": 1,
        })
    assert result["status"] == "controlled_session_completed"
    assert result["controlled_session"]["executed_motor_steps"] == 1
    assert calls == ["/ready", "/actions", "/finish", "/terminal", "/terminal"]


def test_bridge_refuses_wrong_image_before_qualification_or_observation():
    contract, request, bridge = _fixture()
    bridge["image_ref"] = "registry.acme.example/wrong@sha256:" + "a" * 64
    calls = []
    factory = RemoteQualifiedSandboxFactory(bridge, request=lambda **kw: calls.append(kw))
    with pytest.raises(ValueError, match="frozen_binding_mismatch"):
        factory(contract=contract, job_request=request, qualified_session=lambda _: {})
    assert calls == []


def test_bridge_refuses_missing_cleanup():
    contract, request, bridge = _fixture()
    def send(*, route, body, **_kwargs):
        if route == "/ready":
            return json.dumps({**body, "status": "qualified_before_first_observation",
                "qualification_receipt_digest": bridge["qualification_receipt_digest"]}).encode()
        if route == "/finish":
            return b'{"status":"cleanup_pending"}'
        return json.dumps({**body, "status": "blocked", "cleanup_complete": False,
            "policy_calls": 0}).encode()
    factory = RemoteQualifiedSandboxFactory(bridge, request=send)
    with pytest.raises(ValueError, match="cleanup_unverified"):
        factory(contract=contract, job_request=request, qualified_session=lambda _: {})


def test_controller_action_order_and_units_match_offered_robot():
    contract = _contract()
    contract["robot"]["joint_names"] = [f"panda_joint{index}" for index in range(1, 8)]
    contract["robot"]["joint_limits"] = [
        {"name": name, "lower": -2.0, "upper": 2.0, "unit": "radian"}
        for name in contract["robot"]["joint_names"]]
    base = dict(contract["action_schema"]["channels"][0])
    contract["action_schema"]["channels"] = [
        {**base, "name": name} for name in contract["robot"]["joint_names"]
    ] + [contract["action_schema"]["channels"][-1]]
    validate_native_controller_interface(contract)
    contract["action_schema"]["channels"][0]["unit"] = "degree"
    with pytest.raises(ValueError, match="controller_interface_mismatch"):
        validate_native_controller_interface(contract)


def test_dedicated_worker_bridge_serves_only_after_qualification_and_cleans(tmp_path):
    if shutil.which("openssl") is None:
        pytest.skip("openssl unavailable")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key), "-out", str(cert), "-days", "1", "-subj", "/CN=localhost"],
        check=True, capture_output=True, timeout=15)
    contract, request, _manifest = _fixture()
    bridge = QualifiedPolicyBridge(contract=contract, job_request=request,
        endpoint_url="https://localhost/v1/controlled-policy", bearer_token="a" * 40,
        manifest_path=tmp_path / "bridge.json", bind_host="127.0.0.1", bind_port=0,
        tls_certificate=cert, tls_private_key=key, maximum_seconds=20)
    port = bridge.server.server_address[1]
    bridge.endpoint_url = f"https://localhost:{port}/v1/controlled-policy"
    qualification = {"status": "qualified_before_first_observation",
        "first_observation_permitted": True, "qualification_digest": "sha256:" + "f" * 64}
    result = {}
    def execute(session):
        session(lambda body, _timeout: b'{"actions":[[0.2,0.5]]}', qualification)
        return {"status": "controlled_session_completed",
            "terminal_receipt": {"cleanup_complete": True, "receipt_digest": "sha256:" + "e" * 64}}
    thread = threading.Thread(target=lambda: result.update(bridge.run(execute)))
    thread.start()
    deadline = time.monotonic() + 5
    while not (tmp_path / "bridge.json").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (tmp_path / "bridge.json").exists()
    manifest = json.loads((tmp_path / "bridge.json").read_text())
    binding = {key: manifest[key] for key in (
        "job_id", "canonical_request_digest", "contract_digest", "image_ref")}

    def post(route, body, token="a" * 40):
        connection = http.client.HTTPSConnection("localhost", port,
            context=ssl._create_unverified_context(), timeout=3)
        connection.request("POST", "/v1/controlled-policy" + route,
            body=json.dumps(body).encode(), headers={"Content-Type": "application/json",
            "Authorization": "Bearer " + token})
        response = connection.getresponse()
        value = response.status, json.loads(response.read())
        connection.close()
        return value

    assert post("/ready", binding, token="wrong")[0] == 401
    assert post("/ready", binding)[1]["qualification_receipt_digest"] == qualification["qualification_digest"]
    assert post("/actions", {**binding, "observation": {"state": {"joint_position": [0.2]}}})[1] == {
        "actions": [[0.2, 0.5]]}
    assert post("/finish", {**binding, "policy_calls": 1})[1]["status"] == "cleanup_pending"
    deadline = time.monotonic() + 5
    terminal = None
    while time.monotonic() < deadline:
        terminal = post("/terminal", binding)[1]
        if terminal["status"] != "terminal_pending":
            break
        time.sleep(0.02)
    assert terminal["status"] == "controlled_session_completed"
    assert terminal["cleanup_complete"] is True
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result["terminal_receipt"]["cleanup_complete"] is True
