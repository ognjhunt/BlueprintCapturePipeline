"""The simulator must not send observations to an unqualified bridge."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from blueprint_pipeline.company_policy_container_contract_v2 import validate_company_policy_container_contract_v2
from blueprint_pipeline.controlled_policy_configuration import canonical_request_digest
from blueprint_pipeline.controlled_policy_remote_sandbox import RemoteQualifiedSandboxFactory
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

