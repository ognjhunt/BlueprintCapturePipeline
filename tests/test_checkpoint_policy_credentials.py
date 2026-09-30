"""Synthetic contract tests; no WebApp, provider, KMS, or real credential access."""
from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import agent_run_executor as executor
from blueprint_pipeline.checkpoint_policy_credentials import CheckpointPolicyCredentialClient
from blueprint_pipeline.controlled_policy_configuration import canonical_request_digest
from blueprint_pipeline.safe_outbound_http import SafeHttpResponse


REFERENCE = "policy-credential-00000000-0000-0000-0000-000000000001"
OWNER = "agent-attempt-" + "a" * 32
TOKEN = "synthetic-sync-token-" + "x" * 32
SECRET = "synthetic-bearer-do-not-log"


@pytest.fixture
def request_value():
    return {"job_id": "canonical-job-1", "customer": {"id": "team-0001"},
        "robot_profile": {"robot_profile_id": "checkpoint-0001"},
        "execution_authorization": {"authorized_by_user_id": "owner-0001"},
        "policy_package": {"policy_api_endpoint": {"endpoint_url": "https://policy.example/action",
            "credential_ref": REFERENCE, "credential_kind": "bearer"}}}


class SyntheticCredentialRoute:
    """Model the old strict schema and the parent-supplied corrected contract.

    This checks caller compatibility; the WebApp's own tests validate its route,
    account/policy bindings, deadline predicate, and decryption ordering.
    """

    def __init__(self, request_value, *, strict_claim=True):
        self.request_value = request_value
        self.strict_claim = strict_claim
        self.owner = OWNER
        self.active = True
        self.now_ms = 1100
        self.original_due_ms = 1000
        self.settlement_due_ms = 2000
        self.requests = []
        self.decryptions = 0
        self.echo_override = {}

    def __call__(self, url, *, method, data, headers, **_kwargs):
        assert url.endswith("/checkpoint-policy-credentials/" + REFERENCE)
        assert method == "POST" and headers["Content-Type"] == "application/json"
        timestamp = headers["X-Blueprint-Pipeline-Timestamp"]
        expected = hmac.new(TOKEN.encode(), timestamp.encode() + b"." + data,
            hashlib.sha256).hexdigest()
        assert headers["X-Blueprint-Pipeline-Signature"] == "sha256=" + expected
        payload = json.loads(data)
        self.requests.append(payload)
        allowed = {"action", "job_id", "canonical_request_digest", "tenant_id", "contract",
            "admission_receipt"} | ({"pipeline_run_id"} if self.strict_claim else set())
        if set(payload) - allowed or (self.strict_claim and not payload.get("pipeline_run_id")):
            raise urllib.error.HTTPError(url, 400, "invalid", {}, None)
        due = self.settlement_due_ms if self.strict_claim else self.original_due_ms
        if not self.active or self.now_ms >= due or (
                self.strict_claim and payload["pipeline_run_id"] != self.owner):
            raise urllib.error.HTTPError(url, 409, "claim required", {}, None)
        assert payload["job_id"] == self.request_value["job_id"]
        assert payload["canonical_request_digest"] == canonical_request_digest(self.request_value)
        self.decryptions += 1
        reply = {"ok": True, "job_id": payload["job_id"],
            "canonical_request_digest": payload["canonical_request_digest"]}
        if self.strict_claim:
            reply["pipeline_run_id"] = payload["pipeline_run_id"]
        if payload["action"] == "access":
            package = self.request_value["policy_package"]
            policy = package.get("policy_api_endpoint") or package["docker_container"]
            reply.update(credential_ref=REFERENCE, kind=policy["credential_kind"])
            if policy["credential_kind"] == "bearer":
                reply["credential"] = {"job_id": payload["job_id"],
                    "endpoint_url": policy["endpoint_url"], "bearer_token": SECRET}
        elif payload["action"] == "registry_lease":
            reply["lease"] = {"lease_id": "policy-registry-lease-" + "b" * 47,
                "run_id": payload["job_id"], "contract_digest": payload["contract"]["contract_digest"],
                "image": payload["contract"]["container"]["image"],
                "status": "active", "single_use": True}
        else:
            reply = {"ok": True, "lease_id": "policy-registry-lease-" + "b" * 47,
                "admission_id": payload["admission_receipt"]["admission_id"]}
        reply.update(self.echo_override)
        return SafeHttpResponse(status=200, body=json.dumps(reply).encode(), url=url, final_url=url)


@pytest.fixture
def route(tmp_path: Path, monkeypatch, request_value):
    # Only a newly written synthetic key is read by the real credential client.
    key = tmp_path / "synthetic-sync-key"
    key.write_text(TOKEN)
    key.chmod(0o600)
    monkeypatch.setenv("BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE", str(key))
    monkeypatch.setenv("BLUEPRINT_WEBAPP_URL", "https://webapp.example")
    synthetic = SyntheticCredentialRoute(request_value)
    monkeypatch.setattr(executor, "safe_request", synthetic)
    return synthetic


def client_for(request_value, *, owner=OWNER, kind="bearer"):
    return CheckpointPolicyCredentialClient(request=request_value,
        payload={"credential_ref": REFERENCE, "credential_kind": kind}, pipeline_run_id=owner)


def test_matched_claim_is_signed_without_mutating_canonical_request(route, request_value, capsys, caplog):
    before = json.dumps(request_value, sort_keys=True)
    credential = client_for(request_value).access()
    assert credential["credential"]["bearer_token"] == SECRET
    assert route.requests == [{"action": "access", "job_id": "canonical-job-1",
        "pipeline_run_id": OWNER, "canonical_request_digest": canonical_request_digest(request_value)}]
    assert json.dumps(request_value, sort_keys=True) == before
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err + caplog.text


def test_all_registry_actions_preserve_claim_and_existing_bindings(route, request_value):
    request_value["policy_package"] = {"docker_container": {"image_ref": "registry.example/policy@sha256:" + "c" * 64,
        "credential_ref": REFERENCE, "credential_kind": "registry"}}
    client = client_for(request_value, kind="registry")
    assert "credential" not in client.access()
    contract = {"contract_digest": "sha256:" + "d" * 64,
        "container": {"image": request_value["policy_package"]["docker_container"]["image_ref"]}}
    lease_id = client.registry_lease(contract=contract, tenant_id="tenant-0001")
    admission = {"admission_id": "admission-0001"}
    client.bind_admission(contract=contract, tenant_id="tenant-0001", admission=admission, lease_id=lease_id)
    assert [row["action"] for row in route.requests] == ["access", "registry_lease", "bind_admission"]
    for row in route.requests:
        assert row["pipeline_run_id"] == OWNER
        assert row["canonical_request_digest"] == canonical_request_digest(request_value)
        assert row["job_id"] == request_value["job_id"]
    assert route.requests[1]["contract"] == route.requests[2]["contract"] == contract
    assert route.requests[1]["tenant_id"] == route.requests[2]["tenant_id"] == "tenant-0001"
    assert route.requests[2]["admission_receipt"] == admission


@pytest.mark.parametrize("owner", [None, "", " ", " padded", "padded ", 123, "a" * 201])
def test_missing_or_invalid_claim_refuses_before_reading_worker_key(monkeypatch, request_value, owner):
    monkeypatch.delenv("BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE", raising=False)
    with pytest.raises(ValueError, match="^checkpoint_policy_credential_claim_invalid$"):
        client_for(request_value, owner=owner)


@pytest.mark.parametrize("field,value", [("pipeline_run_id", None), ("pipeline_run_id", "another-owner"),
    ("job_id", "another-job"), ("canonical_request_digest", "sha256:" + "f" * 64)])
def test_response_binding_mismatch_refuses_secret(route, request_value, field, value, capsys, caplog):
    route.echo_override[field] = value
    with pytest.raises(ValueError, match="^checkpoint_policy_credential_job_binding_mismatch$"):
        client_for(request_value).access()
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("action", ["access", "registry_lease", "bind_admission"])
def test_other_claim_is_refused_without_decryption_or_automatic_retry(route, request_value, action):
    client = client_for(request_value, owner="another-owner")
    with pytest.raises(ValueError, match="^checkpoint_policy_credential_delivery_unavailable$") as error:
        client._request(action)
    assert error.value.__context__.code == 409
    assert len(route.requests) == 1 and route.decryptions == 0


@pytest.mark.parametrize("inactive", [True, False])
def test_inactive_or_expired_dispatch_refuses(route, request_value, inactive):
    route.active = not inactive
    if not inactive:
        route.now_ms = route.settlement_due_ms
    with pytest.raises(ValueError, match="^checkpoint_policy_credential_delivery_unavailable$"):
        client_for(request_value).access()
    assert route.decryptions == 0


def test_retry_reuses_owner_after_original_deadline_until_effective_deadline(route, request_value):
    client = client_for(request_value)
    assert route.original_due_ms < route.now_ms < route.settlement_due_ms
    client.access()
    client.access()
    assert route.requests[0] == route.requests[1]
    route.now_ms = route.settlement_due_ms
    with pytest.raises(ValueError, match="^checkpoint_policy_credential_delivery_unavailable$"):
        client.access()
    assert route.decryptions == 2


@pytest.mark.parametrize("strict_server,new_caller,status", [(False, False, 200),
    (False, True, 400), (True, False, 400), (True, True, 200)])
def test_compatibility_requires_matched_versions(route, request_value, strict_server, new_caller, status):
    route.strict_claim = strict_server
    route.now_ms = 500
    client = client_for(request_value)
    if not new_caller:
        client.binding.pop("pipeline_run_id")
    if status == 200:
        assert client.access()["ok"] is True
    else:
        with pytest.raises(ValueError, match="^checkpoint_policy_credential_delivery_unavailable$") as error:
            client.access()
        assert error.value.__context__.code == status
    assert len(route.requests) == 1
    assert route.decryptions == (1 if status == 200 else 0)


def test_transport_exception_text_cannot_enter_native_failure_receipt(route, request_value, monkeypatch):
    client = client_for(request_value)

    def contaminated_transport(*_args, **_kwargs):
        raise urllib.error.URLError(SECRET)

    monkeypatch.setattr(client.client, "_json", contaminated_transport)
    with pytest.raises(ValueError, match="^checkpoint_policy_credential_delivery_unavailable$") as error:
        client.access()
    assert SECRET not in str(error.value)
    assert error.value.__suppress_context__
