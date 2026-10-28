"""The beta finalist HTTP boundary authenticates before unpaid preparation or authority."""
from __future__ import annotations

from datetime import datetime, timezone
import hmac
import itertools
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from blueprint_pipeline import beta_policy_finalist_routes as routes
from blueprint_pipeline import beta_policy_finalists as finalists
from blueprint_pipeline import live_pipeline_intake_service as intake

PREFIX = "/api/live-pipeline/beta-policy-finalists"
SECRET = "synthetic-finalist-route-signing-secret"
CLIENT = "finalist-route-test"
FID = "finalist-" + "1" * 32
DIGEST = "sha256:" + "2" * 64


def _headers(body: str, nonce: str, *, secret=SECRET, client=CLIENT):
    timestamp = datetime.now(timezone.utc).isoformat()
    signature = hmac.new(secret.encode(), f"{timestamp}.{client}.{nonce}.{body}".encode(), "sha256").hexdigest()
    return {"content-type": "application/json", "x-blueprint-pipeline-timestamp": timestamp,
            "x-blueprint-pipeline-client-id": client, "x-blueprint-pipeline-nonce": nonce,
            "x-blueprint-pipeline-signature": "sha256=" + signature}


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    # Keep the actual signature, nonce-replay and intake-admission implementation.
    # Only finalist storage/execution actions are replaced below.
    monkeypatch.setenv(intake.INTAKE_CLIENT_SECRETS_ENV, json.dumps({CLIENT: SECRET}))
    monkeypatch.setenv(intake.INTAKE_TOKEN_ENV, SECRET)
    monkeypatch.setenv(intake.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    monkeypatch.setenv(intake.INTAKE_ALLOW_LEGACY_WEBAPP_HMAC_ENV, "false")
    monkeypatch.setenv(intake.INTAKE_WORK_DIR_ENV, str(tmp_path / "admission"))
    monkeypatch.setenv(intake.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    intake._INTAKE_NONCE_CACHE.clear()
    app = FastAPI()
    routes.register_beta_policy_finalist_routes(app, require_admission=intake._require_admission)
    calls = []
    monkeypatch.setattr(routes, "prepare_finalist", lambda value: calls.append(("prepare", value)) or {"prepared": True})
    monkeypatch.setattr(routes, "inspect_finalist", lambda fid: calls.append(("inspect", fid)) or {"finalist_id": fid})
    monkeypatch.setattr(routes, "authorize_finalist", lambda fid, value: calls.append(("authorize", fid, value)) or {"authorized": True})
    counter = itertools.count()
    client = TestClient(app)

    def post(path=PREFIX, value=None, *, raw=None):
        body = raw if raw is not None else json.dumps(value if value is not None else {})
        return client.post(path, content=body, headers=_headers(body, f"finalist-route-{next(counter):04}"))

    def get(path):
        return client.get(path, headers=_headers("", f"finalist-route-{next(counter):04}"))

    yield SimpleNamespace(client=client, calls=calls, post=post, get=get)
    intake._INTAKE_NONCE_CACHE.clear()


@pytest.mark.parametrize("suffix", ["", f"/{FID}/authorize"])
def test_signature_admission_rejects_before_prepare_or_authorize(boundary, suffix):
    path = PREFIX + suffix
    assert boundary.client.post(path, json={}).status_code == 401
    assert boundary.client.post(path, content="{}", headers=_headers("{}", "invalid-secret-0001", secret="wrong")).status_code == 401
    assert boundary.client.post(path, content="{}", headers=_headers("{}", "unknown-client-0001", client="unregistered-client")).status_code == 401
    # The signature binds the actual body, so a changed owner cap is not admitted.
    signed = '{"approval":{"sponsorCapUsd":1}}'
    changed = '{"approval":{"sponsorCapUsd":20}}'
    assert boundary.client.post(path, content=changed, headers=_headers(signed, "changed-body-0001")).status_code == 401
    assert boundary.calls == []


def test_authenticated_nonce_cannot_replay_a_preparation(boundary):
    headers = _headers("{}", "single-use-nonce-0001")
    assert boundary.client.post(PREFIX, content="{}", headers=headers).status_code == 200
    replay = boundary.client.post(PREFIX, content="{}", headers=headers)
    assert replay.status_code == 401 and "replayed" in replay.json()["detail"]
    assert boundary.calls == [("prepare", {})]


@pytest.mark.parametrize("suffix", ["", f"/{FID}/authorize"])
@pytest.mark.parametrize("raw,status,detail", [
    ("{broken", 400, "beta_finalist_invalid_json"),
    ("[]", 422, "beta_finalist_object_required"),
    ("null", 422, "beta_finalist_object_required"),
    ('"text"', 422, "beta_finalist_object_required"),
])
def test_malformed_signed_body_never_reaches_core(boundary, suffix, raw, status, detail):
    response = boundary.post(PREFIX + suffix, raw=raw)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert boundary.calls == []


@pytest.mark.parametrize("suffix", ["", f"/{FID}/authorize"])
def test_exact_route_body_bound_accepts_32kib_and_rejects_one_more_byte(boundary, suffix):
    exact = json.dumps({"padding": "x" * (32_768 - len(json.dumps({"padding": ""})))})
    assert len(exact.encode()) == 32_768
    assert boundary.post(PREFIX + suffix, raw=exact).status_code == 200
    assert len(boundary.calls) == 1
    oversized = boundary.post(PREFIX + suffix, raw=exact + " ")
    assert oversized.status_code == 413
    assert oversized.json() == {"detail": "beta_finalist_request_too_large"}
    assert len(boundary.calls) == 1


def _record(*, authorized=False):
    return {"plan": {"campaign_id": FID, "source": {"capture_id": "capture-1"},
                     "finalist_binding": {"job_id": "job-1", "brief_digest": DIGEST, "source_fingerprint": DIGEST},
                     "candidates": [{"candidate_id": "pi05_droid"}], "maximum_planned_cost_usd": "13.74"},
            "prepared_plan_digest": DIGEST, "state": "authorized" if authorized else "awaiting_owner_authority",
            **({"authority": {"authority_digest": DIGEST}} if authorized else {})}


def test_signed_prepare_inspect_authorize_forward_exact_inputs_without_provider_effects(boundary, monkeypatch):
    request = {"schema_version": "beta_policy_finalist_request.v1", "campaign_id": "screening-1",
               "publication_digest": DIGEST, "candidate_id": "pi05_droid", "job_id": "job-1",
               "brief_digest": DIGEST, "source_fingerprint": DIGEST}
    approval = {"plan_digest": DIGEST, "approved_by": "named-owner", "approval": {
        "episodes": 200, "sponsorCapUsd": 20, "expiresAtIso": "2026-11-01T00:00:00Z",
        "maxAttempts": 1, "evidenceScope": "development_only"}}
    calls = []
    monkeypatch.setattr(routes, "prepare_finalist", lambda payload: calls.append(("prepare", payload)) or finalists._receipt(_record()))
    monkeypatch.setattr(routes, "inspect_finalist", lambda fid: calls.append(("inspect", fid)) or finalists._receipt(_record()))
    monkeypatch.setattr(routes, "authorize_finalist", lambda fid, payload:
        calls.append(("authorize", fid, payload)) or finalists._receipt(_record(authorized=True)))
    prepared = boundary.post(value=request)
    inspected = boundary.get(f"{PREFIX}/{FID}")
    authorized = boundary.post(f"{PREFIX}/{FID}/authorize", value=approval)
    assert [prepared.status_code, inspected.status_code, authorized.status_code] == [200, 200, 200]
    assert calls == [("prepare", request), ("inspect", FID), ("authorize", FID, approval)]
    assert prepared.json()["paid_execution_requested"] is False
    assert inspected.json()["paid_execution_requested"] is False
    # Accepted owner spending authority is distinct from provider execution.
    assert authorized.json()["paid_execution_requested"] is True
    assert authorized.json()["owner_authority_digest"] == DIGEST
    for response in (prepared, inspected, authorized):
        assert response.json()["provider_mutation_performed"] is False
        assert response.json()["recommendation_eligible"] is False


@pytest.mark.parametrize("suffix,action", [("", "prepare_finalist"), (f"/{FID}", "inspect_finalist"),
                                           (f"/{FID}/authorize", "authorize_finalist")])
@pytest.mark.parametrize("error,status,detail", [
    (FileNotFoundError("/private/customer-input.json"), 404, "beta_finalist_not_found"),
    (ValueError("beta_finalist_owner_cap_insufficient"), 409, "beta_finalist_owner_cap_insufficient"),
    (ValueError("/private/customer-input.json: credential value"), 409, "beta_finalist_source_unavailable"),
    (KeyError("/private/customer-input.json"), 409, "beta_finalist_source_unavailable"),
    (TypeError("credential value"), 409, "beta_finalist_source_unavailable"),
])
def test_core_failures_have_fixed_status_and_safe_error_mapping(boundary, monkeypatch, suffix, action, error, status, detail):
    def refuse(*args):
        raise error
    monkeypatch.setattr(routes, action, refuse)
    response = boundary.get(PREFIX + suffix) if action == "inspect_finalist" else boundary.post(PREFIX + suffix)
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert "private" not in response.text and "credential value" not in response.text
