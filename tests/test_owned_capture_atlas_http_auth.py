"""Actual native HMAC controls for private owned-Atlas read commands; no network."""
from __future__ import annotations

import json
import hashlib
from io import BytesIO
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline import website_atlas_pipeline as controller
from blueprint_pipeline import task_evaluation_configured_scene_object_store as archives
from blueprint_pipeline.owned_capture_atlas_http import register_owned_capture_atlas_routes
from blueprint_pipeline.owned_capture_atlas_worker import enqueue_owned_atlas
from blueprint_pipeline.reconstruction_control_plane import prepare_owned_capture_atlas_plan
from tests.test_live_pipeline_intake_service import _signed_intake_headers
from tests.test_owned_capture_atlas_project import configuration as configuration
from tests.test_reconstruction_control_plane import _seed_owned_atlas_source


@pytest.fixture
def owned(tmp_path):
    state, store = tmp_path / "state", tmp_path / "store"
    session, intake, digest = _seed_owned_atlas_source(store)
    plan = prepare_owned_capture_atlas_plan(state_root=state, capture_store_root=store,
        capture_session_id=session, intake_id=intake, requested_claim_types=["navigation_clearance"],
        idempotency_key="owned-plan-1")
    return SimpleNamespace(state=state, store=store, plan=plan, digest=digest)


@pytest.fixture
def context(owned):
    return owned.plan["source_context"]


@pytest.fixture
def client(owned, configuration, monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "load_configuration", lambda *_, **__: configuration)
    enqueue_owned_atlas(state_root=owned.state, capture_store_root=owned.store,
        plan_id=owned.plan["plan_id"], task_context_digest=owned.plan["source_context"]["context_digest"],
        source_video_digest=owned.digest, owner_user_id="owner", idempotency_key="owned-execute")
    monkeypatch.delenv(service.INTAKE_TOKEN_ENV, raising=False)
    monkeypatch.setenv(service.INTAKE_CLIENT_SECRETS_ENV, json.dumps({
        "blueprint-webapp": "hermetic-signing-key", "another-client": "hermetic-signing-key"}))
    monkeypatch.setenv(service.INTAKE_ALLOW_LEGACY_BEARER_ENV, "false")
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "admission"))
    service._INTAKE_NONCE_CACHE.clear()
    app = FastAPI()
    register_owned_capture_atlas_routes(app, require_admission=service._require_admission,
                                        roots=lambda: (owned.state, owned.store))
    return TestClient(app)


def _read_command(owned, *, bundle=False):
    return {"schema_version": ("owned_capture_atlas_bundle_submission.v1" if bundle
                               else "owned_capture_atlas_inspection_submission.v1"),
            "plan_id": owned.plan["plan_id"], "owner_user_id": "owner",
            **({"digest": "sha256:" + "f" * 64} if bundle else {})}


def _post(client, operation, value, *, nonce="owned-read", issuer="blueprint-webapp"):
    body = json.dumps(value)
    return client.post("/api/live-pipeline/reconstructions/atlas-" + operation, content=body,
        headers=_signed_intake_headers("hermetic-signing-key", body, client_id=issuer, nonce=nonce))


def test_signed_read_body_cannot_be_retargeted_through_unsigned_get(owned, client):
    # This authenticated body explicitly names a different owner and plan.
    # The old GET implementation ignores it and executes the unsigned target.
    body = json.dumps({"schema_version": "owned_capture_atlas_inspection_submission.v1",
                       "plan_id": "reconstruction-atlas-" + "a" * 24, "owner_user_id": "another-owner"})
    headers = _signed_intake_headers("hermetic-signing-key", body, client_id="blueprint-webapp",
                                    nonce="owned-read-target-substitution")
    response = client.request("GET", f'/api/live-pipeline/reconstructions/atlas/{owned.plan["plan_id"]}',
                              params={"owner_user_id": "owner"}, content=body, headers=headers)
    assert response.status_code in {401, 403, 404, 405, 409, 422}, response.text


def test_actual_hmac_read_and_replay_controls(owned, client):
    value = _read_command(owned)
    body = json.dumps(value)
    headers = _signed_intake_headers("hermetic-signing-key", body, client_id="blueprint-webapp", nonce="read-replay")
    response = client.post("/api/live-pipeline/reconstructions/atlas-inspect", content=body, headers=headers,
                           params={"owner_user_id": "unsigned-other-owner", "plan_id": "unsigned-other-plan"})
    assert response.status_code == 200, response.text
    assert response.json()["plan_id"] == owned.plan["plan_id"]
    assert response.json()["state"] == "processing"
    assert client.post("/api/live-pipeline/reconstructions/atlas-inspect", content=body,
                       headers=headers).status_code == 401


@pytest.mark.parametrize("bundle,field,value", [
    (False, "owner_user_id", "another-owner"),
    (False, "plan_id", "reconstruction-atlas-" + "a" * 24),
    (True, "owner_user_id", "another-owner"),
    (True, "plan_id", "reconstruction-atlas-" + "a" * 24),
    (True, "digest", "sha256:" + "a" * 64),
])
def test_changed_signed_target_fields_fail_actual_hmac(owned, client, bundle, field, value):
    command = _read_command(owned, bundle=bundle)
    body = json.dumps(command)
    headers = _signed_intake_headers("hermetic-signing-key", body, client_id="blueprint-webapp", nonce="tampered-read")
    response = client.post("/api/live-pipeline/reconstructions/atlas-" + ("bundle" if bundle else "inspect"),
                           content=json.dumps({**command, field: value}), headers=headers)
    assert response.status_code == 401


@pytest.mark.parametrize("operation", ["inspect", "bundle"])
def test_valid_other_service_signature_is_not_webapp_owner_authority(owned, client, operation):
    response = _post(client, operation, _read_command(owned, bundle=operation == "bundle"), issuer="another-client")
    assert response.status_code == 403


@pytest.mark.parametrize("operation", ["inspect", "bundle"])
def test_valid_signature_for_other_read_operation_cannot_cross_routes(owned, client, operation):
    command = _read_command(owned, bundle=operation == "inspect")
    assert _post(client, operation, command).status_code == 422


def test_signed_wrong_owner_cannot_read_actual_job(owned, client):
    command = {**_read_command(owned), "owner_user_id": "another-owner"}
    assert _post(client, "inspect", command).status_code == 409


def test_actual_hmac_owner_bundle_stream_and_digest_binding(owned, client, monkeypatch):
    payload = b"hermetic-private-reconstruction-bundle"
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    job_path = owned.state / "plans" / owned.plan["plan_id"] / "atlas" / "owned_job.json"
    job = json.loads(job_path.read_text())
    job.update(state="ready", stage="archived", archive={"digest": digest, "size_bytes": len(payload),
        "filename": "capture-session-1.zip", "manifest_digest": "sha256:" + "b" * 64})
    job_path.write_text(json.dumps(job))
    reads = []

    class Body(BytesIO):
        def iter_chunks(self, size):
            while data := self.read(size):
                yield data

    body = Body(payload)

    class Store:
        def head_object(self, *, Bucket, Key):
            reads.append(("head", Key))
            assert f"/sha256/{digest[7:]}/capture-session-1.zip" in Key
            return {"Metadata": {"sha256": digest[7:]}, "ContentLength": len(payload)}

        def get_object(self, *, Bucket, Key):
            reads.append(("get", Key))
            return {"Body": body}

    monkeypatch.setattr(archives, "_artifact_object_store_client", lambda: (Store(), "hermetic-private-bucket"))
    command = {**_read_command(owned, bundle=True), "digest": digest}
    response = _post(client, "bundle", command, nonce="valid-private-download")
    assert response.status_code == 200, response.text
    assert response.content == payload
    assert response.headers["X-Blueprint-Artifact-SHA256"] == digest
    assert response.headers["Cache-Control"] == "private, no-store"
    assert body.closed
    assert len(reads) == 2
    wrong_digest = _post(client, "bundle", {**command, "digest": "sha256:" + "a" * 64}, nonce="wrong-download-digest")
    wrong_owner = _post(client, "bundle", {**command, "owner_user_id": "another-owner"}, nonce="wrong-download-owner")
    assert wrong_digest.status_code == 404 and wrong_owner.status_code == 409
    assert len(reads) == 2
