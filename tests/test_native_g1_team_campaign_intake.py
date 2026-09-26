"""G1 team intake binds a signed choice to trusted packet paths without spend."""

from __future__ import annotations

import json
import hmac
import time
from copy import deepcopy
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.native_g1_development_pair import PAIR_ORDER
from blueprint_pipeline.native_g1_team_campaign_intake import (
    QUEUE_ENV,
    REGISTRY_ENV,
    REGISTRY_SCHEMA,
    stage_g1_team_campaign,
)
from blueprint_pipeline.task_evaluation_scene_intake import CLIENTS_ENV
from tests.test_native_g1_team_campaign_request import NOW, OWNER, _request, _reseal


def _registry(tmp_path, request):
    source = tmp_path / "source"
    source.mkdir()
    directories = {name: tmp_path / name for name in (
        "manipulation_packet_dir", "movement_packet_dir", "publisher_source_dir"
    )}
    for path in directories.values():
        path.mkdir()
    files = {name: tmp_path / name for name in (
        "navigation_authority_path", "runtime_source_receipt_path"
    )}
    for path in files.values():
        path.write_text("{}", encoding="utf-8")
    rights = {name: tmp_path / f"rights-{name}.json" for name in PAIR_ORDER}
    for path in rights.values():
        path.write_text("{}", encoding="utf-8")
    binding = {
        "owner": OWNER,
        "scene_id": request["scene_id"],
        "task_id": request["task_id"],
        "source_packet_receipt_digest": request["source_packet_receipt_digest"],
        "source_packet_dir": str(source),
        **{name: str(path) for name, path in directories.items()},
        **{name: str(path) for name, path in files.items()},
        "rights_review_paths": {name: str(path) for name, path in rights.items()},
    }
    registry = {"schema_version": REGISTRY_SCHEMA, "bindings": [binding]}
    registry["registry_digest"] = digest(registry, digest_field="registry_digest")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path, binding


def test_stages_one_immutable_no_spend_intent(tmp_path, monkeypatch):
    setup, request = _request(tmp_path, monkeypatch)
    registry, binding = _registry(tmp_path, request)
    seen = []
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_intake.make_packet_planning_setup",
        lambda *, source_packet_dir: seen.append(source_packet_dir) or setup,
    )
    args = dict(
        value=request, registry_path=registry, queue_root=tmp_path / "queue",
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"},
        now_epoch=NOW,
    )
    receipt = stage_g1_team_campaign(**args)
    assert receipt["status"] == "accepted_not_dispatched"
    assert receipt["provider_mutation_performed_inside_http_request"] is False
    assert seen == [tmp_path / "source"]
    intent = json.loads((args["queue_root"] / receipt["intent_id"] / "intent.json").read_text())
    assert intent["request"] == request
    assert intent["binding"] == binding
    assert intent["provider_mutation_performed"] is False
    assert intent["intent_digest"] == receipt["intent_digest"]
    assert stage_g1_team_campaign(**args) == receipt


def test_rejects_changed_request_for_same_run_id(tmp_path, monkeypatch):
    setup, request = _request(tmp_path, monkeypatch)
    registry, _ = _registry(tmp_path, request)
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_intake.make_packet_planning_setup",
        lambda **_: setup,
    )
    args = dict(
        registry_path=registry, queue_root=tmp_path / "queue",
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"},
        now_epoch=NOW,
    )
    stage_g1_team_campaign(value=request, **args)
    changed = deepcopy(request)
    changed["authorization"]["expires_at_epoch"] += 1
    _reseal(changed)
    with pytest.raises(ValueError, match="idempotency_conflict"):
        stage_g1_team_campaign(value=changed, **args)


@pytest.mark.parametrize("change, error", [
    ("issuer", "issuer_not_authorized"),
    ("owner", "binding_unavailable"),
    ("packet", "binding_unavailable"),
    ("registry_digest", "record_invalid"),
    ("binding_packet", "binding_packet_mismatch"),
])
def test_rejects_untrusted_intake(tmp_path, monkeypatch, change, error):
    setup, request = _request(tmp_path, monkeypatch)
    registry, _ = _registry(tmp_path, request)
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_intake.make_packet_planning_setup",
        lambda **_: setup,
    )
    if change in {"owner", "packet"}:
        request = deepcopy(request)
        if change == "owner":
            request["owner"] = {"user_id": "other", "organization_id": "team-org"}
        else:
            request["source_packet_receipt_digest"] = "sha256:" + "9" * 64
        _reseal(request)
    elif change == "registry_digest":
        value = json.loads(registry.read_text())
        value["bindings"][0]["task_id"] = "other-task"
        registry.write_text(json.dumps(value), encoding="utf-8")
    elif change == "binding_packet":
        value = json.loads(registry.read_text())
        value["bindings"][0]["task_id"] = "other-task"
        value["registry_digest"] = digest(value, digest_field="registry_digest")
        registry.write_text(json.dumps(value), encoding="utf-8")
    client = "untrusted" if change == "issuer" else "blueprint-webapp"
    with pytest.raises(ValueError, match=error):
        stage_g1_team_campaign(
            value=request, registry_path=registry, queue_root=tmp_path / "queue",
            authenticated_client=client, trusted_clients={"blueprint-webapp"},
            now_epoch=NOW,
        )


def test_http_requires_signature_and_stages_without_provider(tmp_path, monkeypatch):
    setup, request = _request(tmp_path, monkeypatch)
    request["authorization"]["expires_at_epoch"] = time.time() + 3600
    _reseal(request)
    registry, _ = _registry(tmp_path, request)
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_intake.make_packet_planning_setup",
        lambda **_: setup,
    )
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    monkeypatch.setenv(QUEUE_ENV, str(tmp_path / "queue"))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "test-token")
    monkeypatch.delenv(service.INTAKE_CLIENT_SECRETS_ENV, raising=False)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "admission"))
    monkeypatch.setenv(CLIENTS_ENV, "webapp")
    monkeypatch.setattr(service, "deployment_identity_payload", lambda: {})
    service._INTAKE_NONCE_CACHE.clear()
    client = TestClient(service.create_app())
    endpoint = "/api/live-pipeline/native-g1-team-campaigns"
    body = json.dumps(request)
    assert client.post(endpoint, content=body).status_code == 401
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "native-g1-team-campaign-test"
    signature = hmac.new(
        b"test-token", f"{timestamp}.webapp.{nonce}.{body}".encode(), "sha256"
    ).hexdigest()
    response = client.post(endpoint, content=body, headers={
        "content-type": "application/json",
        "x-blueprint-pipeline-client-id": "webapp",
        "x-blueprint-pipeline-timestamp": timestamp,
        "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-signature": "sha256=" + signature,
    })
    assert response.status_code == 202, response.text
    assert response.json()["provider_mutation_performed_inside_http_request"] is False
    assert len(list((tmp_path / "queue").glob("g1-*/intent.json"))) == 1
    assert client.post(endpoint, content=body, headers={
        "content-type": "application/json",
        "x-blueprint-pipeline-client-id": "webapp",
        "x-blueprint-pipeline-timestamp": timestamp,
        "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-signature": "sha256=" + signature,
    }).status_code == 401
