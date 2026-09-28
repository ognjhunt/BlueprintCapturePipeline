"""Signed team policy selection is durable, owner-bound and no-spend."""

from __future__ import annotations

import hmac
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from blueprint_pipeline.native_g1_team_campaign_intake import REGISTRY_ENV
from blueprint_pipeline.native_g1_team_policy_run_intake import (
    QUEUE_ENV,
    stage_g1_team_policy_run,
)
from blueprint_pipeline.task_evaluation_scene_intake import CLIENTS_ENV
from tests.test_native_g1_team_campaign_intake import _registry
from tests.test_native_g1_team_policy_run_request import NOW, _request
from tests.test_native_g1_team_scored_scene_episode import _inputs
from tests.test_team_policy_delivery_profile import _profile


def _setup(tmp_path: Path, monkeypatch):
    setup, _, _ = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "container",
        "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
        "protocol": "jsonl_observation_action_v1",
    })
    request = _request(setup, profile)
    registry, binding = _registry(tmp_path, request)
    binding["owner"] = request["owner"]
    retained = {"schema_version": "native_g1_team_campaign_registry.v1", "bindings": [binding]}
    retained["registry_digest"] = digest(retained, digest_field="registry_digest")
    registry.write_text(json.dumps(retained), encoding="utf-8")
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_policy_run_intake.make_packet_planning_setup",
        lambda *, source_packet_dir: setup,
    )
    return setup, request, registry, binding


def test_team_policy_intake_is_one_use_and_does_not_open_runtime(tmp_path: Path, monkeypatch) -> None:
    _, request, registry, binding = _setup(tmp_path, monkeypatch)
    queue = tmp_path / "team-policy-queue"
    args = dict(
        value=request, registry_path=registry, queue_root=queue,
        authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"},
        now_epoch=NOW,
    )
    receipt = stage_g1_team_policy_run(**args)
    assert receipt["status"] == "accepted_pending_operator_approval"
    assert receipt["provider_mutation_performed_inside_http_request"] is False
    assert receipt["receipt_digest"] == digest(receipt, digest_field="receipt_digest")
    intent = json.loads((queue / receipt["intent_id"] / "intent.json").read_text())
    assert intent["request"] == request
    assert intent["binding"] == binding
    assert intent["provider_mutation_performed"] is False
    assert stage_g1_team_policy_run(**args) == receipt
    changed = {**request, "run_id": request["run_id"]}
    changed["authorization"] = {**request["authorization"], "maximum_cost_usd": 11}
    changed["request_digest"] = digest(changed, digest_field="request_digest")
    with pytest.raises(ValueError, match="idempotency_conflict"):
        stage_g1_team_policy_run(**{**args, "value": changed})


def test_team_policy_http_requires_signature_and_retains_no_spend_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    _, request, registry, _ = _setup(tmp_path, monkeypatch)
    request["authorization"]["expires_at_epoch"] = time.time() + 3600
    request["request_digest"] = digest(request, digest_field="request_digest")
    monkeypatch.setenv(REGISTRY_ENV, str(registry))
    monkeypatch.setenv(QUEUE_ENV, str(tmp_path / "team-policy-queue"))
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "test-token")
    monkeypatch.delenv(service.INTAKE_CLIENT_SECRETS_ENV, raising=False)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "admission"))
    monkeypatch.setenv(CLIENTS_ENV, "webapp")
    monkeypatch.setattr(service, "deployment_identity_payload", lambda: {})
    service._INTAKE_NONCE_CACHE.clear()
    client = TestClient(service.create_app())
    endpoint = "/api/live-pipeline/native-g1-team-policy-runs"
    body = json.dumps(request)
    assert client.post(endpoint, content=body).status_code == 401
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "native-g1-team-policy-test"
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
    assert response.json()["status"] == "accepted_pending_operator_approval"
    assert response.json()["provider_mutation_performed_inside_http_request"] is False
    assert len(list((tmp_path / "team-policy-queue").glob("g1-team-policy-*/intent.json"))) == 1
