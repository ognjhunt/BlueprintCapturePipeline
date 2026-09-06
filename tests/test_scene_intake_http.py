"""The owner-intent endpoint requires exact-byte HMAC and an admitted issuer."""

import hmac
import json
import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline import task_evaluation_scene_intake as intake
from tests.test_task_evaluation_scene_intake import request


def test_signed_intake_and_nonce_replay(tmp_path, monkeypatch):
    monkeypatch.setenv(service.INTAKE_TOKEN_ENV, "test-token")
    monkeypatch.delenv(service.INTAKE_CLIENT_SECRETS_ENV, raising=False)
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "admission"))
    monkeypatch.setenv(intake.ROOT_ENV, str(tmp_path / "queue"))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    service._INTAKE_NONCE_CACHE.clear()
    monkeypatch.setattr(service, "deployment_identity_payload", lambda: {})
    value = request()
    now = time.time()
    value["execution"]["expires_at_epoch"] = now + 1000
    value["consent"]["accepted_at_epoch"] = now - 1
    body = json.dumps(value)
    timestamp = datetime.now(timezone.utc).isoformat()
    nonce = "scene-intake-nonce-1"
    signature = hmac.new(b"test-token", f"{timestamp}.webapp.{nonce}.{body}".encode(), "sha256").hexdigest()
    headers = {"Content-Type": "application/json", "x-blueprint-pipeline-client-id": "webapp",
        "x-blueprint-pipeline-timestamp": timestamp, "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-signature": "sha256=" + signature}
    client = TestClient(service.create_app())
    endpoint = "/api/live-pipeline/task-evaluation-scene-intents"
    assert client.post(endpoint, content=body).status_code == 401
    response = client.post(endpoint, content=body, headers=headers)
    assert response.status_code == 202, response.text
    assert response.json()["provider_mutation_performed_inside_http_request"] is False
    assert client.post(endpoint, content=body, headers=headers).status_code == 401
    assert len(list((tmp_path / "queue").glob("scene-*"))) == 1
