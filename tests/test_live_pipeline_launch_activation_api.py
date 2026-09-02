from __future__ import annotations

import hmac
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.live_pipeline_intake_service import create_app
from tests.test_task_evaluation_launch_activation_contract import request


def signed_headers(body: str, *, nonce: str) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).isoformat()
    client_id = "blueprint-webapp"
    signature = hmac.new(
        b"test-intake-token",
        f"{timestamp}.{client_id}.{nonce}.{body}".encode(),
        "sha256",
    ).hexdigest()
    return {
        "content-type": "application/json",
        "x-blueprint-pipeline-timestamp": timestamp,
        "x-blueprint-pipeline-nonce": nonce,
        "x-blueprint-pipeline-client-id": client_id,
        "x-blueprint-pipeline-signature": f"sha256={signature}",
    }


def configure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        service,
        "deployment_identity_payload",
        lambda: {
            "commit_proven": True,
            "source_commit": "a" * 40,
            "blockers": [],
        },
    )
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "intake"))
    monkeypatch.setenv(
        service.INTAKE_CLIENT_SECRETS_ENV,
        json.dumps({"blueprint-webapp": "test-intake-token"}),
    )
    monkeypatch.setenv(
        service.INTAKE_NONCE_STORE_DIR_ENV,
        str(tmp_path / "nonces"),
    )
    monkeypatch.setenv(
        service.TASK_EVALUATION_LAUNCH_ACTIVATION_QUEUE_ROOT_ENV,
        str(tmp_path / "activations"),
    )


def test_authenticated_webapp_queues_and_reads_activation_without_mutation(
    monkeypatch, tmp_path
) -> None:
    configure(monkeypatch, tmp_path)
    value = request()
    body = json.dumps(value, separators=(",", ":"))
    client = TestClient(create_app())
    response = client.post(
        "/api/live-pipeline/task-evaluation-launch-activations",
        data=body,
        headers=signed_headers(body, nonce="activation-submit-001"),
    )
    assert response.status_code == 202
    receipt = response.json()
    assert receipt["status"] == "queued_for_authority_gated_activation"
    assert receipt["activation_id"] == value["activation_id"]
    assert receipt["provider_mutation_performed_inside_http_request"] is False
    assert receipt["catalog_mutation_performed_inside_http_request"] is False
    assert receipt["standing_authorization_published_inside_http_request"] is False
    assert receipt["paid_execution_requested"] is False
    assert "queue_path" not in receipt
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    status = client.get(
        "/api/live-pipeline/task-evaluation-launch-activations/"
        + value["activation_id"],
        headers=signed_headers("", nonce="activation-status-001"),
    )
    assert status.status_code == 200
    assert status.json()["status"] == "pending"
    assert status.json()["request_digest"] == receipt["request_digest"]


def test_activation_api_fails_before_queue_on_stale_commit_or_host_path(
    monkeypatch, tmp_path
) -> None:
    configure(monkeypatch, tmp_path)
    client = TestClient(create_app())
    stale = request()
    stale["expected_production_commit"] = "b" * 40
    body = json.dumps(stale, separators=(",", ":"))
    response = client.post(
        "/api/live-pipeline/task-evaluation-launch-activations",
        data=body,
        headers=signed_headers(body, nonce="activation-stale-001"),
    )
    assert response.status_code == 409
    assert response.json()["blockers"] == [
        "launch_activation_production_commit_mismatch"
    ]

    unsafe = request()
    unsafe["release_window"]["uri"] = "/etc/blueprint/window.json"
    body = json.dumps(unsafe, separators=(",", ":"))
    response = client.post(
        "/api/live-pipeline/task-evaluation-launch-activations",
        data=body,
        headers=signed_headers(body, nonce="activation-unsafe-001"),
    )
    assert response.status_code == 400
    assert response.json()["accepted"] is False
    assert not list((tmp_path / "activations" / "pending").glob("*.json"))


def test_activation_api_refuses_exhausted_disk_before_queueing(
    monkeypatch, tmp_path
) -> None:
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(
        service,
        "deployment_identity_payload",
        lambda: {
            "commit_proven": True,
            "source_commit": "a" * 40,
            "blockers": [],
            "disk_headroom": {
                "status": "exhausted",
                "refused_roles": ["launch_activation"],
            },
        },
    )
    body = json.dumps(request(), separators=(",", ":"))
    response = TestClient(create_app()).post(
        "/api/live-pipeline/task-evaluation-launch-activations",
        data=body,
        headers=signed_headers(body, nonce="activation-disk-001"),
    )

    assert response.status_code == 503
    assert response.json()["blockers"] == [
        "launch_activation_disk_headroom_exhausted"
    ]
    assert not list((tmp_path / "activations" / "pending").glob("*.json"))
