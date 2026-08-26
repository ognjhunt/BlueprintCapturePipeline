from __future__ import annotations

import json
from datetime import datetime, timezone
import hmac

from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
from blueprint_pipeline.live_pipeline_intake_service import create_app
from blueprint_pipeline.scene_object_discovery_queue import (
    seal_scene_object_discovery_result,
)
from tests.test_scene_object_discovery_contract import request


def _digest(character: str) -> str:
    return "sha256:" + character * 64


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
    monkeypatch.setenv(service.INTAKE_NONCE_STORE_DIR_ENV, str(tmp_path / "nonces"))
    monkeypatch.setenv(
        service.SCENE_OBJECT_DISCOVERY_QUEUE_ROOT_ENV,
        str(tmp_path / "discoveries"),
    )


def test_webapp_can_stage_poll_and_select_scene_object_discovery(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)
    value = request()
    body = json.dumps(value, separators=(",", ":"))
    client = TestClient(create_app())
    queued = client.post(
        "/api/live-pipeline/scene-object-discoveries",
        content=body,
        headers=signed_headers(body, nonce="discovery-submit-001"),
    )
    assert queued.status_code == 202
    receipt = queued.json()
    assert receipt["status"] == "queued_for_no_spend_discovery_preparation"
    assert receipt["provider_mutation_performed_inside_http_request"] is False
    assert receipt["paid_execution_requested"] is False

    pending = client.get(
        f"/api/live-pipeline/scene-object-discoveries/{value['discovery_id']}",
        headers=signed_headers("", nonce="discovery-status-001"),
    )
    assert pending.status_code == 200
    assert pending.json()["status"] == "pending"

    candidates = []
    for index in range(2):
        candidates.append(
            {
                "candidate_id": f"sam31-tote-{index + 1:03d}",
                "label": "red tote",
                "backend": "sam31",
                "confidence": 0.9,
                "task_match_score": 0.9,
                "eligible_for_automatic_source_object": True,
                "candidate_claim_boundary": "metric_source_object_candidate",
                "metric_geometry_authority": "production_semantic_gaussian_obb",
                "metric_geometry": {"evidence_digest": _digest(str(index + 1))},
                "source_object_artifact": {
                    "uri": f"https://objects.example/source-{index}.json",
                    "digest": _digest(str(index + 3)),
                    "size_bytes": 100,
                },
            }
        )
    discovery = {
        "schema_version": "scene_object_discovery.v1",
        "status": "selection_required",
        "discovery_digest": _digest("7"),
        "candidates": candidates,
        "selected_candidate_id": None,
        "source_object": None,
        "coverage": {"unseen_regions": ["behind_partition"]},
    }
    seal_scene_object_discovery_result(
        queue_root=tmp_path / "discoveries",
        discovery_id=value["discovery_id"],
        request_digest=receipt["request_digest"],
        source_commit=value["expected_production_commit"],
        discovery=discovery,
    )
    ready_for_selection = client.get(
        f"/api/live-pipeline/scene-object-discoveries/{value['discovery_id']}",
        headers=signed_headers("", nonce="discovery-status-002"),
    )
    assert ready_for_selection.status_code == 200
    assert ready_for_selection.json()["status"] == "selection_required"
    assert ready_for_selection.json()["candidates"][0]["candidate_id"] == "sam31-tote-001"

    selection = {
        "schema_version": "scene_object_discovery_selection_request.v1",
        "discovery_id": value["discovery_id"],
        "expected_production_commit": value["expected_production_commit"],
        "request_digest": receipt["request_digest"],
        "discovery_digest": _digest("7"),
        "candidate_id": "sam31-tote-001",
        "confirm_selection": True,
    }
    selection_body = json.dumps(selection, separators=(",", ":"))
    selected = client.post(
        f"/api/live-pipeline/scene-object-discoveries/{value['discovery_id']}/selection",
        content=selection_body,
        headers=signed_headers(selection_body, nonce="discovery-select-001"),
    )
    assert selected.status_code == 202
    assert selected.json()["status"] == "selection_sealed"
    assert selected.json()["paid_execution_requested"] is False

    final = client.get(
        f"/api/live-pipeline/scene-object-discoveries/{value['discovery_id']}",
        headers=signed_headers("", nonce="discovery-status-003"),
    )
    assert final.status_code == 200
    assert final.json()["status"] == "ready_auto_selected"
    assert final.json()["source_object"]["source_object_artifact"]["digest"] == _digest("3")


def test_discovery_api_rejects_stale_commit_before_queueing(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)
    value = request()
    value["expected_production_commit"] = "b" * 40
    body = json.dumps(value, separators=(",", ":"))
    response = TestClient(create_app()).post(
        "/api/live-pipeline/scene-object-discoveries",
        content=body,
        headers=signed_headers(body, nonce="discovery-stale-001"),
    )
    assert response.status_code == 409
    assert response.json()["blockers"] == ["scene_object_discovery_production_commit_mismatch"]
    assert not list((tmp_path / "discoveries" / "pending").glob("*.json"))
