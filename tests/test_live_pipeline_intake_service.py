from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from blueprint_pipeline.live_pipeline_control_plane import (
    CONTROL_PLANE_OUTPUT_PATH_ENV,
    run_live_pipeline_control_plane,
)
from blueprint_pipeline.live_pipeline_intake_service import (
    INTAKE_TOKEN_ENV,
    create_app,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def _control_manifest(tmp_path: Path, capture_root: Path) -> Path:
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=output_path,
    )
    return output_path


def _webapp_request(capture_root: Path, *, job_id: str = "webapp-job-1") -> dict[str, object]:
    buyer_request_id = "buyer-request-1"
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "buyer_request_id": buyer_request_id,
            "site_package": {
                "capture_root": str(capture_root),
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
                "buyer_request_id": buyer_request_id,
                "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
            },
            "owner_system": {
                "name": "Blueprint-WebApp",
                "request_id": job_id,
                "buyer_request_id": buyer_request_id,
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
            },
            "source": {
                "system": "Blueprint-WebApp",
                "selection_state": {
                    "buyer_request_id": buyer_request_id,
                    "site_submission_id": "site-submission-1",
                    "capture_job_id": "capture-job-1",
                },
            },
        },
    }


def test_live_pipeline_intake_service_requires_token(tmp_path: Path, monkeypatch) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.delenv(INTAKE_TOKEN_ENV, raising=False)
    client = TestClient(create_app())

    response = client.post("/api/live-pipeline/job-requests", json=_webapp_request(capture_root))

    assert response.status_code == 503


def test_live_pipeline_intake_service_stages_webapp_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/job-requests",
        json=_webapp_request(capture_root),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["webapp_job_request"]["missing_fields"] == []
    assert payload["webapp_staging"]["performed"] is True
    assert Path(payload["webapp_staging"]["target_path"]).is_file()
    assert payload["trigger"]["status"] == "not_configured"
    assert payload["proof_boundary"]["intake_sets_proof_booleans"] is False


def test_live_pipeline_intake_service_exposes_latest_audit(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())
    client.post(
        "/api/live-pipeline/job-requests",
        json=_webapp_request(capture_root),
        headers={"x-blueprint-intake-token": "test-intake-token"},
    )

    response = client.get(
        "/api/live-pipeline/intake-audit",
        headers={"x-blueprint-intake-token": "test-intake-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["webapp_staging"]["performed"] is True
