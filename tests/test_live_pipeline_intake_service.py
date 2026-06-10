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


def _live_closure_evidence(job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "schema_version": "live_robot_eval_closure_evidence.v1",
        "job_id": job_id,
        "review_acceptance": {"accepted": True, "reviewer": "owner-reviewer"},
        "delivery": {
            "storage_upload_performed": True,
            "signed_urls": ["https://delivery.example/signed/package-1"],
            "entitlement_verified": True,
        },
        "safety_contact_physics": {
            "physics_contact_validated": True,
            "safety_validated": True,
            "robot_readiness_proven": True,
            "methodology_uri_or_path": "owner://methodology",
            "contact_validation_uri_or_path": "owner://contact",
            "safety_validation_uri_or_path": "owner://safety",
            "operator_attestation": {
                "attested_by": "safety-owner",
                "attestation": "Owner accepted contact, physics, and safety evidence.",
            },
        },
    }


def _deployment_outcomes(
    job_id: str = "webapp-job-1",
    *,
    include_evidence: bool = True,
) -> dict[str, object]:
    record: dict[str, object] = {
        "outcome_id": "pilot-outcome-1",
        "task_id": "place_return_in_bin",
        "scenario_id": "scenario_place_return_in_bin_mobile",
        "actual_success": False,
        "failure_mode_ids": ["missed_blocked_path"],
    }
    if include_evidence:
        record["evidence_refs"] = {"pilot_log": "owner://pilot/pilot-outcome-1"}
    return {
        "schema_version": "deployment_outcome_manifest.v1",
        "job_id": job_id,
        "records": [record],
    }


def _policy_package(job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "schema_version": "robot_team_policy_package.v1",
        "job_id": job_id,
        "policy_package": {
            "policy_api_endpoint": {
                "endpoint_url": "https://robot-team.example/policy",
                "observation_schema_ref": "schemas/obs-v1.json",
                "action_schema_ref": "schemas/action-v1.json",
            }
        },
    }


def _real_robot_pov_manifest(job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "schema_version": "real_robot_pov_manifest.v1",
        "job_id": job_id,
        "owner_system": "robot-team-owner-system",
        "records": [
            {
                "evidence_id": "real-pov-1",
                "task_id": "place_return_in_bin",
                "scenario_id": "scenario_place_return_in_bin_mobile",
                "scenario_eval_run_id": "scenario-run-1",
                "scenario_variation_instance_id": "scenario-variation-1",
                "robot_camera_video_uri": "owner://pov/scenario-run-1.mp4",
                "action_log_uri": "owner://actions/scenario-run-1.jsonl",
                "timestamp_alignment": "aligned_to_scenario_eval_run",
                "owner_evidence_refs": {
                    "camera": "owner://pov/scenario-run-1.mp4",
                    "action_log": "owner://actions/scenario-run-1.jsonl",
                },
                "operator_attestation": {
                    "attested_by": "robot-team-ops",
                    "attestation": "Robot POV and action logs are aligned to this eval run.",
                },
            }
        ],
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


def test_live_pipeline_intake_service_records_blocked_webapp_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    other_capture_root = (
        tmp_path
        / "storage"
        / "bucket"
        / "scenes"
        / "scene-2"
        / "captures"
        / "capture-2"
    )
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/job-requests",
        json=_webapp_request(other_capture_root),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert (
        "webapp:request_capture_root_does_not_match_control_plane"
        in payload["input_blockers"]
    )
    assert payload["webapp_job_request"]["capture_root_matches_control_plane"] is False
    assert payload["webapp_staging"]["performed"] is False
    assert payload["trigger"]["status"] == "not_run"
    assert Path(payload["candidate"]["path"]).is_file()


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


def test_live_pipeline_intake_service_stages_deployment_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/deployment-outcomes",
        json=_deployment_outcomes(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["deployment_outcomes_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert payload["deployment_outcomes"]["record_count"] == 1
    assert payload["deployment_outcomes"]["owner_evidence_ready"] is True
    assert payload["deployment_outcomes"]["owner_evidence_record_count"] == 1
    assert payload["deployment_outcomes"]["missing_owner_evidence_record_ids"] == []
    assert payload["deployment_outcomes_staging"]["performed"] is True
    assert target_path == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "deployment_outcomes"
        / "inbox"
        / "pilot-outcome-1.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["real_world_outcome_proven"] is False


def test_live_pipeline_intake_service_accepts_outcome_records_without_owner_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/deployment-outcomes",
        json=_deployment_outcomes(include_evidence=False),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert payload["deployment_outcomes"]["record_count"] == 1
    assert payload["deployment_outcomes"]["owner_evidence_ready"] is False
    assert payload["deployment_outcomes"]["owner_evidence_record_count"] == 0
    assert payload["deployment_outcomes"]["missing_owner_evidence_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert payload["deployment_outcomes_staging"]["performed"] is True


def test_live_pipeline_intake_service_stages_policy_package(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/policy-packages",
        json=_policy_package(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["policy_package_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["policy_package"]["status"] == "ready_for_robot_eval_job"
    assert payload["policy_package"]["selected_modalities"] == ["policy_api_endpoint"]
    assert payload["policy_package_staging"]["performed"] is True
    assert target_path == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "policy_package.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["robot_policy_execution_proven"] is False


def test_live_pipeline_intake_service_records_blocked_policy_package(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/policy-packages",
        json={
            "schema_version": "robot_team_policy_package.v1",
            "job_id": "../escape",
            "policy_package": {"docker_container": {"image_ref": "registry.example/policy:latest"}},
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "policy_package:policy_package_job_id_unsafe" in payload["input_blockers"]
    assert "policy_package:policy_package.docker_container.digest" in payload[
        "input_blockers"
    ]
    assert payload["policy_package_staging"]["performed"] is False


def test_live_pipeline_intake_service_stages_real_robot_pov_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/real-robot-pov",
        json=_real_robot_pov_manifest(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["real_robot_pov_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["real_robot_pov"]["status"] == "ready_for_robot_eval_job"
    assert payload["real_robot_pov"]["record_count"] == 1
    assert payload["real_robot_pov"]["missing_exact_key_record_ids"] == []
    assert payload["real_robot_pov"]["missing_evidence_record_ids"] == []
    assert payload["real_robot_pov_staging"]["performed"] is True
    assert target_path == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "real_robot_pov_manifest.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["robot_pov_evidence_proven"] is False


def test_live_pipeline_intake_service_records_blocked_real_robot_pov_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/real-robot-pov",
        json={
            "schema_version": "real_robot_pov_manifest.v1",
            "job_id": "../escape",
            "records": [
                {
                    "evidence_id": "real-pov-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "robot_camera_video_uri": "owner://pov/scenario-run-1.mp4",
                }
            ],
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "real_robot_pov:real_robot_pov_job_id_unsafe" in payload["input_blockers"]
    assert "real_robot_pov:real_robot_pov_missing_exact_keys" in payload[
        "input_blockers"
    ]
    assert "real_robot_pov:real_robot_pov_missing_action_logs" in payload[
        "input_blockers"
    ]
    assert payload["real_robot_pov_staging"]["performed"] is False


def test_live_pipeline_intake_service_records_blocked_deployment_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/deployment-outcomes",
        json={
            "schema_version": "deployment_outcome_manifest.v1",
            "job_id": "../escape",
            "records": [{"task_id": "task-only"}],
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "deployment_outcomes:deployment_outcomes_job_id_unsafe" in payload[
        "input_blockers"
    ]
    assert payload["deployment_outcomes_staging"]["performed"] is False


def test_live_pipeline_intake_service_stages_live_closure_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/live-closure-evidence",
        json=_live_closure_evidence(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    target_path = Path(payload["live_closure_evidence_staging"]["target_path"])
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["live_closure_evidence"]["status"] == "ready_for_closure_audit"
    assert payload["live_closure_evidence_staging"]["performed"] is True
    assert target_path == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "live_eval_closure_evidence.json"
    )
    assert target_path.is_file()
    assert payload["proof_boundary"]["intake_sets_proof_booleans"] is False


def test_live_pipeline_intake_service_records_blocked_closure_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/live-closure-evidence",
        json={
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "job_id": "../escape",
        },
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "live_closure_evidence:live_closure_evidence_job_id_unsafe" in payload[
        "input_blockers"
    ]
    assert payload["live_closure_evidence_staging"]["performed"] is False
