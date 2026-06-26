from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline import live_pipeline_intake_service as service
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


def _seed_robot_eval_dataset_cards(capture_root: Path) -> None:
    dataset_dir = capture_root / "pipeline" / "robot_eval_dataset"
    _write_json(
        dataset_dir / "task_cards.json",
        {
            "schema_version": "robot_eval_task_cards.v1",
            "cards": [
                {
                    "task_id": "scene_anchor_geometry_0",
                    "task_card_id": "task-card-1",
                }
            ],
        },
    )
    _write_json(
        dataset_dir / "scenario_cards.json",
        {
            "schema_version": "robot_eval_scenario_cards.v1",
            "cards": [
                {
                    "scenario_id": "scenario_scene_anchor_geometry_0_unitree_g1",
                    "task_id": "scene_anchor_geometry_0",
                    "robot_profile_id": "unitree_g1",
                }
            ],
        },
    )


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


def _capture_handoff() -> dict[str, object]:
    return {
        "schema_version": "blueprint_capture_pipeline_handoff.v1",
        "scene_id": "scene-1",
        "capture_id": "capture-1",
        "site_submission_id": "site-submission-1",
        "buyer_request_id": "buyer-request-1",
        "capture_job_id": "capture-job-1",
        "requested_outputs": ["robot_eval_dataset", "task_evaluation_run"],
        "requested_lanes": ["evaluation_prep", "robot_eval_dataset", "task_evaluation_run"],
        "robot_eval_dataset_requested": True,
        "capture_descriptor_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/capture_descriptor.json",
        "pipeline_handoff_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline_handoff.json",
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
            "non_ranking_operational_claim_validated": True,
            "rank_fidelity_result_proven": True,
            "methodology_uri_or_path": "owner://methodology",
            "contact_validation_uri_or_path": "owner://contact",
            "non_ranking_operational_claim_uri_or_path": "owner://safety",
            "operator_attestation": {
                "attested_by": "safety-owner",
                "attestation": "Owner accepted contact, physics, and safety evidence.",
            },
        },
    }


def test_live_pipeline_intake_service_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "control" / "manifest.json"
    monkeypatch.setenv(service.INTAKE_WORK_DIR_ENV, str(tmp_path / "custom-work"))
    assert service._work_dir(manifest_path) == tmp_path / "custom-work"
    assert service._request_from_payload({"schema_version": "robot_eval_job_request.v1"})["schema_version"] == "robot_eval_job_request.v1"
    assert service._request_from_payload({"schema_version": "other"}) == {}
    assert service._first_string("", None) == ""
    assert service._list_from_payload(("a", "b")) == ["a", "b"]
    assert service._list_from_payload("bad") == []
    cards_path = tmp_path / "cards.json"
    _write_json(cards_path, [{"task_id": "task-1"}, "bad"])
    assert service._cards_from_file(cards_path) == [{"task_id": "task-1"}]

    missing_root = tmp_path / "missing-cards"
    assert service._select_dataset_task(missing_root) == (
        None,
        ["robot_eval_task_cards_missing", "robot_eval_scenario_cards_missing"],
    )
    empty_root = tmp_path / "empty-cards"
    _write_json(empty_root / "pipeline" / "robot_eval_dataset" / "task_cards.json", {"cards": []})
    _write_json(empty_root / "pipeline" / "robot_eval_dataset" / "scenario_cards.json", {"cards": []})
    assert service._select_dataset_task(empty_root) == (
        None,
        ["robot_eval_task_cards_empty", "robot_eval_scenario_cards_empty"],
    )
    unmatched_root = tmp_path / "unmatched-cards"
    _write_json(
        unmatched_root / "pipeline" / "robot_eval_dataset" / "task_cards.json",
        {"cards": [{}, {"task_id": "task-1"}]},
    )
    _write_json(
        unmatched_root / "pipeline" / "robot_eval_dataset" / "scenario_cards.json",
        {"cards": [{"task_id": "other", "scenario_id": "scenario-1"}]},
    )
    assert service._select_dataset_task(unmatched_root) == (
        None,
        ["robot_eval_no_task_scenario_pair"],
    )


def test_capture_handoff_blocker_edges(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    payload = {
        "scene_id": "other-scene",
        "capture_id": "other-capture",
        "requested_outputs": ["task_evaluation_run"],
    }
    envelope, audit = service._capture_handoff_to_webapp_request(payload=payload, capture_root=capture_root)

    assert envelope is None
    assert "capture_handoff_scene_id_mismatch" in audit["blockers"]
    assert "capture_handoff_capture_id_mismatch" in audit["blockers"]
    assert "capture_handoff_missing_site_submission_id" in audit["blockers"]


def test_trigger_control_plane_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(service.INTAKE_TRIGGER_ENV, "echo trigger")
    monkeypatch.delenv(service.INTAKE_ALLOW_TRIGGER_ENV, raising=False)
    assert service._trigger_control_plane()["status"] == "blocked"

    class Completed:
        returncode = 0
        stdout = "x" * 2100
        stderr = "err"

    monkeypatch.setenv(service.INTAKE_ALLOW_TRIGGER_ENV, "true")
    monkeypatch.setattr(service.subprocess, "run", lambda *_args, **_kwargs: Completed())
    triggered = service._trigger_control_plane()
    assert triggered["status"] == "triggered"
    assert len(triggered["stdout_tail"]) == 2000


def test_live_pipeline_intake_service_error_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    missing_manifest = tmp_path / "missing-manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(missing_manifest))
    headers = {"x-blueprint-intake-token": "token"}

    assert client.get("/health").json()["manifest_path"] == str(missing_manifest)
    assert client.get("/health").json()["manifest_exists"] is False
    assert client.post("/api/live-pipeline/job-requests", json={}, headers={"x-blueprint-intake-token": "bad"}).status_code == 401

    endpoints = [
        "/api/live-pipeline/job-requests",
        "/api/live-pipeline/capture-handoffs",
        "/api/live-pipeline/policy-packages",
        "/api/live-pipeline/real-robot-pov",
        "/api/live-pipeline/deployment-outcomes",
        "/api/live-pipeline/live-closure-evidence",
    ]
    for endpoint in endpoints:
        assert client.post(endpoint, data="{", headers={**headers, "content-type": "application/json"}).status_code == 400
        assert client.post(endpoint, json=[], headers=headers).status_code == 400
        assert client.post(endpoint, json={}, headers=headers).status_code == 503

    manifest_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    _write_json(manifest_path, ["bad"])
    assert client.post("/api/live-pipeline/capture-handoffs", json={}, headers=headers).status_code == 503
    _write_json(manifest_path, {})
    assert client.post("/api/live-pipeline/capture-handoffs", json={}, headers=headers).status_code == 503

    assert client.get("/api/live-pipeline/intake-audit", headers=headers).status_code == 404
    _write_json(manifest_path.parent / "live_pipeline_input_intake_audit.json", ["bad"])
    assert client.get("/api/live-pipeline/intake-audit", headers=headers).status_code == 500


def test_capture_handoff_blocked_after_conversion_and_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "token")
    monkeypatch.setattr(
        service,
        "build_live_pipeline_input_intake",
        lambda **_kwargs: {"status": "blocked", "input_blockers": ["blocked_after_conversion"]},
    )

    response = TestClient(create_app()).post(
        "/api/live-pipeline/capture-handoffs",
        json=_capture_handoff(),
        headers={"x-blueprint-intake-token": "token"},
    )

    assert response.status_code == 202
    assert response.json()["capture_handoff"]["converted_to_job_request"] is True

    calls: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda app, host, port: calls.update({"app": app, "host": host, "port": port})),
    )
    assert service.main(["--host", "0.0.0.0", "--port", "9999"]) == 0
    assert calls == {
        "app": "blueprint_pipeline.live_pipeline_intake_service:app",
        "host": "0.0.0.0",
        "port": 9999,
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


def test_live_pipeline_intake_service_converts_capture_handoff_to_webapp_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    monkeypatch.setenv("BLUEPRINT_LIVE_PIPELINE_INTAKE_OVERWRITE", "true")
    client = TestClient(create_app())

    response = client.post(
        "/api/live-pipeline/capture-handoffs",
        json=_capture_handoff(),
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "staged_for_control_plane"
    assert payload["accepted"] is True
    assert payload["capture_handoff"]["converted_to_job_request"] is True
    assert payload["capture_handoff"]["dataset_selection"]["task_id"] == "scene_anchor_geometry_0"
    target_path = Path(payload["webapp_staging"]["target_path"])
    assert target_path.is_file()
    envelope = json.loads(target_path.read_text(encoding="utf-8"))
    job_request = envelope["job_request"]
    assert envelope["source_kind"] == "capture_pipeline_handoff"
    assert job_request["source_kind"] == "capture_pipeline_handoff"
    assert job_request["requested_tasks"] == [
        {
            "task_id": "scene_anchor_geometry_0",
            "scenario_ids": ["scenario_scene_anchor_geometry_0_unitree_g1"],
        }
    ]
    assert job_request["source"]["pipeline_handoff_uri"].endswith("pipeline_handoff.json")
    assert job_request["policy_package"]["high_level_skill_trace"]["ordered_skill_sequence"] == [
        "walk_to_target"
    ]


def test_live_pipeline_intake_service_blocks_capture_handoff_without_robot_eval_request(
    tmp_path: Path, monkeypatch
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_robot_eval_dataset_cards(capture_root)
    manifest_path = _control_manifest(tmp_path, capture_root)
    monkeypatch.setenv(CONTROL_PLANE_OUTPUT_PATH_ENV, str(manifest_path))
    monkeypatch.setenv(INTAKE_TOKEN_ENV, "test-intake-token")
    client = TestClient(create_app())
    handoff = {
        **_capture_handoff(),
        "requested_outputs": ["qualification"],
        "requested_lanes": ["qualification"],
        "robot_eval_dataset_requested": False,
    }

    response = client.post(
        "/api/live-pipeline/capture-handoffs",
        json=handoff,
        headers={"authorization": "Bearer test-intake-token"},
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["status"] == "blocked"
    assert payload["accepted"] is False
    assert "capture_handoff:capture_handoff_robot_eval_not_requested" in payload[
        "input_blockers"
    ]


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
