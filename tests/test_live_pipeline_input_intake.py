from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.live_pipeline_control_plane import run_live_pipeline_control_plane
from blueprint_pipeline.live_pipeline_input_intake import (
    LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION,
    build_live_pipeline_input_intake,
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
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "site_package": {
                "capture_root": str(capture_root),
                "site_id": "site-1",
                "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
            },
            "source": {
                "system": "Blueprint-WebApp",
                "site_submission_id": "site-submission-1",
                "request_id": "request-1",
                "buyer_request_id": "buyer-request-1",
                "capture_job_id": "capture-job-1",
            },
        },
    }


def _webapp_site_library_request(
    capture_root: Path, *, job_id: str = "webapp-job-1"
) -> dict[str, object]:
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


def _arena_results(results_dir: Path) -> Path:
    _write_json(
        results_dir / "rollout_manifest.json",
        {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "scenario_id": "scenario-1",
                    "status": "success",
                    "success": True,
                }
            ]
        },
    )
    return results_dir


def _live_closure_evidence(path: Path, *, job_id: str = "webapp-job-1") -> Path:
    _write_json(
        path,
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "job_id": job_id,
            "review_acceptance": {
                "accepted": True,
                "reviewer": "owner-reviewer",
            },
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
        },
    )
    return path


def _deployment_outcomes(
    path: Path,
    *,
    job_id: str = "webapp-job-1",
    include_evidence: bool = True,
    include_prediction_match_key: bool = True,
) -> Path:
    record: dict[str, object] = {
        "outcome_id": "pilot-outcome-1",
        "task_id": "place_return_in_bin",
        "scenario_id": "scenario_place_return_in_bin_mobile",
        "policy_id": "policy-live",
        "actual_success": False,
        "failure_mode_ids": ["missed_blocked_path"],
        "cycle_time_seconds": 31.0,
        "intervention_count": 1,
        "tuning_hours": 0.75,
        "tuning_iterations": 1,
        "site_modifications": ["moved_cart_from_approach_lane"],
        "site_modifications_helped": True,
    }
    if include_prediction_match_key:
        record["scenario_eval_run_id"] = "scenario-run-1"
    if include_evidence:
        record["evidence_refs"] = {"pilot_log": "owner://pilot/pilot-outcome-1"}
    _write_json(
        path,
        {
            "schema_version": "deployment_outcome_manifest.v1",
            "job_id": job_id,
            "records": [record],
        },
    )
    return path


def _policy_package(path: Path, *, job_id: str = "webapp-job-1") -> Path:
    _write_json(
        path,
        {
            "schema_version": "robot_team_policy_package.v1",
            "job_id": job_id,
            "policy_package": {
                "policy_api_endpoint": {
                    "endpoint_url": "https://robot-team.example/policy",
                    "observation_schema_ref": "schemas/obs-v1.json",
                    "action_schema_ref": "schemas/action-v1.json",
                }
            },
        },
    )
    return path


def test_live_pipeline_input_intake_validates_inputs_without_staging(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    results_dir = _arena_results(tmp_path / "arena-results")
    _write_json(request_path, _webapp_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        arena_results_dir=results_dir,
    )

    assert result["schema_version"] == LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION
    assert result["status"] == "ready_for_control_plane"
    assert result["webapp_job_request"]["status"] == "ready"
    assert result["arena_results"]["status"] == "ready_for_ingest"
    assert result["webapp_staging"]["status"] == "not_requested"
    assert result["proof_boundary"]["simulator_execution_proven"] is False
    assert Path(str(result["output_path"])).is_file()


def test_live_pipeline_input_intake_stages_valid_webapp_request(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    target_path = Path(str(result["webapp_staging"]["target_path"]))
    assert result["status"] == "staged_for_control_plane"
    assert result["webapp_staging"]["performed"] is True
    assert result["staged_inputs"]["status"] == "staged"
    assert Path(str(result["staged_inputs"]["path"])).is_file()
    assert target_path.is_file()
    assert json.loads(target_path.read_text(encoding="utf-8"))["job_id"] == "webapp-job-1"


def test_live_pipeline_input_intake_accepts_webapp_site_library_id_locations(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_site_library_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    assert result["status"] == "staged_for_control_plane"
    assert result["webapp_job_request"]["missing_fields"] == []
    assert result["webapp_job_request"]["fields_present"] == {
        "site_submission_id": True,
        "request_id": True,
        "buyer_request_id": True,
        "capture_job_id": True,
    }


def test_live_pipeline_input_intake_staged_arena_results_feed_control_plane(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    results_dir = _arena_results(tmp_path / "arena-results")

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        arena_results_dir=results_dir,
        stage_arena_results=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}

    assert intake["status"] == "staged_for_control_plane"
    assert intake["staged_inputs"]["arena_results_staged"] is True
    assert rerun["staged_inputs"]["arena_results_ready"] is True
    assert rerun["setup_status"] == "local_ready_live_external_blocked"
    assert required_input_ids == {
        "webapp_upstream_truth",
        "live_robot_eval_closure_evidence",
        "real_world_deployment_outcomes",
        "robot_team_policy_package",
    }
    assert "Isaac Lab-Arena" not in " ".join(rerun["next_inputs_needed"])


def test_live_pipeline_input_intake_stages_live_closure_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    evidence_path = _live_closure_evidence(tmp_path / "incoming" / "closure.json")
    _write_json(request_path, _webapp_request(capture_root))

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        live_closure_evidence=evidence_path,
        stage_live_closure_evidence=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    staged_target = Path(str(intake["live_closure_evidence_staging"]["target_path"]))

    assert intake["status"] == "staged_for_control_plane"
    assert intake["live_closure_evidence"]["status"] == "ready_for_closure_audit"
    assert intake["staged_inputs"]["live_closure_evidence_staged"] is True
    assert staged_target == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "live_eval_closure_evidence.json"
    )
    assert staged_target.is_file()
    assert rerun["staged_inputs"]["live_closure_evidence_ready"] is True
    assert "live_robot_eval_closure_evidence" not in required_input_ids


def test_live_pipeline_input_intake_accepts_camel_case_live_closure_evidence(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    evidence_path = tmp_path / "incoming" / "closure-camel.json"
    _write_json(request_path, _webapp_request(capture_root))
    _write_json(
        evidence_path,
        {
            "schema_version": "live_robot_eval_closure_evidence.v1",
            "jobId": "webapp-job-1",
            "reviewAcceptance": {
                "accepted": True,
                "ownerAttestation": {
                    "attestedBy": "owner-reviewer",
                    "acceptedClaimBoundary": "Owner accepted review evidence.",
                },
            },
            "delivery": {
                "storageUploadPerformed": True,
                "signedUrls": ["https://delivery.example/signed/package-1"],
                "entitlementVerified": True,
            },
            "safetyContactPhysics": {
                "physicsContactValidated": True,
                "safetyValidated": True,
                "robotReadinessProven": True,
                "methodologyUriOrPath": "owner://methodology",
                "contactValidationUriOrPath": "owner://contact",
                "safetyValidationUriOrPath": "owner://safety",
                "ownerAttestation": {
                    "attestedBy": "safety-owner",
                    "acceptedClaimBoundary": (
                        "Owner accepted contact, physics, and safety evidence."
                    ),
                },
            },
        },
    )

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        live_closure_evidence=evidence_path,
        stage_live_closure_evidence=True,
    )

    assert intake["status"] == "staged_for_control_plane"
    assert intake["live_closure_evidence"]["status"] == "ready_for_closure_audit"
    assert intake["live_closure_evidence"]["blockers"] == []
    assert intake["live_closure_evidence_staging"]["performed"] is True


def test_live_pipeline_input_intake_stages_deployment_outcomes(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    outcomes_path = _deployment_outcomes(tmp_path / "incoming" / "deployment.json")
    _write_json(request_path, _webapp_request(capture_root))

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    staged_target = Path(str(intake["deployment_outcomes_staging"]["target_path"]))
    staged_payload = json.loads(
        Path(str(intake["staged_inputs"]["path"])).read_text(encoding="utf-8")
    )

    assert intake["status"] == "staged_for_control_plane"
    assert intake["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert intake["deployment_outcomes"]["record_count"] == 1
    assert intake["deployment_outcomes"]["records_ready_for_calibration"] is True
    assert intake["deployment_outcomes"]["prediction_match_key_record_count"] == 1
    assert intake["deployment_outcomes"]["missing_prediction_match_key_record_ids"] == []
    assert intake["deployment_outcomes"]["owner_evidence_ready"] is True
    assert intake["deployment_outcomes"]["owner_evidence_record_count"] == 1
    assert intake["deployment_outcomes"]["missing_owner_evidence_record_ids"] == []
    assert intake["staged_inputs"]["deployment_outcomes_staged"] is True
    assert staged_payload["deployment_outcomes"]["records_ready_for_calibration"] is True
    assert staged_payload["deployment_outcomes"]["prediction_match_keys_ready"] is True
    assert staged_payload["deployment_outcomes"]["owner_evidence_ready"] is True
    assert staged_target == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "deployment_outcomes"
        / "inbox"
        / "pilot-outcome-1.json"
    )
    assert staged_target.is_file()
    assert rerun["staged_inputs"]["deployment_outcomes_ready"] is True
    assert rerun["staged_inputs"]["deployment_outcomes_records_ready_for_calibration"] is True
    assert rerun["staged_inputs"]["deployment_outcomes_owner_evidence_ready"] is True
    assert "real_world_deployment_outcomes" not in required_input_ids
    assert "predicted_vs_actual_exact_match_keys" not in required_input_ids
    assert "real_world_deployment_outcome_owner_evidence" not in required_input_ids


def test_live_pipeline_input_intake_keeps_owner_evidence_blocker_for_outcome_records(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    outcomes_path = _deployment_outcomes(
        tmp_path / "incoming" / "deployment.json",
        include_evidence=False,
    )
    _write_json(request_path, _webapp_request(capture_root))

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}

    assert intake["status"] == "staged_for_control_plane"
    assert intake["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert intake["deployment_outcomes"]["ready"] is True
    assert intake["deployment_outcomes"]["records_ready_for_calibration"] is True
    assert intake["deployment_outcomes"]["prediction_match_keys_ready"] is True
    assert intake["deployment_outcomes"]["owner_evidence_ready"] is False
    assert intake["deployment_outcomes"]["owner_evidence_record_count"] == 0
    assert intake["deployment_outcomes"]["missing_owner_evidence_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert "real_world_deployment_outcomes" not in required_input_ids
    assert "predicted_vs_actual_exact_match_keys" not in required_input_ids
    assert "real_world_deployment_outcome_owner_evidence" in required_input_ids
    assert rerun["staged_inputs"]["deployment_outcomes_ready"] is True
    assert rerun["staged_inputs"]["deployment_outcomes_owner_evidence_ready"] is False
    assert rerun["staged_inputs"]["deployment_outcome_owner_evidence_record_count"] == 0
    assert rerun["staged_inputs"]["deployment_outcome_missing_owner_evidence_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert "owner evidence" in " ".join(rerun["next_inputs_needed"])


def test_live_pipeline_input_intake_keeps_calibration_key_blocker_for_weak_outcome_matches(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    outcomes_path = _deployment_outcomes(
        tmp_path / "incoming" / "deployment.json",
        include_prediction_match_key=False,
    )
    _write_json(request_path, _webapp_request(capture_root))

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    calibration_input = next(
        item
        for item in packet["required_inputs"]
        if item["id"] == "predicted_vs_actual_exact_match_keys"
    )

    assert intake["status"] == "staged_for_control_plane"
    assert intake["deployment_outcomes"]["status"] == "ready_for_real_world_validation"
    assert intake["deployment_outcomes"]["ready"] is True
    assert intake["deployment_outcomes"]["records_ready_for_calibration"] is False
    assert intake["deployment_outcomes"]["prediction_match_keys_ready"] is False
    assert intake["deployment_outcomes"]["prediction_match_key_record_count"] == 0
    assert intake["deployment_outcomes"]["missing_prediction_match_key_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert rerun["staged_inputs"]["deployment_outcomes_ready"] is True
    assert rerun["staged_inputs"]["deployment_outcomes_records_ready_for_calibration"] is False
    assert rerun["staged_inputs"][
        "deployment_outcome_missing_prediction_match_key_record_ids"
    ] == ["pilot-outcome-1"]
    assert "real_world_deployment_outcomes" not in required_input_ids
    assert "predicted_vs_actual_exact_match_keys" in required_input_ids
    assert "real_world_deployment_outcome_owner_evidence" not in required_input_ids
    assert calibration_input["required_record_fields"] == [
        "scenario_eval_run_id or scenario_variation_instance_id"
    ]


def test_live_pipeline_input_intake_stages_policy_package(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    package_path = _policy_package(tmp_path / "incoming" / "policy-package.json")
    _write_json(request_path, _webapp_request(capture_root))

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        policy_package=package_path,
        stage_policy_package=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}
    staged_target = Path(str(intake["policy_package_staging"]["target_path"]))

    assert intake["status"] == "staged_for_control_plane"
    assert intake["policy_package"]["status"] == "ready_for_robot_eval_job"
    assert intake["policy_package"]["selected_modalities"] == ["policy_api_endpoint"]
    assert intake["staged_inputs"]["policy_package_staged"] is True
    assert staged_target == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "policy_package.json"
    )
    assert staged_target.is_file()
    assert rerun["staged_inputs"]["policy_package_ready"] is True
    assert "robot_team_policy_package" not in required_input_ids


def test_live_pipeline_input_intake_rejects_invalid_policy_package(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    package_path = tmp_path / "incoming" / "policy-package.json"
    _write_json(
        package_path,
        {
            "schema_version": "robot_team_policy_package.v1",
            "job_id": "../escape",
            "policy_package": {"docker_container": {"image_ref": "registry.example/policy:latest"}},
        },
    )

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        policy_package=package_path,
        stage_policy_package=True,
    )

    assert result["status"] == "blocked"
    assert "policy_package:policy_package_job_id_unsafe" in result["input_blockers"]
    assert "policy_package:policy_package.docker_container.digest" in result[
        "input_blockers"
    ]
    assert result["policy_package_staging"]["performed"] is False


def test_live_pipeline_input_intake_rejects_invalid_deployment_outcomes(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    outcomes_path = tmp_path / "incoming" / "deployment.json"
    _write_json(
        outcomes_path,
        {
            "schema_version": "deployment_outcome_manifest.v1",
            "job_id": "../escape",
            "records": [{"task_id": "task-only"}],
        },
    )

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        deployment_outcomes=outcomes_path,
        stage_deployment_outcomes=True,
    )

    assert result["status"] == "blocked"
    assert "deployment_outcomes:deployment_outcomes_job_id_unsafe" in result[
        "input_blockers"
    ]
    assert "deployment_outcomes:deployment_outcomes_missing_task_or_scenario" in result[
        "input_blockers"
    ]
    assert "deployment_outcomes:deployment_outcomes_missing_actual_result_signal" in result[
        "input_blockers"
    ]
    assert result["deployment_outcomes_staging"]["performed"] is False


def test_live_pipeline_input_intake_rejects_mismatched_live_closure_evidence_job(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    evidence_path = _live_closure_evidence(
        tmp_path / "incoming" / "closure.json",
        job_id="other-job",
    )
    _write_json(request_path, _webapp_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        live_closure_evidence=evidence_path,
        stage_live_closure_evidence=True,
    )

    assert result["status"] == "blocked"
    assert "live_closure_evidence:live_closure_evidence_job_id_mismatch" in result[
        "input_blockers"
    ]
    assert result["live_closure_evidence_staging"]["performed"] is False


def test_live_pipeline_input_intake_rejects_unsafe_live_closure_evidence_job_id(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    evidence_path = _live_closure_evidence(
        tmp_path / "incoming" / "closure.json",
        job_id="../escape",
    )

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        live_closure_evidence=evidence_path,
        stage_live_closure_evidence=True,
    )

    assert result["status"] == "blocked"
    assert "live_closure_evidence:live_closure_evidence_job_id_unsafe" in result[
        "input_blockers"
    ]
    assert result["live_closure_evidence_staging"]["performed"] is False


def test_live_pipeline_input_intake_rejects_mismatched_capture_root(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    other_capture_root = tmp_path / "other" / "captures" / "capture-2"
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_request(other_capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    assert result["status"] == "blocked"
    assert "webapp:request_capture_root_does_not_match_control_plane" in result[
        "input_blockers"
    ]
    assert result["webapp_staging"]["performed"] is False


def test_live_pipeline_input_intake_module_cli(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_request(capture_root))
    env = os.environ.copy()
    src_root = Path.cwd() / "src"
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_input_intake",
            "--manifest-path",
            str(manifest_path),
            "--webapp-job-request",
            str(request_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=ready_for_control_plane" in completed.stdout
    audit = json.loads(
        (tmp_path / "control" / "live_pipeline_input_intake_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "ready_for_control_plane"
