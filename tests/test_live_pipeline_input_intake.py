from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import blueprint_pipeline.live_pipeline_input_intake as intake
from blueprint_pipeline.live_pipeline_control_plane import run_live_pipeline_control_plane
from blueprint_pipeline.live_pipeline_input_intake import (
    LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION,
    build_live_pipeline_input_intake,
)


pytestmark = [pytest.mark.slow, pytest.mark.integration]


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


def _decision_evidence_envelope(capture_root: Path) -> dict[str, object]:
    return {
        "queue_contract": "blueprint.decision_evidence_request_inbox.v1",
        "request_id": "decision-request-1",
        "decision_id": "buyer-decision-1",
        "decision_request": {
            "schema_version": "blueprint.decision_evidence_request.v1",
            "request_id": "decision-request-1",
            "decision_id": "buyer-decision-1",
            "testbed": {
                "testbed_id": "site-1",
                "version": "v1",
                "digest_sha256": "sha256:" + "a" * 64,
                "manifest_uri": str(capture_root / "pipeline" / "robot_eval_dataset" / "robot_eval_dataset_manifest.json"),
            },
            "decision_question": "Can policy-a complete task-a?",
            "site_task": {
                "site_id": "site-1",
                "site_name": "site-one",
                "task_id": "task-a",
                "task_description": "Complete task A",
                "conditions": ["fixture"],
            },
            "candidates": [
                {
                    "candidate_id": "policy-a",
                    "kind": "policy",
                    "label": "Policy A",
                    "reference": {"external_id": "policy-a"},
                }
            ],
            "routing_authority": {
                "system": "BlueprintCapturePipeline",
                "method_selection": "pipeline_qualified_least_cost_sufficient_evidence",
                "webapp_backend_selection_allowed": False,
            },
            "authorization": {
                "entitlement_id": "entitlement-1",
                "access_state": "provisioned",
                "verified_by": "server_marketplace_entitlement",
            },
        },
    }


def test_decision_evidence_envelope_stages_bounded_legacy_execution_adapter(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "decision-envelope.json"
    _write_json(request_path, _decision_evidence_envelope(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
        overwrite=True,
        staged_inputs_path=tmp_path / "staged-inputs.json",
    )

    assert result["status"] == "staged_for_control_plane"
    staged_path = Path(result["webapp_staging"]["target_path"])
    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    assert staged["queue_contract"] == "robot_eval_job_request_inbox.v1"
    request = staged["job_request"]
    assert request["schema_version"] == "robot_eval_job_request.v1"
    assert request["job_id"] == "decision-request-1"
    assert request["requested_tasks"] == [{"task_id": "task-a", "scenario_ids": []}]
    assert request["site_package"]["capture_root"] == str(capture_root)
    assert request["source"]["source_kind"] == (
        "decision_evidence_request_legacy_execution_adapter"
    )
    assert request["proof_boundary"]["translation_grants_method_qualification"] is False


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


def _local_webapp_rehearsal_request(capture_root: Path) -> dict[str, object]:
    envelope = _webapp_site_library_request(capture_root)
    envelope["source_kind"] = "local_first_gpu_rehearsal_request"
    envelope["local_rehearsal_only"] = True
    job_request = envelope["job_request"]
    assert isinstance(job_request, dict)
    job_request["source_kind"] = "local_first_gpu_rehearsal_request"
    return envelope


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
                "buyer_access_check": {
                    "buyer_access_checked": True,
                    "buyer_accessible": True,
                    "status": "ok",
                },
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
        },
    )
    return path


def _deployment_outcomes(
    path: Path,
    *,
    job_id: str = "webapp-job-1",
    include_evidence: bool = True,
    include_prediction_match_key: bool = True,
    include_variation_match_key: bool = True,
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
    if include_variation_match_key:
        record["scenario_variation_instance_id"] = "scenario-variation-1"
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


def _real_robot_pov_manifest(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema_version": "real_robot_pov_manifest.v1",
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


def test_live_pipeline_input_intake_preserves_local_rehearsal_boundary(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-rehearsal.json"
    _write_json(request_path, _local_webapp_rehearsal_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    staged_inputs_path = Path(str(result["staged_inputs"]["path"]))
    staged_inputs = json.loads(staged_inputs_path.read_text(encoding="utf-8"))
    assert result["status"] == "staged_for_control_plane"
    assert result["webapp_job_request"]["status"] == "ready"
    assert result["webapp_job_request"]["local_rehearsal_only"] is True
    assert result["webapp_request_metadata_valid"] is True
    assert result["webapp_truth_proven"] is False
    assert result["local_webapp_rehearsal_only"] is True
    assert result["proof_boundary"]["webapp_request_metadata_valid"] is True
    assert result["proof_boundary"]["webapp_truth_proven"] is False
    assert result["proof_boundary"]["local_webapp_rehearsal_only"] is True
    assert result["proof_boundary"]["live_webapp_forwarding_proven"] is False
    assert staged_inputs["local_rehearsal_only"] is True
    assert staged_inputs["source_kind"] == "local_first_gpu_rehearsal_request"
    assert staged_inputs["webapp_request"]["source_kind"] == "local_first_gpu_rehearsal_request"
    assert staged_inputs["proof_boundary"]["local_webapp_rehearsal_only"] is True
    assert staged_inputs["proof_boundary"]["live_webapp_forwarding_proven"] is False


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The local deterministic lane requires host ffmpeg for clip keyframe paths; this test
    # asserts control-plane gating semantics, not host tool installs (CI has no ffmpeg).
    import shutil as _shutil

    real_which = _shutil.which
    monkeypatch.setattr(
        _shutil,
        "which",
        lambda cmd, *a, **kw: "/usr/bin/ffmpeg" if cmd == "ffmpeg" else real_which(cmd, *a, **kw),
    )
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
        "robot_team_policy_package",
    }
    assert "Isaac Lab-Arena" not in " ".join(rerun["next_inputs_needed"])


def test_live_pipeline_input_intake_stages_real_robot_pov_manifest(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    pov_path = _real_robot_pov_manifest(tmp_path / "incoming" / "real-pov.json")

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        real_robot_pov=pov_path,
        stage_real_robot_pov=True,
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
    staged_target = Path(str(intake["real_robot_pov_staging"]["target_path"]))
    staged_payload = json.loads(
        Path(str(intake["staged_inputs"]["path"])).read_text(encoding="utf-8")
    )

    assert intake["status"] == "staged_for_control_plane"
    assert intake["real_robot_pov"]["status"] == "ready_for_robot_eval_job"
    assert intake["real_robot_pov"]["record_count"] == 1
    assert intake["real_robot_pov"]["missing_exact_key_record_ids"] == []
    assert intake["real_robot_pov"]["missing_evidence_record_ids"] == []
    assert intake["staged_inputs"]["real_robot_pov_staged"] is True
    assert staged_payload["real_robot_pov"]["ready"] is True
    assert staged_payload["real_robot_pov"]["record_count"] == 1
    assert staged_target == (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "real_robot_pov_manifest.json"
    )
    assert staged_target.is_file()
    assert rerun["staged_inputs"]["real_robot_pov_ready"] is True
    assert "real_robot_pov_evidence" not in required_input_ids


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
                "buyerAccessCheck": {
                    "buyerAccessChecked": True,
                    "buyerAccessible": True,
                    "status": "ok",
                },
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
    assert "real_world_deployment_outcome_owner_evidence" not in required_input_ids
    assert rerun["staged_inputs"]["deployment_outcomes_ready"] is True
    assert rerun["staged_inputs"]["deployment_outcomes_owner_evidence_ready"] is False
    assert rerun["staged_inputs"]["deployment_outcome_owner_evidence_record_count"] == 0
    assert rerun["staged_inputs"]["deployment_outcome_missing_owner_evidence_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert "owner evidence" not in " ".join(rerun["next_inputs_needed"])


def test_live_pipeline_input_intake_keeps_calibration_key_blocker_for_run_only_outcomes(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    outcomes_path = _deployment_outcomes(
        tmp_path / "incoming" / "deployment.json",
        include_variation_match_key=False,
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

    assert intake["deployment_outcomes"]["records_ready_for_calibration"] is False
    assert intake["deployment_outcomes"]["prediction_match_keys_ready"] is False
    assert intake["deployment_outcomes"]["missing_prediction_match_key_record_ids"] == [
        "pilot-outcome-1"
    ]
    assert rerun["staged_inputs"][
        "deployment_outcome_missing_prediction_match_key_record_ids"
    ] == ["pilot-outcome-1"]
    assert "predicted_vs_actual_exact_match_keys" not in required_input_ids


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
    assert "predicted_vs_actual_exact_match_keys" not in required_input_ids
    assert "real_world_deployment_outcome_owner_evidence" not in required_input_ids


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


def test_live_pipeline_input_intake_rejects_invalid_real_robot_pov_manifest(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    pov_path = tmp_path / "incoming" / "real-pov.json"
    _write_json(
        pov_path,
        {
            "schema_version": "real_robot_pov_manifest.v1",
            "records": [
                {
                    "evidence_id": "real-pov-1",
                    "scenario_eval_run_id": "scenario-run-1",
                    "robot_camera_video_uri": "owner://pov/scenario-run-1.mp4",
                }
            ],
        },
    )

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        real_robot_pov=pov_path,
        stage_real_robot_pov=True,
    )

    assert result["status"] == "blocked"
    assert "real_robot_pov:real_robot_pov_missing_exact_keys" in result[
        "input_blockers"
    ]
    assert "real_robot_pov:real_robot_pov_missing_action_logs" in result[
        "input_blockers"
    ]
    assert result["real_robot_pov_staging"]["performed"] is False


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
    assert "webapp_request_metadata_valid=true" in completed.stdout
    assert "local_webapp_rehearsal_only=false" in completed.stdout
    assert "webapp_truth_proven=true" in completed.stdout
    audit = json.loads(
        (tmp_path / "control" / "live_pipeline_input_intake_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "ready_for_control_plane"
    assert audit["webapp_request_metadata_valid"] is True
    assert audit["local_webapp_rehearsal_only"] is False
    assert audit["webapp_truth_proven"] is True


def test_live_pipeline_input_intake_helper_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert intake._boolish(True) is True
    assert intake._attestation_ok("owner accepted") is True
    assert intake._attestation_ok(None) is False
    assert (
        intake._attestation_ok(
            {"operatorId": "ops-1", "acceptedClaimBoundary": "owner accepted"}
        )
        is True
    )
    assert intake._delivery_access_ready({}) is True
    assert intake._delivery_access_ready({"status": "ready"}) is True
    assert intake._delivery_access_ready({"artifactRefs": ["owner://artifact"]}) is True
    assert intake._request_from_payload({"queue_contract": "robot_eval_job_request_inbox.v1"}) is None
    direct_request = {
        "schema_version": "robot_eval_job_request.v1",
        "job_id": "direct-job",
    }
    assert intake._request_from_payload(direct_request) == direct_request
    assert intake._request_from_payload({"schema_version": "other"}) is None
    assert (
        intake._source_kind_from_request(
            {"source": {"selection_state": {"source_kind": "site_library"}}}
        )
        == "site_library"
    )
    assert intake._field_value({"source": {}}, "buyer_request_id") is None
    assert intake._path_matches(None, tmp_path) is False
    assert intake._policy_package_from_payload(
        {"policyApiEndpoint": {"url": "https://robot-team.example/policy"}}
    ) == {"policy_api_endpoint": {"url": "https://robot-team.example/policy"}}
    assert intake._record_has_owner_evidence({"evidence_refs": ["owner://record"]}) is True

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        intake._read_mapping(array_path)

    wrong_manifest = tmp_path / "manifest.json"
    _write_json(wrong_manifest, {"schema_version": "other"})
    with pytest.raises(ValueError, match="blueprint_live_pipeline_control_plane_run.v1"):
        intake._load_control_plane_manifest(wrong_manifest)

    class BrokenPath:
        def __init__(self, value: str) -> None:
            self.value = value

        def resolve(self) -> Path:
            raise OSError("cannot resolve")

    monkeypatch.setattr(intake, "Path", BrokenPath)
    assert intake._path_matches("unresolvable", tmp_path) is False


def test_live_pipeline_input_intake_audit_missing_and_malformed_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    missing_request = tmp_path / "missing" / "request.json"
    malformed_request = tmp_path / "incoming" / "malformed-request.json"
    malformed_request.parent.mkdir(parents=True, exist_ok=True)
    malformed_request.write_text("[]", encoding="utf-8")
    non_request = tmp_path / "incoming" / "not-request.json"
    _write_json(non_request, {"schema_version": "not-a-request"})
    incomplete_request = tmp_path / "incoming" / "incomplete-request.json"
    _write_json(
        incomplete_request,
        {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": "incomplete-job",
            "site_package": {"capture_root": str(capture_root)},
        },
    )
    empty_arena = tmp_path / "empty-arena"
    empty_arena.mkdir()

    assert intake._audit_webapp_request(
        request_path=None,
        expected_capture_root=capture_root,
        configured_inbox=None,
    )["blockers"] == ["webapp_job_request_not_provided"]
    assert intake._audit_webapp_request(
        request_path=missing_request,
        expected_capture_root=capture_root,
        configured_inbox=None,
    )["blockers"] == ["webapp_job_request_missing"]
    assert intake._audit_webapp_request(
        request_path=malformed_request,
        expected_capture_root=capture_root,
        configured_inbox=None,
    )["blockers"] == ["webapp_job_request_read_failed:ValueError"]
    assert intake._audit_webapp_request(
        request_path=non_request,
        expected_capture_root=capture_root,
        configured_inbox=None,
    )["blockers"] == ["not_robot_eval_job_request_v1_or_queue_envelope"]
    missing_fields_audit = intake._audit_webapp_request(
        request_path=incomplete_request,
        expected_capture_root=capture_root,
        configured_inbox=None,
    )
    assert missing_fields_audit["status"] == "blocked"
    assert "missing_required_webapp_ids" in missing_fields_audit["blockers"]

    assert intake._audit_arena_results(tmp_path / "missing-arena")["blockers"] == [
        "arena_results_dir_missing"
    ]
    assert intake._audit_arena_results(empty_arena)["blockers"] == [
        "arena_results_dir_has_no_json_artifacts"
    ]


def test_live_pipeline_input_intake_optional_artifact_audit_blockers(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")

    deployment_missing = intake._audit_deployment_outcomes(
        path=tmp_path / "missing-deployment.json",
        expected_job_id="webapp-job-1",
    )
    assert deployment_missing["blockers"] == ["deployment_outcomes_missing"]
    deployment_read_failed = intake._audit_deployment_outcomes(
        path=malformed,
        expected_job_id="webapp-job-1",
    )
    assert deployment_read_failed["blockers"] == [
        "deployment_outcomes_read_failed:ValueError"
    ]
    bad_deployment = tmp_path / "bad-deployment.json"
    _write_json(
        bad_deployment,
        {
            "schema_version": "deployment_outcome_manifest.v0",
            "job_id": "other-job",
            "records": {"not": "a list"},
        },
    )
    deployment_audit = intake._audit_deployment_outcomes(
        path=bad_deployment,
        expected_job_id="webapp-job-1",
    )
    assert "deployment_outcomes_schema_mismatch" in deployment_audit["blockers"]
    assert "deployment_outcomes_job_id_mismatch" in deployment_audit["blockers"]
    assert "deployment_outcomes_no_records" in deployment_audit["blockers"]
    missing_job_deployment = tmp_path / "missing-job-deployment.json"
    _write_json(
        missing_job_deployment,
        {
            "schema_version": "deployment_outcome_manifest.v1",
            "records": [
                {
                    "task_id": "task-1",
                    "scenario_id": "scenario-1",
                    "actual_success": True,
                    "scenario_eval_run_id": "run-1",
                    "scenario_variation_instance_id": "variation-1",
                    "evidence_refs": {"log": "owner://log"},
                }
            ],
        },
    )
    assert intake._audit_deployment_outcomes(
        path=missing_job_deployment,
        expected_job_id=None,
    )["blockers"] == ["deployment_outcomes_job_id_missing"]

    assert intake._audit_real_robot_pov(
        path=tmp_path / "missing-pov.json",
        expected_job_id="webapp-job-1",
    )["blockers"] == ["real_robot_pov_missing"]
    assert intake._audit_real_robot_pov(
        path=malformed,
        expected_job_id="webapp-job-1",
    )["blockers"] == ["real_robot_pov_read_failed:ValueError"]
    bad_pov = tmp_path / "bad-pov.json"
    _write_json(
        bad_pov,
        {
            "schema_version": "real_robot_pov_manifest.v0",
            "job_id": "../escape",
            "records": [],
        },
    )
    pov_audit = intake._audit_real_robot_pov(path=bad_pov, expected_job_id=None)
    assert "real_robot_pov_schema_mismatch" in pov_audit["blockers"]
    assert "real_robot_pov_job_id_unsafe" in pov_audit["blockers"]
    assert "real_robot_pov_no_records" in pov_audit["blockers"]
    camera_missing = tmp_path / "camera-missing-pov.json"
    _write_json(
        camera_missing,
        {
            "schema_version": "real_robot_pov_manifest.v1",
            "job_id": "other-job",
            "records": [
                {
                    "task_id": "task-1",
                    "scenario_id": "scenario-1",
                    "scenario_eval_run_id": "run-1",
                    "scenario_variation_instance_id": "variation-1",
                    "action_log_uri": "owner://actions.jsonl",
                    "timestamp_alignment": "aligned",
                    "owner_evidence_refs": {"action": "owner://actions.jsonl"},
                }
            ],
        },
    )
    camera_missing_audit = intake._audit_real_robot_pov(
        path=camera_missing,
        expected_job_id="webapp-job-1",
    )
    assert "real_robot_pov_job_id_mismatch" in camera_missing_audit["blockers"]
    assert "real_robot_pov_missing_camera_video" in camera_missing_audit["blockers"]


def test_live_pipeline_input_intake_policy_and_closure_blocker_edges(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed-policy.json"
    malformed.write_text("[]", encoding="utf-8")

    assert intake._audit_policy_modality(
        modality="policy_api_endpoint",
        payload={},
    ) == []
    assert intake._audit_policy_modality(
        modality="policy_api_endpoint",
        payload={"endpoint_url": "ftp://policy"},
    ) == ["policy_package.policy_api_endpoint.endpoint_url"]
    assert intake._audit_policy_modality(
        modality="docker_container",
        payload={"name": "missing-inputs"},
    ) == [
        "policy_package.docker_container.image_ref",
        "policy_package.docker_container.digest",
    ]
    assert intake._audit_policy_modality(
        modality="recorded_action_trace",
        payload={"name": "missing-inputs"},
    ) == [
        "policy_package.recorded_action_trace.trace_manifest_uri",
        "policy_package.recorded_action_trace.timestamp_alignment",
    ]
    assert intake._audit_policy_modality(
        modality="high_level_skill_trace",
        payload={"name": "missing-inputs"},
    ) == ["policy_package.high_level_skill_trace.ordered_skill_sequence"]
    assert intake._audit_policy_modality(
        modality="teleop_demo",
        payload={"name": "missing-inputs"},
    ) == [
        "policy_package.teleop_demo.demo_artifact_uri",
        "policy_package.teleop_demo.rights_privacy_attestation",
    ]
    assert intake._audit_policy_modality(
        modality="sim_controller_plugin",
        payload={"name": "missing-inputs"},
    ) == [
        "policy_package.sim_controller_plugin.simulator_framework",
        "policy_package.sim_controller_plugin.plugin_uri",
    ]

    assert intake._audit_policy_package(
        path=tmp_path / "missing-policy.json",
        expected_job_id="webapp-job-1",
    )["blockers"] == ["policy_package_missing"]
    assert intake._audit_policy_package(
        path=malformed,
        expected_job_id="webapp-job-1",
    )["blockers"] == ["policy_package_read_failed:ValueError"]
    bad_policy = tmp_path / "bad-policy.json"
    _write_json(bad_policy, {"schema_version": "bad"})
    policy_audit = intake._audit_policy_package(path=bad_policy, expected_job_id=None)
    assert "policy_package_schema_mismatch" in policy_audit["blockers"]
    assert "policy_package_job_id_missing" in policy_audit["blockers"]
    assert "policy_package_no_supported_modality" in policy_audit["blockers"]
    mismatched_policy = tmp_path / "mismatched-policy.json"
    _write_json(
        mismatched_policy,
        {
            "schema_version": "robot_team_policy_package.v1",
            "job_id": "other-job",
            "policy_package": {
                "policy_api_endpoint": {
                    "endpoint_url": "https://robot-team.example/policy"
                }
            },
        },
    )
    assert intake._audit_policy_package(
        path=mismatched_policy,
        expected_job_id="webapp-job-1",
    )["blockers"] == ["policy_package_job_id_mismatch"]

    assert intake._audit_live_closure_evidence(
        path=tmp_path / "missing-closure.json",
        expected_job_id="webapp-job-1",
    )["blockers"] == ["live_closure_evidence_missing"]
    assert intake._audit_live_closure_evidence(
        path=malformed,
        expected_job_id="webapp-job-1",
    )["blockers"] == ["live_closure_evidence_read_failed:ValueError"]
    bad_closure = tmp_path / "bad-closure.json"
    _write_json(
        bad_closure,
        {
            "schema_version": "live_robot_eval_closure_evidence.v0",
            "delivery": {"status": "pending"},
            "rights_privacy": {"accepted": False},
        },
    )
    closure_audit = intake._audit_live_closure_evidence(
        path=bad_closure,
        expected_job_id=None,
    )
    assert "live_closure_evidence_schema_mismatch" in closure_audit["blockers"]
    assert "live_closure_evidence_job_id_missing" in closure_audit["blockers"]
    assert "delivery_access_evidence_incomplete" in closure_audit["blockers"]
    assert "rights_privacy_evidence_blocked" in closure_audit["blockers"]


def test_live_pipeline_input_intake_staging_blocker_edges(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    evidence_path = _live_closure_evidence(tmp_path / "incoming" / "closure.json")
    outcome_path = _deployment_outcomes(tmp_path / "incoming" / "deployment.json")
    policy_path = _policy_package(tmp_path / "incoming" / "policy.json")
    pov_path = _real_robot_pov_manifest(tmp_path / "incoming" / "pov.json")
    _write_json(request_path, _webapp_request(capture_root))

    assert intake._stage_webapp_request(
        request_path=request_path,
        audit={"ready": False},
        inbox=tmp_path / "inbox",
        overwrite=False,
    )["blockers"] == ["webapp_request_not_ready_for_staging"]
    assert intake._stage_webapp_request(
        request_path=request_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        inbox=None,
        overwrite=False,
    )["blockers"] == ["missing_env_or_manifest_BLUEPRINT_ROBOT_EVAL_JOB_REQUEST_INBOX"]
    inbox = tmp_path / "inbox"
    _write_json(inbox / "webapp-job-1.json", {"already": True})
    assert intake._stage_webapp_request(
        request_path=request_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        inbox=inbox,
        overwrite=False,
    )["blockers"] == ["target_request_already_exists"]

    assert intake._stage_live_closure_evidence(
        evidence_path=evidence_path,
        audit={"ready": False},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["live_closure_evidence_not_ready_for_staging"]
    assert intake._stage_live_closure_evidence(
        evidence_path=evidence_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=None,
        overwrite=False,
    )["blockers"] == ["missing_control_plane_capture_root"]
    assert intake._stage_live_closure_evidence(
        evidence_path=evidence_path,
        audit={"ready": True},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["live_closure_evidence_job_id_missing"]
    assert intake._stage_live_closure_evidence(
        evidence_path=evidence_path,
        audit={"ready": True, "job_id": "../escape"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["live_closure_evidence_job_id_unsafe"]
    closure_target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "live_eval_closure_evidence.json"
    )
    _write_json(closure_target, {"already": True})
    assert intake._stage_live_closure_evidence(
        evidence_path=evidence_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["target_live_closure_evidence_already_exists"]

    assert intake._stage_deployment_outcomes(
        outcome_path=outcome_path,
        audit={"ready": False},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["deployment_outcomes_not_ready_for_staging"]
    assert intake._stage_deployment_outcomes(
        outcome_path=outcome_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=None,
        overwrite=False,
    )["blockers"] == ["missing_control_plane_capture_root"]
    assert intake._stage_deployment_outcomes(
        outcome_path=outcome_path,
        audit={"ready": True},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["deployment_outcomes_job_id_missing"]
    assert intake._stage_deployment_outcomes(
        outcome_path=outcome_path,
        audit={"ready": True, "job_id": "../escape"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["deployment_outcomes_job_id_unsafe"]
    deployment_target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "deployment_outcomes"
        / "inbox"
        / "pilot-outcome-1.json"
    )
    _write_json(deployment_target, {"already": True})
    assert intake._stage_deployment_outcomes(
        outcome_path=outcome_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["target_deployment_outcome_already_exists"]

    assert intake._stage_real_robot_pov(
        pov_path=pov_path,
        audit={"ready": False},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["real_robot_pov_not_ready_for_staging"]
    assert intake._stage_real_robot_pov(
        pov_path=pov_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=None,
        overwrite=False,
    )["blockers"] == ["missing_control_plane_capture_root"]
    pov_target = capture_root / "pipeline" / "robot_eval_inputs" / "real_robot_pov_manifest.json"
    _write_json(pov_target, {"already": True})
    assert intake._stage_real_robot_pov(
        pov_path=pov_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["target_real_robot_pov_already_exists"]

    assert intake._stage_policy_package(
        policy_path=policy_path,
        audit={"ready": False},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["policy_package_not_ready_for_staging"]
    assert intake._stage_policy_package(
        policy_path=policy_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=None,
        overwrite=False,
    )["blockers"] == ["missing_control_plane_capture_root"]
    assert intake._stage_policy_package(
        policy_path=policy_path,
        audit={"ready": True},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["policy_package_job_id_missing"]
    assert intake._stage_policy_package(
        policy_path=policy_path,
        audit={"ready": True, "job_id": "../escape"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["policy_package_job_id_unsafe"]
    policy_target = (
        capture_root
        / "pipeline"
        / "robot_eval_inputs"
        / "webapp-job-1"
        / "policy_package.json"
    )
    _write_json(policy_target, {"already": True})
    assert intake._stage_policy_package(
        policy_path=policy_path,
        audit={"ready": True, "job_id": "webapp-job-1"},
        capture_root=capture_root,
        overwrite=False,
    )["blockers"] == ["target_policy_package_already_exists"]


def test_live_pipeline_input_intake_waiting_and_blocked_staged_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    output_path = tmp_path / "custom" / "audit.json"

    waiting = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        output_path=output_path,
    )
    assert waiting["status"] == "waiting_for_inputs"
    assert waiting["output_path"] == str(output_path.resolve())

    blocked = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        arena_results_dir=tmp_path / "missing-arena",
        stage_arena_results=True,
        staged_inputs_path=tmp_path / "custom" / "staged.json",
    )
    assert blocked["status"] == "blocked"
    assert "arena:arena_results_dir_missing" in blocked["input_blockers"]
    assert "staged_inputs:arena_results_not_ready_for_staging" in blocked[
        "input_blockers"
    ]
    assert blocked["staged_inputs"]["blockers"] == ["arena_results_not_ready_for_staging"]


def test_live_pipeline_input_intake_main_covers_success_and_blocker_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    results_dir = _arena_results(tmp_path / "arena-results")
    evidence_path = _live_closure_evidence(tmp_path / "incoming" / "closure.json")
    outcome_path = _deployment_outcomes(tmp_path / "incoming" / "deployment.json")
    policy_path = _policy_package(tmp_path / "incoming" / "policy.json")
    pov_path = _real_robot_pov_manifest(tmp_path / "incoming" / "pov.json")
    _write_json(request_path, _webapp_request(capture_root))

    assert intake.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--webapp-job-request",
            str(request_path),
            "--arena-results-dir",
            str(results_dir),
            "--live-closure-evidence",
            str(evidence_path),
            "--deployment-outcomes",
            str(outcome_path),
            "--policy-package",
            str(policy_path),
            "--real-robot-pov",
            str(pov_path),
            "--stage-webapp-request",
            "--stage-arena-results",
            "--stage-live-closure-evidence",
            "--stage-deployment-outcomes",
            "--stage-policy-package",
            "--stage-real-robot-pov",
            "--overwrite",
            "--output-path",
            str(tmp_path / "cli" / "audit.json"),
            "--staged-inputs-path",
            str(tmp_path / "cli" / "staged.json"),
        ]
    ) == 0
    success = capsys.readouterr().out
    assert "status=staged_for_control_plane" in success
    assert "webapp_truth_proven=true" in success

    assert intake.main(
        [
            "--manifest-path",
            str(manifest_path),
            "--arena-results-dir",
            str(tmp_path / "missing-arena"),
            "--stage-arena-results",
        ]
    ) == 1
    blocked = capsys.readouterr().out
    assert "status=blocked" in blocked
    assert "blockers=" in blocked
