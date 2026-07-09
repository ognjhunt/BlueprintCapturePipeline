from __future__ import annotations

from pathlib import Path

import scripts.run_beta_intake_soak_test as soak
from scripts.run_beta_intake_soak_test import build_capacity_cost_summary, build_dry_run_report
from scripts.validate_capture_truth_backup_policy import (
    validate_backup_policy,
    validate_restore_drill_artifact,
)
from scripts.validate_beta_capacity_storage import validate_files


def test_beta_capacity_storage_artifacts_are_complete() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = validate_files(repo_root)

    assert result["status"] == "passed"
    assert result["external_users"] == 100
    assert result["modeled_captures_per_month"] == 300
    assert result["target_concurrent_uploaders"] == 25
    assert result["retention_policy_path"].endswith(
        "docs/beta_data_retention_policy_2026-07-09.json"
    )


def test_beta_capacity_storage_validator_checks_capture_swift_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    capture_policy = repo_root.parent / "BlueprintCapture" / "BlueprintCapture" / "Services" / "CaptureUploadFilePlan.swift"
    result = validate_files(repo_root, capture_policy if capture_policy.exists() else None)

    assert result["status"] == "passed"


def test_beta_intake_soak_dry_run_report_preserves_claim_boundary() -> None:
    model = {
        "schema_version": "blueprint.beta_capacity_cost_storage_model.v1",
        "beta_target": {
            "external_users": 100,
            "modeled_captures_per_month": 300,
            "target_concurrent_uploaders": 25,
        },
        "per_capture_limits": {
            "max_upload_payload_bytes": 20 * 1024 * 1024 * 1024,
            "max_duration_seconds": 45 * 60,
        },
    }

    report = build_dry_run_report(model, concurrency=25, duration_seconds=900)

    assert report["status"] == "dry_run"
    assert report["planned_concurrency"] == 25
    assert report["target_concurrency_met"] is True
    assert report["blockers"] == []
    assert report["planned_duration_seconds"] == 900
    assert report["claim_boundary"] == "dry_run_only_no_network_requests_were_sent"
    assert report["cost_per_capture_model"]["budget_cap_usd_per_capture"] == 16.67
    assert report["firestore_latency_observation"]["schema_version"] == (
        "blueprint.firestore_latency_observation.v1"
    )
    assert report["firestore_latency_observation"]["status"] == "not_provided"
    assert report["firestore_created_at_hotspot_policy"]["shard_field"] == "createdAtShard"


def test_beta_intake_soak_blocks_below_target_concurrency() -> None:
    model = {
        "schema_version": "blueprint.beta_capacity_cost_storage_model.v1",
        "beta_target": {
            "external_users": 100,
            "modeled_captures_per_month": 300,
            "target_concurrent_uploaders": 25,
        },
        "per_capture_limits": {
            "max_upload_payload_bytes": 20 * 1024 * 1024 * 1024,
            "max_duration_seconds": 45 * 60,
        },
    }

    report = build_dry_run_report(model, concurrency=10, duration_seconds=900)

    assert report["status"] == "blocked"
    assert report["target_concurrency_met"] is False
    assert report["blockers"] == ["concurrency_below_beta_target:10<target:25"]


def test_beta_cost_per_capture_model_is_carried_into_soak_reports() -> None:
    model = {
        "schema_version": "blueprint.beta_capacity_cost_storage_model.v1",
        "beta_target": {
            "external_users": 100,
            "modeled_captures_per_month": 300,
            "target_concurrent_uploaders": 25,
        },
        "budget_guardrails": {"cohort_hard_stop_threshold_usd": 5000},
        "monthly_projection": {"total_new_storage_gib_p50": 1260},
        "per_capture_limits": {
            "max_upload_payload_bytes": 20 * 1024 * 1024 * 1024,
            "max_duration_seconds": 45 * 60,
        },
        "cost_per_capture_model": {
            "schema_version": "blueprint.beta_cost_per_capture_model.v1",
            "status": "planning_estimate_not_live_billing_proof",
            "modeled_captures_per_month": 300,
            "budget_cap_usd_per_capture": 16.67,
            "budget_cap_usd_per_100_user_month": 5000,
            "estimated_usd_per_capture_p50": 3.56,
        },
    }

    summary = build_capacity_cost_summary(model)
    report = build_dry_run_report(model, concurrency=25, duration_seconds=900)

    assert summary["schema_version"] == "blueprint.beta_cost_per_capture_model.v1"
    assert summary["budget_cap_usd_per_capture"] == 16.67
    assert report["cost_per_capture_model"]["estimated_usd_per_capture_p50"] == 3.56


def test_beta_intake_soak_executes_concurrent_request_cap_without_network(
    monkeypatch,
) -> None:
    model = {
        "schema_version": "blueprint.beta_capacity_cost_storage_model.v1",
        "beta_target": {
            "external_users": 100,
            "modeled_captures_per_month": 300,
            "target_concurrent_uploaders": 3,
        },
        "budget_guardrails": {"cohort_hard_stop_threshold_usd": 5000},
        "monthly_projection": {"total_new_storage_gib_p50": 1260},
        "per_capture_limits": {
            "max_upload_payload_bytes": 20 * 1024 * 1024 * 1024,
            "max_duration_seconds": 45 * 60,
        },
    }
    calls: list[dict[str, object]] = []

    def fake_request_once(
        url: str,
        payload: bytes,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout": timeout,
                "headers": headers,
            }
        )
        return {"ok": True, "status": 202, "latency_ms": 12.0}

    monkeypatch.setattr(soak, "_request_once", fake_request_once)

    report = soak.run_soak(
        "https://intake.example.test/probe",
        model,
        concurrency=3,
        request_count=6,
        duration_seconds=60,
        timeout=2.5,
        headers={"x-blueprint-test": "yes"},
    )

    assert report["status"] == "passed"
    assert report["requests_executed"] == 6
    assert report["ok_count"] == 6
    assert report["failure_count"] == 0
    assert report["target_concurrency_met"] is True
    assert report["latency_ms"]["p95"] == 12.0
    assert report["cost_per_capture_model"]["budget_cap_usd_per_capture"] == 16.67
    assert len(calls) == 6
    assert all(call["headers"] == {"x-blueprint-test": "yes"} for call in calls)


def test_beta_intake_soak_requires_firestore_latency_observation_when_enabled(
    monkeypatch,
) -> None:
    model = {
        "schema_version": "blueprint.beta_capacity_cost_storage_model.v1",
        "beta_target": {
            "external_users": 100,
            "modeled_captures_per_month": 300,
            "target_concurrent_uploaders": 2,
        },
        "per_capture_limits": {
            "max_upload_payload_bytes": 20 * 1024 * 1024 * 1024,
            "max_duration_seconds": 45 * 60,
        },
        "runtime_capacity": {
            "firestore_created_at_hotspot_policy": {
                "latency_metric": "serviceruntime.googleapis.com/api/request_latencies",
                "monitoring_alert_policy": "google_monitoring_alert_policy.firestore_request_latency",
                "p99_alert_threshold_seconds": 0.25,
                "p99_alert_duration_seconds": 300,
            }
        },
    }

    monkeypatch.setattr(
        soak,
        "_request_once",
        lambda *args, **kwargs: {"ok": True, "status": 202, "latency_ms": 10.0},
    )

    missing = soak.run_soak(
        "https://intake.example.test/probe",
        model,
        concurrency=2,
        request_count=2,
        duration_seconds=60,
        timeout=2.5,
        require_firestore_latency=True,
    )
    observed = soak.run_soak(
        "https://intake.example.test/probe",
        model,
        concurrency=2,
        request_count=2,
        duration_seconds=60,
        timeout=2.5,
        require_firestore_latency=True,
        firestore_p99_latency_seconds=0.12,
        firestore_latency_source="output/beta_capacity/firestore-p99-latency.txt",
    )
    exceeded = soak.run_soak(
        "https://intake.example.test/probe",
        model,
        concurrency=2,
        request_count=2,
        duration_seconds=60,
        timeout=2.5,
        require_firestore_latency=True,
        firestore_p99_latency_seconds=0.4,
        firestore_latency_source="output/beta_capacity/firestore-p99-latency.txt",
    )

    assert missing["status"] == "failed"
    assert "firestore_latency_observation_missing" in missing["blockers"]
    assert observed["status"] == "passed"
    assert observed["firestore_latency_observation"]["status"] == "passed"
    assert exceeded["status"] == "failed"
    assert "firestore_p99_latency_exceeded:0.400>0.250" in exceeded["blockers"]


def test_capture_truth_backup_policy_artifacts_are_complete() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = validate_backup_policy(repo_root)

    assert result["status"] == "passed"
    assert result["script"].endswith("scripts/apply_capture_truth_backup_policy.sh")
    assert result["runbook"].endswith("docs/CAPTURE_TRUTH_BACKUP_DR_RUNBOOK_2026-07-08.md")


def test_capture_truth_restore_drill_artifact_validator(tmp_path: Path) -> None:
    artifact = tmp_path / "capture_truth_restore_drill.json"
    artifact.write_text(
        """
{
  "schema_version": "capture_truth_restore_drill.v1",
  "status": "passed",
  "source_project_id": "blueprint-prod",
  "restore_project_id": "blueprint-restore-drill",
  "non_production_restore_project": true,
  "firestore_restore": {
    "backup_id": "projects/blueprint-prod/databases/(default)/backups/backup-1",
    "validation_status": "passed",
    "restored_document_paths": ["capture_submissions/capture-1"]
  },
  "storage_restore": {
    "bucket": "gs://primary-capture-bucket",
    "restored_object": "restore-drill/scenes/scene-1/captures/capture-1/raw/manifest.json",
    "raw_manifest_generation": "1700000000000000",
    "restored_checksum_sha256": "0123456789abcdef",
    "validation_status": "passed"
  },
  "transcript": {
    "path": "output/beta_capacity/backup_drill/transcript.redacted.txt",
    "secrets_redacted": true
  },
  "claim_boundary": {
    "live_restore_drill_executed": true,
    "production_restore_performed": false
  }
}
""".strip(),
        encoding="utf-8",
    )

    result = validate_restore_drill_artifact(artifact)

    assert result["status"] == "passed"
    assert result["restore_drill_artifact"] == str(artifact)
    assert result["source_project_id"] == "blueprint-prod"
    assert result["restore_project_id"] == "blueprint-restore-drill"
