from __future__ import annotations

from pathlib import Path

from scripts.run_beta_intake_soak_test import build_dry_run_report
from scripts.validate_capture_truth_backup_policy import validate_backup_policy
from scripts.validate_beta_capacity_storage import validate_files


def test_beta_capacity_storage_artifacts_are_complete() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = validate_files(repo_root)

    assert result["status"] == "passed"
    assert result["external_users"] == 100
    assert result["modeled_captures_per_month"] == 300
    assert result["target_concurrent_uploaders"] == 25


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


def test_capture_truth_backup_policy_artifacts_are_complete() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = validate_backup_policy(repo_root)

    assert result["status"] == "passed"
    assert result["script"].endswith("scripts/apply_capture_truth_backup_policy.sh")
    assert result["runbook"].endswith("docs/CAPTURE_TRUTH_BACKUP_DR_RUNBOOK_2026-07-08.md")
