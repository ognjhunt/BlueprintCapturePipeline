from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import arena_package_audit as audit


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / "site-1" / "captures" / "cap-1"
    root.mkdir(parents=True)
    return root


def _write_required_package(package_dir: Path, *, scenario_count: int = 500, attempt_count: int = 2) -> None:
    for name in audit.REQUIRED_ARTIFACTS:
        path = package_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".md"):
            path.write_text("# handoff\n", encoding="utf-8")
        elif name.endswith(".jsonl"):
            path.write_text('{"row": 1}\n', encoding="utf-8")
        else:
            _write_json(path, {})
    for name in audit.JOB_ARTIFACTS:
        _write_json(package_dir / name, {})

    _write_json(package_dir / "arena_eval_schedule.json", {"scenario_count": scenario_count, "shard_count": 1})
    _write_json(package_dir / "normalized_attempt_trace.json", {"attempt_count": attempt_count})
    _write_json(package_dir / "failure_labels.json", {"label_count": 1})
    _write_json(package_dir / "clips_manifest.json", {"clip_count": attempt_count})
    _write_json(package_dir / "post_training_data_package_export_manifest.json", {"status": "export_ready_review_required"})
    _write_json(package_dir / "archive_manifest.json", {"archive": {"exists": True}})
    _write_json(package_dir / "delivery_manifest.json", {"status": "local_delivery_bundle_ready"})
    _write_json(package_dir / "signed_access_manifest.json", {"status": "ready"})
    _write_json(package_dir / "live_operator_ledger.json", {"status": "completed"})
    _write_json(package_dir / "review_resolution_ledger.json", {"status": "accepted_labels_ready"})
    _write_json(package_dir / "arena_rerun_plan.json", {"status": "no_eligible_reruns"})


def test_arena_package_audit_helpers_and_package_resolution(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    context_pipeline = capture_root / "pipeline"
    assert audit._string_list(None) == []
    assert audit._string_list("one") == ["one"]
    assert audit._string_list(["one", "one", "", "two"]) == ["one", "two"]
    assert audit._string_list(3) == ["3"]
    assert audit._relative_to(capture_root, capture_root / "pipeline" / "file.json") == "pipeline/file.json"
    assert audit._latest_arena_job_dir(context_pipeline) is None

    jobs_root = context_pipeline / "robot_eval_jobs"
    (jobs_root / "empty").mkdir(parents=True)
    assert audit._latest_arena_job_dir(context_pipeline) is None
    _write_json(jobs_root / "job-1" / "arena_result_ingest_run_manifest.json", {"status": "completed"})
    assert audit._latest_arena_job_dir(context_pipeline) == jobs_root / "job-1"
    assert audit._resolve_package_dir(capture_root, None) == jobs_root / "job-1"

    explicit = tmp_path / "explicit-package"
    assert audit._resolve_package_dir(capture_root, explicit) == explicit.resolve()

    (jobs_root / "job-1" / "arena_result_ingest_run_manifest.json").unlink()
    assert audit._resolve_package_dir(capture_root, None) == context_pipeline / "arena_eval_package"


def test_arena_package_audit_passes_valid_package_and_cli(tmp_path: Path, capsys) -> None:
    capture_root = _capture_root(tmp_path)
    package_dir = tmp_path / "arena-package"
    _write_required_package(package_dir)
    output_path = tmp_path / "audit.json"

    result = audit.build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=package_dir,
        expected_scenario_count=500,
        require_job_artifacts=True,
        output_path=output_path,
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["summary"]["scenario_count"] == 500
    assert result["artifact_assertions"]["arena_eval_schedule.json"]["exists"] is True
    assert output_path.is_file()

    assert audit.main(
        [
            "--capture-root",
            str(capture_root),
            "--package-dir",
            str(package_dir),
            "--expected-scenario-count",
            "500",
            "--require-job-artifacts",
            "--output-path",
            str(tmp_path / "audit-cli.json"),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "[arena-package-audit] manifest=" in output
    assert "status=passed" in output


def test_arena_package_audit_blocks_bad_package_and_proof_boundary_violations(tmp_path: Path, capsys) -> None:
    capture_root = _capture_root(tmp_path)
    empty_package = tmp_path / "empty-package"
    empty_package.mkdir()
    empty_result = audit.build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=empty_package,
        expected_scenario_count=500,
    )
    assert "normalized_attempt_trace_empty" in empty_result["blockers"]
    assert "signed_access_manifest_missing" in empty_result["blockers"]
    assert "live_operator_ledger_missing" in empty_result["blockers"]

    package_dir = tmp_path / "bad-package"
    package_dir.mkdir()
    _write_json(package_dir / "arena_eval_schedule.json", {"scenario_count": 3, "shard_count": 0})
    _write_json(package_dir / "normalized_attempt_trace.json", {"attempt_count": 2})
    _write_json(package_dir / "clips_manifest.json", {"clip_count": 1})
    _write_json(package_dir / "policy_adapter_manifest.json", {"rank_fidelity_result_proven": True, "claim_boundary": {"non_ranking_operational_claim_validated": True}})
    _write_json(package_dir / "signed_access_manifest.json", {"status": "blocked", "blockers": ["missing_command", "missing_command"]})
    _write_json(package_dir / "live_operator_ledger.json", {"status": "blocked", "blockers": "missing_agents_sdk"})
    (package_dir / "arena_eval_metrics.json").write_text("{bad-json", encoding="utf-8")

    result = audit.build_arena_package_proof_boundary_audit(
        capture_root=capture_root,
        package_dir=package_dir,
        expected_scenario_count=500,
    )

    expected_blockers = {
        "arena_schedule_scenario_count_mismatch",
        "arena_schedule_shards_missing",
        "clip_count_does_not_match_attempt_count",
        "failure_labels_manifest_missing_or_invalid",
        "post_training_data_package_not_export_ready_review_required",
        "post_training_data_package_archive_missing",
        "local_delivery_bundle_not_ready",
        "review_resolution_status_unexpected",
        "rerun_plan_status_unexpected",
        "proof_boundary_violation:policy_adapter_manifest.json:rank_fidelity_result_proven",
        "proof_boundary_violation:policy_adapter_manifest.json:claim_boundary.non_ranking_operational_claim_validated",
    }
    assert expected_blockers.issubset(set(result["blockers"]))
    assert result["status"] == "blocked"
    assert result["external_blockers"] == [
        {
            "system": "storage_delivery",
            "status": "blocked_or_not_requested",
            "blockers": ["missing_command"],
            "next_input_needed": (
                "Set BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD=true, pass "
                "--allow-delivery-upload, and provide a delivery command when live upload "
                "or signed URLs are required."
            ),
        },
        {
            "system": "live_agents_codex_operators",
            "status": "blocked",
            "blockers": ["missing_agents_sdk"],
            "next_input_needed": (
                "Set the live operator env gates, provide SDK dependencies and credentials, "
                "and pass the live operator CLI flags when real SDK execution is required."
            ),
        },
    ]

    assert audit.main(["--capture-root", str(capture_root), "--package-dir", str(package_dir)]) == 1
    assert "status=blocked" in capsys.readouterr().out


def test_arena_package_audit_allows_live_closure_proven_fields(tmp_path: Path) -> None:
    package_dir = tmp_path / "closure-package"
    _write_json(
        package_dir / "live_eval_closure_manifest.json",
        {
            "status": "live_end_to_end_verified",
            "live_end_to_end_verified": True,
            "proof_boundary": {"non_ranking_operational_claim_validated": True},
        },
    )
    _write_json(package_dir / "proof.json", {"claim_boundary": {"non_ranking_operational_claim_validated": True}})
    _write_json(package_dir / "proof_top.json", {"non_ranking_operational_claim_validated": True})

    assert audit._proof_field_violations(package_dir, ["proof.json", "proof_top.json"]) == []
