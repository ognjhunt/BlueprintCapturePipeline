"""AUT-20: an absent or stale audit cannot prove even local package readiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.live_pipeline_proof_audit import _goal_requirement_audit


def _goals(package_audit: object, paths: dict | None = None) -> dict:
    return _goal_requirement_audit(
        manifest={}, packet={"configured_paths": paths or {}},
        setup_manifest={"sections": {"local_deterministic_lane": {"package_audit": package_audit}}},
    )


@pytest.mark.parametrize("package_audit", [
    None, [], {}, {"blockers": []}, {"audit_status": "unknown", "blockers": []},
    {"audit_status": "failed", "blockers": []},
    {"audit_status": "passed", "blockers": ["missing_source"]},
    {"audit_status": "passed", "blockers": []},
])
def test_missing_or_contradictory_audit_does_not_prove_package(package_audit: object) -> None:
    goals = _goals(package_audit)
    for name in ("arena_result_ingest", "500_scenario_scheduler", "policy_adapters", "clips",
                 "review_resolution", "dataset_packaging", "customer_handoff_report", "rerun_loop"):
        assert goals[name]["status"] == "not_proven_in_current_control_plane"
        assert goals[name]["reason"]


def _passed_audit(tmp_path: Path) -> tuple[dict, dict, dict]:
    capture = tmp_path / "capture"
    package = capture / "pipeline" / "package"
    package.mkdir(parents=True)
    files = (
        "arena_result_ingest_ledger.json", "arena_result_ingest_run_manifest.json",
        "arena_eval_schedule.json", "arena_eval_retry_queue.json", "arena_eval_resume_manifest.json",
        "policy_adapter_manifest.json", "clips_manifest.json", "review_resolution_ledger.json",
        "package_index.json", "dataset_card.json", "customer_handoff_report.json",
        "customer_handoff_report.md", "arena_rerun_plan.json", "arena_rerun_lineage.json",
    )
    assertions = {}
    for name in files:
        payload = b"{}\n"
        (package / name).write_bytes(payload)
        assertions[name] = {"path": name, "exists": True, "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()}
    document = {
        "schema_version": "arena_package_proof_boundary_audit.v1", "status": "passed",
        "blockers": [], "capture_root": str(capture), "package_dir": str(package),
        "artifact_assertions": assertions, "proof_boundary_violations": [],
    }
    path = package / "arena_package_proof_boundary_audit.json"
    path.write_text(json.dumps(document))
    summary = {"audit_status": "passed", "blockers": [],
               "artifact": {"path": str(path), "exists": True}}
    return summary, {"package_dir": str(package), "capture_root": str(capture)}, document


def test_explicit_pass_with_source_and_artifact_bytes_proves_only_local_contracts(tmp_path: Path) -> None:
    summary, paths, _ = _passed_audit(tmp_path)
    goals = _goals(summary, paths)
    assert goals["clips"]["status"] == "ready"
    assert goals["clips"]["scope"] == "local_package_contract"
    assert goals["clips"]["execution_proven"] is False
    assert goals["clips"]["evidence_sha256"]
    assert goals["clips"]["validated_artifacts"] == ["clips_manifest.json"]


@pytest.mark.parametrize("change", ["schema", "capture", "package", "artifact", "missing", "blockers"])
def test_changed_evidence_cannot_keep_a_ready_projection(tmp_path: Path, change: str) -> None:
    summary, paths, document = _passed_audit(tmp_path)
    package = Path(paths["package_dir"])
    if change == "artifact":
        (package / "clips_manifest.json").write_text('{"changed": true}')
    elif change == "missing":
        (package / "clips_manifest.json").unlink()
    else:
        field, value = {"schema": ("schema_version", "other"), "capture": ("capture_root", "/other/capture"),
                        "package": ("package_dir", "/other/package"),
                        "blockers": ("blockers", ["contradiction"])}[change]
        document[field] = value
        Path(summary["artifact"]["path"]).write_text(json.dumps(document))
    goal = _goals(summary, paths)["clips"]
    assert goal["status"] == "not_proven_in_current_control_plane"
    assert goal["reason"]
