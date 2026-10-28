"""AUT-20: an absent or stale audit cannot prove even local package readiness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.live_pipeline_proof_audit import _goal_requirement_audit
from blueprint_pipeline.arena_package_audit import REQUIRED_ARTIFACTS, build_arena_package_proof_boundary_audit


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
    capture = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    package = capture / "pipeline" / "package"
    package.mkdir(parents=True)
    (capture / "capture_descriptor.json").write_text(json.dumps({"scene_id": "scene-1", "capture_id": "capture-1"}))
    values = {
        "arena_eval_schedule.json": {"scenario_count": 500, "shard_count": 1},
        "normalized_attempt_trace.json": {"attempt_count": 1},
        "clips_manifest.json": {"clip_count": 1},
        "failure_labels.json": {"label_count": 0},
        "post_training_data_package_export_manifest.json": {"status": "export_ready_review_required"},
        "archive_manifest.json": {"archive": {"exists": True}},
        "delivery_manifest.json": {"status": "local_delivery_bundle_ready"},
        "signed_access_manifest.json": {"status": "not_requested"},
        "live_operator_ledger.json": {"status": "not_requested"},
        "review_resolution_ledger.json": {"status": "no_review_required"},
        "arena_rerun_plan.json": {"status": "no_eligible_reruns"},
    }
    for name in REQUIRED_ARTIFACTS:
        payload = (json.dumps(values.get(name, {})) + "\n").encode()
        (package / name).write_bytes(payload)
    document = build_arena_package_proof_boundary_audit(capture_root=capture, package_dir=package)
    assert document["status"] == "passed", document["blockers"]
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


def test_changed_cross_capability_dependency_invalidates_canonical_audit_projection(tmp_path: Path) -> None:
    summary, paths, _ = _passed_audit(tmp_path)
    assert _goals(summary, paths)["clips"]["status"] == "ready"
    package = Path(paths["package_dir"])
    (package / "normalized_attempt_trace.json").write_text(json.dumps({"attempt_count": 2}))
    canonical = build_arena_package_proof_boundary_audit(capture_root=paths["capture_root"], package_dir=package,
                                                        output_path=tmp_path / "canonical-recheck.json")
    assert "clip_count_does_not_match_attempt_count" in canonical["blockers"]
    goal = _goals(summary, paths)["clips"]
    assert goal["status"] == "not_proven_in_current_control_plane"
    assert goal["reason"] == "package_audit_dependency_digest_mismatch"
