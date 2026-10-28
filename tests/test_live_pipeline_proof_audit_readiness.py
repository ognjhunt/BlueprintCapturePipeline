"""AUT-20: an absent or stale audit cannot prove even local package readiness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.live_pipeline_control_plane import PACKAGE_DIR_ENV, run_live_pipeline_control_plane
from blueprint_pipeline.live_pipeline_proof_audit import _goal_requirement_audit, build_live_pipeline_proof_audit
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


def _passed_audit(tmp_path: Path, *, package_name: str = "package") -> tuple[dict, dict, dict]:
    capture = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    package = capture / "pipeline" / package_name
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


_LOCAL_GOALS = ("arena_result_ingest", "500_scenario_scheduler", "policy_adapters", "clips",
                "review_resolution", "dataset_packaging", "customer_handoff_report", "rerun_loop")


def _run_control_plane_and_audit(capture: Path, output: Path, *, package: Path | None = None) -> tuple[dict, dict, dict]:
    run = run_live_pipeline_control_plane(
        capture_root=capture, package_dir=package, output_path=output, process_inbox=False,
        load_local_env=False, allow_digitalocean_read=False, allow_live_agent_operator=False,
        allow_gpu_provisioning=False, allow_simulator_execution=False, allow_cpu_simulator_preflight=False,
        allow_cpu_preflight_render=False, allow_training=False, allow_rollout_vision_labeling=False,
        allow_delivery_upload=False, allow_live_agents_sdk=False, allow_live_codex_sdk=False,
    )
    setup = json.loads(Path(run["setup_manifest_path"]).read_text())
    packet = json.loads(Path(run["external_input_packet"]["path"]).read_text())
    audit = build_live_pipeline_proof_audit(manifest_path=output)
    return setup, packet, audit


@pytest.mark.parametrize("selection", ("default", "argument", "environment"))
def test_control_plane_preserves_actual_package_binding_and_bounded_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str,
) -> None:
    monkeypatch.delenv(PACKAGE_DIR_ENV, raising=False)
    summary, paths, _ = _passed_audit(tmp_path, package_name="arena_eval_package" if selection == "default" else "selected-package")
    package = Path(paths["package_dir"])
    if selection == "environment":
        monkeypatch.setenv(PACKAGE_DIR_ENV, str(package))
    elif selection == "argument":
        monkeypatch.setenv(PACKAGE_DIR_ENV, str(tmp_path / "unselected-package"))
    setup, packet, audit = _run_control_plane_and_audit(
        Path(paths["capture_root"]), tmp_path / "control-plane.json",
        package=package if selection == "argument" else None,
    )
    accepted = setup["sections"]["local_deterministic_lane"]["package_audit"]
    assert accepted["audit_status"] == "passed" and accepted["artifact"]["path"] == summary["artifact"]["path"]
    assert setup["package_dir"] == str(package)
    assert packet["configured_paths"]["package_dir"] == str(package)
    assert packet["configured_paths"]["capture_root"] == paths["capture_root"]
    for name in _LOCAL_GOALS:
        goal = audit["goal_requirement_audit"][name]
        assert goal["status"] == "ready", goal
        assert goal["scope"] == "local_package_contract" and goal["execution_proven"] is False
        assert goal["evidence_sha256"] and goal["validated_artifacts"]
    assert audit["proof_boundary"]["audit_performs_live_actions"] is False
    assert audit["proof_boundary"]["simulator_execution_proven"] is False
    assert audit["proof_boundary"]["public_claim_upgrade_allowed"] is False


@pytest.mark.parametrize("damage", ("absent", "capture_binding", "package_binding"))
def test_default_package_binding_does_not_promote_missing_or_mismatched_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    monkeypatch.delenv(PACKAGE_DIR_ENV, raising=False)
    summary, paths, document = _passed_audit(tmp_path, package_name="arena_eval_package")
    audit_path = Path(summary["artifact"]["path"])
    if damage == "absent":
        audit_path.unlink()
    else:
        document["capture_root" if damage == "capture_binding" else "package_dir"] = str(tmp_path / "other-source")
        audit_path.write_text(json.dumps(document))
    _setup, _packet, audit = _run_control_plane_and_audit(Path(paths["capture_root"]), tmp_path / "control-plane.json")
    for name in _LOCAL_GOALS:
        goal = audit["goal_requirement_audit"][name]
        assert goal["status"] == "not_proven_in_current_control_plane"
        assert goal["reason"]
        if damage != "absent":
            assert goal["reason"] == "package_audit_source_binding_mismatch"


def test_explicit_missing_package_does_not_fall_back_to_a_valid_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PACKAGE_DIR_ENV, raising=False)
    _summary, paths, _document = _passed_audit(tmp_path, package_name="arena_eval_package")
    missing_package = tmp_path / "selected-missing-package"
    setup, packet, audit = _run_control_plane_and_audit(
        Path(paths["capture_root"]), tmp_path / "control-plane.json", package=missing_package,
    )
    assert setup["package_dir"] == packet["configured_paths"]["package_dir"] == str(missing_package)
    assert all(audit["goal_requirement_audit"][name]["status"] == "not_proven_in_current_control_plane"
               for name in _LOCAL_GOALS)
