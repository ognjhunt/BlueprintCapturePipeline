"""Retained source admission remains fail-closed after the read-only extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_controls_source_evidence as source
from blueprint_pipeline import task_evaluation_controls_context_contracts as metadata
from blueprint_pipeline import task_evaluation_launch_evidence_contracts as launch
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _seal(value: dict, field: str) -> dict:
    return {**value, field: canonical_digest(value, digest_field=field)}


def _source(tmp_path: Path, *, cross_runtime: bool = False,
            run_root: Path | None = None, profile_digest: str | None = None) -> Path:
    root = run_root or tmp_path / "launch" / "source-launch"
    result_path = tmp_path / "terminal" / "result.json"
    _write(result_path, {"configured_scene_revision_digest": "sha256:" + "a" * 64})
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": "completed",
        "launch_id": root.name,
        "run_id": "scene-run",
        "request_digest": "sha256:" + "1" * 64,
        "launch_profile_digest": profile_digest or "sha256:" + "2" * 64,
        "terminal_evidence": {
            "status": "passed",
            "scene_configuration": {"configuration_completed": True},
            "result": {
                "path": str(result_path),
                "exists": True,
                "digest": "sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest(),
            },
        },
    }
    if cross_runtime:
        receipt.update(receipt_digest_canonicalization="rfc8785", retained_number=1.0)
    digest = cross_runtime_canonical_digest if cross_runtime else canonical_digest
    receipt["receipt_digest"] = digest(receipt, digest_field="receipt_digest")
    _write(root / "launch_receipt.json", receipt)
    identity = {key: receipt[key] for key in ("launch_id", "run_id", "request_digest", "receipt_digest")}
    sync = _seal({
        "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
        "status": "succeeded",
        **identity,
        "attempt_number": 1,
        "attempted_at": "2026-09-29T12:00:00Z",
        "provider_mutation_performed": False,
        "response": {
            "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
            "status": "completed",
            "already_exists": False,
            **identity,
        },
    }, "sync_result_digest")
    _write(root / "webapp_sync_succeeded.json", sync)
    zero = _seal({
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        **identity,
        "launch_profile_digest": receipt["launch_profile_digest"],
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "allocator_invoked": False,
        "provider_mutation_performed": False,
        "automatic_retry_performed": False,
        "blockers": [],
    }, "provider_zero_receipt_digest")
    _write(root / "post_teardown_provider_zero_receipt.json", zero)
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("cross_runtime", [False, True])
def test_valid_source_matches_worker_without_writing(tmp_path, cross_runtime):
    from blueprint_pipeline import task_evaluation_configured_controls_progression_worker as worker
    from blueprint_pipeline.configured_controls_plan_validation import (
        TaskEvaluationConfiguredControlsProgressionWorkerError,
    )

    root = _source(tmp_path, cross_runtime=cross_runtime)
    before = _snapshot(tmp_path)
    result, receipt, zero = source.validate_source(root)
    assert worker._validate_source(root) == (result, receipt, zero)
    assert result["configured_scene_revision_digest"] == "sha256:" + "a" * 64
    assert zero["receipt_digest"] == receipt["receipt_digest"]
    assert _snapshot(tmp_path) == before
    assert TaskEvaluationConfiguredControlsProgressionWorkerError is source.TaskEvaluationConfiguredControlsProgressionWorkerError


@pytest.mark.parametrize("mutation,blocker", [
    ("receipt_digest", "qualifying_terminal_missing"),
    ("failed_terminal", "qualifying_terminal_missing"),
    ("terminal_bytes", "terminal_artifact_invalid"),
    ("terminal_symlink", "terminal_artifact_invalid"),
    ("sync_receipt", "webapp_sync_invalid"),
    ("sync_response", "webapp_sync_invalid"),
    ("sync_attempt_bool", "webapp_sync_invalid"),
    ("sync_time_without_zone", "webapp_sync_invalid"),
    ("sync_mutation", "webapp_sync_invalid"),
    ("zero_missing", "post_teardown_provider_zero_missing"),
    ("zero_digest", "post_teardown_provider_zero_invalid"),
    ("zero_binding", "post_teardown_provider_zero_invalid"),
    ("zero_burn", "post_teardown_provider_zero_invalid"),
    ("zero_retry", "post_teardown_provider_zero_invalid"),
])
def test_source_refuses_changed_evidence_without_writing(tmp_path, mutation, blocker):
    root = _source(tmp_path)
    receipt_path = root / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    sync_path = root / "webapp_sync_succeeded.json"
    sync = json.loads(sync_path.read_text())
    zero_path = root / "post_teardown_provider_zero_receipt.json"
    zero = json.loads(zero_path.read_text())
    if mutation == "receipt_digest":
        receipt["run_id"] = "forged"
        _write(receipt_path, receipt)
    elif mutation == "failed_terminal":
        receipt["terminal_evidence"]["status"] = "failed"
        _write(receipt_path, _seal(receipt, "receipt_digest"))
    elif mutation.startswith("terminal_"):
        terminal = Path(receipt["terminal_evidence"]["result"]["path"])
        if mutation == "terminal_bytes":
            terminal.write_text("{}", encoding="utf-8")
        else:
            destination = terminal.with_name("actual.json")
            terminal.rename(destination)
            terminal.symlink_to(destination)
    elif mutation.startswith("sync_"):
        if mutation == "sync_receipt":
            sync["receipt_digest"] = "sha256:" + "3" * 64
        elif mutation == "sync_response":
            sync["response"]["request_digest"] = "sha256:" + "4" * 64
        elif mutation == "sync_attempt_bool":
            sync["attempt_number"] = True
        elif mutation == "sync_time_without_zone":
            sync["attempted_at"] = "2026-09-29T12:00:00"
        else:
            sync["provider_mutation_performed"] = True
        _write(sync_path, _seal(sync, "sync_result_digest"))
    elif mutation == "zero_missing":
        zero_path.unlink()
    else:
        if mutation == "zero_digest":
            zero["provider_zero_verified"] = False
            _write(zero_path, zero)
        else:
            field = {"zero_binding": "launch_profile_digest", "zero_burn": "continuing_spend_from_this_run", "zero_retry": "automatic_retry_performed"}[mutation]
            zero[field] = "sha256:" + "5" * 64 if mutation == "zero_binding" else True
            _write(zero_path, _seal(zero, "provider_zero_receipt_digest"))
    before = _snapshot(tmp_path)
    with pytest.raises(source.TaskEvaluationConfiguredControlsProgressionWorkerError, match="configured_controls_worker_" + blocker):
        source.validate_source(root)
    assert _snapshot(tmp_path) == before


def test_metadata_reexports_retain_number_semantics_and_symlink_refusal(tmp_path):
    from blueprint_pipeline import task_evaluation_controls_autoprovision as worker
    from blueprint_pipeline import task_evaluation_scene_intent_contracts as intake

    path = tmp_path / "intent.json"
    value = intake._seal({"accepted_at_epoch": 1.0}, "intent_digest")
    _write(path, value)
    assert metadata._scene_intent(path) == worker._scene_intent(path) == value
    assert worker._json is metadata._json
    assert worker._sealed is metadata._sealed
    assert worker.CONFIG_ENV == metadata.CONFIG_ENV
    assert worker.CATALOG_SCHEMA == metadata.CATALOG_SCHEMA
    assert worker.CONTENT_CATALOG_SCHEMA == metadata.CONTENT_CATALOG_SCHEMA
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="^controls_autoprovision_symlink_refused$"):
        metadata._scene_intent(alias / path.name)
    value["accepted_at_epoch"] = 2
    _write(path, value)
    with pytest.raises(intake.SceneIntakeError):
        metadata._scene_intent(path)


def test_sync_reexports_keep_exact_offering_acknowledgement(tmp_path):
    from blueprint_pipeline import task_evaluation_launch_reconciler as reconciler

    root = _source(tmp_path)
    receipt = json.loads((root / "launch_receipt.json").read_text())
    sync = json.loads((root / "webapp_sync_succeeded.json").read_text())
    offering = {"status": "configured_controls_pending", "offering_digest": "sha256:" + "6" * 64}
    receipt["terminal_evidence"]["scene_configuration"]["configured_scene_offering"] = offering
    for row in (sync, sync["response"]):
        row["configured_scene_offering_digest"] = offering["offering_digest"]
        row["configured_scene_offering_status"] = offering["status"]
    sync = _seal(sync, "sync_result_digest")
    assert reconciler.validated_succeeded_webapp_sync_row is launch.validated_succeeded_webapp_sync_row
    assert reconciler.validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)["website_trigger_proven"] is True
    sync["response"]["configured_scene_offering_digest"] = "sha256:" + "7" * 64
    sync = _seal(sync, "sync_result_digest")
    with pytest.raises(launch.TaskEvaluationLaunchError, match="webapp_sync_succeeded_invalid"):
        launch.validated_succeeded_webapp_sync_row(receipt=receipt, attempt=sync)


@pytest.mark.parametrize("seed", [
    "task_evaluation_controls_context_contracts",
    "task_evaluation_configured_controls_source_evidence",
    "task_evaluation_launch_evidence_contracts",
    "task_evaluation_team_run_context",
])
def test_read_only_closures_cannot_reach_active_workers(seed):
    from tests.test_live_pipeline_import_isolation import (
        HOT_LANE_MODULES,
        PKG_DIR,
        _transitive_local_modules,
    )

    active = {
        "task_evaluation_controls_autoprovision",
        "task_evaluation_configured_controls_progression_worker",
        "task_evaluation_launch_dispatcher",
        "task_evaluation_launch_reconciler",
        "task_evaluation_scene_intake",
        "task_evaluation_configured_controls_autostart",
        "task_evaluation_robot_placement_warm_executor",
        "paid_resource_allocator",
        "native_task_arena_direct_execution_closeout",
        "paid_attempt_authority",
    }
    assert (PKG_DIR / f"{seed}.py").is_file()
    assert not _transitive_local_modules((seed,)) & (HOT_LANE_MODULES | active)
