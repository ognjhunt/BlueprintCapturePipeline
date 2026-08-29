from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_terminal_scene_payload_retention import (
    APPLY_ACK,
    TaskEvaluationTerminalScenePayloadRetentionError,
    apply_terminal_scene_payload_retention,
    archive_terminal_scene_payload_to_b2,
    plan_terminal_scene_payload_retention,
)
from blueprint_pipeline import task_evaluation_terminal_scene_payload_retention as retention


EXPECTED_BUCKET = "blueprint-task-evaluation-artifacts-prod"


def test_cli_serializes_domain_refusal_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(**_: object) -> dict[str, object]:
        raise TaskEvaluationTerminalScenePayloadRetentionError(
            "terminal_scene_payload_provider_nonzero"
        )

    monkeypatch.setattr(retention, "archive_terminal_scene_payload_to_b2", refuse)

    status = retention.main(
        [
            "archive",
            "--scope-kind",
            "diagnostic",
            "--scope-root",
            "/nonexistent/scope",
            "--managed-root",
            "/nonexistent/managed",
            "--expected-bucket",
            EXPECTED_BUCKET,
            "--b2-index-out",
            "/nonexistent/index.json",
        ]
    )

    assert status == 2
    assert json.loads(capsys.readouterr().out) == {
        "blockers": ["terminal_scene_payload_provider_nonzero"],
        "removed_bytes": 0,
        "removed_count": 0,
        "status": "blocked",
    }


def test_cli_does_not_mask_unexpected_programmer_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_: object) -> dict[str, object]:
        raise RuntimeError("unexpected programmer fault")

    monkeypatch.setattr(retention, "archive_terminal_scene_payload_to_b2", fail)

    with pytest.raises(RuntimeError, match="unexpected programmer fault"):
        retention.main(
            [
                "archive",
                "--scope-kind",
                "diagnostic",
                "--scope-root",
                "/nonexistent/scope",
                "--managed-root",
                "/nonexistent/managed",
                "--expected-bucket",
                EXPECTED_BUCKET,
                "--b2-index-out",
                "/nonexistent/index.json",
            ]
        )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, object], *, digest_field: str | None = None) -> None:
    payload = dict(value)
    if digest_field:
        payload[digest_field] = ""
        payload[digest_field] = canonical_digest(payload, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _remote_reference(
    path: Path,
    *,
    artifact_kind: str = "provider-output",
    bucket: str = EXPECTED_BUCKET,
) -> dict[str, object]:
    digest = _sha256(path)
    return {
        "schema_version": "task_evaluation_scene_artifact_reference.v1",
        "status": "remote_verified",
        "artifact_kind": artifact_kind,
        "uri": (
            f"s3://{bucket}/blueprint/arm-decision-proof-v1/configured-scenes/"
            f"artifacts/{artifact_kind}/sha256/{digest.removeprefix('sha256:')}/"
            f"{path.name}"
        ),
        "digest": digest,
        "size_bytes": path.stat().st_size,
        "cache_hit": False,
        "upload_performed": True,
        "content_addressed_key": True,
        "remote_identity_verified": True,
        "full_byte_service_account_readback_passed": True,
        "remote_verified_at": "2026-08-28T22:00:00Z",
        "readback_digest": digest,
        "readback_size_bytes": path.stat().st_size,
        "raw_secret_values_recorded": False,
    }


def _build_scope(tmp_path: Path, *, kind: str) -> dict[str, Path]:
    controls_plans = tmp_path / "configured-controls-plans"
    controls_state = tmp_path / "configured-controls-state"
    controls_plans.mkdir()
    controls_state.mkdir()
    if kind == "launch":
        managed = tmp_path / "task-evaluation-launch-runs"
        scope = managed / "launch-1"
        job = scope / "allocator" / "scene-configuration-job"
        result_schema = "task_evaluation_scene_configuration_vast_result.v1"
        result_status = "blocked"
    else:
        managed = tmp_path / "scene-configuration-diagnostics"
        scope = managed / "attempt-1" / "job"
        job = scope
        result_schema = "task_evaluation_scene_configuration_diagnostic_vast_result.v1"
        result_status = "completed_diagnostic_only"
    execution = job / "immutable_execution"
    payload = execution / "stages" / "stage-1" / "configured_appearance.usdz"
    receipt = execution / "stages" / "stage-1" / "appearance_receipt.json"
    manifest = job / "artifact_manifest.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"large configured appearance")
    receipt.write_text("{}\n", encoding="utf-8")
    _write(
        manifest,
        {
            "schema_version": "task_evaluation_artifact_manifest.v1",
            "status": "completed",
            "blockers": [],
            "raw_secret_values_recorded": False,
        },
    )
    output_zip = job / "vast_provider_run" / "vast_provider_runtime_output.zip"
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(payload, "stages/stage-1/configured_appearance.usdz")
        archive.write(receipt, "stages/stage-1/appearance_receipt.json")
    source_commit = "a" * 40
    run_id = "scene-run-1"
    bundle: Path | None = None
    bundle_receipt: Path | None = None
    if kind == "diagnostic":
        bundle_root = job.parent / "bundle"
        bundle = (
            bundle_root
            / "task_evaluation_scene_configuration_provider_bundle.zip"
        )
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"exact provider bundle bytes")
        bundle_digest = _sha256(bundle)
        bundle_receipt = (
            bundle_root
            / "task_evaluation_scene_configuration_provider_bundle.v1.receipt.json"
        )
        _write(
            bundle_receipt,
            {
                "schema_version": (
                    "task_evaluation_scene_configuration_provider_bundle.v1"
                ),
                "status": "ready",
                "diagnostic_only": True,
                "qualification_eligible": False,
                "run_id": run_id,
                "source_commit": source_commit,
                "bundle_sha256": bundle_digest,
                "bundle_path": str(bundle),
                "bundle_size_bytes": bundle.stat().st_size,
            },
            digest_field="receipt_digest",
        )
    else:
        bundle_digest = "sha256:" + "b" * 64
    result_path = job / "task_evaluation_scene_configuration_vast_result.v1.json"
    _write(
        result_path,
        {
            "schema_version": result_schema,
            "status": result_status,
            "run_id": run_id,
            "source_commit": source_commit,
            "bundle_sha256": bundle_digest,
            "provider_runtime_output_zip_path": str(output_zip),
            "provider_runtime_output_zip_sha256": _sha256(output_zip),
            "continuing_spend_from_this_run": False,
            "runtime_secret_cleanup_completed": True,
            "raw_secret_values_recorded": False,
            "blockers": [] if kind == "diagnostic" else ["stage_blocked"],
        },
        digest_field="result_digest",
    )
    teardown = job / "vast_provider_run" / "vast_teardown_manifest.json"
    _write(
        teardown,
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "raw_secret_values_recorded": False,
        },
    )
    cleanup = job / "object_store_staging" / "wam_provider_object_store_cleanup.json"
    _write(
        cleanup,
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "blockers": [],
            "raw_secret_values_recorded": False,
        },
    )
    if kind == "launch":
        launch_receipt = scope / "launch_receipt.json"
        _write(
            launch_receipt,
            {
                "schema_version": "task_evaluation_launch_receipt.v1",
                "status": "blocked",
                "launch_id": scope.name,
                "run_id": scope.name,
                "request_digest": "sha256:" + "c" * 64,
                "raw_secret_values_recorded": False,
                "retain_processing_for_reconciliation": False,
            },
            digest_field="receipt_digest",
        )
        launch = json.loads(launch_receipt.read_text(encoding="utf-8"))
        _write(
            scope / "webapp_sync_succeeded.json",
            {
                "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
                "status": "succeeded",
                "launch_id": scope.name,
                "run_id": scope.name,
                "request_digest": launch["request_digest"],
                "receipt_digest": launch["receipt_digest"],
                "raw_secret_values_recorded": False,
            },
            digest_field="sync_result_digest",
        )
        _write(
            scope / "post_teardown_provider_zero_receipt.json",
            {
                "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
                "status": "provider_zero_confirmed",
                "launch_id": scope.name,
                "run_id": scope.name,
                "request_digest": launch["request_digest"],
                "receipt_digest": launch["receipt_digest"],
                "provider_zero_verified": True,
                "continuing_spend_from_this_run": False,
                "allocator_invoked": False,
                "provider_mutation_performed": False,
                "automatic_retry_performed": False,
                "blockers": [],
                "raw_secret_values_recorded": False,
            },
            digest_field="provider_zero_receipt_digest",
        )
    else:
        _write(
            job / "scene_artifact_lease.v1.json",
            {
                "schema_version": "task_evaluation_scene_artifact_lease.v1",
                "run_id": run_id,
                "lifecycle_state": "completed",
                "expires_at": None,
                "artifact_references": [
                    {
                        "artifact_kind": "provider-output",
                        "digest": _sha256(output_zip),
                        "size_bytes": output_zip.stat().st_size,
                        "uri": "s3://legacy/object",
                    }
                ],
                "raw_secret_values_recorded": False,
            },
            digest_field="lease_digest",
        )
    return {
        "managed": managed,
        "scope": scope,
        "job": job,
        "payload": payload,
        "receipt": receipt,
        "manifest": manifest,
        "output_zip": output_zip,
        "result": result_path,
        "controls_plans": controls_plans,
        "controls_state": controls_state,
        "bundle": bundle,
        "bundle_receipt": bundle_receipt,
    }


def _archive(scope: dict[str, Path], *, kind: str, bucket: str = EXPECTED_BUCKET) -> Path:
    index_path = scope["job"] / "scene_artifact_b2_retention_index.v1.json"

    def publish(*, path: Path, artifact_kind: str) -> dict[str, object]:
        expected = {
            "provider-output": scope["output_zip"],
            "provider-bundle": scope["bundle"],
        }
        assert path == expected[artifact_kind]
        return _remote_reference(
            path, artifact_kind=artifact_kind, bucket=bucket
        )

    archive_terminal_scene_payload_to_b2(
        scope_root=scope["scope"],
        scope_kind=kind,
        managed_root=scope["managed"],
        expected_bucket=EXPECTED_BUCKET,
        index_destination=index_path,
        publisher=publish,
        configured_controls_plan_root=scope["controls_plans"],
        configured_controls_progression_root=scope["controls_state"],
    )
    return index_path


@pytest.mark.parametrize("kind", ["launch", "diagnostic"])
def test_terminal_scope_archives_exact_output_to_expected_b2_bucket(
    tmp_path: Path, kind: str
) -> None:
    scope = _build_scope(tmp_path, kind=kind)

    index_path = _archive(scope, kind=kind)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["status"] == "completed"
    references = {
        reference["artifact_kind"]: reference
        for reference in index["artifact_references"]
    }
    assert set(references) == (
        {"provider-output", "provider-bundle"}
        if kind == "diagnostic"
        else {"provider-output"}
    )
    assert references["provider-output"]["uri"].startswith(
        f"s3://{EXPECTED_BUCKET}/"
    )
    assert references["provider-output"]["digest"] == _sha256(scope["output_zip"])
    if kind == "diagnostic":
        assert references["provider-bundle"]["digest"] == _sha256(scope["bundle"])
    assert index_path.stat().st_mode & 0o777 == 0o440


def test_launch_plan_reclaims_only_archive_backed_binary_payloads(
    tmp_path: Path,
) -> None:
    scope = _build_scope(tmp_path, kind="launch")
    index = _archive(scope, kind="launch")

    plan = plan_terminal_scene_payload_retention(
        scope_root=scope["scope"],
        scope_kind="launch",
        managed_root=scope["managed"],
        expected_bucket=EXPECTED_BUCKET,
        b2_index_path=index,
        configured_controls_plan_root=scope["controls_plans"],
        configured_controls_progression_root=scope["controls_state"],
    )

    assert plan["status"] == "completed"
    assert {Path(row["local_path"]) for row in plan["candidates"]} == {
        scope["payload"],
        scope["output_zip"],
    }
    assert scope["receipt"].is_file()
    assert scope["manifest"].is_file()
    assert plan["lifecycle_proof"]["provider_zero_verified"] is True
    assert plan["lifecycle_proof"]["reconciled"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_sync", "terminal_scene_payload_unreconciled"),
        ("provider_nonzero", "terminal_scene_payload_provider_nonzero"),
    ],
)
def test_launch_retention_refuses_unreconciled_or_provider_nonzero(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    scope = _build_scope(tmp_path, kind="launch")
    if mutation == "missing_sync":
        (scope["scope"] / "webapp_sync_succeeded.json").unlink()
    else:
        path = scope["scope"] / "post_teardown_provider_zero_receipt.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["provider_zero_verified"] = False
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")

    with pytest.raises(TaskEvaluationTerminalScenePayloadRetentionError, match=expected):
        archive_terminal_scene_payload_to_b2(
            scope_root=scope["scope"],
            scope_kind="launch",
            managed_root=scope["managed"],
            expected_bucket=EXPECTED_BUCKET,
            index_destination=scope["job"] / "b2-index.json",
            publisher=lambda **_: _remote_reference(scope["output_zip"]),
            configured_controls_plan_root=scope["controls_plans"],
            configured_controls_progression_root=scope["controls_state"],
        )


def test_diagnostic_retention_refuses_active_lease(tmp_path: Path) -> None:
    scope = _build_scope(tmp_path, kind="diagnostic")
    lease_path = scope["job"] / "scene_artifact_lease.v1.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["lifecycle_state"] = "in_flight"
    lease["lease_digest"] = canonical_digest(lease, digest_field="lease_digest")
    lease_path.write_text(canonical_json(lease) + "\n", encoding="utf-8")

    with pytest.raises(
        TaskEvaluationTerminalScenePayloadRetentionError,
        match="terminal_scene_payload_scope_active",
    ):
        _archive(scope, kind="diagnostic")


def test_enabled_configured_controls_plan_pins_source_launch(tmp_path: Path) -> None:
    scope = _build_scope(tmp_path, kind="launch")
    _write(
        scope["controls_plans"] / "scene-839873.json",
        {
            "schema_version": "task_evaluation_configured_controls_progression_plan.v1",
            "enabled": True,
            "source_launch_id": scope["scope"].name,
        },
        digest_field="plan_digest",
    )

    with pytest.raises(
        TaskEvaluationTerminalScenePayloadRetentionError,
        match="terminal_scene_payload_scope_active",
    ):
        _archive(scope, kind="launch")


def test_incomplete_configured_controls_state_pins_source_launch(tmp_path: Path) -> None:
    scope = _build_scope(tmp_path, kind="launch")
    state = scope["controls_state"] / scope["scope"].name
    state.mkdir()
    _write(
        state / "configured_controls_progression.v1.json",
        {
            "schema_version": "task_evaluation_configured_controls_progression.v1",
            "status": "episode_preparation_queued",
        },
        digest_field="progression_digest",
    )

    with pytest.raises(
        TaskEvaluationTerminalScenePayloadRetentionError,
        match="terminal_scene_payload_scope_active",
    ):
        _archive(scope, kind="launch")


def test_b2_bucket_or_readback_mismatch_refuses_retention(tmp_path: Path) -> None:
    scope = _build_scope(tmp_path, kind="launch")

    with pytest.raises(
        TaskEvaluationTerminalScenePayloadRetentionError,
        match="terminal_scene_payload_b2_reference_invalid",
    ):
        _archive(scope, kind="launch", bucket="wrong-bucket")


def test_apply_rechecks_every_candidate_and_preserves_receipts(tmp_path: Path) -> None:
    scope = _build_scope(tmp_path, kind="diagnostic")
    index = _archive(scope, kind="diagnostic")
    plan = plan_terminal_scene_payload_retention(
        scope_root=scope["scope"],
        scope_kind="diagnostic",
        managed_root=scope["managed"],
        expected_bucket=EXPECTED_BUCKET,
        b2_index_path=index,
        configured_controls_plan_root=scope["controls_plans"],
        configured_controls_progression_root=scope["controls_state"],
    )
    scope["payload"].write_bytes(b"changed after plan")

    with pytest.raises(
        TaskEvaluationTerminalScenePayloadRetentionError,
        match="terminal_scene_payload_candidate_changed",
    ):
        apply_terminal_scene_payload_retention(
            plan=plan,
            acknowledgement=APPLY_ACK,
            expected_bucket=EXPECTED_BUCKET,
        )
    assert scope["output_zip"].is_file()
    assert scope["bundle"].is_file()
    assert scope["receipt"].is_file()


def test_apply_removes_only_exact_binary_payload_and_archive(tmp_path: Path) -> None:
    scope = _build_scope(tmp_path, kind="diagnostic")
    index = _archive(scope, kind="diagnostic")
    plan = plan_terminal_scene_payload_retention(
        scope_root=scope["scope"],
        scope_kind="diagnostic",
        managed_root=scope["managed"],
        expected_bucket=EXPECTED_BUCKET,
        b2_index_path=index,
        configured_controls_plan_root=scope["controls_plans"],
        configured_controls_progression_root=scope["controls_state"],
    )

    result = apply_terminal_scene_payload_retention(
        plan=plan,
        acknowledgement=APPLY_ACK,
        expected_bucket=EXPECTED_BUCKET,
    )

    assert result["status"] == "completed"
    assert result["removed_count"] == 3
    assert not scope["payload"].exists()
    assert not scope["output_zip"].exists()
    assert not scope["bundle"].exists()
    assert scope["bundle_receipt"].is_file()
    assert scope["receipt"].is_file()
    assert scope["manifest"].is_file()
    assert index.is_file()


def test_completed_diagnostic_can_reclaim_exact_bundle_when_output_zip_is_absent(
    tmp_path: Path,
) -> None:
    scope = _build_scope(tmp_path, kind="diagnostic")
    scope["output_zip"].unlink()
    index = _archive(scope, kind="diagnostic")

    plan = plan_terminal_scene_payload_retention(
        scope_root=scope["scope"],
        scope_kind="diagnostic",
        managed_root=scope["managed"],
        expected_bucket=EXPECTED_BUCKET,
        b2_index_path=index,
        configured_controls_plan_root=scope["controls_plans"],
        configured_controls_progression_root=scope["controls_state"],
    )

    assert plan["candidate_count"] == 1
    assert plan["candidates"][0]["local_path"] == str(scope["bundle"])
    assert plan["candidates"][0]["artifact_kind"] == "provider-bundle"
    assert scope["bundle_receipt"].is_file()
