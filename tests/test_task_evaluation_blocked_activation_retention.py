from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_blocked_activation_retention import (
    APPLY_ACKNOWLEDGEMENT,
    BlockedActivationRetentionError,
    apply_blocked_activation_retention_plan,
    build_blocked_activation_retention_plan,
)
from blueprint_pipeline.task_evaluation_launch_activation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    BUNDLE_SCHEMA_VERSION,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed(path: Path, value: dict, *, digest_field: str) -> dict:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _fixture(tmp_path: Path) -> dict:
    state = tmp_path / "state"
    queue = state / "task-evaluation-launch-activations"
    for child in ("pending", "processing", "prepared", "blocked", "results"):
        (queue / child).mkdir(parents=True, exist_ok=True)
    for name in ("launches", "terminal"):
        (state / name).mkdir()
    profiles = tmp_path / "profiles"
    standing = tmp_path / "standing"
    profiles.mkdir()
    standing.mkdir()
    catalog = state / "catalog.json"
    catalog.write_text('{"profiles":[]}\n', encoding="utf-8")

    activation_base = tmp_path / "activations"
    activation_id = "scene-839873-activation-a"
    activation = activation_base / activation_id
    bundle_root = activation / "launch-set" / "bundle"
    stage = bundle_root / "stage"
    (stage / "provider_runtime").mkdir(parents=True)
    (stage / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (stage / "run.sh").chmod(0o555)
    (stage / "provider_runtime" / "payload.bin").write_bytes(b"provider payload")
    bundle = bundle_root / "task_evaluation_scene_configuration_provider_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
    receipt_path = bundle_root / f"{BUNDLE_SCHEMA_VERSION}.receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    receipt = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "status": "ready",
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_size_bytes": bundle.stat().st_size,
        "nested_provider_mutations_performed": 0,
        "evaluation_episode_executed": False,
    }

    (activation / "references").mkdir(parents=True)
    (activation / "references" / "release-window.json").write_text(
        '{"window":"preserved"}\n', encoding="utf-8"
    )
    (activation / "activation-context.json").write_text(
        '{"context":"preserved"}\n', encoding="utf-8"
    )
    launch_set = activation / "launch-set"
    log_rows = []
    for step_id, produces in (
        ("provider_bundle", receipt_path),
        ("immutable_manifest", launch_set / "manifest_publication_receipt.v1.json"),
    ):
        stdout = produces.with_name(f"{produces.name}.{step_id}.stdout.log")
        stderr = produces.with_name(f"{produces.name}.{step_id}.stderr.log")
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout.write_text(f"{step_id} preserved stdout\n", encoding="utf-8")
        stderr.write_text(f"{step_id} preserved stderr\n", encoding="utf-8")
        log_rows.append(
            {
                "step_id": step_id,
                "stdout_path": str(stdout),
                "stdout_sha256": _sha256(stdout),
                "stderr_path": str(stderr),
                "stderr_sha256": _sha256(stderr),
                "credential_redaction_applied": True,
            }
        )
    preparation_path = activation / "paid_lane_launch_preparation.v1.json"
    preparation_path.write_text(
        json.dumps(
            {
                "schema_version": "paid_lane_launch_preparation.v1",
                "lane": "task_evaluation_scene_configuration",
                "status": "blocked",
                "completed_steps": [
                    {
                        "step_id": "provider_bundle",
                        "artifact_path": str(receipt_path),
                        "artifact_sha256": _sha256(receipt_path),
                    }
                ],
                "step_logs": log_rows,
                "blockers": ["immutable_manifest:exit_2"],
                "provider_allocation_performed": False,
                "paid_inference_performed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    request_digest = "sha256:" + "a" * 64
    filename = f"{activation_id}-{request_digest.removeprefix('sha256:')}.json"
    envelope = _write_sealed(
        queue / "blocked" / filename,
        {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "request_digest": request_digest,
            "request": {"activation_id": activation_id},
            "provider_mutation_performed_inside_intake": False,
            "paid_execution_requested": False,
            "envelope_digest": "",
        },
        digest_field="envelope_digest",
    )
    result = _write_sealed(
        queue / "results" / filename,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "activation_id": activation_id,
            "blockers": ["launch_activation_preparation_graph_blocked"],
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "result_digest": "",
        },
        digest_field="result_digest",
    )

    def validator(path: Path):
        assert path == receipt_path
        return receipt

    return {
        "activation_root": activation,
        "activation_base_root": activation_base,
        "state_root": state,
        "activation_queue_root": queue,
        "profile_dir": profiles,
        "public_catalog": catalog,
        "standing_authorization_dir": standing,
        "live_reference_roots": (state / "launches", state / "terminal"),
        "bundle_validator": validator,
        "bundle": bundle,
        "stage": stage,
        "receipt": receipt_path,
        "preparation": preparation_path,
        "envelope": queue / "blocked" / filename,
        "result": queue / "results" / filename,
        "envelope_digest": envelope["envelope_digest"],
        "result_digest": result["result_digest"],
    }


def _complete_immutable_manifest(fixture: dict) -> Path:
    launch_set = fixture["activation_root"] / "launch-set"
    manifest = launch_set / "manifest_publication_receipt.v1.json"
    manifest.write_text(
        '{"schema_version":"immutable_manifest_publication.v1","status":"published"}\n',
        encoding="utf-8",
    )
    preparation = json.loads(fixture["preparation"].read_text(encoding="utf-8"))
    preparation["completed_steps"].append(
        {
            "step_id": "immutable_manifest",
            "artifact_path": str(manifest),
            "artifact_sha256": _sha256(manifest),
        }
    )
    authority = launch_set / "task_evaluation_scene_configuration_paid_authority.v1.json"
    stdout = authority.with_name(f"{authority.name}.paid_authority.stdout.log")
    stderr = authority.with_name(f"{authority.name}.paid_authority.stderr.log")
    stdout.write_text("paid authority preserved redacted stdout\n", encoding="utf-8")
    stderr.write_text("paid authority preserved redacted stderr\n", encoding="utf-8")
    preparation["step_logs"].append(
        {
            "step_id": "paid_authority",
            "stdout_path": str(stdout),
            "stdout_sha256": _sha256(stdout),
            "stderr_path": str(stderr),
            "stderr_sha256": _sha256(stderr),
            "credential_redaction_applied": True,
        }
    )
    preparation["blockers"] = ["paid_authority:exit_1"]
    fixture["preparation"].write_text(
        json.dumps(preparation, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def test_plan_apply_removes_only_exact_rebuildables_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    plan = build_blocked_activation_retention_plan(
        **{
            key: fixture[key]
            for key in (
                "activation_root",
                "activation_base_root",
                "state_root",
                "activation_queue_root",
                "profile_dir",
                "public_catalog",
                "standing_authorization_dir",
                "live_reference_roots",
                "bundle_validator",
            )
        }
    )
    assert plan["completed_preparation_steps"] == ["provider_bundle"]
    assert plan["terminal_envelope_digest"] == fixture["envelope_digest"]
    assert plan["terminal_result_digest"] == fixture["result_digest"]
    assert plan["removable_stage_tree"]["archive_byte_identity_proven"] is True
    plan_path = tmp_path / "retention-plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")

    applied = apply_blocked_activation_retention_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=tmp_path / "retention-receipt.json",
        bundle_validator=fixture["bundle_validator"],
    )

    assert applied["removed_bytes"] == plan["predicted_removed_bytes"]
    assert not fixture["bundle"].exists()
    assert not fixture["stage"].exists()
    for key in ("receipt", "preparation", "envelope", "result"):
        assert fixture[key].is_file()
    assert (fixture["activation_root"] / "references/release-window.json").is_file()
    assert (
        fixture["activation_root"]
        / "launch-set/bundle/task_evaluation_scene_configuration_provider_bundle.v1.receipt.json.provider_bundle.stdout.log"
    ).is_file()
    assert applied["evidence_artifacts_removed"] is False


def test_two_step_pre_authority_prefix_and_failed_authority_logs_are_preserved(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = _complete_immutable_manifest(fixture)
    keys = (
        "activation_root",
        "activation_base_root",
        "state_root",
        "activation_queue_root",
        "profile_dir",
        "public_catalog",
        "standing_authorization_dir",
        "live_reference_roots",
        "bundle_validator",
    )
    plan = build_blocked_activation_retention_plan(**{key: fixture[key] for key in keys})

    assert plan["completed_preparation_steps"] == [
        "provider_bundle",
        "immutable_manifest",
    ]
    assert plan["immutable_manifest"]["sha256"] == _sha256(manifest)
    assert [row["step_id"] for row in plan["step_logs"]] == [
        "provider_bundle",
        "immutable_manifest",
        "paid_authority",
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    applied = apply_blocked_activation_retention_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=tmp_path / "apply.json",
        bundle_validator=fixture["bundle_validator"],
    )

    assert applied["status"] == "applied"
    assert manifest.is_file()
    for record in plan["step_logs"]:
        assert Path(record["stdout"]["path"]).is_file()
        assert Path(record["stderr"]["path"]).is_file()


def test_plan_refuses_provider_mutation_claim(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    value = json.loads(fixture["result"].read_text(encoding="utf-8"))
    value["provider_mutation_performed"] = True
    _write_sealed(fixture["result"], value, digest_field="result_digest")

    with pytest.raises(
        BlockedActivationRetentionError,
        match="blocked_activation_retention_terminal_claim_invalid",
    ):
        build_blocked_activation_retention_plan(
            **{
                key: fixture[key]
                for key in (
                    "activation_root",
                    "activation_base_root",
                    "state_root",
                    "activation_queue_root",
                    "profile_dir",
                    "public_catalog",
                    "standing_authorization_dir",
                    "live_reference_roots",
                    "bundle_validator",
                )
            }
        )


def test_plan_refuses_paid_authority_or_live_reference(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["activation_root"] / "launch-set/task_evaluation_paid_authority.v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(
        BlockedActivationRetentionError,
        match="authority_or_execution_artifact_present",
    ):
        build_blocked_activation_retention_plan(
            **{
                key: fixture[key]
                for key in (
                    "activation_root",
                    "activation_base_root",
                    "state_root",
                    "activation_queue_root",
                    "profile_dir",
                    "public_catalog",
                    "standing_authorization_dir",
                    "live_reference_roots",
                    "bundle_validator",
                )
            }
        )

    (fixture["activation_root"] / "launch-set/task_evaluation_paid_authority.v1.json").unlink()
    (fixture["live_reference_roots"][0] / "request.json").write_text(
        json.dumps({"activation_id": fixture["activation_root"].name}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        BlockedActivationRetentionError,
        match="blocked_activation_retention_live_reference_present",
    ):
        build_blocked_activation_retention_plan(
            **{
                key: fixture[key]
                for key in (
                    "activation_root",
                    "activation_base_root",
                    "state_root",
                    "activation_queue_root",
                    "profile_dir",
                    "public_catalog",
                    "standing_authorization_dir",
                    "live_reference_roots",
                    "bundle_validator",
                )
            }
        )


def test_plan_refuses_stage_archive_mismatch_or_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["stage"] / "provider_runtime/payload.bin").write_bytes(b"changed")
    with pytest.raises(
        BlockedActivationRetentionError,
        match="blocked_activation_retention_stage_archive_mismatch",
    ):
        build_blocked_activation_retention_plan(
            **{
                key: fixture[key]
                for key in (
                    "activation_root",
                    "activation_base_root",
                    "state_root",
                    "activation_queue_root",
                    "profile_dir",
                    "public_catalog",
                    "standing_authorization_dir",
                    "live_reference_roots",
                    "bundle_validator",
                )
            }
        )

    fixture = _fixture(tmp_path / "symlink")
    os.symlink(fixture["receipt"], fixture["stage"] / "unsafe-link")
    with pytest.raises(
        BlockedActivationRetentionError,
        match="blocked_activation_retention_stage_tree_symlink",
    ):
        build_blocked_activation_retention_plan(
            **{
                key: fixture[key]
                for key in (
                    "activation_root",
                    "activation_base_root",
                    "state_root",
                    "activation_queue_root",
                    "profile_dir",
                    "public_catalog",
                    "standing_authorization_dir",
                    "live_reference_roots",
                    "bundle_validator",
                )
            }
        )


def test_apply_revalidates_under_lock_and_refuses_changed_plan(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    keys = (
        "activation_root",
        "activation_base_root",
        "state_root",
        "activation_queue_root",
        "profile_dir",
        "public_catalog",
        "standing_authorization_dir",
        "live_reference_roots",
        "bundle_validator",
    )
    plan = build_blocked_activation_retention_plan(**{key: fixture[key] for key in keys})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    fixture["public_catalog"].write_text('{"profiles":[],"changed":true}\n', encoding="utf-8")

    with pytest.raises(
        BlockedActivationRetentionError,
        match="blocked_activation_retention_plan_changed",
    ):
        apply_blocked_activation_retention_plan(
            dry_run_plan_path=plan_path,
            acknowledgement=APPLY_ACKNOWLEDGEMENT,
            receipt_out=tmp_path / "receipt.json",
            bundle_validator=fixture["bundle_validator"],
        )
    assert fixture["bundle"].is_file()
    assert fixture["stage"].is_dir()
