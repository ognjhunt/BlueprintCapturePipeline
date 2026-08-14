from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import task_evaluation_launch_dispatcher as dispatcher
from blueprint_pipeline.semantic_teacher_image_edit_bundle import (
    BUNDLE_RECEIPT_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_semantic_teacher_image_edit_live_profile",
    REPO_ROOT / "scripts" / "build_semantic_teacher_image_edit_live_profile.py",
)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)

COMMIT = "a" * 40
IMAGE = "registry.example/semantic-teacher@sha256:" + "b" * 64


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    bundle = tmp_path / "semantic-teacher.zip"
    bundle.write_bytes(b"semantic-teacher-bundle")
    bundle_digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt = {
        "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "completed_no_upload_no_inference",
        "source_commit_sha": COMMIT,
        "bundle": {
            "path": str(bundle),
            "size_bytes": bundle.stat().st_size,
            "sha256": bundle_digest,
        },
        "backend_entry_digest": "sha256:" + "c" * 64,
        "task_count": 2,
        "camera_count": 16,
        "provider_mutations_performed": 0,
        "secret_values_stored": False,
        "raw_nonredistributable_source_bytes_included": False,
        "rehearsal": {
            "status": "passed",
            "token_lookup_performed": False,
            "upload_performed": False,
            "provider_mutations_performed": 0,
        },
    }
    receipt_path = tmp_path / "bundle-receipt.json"
    _write(receipt_path, receipt)
    prior_spend = tmp_path / "prior-spend.json"
    prior_spend.write_text(
        json.dumps({"schema_version": "fixture_prior_spend.v1"}) + "\n",
        encoding="utf-8",
    )
    authority = {
        "authorization_digest": "sha256:" + "d" * 64,
        "runtime_image_identity": IMAGE,
        "backend_entry_digest": receipt["backend_entry_digest"],
        "task_count": 2,
        "camera_count": 16,
        "maximum_hourly_rate_usd": 0.5,
        "hard_total_spend_cap_usd": 1.0,
        "hard_ttl_seconds": 600,
        "prior_spend_reconciliation": {
            "path": str(prior_spend),
            "size_bytes": prior_spend.stat().st_size,
            "sha256": "sha256:"
            + hashlib.sha256(prior_spend.read_bytes()).hexdigest(),
        },
    }
    authority_path = tmp_path / "authority.json"
    _write(authority_path, authority)
    dry_run = {
        "schema_version": "semantic_teacher_image_edit_allocator_dry_run.v1",
        "status": "dry_run_ready",
        "source_commit_sha": COMMIT,
        "authorization_digest": authority["authorization_digest"],
        "bundle_sha256": bundle_digest,
        "bundle_size_bytes": bundle.stat().st_size,
        "backend_entry_digest": receipt["backend_entry_digest"],
        "task_count": 2,
        "camera_count": 16,
        "maximum_provider_allocations": 1,
        "automatic_retry_count": 0,
        "provider_inventory_api_zero": True,
        "provider_mutations_performed": 0,
        "dry_run_digest": "",
    }
    dry_run["dry_run_digest"] = canonical_digest(
        dry_run, digest_field="dry_run_digest"
    )
    dry_run_path = tmp_path / "dry-run.json"
    _write(dry_run_path, dry_run)
    token = tmp_path / "token"
    token.write_text("fixture-secret", encoding="utf-8")
    token.chmod(0o600)
    return {
        "receipt": receipt_path,
        "authority": authority_path,
        "dry_run": dry_run_path,
        "token": token,
        "prior_spend": prior_spend,
    }


def _build(tmp_path: Path, monkeypatch):
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        builder,
        "validate_semantic_teacher_image_edit_paid_authority",
        lambda authority, **_kwargs: authority,
    )
    profile = builder.build_semantic_teacher_image_edit_live_profile(
        bundle_receipt_path=paths["receipt"],
        attempt_authority_path=paths["authority"],
        dry_run_receipt_path=paths["dry_run"],
        token_file_path=paths["token"],
        source_commit=COMMIT,
        raw_manifest_uri=f"https://example.invalid/{COMMIT}/semantic-teacher.json",
    )
    return profile, paths


def test_profile_uses_admitted_dispatch_and_binds_private_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    profile, paths = _build(tmp_path, monkeypatch)
    argv = profile["allocator"]["argv"]
    assert profile["allocator"]["retry_cap"] == 0
    assert "--probe-kind" in argv
    assert "semantic-teacher-image-edit" in argv
    assert "--execute" not in argv
    assert "--semantic-teacher-dry-run-receipt" in argv
    assert "--semantic-teacher-dry-run-output" in argv
    assert "--semantic-teacher-runtime-image-identity" in argv
    assert IMAGE in argv
    immutable = {row["name"] for row in profile["immutable_inputs"]}
    assert "semantic_teacher_paid_attempt_authority" in immutable
    assert "semantic_teacher_prior_spend_reconciliation" in immutable
    assert "evaluation_run_spec" in immutable
    assert all(str(paths["token"]) != row["path"] for row in profile["immutable_inputs"])
    prior = next(
        row
        for row in profile["immutable_inputs"]
        if row["name"] == "semantic_teacher_prior_spend_reconciliation"
    )
    assert prior["path"] == str(paths["prior_spend"])
    paths["prior_spend"].write_text("changed\n", encoding="utf-8")
    assert (
        "launch_profile_immutable_input_digest_mismatch:"
        "semantic_teacher_prior_spend_reconciliation"
        in dispatcher.verify_profile_immutable_inputs(profile)
    )
    assert profile["terminal_contract"]["required_values"] == {
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
    }


def test_semantic_terminal_result_satisfies_shared_dispatcher_contract(
    tmp_path: Path, monkeypatch
) -> None:
    profile, _paths = _build(tmp_path, monkeypatch)
    run_root = tmp_path / "launch-run"
    allocator = run_root / "allocator"
    allocator.mkdir(parents=True)
    artifact = allocator / "artifact-manifest.json"
    teardown = allocator / "teardown-manifest.json"
    artifact.write_text("{}\n", encoding="utf-8")
    teardown.write_text("{}\n", encoding="utf-8")
    result_path = allocator / "result.json"
    _write(
        result_path,
        {
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "retry_cap": 0,
            "artifact_manifest_path": str(artifact),
            "teardown_manifest_path": str(teardown),
        },
    )
    terminal = dispatcher._terminal_evidence(profile, execute=True, run_root=run_root)
    assert terminal["status"] == "passed"
    assert terminal["blockers"] == []

    changed = json.loads(result_path.read_text(encoding="utf-8"))
    changed.pop("retry_cap")
    _write(result_path, changed)
    refused = dispatcher._terminal_evidence(profile, execute=True, run_root=run_root)
    assert "allocator_terminal_value_mismatch:retry_cap" in refused["blockers"]
