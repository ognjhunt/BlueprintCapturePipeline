from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_gpu_admission import REQUEST_SCHEMA_VERSION
from blueprint_pipeline.sam31_paid_attempt_authority import (
    consume_sam31_paid_attempt_authority_once,
    materialize_sam31_paid_attempt_authority,
    validate_sam31_paid_attempt_authority,
)
from blueprint_pipeline.sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION


COMMIT = "a" * 40
SHA = "sha256:" + "b" * 64


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    bundle = tmp_path / "input.zip"
    bundle.write_bytes(b"deterministic-sam31-bundle")
    import hashlib

    bundle_digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_commit_sha": COMMIT,
        "worker_image_digest": "registry.example/sam31@" + SHA,
        "input_bundle_digest": bundle_digest,
        "input_bundle_size_bytes": bundle.stat().st_size,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
        "authority_id": "goal-authority-1",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt = {
        "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle": {
            "filename": bundle.name,
            "sha256": bundle_digest,
            "size_bytes": bundle.stat().st_size,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return request_path, bundle, receipt_path, request, receipt


def _authority(tmp_path: Path) -> tuple[dict, Path, Path, dict, dict]:
    request_path, bundle, receipt_path, request, receipt = _inputs(tmp_path)
    output = tmp_path / "authority.json"
    authority = materialize_sam31_paid_attempt_authority(
        request_path=request_path,
        bundle_path=bundle,
        bundle_receipt_path=receipt_path,
        authorization_reference="User directed one bounded SAM source-track run",
        authorized_by="fixture-user",
        authorized_on="2026-08-13",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=1.0,
        hard_ttl_seconds=600,
        aggregate_goal_spend_before_attempt_usd=0.0,
        aggregate_goal_spend_cap_usd=12.0,
        output_path=output,
    )
    return authority, bundle, output, request, receipt


def test_authority_binds_request_bundle_budget_and_zero_retry(tmp_path: Path) -> None:
    authority, bundle, _output, request, receipt = _authority(tmp_path)
    validated = validate_sam31_paid_attempt_authority(
        authority,
        request=request,
        bundle_path=bundle,
        bundle_receipt=receipt,
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=1.0,
        hard_ttl_seconds=600,
    )
    assert validated["maximum_provider_allocations"] == 1
    assert validated["maximum_automatic_retries"] == 0
    assert validated["automatic_paid_retry_authorized"] is False
    assert validated["bundle"]["sha256"] == request["input_bundle_digest"]


def test_authority_rejects_self_digest_valid_bundle_tamper(tmp_path: Path) -> None:
    authority, bundle, _output, request, receipt = _authority(tmp_path)
    tampered = json.loads(json.dumps(authority))
    tampered["bundle"]["sha256"] = SHA
    tampered["authorization_digest"] = canonical_digest(
        tampered, digest_field="authorization_digest"
    )
    with pytest.raises(ValueError, match="sam31_paid_authority_invalid"):
        validate_sam31_paid_attempt_authority(
            tampered,
            request=request,
            bundle_path=bundle,
            bundle_receipt=receipt,
            blueprint_commit=COMMIT,
            max_hourly_rate_usd=0.5,
            hard_cap_usd=1.0,
            hard_ttl_seconds=600,
        )


def test_authority_is_atomically_single_use(tmp_path: Path, monkeypatch) -> None:
    authority, _bundle, _output, _request, _receipt = _authority(tmp_path)
    ledger = tmp_path / "ledger" / "consumed"

    def prepare_ledger() -> Path:
        ledger.mkdir(mode=0o700, parents=True, exist_ok=True)
        return ledger

    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_attempt_authority.prepare_consumption_root",
        prepare_ledger,
    )
    first = consume_sam31_paid_attempt_authority_once(authority, blueprint_commit=COMMIT)
    second = consume_sam31_paid_attempt_authority_once(authority, blueprint_commit=COMMIT)
    assert first["status"] == "consumed"
    assert second == {"status": "blocked", "blockers": ["sam31_paid_authority_consumed"]}
    records = list(ledger.glob("sam31-*.json"))
    assert len(records) == 1
    assert records[0].stat().st_mode & 0o077 == 0


def test_nonzero_prior_spend_requires_digest_bound_reconciliation(tmp_path: Path) -> None:
    request_path, bundle, receipt_path, _request, _receipt = _inputs(tmp_path)
    with pytest.raises(ValueError, match="sam31_paid_authority_configuration_invalid"):
        materialize_sam31_paid_attempt_authority(
            request_path=request_path,
            bundle_path=bundle,
            bundle_receipt_path=receipt_path,
            authorization_reference="goal",
            authorized_by="fixture",
            authorized_on="2026-08-13",
            blueprint_commit=COMMIT,
            max_hourly_rate_usd=0.5,
            hard_cap_usd=1.0,
            hard_ttl_seconds=600,
            aggregate_goal_spend_before_attempt_usd=0.1,
            aggregate_goal_spend_cap_usd=12.0,
            output_path=tmp_path / "authority.json",
        )
