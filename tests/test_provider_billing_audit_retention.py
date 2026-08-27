import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.provider_billing_audit_retention import (
    APPLY_ACKNOWLEDGEMENT,
    ProviderBillingAuditRetentionError,
    apply_provider_billing_audit_retention_plan,
    build_provider_billing_audit_retention_plan,
)
from blueprint_pipeline.vast_official_billing_extractor import (
    _load_vast_responses,
    _validate_source_receipt,
)


def _canonical_digest(value: dict, *, digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field, None)
    payload = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_old_audit(
    audit_root: Path, *, timestamp: str, response: bytes
) -> tuple[Path, Path, bytes]:
    directory = audit_root / timestamp
    directory.mkdir(mode=0o700)
    response_path = directory / "response-001-vast.json"
    response_path.write_bytes(response)
    response_path.chmod(0o600)
    receipt = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "generated_at": "2026-08-27T00:00:00+00:00",
        "cohort_start_at": "2026-01-01T00:00:00+00:00",
        "cohort_end_at": "2026-08-27T00:00:00+00:00",
        "provider_totals_usd": {"vast": 1.0},
        "required_provider_ids": ["vast"],
        "covered_provider_ids": ["vast"],
        "uncovered_provider_ids": [],
        "optional_provider_failures": {},
        "sources": [
            {
                "provider": "vast",
                "endpoint": "https://console.vast.ai/api/v0/charges/",
                "request_query_digest": "sha256:" + "0" * 64,
                "response_digest": "sha256:" + hashlib.sha256(response).hexdigest(),
                "response_size_bytes": len(response),
                "retained_path": str(response_path.resolve()),
            }
        ],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = _canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = directory / "provider_billing_source_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    receipt_path.chmod(0o600)
    return receipt_path, response_path, receipt_path.read_bytes()


def _fixture(tmp_path: Path) -> dict[str, object]:
    audit_root = tmp_path / "billing-audit"
    audit_root.mkdir(mode=0o700)
    response = json.dumps(
        {"success": True, "results": [{"amount": 1.0}], "next_token": None}
    ).encode()
    first = _write_old_audit(
        audit_root,
        timestamp="20260827T000000.000001Z",
        response=response,
    )
    second = _write_old_audit(
        audit_root,
        timestamp="20260827T001000.000001Z",
        response=response,
    )
    return {
        "audit_root": audit_root,
        "response": response,
        "first": first,
        "second": second,
    }


def test_dry_run_is_non_mutating_and_reports_exact_duplicate_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit_root = fixture["audit_root"]
    assert isinstance(audit_root, Path)
    before = sorted(str(path.relative_to(audit_root)) for path in audit_root.rglob("*"))

    plan = build_provider_billing_audit_retention_plan(audit_root=audit_root)

    assert plan["status"] == "dry_run"
    assert plan["receipt_count"] == 2
    assert plan["response_path_count"] == 2
    assert plan["predicted_relinked_bytes"] == len(fixture["response"])
    assert plan["production_artifact_mutation_performed"] is False
    assert sorted(str(path.relative_to(audit_root)) for path in audit_root.rglob("*")) == before
    assert not (audit_root / "objects").exists()


def test_apply_preserves_receipt_bytes_and_paths_and_passes_vast_validator(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit_root = fixture["audit_root"]
    assert isinstance(audit_root, Path)
    first_receipt, first_response, first_receipt_bytes = fixture["first"]
    second_receipt, second_response, second_receipt_bytes = fixture["second"]
    assert isinstance(first_receipt, Path)
    assert isinstance(first_response, Path)
    assert isinstance(first_receipt_bytes, bytes)
    assert isinstance(second_receipt, Path)
    assert isinstance(second_response, Path)
    assert isinstance(second_receipt_bytes, bytes)
    assert first_response.stat().st_ino != second_response.stat().st_ino
    plan = build_provider_billing_audit_retention_plan(audit_root=audit_root)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    result = apply_provider_billing_audit_retention_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=tmp_path / "apply.json",
    )

    assert result["status"] == "applied"
    assert result["relinked_path_count"] == 1
    assert result["relinked_bytes"] == len(fixture["response"])
    assert first_response.stat().st_ino == second_response.stat().st_ino
    assert first_receipt.read_bytes() == first_receipt_bytes
    assert second_receipt.read_bytes() == second_receipt_bytes
    for receipt_path in (first_receipt, second_receipt):
        source_path, source, _source_bytes = _validate_source_receipt(receipt_path)
        responses = _load_vast_responses(
            source_receipt_path=source_path, source_receipt=source
        )
        assert len(responses) == 1
        assert responses[0][1] == Path(source["sources"][0]["retained_path"])


def test_apply_requires_ack_and_blocks_changed_response_before_object_creation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit_root = fixture["audit_root"]
    assert isinstance(audit_root, Path)
    plan = build_provider_billing_audit_retention_plan(audit_root=audit_root)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(
        ProviderBillingAuditRetentionError,
        match="provider_billing_audit_apply_acknowledgement_missing",
    ):
        apply_provider_billing_audit_retention_plan(
            dry_run_plan_path=plan_path,
            acknowledgement="",
            receipt_out=tmp_path / "apply-no-ack.json",
        )

    _receipt, response_path, _bytes = fixture["second"]
    assert isinstance(response_path, Path)
    response_path.write_bytes(b"changed after review")
    response_path.chmod(0o600)
    with pytest.raises(
        ProviderBillingAuditRetentionError,
        match="provider_billing_audit_(response_(metadata_invalid|digest_mismatch)|plan_changed_since_dry_run)",
    ):
        apply_provider_billing_audit_retention_plan(
            dry_run_plan_path=plan_path,
            acknowledgement=APPLY_ACKNOWLEDGEMENT,
            receipt_out=tmp_path / "apply-changed.json",
        )
    assert not (audit_root / "objects").exists()


def test_symlink_response_blocks_entire_dry_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _receipt, response_path, _bytes = fixture["second"]
    assert isinstance(response_path, Path)
    target = tmp_path / "outside.json"
    target.write_bytes(response_path.read_bytes())
    target.chmod(0o600)
    response_path.unlink()
    response_path.symlink_to(target)

    with pytest.raises(
        ProviderBillingAuditRetentionError,
        match="provider_billing_audit_response_file_invalid",
    ):
        build_provider_billing_audit_retention_plan(
            audit_root=fixture["audit_root"]
        )


def test_secure_legacy_response_metadata_is_preserved_and_excluded(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _receipt, response_path, _bytes = fixture["first"]
    assert isinstance(response_path, Path)
    response_path.chmod(0o440)

    plan = build_provider_billing_audit_retention_plan(
        audit_root=fixture["audit_root"]
    )

    assert plan["response_path_count"] == 2
    assert plan["predicted_relinked_bytes"] == 0
    assert [row["path"] for row in plan["metadata_excluded_response_paths"]] == [
        str(response_path)
    ]
    assert response_path.stat().st_mode & 0o777 == 0o440


def test_apply_repairs_reviewed_directory_metadata_before_relink(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    second_receipt, _response, _bytes = fixture["second"]
    assert isinstance(second_receipt, Path)
    second_receipt.parent.chmod(0o775)
    plan = build_provider_billing_audit_retention_plan(
        audit_root=fixture["audit_root"]
    )
    assert [row["path"] for row in plan["directory_repairs"]] == [
        str(second_receipt.parent)
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    result = apply_provider_billing_audit_retention_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=tmp_path / "apply.json",
    )

    assert result["repaired_directories"] == [str(second_receipt.parent)]
    assert second_receipt.parent.stat().st_mode & 0o777 == 0o700


def test_apply_rejects_plan_or_receipt_inside_managed_root(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    audit_root = fixture["audit_root"]
    assert isinstance(audit_root, Path)
    plan = build_provider_billing_audit_retention_plan(audit_root=audit_root)
    plan_path = audit_root / "plan.json"
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(
        ProviderBillingAuditRetentionError,
        match="provider_billing_audit_receipt_inside_audit_root",
    ):
        apply_provider_billing_audit_retention_plan(
            dry_run_plan_path=plan_path,
            acknowledgement=APPLY_ACKNOWLEDGEMENT,
            receipt_out=tmp_path / "apply.json",
        )


def test_audit_root_requires_invoking_owner_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)

    with pytest.raises(
        ProviderBillingAuditRetentionError,
        match="provider_billing_audit_executor_identity_mismatch",
    ):
        build_provider_billing_audit_retention_plan(
            audit_root=fixture["audit_root"]
        )
