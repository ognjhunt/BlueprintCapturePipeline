from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.semantic_teacher_image_edit_paid_authority import (
    _validate_prior_spend_reconciliation,
)
from blueprint_pipeline.semantic_teacher_pending_spend import (
    materialize_semantic_teacher_pending_spend,
)


DIGEST = "sha256:" + "a" * 64
BUNDLE = "sha256:" + "b" * 64


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
    authority = {
        "schema_version": "semantic_teacher_image_edit_paid_authority.v1",
        "authorization_digest": "",
        "bundle": {"sha256": BUNDLE},
        "hard_total_spend_cap_usd": 5.0,
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    consumption = {
        "schema_version": "semantic_teacher_image_edit_authority_consumption.v1",
        "authorization_digest": authority["authorization_digest"],
        "bundle_sha256": BUNDLE,
    }
    teardown = {
        "status": "PASS",
        "provider": "vast",
        "authorization_digest": authority["authorization_digest"],
        "bundle_sha256": BUNDLE,
        "instance_id": "42",
        "global_provider_zero": True,
        "scoped_provider_zero": True,
        "continuing_spend_from_this_run": False,
        "timestamp": "2026-08-16T13:42:38+00:00",
        "teardown_digest": "",
    }
    teardown["teardown_digest"] = canonical_digest(
        teardown, digest_field="teardown_digest"
    )
    watchdog = {
        "status": "provider_terminal",
        "instance_ids": [42],
        "provider_absence_confirmed": True,
    }
    zero = {
        "schema_version": "gpu_spend_guard.v1",
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "provider_zero": {"status": "verified"},
        "receipt_digest": "",
    }
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    billing = {"results": [{"source": "instance-41", "amount": 0.01}]}
    paths = {
        "authority": _write(tmp_path / "authority.json", authority),
        "consumption": _write(tmp_path / "consumption.json", consumption),
        "teardown": _write(tmp_path / "teardown.json", teardown),
        "watchdog": _write(tmp_path / "watchdog.json", watchdog),
        "zero": _write(tmp_path / "zero.json", zero),
        "billing": _write(tmp_path / "billing.json", billing),
    }
    source = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "generated_at": "2026-08-16T13:50:00+00:00",
        "provider_mutation_performed": False,
        "sources": [
            {
                "provider": "vast",
                "retained_path": str(paths["billing"]),
                "response_digest": _sha(paths["billing"]),
                "response_size_bytes": paths["billing"].stat().st_size,
            }
        ],
        "receipt_digest": "",
    }
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    paths["billing_source"] = _write(tmp_path / "billing-source.json", source)
    return paths


def _run(tmp_path: Path, paths: dict[str, Path]):
    return materialize_semantic_teacher_pending_spend(
        attempt_id="website-launch-1",
        authority_path=paths["authority"],
        consumption_path=paths["consumption"],
        teardown_path=paths["teardown"],
        watchdog_path=paths["watchdog"],
        provider_zero_path=paths["zero"],
        official_billing_response_paths=[paths["billing"]],
        provider_billing_source_receipt_path=paths["billing_source"],
        reservation_output_path=tmp_path / "reservation.json",
        reconciliation_output_path=tmp_path / "reconciliation.json",
    )


def test_pending_statement_reserves_full_cap_and_validates_for_next_authority(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    reservation, reconciliation = _run(tmp_path, paths)
    assert reservation["cost_usd"] == 5.0
    assert reservation["official_billing_pending"] is True
    assert reservation["provider_zero_confirmed"] is True
    assert reconciliation["total_cost_usd"] == 5.0
    reopened, record = _validate_prior_spend_reconciliation(
        tmp_path / "reconciliation.json", expected_total_cost_usd=5.0
    )
    assert reopened == reconciliation
    assert record["total_cost_usd"] == 5.0


def test_posted_charge_cannot_be_mislabeled_pending(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    billing = json.loads(paths["billing"].read_text())
    billing["results"].append({"source": "instance-42", "amount": 0.001})
    _write(paths["billing"], billing)
    source = json.loads(paths["billing_source"].read_text())
    source["sources"][0]["response_digest"] = _sha(paths["billing"])
    source["sources"][0]["response_size_bytes"] = paths["billing"].stat().st_size
    source["receipt_digest"] = canonical_digest(source, digest_field="receipt_digest")
    _write(paths["billing_source"], source)
    with pytest.raises(ValueError, match="official_charge_not_pending"):
        _run(tmp_path, paths)


def test_legacy_gpu_spend_guard_without_embedded_digest_is_bound_by_file_sha(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    zero = json.loads(paths["zero"].read_text())
    zero.pop("receipt_digest")
    _write(paths["zero"], zero)

    reservation, _reconciliation = _run(tmp_path, paths)

    source = reservation["sources"]["provider_zero"]
    assert source["sha256"] == _sha(paths["zero"])
    assert "receipt_digest" not in source


def test_invalid_optional_gpu_spend_guard_digest_fails_closed(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    zero = json.loads(paths["zero"].read_text())
    zero["receipt_digest"] = DIGEST
    _write(paths["zero"], zero)

    with pytest.raises(ValueError, match="terminal_evidence_invalid"):
        _run(tmp_path, paths)
