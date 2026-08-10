from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_paid_closeout_recovery import (
    recover_native_task_paid_closeout,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    attempt = tmp_path / "attempt_001"
    provider = attempt / "vast_provider_run"
    staging = attempt / "object_store_staging"
    provider.mkdir(parents=True)
    staging.mkdir()
    instance_id = 47382238
    phase = provider / "vast_runtime_phase_log.jsonl"
    phase.write_text(
        json.dumps(
            {
                "phase": "vast_instance_create_requested",
                "status": "completed",
                "instance_id": instance_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        provider / "vast_budget_ledger.json",
        {"vast_instance_ids": [instance_id]},
    )
    write_json(
        provider / "vast_all_in_cost_binding.json",
        {"instance_id": instance_id},
    )
    (provider / "vast_onstart_container.log").write_text(
        "ModuleNotFoundError: No module named 'torch'\n",
        encoding="utf-8",
    )
    write_json(
        staging / "wam_provider_object_store_cleanup.json",
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "all_objects_absent": True,
            "signed_url_files_removed": True,
            "exact_object_count": 2,
            "blockers": [],
        },
    )
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"immutable-bundle")
    bundle_receipt = tmp_path / "bundle-receipt.json"
    write_json(
        bundle_receipt,
        {
            "status": "ready",
            "bundle_path": str(bundle),
            "bundle_sha256": _sha256(bundle),
            "bundle_size_bytes": bundle.stat().st_size,
        },
    )
    zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "api_confirmed": True,
        "provider_zero": True,
        "global_live_resource_count": 0,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(zero, digest_field="provider_zero_digest")
    zero_path = tmp_path / "provider-zero.json"
    write_json(zero_path, zero)
    return {
        "attempt": attempt,
        "bundle_receipt": bundle_receipt,
        "zero": zero_path,
    }


def _invoices(command, **kwargs):
    assert kwargs == {"check": False, "capture_output": True, "text": True}
    assert command[-2:] == ["--instance_label", "fixture-label"]
    rows = [
        {
            "amount": "0.042",
            "instance_id": 47382238,
            "type": "charge",
            "description": "GPU",
        },
        {
            "amount": "0.025",
            "instance_id": 47382238,
            "type": "charge",
            "description": "storage",
        },
        {
            "amount": "0.006",
            "instance_id": 47382238,
            "type": "charge",
            "description": "download",
        },
    ]
    return subprocess.CompletedProcess(command, 0, stdout=json.dumps(rows), stderr="")


def test_recovery_joins_independent_zero_cost_and_scientific_failure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "recovery.json"

    result = recover_native_task_paid_closeout(
        attempt_root=fixture["attempt"],
        provider_zero_receipt_path=fixture["zero"],
        bundle_receipt_path=fixture["bundle_receipt"],
        output_path=output,
        expected_instance_id=47382238,
        instance_label="fixture-label",
        run_command=_invoices,
        generated_at="fixed",
    )

    assert result["status"] == "recovered_blocked_attempt_closeout"
    assert result["scientific_status"] == "blocked_before_simulation_app"
    assert result["scientific_blockers"] == ["native_task_pre_app_dependency_missing:torch"]
    assert result["provider_terminal_evidence"]["continuing_spend_from_this_run"] is False
    assert result["object_store_terminal_evidence"]["all_objects_absent"] is True
    assert result["provider_invoice"]["provider_reported_total_usd"] == "0.073"
    assert result["normal_teardown_reconstructed"] is False
    assert result["manual_destroy_command_receipt_present"] is False
    assert result["recovery_receipt_digest"] == canonical_digest(
        result, digest_field="recovery_receipt_digest"
    )
    assert json.loads(output.read_text()) == result


def test_recovery_rejects_instance_mismatch_or_unproven_provider_zero(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="instance_binding_invalid"):
        recover_native_task_paid_closeout(
            attempt_root=fixture["attempt"],
            provider_zero_receipt_path=fixture["zero"],
            bundle_receipt_path=fixture["bundle_receipt"],
            output_path=tmp_path / "wrong-id.json",
            expected_instance_id=7,
            instance_label="fixture-label",
            run_command=_invoices,
        )

    zero = json.loads(fixture["zero"].read_text())
    zero["provider_zero"] = False
    zero["provider_zero_digest"] = canonical_digest(zero, digest_field="provider_zero_digest")
    write_json(fixture["zero"], zero)
    with pytest.raises(ValueError, match="provider_zero_invalid"):
        recover_native_task_paid_closeout(
            attempt_root=fixture["attempt"],
            provider_zero_receipt_path=fixture["zero"],
            bundle_receipt_path=fixture["bundle_receipt"],
            output_path=tmp_path / "no-zero.json",
            expected_instance_id=47382238,
            instance_label="fixture-label",
            run_command=_invoices,
        )
