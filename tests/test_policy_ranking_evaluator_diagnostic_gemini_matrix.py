from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_matrix import (
    _ledger_core,
    submit_matrix_batch,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def test_matrix_media_ledger_is_digest_bound_and_redacted() -> None:
    inventory = {"inventory_sha256": "a" * 64}
    manifest = {"manifest_sha256": "b" * 64}
    ledger = _ledger_core(
        status="ready",
        inventory=inventory,
        manifest=manifest,
        receipts=[
            {
                "request_id": "request-1",
                "local_sha256": "c" * 64,
                "provider_file_name": "files/test",
            }
        ],
        source_commit="d" * 40,
    )

    assert ledger["uploaded_video_count"] == 1
    assert ledger["policy_identity_sent_to_provider"] is False
    assert ledger["physical_ground_truth_pixels_uploaded"] is False
    assert ledger["credential_path_or_value_persisted"] is False
    assert len(ledger["ledger_sha256"]) == 64


def test_matrix_submission_is_idempotent_from_existing_valid_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    previous = {
        "status": "JOB_STATE_PENDING",
        "batch_name": "batches/existing",
        "arm_id": "gemini36_flash_native_video",
        "request_count": 441,
    }
    previous["receipt_sha256"] = canonical_sha256(previous)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(previous))

    result = submit_matrix_batch(
        {
            "status": "ready",
            "pair_count": 441,
            "protocol_sha256": "wrong_but_not_reached_after_inventory_validation",
            "pairs": [],
        },
        {},
        api_key_file=tmp_path / "unused",
        receipt_path=receipt,
        source_commit="e" * 40,
    )

    assert result == previous
