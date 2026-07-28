from __future__ import annotations

from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_matrix import (
    _ledger_core,
)


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
