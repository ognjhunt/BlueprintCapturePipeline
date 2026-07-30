from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_matrix import (
    GeminiMatrixError,
    _ledger_core,
    _media_ledger_ready_age_seconds,
    _provider_error_payload,
    _require_complete_graph_paid_admission,
    _shard_pairs,
    _validate_inventory,
    build_complete_graph_paid_admission,
    submit_matrix_batch,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic import (
    complete_graph_diagnostic_protocol,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def _inventory(count: int, *, parent: str | None = None) -> dict:
    protocol = complete_graph_diagnostic_protocol()
    result = {
        "status": "ready",
        "pair_count": count,
        "protocol_sha256": protocol["protocol_sha256"],
        "pairs": [{"pair_id": f"pair-{index}"} for index in range(count)],
        "outcome_labels_accessed_to_build_pairs": False,
    }
    if parent is not None:
        result["parent_inventory_sha256"] = parent
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def _manifest() -> dict:
    result = {
        "status": "passed",
        "video_count": 441,
        "all_physical_right_half_pixels_excluded": True,
        "receipts": [{"request_id": f"request-{index}"} for index in range(441)],
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _admission_inputs() -> tuple[dict, dict, dict, dict, dict]:
    complete = _inventory(1323)
    missing = _inventory(882, parent=complete["inventory_sha256"])
    manifest = _manifest()
    prior = {
        "status": "completed",
        "result_count": 441,
        "error_count": 0,
        "report_sha256": "b" * 64,
        "results": [
            {
                "model": "gemini-3.6-flash",
                "arm_id": "gemini36_flash_native_video",
            }
            for _ in range(441)
        ],
    }
    reuse = {
        "status": "passed",
        "reused_pair_count": 441,
        "missing_pair_count": 882,
        "complete_inventory_sha256": complete["inventory_sha256"],
        "prior_collection_file_sha256": "a" * 64,
        "prior_collection_report_sha256": prior["report_sha256"],
        "report_sha256": "c" * 64,
        "mappings": [{"complete_pair_id": f"pair-{index}"} for index in range(882, 1323)],
    }
    missing["pairs"] = [{"pair_id": f"pair-{index}"} for index in range(882)]
    missing["inventory_sha256"] = canonical_sha256(
        {key: value for key, value in missing.items() if key != "inventory_sha256"}
    )
    return missing, complete, manifest, reuse, prior


def _paid_admission(*, projection: float = 8.3633445) -> dict:
    missing, complete, manifest, reuse, prior = _admission_inputs()
    return build_complete_graph_paid_admission(
        missing,
        complete,
        manifest,
        reuse,
        prior,
        missing_inventory_file_sha256="d" * 64,
        complete_inventory_file_sha256="e" * 64,
        manifest_file_sha256="f" * 64,
        reuse_audit_file_sha256="1" * 64,
        prior_collection_file_sha256="a" * 64,
        source_commit="2" * 40,
        realized_api_spend_usd=8.418512,
        projected_missing_arm_cost_usd=projection,
        missing_arm_cap_usd=9.0,
        campaign_api_cap_usd=25.0,
        credential_ready=True,
    )


def test_complete_graph_paid_admission_binds_inputs_budget_and_claim_boundary() -> None:
    admission = _paid_admission()

    assert admission["status"] == "admitted"
    assert admission["shared_paid_lane_admission"]["resource_class"] == "evaluator_api"
    assert admission["request_count"] == 882
    assert admission["cost_boundary"]["projected_api_spend_after_stage_usd"] == pytest.approx(
        16.7818565
    )
    assert admission["execution_contract"]["partial_matrix_ranking_credit"] is False
    assert admission["execution_contract"]["physical_ground_truth_pixels_uploaded"] is False
    payload = {key: value for key, value in admission.items() if key != "admission_sha256"}
    assert canonical_sha256(payload) == admission["admission_sha256"]


def test_complete_graph_paid_admission_fails_closed_when_projection_exceeds_arm_cap() -> None:
    admission = _paid_admission(projection=9.01)

    assert admission["status"] == "blocked"
    assert "projected_missing_arm_cost_exceeds_arm_cap" in admission["blockers"]
    assert admission["shared_paid_lane_admission"]["status"] == "blocked"


def test_complete_graph_paid_admission_issues_only_exact_bound_capability() -> None:
    missing, _, manifest, _, _ = _admission_inputs()
    admission = _paid_admission()

    grant = _require_complete_graph_paid_admission(
        admission,
        inventory=missing,
        manifest=manifest,
        source_commit="2" * 40,
    )
    assert grant.resource_class == "evaluator_api"

    tampered = json.loads(json.dumps(admission))
    tampered["cost_boundary"]["projected_missing_arm_cost_usd"] = 0.0
    with pytest.raises(GeminiMatrixError, match="complete_graph_paid_admission_digest_invalid"):
        _require_complete_graph_paid_admission(
            tampered,
            inventory=missing,
            manifest=manifest,
            source_commit="2" * 40,
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


def test_matrix_submission_is_idempotent_from_existing_valid_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    protocol = complete_graph_diagnostic_protocol()
    pairs = [{"pair_id": f"pair-{index}"} for index in range(882)]
    inventory = {
        "status": "ready",
        "pair_count": 882,
        "protocol_sha256": protocol["protocol_sha256"],
        "pairs": pairs,
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    previous = {
        "status": "JOB_STATE_PENDING",
        "batch_name": "batches/existing",
        "arm_id": "gemini36_flash_complete_graph",
        "request_count": 63,
        "full_inventory_request_count": 882,
        "inventory_sha256": inventory["inventory_sha256"],
        "shard_index": 0,
        "shard_count": 14,
        "cleanup_uploads_on_terminal": False,
    }
    previous["receipt_sha256"] = canonical_sha256(previous)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(previous))

    result = submit_matrix_batch(
        inventory,
        {},
        api_key_file=tmp_path / "unused",
        receipt_path=receipt,
        source_commit="e" * 40,
    )

    assert result == previous


def test_complete_graph_shards_are_exact_disjoint_inventory_partition() -> None:
    pairs = [{"pair_id": f"pair-{index}"} for index in range(882)]
    shards = [_shard_pairs(pairs, shard_index=index, shard_count=14) for index in range(14)]

    assert all(len(shard) == 63 for shard in shards)
    assert [row for shard in shards for row in shard] == pairs
    assert len({row["pair_id"] for shard in shards for row in shard}) == 882


def test_nonfinal_shard_cannot_request_early_cleanup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    inventory = _inventory(882)

    with pytest.raises(
        GeminiMatrixError, match="complete_graph_batch_shard_cleanup_contract_invalid"
    ):
        submit_matrix_batch(
            inventory,
            {},
            api_key_file=tmp_path / "unused",
            receipt_path=tmp_path / "new-receipt.json",
            source_commit="2" * 40,
            shard_index=0,
            shard_count=14,
            cleanup_uploads_on_terminal=True,
        )


def test_provider_error_payload_preserves_sanitized_code_status_and_message() -> None:
    class ExampleClientError(Exception):
        code = 400
        status = "INVALID_ARGUMENT"
        message = "request exceeds an active provider limit"
        details = {"error": {"reason": "EXAMPLE_PRECONDITION"}}

    payload = _provider_error_payload(ExampleClientError("raw response object omitted"))

    assert payload == {
        "error_type": "ExampleClientError",
        "provider_code": 400,
        "provider_status": "INVALID_ARGUMENT",
        "provider_message": "request exceeds an active provider limit",
        "provider_details": {"error": {"reason": "EXAMPLE_PRECONDITION"}},
    }


def test_media_ledger_ready_age_is_measured_from_frozen_utc_timestamp() -> None:
    age = _media_ledger_ready_age_seconds(
        {"updated_at_utc": "2026-07-30T07:00:00Z"},
        now=datetime(2026, 7, 30, 7, 1, 31, tzinfo=timezone.utc),
    )

    assert age == 91.0


def test_matrix_validator_accepts_registered_882_pair_subset() -> None:
    protocol = complete_graph_diagnostic_protocol()
    inventory = {
        "status": "ready",
        "pair_count": 882,
        "protocol_sha256": protocol["protocol_sha256"],
        "pairs": [{"pair_id": f"pair-{index}"} for index in range(882)],
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)

    assert len(_validate_inventory(inventory)) == 882


def test_matrix_submission_refuses_provider_path_without_paid_admission(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    inventory = _inventory(882)

    with pytest.raises(GeminiMatrixError, match="complete_graph_paid_admission_missing"):
        submit_matrix_batch(
            inventory,
            {"native_video_manifest_sha256": "f" * 64},
            api_key_file=tmp_path / "unused",
            receipt_path=tmp_path / "new-receipt.json",
            source_commit="2" * 40,
        )
