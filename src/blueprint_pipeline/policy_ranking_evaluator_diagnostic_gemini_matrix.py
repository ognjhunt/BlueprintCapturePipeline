"""Resumable media staging for the full Gemini native-video diagnostic matrix."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
    require_paid_resource_admission_grant,
)
from .policy_ranking_evaluator_diagnostic import (
    complete_graph_diagnostic_protocol,
    diagnostic_protocol,
)
from .policy_ranking_evaluator_diagnostic_gemini import (
    GATE_ENV,
    _secure_file,
    _upload_video,
    _validated_manifest_rows,
)
from .policy_ranking_evaluator_diagnostic_gemini_batch import _build_inline_request, _job_state
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


LEDGER_SCHEMA = "policy_ranking_gemini_matrix_media_ledger.v1"
PAID_ADMISSION_SCHEMA = "policy_ranking_gemini_complete_graph_paid_admission.v1"
PAID_RESOURCE_CLASS = "evaluator_api"
COMPLETE_GRAPH_ARM_ID = "gemini36_flash_complete_graph"
COMPLETE_GRAPH_MODEL = "gemini-3.6-flash"
MISSING_PAIR_COUNT = 882
COMPLETE_PAIR_COUNT = 1323
NATIVE_VIDEO_COUNT = 441


class GeminiMatrixError(ValueError):
    """The full-matrix Gemini media stage is invalid."""


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def build_complete_graph_paid_admission(
    missing_inventory: Mapping[str, Any],
    complete_inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    reuse_audit: Mapping[str, Any],
    prior_collection: Mapping[str, Any],
    *,
    missing_inventory_file_sha256: str,
    complete_inventory_file_sha256: str,
    manifest_file_sha256: str,
    reuse_audit_file_sha256: str,
    prior_collection_file_sha256: str,
    source_commit: str,
    realized_api_spend_usd: float,
    projected_missing_arm_cost_usd: float,
    missing_arm_cap_usd: float,
    campaign_api_cap_usd: float,
    credential_ready: bool,
) -> dict[str, Any]:
    """Build the immutable, diagnostic-only admission for the missing Gemini edges."""

    blockers: list[str] = []
    try:
        missing_pairs = _validate_inventory(missing_inventory)
        complete_pairs = _validate_inventory(complete_inventory)
        _validated_manifest_rows(manifest)
    except (GeminiMatrixError, ValueError) as exc:
        blockers.append(f"input_contract_invalid:{type(exc).__name__}")
        missing_pairs = []
        complete_pairs = []
    if len(missing_pairs) != MISSING_PAIR_COUNT:
        blockers.append("missing_pair_count_not_882")
    if len(complete_pairs) != COMPLETE_PAIR_COUNT:
        blockers.append("complete_pair_count_not_1323")
    missing_pair_ids = {str(pair.get("pair_id")) for pair in missing_pairs}
    complete_pair_ids = {str(pair.get("pair_id")) for pair in complete_pairs}
    reuse_mappings = reuse_audit.get("mappings")
    reused_complete_pair_ids = (
        {str(row.get("complete_pair_id")) for row in reuse_mappings}
        if isinstance(reuse_mappings, list)
        else set()
    )
    if (
        len(reused_complete_pair_ids) != NATIVE_VIDEO_COUNT
        or missing_pair_ids & reused_complete_pair_ids
        or missing_pair_ids | reused_complete_pair_ids != complete_pair_ids
    ):
        blockers.append("missing_and_reused_pairs_do_not_partition_complete_graph")
    if missing_inventory.get("parent_inventory_sha256") != complete_inventory.get(
        "inventory_sha256"
    ):
        blockers.append("missing_inventory_parent_mismatch")
    if missing_inventory.get("outcome_labels_accessed_to_build_pairs") is not False:
        blockers.append("outcome_labels_used_to_build_missing_inventory")
    if manifest.get("all_physical_right_half_pixels_excluded") is not True:
        blockers.append("physical_ground_truth_pixels_not_excluded")
    if (
        reuse_audit.get("status") != "passed"
        or reuse_audit.get("reused_pair_count") != NATIVE_VIDEO_COUNT
        or reuse_audit.get("missing_pair_count") != MISSING_PAIR_COUNT
        or reuse_audit.get("complete_inventory_sha256")
        != complete_inventory.get("inventory_sha256")
        or reuse_audit.get("prior_collection_file_sha256") != prior_collection_file_sha256
        or reuse_audit.get("prior_collection_report_sha256")
        != prior_collection.get("report_sha256")
    ):
        blockers.append("frozen_subset_reuse_evidence_invalid")
    prior_results = prior_collection.get("results")
    if (
        prior_collection.get("status") != "completed"
        or prior_collection.get("result_count") != NATIVE_VIDEO_COUNT
        or prior_collection.get("error_count") != 0
        or not isinstance(prior_results, list)
        or len(prior_results) != NATIVE_VIDEO_COUNT
        or any(result.get("model") != COMPLETE_GRAPH_MODEL for result in prior_results)
        or any(result.get("arm_id") != "gemini36_flash_native_video" for result in prior_results)
    ):
        blockers.append("prior_same_configuration_collection_invalid")
    if not credential_ready:
        blockers.append("gemini_credential_not_ready")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        blockers.append("source_commit_not_full_lowercase_sha")
    costs = (
        realized_api_spend_usd,
        projected_missing_arm_cost_usd,
        missing_arm_cap_usd,
        campaign_api_cap_usd,
    )
    if any(not isinstance(value, (int, float)) or value < 0 for value in costs):
        blockers.append("cost_value_invalid")
    if not 0 < missing_arm_cap_usd <= campaign_api_cap_usd <= 25.0:
        blockers.append("cost_caps_invalid")
    if projected_missing_arm_cost_usd > missing_arm_cap_usd:
        blockers.append("projected_missing_arm_cost_exceeds_arm_cap")
    if realized_api_spend_usd + projected_missing_arm_cost_usd > campaign_api_cap_usd:
        blockers.append("projected_campaign_api_cost_exceeds_category_cap")
    expected_file_hashes = {
        "missing_inventory": missing_inventory_file_sha256,
        "complete_inventory": complete_inventory_file_sha256,
        "native_video_manifest": manifest_file_sha256,
        "reuse_audit": reuse_audit_file_sha256,
        "prior_collection": prior_collection_file_sha256,
    }
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in expected_file_hashes.values()
    ):
        blockers.append("source_file_sha256_invalid")
    shared = build_paid_lane_admission(
        resource_class=PAID_RESOURCE_CLASS,
        blockers=blockers,
    )
    admission: dict[str, Any] = {
        "schema_version": PAID_ADMISSION_SCHEMA,
        "status": shared["status"],
        "arm_id": COMPLETE_GRAPH_ARM_ID,
        "model": COMPLETE_GRAPH_MODEL,
        "claim_class": "post_unseal_diagnostic_only",
        "source_commit": source_commit,
        "request_count": MISSING_PAIR_COUNT,
        "unique_video_count": NATIVE_VIDEO_COUNT,
        "missing_inventory_sha256": missing_inventory.get("inventory_sha256"),
        "complete_inventory_sha256": complete_inventory.get("inventory_sha256"),
        "native_video_manifest_sha256": manifest.get("manifest_sha256"),
        "reuse_audit_report_sha256": reuse_audit.get("report_sha256"),
        "prior_collection_report_sha256": prior_collection.get("report_sha256"),
        "source_file_sha256": expected_file_hashes,
        "cost_boundary": {
            "realized_api_spend_before_stage_usd": realized_api_spend_usd,
            "projected_missing_arm_cost_usd": projected_missing_arm_cost_usd,
            "missing_arm_hard_cap_usd": missing_arm_cap_usd,
            "projected_api_spend_after_stage_usd": (
                realized_api_spend_usd + projected_missing_arm_cost_usd
            ),
            "campaign_api_hard_cap_usd": campaign_api_cap_usd,
            "contingency_does_not_expand_category_cap": True,
        },
        "execution_contract": {
            "frozen_prior_results_reused": NATIVE_VIDEO_COUNT,
            "prospective_requests": MISSING_PAIR_COUNT,
            "temporary_uploads_expected": NATIVE_VIDEO_COUNT,
            "upload_idempotency_required": True,
            "cleanup_on_submission_or_collection_terminal": True,
            "policy_identity_sent_to_provider": False,
            "physical_outcome_sent_to_provider": False,
            "physical_ground_truth_pixels_uploaded": False,
            "partial_matrix_ranking_credit": False,
            "aggregate_only_after_complete_1323_predictions_frozen": True,
        },
        "shared_paid_lane_admission": shared,
        "blockers": sorted(set(blockers)),
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    admission["admission_sha256"] = canonical_sha256(admission)
    return admission


def _require_complete_graph_paid_admission(
    admission: Mapping[str, Any] | None,
    *,
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source_commit: str,
) -> PaidResourceAdmissionGrant:
    if not isinstance(admission, Mapping):
        raise GeminiMatrixError("complete_graph_paid_admission_missing")
    payload = _without_digest(admission, "admission_sha256")
    cost = admission.get("cost_boundary") or {}
    contract = admission.get("execution_contract") or {}
    blockers: list[str] = []
    if canonical_sha256(payload) != admission.get("admission_sha256"):
        blockers.append("complete_graph_paid_admission_digest_invalid")
    if admission.get("schema_version") != PAID_ADMISSION_SCHEMA:
        blockers.append("complete_graph_paid_admission_schema_invalid")
    if admission.get("status") != "admitted" or admission.get("blockers") not in ([], None):
        blockers.append("complete_graph_paid_admission_not_admitted")
    if (
        admission.get("arm_id") != COMPLETE_GRAPH_ARM_ID
        or admission.get("model") != COMPLETE_GRAPH_MODEL
        or admission.get("claim_class") != "post_unseal_diagnostic_only"
        or admission.get("request_count") != MISSING_PAIR_COUNT
        or admission.get("unique_video_count") != NATIVE_VIDEO_COUNT
    ):
        blockers.append("complete_graph_paid_admission_arm_contract_mismatch")
    if (
        admission.get("source_commit") != source_commit
        or admission.get("missing_inventory_sha256") != inventory.get("inventory_sha256")
        or admission.get("native_video_manifest_sha256") != manifest.get("manifest_sha256")
    ):
        blockers.append("complete_graph_paid_admission_input_binding_mismatch")
    if (
        cost.get("projected_missing_arm_cost_usd", float("inf"))
        > cost.get("missing_arm_hard_cap_usd", -1)
        or cost.get("projected_api_spend_after_stage_usd", float("inf"))
        > cost.get("campaign_api_hard_cap_usd", -1)
        or cost.get("campaign_api_hard_cap_usd", float("inf")) > 25.0
    ):
        blockers.append("complete_graph_paid_admission_cost_boundary_invalid")
    if (
        contract.get("partial_matrix_ranking_credit") is not False
        or contract.get("aggregate_only_after_complete_1323_predictions_frozen") is not True
        or contract.get("physical_ground_truth_pixels_uploaded") is not False
    ):
        blockers.append("complete_graph_paid_admission_claim_boundary_invalid")
    if blockers:
        raise GeminiMatrixError(";".join(sorted(set(blockers))))
    try:
        return require_paid_resource_admission(
            admission["shared_paid_lane_admission"],
            resource_class=PAID_RESOURCE_CLASS,
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except Exception as exc:
        raise GeminiMatrixError("shared_paid_resource_admission_rejected") from exc


def _arm_id(inventory: Mapping[str, Any]) -> str:
    if inventory.get("protocol_sha256") == complete_graph_diagnostic_protocol()["protocol_sha256"]:
        return "gemini36_flash_complete_graph"
    return "gemini36_flash_native_video"


def _ledger_core(
    *,
    status: str,
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    source_commit: str,
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA,
        "status": status,
        "arm_id": _arm_id(inventory),
        "inventory_sha256": inventory["inventory_sha256"],
        "native_video_manifest_sha256": manifest["manifest_sha256"],
        "expected_video_count": 441,
        "uploaded_video_count": len(receipts),
        "uploads": sorted((dict(row) for row in receipts), key=lambda row: row["request_id"]),
        "source_commit": source_commit,
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_uploaded": False,
        "credential_path_or_value_persisted": False,
        "blockers": sorted(set(blockers)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    return ledger


def _validate_inventory(inventory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    allowed_protocol_digests = {
        diagnostic_protocol()["protocol_sha256"],
        complete_graph_diagnostic_protocol()["protocol_sha256"],
    }
    pairs = inventory.get("pairs")
    pair_count = inventory.get("pair_count")
    payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if (
        inventory.get("status") != "ready"
        or pair_count not in {441, 882, 1323}
        or inventory.get("protocol_sha256") not in allowed_protocol_digests
        or canonical_sha256(payload) != inventory.get("inventory_sha256")
        or not isinstance(pairs, list)
        or len(pairs) != pair_count
        or len({str(pair.get("pair_id")) for pair in pairs}) != pair_count
    ):
        raise GeminiMatrixError("pair_inventory_not_ready_bound_and_valid_supported_matrix")
    return pairs


def _delete_receipts(client: Any, receipts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt in receipts:
        name = str(receipt["provider_file_name"])
        try:
            client.files.delete(name=name)
            rows.append({"provider_file_name": name, "deleted": True})
        except Exception as exc:
            rows.append(
                {"provider_file_name": name, "deleted": False, "error_type": type(exc).__name__}
            )
    return rows


def upload_matrix_media(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    ledger_path: str | Path,
    source_commit: str,
    workers: int = 4,
    paid_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        raise GeminiMatrixError(f"missing_env_{GATE_ENV}")
    pairs = _validate_inventory(inventory)
    manifest_rows = _validated_manifest_rows(manifest)
    required_ids = sorted(
        {
            str(pair[side]["source_request_id"])
            for pair in pairs
            for side in ("episode_a", "episode_b")
        }
    )
    if len(required_ids) != 441 or set(required_ids) != set(manifest_rows):
        raise GeminiMatrixError("matrix_video_identity_set_not_exact_441")
    paid_grant = _require_complete_graph_paid_admission(
        paid_admission,
        inventory=inventory,
        manifest=manifest,
        source_commit=source_commit,
    )
    key_path = _secure_file(api_key_file)
    api_key = key_path.read_text(encoding="utf-8").strip()
    target = Path(ledger_path)
    existing: list[dict[str, Any]] = []
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        previous_payload = {key: value for key, value in previous.items() if key != "ledger_sha256"}
        if (
            canonical_sha256(previous_payload) != previous.get("ledger_sha256")
            or previous.get("inventory_sha256") != inventory["inventory_sha256"]
            or previous.get("native_video_manifest_sha256") != manifest["manifest_sha256"]
        ):
            raise GeminiMatrixError("existing_media_ledger_invalid_or_wrong_inputs")
        existing = [dict(row) for row in previous.get("uploads") or []]
    existing_by_id = {str(row["request_id"]): row for row in existing}
    pending = [request_id for request_id in required_ids if request_id not in existing_by_id]
    receipts = list(existing_by_id.values())
    failures: list[str] = []

    def upload_one(request_id: str) -> dict[str, Any]:
        from google import genai

        require_paid_resource_admission_grant(
            paid_grant,
            resource_class=PAID_RESOURCE_CLASS,
        )
        client = genai.Client(api_key=api_key)
        _, receipt = _upload_video(client, manifest_rows[request_id])
        return receipt

    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8))) as pool:
        futures = {pool.submit(upload_one, request_id): request_id for request_id in pending}
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                receipts.append(future.result())
            except Exception as exc:
                failures.append(f"{request_id}:{type(exc).__name__}")
            progress = _ledger_core(
                status="uploading" if not failures else "cleanup_required",
                inventory=inventory,
                manifest=manifest,
                receipts=receipts,
                source_commit=source_commit,
                blockers=failures,
            )
            write_json(target, progress)
    if failures:
        from google import genai

        cleanup_client = genai.Client(api_key=api_key)
        deletions = _delete_receipts(cleanup_client, receipts)
        cleanup_failed = [row for row in deletions if not row["deleted"]]
        failed = _ledger_core(
            status="failed_cleaned" if not cleanup_failed else "blocked_cleanup_incomplete",
            inventory=inventory,
            manifest=manifest,
            receipts=[],
            source_commit=source_commit,
            blockers=failures + ["provider_file_cleanup_incomplete"] * bool(cleanup_failed),
        )
        failed["deletions"] = deletions
        failed["ledger_sha256"] = canonical_sha256(
            {key: value for key, value in failed.items() if key != "ledger_sha256"}
        )
        write_json(target, failed)
        return failed
    ready = _ledger_core(
        status="ready",
        inventory=inventory,
        manifest=manifest,
        receipts=receipts,
        source_commit=source_commit,
    )
    if ready["uploaded_video_count"] != 441:
        raise GeminiMatrixError("media_upload_completed_without_441_receipts")
    write_json(target, ready)
    return ready


def cleanup_matrix_media(
    ledger: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if canonical_sha256(payload) != ledger.get("ledger_sha256"):
        raise GeminiMatrixError("media_ledger_digest_invalid")
    key_path = _secure_file(api_key_file)
    from google import genai

    client = genai.Client(api_key=key_path.read_text(encoding="utf-8").strip())
    deletions = _delete_receipts(client, ledger.get("uploads") or [])
    deleted_all = all(row["deleted"] for row in deletions)
    report = {
        "schema_version": "policy_ranking_gemini_matrix_media_cleanup.v1",
        "status": "passed" if deleted_all else "blocked",
        "ledger_sha256": ledger["ledger_sha256"],
        "expected_deletion_count": len(ledger.get("uploads") or []),
        "deletions": deletions,
        "all_task_media_deleted": deleted_all,
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(Path(output_path), report)
    return report


def submit_matrix_batch(
    inventory: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    receipt_path: str | Path,
    source_commit: str,
    paid_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        raise GeminiMatrixError(f"missing_env_{GATE_ENV}")
    pairs = _validate_inventory(inventory)
    pair_count = len(pairs)
    target = Path(receipt_path)
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        previous_payload = {
            key: value for key, value in previous.items() if key != "receipt_sha256"
        }
        if canonical_sha256(previous_payload) != previous.get("receipt_sha256"):
            raise GeminiMatrixError("existing_matrix_batch_receipt_digest_invalid")
        if (
            previous.get("batch_name")
            and previous.get("arm_id") == _arm_id(inventory)
            and previous.get("request_count") == pair_count
            and (
                pair_count == 441
                or previous.get("inventory_sha256") == inventory["inventory_sha256"]
            )
        ):
            return previous
        raise GeminiMatrixError("existing_matrix_batch_failure_receipt_requires_review")
    paid_grant = _require_complete_graph_paid_admission(
        paid_admission,
        inventory=inventory,
        manifest={
            "manifest_sha256": ledger.get("native_video_manifest_sha256"),
        },
        source_commit=source_commit,
    )
    ledger_payload = {key: value for key, value in ledger.items() if key != "ledger_sha256"}
    if (
        canonical_sha256(ledger_payload) != ledger.get("ledger_sha256")
        or ledger.get("status") != "ready"
        or ledger.get("inventory_sha256") != inventory["inventory_sha256"]
        or ledger.get("uploaded_video_count") != 441
    ):
        raise GeminiMatrixError("matrix_media_ledger_not_ready_bound_and_valid_441")
    key_path = _secure_file(api_key_file)
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key_path.read_text(encoding="utf-8").strip())
    upload_by_id = {str(row["request_id"]): row for row in ledger["uploads"]}
    files_by_id = {
        request_id: types.File(
            name=str(row["provider_file_name"]),
            uri=str(row["provider_file_uri"]),
            mime_type=str(row.get("provider_mime_type") or "video/mp4"),
        )
        for request_id, row in upload_by_id.items()
    }
    requests = [
        _build_inline_request(
            pair,
            files_by_id[str(pair["episode_a"]["source_request_id"])],
            files_by_id[str(pair["episode_b"]["source_request_id"])],
            types_module=types,
        )
        for pair in pairs
    ]
    try:
        require_paid_resource_admission_grant(
            paid_grant,
            resource_class=PAID_RESOURCE_CLASS,
        )
        job = client.batches.create(
            model="gemini-3.6-flash",
            src=requests,
            config=types.CreateBatchJobConfig(
                display_name="blueprint-roboarena-gemini36-complete-graph-v1"
            ),
        )
    except Exception as exc:
        deletions = _delete_receipts(client, ledger["uploads"])
        failed: dict[str, Any] = {
            "schema_version": "policy_ranking_gemini_matrix_batch_submission.v1",
            "status": "failed_before_batch_creation",
            "batch_name": None,
            "request_count": pair_count,
            "inventory_sha256": inventory["inventory_sha256"],
            "media_ledger_sha256": ledger["ledger_sha256"],
            "error_type": type(exc).__name__,
            "deletions": deletions,
            "all_task_media_deleted": all(row["deleted"] for row in deletions),
            "provider_generation_rows_created": 0,
            "credential_path_or_value_persisted": False,
        }
        failed["receipt_sha256"] = canonical_sha256(failed)
        write_json(target, failed)
        return failed
    receipt: dict[str, Any] = {
        "schema_version": "policy_ranking_gemini_matrix_batch_submission.v1",
        "status": _job_state(job),
        "batch_name": str(job.name),
        "model": "gemini-3.6-flash",
        "arm_id": _arm_id(inventory),
        "pair_ids": [str(pair["pair_id"]) for pair in pairs],
        "request_count": pair_count,
        "unique_video_count": 441,
        "inventory_sha256": inventory["inventory_sha256"],
        "uploads": ledger["uploads"],
        "media_ledger_sha256": ledger["ledger_sha256"],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "media_staging_source_commit": ledger["source_commit"],
        "submission_source_commit": source_commit,
        "provider_called": True,
        "data_uploaded": True,
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_uploaded": False,
        "credential_path_or_value_persisted": False,
        "duplicate_submission_refused_when_receipt_exists": True,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(target, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    admit = commands.add_parser("admit")
    admit.add_argument("--inventory", required=True)
    admit.add_argument("--complete-inventory", required=True)
    admit.add_argument("--native-video-manifest", required=True)
    admit.add_argument("--reuse-audit", required=True)
    admit.add_argument("--prior-collection", required=True)
    admit.add_argument("--api-key-file", required=True)
    admit.add_argument("--source-commit", required=True)
    admit.add_argument("--realized-api-spend-usd", type=float, required=True)
    admit.add_argument("--projected-missing-arm-cost-usd", type=float, required=True)
    admit.add_argument("--missing-arm-cap-usd", type=float, required=True)
    admit.add_argument("--campaign-api-cap-usd", type=float, required=True)
    admit.add_argument("--output", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--inventory", required=True)
    upload.add_argument("--native-video-manifest", required=True)
    upload.add_argument("--api-key-file", required=True)
    upload.add_argument("--ledger", required=True)
    upload.add_argument("--source-commit", required=True)
    upload.add_argument("--paid-admission", required=True)
    upload.add_argument("--workers", type=int, default=4)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--ledger", required=True)
    cleanup.add_argument("--api-key-file", required=True)
    cleanup.add_argument("--output", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--inventory", required=True)
    submit.add_argument("--ledger", required=True)
    submit.add_argument("--api-key-file", required=True)
    submit.add_argument("--receipt", required=True)
    submit.add_argument("--source-commit", required=True)
    submit.add_argument("--paid-admission", required=True)
    args = parser.parse_args(argv)
    if args.command == "admit":
        paths = {
            "missing_inventory": Path(args.inventory),
            "complete_inventory": Path(args.complete_inventory),
            "manifest": Path(args.native_video_manifest),
            "reuse_audit": Path(args.reuse_audit),
            "prior_collection": Path(args.prior_collection),
        }
        try:
            _secure_file(args.api_key_file)
            credential_ready = True
        except ValueError:
            credential_ready = False
        result = build_complete_graph_paid_admission(
            json.loads(paths["missing_inventory"].read_text(encoding="utf-8")),
            json.loads(paths["complete_inventory"].read_text(encoding="utf-8")),
            json.loads(paths["manifest"].read_text(encoding="utf-8")),
            json.loads(paths["reuse_audit"].read_text(encoding="utf-8")),
            json.loads(paths["prior_collection"].read_text(encoding="utf-8")),
            missing_inventory_file_sha256=file_sha256(paths["missing_inventory"]),
            complete_inventory_file_sha256=file_sha256(paths["complete_inventory"]),
            manifest_file_sha256=file_sha256(paths["manifest"]),
            reuse_audit_file_sha256=file_sha256(paths["reuse_audit"]),
            prior_collection_file_sha256=file_sha256(paths["prior_collection"]),
            source_commit=args.source_commit,
            realized_api_spend_usd=args.realized_api_spend_usd,
            projected_missing_arm_cost_usd=args.projected_missing_arm_cost_usd,
            missing_arm_cap_usd=args.missing_arm_cap_usd,
            campaign_api_cap_usd=args.campaign_api_cap_usd,
            credential_ready=credential_ready,
        )
        write_json(Path(args.output), result)
    elif args.command == "upload":
        result = upload_matrix_media(
            json.loads(Path(args.inventory).read_text(encoding="utf-8")),
            json.loads(Path(args.native_video_manifest).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            ledger_path=args.ledger,
            source_commit=args.source_commit,
            workers=args.workers,
            paid_admission=json.loads(Path(args.paid_admission).read_text(encoding="utf-8")),
        )
    elif args.command == "cleanup":
        result = cleanup_matrix_media(
            json.loads(Path(args.ledger).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            output_path=args.output,
        )
    else:
        result = submit_matrix_batch(
            json.loads(Path(args.inventory).read_text(encoding="utf-8")),
            json.loads(Path(args.ledger).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            receipt_path=args.receipt,
            source_commit=args.source_commit,
            paid_admission=json.loads(Path(args.paid_admission).read_text(encoding="utf-8")),
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"uploads", "deletions"}}
        )
    )
    return (
        0
        if result.get("status")
        in {"admitted", "ready", "passed", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
