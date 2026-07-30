"""Paid, label-blind Gemini transport canary for the frozen missing comparison graph."""

from __future__ import annotations

import argparse
import json
import os
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
from .policy_ranking_evaluator_diagnostic import complete_graph_diagnostic_protocol
from .policy_ranking_evaluator_diagnostic_gemini import (
    GATE_ENV,
    _secure_file,
    _upload_video,
    _validated_manifest_rows,
)
from .policy_ranking_evaluator_diagnostic_gemini_batch import (
    _build_inline_request,
    _delete_uploaded_file_best_effort,
    _job_state,
)
from .policy_ranking_evaluator_diagnostic_gemini_matrix import (
    COMPLETE_GRAPH_ARM_ID,
    COMPLETE_GRAPH_MODEL,
    MINIMUM_MEDIA_READY_AGE_SECONDS,
    PAID_RESOURCE_CLASS,
    _provider_error_payload,
    _validate_inventory,
)
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


ADMISSION_SCHEMA = "policy_ranking_gemini_transport_canary_paid_admission.v1"
LEDGER_SCHEMA = "policy_ranking_gemini_transport_canary_media_ledger.v1"
RECEIPT_SCHEMA = "policy_ranking_gemini_transport_canary_submission.v1"
CANARY_PAIR_INDEX = 0
CANARY_REQUEST_COUNT = 1
CANARY_VIDEO_COUNT = 2


class GeminiTransportCanaryError(ValueError):
    """The frozen Gemini transport canary contract is invalid."""


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _selected_pair(inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    pairs = _validate_inventory(inventory)
    pair = pairs[CANARY_PAIR_INDEX]
    request_ids = [str(pair[side]["source_request_id"]) for side in ("episode_a", "episode_b")]
    if len(set(request_ids)) != CANARY_VIDEO_COUNT:
        raise GeminiTransportCanaryError("canary_pair_does_not_reference_two_distinct_videos")
    return pair


def build_transport_canary_paid_admission(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    inventory_file_sha256: str,
    manifest_file_sha256: str,
    source_commit: str,
    realized_api_spend_usd: float,
    realized_missing_graph_spend_usd: float,
    projected_canary_cost_usd: float,
    canary_hard_cap_usd: float,
    missing_graph_hard_cap_usd: float,
    campaign_api_hard_cap_usd: float,
    credential_ready: bool,
) -> dict[str, Any]:
    """Build a source- and input-bound capability for exactly one comparison request."""

    blockers: list[str] = []
    try:
        pair = _selected_pair(inventory)
        manifest_rows = _validated_manifest_rows(manifest)
    except (GeminiTransportCanaryError, ValueError, KeyError) as exc:
        blockers.append(f"canary_input_contract_invalid:{type(exc).__name__}")
        pair = {}
        manifest_rows = {}
    request_ids = [
        str(pair.get(side, {}).get("source_request_id") or "")
        for side in ("episode_a", "episode_b")
    ]
    if len(set(request_ids)) != CANARY_VIDEO_COUNT or any(
        request_id not in manifest_rows for request_id in request_ids
    ):
        blockers.append("canary_pair_media_not_exactly_bound")
    if inventory.get("outcome_labels_accessed_to_build_pairs") is not False:
        blockers.append("outcome_labels_used_to_build_canary_inventory")
    if manifest.get("all_physical_right_half_pixels_excluded") is not True:
        blockers.append("physical_ground_truth_pixels_not_excluded")
    if not credential_ready:
        blockers.append("gemini_credential_not_ready")
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        blockers.append("source_commit_not_full_lowercase_sha")
    hashes = (inventory_file_sha256, manifest_file_sha256)
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in hashes
    ):
        blockers.append("source_file_sha256_invalid")
    costs = (
        realized_api_spend_usd,
        realized_missing_graph_spend_usd,
        projected_canary_cost_usd,
        canary_hard_cap_usd,
        missing_graph_hard_cap_usd,
        campaign_api_hard_cap_usd,
    )
    if any(not isinstance(value, (int, float)) or value < 0 for value in costs):
        blockers.append("cost_value_invalid")
    if not 0 < canary_hard_cap_usd <= missing_graph_hard_cap_usd <= 9.0:
        blockers.append("canary_or_missing_graph_cost_cap_invalid")
    if not 0 < campaign_api_hard_cap_usd <= 25.0:
        blockers.append("campaign_api_cost_cap_invalid")
    if projected_canary_cost_usd > canary_hard_cap_usd:
        blockers.append("projected_canary_cost_exceeds_cap")
    if realized_missing_graph_spend_usd + projected_canary_cost_usd > missing_graph_hard_cap_usd:
        blockers.append("projected_missing_graph_cost_exceeds_cap")
    if realized_api_spend_usd + projected_canary_cost_usd > campaign_api_hard_cap_usd:
        blockers.append("projected_campaign_api_cost_exceeds_category_cap")
    shared = build_paid_lane_admission(
        resource_class=PAID_RESOURCE_CLASS,
        blockers=blockers,
    )
    protocol = complete_graph_diagnostic_protocol()
    admission: dict[str, Any] = {
        "schema_version": ADMISSION_SCHEMA,
        "status": shared["status"],
        "arm_id": COMPLETE_GRAPH_ARM_ID,
        "model": COMPLETE_GRAPH_MODEL,
        "claim_class": "post_unseal_transport_diagnostic_only",
        "source_commit": source_commit,
        "inventory_sha256": inventory.get("inventory_sha256"),
        "inventory_file_sha256": inventory_file_sha256,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_sha256": manifest_file_sha256,
        "protocol_sha256": protocol["protocol_sha256"],
        "prompt_sha256": protocol["shared_judging_contract"]["prompt_sha256"],
        "pair_index": CANARY_PAIR_INDEX,
        "pair_id": pair.get("pair_id"),
        "request_ids": request_ids,
        "request_count": CANARY_REQUEST_COUNT,
        "video_count": CANARY_VIDEO_COUNT,
        "cost_boundary": {
            "realized_api_spend_before_canary_usd": realized_api_spend_usd,
            "realized_missing_graph_spend_before_canary_usd": realized_missing_graph_spend_usd,
            "projected_canary_cost_usd": projected_canary_cost_usd,
            "canary_hard_cap_usd": canary_hard_cap_usd,
            "missing_graph_hard_cap_usd": missing_graph_hard_cap_usd,
            "campaign_api_hard_cap_usd": campaign_api_hard_cap_usd,
        },
        "execution_contract": {
            "minimum_media_ready_age_seconds": MINIMUM_MEDIA_READY_AGE_SECONDS,
            "batch_request_count": CANARY_REQUEST_COUNT,
            "temporary_upload_count": CANARY_VIDEO_COUNT,
            "cleanup_on_submission_failure": True,
            "cleanup_on_terminal_collection": True,
            "policy_identity_sent_to_provider": False,
            "physical_outcome_sent_to_provider": False,
            "physical_ground_truth_pixels_uploaded": False,
            "ranking_or_confirmation_credit": False,
        },
        "shared_paid_lane_admission": shared,
        "blockers": sorted(set(blockers)),
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
    admission["admission_sha256"] = canonical_sha256(admission)
    return admission


def _require_transport_canary_paid_admission(
    admission: Mapping[str, Any] | None,
    *,
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source_commit: str,
) -> PaidResourceAdmissionGrant:
    if not isinstance(admission, Mapping):
        raise GeminiTransportCanaryError("transport_canary_paid_admission_missing")
    pair = _selected_pair(inventory)
    request_ids = [str(pair[side]["source_request_id"]) for side in ("episode_a", "episode_b")]
    contract = admission.get("execution_contract") or {}
    cost = admission.get("cost_boundary") or {}
    protocol = complete_graph_diagnostic_protocol()
    blockers: list[str] = []
    if canonical_sha256(_without_digest(admission, "admission_sha256")) != admission.get(
        "admission_sha256"
    ):
        blockers.append("transport_canary_paid_admission_digest_invalid")
    if admission.get("schema_version") != ADMISSION_SCHEMA:
        blockers.append("transport_canary_paid_admission_schema_invalid")
    if admission.get("status") != "admitted" or admission.get("blockers") not in ([], None):
        blockers.append("transport_canary_paid_admission_not_admitted")
    if (
        admission.get("source_commit") != source_commit
        or admission.get("inventory_sha256") != inventory.get("inventory_sha256")
        or admission.get("manifest_sha256") != manifest.get("manifest_sha256")
        or admission.get("pair_id") != pair.get("pair_id")
        or admission.get("request_ids") != request_ids
    ):
        blockers.append("transport_canary_paid_admission_input_binding_mismatch")
    if (
        admission.get("arm_id") != COMPLETE_GRAPH_ARM_ID
        or admission.get("model") != COMPLETE_GRAPH_MODEL
        or admission.get("claim_class") != "post_unseal_transport_diagnostic_only"
        or admission.get("protocol_sha256") != protocol["protocol_sha256"]
        or admission.get("prompt_sha256") != protocol["shared_judging_contract"]["prompt_sha256"]
        or admission.get("request_count") != CANARY_REQUEST_COUNT
        or admission.get("video_count") != CANARY_VIDEO_COUNT
        or contract.get("minimum_media_ready_age_seconds") != MINIMUM_MEDIA_READY_AGE_SECONDS
        or contract.get("ranking_or_confirmation_credit") is not False
        or contract.get("physical_ground_truth_pixels_uploaded") is not False
    ):
        blockers.append("transport_canary_paid_admission_claim_boundary_invalid")
    cost_values = (
        cost.get("realized_api_spend_before_canary_usd"),
        cost.get("realized_missing_graph_spend_before_canary_usd"),
        cost.get("projected_canary_cost_usd"),
        cost.get("canary_hard_cap_usd"),
        cost.get("missing_graph_hard_cap_usd"),
        cost.get("campaign_api_hard_cap_usd"),
    )
    if any(not isinstance(value, (int, float)) or value < 0 for value in cost_values) or not (
        0 < cost.get("canary_hard_cap_usd", 0) <= cost.get("missing_graph_hard_cap_usd", 0)
        and 0 < cost.get("campaign_api_hard_cap_usd", 0)
    ):
        blockers.append("transport_canary_paid_admission_cost_value_invalid")
    elif (
        cost.get("projected_canary_cost_usd", float("inf")) > cost.get("canary_hard_cap_usd", -1)
        or cost.get("realized_missing_graph_spend_before_canary_usd", float("inf"))
        + cost.get("projected_canary_cost_usd", float("inf"))
        > cost.get("missing_graph_hard_cap_usd", -1)
        or cost.get("realized_api_spend_before_canary_usd", float("inf"))
        + cost.get("projected_canary_cost_usd", float("inf"))
        > cost.get("campaign_api_hard_cap_usd", -1)
        or cost.get("canary_hard_cap_usd", float("inf")) > 0.05
        or cost.get("missing_graph_hard_cap_usd", float("inf")) > 9.0
        or cost.get("campaign_api_hard_cap_usd", float("inf")) > 25.0
    ):
        blockers.append("transport_canary_paid_admission_cost_boundary_invalid")
    if blockers:
        raise GeminiTransportCanaryError(";".join(sorted(set(blockers))))
    try:
        return require_paid_resource_admission(
            admission["shared_paid_lane_admission"],
            resource_class=PAID_RESOURCE_CLASS,
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except Exception as exc:
        raise GeminiTransportCanaryError("shared_paid_resource_admission_rejected") from exc


def stage_transport_canary_media(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    ledger_path: str | Path,
    source_commit: str,
    paid_admission: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Upload exactly the two videos used by the frozen first missing edge."""

    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        raise GeminiTransportCanaryError(f"missing_env_{GATE_ENV}")
    target = Path(ledger_path)
    if target.exists():
        raise GeminiTransportCanaryError("transport_canary_media_ledger_already_exists")
    pair = _selected_pair(inventory)
    manifest_rows = _validated_manifest_rows(manifest)
    paid_grant = _require_transport_canary_paid_admission(
        paid_admission,
        inventory=inventory,
        manifest=manifest,
        source_commit=source_commit,
    )
    key_path = _secure_file(api_key_file)
    from google import genai

    client = genai.Client(api_key=key_path.read_text(encoding="utf-8").strip())
    uploads: list[dict[str, Any]] = []
    uploaded_objects: list[Any] = []
    try:
        for side in ("episode_a", "episode_b"):
            require_paid_resource_admission_grant(
                paid_grant,
                resource_class=PAID_RESOURCE_CLASS,
            )
            request_id = str(pair[side]["source_request_id"])
            uploaded, receipt = _upload_video(client, manifest_rows[request_id])
            uploaded_objects.append(uploaded)
            uploads.append(receipt)
        expected_ids = {str(pair[side]["source_request_id"]) for side in ("episode_a", "episode_b")}
        if (
            len(uploads) != CANARY_VIDEO_COUNT
            or {str(row.get("request_id") or "") for row in uploads} != expected_ids
            or any(
                not row.get("provider_file_name")
                or not row.get("provider_file_uri")
                or row.get("provider_mime_type") != "video/mp4"
                for row in uploads
            )
        ):
            raise GeminiTransportCanaryError("transport_canary_upload_receipts_invalid")
        ledger: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "status": "ready",
            "arm_id": COMPLETE_GRAPH_ARM_ID,
            "pair_id": pair["pair_id"],
            "inventory_sha256": inventory["inventory_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "paid_admission_sha256": paid_admission["admission_sha256"],
            "uploaded_video_count": len(uploads),
            "uploads": uploads,
            "source_commit": source_commit,
            "policy_identity_sent_to_provider": False,
            "physical_outcome_sent_to_provider": False,
            "physical_ground_truth_pixels_uploaded": False,
            "credential_path_or_value_persisted": False,
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        ledger["ledger_sha256"] = canonical_sha256(ledger)
        write_json(target, ledger)
        return ledger
    except Exception:
        for uploaded in uploaded_objects:
            _delete_uploaded_file_best_effort(client, uploaded)
        raise


def _ledger_age_seconds(ledger: Mapping[str, Any], *, now: datetime | None = None) -> float:
    try:
        ready_at = datetime.fromisoformat(
            str(ledger.get("updated_at_utc") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise GeminiTransportCanaryError("transport_canary_media_ready_time_invalid") from exc
    if ready_at.tzinfo is None:
        raise GeminiTransportCanaryError("transport_canary_media_ready_time_not_utc")
    return ((now or datetime.now(timezone.utc)) - ready_at).total_seconds()


def _delete_upload_receipts(
    client: Any, uploads: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for upload in uploads:
        name = str(upload["provider_file_name"])
        try:
            client.files.delete(name=name)
            results.append({"provider_file_name": name, "deleted": True})
        except Exception as exc:
            results.append(
                {"provider_file_name": name, "deleted": False, "error_type": type(exc).__name__}
            )
    return results


def submit_transport_canary(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    receipt_path: str | Path,
    source_commit: str,
    paid_admission: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create exactly one paid Batch request after the frozen propagation grace."""

    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        raise GeminiTransportCanaryError(f"missing_env_{GATE_ENV}")
    target = Path(receipt_path)
    if target.exists():
        raise GeminiTransportCanaryError("transport_canary_submission_receipt_already_exists")
    pair = _selected_pair(inventory)
    paid_grant = _require_transport_canary_paid_admission(
        paid_admission,
        inventory=inventory,
        manifest=manifest,
        source_commit=source_commit,
    )
    ledger_payload = _without_digest(ledger, "ledger_sha256")
    ledger_uploads = ledger.get("uploads")
    expected_request_ids = {
        str(pair[side]["source_request_id"]) for side in ("episode_a", "episode_b")
    }
    observed_request_ids = (
        {str(row.get("request_id") or "") for row in ledger_uploads}
        if isinstance(ledger_uploads, list)
        else set()
    )
    if (
        canonical_sha256(ledger_payload) != ledger.get("ledger_sha256")
        or ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("status") != "ready"
        or ledger.get("pair_id") != pair.get("pair_id")
        or ledger.get("inventory_sha256") != inventory.get("inventory_sha256")
        or ledger.get("manifest_sha256") != manifest.get("manifest_sha256")
        or ledger.get("paid_admission_sha256") != paid_admission.get("admission_sha256")
        or ledger.get("uploaded_video_count") != CANARY_VIDEO_COUNT
        or not isinstance(ledger_uploads, list)
        or len(ledger_uploads) != CANARY_VIDEO_COUNT
        or observed_request_ids != expected_request_ids
        or any(
            not row.get("provider_file_name")
            or not row.get("provider_file_uri")
            or row.get("provider_mime_type") != "video/mp4"
            for row in ledger_uploads
        )
        or ledger.get("source_commit") != source_commit
    ):
        raise GeminiTransportCanaryError("transport_canary_media_ledger_invalid_or_unbound")
    if _ledger_age_seconds(ledger) < MINIMUM_MEDIA_READY_AGE_SECONDS:
        raise GeminiTransportCanaryError("transport_canary_media_propagation_grace_not_elapsed")
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
    request = _build_inline_request(
        pair,
        files_by_id[str(pair["episode_a"]["source_request_id"])],
        files_by_id[str(pair["episode_b"]["source_request_id"])],
        types_module=types,
    )
    try:
        require_paid_resource_admission_grant(paid_grant, resource_class=PAID_RESOURCE_CLASS)
        job = client.batches.create(
            model=COMPLETE_GRAPH_MODEL,
            src=[request],
            config=types.CreateBatchJobConfig(
                display_name="blueprint-roboarena-gemini36-complete-graph-canary-v1"
            ),
        )
    except Exception as exc:
        deletions = _delete_upload_receipts(client, ledger["uploads"])
        failed: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "failed_before_batch_creation",
            "batch_name": None,
            "model": COMPLETE_GRAPH_MODEL,
            "arm_id": COMPLETE_GRAPH_ARM_ID,
            "pair_ids": [str(pair["pair_id"])],
            "request_count": CANARY_REQUEST_COUNT,
            "unique_video_count": CANARY_VIDEO_COUNT,
            "provider_error": _provider_error_payload(exc),
            "deletions": deletions,
            "all_task_media_deleted": all(row["deleted"] for row in deletions),
            "provider_generation_rows_created": 0,
            "inventory_sha256": inventory["inventory_sha256"],
            "media_ledger_sha256": ledger["ledger_sha256"],
            "paid_admission_sha256": paid_admission["admission_sha256"],
            "source_commit": source_commit,
            "failed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "policy_identity_sent_to_provider": False,
            "physical_outcome_sent_to_provider": False,
            "physical_ground_truth_pixels_uploaded": False,
            "ranking_or_confirmation_credit": False,
            "credential_path_or_value_persisted": False,
        }
        failed["receipt_sha256"] = canonical_sha256(failed)
        write_json(target, failed)
        return failed
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": _job_state(job),
        "batch_name": str(job.name),
        "model": COMPLETE_GRAPH_MODEL,
        "arm_id": COMPLETE_GRAPH_ARM_ID,
        "pair_ids": [str(pair["pair_id"])],
        "request_count": CANARY_REQUEST_COUNT,
        "unique_video_count": CANARY_VIDEO_COUNT,
        "uploads": ledger["uploads"],
        "cleanup_uploads_on_terminal": True,
        "inventory_sha256": inventory["inventory_sha256"],
        "media_ledger_sha256": ledger["ledger_sha256"],
        "paid_admission_sha256": paid_admission["admission_sha256"],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "provider_called": True,
        "data_uploaded": True,
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_uploaded": False,
        "ranking_or_confirmation_credit": False,
        "credential_path_or_value_persisted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(target, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    admit = commands.add_parser("admit")
    admit.add_argument("--inventory", required=True)
    admit.add_argument("--native-video-manifest", required=True)
    admit.add_argument("--api-key-file", required=True)
    admit.add_argument("--source-commit", required=True)
    admit.add_argument("--realized-api-spend-usd", type=float, required=True)
    admit.add_argument("--realized-missing-graph-spend-usd", type=float, required=True)
    admit.add_argument("--projected-canary-cost-usd", type=float, required=True)
    admit.add_argument("--canary-hard-cap-usd", type=float, required=True)
    admit.add_argument("--missing-graph-hard-cap-usd", type=float, required=True)
    admit.add_argument("--campaign-api-hard-cap-usd", type=float, required=True)
    admit.add_argument("--output", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--inventory", required=True)
    stage.add_argument("--native-video-manifest", required=True)
    stage.add_argument("--api-key-file", required=True)
    stage.add_argument("--ledger", required=True)
    stage.add_argument("--source-commit", required=True)
    stage.add_argument("--paid-admission", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--inventory", required=True)
    submit.add_argument("--native-video-manifest", required=True)
    submit.add_argument("--ledger", required=True)
    submit.add_argument("--api-key-file", required=True)
    submit.add_argument("--receipt", required=True)
    submit.add_argument("--source-commit", required=True)
    submit.add_argument("--paid-admission", required=True)
    args = parser.parse_args(argv)
    inventory_path = Path(args.inventory)
    manifest_path = Path(args.native_video_manifest)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.command == "admit":
        try:
            _secure_file(args.api_key_file)
            credential_ready = True
        except ValueError:
            credential_ready = False
        result = build_transport_canary_paid_admission(
            inventory,
            manifest,
            inventory_file_sha256=file_sha256(inventory_path),
            manifest_file_sha256=file_sha256(manifest_path),
            source_commit=args.source_commit,
            realized_api_spend_usd=args.realized_api_spend_usd,
            realized_missing_graph_spend_usd=args.realized_missing_graph_spend_usd,
            projected_canary_cost_usd=args.projected_canary_cost_usd,
            canary_hard_cap_usd=args.canary_hard_cap_usd,
            missing_graph_hard_cap_usd=args.missing_graph_hard_cap_usd,
            campaign_api_hard_cap_usd=args.campaign_api_hard_cap_usd,
            credential_ready=credential_ready,
        )
        write_json(Path(args.output), result)
    elif args.command == "stage":
        result = stage_transport_canary_media(
            inventory,
            manifest,
            api_key_file=args.api_key_file,
            ledger_path=args.ledger,
            source_commit=args.source_commit,
            paid_admission=json.loads(Path(args.paid_admission).read_text(encoding="utf-8")),
        )
    else:
        result = submit_transport_canary(
            inventory,
            manifest,
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
        if result.get("status") in {"admitted", "ready", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
