"""Gemini Batch transport for native-video pairwise evaluator diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_evaluator_diagnostic import (
    GEMINI_MODEL,
    PAIR_OUTPUT_SCHEMA,
    PAIR_PROMPT,
    PAIR_RESULT_SCHEMA_VERSION,
    diagnostic_protocol,
)
from .policy_ranking_evaluator_diagnostic_gemini import (
    GATE_ENV,
    MAX_OUTPUT_TOKENS,
    _response_usage,
    _secure_file,
    _upload_video,
    _validate_payload,
    _validated_manifest_rows,
)
from .policy_ranking_roboarena_calibration import canonical_sha256


SCHEMA_VERSION = "policy_ranking_gemini_pair_batch.v1"
BATCH_RESPONSE_SCHEMA = {
    key: value for key, value in PAIR_OUTPUT_SCHEMA.items() if key != "additionalProperties"
}
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class GeminiBatchDiagnosticError(ValueError):
    """The Gemini Batch job or result set is invalid."""


def _delete_uploaded_file_best_effort(client: Any, uploaded: Any) -> bool:
    try:
        client.files.delete(name=uploaded.name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _job_state(job: Any) -> str:
    state = getattr(job, "state", "")
    return str(getattr(state, "name", state) or "")


def _build_inline_request(
    pair: Mapping[str, Any], video_a: Any, video_b: Any, *, types_module: Any
) -> Any:
    metadata = {
        "task_instruction": pair["task_instruction"],
        "episode_a": "first attached complete video",
        "episode_b": "second attached complete video",
        "claim_boundary": "generated_episode_pair_diagnostic_not_physical_success",
    }
    return types_module.InlinedRequest(
        contents=[
            PAIR_PROMPT + "\n" + json.dumps(metadata, sort_keys=True),
            "EPISODE A — complete chronological generated-only video",
            video_a,
            "EPISODE B — complete chronological generated-only video",
            video_b,
        ],
        metadata={"pair_id": str(pair["pair_id"])},
        config=types_module.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BATCH_RESPONSE_SCHEMA,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types_module.ThinkingConfig(
                thinking_level="MEDIUM", include_thoughts=False
            ),
        ),
    )


def submit_pilot(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    receipt_path: str | Path,
    source_commit: str,
    count: int = 7,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        raise GeminiBatchDiagnosticError(f"missing_env_{GATE_ENV}")
    protocol = diagnostic_protocol()
    if (
        inventory.get("status") != "ready"
        or inventory.get("pair_count") != 441
        or inventory.get("protocol_sha256") != protocol["protocol_sha256"]
        or count != 7
    ):
        raise GeminiBatchDiagnosticError("pair_inventory_or_pilot_count_invalid")
    rows = _validated_manifest_rows(manifest)
    key = _secure_file(api_key_file)
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key.read_text(encoding="utf-8").strip())
    pairs = inventory["pairs"][:count]
    request_ids = sorted(
        {
            str(pair[side]["source_request_id"])
            for pair in pairs
            for side in ("episode_a", "episode_b")
        }
    )
    uploaded_by_id: dict[str, Any] = {}
    upload_receipts: list[dict[str, Any]] = []
    try:
        for request_id in request_ids:
            uploaded, upload_receipt = _upload_video(client, rows[request_id])
            uploaded_by_id[request_id] = uploaded
            upload_receipts.append(upload_receipt)
        requests = [
            _build_inline_request(
                pair,
                uploaded_by_id[str(pair["episode_a"]["source_request_id"])],
                uploaded_by_id[str(pair["episode_b"]["source_request_id"])],
                types_module=types,
            )
            for pair in pairs
        ]
        job = client.batches.create(
            model=GEMINI_MODEL,
            src=requests,
            config=types.CreateBatchJobConfig(
                display_name="blueprint-roboarena-gemini36-pilot-v1"
            ),
        )
    except Exception:
        for uploaded in uploaded_by_id.values():
            _delete_uploaded_file_best_effort(client, uploaded)
        raise
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": _job_state(job),
        "batch_name": str(job.name),
        "model": GEMINI_MODEL,
        "arm_id": "gemini36_flash_native_video",
        "pair_ids": [str(pair["pair_id"]) for pair in pairs],
        "request_count": count,
        "unique_video_count": len(uploaded_by_id),
        "uploads": upload_receipts,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "provider_called": True,
        "data_uploaded": True,
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_uploaded": False,
        "credential_path_or_value_persisted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(Path(receipt_path), receipt)
    return receipt


def _delete_uploads(client: Any, receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    deletions: list[dict[str, Any]] = []
    for upload in receipt["uploads"]:
        name = str(upload["provider_file_name"])
        try:
            client.files.delete(name=name)
            deletions.append({"provider_file_name": name, "deleted": True})
        except Exception as exc:
            deletions.append(
                {"provider_file_name": name, "deleted": False, "error_type": type(exc).__name__}
            )
    return deletions


def collect_pilot(
    receipt: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if canonical_sha256(payload) != receipt.get("receipt_sha256"):
        raise GeminiBatchDiagnosticError("batch_receipt_digest_invalid")
    key = _secure_file(api_key_file)
    from google import genai

    client = genai.Client(api_key=key.read_text(encoding="utf-8").strip())
    job = client.batches.get(name=str(receipt["batch_name"]))
    state = _job_state(job)
    terminal = state in TERMINAL_STATES
    if state != "JOB_STATE_SUCCEEDED":
        deletions = _delete_uploads(client, receipt) if terminal else []
        report = {
            "schema_version": "policy_ranking_gemini_pair_batch_collection.v1",
            "status": state,
            "batch_name": receipt["batch_name"],
            "completed": False,
            "terminal": terminal,
            "provider_error": (
                job.error.model_dump(mode="json") if getattr(job, "error", None) else None
            ),
            "completion_stats": (
                job.completion_stats.model_dump(mode="json")
                if getattr(job, "completion_stats", None)
                else None
            ),
            "deletions": deletions,
        }
        report["report_sha256"] = canonical_sha256(report)
        write_json(Path(output_path), report)
        return report
    responses = list(getattr(getattr(job, "dest", None), "inlined_responses", None) or [])
    if len(responses) != len(receipt["pair_ids"]):
        raise GeminiBatchDiagnosticError("batch_inline_response_count_invalid")
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    all_usage: list[dict[str, Any]] = []
    for pair_id, inline in zip(receipt["pair_ids"], responses, strict=True):
        response = getattr(inline, "response", None)
        error = getattr(inline, "error", None)
        if response is None or error is not None:
            errors.append(
                {
                    "pair_id": pair_id,
                    "error": error.model_dump(mode="json") if error is not None else None,
                }
            )
            continue
        raw_text = str(response.text or "")
        usage = _response_usage(response)
        all_usage.append(usage)
        try:
            structured = json.loads(raw_text)
            _validate_payload(structured)
        except Exception as exc:
            errors.append(
                {
                    "pair_id": pair_id,
                    "error_type": type(exc).__name__,
                    "error_code": "structured_output_invalid",
                    "response_id": str(getattr(response, "response_id", "") or ""),
                    "raw_response_text": raw_text,
                    "usage": usage,
                }
            )
            continue
        result: dict[str, Any] = {
            "schema_version": PAIR_RESULT_SCHEMA_VERSION,
            "pair_id": pair_id,
            "arm_id": "gemini36_flash_native_video",
            "provider": "google_gemini_api",
            "model": GEMINI_MODEL,
            "response_id": str(getattr(response, "response_id", "") or ""),
            "structured_response": structured,
            "usage": usage,
            "transport": "gemini_batch_api_native_video",
            "policy_identity_sent_to_provider": False,
            "physical_outcome_sent_to_provider": False,
            "physical_ground_truth_pixels_sent_to_provider": False,
            "claim_class": "post_unseal_diagnostic_only",
        }
        result["result_sha256"] = canonical_sha256(result)
        results.append(result)
    deletions = _delete_uploads(client, receipt)
    deleted_all = len(deletions) == receipt["unique_video_count"] and all(
        row["deleted"] for row in deletions
    )
    report = {
        "schema_version": "policy_ranking_gemini_pair_batch_collection.v1",
        "status": (
            "completed"
            if len(results) == len(receipt["pair_ids"]) and not errors and deleted_all
            else "failed"
        ),
        "provider_job_state": state,
        "batch_name": receipt["batch_name"],
        "completed": True,
        "result_count": len(results),
        "error_count": len(errors),
        "estimated_batch_cost_usd": sum(
            usage["projected_batch_cost_same_usage_usd"] for usage in all_usage
        ),
        "results": results,
        "errors": errors,
        "deletions": deletions,
        "temporary_video_files_deleted": deleted_all,
        "batch_result_retention": "provider_default_up_to_six_weeks",
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(Path(output_path), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--inventory", required=True)
    submit.add_argument("--native-video-manifest", required=True)
    submit.add_argument("--api-key-file", required=True)
    submit.add_argument("--receipt", required=True)
    submit.add_argument("--source-commit", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--receipt", required=True)
    collect.add_argument("--api-key-file", required=True)
    collect.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "submit":
        result = submit_pilot(
            json.loads(Path(args.inventory).read_text(encoding="utf-8")),
            json.loads(Path(args.native_video_manifest).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            receipt_path=args.receipt,
            source_commit=args.source_commit,
        )
    else:
        result = collect_pilot(
            json.loads(Path(args.receipt).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            output_path=args.output,
        )
    print(json.dumps({key: value for key, value in result.items() if key != "results"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
