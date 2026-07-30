"""Asynchronous OpenAI Batch transport for pairwise diagnostic matrices."""

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
    PAIR_RESULT_SCHEMA_VERSION,
    complete_graph_diagnostic_protocol,
)
from .policy_ranking_evaluator_diagnostic_openai import (
    GATE_ENV,
    MODEL_PRICES_STANDARD,
    _secure_file,
    _request_config,
    _validate_payload,
    build_response_body,
)
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256
from .policy_ranking_roboarena_evaluator_openai import validate_rotation_attestation


SCHEMA_VERSION = "policy_ranking_openai_pair_batch.v1"
COMPLETE_GRAPH_PAIR_COUNT = 1323
TRANSPORT_REPAIR_GRAPH_KIND = "registered_complete_graph_subset"
TRANSPORT_REPAIR_ARM_ID = "gpt5_complete_graph_transport_repair"
TRANSPORT_REPAIR_MAX_PAIR_COUNT = 42
TRANSPORT_REPAIR_MAX_OUTPUT_TOKENS = 6000


class OpenAIBatchDiagnosticError(ValueError):
    """The batch shard, job, or response set is invalid."""


def _validate_inventory(inventory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if canonical_sha256(payload) != inventory.get("inventory_sha256"):
        raise OpenAIBatchDiagnosticError("pair_inventory_digest_invalid")
    pairs = inventory.get("pairs")
    pair_count = inventory.get("pair_count")
    complete_protocol_sha256 = complete_graph_diagnostic_protocol()["protocol_sha256"]
    standard_matrix = pair_count in {441, COMPLETE_GRAPH_PAIR_COUNT}
    registered_repair_subset = (
        inventory.get("comparison_graph_kind") == TRANSPORT_REPAIR_GRAPH_KIND
        and isinstance(pair_count, int)
        and not isinstance(pair_count, bool)
        and 1 <= pair_count <= TRANSPORT_REPAIR_MAX_PAIR_COUNT
        and inventory.get("protocol_sha256") == complete_protocol_sha256
        and isinstance(inventory.get("parent_inventory_sha256"), str)
        and len(inventory["parent_inventory_sha256"]) == 64
        and inventory.get("outcome_labels_accessed_to_build_pairs") is False
    )
    if (
        inventory.get("status") != "ready"
        or not (standard_matrix or registered_repair_subset)
        or not isinstance(pairs, list)
        or len(pairs) != pair_count
        or len({str(pair.get("pair_id")) for pair in pairs}) != pair_count
        or (
            pair_count == COMPLETE_GRAPH_PAIR_COUNT
            and inventory.get("protocol_sha256") != complete_protocol_sha256
        )
    ):
        raise OpenAIBatchDiagnosticError("pair_inventory_not_ready_supported_matrix")
    return pairs


def prepare_shard(
    inventory: Mapping[str, Any],
    *,
    model: str,
    offset: int,
    count: int,
    jsonl_path: str | Path,
    manifest_path: str | Path,
    source_commit: str,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    arm_id: str | None = None,
) -> dict[str, Any]:
    pairs = _validate_inventory(inventory)
    if model not in MODEL_PRICES_STANDARD:
        raise OpenAIBatchDiagnosticError("unregistered_openai_model")
    if offset < 0 or count <= 0 or offset + count > len(pairs) or count > 24:
        raise OpenAIBatchDiagnosticError("batch_shard_bounds_invalid")
    resolved_effort, resolved_tokens, resolved_arm = _request_config(
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        arm_id=arm_id,
    )
    if len(pairs) == COMPLETE_GRAPH_PAIR_COUNT and (
        model != "gpt-5-2025-08-07"
        or resolved_effort != "medium"
        or resolved_tokens != 4000
        or resolved_arm != "gpt5_complete_graph"
    ):
        raise OpenAIBatchDiagnosticError("complete_graph_gpt5_config_mismatch")
    if inventory.get("comparison_graph_kind") == TRANSPORT_REPAIR_GRAPH_KIND and (
        model != "gpt-5-2025-08-07"
        or resolved_effort != "medium"
        or resolved_tokens != TRANSPORT_REPAIR_MAX_OUTPUT_TOKENS
        or resolved_arm != TRANSPORT_REPAIR_ARM_ID
    ):
        raise OpenAIBatchDiagnosticError("complete_graph_transport_repair_config_mismatch")
    selected = pairs[offset : offset + count]
    target = Path(jsonl_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for pair in selected:
            line = {
                "custom_id": pair["pair_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": build_response_body(
                    pair,
                    model=model,
                    reasoning_effort=resolved_effort,
                    max_output_tokens=resolved_tokens,
                ),
            }
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
    shard_core = {
        "model": model,
        "offset": offset,
        "count": count,
        "pair_ids": [pair["pair_id"] for pair in selected],
        "inventory_sha256": inventory["inventory_sha256"],
        "reasoning_effort": resolved_effort,
        "max_output_tokens": resolved_tokens,
        "arm_id": resolved_arm,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "arm_id": shard_core["arm_id"],
        **shard_core,
        "shard_id": canonical_sha256(shard_core),
        "jsonl_path": str(target),
        "jsonl_sha256": file_sha256(target),
        "jsonl_size_bytes": target.stat().st_size,
        "source_commit": source_commit,
        "policy_identity_in_batch_body": False,
        "physical_outcome_in_batch_body": False,
        "physical_ground_truth_pixels_in_batch_body": False,
        "provider_called": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(Path(manifest_path), manifest)
    return manifest


def submit_shard(
    manifest: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    rotation_attestation_file: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        raise OpenAIBatchDiagnosticError(f"missing_env_{GATE_ENV}")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != manifest.get("manifest_sha256"):
        raise OpenAIBatchDiagnosticError("batch_manifest_digest_invalid")
    jsonl = Path(str(manifest["jsonl_path"]))
    if not jsonl.is_file() or file_sha256(jsonl) != manifest.get("jsonl_sha256"):
        raise OpenAIBatchDiagnosticError("batch_jsonl_changed_before_upload")
    key = _secure_file(api_key_file)
    attestation = validate_rotation_attestation(rotation_attestation_file)
    from openai import OpenAI

    client = OpenAI(api_key=key.read_text(encoding="utf-8").strip())
    uploaded = client.files.create(
        file=jsonl,
        purpose="batch",
        expires_after={"anchor": "created_at", "seconds": 172_800},
    )
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "experiment": "roboarena-evaluator-diagnostic",
            "shard_id": manifest["shard_id"],
            "arm_id": manifest["arm_id"],
        },
        output_expires_after={"anchor": "created_at", "seconds": 172_800},
    )
    receipt: dict[str, Any] = {
        "schema_version": "policy_ranking_openai_pair_batch_submission.v1",
        "status": str(batch.status),
        "shard_id": manifest["shard_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "model": manifest["model"],
        "arm_id": manifest["arm_id"],
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_count": manifest["count"],
        "provider_called": True,
        "data_uploaded": True,
        "physical_ground_truth_pixels_uploaded": False,
        "credential_path_or_value_persisted": False,
        "rotation_attestation": attestation,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    write_json(Path(receipt_path), receipt)
    return receipt


def _output_text(body: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                chunks.append(str(content.get("text") or ""))
    return "".join(chunks)


def _usage(body: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    details = usage.get("input_tokens_details")
    cached = int(details.get("cached_tokens") or 0) if isinstance(details, Mapping) else 0
    prices = MODEL_PRICES_STANDARD[model]
    standard = (
        (input_tokens - cached) * prices["input"]
        + cached * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output_tokens,
        "batch_cost_usd": standard * 0.5,
    }


def _write_collection_report(
    output_path: str | Path,
    report: dict[str, Any],
    *,
    client: Any,
    input_file_id: str,
    delete_input: bool,
) -> dict[str, Any]:
    """Persist provider evidence before attempting input-file cleanup."""

    target = Path(output_path)
    report["input_file_deleted"] = False
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    write_json(target, report)
    if not delete_input:
        return report
    try:
        client.files.delete(input_file_id)
        report["input_file_deleted"] = True
        report.pop("input_file_deletion_error_type", None)
    except Exception as exc:  # noqa: BLE001 - cleanup failure is retained as evidence
        report["input_file_deletion_error_type"] = type(exc).__name__
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    write_json(target, report)
    return report


def collect_shard(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    key = _secure_file(api_key_file)
    from openai import OpenAI

    client = OpenAI(api_key=key.read_text(encoding="utf-8").strip())
    batch = client.batches.retrieve(str(receipt["batch_id"]))
    if batch.status != "completed":
        errors = getattr(batch, "errors", None)
        if hasattr(errors, "model_dump"):
            errors = errors.model_dump(mode="json")
        counts = getattr(batch, "request_counts", None)
        if hasattr(counts, "model_dump"):
            counts = counts.model_dump(mode="json")
        usage = getattr(batch, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump(mode="json")
        terminal = str(batch.status) in {"failed", "expired", "cancelled"}
        report = {
            "schema_version": "policy_ranking_openai_pair_batch_collection.v1",
            "status": str(batch.status),
            "batch_id": batch.id,
            "shard_id": receipt["shard_id"],
            "completed": False,
            "provider_errors": errors,
            "request_counts": counts,
            "usage": usage,
            "terminal": terminal,
            "provider_output_file_id": getattr(batch, "output_file_id", None),
            "provider_error_file_id": getattr(batch, "error_file_id", None),
        }
        return _write_collection_report(
            output_path,
            report,
            client=client,
            input_file_id=str(receipt["input_file_id"]),
            delete_input=terminal,
        )
    provider_files = {
        "output": getattr(batch, "output_file_id", None),
        "error": getattr(batch, "error_file_id", None),
    }
    if not any(provider_files.values()):
        raise OpenAIBatchDiagnosticError("completed_batch_result_files_missing")
    pair_by_id = {pair["pair_id"]: pair for pair in _validate_inventory(inventory)}
    expected = set(manifest["pair_ids"])
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    all_usage: list[dict[str, Any]] = []
    observed: set[str] = set()
    for provider_file_role, provider_file_id in provider_files.items():
        if not provider_file_id:
            continue
        raw = client.files.content(provider_file_id).text
        for raw_line in raw.splitlines():
            if not raw_line.strip():
                continue
            line = json.loads(raw_line)
            pair_id = str(line.get("custom_id") or "")
            response = (
                line.get("response") if isinstance(line.get("response"), Mapping) else {}
            )
            body = response.get("body") if isinstance(response.get("body"), Mapping) else {}
            if pair_id not in expected or pair_id not in pair_by_id:
                raise OpenAIBatchDiagnosticError("batch_output_pair_id_unexpected")
            if pair_id in observed:
                raise OpenAIBatchDiagnosticError("batch_output_pair_id_duplicate")
            observed.add(pair_id)
            provenance = {
                "provider_file_role": provider_file_role,
                "provider_file_id": str(provider_file_id),
                "batch_row_id": line.get("id"),
                "request_id": response.get("request_id"),
            }
            top_level_error = line.get("error")
            response_body_error = body.get("error")
            if (
                response.get("status_code") != 200
                or top_level_error
                or response_body_error
            ):
                error_code = None
                if isinstance(response_body_error, Mapping):
                    error_code = response_body_error.get("code")
                elif isinstance(top_level_error, Mapping):
                    error_code = top_level_error.get("code")
                errors.append(
                    {
                        "pair_id": pair_id,
                        "error_type": "provider_batch_row_error",
                        "error_code": error_code,
                        "status_code": response.get("status_code"),
                        "top_level_error": top_level_error,
                        "response_body_error": response_body_error,
                        **provenance,
                    }
                )
                continue
            usage = _usage(body, model=manifest["model"])
            all_usage.append(usage)
            raw_text = _output_text(body)
            try:
                if body.get("status") != "completed":
                    raise OpenAIBatchDiagnosticError("batch_row_response_incomplete")
                structured = json.loads(raw_text)
                _validate_payload(structured)
            except Exception as exc:
                errors.append(
                    {
                        "pair_id": pair_id,
                        "error_type": type(exc).__name__,
                        "error_code": "structured_output_invalid_or_incomplete",
                        "response_id": body.get("id"),
                        "response_status": body.get("status"),
                        "incomplete_details": body.get("incomplete_details"),
                        "raw_response_text": raw_text,
                        "usage": usage,
                        **provenance,
                    }
                )
                continue
            result: dict[str, Any] = {
                "schema_version": PAIR_RESULT_SCHEMA_VERSION,
                "pair_id": pair_id,
                "arm_id": manifest["arm_id"],
                "provider": "openai",
                "model": manifest["model"],
                "response_id": body.get("id"),
                "response_status": body.get("status"),
                "structured_response": structured,
                "usage": usage,
                "transport": "openai_batch_api",
                "policy_identity_sent_to_provider": False,
                "physical_outcome_sent_to_provider": False,
                "physical_ground_truth_pixels_sent_to_provider": False,
                "claim_class": "post_unseal_diagnostic_only",
                **provenance,
            }
            result["result_sha256"] = canonical_sha256(result)
            results.append(result)
    if observed != expected:
        raise OpenAIBatchDiagnosticError("batch_output_pair_coverage_incomplete")
    results.sort(key=lambda value: value["pair_id"])
    report = {
        "schema_version": "policy_ranking_openai_pair_batch_collection.v1",
        "status": "completed" if len(results) == len(expected) and not errors else "failed",
        "batch_id": batch.id,
        "shard_id": receipt["shard_id"],
        "completed": True,
        "result_count": len(results),
        "error_count": len(errors),
        "estimated_batch_cost_usd": sum(row["batch_cost_usd"] for row in all_usage),
        "results": results,
        "errors": errors,
        "provider_output_file_id": provider_files["output"],
        "provider_error_file_id": provider_files["error"],
        "exact_pair_coverage_across_provider_files": True,
        "output_file_expires_automatically": True,
    }
    return _write_collection_report(
        output_path,
        report,
        client=client,
        input_file_id=str(receipt["input_file_id"]),
        delete_input=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--inventory", required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--offset", type=int, required=True)
    prepare.add_argument("--count", type=int, required=True)
    prepare.add_argument("--jsonl", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--reasoning-effort")
    prepare.add_argument("--max-output-tokens", type=int)
    prepare.add_argument("--arm-id")
    submit = commands.add_parser("submit")
    submit.add_argument("--manifest", required=True)
    submit.add_argument("--api-key-file", required=True)
    submit.add_argument("--rotation-attestation", required=True)
    submit.add_argument("--receipt", required=True)
    collect = commands.add_parser("collect")
    collect.add_argument("--receipt", required=True)
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--inventory", required=True)
    collect.add_argument("--api-key-file", required=True)
    collect.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_shard(
            json.loads(Path(args.inventory).read_text()),
            model=args.model,
            offset=args.offset,
            count=args.count,
            jsonl_path=args.jsonl,
            manifest_path=args.manifest,
            source_commit=args.source_commit,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            arm_id=args.arm_id,
        )
    elif args.command == "submit":
        result = submit_shard(
            json.loads(Path(args.manifest).read_text()),
            api_key_file=args.api_key_file,
            rotation_attestation_file=args.rotation_attestation,
            receipt_path=args.receipt,
        )
    else:
        result = collect_shard(
            json.loads(Path(args.receipt).read_text()),
            json.loads(Path(args.manifest).read_text()),
            json.loads(Path(args.inventory).read_text()),
            api_key_file=args.api_key_file,
            output_path=args.output,
        )
    print(json.dumps({key: value for key, value in result.items() if key != "results"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
