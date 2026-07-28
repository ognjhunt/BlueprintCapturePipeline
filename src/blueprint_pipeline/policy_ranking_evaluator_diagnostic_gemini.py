"""Gemini native-video transport for the post-unseal evaluator diagnostic."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import stat
import time
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
from .policy_ranking_roboarena_calibration import canonical_sha256, file_sha256


GATE_ENV = "BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI"
INPUT_USD_PER_MILLION = 1.5
OUTPUT_USD_PER_MILLION = 7.5
MAX_OUTPUT_TOKENS = 3000


class GeminiDiagnosticError(ValueError):
    """The Gemini diagnostic request or media evidence is invalid."""


def _secure_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise GeminiDiagnosticError("gemini_key_missing_or_mode_not_0600")
    return resolved


def _validated_manifest_rows(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if canonical_sha256(payload) != manifest.get("manifest_sha256"):
        raise GeminiDiagnosticError("native_video_manifest_digest_invalid")
    if (
        manifest.get("status") != "passed"
        or manifest.get("video_count") != 441
        or manifest.get("all_physical_right_half_pixels_excluded") is not True
    ):
        raise GeminiDiagnosticError("native_video_manifest_not_passed_441")
    rows = {str(row["request_id"]): row for row in manifest["receipts"]}
    if len(rows) != 441:
        raise GeminiDiagnosticError("native_video_receipts_not_unique_441")
    return rows


def _wait_file_active(client: Any, uploaded: Any, *, timeout_seconds: float = 600) -> Any:
    deadline = time.monotonic() + timeout_seconds
    current = uploaded
    while True:
        state = str(getattr(current, "state", "") or "").upper()
        if state.endswith("ACTIVE") or not state:
            return current
        if state.endswith("FAILED"):
            raise GeminiDiagnosticError("gemini_uploaded_video_processing_failed")
        if time.monotonic() >= deadline:
            raise GeminiDiagnosticError("gemini_uploaded_video_processing_timeout")
        time.sleep(2)
        current = client.files.get(name=current.name)


def _upload_video(client: Any, receipt: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    path = Path(str(receipt["output_path"]))
    if not path.is_file() or file_sha256(path) != receipt.get("output_sha256"):
        raise GeminiDiagnosticError("native_video_changed_before_upload")
    uploaded = client.files.upload(
        file=path,
        config={
            "mime_type": "video/mp4",
            "display_name": f"blueprint-diag-{receipt['request_id'][:20]}",
        },
    )
    active = _wait_file_active(client, uploaded)
    return active, {
        "request_id": receipt["request_id"],
        "local_sha256": receipt["output_sha256"],
        "local_size_bytes": receipt["output_size_bytes"],
        "provider_file_name": str(active.name),
        "provider_file_state": str(getattr(active, "state", "")),
        "physical_ground_truth_pixels_uploaded": False,
    }


def _response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    candidates = int(getattr(usage, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
    cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
    total = int(getattr(usage, "total_token_count", 0) or 0)
    billable_output = max(candidates + thoughts, total - prompt, 0)
    standard_cost = (
        prompt * INPUT_USD_PER_MILLION + billable_output * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "candidate_tokens": candidates,
        "thinking_tokens": thoughts,
        "cached_tokens": cached,
        "total_tokens": total,
        "billable_output_tokens_conservative": billable_output,
        "standard_cost_usd": standard_cost,
        "projected_batch_cost_same_usage_usd": standard_cost * 0.5,
    }


def _validate_payload(payload: Mapping[str, Any]) -> None:
    expected = set(PAIR_OUTPUT_SCHEMA["required"])
    if set(payload) != expected:
        raise GeminiDiagnosticError("structured_fields_not_exact")
    for field in PAIR_OUTPUT_SCHEMA["required"]:
        if field not in payload:
            raise GeminiDiagnosticError(f"structured_field_missing:{field}")
    preference = payload.get("preferred_episode")
    if preference not in {"A", "B", "tie", "abstain"}:
        raise GeminiDiagnosticError("structured_preference_invalid")
    if preference == "abstain" and not payload.get("abstention_factors"):
        raise GeminiDiagnosticError("structured_abstention_factor_missing")


def score_canary(
    client: Any,
    pair: Mapping[str, Any],
    *,
    video_a: Any,
    video_b: Any,
    types_module: Any | None = None,
) -> dict[str, Any]:
    if types_module is None:
        from google.genai import types as types_module

    metadata = {
        "task_instruction": pair["task_instruction"],
        "episode_a": "first attached complete video",
        "episode_b": "second attached complete video",
        "claim_boundary": "generated_episode_pair_diagnostic_not_physical_success",
    }
    contents = [
        PAIR_PROMPT + "\n" + json.dumps(metadata, sort_keys=True),
        "EPISODE A — complete chronological generated-only video",
        video_a,
        "EPISODE B — complete chronological generated-only video",
        video_b,
    ]
    started = time.monotonic()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types_module.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=PAIR_OUTPUT_SCHEMA,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types_module.ThinkingConfig(
                thinking_level="MEDIUM", include_thoughts=False
            ),
        ),
    )
    payload = json.loads(str(response.text or ""))
    _validate_payload(payload)
    result: dict[str, Any] = {
        "schema_version": PAIR_RESULT_SCHEMA_VERSION,
        "pair_id": pair["pair_id"],
        "arm_id": "gemini36_flash_native_video",
        "provider": "google_gemini_api",
        "model": GEMINI_MODEL,
        "response_id": str(getattr(response, "response_id", "") or ""),
        "structured_response": payload,
        "usage": _response_usage(response),
        "latency_seconds": time.monotonic() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "transport": "synchronous_native_video_schema_canary_only",
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_sent_to_provider": False,
        "claim_class": "post_unseal_transport_diagnostic_only",
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def run_canary(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    api_key_file: str | Path,
    output: str | Path,
    source_commit: str,
) -> dict[str, Any]:
    if os.getenv(GATE_ENV, "").lower() not in {"1", "true", "yes"}:
        return {"status": "blocked", "blockers": [f"missing_env_{GATE_ENV}"]}
    protocol = diagnostic_protocol()
    if (
        inventory.get("status") != "ready"
        or inventory.get("pair_count") != 441
        or inventory.get("protocol_sha256") != protocol["protocol_sha256"]
    ):
        return {"status": "blocked", "blockers": ["pair_inventory_not_ready_or_bound"]}
    rows = _validated_manifest_rows(manifest)
    key = _secure_file(api_key_file)
    from google import genai

    client = genai.Client(api_key=key.read_text(encoding="utf-8").strip())
    pair = inventory["pairs"][0]
    request_a = pair["episode_a"]["source_request_id"]
    request_b = pair["episode_b"]["source_request_id"]
    uploaded: list[Any] = []
    upload_receipts: list[dict[str, Any]] = []
    delete_receipts: list[dict[str, Any]] = []
    try:
        for request_id in (request_a, request_b):
            current, receipt = _upload_video(client, rows[request_id])
            uploaded.append(current)
            upload_receipts.append(receipt)
        result = score_canary(client, pair, video_a=uploaded[0], video_b=uploaded[1])
    finally:
        for current in uploaded:
            try:
                client.files.delete(name=current.name)
                delete_receipts.append(
                    {"provider_file_name": str(current.name), "deleted": True}
                )
            except Exception as exc:
                delete_receipts.append(
                    {
                        "provider_file_name": str(current.name),
                        "deleted": False,
                        "error_type": type(exc).__name__,
                    }
                )
    projected_batch = result["usage"]["projected_batch_cost_same_usage_usd"] * 441
    arm_cap = next(
        arm["full_matrix_cap_usd"]
        for arm in protocol["arms"]
        if arm["arm_id"] == "gemini36_flash_native_video"
    )
    deleted_all = len(delete_receipts) == 2 and all(row["deleted"] for row in delete_receipts)
    report: dict[str, Any] = {
        "schema_version": "policy_ranking_gemini_pair_canary.v1",
        "status": "passed" if projected_batch <= arm_cap and deleted_all else "blocked",
        "model": GEMINI_MODEL,
        "result": result,
        "uploads": upload_receipts,
        "deletions": delete_receipts,
        "projected_full_batch_cost_usd_from_one_canary": projected_batch,
        "frozen_arm_cap_usd": arm_cap,
        "full_matrix_admitted_from_canary_only": False,
        "next_gate": "seven_pair_batch_transport_and_cost_pilot",
        "source_commit": source_commit,
        "google_genai_sdk_version": importlib.metadata.version("google-genai"),
        "credential_path_or_value_persisted": False,
        "physical_ground_truth_pixels_uploaded": False,
        "temporary_canary_files_deleted": deleted_all,
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(Path(output), report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--native-video-manifest", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.native_video_manifest).read_text(encoding="utf-8"))
    result = run_canary(
        inventory,
        manifest,
        api_key_file=args.api_key_file,
        output=args.output,
        source_commit=args.source_commit,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "result"}))
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
