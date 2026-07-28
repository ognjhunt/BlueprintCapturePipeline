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
from .policy_ranking_evaluator_diagnostic import diagnostic_protocol
from .policy_ranking_evaluator_diagnostic_gemini import (
    GATE_ENV,
    _secure_file,
    _upload_video,
    _validated_manifest_rows,
)
from .policy_ranking_roboarena_calibration import canonical_sha256


LEDGER_SCHEMA = "policy_ranking_gemini_matrix_media_ledger.v1"


class GeminiMatrixError(ValueError):
    """The full-matrix Gemini media stage is invalid."""


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
        "arm_id": "gemini36_flash_native_video",
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
    protocol = diagnostic_protocol()
    pairs = inventory.get("pairs")
    payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if (
        inventory.get("status") != "ready"
        or inventory.get("pair_count") != 441
        or inventory.get("protocol_sha256") != protocol["protocol_sha256"]
        or canonical_sha256(payload) != inventory.get("inventory_sha256")
        or not isinstance(pairs, list)
        or len(pairs) != 441
    ):
        raise GeminiMatrixError("pair_inventory_not_ready_bound_and_valid_441")
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
    key_path = _secure_file(api_key_file)
    api_key = key_path.read_text(encoding="utf-8").strip()
    target = Path(ledger_path)
    existing: list[dict[str, Any]] = []
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        previous_payload = {
            key: value for key, value in previous.items() if key != "ledger_sha256"
        }
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload")
    upload.add_argument("--inventory", required=True)
    upload.add_argument("--native-video-manifest", required=True)
    upload.add_argument("--api-key-file", required=True)
    upload.add_argument("--ledger", required=True)
    upload.add_argument("--source-commit", required=True)
    upload.add_argument("--workers", type=int, default=4)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--ledger", required=True)
    cleanup.add_argument("--api-key-file", required=True)
    cleanup.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "upload":
        result = upload_matrix_media(
            json.loads(Path(args.inventory).read_text(encoding="utf-8")),
            json.loads(Path(args.native_video_manifest).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            ledger_path=args.ledger,
            source_commit=args.source_commit,
            workers=args.workers,
        )
    else:
        result = cleanup_matrix_media(
            json.loads(Path(args.ledger).read_text(encoding="utf-8")),
            api_key_file=args.api_key_file,
            output_path=args.output,
        )
    print(json.dumps({key: value for key, value in result.items() if key not in {"uploads", "deletions"}}))
    return 0 if result.get("status") in {"ready", "passed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
