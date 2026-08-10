"""Recover an abnormal native-task closeout without inventing teardown evidence.

This path is intentionally narrower than the normal Vast adapter closeout.  It
is used only when local receipt writing failed after an instance was created,
and it may conclude provider/object-store zero plus billed cost from independent
API evidence.  It never reconstructs a missing adapter or teardown manifest and
never upgrades the scientific run to completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "native_task_paid_closeout_recovery.v1"
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return dict(value)


def _phase_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("native_task_closeout_phase_log_invalid") from exc
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("native_task_closeout_phase_log_invalid")
    return [dict(row) for row in rows]


def recover_native_task_paid_closeout(
    *,
    attempt_root: str | Path,
    provider_zero_receipt_path: str | Path,
    bundle_receipt_path: str | Path,
    output_path: str | Path,
    expected_instance_id: int,
    instance_label: str,
    run_command: RunCommand = subprocess.run,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Join immutable run bytes to fresh read-only provider billing/zero APIs."""

    attempt = Path(attempt_root).expanduser().resolve()
    provider_run = attempt / "vast_provider_run"
    output = Path(output_path).expanduser().resolve()
    instance_id = int(expected_instance_id)
    if instance_id < 1 or not str(instance_label).strip():
        raise ValueError("native_task_closeout_instance_binding_invalid")
    if (provider_run / "vast_provider_adapter_result.json").exists() or (
        provider_run / "vast_teardown_manifest.json"
    ).exists():
        raise ValueError("native_task_closeout_normal_receipts_present")

    bundle = _mapping(
        Path(bundle_receipt_path).expanduser().resolve(),
        error="native_task_closeout_bundle_receipt_invalid",
    )
    bundle_path = Path(str(bundle.get("bundle_path") or "")).expanduser().resolve()
    if (
        bundle.get("status") != "ready"
        or not bundle_path.is_file()
        or bundle.get("bundle_sha256") != _sha256(bundle_path)
        or bundle.get("bundle_size_bytes") != bundle_path.stat().st_size
    ):
        raise ValueError("native_task_closeout_bundle_receipt_invalid")

    phase_path = provider_run / "vast_runtime_phase_log.jsonl"
    phases = _phase_rows(phase_path)
    created_ids = {
        int(row["instance_id"])
        for row in phases
        if row.get("phase") == "vast_instance_create_requested"
        and row.get("status") == "completed"
        and isinstance(row.get("instance_id"), int)
    }
    budget = _mapping(
        provider_run / "vast_budget_ledger.json",
        error="native_task_closeout_budget_ledger_invalid",
    )
    all_in = _mapping(
        provider_run / "vast_all_in_cost_binding.json",
        error="native_task_closeout_cost_binding_invalid",
    )
    if (
        created_ids != {instance_id}
        or budget.get("vast_instance_ids") != [instance_id]
        or all_in.get("instance_id") != instance_id
    ):
        raise ValueError("native_task_closeout_instance_binding_invalid")

    cleanup_path = attempt / "object_store_staging/wam_provider_object_store_cleanup.json"
    cleanup = _mapping(
        cleanup_path,
        error="native_task_closeout_object_store_cleanup_invalid",
    )
    if (
        cleanup.get("schema_version") != "wam_provider_object_store_cleanup.v1"
        or cleanup.get("status") != "completed"
        or cleanup.get("all_objects_absent") is not True
        or cleanup.get("signed_url_files_removed") is not True
        or cleanup.get("blockers") != []
    ):
        raise ValueError("native_task_closeout_object_store_cleanup_invalid")

    zero_path = Path(provider_zero_receipt_path).expanduser().resolve()
    zero = _mapping(zero_path, error="native_task_closeout_provider_zero_invalid")
    if (
        zero.get("schema_version") != "adp_paid_provider_zero.v1"
        or zero.get("provider") != "vast"
        or zero.get("api_confirmed") is not True
        or zero.get("provider_zero") is not True
        or zero.get("global_live_resource_count") != 0
        or zero.get("provider_zero_digest")
        != canonical_digest(zero, digest_field="provider_zero_digest")
    ):
        raise ValueError("native_task_closeout_provider_zero_invalid")

    command = [
        "vastai",
        "show",
        "invoices",
        "--raw",
        "--only_charges",
        "--instance_label",
        str(instance_label),
    ]
    completed = run_command(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        invoice_rows = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("native_task_closeout_provider_invoice_invalid") from exc
    if (
        completed.returncode != 0
        or not isinstance(invoice_rows, list)
        or not invoice_rows
        or any(
            not isinstance(row, Mapping)
            or row.get("instance_id") != instance_id
            or row.get("type") != "charge"
            for row in invoice_rows
        )
    ):
        raise ValueError("native_task_closeout_provider_invoice_invalid")
    try:
        cost = sum((Decimal(str(row["amount"])) for row in invoice_rows), Decimal("0"))
    except (InvalidOperation, KeyError, TypeError) as exc:
        raise ValueError("native_task_closeout_provider_invoice_invalid") from exc

    scientific_log = provider_run / "vast_onstart_container.log"
    try:
        log_text = scientific_log.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("native_task_closeout_scientific_log_invalid") from exc
    missing_torch = "ModuleNotFoundError: No module named 'torch'" in log_text
    scientific_blockers = (
        ["native_task_pre_app_dependency_missing:torch"]
        if missing_torch
        else ["native_task_runtime_failure_unclassified"]
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "recovered_blocked_attempt_closeout",
        "attempt_root": str(attempt),
        "instance_id": instance_id,
        "instance_label": str(instance_label),
        "bundle_sha256": bundle["bundle_sha256"],
        "bundle_size_bytes": bundle["bundle_size_bytes"],
        "phase_log": {
            "path": str(phase_path),
            "sha256": _sha256(phase_path),
            "row_count": len(phases),
        },
        "scientific_log": {
            "path": str(scientific_log),
            "sha256": _sha256(scientific_log),
            "size_bytes": scientific_log.stat().st_size,
            "missing_torch_observed": missing_torch,
        },
        "scientific_status": "blocked_before_simulation_app",
        "scientific_blockers": scientific_blockers,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "normal_adapter_receipt_present": False,
        "normal_teardown_manifest_present": False,
        "normal_teardown_reconstructed": False,
        "manual_destroy_command_receipt_present": False,
        "provider_terminal_evidence": {
            "method": "global_provider_api_after_external_recovery",
            "provider_zero_receipt_path": str(zero_path),
            "provider_zero_digest": zero["provider_zero_digest"],
            "global_live_resource_count": 0,
            "continuing_spend_from_this_run": False,
        },
        "object_store_terminal_evidence": {
            "cleanup_receipt_path": str(cleanup_path),
            "exact_object_count": cleanup.get("exact_object_count"),
            "all_objects_absent": True,
            "signed_url_files_removed": True,
        },
        "provider_invoice": {
            "api_command": command,
            "returncode": completed.returncode,
            "rows": [dict(row) for row in invoice_rows],
            "provider_reported_total_usd": format(cost, "f"),
            "row_count": len(invoice_rows),
        },
        "remaining_ambiguities": [
            "exact_external_destroy_method_not_retained_in_run_bytes",
            "local_closeout_failure_cause_not_retained_in_run_bytes",
        ],
        "blockers": [
            *scientific_blockers,
            "native_task_arena_normal_closeout_receipts_missing",
        ],
        "provider_mutations_performed_by_recovery": 0,
        "raw_secret_values_recorded": False,
        "recovery_receipt_digest": "",
    }
    result["recovery_receipt_digest"] = canonical_digest(
        result, digest_field="recovery_receipt_digest"
    )
    write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--provider-zero-receipt", required=True)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instance-id", required=True, type=int)
    parser.add_argument("--instance-label", required=True)
    args = parser.parse_args(argv)
    recover_native_task_paid_closeout(
        attempt_root=args.attempt_root,
        provider_zero_receipt_path=args.provider_zero_receipt,
        bundle_receipt_path=args.bundle_receipt,
        output_path=args.output,
        expected_instance_id=args.instance_id,
        instance_label=args.instance_label,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI seam
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "main", "recover_native_task_paid_closeout"]
