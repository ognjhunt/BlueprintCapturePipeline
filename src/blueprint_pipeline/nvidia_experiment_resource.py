"""Paid-resource evidence boundary for optional NVIDIA experiments.

The experiment adapters never allocate infrastructure.  Local, owner-supplied
hardware is explicitly non-paid.  A provider-backed worker must carry evidence
that it entered through :mod:`blueprint_pipeline.paid_resource_allocator`, and
its result remains open until exact-attempt and global provider absence are
recorded in a separate closeout artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256


SCHEMA_VERSION = "nvidia_experiment_resource_context.v1"
CLOSEOUT_SCHEMA_VERSION = "nvidia_experiment_resource_closeout.v1"
TEARDOWN_SCHEMA_VERSION = "nvidia_experiment_paid_teardown.v1"
ALLOCATOR_MODULE = "blueprint_pipeline.paid_resource_allocator"
PAID_ALLOCATION_KINDS = {"cpu-build", "model-volume", "gpu-canary"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _digest(value: Any) -> bool:
    text = _string(value).lower()
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def local_unpaid_resource_context() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "resource_origin": "local_unpaid",
        "paid_resource": False,
        "allocation": None,
        "admission": {
            "status": "not_applicable_local_unpaid",
            "allocator_module": None,
            "allocation_kind": None,
        },
        "claim_boundary": {
            "provider_allocation_performed": False,
            "provider_teardown_required": False,
            "local_unpaid_is_not_provider_execution_proof": True,
        },
    }
    payload["context_fingerprint"] = canonical_sha256(payload)
    return payload


def load_resource_context(path: str | Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return local_unpaid_resource_context(), []
    resource_path = Path(path).resolve()
    loaded = read_json_any(resource_path)
    payload = _mapping(loaded)
    blockers = validate_resource_context(payload)
    payload = dict(payload)
    payload["source_path"] = str(resource_path)
    payload["source_sha256"] = sha256_file(resource_path) if resource_path.is_file() else None
    return payload, blockers


def validate_resource_context(payload: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("nvidia_resource_context_schema_invalid")
    origin = _string(payload.get("resource_origin"))
    paid = payload.get("paid_resource")
    if origin == "local_unpaid":
        if paid is not False:
            blockers.append("local_resource_context_must_be_unpaid")
        if payload.get("allocation") not in (None, {}):
            blockers.append("local_resource_context_must_not_claim_provider_allocation")
        return blockers
    if origin != "paid_provider" or paid is not True:
        blockers.append("resource_origin_must_be_local_unpaid_or_paid_provider")
        return blockers
    allocation = _mapping(payload.get("allocation"))
    admission = _mapping(payload.get("admission"))
    if _string(admission.get("allocator_module")) != ALLOCATOR_MODULE:
        blockers.append("paid_resource_must_use_shared_paid_resource_allocator")
    if _string(admission.get("allocation_kind")) not in PAID_ALLOCATION_KINDS:
        blockers.append("paid_resource_allocation_kind_invalid")
    if admission.get("status") != "PASS" or admission.get("spend_allowed") is not True:
        blockers.append("paid_resource_pre_spend_admission_not_proven")
    for field in ("pre_spend_preflight_sha256", "allocation_receipt_sha256"):
        if not _digest(admission.get(field)):
            blockers.append(f"paid_resource_admission_digest_invalid:{field}")
    for field in ("provider_id", "allocation_id", "attempt_id"):
        if not _string(allocation.get(field)):
            blockers.append(f"paid_resource_allocation_identity_missing:{field}")
    try:
        spend_cap = float(admission.get("spend_cap_usd"))
    except (TypeError, ValueError):
        spend_cap = -1.0
    if spend_cap < 0.0:
        blockers.append("paid_resource_spend_cap_missing_or_invalid")
    return list(dict.fromkeys(blockers))


def resource_stop_evidence(
    context: Mapping[str, Any], closeout: Mapping[str, Any] | None = None
) -> dict[str, bool]:
    local = (
        context.get("resource_origin") == "local_unpaid" and context.get("paid_resource") is False
    )
    admission_ok = not validate_resource_context(context)
    closeout_payload = _mapping(closeout)
    teardown_ok = bool(
        local
        or (
            closeout_payload.get("schema_version") == CLOSEOUT_SCHEMA_VERSION
            and closeout_payload.get("status") == "proven_zero"
            and closeout_payload.get("exact_attempt_zero_proven") is True
            and closeout_payload.get("global_provider_zero_proven") is True
        )
    )
    return {
        "paid_resource_admission_enforced": admission_ok,
        "provider_teardown_provable": teardown_ok,
    }


def load_resource_closeout(
    context: Mapping[str, Any], path: str | Path | None
) -> tuple[dict[str, Any], list[str]]:
    """Load and bind a post-attempt closeout to the admitted allocation."""

    if path is None:
        return {}, []
    source = Path(path).resolve()
    loaded = read_json_any(source)
    payload = _mapping(loaded)
    blockers: list[str] = []
    if payload.get("schema_version") != CLOSEOUT_SCHEMA_VERSION:
        blockers.append("nvidia_resource_closeout_schema_invalid")
    if payload.get("status") != "proven_zero" or payload.get("blockers"):
        blockers.append("nvidia_resource_closeout_not_proven_zero")
    allocation = _mapping(context.get("allocation"))
    if context.get("resource_origin") != "paid_provider":
        blockers.append("nvidia_resource_closeout_only_valid_for_paid_provider")
    for field in ("provider_id", "allocation_id", "attempt_id"):
        if _string(payload.get(field)) != _string(allocation.get(field)):
            blockers.append(f"nvidia_resource_closeout_identity_mismatch:{field}")
    expected_context_sha = _string(context.get("source_sha256"))
    if not expected_context_sha or payload.get("resource_context_sha256") != expected_context_sha:
        blockers.append("nvidia_resource_closeout_context_digest_mismatch")
    payload = dict(payload)
    payload["source_path"] = str(source)
    payload["source_sha256"] = sha256_file(source) if source.is_file() else None
    return payload, list(dict.fromkeys(blockers))


def build_resource_closeout(
    *,
    resource_context_path: str | Path,
    teardown_evidence_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    context_path = Path(resource_context_path).resolve()
    teardown_path = Path(teardown_evidence_path).resolve()
    context = _mapping(read_json_any(context_path))
    teardown = _mapping(read_json_any(teardown_path))
    blockers = validate_resource_context(context)
    if context.get("resource_origin") != "paid_provider":
        blockers.append("paid_resource_context_required_for_provider_closeout")
    if teardown.get("schema_version") != TEARDOWN_SCHEMA_VERSION:
        blockers.append("nvidia_paid_teardown_schema_invalid")
    allocation = _mapping(context.get("allocation"))
    for field in ("provider_id", "attempt_id"):
        if _string(teardown.get(field)) != _string(allocation.get(field)):
            blockers.append(f"teardown_identity_mismatch:{field}")
    allocation_id = _string(allocation.get("allocation_id"))
    teardown_ids = {
        _string(value)
        for value in teardown.get("exact_attempt_allocation_ids", [])
        if _string(value)
    }
    exact_zero = bool(
        teardown_ids == {allocation_id} and teardown.get("exact_attempt_active_resource_count") == 0
    )
    if not exact_zero:
        blockers.append("teardown_exact_attempt_zero_not_proven")
    inventories = teardown.get("global_provider_inventory")
    if not isinstance(inventories, list) or not inventories:
        inventories = []
        blockers.append("teardown_global_provider_inventory_missing")
    providers = set()
    for index, value in enumerate(inventories):
        row = _mapping(value)
        provider_id = _string(row.get("provider_id"))
        if not provider_id or provider_id in providers:
            blockers.append(f"teardown_provider_identity_missing_or_duplicate:{index}")
        providers.add(provider_id)
        if row.get("active_resource_count") != 0:
            blockers.append(f"teardown_global_provider_not_zero:{provider_id or index}")
        try:
            burn = float(row.get("hourly_allocation_burn_usd"))
        except (TypeError, ValueError):
            burn = -1.0
        if burn != 0.0:
            blockers.append(f"teardown_global_provider_burn_not_zero:{provider_id or index}")
        if not _digest(row.get("inventory_report_sha256")):
            blockers.append(f"teardown_inventory_digest_invalid:{provider_id or index}")
    global_zero = bool(
        providers
        and _string(allocation.get("provider_id")) in providers
        and not any(item.startswith("teardown_global_provider_") for item in blockers)
    )
    if not _digest(teardown.get("teardown_report_sha256")):
        blockers.append("teardown_report_digest_invalid")
    billing = _mapping(teardown.get("billing_reconciliation"))
    if billing.get("status") != "reconciled" or not _digest(billing.get("billing_export_sha256")):
        blockers.append("teardown_billing_reconciliation_not_proven")
    try:
        spend = float(billing.get("total_spend_usd"))
    except (TypeError, ValueError):
        spend = -1.0
    if spend < 0.0:
        blockers.append("teardown_total_spend_missing_or_invalid")
    observed_at = _string(teardown.get("observed_at"))
    try:
        datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        blockers.append("teardown_observed_at_invalid")
    payload = {
        "schema_version": CLOSEOUT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "proven_zero" if not blockers else "blocked",
        "resource_context_path": str(context_path),
        "resource_context_sha256": sha256_file(context_path) if context_path.is_file() else None,
        "teardown_evidence_path": str(teardown_path),
        "teardown_evidence_sha256": sha256_file(teardown_path) if teardown_path.is_file() else None,
        "provider_id": allocation.get("provider_id"),
        "allocation_id": allocation_id or None,
        "attempt_id": allocation.get("attempt_id"),
        "exact_attempt_zero_proven": exact_zero,
        "global_provider_zero_proven": global_zero,
        "total_spend_usd": spend if spend >= 0.0 else None,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "provider_absence_proven": not blockers,
            "runtime_success_proven": False,
            "semantic_success_proven": False,
            "ranking_success_proven": False,
        },
    }
    payload["closeout_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Close an NVIDIA experiment paid-resource attempt")
    parser.add_argument("--resource-context", required=True)
    parser.add_argument("--teardown-evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_resource_closeout(
        resource_context_path=args.resource_context,
        teardown_evidence_path=args.teardown_evidence,
        output_path=args.output,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "proven_zero" else 2


if __name__ == "__main__":
    raise SystemExit(main())
