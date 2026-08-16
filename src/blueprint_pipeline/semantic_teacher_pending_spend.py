"""Conservatively reserve a semantic-teacher charge pending official billing.

This materializer is used only after an allocated attempt is terminal, provider
zero is independently proven, and a later official Vast billing snapshot still
does not contain the instance charge.  It reserves the attempt's full authority
cap for the next authority; it never estimates a smaller charge or treats the
pending statement as final billing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .semantic_teacher_image_edit_paid_authority import (
    AUTHORITY_SCHEMA_VERSION,
    CONSUMPTION_SCHEMA_VERSION,
    PRIOR_SPEND_ENTRY_SCHEMA_VERSION,
    PRIOR_SPEND_RECONCILIATION_SCHEMA_VERSION,
)


RESERVATION_SCHEMA_VERSION = (
    "semantic_teacher_pending_official_billing_reservation.v1"
)
RECONCILIATION_STATUS = "all_same_goal_paid_attempts_terminal_and_provider_zero"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: str | Path, *, code: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if source.is_symlink() or not source.is_file() or not isinstance(value, dict):
        raise ValueError(code)
    return source, value


def _record(path: Path, *, digest: str | None = None) -> dict[str, Any]:
    value = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if digest is not None:
        value["receipt_digest"] = digest
    return value


def _time(value: Any, *, code: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(code) from exc


def materialize_semantic_teacher_pending_spend(
    *,
    attempt_id: str,
    authority_path: str | Path,
    consumption_path: str | Path,
    teardown_path: str | Path,
    watchdog_path: str | Path,
    provider_zero_path: str | Path,
    official_billing_response_paths: Sequence[str | Path],
    provider_billing_source_receipt_path: str | Path,
    reservation_output_path: str | Path,
    reconciliation_output_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one full-cap reserve and the prior-spend ledger that binds it."""

    authority_file, authority = _read(
        authority_path, code="semantic_teacher_pending_authority_invalid"
    )
    consumption_file, consumption = _read(
        consumption_path, code="semantic_teacher_pending_consumption_invalid"
    )
    teardown_file, teardown = _read(
        teardown_path, code="semantic_teacher_pending_teardown_invalid"
    )
    watchdog_file, watchdog = _read(
        watchdog_path, code="semantic_teacher_pending_watchdog_invalid"
    )
    zero_file, zero = _read(
        provider_zero_path, code="semantic_teacher_pending_provider_zero_invalid"
    )
    billing_source_file, billing_source = _read(
        provider_billing_source_receipt_path,
        code="semantic_teacher_pending_billing_source_invalid",
    )
    billing_files = [
        _read(path, code="semantic_teacher_pending_billing_response_invalid")
        for path in official_billing_response_paths
    ]
    authority_digest = authority.get("authorization_digest")
    bundle_sha256 = (authority.get("bundle") or {}).get("sha256")
    instance_id = str(teardown.get("instance_id") or "")
    reserve = authority.get("hard_total_spend_cap_usd")
    if (
        not attempt_id.strip()
        or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority_digest
        != canonical_digest(authority, digest_field="authorization_digest")
        or consumption.get("schema_version") != CONSUMPTION_SCHEMA_VERSION
        or consumption.get("authorization_digest") != authority_digest
        or consumption.get("bundle_sha256") != bundle_sha256
        or teardown.get("status") != "PASS"
        or teardown.get("provider") != "vast"
        or teardown.get("authorization_digest") != authority_digest
        or teardown.get("bundle_sha256") != bundle_sha256
        or not instance_id.isdigit()
        or teardown.get("global_provider_zero") is not True
        or teardown.get("scoped_provider_zero") is not True
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("teardown_digest")
        != canonical_digest(teardown, digest_field="teardown_digest")
        or watchdog.get("status") != "provider_terminal"
        or int(instance_id) not in (watchdog.get("instance_ids") or [])
        or watchdog.get("provider_absence_confirmed") is not True
        or zero.get("provider_zero_verified") is not True
        or zero.get("live_instance_count") != 0
        or (zero.get("provider_zero") or {}).get("status") != "verified"
        or isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not 0 < float(reserve) <= 5.0
    ):
        raise ValueError("semantic_teacher_pending_terminal_evidence_invalid")
    billing_digest = billing_source.get("receipt_digest")
    if (
        billing_source.get("schema_version")
        != "blueprint.provider_billing_source_receipt.v1"
        or billing_source.get("status") != "reconciled"
        or billing_source.get("provider_mutation_performed") is not False
        or billing_digest
        != canonical_digest(billing_source, digest_field="receipt_digest")
        or _time(billing_source.get("generated_at"), code="billing_time_invalid")
        <= _time(teardown.get("timestamp"), code="teardown_time_invalid")
        or not billing_files
    ):
        raise ValueError("semantic_teacher_pending_billing_source_invalid")
    linked_paths: set[Path] = set()
    for source in billing_source.get("sources") or []:
        if not isinstance(source, Mapping) or source.get("provider") != "vast":
            continue
        linked = Path(str(source.get("retained_path") or "")).expanduser().resolve()
        if any(
            linked == path
            and source.get("response_digest") == _sha256(path)
            and source.get("response_size_bytes") == path.stat().st_size
            for path, _value in billing_files
        ):
            linked_paths.add(linked)
    if linked_paths != {path for path, _value in billing_files}:
        raise ValueError("semantic_teacher_pending_billing_response_unbound")
    billing_sources: list[dict[str, Any]] = []
    needle = f"instance-{instance_id}"
    for path, value in billing_files:
        results = value.get("results")
        if not isinstance(results, list) or any(
            isinstance(row, Mapping) and row.get("source") == needle
            for row in results
        ):
            raise ValueError("semantic_teacher_official_charge_not_pending")
        billing_sources.append(_record(path))

    source_records = {
        "authority": _record(authority_file, digest=str(authority_digest)),
        "consumption": _record(consumption_file),
        "teardown": _record(teardown_file, digest=str(teardown["teardown_digest"])),
        "watchdog": _record(watchdog_file),
        "provider_zero": _record(zero_file, digest=str(zero["receipt_digest"])),
        "billing_source": _record(billing_source_file, digest=str(billing_digest)),
        "billing_responses": billing_sources,
    }
    reservation: dict[str, Any] = {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "status": "pending_official_billing_conservative_reserve",
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": attempt_id.strip(),
        "lane": "semantic_teacher_image_edit_gpu_canary",
        "cost_usd": float(reserve),
        "cost_basis": "full_authorized_attempt_cap_until_official_charge_posts",
        "official_billing_pending": True,
        "provider_zero_confirmed": True,
        "continuing_spend_from_this_run": False,
        "provider_instance_id": int(instance_id),
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_sha256,
        "sources": source_records,
        "supersession_required": "replace_with_fully_bound_official_billing_before_final_campaign_closeout",
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "generated_at": utc_now_iso(),
        "receipt_digest": "",
    }
    reservation["receipt_digest"] = canonical_digest(
        reservation, digest_field="receipt_digest"
    )
    reservation_output = Path(reservation_output_path).expanduser().resolve()
    reconciliation_output = Path(reconciliation_output_path).expanduser().resolve()
    if any(path.exists() or path.is_symlink() for path in (reservation_output, reconciliation_output)):
        raise ValueError("semantic_teacher_pending_output_exists")
    write_json(reservation_output, reservation)
    reservation_source = {
        "role": "pending_official_billing_reservation",
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "digest_field": "receipt_digest",
        "record": _record(
            reservation_output, digest=str(reservation["receipt_digest"])
        ),
    }
    bindings = [
        ("cost_usd", "cost_usd", float(reserve)),
        ("provider_zero", "provider_zero_confirmed", True),
        ("continuing_spend", "continuing_spend_from_this_run", False),
        ("instance_id", "provider_instance_id", int(instance_id)),
        ("authority_digest", "authority_digest", authority_digest),
        ("bundle_sha256", "bundle_sha256", bundle_sha256),
    ]
    entry: dict[str, Any] = {
        "schema_version": PRIOR_SPEND_ENTRY_SCHEMA_VERSION,
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": attempt_id.strip(),
        "lane": "semantic_teacher_image_edit_gpu_canary",
        "evidence_kind": "pending_official_billing_conservative_reserve",
        "provider_instance_id": int(instance_id),
        "cost_usd": float(reserve),
        "authority_digest": authority_digest,
        "bundle_sha256": bundle_sha256,
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "source_receipts": [reservation_source],
        "bindings": [
            {
                "kind": kind,
                "source_role": "pending_official_billing_reservation",
                "json_path": [field],
                "expected_value": expected,
            }
            for kind, field, expected in bindings
        ],
        "entry_digest": "",
    }
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    reconciliation: dict[str, Any] = {
        "schema_version": PRIOR_SPEND_RECONCILIATION_SCHEMA_VERSION,
        "status": RECONCILIATION_STATUS,
        "goal_id": "arm-decision-proof-v1",
        "entries": [entry],
        "entry_count": 1,
        "total_cost_usd": float(reserve),
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    reconciliation["receipt_digest"] = canonical_digest(
        reconciliation, digest_field="receipt_digest"
    )
    write_json(reconciliation_output, reconciliation)
    return reservation, reconciliation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--consumption", required=True)
    parser.add_argument("--teardown", required=True)
    parser.add_argument("--watchdog", required=True)
    parser.add_argument("--provider-zero", required=True)
    parser.add_argument("--official-billing-response", action="append", required=True)
    parser.add_argument("--provider-billing-source-receipt", required=True)
    parser.add_argument("--reservation-output", required=True)
    parser.add_argument("--reconciliation-output", required=True)
    args = parser.parse_args(argv)
    materialize_semantic_teacher_pending_spend(
        attempt_id=args.attempt_id,
        authority_path=args.authority,
        consumption_path=args.consumption,
        teardown_path=args.teardown,
        watchdog_path=args.watchdog,
        provider_zero_path=args.provider_zero,
        official_billing_response_paths=args.official_billing_response,
        provider_billing_source_receipt_path=args.provider_billing_source_receipt,
        reservation_output_path=args.reservation_output,
        reconciliation_output_path=args.reconciliation_output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
