"""Fail-closed paid-work admission lock for the beta cohort spend ceiling.

The lock combines the conservative allocation ledger with a current provider
billing export.  It never treats a local estimate as invoice truth, and it
never treats a generated page event or teardown plan as proof that an operator
was paged or that a provider resource is gone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = "blueprint.paid_spend_admission_lock.v1"
OVERRIDE_SCHEMA_VERSION = "blueprint.paid_spend_override.v1"
HARD_STOP_USD = 5000.0
REQUIRED_BILLING_PROVIDERS = {"runpod", "vast", "digitalocean"}
SUPPORTED_BILLING_PROVIDERS = REQUIRED_BILLING_PROVIDERS | {"gcp", "aws"}
BILLING_EXPORT_SCHEMA_VERSION = "blueprint.provider_billing_export.v1"
BILLING_EXPORT_SCOPE = "blueprint_beta_100_user_cohort"
MAX_LOCK_AGE_SECONDS = 5 * 60
MAX_BILLING_AGE_SECONDS = 24 * 60 * 60
MAX_OVERRIDE_DURATION_SECONDS = 4 * 60 * 60
MAX_OVERRIDE_BYTES = 64 * 1024
OVERRIDE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _durable_ticket_uri(value: object) -> bool:
    parsed = urlsplit(str(value or "").strip())
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def _mode_from_octal(value: object) -> int | None:
    text = str(value or "").strip()
    if re.fullmatch(r"[0-7]{3,4}", text) is None:
        return None
    return int(text, 8)


def _string_items(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _inventory_contract_blockers(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ["provider_inventory_contract_invalid"]
    rows = [_mapping(item) for item in value]
    providers = [str(row.get("provider") or "").strip() for row in rows]
    blockers: list[str] = []
    provider_set = set(providers)
    if (
        len(rows) != len(provider_set)
        or not REQUIRED_BILLING_PROVIDERS.issubset(provider_set)
        or not provider_set.issubset(SUPPORTED_BILLING_PROVIDERS)
    ):
        blockers.append("provider_inventory_coverage_incomplete")
    for row, provider in zip(rows, providers, strict=False):
        if provider not in SUPPORTED_BILLING_PROVIDERS:
            blockers.append(f"provider_inventory_unexpected:{provider or 'missing'}")
            continue
        if row.get("required") is not True:
            blockers.append(f"provider_inventory_not_required:{provider}")
        if row.get("credential_present") is not True:
            blockers.append(f"provider_inventory_credential_missing:{provider}")
        if row.get("status") != "succeeded":
            blockers.append(f"provider_inventory_not_succeeded:{provider}")
        row_count = row.get("row_count")
        if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
            blockers.append(f"provider_inventory_row_count_invalid:{provider}")
        blockers.extend(
            f"provider_inventory:{provider}:{item}"
            for item in _string_items(row.get("blockers"))
        )
    return sorted(set(blockers))


def _embedded_override_blockers(
    value: object,
    *,
    now: datetime,
) -> list[str]:
    override = _mapping(value)
    blockers: list[str] = []
    if override.get("schema_version") != OVERRIDE_SCHEMA_VERSION:
        blockers.append("spend_override_schema_invalid")
    if override.get("status") != "valid":
        blockers.append("spend_override_not_valid")
    if override.get("scope") != "paid_spend_hard_stop":
        blockers.append("spend_override_scope_invalid")
    if override.get("allow_new_paid_work") is not True:
        blockers.append("spend_override_admission_not_explicit")
    if _finite_number(override.get("hard_stop_usd")) != HARD_STOP_USD:
        blockers.append("spend_override_hard_stop_mismatch")
    override_id = str(override.get("override_id") or "")
    if OVERRIDE_ID_PATTERN.fullmatch(override_id) is None:
        blockers.append("spend_override_id_invalid")
    requested_by = str(override.get("requested_by") or "").strip()
    approved_by = str(override.get("approved_by") or "").strip()
    if (
        len(requested_by) < 3
        or len(approved_by) < 3
        or len(requested_by) > 128
        or len(approved_by) > 128
    ):
        blockers.append("spend_override_two_person_audit_missing")
    elif requested_by == approved_by:
        blockers.append("spend_override_approver_must_differ")
    reason = str(override.get("reason") or "").strip()
    if len(reason) < 20 or len(reason) > 500:
        blockers.append("spend_override_reason_missing")
    if not _durable_ticket_uri(override.get("ticket_uri")):
        blockers.append("spend_override_ticket_uri_invalid")
    issued_at = _parse_time(override.get("issued_at"))
    expires_at = _parse_time(override.get("expires_at"))
    if issued_at is None or expires_at is None or issued_at >= expires_at:
        blockers.append("spend_override_interval_invalid")
    else:
        if issued_at > now + timedelta(minutes=5):
            blockers.append("spend_override_from_future")
        if expires_at <= now:
            blockers.append("spend_override_expired")
        if expires_at - issued_at > timedelta(seconds=MAX_OVERRIDE_DURATION_SECONDS):
            blockers.append("spend_override_duration_exceeds_policy")
    if SHA256_PATTERN.fullmatch(
        str(override.get("source_artifact_digest") or "")
    ) is None:
        blockers.append("spend_override_source_digest_invalid")
    source_mode = _mode_from_octal(override.get("source_mode_octal"))
    if source_mode is None:
        blockers.append("spend_override_source_mode_invalid")
    elif source_mode & (stat.S_IWGRP | stat.S_IWOTH):
        blockers.append("spend_override_source_mode_unsafe")
    return sorted(set(blockers))


def validate_spend_override(
    override_path: Path | None,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Validate a short-lived, two-person, non-writable override artifact."""

    if override_path is None:
        return {
            "schema_version": OVERRIDE_SCHEMA_VERSION,
            "status": "not_configured",
            "blockers": [],
        }
    path = override_path.expanduser()
    blockers: list[str] = []
    payload: dict[str, Any] = {}
    digest: str | None = None
    mode: int | None = None
    owner_uid: int | None = None
    if path.is_symlink() or not path.is_file():
        blockers.append("spend_override_missing_or_symlink")
    else:
        try:
            metadata = path.stat()
            mode = stat.S_IMODE(metadata.st_mode)
            owner_uid = metadata.st_uid
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append("spend_override_unreadable")
        else:
            if mode & (stat.S_IWGRP | stat.S_IWOTH):
                blockers.append("spend_override_writable_by_group_or_world")
            if owner_uid not in {0, os.geteuid()}:
                blockers.append("spend_override_owner_untrusted")
            if metadata.st_size > MAX_OVERRIDE_BYTES:
                blockers.append("spend_override_too_large")
            else:
                try:
                    digest = _sha256(path)
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    blockers.append("spend_override_unreadable")
                else:
                    payload = _mapping(raw)
                    if not payload:
                        blockers.append("spend_override_not_object")
    if payload.get("schema_version") != OVERRIDE_SCHEMA_VERSION:
        blockers.append("spend_override_schema_invalid")
    if payload.get("status") != "approved":
        blockers.append("spend_override_not_approved")
    if payload.get("scope") != "paid_spend_hard_stop":
        blockers.append("spend_override_scope_invalid")
    if payload.get("allow_new_paid_work") is not True:
        blockers.append("spend_override_admission_not_explicit")
    if _finite_number(payload.get("hard_stop_usd")) != HARD_STOP_USD:
        blockers.append("spend_override_hard_stop_mismatch")
    override_id = str(payload.get("override_id") or "")
    if OVERRIDE_ID_PATTERN.fullmatch(override_id) is None:
        blockers.append("spend_override_id_invalid")
    requested_by = str(payload.get("requested_by") or "").strip()
    approved_by = str(payload.get("approved_by") or "").strip()
    if (
        len(requested_by) < 3
        or len(approved_by) < 3
        or len(requested_by) > 128
        or len(approved_by) > 128
    ):
        blockers.append("spend_override_two_person_audit_missing")
    elif requested_by == approved_by:
        blockers.append("spend_override_approver_must_differ")
    reason = str(payload.get("reason") or "").strip()
    if len(reason) < 20 or len(reason) > 500:
        blockers.append("spend_override_reason_missing")
    if not _durable_ticket_uri(payload.get("ticket_uri")):
        blockers.append("spend_override_ticket_uri_invalid")
    issued_at = _parse_time(payload.get("issued_at"))
    expires_at = _parse_time(payload.get("expires_at"))
    current = now.astimezone(timezone.utc)
    if issued_at is None or expires_at is None or issued_at >= expires_at:
        blockers.append("spend_override_interval_invalid")
    else:
        if issued_at > current + timedelta(minutes=5):
            blockers.append("spend_override_from_future")
        if expires_at <= current:
            blockers.append("spend_override_expired")
        if expires_at - issued_at > timedelta(seconds=MAX_OVERRIDE_DURATION_SECONDS):
            blockers.append("spend_override_duration_exceeds_policy")
    blockers = sorted(set(blockers))
    return {
        "schema_version": OVERRIDE_SCHEMA_VERSION,
        "status": "valid" if not blockers else "blocked",
        "scope": payload.get("scope"),
        "hard_stop_usd": payload.get("hard_stop_usd"),
        "allow_new_paid_work": payload.get("allow_new_paid_work"),
        "override_id": override_id or None,
        "requested_by": requested_by or None,
        "approved_by": approved_by or None,
        "reason": reason[:500] or None,
        "ticket_uri": str(payload.get("ticket_uri") or "")[:500] or None,
        "issued_at": payload.get("issued_at"),
        "expires_at": payload.get("expires_at"),
        "source_artifact_digest": digest,
        "source_mode_octal": f"{mode:04o}" if mode is not None else None,
        "blockers": blockers,
    }


def _billing_total(reconciliation: Mapping[str, Any]) -> float | None:
    totals = reconciliation.get("provider_totals_usd")
    if not isinstance(totals, Mapping):
        return None
    values = [_finite_number(value) for value in totals.values()]
    if not values or any(value is None or value < 0 for value in values):
        return None
    return round(sum(value for value in values if value is not None), 4)


def build_spend_admission_lock(
    *,
    fleet_budget: Mapping[str, Any],
    billing_reconciliation: Mapping[str, Any],
    instances: Sequence[Mapping[str, Any]],
    reap_results: Sequence[Mapping[str, Any]],
    inventory_results: Sequence[Mapping[str, Any]],
    override_path: Path | None,
    now: datetime,
) -> dict[str, Any]:
    """Build the one artifact every production paid-lane admission must pass."""

    current = now.astimezone(timezone.utc)
    fleet = _mapping(fleet_budget)
    billing = _mapping(billing_reconciliation)
    override = validate_spend_override(override_path, now=current)
    blockers: list[str] = []
    inventory_rows = [dict(result) for result in inventory_results]
    inventory_blockers = _inventory_contract_blockers(inventory_rows)
    blockers.extend(inventory_blockers)
    if billing.get("status") != "reconciled":
        blockers.append("billing_reconciliation_not_current")
        blockers.extend(
            f"billing:{item}"
            for item in _string_items(billing.get("blockers"))
        )
    if billing.get("required") is not True:
        blockers.append("billing_reconciliation_not_required")
    if billing.get("currency") != "USD":
        blockers.append("billing_reconciliation_currency_invalid")
    if billing.get("billing_export_schema_version") != BILLING_EXPORT_SCHEMA_VERSION:
        blockers.append("billing_reconciliation_schema_invalid")
    if billing.get("scope") != BILLING_EXPORT_SCOPE:
        blockers.append("billing_reconciliation_scope_invalid")
    billing_totals = billing.get("provider_totals_usd")
    inventory_provider_set = {
        str(row.get("provider") or "").strip() for row in inventory_rows
    }
    if not isinstance(billing_totals, Mapping) or set(billing_totals) != inventory_provider_set:
        blockers.append("billing_reconciliation_provider_coverage_incomplete")
    billing_generated_at = _parse_time(billing.get("generated_at"))
    if (
        billing_generated_at is None
        or billing_generated_at > current + timedelta(minutes=5)
        or current - billing_generated_at
        > timedelta(seconds=MAX_BILLING_AGE_SECONDS)
    ):
        blockers.append("billing_reconciliation_generated_at_stale_or_invalid")
    if SHA256_PATTERN.fullmatch(
        str(billing.get("billing_export_sha256") or "")
    ) is None:
        blockers.append("billing_reconciliation_source_digest_invalid")
    billing_mode = _mode_from_octal(billing.get("billing_export_mode_octal"))
    if billing_mode is None:
        blockers.append("billing_reconciliation_source_mode_invalid")
    elif billing_mode & (stat.S_IWGRP | stat.S_IWOTH):
        blockers.append("billing_reconciliation_source_mode_unsafe")
    billing_total = _billing_total(billing)
    if billing_total is None:
        blockers.append("billing_reconciled_total_invalid")
    ledger_total = _finite_number(fleet.get("total_spend_usd"))
    if ledger_total is None or ledger_total < 0:
        blockers.append("allocation_ledger_total_invalid")
    effective_values = [
        value for value in (billing_total, ledger_total) if value is not None
    ]
    effective_spend = max(effective_values) if effective_values else None
    threshold_crossed = bool(
        effective_spend is not None and effective_spend >= HARD_STOP_USD
    )
    override_valid = override.get("status") == "valid"
    original_fleet_blockers = _string_items(fleet.get("blockers"))
    fleet_blockers = list(original_fleet_blockers)
    if override_valid and threshold_crossed:
        fleet_blockers = [
            item for item in fleet_blockers if item != "fleet_total_spend_limit_exceeded"
        ]
    blockers.extend(f"fleet:{item}" for item in fleet_blockers)
    override_covers_fleet_status = bool(
        override_valid
        and threshold_crossed
        and original_fleet_blockers
        and set(original_fleet_blockers) == {"fleet_total_spend_limit_exceeded"}
    )
    if fleet.get("status") != "passed" and not override_covers_fleet_status:
        blockers.append("fleet_budget_status_not_passed")
    if override.get("status") == "blocked":
        blockers.extend(f"override:{item}" for item in override.get("blockers") or [])
    if threshold_crossed and not override_valid:
        blockers.append("cohort_hard_stop_reached")

    blockers = sorted(set(blockers))
    admission_allowed = not blockers
    live_instances = [row for row in instances if _mapping(row).get("live") is True]
    reap_candidates = [
        row for row in instances if _mapping(row).get("reap_candidate") is True
    ]
    successful_teardowns = {
        (str(_mapping(row).get("provider") or ""), str(_mapping(row).get("id") or ""))
        for row in reap_results
        if _mapping(row).get("status") == "terminated"
    }
    all_candidates_terminated = all(
        (str(_mapping(row).get("provider") or ""), str(_mapping(row).get("id") or ""))
        in successful_teardowns
        for row in reap_candidates
    )
    inventory_known = not inventory_blockers
    if admission_allowed:
        drain_status = "not_required"
    elif not inventory_known:
        drain_status = "inventory_unknown"
    elif live_instances:
        drain_status = "draining"
    else:
        drain_status = "drained"
    teardown_complete = bool(
        not admission_allowed
        and inventory_known
        and not live_instances
        and all_candidates_terminated
    )
    event_seed = json.dumps(
        {
            "effective_spend_usd": effective_spend,
            "threshold_crossed": threshold_crossed,
            "blockers": blockers,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # An override may reopen admission, but it never suppresses the operator
    # notification required by a cohort hard-stop crossing.
    page_required = threshold_crossed or not admission_allowed
    status = "open"
    if override_valid and threshold_crossed and admission_allowed:
        status = "override_open"
    elif not admission_allowed:
        status = "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": current.isoformat(),
        "status": status,
        "admission_allowed": admission_allowed,
        "hard_stop_usd": HARD_STOP_USD,
        "threshold_crossed": threshold_crossed,
        "allocation_ledger_total_usd": ledger_total,
        "reconciled_billing_total_usd": billing_total,
        "effective_spend_usd": effective_spend,
        "billing_reconciliation": billing,
        "provider_inventory": inventory_rows,
        "override": override,
        "controlled_drain": {
            "status": drain_status,
            "new_paid_work_stopped": not admission_allowed,
            "live_instance_count": len(live_instances),
            "live_instance_ids": sorted(
                str(_mapping(row).get("id") or "") for row in live_instances
            ),
            "reap_candidate_count": len(reap_candidates),
            "reap_results": [dict(row) for row in reap_results],
            "all_reap_candidates_terminated": all_candidates_terminated,
            "provider_inventory_complete": inventory_known,
            "teardown_evidence_complete": teardown_complete,
        },
        "page_event": {
            "event_id": f"spend-lock-{hashlib.sha256(event_seed).hexdigest()[:24]}",
            "event_type": "paid_spend_admission_locked",
            "severity": "critical",
            "required": page_required,
            "delivery_status": "external_pending" if page_required else "not_required",
        },
        "blockers": blockers,
        "claim_boundary": {
            "billing_export_is_external_input_not_live_api_proof": True,
            "page_event_is_not_notification_delivery_proof": True,
            "draining_is_not_provider_teardown_proof": True,
            "override_is_short_lived_and_two_person_audited": True,
        },
    }


def validate_spend_admission_lock(
    evidence: Mapping[str, Any] | None,
    *,
    now: datetime,
    max_age_seconds: int = MAX_LOCK_AGE_SECONDS,
) -> list[str]:
    """Validate a lock artifact at the shared paid-lane chokepoint."""

    row = _mapping(evidence)
    blockers: list[str] = []
    if row.get("schema_version") != SCHEMA_VERSION:
        blockers.append("spend_admission_lock_schema_invalid")
    generated_at = _parse_time(row.get("generated_at"))
    current = now.astimezone(timezone.utc)
    if generated_at is None:
        blockers.append("spend_admission_lock_generated_at_invalid")
    elif generated_at > current + timedelta(minutes=5):
        blockers.append("spend_admission_lock_from_future")
    elif current - generated_at > timedelta(seconds=max(1, max_age_seconds)):
        blockers.append("spend_admission_lock_stale")
    if _finite_number(row.get("hard_stop_usd")) != HARD_STOP_USD:
        blockers.append("spend_admission_lock_hard_stop_mismatch")
    effective_spend = _finite_number(row.get("effective_spend_usd"))
    if effective_spend is None or effective_spend < 0:
        blockers.append("spend_admission_lock_effective_spend_invalid")
    if _mapping(row.get("billing_reconciliation")).get("status") != "reconciled":
        blockers.append("spend_admission_lock_billing_not_reconciled")
    billing = _mapping(row.get("billing_reconciliation"))
    if billing.get("required") is not True:
        blockers.append("spend_admission_lock_billing_not_required")
    if (
        billing.get("currency") != "USD"
        or billing.get("scope") != BILLING_EXPORT_SCOPE
        or billing.get("billing_export_schema_version")
        != BILLING_EXPORT_SCHEMA_VERSION
    ):
        blockers.append("spend_admission_lock_billing_contract_invalid")
    totals = billing.get("provider_totals_usd")
    inventory_provider_set = {
        str(item.get("provider") or "").strip()
        for item in row.get("provider_inventory") or []
        if isinstance(item, Mapping)
    }
    if not isinstance(totals, Mapping) or set(totals) != inventory_provider_set:
        blockers.append("spend_admission_lock_billing_provider_coverage_incomplete")
    billing_generated_at = _parse_time(billing.get("generated_at"))
    if (
        billing_generated_at is None
        or billing_generated_at > current + timedelta(minutes=5)
        or current - billing_generated_at
        > timedelta(seconds=MAX_BILLING_AGE_SECONDS)
    ):
        blockers.append("spend_admission_lock_billing_stale_or_invalid_time")
    if SHA256_PATTERN.fullmatch(
        str(billing.get("billing_export_sha256") or "")
    ) is None:
        blockers.append("spend_admission_lock_billing_source_digest_invalid")
    billing_mode = _mode_from_octal(billing.get("billing_export_mode_octal"))
    if billing_mode is None:
        blockers.append("spend_admission_lock_billing_source_mode_invalid")
    elif billing_mode & (stat.S_IWGRP | stat.S_IWOTH):
        blockers.append("spend_admission_lock_billing_source_mode_unsafe")
    blockers.extend(
        f"spend_admission_lock_{item}"
        for item in _inventory_contract_blockers(row.get("provider_inventory"))
    )
    computed_billing_total = _billing_total(billing)
    recorded_billing_total = _finite_number(row.get("reconciled_billing_total_usd"))
    ledger_total = _finite_number(row.get("allocation_ledger_total_usd"))
    if (
        computed_billing_total is None
        or recorded_billing_total is None
        or abs(computed_billing_total - recorded_billing_total) > 0.0001
    ):
        blockers.append("spend_admission_lock_billing_total_mismatch")
    if ledger_total is None or ledger_total < 0:
        blockers.append("spend_admission_lock_ledger_total_invalid")
    if (
        effective_spend is not None
        and computed_billing_total is not None
        and ledger_total is not None
        and abs(effective_spend - max(computed_billing_total, ledger_total)) > 0.0001
    ):
        blockers.append("spend_admission_lock_effective_total_mismatch")
    expected_crossing = bool(
        effective_spend is not None and effective_spend >= HARD_STOP_USD
    )
    if row.get("threshold_crossed") is not expected_crossing:
        blockers.append("spend_admission_lock_threshold_state_mismatch")
    status = row.get("status")
    if status not in {"open", "override_open"}:
        blockers.append(f"spend_admission_lock_closed:{status or 'missing'}")
    if row.get("admission_allowed") is not True:
        blockers.append("spend_admission_lock_admission_denied")
    if row.get("blockers") not in ([], None):
        blockers.append("spend_admission_lock_has_blockers")
    override = _mapping(row.get("override"))
    if effective_spend is not None and effective_spend >= HARD_STOP_USD:
        if status != "override_open" or override.get("status") != "valid":
            blockers.append("spend_admission_lock_threshold_without_valid_override")
        blockers.extend(
            f"spend_admission_lock_override:{item}"
            for item in _embedded_override_blockers(override, now=current)
        )
    elif status != "open":
        blockers.append("spend_admission_lock_status_inconsistent")
    page_event = _mapping(row.get("page_event"))
    expected_page_required = expected_crossing
    if page_event.get("required") is not expected_page_required:
        blockers.append("spend_admission_lock_page_required_state_mismatch")
    expected_delivery_status = (
        "external_pending" if expected_page_required else "not_required"
    )
    if page_event.get("delivery_status") != expected_delivery_status:
        blockers.append("spend_admission_lock_page_delivery_state_inconsistent")
    if page_event.get("event_type") != "paid_spend_admission_locked":
        blockers.append("spend_admission_lock_page_event_type_invalid")
    if page_event.get("severity") != "critical":
        blockers.append("spend_admission_lock_page_event_severity_invalid")
    drain = _mapping(row.get("controlled_drain"))
    if (
        drain.get("status") != "not_required"
        or drain.get("new_paid_work_stopped") is not False
        or drain.get("provider_inventory_complete") is not True
    ):
        blockers.append("spend_admission_lock_drain_state_inconsistent")
    claim_boundary = _mapping(row.get("claim_boundary"))
    required_claim_boundaries = {
        "billing_export_is_external_input_not_live_api_proof",
        "page_event_is_not_notification_delivery_proof",
        "draining_is_not_provider_teardown_proof",
        "override_is_short_lived_and_two_person_audited",
    }
    if any(claim_boundary.get(key) is not True for key in required_claim_boundaries):
        blockers.append("spend_admission_lock_claim_boundary_invalid")
    return sorted(set(blockers))
