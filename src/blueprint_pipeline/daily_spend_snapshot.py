"""Offline daily spend projection from retained billing evidence. No network imports.

This observer never writes billing v1, admission, reservations, or repriced scene
costs. Cumulative deltas have posting time only; they are never usage-day costs.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from blueprint_pipeline.metered_call_evidence import digest

SCHEMA = "blueprint.daily_spend_snapshot.v1"
CHICAGO = ZoneInfo("America/Chicago")
COMPONENTS = {"gpu": "gpu", "disk": "disk", "bwd": "download", "bwu": "upload"}


def _hash(value: Any) -> str:
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def _time(value: Any) -> datetime | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (float, int)) and math.isfinite(value):
            return datetime.fromtimestamp(value, timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (ValueError, OverflowError, OSError):
        return None


def _iso(value: Any) -> str | None:
    parsed = _time(value)
    return parsed.isoformat() if parsed else None


def _exact_observation_time(value: Any) -> tuple[datetime, str] | None:
    # datetime truncates fractions beyond six digits. Retain that remainder for
    # binding comparisons, without rounding or rewriting the original evidence.
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}:\d{2})(?:[.,](\d+))?"
        r"(Z|[+-]\d{2}:?\d{2})", value, re.IGNORECASE,
    )
    if not match:
        return None
    whole_time = _time(match[1].upper() + match[3].upper())
    if whole_time is None:
        return None
    return whole_time, (match[2] or "").rstrip("0")


def _money(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, float, int)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _currency(value: Any) -> str | None:
    return value.upper() if isinstance(value, str) and re.fullmatch(r"[A-Za-z]{3}", value) else None


def _identity(value: Any) -> str | None:
    # Account/project/resource identifiers are pseudonymous in the reporting artifact.
    return (
        _hash(str(value))
        if value is not None and isinstance(value, (str, int)) and not isinstance(value, bool)
        else None
    )


def chicago_day(start: Any, end: Any) -> str | None:
    """Allocate only intervals wholly inside one Chicago day (end exclusive).

    This handles 23/25-hour DST days without spreading UTC buckets across days.
    """
    lower, upper = _time(start), _time(end)
    if not lower or not upper or upper <= lower:
        return None
    first = lower.astimezone(CHICAGO).date()
    last = (upper - timedelta(microseconds=1)).astimezone(CHICAGO).date()
    return first.isoformat() if first == last else None


def _period(month: Any) -> tuple[str | None, str | None]:
    if not isinstance(month, str) or not re.fullmatch(r"\d{4}-\d{2}", month):
        return None, None
    try:
        lower = datetime.fromisoformat(month + "-01").replace(tzinfo=timezone.utc)
        return lower.isoformat(), (
            lower + timedelta(days=calendar.monthrange(lower.year, lower.month)[1])
        ).isoformat()
    except ValueError:
        return None, None


def project_daily_snapshot(
    export: dict,
    receipt: dict,
    responses: list[tuple[dict, Any]],
    *,
    now: datetime,
    evidence_origin: str = "unverified_local_export",
    previous: dict | None = None,
) -> dict:
    """Pure projection over already received data; sources never imply total coverage."""
    rows: list[dict] = []
    gaps: set[str] = {
        "all_provider_all_day_coverage_unknown",
        "aws_collection_removed",
        "gcp_source_missing",
    }
    pages: dict[str, list[dict]] = {}
    collected = _iso(receipt.get("generated_at"))
    receipt_digest = receipt.get("receipt_digest")
    export_currency = _currency(export.get("currency"))

    def add(
        provider: str,
        service: str,
        raw: dict,
        *,
        kind: str,
        start: Any,
        end: Any,
        observed: Any,
        source: dict,
        charge_id: Any = None,
        components: dict | None = None,
        amount: Any = None,
        currency: Any = None,
        grain: str = "interval",
        invalid: bool = False,
    ) -> None:
        lower, upper = _iso(start), _iso(end)
        day = chicago_day(start, end) if kind == "actual" and grain == "interval" else None
        curr = _currency(currency)
        value = _money(amount)
        resource = _identity(
            raw.get("source")
            or raw.get("resourceId")
            or raw.get("resource_id")
            or raw.get("podId")
            or raw.get("endpointId")
            or raw.get("networkVolumeId")
        )
        account = _identity(raw.get("account_id"))
        project = _identity(raw.get("project_id"))
        charge_identity = _identity(charge_id)
        provider_charge_key = (
            provider == "vast" and service == "charges" and charge_identity is not None
        )
        identity_unknown = provider == "vast" and service == "charges" and charge_identity is None
        if identity_unknown:
            gaps.add("vast_stable_charge_id_missing_corrections_unknown")
            charge_identity = _hash(
                raw
            )  # Retain distinct rows; never assume a correction relationship.
        # Provider charge IDs are account-scoped. Interval/project/resource fields
        # can be corrected and belong to the revision, not the event identity.
        # A missing account retains the existing single-account collector scope;
        # do not merge records from additional accounts without their namespace.
        if provider_charge_key:
            key = _hash([provider, service, kind, account, charge_identity])
        else:
            key = _hash(
                [
                    provider,
                    service,
                    kind,
                    account,
                    project,
                    resource,
                    lower,
                    None if kind == "cumulative_actual" else upper,
                    charge_identity,
                ]
            )
        partial = (
            invalid
            or identity_unknown
            or value is None
            or curr is None
            or lower is None
            or upper is None
            or _time(end) <= _time(start)
        )
        source_observed = _iso(observed)
        observed_time = _time(observed)
        age = (now - observed_time).total_seconds() if observed_time else None
        freshness = (
            "unknown" if age is None else "stale" if age > 48 * 3600 or age < 0 else "current"
        )
        if freshness != "current":
            gaps.add(f"{provider}_provider_freshness_{freshness}")
        if partial:
            gaps.add(f"{provider}_invalid_amount_currency_or_interval")
        row = {
            "source_key": key,
            "provider": provider,
            "account": account,
            "project": project,
            "resource": resource,
            "service": service,
            "kind": kind,
            "amount": value,
            "currency": curr,
            "interval_start": lower,
            "interval_end": upper,
            "source_timezone": "UTC" if lower and upper else "unknown",
            "source_day": raw.get("day")
            if _time(raw.get("day"))
            or (isinstance(raw.get("day"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw["day"]))
            else None,
            "grain": grain,
            "reporting_timezone": "America/Chicago",
            "reporting_day": day,
            "allocation_gap": None
            if day
            else "source_interval_not_exactly_allocatable_to_Chicago_day",
            "components": components or {},
            "source_observed_at": source_observed,
            "collected_at": collected,
            "provider_freshness": freshness,
            "provider_age_seconds": age,
            "partial": partial,
            "evidence_origin": evidence_origin,
            "stable_correction_identity": not identity_unknown,
            "provenance": {
                "receipt_digest": receipt_digest,
                "response_digest": source.get("response_digest"),
                "request_query_digest": source.get("request_query_digest"),
            },
            "expense_rollup": "reference_only",
            "billed_cost": value
            if kind == "actual" and curr == "USD" and grain == "interval" and not partial
            else None,
            "queried_cohort_start": _iso(receipt.get("cohort_start_at")),
            "queried_cohort_end": _iso(receipt.get("cohort_end_at")),
            "adjustment_kind": "credit"
            if value is not None and value < 0
            else "charge"
            if value is not None
            else "unknown",
        }
        revision = {k: row[k] for k in ("source_key", "amount", "currency", "components")}
        if provider_charge_key:
            revision.update(
                {k: row[k] for k in (
                    "account", "project", "resource", "interval_start", "interval_end",
                )}
            )
        row["revision_id"] = _hash(revision)
        rows.append(row)

    # This reference is a cumulative cohort amount, even when the file was just refreshed.
    for provider, total in export.get("provider_totals_usd", {}).items():
        if provider not in {"runpod", "vast", "digitalocean"}:
            continue
        add(
            provider,
            "cohort_total",
            {},
            kind="cumulative_actual",
            start=receipt.get("cohort_start_at"),
            end=receipt.get("cohort_end_at"),
            observed=None,
            source={},
            amount=total,
            currency=export_currency,
            grain="cumulative",
        )
    balances = [
        payload
        for source, payload in responses
        if source.get("provider") == "digitalocean"
        and str(source.get("endpoint", "")).endswith("/balance")
        and isinstance(payload, dict)
    ]
    preview_observed = balances[-1].get("generated_at") if balances else None
    finalized_do_periods = {
        r.get("invoice_period")
        for src, body in responses
        if src.get("provider") == "digitalocean" and isinstance(body, dict)
        for r in (body.get("invoices") if isinstance(body.get("invoices"), list) else [])
        if isinstance(r, dict)
    }
    for source, payload in responses:
        provider = source.get("provider")
        endpoint = str(source.get("endpoint", ""))
        if provider not in {"runpod", "vast", "digitalocean"}:
            continue
        pages.setdefault(provider, []).append(
            {
                "response_digest": source.get("response_digest"),
                "request_query_digest": source.get("request_query_digest"),
                "complete": None,
            }
        )
        if provider == "runpod" and isinstance(payload, list):
            service = endpoint.rsplit("/", 1)[-1]
            if service not in {"pods", "endpoints", "networkvolumes"}:
                gaps.add("runpod_endpoint_unknown")
                continue
            for index, raw in enumerate(payload):
                if not isinstance(raw, dict):
                    gaps.add("runpod_row_invalid")
                    continue
                components = {
                    "amount": _money(raw.get("amount")),
                    "high_performance_storage": _money(raw.get("highPerformanceStorageAmount")),
                }
                storage_invalid = (
                    raw.get("highPerformanceStorageAmount") is not None
                    and components["high_performance_storage"] is None
                )
                values = [v for v in components.values() if v is not None]
                amount = (
                    sum(values)
                    if components["amount"] is not None and not storage_invalid
                    else None
                )
                add(
                    provider,
                    service,
                    raw,
                    kind="actual",
                    start=raw.get("startTime") or raw.get("start"),
                    end=raw.get("endTime") or raw.get("end"),
                    observed=raw.get("generated_at"),
                    source=source,
                    amount=amount,
                    currency=raw.get("currency", export_currency),
                    grain="year",
                    charge_id=raw.get("id") if raw.get("id") is not None else f"row:{index}",
                    components=components,
                )
            gaps.add("runpod_year_grain_has_no_daily_breakdown")
        elif (
            provider == "vast"
            and isinstance(payload, dict)
            and isinstance(payload.get("results"), list)
        ):
            pages[provider][-1]["complete"] = (
                payload.get("success") is True
                and "next_token" in payload
                and payload.get("next_token") is None
            )
            for raw in payload["results"]:
                if not isinstance(raw, dict):
                    gaps.add("vast_row_invalid")
                    continue
                components = {}
                component_valid = True
                for item in raw.get("items", []) if isinstance(raw.get("items", []), list) else []:
                    if isinstance(item, dict) and item.get("type") in COMPONENTS:
                        name = COMPONENTS[item["type"]]
                        if name in components:
                            component_valid = False
                        components[name] = _money(item.get("amount"))
                    else:
                        component_valid = False
                amount = _money(raw.get("amount"))
                components_invalid = (
                    not isinstance(raw.get("items"), list)
                    or set(components) != set(COMPONENTS.values())
                    or (
                        any(v is None for v in components.values())
                        or not component_valid
                        or not math.isclose(
                            sum(v or 0 for v in components.values()), amount or 0, abs_tol=1e-6
                        )
                    )
                )
                if components_invalid:
                    gaps.add("vast_components_partial_or_conflicting")
                add(
                    provider,
                    "charges",
                    raw,
                    kind="actual",
                    start=raw.get("start"),
                    end=raw.get("end"),
                    observed=payload.get("generated_at"),
                    source=source,
                    charge_id=raw.get("id"),
                    amount=raw.get("amount"),
                    currency=raw.get("currency", payload.get("currency", export_currency)),
                    components=components,
                    invalid=components_invalid or payload.get("success") is not True,
                )
        elif (
            provider == "digitalocean"
            and isinstance(payload, dict)
            and endpoint.endswith("/invoices")
        ):
            links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
            page_links = links.get("pages") if isinstance(links.get("pages"), dict) else {}
            next_page = page_links.get("next")
            pages[provider][-1]["complete"] = not next_page and isinstance(
                payload.get("invoices"), list
            )
            invoices = (
                payload.get("invoices", []) if isinstance(payload.get("invoices", []), list) else []
            )
            for raw in invoices + (
                [payload["invoice_preview"]]
                if isinstance(payload.get("invoice_preview"), dict)
                else []
            ):
                if not isinstance(raw, dict):
                    gaps.add("digitalocean_row_invalid")
                    continue
                preview = raw is payload.get("invoice_preview")
                overlap = preview and raw.get("invoice_period") in finalized_do_periods
                if overlap:
                    gaps.add("digitalocean_preview_overlaps_finalized_invoice")
                start, end = _period(raw.get("invoice_period") or raw.get("billing_period"))
                add(
                    provider,
                    "invoice_preview" if preview else "invoice",
                    raw,
                    kind="estimate" if preview else "invoice_reference",
                    start=start,
                    end=end,
                    observed=preview_observed if preview else raw.get("updated_at"),
                    source=source,
                    amount=raw.get("amount"),
                    currency=raw.get("currency", export_currency),
                    charge_id=raw.get("invoice_uuid"),
                    grain="month",
                    invalid=overlap,
                )
    for provider in ("runpod", "vast", "digitalocean"):
        if provider not in pages:
            gaps.add(f"{provider}_source_missing")
        elif provider != "runpod" and not any(p["complete"] for p in pages[provider]):
            gaps.add(f"{provider}_pagination_partial_or_unknown")
    # Immutable revisions with a stable source key. Conflicts within one source are unknown.
    by_key: dict[str, dict] = {}
    history: dict[str, dict] = {
        r["revision_id"]: r for r in (previous or {}).get("revision_history", [])
    }
    for row in rows:
        history.setdefault(row["revision_id"], row)
        prior = by_key.get(row["source_key"])
        if prior and (
            prior.get("conflicting_revisions") or prior["revision_id"] != row["revision_id"]
        ):
            prior = prior | {
                "amount": None,
                "billed_cost": None,
                "partial": True,
                "conflicting_revisions": sorted(
                    set(prior.get("conflicting_revisions", [prior["revision_id"]]))
                    | {row["revision_id"]}
                ),
            }
            gaps.add("conflicting_source_rows")
            by_key[row["source_key"]] = prior
        else:
            by_key[row["source_key"]] = row
    posting_changes = []
    prior_totals = {
        r["provider"]: r
        for r in (previous or {}).get("rows", [])
        if r.get("kind") == "cumulative_actual"
    }
    for row in by_key.values():
        prior = prior_totals.get(row["provider"])
        if (
            row["kind"] == "cumulative_actual"
            and prior
            and row["amount"] is not None
            and prior.get("amount") is not None
            and row["currency"] == prior.get("currency")
        ):
            change = row["amount"] - prior["amount"]
            if change:
                posting_changes.append(
                    {
                        "provider": row["provider"],
                        "amount": change,
                        "currency": row["currency"],
                        "kind": "cumulative_posting_change",
                        "posted_at": collected,
                        "reporting_day": None,
                        "usage_day_proven": False,
                        "prior_revision": prior["revision_id"],
                        "revision": row["revision_id"],
                    }
                )
    # Observed slices never imply all-day coverage. Invoices/cumulative/reservations do not enter this sum.
    slices: dict[tuple, float] = {}
    for row in by_key.values():
        if row["kind"] == "actual" and row["reporting_day"] and not row["partial"]:
            key = (row["provider"], row["reporting_day"], row["currency"])
            slices[key] = slices.get(key, 0) + row["amount"]
    # Deliberately no all-provider / daily total: empty or partial inputs are unknown.
    return {
        "schema_version": SCHEMA,
        "generated_at": now.isoformat(),
        "source_collected_at": collected,
        "accounting_pointer_freshness_is_provider_freshness": False,
        "provider_requests_added": 0,
        "evidence_origin": evidence_origin,
        "coverage": {
            "partial": True,
            "gaps": sorted(gaps),
            "pages": pages,
            "daily_total": None,
            "all_provider_coverage": "unknown",
        },
        "rows": list(by_key.values()),
        "revision_history": list(history.values()),
        "posting_changes": posting_changes,
        "observed_daily_slices": [
            {
                "provider": k[0],
                "day": k[1],
                "currency": k[2],
                "amount": v,
                "partial_day_coverage": True,
            }
            for k, v in sorted(slices.items())
        ],
        "snapshot_digest": _hash(list(by_key.values())),
    }


def _load_bound(export_path: Path, receipt_path: Path) -> tuple[dict, dict, list[tuple[dict, Any]]]:
    export = json.loads(export_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_digest"}
    if receipt.get("schema_version") != "blueprint.provider_billing_source_receipt.v1" or _hash(
        unsigned
    ) != receipt.get("receipt_digest"):
        raise ValueError("billing_receipt_digest_or_schema_invalid")
    # Equivalent timezone/fraction formatting is not a different observation.
    # Parse for this comparison only; retain the exact source bytes and hashes.
    export_observed = _exact_observation_time(export.get("generated_at"))
    receipt_observed = _exact_observation_time(receipt.get("generated_at"))
    if (
        export.get("schema_version") != "blueprint.provider_billing_export.v1"
        or export_observed is None
        or receipt_observed is None
        or export_observed != receipt_observed
        or export.get("provider_totals_usd") != receipt.get("provider_totals_usd")
    ):
        raise ValueError("billing_export_receipt_binding_invalid")
    responses = []
    for source in receipt.get("sources", []):
        if source.get("provider") not in {"runpod", "vast", "digitalocean"}:
            continue  # Historical AWS monthly data is excluded, never replayed.
        name = Path(str(source.get("retained_path", ""))).name
        if not re.fullmatch(r"response-\d+-[a-z]+\.json", name):
            raise ValueError("billing_response_filename_invalid")
        path = receipt_path.parent / name
        if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("billing_response_file_invalid")
        raw = path.read_bytes()
        if digest(raw) != source.get("response_digest") or len(raw) != source.get(
            "response_size_bytes"
        ):
            raise ValueError("billing_response_digest_or_size_invalid")
        responses.append((source, json.loads(raw)))
    return export, receipt, responses


def write_daily_snapshot(
    export_path: Path,
    receipt_path: Path,
    audit_root: Path,
    *,
    evidence_origin: str = "unverified_local_export",
) -> dict:
    output = export_path.parent / "daily_spend_snapshot.json"
    try:
        export, receipt, responses = _load_bound(export_path, receipt_path)
        previous = json.loads(output.read_bytes()) if output.is_file() else None
        snapshot = project_daily_snapshot(
            export,
            receipt,
            responses,
            now=datetime.now(timezone.utc),
            previous=previous,
            evidence_origin=evidence_origin,
        )
        from blueprint_pipeline.metered_call_evidence import summarize_attempts
        from blueprint_pipeline.notion_spend_projection import build_notion_projection

        snapshot["local_metered_calls"] = summarize_attempts(
            audit_root.parent / "metered-call-events"
        )
        snapshot["notion_projection"] = build_notion_projection(snapshot)
        descriptor, temp = tempfile.mkstemp(prefix=".daily-spend-", dir=output.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(snapshot, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, output)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        return {
            "status": "written",
            "snapshot_digest": snapshot["snapshot_digest"],
            "partial": True,
            "snapshot_path": str(output),
            "provider_requests_added": 0,
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Visibility failure is surfaced independently; it cannot change admission.
        return {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "provider_requests_added": 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--billing-export", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = write_daily_snapshot(
        args.billing_export, args.source_receipt, args.source_receipt.parent.parent
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "written" else 2


if __name__ == "__main__":
    raise SystemExit(main())
