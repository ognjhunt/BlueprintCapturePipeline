"""Compute the account's month and write the report the paid chokepoint reads.

The GPU spend guard calls ``refresh_report`` every minute. This is the
computing half of ``account_monthly_spend``; that module holds the owner's
configuration, the admission check and the alerts, and stays light because
every paid admission imports it. Nothing here admits, refuses, reserves or
calls a provider: it reads retained files only.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from .account_monthly_spend import (
    ACCOUNT_PROVIDERS,
    ALERT_PERCENTS,
    BILLING_AUDIT_ROOT_ENV,
    BILLING_EXPORT_ENV,
    BILLING_EXPORT_SCHEMA,
    BILLING_EXPORT_SCOPE,
    BILLING_MAX_AGE_SECONDS,
    BILLING_RECEIPT_SCHEMA,
    CONFIG_INVALID,
    LIMIT_REACHED,
    MAX_BASELINE_RECEIPTS,
    MAX_EXPORT_BYTES,
    MAX_RECEIPT_BYTES,
    OBSERVATION_MAX_AGE_SECONDS,
    OBSERVATION_SCHEMA,
    REPORT_MAX_JOBS,
    REPORT_SCHEMA,
    _AUDIT_DIRECTORY,
    _epoch,
    _json_object,
    _month_start,
    _regular_bytes,
    _utc,
    _write_public,
    configured_policy,
    effective_limit,
    read_provider_observation,
    report_path,
)
from . import account_monthly_spend as _account
from .paid_spend_meter import money, month_of, usd

#: The capacity tick reads the GCP budget report on the same rule.
GCP_REPORT_MAX_AGE_SECONDS = 30 * 60
_AUDIT_NAME_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def _reading(status: str, month_usd: Decimal | None = None, **extra: Any) -> dict[str, Any]:
    return {"status": status, "month_usd": month_usd, **extra}


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    body = {key: item for key, item in value.items() if key != digest_field}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _receipt_totals(directory: Path) -> dict[str, Decimal] | None:
    try:
        receipt = _json_object(_regular_bytes(directory / "provider_billing_source_receipt.json", MAX_RECEIPT_BYTES))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    totals = receipt.get("provider_totals_usd")
    if (receipt.get("schema_version") != BILLING_RECEIPT_SCHEMA or not isinstance(totals, dict)
            or receipt.get("receipt_digest") != _canonical_digest(receipt, digest_field="receipt_digest")):
        return None
    parsed = {str(name): money(value) for name, value in totals.items()}
    return {name: value for name, value in parsed.items() if value is not None}


def _audit_directories(audit_root: Path) -> list[str]:
    """Receipt directory names, oldest first; the names sort as their UTC times do."""

    try:
        with os.scandir(audit_root) as entries:
            return sorted(entry.name for entry in entries
                          if _AUDIT_DIRECTORY.fullmatch(entry.name) and not entry.is_symlink() and entry.is_dir())
    except OSError:
        return []


def _directory_time(name: str) -> float:
    return datetime.strptime(name, _AUDIT_NAME_FORMAT).replace(tzinfo=timezone.utc).timestamp()


def billing_month_readings(export_path: Path | None, audit_root: Path | None, *, now: float) -> dict[str, dict[str, Any]]:
    """This month's spend for each provider in the reconciled billing export.

    The export is cumulative since the cohort start. The month's spend is the
    export's total less the same provider's total in the last retained
    reconciliation receipt written before the month began. Without one, the
    first receipt of the month stands in and the reading says so: the spend
    between the month's start and that receipt is not counted.
    """

    if export_path is None:
        return {}
    try:
        export = _json_object(_regular_bytes(Path(export_path), MAX_EXPORT_BYTES))
    except (OSError, ValueError, UnicodeDecodeError):
        return {"vast": _reading("unreadable", reason="billing_export_unreadable"),
                "digitalocean": _reading("unreadable", reason="billing_export_unreadable")}
    totals = export.get("provider_totals_usd")
    generated = _epoch(export.get("generated_at"))
    if (export.get("schema_version") != BILLING_EXPORT_SCHEMA or export.get("scope") != BILLING_EXPORT_SCOPE
            or export.get("currency") != "USD" or not isinstance(totals, dict) or generated is None):
        return {"vast": _reading("invalid", reason="billing_export_invalid"),
                "digitalocean": _reading("invalid", reason="billing_export_invalid")}
    current = {str(name): money(value) for name, value in totals.items()}
    month_start = _month_start(now)
    if generated < month_start:
        return {name: _reading("other_month", reason="billing_export_from_previous_month") for name in current}
    stale = not 0 <= now - generated <= BILLING_MAX_AGE_SECONDS + 300
    baselines: dict[str, tuple[Decimal, float, bool]] = {}
    names = _audit_directories(Path(audit_root)) if audit_root is not None else []
    split = bisect.bisect_left(names, _utc(month_start).strftime(_AUDIT_NAME_FORMAT))
    before = names[max(0, split - MAX_BASELINE_RECEIPTS):split][::-1]
    after = [name for name in names[split:split + MAX_BASELINE_RECEIPTS] if _directory_time(name) <= generated]
    for candidates, late in ((before, False), (after, True)):
        for directory in candidates:
            if all(name in baselines for name in current):
                break
            receipt = _receipt_totals(Path(audit_root) / directory)
            for name, value in (receipt or {}).items():
                baselines.setdefault(name, (value, _directory_time(directory), late))
    readings = {}
    for name, total in sorted(current.items()):
        if total is None:
            readings[name] = _reading("invalid", reason="billing_export_total_invalid")
            continue
        baseline = baselines.get(name)
        if baseline is None:
            readings[name] = _reading("baseline_unavailable", reason="billing_month_baseline_unavailable")
            continue
        value, moment, late = baseline
        delta = total - value
        extra: dict[str, Any] = {"baseline_at": _utc(moment).isoformat(), "observed_at": export.get("generated_at")}
        reason = None
        if delta < 0:
            delta, reason = Decimal(0), "billing_total_below_month_baseline"
        elif late:
            reason = "billing_month_baseline_after_month_start"
        elif stale:
            reason = "billing_export_stale"
        status = "observed" if reason is None else "partial"
        readings[name] = _reading(status, delta, **({"reason": reason} if reason else {}), **extra)
    return readings


def gcp_budget_reading(*, now: float, environ: Mapping[str, str]) -> dict[str, Any]:
    """The project budget guard's month cost, when its billing month is this UTC month.

    Read from the guard's own report, as ``provider_spend_alerts.observe_gcp_spend``
    reads it. Google bills by the Los Angeles month, so for the first hours of a
    UTC month the report still names the previous one and is not counted.
    """

    from .gcp_monthly_budget_admission import CONFIG_ENV as GCP_CONFIG_ENV
    from .gcp_monthly_budget_admission import SCHEMA, _read, _time, load_policy

    configured = str(environ.get(GCP_CONFIG_ENV) or "").strip()
    if not configured:
        return _reading("unobserved", reason="gcp_budget_guard_not_configured")
    try:
        policy = load_policy(Path(configured))
        report = _read(Path(policy["report_path"]))
    except Exception:  # noqa: BLE001 - any unreadable policy or report is one reason
        return _reading("unreadable", reason="gcp_budget_report_unreadable")
    if report.get("schema_version") != SCHEMA or report.get("project_id") != policy["project_id"]:
        return _reading("invalid", reason="gcp_budget_report_invalid")
    cost = money(report.get("observed_cost_usd"))
    if cost is None:
        return _reading("unobserved", reason="gcp_month_cost_not_reported")
    if report.get("period") != month_of(now):
        return _reading("other_month", reason="gcp_billing_month_differs_from_utc_month")
    try:
        age = now - _time(str(report.get("generated_at"))).timestamp()
    except (TypeError, ValueError):
        age = math.inf
    observed_at = str(report.get("generated_at"))
    if not 0 <= age <= GCP_REPORT_MAX_AGE_SECONDS:
        return _reading("partial", cost, reason="gcp_budget_report_stale", observed_at=observed_at)
    return _reading("observed", cost, observed_at=observed_at)


def observation_readings(observation: Mapping[str, Any] | None, *, now: float) -> dict[str, dict[str, Any]]:
    """OpenAI and World Labs month-to-date from the capacity tick's last observation."""

    if not isinstance(observation, Mapping) or observation.get("schema_version") != OBSERVATION_SCHEMA:
        return {name: _reading("unobserved", reason="provider_observation_missing") for name in ("openai", "world_labs")}
    observed = observation.get("observed_at_epoch")
    known = isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isfinite(observed)
    age = now - observed if known else math.inf
    stamp = {"observed_at": _utc(observed).isoformat()} if known else {}
    readings = {}
    providers = observation.get("providers") if isinstance(observation.get("providers"), Mapping) else {}
    for name in ("openai", "world_labs"):
        row = providers.get(name)
        amount = money(row.get("month_to_date_usd")) if isinstance(row, Mapping) else None
        if not isinstance(row, Mapping) or amount is None:
            readings[name] = _reading("unobserved", reason=",".join((row or {}).get("blockers") or []) or "provider_month_unreported")
        elif row.get("month") != month_of(now):
            readings[name] = _reading("other_month", reason="provider_observation_from_previous_month")
        elif not 0 <= age <= OBSERVATION_MAX_AGE_SECONDS:
            readings[name] = _reading("stale", amount, reason="provider_observation_stale", **stamp)
        elif row.get("status") != "observed" or row.get("blockers"):
            readings[name] = _reading("partial", amount, **stamp,
                                      reason=",".join(row.get("blockers") or []) or "provider_reading_partial")
        else:
            readings[name] = _reading("observed", amount, **stamp)
    return readings


def collect_readings(*, now: float, environ: Mapping[str, str], observation: Mapping[str, Any] | None,
                     billing_export_path: Path | None = None, audit_root: Path | None = None) -> dict[str, dict[str, Any]]:
    export = billing_export_path or (Path(environ[BILLING_EXPORT_ENV]) if environ.get(BILLING_EXPORT_ENV) else None)
    audit = audit_root or (Path(environ[BILLING_AUDIT_ROOT_ENV]) if environ.get(BILLING_AUDIT_ROOT_ENV) else None)
    billing = billing_month_readings(export, audit, now=now)
    readings: dict[str, dict[str, Any]] = {name: {**row, "source": "provider_billing_export"} for name, row in billing.items()
                                           if name in {"vast", "digitalocean"}}
    budget = gcp_budget_reading(now=now, environ=environ)
    exported = billing.get("gcp")
    candidates = [row for row in ({**budget, "source": "gcp_monthly_budget_guard"},
                                  {**exported, "source": "provider_billing_export"} if exported else None)
                  if row is not None and row.get("month_usd") is not None]
    readings["gcp"] = max(candidates, key=lambda row: row["month_usd"]) if candidates else {**budget, "source": "gcp_monthly_budget_guard"}
    for name, row in observation_readings(observation, now=now).items():
        readings[name] = {**row, "source": "provider_spend_observation"}
    return readings


#: Meter gaps that mean a source of holds could not be read, by the source they belong to.
_HOLD_SOURCE_GAPS = {"beta_policy_screening": "paid_spend_meter_screening_",
                     "paid_resource_admission": "paid_spend_meter_journal_"}


def _same_month(previous: Mapping[str, Any] | None, period: str) -> Mapping[str, Any]:
    """The previous report, when it is this month's; else nothing to hold."""

    if isinstance(previous, Mapping) and previous.get("schema_version") == REPORT_SCHEMA and previous.get("period") == period:
        return previous
    return {}


def _previous_holds(previous: Mapping[str, Any]) -> dict[str, Decimal]:
    meter = previous.get("meter") if isinstance(previous.get("meter"), Mapping) else {}
    sources = (meter.get("totals") or {}).get("by_source") if isinstance(meter.get("totals"), Mapping) else None
    held = {}
    for name, row in (sources.items() if isinstance(sources, Mapping) else ()):
        amount = money(row.get("held_usd")) if isinstance(row, Mapping) else None
        if amount is not None:
            held[str(name)] = amount
    return held


def build_report(*, now: float, policy: Mapping[str, Any] | None, policy_error: str | None, config: Path,
                 readings: Mapping[str, Mapping[str, Any]], meter: Mapping[str, Any],
                 previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The account's month: what counts, against which limit, and whether admission is open.

    ``previous`` is the last report written. Within one month a monitoring
    failure never lowers the count: a provider never counts less than it did
    in the previous report, and a source of holds that cannot be read keeps the
    holds it last had. A held figure is marked ``stale`` (its reading failed)
    or ``partial`` (its reading came back lower), with the time it was observed.
    """

    from .paid_spend_meter import SOURCE_ATLAS, public_view

    period = month_of(now)
    prior = _same_month(previous, period)
    prior_providers = prior.get("providers") if isinstance(prior.get("providers"), Mapping) else {}
    exact = meter["_exact"]
    metered_by_provider: Mapping[str, Decimal] = exact["by_provider_settled_usd"]
    providers: dict[str, dict[str, Any]] = {}
    gaps = list(meter.get("gaps") or [])
    spend = Decimal(0)
    for name in sorted(set(ACCOUNT_PROVIDERS) | set(metered_by_provider) | set(prior_providers)):
        reading = readings.get(name) or _reading("unobserved", reason="no_reading_for_provider")
        reconciled = reading.get("month_usd")
        metered = metered_by_provider.get(name, Decimal(0))
        counted = max(reconciled if reconciled is not None else Decimal(0), metered)
        status = str(reading.get("status"))
        if name not in ACCOUNT_PROVIDERS and reconciled is None:
            status = "metered_only"
        observed_at = reading.get("observed_at") if reconciled is not None and counted == reconciled else None
        row = {"reconciled_month_usd": None if reconciled is None else usd(reconciled),
               "metered_settled_usd": usd(metered),
               **{key: reading[key] for key in ("reason", "baseline_at", "source") if reading.get(key)}}
        before = prior_providers.get(name) if isinstance(prior_providers.get(name), Mapping) else {}
        floor = money(before.get("counted_usd"))
        if floor is not None and floor > counted:
            # Keep the last count: a failed or lower reading is a monitoring gap, not a refund.
            status = "stale" if reconciled is None else "partial"
            counted = floor
            observed_at = before.get("observed_at")
            row["held_from_previous_report"] = True
            gaps.append(f"account_monthly_spend_provider_held_at_last_count:{name}")
        if status not in {"observed", "metered_only"}:
            gaps.append(f"account_monthly_spend_provider_{status}:{name}")
        providers[name] = {"status": status, **row, "counted_usd": usd(counted), "observed_at": observed_at}
        spend += counted
    by_source: dict[str, Decimal] = {name: amounts["held_usd"] for name, amounts in exact["by_source"].items()}
    prior_holds = _previous_holds(prior)
    held_sources: dict[str, dict[str, Any]] = {}
    for source, prefix in _HOLD_SOURCE_GAPS.items():
        unread = any(str(gap).startswith(prefix) for gap in meter.get("gaps") or [])
        current = by_source.get(source, Decimal(0))
        if unread and prior_holds.get(source, Decimal(0)) > current:
            by_source[source] = prior_holds[source]
            held_sources[source] = {"status": "stale", "held_usd": usd(prior_holds[source]),
                                    "held_from_previous_report": True}
            gaps.append(f"account_monthly_spend_holds_held_at_last_count:{source}")
    held = sum((amount for name, amount in by_source.items() if name != SOURCE_ATLAS), Decimal(0))
    counted_total = spend + held
    limit = effective_limit(policy, period) if policy is not None else None
    fraction = float(counted_total / limit) if limit else None
    reached = [percent for percent in ALERT_PERCENTS if limit is not None and counted_total * 100 >= limit * percent]
    limit_reached = limit is not None and counted_total >= limit
    blockers = [CONFIG_INVALID] if policy_error else ([LIMIT_REACHED] if limit_reached else [])
    return {
        "schema_version": REPORT_SCHEMA, "generated_at": _utc(now).isoformat(), "period": period,
        "cap_status": "config_invalid" if policy_error else ("set" if policy is not None else "not_set"),
        "config_path": str(config),
        "policy_sha256": policy.get("policy_sha256") if policy is not None else None,
        "limit_usd": None if limit is None else str(limit),
        "alert_lines_usd": {str(percent): usd(limit * percent / 100) for percent in ALERT_PERCENTS} if limit else None,
        "spend_usd": usd(spend), "held_usd": usd(held), "counted_usd": usd(counted_total),
        "held_by_source_usd": {name: usd(amount) for name, amount in sorted(by_source.items()) if name != SOURCE_ATLAS},
        **({"held_sources_kept": held_sources} if held_sources else {}),
        "fraction": None if fraction is None else round(fraction, 6),
        "alert_threshold_percent": reached[-1] if reached else None,
        "limit_reached": limit_reached, "admission_allowed": not blockers, "blockers": blockers,
        "providers": providers,
        "meter": public_view(meter, max_jobs=REPORT_MAX_JOBS),
        "complete": not gaps, "gaps": sorted(set(gaps)),
        "claim_boundary": ("Provider readings lag their bills and held reservations are upper bounds; the limit "
                           "stops new paid admissions, not provider charges already running."),
    }


def refresh_report(*, now: float | None = None, environ: Mapping[str, str] | None = None,
                   billing_export_path: Path | None = None, audit_root: Path | None = None) -> dict[str, Any]:
    """Recompute the account's month and rewrite the admission report (the spend guard calls this)."""

    from .paid_spend_meter_sources import collect_meter_view

    moment = _account._clock() if now is None else float(now)
    env = os.environ if environ is None else environ
    policy, error, config = configured_policy(env)
    observation = read_provider_observation(env)
    readings = collect_readings(now=moment, environ=env, observation=observation,
                                billing_export_path=billing_export_path, audit_root=audit_root)
    meter = collect_meter_view(now=moment, world_labs=_account.current_world_labs(observation, now=moment), environ=env)
    destination = report_path(policy, env)
    previous = None
    if destination is not None:
        try:
            previous = _account.read_report(destination)
        except (OSError, ValueError, UnicodeDecodeError):
            previous = None
    report = build_report(now=moment, policy=policy, policy_error=error, config=config, readings=readings,
                          meter=meter, previous=previous)
    report["report_path"] = str(destination) if destination is not None else None
    if destination is not None:
        _write_public(destination, report)
    return report


__all__ = ["billing_month_readings", "build_report", "collect_readings", "gcp_budget_reading",
           "observation_readings", "refresh_report"]
