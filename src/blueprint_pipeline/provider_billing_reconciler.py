"""Read-only provider billing reconciliation for the paid-spend admission lock.

The synchronizer reads the same file-backed provider credentials as the GPU
spend guard, fetches cumulative cohort charges from fixed official endpoints,
retains the exact private response bytes, and atomically writes the narrow
``blueprint.provider_billing_export.v1`` input consumed by the guard.  It never
creates, changes, or deletes provider resources. GCP can use a fresh retained
official project CSV with explicit account and inventory scope. AWS billing
remains uncovered: automatic paid Cost Explorer queries have been removed.
RunPod billing is no longer collected: RunPod is retired (2026-11-05).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from blueprint_pipeline.metered_call_evidence import MeteredCallEvidenceError, MeteredTransport
from blueprint_pipeline.provider_billing_audit_retention import (
    ProviderBillingAuditRetentionError,
    audit_root_lock,
    retain_billing_response,
)
from .vast_official_billing_serialization import (
    BILLING_SOURCE_SCHEMA_VERSION as BILLING_SOURCE_SCHEMA_VERSION,
    VAST_CHARGES_URL as VAST_CHARGES_URL,
    MAX_RESPONSE_BYTES as MAX_RESPONSE_BYTES,
)

BILLING_EXPORT_SCHEMA_VERSION = "blueprint.provider_billing_export.v1"
BILLING_SCOPE = "blueprint_beta_100_user_cohort"
#: Known provider ids. RunPod and AWS stay so retained exports still read.
PROVIDERS = ("runpod", "vast", "digitalocean", "aws", "gcp")
DIGITALOCEAN_API_URL = "https://api.digitalocean.com/v2"
MAX_PAGES = 100
AWS_BILLING_UNAVAILABLE_REASON = "aws_billing_unavailable_paid_api_collection_removed"
#: RunPod is retired (owner decision 2026-11-05): no lane can allocate it, and
#: its billing request failed on every call (85 of 85 on 2026-11-05), so it is not
#: requested. The spend guard still inventories RunPod while a key exists; a live
#: RunPod pod keeps paid admission blocked because this export cannot cover it.
RUNPOD_BILLING_RETIRED_REASON = "runpod_billing_retired_provider_not_collected"
#: The longest a caller may ask an existing export to stand in for a refresh.
#: The spend guard and the paid-spend admission lock both refuse billing
#: evidence older than 24 hours; an hour keeps any interval far inside that.
MAX_MIN_REFRESH_INTERVAL_SECONDS = 3600
#: The spend guard refuses a larger export, so a larger one is never retained.
MAX_EXPORT_BYTES = 1024 * 1024
#: Upper bound on the spend guard's ``--json-report`` read to vouch for coverage.
MAX_GUARD_REPORT_BYTES = 16 * 1024 * 1024
# DigitalOcean documents its current invoice preview as a daily artifact.  A
# strict 24-hour wall-clock cutoff therefore races that provider cadence: the
# same official response becomes inadmissible in the interval between crossing
# 24 hours and the next daily refresh.  Keep the fail-closed bound, but allow
# one complete daily refresh interval of scheduling tolerance.
DIGITALOCEAN_BALANCE_MAX_AGE = timedelta(hours=48)


class ProviderBillingReconciliationError(RuntimeError):
    """Raised when current provider billing cannot be proven."""


Transport = Callable[[urllib.request.Request, float], bytes]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return _digest_bytes(_canonical_json(payload).encode("utf-8"))


def _parse_time(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProviderBillingReconciliationError(f"{field}_invalid") from exc
    if parsed.tzinfo is None:
        raise ProviderBillingReconciliationError(f"{field}_timezone_missing")
    return parsed.astimezone(timezone.utc)


def _money(value: Any, *, field: str) -> float:
    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderBillingReconciliationError(f"{field}_invalid") from exc
    if not math.isfinite(amount) or amount < 0:
        raise ProviderBillingReconciliationError(f"{field}_invalid")
    return amount


def _read_secret(path: Path) -> str:
    if path.is_symlink():
        raise ProviderBillingReconciliationError(f"secret_symlink_forbidden:{path.name}")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise ProviderBillingReconciliationError(f"secret_missing:{path.name}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 64 * 1024:
        raise ProviderBillingReconciliationError(f"secret_file_invalid:{path.name}")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        raise ProviderBillingReconciliationError(
            f"secret_file_writable_by_group_or_world:{path.name}"
        )
    value = path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ProviderBillingReconciliationError(f"secret_value_invalid:{path.name}")
    return value


def _default_transport(request: urllib.request.Request, timeout: float) -> bytes:
    parsed = urllib.parse.urlsplit(request.full_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ProviderBillingReconciliationError("provider_billing_endpoint_not_public_https")
    try:
        with urllib.request.urlopen(  # nosec B310 - public HTTPS endpoint validated above
            request, timeout=timeout
        ) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise ProviderBillingReconciliationError("provider_billing_request_failed") from exc
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ProviderBillingReconciliationError("provider_billing_response_too_large")
    return payload


def _request_json(
    *,
    provider: str,
    url: str,
    token: str,
    timeout: float,
    transport: Transport,
    audit_rows: list[dict[str, Any]],
) -> Any:
    if not url.startswith((VAST_CHARGES_URL, DIGITALOCEAN_API_URL)):
        raise ProviderBillingReconciliationError("provider_billing_endpoint_not_allowlisted")
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        method="GET",
    )
    payload = transport(request, timeout)
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProviderBillingReconciliationError(
            f"provider_billing_response_invalid_json:{provider}"
        ) from exc
    audit_rows.append(
        {
            "provider": provider,
            "endpoint": urllib.parse.urlsplit(url)._replace(query="").geturl(),
            "request_query_digest": _digest_bytes(urllib.parse.urlsplit(url).query.encode("utf-8")),
            "response_digest": _digest_bytes(payload),
            "response_size_bytes": len(payload),
            "response_bytes": payload,
        }
    )
    return decoded




def _vast_total(
    *,
    token: str,
    start_at: datetime,
    end_at: datetime,
    timeout: float,
    transport: Transport,
    audit_rows: list[dict[str, Any]],
) -> float:
    total = 0.0
    after_token: str | None = None
    seen_tokens: set[str] = set()
    for _page in range(MAX_PAGES):
        query: dict[str, Any] = {
            "select_filters": _canonical_json(
                {
                    "day": {
                        "gte": int(start_at.timestamp()),
                        "lte": int(end_at.timestamp()),
                    }
                }
            ),
            "limit": 500,
        }
        if after_token:
            query["after_token"] = after_token
        payload = _request_json(
            provider="vast",
            url=f"{VAST_CHARGES_URL}?{urllib.parse.urlencode(query)}",
            token=token,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        )
        if not isinstance(payload, Mapping) or payload.get("success") is not True:
            raise ProviderBillingReconciliationError("vast_billing_response_invalid")
        rows = payload.get("results")
        if not isinstance(rows, list):
            raise ProviderBillingReconciliationError("vast_billing_results_invalid")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ProviderBillingReconciliationError(f"vast_billing_row_invalid:{index}")
            total += _money(row.get("amount"), field="vast_charge_amount")
        next_token = payload.get("next_token")
        if next_token is None:
            return total
        if not isinstance(next_token, str) or not next_token or next_token in seen_tokens:
            raise ProviderBillingReconciliationError("vast_billing_cursor_invalid")
        seen_tokens.add(next_token)
        after_token = next_token
    raise ProviderBillingReconciliationError("vast_billing_page_cap_exceeded")


def _digitalocean_total(
    *,
    token: str,
    start_at: datetime,
    end_at: datetime,
    timeout: float,
    transport: Transport,
    audit_rows: list[dict[str, Any]],
) -> float:
    balance = _request_json(
        provider="digitalocean",
        url=f"{DIGITALOCEAN_API_URL}/customers/my/balance",
        token=token,
        timeout=timeout,
        transport=transport,
        audit_rows=audit_rows,
    )
    if not isinstance(balance, Mapping):
        raise ProviderBillingReconciliationError("digitalocean_balance_invalid")
    generated_at = _parse_time(
        balance.get("generated_at"), field="digitalocean_balance_generated_at"
    )
    if generated_at > end_at or end_at - generated_at > DIGITALOCEAN_BALANCE_MAX_AGE:
        raise ProviderBillingReconciliationError("digitalocean_balance_stale")

    total = 0.0
    seen_invoice_ids: set[str] = set()
    start_period = start_at.strftime("%Y-%m")
    for page in range(1, MAX_PAGES + 1):
        query = urllib.parse.urlencode({"per_page": 200, "page": page})
        payload = _request_json(
            provider="digitalocean",
            url=f"{DIGITALOCEAN_API_URL}/customers/my/invoices?{query}",
            token=token,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("invoices"), list):
            raise ProviderBillingReconciliationError("digitalocean_invoices_invalid")
        invoices = payload["invoices"]
        for index, row in enumerate(invoices):
            if not isinstance(row, Mapping):
                raise ProviderBillingReconciliationError(f"digitalocean_invoice_invalid:{index}")
            invoice_id = str(row.get("invoice_uuid") or "")
            period = str(row.get("invoice_period") or "")
            if not invoice_id or invoice_id in seen_invoice_ids:
                raise ProviderBillingReconciliationError("digitalocean_invoice_identity_invalid")
            seen_invoice_ids.add(invoice_id)
            if period >= start_period:
                total += _money(row.get("amount"), field="digitalocean_invoice_amount")
        if page == 1:
            preview = payload.get("invoice_preview")
            if (
                isinstance(preview, Mapping)
                and str(preview.get("invoice_period") or "") >= start_period
            ):
                total += _money(preview.get("amount"), field="digitalocean_invoice_preview_amount")
        links = payload.get("links")
        pages = links.get("pages") if isinstance(links, Mapping) else None
        if not isinstance(pages, Mapping) or not pages.get("next"):
            return total
    raise ProviderBillingReconciliationError("digitalocean_billing_page_cap_exceeded")


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def reconcile_provider_billing(
    *,
    secrets_dir: str | Path,
    billing_export_path: str | Path,
    audit_root: str | Path,
    start_at: str,
    now: datetime | None = None,
    timeout: float = 30.0,
    transport: Transport = _default_transport,
    aws_account_id: str | None = None,
    aws_credentials_file: str | Path | None = None,
    aws_profile: str | None = None,
    aws_credentials: Any | None = None,
    required_providers: Sequence[str] = (),
    gcp_report_receipt: str | Path | None = None,
    gcp_billing_account_id: str = "",
    gcp_project_ids: Sequence[str] = (),
) -> dict[str, Any]:
    # Preserve legacy call signatures without reading credentials or making AWS requests.
    del aws_account_id, aws_credentials_file, aws_profile, aws_credentials
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = _parse_time(start_at, field="cohort_start_at")
    if start >= current:
        raise ProviderBillingReconciliationError("cohort_start_at_not_before_end")
    secret_root = Path(secrets_dir).expanduser().resolve()
    requested_required = {str(provider).strip() for provider in required_providers}
    if not requested_required.issubset(PROVIDERS):
        raise ProviderBillingReconciliationError("required_provider_invalid")
    required_provider_ids = requested_required
    if "aws" in required_provider_ids:
        raise ProviderBillingReconciliationError(AWS_BILLING_UNAVAILABLE_REASON)
    if "runpod" in required_provider_ids:
        raise ProviderBillingReconciliationError(RUNPOD_BILLING_RETIRED_REASON)
    # Same dispatch seam and call count; the journal survives collector failure/crash.
    evidence_origin = "retained_provider_response" if transport is _default_transport else "development_only"
    transport = MeteredTransport(transport, Path(audit_root).expanduser().resolve().parent / "metered-call-events")
    audit_rows: list[dict[str, Any]] = []
    totals: dict[str, float] = {}
    source_times = [current]
    optional_provider_failures: dict[str, str] = {
        "aws": AWS_BILLING_UNAVAILABLE_REASON,
        "runpod": RUNPOD_BILLING_RETIRED_REASON,
    }
    security_blockers = (
        "secret_symlink_forbidden:",
        "secret_file_invalid:",
        "secret_file_writable_by_group_or_world:",
    )

    def record_provider(provider: str, collect: Callable[[], float]) -> None:
        try:
            totals[provider] = collect()
        except (ProviderBillingReconciliationError, MeteredCallEvidenceError) as exc:
            error = str(exc)
            if provider in required_provider_ids or error.startswith(security_blockers):
                raise
            optional_provider_failures[provider] = error

    record_provider(
        "vast",
        lambda: _vast_total(
            token=_read_secret(secret_root / "vast_api_key"),
            start_at=start,
            end_at=current,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        ),
    )
    record_provider(
        "digitalocean",
        lambda: _digitalocean_total(
            token=_read_secret(secret_root / "digitalocean_api_token"),
            start_at=start,
            end_at=current,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        ),
    )

    def collect_gcp() -> float:
        from blueprint_pipeline.gcp_billing_report import GCPBillingReportError, read_report

        if not gcp_report_receipt:
            raise ProviderBillingReconciliationError("gcp_billing_report_unconfigured")
        try:
            total, captured, row = read_report(
                gcp_report_receipt, billing_account_id=gcp_billing_account_id,
                project_ids=list(gcp_project_ids), cohort_start=start, now=current,
            )
        except GCPBillingReportError as exc:
            raise ProviderBillingReconciliationError(str(exc)) from exc
        source_times.append(captured)
        audit_rows.append(row)
        return total

    record_provider("gcp", collect_gcp)

    if not totals:
        raise ProviderBillingReconciliationError("provider_billing_no_provider_covered")
    totals = {provider: round(value, 6) for provider, value in totals.items()}
    covered_provider_ids = sorted(totals)
    uncovered_provider_ids = sorted(set(PROVIDERS) - set(totals))
    generated_at = current.isoformat()
    audit_root_path = Path(audit_root).expanduser().resolve()
    try:
        with audit_root_lock(audit_root_path, create=True) as (
            locked_audit_root,
            root_info,
        ):
            audit_directory = locked_audit_root / current.strftime(
                "%Y%m%dT%H%M%S.%fZ"
            )
            try:
                audit_directory.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_directory_exists"
                ) from exc
            source_rows: list[dict[str, Any]] = []
            response_paths: list[Path] = []
            try:
                for index, row in enumerate(audit_rows, start=1):
                    response_name = f"response-{index:03d}-{row['provider']}.json"
                    response_path = retain_billing_response(
                        audit_root=locked_audit_root,
                        audit_directory=audit_directory,
                        response_name=response_name,
                        payload=row["response_bytes"],
                        root_info=root_info,
                    )
                    response_paths.append(response_path)
                    source_rows.append(
                        {
                            key: value
                            for key, value in row.items()
                            if key != "response_bytes"
                        }
                        | {"retained_path": str(response_path)}
                    )
                source_receipt = {
                    "schema_version": BILLING_SOURCE_SCHEMA_VERSION,
                    "status": "reconciled",
                    "generated_at": generated_at,
                    "cohort_start_at": start.isoformat(),
                    "cohort_end_at": min(source_times).isoformat(),
                    "provider_totals_usd": totals,
                    "required_provider_ids": sorted(required_provider_ids),
                    "covered_provider_ids": covered_provider_ids,
                    "uncovered_provider_ids": uncovered_provider_ids,
                    "optional_provider_failures": optional_provider_failures,
                    "sources": source_rows,
                    "provider_mutation_performed": False,
                    "raw_secret_values_recorded": False,
                }
                if min(source_times) != current:
                    # Bind the older export observation without re-dating the
                    # provider report or the actual collection receipt.
                    source_receipt["billing_export_generated_at"] = min(source_times).isoformat()
                source_receipt["receipt_digest"] = _canonical_digest(
                    source_receipt, digest_field="receipt_digest"
                )
                source_path = audit_directory / "provider_billing_source_receipt.json"
                _atomic_write(
                    source_path,
                    (_canonical_json(source_receipt) + "\n").encode("utf-8"),
                )
            except BaseException:
                for response_path in reversed(response_paths):
                    if (
                        response_path.parent == audit_directory
                        and not response_path.is_symlink()
                        and response_path.is_file()
                    ):
                        response_path.unlink()
                if audit_directory.is_dir() and not any(audit_directory.iterdir()):
                    audit_directory.rmdir()
                raise
    except ProviderBillingAuditRetentionError as exc:
        raise ProviderBillingReconciliationError(str(exc)) from exc
    billing_export = {
        "schema_version": BILLING_EXPORT_SCHEMA_VERSION,
        # Re-reading a downloaded report must never renew its freshness lease.
        "generated_at": min(source_times).isoformat(),
        "currency": "USD",
        "scope": BILLING_SCOPE,
        "provider_totals_usd": totals,
    }
    export_path = Path(billing_export_path).expanduser().resolve()
    _atomic_write(export_path, (_canonical_json(billing_export) + "\n").encode("utf-8"))
    # Reporting is an offline observer, never an admission or reservation input.
    try:
        from blueprint_pipeline.daily_spend_snapshot import write_daily_snapshot
        daily_status = write_daily_snapshot(export_path, Path(source_path), Path(audit_root).expanduser().resolve(), evidence_origin=evidence_origin)
    except Exception as exc:
        # Observer defects cannot invalidate already committed billing evidence.
        daily_status = {"status": "unavailable", "error_type": type(exc).__name__, "provider_requests_added": 0}

    return {
        "schema_version": "blueprint.provider_billing_reconciliation_run.v1",
        "status": "reconciled",
        "generated_at": generated_at,
        "provider_totals_usd": totals,
        "required_provider_ids": sorted(required_provider_ids),
        "covered_provider_ids": covered_provider_ids,
        "uncovered_provider_ids": uncovered_provider_ids,
        "optional_provider_failures": optional_provider_failures,
        "billing_export_path": str(export_path),
        "billing_export_digest": _digest_bytes(export_path.read_bytes()),
        "source_receipt_path": str(source_path),
        "source_receipt_digest": source_receipt["receipt_digest"],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "daily_spend_snapshot": daily_status,
    }


def _euid() -> int:
    return os.geteuid()


def _own_file_bytes(path: Path, limit: int) -> bytes | None:
    """A regular file this account wrote, not group- or world-writable, read without following links."""
    try:
        observed = path.lstat()
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_size > limit
            or observed.st_uid != _euid()
            or stat.S_IMODE(observed.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            return None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
                return None
            raw = stream.read(limit + 1)
    except OSError:
        return None
    return raw if len(raw) <= limit else None


def _guard_live_providers(
    report_path: str | Path,
    *,
    export_digest: str,
    export_providers: set[str],
    now: datetime,
    max_age_seconds: float,
) -> list[str] | None:
    """The providers the spend guard last required, when this export covers every one.

    The guard blocks paid admission while a provider with live instances is
    missing from the export (``reconcile_billing_export``), and only the guard
    sees live inventory. Its own ``--json-report``, written by this account
    within the interval, names them. A report that judged this very export and
    found anything wrong is honoured too. Anything else returns ``None``.
    """

    raw = _own_file_bytes(Path(report_path).expanduser(), MAX_GUARD_REPORT_BYTES)
    if raw is None:
        return None
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    billing = report.get("billing_reconciliation") if isinstance(report, dict) else None
    if not isinstance(billing, dict):
        return None
    try:
        generated_at = _parse_time(report.get("generated_at"), field="spend_guard_report_generated_at")
    except ProviderBillingReconciliationError:
        return None
    if not 0 <= (now.astimezone(timezone.utc) - generated_at).total_seconds() <= max_age_seconds:
        return None
    live = billing.get("live_providers")
    if not isinstance(live, list) or not all(isinstance(provider, str) for provider in live):
        return None
    if not set(live).issubset(export_providers):
        return None
    if billing.get("billing_export_sha256") == export_digest and billing.get("blockers"):
        return None
    return sorted(set(live))


def retained_fresh_export(
    billing_export_path: str | Path,
    *,
    now: datetime,
    min_refresh_interval_seconds: float,
    required_providers: Sequence[str] = (),
    spend_guard_report: str | Path | None = None,
) -> dict[str, Any] | None:
    """The current export, when a refresh inside the interval would only repeat it.

    ``blueprint-gpu-spend-guard.service`` wants and orders after this unit, so
    every guard start -- its own one-minute timer and every unit that wants the
    guard -- ran a full refresh first: 31 runs and 1,279 s of CPU in one hour on
    2026-11-02, each up to 920 MB, against a designed ten-minute cadence. The
    dependency stays, because it is what guarantees a refresh attempt before the
    guard reads older evidence; the run itself now ends here while the export
    this account wrote is still younger than the interval.

    Age is the export's own ``generated_at``, the time the guard judges, so an
    export bound to an older source (a retained GCP report) is never treated as
    fresh. Coverage is what the guard requires: every provider with live
    instances in the guard's own last report (review of #3278: after one
    provider's API failed, a skip keyed on age alone kept paid admission blocked
    for ten minutes instead of about one). Anything but a regular, readable,
    well-formed export owned by this account, not group- or world-writable,
    covering every required and every live provider, generated within the
    interval, with a current guard report returns ``None``: refresh.
    """

    try:
        interval = min(float(min_refresh_interval_seconds), MAX_MIN_REFRESH_INTERVAL_SECONDS)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(interval) or interval <= 0 or not spend_guard_report:
        return None
    path = Path(billing_export_path).expanduser()
    raw = _own_file_bytes(path, MAX_EXPORT_BYTES)
    if raw is None:
        return None
    try:
        export = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if (
        not isinstance(export, dict)
        or export.get("schema_version") != BILLING_EXPORT_SCHEMA_VERSION
        or export.get("scope") != BILLING_SCOPE
        or export.get("currency") != "USD"
    ):
        return None
    totals = export.get("provider_totals_usd")
    if not isinstance(totals, dict) or not totals or not set(totals).issubset(PROVIDERS):
        return None
    for value in totals.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or value < 0:
            return None
    if not {str(provider).strip() for provider in required_providers}.issubset(totals):
        return None
    try:
        generated_at = _parse_time(export.get("generated_at"), field="billing_export_generated_at")
    except ProviderBillingReconciliationError:
        return None
    age = (now.astimezone(timezone.utc) - generated_at).total_seconds()
    if not 0 <= age < interval:
        return None
    live = _guard_live_providers(
        spend_guard_report, export_digest=_digest_bytes(raw), export_providers=set(totals),
        now=now, max_age_seconds=interval,
    )
    if live is None:
        return None
    return {
        "billing_export_path": str(path),
        "billing_export_generated_at": generated_at.isoformat(),
        "billing_export_age_seconds": round(age, 3),
        "billing_export_digest": _digest_bytes(raw),
        "min_refresh_interval_seconds": interval,
        "spend_guard_live_providers": live,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets-dir", required=True)
    parser.add_argument("--billing-export", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--cohort-start-at", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--aws-account-id", help="legacy option; automatic AWS billing is unavailable")
    parser.add_argument(
        "--aws-credentials-file", help="legacy option; no AWS credentials are read"
    )
    parser.add_argument("--aws-profile", help="legacy option; no AWS credentials are read")
    parser.add_argument("--gcp-report-receipt", default=os.getenv("BLUEPRINT_GCP_BILLING_REPORT_RECEIPT"))
    parser.add_argument("--gcp-billing-account-id", default=os.getenv("BLUEPRINT_GCP_BILLING_ACCOUNT_ID", ""))
    parser.add_argument(
        "--require-provider",
        action="append",
        default=[],
        choices=PROVIDERS,
        help=(
            "make this provider mandatory for the refresh; "
            "unspecified providers are optional; AWS is unavailable; RunPod is retired"
        ),
    )
    parser.add_argument(
        "--min-refresh-interval-seconds",
        type=float,
        default=0.0,
        help=(
            "exit without any provider request while this account's current export "
            f"is younger than this (capped at {MAX_MIN_REFRESH_INTERVAL_SECONDS} s); "
            "0, the default, refreshes on every run"
        ),
    )
    parser.add_argument(
        "--spend-guard-report",
        default=os.getenv("BLUEPRINT_GPU_SPEND_GUARD_REPORT"),
        help=(
            "the spend guard's --json-report; an export is retained only while that "
            "report is current and every provider it found live is covered"
        ),
    )
    args = parser.parse_args(argv)
    retained = retained_fresh_export(
        args.billing_export,
        now=datetime.now(timezone.utc),
        min_refresh_interval_seconds=args.min_refresh_interval_seconds,
        required_providers=args.require_provider,
        spend_guard_report=args.spend_guard_report,
    )
    if retained is not None:
        print(
            json.dumps(
                {
                    "schema_version": "blueprint.provider_billing_reconciliation_run.v1",
                    "status": "fresh_export_retained",
                    **retained,
                    "provider_requests_added": 0,
                    "provider_mutation_performed": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 0
    from blueprint_pipeline.cloud_vm_render_providers import GCPRenderProvider

    try:
        result = reconcile_provider_billing(
            secrets_dir=args.secrets_dir,
            billing_export_path=args.billing_export,
            audit_root=args.audit_root,
            start_at=args.cohort_start_at,
            timeout=args.timeout,
            aws_account_id=args.aws_account_id,
            aws_credentials_file=args.aws_credentials_file,
            aws_profile=args.aws_profile,
            required_providers=args.require_provider,
            gcp_report_receipt=args.gcp_report_receipt,
            gcp_billing_account_id=args.gcp_billing_account_id,
            gcp_project_ids=GCPRenderProvider().inventory_projects(),
        )
    except (OSError, ProviderBillingReconciliationError, MeteredCallEvidenceError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "blueprint.provider_billing_reconciliation_run.v1",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                    "raw_secret_values_recorded": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
