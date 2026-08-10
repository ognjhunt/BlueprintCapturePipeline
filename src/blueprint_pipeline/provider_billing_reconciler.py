"""Read-only provider billing reconciliation for the paid-spend admission lock.

The synchronizer reads the same file-backed provider credentials as the GPU
spend guard, fetches cumulative cohort charges from fixed official endpoints,
retains the exact private response bytes, and atomically writes the narrow
``blueprint.provider_billing_export.v1`` input consumed by the guard.  It never
creates, changes, or deletes provider resources.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


BILLING_EXPORT_SCHEMA_VERSION = "blueprint.provider_billing_export.v1"
BILLING_SOURCE_SCHEMA_VERSION = "blueprint.provider_billing_source_receipt.v1"
BILLING_SCOPE = "blueprint_beta_100_user_cohort"
PROVIDERS = ("runpod", "vast", "digitalocean")
RUNPOD_BILLING_URL = "https://rest.runpod.io/v1/billing"
VAST_CHARGES_URL = "https://console.vast.ai/api/v0/charges/"
DIGITALOCEAN_API_URL = "https://api.digitalocean.com/v2"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_PAGES = 100


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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
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
    if not url.startswith((RUNPOD_BILLING_URL, VAST_CHARGES_URL, DIGITALOCEAN_API_URL)):
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


def _runpod_total(
    *,
    token: str,
    start_at: datetime,
    end_at: datetime,
    timeout: float,
    transport: Transport,
    audit_rows: list[dict[str, Any]],
) -> float:
    total = 0.0
    query = {
        "startTime": start_at.isoformat().replace("+00:00", "Z"),
        "endTime": end_at.isoformat().replace("+00:00", "Z"),
        "bucketSize": "year",
    }
    for resource in ("pods", "endpoints", "networkvolumes"):
        url = f"{RUNPOD_BILLING_URL}/{resource}?{urllib.parse.urlencode(query)}"
        rows = _request_json(
            provider="runpod",
            url=url,
            token=token,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        )
        if not isinstance(rows, list):
            raise ProviderBillingReconciliationError(f"runpod_billing_response_invalid:{resource}")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ProviderBillingReconciliationError(
                    f"runpod_billing_row_invalid:{resource}:{index}"
                )
            total += _money(row.get("amount", 0), field=f"runpod_{resource}_amount")
            high_performance = row.get("highPerformanceStorageAmount")
            if high_performance is not None:
                total += _money(
                    high_performance,
                    field=f"runpod_{resource}_high_performance_storage_amount",
                )
    return total


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
    if generated_at > end_at or end_at - generated_at > timedelta(hours=24):
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
) -> dict[str, Any]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    start = _parse_time(start_at, field="cohort_start_at")
    if start >= current:
        raise ProviderBillingReconciliationError("cohort_start_at_not_before_end")
    secret_root = Path(secrets_dir).expanduser().resolve()
    keys = {
        "runpod": _read_secret(secret_root / "runpod_api_key"),
        "vast": _read_secret(secret_root / "vast_api_key"),
        "digitalocean": _read_secret(secret_root / "digitalocean_api_token"),
    }
    audit_rows: list[dict[str, Any]] = []
    totals = {
        "runpod": _runpod_total(
            token=keys["runpod"],
            start_at=start,
            end_at=current,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        ),
        "vast": _vast_total(
            token=keys["vast"],
            start_at=start,
            end_at=current,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        ),
        "digitalocean": _digitalocean_total(
            token=keys["digitalocean"],
            start_at=start,
            end_at=current,
            timeout=timeout,
            transport=transport,
            audit_rows=audit_rows,
        ),
    }
    totals = {provider: round(value, 6) for provider, value in totals.items()}
    generated_at = current.isoformat()
    audit_directory = Path(audit_root).expanduser().resolve() / current.strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    source_rows: list[dict[str, Any]] = []
    for index, row in enumerate(audit_rows, start=1):
        response_path = audit_directory / f"response-{index:03d}-{row['provider']}.json"
        _atomic_write(response_path, row["response_bytes"])
        source_rows.append(
            {key: value for key, value in row.items() if key != "response_bytes"}
            | {"retained_path": str(response_path)}
        )
    source_receipt = {
        "schema_version": BILLING_SOURCE_SCHEMA_VERSION,
        "status": "reconciled",
        "generated_at": generated_at,
        "cohort_start_at": start.isoformat(),
        "cohort_end_at": generated_at,
        "provider_totals_usd": totals,
        "sources": source_rows,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }
    source_receipt["receipt_digest"] = _canonical_digest(
        source_receipt, digest_field="receipt_digest"
    )
    source_path = audit_directory / "provider_billing_source_receipt.json"
    _atomic_write(source_path, (_canonical_json(source_receipt) + "\n").encode("utf-8"))
    billing_export = {
        "schema_version": BILLING_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "currency": "USD",
        "scope": BILLING_SCOPE,
        "provider_totals_usd": totals,
    }
    export_path = Path(billing_export_path).expanduser().resolve()
    _atomic_write(export_path, (_canonical_json(billing_export) + "\n").encode("utf-8"))
    return {
        "schema_version": "blueprint.provider_billing_reconciliation_run.v1",
        "status": "reconciled",
        "generated_at": generated_at,
        "provider_totals_usd": totals,
        "billing_export_path": str(export_path),
        "billing_export_digest": _digest_bytes(export_path.read_bytes()),
        "source_receipt_path": str(source_path),
        "source_receipt_digest": source_receipt["receipt_digest"],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets-dir", required=True)
    parser.add_argument("--billing-export", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--cohort-start-at", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        result = reconcile_provider_billing(
            secrets_dir=args.secrets_dir,
            billing_export_path=args.billing_export,
            audit_root=args.audit_root,
            start_at=args.cohort_start_at,
            timeout=args.timeout,
        )
    except (OSError, ProviderBillingReconciliationError) as exc:
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
