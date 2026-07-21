#!/usr/bin/env python3
"""GPU spend guard — report configured GPU-provider spend and reap orphans.

A standalone cost watchdog. It reads file-based credentials from
``~/.blueprint-secrets`` (``runpod_api_key``, ``vast_api_key``, and
``digitalocean_api_token``), lists every live
pod/instance with its id, name, age, runtime/boot state and ``$/hr`` (plus a total
burn estimate), and — only with ``--reap`` — terminates pods that are clearly
orphaned: allocated but never booted (``runtime`` absent) past
``--max-boot-seconds`` (default 480s), the classic "stuck dud that keeps billing",
or booted allocations with no live owner past ``--max-booted-orphan-seconds``.
Configured inventory is fail-closed: missing credentials, API errors, and
unverified deletion are blockers rather than green empty fleets.

Safety rails (the whole point of the tool is to never kill live work):

* Default is **dry-run** — it reports and would-reap, but changes nothing.
* A pod whose id appears in a registered render-job owner file, or in a validated
  persistent-qualification owner file, **that still has a live owning process**
  is never reaped, no matter how stuck it looks. Only orphans whose launching run
  has died are eligible.
* Healthy booted pods with a live owning process are never auto-reaped.
* Booted pods without a live owner are only eligible after a separate hard-age
  TTL, so a crashed launcher cannot leave a billing allocation running forever.

Conventions mirror :mod:`blueprint_pipeline.gpu_render_providers`: file-based
secrets under ``~/.blueprint-secrets`` that are never logged, RunPod REST pods at
``https://rest.runpod.io/v1`` and Vast instances at
``https://console.vast.ai/api/v0`` (Bearer auth). This module deliberately does
not import the Isaac render/parity job code — it only *reads* the pod-id files
those jobs write.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from blueprint_pipeline.common import write_json
from blueprint_pipeline.spend_admission_lock import build_spend_admission_lock

SCHEMA_VERSION = "gpu_spend_guard.v1"
SPEND_LEDGER_SCHEMA_VERSION = "gpu_spend_ledger.v1"
BILLING_EXPORT_SCHEMA_VERSION = "blueprint.provider_billing_export.v1"
BILLING_EXPORT_SCOPE = "blueprint_beta_100_user_cohort"
SECRETS_DIR = Path.home() / ".blueprint-secrets"
RUNPOD_API = "https://rest.runpod.io/v1"
VAST_API = "https://console.vast.ai/api/v0"
DEFAULT_MAX_BOOT_SECONDS = 480
DEFAULT_MAX_BOOTED_ORPHAN_SECONDS = 4 * 60 * 60
DEFAULT_WARM_LEASE_SECONDS = 15 * 60
PROVIDERS = ("runpod", "vast", "digitalocean", "gcp", "aws")
BILLING_BASE_PROVIDERS = ("runpod", "vast", "digitalocean")
MAX_BILLING_EXPORT_BYTES = 1024 * 1024
ALLOWED_PROVIDER_API_HOSTS = frozenset(
    {"rest.runpod.io", "console.vast.ai", "api.digitalocean.com"}
)

# Vast statuses that mean the instance is no longer billing compute.
VAST_TERMINAL_STATUSES = frozenset(
    {"stopped", "exited", "failed", "destroyed", "deleted", "inactive", "completed"}
)
# RunPod desired-states that mean the pod is no longer a live compute allocation.
RUNPOD_TERMINAL_STATUSES = frozenset({"EXITED", "TERMINATED", "TERMINATING"})
RUNPOD_STOPPED_STATUSES = frozenset({"STOPPED", "PAUSED"})
# Historical warm-candidate ids remain visible for diagnostics only. Reap protection
# now comes from live warm_serve_pod.json markers, not this static list.
DEFAULT_WARM_CANDIDATE_IDS = frozenset(
    {
        "pwbu7wxsvxpr0x",
        "9zxerj0nm3ow76",
        "qzgtsh4t27hi7f",
        "v4bd9u2qhwivb8",
        "y3n5n7t6wvaawe",
        "1gx9uri0mkrxg9",
        "ajxbj2ysyow3n9",
        "usjvua1bwlwhyj",
    }
)


def _now() -> float:
    """Wall-clock epoch seconds. Indirected so tests can pin the clock."""
    return time.time()


# ----------------------------- file-based secrets -----------------------------


def _read_secret(name: str, *, secrets_dir: Path = SECRETS_DIR) -> str | None:
    """Return the stripped contents of ``<secrets_dir>/<name>`` or ``None``.

    Values are never logged or echoed — only their presence drives behaviour.
    """
    path = Path(secrets_dir) / name
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _redact(text: Any, *secrets: str | None) -> Any:
    """Replace any secret substring with ``<redacted>`` before it can be printed."""
    if not isinstance(text, str):
        return text
    out = text
    for secret in secrets:
        if secret:
            out = out.replace(secret, "<redacted>")
    return out


# ----------------------------- value coercion -----------------------------


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _iso_to_epoch(value: Any) -> float | None:
    """Parse an ISO-8601 timestamp (``...Z`` accepted) into epoch seconds (UTC)."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


# ----------------------------- normalized instance -----------------------------


@dataclass
class GpuInstance:
    """A provider-neutral view of one GPU pod/instance."""

    provider: str
    id: str
    name: str
    state: str
    booted: bool
    live: bool
    cost_per_hr: float = 0.0
    age_seconds: float | None = None


def _parse_runpod_pod(pod: Mapping[str, Any], *, now: float) -> GpuInstance:
    pod_id = str(pod.get("id") or "")
    name = str(pod.get("name") or "")
    desired = str(pod.get("desiredStatus") or "").upper()
    runtime = pod.get("runtime")
    booted = bool(runtime)
    live = booted or desired not in (RUNPOD_TERMINAL_STATUSES | RUNPOD_STOPPED_STATUSES)

    if booted:
        state = "running"
    elif desired == "RUNNING":
        state = "booting"
    elif desired in RUNPOD_STOPPED_STATUSES:
        state = "stopped"
    else:
        state = desired.lower() or "unknown"

    cost_per_hr = _coerce_float(pod.get("costPerHr"))
    if cost_per_hr is None:
        cost_per_hr = _coerce_float(pod.get("costPerHour")) or 0.0

    age: float | None = None
    if isinstance(runtime, Mapping):
        age = _coerce_float(runtime.get("uptimeInSeconds"))
    if age is None:
        started = _iso_to_epoch(
            pod.get("lastStartedAt") or pod.get("createdAt") or pod.get("lastStatusChange")
        )
        if started is not None:
            age = max(0.0, now - started)

    return GpuInstance(
        provider="runpod",
        id=pod_id,
        name=name,
        state=state,
        booted=booted,
        live=live,
        cost_per_hr=cost_per_hr,
        age_seconds=age,
    )


def _parse_vast_instance(inst: Mapping[str, Any], *, now: float) -> GpuInstance:
    iid = str(inst.get("id") or inst.get("instance_id") or inst.get("contract_id") or "")
    name = str(inst.get("label") or inst.get("name") or "")
    status = str(
        inst.get("actual_status")
        or inst.get("cur_state")
        or inst.get("status")
        or inst.get("intended_status")
        or ""
    ).lower()
    booted = status == "running"
    live = status not in VAST_TERMINAL_STATUSES

    cost_per_hr = _coerce_float(
        inst.get("dph_total")
        or inst.get("discounted_dph_total")
        or inst.get("min_bid")
        or inst.get("price_per_hour")
    ) or 0.0

    started = _coerce_float(inst.get("start_date"))
    age = max(0.0, now - started) if started is not None else None

    return GpuInstance(
        provider="vast",
        id=iid,
        name=name,
        state=status or "unknown",
        booted=booted,
        live=live,
        cost_per_hr=cost_per_hr,
        age_seconds=age,
    )


DO_API = "https://api.digitalocean.com/v2"
DO_TERMINAL_STATUSES = {"archive"}


def _parse_do_droplet(droplet: Mapping[str, Any], *, now: float) -> GpuInstance:
    """GPU droplets bill until DESTROYED — powered-off ("off") is still live spend."""
    raw_size = droplet.get("size")
    size = dict(raw_size) if isinstance(raw_size, Mapping) else {}
    status = str(droplet.get("status") or "").lower()
    started = _iso_to_epoch(droplet.get("created_at"))
    return GpuInstance(
        provider="digitalocean",
        id=str(droplet.get("id") or ""),
        name=str(droplet.get("name") or ""),
        state=status or "unknown",
        booted=status == "active",
        live=status not in DO_TERMINAL_STATUSES,
        cost_per_hr=_coerce_float(size.get("price_hourly")) or 0.0,
        age_seconds=max(0.0, now - started) if started is not None else None,
    )


def _parse_cloud_vm(resource: Mapping[str, Any], *, provider: str, now: float) -> GpuInstance:
    """Normalize a GCP/AWS resource returned by the first-class render adapter."""
    status = str(resource.get("status") or "").lower()
    terminal = {"terminated", "shutting-down"}
    booted = status == "running"
    started = _iso_to_epoch(resource.get("created_at"))
    return GpuInstance(
        provider=provider,
        id=str(resource.get("instance_id") or ""),
        name=str(resource.get("name") or ""),
        state=status or "unknown",
        booted=booted,
        # Stopped VMs retain billable disks and remain inventory until deleted.
        live=status not in terminal,
        cost_per_hr=_coerce_float(resource.get("cost_per_hour")) or 0.0,
        age_seconds=max(0.0, now - started) if started is not None else None,
    )


def collect_instances(
    *,
    now: float,
    runpod_pods: Sequence[Mapping[str, Any]] | None = None,
    vast_instances: Sequence[Mapping[str, Any]] | None = None,
    do_droplets: Sequence[Mapping[str, Any]] | None = None,
    gcp_instances: Sequence[Mapping[str, Any]] | None = None,
    aws_instances: Sequence[Mapping[str, Any]] | None = None,
) -> list[GpuInstance]:
    """Parse raw provider JSON rows into :class:`GpuInstance` records (no network)."""
    instances: list[GpuInstance] = []
    for pod in runpod_pods or []:
        if isinstance(pod, Mapping):
            instances.append(_parse_runpod_pod(pod, now=now))
    for inst in vast_instances or []:
        if isinstance(inst, Mapping):
            instances.append(_parse_vast_instance(inst, now=now))
    for droplet in do_droplets or []:
        if isinstance(droplet, Mapping):
            instances.append(_parse_do_droplet(droplet, now=now))
    for resource in gcp_instances or []:
        if isinstance(resource, Mapping):
            instances.append(_parse_cloud_vm(resource, provider="gcp", now=now))
    for resource in aws_instances or []:
        if isinstance(resource, Mapping):
            instances.append(_parse_cloud_vm(resource, provider="aws", now=now))
    return instances


def total_burn_per_hour(instances: Iterable[GpuInstance]) -> float:
    return sum(i.cost_per_hr for i in instances if i.live)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _utc_day_bounds(now: float) -> tuple[str, float]:
    current = datetime.fromtimestamp(now, timezone.utc)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return current.date().isoformat(), start.timestamp()


def update_spend_ledger(
    instances: Sequence[GpuInstance],
    *,
    ledger_path: Path,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist cumulative fleet spend estimates for daily/total budget gates.

    First observation of a live allocation conservatively counts the instance's
    known age. Later guard runs add only the elapsed time since last observation,
    so a scheduled timer can enforce aggregate spend without provider-specific
    billing exports.
    """
    observed_at = _now() if now is None else float(now)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    lock_file = lock_path.open("a+b")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    previous = _load_json_mapping(ledger_path) if ledger_path.is_file() else {}
    if ledger_path.is_file() and not previous:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        return {
            "schema_version": SPEND_LEDGER_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": ["spend_ledger_existing_state_invalid"],
            "ledger_path": str(ledger_path),
            "prior_state_preserved": True,
        }
    raw_previous_instances = previous.get("instances")
    previous_instances = (
        dict(raw_previous_instances)
        if isinstance(raw_previous_instances, Mapping)
        else {}
    )
    day, day_start = _utc_day_bounds(observed_at)
    previous_day = str(previous.get("daily_budget_day") or "")
    previous_daily = (
        _coerce_float(previous.get("daily_spend_usd")) if previous_day == day else 0.0
    ) or 0.0
    previous_total = _coerce_float(previous.get("total_spend_usd")) or 0.0
    daily_increment = 0.0
    total_increment = 0.0
    active_instances: dict[str, dict[str, Any]] = {}
    for inst in instances:
        if not inst.live:
            continue
        key = f"{inst.provider}:{inst.id}"
        raw_prior = previous_instances.get(key)
        prior = dict(raw_prior) if isinstance(raw_prior, Mapping) else {}
        last_seen = _coerce_float(prior.get("last_seen_epoch"))
        inferred_start = (
            max(0.0, observed_at - float(inst.age_seconds))
            if inst.age_seconds is not None
            else observed_at
        )
        total_start = last_seen if last_seen is not None else inferred_start
        daily_start = max(total_start, day_start)
        total_seconds = max(0.0, observed_at - total_start)
        daily_seconds = max(0.0, observed_at - daily_start)
        total_increment += inst.cost_per_hr * total_seconds / 3600.0
        daily_increment += inst.cost_per_hr * daily_seconds / 3600.0
        active_instances[key] = {
            "provider": inst.provider,
            "id": inst.id,
            "name": inst.name,
            "last_seen_epoch": observed_at,
            "last_seen_at": datetime.fromtimestamp(
                observed_at, timezone.utc
            ).isoformat(),
            "cost_per_hr_usd": inst.cost_per_hr,
        }
    ledger = {
        "schema_version": SPEND_LEDGER_SCHEMA_VERSION,
        "revision": int(previous.get("revision") or 0) + 1,
        "status": "updated",
        "generated_at": datetime.fromtimestamp(observed_at, timezone.utc).isoformat(),
        "daily_budget_day": day,
        "daily_spend_usd": round(previous_daily + daily_increment, 4),
        "total_spend_usd": round(previous_total + total_increment, 4),
        "daily_increment_usd": round(daily_increment, 4),
        "total_increment_usd": round(total_increment, 4),
        "active_instance_count": len(active_instances),
        "instances": active_instances,
        "claim_boundary": (
            "Spend ledger is a conservative allocation-cost estimate from guard "
            "polling. It is a budget gate, not provider invoice reconciliation."
        ),
    }
    try:
        write_json(ledger_path, ledger)
    except OSError as exc:
        ledger["status"] = "blocked"
        ledger["blockers"] = [f"spend_ledger_write_failed:{type(exc).__name__}"]
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    return ledger


def build_fleet_budget_guard(
    instances: Sequence[GpuInstance],
    *,
    max_live_instances: int | None = None,
    max_burn_usd_per_hour: float | None = None,
    spend_ledger: Mapping[str, Any] | None = None,
    max_daily_spend_usd: float | None = None,
    max_total_spend_usd: float | None = None,
) -> dict[str, Any]:
    live_count = sum(1 for inst in instances if inst.live)
    burn = total_burn_per_hour(instances)
    blockers: list[str] = []
    if max_live_instances is not None and live_count > max_live_instances:
        blockers.append("fleet_live_gpu_instance_limit_exceeded")
    if max_burn_usd_per_hour is not None and burn > max_burn_usd_per_hour:
        blockers.append("fleet_burn_rate_limit_exceeded")
    ledger = dict(spend_ledger) if isinstance(spend_ledger, Mapping) else {}
    if max_daily_spend_usd is not None or max_total_spend_usd is not None:
        if not ledger:
            blockers.append("fleet_cumulative_spend_ledger_missing")
        elif ledger.get("status") != "updated":
            blockers.append("fleet_cumulative_spend_ledger_not_updated")
            blockers.extend(
                f"spend_ledger:{blocker}"
                for blocker in ledger.get("blockers") or []
                if isinstance(blocker, str)
            )
    daily_spend = _coerce_float(ledger.get("daily_spend_usd")) if ledger else None
    total_spend = _coerce_float(ledger.get("total_spend_usd")) if ledger else None
    if (
        max_daily_spend_usd is not None
        and daily_spend is not None
        and daily_spend > max_daily_spend_usd
    ):
        blockers.append("fleet_daily_spend_limit_exceeded")
    if (
        max_total_spend_usd is not None
        and total_spend is not None
        and total_spend > max_total_spend_usd
    ):
        blockers.append("fleet_total_spend_limit_exceeded")
    return {
        "schema_version": "gpu_fleet_budget_guard.v1",
        "status": "passed" if not blockers else "blocked",
        "live_instance_count": live_count,
        "total_burn_per_hour_usd": round(burn, 4),
        "max_live_instances": max_live_instances,
        "max_burn_usd_per_hour": max_burn_usd_per_hour,
        "daily_spend_usd": daily_spend,
        "total_spend_usd": total_spend,
        "max_daily_spend_usd": max_daily_spend_usd,
        "max_total_spend_usd": max_total_spend_usd,
        "blockers": blockers,
        "claim_boundary": (
            "Fleet budget status is a cost/allocation gate only. It is not provider "
            "runtime proof, task success, or artifact quality evidence."
        ),
    }


# ----------------------------- HTTP -----------------------------


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _validated_provider_api_url(value: str) -> str:
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider API URL is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ALLOWED_PROVIDER_API_HOSTS
        or port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("provider API URL is outside the pinned HTTPS origins")
    return url


def _http_request(
    method: str,
    url: str,
    *,
    key: str | None = None,
    body: Mapping[str, Any] | None = None,
    timeout: int = 30,
) -> tuple[int, Any]:
    """Issue one Bearer-authenticated JSON request. Returns ``(status, payload)``.

    Network/HTTP errors are caught and returned as ``(code, {"error": ...})`` with
    the API key redacted from any error text — never raised, never logged raw.
    """
    data = json.dumps(dict(body)).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        validated_url = _validated_provider_api_url(url)
    except ValueError as exc:
        return 0, {"error": str(exc)}
    request = urllib.request.Request(
        validated_url,
        data=data,
        method=method,
        headers=headers,
    )
    try:
        opener = urllib.request.build_opener(_RejectRedirects)
        with opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            return status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            detail = ""
        return exc.code, {"error": _redact(detail, key)}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": _redact(repr(exc)[:300], key)}


def _rows_from_payload(payload: Any, *list_keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, Mapping):
        for key in list_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            if isinstance(value, Mapping):
                return [row for row in value.values() if isinstance(row, dict)]
    return []


class ProviderInventoryError(RuntimeError):
    def __init__(self, provider: str, status: int) -> None:
        super().__init__(f"{provider}_inventory_query_failed:http_{status}")
        self.provider = provider
        self.status = status


def fetch_runpod_pods(key: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    status, payload = _http_request("GET", f"{RUNPOD_API}/pods", key=key, timeout=timeout)
    if status not in (200, 201):
        _warn(f"runpod pod query failed (http={status})")
        raise ProviderInventoryError("runpod", status)
    return _rows_from_payload(payload, "pods", "data", "items")


def fetch_vast_instances(key: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    status, payload = _http_request("GET", f"{VAST_API}/instances/", key=key, timeout=timeout)
    if status not in (200, 201):
        _warn(f"vast instance query failed (http={status})")
        raise ProviderInventoryError("vast", status)
    return _rows_from_payload(payload, "instances", "results", "data")


def fetch_do_droplets(token: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    """Only GPU droplets (size slug gpu-*) count toward render-lane spend."""
    status, payload = _http_request(
        "GET", f"{DO_API}/droplets?per_page=200", key=token, timeout=timeout
    )
    if status not in (200, 201):
        _warn(f"digitalocean droplet query failed (http={status})")
        raise ProviderInventoryError("digitalocean", status)
    rows = _rows_from_payload(payload, "droplets", "data")
    return [
        r for r in rows
        if str((r.get("size") or {}).get("slug") or r.get("size_slug") or "").startswith("gpu-")
    ]


def cloud_provider_configured(provider: str) -> bool:
    """Return whether the explicit account/location contract is configured."""
    try:
        from blueprint_pipeline.gpu_render_providers import get_render_provider

        availability = get_render_provider(provider).available()
    except Exception:  # noqa: BLE001 - configuration uncertainty is false
        return False
    return availability.get("available") is True


def fetch_cloud_vm_instances(provider: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    """Use the provider adapter's authenticated, scoped inventory API."""
    del timeout  # provider adapters own their bounded per-call timeouts
    from blueprint_pipeline.gpu_render_providers import get_render_provider

    inventory = get_render_provider(provider).billable_inventory(name_prefix="blueprint")
    if inventory.get("api_confirmed") is not True:
        raise ProviderInventoryError(provider, int(inventory.get("http") or 0))
    rows = inventory.get("resources")
    if not isinstance(rows, list):
        raise ProviderInventoryError(provider, 0)
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _inventory_query(
    *,
    provider: str,
    credential: str | None,
    fetch: Any,
    timeout: int,
    required: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not credential:
        return [], {
            "provider": provider,
            "status": "blocked_missing_credential" if required else "not_configured",
            "required": required,
            "credential_present": False,
            "row_count": 0,
            "blockers": [f"{provider}_inventory_credential_missing"] if required else [],
        }
    try:
        rows = fetch(credential, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - inventory uncertainty must fail closed
        return [], {
            "provider": provider,
            "status": "failed",
            "required": required,
            "credential_present": True,
            "row_count": 0,
            "error_type": type(exc).__name__,
            "blockers": [f"{provider}_inventory_query_failed"],
        }
    return rows, {
        "provider": provider,
        "status": "succeeded",
        "required": required,
        "credential_present": True,
        "row_count": len(rows),
        "blockers": [],
    }


def reconcile_billing_export(
    *,
    billing_export_path: Path | None,
    instances: Sequence[GpuInstance],
    now: float,
    required: bool,
    max_age_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    if billing_export_path is None:
        return {
            "status": "blocked" if required else "not_configured",
            "required": required,
            "blockers": ["provider_billing_export_missing"] if required else [],
        }
    blockers: list[str] = []
    billing_path_valid = False
    source_mode: int | None = None
    billing_export_digest: str | None = None
    if billing_export_path.is_symlink():
        blockers.append("provider_billing_export_symlink")
    else:
        try:
            metadata = billing_export_path.stat()
        except FileNotFoundError:
            blockers.append("provider_billing_export_missing")
        except OSError:
            blockers.append("provider_billing_export_unreadable")
        else:
            source_mode = stat.S_IMODE(metadata.st_mode)
            if not stat.S_ISREG(metadata.st_mode):
                blockers.append("provider_billing_export_not_regular_file")
            elif metadata.st_size > MAX_BILLING_EXPORT_BYTES:
                blockers.append("provider_billing_export_too_large")
            else:
                billing_path_valid = True
                if source_mode & (stat.S_IWGRP | stat.S_IWOTH):
                    blockers.append(
                        "provider_billing_export_writable_by_group_or_world"
                    )
                if metadata.st_uid not in {0, os.geteuid()}:
                    blockers.append("provider_billing_export_owner_untrusted")

    payload = _load_json_mapping(billing_export_path) if billing_path_valid else {}
    if billing_path_valid:
        try:
            billing_export_digest = _sha256_file(billing_export_path)
        except OSError:
            blockers.append("provider_billing_export_unreadable")
    generated = _iso_to_epoch(payload.get("generated_at"))
    totals = payload.get("provider_totals_usd")
    live_providers = {instance.provider for instance in instances if instance.live}
    if not payload:
        blockers.append("provider_billing_export_invalid")
    if payload.get("schema_version") != BILLING_EXPORT_SCHEMA_VERSION:
        blockers.append("provider_billing_export_schema_invalid")
    if payload.get("currency") != "USD":
        blockers.append("provider_billing_export_currency_invalid")
    if payload.get("scope") != BILLING_EXPORT_SCOPE:
        blockers.append("provider_billing_export_scope_invalid")
    if generated is None or now - generated > max_age_seconds or generated > now + 300:
        blockers.append("provider_billing_export_stale_or_invalid_time")
    if not isinstance(totals, Mapping):
        blockers.append("provider_billing_export_totals_missing")
        totals = {}
    # Backward-compatible exports always cover the original fleet. Newly added
    # providers become mandatory as soon as they have live inventory; an
    # unconfigured provider does not make every historical export invalid.
    required_providers = (
        set(BILLING_BASE_PROVIDERS) | live_providers if required else live_providers
    )
    missing = sorted(provider for provider in required_providers if provider not in totals)
    if missing:
        blockers.extend(f"provider_billing_export_missing:{provider}" for provider in missing)
    if required:
        unexpected = sorted(provider for provider in totals if provider not in PROVIDERS)
        if unexpected:
            blockers.extend(
                f"provider_billing_export_unexpected:{provider}"
                for provider in unexpected
            )
    for provider, value in totals.items():
        amount = _coerce_float(value)
        if amount is None or amount < 0:
            blockers.append(f"provider_billing_export_invalid_total:{provider}")
    return {
        "status": "reconciled" if not blockers else "blocked",
        "required": required,
        "billing_export_artifact_name": billing_export_path.name,
        "billing_export_sha256": billing_export_digest,
        "billing_export_mode_octal": f"{source_mode:04o}"
        if source_mode is not None
        else None,
        "currency": payload.get("currency"),
        "scope": payload.get("scope"),
        "billing_export_schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "provider_totals_usd": dict(totals),
        "live_providers": sorted(live_providers),
        "blockers": blockers,
        "claim_boundary": "Billing export reconciliation is spend evidence, not task success.",
    }


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


# ----------------------------- ownership / live owner -----------------------------


# The render job writes the launched provider id into the run's
# object_store_real_run dir: started_pod_id.txt (RunPod) / started_vast_instance_id.txt
# (Vast). Both confer ownership — protect either against reaping.
OWNER_ID_FILENAMES = (
    "started_pod_id.txt",
    "started_vast_instance_id.txt",
    "started_do_droplet_id.txt",
    "started_gcp_instance_name.txt",
    "started_aws_instance_id.txt",
)
MAX_OWNER_ID_BYTES = 1024
MAX_QUALIFICATION_MANIFEST_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class _OwnerBinding:
    id_path: Path
    launched_id: str
    qualification_manifest_path: Path | None = None
    qualification_manifest_alias_path: Path | None = None
    qualification_resource_name_prefix: str | None = None
    qualification_watchdog_deadline_epoch: str | None = None


def _read_bounded_regular_text(
    path: Path,
    *,
    max_bytes: int,
    required_mode: int | None = None,
) -> str | None:
    """Read a small regular file without following a symlink or device node."""

    try:
        before = path.lstat()
    except OSError:
        return None
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > max_bytes
        or (
            required_mode is not None
            and stat.S_IMODE(before.st_mode) != required_mode
        )
    ):
        return None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size <= 0
            or opened.st_size > max_bytes
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (
                required_mode is not None
                and stat.S_IMODE(opened.st_mode) != required_mode
            )
        ):
            return None
        payload = os.read(descriptor, max_bytes + 1)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    if not payload or len(payload) > max_bytes:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeError:
        return None


def _validated_qualification_manifest(
    id_path: Path, launched_id: str
) -> tuple[Path, Path, str, str] | None:
    """Return an adjacent, fully bound live Vast qualification manifest."""

    # Import lazily so ordinary render-owner scans do not load the qualification
    # runtime and its provider adapters.
    try:
        from blueprint_pipeline.single_g1_kitchen_qualification_session import (
            _validate_manifest_binding,
        )
    except Exception:  # noqa: BLE001 - an optional owner must not disable the guard
        return None

    for manifest_path in sorted(id_path.parent.glob("*.json")):
        encoded = _read_bounded_regular_text(
            manifest_path,
            max_bytes=MAX_QUALIFICATION_MANIFEST_BYTES,
            required_mode=0o600,
        )
        if encoded is None:
            continue
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        manifest = dict(value)
        discovered_source = manifest_path.expanduser().absolute()
        try:
            source = manifest_path.parent.resolve(strict=True) / manifest_path.name
        except OSError:
            source = manifest_path.expanduser().absolute()
        try:
            _validate_manifest_binding(source, manifest)
        except Exception:  # noqa: BLE001 - invalid/unavailable binding fails closed
            continue
        try:
            watchdog_deadline_epoch = float(manifest.get("watchdog_deadline_epoch"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not 0 < watchdog_deadline_epoch < 1e20:
            continue
        if (
            manifest.get("release_binding_status") == "bound"
            and str(manifest.get("instance_id") or "") == launched_id
            and manifest.get("continuing_spend") is True
        ):
            return (
                source,
                discovered_source,
                str(manifest.get("resource_name_prefix") or ""),
                str(watchdog_deadline_epoch),
            )
    return None


def _iter_owner_bindings(
    output_roots: Iterable[Path | str], filename: str
) -> list[_OwnerBinding]:
    """Return validated provider-owner bindings under configured output roots."""

    found: list[_OwnerBinding] = []
    seen: set[tuple[str, str, str]] = set()
    for root in output_roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        candidates = list(base.glob(f"**/object_store_real_run/{filename}"))
        if filename == "started_vast_instance_id.txt":
            candidates.extend(base.glob(f"**/{filename}"))
        for path in candidates:
            payload = _read_bounded_regular_text(path, max_bytes=MAX_OWNER_ID_BYTES)
            launched_id = str(payload or "").strip()
            if not launched_id:
                continue
            render_owner = path.parent.name == "object_store_real_run" and (
                "pipeline" in path.parts
            )
            qualification_manifest_path = None
            qualification_manifest_alias_path = None
            qualification_resource_name_prefix = None
            qualification_watchdog_deadline_epoch = None
            if filename == "started_vast_instance_id.txt" and not render_owner:
                qualification_manifest = _validated_qualification_manifest(
                    path, launched_id
                )
                if qualification_manifest is not None:
                    (
                        qualification_manifest_path,
                        qualification_manifest_alias_path,
                        qualification_resource_name_prefix,
                        qualification_watchdog_deadline_epoch,
                    ) = qualification_manifest
            if not render_owner and qualification_manifest_path is None:
                continue
            dedupe_key = (
                str(path),
                launched_id,
                str(qualification_manifest_path or ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            found.append(
                _OwnerBinding(
                    id_path=path,
                    launched_id=launched_id,
                    qualification_manifest_path=qualification_manifest_path,
                    qualification_manifest_alias_path=qualification_manifest_alias_path,
                    qualification_resource_name_prefix=(
                        qualification_resource_name_prefix
                    ),
                    qualification_watchdog_deadline_epoch=(
                        qualification_watchdog_deadline_epoch
                    ),
                )
            )
    return found


def _iter_owner_id_files(
    output_roots: Iterable[Path | str], filename: str
) -> list[tuple[Path, str]]:
    """Return validated provider-owner id files under configured output roots."""
    return [
        (binding.id_path, binding.launched_id)
        for binding in _iter_owner_bindings(output_roots, filename)
    ]


def iter_started_pod_id_files(output_roots: Iterable[Path | str]) -> list[tuple[Path, str]]:
    """RunPod ``(file_path, pod_id)`` pairs (see :func:`_iter_owner_id_files`)."""
    return _iter_owner_id_files(output_roots, "started_pod_id.txt")


WARM_SERVE_MARKER_FILENAME = "warm_serve_pod.json"


def find_expected_serve_pod_ids(
    output_roots: Iterable[Path | str],
    *,
    now: float | None = None,
    max_lease_seconds: int = DEFAULT_WARM_LEASE_SECONDS,
) -> set[str]:
    """Return only warm workers whose owner lease is still fresh."""

    observed_at = _now() if now is None else float(now)
    expected: set[str] = set()
    for root in output_roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for path in base.glob(f"**/{WARM_SERVE_MARKER_FILENAME}"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            pod_id = str(payload.get("pod_id") or "").strip()
            expires_at = _iso_to_epoch(payload.get("lease_expires_at"))
            heartbeat_at = _iso_to_epoch(
                payload.get("heartbeat_at") or payload.get("generated_at")
            )
            lease_fresh = bool(
                (expires_at is not None and expires_at > observed_at)
                or (
                    heartbeat_at is not None
                    and 0 <= observed_at - heartbeat_at <= max(1, max_lease_seconds)
                )
            )
            if pod_id and payload.get("status") == "serving" and lease_fresh:
                expected.add(pod_id)
    return expected


def list_process_cmdlines() -> list[str]:
    """Return the command line of every running process (best-effort, injectable)."""
    try:
        completed = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def find_protected_pod_ids(
    output_roots: Iterable[Path | str],
    *,
    process_cmdlines: Sequence[str],
) -> set[str]:
    """Launched ids (RunPod pod / Vast instance) that must never be reaped because a
    *live* owning process exists.

    An owner-id file (``started_pod_id.txt`` / ``started_vast_instance_id.txt``) has
    a live render owner when some running process's command line references that run.
    A qualification owner additionally requires the exact validated manifest as the
    value of ``--qualification-session-manifest``; directory prefixes and instance-id
    substrings never establish qualification ownership. If the file exists but no
    process references it, the launching run has died and the instance is an orphan
    (eligible for reaping), not protected.
    """
    protected: set[str] = set()
    cmdlines = [c for c in process_cmdlines if isinstance(c, str)]
    owner_files: list[_OwnerBinding] = []
    for filename in OWNER_ID_FILENAMES:
        owner_files.extend(_iter_owner_bindings(output_roots, filename))
    qualification_targets = tuple(
        str(candidate)
        for binding in owner_files
        for candidate in (
            binding.qualification_manifest_path,
            binding.qualification_manifest_alias_path,
        )
        if candidate is not None
    )
    qualification_out_dirs = tuple(
        str(Path(candidate).parent)
        for candidate in qualification_targets
    )
    for binding in owner_files:
        path = binding.id_path
        launched_id = binding.launched_id
        qualification_manifest_path = binding.qualification_manifest_path
        if qualification_manifest_path is not None:
            allocator_owner = any(
                _cmd_option_references_exact_path(
                    cmd,
                    "--qualification-session-manifest",
                    str(qualification_manifest_path),
                    candidate_targets=qualification_targets,
                )
                for cmd in cmdlines
            )
            watchdog_owner = any(
                _cmd_option_references_exact_value(
                    cmd,
                    "-m",
                    "blueprint_pipeline.groot_oscar_runpod_watchdog",
                )
                and _cmd_option_references_exact_path(
                    cmd,
                    "--out-dir",
                    str(qualification_manifest_path.parent),
                    candidate_targets=qualification_out_dirs,
                    reject_final_symlink=False,
                )
                and _cmd_option_references_exact_value(cmd, "--provider", "vast")
                and _cmd_option_references_exact_value(
                    cmd,
                    "--pod-name-prefix",
                    str(binding.qualification_resource_name_prefix or ""),
                )
                and _cmd_option_references_exact_value(
                    cmd,
                    "--deadline-epoch",
                    str(binding.qualification_watchdog_deadline_epoch or ""),
                )
                for cmd in cmdlines
            )
            if allocator_owner or watchdog_owner:
                protected.add(launched_id)
            continue
        job_dir = str(path.parent)
        render_out_dir = (
            str(path.parent.parent)
            if path.parent.name == "object_store_real_run"
            else ""
        )
        for cmd in cmdlines:
            if (
                _cmd_references_exact_token(cmd, launched_id)
                or _cmd_references_path(cmd, job_dir)
                or (
                    render_out_dir
                    and _cmd_references_path(cmd, render_out_dir)
                )
            ):
                protected.add(launched_id)
                break
    return protected


def _cmd_references_exact_token(cmd: str, target: str) -> bool:
    """Whether a command line contains ``target`` as a complete argument value."""

    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    for token in tokens:
        if token == target:
            return True
        if token.startswith("--") and "=" in token:
            _, value = token.split("=", 1)
            if value == target:
                return True
    return False


def _resolved_path_identity(value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve(strict=True)
    except OSError:
        return Path(os.path.abspath(path))


def _cmd_option_values(cmd: str, option: str) -> list[str]:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == option and index + 1 < len(tokens):
            values.append(tokens[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    return values


def _cmd_option_references_exact_value(cmd: str, option: str, target: str) -> bool:
    return bool(target) and target in _cmd_option_values(cmd, option)


def _cmd_option_references_exact_path(
    cmd: str,
    option: str,
    target: str,
    *,
    candidate_targets: Sequence[str],
    reject_final_symlink: bool = True,
) -> bool:
    """Whether ``option`` names the exact absolute or component-relative path."""

    values = _cmd_option_values(cmd, option)
    target_path = _resolved_path_identity(target)
    for value in values:
        candidate = Path(value).expanduser()
        # Directory symlink components are supported, but the manifest option
        # itself must not be a symlink because qualification evidence forbids it.
        if reject_final_symlink and candidate.is_symlink():
            continue
        if candidate.is_absolute():
            if _resolved_path_identity(candidate) == target_path:
                return True
            continue
        # ``ps`` exposes argv but not the launcher's cwd portably. Accept only
        # an unambiguous multi-component suffix; component equality prevents
        # attempt_047 from aliasing attempt_047_retry.
        normalized = Path(os.path.normpath(str(candidate)))
        parts = tuple(part for part in normalized.parts if part not in ("", "."))
        if ".." in parts:
            continue
        if len(parts) < 3:
            continue
        matching_targets = set()
        for item in candidate_targets:
            # Retain the lexical path used to discover a manifest (including a
            # configured symlink-root alias) for suffix matching, but collapse
            # matches to their resolved identity before deciding uniqueness.
            lexical = Path(os.path.abspath(Path(item).expanduser()))
            if lexical.parts[-len(parts) :] == parts:
                matching_targets.add(_resolved_path_identity(item))
        if matching_targets == {target_path}:
            return True
    return False


def _cmd_references_path(cmd: str, target: str) -> bool:
    """Whether a command line refers to ``target`` (an absolute run path).

    Matches the absolute path directly, and also a *relative* path token whose
    suffix the absolute ``target`` ends with — so a job launched with
    ``--out-dir output/site/...`` is still recognized as the owner of a run the
    guard discovered by its absolute path. Erring toward a match keeps owned work
    safe from reaping.
    """
    if target in cmd:
        return True
    for token in cmd.split():
        if "/" not in token:
            continue
        candidate = token.strip().rstrip("/")
        if len(candidate) >= 8 and (
            target.endswith(candidate) or target.endswith("/" + candidate)
        ):
            return True
    return False


# ----------------------------- reap decision + action -----------------------------


def is_reapable(
    inst: GpuInstance,
    *,
    max_boot_seconds: int,
    protected_ids: set[str],
    max_booted_orphan_seconds: int = DEFAULT_MAX_BOOTED_ORPHAN_SECONDS,
) -> bool:
    """True only for unprotected live allocations past their orphan TTL."""
    if inst.id in protected_ids:
        return False
    if not inst.live:
        return False
    if inst.age_seconds is None:
        return False
    if inst.booted:
        return inst.age_seconds > max_booted_orphan_seconds
    if inst.provider == "runpod" and inst.state != "booting":
        return False
    return inst.age_seconds > max_boot_seconds


def reap_candidate_reason(
    inst: GpuInstance,
    *,
    max_boot_seconds: int,
    protected_ids: set[str],
    max_booted_orphan_seconds: int = DEFAULT_MAX_BOOTED_ORPHAN_SECONDS,
) -> str | None:
    if not is_reapable(
        inst,
        max_boot_seconds=max_boot_seconds,
        protected_ids=protected_ids,
        max_booted_orphan_seconds=max_booted_orphan_seconds,
    ):
        return None
    if inst.booted:
        return "booted_orphan_past_hard_ttl"
    return "unbooted_dud_past_boot_ttl"


def terminate_instance(
    inst: GpuInstance,
    *,
    runpod_key: str | None,
    vast_key: str | None,
    do_token: str | None = None,
    verification_attempts: int = 3,
    verification_delay_seconds: float = 0.5,
) -> dict[str, Any]:
    """Permanently delete an orphaned instance (releases its disk too)."""
    if inst.provider == "runpod":
        if not runpod_key:
            return {"status": "blocked", "reason": "runpod_api_key_missing"}
        status, _ = _http_request("DELETE", f"{RUNPOD_API}/pods/{inst.id}", key=runpod_key)
        ok = status in (200, 201, 204)
        return {"status": "terminated" if ok else "terminate_failed", "http": status}
    if inst.provider == "vast":
        if not vast_key:
            return {"status": "blocked", "reason": "vast_api_key_missing"}
        status, _ = _http_request(
            "DELETE", f"{VAST_API}/instances/{inst.id}/", key=vast_key
        )
        ok = status in (200, 201, 204)
        return {"status": "terminated" if ok else "terminate_failed", "http": status}
    if inst.provider == "digitalocean":
        if not do_token:
            return {"status": "blocked", "reason": "digitalocean_api_token_missing"}
        status, _ = _http_request(
            "DELETE",
            f"{DO_API}/droplets/{inst.id}",
            key=do_token,
        )
        if status not in (200, 202, 204, 404):
            return {
                "status": "terminate_failed",
                "http": status,
                "absence_verified": False,
            }
        verification_http: int | None = None
        for attempt in range(max(1, verification_attempts)):
            verification_http, _ = _http_request(
                "GET",
                f"{DO_API}/droplets/{inst.id}",
                key=do_token,
            )
            if verification_http in (404, 410):
                return {
                    "status": "terminated",
                    "http": status,
                    "verification_http": verification_http,
                    "absence_verified": True,
                }
            if attempt + 1 < max(1, verification_attempts):
                time.sleep(max(0.0, verification_delay_seconds))
        return {
            "status": "terminate_unverified",
            "http": status,
            "verification_http": verification_http,
            "absence_verified": False,
            "reason": "digitalocean_droplet_absence_not_verified",
        }
    if inst.provider in {"gcp", "aws"}:
        try:
            from blueprint_pipeline.gpu_render_providers import get_render_provider

            provider = get_render_provider(inst.provider)
            result = provider.terminate(inst.id)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "terminate_failed",
                "absence_verified": False,
                "error_type": type(exc).__name__,
            }
        if result.get("status") != "terminated":
            return {
                "status": "terminate_failed",
                "absence_verified": False,
                "provider_result_status": result.get("status"),
            }
        for attempt in range(max(1, verification_attempts)):
            inventory = provider.billable_inventory(name_prefix="blueprint")
            resources = inventory.get("resources")
            if inventory.get("api_confirmed") is True and isinstance(resources, list):
                live_ids = {
                    str(row.get("instance_id") or "")
                    for row in resources
                    if isinstance(row, Mapping)
                }
                if inst.id not in live_ids:
                    return {
                        "status": "terminated",
                        "absence_verified": True,
                        "verification_attempts": attempt + 1,
                    }
            if attempt + 1 < max(1, verification_attempts):
                time.sleep(max(0.0, verification_delay_seconds))
        return {
            "status": "terminate_unverified",
            "absence_verified": False,
            "reason": f"{inst.provider}_instance_absence_not_verified",
        }
    return {"status": "blocked", "reason": "unknown_provider"}


# ----------------------------- reporting -----------------------------


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_report(
    instances: Sequence[GpuInstance],
    *,
    protected_ids: set[str],
    max_boot_seconds: int,
    max_booted_orphan_seconds: int = DEFAULT_MAX_BOOTED_ORPHAN_SECONDS,
    serve_pod_ids: set[str] | frozenset[str] = frozenset(),
    fleet_budget: Mapping[str, Any] | None = None,
) -> str:
    live = [i for i in instances if i.live]
    lines: list[str] = []
    lines.append(f"GPU spend guard — {len(live)} live instance(s)")
    if live:
        lines.append(
            f"{'PROVIDER':8} {'ID':16} {'NAME':22} {'AGE':>7} {'STATE':9} {'$/HR':>7}"
        )
        for inst in live:
            if inst.id in serve_pod_ids:
                owned = " [warm-serve worker (expected)]"
            elif inst.id in protected_ids:
                owned = " [owned]"
            else:
                owned = ""
            lines.append(
                f"{inst.provider:8} {inst.id:16.16} {inst.name:22.22} "
                f"{_fmt_age(inst.age_seconds):>7} {inst.state:9} "
                f"{inst.cost_per_hr:7.3f}{owned}"
            )
    burn = total_burn_per_hour(instances)
    lines.append(f"Total burn estimate: ${burn:.3f}/hr (${burn * 24:.2f}/day)")
    budget = dict(fleet_budget) if isinstance(fleet_budget, Mapping) else None
    if budget:
        lines.append(
            "Fleet budget guard: "
            f"{budget.get('status')} "
            f"(live={budget.get('live_instance_count')}/"
            f"{budget.get('max_live_instances')}, "
            f"burn=${budget.get('total_burn_per_hour_usd')}/hr/"
            f"{budget.get('max_burn_usd_per_hour')})"
        )

    candidates = [
        i
        for i in instances
        if is_reapable(
            i,
            max_boot_seconds=max_boot_seconds,
            protected_ids=protected_ids,
            max_booted_orphan_seconds=max_booted_orphan_seconds,
        )
    ]
    if candidates:
        lines.append(
            f"Orphan reap candidates ({len(candidates)}): "
            f"unbooted past {max_boot_seconds}s or booted past "
            f"{max_booted_orphan_seconds}s, no live owner"
        )
        for inst in candidates:
            reason = reap_candidate_reason(
                inst,
                max_boot_seconds=max_boot_seconds,
                protected_ids=protected_ids,
                max_booted_orphan_seconds=max_booted_orphan_seconds,
            )
            lines.append(
                f"  - {inst.provider} {inst.id} ({inst.name}) "
                f"age={_fmt_age(inst.age_seconds)} ${inst.cost_per_hr:.3f}/hr "
                f"reason={reason}"
            )
    else:
        lines.append("Orphan reap candidates: none")
    return "\n".join(lines)


def build_json_report(
    instances: Sequence[GpuInstance],
    *,
    protected_ids: set[str],
    max_boot_seconds: int,
    max_booted_orphan_seconds: int = DEFAULT_MAX_BOOTED_ORPHAN_SECONDS,
    fleet_budget: Mapping[str, Any] | None = None,
    spend_ledger: Mapping[str, Any] | None = None,
    reap_mode: bool = False,
    reap_results: Sequence[Mapping[str, Any]] = (),
    inventory_results: Sequence[Mapping[str, Any]] = (),
    billing_reconciliation: Mapping[str, Any] | None = None,
    spend_admission_lock: Mapping[str, Any] | None = None,
) -> dict:
    """Machine-readable spend snapshot so ops never re-derives state from stdout.

    Persisted with ``--json-report``. Records every live allocation, the burn
    estimate, which ids were protected/reapable, and — when reaping ran — the
    per-instance termination result as teardown evidence.
    """
    live = [i for i in instances if i.live]
    candidates = [
        i
        for i in instances
        if is_reapable(
            i,
            max_boot_seconds=max_boot_seconds,
            protected_ids=protected_ids,
            max_booted_orphan_seconds=max_booted_orphan_seconds,
        )
    ]
    candidate_ids = {i.id for i in candidates}
    inventory_blockers = [
        str(blocker)
        for result in inventory_results
        for blocker in result.get("blockers") or []
    ]
    billing_blockers = (
        [str(item) for item in billing_reconciliation.get("blockers") or []]
        if isinstance(billing_reconciliation, Mapping)
        else []
    )
    reap_blockers = [
        f"reap_failed:{row.get('provider')}:{row.get('id')}:{row.get('status')}"
        for row in reap_results
        if row.get("status") != "terminated"
    ]
    blockers = [*inventory_blockers, *billing_blockers, *reap_blockers]
    admission = (
        dict(spend_admission_lock)
        if isinstance(spend_admission_lock, Mapping)
        else None
    )
    if admission is not None and admission.get("admission_allowed") is not True:
        blockers.extend(
            f"spend_admission:{item}"
            for item in admission.get("blockers") or ["paid_work_admission_locked"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "inventory_results": [dict(result) for result in inventory_results],
        "billing_reconciliation": dict(billing_reconciliation)
        if isinstance(billing_reconciliation, Mapping)
        else None,
        "spend_admission_lock": admission,
        "live_instance_count": len(live),
        "total_burn_per_hour_usd": round(total_burn_per_hour(instances), 4),
        "max_boot_seconds": int(max_boot_seconds),
        "max_booted_orphan_seconds": int(max_booted_orphan_seconds),
        "fleet_budget": dict(fleet_budget)
        if isinstance(fleet_budget, Mapping)
        else build_fleet_budget_guard(instances),
        "spend_ledger": dict(spend_ledger)
        if isinstance(spend_ledger, Mapping)
        else None,
        "reap_mode": bool(reap_mode),
        "instances": [
            {
                "provider": inst.provider,
                "id": inst.id,
                "name": inst.name,
                "state": inst.state,
                "live": inst.live,
                "booted": inst.booted,
                "age_seconds": inst.age_seconds,
                "cost_per_hr_usd": inst.cost_per_hr,
                "protected": inst.id in protected_ids,
                "reap_candidate": inst.id in candidate_ids,
                "reap_candidate_reason": reap_candidate_reason(
                    inst,
                    max_boot_seconds=max_boot_seconds,
                    protected_ids=protected_ids,
                    max_booted_orphan_seconds=max_booted_orphan_seconds,
                ),
            }
            for inst in instances
        ],
        "reap_candidate_ids": sorted(candidate_ids),
        "reap_results": [dict(r) for r in reap_results],
        "claim_boundary": (
            "This snapshot is billing/allocation state at one moment. It is not run "
            "success, artifact quality, or task evidence; booted-but-stalled pods are "
            "reported live and are only auto-reaped after the booted-orphan hard TTL."
        ),
    }


def default_output_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    roots: list[Path] = []
    for candidate in (repo_root / "output", Path.cwd() / "output"):
        if candidate not in roots:
            roots.append(candidate)
    return roots


# ----------------------------- CLI -----------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report live RunPod/Vast/DigitalOcean GPU allocations with a burn estimate, and optionally "
            "reap orphaned not-booted duds (never owned, never healthy pods)."
        )
    )
    parser.add_argument(
        "--reap",
        action="store_true",
        help="terminate orphaned not-booted duds (default: dry-run report only)",
    )
    parser.add_argument(
        "--max-boot-seconds",
        type=int,
        default=DEFAULT_MAX_BOOT_SECONDS,
        help=(
            "a pod allocated but never booted (runtime absent) older than this is a "
            f"dud (default {DEFAULT_MAX_BOOT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-booted-orphan-seconds",
        type=int,
        default=DEFAULT_MAX_BOOTED_ORPHAN_SECONDS,
        help=(
            "a booted live allocation with no live owner older than this is an "
            "orphan eligible for --reap "
            f"(default {DEFAULT_MAX_BOOTED_ORPHAN_SECONDS})"
        ),
    )
    parser.add_argument(
        "--output-root",
        action="append",
        default=None,
        metavar="DIR",
        help=(
            "output tree to scan for object_store_real_run/started_pod_id.txt "
            "ownership (repeatable; defaults to ./output and the repo output dir)"
        ),
    )
    parser.add_argument("--timeout", type=int, default=30, help="per-request timeout seconds")
    parser.add_argument(
        "--require-provider",
        action="append",
        choices=PROVIDERS,
        default=[],
        help="Provider inventory that must be credentialed and query successfully; repeatable.",
    )
    parser.add_argument(
        "--warm-lease-seconds",
        type=int,
        default=DEFAULT_WARM_LEASE_SECONDS,
        help="Maximum age of a warm-worker heartbeat without an explicit lease expiry.",
    )
    parser.add_argument(
        "--billing-export",
        default=os.getenv("BLUEPRINT_GPU_BILLING_EXPORT"),
        help="Current provider billing export JSON for reconciliation.",
    )
    parser.add_argument(
        "--require-billing-reconciliation",
        action="store_true",
        help="Fail closed unless a current billing export covers every live provider.",
    )
    parser.add_argument(
        "--admission-lock-report",
        default=os.getenv("BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH"),
        help=(
            "Write the current $5,000 paid-work admission lock. Supplying this "
            "enables fail-closed billing reconciliation and exit status."
        ),
    )
    parser.add_argument(
        "--admission-override",
        default=os.getenv("BLUEPRINT_PAID_SPEND_OVERRIDE_PATH"),
        help="Optional short-lived audited override JSON; absence is normal.",
    )
    parser.add_argument(
        "--max-live-instances",
        type=int,
        default=None,
        help="exit 2 when live GPU allocations exceed this fleet ceiling",
    )
    parser.add_argument(
        "--max-burn-usd-per-hour",
        type=float,
        default=None,
        help="exit 2 when estimated live GPU burn exceeds this hourly fleet ceiling",
    )
    parser.add_argument(
        "--spend-ledger",
        default=os.getenv("BLUEPRINT_GPU_SPEND_LEDGER"),
        metavar="PATH",
        help=(
            "persist a gpu_spend_ledger.v1 daily/total spend estimate and include "
            "it in the fleet budget guard"
        ),
    )
    parser.add_argument(
        "--max-daily-spend-usd",
        type=float,
        default=None,
        help="exit 2 when the spend ledger's daily estimate exceeds this ceiling",
    )
    parser.add_argument(
        "--max-total-spend-usd",
        type=float,
        default=None,
        help="exit 2 when the spend ledger's total estimate exceeds this ceiling",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        metavar="PATH",
        help=(
            "also write a machine-readable gpu_spend_guard.v1 snapshot (instances, "
            "burn, protections, reap candidates, and reap results) to this file"
        ),
    )
    args = parser.parse_args(argv)

    secrets_dir = Path(
        str(os.getenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR") or SECRETS_DIR)
    ).expanduser()
    runpod_key = _read_secret("runpod_api_key", secrets_dir=secrets_dir)
    vast_key = _read_secret("vast_api_key", secrets_dir=secrets_dir)
    do_token = _read_secret("digitalocean_api_token", secrets_dir=secrets_dir)
    now = _now()
    credentials = {
        "runpod": runpod_key,
        "vast": vast_key,
        "digitalocean": do_token,
        "gcp": "configured" if cloud_provider_configured("gcp") else None,
        "aws": "configured" if cloud_provider_configured("aws") else None,
    }
    configured = {provider for provider, value in credentials.items() if value}
    required_providers = (
        configured
        if args.admission_lock_report
        else (set(args.require_provider) or configured)
    )
    if not configured and not required_providers:
        required_providers = set(PROVIDERS)
    runpod_pods, runpod_inventory = _inventory_query(
        provider="runpod",
        credential=runpod_key,
        fetch=fetch_runpod_pods,
        timeout=args.timeout,
        required="runpod" in required_providers,
    )
    vast_instances, vast_inventory = _inventory_query(
        provider="vast",
        credential=vast_key,
        fetch=fetch_vast_instances,
        timeout=args.timeout,
        required="vast" in required_providers,
    )
    do_droplets, do_inventory = _inventory_query(
        provider="digitalocean",
        credential=do_token,
        fetch=fetch_do_droplets,
        timeout=args.timeout,
        required="digitalocean" in required_providers,
    )
    gcp_instances, gcp_inventory = _inventory_query(
        provider="gcp",
        credential=credentials["gcp"],
        fetch=lambda _credential, *, timeout: fetch_cloud_vm_instances("gcp", timeout=timeout),
        timeout=args.timeout,
        required="gcp" in required_providers,
    )
    aws_instances, aws_inventory = _inventory_query(
        provider="aws",
        credential=credentials["aws"],
        fetch=lambda _credential, *, timeout: fetch_cloud_vm_instances("aws", timeout=timeout),
        timeout=args.timeout,
        required="aws" in required_providers,
    )
    inventory_results = [
        runpod_inventory,
        vast_inventory,
        do_inventory,
        gcp_inventory,
        aws_inventory,
    ]
    inventory_blocked = any(result.get("blockers") for result in inventory_results)
    if not configured:
        print(
            "No file-based GPU credentials or configured cloud identity found; "
            "provider inventory is unknown and blocked.",
            file=sys.stderr,
        )
    instances = collect_instances(
        now=now, runpod_pods=runpod_pods, vast_instances=vast_instances,
        do_droplets=do_droplets,
        gcp_instances=gcp_instances,
        aws_instances=aws_instances,
    )

    roots = [Path(p) for p in (args.output_root or default_output_roots())]
    protected = find_protected_pod_ids(roots, process_cmdlines=list_process_cmdlines())
    serve_pods = find_expected_serve_pod_ids(
        roots,
        now=now,
        max_lease_seconds=max(1, args.warm_lease_seconds),
    )
    protected = protected | serve_pods
    spend_ledger = (
        update_spend_ledger(instances, ledger_path=Path(args.spend_ledger), now=now)
        if args.spend_ledger
        else None
    )
    fleet_budget = build_fleet_budget_guard(
        instances,
        max_live_instances=args.max_live_instances,
        max_burn_usd_per_hour=args.max_burn_usd_per_hour,
        spend_ledger=spend_ledger,
        max_daily_spend_usd=args.max_daily_spend_usd,
        max_total_spend_usd=args.max_total_spend_usd,
    )
    billing_reconciliation = reconcile_billing_export(
        billing_export_path=Path(args.billing_export) if args.billing_export else None,
        instances=instances,
        now=now,
        required=bool(args.require_billing_reconciliation or args.admission_lock_report),
    )
    billing_blocked = bool(billing_reconciliation.get("blockers"))

    print(
        build_report(
            instances,
            protected_ids=protected,
            max_boot_seconds=args.max_boot_seconds,
            max_booted_orphan_seconds=args.max_booted_orphan_seconds,
            serve_pod_ids=serve_pods,
            fleet_budget=fleet_budget,
        )
    )

    candidates = [
        i
        for i in instances
        if is_reapable(
            i,
            max_boot_seconds=args.max_boot_seconds,
            protected_ids=protected,
            max_booted_orphan_seconds=args.max_booted_orphan_seconds,
        )
    ]
    reap_results: list[dict] = []

    def _write_outputs() -> dict[str, Any] | None:
        admission_lock: dict[str, Any] | None = None
        report = build_json_report(
            instances,
            protected_ids=protected,
            max_boot_seconds=args.max_boot_seconds,
            max_booted_orphan_seconds=args.max_booted_orphan_seconds,
            fleet_budget=fleet_budget,
            spend_ledger=spend_ledger,
            reap_mode=bool(args.reap),
            reap_results=reap_results,
            inventory_results=inventory_results,
            billing_reconciliation=billing_reconciliation,
        )
        if args.admission_lock_report:
            admission_lock = build_spend_admission_lock(
                fleet_budget=fleet_budget,
                billing_reconciliation=billing_reconciliation,
                instances=report["instances"],
                reap_results=reap_results,
                inventory_results=[
                    result for result in inventory_results
                    if result.get("required") is True
                ],
                override_path=Path(args.admission_override)
                if args.admission_override
                else None,
                now=datetime.fromtimestamp(now, timezone.utc),
            )
            admission_path = Path(args.admission_lock_report)
            admission_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(admission_path, admission_lock)
            print(f"Paid-work admission lock written: {admission_path}")
            report = build_json_report(
                instances,
                protected_ids=protected,
                max_boot_seconds=args.max_boot_seconds,
                max_booted_orphan_seconds=args.max_booted_orphan_seconds,
                fleet_budget=fleet_budget,
                spend_ledger=spend_ledger,
                reap_mode=bool(args.reap),
                reap_results=reap_results,
                inventory_results=inventory_results,
                billing_reconciliation=billing_reconciliation,
                spend_admission_lock=admission_lock,
            )
        if args.json_report:
            path = Path(args.json_report)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, report)
            print(f"JSON snapshot written: {path}")
        return admission_lock

    def _fleet_blocks_exit(admission_lock: Mapping[str, Any] | None) -> bool:
        if fleet_budget.get("status") != "blocked":
            return False
        return not (
            isinstance(admission_lock, Mapping)
            and admission_lock.get("status") == "override_open"
            and set(fleet_budget.get("blockers") or [])
            == {"fleet_total_spend_limit_exceeded"}
        )

    if not candidates:
        admission_lock = _write_outputs()
        return 2 if (
            _fleet_blocks_exit(admission_lock)
            or inventory_blocked
            or billing_blocked
            or (
                admission_lock is not None
                and admission_lock.get("admission_allowed") is not True
            )
        ) else 0
    if not args.reap:
        print(
            f"\n(dry-run) {len(candidates)} orphan(s) would be reaped. "
            "Re-run with --reap to terminate."
        )
        admission_lock = _write_outputs()
        return 2 if (
            _fleet_blocks_exit(admission_lock)
            or inventory_blocked
            or billing_blocked
            or (
                admission_lock is not None
                and admission_lock.get("admission_allowed") is not True
            )
        ) else 0

    print(f"\nReaping {len(candidates)} orphan(s)...")
    for inst in candidates:
        result = terminate_instance(
            inst,
            runpod_key=runpod_key,
            vast_key=vast_key,
            do_token=do_token,
        )
        reap_results.append(
            {"provider": inst.provider, "id": inst.id, **{k: result.get(k) for k in ("status", "http")}}
        )
        print(
            f"  reap {inst.provider} {inst.id}: "
            f"{result.get('status')} (http={result.get('http')})"
        )
    admission_lock = _write_outputs()
    reap_failed = any(result.get("status") != "terminated" for result in reap_results)
    return 2 if (
        _fleet_blocks_exit(admission_lock)
        or inventory_blocked
        or billing_blocked
        or reap_failed
        or (
            admission_lock is not None
            and admission_lock.get("admission_allowed") is not True
        )
    ) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
