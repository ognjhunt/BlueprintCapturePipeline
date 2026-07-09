#!/usr/bin/env python3
"""GPU spend guard — report live RunPod/Vast GPU pods and reap orphaned duds.

A standalone cost watchdog. It reads file-based credentials from
``~/.blueprint-secrets`` (``runpod_api_key``, ``vast_api_key``), lists every live
pod/instance with its id, name, age, runtime/boot state and ``$/hr`` (plus a total
burn estimate), and — only with ``--reap`` — terminates pods that are clearly
orphaned: allocated but never booted (``runtime`` absent) past
``--max-boot-seconds`` (default 480s), the classic "stuck dud that keeps billing",
or booted allocations with no live owner past ``--max-booted-orphan-seconds``.

Safety rails (the whole point of the tool is to never kill live work):

* Default is **dry-run** — it reports and would-reap, but changes nothing.
* A pod whose id appears in any ``*/object_store_real_run/started_pod_id.txt``
  under an ``output/.../pipeline/`` tree **that still has a live owning process**
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
import json
import os
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

SCHEMA_VERSION = "gpu_spend_guard.v1"
SPEND_LEDGER_SCHEMA_VERSION = "gpu_spend_ledger.v1"
SECRETS_DIR = Path.home() / ".blueprint-secrets"
RUNPOD_API = "https://rest.runpod.io/v1"
VAST_API = "https://console.vast.ai/api/v0"
DEFAULT_MAX_BOOT_SECONDS = 480
DEFAULT_MAX_BOOTED_ORPHAN_SECONDS = 4 * 60 * 60

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
    size = droplet.get("size") if isinstance(droplet.get("size"), Mapping) else {}
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


def collect_instances(
    *,
    now: float,
    runpod_pods: Sequence[Mapping[str, Any]] | None = None,
    vast_instances: Sequence[Mapping[str, Any]] | None = None,
    do_droplets: Sequence[Mapping[str, Any]] | None = None,
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
    return instances


def total_burn_per_hour(instances: Iterable[GpuInstance]) -> float:
    return sum(i.cost_per_hr for i in instances if i.live)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


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
    previous = _load_json_mapping(ledger_path) if ledger_path.is_file() else {}
    previous_instances = (
        dict(previous.get("instances"))
        if isinstance(previous.get("instances"), Mapping)
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
        prior = (
            dict(previous_instances.get(key))
            if isinstance(previous_instances.get(key), Mapping)
            else {}
        )
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
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except OSError as exc:
        ledger["status"] = "blocked"
        ledger["blockers"] = [f"spend_ledger_write_failed:{type(exc).__name__}"]
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
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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


def fetch_runpod_pods(key: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    status, payload = _http_request("GET", f"{RUNPOD_API}/pods", key=key, timeout=timeout)
    if status not in (200, 201):
        _warn(f"runpod pod query failed (http={status})")
        return []
    return _rows_from_payload(payload, "pods", "data", "items")


def fetch_vast_instances(key: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    status, payload = _http_request("GET", f"{VAST_API}/instances/", key=key, timeout=timeout)
    if status not in (200, 201):
        _warn(f"vast instance query failed (http={status})")
        return []
    return _rows_from_payload(payload, "instances", "results", "data")


def fetch_do_droplets(token: str, *, timeout: int = 30) -> list[dict[str, Any]]:
    """Only GPU droplets (size slug gpu-*) count toward render-lane spend."""
    status, payload = _http_request(
        "GET", f"{DO_API}/droplets?per_page=200", key=token, timeout=timeout
    )
    if status not in (200, 201):
        _warn(f"digitalocean droplet query failed (http={status})")
        return []
    rows = _rows_from_payload(payload, "droplets", "data")
    return [
        r for r in rows
        if str((r.get("size") or {}).get("slug") or r.get("size_slug") or "").startswith("gpu-")
    ]


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


# ----------------------------- ownership / live owner -----------------------------


# The render job writes the launched provider id into the run's
# object_store_real_run dir: started_pod_id.txt (RunPod) / started_vast_instance_id.txt
# (Vast). Both confer ownership — protect either against reaping.
OWNER_ID_FILENAMES = ("started_pod_id.txt", "started_vast_instance_id.txt", "started_do_droplet_id.txt")


def _iter_owner_id_files(
    output_roots: Iterable[Path | str], filename: str
) -> list[tuple[Path, str]]:
    """``(file_path, launched_id)`` for every ``object_store_real_run/<filename>``
    under a ``pipeline`` tree in any output root."""
    found: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for root in output_roots:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        for path in base.glob(f"**/object_store_real_run/{filename}"):
            if "pipeline" not in path.parts:
                continue
            try:
                launched_id = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not launched_id:
                continue
            dedupe_key = (str(path), launched_id)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            found.append((path, launched_id))
    return found


def iter_started_pod_id_files(output_roots: Iterable[Path | str]) -> list[tuple[Path, str]]:
    """RunPod ``(file_path, pod_id)`` pairs (see :func:`_iter_owner_id_files`)."""
    return _iter_owner_id_files(output_roots, "started_pod_id.txt")


WARM_SERVE_MARKER_FILENAME = "warm_serve_pod.json"


def find_expected_serve_pod_ids(output_roots: Iterable[Path | str]) -> set[str]:
    """Pod ids recorded as live warm serve workers (marker status == 'serving')."""
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
            if pod_id and payload.get("status") == "serving":
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
    a live owner when some running process's command line references that run — by the
    launched id, by the ``object_store_real_run`` dir, or by the ``--out-dir`` (its
    parent) the render job was launched with. If the file exists but no process
    references it, the launching run has died and the instance is an orphan (eligible
    for reaping), not protected.
    """
    protected: set[str] = set()
    cmdlines = [c for c in process_cmdlines if isinstance(c, str)]
    owner_files: list[tuple[Path, str]] = []
    for filename in OWNER_ID_FILENAMES:
        owner_files.extend(_iter_owner_id_files(output_roots, filename))
    for path, launched_id in owner_files:
        job_dir = str(path.parent)  # .../object_store_real_run
        out_dir = str(path.parent.parent)  # the render job's --out-dir
        for cmd in cmdlines:
            if (
                launched_id in cmd
                or _cmd_references_path(cmd, out_dir)
                or _cmd_references_path(cmd, job_dir)
            ):
                protected.add(launched_id)
                break
    return protected


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
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
            "Report live RunPod/Vast GPU pods with a burn estimate, and optionally "
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

    runpod_key = _read_secret("runpod_api_key")
    vast_key = _read_secret("vast_api_key")
    do_token = _read_secret("digitalocean_api_token")
    if not runpod_key and not vast_key and not do_token:
        print(
            "No file-based GPU credentials found in ~/.blueprint-secrets "
            "(runpod_api_key, vast_api_key); nothing to check."
        )
        return 0

    now = _now()
    runpod_pods = fetch_runpod_pods(runpod_key, timeout=args.timeout) if runpod_key else []
    vast_instances = (
        fetch_vast_instances(vast_key, timeout=args.timeout) if vast_key else []
    )
    do_droplets = fetch_do_droplets(do_token, timeout=args.timeout) if do_token else []
    instances = collect_instances(
        now=now, runpod_pods=runpod_pods, vast_instances=vast_instances,
        do_droplets=do_droplets,
    )

    roots = [Path(p) for p in (args.output_root or default_output_roots())]
    protected = find_protected_pod_ids(roots, process_cmdlines=list_process_cmdlines())
    serve_pods = find_expected_serve_pod_ids(roots)
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

    def _write_json_report() -> None:
        if not args.json_report:
            return
        report = build_json_report(
            instances,
            protected_ids=protected,
            max_boot_seconds=args.max_boot_seconds,
            max_booted_orphan_seconds=args.max_booted_orphan_seconds,
            fleet_budget=fleet_budget,
            spend_ledger=spend_ledger,
            reap_mode=bool(args.reap),
            reap_results=reap_results,
        )
        path = Path(args.json_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON snapshot written: {path}")

    if not candidates:
        _write_json_report()
        return 2 if fleet_budget.get("status") == "blocked" else 0
    if not args.reap:
        print(
            f"\n(dry-run) {len(candidates)} orphan(s) would be reaped. "
            "Re-run with --reap to terminate."
        )
        _write_json_report()
        return 2 if fleet_budget.get("status") == "blocked" else 0

    print(f"\nReaping {len(candidates)} orphan(s)...")
    for inst in candidates:
        result = terminate_instance(inst, runpod_key=runpod_key, vast_key=vast_key)
        reap_results.append(
            {"provider": inst.provider, "id": inst.id, **{k: result.get(k) for k in ("status", "http")}}
        )
        print(
            f"  reap {inst.provider} {inst.id}: "
            f"{result.get('status')} (http={result.get('http')})"
        )
    _write_json_report()
    return 2 if fleet_budget.get("status") == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
