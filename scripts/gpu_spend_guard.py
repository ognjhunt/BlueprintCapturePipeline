#!/usr/bin/env python3
"""GPU spend guard — report live RunPod/Vast GPU pods and reap orphaned duds.

A standalone cost watchdog. It reads file-based credentials from
``~/.blueprint-secrets`` (``runpod_api_key``, ``vast_api_key``), lists every live
pod/instance with its id, name, age, runtime/boot state and ``$/hr`` (plus a total
burn estimate), and — only with ``--reap`` — terminates pods that are clearly
orphaned: allocated but never booted (``runtime`` absent) past
``--max-boot-seconds`` (default 480s), the classic "stuck dud that keeps billing".

Safety rails (the whole point of the tool is to never kill live work):

* Default is **dry-run** — it reports and would-reap, but changes nothing.
* A pod whose id appears in any ``*/object_store_real_run/started_pod_id.txt``
  under an ``output/.../pipeline/`` tree **that still has a live owning process**
  is never reaped, no matter how stuck it looks. Only orphans whose launching run
  has died are eligible.
* Healthy booted pods are never auto-reaped — boot-timeout duds only.

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
SECRETS_DIR = Path.home() / ".blueprint-secrets"
RUNPOD_API = "https://rest.runpod.io/v1"
VAST_API = "https://console.vast.ai/api/v0"
DEFAULT_MAX_BOOT_SECONDS = 480

# Vast statuses that mean the instance is no longer billing compute.
VAST_TERMINAL_STATUSES = frozenset(
    {"stopped", "exited", "failed", "destroyed", "deleted", "inactive", "completed"}
)
# RunPod desired-states that mean the pod is no longer a live compute allocation.
RUNPOD_TERMINAL_STATUSES = frozenset({"EXITED", "TERMINATED", "TERMINATING"})


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
    live = booted or desired not in RUNPOD_TERMINAL_STATUSES

    if booted:
        state = "running"
    elif desired == "RUNNING":
        state = "booting"
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


def collect_instances(
    *,
    now: float,
    runpod_pods: Sequence[Mapping[str, Any]] | None = None,
    vast_instances: Sequence[Mapping[str, Any]] | None = None,
) -> list[GpuInstance]:
    """Parse raw provider JSON rows into :class:`GpuInstance` records (no network)."""
    instances: list[GpuInstance] = []
    for pod in runpod_pods or []:
        if isinstance(pod, Mapping):
            instances.append(_parse_runpod_pod(pod, now=now))
    for inst in vast_instances or []:
        if isinstance(inst, Mapping):
            instances.append(_parse_vast_instance(inst, now=now))
    return instances


def total_burn_per_hour(instances: Iterable[GpuInstance]) -> float:
    return sum(i.cost_per_hr for i in instances if i.live)


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


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


# ----------------------------- ownership / live owner -----------------------------


# The render job writes the launched provider id into the run's
# object_store_real_run dir: started_pod_id.txt (RunPod) / started_vast_instance_id.txt
# (Vast). Both confer ownership — protect either against reaping.
OWNER_ID_FILENAMES = ("started_pod_id.txt", "started_vast_instance_id.txt")


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
) -> bool:
    """True only for an orphaned dud: live, never booted, older than the boot
    threshold, and not protected by a live owning process."""
    if inst.id in protected_ids:
        return False
    if not inst.live:
        return False
    if inst.booted:
        return False
    if inst.age_seconds is None:
        return False
    return inst.age_seconds > max_boot_seconds


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
) -> str:
    live = [i for i in instances if i.live]
    lines: list[str] = []
    lines.append(f"GPU spend guard — {len(live)} live instance(s)")
    if live:
        lines.append(
            f"{'PROVIDER':8} {'ID':16} {'NAME':22} {'AGE':>7} {'STATE':9} {'$/HR':>7}"
        )
        for inst in live:
            owned = " [owned]" if inst.id in protected_ids else ""
            lines.append(
                f"{inst.provider:8} {inst.id:16.16} {inst.name:22.22} "
                f"{_fmt_age(inst.age_seconds):>7} {inst.state:9} "
                f"{inst.cost_per_hr:7.3f}{owned}"
            )
    burn = total_burn_per_hour(instances)
    lines.append(f"Total burn estimate: ${burn:.3f}/hr (${burn * 24:.2f}/day)")

    candidates = [
        i
        for i in instances
        if is_reapable(i, max_boot_seconds=max_boot_seconds, protected_ids=protected_ids)
    ]
    if candidates:
        lines.append(
            f"Orphan reap candidates ({len(candidates)}): "
            f"not booted past {max_boot_seconds}s, no live owner"
        )
        for inst in candidates:
            lines.append(
                f"  - {inst.provider} {inst.id} ({inst.name}) "
                f"age={_fmt_age(inst.age_seconds)} ${inst.cost_per_hr:.3f}/hr"
            )
    else:
        lines.append("Orphan reap candidates: none")
    return "\n".join(lines)


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
    args = parser.parse_args(argv)

    runpod_key = _read_secret("runpod_api_key")
    vast_key = _read_secret("vast_api_key")
    if not runpod_key and not vast_key:
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
    instances = collect_instances(
        now=now, runpod_pods=runpod_pods, vast_instances=vast_instances
    )

    roots = [Path(p) for p in (args.output_root or default_output_roots())]
    protected = find_protected_pod_ids(roots, process_cmdlines=list_process_cmdlines())

    print(build_report(instances, protected_ids=protected, max_boot_seconds=args.max_boot_seconds))

    candidates = [
        i
        for i in instances
        if is_reapable(i, max_boot_seconds=args.max_boot_seconds, protected_ids=protected)
    ]
    if not candidates:
        return 0
    if not args.reap:
        print(
            f"\n(dry-run) {len(candidates)} orphan(s) would be reaped. "
            "Re-run with --reap to terminate."
        )
        return 0

    print(f"\nReaping {len(candidates)} orphan(s)...")
    for inst in candidates:
        result = terminate_instance(inst, runpod_key=runpod_key, vast_key=vast_key)
        print(
            f"  reap {inst.provider} {inst.id}: "
            f"{result.get('status')} (http={result.get('http')})"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
