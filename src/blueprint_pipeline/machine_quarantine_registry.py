"""Persistent cross-run machine quarantine registry (startup reliability P0-2).

Pre-runtime machine quarantine used to live only inside a single
``launch_with_marker_retry`` call, so a later command could re-select the same
dead or driver-incompatible provider machine. This registry makes machine
failures durable across runs, keyed by:

- provider;
- provider machine ID;
- image digest;
- Isaac version;
- failure class.

Fail-closed rules:

- Only machine-attributable startup/runtime-canary failures may be recorded.
  Placement, policy, kitchen-asset, or task-validation failures are refused —
  those say nothing about the host.
- Entries never contain secrets or raw provider API payloads; evidence is
  recorded as paths plus checksums only.
- Quarantine is advisory at create time: most providers cannot exclude a
  machine during create, and every entry records that limitation explicitly
  (``provider_exclusion_supported``) so callers do not pretend a quarantine
  guarantees a different host.
- A corrupted registry file is skipped and surfaced, never a crash and never a
  silently honored match.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import ensure_dir, utc_now_iso, write_json

SCHEMA_VERSION = "machine_quarantine_registry.v1"
# The one Isaac generation this lane targets; keys must match across writers.
DEFAULT_ISAAC_VERSION = "6.0.0"
REGISTRY_DIR_ENV = "BLUEPRINT_MACHINE_QUARANTINE_DIR"
DEFAULT_REGISTRY_DIR = Path.home() / ".blueprint-machine-quarantine"
# Provider fleets churn; a two-week TTL keeps dead-host memory useful without
# permanently blacklisting hardware that may have been re-imaged.
DEFAULT_TTL_SECONDS = 14 * 24 * 3600

PHASE_PRE_RUNTIME = "pre_runtime"
PHASE_RUNTIME_CANARY = "runtime_canary"
VALID_PHASES = frozenset({PHASE_PRE_RUNTIME, PHASE_RUNTIME_CANARY})

# Machine-attributable failure classes. The registry stays open to new classes
# (slug-validated) but refuses the classes below outright: they are properties
# of the scene/task/policy, not of the provider host.
FORBIDDEN_FAILURE_CLASS_MARKERS = (
    "placement",
    "policy",
    "kitchen",
    "task_validation",
    "stance",
    "scene_load",
)
KNOWN_FAILURE_CLASSES = frozenset(
    {
        "container_never_started",
        "image_pull_stalled",
        "no_runtime",
        "stale_marker",
        "driver_incompatible",
        "cuda_unavailable",
        "rtx_init_failed",
        "empty_frame",
        "gpu_unhealthy",
    }
)

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,79}$")
_SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "authorization",
    "credential",
    "signature",
)
_SECRET_VALUE_MARKERS = ("x-amz-", "bearer ", "?token=", "&token=", "signature=")


class QuarantineRefused(ValueError):
    """Raised when an entry must not be recorded (wrong class or unsafe data)."""


def _registry_dir(registry_dir: str | Path | None) -> Path:
    if registry_dir:
        return Path(registry_dir).expanduser()
    env_dir = str(os.getenv(REGISTRY_DIR_ENV) or "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return DEFAULT_REGISTRY_DIR


def _slug(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise QuarantineRefused(f"machine_quarantine_{field}_missing")
    return text


def _validate_failure_class(failure_class: str) -> str:
    text = str(failure_class or "").strip().lower()
    if not _SLUG_RE.match(text):
        raise QuarantineRefused("machine_quarantine_failure_class_invalid")
    for marker in FORBIDDEN_FAILURE_CLASS_MARKERS:
        if marker in text:
            raise QuarantineRefused(
                "machine_quarantine_refused_non_machine_failure_class:" + text
            )
    return text


def _reject_secret_text(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    for marker in _SECRET_VALUE_MARKERS:
        if marker in lowered:
            raise QuarantineRefused(f"machine_quarantine_secretlike_value:{field}")
    if len(text) > 512:
        raise QuarantineRefused(f"machine_quarantine_value_too_large:{field}")
    return text


def _entry_key(
    provider: str, machine_id: str, image_digest: str, isaac_version: str, failure_class: str
) -> str:
    joined = "\n".join((provider, machine_id, image_digest, isaac_version, failure_class))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def _entry_path(registry: Path, key: str) -> str:
    return str(registry / f"quarantine-{key}.json")


def _evidence_records(evidence_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in evidence_paths:
        path = Path(raw)
        record: dict[str, Any] = {"path": str(path), "sha256": None, "bytes": None}
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            record["sha256"] = digest.hexdigest()
            record["bytes"] = path.stat().st_size
        records.append(record)
    return records


def _read_entry(path: Path) -> dict[str, Any] | None:
    """Return the parsed entry, or None when the file is corrupted/foreign."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    entry = dict(payload)
    if entry.get("schema_version") != SCHEMA_VERSION:
        return None
    return entry


def _entry_expired(entry: Mapping[str, Any], now_epoch: float) -> bool:
    try:
        expires = float(entry.get("expires_at_epoch") or 0.0)
    except (TypeError, ValueError):
        return True
    return now_epoch >= expires


def record_machine_quarantine(
    *,
    provider: str,
    machine_id: str,
    image_digest: str,
    isaac_version: str,
    failure_class: str,
    phase: str,
    gpu_name: str | None = None,
    driver_version: str | None = None,
    evidence_paths: Iterable[str | Path] = (),
    run_id: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    registry_dir: str | Path | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Create or refresh a durable quarantine entry for one machine identity.

    Concurrent writers are safe at the filesystem level: each update is an
    atomic whole-file replace under an exclusive advisory lock, so readers
    never observe a torn entry and two racing writers never corrupt the file.
    """
    provider_slug = _slug(provider, field="provider").lower()
    machine = _slug(machine_id, field="machine_id")
    digest = _slug(image_digest, field="image_digest")
    isaac = _slug(isaac_version, field="isaac_version")
    fail_class = _validate_failure_class(failure_class)
    phase_name = str(phase or "").strip().lower()
    if phase_name not in VALID_PHASES:
        raise QuarantineRefused("machine_quarantine_phase_invalid:" + phase_name)
    gpu = _reject_secret_text(gpu_name or "", field="gpu_name") or None
    driver = _reject_secret_text(driver_version or "", field="driver_version") or None
    _reject_secret_text(machine, field="machine_id")

    now = time.time() if now_epoch is None else float(now_epoch)
    ttl = max(60, int(ttl_seconds))
    registry = _registry_dir(registry_dir)
    ensure_dir(registry)
    key = _entry_key(provider_slug, machine, digest, isaac, fail_class)
    path = Path(_entry_path(registry, key))
    lock_path = registry / f"quarantine-{key}.lock"

    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            import fcntl

            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        existing = _read_entry(path) if path.exists() else None
        stale = existing is not None and _entry_expired(existing, now)
        if existing is None or stale:
            first_epoch = now
            first_at = (
                datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
            )
            attempt_count = 1
        else:
            first_epoch = float(existing.get("first_observed_epoch") or now)
            first_at = str(existing.get("first_observed_at") or utc_now_iso())
            attempt_count = int(existing.get("attempt_count") or 0) + 1
        entry: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "provider": provider_slug,
            "machine_id": machine,
            "image_digest": digest,
            "isaac_version": isaac,
            "failure_class": fail_class,
            "failure_class_registered": fail_class in KNOWN_FAILURE_CLASSES,
            "phase": phase_name,
            "gpu_name": gpu,
            "driver_version": driver,
            "run_id": str(run_id or "").strip() or None,
            "first_observed_at": first_at,
            "first_observed_epoch": first_epoch,
            "last_observed_at": utc_now_iso(),
            "last_observed_epoch": now,
            "attempt_count": attempt_count,
            "ttl_seconds": ttl,
            "expires_at_epoch": now + ttl,
            "expires_at": datetime.fromtimestamp(now + ttl, tz=timezone.utc).isoformat(),
            "evidence": _evidence_records(evidence_paths),
            "provider_exclusion_supported": False,
            "raw_provider_payload_recorded": False,
            "path": str(path),
            "claim_boundary": (
                "This entry marks one provider machine identity as a known "
                "startup/runtime-canary failure for one image digest and Isaac "
                "version. It cannot force the provider to allocate a different "
                "host and says nothing about scene, placement, policy, or task "
                "outcomes."
            ),
        }
        write_json(path, entry)
    finally:
        os.close(lock_fd)
    return entry


def find_active_quarantine(
    *,
    provider: str,
    machine_id: str,
    image_digest: str,
    isaac_version: str,
    registry_dir: str | Path | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any] | None:
    """Return the newest still-valid quarantine entry for this machine identity.

    The match requires provider + machine + image digest + Isaac version: a new
    image digest or Isaac version gets a fresh chance on the same host. Expired
    entries never match.
    """
    now = time.time() if now_epoch is None else float(now_epoch)
    matches = [
        entry
        for entry in load_quarantine_entries(
            registry_dir=registry_dir, include_expired=False, now_epoch=now
        )
        if entry.get("provider") == str(provider or "").strip().lower()
        and entry.get("machine_id") == str(machine_id or "").strip()
        and entry.get("image_digest") == str(image_digest or "").strip()
        and entry.get("isaac_version") == str(isaac_version or "").strip()
    ]
    if not matches:
        return None
    return max(matches, key=lambda e: float(e.get("last_observed_epoch") or 0.0))


def load_quarantine_entries(
    *,
    registry_dir: str | Path | None = None,
    include_expired: bool = False,
    now_epoch: float | None = None,
) -> list[dict[str, Any]]:
    now = time.time() if now_epoch is None else float(now_epoch)
    registry = _registry_dir(registry_dir)
    if not registry.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(registry.glob("quarantine-*.json")):
        entry = _read_entry(path)
        if entry is None:
            continue
        entry["path"] = str(path)
        entry["expired"] = _entry_expired(entry, now)
        if include_expired or not entry["expired"]:
            entries.append(entry)
    return entries


def registry_health(
    *, registry_dir: str | Path | None = None, now_epoch: float | None = None
) -> dict[str, Any]:
    """Report readable/corrupted/expired counts without failing on bad files."""
    now = time.time() if now_epoch is None else float(now_epoch)
    registry = _registry_dir(registry_dir)
    corrupted: list[str] = []
    active = 0
    expired = 0
    if registry.is_dir():
        for path in sorted(registry.glob("quarantine-*.json")):
            entry = _read_entry(path)
            if entry is None:
                corrupted.append(str(path))
            elif _entry_expired(entry, now):
                expired += 1
            else:
                active += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_dir": str(registry),
        "active_count": active,
        "expired_count": expired,
        "corrupted_files": corrupted,
        "generated_at": utc_now_iso(),
    }


def purge_expired(
    *, registry_dir: str | Path | None = None, now_epoch: float | None = None
) -> dict[str, Any]:
    now = time.time() if now_epoch is None else float(now_epoch)
    registry = _registry_dir(registry_dir)
    removed: list[str] = []
    if registry.is_dir():
        for path in sorted(registry.glob("quarantine-*.json")):
            entry = _read_entry(path)
            if entry is not None and _entry_expired(entry, now):
                path.unlink(missing_ok=True)
                Path(str(path)[: -len(".json")] + ".lock").unlink(missing_ok=True)
                removed.append(str(path))
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_dir": str(registry),
        "removed": removed,
        "generated_at": utc_now_iso(),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Durable cross-run provider machine quarantine registry."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    lister = sub.add_parser("list", help="List active quarantine entries.")
    lister.add_argument("--registry-dir", default=None)
    lister.add_argument("--include-expired", action="store_true")
    purger = sub.add_parser("purge-expired", help="Delete expired entries.")
    purger.add_argument("--registry-dir", default=None)
    args = parser.parse_args(argv)

    if args.command == "list":
        entries = load_quarantine_entries(
            registry_dir=args.registry_dir, include_expired=args.include_expired
        )
        print(json.dumps(entries, indent=2, default=str))
        return 0
    report = purge_expired(registry_dir=args.registry_dir)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
