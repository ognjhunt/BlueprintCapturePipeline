"""Atomic cross-process lease for a paid provider lane's billable mutations.

Pending-teardown records protect billing cleanup per attempt, but they do not
prove exclusive local ownership: two concurrent local agents can both open
their own records and both create pods. This module provides the missing
single-owner lease for a (provider, lane) pair.

Semantics, deliberately narrow:

- Scope is billable provider mutation (create/start/rebuild) for one lane and
  provider account on this machine — not pure preparation or tests.
- Acquisition is atomic (hardlink-from-temp, O_EXCL-equivalent): the second
  concurrent process is blocked with ``paid_provider_lane_already_owned``
  before any provider API call.
- A stale lease (dead owner PID on this host, or past expiry) may be reclaimed
  ONLY with explicit reconciliation evidence that provider inventory and open
  pending-teardown records were reviewed and reclaiming cannot orphan an
  allocation. A crashed owner therefore keeps the lane closed until a human or
  runner reconciles — that is the fail-closed intent, not a bug.
- Releasing the lease never proves billing stopped; teardown proofs do.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .paid_lane_guard import load_pending_teardowns
from .render_lock import _atomic_create_with_content, _hostname, _pid_is_alive

LEASE_SCHEMA_VERSION = "paid_provider_lane_lease.v1"
LEASE_HANDOFF_SCHEMA_VERSION = "paid_provider_lane_lease_handoff.v1"
RECONCILIATION_SCHEMA_VERSION = "paid_provider_lane_reconciliation.v1"
LEASE_DIR_ENV = "BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR"
DEFAULT_LEASE_TTL_SECONDS = 4 * 3600
BLOCKER_ALREADY_OWNED = "paid_provider_lane_already_owned"
BLOCKER_STALE_REQUIRES_RECONCILIATION = (
    "paid_provider_lane_stale_lease_requires_reconciliation"
)
BLOCKER_RECONCILIATION_UNAVAILABLE = "paid_provider_lane_reconciliation_unavailable"
BLOCKER_TEARDOWN_UNVERIFIED = "paid_provider_lane_teardown_unverified"
#: Every key the caller must assert (truthfully) before a stale lease may be
#: replaced. The lease module cannot query providers itself; it demands the
#: caller's reconciliation evidence instead of silently reclaiming.
STALE_RECLAIM_REQUIRED_EVIDENCE = (
    "provider_inventory_checked",
    "open_pending_teardowns_reviewed",
    "reclaim_cannot_orphan_allocation",
)
_CLAIM_BOUNDARY = (
    "Lease ownership proves exclusive local mutation intent only. It does not "
    "prove any allocation exists, was torn down, or that billing stopped."
)
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
MIN_HANDOFF_REMAINING_SECONDS = 60


def default_lease_dir() -> Path:
    override = os.environ.get(LEASE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".blueprint-state" / "paid-provider-lane-leases"


def _slug(value: str) -> str:
    return (_SAFE.sub("-", value.strip()).strip("-._") or "lane")[:120]


def lease_path(provider: str, lane: str, lease_dir: str | os.PathLike[str] | None = None) -> Path:
    base = Path(lease_dir).expanduser() if lease_dir is not None else default_lease_dir()
    return base / f"{_slug(provider)}-{_slug(lane)}.lease.json"


def read_lease(
    provider: str, lane: str, lease_dir: str | os.PathLike[str] | None = None
) -> dict[str, Any] | None:
    path = lease_path(provider, lane, lease_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _lease_is_stale(lease: Mapping[str, Any]) -> tuple[bool, str | None]:
    hostname = str(lease.get("hostname") or "")
    same_host = (not hostname) or hostname == _hostname()
    try:
        owner_pid = int(lease.get("owner_pid") or 0)
    except (TypeError, ValueError):
        owner_pid = 0
    if same_host:
        # A live same-host owner is NEVER stale — not even past expiry. Only
        # a dead owner leaves a reclaimable lease on this host.
        if _pid_is_alive(owner_pid):
            return False, None
        try:
            retained_owner_pid = int(lease.get("retained_teardown_owner_pid") or 0)
        except (TypeError, ValueError):
            retained_owner_pid = 0
        if retained_owner_pid and _pid_is_alive(retained_owner_pid):
            return False, None
        return True, "owner_pid_not_alive"
    # Cross-host liveness is unknowable here; expiry is the only stale signal.
    try:
        expires = float(lease.get("expires_at_epoch") or 0.0)
    except (TypeError, ValueError):
        expires = 0.0
    if expires and time.time() > expires:
        return True, "lease_expired"
    return False, None


@contextmanager
def _reclaim_mutex(path: Path) -> Iterator[None]:
    """Serialize stale-lease reclaim so two reconciled processes cannot both
    judge the same lease stale and have the loser unlink the winner's fresh
    live lease (check-then-unlink TOCTOU)."""
    mutex = path.with_name(path.name + ".reclaim-mutex")
    with open(mutex, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reconciliation_complete(
    reconciliation: Mapping[str, Any] | None,
    *,
    provider: str,
    lane: str,
) -> bool:
    if not isinstance(reconciliation, Mapping):
        return False
    return bool(
        reconciliation.get("schema_version") == RECONCILIATION_SCHEMA_VERSION
        and reconciliation.get("status") == "passed"
        and str(reconciliation.get("provider") or "").strip().lower()
        == str(provider).strip().lower()
        and str(reconciliation.get("lane") or "") == str(lane)
        and all(
            reconciliation.get(key) is True
            for key in STALE_RECLAIM_REQUIRED_EVIDENCE
        )
        and reconciliation.get("provider_live_resource_count") == 0
        and reconciliation.get("open_pending_teardown_count") == 0
    )


def build_paid_provider_lane_reconciliation(
    *,
    provider: str,
    lane: str,
    provider_inventory: Mapping[str, Any] | None,
    open_pending_teardowns: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind read-only provider inventory and pending-teardown state to a lane.

    This evidence is required both before initial acquisition (so a legacy or
    manual allocation without a lease is still visible) and before releasing a
    mutated lane. Raw provider responses and credentials are never recorded.
    """
    provider_name = str(provider).strip().lower()
    lane_name = str(lane)
    inventory = dict(provider_inventory or {})
    inventory_confirmed = inventory.get("api_confirmed") is True
    live_count = inventory.get("live_resource_count")
    live_count_valid = type(live_count) is int and live_count >= 0
    matching_pending = [
        record
        for record in open_pending_teardowns
        if str(record.get("provider") or "").strip().lower() == provider_name
        and record.get("status") == "open"
    ]
    blockers: list[str] = []
    if not inventory_confirmed or not live_count_valid:
        blockers.append(BLOCKER_RECONCILIATION_UNAVAILABLE)
    if (live_count_valid and live_count > 0) or matching_pending:
        blockers.append(BLOCKER_ALREADY_OWNED)
    clean = not blockers
    return {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "passed" if clean else "blocked",
        "provider": provider_name,
        "lane": lane_name,
        "provider_inventory_checked": inventory_confirmed and live_count_valid,
        "provider_inventory_api_confirmed": inventory_confirmed,
        "provider_live_resource_count": live_count if live_count_valid else None,
        "provider_resources": [
            {
                key: row.get(key)
                for key in (
                    "instance_id",
                    "name",
                    "desired_status",
                    "status",
                    "cost_per_hour",
                )
                if key in row
            }
            for row in inventory.get("resources", [])
            if isinstance(row, Mapping)
        ],
        "open_pending_teardowns_reviewed": True,
        "open_pending_teardown_count": len(matching_pending),
        "open_pending_teardowns": [
            {
                "path": record.get("path"),
                "run_id": record.get("run_id"),
                "instance_id": record.get("instance_id"),
                "job_dir": record.get("job_dir"),
            }
            for record in matching_pending
        ],
        "reclaim_cannot_orphan_allocation": clean,
        "blockers": blockers,
        "raw_provider_response_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def paid_launch_pending_teardown_max_age(
    *,
    marker_timeout: int,
    startup_no_runtime_timeout: int,
    max_attempts: int,
    max_seconds: int = 0,
) -> int:
    per_attempt = max(
        int(marker_timeout or 0),
        int(startup_no_runtime_timeout or 0),
        60,
    )
    return max(
        300,
        per_attempt * max(1, int(max_attempts or 1))
        + max(0, int(max_seconds or 0))
        + 1800,
    )


def acquire_paid_provider_lane_lease(
    *,
    provider: str,
    lane: str,
    job_dir: str,
    intended_provider: str | None = None,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    lease_dir: str | os.PathLike[str] | None = None,
    reconciliation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically acquire the single-owner lease for (provider, lane).

    Returns ``{"status": "acquired", ...}`` with the recorded lease, or a
    blocked result carrying a stable blocker. Never mutates a live owner's
    lease and never reclaims a stale lease without complete reconciliation
    evidence.
    """
    path = lease_path(provider, lane, lease_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    record = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "provider": str(provider).strip().lower(),
        "lane": str(lane),
        "intended_provider": str(intended_provider or provider).strip().lower(),
        "owner_pid": os.getpid(),
        "hostname": _hostname(),
        "job_dir": str(job_dir),
        "started_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "started_at_epoch": now,
        "expires_at_epoch": now + max(60, int(ttl_seconds)),
        "heartbeat_at_epoch": now,
        "preacquire_reconciliation": dict(reconciliation or {}),
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    if not path.exists() and not _reconciliation_complete(
        reconciliation, provider=provider, lane=lane
    ):
        reconciliation_blockers = list(
            dict(reconciliation or {}).get("blockers") or []
        )
        blocker = (
            BLOCKER_ALREADY_OWNED
            if BLOCKER_ALREADY_OWNED in reconciliation_blockers
            else BLOCKER_RECONCILIATION_UNAVAILABLE
        )
        return {
            "status": "blocked",
            "path": str(path),
            "blockers": [blocker],
            "reconciliation": dict(reconciliation or {}),
            "claim_boundary": _CLAIM_BOUNDARY,
        }
    payload = json.dumps(record, indent=2, sort_keys=True).encode("utf-8")
    for _ in range(3):
        if _atomic_create_with_content(path, payload):
            return {
                "status": "acquired",
                "path": str(path),
                "lease": record,
                "blockers": [],
                "claim_boundary": _CLAIM_BOUNDARY,
            }
        holder = read_lease(provider, lane, lease_dir)
        if holder is None:
            # Vanished or unreadable between create and read; retry the create.
            if not path.exists():
                continue
            return {
                "status": "blocked",
                "path": str(path),
                "blockers": [BLOCKER_ALREADY_OWNED],
                "holder": None,
                "holder_unreadable": True,
                "claim_boundary": _CLAIM_BOUNDARY,
            }
        stale, stale_reason = _lease_is_stale(holder)
        if not stale:
            return {
                "status": "blocked",
                "path": str(path),
                "blockers": [BLOCKER_ALREADY_OWNED],
                "holder": holder,
                "claim_boundary": _CLAIM_BOUNDARY,
            }
        if not _reconciliation_complete(
            reconciliation, provider=provider, lane=lane
        ):
            return {
                "status": "blocked",
                "path": str(path),
                "blockers": [BLOCKER_STALE_REQUIRES_RECONCILIATION],
                "holder": holder,
                "stale_reason": stale_reason,
                "required_reconciliation_evidence": list(STALE_RECLAIM_REQUIRED_EVIDENCE),
                "claim_boundary": _CLAIM_BOUNDARY,
            }
        # Reconciled stale lease: reclaim under a mutex so a racing reclaimer
        # cannot unlink the lease we are about to (or just did) create, and
        # only if the on-disk lease still matches the holder judged stale.
        with _reclaim_mutex(path):
            current = read_lease(provider, lane, lease_dir)
            if current == holder and _lease_is_stale(current)[0]:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                if _atomic_create_with_content(path, payload):
                    return {
                        "status": "acquired",
                        "path": str(path),
                        "lease": record,
                        "blockers": [],
                        "reclaimed_stale_lease": {
                            "previous_owner_pid": holder.get("owner_pid"),
                            "stale_reason": stale_reason,
                        },
                        "claim_boundary": _CLAIM_BOUNDARY,
                    }
    return {
        "status": "blocked",
        "path": str(path),
        "blockers": [BLOCKER_ALREADY_OWNED],
        "holder": read_lease(provider, lane, lease_dir),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def release_paid_provider_lane_lease(
    acquisition: Mapping[str, Any],
    *,
    reason: str,
    provider_mutation_started: bool = True,
    terminal_reconciliation: Mapping[str, Any] | None = None,
    lease_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Release a lease this process acquired. Refuses to delete another owner's.

    ``teardown_verified`` records whether a provider-terminal teardown proof
    accompanied the release; it is evidence bookkeeping, not billing proof.
    """
    lease = acquisition.get("lease") if isinstance(acquisition, Mapping) else None
    if not isinstance(lease, Mapping):
        return {"status": "not_held", "reason": reason, "released": False}
    path = lease_path(str(lease.get("provider") or ""), str(lease.get("lane") or ""), lease_dir)
    current = None
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        current = None
    if not isinstance(current, dict):
        return {"status": "already_released", "reason": reason, "released": False}
    if (
        current.get("owner_pid") != lease.get("owner_pid")
        or current.get("started_at_epoch") != lease.get("started_at_epoch")
    ):
        return {
            "status": "refused_not_owner",
            "reason": reason,
            "released": False,
            "holder": current,
        }
    teardown_verified = _reconciliation_complete(
        terminal_reconciliation,
        provider=str(lease.get("provider") or ""),
        lane=str(lease.get("lane") or ""),
    )
    if provider_mutation_started and not teardown_verified:
        return {
            "status": "retained_unverified_teardown",
            "reason": reason,
            "released": False,
            "blockers": [BLOCKER_TEARDOWN_UNVERIFIED],
            "provider_mutation_started": True,
            "terminal_reconciliation": dict(terminal_reconciliation or {}),
            "claim_boundary": _CLAIM_BOUNDARY,
        }
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    return {
        "status": "released",
        "reason": reason,
        "released": True,
        "provider_mutation_started": bool(provider_mutation_started),
        "teardown_verified": teardown_verified if provider_mutation_started else None,
        "terminal_reconciliation": dict(terminal_reconciliation or {}),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _canonical_handoff_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "provider": str(binding.get("provider") or "").strip().lower(),
        "lane": str(binding.get("lane") or "").strip(),
        "volume_id": str(binding.get("volume_id") or "").strip(),
        "pending_teardown_record": str(
            binding.get("pending_teardown_record") or ""
        ).strip(),
        "watchdog_nonce": str(binding.get("watchdog_nonce") or "").strip(),
        "watchdog_deadline_epoch": binding.get("watchdog_deadline_epoch"),
    }


def _handoff_capability_digest(token: bytes, binding: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_handoff_binding(binding),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(token + b"\0" + payload).hexdigest()


def _handoff_capability_payload(token: bytes, binding: Mapping[str, Any]) -> bytes:
    return json.dumps(
        {
            "schema_version": LEASE_HANDOFF_SCHEMA_VERSION,
            "binding": _canonical_handoff_binding(binding),
            "token_hex": token.hex(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_handoff_capability(path: Path) -> tuple[bytes, dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        token = bytes.fromhex(str(payload.get("token_hex") or ""))
    except (OSError, ValueError, AttributeError):
        return None
    if (
        payload.get("schema_version") != LEASE_HANDOFF_SCHEMA_VERSION
        or len(token) != 32
        or not isinstance(payload.get("binding"), Mapping)
    ):
        return None
    return token, _canonical_handoff_binding(payload["binding"])


def _write_lease(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_secret_file(path: Path, token: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _read_process_argv(pid: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        try:
            command = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ()
        try:
            return tuple(shlex.split(command))
        except ValueError:
            return ()
    return tuple(
        part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part
    )


def _canary_watchdog_identity_valid(
    watchdog: Mapping[str, Any],
    *,
    process_argv_probe: Any,
    clock: Any,
) -> bool:
    pid = watchdog.get("watchdog_pid")
    prefix = str(watchdog.get("watchdog_pod_name_prefix") or "").strip()
    deadline = watchdog.get("watchdog_deadline_epoch")
    if (
        type(pid) is not int
        or pid <= 0
        or not _pid_is_alive(pid)
        or not prefix.startswith("blueprint-groot-oscar-canary-")
        or not isinstance(deadline, (int, float))
        or float(deadline) <= float(clock()) + MIN_HANDOFF_REMAINING_SECONDS
        or watchdog.get("watchdog_process_identity_verified") is not True
        or watchdog.get("independent_teardown_watchdog") is not True
    ):
        return False
    tokens = tuple(str(token) for token in process_argv_probe(pid))
    try:
        module_index = tokens.index("blueprint_pipeline.groot_oscar_runpod_watchdog")
        prefix_index = tokens.index("--pod-name-prefix", module_index + 1) + 1
        deadline_index = tokens.index("--deadline-epoch", module_index + 1) + 1
        observed_deadline = float(tokens[deadline_index])
    except (ValueError, IndexError):
        return False
    return bool(
        module_index > 0
        and tokens[module_index - 1] == "-m"
        and tokens[prefix_index] == prefix
        and observed_deadline == float(deadline)
    )


def transfer_paid_provider_lane_lease_to_watchdog(
    acquisition: Mapping[str, Any],
    *,
    watchdog_pid: int,
    capability_path: str | os.PathLike[str],
    binding: Mapping[str, Any],
    clock: Any = time.time,
) -> dict[str, Any]:
    """Atomically transfer a held lane to a live retained teardown watchdog."""

    lease = acquisition.get("lease") if isinstance(acquisition, Mapping) else None
    if not isinstance(lease, Mapping):
        return {"status": "blocked", "blockers": ["paid_provider_lane_handoff_not_held"]}
    canonical = _canonical_handoff_binding(binding)
    expected = {
        "provider": str(lease.get("provider") or "").strip().lower(),
        "lane": str(lease.get("lane") or "").strip(),
    }
    if (
        canonical["provider"] != expected["provider"]
        or canonical["lane"] != expected["lane"]
        or not canonical["volume_id"]
        or not canonical["pending_teardown_record"]
        or not canonical["watchdog_nonce"]
        or not isinstance(canonical["watchdog_deadline_epoch"], (int, float))
        or float(canonical["watchdog_deadline_epoch"])
        <= float(clock()) + MIN_HANDOFF_REMAINING_SECONDS
    ):
        return {
            "status": "blocked",
            "blockers": ["paid_provider_lane_handoff_binding_invalid"],
        }
    if type(watchdog_pid) is not int or watchdog_pid <= 0 or not _pid_is_alive(watchdog_pid):
        return {
            "status": "blocked",
            "blockers": ["paid_provider_lane_handoff_watchdog_not_alive"],
        }
    path = Path(str(acquisition.get("path") or ""))
    secret_path = Path(capability_path)
    job_dir = Path(str(lease.get("job_dir") or "")).expanduser().resolve()
    if (
        secret_path.expanduser().resolve().parent != job_dir
        or secret_path.name != "provider_lane_handoff.capability"
    ):
        return {
            "status": "blocked",
            "blockers": ["paid_provider_lane_handoff_capability_path_invalid"],
        }
    token = secrets.token_bytes(32)
    digest = _handoff_capability_digest(token, canonical)
    with _reclaim_mutex(path):
        current = read_lease(expected["provider"], expected["lane"], path.parent)
        if not isinstance(current, dict) or (
            current.get("owner_pid") != lease.get("owner_pid")
            or current.get("started_at_epoch") != lease.get("started_at_epoch")
        ):
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_not_current_owner"],
            }
        try:
            pending = json.loads(
                Path(canonical["pending_teardown_record"]).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            pending = {}
        if not isinstance(pending, Mapping) or not bool(
            pending.get("status") == "open"
            and str(pending.get("provider") or "").strip().lower()
            == canonical["provider"]
            and str(pending.get("lane") or "") == canonical["lane"]
            and pending.get("resource_kind") == "network_volume"
            and str(pending.get("instance_id") or "") == canonical["volume_id"]
        ):
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_pending_teardown_invalid"],
            }
        if secret_path.exists() or secret_path.is_symlink():
            recorded_handoff = current.get("handoff")
            if isinstance(recorded_handoff, Mapping) and str(
                recorded_handoff.get("capability_path") or ""
            ) == str(secret_path):
                return {
                    "status": "blocked",
                    "blockers": ["paid_provider_lane_handoff_already_pending"],
                }
            try:
                secret_stat = secret_path.lstat()
            except OSError:
                secret_stat = None
            if (
                secret_stat is None
                or secret_path.is_symlink()
                or not secret_path.is_file()
                or secret_stat.st_mode & 0o077
            ):
                return {
                    "status": "blocked",
                    "blockers": ["paid_provider_lane_handoff_capability_unsafe"],
                }
            prior = _read_handoff_capability(secret_path)
            if prior is None or prior[1] != canonical:
                return {
                    "status": "blocked",
                    "blockers": ["paid_provider_lane_handoff_capability_unowned"],
                }
            secret_path.unlink()
        if not _create_secret_file(
            secret_path, _handoff_capability_payload(token, canonical)
        ):
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_capability_exists"],
            }
        handoff = {
            "schema_version": LEASE_HANDOFF_SCHEMA_VERSION,
            "status": "pending_canary_acceptance",
            "lease_path": str(path),
            "capability_path": str(secret_path),
            "capability_digest": digest,
            "source_owner_pid": watchdog_pid,
            "binding": canonical,
            "created_at_epoch": time.time(),
            "raw_capability_recorded": False,
        }
        current.update(
            {
                "owner_pid": watchdog_pid,
                "owner_role": "retained_network_volume_watchdog",
                "retained_teardown_owner_pid": watchdog_pid,
                "heartbeat_at_epoch": time.time(),
                "handoff": handoff,
            }
        )
        try:
            _write_lease(path, current)
        except Exception:
            secret_path.unlink(missing_ok=True)
            raise
    return handoff


def accept_paid_provider_lane_lease_handoff(
    handoff: Mapping[str, Any],
    *,
    canary_watchdog: Mapping[str, Any],
    expected_binding: Mapping[str, Any],
    process_argv_probe: Any = _read_process_argv,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Consume the one-time handoff and transfer the lane without an owner gap."""

    canonical = _canonical_handoff_binding(expected_binding)
    recorded_binding = _canonical_handoff_binding(
        handoff.get("binding") if isinstance(handoff.get("binding"), Mapping) else {}
    )
    if (
        handoff.get("schema_version") != LEASE_HANDOFF_SCHEMA_VERSION
        or handoff.get("status") != "pending_canary_acceptance"
        or recorded_binding != canonical
    ):
        return {
            "status": "blocked",
            "blockers": ["paid_provider_lane_handoff_contract_invalid"],
        }
    if not _canary_watchdog_identity_valid(
        canary_watchdog,
        process_argv_probe=process_argv_probe,
        clock=clock,
    ):
        return {
            "status": "blocked",
            "blockers": ["paid_provider_lane_canary_watchdog_identity_invalid"],
        }
    canary_watchdog_pid = int(canary_watchdog["watchdog_pid"])
    canary_deadline = float(canary_watchdog["watchdog_deadline_epoch"])
    if (
        float(canonical.get("watchdog_deadline_epoch") or 0)
        < canary_deadline + MIN_HANDOFF_REMAINING_SECONDS
    ):
        return {
            "status": "blocked",
            "blockers": ["paid_provider_lane_handoff_deadline_insufficient"],
        }
    path = Path(str(handoff.get("lease_path") or ""))
    capability_path = Path(str(handoff.get("capability_path") or ""))
    with _reclaim_mutex(path):
        current = read_lease(canonical["provider"], canonical["lane"], path.parent)
        if not isinstance(current, dict) or current.get("handoff") != dict(handoff):
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_not_current"],
            }
        source_owner_pid = handoff.get("source_owner_pid")
        if current.get("owner_pid") != source_owner_pid or not _pid_is_alive(
            int(source_owner_pid or 0)
        ):
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_source_watchdog_not_alive"],
            }
        try:
            pending = json.loads(
                Path(canonical["pending_teardown_record"]).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            pending = {}
        if not isinstance(pending, Mapping) or not bool(
            pending.get("status") == "open"
            and str(pending.get("provider") or "").strip().lower()
            == canonical["provider"]
            and str(pending.get("lane") or "") == canonical["lane"]
            and pending.get("resource_kind") == "network_volume"
            and str(pending.get("instance_id") or "") == canonical["volume_id"]
        ):
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_pending_teardown_invalid"],
            }
        try:
            stat_result = capability_path.lstat()
        except OSError:
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_capability_missing"],
            }
        if not capability_path.is_file() or capability_path.is_symlink() or stat_result.st_mode & 0o077:
            return {
                "status": "blocked",
                "blockers": ["paid_provider_lane_handoff_capability_unsafe"],
            }
        consumed = capability_path.with_name(
            f".{capability_path.name}.consumed-{os.getpid()}-{time.monotonic_ns()}"
        )
        os.replace(capability_path, consumed)
        try:
            capability = _read_handoff_capability(consumed)
            if capability is None or capability[1] != canonical:
                return {
                    "status": "blocked",
                    "blockers": ["paid_provider_lane_handoff_capability_invalid"],
                }
            token = capability[0]
            observed = _handoff_capability_digest(token, canonical)
            if not secrets.compare_digest(
                observed, str(handoff.get("capability_digest") or "")
            ):
                return {
                    "status": "blocked",
                    "blockers": ["paid_provider_lane_handoff_capability_invalid"],
                }
            accepted_at = time.time()
            current.update(
                {
                    "owner_pid": canary_watchdog_pid,
                    "owner_role": "gpu_canary_watchdog",
                    "retained_teardown_owner_pid": source_owner_pid,
                    "heartbeat_at_epoch": accepted_at,
                    "handoff": {
                        **dict(handoff),
                        "status": "accepted",
                        "accepted_at_epoch": accepted_at,
                        "canary_watchdog_pid": canary_watchdog_pid,
                    },
                }
            )
            _write_lease(path, current)
        finally:
            consumed.unlink(missing_ok=True)
    return {
        "schema_version": LEASE_HANDOFF_SCHEMA_VERSION,
        "status": "accepted",
        "lease_path": str(path),
        "owner_pid": canary_watchdog_pid,
        "retained_teardown_owner_pid": source_owner_pid,
        "binding": canonical,
        "capability_digest": handoff.get("capability_digest"),
        "capability_consumed": True,
        "raw_capability_recorded": False,
    }


def release_transferred_paid_provider_lane_lease(
    *,
    lease_path_value: str | os.PathLike[str],
    teardown_owner_pid: int,
    terminal_reconciliation: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Release a transferred lease only after provider-global terminal proof."""

    path = Path(lease_path_value)
    with _reclaim_mutex(path):
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"status": "already_released", "released": False, "reason": reason}
        if teardown_owner_pid not in {
            current.get("owner_pid"),
            current.get("retained_teardown_owner_pid"),
        }:
            return {
                "status": "refused_not_teardown_owner",
                "released": False,
                "reason": reason,
            }
        if not _reconciliation_complete(
            terminal_reconciliation,
            provider=str(current.get("provider") or ""),
            lane=str(current.get("lane") or ""),
        ):
            return {
                "status": "retained_unverified_teardown",
                "released": False,
                "reason": reason,
                "blockers": [BLOCKER_TEARDOWN_UNVERIFIED],
            }
        path.unlink(missing_ok=True)
    return {
        "status": "released",
        "released": True,
        "reason": reason,
        "teardown_verified": True,
    }


def restore_paid_provider_lane_lease_to_retained_watchdog(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(str(receipt.get("lease_path") or ""))
    with _reclaim_mutex(path):
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"status": "already_released", "restored": False}
        retained_pid = int(current.get("retained_teardown_owner_pid") or 0)
        canary_pid = int(receipt.get("owner_pid") or 0)
        accepted_handoff = current.get("handoff")
        accepted_handoff = (
            accepted_handoff if isinstance(accepted_handoff, Mapping) else {}
        )
        if (
            current.get("owner_pid") != canary_pid
            or accepted_handoff.get("status") != "accepted"
            or accepted_handoff.get("canary_watchdog_pid") != canary_pid
            or accepted_handoff.get("capability_digest")
            != receipt.get("capability_digest")
            or _canonical_handoff_binding(
                accepted_handoff.get("binding")
                if isinstance(accepted_handoff.get("binding"), Mapping)
                else {}
            )
            != _canonical_handoff_binding(
                receipt.get("binding")
                if isinstance(receipt.get("binding"), Mapping)
                else {}
            )
            or not _pid_is_alive(retained_pid)
        ):
            return {"status": "refused_identity_mismatch", "restored": False}
        current.update(
            {
                "owner_pid": retained_pid,
                "owner_role": "retained_network_volume_watchdog",
                "heartbeat_at_epoch": time.time(),
                "canary_owner_returned_at_epoch": time.time(),
            }
        )
        _write_lease(path, current)
    return {"status": "restored", "restored": True, "owner_pid": retained_pid}


class PaidProviderLaneLeaseSet:
    """Acquire/release one mutation lease for every provider in a paid race.

    The set aggregates terminal state: if any contender remains billable or
    unverified, every mutated lease is retained so another provider cannot
    enter the same logical lane alongside the residual allocation.
    """

    def __init__(
        self,
        *,
        providers: Mapping[str, Any],
        lane: str,
        job_dir: str,
        resource_name_prefix: str,
    ) -> None:
        self.providers = dict(providers)
        self.lane = str(lane)
        self.job_dir = str(job_dir)
        self.resource_name_prefix = str(resource_name_prefix)
        self.acquisitions: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {"status": "acquired", "leases": []}

    def _reconciliation(self, provider_name: str) -> dict[str, Any]:
        provider = self.providers[provider_name]
        try:
            inventory = provider.billable_inventory(
                name_prefix=self.resource_name_prefix
            )
        except Exception as exc:  # noqa: BLE001 - unavailable proof blocks
            inventory = {
                "status": "blocked",
                "api_confirmed": False,
                "live_resource_count": None,
                "resources": [],
                "blockers": ["paid_provider_lane_inventory_query_failed"],
                "error_type": type(exc).__name__,
                "raw_provider_response_recorded": False,
            }
        return build_paid_provider_lane_reconciliation(
            provider=provider_name,
            lane=self.lane,
            provider_inventory=inventory,
            open_pending_teardowns=load_pending_teardowns(),
        )

    def acquire(self) -> dict[str, Any]:
        for provider_name in self.providers:
            acquired = acquire_paid_provider_lane_lease(
                provider=provider_name,
                lane=self.lane,
                job_dir=self.job_dir,
                reconciliation=self._reconciliation(provider_name),
            )
            self.summary["leases"].append(
                {
                    key: acquired.get(key)
                    for key in (
                        "status",
                        "path",
                        "blockers",
                        "holder",
                        "stale_reason",
                        "reconciliation",
                    )
                    if key in acquired
                }
            )
            if acquired.get("status") != "acquired":
                for prior in self.acquisitions:
                    release_paid_provider_lane_lease(
                        prior,
                        reason="sibling_lane_lease_acquisition_blocked",
                        provider_mutation_started=False,
                    )
                self.summary["status"] = "blocked"
                self.summary["blockers"] = list(acquired.get("blockers") or [])
                return self.summary
            self.acquisitions.append(acquired)
        return self.summary

    def release(self, reason: str, *, provider_mutation_started: bool) -> dict[str, Any]:
        terminal = {
            provider_name: self._reconciliation(provider_name)
            for provider_name in self.providers
        }
        all_terminal = all(item.get("status") == "passed" for item in terminal.values())
        release = {
            "reason": reason,
            "provider_mutation_started": provider_mutation_started,
            "all_providers_terminal": all_terminal,
            "terminal_reconciliations": terminal,
            "results": [
                release_paid_provider_lane_lease(
                    acquired,
                    reason=reason,
                    provider_mutation_started=provider_mutation_started,
                    terminal_reconciliation=(
                        terminal.get(
                            str((acquired.get("lease") or {}).get("provider") or "")
                        )
                        if all_terminal
                        else None
                    ),
                )
                for acquired in self.acquisitions
            ],
        }
        self.summary["release"] = release
        return release
