"""Lease remote CPU attempts by worker identity and hold capacity until provider zero (plan 14 §7).

One lease per queue row lives at ``<root>/leases/<job_id>.json``, guarded by a non-blocking
``flock`` on ``<job_id>.lock``, with a ``<root>/live/<job_id>`` marker until it is terminal.
The attempt number is the fencing generation: a heartbeat, a receipt or a transition counts
only under the current attempt id and the execution recorded as that attempt's worker identity.

A lease is live while it is dispatching, dispatched or running and ``now`` is before both its
hard deadline and its last heartbeat plus the stale window.  No host PID is involved, so a
dispatcher restart never changes liveness.  Capacity counts every attempt that may have
started an execution until its provider zero is proven, live or not, and a lease turns
terminal only once every such attempt is provider-zero.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_job_contract import (
    MAX_ATTEMPTS_CAP,
    RemoteCpuContractError,
    execution_name_of,
    record_bytes,
    safe_label,
    validate_descriptor,
    validate_heartbeat,
    worker_identity_for,
)
from .remote_cpu_job_records import compute_zero_proven, fsync_directory, validate_teardown

LEASE_SCHEMA_VERSION = "remote_cpu_job_lease.v1"
HARD_DEADLINE_GRACE_SECONDS = 120
LIVE_STATES = frozenset({"dispatching", "dispatched", "running"})
TERMINAL_STATES = frozenset({"completed", "blocked", "fallback_host", "shadow_compared", "abandoned_dispatch"})
# ``expired`` closes an attempt; the next attempt re-enters ``claimed`` through ``claim_handoff``.
TRANSITIONS: Mapping[str, frozenset[str]] = {
    "claimed": frozenset({"awaiting_capacity", "dispatching", "fallback_host"}),
    "awaiting_capacity": frozenset({"dispatching", "fallback_host"}),
    "dispatching": frozenset({"dispatched", "expired", "abandoned_dispatch"}),
    "dispatched": frozenset({"running", "collecting", "expired"}),
    "running": frozenset({"collecting", "expired"}),
    "collecting": frozenset({"completed", "blocked", "shadow_compared", "fallback_host", "expired"}),
    "expired": frozenset({"fallback_host", "abandoned_dispatch"}),
}
STATES = frozenset(TRANSITIONS) | TERMINAL_STATES
UPDATES = frozenset({"worker_identity", "transport_object", "transport_generation", "compute_zero", "teardown", "outcome"})
_ATTEMPT_FIELDS = (
    "attempt", "attempt_id", "descriptor_digest", "limits", "dispatch_started", "worker_identity",
    "transport_object", "transport_generation", "compute_zero_proven", "provider_zero_proven",
    "teardown_digest", "outcome",
)
_LEASE_KEYS = frozenset({
    *_ATTEMPT_FIELDS, "schema_version", "job_id", "stage", "queue_row", "execution", "transport_bucket", "state",
    "deadlines", "heartbeat", "lease_expires_at_epoch", "prior_attempts", "transitions", "lease_digest",
})
_JOB_ID = re.compile(r"rcj-[a-z]{2}-[0-9a-f]{24}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
_OUTCOME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,255}")
_MAX_LEASE_BYTES = 256 * 1024


class RemoteCpuLeaseError(RemoteCpuContractError):
    """A lease operation was refused: locked, fenced, out of order, or unproven."""


def hard_deadline_epoch(dispatch_started_at_epoch: float, limits: Mapping[str, Any]) -> float:
    """Dispatch + start allowance + task timeout + grace; every presigned URL expires here too."""

    return (float(dispatch_started_at_epoch) + limits["start_allowance_seconds"] + limits["task_timeout_seconds"]
            + HARD_DEADLINE_GRACE_SECONDS)


def _paths(root: str | Path, job_id: str) -> tuple[Path, Path, Path]:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise RemoteCpuLeaseError("remote_cpu_lease_job_id_invalid")
    base = Path(root)
    return base / "leases" / f"{job_id}.json", base / "leases" / f"{job_id}.lock", base / "live" / job_id


@contextmanager
def _locked(root: str | Path, job_id: str) -> Iterator[tuple[Path, Path]]:
    """Hold ``<job_id>.lock`` exclusively; a second holder is refused rather than queued."""

    lease_path, lock_path, marker = _paths(root, job_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(lock_path, flags, 0o640)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RemoteCpuLeaseError(f"remote_cpu_lease_locked:{job_id}") from exc
        yield lease_path, marker
    finally:
        os.close(descriptor)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RemoteCpuLeaseError(f"remote_cpu_lease_unreadable:{path.name}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        payload = stream.read(_MAX_LEASE_BYTES + 1) if stat.S_ISREG(os.fstat(stream.fileno()).st_mode) else b""
    try:
        lease = json.loads(payload)
    except ValueError:
        lease = None
    if (not isinstance(lease, dict) or set(lease) != _LEASE_KEYS or len(payload) > _MAX_LEASE_BYTES
            or lease["schema_version"] != LEASE_SCHEMA_VERSION or lease["state"] not in STATES
            or lease["lease_digest"] != canonical_digest(lease, digest_field="lease_digest")):
        raise RemoteCpuLeaseError(f"remote_cpu_lease_unreadable:{path.name}")
    return lease


def _write(path: Path, marker: Path, lease: dict[str, Any]) -> dict[str, Any]:
    """Atomically replace the lease; the record guard refuses transports, URLs and credentials."""

    lease["lease_digest"] = canonical_digest(lease, digest_field="lease_digest")
    payload = record_bytes(lease)
    if lease["state"] not in TERMINAL_STATES:
        # The marker precedes the lease, so a lease is never live without its marker.
        marker.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        with suppress(FileExistsError):
            os.close(os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o640))
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "xb") as stream:
            os.fchmod(stream.fileno(), 0o640)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)
    if lease["state"] in TERMINAL_STATES:
        marker.unlink(missing_ok=True)
    return lease


def _attempt(descriptor: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """The fields each attempt starts afresh, bound to its own validated descriptor and config."""

    limits, execution = descriptor["limits"], descriptor["execution"]
    return {
        "execution": {name: execution[name] for name in ("project", "region", "job")},
        "transport_bucket": config["transport_bucket"],
        "attempt": descriptor["attempt"], "attempt_id": descriptor["attempt_id"],
        "descriptor_digest": descriptor["descriptor_digest"],
        "limits": {name: limits[name] for name in (
            "start_allowance_seconds", "task_timeout_seconds", "heartbeat_stale_seconds")},
        "dispatch_started": False, "worker_identity": None, "transport_object": None, "transport_generation": None,
        "compute_zero_proven": False, "provider_zero_proven": False, "teardown_digest": None, "outcome": None,
        "state": "claimed", "deadlines": None, "heartbeat": None, "lease_expires_at_epoch": None,
    }


def claim_handoff(root: str | Path, *, descriptor: Mapping[str, Any], config: Mapping[str, Any],
                  now: float) -> dict[str, Any]:
    """Lease a hand-off for the descriptor's attempt.

    Attempt 1 creates the lease exclusively.  A later attempt re-claims the lease only when the
    previous attempt has expired and is compute-zero; that attempt moves to ``prior_attempts``
    and keeps its capacity slot until its provider zero is proven.  Re-claiming the same
    attempt returns the lease unchanged.
    """

    checked = validate_descriptor(descriptor, config=config)
    job_id = checked["job_id"]
    if not isinstance(config.get("transport_bucket"), str) or not _BUCKET.fullmatch(config["transport_bucket"]):
        raise RemoteCpuLeaseError("remote_cpu_lease_config_invalid:transport_bucket")
    with _locked(root, job_id) as (path, marker):
        current = _read(path)
        if current is not None and (current["attempt_id"], current["descriptor_digest"]) == (
                checked["attempt_id"], checked["descriptor_digest"]):
            return current
        if current is None and checked["attempt"] == 1:
            lease = {
                "schema_version": LEASE_SCHEMA_VERSION, "job_id": job_id, "stage": checked["stage"],
                "queue_row": checked["queue_row"], **_attempt(checked, config), "prior_attempts": [],
                "transitions": [], "lease_digest": "",
            }
        elif current is None:
            raise RemoteCpuLeaseError("remote_cpu_lease_prior_attempt_missing")
        elif checked["attempt"] != current["attempt"] + 1:
            raise RemoteCpuLeaseError(f"remote_cpu_lease_exists:{job_id}")
        elif current["state"] != "expired":
            raise RemoteCpuLeaseError(f"remote_cpu_lease_transition_refused:{current['state']}->claimed")
        elif current["dispatch_started"] and not current["compute_zero_proven"]:
            raise RemoteCpuLeaseError("remote_cpu_lease_prior_attempt_not_compute_zero")
        else:
            prior = {name: current[name] for name in _ATTEMPT_FIELDS}
            lease = {**current, **_attempt(checked, config), "prior_attempts": [*current["prior_attempts"], prior]}
        lease["transitions"] = [*lease["transitions"], {
            "from": None if current is None else current["state"], "to": "claimed",
            "attempt": checked["attempt"], "at_epoch": float(now)}]
        return _write(path, marker, lease)


def _apply_updates(lease: dict[str, Any], updates: Mapping[str, Any]) -> None:
    unknown = sorted(set(updates) - UPDATES, key=str)
    if unknown:
        raise RemoteCpuLeaseError(f"remote_cpu_lease_update_invalid:{safe_label(unknown[0])}")
    if "worker_identity" in updates:
        identity = updates["worker_identity"]
        if lease["worker_identity"] not in {None, identity}:
            raise RemoteCpuLeaseError("remote_cpu_lease_worker_identity_immutable")
        if identity != worker_identity_for(lease["execution"], execution_name_of(identity)):
            raise RemoteCpuLeaseError("remote_cpu_lease_worker_identity_unbound")
        lease["worker_identity"] = identity
    if {"transport_object", "transport_generation"} & set(updates):
        name, generation = updates.get("transport_object"), updates.get("transport_generation")
        pattern = (rf"gs://{re.escape(lease['transport_bucket'])}/transport/{lease['job_id']}/"
                   rf"{re.escape(lease['attempt_id'])}-[0-9a-f]{{32}}\.json")
        if (not isinstance(name, str) or not re.fullmatch(pattern, name) or not isinstance(generation, int)
                or isinstance(generation, bool) or generation < 1):
            raise RemoteCpuLeaseError("remote_cpu_lease_transport_invalid")
        if (lease["transport_object"], lease["transport_generation"]) not in {(None, None), (name, generation)}:
            raise RemoteCpuLeaseError("remote_cpu_lease_transport_immutable")
        lease["transport_object"], lease["transport_generation"] = name, generation
    if "compute_zero" in updates:
        if not compute_zero_proven(updates["compute_zero"], worker_identity=lease["worker_identity"]):
            raise RemoteCpuLeaseError("remote_cpu_lease_compute_zero_unproven")
        lease["compute_zero_proven"] = True
    if "teardown" in updates:
        record = validate_teardown(updates["teardown"])
        targets = [attempt for attempt in [*lease["prior_attempts"], lease]
                   if (attempt["attempt_id"], attempt["descriptor_digest"], attempt["worker_identity"])
                   == (record["attempt_id"], record["descriptor_digest"], record["worker_identity"])]
        if len(targets) != 1 or record["job_id"] != lease["job_id"]:
            raise RemoteCpuLeaseError("remote_cpu_lease_teardown_unbound")
        target = targets[0]
        target["compute_zero_proven"] = target["compute_zero_proven"] or record["compute_zero_proven"]
        if record["provider_zero_proven"]:
            target["provider_zero_proven"], target["teardown_digest"] = True, record["teardown_digest"]
    if "outcome" in updates:
        if not isinstance(updates["outcome"], str) or not _OUTCOME.fullmatch(updates["outcome"]):
            raise RemoteCpuLeaseError("remote_cpu_lease_outcome_invalid")
        lease["outcome"] = updates["outcome"]


def _enter(lease: dict[str, Any], state: str, now: float) -> None:
    if state == "dispatching":
        limits, started = lease["limits"], float(now)
        hard = hard_deadline_epoch(started, limits)
        start_by = started + limits["start_allowance_seconds"]
        lease["dispatch_started"] = True
        lease["deadlines"] = {"dispatch_started_at_epoch": started, "start_by_epoch": start_by,
                              "hard_deadline_epoch": hard}
        lease["lease_expires_at_epoch"] = min(start_by, hard)
    if state in {"dispatched", "running", "collecting"} and lease["worker_identity"] is None:
        raise RemoteCpuLeaseError("remote_cpu_lease_worker_identity_missing")
    if state == "dispatched" and lease["transport_object"] is None:
        # Compute-zero needs the transport deleted at its generation, so it must be on record.
        raise RemoteCpuLeaseError("remote_cpu_lease_transport_missing")
    if state == "running" and lease["heartbeat"] is None:
        raise RemoteCpuLeaseError("remote_cpu_lease_heartbeat_missing")
    if (state == "expired" or state in TERMINAL_STATES) and not lease["outcome"]:
        raise RemoteCpuLeaseError("remote_cpu_lease_outcome_missing")
    if state in TERMINAL_STATES and any(attempt["dispatch_started"] and not attempt["provider_zero_proven"]
                                        for attempt in [*lease["prior_attempts"], lease]):
        raise RemoteCpuLeaseError("remote_cpu_lease_provider_zero_unproven")
    lease["transitions"].append({"from": lease["state"], "to": state, "attempt": lease["attempt"],
                                 "at_epoch": float(now)})
    lease["state"] = state


def transition(root: str | Path, job_id: str, *, attempt_id: str, to_state: str | None, now: float,
               updates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Move the current attempt to ``to_state`` (or only record ``updates`` when it is None).

    The caller names the attempt it believes is current; any other attempt id is fenced.
    Updates: ``worker_identity`` and the transport name and generation (each set once),
    ``compute_zero`` evidence, a sealed ``teardown`` for the current or a prior attempt, and
    an ``outcome``.  A terminal state requires provider zero for every started attempt.
    """

    with _locked(root, job_id) as (path, marker):
        lease = _read(path)
        if lease is None:
            raise RemoteCpuLeaseError(f"remote_cpu_lease_missing:{job_id}")
        if attempt_id != lease["attempt_id"]:
            raise RemoteCpuLeaseError("remote_cpu_lease_attempt_fenced")
        state = lease["state"]
        if state in TERMINAL_STATES or (to_state not in {None, state} and to_state not in TRANSITIONS[state]):
            raise RemoteCpuLeaseError(f"remote_cpu_lease_transition_refused:{state}->{to_state or state}")
        _apply_updates(lease, updates or {})
        if to_state not in {None, state}:
            _enter(lease, to_state, now)
        return _write(path, marker, lease)


def observe_heartbeat(root: str | Path, job_id: str, heartbeat: Mapping[str, Any], *, execution_running: bool,
                      now: float) -> dict[str, Any]:
    """Renew a live attempt when its heartbeat sequence advances while the execution runs.

    A heartbeat for another attempt or execution, for a lease that is not dispatched or running,
    or for a lease already past its expiry is fenced; it never revives an attempt.
    """

    with _locked(root, job_id) as (path, marker):
        lease = _read(path)
        if lease is None:
            raise RemoteCpuLeaseError(f"remote_cpu_lease_missing:{job_id}")
        name = execution_name_of(lease["worker_identity"]) if lease["worker_identity"] else ""
        checked = validate_heartbeat(heartbeat, attempt_id=lease["attempt_id"], execution_name=name)
        if lease["state"] not in {"dispatched", "running"}:
            raise RemoteCpuLeaseError(f"remote_cpu_lease_heartbeat_fenced:state_{lease['state']}")
        if float(now) >= lease["lease_expires_at_epoch"]:
            raise RemoteCpuLeaseError("remote_cpu_lease_heartbeat_fenced:stale")
        if execution_running is not True or checked["sequence"] <= (lease["heartbeat"] or {}).get("sequence", 0):
            return {"renewed": False, "state": lease["state"], "lease_expires_at_epoch": lease["lease_expires_at_epoch"]}
        lease["heartbeat"] = {
            **{field: checked[field] for field in ("sequence", "phase", "elapsed_seconds", "bytes_fetched",
                                                   "bytes_uploaded")},
            "observed_at_epoch": float(now),
        }
        lease["lease_expires_at_epoch"] = min(lease["deadlines"]["hard_deadline_epoch"],
                                              float(now) + lease["limits"]["heartbeat_stale_seconds"])
        if lease["state"] == "dispatched":
            _enter(lease, "running", now)
        _write(path, marker, lease)
        return {"renewed": True, "state": lease["state"], "lease_expires_at_epoch": lease["lease_expires_at_epoch"]}


def _marked(root: str | Path) -> Iterator[tuple[str, dict[str, Any] | None, bool]]:
    """Every lease with a live marker, as ``(job_id, lease or None, unreadable)``."""

    directory = Path(root) / "live"
    if not directory.is_dir():
        return
    for marker in sorted(directory.iterdir()):
        try:
            yield marker.name, _read(_paths(root, marker.name)[0]), False
        except RemoteCpuContractError:
            yield marker.name, None, True


def live_leases(root: str | Path, *, now: float) -> list[dict[str, Any]]:
    """Leases whose attempt is live: dispatching, dispatched or running, and not yet expired."""

    return [lease for _, lease, _ in _marked(root)
            if lease is not None and lease["state"] in LIVE_STATES and float(now) < lease["lease_expires_at_epoch"]]


def slots_in_use(root: str | Path) -> int:
    """Attempts that may have started an execution and are not yet provider-zero, live or not.

    An unreadable lease is counted as holding every attempt it could have started.
    """

    total = 0
    for _, lease, unreadable in _marked(root):
        if unreadable:
            total += MAX_ATTEMPTS_CAP
        elif lease is not None:
            total += sum(1 for attempt in [*lease["prior_attempts"], lease]
                         if attempt["dispatch_started"] and not attempt["provider_zero_proven"])
    return total


def expire_stale(root: str | Path, *, now: float) -> list[str]:
    """Close every live-state attempt past its expiry; a locked lease is left to its holder."""

    expired: list[str] = []
    for job_id, snapshot, _ in _marked(root):
        if snapshot is None or snapshot["state"] not in LIVE_STATES or float(now) < snapshot["lease_expires_at_epoch"]:
            continue
        try:
            with _locked(root, job_id) as (path, marker):
                lease = _read(path)
                if lease is None or lease["state"] not in LIVE_STATES or float(now) < lease["lease_expires_at_epoch"]:
                    continue
                if float(now) >= lease["deadlines"]["hard_deadline_epoch"]:
                    lease["outcome"] = "hard_deadline_passed"
                else:
                    lease["outcome"] = "heartbeat_stale" if lease["state"] == "running" else "start_timeout"
                _enter(lease, "expired", now)
                _write(path, marker, lease)
                expired.append(lease["attempt_id"])
        except RemoteCpuLeaseError as exc:
            if not str(exc).startswith("remote_cpu_lease_locked:"):
                raise
    return expired
