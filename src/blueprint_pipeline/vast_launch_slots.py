"""Vast paid-launch concurrency: N slot locks and the deploy gate.

Extracted from `vast_provider_adapter`, which re-exports every name here, so
lanes and tests that reach these through the adapter keep working.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .common import ensure_dir, utc_now_iso, write_json


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


#: How many paid launches may hold a provider at once, fleet-wide.
#:
#: This is a spend policy, not a technical limit. It was 1 for a long time and
#: that made every lane queue behind the slowest run in flight -- a Content
#: Agents run held the provider for 78 minutes on 2026-08-13 while three other
#: lanes waited. Raised to 3 on explicit authorization the same day.
#:
#: What it does NOT change: each attempt still carries its own hard cap, TTL,
#: and watchdog, so the worst case is N times one attempt's ceiling rather than
#: an unbounded fleet. And a run still proves teardown from its own receipt.
#: Fleet-wide provider zero simply becomes provable between batches rather than
#: after every run, which is where the reconciler already looks for it.
DEFAULT_MAX_CONCURRENT_PAID_LAUNCHES = 3
MAX_CONCURRENT_PAID_LAUNCHES_ENV = "BLUEPRINT_VAST_MAX_CONCURRENT_PAID_LAUNCHES"


def _max_concurrent_paid_launches() -> int:
    raw = _string(os.environ.get(MAX_CONCURRENT_PAID_LAUNCHES_ENV))
    if not raw:
        return DEFAULT_MAX_CONCURRENT_PAID_LAUNCHES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_PAID_LAUNCHES
    # Never widen past the compiled policy from an environment variable, and
    # never fall below one: an env typo must not silently authorize more
    # concurrent spend, nor deadlock every lane.
    return max(1, min(value, DEFAULT_MAX_CONCURRENT_PAID_LAUNCHES))


def vast_launch_lock_paths(lock_path: Path | None = None) -> list[Path]:
    """One path per concurrency slot.

    Slot 0 keeps the historical filename, so a host, a reaper, or an operator
    that knows only `vast_paid_launch.lock` still sees a real lock rather than
    nothing.
    """

    base = lock_path or _vast_launch_lock_path()
    paths = [base]
    for slot in range(1, _max_concurrent_paid_launches()):
        paths.append(base.with_name(f"{base.stem}.slot{slot}{base.suffix}"))
    return paths


def vast_launch_gate_path(lock_path: Path | None = None) -> Path:
    """The deploy gate beside the slots.

    A launch holds it shared only while it takes a slot; a deploy holds it
    exclusively for the whole release swap. So no launch can start during a
    deploy, while launches that started before it keep their slots and run on
    their own immutable release tree.
    """

    base = lock_path or _vast_launch_lock_path()
    return base.with_name(f"{base.stem}.gate{base.suffix}")


def _vast_launch_lock_path() -> Path:
    from .vast_provider_adapter import (
        DEFAULT_VAST_API_KEY_FILE,
        DEFAULT_VAST_LAUNCH_LOCK_FILENAME,
        VAST_API_KEY_FILE_ENV,
        VAST_LAUNCH_LOCK_FILE_ENV,
    )

    configured = _string(os.environ.get(VAST_LAUNCH_LOCK_FILE_ENV))
    if configured:
        return Path(configured).expanduser().resolve()
    api_key_path = Path(
        os.environ.get(VAST_API_KEY_FILE_ENV, DEFAULT_VAST_API_KEY_FILE)
    ).expanduser()
    return (api_key_path.parent / DEFAULT_VAST_LAUNCH_LOCK_FILENAME).resolve()


def _try_acquire_vast_launch_lock(
    *,
    job_dir: Path,
    generated_at: str,
    lock_path: Path | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    slots = vast_launch_lock_paths(lock_path)
    handle = None
    held_path: Path | None = None
    last_holder = ""
    unusable: list[str] = []
    gate_path = vast_launch_gate_path(lock_path)
    gate = None
    gate_closed_for_deploy = False
    ensure_dir(gate_path.parent)
    try:
        gate = gate_path.open("a+", encoding="utf-8")
        gate_path.chmod(0o600)
    except OSError as exc:
        if gate is not None:
            gate.close()
            gate = None
        unusable.append(f"{gate_path.name}:{type(exc).__name__}")
    if gate is not None:
        try:
            fcntl.flock(gate.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            gate.close()
            gate = None
            gate_closed_for_deploy = True
    for candidate in slots if gate is not None else ():
        ensure_dir(candidate.parent)
        # A slot the launching account cannot open is a provisioning fault, not
        # a busy slot, and no amount of waiting clears it. Production reached
        # this state when a tool run as root created `slot1`/`slot2` owned
        # `root:root` at 0644 while the adapter runs as `blueprint`. Both calls
        # sat outside the `try:` below, which catches only `BlockingIOError`,
        # so the `PermissionError` escaped as an unhandled traceback at the
        # money boundary -- and only when slot 0 was already held, because slot
        # 0 is tried first and is usually fine.
        try:
            attempt = candidate.open("a+", encoding="utf-8")
        except OSError as exc:
            unusable.append(f"{candidate.name}:{type(exc).__name__}")
            continue
        try:
            candidate.chmod(0o600)
        except OSError as exc:
            unusable.append(f"{candidate.name}:{type(exc).__name__}")
            attempt.close()
            continue
        try:
            fcntl.flock(attempt.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            attempt.seek(0)
            last_holder = attempt.read()[:1000]
            attempt.close()
            continue
        handle = attempt
        held_path = candidate
        break
    if gate is not None:
        # The slot is the run's for its whole life; the gate only spans taking it.
        fcntl.flock(gate.fileno(), fcntl.LOCK_UN)
        gate.close()
    if handle is None or held_path is None:
        # Two different refusals share this exit. "Busy" is the fleet at its
        # authorized concurrency and says nothing about whether a *particular*
        # run may proceed. "Unusable" is a host that cannot honour its own
        # semaphore, which an operator has to repair.
        every_slot_unusable = bool(unusable) and not gate_closed_for_deploy and (
            len(unusable) == len(slots) or any(item.startswith(gate_path.name) for item in unusable)
        )
        manifest = {
            "schema_version": "vast_launch_lock_manifest.v1",
            "generated_at": generated_at,
            "status": "blocked",
            "lock_path": str(slots[0]),
            "lock_slots": [str(item) for item in slots],
            "lock_acquired": False,
            "blockers": [
                "vast_paid_launch_lock_unusable"
                if every_slot_unusable
                else "vast_paid_launch_lock_busy"
            ],
            "existing_lock_record_prefix": last_holder,
            "unusable_lock_slots": unusable,
            # Busy for the length of a deploy, then retried like any busy slot.
            "gate_closed_for_deploy": gate_closed_for_deploy,
            "raw_secret_values_recorded": False,
        }
        write_json(job_dir / "vast_launch_lock_manifest.json", manifest)
        return None, manifest
    lock_path = held_path
    record = {
        "pid": os.getpid(),
        "job_dir": str(job_dir),
        "acquired_at": generated_at,
        "purpose": "vast_paid_instance_launch_single_flight_guard",
    }
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    manifest = {
        "schema_version": "vast_launch_lock_manifest.v1",
        "generated_at": generated_at,
        "status": "acquired",
        "lock_path": str(lock_path),
        "lock_slots": [str(item) for item in slots],
        "lock_acquired": True,
        "lock_record": record,
        "blockers": [],
        # Recorded on the success path too: a fleet silently running at lower
        # concurrency than it is authorized for is the failure this hides.
        "unusable_lock_slots": unusable,
        "raw_secret_values_recorded": False,
    }
    write_json(job_dir / "vast_launch_lock_manifest.json", manifest)
    return handle, manifest


def _release_vast_launch_lock(
    handle: Any | None,
    *,
    job_dir: Path | None = None,
    generated_at: str | None = None,
) -> dict[str, Any] | None:
    if handle is None:
        return None
    lock_path = Path(handle.name).expanduser()
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
    manifest = {
        "schema_version": "vast_launch_lock_manifest.v1",
        "generated_at": generated_at or utc_now_iso(),
        "status": "released",
        "lock_path": str(lock_path),
        "lock_released": True,
        "raw_secret_values_recorded": False,
    }
    if job_dir is not None:
        write_json(job_dir / "vast_launch_lock_manifest.json", manifest)
    return manifest
