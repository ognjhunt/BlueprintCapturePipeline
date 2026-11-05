"""Retire a launch-activation claim whose worker died, so it stops holding disk.

The activation worker (``process_launch_activation_queue``) claims an envelope
by moving it from ``pending/`` to ``processing/`` and holds an exclusive
``flock`` on it until the envelope has left ``processing/`` for ``prepared/`` or
``blocked/`` beside its sealed result in ``results/``. Every later run claims
only from ``pending/``. So a worker killed mid-claim (the unit's start timeout,
a restart, OOM) leaves an unlocked envelope in ``processing/`` forever, and the
readers that count pending and processing rows as live keep everything it
names: the storage GC keeps the activation's directory as ``queue_referenced``
and release retention keeps its release. On 2026-10-31 the then one-hour
``TimeoutStartSec`` killed capture-447's scene-configuration activation this
way, and its 14.8 GB directory stayed on the work volume.

A ``processing/*.json`` row is abandoned only when all of these hold:

* it is a regular file, not a link, of at most ``MAX_QUEUE_MESSAGE_BYTES``;
* it was claimed more than ``ABANDONED_AFTER_SECONDS`` ago, by both its change
  and its modification time: the activation unit's ``TimeoutStartSec`` plus a
  margin for systemd's stop timeout, after which no worker the unit started can
  still be running;
* its ``flock`` is free when taken without waiting: no live process holds it;
* it is a digest-valid activation envelope named for its own request, and no
  other queue state holds the same name;
* no sealed result exists for it.

Holding the lock, recovery seals a ``blocked`` result in the worker's own
schema (``INTERRUPTED_CLAIM_BLOCKER``; ``source_commit`` the release the claim
was made under; the recovering release under ``claim_recovery``) and moves the
envelope, byte for byte, to ``blocked/``. When a sealed result already exists
(the worker died between its result and its move), the envelope is moved where
that result says instead. The result always comes first and is written once,
so a recovery that dies between the two leaves a result the next pass only has
to finish moving. A worker killed between the two steps of its claim leaves an
empty placeholder in ``processing/`` that keeps its still pending row from ever
being claimed; an abandoned placeholder is removed.

A young or locked row is never touched. Nothing here re-runs a request, moves a
row back to ``pending/``, or touches pins, profiles, catalogs or authorizations.

Recovery runs at the start of every activation worker run, as the service
account that owns the queue; it refuses to act as root. The worker's path unit
starts it when a ``pending/`` row appears, and its hourly timer
(``blueprint-task-evaluation-launch-activation.timer``) starts it on an idle
queue too. The storage GC then reclaims what the retired claim kept on its
next pass.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .control_plane_storage_references import MAX_QUEUE_MESSAGE_BYTES
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_activation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    QUEUE_STATES,
    RESULT_SCHEMA_VERSION,
    _queue_filename,
)
from .task_evaluation_launch_preparation_storage import write_launch_preparation_record_exclusive

RECOVERY_SCHEMA_VERSION = "task_evaluation_launch_activation_claim_recovery.v1"
INTERRUPTED_CLAIM_BLOCKER = "launch_activation_claim_interrupted:worker_process_terminated"
#: ``TimeoutStartSec`` of blueprint-task-evaluation-launch-activation.service, which a test
#: reads; no worker run the unit starts outlives it.
ACTIVATION_UNIT_TIMEOUT_SECONDS = 255 * 60
#: After the start timeout's SIGTERM, systemd waits ``TimeoutStopSec`` (90 s by default)
#: before SIGKILL; the rest of the margin absorbs clock steps.
STOP_MARGIN_SECONDS = 15 * 60
ABANDONED_AFTER_SECONDS = ACTIVATION_UNIT_TIMEOUT_SECONDS + STOP_MARGIN_SECONDS
RECOVERED_BY = "task_evaluation_launch_activation_worker"
#: What the worker seals for an activation it prepared (``control_plane_pin_proofs``).
PREPARED_STATUSES = frozenset({
    "policy_campaign_queue_materialized_no_execution",
    "profile_authority_materialized_no_execution",
})
_COMMIT = re.compile(r"[0-9a-f]{40}")
_clock: Callable[[], float] = time.time


def retire_abandoned_activation_claims(
    queue_root: str | Path,
    *,
    recovering_source_commit: str | None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Recover every abandoned ``processing/`` row of one activation queue.

    Returns one entry per row looked at: ``retired``, ``finished`` (with its
    ``terminal_state``), ``placeholder_removed``, ``kept`` (with its ``reason``)
    or ``failed`` (with an ``error`` type). A row that fails stays claimed and
    never stops another, nor the worker run.
    """

    queue = Path(queue_root)
    moment = _clock() if now is None else float(now)
    commit = recovering_source_commit if _COMMIT.fullmatch(str(recovering_source_commit or "")) else None
    try:
        with os.scandir(queue / "processing") as entries:
            names = sorted(entry.name for entry in entries
                           if entry.name.endswith(".json") and not entry.name.startswith("."))
    except FileNotFoundError:
        return []
    if os.geteuid() == 0:
        # The queue belongs to the service account: root never writes into it.
        return [{"name": name, **_kept("running_as_root")} for name in names]
    rows: list[dict[str, Any]] = []
    for name in names:
        try:
            row = _recover(queue, name, moment=moment, commit=commit)
        except Exception as exc:  # noqa: BLE001 - one row never costs the others or the worker run
            row = {"action": "failed", "error": type(exc).__name__}
        rows.append({"name": name, **row})
    return rows


def _kept(reason: str) -> dict[str, Any]:
    return {"action": "kept", "reason": reason}


def _same_file(path: Path, info: os.stat_result) -> bool:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino)


def _recover(queue: Path, name: str, *, moment: float, commit: str | None) -> dict[str, Any]:
    claimed = queue / "processing" / name
    try:
        observed = claimed.lstat()
    except FileNotFoundError:  # finished by its worker since the listing
        return _kept("moved")
    if stat.S_ISLNK(observed.st_mode):
        return _kept("linked")
    if not stat.S_ISREG(observed.st_mode):
        return _kept("not_regular")
    if observed.st_size > MAX_QUEUE_MESSAGE_BYTES:
        return _kept("oversized")
    # The rename into processing/ is the claim, and sets the change time. Age is judged
    # before any lock is taken, so a claim a worker has just made is never contended.
    claimed_at = max(observed.st_mtime, observed.st_ctime)
    if moment - claimed_at <= ABANDONED_AFTER_SECONDS:
        return _kept("young")
    try:
        descriptor = os.open(claimed, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return _kept("moved")
    try:
        held = os.fstat(descriptor)
        if not _same_file(claimed, held) or (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
            return _kept("moved")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return _kept("lease_held")
        if not _same_file(claimed, held):  # moved out by whoever held the lock until now
            return _kept("moved")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            raw = stream.read(MAX_QUEUE_MESSAGE_BYTES + 1)
        if len(raw) > MAX_QUEUE_MESSAGE_BYTES:
            return _kept("oversized")
        return _settle(queue, name, raw, claimed_at=claimed_at, moment=moment, commit=commit)
    finally:
        os.close(descriptor)  # the lock is released only after the move


def _settle(queue: Path, name: str, raw: bytes, *, claimed_at: float, moment: float,
            commit: str | None) -> dict[str, Any]:
    claimed = queue / "processing" / name
    if not raw and os.path.lexists(queue / "pending" / name):
        claimed.unlink()
        return {"action": "placeholder_removed"}
    envelope = _envelope(raw, name)
    if envelope is None:
        return _kept("envelope_invalid")
    if any(os.path.lexists(queue / state / name) for state in QUEUE_STATES if state != "processing"):
        return _kept("identity_ambiguous")
    activation_id = envelope["request"]["activation_id"]
    result_path = queue / "results" / name
    if os.path.lexists(result_path):
        return _finish(queue, name, activation_id)
    result = _blocked_result(envelope, raw, name, claimed_at=claimed_at, moment=moment, commit=commit)
    try:
        write_launch_preparation_record_exclusive(result_path, result)
    except FileExistsError:
        return _finish(queue, name, activation_id)
    os.replace(claimed, queue / "blocked" / name)
    return {"action": "retired", "activation_id": activation_id, "result_digest": result["result_digest"]}


def _envelope(raw: bytes, name: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    request = value.get("request") if isinstance(value, dict) else None
    if (
        not isinstance(request, dict)
        or value.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or value.get("envelope_digest") != canonical_digest(value, digest_field="envelope_digest")
        or value.get("request_digest") != canonical_digest(request)
        or not isinstance(request.get("activation_id"), str)
        or not _COMMIT.fullmatch(str(request.get("expected_production_commit") or ""))
        or _queue_filename(activation_id=request["activation_id"], request_digest=value["request_digest"]) != name
    ):
        return None
    return value


def _finish(queue: Path, name: str, activation_id: str) -> dict[str, Any]:
    """Move the claim where the result its worker sealed says, as the worker would have."""

    path = queue / "results" / name
    try:
        info = path.lstat()
        value = (json.loads(path.read_bytes())
                 if stat.S_ISREG(info.st_mode) and info.st_size <= MAX_QUEUE_MESSAGE_BYTES else None)
    except (OSError, ValueError, RecursionError):
        value = None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != RESULT_SCHEMA_VERSION
        or value.get("activation_id") != activation_id
        or value.get("result_digest") != canonical_digest(value, digest_field="result_digest")
    ):
        return _kept("result_invalid")
    status = value.get("status")
    if status != "blocked" and status not in PREPARED_STATUSES:
        return _kept("result_status_unknown")
    if value.get("run_kind") == "internal_policy_canary" and status == "policy_campaign_queue_materialized_no_execution":
        # The worker writes this campaign's dispatch envelope between its result and its
        # move; nothing here can prove that write happened, so the claim stays for review.
        return _kept("policy_canary_dispatch_unverified")
    state = "blocked" if status == "blocked" else "prepared"
    os.replace(queue / "processing" / name, queue / state / name)
    return {"action": "finished", "terminal_state": state, "activation_id": activation_id}


def _blocked_result(envelope: Mapping[str, Any], raw: bytes, name: str, *, claimed_at: float, moment: float,
                    commit: str | None) -> dict[str, Any]:
    request = envelope["request"]
    recovery = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "envelope_name": name,
        "request_digest": envelope["request_digest"],
        "envelope_digest": envelope["envelope_digest"],
        "envelope_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "envelope_size_bytes": len(raw),
        "claimed_at_epoch": claimed_at,
        "abandoned_after_seconds": ABANDONED_AFTER_SECONDS,
        "recovered_at_epoch": moment,
        "claim_source_commit": request["expected_production_commit"],
        "recovered_by": RECOVERED_BY,
        "recovered_by_source_commit": commit,
        "worker_lease_held": False,
        "request_re_executed": False,
        "envelope_bytes_modified": False,
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "activation_id": request["activation_id"],
        "blockers": [INTERRUPTED_CLAIM_BLOCKER],
        # The release the claim was made under, which the worker binds a result to.
        "source_commit": request["expected_production_commit"],
        "claim_recovery": recovery,
        # The killed worker may have started the preparation command, as the worker's own
        # blocked result says of any failure.
        "catalog_mutation_state": "unknown_if_preparation_started",
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "observed_at_iso": datetime.fromtimestamp(moment, timezone.utc).isoformat(),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


__all__ = [
    "ABANDONED_AFTER_SECONDS",
    "ACTIVATION_UNIT_TIMEOUT_SECONDS",
    "INTERRUPTED_CLAIM_BLOCKER",
    "PREPARED_STATUSES",
    "RECOVERED_BY",
    "RECOVERY_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "STOP_MARGIN_SECONDS",
    "retire_abandoned_activation_claims",
]
