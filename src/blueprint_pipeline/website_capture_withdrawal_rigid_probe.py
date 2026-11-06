"""The rigid robot-free probe copies of one withdrawn site's team scene halves, for withdrawal deletion.

Beta journey step: site withdrawal after the site's compiled half queued its robot-free probe
(``rigid_robot_free_probe_runner``). Each attempt of a probe job (``<queue>/jobs/<capture key>-*``)
keeps bytes derived from the half: the sealed bundle built from the half's native task packet, the
sealed probe input (the half's resolved scene plan) and the provider run's output archive with its
extraction (the probe's trace). Beside them are the records its website-grant hold is settled from
(``rigid_robot_free_probe_settlement``). The owner's rule (2026-11-06) is to delete them once settled,
the rule withdrawal applies to screening attempts (``website_capture_withdrawal_screening``).

A job's money is final when every attempt's is:

- ``settled``: the runner's settlement record (``rigid_robot_free_probe_spend_settlement.json`` in
  the attempt's queue directory) is there, sealed, and names the attempt. ``settle_attempt_spend``
  writes it once, after the WebApp settled the hold, and reads it as settled from then on;
- ``never_funded``: the allocator ended (its result is there) and no reservation of the website
  grant (``allocator/website_preparation_spend_admission.json``) was written in its job directory;
- ``never_launched``: no allocator ran from it (no launch intent, or a spawn failure with no
  process).

Anything else waits, by name: a reservation with no settlement (``settlement_pending``: not settled
yet, held because its teardown is unproven, or refused by the WebApp), an allocator still running or
of unknown fate, a settlement record that is unreadable, unsealed or names another attempt, an
unreadable attempt record, and a work directory that is not where the runner puts it, not mounted,
or not named by any attempt. A waiting job stays whole, is named in ``remaining``, keeps the deletion
incomplete and never blocks the rest of it.

For a job whose money is final, each attempt's derived bytes are planned with an inventory and its
digest: its work directory's ``bundle/``, every archive under its ``allocator/``, the extracted
output (``immutable_execution/``, ``vast_provider_runtime_output_videos/``) and the probe input.
Every other file stays: the settlement record and the reservation, the paid admission, the
allocator's results, the Vast adapter's teardown manifest, adapter result and budget ledgers, the
machine avoidlists, and the authority, ingest, terminal, preflight and artifact-manifest records.
Their writers put amounts, ids, statuses, paths and digests in them, and settlement replays from
them. A link, special file or unreadable entry in what would be deleted fails closed into
``unresolved``. Apply holds each job's runner lock without waiting (``rigid_probe_job_busy``),
plans the job again, and refuses unless it is exactly the reviewed one (``rigid_probe_job_changed``).
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .rigid_robot_free_probe_queue_root import (
    ATTEMPT_FILENAME,
    ATTEMPTS_DIRECTORY,
    FUNDING_FILENAME,
    INPUTS_FILENAME,
    INTENT_FILENAME,
    JOB_LOCK_FILENAME,
    JOBS_DIRECTORY,
    PROCESS_FILENAME,
    RESULT_FILENAME,
    SETTLEMENT_FILENAME,
    SETTLEMENT_SCHEMA,
    SPAWN_FAILED_FILENAME,
    WORK_DIR_FIELD,
)

_PREFIX = "website_withdrawal_deletion_"
# The derived bytes in an attempt's allocator job directory: the provider output archive (and any
# bundle copy) and what was extracted from it.
_EXTRACTED = ("immutable_execution", "vast_provider_runtime_output_videos")
_ARCHIVE_SUFFIX = ".zip"


class _Unsafe(Exception):
    """What would be deleted holds a link, a special file or an unreadable entry."""


def _unsafe(path: Path) -> bool:
    return any(part.is_symlink() for part in (path, *path.parents))


def _record(path: Path) -> dict[str, Any] | None:
    """A regular JSON object, or None when it cannot be read as one."""
    try:
        if path.is_symlink() or not stat.S_ISREG(os.lstat(path).st_mode):
            return None
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _settled(path: Path, attempt: str) -> bool:
    record = _record(path)
    try:
        return (record is not None and record.get("schema_version") == SETTLEMENT_SCHEMA
                and record.get("status") == "settled" and record.get("attempt") == attempt
                and record.get("record_digest") == canonical_digest(record, digest_field="record_digest"))
    except (TypeError, ValueError):
        return False


def _work_dir(attempt: Path, job: Path) -> tuple[Path | None, str | None]:
    """The directory the attempt's record says its bulk is in (itself without one), or why it cannot be told."""
    path = attempt / ATTEMPT_FILENAME
    if not (path.exists() or path.is_symlink()):
        return attempt, None
    record = _record(path)
    if record is None:
        return None, "attempt_record_unreadable"
    recorded = record.get(WORK_DIR_FIELD)
    if recorded is None:
        return attempt, None
    work = Path(recorded) if isinstance(recorded, str) else Path()
    # Only where the runner puts it (``new_attempt_work_dir``): <work root>/jobs/<job>/attempts/<NN>.
    if (not work.is_absolute() or ".." in work.parts or work.name != attempt.name
            or work.parent.name != ATTEMPTS_DIRECTORY or work.parent.parent.name != job.name
            or work.parent.parent.parent.name != JOBS_DIRECTORY):
        return None, "work_dir_invalid"
    if _unsafe(work):
        return None, "work_dir_invalid"
    if not work.is_dir():
        return None, "work_dir_unavailable"
    return work, None


def _money(attempt: Path, work: Path) -> tuple[str | None, str | None]:
    """(how its money ended, None), or (None, why it must wait)."""
    settlement = attempt / SETTLEMENT_FILENAME
    if settlement.exists() or settlement.is_symlink():
        return ("settled", None) if _settled(settlement, attempt.name) else (None, "settlement_unreadable")
    funding = work / "allocator" / FUNDING_FILENAME
    if funding.exists() or funding.is_symlink():
        # A reservation with no settlement: not settled yet, held, or refused by the WebApp.
        return None, "settlement_pending" if (attempt / RESULT_FILENAME).exists() else "allocator_running_or_unknown"
    if (attempt / RESULT_FILENAME).exists():
        return "never_funded", None
    if (attempt / SPAWN_FAILED_FILENAME).exists() and not (attempt / PROCESS_FILENAME).exists():
        return "never_launched", None
    if (attempt / INTENT_FILENAME).exists():
        return None, "allocator_running_or_unknown"
    return "never_launched", None


def _bulk(work: Path) -> list[Path]:
    """The derived bytes in one work directory: its bundle, and each archive and extraction of its run."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError

    roots = [work / "bundle"] if (work / "bundle").exists() or (work / "bundle").is_symlink() else []

    def unreadable(error: OSError) -> None:
        raise _Unsafe(_PREFIX + "record_unreadable") from error

    allocator = work / "allocator"
    if allocator.is_symlink():
        raise WithdrawalDeletionError(_PREFIX + "unsafe_entry")
    for directory, directories, files in os.walk(allocator, onerror=unreadable) if allocator.is_dir() else ():
        here = Path(directory)
        for name in [*directories, *files]:
            info = os.lstat(here / name)
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise WithdrawalDeletionError(_PREFIX + "unsafe_entry")
        roots += [here / name for name in directories if name in _EXTRACTED]
        roots += [here / name for name in files if name.endswith(_ARCHIVE_SUFFIX)]
        directories[:] = sorted(name for name in directories if name not in _EXTRACTED)
    return sorted(roots)


def _inventory(roots: list[Path]) -> list[list[Any]]:
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _inventory as inventory

    rows = []
    for root in roots:
        info = os.lstat(root)
        if stat.S_ISREG(info.st_mode):
            rows.append([str(root), info.st_size, info.st_ino, info.st_mtime_ns])
        elif stat.S_ISDIR(info.st_mode):
            rows += [[str(root / row[0]), *row[1:]] for row in inventory(root)]
        else:
            raise WithdrawalDeletionError(_PREFIX + "unsafe_entry")
    return sorted(rows)


def plan_job(job: Path, work_root: Path | None) -> dict[str, Any]:
    """One job: its attempts' money, and for a final job each attempt's derived bytes; raises on an unsafe entry."""
    attempts_dir = job / ATTEMPTS_DIRECTORY
    attempts = sorted(path for path in attempts_dir.iterdir() if path.name.isdigit() and len(path.name) == 2) \
        if attempts_dir.is_dir() and not attempts_dir.is_symlink() else []
    rows, waiting = [], []
    conventional = (Path(work_root) / JOBS_DIRECTORY / job.name / ATTEMPTS_DIRECTORY) if work_root else None
    bound: set[str] = set()
    for attempt in attempts:
        if attempt.is_symlink() or not attempt.is_dir():
            raise _Unsafe(_PREFIX + "unsafe_entry")
        work, reason = _work_dir(attempt, job)
        money = None
        if reason is None:
            money, reason = _money(attempt, work)
        if reason is not None:
            waiting.append({"attempt": attempt.name, "reason": reason})
            continue
        works = [work]
        unrecorded = conventional / attempt.name if conventional else None
        if unrecorded is not None and unrecorded != work and (unrecorded.exists() or unrecorded.is_symlink()):
            # A staging that failed before its record named it: no allocator ran from it, so it holds no money.
            if unrecorded.is_symlink() or (unrecorded / "allocator" / FUNDING_FILENAME).exists():
                waiting.append({"attempt": attempt.name, "reason": "work_dir_unrecorded"})
                continue
            works.append(unrecorded)
        bound.update(str(path) for path in works)
        roots = [root for path in works for root in _bulk(path)]
        if (attempt / INPUTS_FILENAME).exists() or (attempt / INPUTS_FILENAME).is_symlink():
            roots.append(attempt / INPUTS_FILENAME)
        inventory = _inventory(sorted(roots))
        rows.append({"attempt": str(attempt), "money": money, "work_dirs": [str(path) for path in works],
                     "remove": [str(root) for root in sorted(roots)],
                     "inventory_digest": canonical_digest({"rows": inventory}), "file_count": len(inventory)})
    if conventional is not None and conventional.is_dir():
        # A work directory no attempt of the queue names cannot be tied to its money.
        waiting += [{"attempt": path.name, "reason": "work_dir_unbound"} for path in sorted(conventional.iterdir())
                    if str(path) not in bound and not (attempts_dir / path.name).is_dir()]
    return {"job": str(job), "final": not waiting, "attempts": rows if not waiting else [], "waiting": waiting}


def plan_probe_jobs(root: Path, work_root: Path | None, keys: set[str],
                    unresolved: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(the jobs whose money is final, with their derived bytes; the jobs that wait, with why)."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError

    jobs_dir = root / JOBS_DIRECTORY
    jobs = sorted(job for job in jobs_dir.iterdir() if any(job.name.startswith(key + "-") for key in keys)) \
        if jobs_dir.is_dir() and not jobs_dir.is_symlink() else []
    names = {job.name for job in jobs}
    final, waiting = [], []
    for job in jobs:
        if job.is_symlink() or not job.is_dir():
            unresolved.append({"path": str(job), "reason": _PREFIX + "unsafe_entry"})
            continue
        try:
            planned = plan_job(job, work_root)
        except (_Unsafe, WithdrawalDeletionError) as error:
            unresolved.append({"path": str(job), "reason": str(error)})
            continue
        except OSError:
            unresolved.append({"path": str(job), "reason": _PREFIX + "record_unreadable"})
            continue
        (final if planned["final"] else waiting).append(planned)
    work_jobs = Path(work_root) / JOBS_DIRECTORY if work_root else None
    if work_jobs is not None and work_jobs.is_dir() and not work_jobs.is_symlink():
        # Bulk of a job the queue no longer has: nothing ties it to its money.
        waiting += [{"job": str(path), "final": False, "attempts": [],
                     "waiting": [{"attempt": None, "reason": "work_dir_unbound"}]}
                    for path in sorted(work_jobs.iterdir())
                    if path.name not in names and any(path.name.startswith(key + "-") for key in keys)]
    return final, waiting


def hold_job(stack: ExitStack, job: Path) -> None:
    """Hold the runner's lock on one job for the deletion, without waiting; nothing is created."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _require

    try:
        descriptor = os.open(job / JOB_LOCK_FILENAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return  # No pass ever advanced this job.
    except OSError:
        raise WithdrawalDeletionError(_PREFIX + "rigid_probe_job_lock_unsafe") from None
    lock = stack.enter_context(os.fdopen(descriptor, "rb"))
    _require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), "rigid_probe_job_lock_unsafe")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise WithdrawalDeletionError(_PREFIX + "rigid_probe_job_busy") from None


def verify_jobs(stack: ExitStack, planned: list[Mapping[str, Any]], work_root: Path | None) -> None:
    """Lock each reviewed job and refuse unless it is planned exactly as reviewed."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _require

    for row in planned:
        job = Path(row["job"])
        _require(not _unsafe(job), "unsafe_entry")
        hold_job(stack, job)
        try:
            now = plan_job(job, work_root)
        except (_Unsafe, WithdrawalDeletionError, OSError):
            raise WithdrawalDeletionError(_PREFIX + "rigid_probe_job_changed") from None
        _require(now == dict(row), "rigid_probe_job_changed")


def remove_jobs(planned: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove each verified attempt's derived bytes; the job's records stay."""
    removed = []
    for row in planned:
        for attempt in row["attempts"]:
            for path in attempt["remove"]:
                target = Path(path)
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            removed.append({"attempt": attempt["attempt"], "money": attempt["money"], "removed": attempt["remove"],
                            "files_deleted": attempt["file_count"]})
    return removed


__all__ = ["hold_job", "plan_job", "plan_probe_jobs", "remove_jobs", "verify_jobs"]
