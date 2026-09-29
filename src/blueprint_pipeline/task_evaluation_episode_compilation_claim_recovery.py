"""Recover interrupted episode-compilation claims instead of stranding them (plan 14 §10).

Only the no-spend unit claims from the episode-compilation queue, and systemd never runs it twice
at once.  A run finishes every claim it takes (the row moves to ``completed/`` or ``blocked/``) or
leaves it in ``processing/`` beside a hand-off for the paid unit, which returns a row only through a
fallback marker; every row the paid unit has touched also has a lease record, which is never
deleted.  So at the start of a run, a ``processing/`` row with no hand-off, shadow marker, fallback
marker or lease record was left by a run that died (a deploy restart, SIGTERM, OOM), and nothing
else will ever move it.  For each such orphan, recovery:

1. renames any partial ``compiled-episodes/<id>`` aside to ``.<id>.interrupted-<epoch>``, which
   nothing deletes and storage GC never selects (it skips dot-prefixed children), so the next
   compile's exclusive ``mkdir`` succeeds;
2. counts the interruption in ``remote-cpu-jobs/recovery/episode_compilation/<name>``;
3. moves the row back to ``pending/``, byte for byte: a link, then an unlink.

On the third interruption the row is sealed ``blocked`` with
``episode_compilation_claim_interrupted:worker_process_terminated`` instead, in the style of
``task_evaluation_preparation_claim_recovery``.  The requeues are recorded under the result's
``claim_recovery``; ``automatic_retry_performed`` keeps its meaning of a paid retry, and stays false.

Two interruptions need no count, because no compile ran: a claim's empty placeholder left beside
its still-pending row is dropped, and a row whose result was written before the row moved is moved
as that result says.  A handed-back row is never requeued, but its compile can be interrupted too:
``prepare_fallback_compile`` sets its partial output aside and counts it against the same budget.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_job_contract import job_id_for
from .remote_cpu_job_records import replace_remote_cpu_record
from .task_evaluation_episode_compilation_remote import QUEUE, STAGE, marker_path
from .task_evaluation_episode_compilation_worker import RESULT_SCHEMA_VERSION, record_compilation

RECOVERY_SCHEMA_VERSION = "task_evaluation_episode_compilation_claim_recovery.v1"
INTERRUPTED_CLAIM_BLOCKER = "episode_compilation_claim_interrupted:worker_process_terminated"
INTERRUPTIONS_BEFORE_BLOCKING = 3
_ROW_NAME = re.compile(r"(?P<compilation_id>[A-Za-z0-9][A-Za-z0-9._-]*)-[0-9a-f]{64}\.json")
_MAX_ENVELOPE_BYTES = 4 * 1024 * 1024


def record_path(jobs_root: str | Path, name: str) -> Path:
    return Path(jobs_root) / "recovery" / STAGE / name


def _owned_by_the_paid_path(jobs_root: Path, name: str) -> bool:
    """A hand-off, shadow or fallback marker, or a lease record: the paid unit decides this row's fate.

    Checked in this order because the paid unit writes a fallback before it drops a hand-off, and
    never deletes a lease record, so a row it is moving can never look unowned in between.
    """

    return (any(marker_path(jobs_root, kind, name).exists() for kind in ("authoritative", "shadow", "fallback"))
            or (jobs_root / "leases" / f"{job_id_for(STAGE, name)}.json").exists())


def set_aside(output_root: Path, name: str, now: float) -> str | None:
    """Rename the row's partial output to ``.<id>.interrupted-<epoch>``; ``None`` when there is none."""

    match = _ROW_NAME.fullmatch(name)
    if match is None:
        return None
    source = output_root / match["compilation_id"]
    if not source.exists() and not source.is_symlink():
        return None
    stem, suffix = f".{match['compilation_id']}.interrupted-{int(now)}", 0
    while (output_root / f"{stem}{f'.{suffix}' if suffix else ''}").exists():
        suffix += 1
    target = output_root / f"{stem}{f'.{suffix}' if suffix else ''}"
    os.rename(source, target)
    return target.name


def _read(jobs_root: Path, name: str) -> dict[str, Any] | None:
    try:
        return json.loads(record_path(jobs_root, name).read_bytes())
    except FileNotFoundError:
        return None


def _count(jobs_root: Path, name: str, *, claim: list[int] | None, aside: str | None, now: float) -> dict[str, Any]:
    """Add one interruption to the row's record, once per claim: a recovery that died is not counted twice."""

    path, current = record_path(jobs_root, name), _read(jobs_root, name)
    history = list(current["history"]) if current else []
    if claim is not None and history and history[-1]["claim"] == claim:
        return current
    history.append({"at_epoch": float(now), "claim": claim, "set_aside": aside})
    record = {"schema_version": RECOVERY_SCHEMA_VERSION, "queue": QUEUE, "name": name,
              "interruptions": len(history), "history": history, "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    replace_remote_cpu_record(path, record, previous_digest=current["record_digest"] if current else None,
                              digest_field="record_digest", mode=0o640)
    return record


def _seal_blocked(queue: Path, name: str, claimed: Path, record: dict[str, Any], *,
                  source_commit: str) -> dict[str, Any]:
    try:
        with claimed.open("rb") as stream:
            envelope = json.loads(stream.read(_MAX_ENVELOPE_BYTES))
    except (OSError, ValueError):
        envelope = None
    envelope = envelope if isinstance(envelope, dict) else {}
    match = _ROW_NAME.fullmatch(name)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "compilation_id": match["compilation_id"] if match else name.removesuffix(".json"),
        **{key: envelope[key] for key in ("run_id", "team_namespace") if isinstance(envelope.get(key), str)},
        "source_commit": source_commit,
        "claim_recovery": {"schema_version": RECOVERY_SCHEMA_VERSION, "interruptions": record["interruptions"],
                           "requeued": record["interruptions"] - 1, "history": record["history"],
                           "record_digest": record["record_digest"]},
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "automatic_retry_performed": False,
        "blockers": [INTERRUPTED_CLAIM_BLOCKER],
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    record_compilation(queue, name, claimed, "blocked", result)
    return result


def _finish_recorded(queue: Path, name: str, claimed: Path) -> None:
    """The result was written and the process died before the row moved: move it as the result says."""

    try:
        status = json.loads((queue / "results" / name).read_bytes()).get("status")
    except (OSError, ValueError, AttributeError):
        status = None  # as ``record_compilation`` treats a result it did not write: blocked
    os.replace(claimed, queue / ("completed" if status == "compiled_for_production_launch" else "blocked") / name)


def recover_interrupted_claims(queue_root: str | Path, *, jobs_root: str | Path, output_root: str | Path,
                               source_commit: str, now: float) -> list[dict[str, Any]]:
    """Recover every orphaned ``processing/`` row; run at the start of each no-spend run, before any claim."""

    queue, jobs, outputs = Path(queue_root), Path(jobs_root), Path(output_root)
    recovered = []
    for claimed in sorted((queue / "processing").glob("*.json")):
        entry: dict[str, Any] = {"name": claimed.name, "action": "", "interruptions": 0, "set_aside": None}
        try:
            if claimed.is_symlink() or not claimed.is_file() or _owned_by_the_paid_path(jobs, claimed.name):
                continue
            _recover(queue, jobs, outputs, claimed, entry, source_commit=source_commit, now=now)
        except (OSError, ValueError, KeyError, TypeError) as exc:  # a record conflict is a ValueError
            # One row that cannot be recovered must not wedge the unit: it stays claimed, and is reported.
            entry["action"] = f"failed:{type(exc).__name__}:{exc}"[:256]
        recovered.append(entry)
    return recovered


def _recover(queue: Path, jobs: Path, outputs: Path, claimed: Path, entry: dict[str, Any], *,
             source_commit: str, now: float) -> None:
    name, pending = claimed.name, queue / "pending" / claimed.name
    if pending.exists():
        if pending.is_file() and os.path.samefile(pending, claimed):
            claimed.unlink()  # a requeue that died between its link and its unlink: already counted
            entry.update(action="requeued", interruptions=(_read(jobs, name) or {}).get("interruptions", 0))
        elif claimed.stat().st_size == 0:
            claimed.unlink()  # ``claim_pending_row`` died between its placeholder and its replace
            entry["action"] = "placeholder_removed"
        else:
            entry["action"] = "ambiguous"  # two different rows under one name: left for an operator
        return
    if (queue / "results" / name).exists():
        _finish_recorded(queue, name, claimed)
        entry["action"] = "finished"
        return
    info = claimed.stat()
    aside = set_aside(outputs, name, now)
    record = _count(jobs, name, claim=[info.st_ino, info.st_ctime_ns], aside=aside, now=now)
    entry.update(interruptions=record["interruptions"], set_aside=aside)
    if record["interruptions"] >= INTERRUPTIONS_BEFORE_BLOCKING:
        _seal_blocked(queue, name, claimed, record, source_commit=source_commit)
        entry["action"] = "blocked"
        return
    os.link(claimed, pending)
    claimed.unlink()
    entry["action"] = "requeued"


def prepare_fallback_compile(queue_root: str | Path, name: str, claimed: Path, *, jobs_root: str | Path,
                             output_root: str | Path, source_commit: str, now: float) -> dict[str, Any] | None:
    """Before a handed-back row compiles on the host: an output already at its path means a compile of it
    died.  Set that aside and count it; on the third interruption seal the row ``blocked`` instead and
    return that result, so the caller compiles nothing."""

    aside = set_aside(Path(output_root), name, now)
    if aside is None:
        return None
    record = _count(Path(jobs_root), name, claim=None, aside=aside, now=now)
    if record["interruptions"] < INTERRUPTIONS_BEFORE_BLOCKING:
        return None
    return _seal_blocked(Path(queue_root), name, claimed, record, source_commit=source_commit)


__all__ = [
    "INTERRUPTED_CLAIM_BLOCKER",
    "INTERRUPTIONS_BEFORE_BLOCKING",
    "RECOVERY_SCHEMA_VERSION",
    "prepare_fallback_compile",
    "record_path",
    "recover_interrupted_claims",
    "set_aside",
]
