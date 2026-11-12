"""Keep a website job's scene run on the host while the job can still need it.

Beta journey step: L5 team policy evaluation. A website job's scene run (the
completed scene-configuration launch under ``task-evaluation-launch-runs``) is
read again after the scene is built. Three things read it:

- the screening pass's re-plan (``beta_policy_screening_worker``), which
  re-checks the launch receipt, its result and its post-teardown receipt;
- a recompile of the team half (``team_scene_packet_worker._compile``), which
  also needs the configured bundle kept inside the run;
- the robot-free probe, whose trace and receipt live beside the launch.

The storage GC moved such a run to cold storage after two idle days, with only
its launch receipt kept on the host. Capture-447ae4f4's run went on
2026-11-06T13:22Z, and every re-plan after that failed with
``configured_controls_worker_launch_receipt_invalid``.

``open_job_source_reason`` is the GC's reason to keep such a run. A run is kept
while all three of these hold:

1. a team-scene-packet request (``pending``, ``processing`` or ``processed``)
   names it as its ``source_launch_root``;
2. that request's scene intent window has not ended: the intent's
   ``request.execution.expires_at_epoch``, raised by its verified window
   extensions (``task_evaluation_scene_execution_window_evidence``). A window
   that cannot be read, or whose extension is invalid, counts as ended, so a
   broken record never keeps bytes for ever;
3. the job's testing is not final. Final means: the intent has a screening
   campaign, and every one of its newest campaigns (not superseded) has all its
   learned rows ended and nothing a re-plan, a deploy or a retry would take up
   again (``beta_policy_screening_coordinator._replan_scope`` at any later
   commit returns nothing). No campaign yet (the half is still compiling, the
   probe is still deciding, a hold the system retries by itself) is not final.
   The screening pass decides this (``open_job_testing_final``: the coordinator
   reaches the paid allocator, which this lane never imports) and writes a
   record sealed to its ledger's digest. A record that is missing, broken or
   computed from another ledger counts as not final: the window in (2) still
   bounds how long the run is kept.

When the job is final or its window has ended, the run goes cold on the normal
two-day rule. Nothing here changes how a withdrawal deletes a site's runs: the
withdrawal deletion never reads this reason and removes a kept run like any
other. Reading only: nothing is written, moved or spent.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

OPEN_JOB_SOURCE = "protected_open_job_source"
# The two queues an open job's scene run is read through; the same names the team-scene-packet and screening
# units read. The storage GC takes them from its environment (``queue_roots_from_env``).
TEAM_SCENE_PACKET_QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
SCREENING_QUEUE_ROOT_ENV = "BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT"
REQUEST_STATES = ("pending", "processing", "processed")
LEDGER_FILENAME = "screening_ledger.v1.json"  # ``beta_policy_screening_coordinator.LEDGER_FILENAME``
TESTING_RECORD_FILENAME = "open_job_testing.v1.json"
TESTING_RECORD_SCHEMA = "blueprint.open_job_testing_record.v1"


def _json(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _same_run(named: Any, directory: Path) -> bool:
    if not isinstance(named, str) or not named:
        return False
    candidate = Path(named)
    if candidate == directory:
        return True
    try:
        return candidate.resolve() == directory.resolve()
    except OSError:
        return False


def naming_requests(directory: Path, *, packet_queue_root: str | Path) -> list[dict[str, Any]]:
    """Every team-scene-packet request that names ``directory`` as its source launch."""

    root = Path(packet_queue_root)
    found = []
    for state in REQUEST_STATES:
        folder = root / state
        if folder.is_symlink() or not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            request = _json(path)
            if request is not None and _same_run(request.get("source_launch_root"), directory):
                found.append(request)
    return found


def window_open(request: Mapping[str, Any], *, intent_root: str | Path, now: float) -> bool:
    """Whether the request's scene intent window is still open; unreadable or invalid reads as ended."""

    from . import task_evaluation_scene_intent_contracts as contracts
    from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry

    intent_id = str(request.get("intent_id") or "")
    directory = Path(intent_root) / intent_id
    if not intent_id or "/" in intent_id or intent_id in {".", ".."} or directory.is_symlink():
        return False
    try:
        intent = contracts._read(directory / "intent.json", "intent_digest")
        if intent.get("intent_id") != intent_id or intent.get("intent_digest") != request.get("intent_digest"):
            return False
        return effective_execution_expiry(directory, intent) > now
    except (OSError, ValueError, KeyError, TypeError):
        return False


def testing_final(intent_id: str, *, screening_queue_root: str | Path) -> bool:
    """Whether the screening pass recorded the intent's testing as final for the ledger as it is now.

    No record, a broken one, or one sealed to another ledger is not final (``open_job_testing_final``).
    """

    root = Path(screening_queue_root)
    record = _json(root / TESTING_RECORD_FILENAME)
    ledger = _json(root / LEDGER_FILENAME)
    try:
        if (root.is_symlink() or record is None or ledger is None
                or record.get("schema_version") != TESTING_RECORD_SCHEMA
                or record.get("record_digest") != canonical_digest(record, digest_field="record_digest")
                or ledger.get("ledger_digest") != canonical_digest(ledger, digest_field="ledger_digest")
                or record.get("ledger_digest") != ledger.get("ledger_digest")):
            return False
        final = record.get("final_by_intent")
        return isinstance(final, dict) and final.get(intent_id) is True
    except (TypeError, ValueError):
        return False


#: A row of the job's screening that is about to use, or using, its scene run: never released at the floor.
#: Pending and waitlisted rows are about to read it too (review 2026-11-12): never moved under the floor.
ACTIVE_TEST_STATUSES = frozenset({"pending", "waitlisted", "preparing", "launching", "running",
                                  "reconciliation_required"})
#: How the work disk is measured; a test replaces it.
measure_disk = None


def job_test_active(intent_id: str, *, screening_queue_root: str | Path | None) -> bool:
    """Whether a row of this job's screening is preparing, launching, running or being reconciled.

    Unreadable reads as active: a run is never released at the floor on a guess.
    """
    if screening_queue_root is None:
        return False
    root = Path(screening_queue_root)
    ledger = _json(root / LEDGER_FILENAME)
    if ledger is None:
        return (root / LEDGER_FILENAME).exists()
    for campaign_id, state in (ledger.get("campaigns") or {}).items():
        rows = (state.get("candidates") or {}).values() if isinstance(state, Mapping) else ()
        if not any(isinstance(row, Mapping) and row.get("status") in ACTIVE_TEST_STATUSES for row in rows):
            continue
        plan = _json(root / "plans" / f"{campaign_id}.json")
        if plan is None or (plan.get("source") or {}).get("scene_intent_id") == intent_id:
            return True
    return False


def _deficit_bytes(directory: Path) -> int:
    """How far the run's filesystem is under the clean-up's floor (``evidence_offload``); 0 at or above it."""
    import shutil

    from .control_plane_disk_budget import floor_bytes

    usage = (measure_disk or shutil.disk_usage)(directory.parent)
    return max(0, floor_bytes(int(usage.total), role="evidence_offload") - int(usage.free))


def _size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def released_at_floor(directory: Path, *, packet_queue_root: str | Path, intent_root: str | Path,
                      screening_queue_root: str | Path | None, now: float) -> bool:
    """Whether ``directory`` is among the open jobs' runs the clean-up may move while under the floor.

    Only while the disk is under its floor: the open jobs' runs, oldest first (their own times), that
    together cover how far it is under, never the run of a job with a test preparing, launching or
    running. A released run comes back on need (``scene_run_restore``).
    """
    deficit = _deficit_bytes(directory)
    if deficit <= 0:
        return False
    runs: dict[Path, float] = {}
    root = Path(packet_queue_root)
    for state in REQUEST_STATES:
        folder = root / state
        if folder.is_symlink() or not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            request = _json(path)
            named = Path(str((request or {}).get("source_launch_root") or ""))
            if (request is None or not named.is_absolute() or not named.is_dir() or named.is_symlink()
                    or not window_open(request, intent_root=intent_root, now=now)
                    or job_test_active(str(request.get("intent_id") or ""), screening_queue_root=screening_queue_root)):
                continue
            try:
                runs[named.resolve()] = named.stat().st_mtime
            except OSError:
                continue
    freed = 0
    for run in sorted(runs, key=lambda item: (runs[item], str(item))):
        if run == directory.resolve():
            return True
        freed += _size(run)
        if freed >= deficit:
            return False
    return False


def open_job_source_reason(directory: Path, *, packet_queue_root: str | Path | None,
                           screening_queue_root: str | Path | None, intent_root: str | Path | None,
                           now: Callable[[], float] = time.time) -> str | None:
    """``OPEN_JOB_SOURCE`` when an open website job still needs this run, else None (see the module docstring).

    Under the disk's floor the oldest open jobs' runs are released first (``released_at_floor``).
    """

    if not packet_queue_root or not intent_root:
        return None
    moment = float(now())
    for request in naming_requests(Path(directory), packet_queue_root=packet_queue_root):
        if not window_open(request, intent_root=intent_root, now=moment):
            continue
        if screening_queue_root is None or not testing_final(str(request.get("intent_id") or ""),
                                                              screening_queue_root=screening_queue_root):
            try:
                if released_at_floor(Path(directory), packet_queue_root=packet_queue_root, intent_root=intent_root,
                                     screening_queue_root=screening_queue_root, now=moment):
                    return None
            except OSError:
                pass  # an unmeasured disk keeps the run
            return OPEN_JOB_SOURCE
    return None


__all__ = ["ACTIVE_TEST_STATUSES", "OPEN_JOB_SOURCE", "REQUEST_STATES", "naming_requests", "open_job_source_reason",
           "released_at_floor", "job_test_active", "testing_final", "window_open"]


def queue_roots_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """The storage GC's ``team_scene_packet_queue_root`` and ``screening_queue_root``; unset is ``None``."""

    env = os.environ if environ is None else environ
    return {"team_scene_packet_queue_root": str(env.get(TEAM_SCENE_PACKET_QUEUE_ROOT_ENV) or "").strip() or None,
            "screening_queue_root": str(env.get(SCREENING_QUEUE_ROOT_ENV) or "").strip() or None}
