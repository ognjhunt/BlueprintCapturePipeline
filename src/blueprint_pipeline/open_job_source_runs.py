"""Keep a website job's scene run on the host while the job can still need it.

Beta journey step: L5 team policy evaluation. A website job's scene run (the
completed scene-configuration launch under ``task-evaluation-launch-runs``) is
read again after the scene is built:

- a recompile of the team half (``team_scene_packet_worker._compile``), which
  needs the configured bundle kept inside the run;
- the robot-free probe, whose trace and receipt live beside the launch;
- the site's success-rule exchange, which checks the launch's terminal result
  and reads the documents kept in the run
  (``task_evaluation_owner_success_confirmation.published_job``).

The early trial's screening re-plan read it too; the owner dropped the trial
on 2026-11-14, so its ledger and its testing record are not read here.

The storage GC moved such a run to cold storage after two idle days, with only
its launch receipt kept on the host. Capture-447ae4f4's run went on
2026-11-06T13:22Z, and every re-plan after that failed with
``configured_controls_worker_launch_receipt_invalid``.

``open_job_source_reason`` is the GC's reason to keep such a run. A run is kept
while both of these hold:

1. a team-scene-packet request (``pending``, ``processing`` or ``processed``)
   names it as its ``source_launch_root``;
2. that request's scene intent window has not ended: the intent's
   ``request.execution.expires_at_epoch``, raised by its verified window
   extensions (``task_evaluation_scene_execution_window_evidence``). A window
   that cannot be read, or whose extension is invalid, counts as ended, so a
   broken record never keeps bytes for ever.

Under the disk's floor the oldest open jobs' runs are released first
(``released_at_floor``); a released run comes back on need
(``scene_run_restore``). When the job's window has ended, the run goes cold on
the normal two-day rule. Nothing here changes how a withdrawal deletes a site's
runs: the withdrawal deletion never reads this reason and removes a kept run
like any other. Reading only: nothing is written, moved or spent.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

OPEN_JOB_SOURCE = "protected_open_job_source"
# The queue an open job's scene run is read through; the same name the team-scene-packet unit reads. The
# storage GC takes it from its environment (``queue_roots_from_env``).
TEAM_SCENE_PACKET_QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
REQUEST_STATES = ("pending", "processing", "processed")


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


#: How the work disk is measured; a test replaces it.
measure_disk = None


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
                      now: float) -> bool:
    """Whether ``directory`` is among the open jobs' runs the clean-up may move while under the floor.

    Only while the disk is under its floor: the open jobs' runs, oldest first (their own times), that
    together cover how far it is under. A released run comes back on need (``scene_run_restore``).
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
                    or not window_open(request, intent_root=intent_root, now=now)):
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
                           intent_root: str | Path | None, now: Callable[[], float] = time.time) -> str | None:
    """``OPEN_JOB_SOURCE`` when an open website job still needs this run, else None (see the module docstring).

    Under the disk's floor the oldest open jobs' runs are released first (``released_at_floor``).
    """

    if not packet_queue_root or not intent_root:
        return None
    moment = float(now())
    for request in naming_requests(Path(directory), packet_queue_root=packet_queue_root):
        if not window_open(request, intent_root=intent_root, now=moment):
            continue
        try:
            if released_at_floor(Path(directory), packet_queue_root=packet_queue_root, intent_root=intent_root,
                                 now=moment):
                return None
        except OSError:
            pass  # an unmeasured disk keeps the run
        return OPEN_JOB_SOURCE
    return None


__all__ = ["OPEN_JOB_SOURCE", "REQUEST_STATES", "naming_requests", "open_job_source_reason", "released_at_floor",
           "window_open"]


def queue_roots_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """The storage GC's ``team_scene_packet_queue_root``; unset is ``None``."""

    env = os.environ if environ is None else environ
    return {"team_scene_packet_queue_root": str(env.get(TEAM_SCENE_PACKET_QUEUE_ROOT_ENV) or "").strip() or None}
