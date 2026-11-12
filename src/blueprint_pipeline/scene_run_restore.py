"""Bring an open website job's scene run back from cold storage by itself, so no job waits on a person.

Beta journey step: L5 team policy evaluation. The storage GC may move a sealed launch run to the artifact
store (``control_plane_evidence_offload``), leaving a pointer that keeps only the launch receipt. Stage 8
reads more of the run than that:

- the screening re-plan: the launch receipt, its terminal result, ``webapp_sync_succeeded.json`` and
  ``post_teardown_provider_zero_receipt.json`` (``task_evaluation_configured_controls_source_evidence``),
  and the robot-free probe's receipt beside the launch (``robot_free_probe_gate.retained_probe_receipt``);
- a team-half recompile: the configured bundle inside the run (``team_scene_packet_worker._compile``);
- a new robot-free probe: the run itself (``rigid_robot_free_probe_bundle``).

Only the launch receipt is in the pointer, so each of them needs the run on disk. ``restore_open_job_run``
restores it in place for an OPEN job (its scene intent window still runs; ``open_job_source_runs``):

- under one disk reservation for the download and its extraction (twice the archive's size), refused
  without spending anything when the disk cannot take it;
- with every restored file and directory's times set to now, so the clean-up counts the run as fresh
  (the archive's own times are the offload's: without this the next pass would offload it again);
- at most ``MAX_ATTEMPTS`` tries, ``RETRY_SECONDS`` apart, recorded beside the pointer; the screening pass
  reads the run again on its next pass and its hold is published as released.

A job whose window has ended is not restored: there is nothing left to run for it, and the screening pass
tells it its window closed (``beta_screening_job_window_closed``). Restoring never deletes the pointer.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .scene_run_cold_storage import POINTER_SUFFIX, offload_pointer

STATE_SUFFIX = ".restore.v1.json"
MAX_ATTEMPTS = 3
RETRY_SECONDS = 3600
ROLE = "evidence_offload"
RESTORED = "restored"
GAVE_UP = "scene_run_restore_attempts_exhausted"


def _state_path(run: Path) -> Path:
    return run.parent / (run.name + STATE_SUFFIX)


def _state(run: Path) -> dict[str, Any]:
    try:
        value = json.loads(_state_path(run).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"attempts": 0}
    return value if isinstance(value, dict) else {"attempts": 0}


def _touch(root: Path, moment: float) -> None:
    for path in (*root.rglob("*"), root):
        if not path.is_symlink():
            os.utime(path, (moment, moment))


WITHDRAWN = "scene_run_restore_job_withdrawn_or_revoked"
BUSY = "scene_run_restore_deletion_lock_busy"


@contextlib.contextmanager
def deletion_lock(packet_queue_root: str | Path | None) -> Iterator[bool]:
    """The team-scene packet queue's lock, the one withdrawal deletion holds throughout (never waits).

    Yields False when another holder has it (a deletion runs now): nothing is placed, and the next pass asks
    again. Yields True with no queue or no lock file, as deletion itself then holds nothing there.
    """
    import fcntl

    path = Path(packet_queue_root) / ".lock" if packet_queue_root else None
    if path is None or not path.is_file() or path.is_symlink():
        yield True
        return
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def restore_open_job_run(run_root: str | Path, *, window_open: bool,
                         still_current: Callable[[], bool] = lambda: True,
                         packet_queue_root: str | Path | None = None,
                         restorer: Callable[..., Mapping[str, Any]] | None = None,
                         reserve: Callable[..., Any] | None = None,
                         now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Restore ``run_root`` from its pointer when its job is open; never raises.

    Privacy first: ``still_current`` (the job is not withdrawn, its consent not revoked) is asked before
    anything is fetched and again, under the deletion's own lock (``deletion_lock``), before anything is
    placed; the run is fetched into a staging folder beside it, and if the job was withdrawn meanwhile the
    staging folder is deleted and nothing lands. Attempts are counted per cold-storage episode: a restore that
    lands resets them, so a later offload of the same run starts a fresh bound.

    Returns ``{"status": ...}``: ``on_disk``, ``window_closed``, ``WITHDRAWN``, ``BUSY``, ``restored``,
    ``waiting_retry``, ``refused`` (this try failed: disk, transfer or digest), or ``GAVE_UP``.
    """
    import shutil

    run = Path(run_root)
    pointer = offload_pointer(run)
    if pointer is None:
        return {"status": "on_disk" if run.is_dir() else "no_pointer"}
    if not window_open:
        return {"status": "window_closed"}
    if not still_current():
        return {"status": WITHDRAWN}
    state, moment = _state(run), float(now())
    if int(state.get("attempts") or 0) >= MAX_ATTEMPTS:
        return {**state, "status": GAVE_UP}
    if moment < float(state.get("retry_not_before") or 0):
        return {**state, "status": "waiting_retry"}
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, reserve_control_plane_disk
    from .control_plane_evidence_offload import restore_offloaded_evidence

    attempt = int(state.get("attempts") or 0) + 1
    record: dict[str, Any] = {"attempts": attempt, "last_attempt_epoch": moment}
    staging = run.parent / f".{run.name}.restoring"
    try:
        shutil.rmtree(staging, ignore_errors=True)
        hold = (reserve or reserve_control_plane_disk)(
            ROLE, target_root=run.parent, expected_bytes=max(1, 2 * int(pointer["size_bytes"])),
            reservation_root=os.environ.get("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT") or DEFAULT_RESERVATION_ROOT,
            workload="scene_run_restore")
        with hold:
            (restorer or restore_offloaded_evidence)(pointer_path=run.parent / (run.name + POINTER_SUFFIX),
                                                     destination=staging)
            with deletion_lock(packet_queue_root) as held:
                if not held:
                    shutil.rmtree(staging, ignore_errors=True)
                    return {**state, "status": BUSY}  # not an attempt: the next pass asks again
                if not still_current():
                    shutil.rmtree(staging, ignore_errors=True)  # withdrawn meanwhile: what was fetched is deleted
                    return {"status": WITHDRAWN}
                _touch(staging, float(now()))
                os.replace(staging, run)
        record = {"attempts": 0, "status": RESTORED, "restored_at_epoch": float(now()),
                  "attempts_this_episode": attempt}
    except Exception as exc:  # noqa: BLE001 - a failed try spends nothing; it is tried again, bounded
        shutil.rmtree(staging, ignore_errors=True)
        record.update(status="refused", error_type=type(exc).__name__,
                      error=str(exc)[:200] if isinstance(exc, ValueError) or "disk" in str(exc) else "",
                      retry_not_before=moment + RETRY_SECONDS)
        if attempt >= MAX_ATTEMPTS:
            record["status"] = GAVE_UP
    try:
        _state_path(run).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass
    return record


__all__ = ["BUSY", "GAVE_UP", "MAX_ATTEMPTS", "RESTORED", "RETRY_SECONDS", "WITHDRAWN", "deletion_lock",
           "restore_open_job_run"]
