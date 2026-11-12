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
- every byte the restore writes before the run lands (the download, its extraction) is inside one staging
  directory named after the run, ``.<run>.restoring``: a restore killed mid-way leaves nothing anywhere
  else, the next pass removes it (``clean_stale_staging``), and a withdrawal deletion counts it as the
  run's own leftover and deletes it with the run (``website_capture_withdrawal_deletion._Units``);
- a transient failed try (disk admission, transfer, store outage, timeout) spends nothing and is tried again
  while the job's window is open: the next try waits ``job_retry_policy.outage_retry_delay`` (15 min,
  doubling, at most 6 h), recorded beside the pointer; the screening pass reads the run again on its next
  pass and its hold is published as released;
- a failure of the archive's own integrity (its digest, a member list that does not match the pointer, a
  symlink, an archive that will not open) cannot clear by waiting: the bytes stored are not the bytes
  offloaded. It ends the tries for this cold-storage episode as ``ARCHIVE_INVALID`` (in the restore record
  and the screening outcome; the job's hold becomes ``team_scene_packet_blocked`` with this code as its
  detail, which the website words as its general "a problem on our side" sentence, never "being restored",
  and reports to staff with its held-evaluation alert) and is tried again only
  when the pointer changes.

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

from .scene_run_cold_pointer import (  # noqa: F401 - ARCHIVE_INVALID and archive_invalid are re-exported
    ARCHIVE_INVALID, POINTER_SUFFIX, RESTORE_STATE_SUFFIX as STATE_SUFFIX, archive_invalid, offload_pointer,
    restore_state,
)

#: The restore's own record beside the run (sealed, so a withdrawal deletion removes it with the run): its
#: attempt count, times, last status and an error type/code. No content of the job.
STATE_SCHEMA = "scene_run_restore_state.v1"
#: The backoff between tries (``job_retry_policy.outage_retry_delay``): the first wait, and the longest.
RETRY_FIRST_SECONDS = 900
RETRY_CAP_SECONDS = 6 * 3600
ROLE = "evidence_offload"
RESTORED = "restored"
#: ``control_plane_evidence_offload.restore_offloaded_evidence``'s refusals of the archive's own bytes.
INTEGRITY_ERRORS = frozenset({"control_plane_evidence_restore_digest_mismatch",
                              "control_plane_evidence_restore_member_mismatch",
                              "control_plane_evidence_restore_symlink"})


def _integrity_failure(exc: BaseException) -> bool:
    import tarfile

    return isinstance(exc, tarfile.TarError) or str(exc) in INTEGRITY_ERRORS


def retry_delay(failures: int) -> int:
    """Seconds before the next try after ``failures`` failed tries in a row; the job's window is the deadline."""
    from .job_retry_policy import outage_retry_delay

    return outage_retry_delay(failures, first=RETRY_FIRST_SECONDS, cap=RETRY_CAP_SECONDS)


def staging_path(run_root: str | Path) -> Path:
    """The one directory a restore of ``run_root`` writes into before the run lands: ``.<run>.restoring``."""
    run = Path(run_root)
    return run.parent / f".{run.name}.restoring"


def _state_path(run: Path) -> Path:
    return run.parent / (run.name + STATE_SUFFIX)


def _state(run: Path) -> dict[str, Any]:
    return restore_state(run)


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


def _restorer_lock(run: Path):
    return run.parent / f".{run.name}.restore.lock"


def clean_stale_staging(run_root: str | Path) -> bool:
    """Remove a crashed restore's staging folder when no restorer holds this run's lock; never raises.

    Run on every pass, whether or not the job is still open. Returns whether anything was removed.
    """
    import fcntl
    import shutil

    run = Path(run_root)
    staging = staging_path(run)
    if not staging.exists() or staging.is_symlink():
        return False
    try:
        with _restorer_lock(run).open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False  # a restorer is working on it now
            shutil.rmtree(staging, ignore_errors=True)
            return True
    except OSError:
        return False


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
    staging folder is deleted and nothing lands. The download and its extraction happen inside that staging
    folder (the restorer's destination is ``.<run>.restoring/<run>``), so nothing of the run is ever written
    under a name a withdrawal could not tie to it. Failed tries are counted per cold-storage episode only to
    space them (``retry_delay``); a restore that lands resets them. Only the window closing or the job's
    withdrawal ends the tries.

    Returns ``{"status": ...}``: ``on_disk``, ``window_closed``, ``WITHDRAWN``, ``BUSY``, ``restored``,
    ``waiting_retry``, ``refused`` (a transient failure: tried again later), or ``ARCHIVE_INVALID``.
    """
    import fcntl
    import shutil

    run = Path(run_root)
    clean_stale_staging(run)
    pointer = offload_pointer(run)
    if pointer is None:
        return {"status": "on_disk" if run.is_dir() else "no_pointer"}
    if not window_open:
        return {"status": "window_closed"}
    if not still_current():
        return {"status": WITHDRAWN}
    state, moment = _state(run), float(now())
    if state.get("pointer_digest") not in (None, pointer.get("pointer_digest")):
        state = {"attempts": 0}  # a changed pointer is a new episode: earlier failures do not count
    if state.get("status") == ARCHIVE_INVALID:
        return {**state, "status": ARCHIVE_INVALID}  # the same archive will not match again
    if moment < float(state.get("retry_not_before") or 0):
        return {**state, "status": "waiting_retry"}
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT, reserve_control_plane_disk
    from .control_plane_evidence_offload import restore_offloaded_evidence

    attempt = int(state.get("attempts") or 0) + 1
    record: dict[str, Any] = {"attempts": attempt, "last_attempt_epoch": moment,
                              "pointer_digest": pointer.get("pointer_digest")}
    staging = staging_path(run)
    landing = staging / run.name
    lock = _restorer_lock(run).open("a+b")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {**state, "status": BUSY}  # another restorer runs now
    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(mode=0o750)
        with contextlib.suppress(OSError):  # the run root's owner, so the restored files are the service's
            owner = run.parent.stat()
            os.chown(staging, owner.st_uid, owner.st_gid)
        hold = (reserve or reserve_control_plane_disk)(
            ROLE, target_root=run.parent, expected_bytes=max(1, 2 * int(pointer["size_bytes"])),
            reservation_root=os.environ.get("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT") or DEFAULT_RESERVATION_ROOT,
            workload="scene_run_restore")
        with hold:
            (restorer or restore_offloaded_evidence)(pointer_path=run.parent / (run.name + POINTER_SUFFIX),
                                                     destination=landing)
            with deletion_lock(packet_queue_root) as held:
                if not held:
                    shutil.rmtree(staging, ignore_errors=True)
                    return {**state, "status": BUSY}  # not an attempt: the next pass asks again
                if not still_current():
                    shutil.rmtree(staging, ignore_errors=True)  # withdrawn meanwhile: what was fetched is deleted
                    return {"status": WITHDRAWN}
                _touch(landing, float(now()))
                os.replace(landing, run)
                shutil.rmtree(staging, ignore_errors=True)
        record = {"attempts": 0, "status": RESTORED, "restored_at_epoch": float(now()),
                  "attempts_this_episode": attempt}
    except Exception as exc:  # noqa: BLE001 - a failed try spends nothing; it is tried again while the window runs
        shutil.rmtree(staging, ignore_errors=True)
        record.update(status="refused", error_type=type(exc).__name__,
                      error=str(exc)[:200] if isinstance(exc, ValueError) or "disk" in str(exc) else "",
                      retry_not_before=moment + retry_delay(attempt))
        if _integrity_failure(exc):
            record.update(status=ARCHIVE_INVALID, error=str(exc)[:200])
            record.pop("retry_not_before")
    finally:
        lock.close()
    _write_state(run, record)
    return record


def _write_state(run: Path, record: Mapping[str, Any]) -> None:
    from .decision_evidence_contracts import canonical_digest

    sealed = {**record, "schema_version": STATE_SCHEMA, "run": run.name, "state_digest": ""}
    sealed["state_digest"] = canonical_digest(sealed, digest_field="state_digest")
    # Atomic: a temporary in the same folder, flushed to disk, then renamed over the record, so a crash leaves
    # the old record or the new one, never a torn one (the withdrawal removes either with the run).
    path = _state_path(run)
    temporary = path.parent / f".{run.name}.restore-state.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(sealed, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        temporary.unlink(missing_ok=True)


__all__ = ["ARCHIVE_INVALID", "BUSY", "archive_invalid", "INTEGRITY_ERRORS", "RESTORED", "RETRY_CAP_SECONDS", "RETRY_FIRST_SECONDS", "WITHDRAWN", "clean_stale_staging",
           "deletion_lock", "restore_open_job_run", "retry_delay", "staging_path"]
