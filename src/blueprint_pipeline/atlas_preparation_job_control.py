"""Source-neutral retained job locking and explicit retry; no provider execution."""
from __future__ import annotations
import fcntl
import os
import re
import stat
import time
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from .common import write_json
from .website_atlas_preview_store import LOCK_FILE, MAXIMUM_JOB_SECONDS
_CODE = re.compile(r"[a-z][a-z0-9_]{0,99}")
_NEEDS_NEW_CAPTURE = frozenset({"atlas_task_and_scene_coverage_gap",
                                "atlas_insufficient_overlapping_distinct_views",
                                "atlas_coverage_anchor_quality_insufficient"})
_NOT_RETRYABLE = frozenset({"atlas_preview_expired", "atlas_preview_source_video_unavailable"})

class _Withdrawn(Exception):
    pass


class PreparationJobBusy(ValueError):
    pass


def _transition(job: Mapping[str, Any], current: int, state: str, reason: str | None,
                **fields: Any) -> dict[str, Any]:
    changed = (job["state"], job["reason"]) != (state, reason) or (
        "archive" in fields and fields["archive"] != job.get("archive"))
    value = {**job, "retry_at_epoch": None, **fields, "state": state, "reason": reason, "updated_at_epoch": current}
    if changed:
        value.update(revision=job["revision"] + 1, revised_at_epoch=current)
    return value


def _reason(error: BaseException) -> str:
    """The typed code of a refusal: the owning service's code, never its text."""
    if isinstance(error, _Withdrawn):
        return "website_capture_withdrawn"
    status = getattr(error, "status", None)
    if isinstance(status, int):
        return f"website_atlas_api_{status}"
    head, _, tail = str(error).partition(":")
    if head.startswith(("website_control_", "owned_capture_atlas_control_")) and "_http_" in head:
        owned = tail.partition(":")[0]
        head = head if owned in {"", "unclassified"} else owned
    code = head.strip().lower().replace("-", "_")
    return code if _CODE.fullmatch(code) else "atlas_preview_" + re.sub(r"[^a-z0-9_]", "_",
                                                                         type(error).__name__.lower())[:60]


@contextmanager
def preparation_job_lock(job_dir: Path):
    """The same nonblocking, non-symlink lock for queueing, stepping and teardown."""
    descriptor = os.open(job_dir / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("atlas_preparation_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreparationJobBusy("atlas_preparation_job_busy") from None
        yield


def retry_preparation_job(path: Path, *, reader: Callable[[Path], dict[str, Any]],
                          source_check: Callable[[dict[str, Any]], None],
                          now: Callable[[], float] = time.time,
                          preflight: Callable[[dict[str, Any]], None] | None = None,
                          request_id: str | None = None) -> dict[str, Any]:
    """Explicit replay using retained stages, with no clock, source or allowance reset."""
    lock_fd = os.open(path.parent / LOCK_FILE, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(lock_fd, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("atlas_preview_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("atlas_preview_job_busy") from None
        job = reader(path)
        source_check(job)
        if request_id is not None and any(row.get("request_id") == request_id for row in job.get("retries", [])):
            return job
        # A finished job reopens only to send the gap-preserving refined-splat
        # forms its adopted minimal splat left untried; any other ready job stays.
        marker = job.get("refined_splat")
        upgrade = (job["state"] == "ready" and isinstance(marker, Mapping)
                   and marker.get("gap_preserving_form_untried") is True)
        if ((job["state"] not in {"failed", "needs_input"} and not upgrade) or job["reason"] in _NOT_RETRYABLE
                or job["reason"] in _NEEDS_NEW_CAPTURE):
            raise ValueError("atlas_preview_retry_requires_recoverable_failure")
        current = int(now())
        if current - job["enqueued_at_epoch"] > MAXIMUM_JOB_SECONDS:
            raise ValueError("atlas_preview_expired")
        # The selected-delivery operator rechecks source, current rights and the
        # reviewed budget snapshot beneath the worker's own lock, before writing.
        if preflight is not None:
            preflight(job)
            current = int(now())
            if current - job["enqueued_at_epoch"] > MAXIMUM_JOB_SECONDS:
                raise ValueError("atlas_preview_expired")
            source_check(job)
        reason = job["reason"]
        if upgrade:
            from .atlas_native_splat import request_gap_preserving_upgrade
            root = Path(str(marker.get("splat_root")))
            if (not root.is_absolute() or root.is_symlink()
                    or not root.resolve().is_relative_to(path.parent.resolve())):
                raise ValueError("atlas_refined_splat_upgrade_invalid")
            request_gap_preserving_upgrade(root, splat_manifest_digest=marker.get("splat_manifest_digest"),
                                           at_epoch=current)
            reason = "refined_splat_gap_preserving_form_untried"
        retries = [*job.get("retries", []), {"reason": reason, "at_epoch": current,
                                             **({"request_id": request_id} if request_id is not None else {})}]
        # The failure's own cleanup completed; the resumed run makes new assets,
        # so cleanup is armed again, keeping what was already deleted.
        cleanup = {**(job.get("provider_cleanup") or {}), "state": "pending", "attempts": 0,
                   "next_attempt_at_epoch": 0}
        job = _transition(job, current, "processing", None, stage="queued", transient_failures=0,
                          next_attempt_at_epoch=current, retries=retries, provider_cleanup=cleanup)
        write_json(path, job)
        return job
