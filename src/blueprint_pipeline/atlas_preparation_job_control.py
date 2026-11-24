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
#: Stops only a new capture moves on: the job ends at once and the website asks the site for a new video. The two
#: mesh codes (2026-11-24) are no finding about the video: the room's mesh, made by the provider from the very splat it
#: is compared with, lies near none of it in the frame of smallest residual (``atlas_mesh_registration``), so this
#: rebuild has no collider, and the room's levelling, its base scene, the room package and the company README all need
#: one. The same inputs always repeat it; before this the job waited, with nothing shipped and nothing asked, until a
#: release or its deadline. A new video is a new rebuild: the only way on that needs no deploy.
#: Two more (stress round 2, 2026-11-24) are facts about the video file that the same file always repeats; before
#: this each took the refusal schedule, ended failed and was reopened by the clock six times, for nothing:
#: ``atlas_insufficient_sharp_static_views``, fewer than two frames of the whole survey sharp and not blown out or
#: too dark (the website words it as frames it could not use), and ``atlas_source_rotation_unsupported``, a file
#: whose display rotation is not a quarter turn (or not a number), which a phone's camera never writes: no upright
#: picture can be told from it, and a video as the phone recorded it moves on.
#: ``atlas_scale_review_unresolved`` too: no reference in the video gave a size. Where a room was built it ships
#: unscaled, flagged (``atlas_unscaled_room``), and the job does not stop; a job that still stops on it (a rebuild,
#: no built background) has no room to ship, the website already asks for a new video with a size reference in
#: view, and before this the clock reopened it six times, each a paid scale review of the same views.
_NEEDS_NEW_CAPTURE = frozenset({"atlas_insufficient_overlapping_distinct_views",
                                "atlas_coverage_anchor_quality_insufficient",
                                "atlas_mesh_not_registered_to_splat", "atlas_mesh_frame_ambiguous",
                                "atlas_insufficient_sharp_static_views", "atlas_source_rotation_unsupported",
                                "atlas_scale_review_unresolved"})
_NOT_RETRYABLE = frozenset({"atlas_preview_expired", "atlas_preview_source_video_unavailable"})
# A planner abstention can be re-examined after a source-review fix. It is not
# proof that the recording must be replaced. Other selection failures keep
# their existing non-retryable boundary.
_NEEDS_TASK_INPUT = frozenset({"atlas_scene_subject_origin_uncertain", "atlas_scene_subject_origin_changed",
                              "atlas_scene_requires_preserved_capture",
                              "atlas_synthetic_subject_requires_preserved_capture"})
_WITHDRAWN = frozenset({"source_revoked", "website_capture_withdrawn", "consent_revoked",
                        "capture_revoked_or_expired"})
#: A reopened review needs room to finish inside the job's own window.
REOPEN_MARGIN_SECONDS = 86_400
#: Written to the job file just before a step runs and removed by the write that follows it.
#: Read under the job lock, it can only be left by a step whose process died before returning.
STEP_IN_FLIGHT = "step_in_flight"
#: The wall a stop's type named (``hard_stops.wall_kind``, or a result's own ``wall``), kept beside its code because a
#: code's words may name none (``website_atlas_pose_disclosure_not_authorized``). It belongs to the stop that wrote it:
#: every other move of the job drops it (``_transition``).
WALL = "wall"
#: Automatic-retry state an operator's reopen starts afresh, with ``transient_failures``.
_AUTOMATIC_RETRY_FIELDS = frozenset({STEP_IN_FLIGHT, "refusal_retries", "refusal_retries_total",
                                     "deterministic_refusal"})
#: The ``retries`` rows of a parked job's automatic reopens (``website_atlas_preview._reopen_parked``): a new
#: release's, or the clock's on ``job_retry_policy``'s refusal schedule.
AUTOMATIC_REOPENS = frozenset({"release_reopened", "parked_retry"})
#: However its stops change, a parked job is reopened by itself at most this many times since an operator last reopened
#: it (three stops' worth of the six-reopen schedule), so nothing loops for ever; each reopen asks every money
#: admission again inside the job's $40, so its spend is bounded too.
MAX_AUTOMATIC_REOPENS = 18


def _trailing_automatic(job: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The job's automatic reopens since an operator last reopened it (an operator starts afresh), oldest first."""
    retries = job.get("retries")
    rows: list[Mapping[str, Any]] = []
    for row in retries if isinstance(retries, list) else ():
        reason = row.get("reason") if isinstance(row, Mapping) else None
        rows = [*rows, row] if reason in AUTOMATIC_REOPENS else []
    return rows


def automatic_reopens(job: Mapping[str, Any]) -> int:
    """How many automatic reopens in a row, since an operator last reopened the job, stopped on the stop it is parked
    on now (``stopped_on``): a job that got past one stop to a new one (a fix, a new release) has the whole schedule
    for the new stop (stress round 2: the sixth reopen got past the pose audit to a later step's stop, which ended the
    job at once, never reopened). A row that names no stop (written before reopens named one) counts as this stop's,
    so an older job is never given more reopens than it had."""
    reason = job.get("reason")
    count = 0
    for row in reversed(_trailing_automatic(job)):
        if row.get("stopped_on") not in (None, reason):
            break
        count += 1
    return count


def reopens_spent(job: Mapping[str, Any]) -> bool:
    """A parked job had the schedule's six automatic reopens for the stop it is on, or ``MAX_AUTOMATIC_REOPENS`` in
    all: it is not reopened by itself again, and its stop is its end (the preparation waiting on it goes on:
    ``website_reconstruction_resume.reopenable``)."""
    from .job_retry_policy import refusal_retry_delay
    return (refusal_retry_delay(automatic_reopens(job)) is None
            or len(_trailing_automatic(job)) >= MAX_AUTOMATIC_REOPENS)



def parked_without_wall(job: Mapping[str, Any]) -> bool:
    """A job ended failed or needs_input with no retry time, on a reason that is no wall: not privacy, rights or
    money (by the stop's type, ``WALL``, or by its words, ``hard_stop_kind``; the spending cap, a withdrawal), not
    the owner's closing of the run, not a stop only a new video or the site's input clears, and not the job's own
    end (expired, source gone)."""
    from .hard_stops import JOB_SPEND_CAP
    from .job_retry_policy import hard_stop_kind, owner_closed
    from .website_job_spend_cap import SPENDING_CAP_REACHED
    from .website_recapture_required import ABSTENTION_CODES
    reason = job.get("reason")
    return (job.get("state") in {"failed", "needs_input"} and not job.get("retry_at_epoch")
            and isinstance(reason, str) and not job.get(WALL) and hard_stop_kind(reason) is None
            and not owner_closed(reason)
            and reason not in (_NOT_RETRYABLE | _NEEDS_NEW_CAPTURE | _NEEDS_TASK_INPUT | _WITHDRAWN
                               | set(ABSTENTION_CODES.values())
                               | {"atlas_scene_planner_abstained", SPENDING_CAP_REACHED, JOB_SPEND_CAP}))


def ended_for_good(job: Mapping[str, Any], *, now: float) -> bool:
    """A job parked on no wall (``parked_without_wall``: failed or needs_input) that nothing reopens by itself any more.

    Such a job is reopened only by the clock or a new release (``website_atlas_preview._reopen_parked``), and only
    while its reopens for that stop are not spent (``reopens_spent``) and its window still leaves a reopened step
    room to finish (``REOPEN_MARGIN_SECONDS``, as ``_may_reopen`` reads it). Past either, its stop is its end: no
    scene can come from it unless an operator reopens it (``retry_failed_job``). A job that waits on a wall, the
    site, a new video or a recheck is never one here: each of those ends by its own path.
    """
    enqueued = job.get("enqueued_at_epoch")
    return parked_without_wall(job) and (
        reopens_spent(job) or isinstance(enqueued, bool) or not isinstance(enqueued, (int, float))
        or now - enqueued > MAXIMUM_JOB_SECONDS - REOPEN_MARGIN_SECONDS)


class _Withdrawn(Exception):
    pass


class PreparationJobBusy(ValueError):
    pass


def _transition(job: Mapping[str, Any], current: int, state: str, reason: str | None,
                **fields: Any) -> dict[str, Any]:
    changed = (job["state"], job["reason"]) != (state, reason) or (
        "archive" in fields and fields["archive"] != job.get("archive"))
    value = {**job, "retry_at_epoch": None, **fields, "state": state, "reason": reason, "updated_at_epoch": current}
    if WALL not in fields:
        value.pop(WALL, None)  # the wall of an earlier stop
    if changed:
        value.update(revision=job["revision"] + 1, revised_at_epoch=current)
    return value


def admission_refusal_code(error: BaseException) -> str | None:
    """The code a refused paid admission (``PaidResourceAdmissionBlocked``, the chokepoint's own type) names, or None.

    Its text is ``paid_resource_admission_blocked:<blockers>``; the code is the account's own (``account_monthly_spend``)
    when one is among them, else the first blocker, so the website is told which limit refused and never the
    chokepoint's name (stress round 2, 2026-11-24).
    """
    from .paid_resource_admission import PaidResourceAdmissionBlocked
    if not isinstance(error, PaidResourceAdmissionBlocked):
        return None
    blockers = [str(code) for code in error.blockers if _CODE.fullmatch(str(code))]
    account = [code for code in blockers if code.startswith("account_monthly_spend_")]
    return (account or blockers or ["paid_resource_admission_blocked"])[0]


def _reason(error: BaseException) -> str:
    """The typed code of a refusal: the owning service's code, never its text."""
    if isinstance(error, _Withdrawn):
        return "website_capture_withdrawn"
    admission = admission_refusal_code(error)
    if admission is not None:
        return admission
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
        # A finished job reopens only to try the reviewed whole-view refined-splat
        # form its adopted minimal splat left untried; any other ready job stays.
        marker = job.get("refined_splat")
        if job["state"] == "ready" and not isinstance(marker, Mapping):
            # Ready before jobs recorded the marker: derive it from the retained records.
            from .atlas_refined_splat_forms import retained_minimal_marker
            marker = retained_minimal_marker(path.parent, job.get("result_digest"))
        upgrade = (job["state"] == "ready" and isinstance(marker, Mapping)
                   and marker.get("gap_preserving_form_untried") is True)
        # Or once to rebuild a refined splat bought from views that disagree (atlas_splat_rebuild).
        rebuild = None
        if job["state"] == "ready" and not upgrade:
            from .atlas_splat_rebuild import rebuild_eligibility
            rebuild = rebuild_eligibility(path.parent, job)
        if ((job["state"] not in {"failed", "needs_input"} and not upgrade and not rebuild)
                or job["reason"] in _NOT_RETRYABLE or job["reason"] in _NEEDS_NEW_CAPTURE):
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
            from .atlas_refined_splat_forms import request_gap_preserving_upgrade
            root = Path(str(marker.get("splat_root")))
            if (not root.is_absolute() or root.is_symlink()
                    or not root.resolve().is_relative_to(path.parent.resolve())):
                raise ValueError("atlas_refined_splat_upgrade_invalid")
            request_gap_preserving_upgrade(root, splat_manifest_digest=marker.get("splat_manifest_digest"),
                                           at_epoch=current)
            reason = "refined_splat_gap_preserving_form_untried"
        elif rebuild:
            from .atlas_splat_rebuild import request_rebuild
            request_rebuild(path.parent, job, at_epoch=current)
            reason = "refined_splat_rebuild_requested"
        retries = [*job.get("retries", []), {"reason": reason, "at_epoch": current,
                                             **({"request_id": request_id} if request_id is not None else {})}]
        # The failure's own cleanup completed; the resumed run makes new assets,
        # so cleanup is armed again, keeping what was already deleted.
        cleanup = {**(job.get("provider_cleanup") or {}), "state": "pending", "attempts": 0,
                   "next_attempt_at_epoch": 0}
        job = {key: value for key, value in job.items() if key not in _AUTOMATIC_RETRY_FIELDS}
        job = _transition(job, current, "processing", None, stage="queued", transient_failures=0,
                          next_attempt_at_epoch=current, retries=retries, provider_cleanup=cleanup)
        write_json(path, job)
        return job
