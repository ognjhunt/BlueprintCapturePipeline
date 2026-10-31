"""Owned-upload source adapter for the existing Atlas preparation lifecycle.

ADP-009D day-14 receipts/day-28 replay, development-only. Queueing is an
explicit authenticated owner action; reading a plan/configuration never queues
or grants anything. Source bytes and worker paths come only from the accepted
Pipeline upload receipt. No site or website-recording consent is invented.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import write_json
from .owned_capture_atlas_runtime import bind_owned_atlas_source, owned_atlas_source_checkpoint
from .reconstruction_control_plane import _source_binding
from .website_atlas_preview import (
    _failed_step, _transition, advance_preparation_job, advance_preparation_step, poll_jobs,
)
from .website_atlas_preview_store import MAXIMUM_JOB_SECONDS

from .owned_capture_atlas_control import (
    _job_path, _check_job_source, _refresh_source, projection,
    JOB_FILE, read_job as read_job, enqueue_owned_atlas as enqueue_owned_atlas,
    inspect_owned_atlas as inspect_owned_atlas, open_owned_bundle as open_owned_bundle,
    retry_owned_atlas as retry_owned_atlas,
)

logger = logging.getLogger(__name__)


def _step(job: dict[str, Any], *, state_root: Path, capture_store_root: Path, path: Path,
          now: Callable[[], float]) -> dict[str, Any]:
    current = int(now())
    job = {**job, "last_tick_epoch": current}
    if current < max(job["next_attempt_at_epoch"], job.get("retry_at_epoch") or 0):
        return job
    if current - job["enqueued_at_epoch"] > MAXIMUM_JOB_SECONDS:
        return _transition(job, current, "failed", "atlas_preview_expired")
    from .owned_capture_atlas_project import bind_owned_atlas_project
    from .website_atlas_pipeline import load_configuration
    try:
        with bind_owned_atlas_source(state_root=state_root, capture_store_root=capture_store_root,
                                     plan_id=job["plan_id"]) as context:
            _check_job_source(job, context=context, path=path)
            checkpoint = owned_atlas_source_checkpoint(context)
            checkpoint("owned_atlas_step")
            configuration = load_configuration(context, job["source_video_digest"], create=False)
            if configuration["authority"]["authorization_digest"] != job["authorization_digest"]:
                raise ValueError("owned_capture_atlas_job_authority_changed")
            with bind_owned_atlas_project(task_context=context, configuration=configuration,
                                          expected_owner_uid=job["owner_user_id"]):
                source = _source_binding(capture_store_root=capture_store_root,
                                         capture_session_id=context["capture_id"], intake_id=context["intake_id"])
                return advance_preparation_step(job, job_dir=path.parent, source_video=source["object_path"],
                    task_context=context, checkpoint=checkpoint, now=now)
    except Exception as error:  # noqa: BLE001 - shared bounded backoff; no uncertain resubmission here
        return _failed_step(job, error, current)


def _deliver(job: dict[str, Any], *, path: Path) -> dict[str, Any]:
    # The normal owner status route reads this retained projection. No separate
    # website record or extra callback authority is invented for an owned job.
    value = projection(job)
    write_json(path.parent / "owned_status.json", value)
    return {**job, "delivered_revision": job["revision"]}


def advance_job(*, state_root: Path, capture_store_root: Path, plan_id: str,
                now: Callable[[], float] = time.time, transport: Any = None) -> str:
    path = _job_path(state_root, plan_id)
    return advance_preparation_job(path, reader=read_job,
        refresh=lambda job: _refresh_source(job, state_root=state_root, capture_store_root=capture_store_root,
                                            path=path, now=now),
        step=lambda job: _step(job, state_root=state_root, capture_store_root=capture_store_root, path=path, now=now),
        deliver=lambda job: _deliver(job, path=path), now=now, transport=transport)


def tick(*, state_root: Path, capture_store_root: Path, now: Callable[[], float] = time.time) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(Path(state_root).glob(f"plans/reconstruction-atlas-*/atlas/{JOB_FILE}")):
        try:
            state = advance_job(state_root=state_root, capture_store_root=capture_store_root,
                                plan_id=path.parent.parent.name, now=now)
        except Exception as error:  # noqa: BLE001 - one unreadable job cannot stop other accepted uploads
            logger.warning("owned_capture_atlas_job_unreadable type=%s", type(error).__name__)
            state = "unreadable"
        counts[state] = counts.get(state, 0) + 1
    return counts


def _roots_from_environment() -> tuple[Path, Path] | None:
    from .capture_upload_intake import CAPTURE_UPLOAD_STORE_ROOT_ENV
    from .live_pipeline_control_plane import CONTROL_PLANE_OUTPUT_PATH_ENV
    from .live_pipeline_intake_runtime_controls import DEFAULT_MANIFEST_PATH, INTAKE_WORK_DIR_ENV
    capture_store = os.getenv(CAPTURE_UPLOAD_STORE_ROOT_ENV, "").strip()
    manifest = Path(os.getenv(CONTROL_PLANE_OUTPUT_PATH_ENV) or DEFAULT_MANIFEST_PATH).expanduser()
    if not capture_store or not manifest.is_file():
        return None
    work = Path(os.getenv(INTAKE_WORK_DIR_ENV) or manifest.parent / "incoming_webapp_job_requests").expanduser()
    return work.resolve() / "reconstruction_control_plane", Path(capture_store).expanduser().resolve()


def tick_from_environment(*, now: Callable[[], float] = time.time) -> dict[str, int]:
    """The existing timer also discovers explicitly queued owned jobs; no new service."""
    roots = _roots_from_environment()
    return {} if roots is None else tick(state_root=roots[0], capture_store_root=roots[1], now=now)


def poll_from_environment(*, now: Callable[[], float] = time.time, transport: Any = None) -> dict[str, int]:
    """The provider poll over queued owned jobs (``website_atlas_preview.poll_jobs``); reads only."""
    roots = _roots_from_environment()
    if roots is None:
        return {}
    state_root, capture_store_root = roots

    def source_live(job: dict[str, Any], path: Path) -> bool:
        refreshed = _refresh_source(job, state_root=state_root, capture_store_root=capture_store_root,
                                    path=path, now=now)
        return refreshed["state"] == "processing"
    paths = sorted(Path(state_root).glob(f"plans/reconstruction-atlas-*/atlas/{JOB_FILE}"))
    return poll_jobs([(path, read_job, lambda job, path=path: source_live(job, path)) for path in paths],
                     now=now, transport=transport)
