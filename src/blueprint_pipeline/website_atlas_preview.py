"""Automatic, consented Atlas previews of website uploads, beside the main preparation.

ADP-040 / Day 28; owner decisions D1-D6 of 2026-10-22. Intake only records a job
(``enqueue_preview``, which never raises); this worker alone calls Atlas, on its
own timer, so a slow provider never holds the shared upload listener and the
existing preparation is unchanged. ``BLUEPRINT_WEBSITE_ATLAS_ROUTING``, read by
intake, selects uploads: ``off`` (the default, and any unreadable value),
``allowlist`` (walkthroughs named in ``BLUEPRINT_WEBSITE_ATLAS_CAPTURE_IDS_JSON``)
or ``consented`` (every walkthrough). The WebApp grant derived from that upload's
own consent still gates every paid call.

Each tick moves a job at most one bounded step and leaves it in one plain state:
processing, ready, needs_input, failed or withdrawn. A finished preview is
archived privately under a key named for its capture, so withdrawal deletion
finds it after the workspace is retired; then every Atlas asset the job uploaded
or received is deleted at World Labs, and the WebApp is told each state. A
withdrawal, re-checked before every paid stage, stops the job, archives nothing
and deletes its Atlas assets.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .website_atlas_preview_store import (
    ARTIFACT_KIND, JOB_FILE, LOCK_FILE, MAXIMUM_JOB_SECONDS, SCHEMA, TERMINAL,
    bundle_filename, capture_name_matches_request, capture_withdrawn as _withdrawn,
    read_job as _read)

logger = logging.getLogger(__name__)

ROUTING_ENV = "BLUEPRINT_WEBSITE_ATLAS_ROUTING"
ALLOWLIST_ENV = "BLUEPRINT_WEBSITE_ATLAS_CAPTURE_IDS_JSON"
ROUTING_MODES = ("off", "allowlist", "consented")
PREVIEW_SCHEMA = "website_atlas_preview.v1"
MAXIMUM_TRANSIENT_FAILURES = 8
MAXIMUM_CLEANUP_ATTEMPTS = 12
AUTHORITY_RECHECK_SECONDS = 1800
_CODE = re.compile(r"[a-z][a-z0-9_]{0,99}")
# Refusals an owner or operator can lift: re-checked, not final, until the job expires.
_AWAITING_AUTHORITY = frozenset({
    "website_atlas_source_consent_required", "website_atlas_consent_expired",
    "provider_terms_not_configured_or_changed", "website_atlas_planner_consent_required",
    "website_atlas_runtime_policy_required", "website_atlas_authorization_invalid", "task_brief_missing",
    "website_atlas_api_key_missing", "website_atlas_api_base_url_missing",
    "website_atlas_provider_retention_decision_required", "atlas_runtime_price_review_expired"})
# Coverage concerns in the current plan do not prove that a new recording is required.
_NEEDS_NEW_CAPTURE = frozenset({"atlas_scene_planner_abstained", "atlas_task_and_scene_coverage_gap",
                                "atlas_insufficient_overlapping_distinct_views",
                                "atlas_coverage_anchor_quality_insufficient"})
_NEEDS_TASK_INPUT = frozenset({"atlas_scene_subject_origin_uncertain", "atlas_scene_subject_origin_changed",
                              "atlas_scene_requires_preserved_capture",
                              "atlas_synthetic_subject_requires_preserved_capture"})
_WITHDRAWN = frozenset({"source_revoked", "website_capture_withdrawn", "consent_revoked"})


class _Withdrawn(Exception):
    pass


def routing_mode() -> str:
    mode = os.getenv(ROUTING_ENV, "").strip() or "off"
    if mode not in ROUTING_MODES:
        logger.warning("website_atlas_routing_invalid; routing off")
        return "off"
    return mode


def routed(capture_id: str) -> bool:
    """Whether intake hands this walkthrough to the worker; an unreadable setting routes none."""
    mode = routing_mode()
    if mode != "allowlist":
        return mode == "consented"
    try:
        values = json.loads(os.getenv(ALLOWLIST_ENV, "[]"))
    except ValueError:
        values = None
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        logger.warning("website_atlas_allowlist_invalid; routing none")
        return False
    return capture_id in values


def enqueue_preview(*, descriptor: Any, capture_root: Path, pipeline_dir: Path,
                    source_video: Callable[[], Path | None]) -> dict[str, Any] | None:
    """Record one job for a routed walkthrough; never raises into intake or holds its work.

    ``source_video`` resolves the raw video only once the upload is routed, so an
    unrouted upload is never read or downloaded for this.
    """
    try:
        request_id = str(descriptor.site_submission_id or descriptor.metadata.get("site_submission_id") or "")
        capture_id, scene_id = descriptor.capture_id, descriptor.scene_id
        if (not capture_name_matches_request(request_id, capture_id) or scene_id != f"site-{request_id}"
                or not routed(capture_id)):
            return None
        root = Path(capture_root).resolve()
        if (root.name != capture_id or root.parent.parent.name != scene_id
                or Path(pipeline_dir).resolve() != root / "pipeline"):
            return None
        path = Path(pipeline_dir) / "atlas" / JOB_FILE
        if path.is_file():
            job = _read(path)
            return job if (job.get("request_id"), job.get("scene_id"), job.get("capture_id")) == (
                request_id, scene_id, capture_id) else None
        if _withdrawn(root):
            return None
        resolved = source_video()
        source = None if resolved is None else Path(resolved).resolve()
        usable = source is not None and source.is_file() and source.is_relative_to(root)
        now = int(time.time())
        job = {"schema_version": SCHEMA, "request_id": request_id, "scene_id": scene_id, "capture_id": capture_id,
               "source_video": str(source.relative_to(root)) if usable else None, "routing_mode": routing_mode(),
               "enqueued_at_epoch": now, "state": "processing" if usable else "failed",
               "reason": None if usable else "atlas_preview_source_video_unavailable", "stage": "queued",
               "revision": 1, "revised_at_epoch": now, "updated_at_epoch": now, "next_attempt_at_epoch": now,
               "retry_at_epoch": None,
               "transient_failures": 0, "archive": None, "provider_cleanup": None, "delivered_revision": 0,
               "delivery_attempts": 0, "delivery_next_at_epoch": 0}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Withdrawal deletion holds this same lock, so it exists before any job does.
        (path.parent / LOCK_FILE).touch()
        write_json(path, job)
        return job
    except Exception:  # noqa: BLE001 - the preview is optional; intake continues without it
        logger.warning("website_atlas_preview_enqueue_failed", exc_info=True)
        return None


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
    if head.startswith("website_control_") and "_http_" in head:
        owned = tail.partition(":")[0]
        head = head if owned in {"", "unclassified"} else owned
    code = head.strip().lower().replace("-", "_")
    return code if _CODE.fullmatch(code) else "atlas_preview_" + re.sub(r"[^a-z0-9_]", "_",
                                                                         type(error).__name__.lower())[:60]


#: The owner confirmed or edited the task while a step ran: what it reserved
#: against the old task context is refused, and the next tick reloads the current one.
_TASK_CONTEXT_CHANGED = frozenset({"website_scene_sponsorship_binding_invalid"})


def _transient(error: BaseException) -> bool:
    status = getattr(error, "status", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    if isinstance(error, ValueError):
        return (_reason(error) in _TASK_CONTEXT_CHANGED
                or re.match(r"website_control_[a-z-]+_http_(429|5\d\d)\b", str(error)) is not None)
    return True  # Typed refusals are ValueErrors; transport, storage and lock contention retry.


def _failed_step(job: Mapping[str, Any], error: BaseException, current: int) -> dict[str, Any]:
    code = _reason(error)
    if code in _WITHDRAWN:
        return _transition(job, current, "withdrawn", "website_capture_withdrawn")
    if code in _AWAITING_AUTHORITY:
        return _transition(job, current, "needs_input", code, retry_at_epoch=current + AUTHORITY_RECHECK_SECONDS)
    if code in _NEEDS_NEW_CAPTURE or code in _NEEDS_TASK_INPUT:
        return _transition(job, current, "needs_input", code)
    if _transient(error):
        failures = job["transient_failures"] + 1
        if failures >= MAXIMUM_TRANSIENT_FAILURES:
            return _transition(job, current, "failed", "atlas_preview_retries_exhausted", transient_failures=failures)
        return {**job, "transient_failures": failures, "last_transient_reason": code,
                "next_attempt_at_epoch": current + min(3600, 60 * 2 ** failures)}
    return _transition(job, current, "failed", code)


def _archive(job: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Publish the bundle to the private B2 store, never the legacy fallback, and prove its bytes."""
    from .local_reconstruction_adapters import _sha256_file
    from .task_evaluation_configured_scene_object_store import (
        publish_configured_scene_artifact, verify_dedicated_artifact_store)
    download = result["download"]
    path = Path(download["zip_path"])
    if path.is_symlink() or _sha256_file(path) != download["zip_sha256"]:
        raise ValueError("atlas_preview_bundle_changed")
    verify_dedicated_artifact_store()
    filename = bundle_filename(job["capture_id"], job["request_id"])
    reference = publish_configured_scene_artifact(path=path, artifact_kind=ARTIFACT_KIND, filename=filename)
    if reference["digest"] != download["zip_sha256"]:
        raise ValueError("atlas_preview_archive_mismatch")
    return {"uri": reference["uri"], "digest": reference["digest"], "size_bytes": reference["size_bytes"],
            "filename": filename, "manifest_digest": download["manifest_digest"],
            "remote_verified_at": reference["remote_verified_at"]}


def _step(job: dict[str, Any], *, job_dir: Path, capture_root: Path, now: Callable[[], float]) -> dict[str, Any]:
    current = int(now())
    # Written every live tick: a changing tree also keeps workspace retirement away.
    job = {**job, "last_tick_epoch": current}
    if _withdrawn(capture_root):
        return _transition(job, current, "withdrawn", "website_capture_withdrawn")
    if current < max(job["next_attempt_at_epoch"], job.get("retry_at_epoch") or 0):
        return job
    if current - job["enqueued_at_epoch"] > MAXIMUM_JOB_SECONDS:
        return _transition(job, current, "failed", "atlas_preview_expired")
    from .website_atlas_pipeline import run_atlas_pipeline
    from .website_task_context import load_current_website_task_context

    def checkpoint(_stage: str) -> None:
        if _withdrawn(capture_root):
            raise _Withdrawn()
    try:
        context = load_current_website_task_context(request_id=job["request_id"], scene_id=job["scene_id"],
                                                    capture_id=job["capture_id"], purpose="scene_preparation")
        from .website_recording_sources import resolve_recording_video
        video = resolve_recording_video(task_context=context, capture_root=capture_root,
            source_video=capture_root / job["source_video"])
        result = run_atlas_pipeline(source_video=video, output_root=job_dir,
                                    task_context=context, max_polls=1, checkpoint=checkpoint)
        if result["status"] == "processing":
            return _transition(job, current, "processing", None, stage=result["stage"], transient_failures=0,
                               next_attempt_at_epoch=current)
        completion = result.get('scene_completion') or {}
        if completion.get('status') not in (None, 'completed'):
            # A downloadable source reconstruction is retained evidence, not a
            # successfully repaired scene. Do not publish it as a ready preview.
            reason = _reason(ValueError(completion.get('reason') or 'atlas_preserved_completion_not_admitted'))
            return _transition(job, current, 'needs_input', reason,
                               stage='preserved_scene_completion', result_digest=result.get('digest'))
        if not result.get("download"):
            reason = (_reason(ValueError(result["reason"]))
                      if result.get("stage") == "source_pose" and result.get("reason")
                      else "atlas_background_review_required")
            return _transition(job, current, "needs_input", reason,
                               stage=result.get("stage", "background_review"), result_digest=result.get("digest"))
        # The last paid-stage checkpoint is behind the final provider call, fusion and export.
        checkpoint("archive")
        archive = _archive(job, result)
    except Exception as error:  # noqa: BLE001 - every outcome is a typed job state, never a raise
        return _failed_step(job, error, current)
    if _withdrawn(capture_root):
        # The withdrawal landed during the upload: the copy is taken back before anything uses it.
        return _transition(job, current, "withdrawn", "website_capture_withdrawn", archive=archive,
                           archive_retraction="pending")
    return _transition(job, current, "ready", None, stage="archived", archive=archive, transient_failures=0,
                       result_digest=result["digest"], splat_available=bool(result.get("splat_available")),
                       mesh_available=bool(result.get("mesh_available")))


def atlas_asset_ids(job_dir: Path) -> list[str]:
    """Every Atlas asset the job uploaded or received, from its own retained records."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "asset_id" and isinstance(item, str):
                    found.add(item)
                elif key == "created_assets" and isinstance(item, list):
                    found.update(row for row in item if isinstance(row, str))
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    # A resubmitted pose sets its failed attempt aside, uploads and all, under atlas_pose_attempts/.
    for path in sorted({*job_dir.rglob("atlas_pose/submission.json"), *job_dir.rglob("result.json"),
                        *job_dir.rglob("atlas_pose_attempts/*/submission.json"),
                        *job_dir.rglob("atlas_posed_rgbd_result.json")}):
        if not path.is_symlink():
            walk(json.loads(path.read_text()))
    return sorted(found)


def _pending_operations(job_dir: Path) -> list[str]:
    """Operations submitted but never collected: only World Labs knows their outputs."""
    found = []
    for path in sorted(job_dir.rglob("atlas_pose/submission.json")):
        state = {} if path.is_symlink() else json.loads(path.read_text())
        if state.get("status") == "submitted" and isinstance(state.get("operation_id"), str):
            found.append(state["operation_id"])
        # A duplicate purchase that reconciliation found is never collected either.
        found.extend(operation_id for row in state.get("reconciliations") or []
                     for operation_id in row.get("duplicate_operation_ids") or [] if isinstance(operation_id, str))
    return found


def _operation_outputs(client: Any, operation_id: str) -> list[str] | None:
    """A finished operation's output asset ids; None while it runs (its cancellation is requested)."""
    from . import website_atlas_pose as pose
    status, payload = client.send("GET", f"/operations/{pose._safe_id(operation_id)}")
    if status in {404, 410}:
        return []  # Past the provider's retention nothing names its outputs any more.
    operation = pose._json_object(status, payload)
    if operation.get("done") is not True:
        try:
            client.call("POST", f"/operations/{operation_id}:cancel", {})
        except Exception:  # noqa: BLE001 - best effort; the next attempt polls it again
            pass
        return None
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "assetId" and isinstance(item, str):
                    found.append(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(operation.get("response"))
    return found


def _delete_assets(client: Any, asset_ids: list[str]) -> tuple[list[str], list[str]]:
    """Delete each asset at World Labs (needs ``assets.delete``); one already gone counts as deleted."""
    from . import website_atlas_pose as pose
    deleted, failed = [], []
    for asset_id in asset_ids:
        try:
            client.call("DELETE", f"/assets/{pose._safe_id(asset_id)}")
        except pose.AtlasHttpError as error:
            if error.status != 404:
                failed.append(asset_id)
                continue
        except Exception:  # noqa: BLE001 - recorded and retried; the provider text may echo URLs
            failed.append(asset_id)
            continue
        deleted.append(asset_id)
    return deleted, failed


def _clean_provider(job: dict[str, Any], *, job_dir: Path, now: Callable[[], float],
                    transport: Any = None) -> dict[str, Any]:
    """Delete every Atlas asset the job made, reconciling operations it never collected first."""
    current = int(now())
    # A record missing a field, as an operator reset to "pending" leaves it,
    # resumes from these defaults instead of failing every later tick.
    cleanup = {"state": "pending", "attempts": 0, "deleted": [], "remaining": [], "operations_running": [],
               "next_attempt_at_epoch": 0, **(job.get("provider_cleanup") or {})}
    if cleanup["state"] != "pending" or current < cleanup["next_attempt_at_epoch"]:
        return job
    operations = _pending_operations(job_dir)
    assets = [asset for asset in atlas_asset_ids(job_dir) if asset not in set(cleanup["deleted"])]
    failed, running = [], []
    if operations or assets:
        try:
            from . import website_atlas_pose as pose
            base, key = pose.atlas_api_config()
            client = pose.AtlasClient(base_url=base, api_key=key, transport=transport)
            for operation_id in operations:
                outputs = _operation_outputs(client, operation_id)
                if outputs is None:
                    running.append(operation_id)
                else:
                    assets.extend(asset for asset in outputs if asset not in set(cleanup["deleted"]))
            deleted, failed = _delete_assets(client, sorted(set(assets)))
        except Exception:  # noqa: BLE001 - e.g. no key on this host yet; retried with backoff
            deleted, failed, running = [], sorted(set(assets)), operations
        cleanup = {**cleanup, "deleted": sorted({*cleanup["deleted"], *deleted})}
    attempts = cleanup["attempts"] + 1
    done = not failed and not running
    state = "complete" if done else ("failed" if attempts >= MAXIMUM_CLEANUP_ATTEMPTS else "pending")
    return {**job, "provider_cleanup": {**cleanup, "state": state, "attempts": attempts, "remaining": failed,
                                        "operations_running": running,
                                        "next_attempt_at_epoch": current + min(6 * 3600, 300 * 2 ** attempts)}}


def _retract_archive(job: dict[str, Any]) -> dict[str, Any]:
    """Delete every version of a bundle a withdrawal overtook; retried each tick until gone."""
    if job.get("archive_retraction") != "pending":
        return job
    try:
        from .website_capture_withdrawal_deletion import B2ObjectVersions
        store, uri = B2ObjectVersions.configured(), job["archive"]["uri"]
        for key, version in store.versions(uri):
            store.delete(key, version)
        if store.versions(uri):
            raise ValueError("atlas_preview_archive_versions_remaining")
    except Exception as error:  # noqa: BLE001 - storage errors retry on the next tick
        return {**job, "archive_retraction_error": _reason(error)}
    return {**job, "archive_retraction": "complete", "archive_retraction_error": None}


def preview(job: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp may show: state, plain reason code and the archived bundle's identity."""
    archive = job.get("archive") if job["state"] == "ready" else None
    value = {"schema_version": PREVIEW_SCHEMA, "capture_id": job["capture_id"], "revision": job["revision"],
             "state": job["state"], "reason": job["reason"], "updated_at_epoch": int(job["revised_at_epoch"]),
             "bundle": None if archive is None else {key: archive[key] for key in ("digest", "size_bytes", "filename")},
             "splat_available": bool(job.get("splat_available")), "mesh_available": bool(job.get("mesh_available")),
             "claim_ceiling": "development_only", "scale_status": "unscaled_provider_world_units"}
    value["preview_digest"] = cross_runtime_canonical_digest(value)
    return value


def _deliver(job: dict[str, Any], *, now: Callable[[], float]) -> dict[str, Any]:
    current = int(now())
    if job["delivered_revision"] >= job["revision"] or current < job["delivery_next_at_epoch"]:
        return job
    from .website_task_context import website_webapp_request
    value = preview(job)
    error = None
    try:
        acceptance = website_webapp_request(capture_id=job["capture_id"], operation="atlas-preview", payload={
            "request_id": job["request_id"], "scene_id": job["scene_id"], "preview": value})
        if (acceptance.get("schema_version") != "website_atlas_preview_acceptance.v1"
                or acceptance.get("accepted") is not True or acceptance.get("revision") != job["revision"]
                or acceptance.get("preview_digest") != value["preview_digest"]):
            error = "website_atlas_preview_acceptance_invalid"
    except Exception as exc:  # noqa: BLE001 - delivery retries; the job state stands
        error = _reason(exc)
        if error in _WITHDRAWN and job["state"] != "withdrawn":
            return _transition(job, current, "withdrawn", "website_capture_withdrawn")
    if error is None:
        return {**job, "delivered_revision": job["revision"], "delivery_attempts": 0, "delivery_next_at_epoch": 0,
                "delivery_error": None}
    attempts = job["delivery_attempts"] + 1
    return {**job, "delivery_error": error, "delivery_attempts": attempts,
            "delivery_next_at_epoch": current + min(3600, 30 * 2 ** attempts)}


def advance_job(path: Path, *, now: Callable[[], float] = time.time, transport: Any = None) -> str:
    """One bounded step for one job: Atlas progress, provider cleanup, WebApp delivery."""
    job_dir, capture_root = path.parent, path.parents[2]
    with (job_dir / LOCK_FILE).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy"
        job = _read(path)
        before = json.dumps(job, sort_keys=True)
        if job["state"] not in TERMINAL or job.get("retry_at_epoch"):
            job = _step(job, job_dir=job_dir, capture_root=capture_root, now=now)
        if job["state"] in TERMINAL and not job.get("retry_at_epoch"):
            job = _retract_archive(job)
            job = _clean_provider(job, job_dir=job_dir, now=now, transport=transport)
        job = _deliver(job, now=now)
        if json.dumps(job, sort_keys=True) != before:
            write_json(path, job)
        return job["state"]


#: Failures a retry cannot cure: the job outlived its window, or its source is gone.
_NOT_RETRYABLE = frozenset({"atlas_preview_expired", "atlas_preview_source_video_unavailable"})


def retry_failed_job(path: Path, *, now: Callable[[], float] = time.time,
                     preflight: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Resume one failed or stopped job once its cause is fixed in the deployed release.

    An operator action, never automatic. The next tick resumes at the first
    unfinished stage, so the survey, plans and other receipts already bought are
    reused, not bought again. A job stopped for input (a background that needs
    review) resumes the same way. A withdrawn capture, a job that only a new
    capture can help, or one that cannot recover, is refused.
    """
    lock_fd = os.open(path.parent / LOCK_FILE, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(lock_fd, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("atlas_preview_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("atlas_preview_job_busy") from None
        job = _read(path)
        if (job["state"] not in {"failed", "needs_input"} or job["reason"] in _NOT_RETRYABLE
                or job["reason"] in _NEEDS_NEW_CAPTURE):
            raise ValueError("atlas_preview_retry_requires_recoverable_failure")
        if _withdrawn(path.parents[2]):
            raise ValueError("website_capture_withdrawn")
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
            if _withdrawn(path.parents[2]):
                raise ValueError("website_capture_withdrawn")
        retries = [*job.get("retries", []), {"reason": job["reason"], "at_epoch": current}]
        # The failure's own cleanup completed; the resumed run makes new assets,
        # so cleanup is armed again, keeping what was already deleted.
        cleanup = {**(job.get("provider_cleanup") or {}), "state": "pending", "attempts": 0,
                   "next_attempt_at_epoch": 0}
        job = _transition(job, current, "processing", None, stage="queued", transient_failures=0,
                          next_attempt_at_epoch=current, retries=retries, provider_cleanup=cleanup)
        write_json(path, job)
        return job


def tick(storage_root: Path, *, now: Callable[[], float] = time.time) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(Path(storage_root).glob(f"*/scenes/site-*/captures/*/pipeline/atlas/{JOB_FILE}")):
        if not capture_name_matches_request(path.parents[4].name.removeprefix("site-"), path.parents[2].name):
            continue
        try:
            outcome = advance_job(path, now=now)
        except Exception as error:  # noqa: BLE001 - one unreadable job never stops the others
            logger.warning("website_atlas_preview_job_unreadable type=%s", type(error).__name__)
            outcome = "unreadable"
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    advance = commands.add_parser("tick", help="advance every preview job one bounded step")
    advance.add_argument("--storage-root", type=Path, required=True)
    retry = commands.add_parser("retry", help="resume one failed job after its cause is fixed (run as the service user)")
    retry.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "retry":
        job = retry_failed_job(args.job)
        print(json.dumps({key: job[key] for key in ("capture_id", "state", "revision", "retries")},
                         sort_keys=True), flush=True)
        return 0
    print(json.dumps(tick(args.storage_root), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
