"""Owned Atlas queue and retained owner reads; never advance provider execution."""
from __future__ import annotations
import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from .common import write_json
from .core.security_controls import strict_identifier
from .decision_evidence_contracts import canonical_json, cross_runtime_canonical_digest
from .owned_capture_atlas_runtime import bind_owned_atlas_source, owned_atlas_source_checkpoint
from .reconstruction_control_plane import _run_root, _write_immutable
from .atlas_preparation_job_control import _reason, _transition, preparation_job_lock, retry_preparation_job
from .website_task_context import load_atlas_configuration as load_configuration
SCHEMA = "owned_capture_atlas_job.v1"
JOB_FILE = "owned_job.json"
_PLAN = re.compile(r"reconstruction-atlas-[0-9a-f]{24}")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")

def _job_path(state_root: Path, plan_id: str) -> Path:
    if not _PLAN.fullmatch(plan_id):
        raise ValueError("owned_capture_atlas_plan_id_invalid")
    return _run_root(state_root, plan_id) / "atlas" / JOB_FILE


def read_job(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink():
        raise ValueError("owned_capture_atlas_job_unsafe")
    job = json.loads(path.read_text())
    if (not isinstance(job, dict) or job.get("schema_version") != SCHEMA
            or job.get("plan_id") != path.parent.parent.name
            or not _PLAN.fullmatch(str(job.get("plan_id") or ""))
            or job.get("request_id") != job.get("plan_id")
            or job.get("state") not in {"processing", "ready", "needs_input", "failed", "withdrawn"}
            or not all(_SHA.fullmatch(str(job.get(key) or "")) for key in
                       ("plan_digest", "task_context_digest", "source_video_digest", "authorization_digest"))
            or not isinstance(job.get("owner_user_id"), str) or not job["owner_user_id"]):
        raise ValueError("owned_capture_atlas_job_invalid")
    return job


def enqueue_owned_atlas(*, state_root: Path, capture_store_root: Path, plan_id: str,
                        task_context_digest: str, source_video_digest: str, owner_user_id: str,
                        idempotency_key: str, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Record one source-bound job after noncreating authority checks; no provider call.

    ``owner_user_id`` is supplied by the authenticated WebApp issuer, never a
    value inferred from task text. Every later purchase rechecks source,
    company terms, the original finite allowance and the canonical allocator.
    """
    from .owned_capture_atlas_project import validate_owned_atlas_configuration

    owner = strict_identifier(owner_user_id, field="owner_user_id", max_length=192)
    key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
    path = _job_path(state_root, plan_id)
    with bind_owned_atlas_source(state_root=state_root, capture_store_root=capture_store_root,
                                 plan_id=plan_id) as context:
        checkpoint = owned_atlas_source_checkpoint(context)
        checkpoint("owned_atlas_enqueue")
        if (context["context_digest"] != task_context_digest
                or context["source_capture"]["capture_digest"] != source_video_digest):
            raise ValueError("owned_capture_atlas_queue_binding_invalid")
        configuration = load_configuration(context, source_video_digest, create=False)
        validate_owned_atlas_configuration(task_context=context, configuration=configuration,
                                           expected_owner_uid=owner, now=now())
        command = {"schema_version": "owned_capture_atlas_queue_binding.v1", "plan_id": plan_id,
                   "capture_id": context["capture_id"], "intake_id": context["intake_id"],
                   "owner_user_id": owner, "task_context_digest": task_context_digest,
                   "source_video_digest": source_video_digest,
                   "authorization_digest": configuration["authority"]["authorization_digest"]}
        request_key = hashlib.sha256(f'{context["capture_id"]}\0{context["intake_id"]}\0{key}'.encode()).hexdigest()
        _write_immutable(Path(state_root) / "owned_atlas_execution_requests" / f"{request_key}.json", command)
        path.parent.mkdir(parents=True, exist_ok=True)
        with preparation_job_lock(path.parent):
            _write_immutable(path.parent / "execution_binding.json", command)
            if not path.exists():
                # Reopen after remote configuration/lock acquisition. No queued work
                # can outlive a revocation which landed during those checks.
                checkpoint("owned_atlas_enqueue_commit")
                current = int(now())
                plan_path = path.parent.parent / "artifacts" / "owned_atlas_preparation_plan.json"
                plan = json.loads(plan_path.read_text())
                job = {**command, "schema_version": SCHEMA, "request_id": plan_id,
                       "plan_digest": plan["plan_digest"], "state": "processing", "reason": None,
                       "stage": "queued", "enqueued_at_epoch": current, "revision": 1,
                       "revised_at_epoch": current, "updated_at_epoch": current,
                       "next_attempt_at_epoch": current, "retry_at_epoch": None, "transient_failures": 0,
                       "archive": None, "provider_cleanup": None, "delivered_revision": 0,
                       "delivery_attempts": 0, "delivery_next_at_epoch": 0}
                write_json(path, job)
    return inspect_owned_atlas(state_root=state_root, capture_store_root=capture_store_root,
                                plan_id=plan_id, owner_user_id=owner)


def _check_job_source(job: Mapping[str, Any], *, context: Mapping[str, Any], path: Path) -> None:
    plan = json.loads((path.parent.parent / "artifacts" / "owned_atlas_preparation_plan.json").read_text())
    if (job["capture_id"] != context["capture_id"] or job["intake_id"] != context["intake_id"]
            or job["task_context_digest"] != context["context_digest"]
            or job["source_video_digest"] != context["source_capture"]["capture_digest"]
            or job["plan_digest"] != plan["plan_digest"]):
        raise ValueError("owned_capture_atlas_job_source_changed")


def _refresh_source(job: dict[str, Any], *, state_root: Path, capture_store_root: Path,
                    path: Path, now: Callable[[], float]) -> dict[str, Any]:
    try:
        with bind_owned_atlas_source(state_root=state_root, capture_store_root=capture_store_root,
                                     plan_id=job["plan_id"]) as context:
            _check_job_source(job, context=context, path=path)
            owned_atlas_source_checkpoint(context)("owned_atlas_tick")
    except Exception as error:  # noqa: BLE001 - preserve the failing typed boundary and clean retained assets
        code = _reason(error)
        state = "withdrawn" if code == "capture_revoked_or_expired" else "needs_input"
        return _transition(job, int(now()), state, code,
                           **({"archive_retraction": "pending"} if job.get("archive") else {}))
    return job


def projection(job: Mapping[str, Any]) -> dict[str, Any]:
    archive = job.get("archive") if job["state"] == "ready" else None
    value = {"schema_version": "owned_capture_atlas_execution.v1",
             **{key: job[key] for key in ("plan_id", "capture_id", "intake_id", "plan_digest",
                "task_context_digest", "source_video_digest", "authorization_digest", "state", "stage", "reason", "revision")},
             "updated_at_epoch": job["updated_at_epoch"], "reconstruction": None,
             "bundle": None if archive is None else {key: archive[key] for key in
                 ("digest", "size_bytes", "filename", "manifest_digest")},
             "splat_available": bool(job.get("splat_available")), "mesh_available": bool(job.get("mesh_available")),
             "provider_cleanup_state": (job.get("provider_cleanup") or {}).get("state", "not_started"),
             "claim_ceiling": "development_only", "scale_status": "unscaled_provider_world_units",
             "proof_boundary": {"derived_reconstruction_upgrades_raw_capture": False,
                                "metric_or_collision_qualification_established": False,
                                "robot_execution_established": False, "physical_task_success_established": False}}
    value["execution_digest"] = cross_runtime_canonical_digest(value)
    return value


def inspect_owned_atlas(*, state_root: Path, capture_store_root: Path, plan_id: str,
                        owner_user_id: str) -> dict[str, Any]:
    path = _job_path(state_root, plan_id)
    job = read_job(path)
    if job["owner_user_id"] != owner_user_id:
        raise ValueError("owned_capture_atlas_owner_mismatch")
    # An owner read may retain a reason after revocation, but it never exposes
    # a bundle whose source is now gone or changed.
    job = _refresh_source(job, state_root=state_root, capture_store_root=capture_store_root,
                          path=path, now=time.time)
    value = projection(job)
    if job["state"] == "ready":
        from .owned_capture_atlas_reconstruction import adopt_owned_atlas_reconstruction
        retained = adopt_owned_atlas_reconstruction(state_root=state_root,
            capture_store_root=capture_store_root, plan_id=plan_id, expected_owner_user_id=owner_user_id)
        # Adoption reopens source/job/bytes beneath the shared lifecycle lock.
        # A status read performs local verification only, never a provider tick.
        value["reconstruction"] = {"plan_id": plan_id, "reconstruction_plan": retained["reconstruction_plan"],
                                   "reconstruction_plan_canonical_json": canonical_json({
                                       key: item for key, item in retained["reconstruction_plan"].items()
                                       if key != "reconstruction_plan_digest"}),
                                   "execution_result": retained["execution_result"]}
        value["execution_digest"] = cross_runtime_canonical_digest(value, digest_field="execution_digest")
    return value


def open_owned_bundle(*, state_root: Path, capture_store_root: Path, plan_id: str,
                      owner_user_id: str, digest: str, client: Any = None,
                      bucket: str | None = None) -> tuple[Any, int]:
    from .website_atlas_preview_store import PreviewUnavailable, open_archived_bundle
    value = inspect_owned_atlas(state_root=state_root, capture_store_root=capture_store_root,
                                plan_id=plan_id, owner_user_id=owner_user_id)
    bundle = value.get("bundle")
    if value["state"] != "ready" or not bundle or bundle["digest"] != digest:
        raise PreviewUnavailable("owned_capture_atlas_bundle_unavailable")
    return open_archived_bundle(filename=bundle["filename"], digest=digest, client=client, bucket=bucket)


def retry_owned_atlas(*, state_root: Path, capture_store_root: Path, plan_id: str,
                      task_context_digest: str, source_video_digest: str, owner_user_id: str,
                      idempotency_key: str, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Resume a repaired stage, never renew a grant or silently change the job."""
    from .owned_capture_atlas_project import validate_owned_atlas_configuration
    key = strict_identifier(idempotency_key, field="idempotency_key", max_length=192)
    owner = strict_identifier(owner_user_id, field="owner_user_id", max_length=192)
    path = _job_path(state_root, plan_id)

    def source_check(job):
        if (job["owner_user_id"] != owner or job["task_context_digest"] != task_context_digest
                or job["source_video_digest"] != source_video_digest):
            raise ValueError("owned_capture_atlas_retry_binding_invalid")
        with bind_owned_atlas_source(state_root=state_root, capture_store_root=capture_store_root,
                                     plan_id=plan_id) as context:
            _check_job_source(job, context=context, path=path)
            owned_atlas_source_checkpoint(context)("owned_atlas_retry")

    def preflight(job):
        with bind_owned_atlas_source(state_root=state_root, capture_store_root=capture_store_root,
                                     plan_id=plan_id) as context:
            configuration = load_configuration(context, source_video_digest, create=False)
            validate_owned_atlas_configuration(context, configuration, expected_owner_uid=owner, now=now())
            if configuration["authority"]["authorization_digest"] != job["authorization_digest"]:
                raise ValueError("owned_capture_atlas_job_authority_changed")

    retry_preparation_job(path, reader=read_job, source_check=source_check, preflight=preflight,
                                 request_id=key, now=now)
    return inspect_owned_atlas(state_root=state_root, capture_store_root=capture_store_root,
                               plan_id=plan_id, owner_user_id=owner)
