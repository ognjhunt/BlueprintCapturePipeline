"""Reopen and validate bounded website preparation authority.

Structural consent/proposal checks live in website_preparation_contracts so
read-only consumers cannot reach this module's current-grant/worker dependencies.
A caller-supplied grant alone cannot confer authenticated issuance or paid admission.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from time import time as current_time

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .website_preparation_contracts import (
    validate_preparation_authority as validate_preparation_authority,
    validate_preparation_request as validate_preparation_request,
)

# False only inside ``retained_sources_only``.
_CURRENT_GRANT_READS = ContextVar("website_preparation_current_grant_reads", default=True)


@contextmanager
def retained_sources_only():
    """Check the retained sources but skip the current-grant reads, which go to the WebApp.

    For the launch catalog's projection only. It runs in the ExecStartPre of every
    guarded unit, so a read there let one WebApp outage, or one job its site edited,
    stop all of them, and its network request kept the runtime env guard from
    caching the pass. Every paid path keeps the reads and refuses on them.
    """
    token = _CURRENT_GRANT_READS.set(False)
    try:
        yield
    finally:
        _CURRENT_GRANT_READS.reset(token)


def _intent_valid_until(*, request, queue_root):
    """The staged intent's verified extended expiry, or None when nothing extends it.

    Only an intent sealing exactly this request counts; its window can carry an
    owner-approved, time-only extension that the grant's raw expiry predates.
    Without one, every check below is exactly the grant's own expiry.
    """
    from pathlib import Path
    from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry
    from .task_evaluation_scene_intent_contracts import _read, canonical_digest

    identity = canonical_digest({"owner": request["owner"], "submission_id": request["submission_id"]})
    directory = Path(queue_root) / ("scene-" + identity.removeprefix("sha256:"))
    path = directory / "intent.json"
    if directory.is_symlink() or not path.is_file():
        return None
    intent = _read(path, "intent_digest")
    if intent.get("request") != request:
        return None
    expiry = effective_execution_expiry(directory, intent)
    return expiry if expiry > request["execution"]["expires_at_epoch"] else None


def _retained_records(*, request, queue_root):
    """The registration, preparation and task context retained for ``request``, digest-checked."""
    from .website_scene_source_registry_paths import binding_root, _index_path
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read

    root = binding_root({"intent_root": str(queue_root)})
    path = _index_path(root, request)
    if not path.is_file() or path.is_symlink():
        raise ValueError("website_preparation_source_missing")
    registration = read(path, digest_field="registration_digest")
    if (
        registration.get("schema_version") != "website_scene_source_registration.v1"
        or registration.get("request_digest") != cross_runtime_canonical_digest(request)
        or registration.get("execution_authority_granted") is not False
    ):
        raise ValueError("website_preparation_source_invalid")
    refs = registration["references"]
    if not {"preparation", "runtime_inputs", "task_context"}.issubset(refs):
        raise ValueError("website_preparation_source_references_missing")
    for ref in refs.values():
        checked_file(ref["path"], ref)
    prepared = read(
        checked_file(refs["preparation"]["path"], refs["preparation"]), digest_field="digest"
    )
    context = read(
        checked_file(refs["task_context"]["path"], refs["task_context"]),
        digest_field="context_digest",
    )
    if (
        prepared.get("intake_request") != request
        or prepared.get("binding", {}).get("task_context_digest") != context["context_digest"]
    ):
        raise ValueError("website_preparation_source_request_changed")
    return registration, prepared, context


def _retained_sources(*, request, queue_root, now):
    """The retained, digest-bound sources for ``request``, read on this host only.

    Returns the task context, the grant it was prepared under and the intent's
    verified extended expiry (``None`` when nothing extends it).
    """
    _registration, prepared, context = _retained_records(request=request, queue_root=queue_root)
    valid_until = _intent_valid_until(request=request, queue_root=queue_root)
    authority = validate_preparation_request(
        request=request,
        task_context=context,
        authority=prepared.get("website_preparation_authority") or {},
        now=now,
        valid_until=valid_until,
    )
    return context, authority, valid_until


#: Where a capture's delivery retains the grant it ran under, beside its preparation
#: (``website_task_evidence``, ``website_task_context.RETAINED_GRANT_FILE``).
CAPTURE_GRANT_FILE = "pipeline/website_scene_sponsorship.json"
_MAX_CAPTURE_GRANT_BYTES = 64 * 1024


def _capture_retained_grant(registration, preparation_path):
    """The grant the capture's delivery retained, read only beside this job's own preparation.

    The registration is sealed, and it names the capture root this
    preparation was written under; a grant anywhere else is not this job's.
    The Pipeline writes the file from the WebApp's signed read; no site or
    deployment company writes to it. The caller validates it like any grant.
    """
    from pathlib import Path
    import json

    capture_root = Path(str((registration.get("capture_source") or {}).get("canonical_path") or ""))
    pipeline = capture_root / "pipeline"
    if (not capture_root.is_absolute() or ".." in capture_root.parts
            or not Path(preparation_path).resolve().is_relative_to(pipeline.resolve())):
        raise ValueError("website_preparation_retained_grant_unbound")
    path = capture_root / CAPTURE_GRANT_FILE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_CAPTURE_GRANT_BYTES:
        raise ValueError("website_preparation_retained_grant_missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("website_preparation_retained_grant_missing")
    return value


def retained_extension_grant(*, request, queue_root, now):
    """The job's own retained grant, for the standing budget extension only.

    First the grant the preparation retained, exactly as ``_retained_sources``
    reads it (a site that has not confirmed its task, a revision, a renewal).
    A preparation that retained none (a site that confirmed its task before
    preparation, 2026-11-12: the only reason the reference job could not
    extend) falls back to the grant its capture's delivery retained, validated
    against this exact request and task context by the same checks, with the
    confirmed-task scope (``confirmed_preparation_scope``). It grants nothing
    but the read of the grant's ``automatic_budget_extension`` flag: consent,
    rights and preparation authority are reopened elsewhere, unchanged.
    """
    try:
        return _retained_sources(request=request, queue_root=queue_root, now=now)[1]
    except ValueError:
        registration, prepared, context = _retained_records(request=request, queue_root=queue_root)
        if prepared.get("website_preparation_authority") is not None:
            raise
    grant = _capture_retained_grant(registration, registration["references"]["preparation"]["path"])
    return validate_preparation_request(
        request=request, task_context=context, authority=grant, now=now,
        valid_until=_intent_valid_until(request=request, queue_root=queue_root), confirmed_task=True)


def require_retained_preparation_authority(*, request, queue_root, now):
    """Reopen server-owned source references; context-only consent is insufficient.

    Signed Web intake and current source registration are both required. The
    read-only current grant check cannot create or renew an allowance. A
    verified time-only extension of the bound intent moves only the "still
    valid now" checks; the grant itself is unchanged. Inside
    ``retained_sources_only`` it stops after the retained sources and returns
    ``now``.
    """
    context, authority, valid_until = _retained_sources(request=request, queue_root=queue_root, now=now)
    if not _CURRENT_GRANT_READS.get():
        return now
    from .website_task_context import (
        load_current_website_task_context,
        load_website_scene_sponsorship,
    )

    current = load_current_website_task_context(
        request_id=context["request_id"],
        scene_id=context["scene_id"],
        capture_id=context["capture_id"],
        purpose="scene_preparation",
    )
    if current != context:
        raise ValueError("website_preparation_source_context_changed")
    fresh = load_website_scene_sponsorship(task_context=current, now=now, create=False,
                                           **({} if valid_until is None else {"valid_until": valid_until}))
    if fresh != authority:
        raise ValueError("website_preparation_source_authority_changed")
    # Signed reads may cross the expiry boundary. Recheck the actual clock
    # and return it for the caller's effective-window/action checks.
    moment = current_time()
    validate_preparation_request(request=request, task_context=current, authority=fresh, now=moment,
                                 valid_until=valid_until)
    return moment
