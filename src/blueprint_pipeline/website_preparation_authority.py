"""Reopen and validate bounded website preparation authority.

Structural consent/proposal checks live in website_preparation_contracts so
read-only consumers cannot reach this module's current-grant/worker dependencies.
A caller-supplied grant alone cannot confer authenticated issuance or paid admission.
"""

from time import time as current_time

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .website_preparation_contracts import (
    validate_preparation_authority as validate_preparation_authority,
    validate_preparation_request as validate_preparation_request,
)


def require_retained_preparation_authority(*, request, queue_root, now):
    """Reopen server-owned source references; context-only consent is insufficient.

    Signed Web intake and current source registration are both required. The
    read-only current grant check cannot create or renew an allowance.
    """
    from .website_scene_source_registry_paths import binding_root, _index_path
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .website_task_context import (
        load_current_website_task_context,
        load_website_scene_sponsorship,
    )

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
    authority = validate_preparation_request(
        request=request,
        task_context=context,
        authority=prepared.get("website_preparation_authority") or {},
        now=now,
    )
    current = load_current_website_task_context(
        request_id=context["request_id"],
        scene_id=context["scene_id"],
        capture_id=context["capture_id"],
        purpose="scene_preparation",
    )
    if current != context:
        raise ValueError("website_preparation_source_context_changed")
    fresh = load_website_scene_sponsorship(task_context=current, now=now, create=False)
    if fresh != authority:
        raise ValueError("website_preparation_source_authority_changed")
    # Signed reads may cross the expiry boundary. Recheck the actual clock
    # and return it for the caller's effective-window/action checks.
    moment = current_time()
    validate_preparation_request(request=request, task_context=current, authority=fresh, now=moment)
    return moment
