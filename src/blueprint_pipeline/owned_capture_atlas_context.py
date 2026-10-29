"""Carry an owned upload's actual source and task into Atlas preparation.

The reconstruction controller supplies the result of its existing
``_source_binding`` read. This is source context, never a recording attestation,
provider-terms acceptance, spend grant, or metric/physics qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .core.security_controls import strict_identifier
from .decision_evidence_contracts import cross_runtime_canonical_digest

SCHEMA = "owned_capture_atlas_context.v1"
_FIELDS = {
    "schema_version", "request_id", "capture_id", "intake_id", "scene_id",
    "source_scene_id", "scene_identity_basis", "purpose", "description",
    "task_specification", "task_family", "task_confirmation_status", "source_capture",
    "governance", "permitted_reconstruction_providers", "permitted_evidence_uses",
    "coordinate_frame_declaration", "authority_issued", "context_digest",
}


def build_owned_capture_atlas_context(
    *, source: Mapping[str, Any], plan_id: str
) -> dict[str, Any]:
    """Use the complete task and governance from the verified intake envelope.

    ``source`` must come from reconstruction_control_plane._source_binding;
    caller-supplied paths or reconstructed evidence are not source authority.
    A missing task is an input gap, not an unsupported scene/task category.
    """
    envelope, receipt = source["envelope"], source["receipt"]
    task = envelope.get("known_task_specification")
    if not isinstance(task, Mapping) or not isinstance(task.get("description"), str) or not task["description"].strip():
        raise ValueError("owned_capture_atlas_task_description_required")
    context: dict[str, Any] = {
        "schema_version": SCHEMA,
        "request_id": strict_identifier(plan_id, field="plan_id", max_length=192),
        "capture_id": receipt["capture_session_id"],
        "intake_id": receipt["intake_id"],
        "scene_id": envelope.get("scene_id") or receipt["capture_session_id"],
        "source_scene_id": envelope.get("scene_id"),
        "scene_identity_basis": "source_scene_id" if envelope.get("scene_id") else "capture_session_identity",
        "purpose": "scene_preparation",
        "description": task["description"],
        "task_specification": deepcopy(dict(task)),
        "task_family": task.get("task_family") or "general_task",
        "task_confirmation_status": task.get("success_parameters_status", "unknown"),
        "source_capture": {
            "capture_digest": receipt["capture_digest"],
            "envelope_digest": receipt["envelope_digest"],
            "qa_report_digest": source["qa"]["qa_report_digest"],
            "object_manifest_digest": source["object_manifest"]["manifest_digest"],
            "capture_authority_profile": envelope["capture_authority_profile"],
        },
        "governance": deepcopy(envelope["governance"]),
        "permitted_reconstruction_providers": deepcopy(envelope.get("permitted_reconstruction_providers", [])),
        "permitted_evidence_uses": deepcopy(envelope.get("permitted_evidence_uses", [])),
        "coordinate_frame_declaration": deepcopy(envelope["coordinate_frame_declaration"]),
        "authority_issued": False,
    }
    context["context_digest"] = cross_runtime_canonical_digest(context, digest_field="context_digest")
    return validate_owned_capture_atlas_context(context)


def validate_owned_capture_atlas_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate retained context without treating its digest as authorization."""
    if value.get("schema_version") != SCHEMA or set(value) != _FIELDS:
        raise ValueError("owned_capture_atlas_context_schema_invalid")
    if value.get("context_digest") != cross_runtime_canonical_digest(value, digest_field="context_digest"):
        raise ValueError("owned_capture_atlas_context_digest_mismatch")
    if value.get("purpose") != "scene_preparation" or value.get("authority_issued") is not False:
        raise ValueError("owned_capture_atlas_context_authority_invalid")
    for key in ("request_id", "capture_id", "intake_id", "scene_id"):
        strict_identifier(value.get(key), field=key, max_length=192)
    source_scene = value.get("source_scene_id")
    if (value["scene_id"] != (source_scene or value["capture_id"])
            or value["scene_identity_basis"] != ("source_scene_id" if source_scene else "capture_session_identity")):
        raise ValueError("owned_capture_atlas_scene_binding_mismatch")
    task = value.get("task_specification")
    if (not isinstance(task, Mapping) or not isinstance(task.get("description"), str)
            or not task["description"].strip() or value.get("description") != task["description"]
            or not isinstance(value.get("task_family"), str) or not value["task_family"].strip()
            or value.get("task_family") != (task.get("task_family") or "general_task")
            or value.get("task_confirmation_status") != task.get("success_parameters_status", "unknown")):
        raise ValueError("owned_capture_atlas_task_binding_mismatch")
    source = value.get("source_capture")
    if not isinstance(source, Mapping):
        raise ValueError("owned_capture_atlas_source_binding_missing")
    for key in ("capture_digest", "envelope_digest", "qa_report_digest", "object_manifest_digest"):
        digest = source.get(key)
        if (not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:")
                or any(char not in "0123456789abcdef" for char in digest[7:])):
            raise ValueError("owned_capture_atlas_source_binding_invalid")
    if not isinstance(value.get("governance"), Mapping):
        raise ValueError("owned_capture_atlas_governance_missing")
    return deepcopy(dict(value))


def require_owned_capture_atlas_source_binding(
    value: Mapping[str, Any], *, source: Mapping[str, Any], plan_id: str
) -> dict[str, Any]:
    """Rebind to a fresh verified source read before obtaining disclosure authority.

    A caller can recompute a JSON digest. Only equality with the server's
    current source envelope establishes this context's source/task binding.
    The source read must perform the existing lifecycle/byte checks, and a
    separate current provider admission must still permit any external call.
    """
    context = validate_owned_capture_atlas_context(value)
    current = build_owned_capture_atlas_context(source=source, plan_id=plan_id)
    if context != current:
        raise ValueError("owned_capture_atlas_source_or_task_changed")
    return current
