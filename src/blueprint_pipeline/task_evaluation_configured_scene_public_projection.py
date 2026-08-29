"""Build the public-safe projection of one configured-scene offering.

The authenticated offering remains the authority for evaluation preparation.
This module emits only an explicitly authorized display projection and never
copies object-store URIs, team namespaces, run identifiers, or raw media.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


AUTHORIZATION_SCHEMA_VERSION = "task_evaluation_configured_scene_public_display_authorization.v1"
PROJECTION_SCHEMA_VERSION = "task_evaluation_configured_scene_public_display.v1"
PUBLIC_DISPLAY_ALLOWED_FIELDS = (
    "status",
    "scene_identity",
    "task_identity",
    "task_kind",
    "task_strategy",
    "public_title",
    "public_summary",
    "public_category",
    "thumbnail",
    "proof_boundary",
)
PUBLIC_OFFERING_STATUSES = frozenset({"configured_controls_pending", "evaluation_ready"})
_PUBLIC_SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,95}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORBIDDEN_PUBLIC_TEXT = (
    "s3://",
    "gs://",
    "http://",
    "https://",
    "file://",
    "/var/",
    "/private/",
    # This is a forbidden-public-text rejection marker, never a filesystem target.
    "/tmp/",  # nosec B108
    "\\",
    "api_key",
    "password",
    "secret",
    "bearer ",
)


class ConfiguredScenePublicProjectionError(ValueError):
    """The request did not authorize a bounded public projection."""


def _identity(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or set(value) != {"id", "version"}:
        return None
    identity = {"id": value.get("id"), "version": value.get("version")}
    if not all(isinstance(part, str) and part for part in identity.values()):
        return None
    return identity  # type: ignore[return-value]


def _safe_public_text(value: Any, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    lowered = text.lower()
    if (
        not 1 <= len(text) <= maximum
        or any(ord(character) < 32 for character in text)
        or any(marker in lowered for marker in _FORBIDDEN_PUBLIC_TEXT)
    ):
        return None
    return text


def validate_public_display_authorization(
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return explicit public-display authority, or ``None`` when absent.

    Absence is intentionally not an error: the scene remains team-private.
    Once the field is present, every binding must validate or publication fails.
    """

    scene = request.get("scene")
    rights = scene.get("rights") if isinstance(scene, Mapping) else None
    authority = rights.get("public_display_authorization") if isinstance(rights, Mapping) else None
    if authority is None:
        return None
    if not isinstance(authority, Mapping):
        raise ConfiguredScenePublicProjectionError(
            "configured_scene_public_display_authorization_invalid"
        )

    task = request.get("task")
    subject = task.get("subject") if isinstance(task, Mapping) else None
    evidence = rights.get("evidence") if isinstance(rights, Mapping) else None
    human_records = [
        row
        for row in evidence or []
        if isinstance(row, Mapping) and row.get("role") == "human_authority_record"
    ]
    allowed_fields = authority.get("allowed_fields")
    safe_metadata = {
        "title": _safe_public_text(authority.get("title"), maximum=120),
        "summary": _safe_public_text(authority.get("summary"), maximum=500),
        "category": _safe_public_text(authority.get("category"), maximum=80),
    }
    valid = (
        authority.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION
        and authority.get("status") == "authorized"
        and authority.get("scope") == "configured_scene_derived_listing"
        and _identity(authority.get("scene_identity"))
        == _identity(scene.get("identity") if isinstance(scene, Mapping) else None)
        and _identity(authority.get("task_identity"))
        == _identity(task.get("identity") if isinstance(task, Mapping) else None)
        and _identity(authority.get("subject_identity"))
        == _identity(subject.get("identity") if isinstance(subject, Mapping) else None)
        and isinstance(rights, Mapping)
        and isinstance(rights.get("admission"), Mapping)
        and authority.get("rights_admission_digest") == rights["admission"].get("digest")
        and len(human_records) == 1
        and isinstance(human_records[0].get("artifact"), Mapping)
        and authority.get("human_authority_record_digest")
        == human_records[0]["artifact"].get("digest")
        and isinstance(authority.get("public_slug"), str)
        and _PUBLIC_SLUG.fullmatch(authority["public_slug"]) is not None
        and all(value is not None for value in safe_metadata.values())
        and list(allowed_fields or []) == list(PUBLIC_DISPLAY_ALLOWED_FIELDS)
        and authority.get("thumbnail_publication_authorized") is True
        and authority.get("derived_metadata_publication_authorized") is True
        and authority.get("private_artifact_uri_publication_authorized") is False
        and authority.get("raw_media_publication_authorized") is False
        and _safe_public_text(authority.get("authority_reference"), maximum=256) is not None
        and _safe_public_text(authority.get("authorized_by"), maximum=192) is not None
        and authority.get("authorization_digest")
        == canonical_digest(authority, digest_field="authorization_digest")
    )
    if not valid:
        raise ConfiguredScenePublicProjectionError(
            "configured_scene_public_display_authorization_invalid"
        )
    return dict(authority)


def build_public_display_projection(
    *,
    request: Mapping[str, Any],
    revision: Mapping[str, Any],
    offering: Mapping[str, Any],
    source_offering_digest: str,
    diagnostic_only: bool,
) -> dict[str, Any] | None:
    """Build a digest-bound, URI-free display projection for the public site."""

    authority = validate_public_display_authorization(request)
    if authority is None:
        return None
    presentation = offering.get("presentation")
    thumbnail = presentation.get("task_thumbnail") if isinstance(presentation, Mapping) else None
    revision_digest = revision.get("revision_digest")
    scene_identity = _identity(offering.get("scene_identity"))
    task = offering.get("task")
    valid = (
        diagnostic_only is False
        and revision.get("status") == "configured"
        and offering.get("status") in PUBLIC_OFFERING_STATUSES
        and _DIGEST.fullmatch(str(source_offering_digest or "")) is not None
        and _DIGEST.fullmatch(str(revision_digest or "")) is not None
        and isinstance(thumbnail, Mapping)
        and _DIGEST.fullmatch(str(thumbnail.get("digest") or "")) is not None
        and scene_identity == _identity(revision.get("scene_identity"))
        and isinstance(task, Mapping)
        and _identity(task.get("identity"))
        == _identity(revision.get("task_template", {}).get("identity"))
        and _identity(task.get("subject_identity"))
        == _identity(revision.get("replacement", {}).get("identity"))
    )
    if not valid:
        raise ConfiguredScenePublicProjectionError(
            "configured_scene_public_projection_nonqualifying"
        )

    projection: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "status": "authorized",
        "source_authorization_digest": authority["authorization_digest"],
        "source_offering_digest": source_offering_digest,
        "public_slug": authority["public_slug"],
        "title": authority["title"].strip(),
        "summary": authority["summary"].strip(),
        "category": authority["category"].strip(),
        "allowed_fields": list(PUBLIC_DISPLAY_ALLOWED_FIELDS),
        "scene_identity_digest": canonical_digest(scene_identity),
        "configured_scene_revision_digest": revision_digest,
        "task_thumbnail_digest": thumbnail["digest"],
        "projection_digest": "",
    }
    projection["projection_digest"] = canonical_digest(projection, digest_field="projection_digest")
    return projection


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "PUBLIC_DISPLAY_ALLOWED_FIELDS",
    "ConfiguredScenePublicProjectionError",
    "build_public_display_projection",
    "validate_public_display_authorization",
]
