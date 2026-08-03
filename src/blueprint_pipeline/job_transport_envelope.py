"""Immutable transport envelope for job handoff — blueprint.job_envelope.v1.

The envelope wraps the existing ``robot_eval_job_request.v1`` contract for
managed-queue delivery (Pub/Sub, Cloud Tasks) without changing it. Managed
queues provide at-least-once delivery only; Blueprint keeps job identity,
idempotent claims, and durable terminal commits. The envelope id is
content-derived (job id + canonical payload digest) so duplicate deliveries
of the same job content are recognizable everywhere, and provider
credentials are refused at build time — credentials stay in the allocator,
never in queue messages or workers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

JOB_ENVELOPE_SCHEMA_VERSION = "blueprint.job_envelope.v1"

# Execution stays with the filesystem inbox until the strangler migration
# promotes a lane; shadow transport is delivery-parity evidence only.
EXECUTION_AUTHORITY_FILESYSTEM = "filesystem"

_REQUIRED_FIELDS = (
    "schema_version",
    "envelope_id",
    "job_id",
    "source_lane",
    "payload_sha256",
    "job_request",
    "created_at",
    "execution_authority",
)

_CREDENTIAL_KEY_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "authorization",
    "private_key",
)


class JobEnvelopeCredentialError(ValueError):
    """A job payload carried credential-shaped content; envelopes refuse it."""


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    # Matches the orchestrator's _sha_payload canonical form (sorted keys,
    # compact separators) so digests agree across surfaces.
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _scan_for_credentials(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in _CREDENTIAL_KEY_MARKERS):
                raise JobEnvelopeCredentialError(
                    f"job_envelope_credential_shaped_key:{path}{key}"
                )
            _scan_for_credentials(item, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_for_credentials(item, f"{path}{index}.")


def build_job_envelope(
    *,
    job_request: Mapping[str, Any],
    job_id: str,
    source_lane: str,
    created_at: str,
) -> dict[str, Any]:
    """Build a deterministic envelope; identity excludes ``created_at``."""

    _scan_for_credentials(job_request, "")
    payload_sha256 = hashlib.sha256(_canonical_payload_bytes(job_request)).hexdigest()
    envelope_id = hashlib.sha256(
        json.dumps(
            {
                "schema_version": JOB_ENVELOPE_SCHEMA_VERSION,
                "job_id": str(job_id),
                "payload_sha256": payload_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": JOB_ENVELOPE_SCHEMA_VERSION,
        "envelope_id": envelope_id,
        "job_id": str(job_id),
        "source_lane": str(source_lane),
        "payload_sha256": payload_sha256,
        "job_request": dict(job_request),
        "created_at": str(created_at),
        "execution_authority": EXECUTION_AUTHORITY_FILESYSTEM,
    }


def validate_job_envelope(envelope: Mapping[str, Any]) -> list[str]:
    """Fail-closed validation; returns blocker strings (empty == valid)."""

    blockers: list[str] = []
    if envelope.get("schema_version") != JOB_ENVELOPE_SCHEMA_VERSION:
        blockers.append("job_envelope_schema_version_invalid")
    for field in _REQUIRED_FIELDS:
        if field == "schema_version":
            continue
        if not envelope.get(field):
            blockers.append(f"job_envelope_field_missing:{field}")
    payload = envelope.get("job_request")
    if isinstance(payload, Mapping) and envelope.get("payload_sha256"):
        expected = hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()
        if expected != envelope.get("payload_sha256"):
            blockers.append("job_envelope_payload_digest_mismatch")
    return blockers


__all__ = [
    "JOB_ENVELOPE_SCHEMA_VERSION",
    "EXECUTION_AUTHORITY_FILESYSTEM",
    "JobEnvelopeCredentialError",
    "build_job_envelope",
    "validate_job_envelope",
]
