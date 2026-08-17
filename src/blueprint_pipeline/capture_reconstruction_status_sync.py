"""Publish one capture's terminal reconstruction status back to the WebApp.

The WebApp already reads ``capture_submissions/{captureId}``, so that document
is where a capturer's app and the operator console will look for the outcome of
their walk.  It has no field for reconstruction today, and its
``operational_state`` map is a closed, QA-specific vocabulary, so this writes a
separate server-owned ``reconstruction`` field.

Three rules hold it honest:

* **Server-owned.** Clients can already only touch an allowlisted key set, and
  the accompanying rules change denies ``reconstruction`` explicitly.  A device
  cannot report its own capture as reconstructed.
* **Digest-bound.** Every published artifact carries its digest and the capture
  digest it came from, so a status row cannot be read as evidence for bytes it
  does not name.
* **Advisory about quality.** A terminal status says the run finished and what
  it produced.  It never asserts appearance fidelity, metric accuracy,
  collision suitability, or task success; those remain separate qualifications.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest

STATUS_SCHEMA_VERSION = "capture_reconstruction_status.v1"
SYNC_RECEIPT_SCHEMA_VERSION = "capture_reconstruction_status_sync_receipt.v1"

#: Terminal states.  ``abstained`` is a first-class outcome, not a failure:
#: it means the run correctly declined for a named missing input.
TERMINAL_STATES = ("published", "abstained", "failed")

#: Field written on capture_submissions/{captureId}.  Server-only.
RECONSTRUCTION_FIELD = "reconstruction"

_DIGEST_PREFIX = "sha256:"


class CaptureReconstructionStatusError(ValueError):
    """Stable fail-closed status error."""


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith(_DIGEST_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def build_terminal_status(
    *,
    capture_id: str,
    capture_digest: str,
    state: str,
    arm: str | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    blockers: Sequence[str] = (),
    campaign_digest: str | None = None,
    completed_at: str,
) -> dict[str, Any]:
    """Compose the digest-bound terminal status for one capture."""

    if state not in TERMINAL_STATES:
        raise CaptureReconstructionStatusError(
            f"capture_reconstruction_status_state_invalid:{state}"
        )
    if not _is_digest(capture_digest):
        raise CaptureReconstructionStatusError(
            "capture_reconstruction_status_capture_digest_invalid"
        )

    normalized: list[dict[str, Any]] = []
    for artifact in artifacts:
        row = dict(artifact)
        name = str(row.get("artifact_id") or "")
        digest = str(row.get("digest") or "")
        uri = str(row.get("uri") or "")
        if not name or not _is_digest(digest) or not uri:
            raise CaptureReconstructionStatusError(
                "capture_reconstruction_status_artifact_binding_incomplete"
            )
        normalized.append({"artifact_id": name, "digest": digest, "uri": uri})
    normalized.sort(key=lambda row: row["artifact_id"])

    if state == "published" and not normalized:
        raise CaptureReconstructionStatusError(
            "capture_reconstruction_status_published_without_artifacts"
        )
    if state != "published" and not blockers:
        raise CaptureReconstructionStatusError(
            "capture_reconstruction_status_nonterminal_requires_blockers"
        )

    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "capture_id": str(capture_id),
        "capture_digest": str(capture_digest),
        "state": state,
        "arm": str(arm) if arm else None,
        "artifacts": normalized,
        "blockers": sorted({str(item) for item in blockers if str(item)}),
        "campaign_digest": str(campaign_digest) if campaign_digest else None,
        "completed_at": str(completed_at),
        # A finished run is not a quality claim.  These stay false until their
        # own gates pass, so a green status row can never be misread.
        "appearance_fidelity_qualified": False,
        "metric_accuracy_qualified": False,
        "collision_suitability_qualified": False,
        "physical_task_success_proven": False,
    }
    status["status_digest"] = canonical_digest(status, digest_field="status_digest")
    return status


def sync_terminal_status(
    *,
    status: Mapping[str, Any],
    writer: Callable[[str, Mapping[str, Any]], Any],
    reader: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Write the status onto capture_submissions, exactly once per digest.

    ``writer`` receives the capture id and the field payload; injecting it keeps
    this testable and keeps Firestore credentials out of this module.  A repeat
    sync of the identical status is a no-op; a *different* status for a capture
    that already reached terminal is refused, because a published capture must
    not be quietly restated.
    """

    payload = dict(status)
    expected = canonical_digest(
        {k: v for k, v in payload.items() if k != "status_digest"},
        digest_field="status_digest",
    )
    if payload.get("status_digest") != expected:
        raise CaptureReconstructionStatusError(
            "capture_reconstruction_status_digest_mismatch"
        )

    capture_id = str(payload["capture_id"])
    if reader is not None:
        existing = reader(capture_id)
        if isinstance(existing, Mapping) and existing:
            if existing.get("status_digest") == payload["status_digest"]:
                return {
                    "schema_version": SYNC_RECEIPT_SCHEMA_VERSION,
                    "capture_id": capture_id,
                    "status_digest": payload["status_digest"],
                    "written": False,
                    "already_synced": True,
                }
            if str(existing.get("state")) in TERMINAL_STATES:
                raise CaptureReconstructionStatusError(
                    "capture_reconstruction_status_terminal_conflict:"
                    f"recorded={existing.get('status_digest')};"
                    f"offered={payload['status_digest']}"
                )

    writer(capture_id, {RECONSTRUCTION_FIELD: payload})
    return {
        "schema_version": SYNC_RECEIPT_SCHEMA_VERSION,
        "capture_id": capture_id,
        "status_digest": payload["status_digest"],
        "written": True,
        "already_synced": False,
    }


def status_from_campaign(
    *,
    capture_id: str,
    capture_digest: str,
    campaign_path: str | Path,
    completed_at: str,
) -> dict[str, Any]:
    """Derive a terminal status from a finalized canonical 3DGS campaign."""

    path = Path(campaign_path).expanduser().resolve()
    try:
        campaign = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureReconstructionStatusError(
            "capture_reconstruction_status_campaign_unreadable"
        ) from exc

    artifacts: list[dict[str, Any]] = []
    for arm in campaign.get("arms", []) or []:
        if not isinstance(arm, Mapping):
            continue
        for artifact in arm.get("artifacts", []) or []:
            if not isinstance(artifact, Mapping):
                continue
            digest = str(artifact.get("digest") or "")
            if not _is_digest(digest):
                continue
            artifacts.append(
                {
                    "artifact_id": f"{arm.get('arm_id')}:{artifact.get('artifact_id')}",
                    "digest": digest,
                    "uri": str(artifact.get("uri") or artifact.get("relative_path") or ""),
                }
            )

    return build_terminal_status(
        capture_id=capture_id,
        capture_digest=capture_digest,
        state="published" if artifacts else "failed",
        arm=str(campaign.get("primary_arm_id") or "") or None,
        artifacts=artifacts,
        blockers=[] if artifacts else ["canonical_3dgs_campaign_produced_no_artifacts"],
        campaign_digest=str(campaign.get("campaign_digest") or "") or None,
        completed_at=completed_at,
    )


__all__ = [
    "RECONSTRUCTION_FIELD",
    "STATUS_SCHEMA_VERSION",
    "SYNC_RECEIPT_SCHEMA_VERSION",
    "TERMINAL_STATES",
    "CaptureReconstructionStatusError",
    "build_terminal_status",
    "status_from_campaign",
    "sync_terminal_status",
]
