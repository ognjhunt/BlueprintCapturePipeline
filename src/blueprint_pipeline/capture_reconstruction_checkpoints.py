"""Durable checkpoints for one capture's reconstruction run.

A reconstruction crosses a laptop, a queue, a paid Windows GPU, and a
publication step, and any of those can die mid-flight.  The ledger exists so a
resumed run can tell the difference between work that already happened and work
that merely looks similar:

* Each ``(capture_digest, stage)`` is written exactly once and never rewritten.
  Re-recording identical evidence is a no-op; re-recording *different* evidence
  for a stage already passed is a conflict, because that means the inputs
  changed underneath a run that already spent something.
* Paid stages additionally carry the provider identity of the work that was
  done, so a resume can refuse to pay twice for the same arm rather than
  silently re-allocating.

The ledger records what happened.  It never authorizes the next step, and a
checkpoint is not evidence of reconstruction quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json

CHECKPOINT_SCHEMA_VERSION = "capture_reconstruction_checkpoint.v1"
LEDGER_SCHEMA_VERSION = "capture_reconstruction_checkpoint_ledger.v1"

#: Ordered stages of one capture's journey from device to downstream analysis.
CHECKPOINT_STAGES = (
    "upload_received",
    "intake_validated",
    "queued",
    "worker_allocated",
    "postshot_import",
    "training",
    "export",
    "publish",
    "downstream_dispatched",
    "terminal",
)

#: Stages whose repetition costs money.  A resume must never redo these.
PAID_STAGES = frozenset({"worker_allocated", "postshot_import", "training", "export"})

_DIGEST_PREFIX = "sha256:"


class CaptureReconstructionCheckpointError(ValueError):
    """Stable fail-closed checkpoint error."""


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith(_DIGEST_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _capture_dir(state_root: str | Path, capture_digest: str) -> Path:
    if not _is_digest(capture_digest):
        raise CaptureReconstructionCheckpointError(
            "capture_reconstruction_checkpoint_capture_digest_invalid"
        )
    key = capture_digest[len(_DIGEST_PREFIX) :]
    return Path(state_root).expanduser().resolve() / key


def record_checkpoint(
    *,
    state_root: str | Path,
    capture_digest: str,
    stage: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Record one stage exactly once.

    Returns the stored checkpoint.  ``already_recorded`` distinguishes a fresh
    write from an idempotent replay so a caller can skip work it already did.
    """

    if stage not in CHECKPOINT_STAGES:
        raise CaptureReconstructionCheckpointError(
            f"capture_reconstruction_checkpoint_stage_unknown:{stage}"
        )
    directory = _capture_dir(state_root, capture_digest)
    directory.mkdir(parents=True, exist_ok=True)
    index = CHECKPOINT_STAGES.index(stage)
    path = directory / f"{index:02d}-{stage}.json"

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "capture_digest": str(capture_digest),
        "stage": stage,
        "stage_index": index,
        "paid_stage": stage in PAID_STAGES,
        "evidence": json.loads(canonical_json(dict(evidence))),
    }
    checkpoint["checkpoint_digest"] = canonical_digest(
        checkpoint, digest_field="checkpoint_digest"
    )
    payload = (canonical_json(checkpoint) + "\n").encode("utf-8")

    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return {**checkpoint, "already_recorded": False}
    except FileExistsError:
        existing_bytes = path.read_bytes()
        if existing_bytes == payload:
            return {**checkpoint, "already_recorded": True}
        try:
            existing = json.loads(existing_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaptureReconstructionCheckpointError(
                f"capture_reconstruction_checkpoint_unreadable:{stage}"
            ) from exc
        raise CaptureReconstructionCheckpointError(
            "capture_reconstruction_checkpoint_conflict:"
            f"{stage};recorded={existing.get('checkpoint_digest')};"
            f"offered={checkpoint['checkpoint_digest']}"
        )


def read_checkpoints(
    *, state_root: str | Path, capture_digest: str
) -> dict[str, Any]:
    """Return every recorded stage for this capture, in stage order."""

    directory = _capture_dir(state_root, capture_digest)
    recorded: list[dict[str, Any]] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CaptureReconstructionCheckpointError(
                    f"capture_reconstruction_checkpoint_unreadable:{path.name}"
                ) from exc
            recorded.append(value)
    recorded.sort(key=lambda row: int(row.get("stage_index", 0)))
    stages = [str(row["stage"]) for row in recorded]
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "capture_digest": str(capture_digest),
        "recorded_stages": stages,
        "checkpoints": recorded,
        "paid_stages_completed": [
            stage for stage in stages if stage in PAID_STAGES
        ],
        "terminal": "terminal" in stages,
    }


def next_stage(*, state_root: str | Path, capture_digest: str) -> str | None:
    """Return the first stage not yet recorded, or None when terminal.

    Resume starts here.  Gaps are not skipped: the first missing stage is the
    resume point even if a later stage was somehow recorded, so an out-of-order
    ledger surfaces rather than letting a run continue past work it never did.
    """

    ledger = read_checkpoints(state_root=state_root, capture_digest=capture_digest)
    recorded = set(ledger["recorded_stages"])
    for stage in CHECKPOINT_STAGES:
        if stage not in recorded:
            return stage
    return None


def assert_paid_stage_not_repeated(
    *, state_root: str | Path, capture_digest: str, stage: str
) -> None:
    """Refuse to redo a paid stage this capture already completed."""

    if stage not in PAID_STAGES:
        return
    ledger = read_checkpoints(state_root=state_root, capture_digest=capture_digest)
    if stage in ledger["recorded_stages"]:
        raise CaptureReconstructionCheckpointError(
            f"capture_reconstruction_paid_stage_already_completed:{stage}"
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STAGES",
    "LEDGER_SCHEMA_VERSION",
    "PAID_STAGES",
    "CaptureReconstructionCheckpointError",
    "assert_paid_stage_not_repeated",
    "next_stage",
    "read_checkpoints",
    "record_checkpoint",
]
