"""Shadow publishing of job envelopes — strangler step for job transport.

Shadow mode publishes each admitted job's envelope to a managed queue and
records delivery-parity evidence WITHOUT executing anything from the queue:
the filesystem inbox remains the only execution authority. Once parity
evidence (duplicate delivery, consumer crash, lease expiry, DLQ, replay,
terminal commit) is proven for a lane, that lane — never the whole
orchestrator — can be promoted deliberately.

Publishers are injectable. The GCP implementation reuses the repository's
existing Pub/Sub dependency; tests and evidence-only shadowing use the
in-memory publisher. A publish failure is contained evidence, never an
admission failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol

from .job_transport_envelope import (
    EXECUTION_AUTHORITY_FILESYSTEM,
    validate_job_envelope,
)

SHADOW_PUBLISH_LEDGER_FILENAME = "shadow_publish_ledger.jsonl"
SHADOW_DELIVERY_LEDGER_FILENAME = "shadow_delivery_ledger.jsonl"
SHADOW_PARITY_SCHEMA_VERSION = "job_transport_shadow_parity.v1"


class EnvelopePublisher(Protocol):
    def publish(self, envelope: Mapping[str, Any]) -> str:
        """Publish one envelope; returns the transport message id."""


class InMemoryEnvelopePublisher:
    """Evidence-only publisher: records envelopes, no external transport."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    def publish(self, envelope: Mapping[str, Any]) -> str:
        self.published.append(dict(envelope))
        return f"inmemory:{envelope.get('envelope_id')}"


class PubsubEnvelopePublisher:
    """Publishes envelopes to a Pub/Sub topic (existing paid dependency)."""

    def __init__(self, *, topic: str, project: str | None = None) -> None:
        from google.cloud import pubsub_v1  # existing base dependency

        self._client = pubsub_v1.PublisherClient()
        if topic.startswith("projects/"):
            self._topic_path = topic
        else:
            if not project:
                raise ValueError("pubsub_project_required_for_short_topic_name")
            self._topic_path = self._client.topic_path(project, topic)

    def publish(self, envelope: Mapping[str, Any]) -> str:
        payload = json.dumps(dict(envelope), sort_keys=True).encode("utf-8")
        future = self._client.publish(
            self._topic_path,
            payload,
            envelope_id=str(envelope.get("envelope_id") or ""),
        )
        return str(future.result(timeout=30))


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def shadow_publish_job_envelope(
    *,
    envelope: Mapping[str, Any],
    publisher: EnvelopePublisher,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Publish one envelope for parity evidence; failures are contained."""

    evidence_dir = Path(evidence_dir)
    record: dict[str, Any] = {
        "envelope_id": envelope.get("envelope_id"),
        "job_id": envelope.get("job_id"),
        "source_lane": envelope.get("source_lane"),
        "execution_authority": EXECUTION_AUTHORITY_FILESYSTEM,
        "status": "publish_failed",
        "transport_message_id": None,
        "error": None,
    }
    blockers = validate_job_envelope(envelope)
    if blockers:
        record["error"] = "job_envelope_invalid:" + ",".join(blockers)
    else:
        try:
            record["transport_message_id"] = publisher.publish(envelope)
            record["status"] = "published"
        except Exception as exc:  # noqa: BLE001 - shadow must never break admission
            record["error"] = f"{type(exc).__name__}:{exc}"
    _append_jsonl(evidence_dir / SHADOW_PUBLISH_LEDGER_FILENAME, record)
    return record


def record_shadow_delivery(
    *, envelope_id: str, evidence_dir: Path, consumer: str
) -> dict[str, Any]:
    """Ack-only receipt from a shadow consumer (which must not execute)."""

    row = {
        "envelope_id": str(envelope_id),
        "consumer": str(consumer),
        "executed": False,
    }
    _append_jsonl(Path(evidence_dir) / SHADOW_DELIVERY_LEDGER_FILENAME, row)
    return row


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def compare_shadow_parity(evidence_dir: Path) -> dict[str, Any]:
    """Delivery-parity report over the shadow ledgers (promotion evidence)."""

    evidence_dir = Path(evidence_dir)
    published_rows = [
        row
        for row in _jsonl_rows(evidence_dir / SHADOW_PUBLISH_LEDGER_FILENAME)
        if row.get("status") == "published"
    ]
    delivered_rows = _jsonl_rows(evidence_dir / SHADOW_DELIVERY_LEDGER_FILENAME)
    published_ids = [str(row.get("envelope_id")) for row in published_rows]
    delivered_ids = [str(row.get("envelope_id")) for row in delivered_rows]
    published_unique = sorted(set(published_ids))
    delivered_unique = sorted(set(delivered_ids))
    missing = sorted(set(published_unique) - set(delivered_unique))
    duplicates = sorted(
        envelope_id
        for envelope_id in set(delivered_ids)
        if delivered_ids.count(envelope_id) > 1
    )
    if missing:
        status = "delivery_gap"
    elif duplicates:
        status = "parity_with_duplicates"
    else:
        status = "parity"
    return {
        "schema_version": SHADOW_PARITY_SCHEMA_VERSION,
        "status": status,
        "published_total": len(published_ids),
        "published_unique": len(published_unique),
        "delivered_total": len(delivered_ids),
        "delivered_unique": len(delivered_unique),
        "missing_delivery": missing,
        "duplicate_deliveries": duplicates,
    }


__all__ = [
    "SHADOW_PUBLISH_LEDGER_FILENAME",
    "SHADOW_DELIVERY_LEDGER_FILENAME",
    "SHADOW_PARITY_SCHEMA_VERSION",
    "EnvelopePublisher",
    "InMemoryEnvelopePublisher",
    "PubsubEnvelopePublisher",
    "shadow_publish_job_envelope",
    "record_shadow_delivery",
    "compare_shadow_parity",
]
