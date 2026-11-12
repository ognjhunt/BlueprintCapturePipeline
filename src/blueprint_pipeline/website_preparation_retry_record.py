"""The website preparation's retry record, as the status reads it: a leaf with no retry, replay or worker import.

``website_preparation_retry`` writes the record and runs the retries; the status reader
(``website_preparation_outcome``) only needs to know whether a job's timed retries are spent.
Standard library only, so the status lane never reaches the retry lane's imports.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

FIELD = "preparation_retry"
SCHEMA = "website_preparation_retry.v1"
#: A failure whose timed retries are spent: it runs again once on each new release.
RETRIES_SPENT = "retries_spent"


def _parse_utc(value: Any) -> datetime | None:
    if type(value) is not str or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def armed(ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    """This ledger's current retry record, or None (another status, another delivery, another error)."""
    record = ledger.get(FIELD)
    if (ledger.get("status") != "failed_retryable" or not isinstance(record, Mapping)
            or record.get("schema_version") != SCHEMA or record.get("error") != ledger.get("last_error")
            or record.get("source_payload_sha256") != ledger.get("source_payload_sha256")
            or record.get("producer_delivery_key") != ledger.get("producer_delivery_key")
            or type(record.get("retry_count")) is not int or _parse_utc(record.get("next_attempt_at")) is None):
        return None
    return dict(record)
