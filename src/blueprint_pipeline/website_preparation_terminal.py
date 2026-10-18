"""Beta journey L3/L6: record that a website capture's preparation has no retry left.

Pub/Sub dead-letters a handoff after its last delivery attempt, and nothing
re-drives a dead-lettered handoff automatically. The ledger used to keep its
last retryable status, so the site was shown "preparing" or "needs review"
forever. This module writes one typed record onto the ledger's current attempt
when that happens:

- the last delivery of the ledger's own source failed or stayed blocked; or
- the last delivery claimed the job and its lease then expired unrenewed (the
  worker died on the final attempt, so Pub/Sub dead-letters it).

The record annotates the attempt; it is never a new job status. Claim, retry,
acknowledgement and workspace retirement are unchanged, and a later claim (an
operator re-drive) starts a fresh attempt without it. Status projection honours
it only while it still describes the ledger's current attempt. Only the ledger
lock is taken here and nothing reaches the network.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .handoff_job_state import (
    JOB_LEDGER_FILENAME,
    JOB_LEDGER_SCHEMA_VERSION,
    _existing_job_ledger_lock,
    _read_job_ledger,
)

TERMINAL_FIELD = "website_preparation_terminal"
SCHEMA_VERSION = "website_preparation_terminal.v1"
# The shared vocabulary of website_preparation_status.v2. This module writes
# retry_exhausted, budget_exhausted and capacity_blocked; the WebApp accepts all.
REASONS = frozenset({"permanent_invalid", "retry_exhausted", "budget_exhausted",
                     "capacity_blocked", "out_of_scope"})
_EXHAUSTED_STATUSES = frozenset({"failed_retryable", "retryable_blocked"})
# A final attempt's lease outlives its 600 s Pub/Sub ack deadline, so once the
# lease has expired the delivery is already dead-lettered. The grace only
# absorbs clock skew between the worker and this tick.
ABANDONED_LEASE_GRACE_SECONDS = 600

# Whole tokens only, as for the authority-ending codes in the listener: a 409
# refusal from any website control operation, e.g. preparation-spend.
_BUDGET_RE = re.compile(r"(?<![A-Za-z0-9_-])website_control_[a-z0-9-]+_http_409:"
                        r"website_scene_preparation_budget_exhausted(?![A-Za-z0-9_-])")
# Disk and staging shortages on the control plane, never something a site provides.
_CAPACITY_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:pubsub_handoff_staging_capacity_insufficient"
                          r"|control_plane_disk_budget_exceeded)(?![A-Za-z0-9_-])")


def _failure_texts(ledger: Mapping[str, Any]) -> tuple[list[str], str]:
    blockers = ledger.get("retry_blockers")
    rows = [item for item in blockers if type(item) is str] if type(blockers) is list else []
    error = ledger.get("last_error")
    return rows, error if type(error) is str else ""


def _budget_refused(ledger: Mapping[str, Any]) -> bool:
    blockers, error = _failure_texts(ledger)
    return bool(_BUDGET_RE.search(error)) or any(_BUDGET_RE.search(item) for item in blockers)


def capacity_only(ledger: Mapping[str, Any]) -> bool:
    """Whether a control-plane capacity shortage is the whole reason this attempt waits."""
    if _budget_refused(ledger):
        return False
    blockers, error = _failure_texts(ledger)
    if ledger.get("status") == "retryable_blocked":
        return bool(blockers) and all(_CAPACITY_RE.search(item) for item in blockers)
    if ledger.get("status") == "failed_retryable":
        return bool(_CAPACITY_RE.search(error))
    return False


def exhausted_reason(ledger: Mapping[str, Any]) -> str:
    """The most specific reason the final retryable attempt could not proceed."""
    if _budget_refused(ledger):
        return "budget_exhausted"
    if capacity_only(ledger):
        return "capacity_blocked"
    return "retry_exhausted"


def current_terminal(ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    """The record, only while it still describes this ledger's current attempt."""
    record = ledger.get(TERMINAL_FIELD)
    if (type(record) is not dict or record.get("schema_version") != SCHEMA_VERSION
            or record.get("reason") not in REASONS
            or record.get("observed_status") not in _EXHAUSTED_STATUSES | {"processing"}
            or record.get("observed_status") != ledger.get("status")
            or type(record.get("attempt_count")) is not int
            or record.get("attempt_count") != ledger.get("attempt_count")
            or type(record.get("source_payload_sha256")) is not str
            or record.get("source_payload_sha256") != ledger.get("source_payload_sha256")
            or record.get("lease_token") != ledger.get("lease_token")):
        return None
    return dict(record)


def _parse_utc(value: Any) -> datetime | None:
    if type(value) is not str or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _commit(capture_root: Path, ledger: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Same revision discipline as the listener's ledger commits; caller holds the lock."""
    revision = ledger.get("revision")
    if type(revision) is not int or revision < 1:
        return None
    committed = {**dict(ledger), TERMINAL_FIELD: dict(record),
                 "schema_version": JOB_LEDGER_SCHEMA_VERSION, "revision": revision + 1}
    write_json(capture_root / JOB_LEDGER_FILENAME, committed)
    return committed


def _record(ledger: Mapping[str, Any], *, reason: str, basis: str, now: datetime) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "reason": reason, "basis": basis,
            "observed_status": ledger.get("status"), "attempt_count": ledger.get("attempt_count"),
            "source_payload_sha256": ledger.get("source_payload_sha256"),
            "lease_token": ledger.get("lease_token"), "recorded_at": _iso(now)}


def record_delivery_exhaustion(capture_root: Path, *, payload_sha256: str,
                               now: datetime | None = None) -> dict[str, Any] | None:
    """After the final delivery of this ledger's own source stayed retryable.

    Returns the committed ledger, or None when nothing was recorded: no ledger,
    another source's delivery (for example a storage-trigger handoff that never
    claimed), an attempt still holding its lease, or an existing record.
    """
    if type(payload_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", payload_sha256) is None:
        return None
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _existing_job_ledger_lock(Path(capture_root)) as state:
        if state != "ledger_present":
            return None
        ledger = _read_job_ledger(Path(capture_root))
        if (ledger.get("status") not in _EXHAUSTED_STATUSES
                or ledger.get("source_payload_sha256") != payload_sha256
                or ledger.get("lease_token") is not None
                or type(ledger.get("attempt_count")) is not int
                or current_terminal(ledger) is not None):
            return None
        return _commit(Path(capture_root), ledger, _record(
            ledger, reason=exhausted_reason(ledger), basis="delivery_attempts_exhausted", now=current))


def record_abandoned_final_delivery(capture_root: Path, *, now: datetime | None = None) -> dict[str, Any] | None:
    """A final-attempt claim whose worker stopped renewing its lease.

    Only a claim made on the last Pub/Sub delivery is marked final, so an
    expired lease on any earlier attempt still waits for its redelivery.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _existing_job_ledger_lock(Path(capture_root)) as state:
        if state != "ledger_present":
            return None
        ledger = _read_job_ledger(Path(capture_root))
        expires = _parse_utc(ledger.get("lease_expires_at"))
        if (ledger.get("status") != "processing" or ledger.get("final_delivery_attempt") is not True
                or type(ledger.get("lease_token")) is not str or not ledger["lease_token"]
                or type(ledger.get("source_payload_sha256")) is not str
                or type(ledger.get("attempt_count")) is not int
                or expires is None
                or expires + timedelta(seconds=ABANDONED_LEASE_GRACE_SECONDS) > current
                or current_terminal(ledger) is not None):
            return None
        return _commit(Path(capture_root), ledger, _record(
            ledger, reason="retry_exhausted", basis="final_delivery_lease_abandoned", now=current))
