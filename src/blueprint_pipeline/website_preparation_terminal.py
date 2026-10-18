"""Beta journey L3/L6: record that a website capture's preparation has no retry left.

Pub/Sub dead-letters a handoff after an approximate number of delivery
attempts, and nothing re-drives a dead-lettered handoff automatically. The
count is best-effort: Pub/Sub may forward early, deliver past the maximum, or
reset the counter (https://docs.cloud.google.com/pubsub/docs/handling-failures),
so a delivery number is no proof that retries are over. The ledger used to keep
its last retryable status, and the site was shown "preparing" or "needs review"
forever.

Silence is the proof used here instead. A retryable outcome, or a lease whose
worker died, ends the preparation once no new claim of that ledger has followed
it for SILENCE_SECONDS of this listener pulling continuously. Pub/Sub redelivers
a deferred handoff within about 20 minutes (600 s deferral plus at most 600 s
backoff), so a redelivery that is still coming is claimed first. A listener
outage or a long run restarts the liveness clock, so a backlog waiting for this
listener is never mistaken for silence.

The record annotates the attempt; it is never a new job status. Claim, retry,
acknowledgement and workspace retirement are unchanged, and a later claim (an
operator re-drive, or a delivery after all) starts a fresh attempt without it.
Only the ledger lock is taken here and nothing reaches the network.
"""
from __future__ import annotations

import json
import os
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
_RETRYABLE_STATUSES = frozenset({"failed_retryable", "retryable_blocked"})
SILENCE_SECONDS_ENV = "BLUEPRINT_WEBSITE_PREPARATION_SILENCE_SECONDS"
DEFAULT_SILENCE_SECONDS = 2 * 60 * 60
_MINIMUM_SILENCE_SECONDS = 30 * 60
# Listener ticks run about a minute apart; a longer gap (an outage, or one tick
# busy processing) means this listener could not have received a redelivery.
LIVENESS_GAP_SECONDS = 15 * 60
LIVENESS_FILE = ".website_preparation_sweep_liveness.json"
LIVENESS_SCHEMA_VERSION = "website_preparation_sweep_liveness.v1"
# The explicit assessment wait is resumed by reconcile_waiting_assessments, not
# by Pub/Sub, so its silence is not the end of preparation.
_ASSESSMENT_PENDING = "website_control_scene-sponsorship_http_409:website_assessment_preparation_pending"

# Whole tokens only, as for the authority-ending codes in the listener: a 409
# refusal from any website control operation, e.g. preparation-spend.
_BUDGET_RE = re.compile(r"(?<![A-Za-z0-9_-])website_control_[a-z0-9-]+_http_409:"
                        r"website_scene_preparation_budget_exhausted(?![A-Za-z0-9_-])")
# Disk and staging shortages on the control plane, never something a site provides.
_CAPACITY_RE = re.compile(r"(?<![A-Za-z0-9_-])(?:pubsub_handoff_staging_capacity_insufficient"
                          r"|control_plane_disk_budget_exceeded)(?![A-Za-z0-9_-])")


def silence_seconds() -> int:
    value = os.getenv(SILENCE_SECONDS_ENV, "").strip()
    seconds = int(value) if value.isdigit() else DEFAULT_SILENCE_SECONDS
    return max(_MINIMUM_SILENCE_SECONDS, seconds)


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
    """The most specific reason the last retryable attempt could not proceed."""
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
            or record.get("observed_status") not in _RETRYABLE_STATUSES | {"processing"}
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


def note_sweep_liveness(storage_root: Path, *, now: datetime | None = None) -> datetime:
    """Record this listener tick; return since when ticks have been continuous.

    A missing, unreadable or stale record restarts the clock at this tick, so
    silence is only ever measured while the listener was able to receive.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    path = Path(storage_root) / LIVENESS_FILE
    since = current
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else {}
        last, start = _parse_utc(value.get("last_tick_at")), _parse_utc(value.get("continuous_since"))
        if (value.get("schema_version") == LIVENESS_SCHEMA_VERSION and last and start
                and start <= last <= current and current - last <= timedelta(seconds=LIVENESS_GAP_SECONDS)):
            since = start
    except (OSError, ValueError, AttributeError):
        since = current
    write_json(path, {"schema_version": LIVENESS_SCHEMA_VERSION, "continuous_since": _iso(since),
                      "last_tick_at": _iso(current)})
    return since


def _commit(capture_root: Path, ledger: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Same revision discipline as the listener's ledger commits; caller holds the lock."""
    revision = ledger.get("revision")
    if type(revision) is not int or revision < 1:
        return None
    committed = {**dict(ledger), TERMINAL_FIELD: dict(record),
                 "schema_version": JOB_LEDGER_SCHEMA_VERSION, "revision": revision + 1}
    write_json(capture_root / JOB_LEDGER_FILENAME, committed)
    return committed


def _silent_since(ledger: Mapping[str, Any], current: datetime) -> datetime | None:
    """When this attempt stopped moving, or None while it may still move on its own."""
    status = ledger.get("status")
    if status in _RETRYABLE_STATUSES:
        if (status == "failed_retryable" and ledger.get("last_error") == _ASSESSMENT_PENDING
                and isinstance(ledger.get("assessment_resume"), Mapping)):
            return None
        return _parse_utc(ledger.get("updated_at"))
    if status == "processing" and type(ledger.get("lease_token")) is str and ledger["lease_token"]:
        expires = _parse_utc(ledger.get("lease_expires_at"))
        return expires if expires is not None and expires <= current else None
    return None


def record_unretried_preparation(capture_root: Path, *, alive_since: datetime,
                                 now: datetime | None = None) -> dict[str, Any] | None:
    """End a website preparation that nothing has retried within the silence window.

    `alive_since` comes from note_sweep_liveness. Returns the committed ledger,
    or None when nothing was recorded: no ledger, not a website source, still
    live or explicitly waiting, not yet a full window of continuous listening
    since the attempt stopped, or an existing record.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _existing_job_ledger_lock(Path(capture_root)) as state:
        if state != "ledger_present":
            return None
        ledger = _read_job_ledger(Path(capture_root))
        silent = _silent_since(ledger, current)
        # The window counts only time after the stop during which this listener
        # was continuously pulling: a redelivery would have been claimed in it.
        if (silent is None or type(ledger.get("attempt_count")) is not int
                or type(ledger.get("source_payload_sha256")) is not str
                or max(silent, alive_since) + timedelta(seconds=silence_seconds()) > current
                or current_terminal(ledger) is not None):
            return None
        status = ledger.get("status")
        record = {"schema_version": SCHEMA_VERSION,
                  "reason": exhausted_reason(ledger) if status in _RETRYABLE_STATUSES else "retry_exhausted",
                  "basis": "no_claim_within_silence_window" if status in _RETRYABLE_STATUSES
                  else "lease_expired_without_reclaim",
                  "observed_status": status, "attempt_count": ledger.get("attempt_count"),
                  "source_payload_sha256": ledger.get("source_payload_sha256"),
                  "lease_token": ledger.get("lease_token"), "silent_since": _iso(silent),
                  "recorded_at": _iso(current)}
        return _commit(Path(capture_root), ledger, record)
