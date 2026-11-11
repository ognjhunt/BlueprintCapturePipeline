"""Beta preparation delay observations and evidence-bound terminal projections.

Broker delivery counts and listener liveness do not prove exhaustion for a
particular message. Silence records an overdue observation only. The existing
v1/v2 status vocabulary stays compatible, and a later current claim supersedes
the observation while its immutable attempt-history entry remains available.
No observation authorizes retry, provider effects or authority renewal.

One silent case is not a delay: a preparation whose retained grant has expired
can never run again: every delivery would end it on the WebApp's
``consent_expired``. ``end_expired_preparation`` asks the WebApp the same
question without effect and, only on that refusal, writes the same ending through
the listener's own terminal function, so the job ends without a redelivery.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .handoff_job_state import (
    JOB_LEDGER_FILENAME,
    JOB_LEDGER_SCHEMA_VERSION,
    _existing_job_ledger_lock,
    _read_job_ledger,
)

TERMINAL_FIELD = "website_preparation_terminal"
SCHEMA_VERSION = "website_preparation_terminal.v1"
OVERDUE_FIELD = "website_preparation_overdue"
OVERDUE_SCHEMA_VERSION = "website_preparation_overdue.v1"
# The shared vocabulary of website_preparation_status.v2. This module writes
# retry_exhausted, budget_exhausted and capacity_blocked; the WebApp accepts all.
REASONS = frozenset({"permanent_invalid", "retry_exhausted", "budget_exhausted",
                     "capacity_blocked", "out_of_scope"})
_RETRYABLE_STATUSES = frozenset({"failed_retryable", "retryable_blocked"})
SILENCE_SECONDS_ENV = "BLUEPRINT_WEBSITE_PREPARATION_SILENCE_SECONDS"
DEFAULT_SILENCE_SECONDS = 2 * 60 * 60
_MINIMUM_SILENCE_SECONDS = 30 * 60
# The grant the job's latest delivery ran under (website_task_evidence).
GRANT_FILE = "pipeline/website_scene_sponsorship.json"
EXPIRY_STAGE = "website_preparation_grant_expiry"
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
    """Silence records cannot establish a terminal retry disposition.

    The existing producer wrote only silence-based annotations. Retain them for
    history, but do not promote even a plausible reason or digest into broker
    evidence. AUT-02's source/attempt-bound durable disposition reader must be
    integrated here before conclusive retry-exhaustion projection is enabled.
    Explicit ``terminal_authority_ended`` is handled separately by the reader.
    """
    return None


def current_overdue(ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    record = ledger.get(OVERDUE_FIELD)
    if (not isinstance(record, Mapping)
            or record.get("schema_version") != OVERDUE_SCHEMA_VERSION
            or record.get("reason") != "recovery_needed"
            or any(record.get(key) != ledger.get(key) for key in
                   ("attempt_count", "source_payload_sha256", "lease_token"))
            or record.get("observed_status") != ledger.get("status")):
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
    history = [item for item in ledger.get("attempt_history", []) if isinstance(item, Mapping)]
    committed = {**dict(ledger), OVERDUE_FIELD: dict(record),
                 "attempt_history": [*history, {"status": "preparation_overdue_observed", **dict(record)}],
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
        stopped = _parse_utc(ledger.get("updated_at"))
        from .website_reconstruction_resume import armed
        if stopped is not None and armed(ledger):
            # Replayed when the capture's Atlas job ends, not by Pub/Sub: its
            # silence starts only after the longest that job may run.
            from .website_atlas_preview_store import MAXIMUM_JOB_SECONDS
            return stopped + timedelta(seconds=MAXIMUM_JOB_SECONDS)
        return stopped
    if status == "processing" and type(ledger.get("lease_token")) is str and ledger["lease_token"]:
        expires = _parse_utc(ledger.get("lease_expires_at"))
        return expires if expires is not None and expires <= current else None
    return None


def record_unretried_preparation(capture_root: Path, *, alive_since: datetime,
                                 now: datetime | None = None) -> dict[str, Any] | None:
    """Observe overdue preparation without claiming that no retry remains.

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
        # Listener liveness only qualifies the delay observation. It says
        # nothing about this message's source/DLQ/recovery delivery state.
        if (silent is None or type(ledger.get("attempt_count")) is not int
                or type(ledger.get("source_payload_sha256")) is not str
                or max(silent, alive_since) + timedelta(seconds=silence_seconds()) > current
                or current_terminal(ledger) is not None or current_overdue(ledger) is not None):
            return None
        status = ledger.get("status")
        cause = ("control_plane_capacity" if capacity_only(ledger) else
                 "capture_budget" if _budget_refused(ledger) else
                 "worker_lease_expired" if status == "processing" else "queue_or_dependency_delay")
        record = {"schema_version": OVERDUE_SCHEMA_VERSION,
                  "reason": "recovery_needed", "known_cause": cause,
                  "basis": "no_claim_within_silence_window" if status in _RETRYABLE_STATUSES
                  else "lease_expired_without_reclaim",
                  "observed_status": status, "attempt_count": ledger.get("attempt_count"),
                  "source_payload_sha256": ledger.get("source_payload_sha256"),
                  "lease_token": ledger.get("lease_token"), "silent_since": _iso(silent),
                  "age_seconds": max(0.0, (current - silent).total_seconds()),
                  "next_action": "reconcile_per_job_recovery_state", "next_retry_at": None,
                  "retry_exhaustion_proven": False,
                  "recorded_at": _iso(current)}
        return _commit(Path(capture_root), ledger, record)


def _grant_may_have_ended(capture_root: Path, ledger: Mapping[str, Any], now_epoch: float) -> bool:
    """Trigger only: the retained grant is sealed, binds this capture, and its own expiry has passed.

    A grant's ``valid_until`` (an owner-approved window extension,
    ``validate_preparation_authority``; the WebApp's ``sponsorshipValidUntil``)
    is never earlier than ``expires_at_epoch``, so a raw expiry still ahead means
    the grant is valid. The extension lives on the scene intent, which a capture
    root does not name, so whether the grant has ended is asked of the WebApp.
    """
    path = Path(capture_root) / GRANT_FILE
    if path.is_symlink() or not path.is_file():
        return False
    try:
        grant = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(grant, dict):
        return False
    expiry = grant.get("expires_at_epoch")
    return not (grant.get("schema_version") != "website_scene_sponsorship.v1"
                or grant.get("authority_digest") != canonical_digest(grant, digest_field="authority_digest")
                or grant.get("scene_id") != ledger.get("scene_id")
                or grant.get("capture_id") != ledger.get("capture_id")
                or isinstance(expiry, bool) or not isinstance(expiry, (int, float))
                or not math.isfinite(expiry) or expiry > now_epoch)


def _webapp_ended(capture_root: Path, ledger: Mapping[str, Any]) -> ValueError | None:
    """The WebApp's own ``consent_expired`` refusal of this job's grant, or None.

    One non-creating, window-honouring ``scene-sponsorship`` read; any other
    answer, an error or a missing retained context is None.
    """
    from .website_task_context import read_scene_sponsorship_ending
    try:
        # The context the job's delivery retained (website_task_evidence), never a fresh one.
        path = Path(capture_root) / "pipeline" / "website_task_context.json"
        if path.is_symlink() or not path.is_file():
            return None
        context = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(context, dict) or type(context.get("request_id")) is not str
                or context.get("context_digest") != canonical_digest(context, digest_field="context_digest")
                or any(context.get(key) != ledger.get(key) for key in ("scene_id", "capture_id"))):
            return None
        return read_scene_sponsorship_ending(task_context=context)
    except Exception:  # noqa: BLE001 - nothing but the typed refusal may end a job
        return None


def end_expired_preparation(capture_root: Path, *, alive_since: datetime, listener: Any,
                            now: datetime | None = None) -> dict[str, Any] | None:
    """End a silent preparation once the WebApp refuses its grant as expired, as a delivery would.

    Applies only where ``record_unretried_preparation`` could record an overdue
    observation (a retryable attempt, or a dead worker's expired lease, silent
    for a full window of continuous listening) and the retained grant's own
    expiry has passed. Then the sweep asks the WebApp, once per tick, and ends
    the job only on its ``consent_expired`` refusal; a still-valid, extended or
    renewed grant, or any other answer, leaves the job as it is. A live lease,
    a completed or ended job and the assessment wait are never asked about. The
    ending is the listener's ``_finish_terminal_authority_ending`` on a claim of
    the revision read here, so it is written once. Nothing is retried, bought,
    renewed or released here. Returns the committed terminal ledger, or None.

    ``listener`` is the running ``pubsub_handoff_listener`` module, passed in as
    ``reconcile_waiting_assessments`` receives it, so this lane never imports it.
    """
    root = Path(capture_root)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    with _existing_job_ledger_lock(root) as state:
        if state != "ledger_present":
            return None
        ledger = _read_job_ledger(root)
        silent = _silent_since(ledger, current)
        if (silent is None or type(ledger.get("revision")) is not int
                or any(type(ledger.get(key)) is not str or not ledger[key] for key in
                       ("scene_id", "capture_id", "producer_delivery_key", "source_payload_sha256"))
                or max(silent, alive_since) + timedelta(seconds=silence_seconds()) > current
                or not _grant_may_have_ended(root, ledger, current.timestamp())):
            return None
    refusal = _webapp_ended(root, ledger)  # No network under the ledger lock.
    ending = listener.authority_ending(refusal) if refusal is not None else None
    if ending is None or ending[1] != "consent_expired":
        return None
    claim, claimed = listener._claim_job_lease(
        root, scene_id=ledger["scene_id"], capture_id=ledger["capture_id"], owner=listener._lease_owner(),
        lease_seconds=listener.DEFAULT_JOB_LEASE_SECONDS, payload_sha256=ledger["source_payload_sha256"],
        producer_delivery_key=ledger["producer_delivery_key"], create_capture_root=False,
        expected_revision=ledger["revision"])
    if claim != "claimed":
        return None
    handoff = listener.HandoffMessage(
        bucket=root.parents[3].name, scene_id=ledger["scene_id"], capture_id=ledger["capture_id"],
        raw_prefix_uri=f"gs://{root.parents[3].name}/scenes/{ledger['scene_id']}"
                       f"/captures/{ledger['capture_id']}/raw/",
        pipeline_handoff_uri=None)
    result = listener._finish_terminal_authority_ending(
        root, handoff=handoff, owner=claimed["lease_owner"], token=claimed["lease_token"],
        operation=ending[0], code=ending[1], error=refusal,
        stage=EXPIRY_STAGE, attempt_count=claimed["attempt_count"],
        attempt_started_at=claimed["last_attempt_started_at"],
        previous_history=listener._attempt_history(claimed),
        payload_digest=ledger["source_payload_sha256"], producer_delivery_key=ledger["producer_delivery_key"])
    return result["job_ledger"]
