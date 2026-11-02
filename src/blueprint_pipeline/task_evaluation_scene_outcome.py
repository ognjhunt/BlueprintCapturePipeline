"""Every website job ends in exactly one terminal state with a plain reason (C-04).

Scene progression re-runs every intent that is not ``completed`` on every pass,
so a job that could never finish was retried forever and the website could only
guess that it had ended. ``decide_outcome`` maps one progression row (status,
phase, blockers) and how long it has been unchanged to one outcome from a single
registry:

``setup_repair``
    Blueprint has to fix something. Not terminal while a code fix can still clear
    it (``pipeline_fault``: an unknown blocker, a failed publication). Terminal
    when a sealed stage failed, a non-extendable retry or recovery allowance ran
    out, the execution window expired (Blueprint sponsors it, so it is never the
    site's fault), or the row has not changed for 24 hours (``stalled``, the
    WebApp's stuck bound).
``awaiting_spend_approval``
    A spend or attempt cap or the job's dollar ceiling waits for the owner's money
    approval. Not terminal.
``awaiting_site``
    The site has to act (renew a consent, write a task brief). Not terminal.
``withdrawn``
    Privacy, rights or consent ended the job: revocation, an expired or withdrawn
    consent, a revoked source. Terminal.
``held``
    We cannot simulate this kind of job yet. Terminal.

Transient, self-retrying rows (work in flight, a capacity wait, a receipt still
pending) have no outcome; neither does ``completed``, which also supersedes any
outcome on the status read. The 24-hour rule ends transient rows and
``pipeline_fault`` rows; a person's wait (money, the site) and an operator's
hold are bounded by the execution window instead, which ends as
``authority_expired``.

``settle_outcome`` records the decision at ``<intent>/outcome.json`` after every
progression write. A terminal outcome is written once and never rewritten; a
non-terminal one is replaced by the next decision, or removed when the row no
longer has one.
"""
from __future__ import annotations

import json
import math
import os
import stat
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .control_plane_stall_alarm import OPERATOR_HOLD_PHASES, _Budget, _unchanged_since
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .validation_file_digests import note_consulted

SCHEMA_VERSION = "task_evaluation_scene_intent_outcome.v1"
FILENAME = "outcome.json"
STALL_SECONDS = 24 * 60 * 60

KINDS = ("setup_repair", "awaiting_spend_approval", "awaiting_site", "withdrawn", "held")
REASON_CATEGORIES = ("blueprint_fault", "stalled", "authority_expired", "spend_approval", "site_action",
                     "privacy_rights", "capability_limit")

#: The closed reason vocabulary: reason_code -> (kind, terminal, reason_category).
REASONS: dict[str, tuple[str, bool, str]] = {
    # (b) A Blueprint fault a code fix can clear; the 24-hour rule ends it.
    "pipeline_fault": ("setup_repair", False, "blueprint_fault"),
    # Sealed Blueprint failures.
    "scene_configuration_failed": ("setup_repair", True, "blueprint_fault"),
    "policy_canary_blocked": ("setup_repair", True, "blueprint_fault"),
    "retry_cap_exhausted": ("setup_repair", True, "blueprint_fault"),
    "submission_retry_cap_exhausted": ("setup_repair", True, "blueprint_fault"),
    "capacity_recovery_cap_exhausted": ("setup_repair", True, "blueprint_fault"),
    "market_miss_recovery_cap_exhausted": ("setup_repair", True, "blueprint_fault"),
    # (g) The execution window is Blueprint's sponsorship.
    "authority_expired": ("setup_repair", True, "authority_expired"),
    # The give-up rule.
    "stalled": ("setup_repair", True, "stalled"),
    # (c) Money: only the owner's budget extension or ceiling raise continues these.
    "spend_cap_reached": ("awaiting_spend_approval", False, "spend_approval"),
    "attempt_cap_reached": ("awaiting_spend_approval", False, "spend_approval"),
    "job_ceiling_reached": ("awaiting_spend_approval", False, "spend_approval"),
    # (d) The site has to act.
    "site_consent_required": ("awaiting_site", False, "site_action"),
    "task_brief_required": ("awaiting_site", False, "site_action"),
    # (e) Privacy, rights and consent.
    "authority_revoked": ("withdrawn", True, "privacy_rights"),
    "consent_expired": ("withdrawn", True, "privacy_rights"),
    "consent_withdrawn": ("withdrawn", True, "privacy_rights"),
    "source_revoked": ("withdrawn", True, "privacy_rights"),
    # (f) Capability limits. task_infeasible and no_candidate are reserved for the
    # screening refusals, which no progression blocker reaches yet.
    "source_kind_unsupported": ("held", True, "capability_limit"),
    "policy_capability_unavailable": ("held", True, "capability_limit"),
    "task_family_unsupported": ("held", True, "capability_limit"),
    "task_infeasible": ("held", True, "capability_limit"),
    "no_candidate": ("held", True, "capability_limit"),
}

#: Blocker codes, as the status endpoint exposes them (up to the first ':'), and
#: where each one comes from.
BLOCKER_REASONS: dict[str, str] = {
    # task_evaluation_scene_intake.reserve_scene_attempt
    "scene_intake_spend_cap_exhausted": "spend_cap_reached",
    "scene_intake_attempt_spend_exceeds_authorized_limit": "spend_cap_reached",
    "scene_intake_attempt_cap_exhausted": "attempt_cap_reached",
    # The website job's dollar ceiling (WebApp #1168): the screening plan and the
    # WebApp's 409 refusal of preparation spend past it.
    "beta_screening_job_ceiling_exceeded": "job_ceiling_reached",
    "website_job_ceiling_insufficient": "job_ceiling_reached",
    "website_scene_preparation_budget_exhausted": "job_ceiling_reached",
    # Non-extendable allowances (max_retries is sealed in the intent; the recovery
    # caps and the submission cap are code and config): scene progression,
    # reserve_scene_attempt, and the bare codes older records carry.
    "scene_progression_retry_cap_exhausted": "retry_cap_exhausted",
    "scene_intake_retry_cap_exhausted": "retry_cap_exhausted",
    "retry_cap_exhausted": "retry_cap_exhausted",
    "scene_progression_submission_retry_cap_exhausted": "submission_retry_cap_exhausted",
    "submission_retry_cap_exhausted": "submission_retry_cap_exhausted",
    "scene_progression_preallocation_capacity_recovery_cap_exhausted": "capacity_recovery_cap_exhausted",
    "scene_intake_preallocation_capacity_recovery_cap_exhausted": "capacity_recovery_cap_exhausted",
    "preallocation_capacity_recovery_cap_exhausted": "capacity_recovery_cap_exhausted",
    "scene_intake_market_miss_recovery_cap_exhausted": "market_miss_recovery_cap_exhausted",
    "market_miss_recovery_cap_exhausted": "market_miss_recovery_cap_exhausted",
    # The authority gate in scene progression, and reopen_scene_intent behind it.
    "scene_intake_authority_expired": "authority_expired",
    "scene_configuration_submission_scene_owner_authority_expired": "authority_expired",
    "scene_intake_authority_revoked": "authority_revoked",
    "scene_configuration_submission_scene_owner_authority_revoked": "authority_revoked",
    "scene_execution_owner_revoked": "authority_revoked",
    # The WebApp's authority-ending refusals (pubsub_handoff_listener) and the
    # Atlas preview's withdrawn set (website_atlas_preview).
    "consent_expired": "consent_expired",
    "consent_revoked": "consent_withdrawn",
    "website_capture_withdrawn": "consent_withdrawn",
    "source_revoked": "source_revoked",
    "capture_revoked_or_expired": "source_revoked",
    # Refusals the site can lift (website_atlas_preview, WebApp 409).
    "website_atlas_source_consent_required": "site_consent_required",
    "website_atlas_planner_consent_required": "site_consent_required",
    "website_atlas_consent_expired": "site_consent_required",
    "task_brief_missing": "task_brief_required",
    # Scene progression, the source resolver and task-family preparation.
    "source_kind_not_supported_by_progression": "source_kind_unsupported",
    "completed_3d_scene_result_required": "source_kind_unsupported",
    "metric_workcell_capture_and_object_geometry_required": "source_kind_unsupported",
    "scene_policy_checkpoint_capability_unavailable": "policy_capability_unavailable",
    "website_task_family_unknown": "task_family_unsupported",
}

#: Phases of a ``blocked`` row that are sealed Blueprint failures: the
#: configuration launch failed in a way no recovery takes (scene progression)
#: and the policy canary's terminal blocked results (the terminal reconciler).
PHASE_REASONS = {
    "scene_configuration_failed": "scene_configuration_failed",
    "policy_canary_blocked": "policy_canary_blocked",
    "policy_canary_preprovider_blocked": "policy_canary_blocked",
}
#: Statuses only the status read projects (no attempt was reserved).
STATUS_REASONS = {"revoked": "authority_revoked", "expired": "authority_expired"}
#: ``blocked`` phases that retry themselves: the configuration capacity wait.
TRANSIENT_PHASES = frozenset({"configuration_capacity"})
#: Statuses that stop and wait; an unclassified blocker there is a Blueprint fault.
STOPPED_STATUSES = frozenset({"blocked", "needs_input"})
#: Operator pauses, which the execution window bounds (scene progression).
OPERATOR_HOLD_BLOCKERS = frozenset({"scene_intent_paused", "scene_submission_paused"})
#: The notes ``_advance_intent`` writes before work it finishes in the same pass.
#: A row never rests on one, so they never clear a decision and the give-up
#: clock steps over them: a failure retried every pass is unchanged.
PROGRESS_MARKER_PHASES = frozenset({"source_preparation", "factory", "publication", "submission", "activation"})

_DECISION_FIELDS = ("kind", "terminal", "reason_category", "reason_code")
_RECORD_FIELDS = frozenset({"schema_version", "intent_id", "intent_digest", *_DECISION_FIELDS,
                            "decided_at_epoch", "outcome_digest"})
_KIND_RANK = {"withdrawn": 0, "held": 1, "awaiting_spend_approval": 3, "awaiting_site": 4}
#: load_progression refuses a longer event chain.
_WALK_EVENTS = 10_000
_WALK_SECONDS = 30.0
_MAX_RECORD_BYTES = 64 * 1024


class SceneOutcomeError(ValueError):
    """A retained outcome record is unreadable or not this intent's."""


def _codes(blockers: Iterable[Any] | None) -> set[str]:
    return {str(code).split(":", 1)[0] for code in blockers or ()}


def _precedence(reason: str) -> tuple[int, str]:
    kind, terminal, _category = REASONS[reason]
    return (2 if terminal else 5) if kind == "setup_repair" else _KIND_RANK[kind], reason


def _classify(status: str, phase: str | None, blockers: Iterable[Any] | None) -> str | None:
    if status == "completed":
        return None
    if status in STATUS_REASONS:
        return STATUS_REASONS[status]
    reasons = {BLOCKER_REASONS[code] for code in _codes(blockers) if code in BLOCKER_REASONS}
    if status == "blocked" and phase in PHASE_REASONS:
        reasons.add(PHASE_REASONS[phase])
    if reasons:
        # Privacy and rights first, then capability, then sealed failures.
        return min(reasons, key=_precedence)
    if status in STOPPED_STATUSES and phase not in TRANSIENT_PHASES:
        return "pipeline_fault"
    return None


def _decision(reason: str) -> dict[str, Any]:
    kind, terminal, category = REASONS[reason]
    return {"kind": kind, "terminal": terminal, "reason_category": category, "reason_code": reason}


def stall_clock_applies(status: str, phase: str | None, blockers: Iterable[Any] | None) -> bool:
    """Whether the 24-hour rule can end this row: no ending yet, no person's wait, no operator hold."""
    if status == "completed" or phase in OPERATOR_HOLD_PHASES or _codes(blockers) & OPERATOR_HOLD_BLOCKERS:
        return False
    return _classify(status, phase, blockers) in {None, "pipeline_fault"}


def decide_outcome(status: str, phase: str | None, blockers: Iterable[Any] | None,
                   unchanged_since_epoch: float | None, now: float) -> dict[str, Any] | None:
    """``None`` or ``{kind, terminal, reason_category, reason_code}`` for one progression row.

    ``unchanged_since_epoch`` is when the row's status, phase and blockers began
    (``None`` when unknown, which never gives up).
    """
    reason = _classify(status, phase, blockers)
    if (unchanged_since_epoch is not None and now - unchanged_since_epoch >= STALL_SECONDS
            and stall_clock_applies(status, phase, blockers)):
        reason = "stalled"
    return None if reason is None else _decision(reason)


def is_progress_marker(status: str | None, phase: str | None, blockers: Iterable[Any] | None) -> bool:
    return status == "preparing" and phase in PROGRESS_MARKER_PHASES and not list(blockers or ())


def _marker_event(event: Mapping[str, Any]) -> bool:
    return is_progress_marker(event.get("status"), event.get("phase"), event.get("blockers"))


def unchanged_since(directory: str | Path, progress: Mapping[str, Any]) -> float:
    """When the row's current status, phase and blockers began, stepping over mid-pass notes.

    The stall alarm's own walk. If the walk stops early, the time it reached is
    later than the true start: a stall it reports is real, and one it misses is
    found on a later pass.
    """
    budget = _Budget(deadline=time.monotonic() + _WALK_SECONDS, events=_WALK_EVENTS, clock=time.monotonic)
    since, _exact = _unchanged_since(Path(directory), progress, budget=budget, max_reads=_WALK_EVENTS,
                                     transparent=_marker_event)
    return since


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _revoked_at(path: Path) -> float | None:
    """When the owner revoked: the sealed receipt's time, else when the receipt was written."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        raw = b""
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if (isinstance(value, dict) and _finite(value.get("revoked_at_epoch"))
            and value.get("receipt_digest") == cross_runtime_canonical_digest(value, digest_field="receipt_digest")):
        return value["revoked_at_epoch"]
    return os.lstat(path).st_mtime


def _decided_at(directory: Path, intent: Mapping[str, Any], reason: str, now: float) -> float:
    # An ending the authority already decided is dated when it happened, so the
    # status read and the driver produce the same record.
    if reason == "authority_revoked":
        revoked = _revoked_at(directory / "revoked.json")
        if revoked is not None:
            return revoked
    if reason == "authority_expired":
        from .task_evaluation_scene_execution_window_evidence import effective_execution_expiry
        expiry = effective_execution_expiry(directory, intent)
        if expiry <= now:
            return expiry
    return now


def outcome_record(directory: str | Path, intent: Mapping[str, Any], decision: Mapping[str, Any], *,
                   now: float) -> dict[str, Any]:
    """The sealed ``outcome.json`` record for ``decision``."""
    value = {"schema_version": SCHEMA_VERSION, "intent_id": intent["intent_id"],
             "intent_digest": intent["intent_digest"], **{key: decision[key] for key in _DECISION_FIELDS},
             "decided_at_epoch": _decided_at(Path(directory), intent, decision["reason_code"], now)}
    value["outcome_digest"] = cross_runtime_canonical_digest(value, digest_field="outcome_digest")
    return value


def read_outcome(directory: str | Path, intent: Mapping[str, Any]) -> dict[str, Any] | None:
    """The intent's retained outcome, ``None`` when it has none; anything else refuses."""
    path = Path(directory) / FILENAME
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SceneOutcomeError("scene_outcome_record_unreadable") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SceneOutcomeError("scene_outcome_record_unreadable")
        raw = os.read(descriptor, _MAX_RECORD_BYTES + 1)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    reason = value.get("reason_code") if isinstance(value, dict) else None
    if not (len(raw) <= _MAX_RECORD_BYTES and isinstance(value, dict) and set(value) == _RECORD_FIELDS
            and value["schema_version"] == SCHEMA_VERSION
            and value["intent_id"] == intent["intent_id"] and value["intent_digest"] == intent["intent_digest"]
            and isinstance(reason, str) and reason in REASONS and type(value["terminal"]) is bool
            and (value["kind"], value["terminal"], value["reason_category"]) == REASONS[reason]
            and _finite(value["decided_at_epoch"])
            and value["outcome_digest"] == cross_runtime_canonical_digest(value, digest_field="outcome_digest")):
        raise SceneOutcomeError("scene_outcome_record_invalid")
    note_consulted(path, raw)
    return value


def terminal_outcome(directory: str | Path, intent: Mapping[str, Any]) -> dict[str, Any] | None:
    value = read_outcome(directory, intent)
    return value if value is not None and value["terminal"] is True else None


def record_outcome(directory: str | Path, intent: Mapping[str, Any], decision: Mapping[str, Any] | None, *,
                   now: float) -> dict[str, Any] | None:
    """Keep ``decision`` as the intent's outcome; the caller holds the intent's progression lock.

    A terminal outcome is never rewritten. The same decision keeps the time it
    was first made; a different one replaces a non-terminal outcome; ``None``
    removes it.
    """
    directory = Path(directory)
    existing = read_outcome(directory, intent)
    if existing is not None and existing["terminal"] is True:
        return existing
    if decision is None:
        if existing is not None:
            (directory / FILENAME).unlink()
        return None
    if existing is not None and all(existing[key] == decision[key] for key in _DECISION_FIELDS):
        return existing
    from .task_evaluation_scene_progression_state import atomic_json
    value = outcome_record(directory, intent, decision, now=now)
    atomic_json(directory / FILENAME, value)
    return value


def settle_outcome(directory: str | Path, intent: Mapping[str, Any], progress: Mapping[str, Any] | None, *,
                   now: float) -> dict[str, Any] | None:
    """Decide and record the outcome after a progression write; returns the retained outcome."""
    existing = read_outcome(directory, intent)
    if progress is None or (existing is not None and existing["terminal"] is True):
        return existing
    status, phase, blockers = progress["status"], progress.get("phase"), progress.get("blockers") or []
    since = unchanged_since(directory, progress) if stall_clock_applies(status, phase, blockers) else None
    decision = decide_outcome(status, phase, blockers, since, now)
    if is_progress_marker(status, phase, blockers) and (decision is None or decision["terminal"] is not True):
        return existing  # a mid-pass note only ever ends a row, never clears or replaces a decision
    return record_outcome(directory, intent, decision, now=now)


def status_outcome(directory: str | Path, intent: Mapping[str, Any], *, status: str,
                   blockers: Iterable[Any], now: float) -> dict[str, Any] | None:
    """The outcome the intent status reports; a status read never writes.

    ``completed`` supersedes any outcome. A revoked or expired projection reports
    the ending its authority already decided, exactly as the driver will record it.
    """
    if status == "completed":
        return None
    retained = read_outcome(directory, intent)
    if status in STATUS_REASONS and (retained is None or retained["terminal"] is not True):
        return outcome_record(directory, intent, decide_outcome(status, None, blockers, None, now), now=now)
    return retained


__all__ = [
    "BLOCKER_REASONS",
    "FILENAME",
    "KINDS",
    "OPERATOR_HOLD_BLOCKERS",
    "PHASE_REASONS",
    "PROGRESS_MARKER_PHASES",
    "REASONS",
    "REASON_CATEGORIES",
    "SCHEMA_VERSION",
    "STALL_SECONDS",
    "STATUS_REASONS",
    "SceneOutcomeError",
    "decide_outcome",
    "is_progress_marker",
    "outcome_record",
    "read_outcome",
    "record_outcome",
    "settle_outcome",
    "stall_clock_applies",
    "status_outcome",
    "terminal_outcome",
    "unchanged_since",
]
