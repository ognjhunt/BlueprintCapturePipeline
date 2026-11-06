"""ADP-010/day14: current source-bound preparation status and durable wake-ups.

A wake-up is never failure authority. The WebApp re-reads the signed status
route, which verifies the current source/context and the latest ledger revision.
No provider is dispatched, and handed_off is not native or assessment completion.

Beta journey L3/L6: v2 is additive. A WebApp that asks for it (request header
below) also learns that preparation ended for good (failed_terminal, with a
reason) and that an attempt only waits for control-plane capacity
(capacity_delayed) rather than for anything the site provides. Every other
caller, including a WebApp that predates v2, keeps the exact v1 receipt.

A preparation-only task stays unconfirmed, so its read also asks the WebApp
where the grant stands. A grant that expired still names the context it was
issued for, so the ledger is reported as for a current one. A preparation with
no grant yet is not read. The read never creates, renews or spends a grant.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SELECTORS = frozenset({"request_id", "scene_id", "capture_id", "completion_marker_generation",
                      "producer_delivery_key", "source_payload_sha256", "task_context_digest"})
DELIVERY_FILE = "website_preparation_status_delivery.json"
_UNAVAILABLE = "website_preparation_status_unavailable"
STATUS_V1 = "website_preparation_status.v1"
STATUS_V2 = "website_preparation_status.v2"
STATUS_V3 = "website_preparation_status.v3"
# v4 adds the one optional question the site can answer inside this job: which
# object the robot moves, when preparation went on with its best guess.
STATUS_V4 = "website_preparation_status.v4"
# v5 adds the job's robot fit table, a spec-bounded estimate (website_robot_fit_projection).
STATUS_V5 = "website_preparation_status.v5"
# v6 adds the job's robot fit precheck: whether a robot the screening catalog runs can do the task in
# this room, decided before any spend, and the hold it names when none can (website_robot_fit_precheck).
STATUS_V6 = "website_preparation_status.v6"
# The WebApp opts in per read; an absent or unknown value is answered with v1.
VERSION_HEADER = "x-blueprint-preparation-status-version"
_STATES_V1 = frozenset({"preparing", "awaiting_inputs", "failed_retryable", "authority_ended", "handed_off"})
_STATES_V2 = _STATES_V1 | {"capacity_delayed", "failed_terminal"}
_ACCEPTANCE = {"website_preparation_status_acceptance.v1": (STATUS_V1, _STATES_V1),
               "website_preparation_status_acceptance.v2": (STATUS_V2, _STATES_V2),
               "website_preparation_status_acceptance.v3": (STATUS_V3, _STATES_V2),
               "website_preparation_status_acceptance.v4": (STATUS_V4, _STATES_V2),
               "website_preparation_status_acceptance.v5": (STATUS_V5, _STATES_V2),
               "website_preparation_status_acceptance.v6": (STATUS_V6, _STATES_V2)}
_CODES = {"preparing": "preparation_in_progress", "awaiting_inputs": "preparation_inputs_pending",
          "capacity_delayed": "preparation_capacity_delayed", "failed_retryable": "preparation_retryable_failure",
          "failed_terminal": "preparation_failed_terminal", "authority_ended": "preparation_authority_ended",
          "handed_off": "preparation_handed_off"}
_LEDGER_STATES = {"processing": "preparing", "failed_retryable": "failed_retryable",
                  "retryable_blocked": "awaiting_inputs", "terminal_authority_ended": "authority_ended"}
# Supplement ids are WebApp-minted digests (captureSupplement.ts); the
# supplement's own birth and owner records still bind it to this request.
_SUPPLEMENT_CAPTURE = re.compile(r"supplement-[0-9a-f]{40}")


def requested_status_version(value: Any) -> str:
    """The newest receipt schema this Pipeline has at or below what the reader asked for."""
    text = value.strip() if type(value) is str else ""
    if re.fullmatch(r"[1-9][0-9]{0,2}", text):
        number = int(text)
        return (STATUS_V6 if number >= 6 else STATUS_V5 if number == 5 else STATUS_V4 if number == 4
                else STATUS_V3 if number == 3 else STATUS_V2 if number == 2 else STATUS_V1)
    return STATUS_V1


class PreparationStatusUnavailable(ValueError):
    """Fixed public refusal; internal paths/errors never cross this boundary.

    ``reason`` names the failed check with a fixed code for the private delivery
    record only (``unavailable_reason``). The message stays the public refusal.
    """

    def __init__(self, message: str = _UNAVAILABLE, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason


def _require(condition: bool, reason: str | None = None) -> None:
    if not condition:
        raise PreparationStatusUnavailable(_UNAVAILABLE, reason=reason)


_REASON_CODE = re.compile(r"[a-z][a-z0-9_.:-]{0,159}")


def unavailable_reason(exc: BaseException) -> str | None:
    """The fixed code behind a refused status read, for the delivery record.

    A named check's ``reason``, else the first fixed-code message of a
    ``ValueError`` or ``RuntimeError`` in the cause chain, such as
    ``website_control_scene-sponsorship_http_409:website_scene_sponsorship_missing``.
    Free text, paths and response bodies never match, so they are never kept.
    """
    current: BaseException | None = exc
    for _ in range(8):
        if current is None:
            return None
        reason = getattr(current, "reason", None)
        if isinstance(current, PreparationStatusUnavailable) and type(reason) is str and _REASON_CODE.fullmatch(reason):
            return reason
        message = current.args[0] if len(current.args) == 1 else None
        if (isinstance(current, (ValueError, RuntimeError)) and type(message) is str
                and message != _UNAVAILABLE and _REASON_CODE.fullmatch(message)):
            return message
        current = current.__cause__ or current.__context__
    return None


def validate_selectors(value: Any) -> dict[str, str]:
    _require(type(value) is dict and set(value) == SELECTORS)
    _require(all(type(item) is str for item in value.values()))
    request = value["request_id"]
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", request) is not None)
    _require(value["scene_id"] == f"site-{request}" and (value["capture_id"] == f"walkthrough-{request}"
             or _SUPPLEMENT_CAPTURE.fullmatch(value["capture_id"]) is not None))
    _require(re.fullmatch(r"[1-9][0-9]{0,19}", value["completion_marker_generation"]) is not None)
    _require(all(re.fullmatch(r"sha256:[0-9a-f]{64}", value[key]) is not None
                 for key in ("producer_delivery_key", "task_context_digest")))
    _require(re.fullmatch(r"[0-9a-f]{64}", value["source_payload_sha256"]) is not None)
    return dict(value)


def _read(path: Path) -> dict[str, Any]:
    _require(not path.is_symlink())
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict)
    return value


#: A parked delivery the owner retired (``website_preparation_delivery_retirement``): terminal for
#: its revision, never due, retried or counted; a newer producer revision starts a new delivery.
RETIRED_STATE = "retired"
RETIRED_REASON = "owner_retired_parked_delivery"
MAX_RETIREMENT_EVIDENCE = 500


def _read_delivery(path: Path) -> dict[str, Any]:
    """Only a valid scheduling record may receive canonical retry bookkeeping."""
    value = _read(path)
    _require(value.get("schema_version") == "website_preparation_status_delivery.v1"
             and value.get("state") in {"pending", "delivered", RETIRED_STATE}
             and type(value.get("revision")) is int and value["revision"] > 0)
    if value["state"] == RETIRED_STATE:
        retired_at, evidence = value.get("retired_at_epoch"), value.get("retirement_approval_evidence")
        _require(type(retired_at) in (int, float) and math.isfinite(retired_at) and retired_at >= 0
                 and value.get("retired_reason") == RETIRED_REASON
                 and isinstance(evidence, str) and 1 <= len(evidence) <= MAX_RETIREMENT_EVIDENCE)
    validate_selectors(value.get("selectors"))
    for field in ("delivery_attempt_count", "delivery_failure_count"):
        counter = value.get(field, 0)
        _require(type(counter) is int and counter >= 0)
    for field in ("pending_since_epoch", "next_attempt_at_epoch", "status_unavailable_since_epoch", "parked_at_epoch"):
        timestamp = value.get(field, 0.0)
        _require(type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp >= 0)
    return value


# D-05 (2026-11-02): seven callbacks whose status read failed on every attempt
# held the due backlog for 4.6 days. A delivery whose status read has failed on
# every attempt for a day is parked, not dropped: it stays pending with all its
# fields, is retried every six hours, and is counted apart from the due backlog.
# One successful read, or a newer revision from the producer, makes it live.
PARK_AFTER_SECONDS = 24 * 3600
PARKED_RETRY_SECONDS = 6 * 3600
PARKED_REASON = "preparation_status_persistently_unavailable"
_READ_STREAK_FIELDS = ("status_unavailable_since_epoch", "last_unavailable_reason", "parked_reason",
                       "parked_at_epoch")


def _instant(value: Any, *, now: float) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= now else None


def _status_read_outcome(updated: Mapping[str, Any], previous: Mapping[str, Any], *, now: float,
                         failure: BaseException | None) -> dict[str, Any]:
    """Failed-read bookkeeping for the status read that precedes each callback.

    A read that succeeds ends the streak and any parking. A failed read extends
    it; a streak a day long parks the delivery until its six-hour retry.
    """
    record = {key: value for key, value in updated.items() if key not in _READ_STREAK_FIELDS}
    if failure is None:
        return record
    since = _instant(previous.get("status_unavailable_since_epoch"), now=now)
    if (since is None and "last_attempt_at_epoch" not in previous
            and previous.get("delivery_failure_count", 0) > 0):
        # Recorded before streaks were kept: every failure the index counted on
        # this record came before any callback, so the streak began when it
        # became pending.
        since = _instant(previous.get("pending_since_epoch"), now=now)
    since = now if since is None else since
    record.update(status_unavailable_since_epoch=since, last_unavailable_reason=unavailable_reason(failure))
    if now - since >= PARK_AFTER_SECONDS:
        parked_at = _instant(previous.get("parked_at_epoch"), now=now)
        record.update(parked_reason=PARKED_REASON, parked_at_epoch=now if parked_at is None else parked_at,
                      next_attempt_at_epoch=now + PARKED_RETRY_SECONDS)
    return record


def _root(root: Path) -> Path:
    root = Path(root).absolute()
    _require(not any(item.is_symlink() for item in (root, *root.parents)))
    _require(root.is_dir())
    return root


def _retained_context(root: Path) -> dict[str, Any]:
    # This is written by the actual website task descriptor producer, before
    # sponsorship/provider work. Never replace it with a newly fetched context.
    return _read(root / "pipeline" / "website_task_context.json")


def _retained_context_or_none(root: Path) -> dict[str, Any] | None:
    """None only when no attempt has retained a context yet (it failed earlier).

    Such an attempt prepared nothing under any context, so its status is bound
    to the site's current context instead; nothing retained is replaced.
    """
    path = root / "pipeline" / "website_task_context.json"
    if path.exists() or path.is_symlink():
        return _retained_context(root)
    return None


def _current_context(*, request_id: str, scene_id: str, capture_id: str,
                     purpose: str | None, allow_unconfirmed_preparation: bool = False) -> dict[str, Any]:
    from .website_task_context import load_current_website_task_context
    arguments = {"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id}
    try:
        return load_current_website_task_context(**arguments, **({"purpose": purpose} if purpose else {}))
    except ValueError as exc:
        # Before a task snapshot exists, an ordinary capture may already have
        # an unconfirmed preparation proposal. Read that explicit scope only;
        # transport, identity and changed-context refusals remain refusals.
        if (not allow_unconfirmed_preparation or purpose is not None
                or str(exc) != "website_task_context_not_confirmed"):
            raise
        return load_current_website_task_context(**arguments, purpose="scene_preparation")


def _v2_state(ledger: Mapping[str, Any]) -> tuple[str, str | None]:
    """Typed v2 meaning of a ledger that is not handed off."""
    from .website_preparation_terminal import capacity_only, current_terminal, current_overdue
    status = ledger.get("status")
    _require(status in _LEDGER_STATES)
    terminal = current_terminal(ledger) if status != "terminal_authority_ended" else None
    if terminal is not None:
        return "failed_terminal", terminal["reason"]
    if status in {"failed_retryable", "retryable_blocked"} and capacity_only(ledger):
        return "capacity_delayed", None
    if status == "processing" and current_overdue(ledger) is not None:
        return "failed_retryable", None
    return _LEDGER_STATES[status], None


def read_preparation_status(*, capture_root: Path, selectors: Mapping[str, Any],
                            schema_version: str = STATUS_V1,
                            grantless_assessment_wait: bool = False) -> dict[str, Any]:
    """Read an as-of snapshot; refuse missing, changed or unverified authority."""
    from .capture_original_owner_observer import load_original_owner_observation
    from .consent_takedown import read_consent_state
    from .handoff_job_state import _read_job_ledger, _output_commit
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_task_context import validate_website_task_context

    try:
        _require(schema_version in {STATUS_V1, STATUS_V2, STATUS_V3, STATUS_V4, STATUS_V5, STATUS_V6})
        selected = validate_selectors(dict(selectors))
        root = _root(capture_root)
        _require(root.name == selected["capture_id"] and root.parent.name == "captures"
                 and root.parent.parent.name == selected["scene_id"])
        _require(not (root / "pipeline_job_ledger.json").is_symlink())
        before = _read_job_ledger(root)
        _require(before.get("schema_version") == "pipeline_job_ledger.v1")
        _require(type(before.get("revision")) is int and before["revision"] > 0
                 and type(before.get("attempt_count")) is int and before["attempt_count"] > 0)
        _require(all(before.get(key) == selected[key] for key in
                     ("scene_id", "capture_id", "producer_delivery_key", "source_payload_sha256")))
        birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
        _require(type(birth) is dict, "capture_birth_missing")
        _require(all(birth.get(key) == selected[key] for key in ("request_id", "scene_id", "capture_id")))
        _require(birth.get("delivery_key") == selected["producer_delivery_key"])
        delivery = _read(Path(birth["birth_delivery_raw_ref"]["path"]))
        purpose = birth.get("capture_observation_purpose")
        if purpose not in {None, "scene_preparation"}:
            raise ValueError("website_preparation_purpose_invalid")
        _require(delivery["source_finalize"]["generation"] == selected["completion_marker_generation"])
        owner = load_original_owner_observation(bucket=delivery["source_finalize"]["bucket"], scene_id=selected["scene_id"],
            capture_id=selected["capture_id"], marker_generation=selected["completion_marker_generation"],
            **({"expected_purpose": purpose} if purpose else {}))
        _require((owner["capture_owner"] or {}).get("user_id") == birth["capture_owner_user_id"],
                 "capture_owner_changed")
        _require(owner["producer_delivery"] == delivery["producer_delivery"], "capture_source_changed")
        _require(owner["producer_delivery"]["delivery_key"] == selected["producer_delivery_key"],
                 "capture_source_changed")
        _require(owner["producer_delivery"]["raw_video"]["generation"] == birth["raw_video"]["generation"],
                 "capture_source_changed")
        _require(owner["producer_delivery"]["raw_video"]["object_name"] == birth["raw_video"]["object_name"],
                 "capture_source_changed")
        # Current fresh owner transport validates rights and complete generation
        # authority; historical birth alone grants no current consent.
        _require(read_consent_state(root)["state"] != "revoked", "consent_revoked")
        raw_retained = _retained_context_or_none(root)
        # Birth scope authenticates the original owner. Task preparation has
        # its own scope and may remain unconfirmed under that same capture.
        context_purpose = raw_retained.get("purpose", purpose) if raw_retained is not None else purpose
        retained = None if raw_retained is None else validate_website_task_context(raw_retained,
                         request_id=selected["request_id"], scene_id=selected["scene_id"],
                         capture_id=selected["capture_id"], purpose=context_purpose)
        current = _current_context(request_id=selected["request_id"],
                        scene_id=selected["scene_id"], capture_id=selected["capture_id"],
                        purpose=context_purpose, allow_unconfirmed_preparation=raw_retained is None)
        _require(current["context_digest"] == selected["task_context_digest"]
                 and (retained is None or retained["context_digest"] == current["context_digest"]),
                 "task_context_changed")
        # A completed job always retained its context; never hand off without it.
        _require(retained is not None or before.get("status") != "completed")
        if current.get("confirmed") is False:
            from .website_assessment_resume import PENDING_CODE
            from .website_task_context import website_scene_sponsorship_standing
            # A status grants nothing, so a grant that expired still binds the
            # context it was issued for. It never changes the reported state.
            # A capture that waited on its assessment never had a grant: only
            # its resume may read it, because that resume requests the first one.
            waits_on_assessment = (grantless_assessment_wait and before.get("status") == "failed_retryable"
                                   and before.get("last_error") == PENDING_CODE
                                   and isinstance(before.get("assessment_resume"), dict))
            website_scene_sponsorship_standing(task_context=current, now=time.time(),
                                               allow_missing=waits_on_assessment)
        terminal_reason = None
        if before.get("status") == "completed":
            from .website_task_context_contracts import (
                holds_native_task_preparation, retains_geometry_only_fit, retains_robot_fit_hold,
                retains_task_family_preparation,
            )
            handoff = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json")
            from .website_target_background_correction import retains_rejected_selection
            from .observed_room_contracts import retains_observed_room_limit
            native_profile_held = (holds_native_task_preparation(handoff, current)
                                   or retains_rejected_selection(handoff, current)
                                   or retains_observed_room_limit(handoff, current)
                                   or retains_geometry_only_fit(handoff, current)
                                   or retains_robot_fit_hold(handoff, current))
            family_prepared = retains_task_family_preparation(handoff, current)
            _require(handoff.get("schema_version") == "website_scene_handoff.v1"
                     and handoff.get("scene_id") == selected["scene_id"]
                     and handoff.get("capture_id") == selected["capture_id"]
                     and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
                     and (native_profile_held or family_prepared or
                          handoff.get("website_intake_outbox", {}).get("state") == "forward_pending"))
            _require(bool(_output_commit(root, scene_id=selected["scene_id"], capture_id=selected["capture_id"])))
            state = ("awaiting_inputs" if native_profile_held or (family_prepared and handoff["status"] == "needs_input")
                     else "handed_off")
        elif schema_version in {STATUS_V2, STATUS_V3, STATUS_V4, STATUS_V5, STATUS_V6}:
            state, terminal_reason = _v2_state(before)
        else:
            _require(before.get("status") in _LEDGER_STATES)
            state = _LEDGER_STATES[before["status"]]
        outcome = question = robot_fit = precheck = None
        if schema_version in {STATUS_V3, STATUS_V4, STATUS_V5, STATUS_V6}:
            from .website_preparation_outcome import read_preparation_outcome
            outcome = read_preparation_outcome(capture_root=root,
                task_context_digest=selected["task_context_digest"], state=state,
                robot_fit_hold=schema_version == STATUS_V6)
        if schema_version in {STATUS_V4, STATUS_V5, STATUS_V6} and state != "authority_ended":
            # Optional: the job continues on its guess whether or not this is answered.
            from .website_task_target_choice import public_question
            question = public_question(root)
        if schema_version in {STATUS_V5, STATUS_V6}:
            from .website_robot_fit_projection import read_robot_fit
            robot_fit = read_robot_fit(capture_root=root, task_context_digest=selected["task_context_digest"])
        if schema_version == STATUS_V6:
            from .website_robot_fit_precheck_projection import read_robot_fit_precheck
            precheck = read_robot_fit_precheck(capture_root=root, task_context_digest=selected["task_context_digest"])
        after = _read_job_ledger(root)
        _require(before == after, "ledger_changed")
        _require(read_consent_state(root)["state"] != "revoked", "consent_revoked")
        result = {"schema_version": schema_version, **selected,
                  "attempt_count": before["attempt_count"], "revision": before["revision"],
                  "state": state, "code": _CODES[state],
                  **({"terminal_reason": terminal_reason} if schema_version != STATUS_V1 else {}),
                  **({"outcome": outcome} if schema_version in {STATUS_V3, STATUS_V4, STATUS_V5, STATUS_V6} else {}),
                  **({"question": question} if schema_version in {STATUS_V4, STATUS_V5, STATUS_V6} else {}),
                  **({"robot_fit": robot_fit} if schema_version in {STATUS_V5, STATUS_V6} else {}),
                  **({"robot_fit_precheck": precheck} if schema_version == STATUS_V6 else {}),
                  "correlation_id": cross_runtime_canonical_digest({**selected,
                      "attempt_count": before["attempt_count"], "revision": before["revision"]})}
        result["status_digest"] = cross_runtime_canonical_digest(result)
        return result
    except PreparationStatusUnavailable:
        raise
    except Exception as exc:
        raise PreparationStatusUnavailable(_UNAVAILABLE) from exc


def retain_preparation_wakeup(capture_root: Path) -> None:
    """Persist only selectors after a committed ledger; no network under lock.

    Reconciliation scans actual ledgers too, closing the crash gap between a
    ledger commit and this separate durable delivery record.
    """
    from .handoff_job_state import _existing_job_ledger_lock, _read_job_ledger
    from .task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from .website_preparation_delivery_index import update as update_index
    root = _root(capture_root)
    birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
    if not birth:
        return
    context = _retained_context_or_none(root)
    if context is None:
        if not (root / "pipeline_job_ledger.json").is_file():
            return  # Nothing claimed yet; the lock below re-checks before any write.
        # An attempt that failed before retaining a context is still reported,
        # bound to the site's current context (one signed read, before the lock).
        purpose = birth.get("capture_observation_purpose")
        _require(purpose in {None, "scene_preparation"})
        context = _current_context(request_id=birth["request_id"], scene_id=birth["scene_id"],
            capture_id=birth["capture_id"], purpose=purpose, allow_unconfirmed_preparation=True)
    with _existing_job_ledger_lock(root) as lock_state:
        if lock_state != "ledger_present":
            return
        ledger = _read_job_ledger(root)
        marker = _read(Path(birth["birth_delivery_raw_ref"]["path"]))["source_finalize"]["generation"]
        selected = validate_selectors({"request_id": birth["request_id"], "scene_id": birth["scene_id"],
            "capture_id": birth["capture_id"], "completion_marker_generation": marker,
            "producer_delivery_key": ledger.get("producer_delivery_key"),
            "source_payload_sha256": ledger.get("source_payload_sha256"),
            "task_context_digest": context.get("context_digest")})
        _require(birth["delivery_key"] == selected["producer_delivery_key"])
        revision = ledger.get("revision")
        _require(type(revision) is int and revision > 0)
        path = root / DELIVERY_FILE
        old = _read_delivery(path) if path.is_file() else {}
        if old.get("state") == RETIRED_STATE and old.get("revision") == revision:
            # The owner retired this revision's delivery: it stays retired, and its hint stays gone.
            update_index(root, old)
            return
        if old.get("selectors") == selected and old.get("revision") == revision:
            if old.get("state") == "pending" and "pending_since_epoch" not in old:
                old = {**old, "pending_since_epoch": time.time(), "next_attempt_at_epoch": 0.0}
                write_json(path, old)
            update_index(root, old)
            return
        delivery = {"schema_version": "website_preparation_status_delivery.v1",
                          "state": "pending", "selectors": selected, "revision": revision,
                          "delivery_attempt_count": 0, "delivery_failure_count": 0,
                          "pending_since_epoch": time.time(), "next_attempt_at_epoch": 0.0}
        write_json(path, delivery)
        update_index(root, delivery)


def validate_acceptance(value: Any, selected: Mapping[str, str]) -> dict[str, Any]:
    """Require the source-bound receipt issued only after the WebApp commit."""
    fields = SELECTORS | {"schema_version", "accepted", "status_digest", "state",
                          "native_execution_complete", "correlation_id"}
    _require(type(value) is dict and set(value) == fields)
    # The acceptance names the receipt version the WebApp read (v1 from a WebApp
    # that predates v2), so postflight re-reads that same version.
    _require(value["schema_version"] in _ACCEPTANCE
             and value["accepted"] is True and value["native_execution_complete"] is False)
    _require(all(value[key] == selected[key] for key in SELECTORS))
    digest = value["status_digest"]
    _require(type(digest) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None)
    _require(value["state"] in _ACCEPTANCE[value["schema_version"]][1])
    _require(value["correlation_id"] == "bp-prep-" + digest.removeprefix("sha256:")[:16])
    return dict(value)


def accepted_status_version(acceptance: Mapping[str, Any]) -> str:
    return _ACCEPTANCE[acceptance["schema_version"]][0]


def reconcile_preparation_wakeups(storage_root: Path, *, limit: int = 50,
                                 now_epoch: float | None = None) -> dict[str, int]:
    """Existing listener tick retries delivery independently of provider work."""
    from .handoff_job_state import _existing_job_ledger_lock, _read_job_ledger
    from .website_preparation_terminal import note_sweep_liveness, record_unretried_preparation
    from .website_preparation_delivery_index import due, defer_unavailable, update as update_index, METRICS_FILE
    from .website_task_context import website_webapp_request
    counts = {"attempted": 0, "delivered": 0, "unavailable": 0}
    storage_root = Path(storage_root)
    try:
        # Liveness qualifies overdue observations, never per-message exhaustion.
        alive_since = note_sweep_liveness(storage_root)
    except OSError:
        alive_since = None
    roots = sorted(path.parent for path in storage_root.glob("*/scenes/site-*/captures/*/pipeline_job_ledger.json")
                   if path.parent.name.startswith(("walkthrough-", "supplement-")))
    cursor_path = storage_root / ".website_preparation_delivery_cursor.json"
    try:
        cursor = _read(cursor_path).get("last_capture_digest") if cursor_path.is_file() else None
    except (OSError, ValueError):
        cursor = None
    keys = [hashlib.sha256(str(root).encode()).hexdigest() for root in roots]
    if cursor in keys:
        pivot = keys.index(cursor) + 1
        roots = roots[pivot:] + roots[:pivot]
    maximum = max(0, min(limit, 50))
    # Historical/index repair has its own budget. Settled history cannot consume
    # the callback budget; newly retained revisions enter the due index directly.
    bounded = roots[:maximum]
    for root in bounded:
        try:
            if alive_since is not None:
                record_unretried_preparation(root, alive_since=alive_since)
            retain_preparation_wakeup(root)
        except Exception:
            counts["unavailable"] += 1
    now = time.time() if now_epoch is None else now_epoch
    try:
        pending_roots, metrics = due(storage_root, now=now, limit=maximum)
    except Exception:
        pending_roots, metrics = [], {"index_unavailable": True}
        counts["unavailable"] += 1
    success_lags = []
    for pending in pending_roots:
        root = pending.capture_root
        delivery = None
        path = root / DELIVERY_FILE
        attempted = False
        try:
            # Retain the canonical pending record before a current-source
            # lookup can fail. Its failure backoff must not monopolize a small
            # dispatch budget or leave unrelated healthy callbacks unvisited.
            if path.is_file():
                delivery = _read_delivery(path)
            retain_preparation_wakeup(root)
            _require(path.is_file())
            delivery = _read_delivery(path)
            if delivery.get("state") != "pending":
                update_index(root, delivery)
                continue
            if delivery.get("next_attempt_at_epoch", 0.0) > now:
                continue
            selected = validate_selectors(delivery["selectors"])
            read_preparation_status(capture_root=root, selectors=selected)
            counts["attempted"] += 1
            attempted = True
            error = None
            acceptance = None
            current_status = None
            try:
                acceptance = validate_acceptance(website_webapp_request(capture_id=selected["capture_id"],
                    operation="preparation-status", payload=selected), selected)
                # The callback may legitimately commit a newer status than our
                # preflight. Verify against fresh authority, never an old receipt.
                current_status = read_preparation_status(capture_root=root, selectors=selected,
                                                         schema_version=accepted_status_version(acceptance))
                _require(acceptance["status_digest"] == current_status["status_digest"]
                         and acceptance["state"] == current_status["state"])
            except Exception as exc:
                error = type(exc).__name__  # Private retry bookkeeping only, never raw response/error.
            # Network is over. Under lock update only the still-current delivery;
            # a concurrent newer producer retains its own pending revision.
            with _existing_job_ledger_lock(root) as lock_state:
                if lock_state != "ledger_present" or _read(path) != delivery:
                    continue
                if error is None:
                    ledger = _read_job_ledger(root)
                    _require(current_status is not None and ledger.get("revision") == current_status["revision"]
                             and ledger.get("attempt_count") == current_status["attempt_count"])
                failures = delivery.get("delivery_failure_count", 0) + int(error is not None)
                updated = {**delivery, "state": "pending" if error else "delivered",
                                  "delivery_attempt_count": delivery.get("delivery_attempt_count", 0) + 1,
                                  "delivery_failure_count": failures,
                                  "next_attempt_at_epoch": now + min(1800, 60 * 2 ** min(failures - 1, 5)) if error else 0.0,
                                  "last_attempt_at_epoch": now,
                                  "delivered_at_epoch": now if error is None else None,
                                  "last_error_type": error, "acceptance": acceptance}
                # The status read succeeded before this callback: unparked and live.
                updated = _status_read_outcome(updated, delivery, now=now, failure=None)
                write_json(path, updated)
                update_index(root, updated)
            counts["delivered"] += int(error is None)
            if error is None:
                success_lags.append(max(0.0, now - delivery.get("pending_since_epoch", now)))
        except Exception as exc:
            counts["unavailable"] += 1
            # Source/control outages also back off. This changes no authority,
            # and cannot acknowledge or overwrite a concurrent newer revision.
            if delivery is not None and delivery.get("state") == "pending":
                try:
                    with _existing_job_ledger_lock(root) as lock_state:
                        if lock_state == "ledger_present" and _read(path) == delivery:
                            failures = delivery.get("delivery_failure_count", 0) + 1
                            updated = {**delivery, "state": "pending",
                                "delivery_attempt_count": delivery.get("delivery_attempt_count", 0) + int(attempted),
                                "delivery_failure_count": failures,
                                "next_attempt_at_epoch": now + min(1800, 60 * 2 ** min(failures - 1, 5)),
                                "last_error_type": type(exc).__name__}
                            # Before the callback this is a failed status read; a day of them parks.
                            updated = _status_read_outcome(updated, delivery, now=now,
                                                           failure=None if attempted else exc)
                            write_json(path, updated)
                            update_index(root, updated)
                except Exception:
                    pass
            # Unreadable canonical bytes cannot receive canonical bookkeeping.
            # Preserve them and defer their exact recoverable hint instead. CAS
            # refuses a concurrent new revision or a changed scheduling hint.
            try:
                defer_unavailable(pending, now=now)
            except Exception:
                pass
    if bounded:
        # Fair bounded scan across restarts; this cursor selects only members of
        # the server-owned glob, never an arbitrary path or a source authority.
        try:
            write_json(cursor_path, {"schema_version": "website_preparation_delivery_cursor.v1",
                       "last_capture_digest": hashlib.sha256(str(bounded[-1]).encode()).hexdigest()})
        except OSError:
            counts["unavailable"] += 1
    try:
        write_json(storage_root / METRICS_FILE, {"schema_version": "website_preparation_delivery_metrics.v1",
                   "observed_at_epoch": now, **metrics, **counts, "repair_scanned": len(bounded),
                   "successful_delivery_lag_seconds": success_lags})
    except OSError:
        counts["unavailable"] += 1
    return counts
