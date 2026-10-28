"""Retained DLQ successors and bounded safe replay in the existing listener.

The default is observe/classify: no replay or acknowledgement. Explicit safe
rollout transfers a message only after its private payload and provenance are
fsynced. Only named pre-lease failures can execute, with the canonical listener's
unattempted-delivery CAS. Unknown effects and existing waits keep their original
reconciler; source counters, budgets and authority are never reset.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest

DIRECTORY = ".handoff_recovery"
MAX_PAYLOAD_BYTES = 1_000_000
RETENTION_SECONDS = 7 * 24 * 3600
SOURCE_ATTRIBUTE = "CloudPubSubDeadLetterSourceSubscription"
STATUS_SCHEMA = "blueprint.handoff_recovery_status.v1"
DISPOSITION_SCHEMA = "blueprint.handoff_recovery_disposition.v1"
_SAFE_CAUSES = {
    ("capture_owner_observation_unavailable_retryable", "capture_owner_response_unavailable"):
        "capture_owner_transport_unavailable",
    ("capture_source_membership_unavailable_retryable", "capture_staging_transport_unavailable"):
        "capture_staging_transport_unavailable",
}
_ACTIONS = {"capture_owner_transport_unavailable": "restore_capture_owner_observation",
            "capture_membership_transport_unavailable": "restore_generation_pinned_membership",
            "capture_staging_transport_unavailable": "restore_capture_staging"}
_SUMMARY_ACTIONS = {**_ACTIONS, "classification_pending": "classify_original_effect_state",
    "broker_source_unproven": "verify_retained_dlq_source",
    "prior_delivery_source_conflict": "reconcile_original_delivery",
    "existing_explicit_processing_wait": "existing_wait_reconciler",
    "original_effect_state_unproven": "reconcile_original_operation",
    "capture_owner_access_denied": "restore_authorized_capture_owner_read",
    "capture_staging_access_denied": "restore_authorized_capture_source_read",
    "capture_staging_source_invalid": "correct_original_source_evidence"}
_HELD_CAUSES = {"capture_owner_response_denied": "capture_owner_access_denied",
                "capture_staging_access_denied": "capture_staging_access_denied",
                "capture_staging_source_invalid": "capture_staging_source_invalid"}


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("handoff_recovery_record_unavailable")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("handoff_recovery_record_invalid")
    return value


def _root(storage_root: Path) -> Path:
    root = Path(storage_root) / DIRECTORY
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("handoff_recovery_root_unsafe")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = root.stat()
    if status.st_uid != os.geteuid() or status.st_mode & 0o077:
        raise ValueError("handoff_recovery_root_unsafe")
    return root


def _once(path: Path, value: Mapping[str, Any]) -> None:
    from .beta_policy_screening_launch import _write
    try:
        _write(path, value)
    except FileExistsError:
        if _read(path) != value:
            raise ValueError("handoff_recovery_record_conflict") from None


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    fd = os.open(root / "recovery.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _selectors(handoff) -> dict[str, Any]:
    return {"bucket": handoff.bucket, "scene_id": handoff.scene_id, "capture_id": handoff.capture_id,
            "source_finalize": dict(handoff.source_finalize) if handoff.source_finalize else None,
            "source_membership_selector": dict(handoff.source_membership_selector)
                if handoff.source_membership_selector else None}


def _source_authority(handoff) -> dict[str, Any]:
    selectors = _selectors(handoff)
    if handoff.source_finalize:
        # Delivery event IDs and JSON layout are not source authority changes.
        selectors["source_finalize"] = {key: handoff.source_finalize[key]
                                        for key in ("bucket", "object_name", "generation")}
    return selectors


def _source_hold(storage_root: Path, handoff) -> dict[str, Any] | None:
    scope = _source_authority(handoff)
    path = Path(storage_root) / DIRECTORY / "source_holds" / (canonical_digest(scope)[7:] + ".json")
    if not path.exists():
        return None
    _root(storage_root)
    value = _read(path)
    if (value.get("schema_version") != "blueprint.handoff_source_authority_hold.v1"
            or value.get("source_authority") != scope or value.get("cause") not in _HELD_CAUSES.values()):
        raise ValueError("handoff_recovery_source_hold_invalid")
    return value


def record_source_result(listener, *, storage_root: Path, subscription: str, message: Any,
                         received: Any, result: Mapping[str, Any] | None) -> None:
    """Private named pre-lease evidence; no exception text or payload in status."""
    _record_result(listener, storage_root=storage_root, subscription=subscription, raw=bytes(message.data),
                   result=result, source_message_id=str(message.message_id),
                   delivery_attempt=getattr(received, "delivery_attempt", None))


def _record_result(listener, *, storage_root: Path, subscription: str, raw: bytes,
                   result: Mapping[str, Any] | None, source_message_id=None, delivery_attempt=None) -> None:
    handoff = listener.parse_handoff_payload(raw)
    code = (result or {}).get("owner_observation_reason") or (result or {}).get("staging_reason")
    cause = _SAFE_CAUSES.get(((result or {}).get("status"), code))
    held = _HELD_CAUSES.get(code)
    # A duplicate can fail its read while the original owns paid effects.
    # That new pre-lease failure never erases the original effect ledger.
    if listener._read_job_ledger(listener._handoff_capture_root(handoff, storage_root=storage_root)):
        cause = None
        held = None
    root = _root(storage_root) / "source_observations"
    root.mkdir(exist_ok=True, mode=0o700)
    if held:
        scope = _source_authority(handoff)
        holds = root.parent / "source_holds"
        holds.mkdir(exist_ok=True, mode=0o700)
        from .beta_policy_screening_launch import _write
        try:
            _write(holds / (canonical_digest(scope)[7:] + ".json"), {
                "schema_version": "blueprint.handoff_source_authority_hold.v1",
                "source_authority": scope, "cause": held})
        except FileExistsError:
            pass  # The first refusal is absorbing for these exact source selectors.
    retained_hold = _source_hold(storage_root, handoff)
    if retained_hold:
        held, cause = retained_hold["cause"], None
    digest = hashlib.sha256(raw).hexdigest()
    descriptor = os.open(root / (digest + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        path = root / (digest + ".json")
        previous = _read(path) if path.exists() else {}
        # A later outage, release, timer or duplicate delivery cannot supply
        # materially new authorization for these exact immutable source bytes.
        if previous.get("effect_state") in {"waiting_for_authorized_action", "permanent_invalid"}:
            return
        write_json(path, {
            "schema_version": "blueprint.handoff_source_recovery_observation.v1", "source_payload_sha256": digest,
            "source_subscription": subscription,
            "source_message_id": source_message_id if source_message_id is not None else previous.get("source_message_id"),
            "source_delivery_attempt": delivery_attempt if source_message_id is not None else previous.get("source_delivery_attempt"),
            **_selectors(handoff),
            "effect_state": "pre_lease" if cause else "permanent_invalid" if held == "capture_staging_source_invalid"
                else "waiting_for_authorized_action" if held else "unproven",
            "cause": cause or held or "original_effect_state_unproven", "observed_at_epoch": time.time()})
    finally:
        os.close(descriptor)


def _published_at(message: Any, fallback: float) -> float | None:
    value = getattr(message, "publish_time", None)
    try:
        epoch = float(value.timestamp()) if callable(getattr(value, "timestamp", None)) else float(value)
        if not math.isfinite(epoch) or epoch <= 0 or epoch > fallback + 60:
            raise ValueError("invalid publish time")
        return epoch
    except (ValueError, TypeError, OverflowError):
        return None


def retained_source_hold(listener, *, storage_root: Path, handoff, payload_digest: str) -> dict[str, Any] | None:
    """An ordinary broker redelivery cannot reread a denied immutable source."""
    path = Path(storage_root) / DIRECTORY / "source_observations" / (payload_digest + ".json")
    retained = _source_hold(storage_root, handoff)
    if retained:
        value = {"source_payload_sha256": payload_digest,
                 "effect_state": "permanent_invalid" if retained["cause"] == "capture_staging_source_invalid"
                    else "waiting_for_authorized_action", **_selectors(handoff), "cause": retained["cause"]}
    elif not path.exists():
        return None
    else:
        _root(storage_root)
        value = _read(path)
    if (value.get("source_payload_sha256") == payload_digest
            and value.get("effect_state") in {"waiting_for_authorized_action", "permanent_invalid"}
            and value.get("cause") in _HELD_CAUSES.values()
            and all(value.get(key) == item for key, item in _selectors(handoff).items())):
        reason = next(code for code, cause in _HELD_CAUSES.items() if cause == value["cause"])
        owner = reason == "capture_owner_response_denied"
        return {"schema_version": "v1", "status": "capture_owner_observation_unavailable_retryable" if owner
                    else "capture_source_membership_unavailable_retryable", "queue_disposition": "retryable",
                "bucket": handoff.bucket, "scene_id": handoff.scene_id, "capture_id": handoff.capture_id,
                "capture_root": str(listener._handoff_capture_root(handoff, storage_root=storage_root)),
                "owner_observation_reason" if owner else "staging_reason": reason,
                "blockers": [value["cause"]], "next_action": _SUMMARY_ACTIONS[value["cause"]]}
    return None


def retain_dlq_message(listener, *, storage_root: Path, subscription: str, source_subscription: str,
                       received: Any, now: float) -> Path:
    """Fsync an exact recoverable successor before any acknowledgement."""
    root = _root(storage_root)
    message, raw = received.message, bytes(received.message.data)
    if not 0 < len(raw) <= MAX_PAYLOAD_BYTES:
        raise ValueError("handoff_recovery_payload_size_refused")
    attributes = dict(message.attributes)
    if (any(not isinstance(k, str) or not isinstance(v, str) for k, v in attributes.items())
            or len(json.dumps(attributes).encode()) > 65_536):
        raise ValueError("handoff_recovery_provenance_size_refused")
    digest = hashlib.sha256(raw).hexdigest()
    entry = root / digest
    entry.mkdir(mode=0o700, exist_ok=True)
    try:
        selectors, valid = _selectors(listener.parse_handoff_payload(raw)), True
    except listener.PipelineError:
        selectors, valid = {}, False
    event = {"schema_version": "blueprint.handoff_recovery_event.v1", "source_payload_sha256": digest,
             "payload_base64": base64.b64encode(raw).decode(), "payload_byte_count": len(raw),
             "payload_valid": valid, **selectors}
    _once(entry / "event.json", event)
    message_id = str(getattr(message, "message_id", "") or "")
    if not message_id:
        raise ValueError("handoff_recovery_broker_identity_missing")
    published = _published_at(message, now)
    forwarded_source = attributes.get(SOURCE_ATTRIBUTE)
    source_matches = forwarded_source == source_subscription
    provenance = {"schema_version": "blueprint.handoff_recovery_broker_provenance.v1",
        "source_payload_sha256": digest, "dlq_subscription": subscription,
        "expected_source_subscription": source_subscription, "broker_source_subscription": forwarded_source,
        "source_subscription_verified": source_matches, "dlq_message_id": message_id,
        "broker_attributes": attributes, "dlq_publish_at_epoch": published,
        "retention_deadline_epoch": published + RETENTION_SECONDS if published is not None else None,
        "publish_time_proven": published is not None}
    identity = hashlib.sha256((subscription + "\0" + message_id).encode()).hexdigest()
    _once(entry / "broker" / (identity + ".json"), provenance)
    state_path = entry / "state.json"
    if not state_path.exists():
        from .beta_policy_screening_launch import _write
        initial = {"schema_version": "blueprint.handoff_recovery_state.v1",
            "source_payload_sha256": digest, "status": "retained", "cause": "classification_pending",
            "next_action": "classify_original_effect_state", "first_retained_at_epoch": now,
            "next_attempt_at_epoch": now, "attempts": 0}
        try:
            _write(state_path, initial)
        except FileExistsError:
            pass  # Never reset a concurrently processing durable successor.
    _state(entry)  # A matching digest alone cannot justify acknowledging an unusable successor.
    return entry


def _event(listener, entry: Path) -> tuple[dict[str, Any], bytes, Any]:
    event = _read(entry / "event.json")
    raw = base64.b64decode(event["payload_base64"], validate=True)
    if (hashlib.sha256(raw).hexdigest() != event.get("source_payload_sha256")
            or event.get("source_payload_sha256") != entry.name or len(raw) != event.get("payload_byte_count")):
        raise ValueError("handoff_recovery_payload_changed")
    try:
        handoff = listener.parse_handoff_payload(raw)
    except listener.PipelineError:
        handoff = None
    if event["payload_valid"] is not (handoff is not None):
        raise ValueError("handoff_recovery_payload_classification_changed")
    if handoff is not None and _selectors(handoff) != {key: event[key] for key in _selectors(handoff)}:
        raise ValueError("handoff_recovery_source_changed")
    return event, raw, handoff


def _provenance(entry: Path, source_subscription: str) -> list[dict[str, Any]]:
    values = [_read(path) for path in (entry / "broker").glob("*.json")]
    return [value for value in values if value.get("source_payload_sha256") == entry.name
            and value.get("source_subscription_verified") is True
            and value.get("broker_source_subscription") == source_subscription
            and value.get("expected_source_subscription") == source_subscription]


def _classify(listener, *, storage_root: Path, entry: Path, source_subscription: str) -> dict[str, Any]:
    event, _raw, handoff = _event(listener, entry)
    if not _provenance(entry, source_subscription):
        return {"status": "orphaned", "cause": "broker_source_unproven", "next_action": "verify_retained_dlq_source"}
    if handoff is None:
        return {"status": "permanent_invalid", "cause": "handoff_payload_invalid", "next_action": "none"}
    capture = listener._handoff_capture_root(handoff, storage_root=storage_root)
    ledger = listener._read_job_ledger(capture)
    ledger_binding = {key: ledger.get(key) for key in ("attempt_count", "lease_token", "revision", "producer_delivery_key")}
    if ledger:
        if (ledger.get("source_payload_sha256") != entry.name
                and entry.name not in listener._ended_payload_digests(ledger)):
            return {"status": "reconciliation_required", "cause": "prior_delivery_source_conflict",
                    "next_action": "reconcile_original_delivery", **ledger_binding}
        if ledger.get("status") == "completed" and listener._output_commit(
                capture, scene_id=handoff.scene_id, capture_id=handoff.capture_id):
            return {"status": "completed", "cause": "canonical_output_committed", "next_action": "none", **ledger_binding}
        if (ledger.get("status") == listener.TERMINAL_AUTHORITY_STATUS
                or entry.name in listener._ended_payload_digests(ledger)):
            return {"status": "authority_ended", "cause": "canonical_authority_ended", "next_action": "none", **ledger_binding}
        from .website_reconstruction_resume import armed
        if armed(ledger) or ledger.get("assessment_resume"):
            return {"status": "waiting_existing_resume", "cause": "existing_explicit_processing_wait",
                    "next_action": "existing_wait_reconciler", **ledger_binding}
        return {"status": "reconciliation_required", "cause": "original_effect_state_unproven",
                "next_action": "reconcile_original_operation", **ledger_binding}
    held = _source_hold(storage_root, handoff)
    if held:
        return {"status": "permanent_invalid" if held["cause"] == "capture_staging_source_invalid" else "waiting_authorized_action",
                "cause": held["cause"], "next_action": _SUMMARY_ACTIONS[held["cause"]]}
    state = _read(entry / "state.json")
    observation_path = _root(storage_root) / "source_observations" / (entry.name + ".json")
    before = _read(observation_path) if observation_path.exists() else {}
    if (before.get("source_subscription") == source_subscription
            and before.get("source_payload_sha256") == entry.name
            and before.get("effect_state") in {"waiting_for_authorized_action", "permanent_invalid"}
            and before.get("cause") in _HELD_CAUSES.values()
            and all(before.get(key) == value for key, value in _selectors(handoff).items())):
        return {"status": "permanent_invalid" if before["effect_state"] == "permanent_invalid" else "waiting_authorized_action",
                "cause": before["cause"], "next_action": _SUMMARY_ACTIONS[before["cause"]]}
    if state.get("effect_reconciliation_required"):
        return {"status": "reconciliation_required", "cause": "original_effect_state_unproven",
                "next_action": "reconcile_original_operation"}
    if (handoff.source_finalize is not None and handoff.source_membership_selector is not None
            and before.get("source_subscription") == source_subscription
            and before.get("source_payload_sha256") == entry.name
            and before.get("effect_state") == "pre_lease" and before.get("cause") in _ACTIONS
            and all(before.get(key) == value for key, value in _selectors(handoff).items())):
        return {"status": "safe_pre_effect", "cause": before["cause"], "next_action": _ACTIONS[before["cause"]]}
    return {"status": "reconciliation_required", "cause": "original_effect_state_unproven",
            "next_action": "classify_original_operation_before_replay"}


def _commit_disposition(entry: Path, event: Mapping[str, Any], classification: Mapping[str, Any], now: float) -> dict[str, Any]:
    value = {"schema_version": DISPOSITION_SCHEMA, "source_payload_sha256": entry.name,
        **{key: event.get(key) for key in ("bucket", "scene_id", "capture_id", "source_finalize", "source_membership_selector")},
        **dict(classification), "retry_exhaustion_proven": False, "recorded_at_epoch": now,
        "disposition_digest": ""}
    value["disposition_digest"] = canonical_digest(value, digest_field="disposition_digest")
    _once(entry / "dispositions" / (value["disposition_digest"][7:] + ".json"), value)
    return value


def _backoff(digest: str, attempts: int) -> float:
    return min(3600, 60 * 2 ** min(attempts, 6)) + int(digest[:4], 16) % 30


def _state(entry: Path) -> dict[str, Any]:
    """One actionable/conservative successor schema for ACK and processing."""
    value = _read(entry / "state.json")
    if (value.get("schema_version") != "blueprint.handoff_recovery_state.v1"
            or value.get("source_payload_sha256") != entry.name
            or any(not isinstance(value.get(key), str) for key in ("status", "cause", "next_action"))
            or value.get("status") not in {"retained", "processing", "safe_pre_effect", "completed", "authority_ended",
                "permanent_invalid", "orphaned", "waiting_existing_resume", "reconciliation_required", "waiting_authorized_action"}
            or value.get("cause") not in set(_SUMMARY_ACTIONS) | {
                "canonical_output_committed", "canonical_authority_ended", "handoff_payload_invalid"}
            or value.get("next_action") not in set(_SUMMARY_ACTIONS.values()) | {
                "none", "classify_original_operation_before_replay"}
            or type(value.get("attempts")) is not int or value["attempts"] < 0
            or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0
                   for key in ("next_attempt_at_epoch", "first_retained_at_epoch"))
            or type(value.get("last_observed_at_epoch", 0)) not in (int, float)
            or not math.isfinite(value.get("last_observed_at_epoch", 0))
            or "effect_reconciliation_required" in value and type(value["effect_reconciliation_required"]) is not bool
            or "retry_exhaustion_proven" in value and value["retry_exhaustion_proven"] is not False):
        raise ValueError("handoff_recovery_state_invalid")
    return value


def _retained_states(root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], int]:
    """Partial/corrupt successors stay unacknowledged and cannot poison peers."""
    values, incomplete = [], 0
    for entry in root.iterdir():
        if not re.fullmatch(r"[0-9a-f]{64}", entry.name) or not entry.is_dir() or entry.is_symlink():
            continue
        try:
            value = _state(entry)
            values.append((entry, value))
        except (OSError, ValueError, TypeError):
            incomplete += 1
    return values, incomplete


def _process_entry(listener, *, entry: Path, state: dict[str, Any], storage_root: Path,
                   source_subscription: str, process_args: Mapping[str, Any], enable_safe: bool,
                   moment: float, dependency_generation: str) -> None:
    event, raw, handoff = _event(listener, entry)
    dependency_changed = (dependency_generation and dependency_generation != state.get("dependency_generation"))
    if moment < state["next_attempt_at_epoch"] and not dependency_changed:
        return
    classification = _classify(listener, storage_root=storage_root, entry=entry,
                               source_subscription=source_subscription)
    if (classification["status"] == "safe_pre_effect" and enable_safe
            and (moment >= state["next_attempt_at_epoch"] or dependency_changed)):
        # The last classifier can race source processing. The real
        # claim CAS refuses every existing ledger before execution.
        state.update(status="processing", attempts=state["attempts"] + 1,
            dependency_generation=dependency_generation, last_attempt_at_epoch=moment,
            next_attempt_at_epoch=moment + _backoff(entry.name, state["attempts"]),
            effect_reconciliation_required=True)
        write_json(entry / "state.json", state)
        try:
            result = listener.process_handoff_payload(raw, storage_root=storage_root,
                payload_digest=entry.name, require_unattempted_delivery=True, **dict(process_args))
        except Exception:
            result = None  # Reclassify the actual effect ledger; never blind retry.
        # Preserve a new known pre-lease failure for the same exact
        # bytes. A generic exception can't turn into retry authority.
        if result is not None and result.get("queue_disposition") == "retryable":
            _record_result(listener, storage_root=storage_root, subscription=source_subscription,
                raw=raw, result=result)
            code = result.get("owner_observation_reason") or result.get("staging_reason")
            if code in _HELD_CAUSES:
                classification = _classify(listener, storage_root=storage_root, entry=entry,
                                           source_subscription=source_subscription)
            elif (result.get("status"), code) not in _SAFE_CAUSES:
                state["cause"] = "original_effect_state_unproven"
                state["effect_reconciliation_required"] = True
                write_json(entry / "state.json", state)
                classification = {"status": "reconciliation_required", "cause": state["cause"],
                    "next_action": "reconcile_original_operation"}
            else:
                state["effect_reconciliation_required"] = False
                write_json(entry / "state.json", state)
                classification = _classify(listener, storage_root=storage_root, entry=entry,
                                           source_subscription=source_subscription)
        else:
            classification = _classify(listener, storage_root=storage_root, entry=entry,
                                       source_subscription=source_subscription)
            if classification["status"] == "safe_pre_effect":
                classification = {"status": "reconciliation_required", "cause": "original_effect_state_unproven",
                                  "next_action": "reconcile_original_operation"}
                state["effect_reconciliation_required"] = True
    if state["next_attempt_at_epoch"] <= moment:
        state["next_attempt_at_epoch"] = moment + _backoff(entry.name, state["attempts"])
    disposition = _commit_disposition(entry, event, classification, moment)
    state.update(classification, disposition_digest=disposition["disposition_digest"],
                 retry_exhaustion_proven=False, last_observed_at_epoch=moment,
                 dependency_generation=dependency_generation)
    write_json(entry / "state.json", state)


def process_retained(listener, *, storage_root: Path, source_subscription: str, process_args: Mapping[str, Any],
                     enable_safe: bool = False, limit: int = 1, now: float | None = None,
                     dependency_generation: str = "") -> dict[str, Any]:
    root = _root(storage_root)
    moment = time.time() if now is None else now
    examined = 0
    try:
        with _lock(root):
            states, _incomplete = _retained_states(root)
            entries = sorted(states, key=lambda item: (item[1]["next_attempt_at_epoch"],
                                                       item[1].get("last_observed_at_epoch", 0)))
            isolated = 0
            for entry, state in entries:
                if examined >= max(0, min(limit, 100)):
                    break
                if (moment < state["next_attempt_at_epoch"] and not
                        (dependency_generation and dependency_generation != state.get("dependency_generation"))):
                    continue
                try:
                    _process_entry(listener, entry=entry, state=state, storage_root=storage_root,
                        source_subscription=source_subscription, process_args=process_args,
                        enable_safe=enable_safe, moment=moment, dependency_generation=dependency_generation)
                    examined += 1
                except (OSError, ValueError, TypeError, KeyError):
                    # Preserve every partial byte and unknown effect. A broker
                    # redelivery can complete a pre-state journal; no synthetic
                    # ack or attempt completion is inferred here.
                    isolated += 1
            result = status(storage_root=storage_root, now=moment)
            if isolated:
                result["isolated_entries"] = isolated
                result["alerts"] = sorted(set(result["alerts"]) | {"handoff_recovery_entry_unavailable"})
                write_json(root / "status.json", result)
            return result
    except BlockingIOError:
        return {"schema_version": STATUS_SCHEMA, "status": "worker_busy"}


def status(*, storage_root: Path, now: float | None = None) -> dict[str, Any]:
    root = _root(storage_root)
    moment = time.time() if now is None else now
    states, incomplete = _retained_states(root)
    rows = [value for _entry, value in states]
    pending = [row for row in rows if row["status"] not in {"completed", "authority_ended", "permanent_invalid"}]
    due = [row for row in pending if row["next_attempt_at_epoch"] <= moment]
    pending_ids = {row["source_payload_sha256"] for row in pending}
    provenance, corrupt_provenance = [], 0
    for path in root.glob("*/broker/*.json"):
        if path.parents[1].name not in pending_ids:
            continue
        try:
            retained = _read(path)
            if (retained.get("source_payload_sha256") != path.parents[1].name
                    or any(retained.get(key) is not None and (type(retained[key]) not in (int, float)
                        or not math.isfinite(retained[key])) for key in (
                            "retention_deadline_epoch", "dlq_publish_at_epoch"))):
                raise ValueError("handoff_recovery_provenance_invalid")
            provenance.append(retained)
        except (OSError, ValueError, TypeError):
            corrupt_provenance += 1
    deadlines = [value["retention_deadline_epoch"] for value in provenance if value.get("retention_deadline_epoch") is not None]
    published = {}
    for value in provenance:
        if value.get("dlq_publish_at_epoch") is not None:
            key = value["source_payload_sha256"]
            published[key] = min(value["dlq_publish_at_epoch"], published.get(key, value["dlq_publish_at_epoch"]))
    alerts = ["handoff_recovery_partial_entry"] if incomplete else []
    if corrupt_provenance:
        alerts.append("handoff_recovery_entry_unavailable")
    if any(row["status"] == "orphaned" for row in rows):
        alerts.append("handoff_recovery_orphaned_dlq")
    if any(not value.get("publish_time_proven") for value in provenance):
        alerts.append("handoff_recovery_retention_deadline_unproven")
    if pending and deadlines and min(deadlines) - moment <= 24 * 3600:
        alerts.append("handoff_recovery_retention_deadline_due")
    if any(row["status"] == "reconciliation_required" for row in rows):
        alerts.append("handoff_recovery_reconciliation_required")
    causes = {cause: [row for row in pending if row["cause"] == cause] for cause in _SUMMARY_ACTIONS}
    value = {"schema_version": STATUS_SCHEMA, "status": "observed", "retained": len(rows),
        "pending": len(pending), "due_backlog": len(due), "incomplete_entries": incomplete,
        "oldest_age_seconds": max([moment - min(row["first_retained_at_epoch"],
            published.get(row["source_payload_sha256"], row["first_retained_at_epoch"])) for row in pending] or [0]),
        "oldest_due_age_seconds": max([moment - row["next_attempt_at_epoch"] for row in due] or [0]),
        "earliest_retention_deadline_epoch": min(deadlines) if deadlines else None,
        "cause_counts": {cause: sum(row["cause"] == cause for row in rows) for cause in sorted({row["cause"] for row in rows})},
        "pending_causes": [{"cause": cause, "count": len(group), "next_action": _SUMMARY_ACTIONS[cause],
                            "next_attempt_at_epoch": min(row["next_attempt_at_epoch"] for row in group)}
                           for cause, group in causes.items() if group],
        "alerts": alerts, "observed_at_epoch": moment, "raw_payload_exposed": False}
    write_json(root / "status.json", value)
    return value


def pull_retained(listener, *, subscriber: Any, storage_root: Path, subscription: str, source_subscription: str,
                  process_args: Mapping[str, Any], enable_safe: bool = False, max_messages: int = 1,
                  now: float | None = None, dependency_generation: str = "") -> dict[str, Any]:
    """Use the existing retained subscription; ack only a committed successor."""
    moment = time.time() if now is None else now
    acknowledged = 0
    for _ in range(max(0, min(max_messages, 100))):
        response = subscriber.pull(request={"subscription": subscription, "max_messages": 1}, timeout=5)
        if not response.received_messages:
            break
        for received in response.received_messages:
            entry = retain_dlq_message(listener, storage_root=storage_root, subscription=subscription,
                                      source_subscription=source_subscription, received=received, now=moment)
            if enable_safe and _provenance(entry, source_subscription):
                subscriber.acknowledge(request={"subscription": subscription, "ack_ids": [received.ack_id]})
                acknowledged += 1
                identity = hashlib.sha256((subscription + "\0" + str(received.message.message_id)).encode()).hexdigest()
                _once(entry / "acknowledged" / (identity + ".json"), {
                    "schema_version": "blueprint.handoff_recovery_acknowledgement.v1",
                    "source_payload_sha256": entry.name, "broker_provenance_id": identity,
                    "dlq_subscription": subscription, "accepted": True})
    return {**process_retained(listener, storage_root=storage_root, source_subscription=source_subscription,
        process_args=process_args, enable_safe=enable_safe, limit=max_messages, now=moment,
        dependency_generation=dependency_generation), "acknowledged": acknowledged, "safe_replay_enabled": enable_safe}


def read_disposition(listener, *, storage_root: Path, source_payload_sha256: str,
                     expected_ledger: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate actual durable bytes, never arbitrary basis/digest assertions."""
    if re.fullmatch(r"[0-9a-f]{64}", source_payload_sha256) is None:
        raise ValueError("handoff_recovery_source_digest_invalid")
    entry = _root(storage_root) / source_payload_sha256
    event, _raw, _handoff = _event(listener, entry)
    state = _read(entry / "state.json")
    digest = state.get("disposition_digest")
    if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("handoff_recovery_disposition_unavailable")
    value = _read(entry / "dispositions" / (digest[7:] + ".json"))
    if (value.get("schema_version") != DISPOSITION_SCHEMA or value.get("disposition_digest") != digest
            or canonical_digest(value, digest_field="disposition_digest") != digest
            or value.get("source_payload_sha256") != source_payload_sha256
            or value.get("retry_exhaustion_proven") is not False
            or any(value.get(key) != event.get(key) for key in (
                "bucket", "scene_id", "capture_id", "source_finalize", "source_membership_selector"))
            or any(value.get(key) != state.get(key) for key in ("status", "cause", "next_action"))
            or expected_ledger is not None and any(value.get(key) != expected_ledger.get(key) for key in (
                "attempt_count", "lease_token", "revision", "producer_delivery_key"))):
        raise ValueError("handoff_recovery_disposition_binding_invalid")
    return value


def recover_from_environment(listener, *, subscriber: Any, storage_root: Path, source_subscription: str,
                             process_args: Mapping[str, Any], max_messages: int = 1) -> dict[str, Any] | None:
    """Opt-in retained-subscription observe tick in the existing listener job."""
    configured = os.getenv("BLUEPRINT_PUBSUB_HANDOFF_DLQ_SUBSCRIPTION", "").strip()
    if not configured:
        return None
    subscription = listener._canonical_subscription_resource(configured)
    if subscription == source_subscription:
        raise ValueError("handoff_recovery_source_is_not_dlq")
    from .task_evaluation_release_identity import running_release_commit
    enabled = os.getenv("BLUEPRINT_PUBSUB_HANDOFF_RECOVERY_SAFE_REPLAY", "").strip().lower() == "true"
    generation = running_release_commit() or ""
    try:
        return pull_retained(listener, subscriber=subscriber, storage_root=storage_root,
            subscription=subscription, source_subscription=source_subscription, process_args=process_args,
            enable_safe=enabled, max_messages=max(1, min(max_messages, 5)), dependency_generation=generation)
    except Exception:
        # A DLQ outage cannot suppress an already durable successor's bounded
        # work, or masquerade as source health. No failed ack is claimed.
        value = process_retained(listener, storage_root=storage_root, source_subscription=source_subscription,
            process_args=process_args, enable_safe=enabled, limit=max(1, min(max_messages, 5)),
            dependency_generation=generation)
        value.update(subscription_read_status="unavailable", next_action="restore_retained_subscription_read")
        write_json(_root(storage_root) / "status.json", value)
        return value
