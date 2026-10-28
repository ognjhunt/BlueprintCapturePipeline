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
    ("capture_source_membership_unavailable_retryable", "capture_staging_owner_observation_unavailable"):
        "capture_owner_transport_unavailable",
    ("capture_source_membership_unavailable_retryable", "capture_staging_membership_unavailable"):
        "capture_membership_transport_unavailable",
    ("capture_source_membership_unavailable_retryable", "capture_staging_unavailable"):
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
    "capture_owner_access_denied": "restore_authorized_capture_owner_read"}


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


def record_source_result(listener, *, storage_root: Path, subscription: str, message: Any,
                         received: Any, result: Mapping[str, Any] | None) -> None:
    """Private named pre-lease evidence; no exception text or payload in status."""
    raw = bytes(message.data)
    handoff = listener.parse_handoff_payload(raw)
    code = (result or {}).get("owner_observation_reason") or (result or {}).get("staging_reason")
    cause = _SAFE_CAUSES.get(((result or {}).get("status"), code))
    denied = code == "capture_owner_response_denied"
    # A duplicate can fail its read while the original owns paid effects.
    # That new pre-lease failure never erases the original effect ledger.
    if listener._read_job_ledger(listener._handoff_capture_root(handoff, storage_root=storage_root)):
        cause = None
        denied = False
    root = _root(storage_root) / "source_observations"
    root.mkdir(exist_ok=True, mode=0o700)
    digest = hashlib.sha256(raw).hexdigest()
    write_json(root / (digest + ".json"), {
        "schema_version": "blueprint.handoff_source_recovery_observation.v1", "source_payload_sha256": digest,
        "source_subscription": subscription, "source_message_id": str(message.message_id),
        "source_delivery_attempt": getattr(received, "delivery_attempt", None), **_selectors(handoff),
        "effect_state": "pre_lease" if cause else "waiting_for_authorized_action" if denied else "unproven",
        "cause": cause or ("capture_owner_access_denied" if denied else "original_effect_state_unproven"),
        "observed_at_epoch": time.time()})


def _published_at(message: Any, fallback: float) -> float | None:
    value = getattr(message, "publish_time", None)
    try:
        epoch = float(value.timestamp()) if callable(getattr(value, "timestamp", None)) else float(value)
        if not math.isfinite(epoch) or epoch <= 0 or epoch > fallback + 60:
            raise ValueError("invalid publish time")
        return epoch
    except (ValueError, TypeError, OverflowError):
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
    if _read(state_path).get("source_payload_sha256") != digest:
        raise ValueError("handoff_recovery_state_binding_changed")
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
    state = _read(entry / "state.json")
    if state.get("effect_reconciliation_required"):
        return {"status": "reconciliation_required", "cause": "original_effect_state_unproven",
                "next_action": "reconcile_original_operation"}
    observation_path = _root(storage_root) / "source_observations" / (entry.name + ".json")
    before = _read(observation_path) if observation_path.exists() else {}
    if (before.get("source_subscription") == source_subscription
            and before.get("source_payload_sha256") == entry.name
            and before.get("effect_state") == "waiting_for_authorized_action"
            and before.get("cause") == "capture_owner_access_denied"):
        return {"status": "waiting_authorized_action", "cause": "capture_owner_access_denied",
                "next_action": "restore_authorized_capture_owner_read"}
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


def process_retained(listener, *, storage_root: Path, source_subscription: str, process_args: Mapping[str, Any],
                     enable_safe: bool = False, limit: int = 1, now: float | None = None,
                     dependency_generation: str = "") -> dict[str, Any]:
    root = _root(storage_root)
    moment = time.time() if now is None else now
    examined = 0
    try:
        with _lock(root):
            entries = sorted((path for path in root.iterdir() if re.fullmatch(r"[0-9a-f]{64}", path.name)
                              and path.is_dir() and not path.is_symlink()),
                             key=lambda path: (_read(path / "state.json").get("next_attempt_at_epoch", 0),
                                               _read(path / "state.json").get("last_observed_at_epoch", 0)))
            for entry in entries:
                if examined >= max(0, min(limit, 100)):
                    break
                event, raw, handoff = _event(listener, entry)
                state = _read(entry / "state.json")
                dependency_changed = (dependency_generation and dependency_generation != state.get("dependency_generation"))
                if moment < state["next_attempt_at_epoch"] and not dependency_changed:
                    continue
                examined += 1
                classification = _classify(listener, storage_root=storage_root, entry=entry,
                                           source_subscription=source_subscription)
                if (classification["status"] == "safe_pre_effect" and enable_safe
                        and (moment >= state["next_attempt_at_epoch"] or dependency_changed)):
                    # The last classifier can race source processing. The real
                    # claim CAS refuses every existing ledger before execution.
                    state.update(status="processing", attempts=state["attempts"] + 1,
                        dependency_generation=dependency_generation, last_attempt_at_epoch=moment,
                        next_attempt_at_epoch=moment + _backoff(entry.name, state["attempts"]))
                    write_json(entry / "state.json", state)
                    try:
                        result = listener.process_handoff_payload(raw, storage_root=storage_root,
                            payload_digest=entry.name, require_unattempted_delivery=True, **dict(process_args))
                    except Exception:
                        result = None  # Reclassify the actual effect ledger; never blind retry.
                    # Preserve a new known pre-lease failure for the same exact
                    # bytes. A generic exception can't turn into retry authority.
                    if result is not None and result.get("queue_disposition") == "retryable":
                        code = result.get("owner_observation_reason") or result.get("staging_reason")
                        if (result.get("status"), code) not in _SAFE_CAUSES:
                            state["cause"] = "original_effect_state_unproven"
                            state["effect_reconciliation_required"] = True
                            write_json(entry / "state.json", state)
                            classification = {"status": "reconciliation_required", "cause": state["cause"],
                                "next_action": "reconcile_original_operation"}
                        else:
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
            return status(storage_root=storage_root, now=moment)
    except BlockingIOError:
        return {"schema_version": STATUS_SCHEMA, "status": "worker_busy"}


def status(*, storage_root: Path, now: float | None = None) -> dict[str, Any]:
    root = _root(storage_root)
    moment = time.time() if now is None else now
    rows = [_read(path) for path in root.glob("*/state.json") if re.fullmatch(r"[0-9a-f]{64}", path.parent.name)]
    pending = [row for row in rows if row["status"] not in {"completed", "authority_ended", "permanent_invalid"}]
    due = [row for row in pending if row["next_attempt_at_epoch"] <= moment]
    pending_ids = {row["source_payload_sha256"] for row in pending}
    provenance = [_read(path) for path in root.glob("*/broker/*.json") if path.parents[1].name in pending_ids]
    deadlines = [value["retention_deadline_epoch"] for value in provenance if value.get("retention_deadline_epoch") is not None]
    published = {}
    for value in provenance:
        if value.get("dlq_publish_at_epoch") is not None:
            key = value["source_payload_sha256"]
            published[key] = min(value["dlq_publish_at_epoch"], published.get(key, value["dlq_publish_at_epoch"]))
    alerts = []
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
        "pending": len(pending), "due_backlog": len(due),
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
