"""Bounded replay of one HTTP503 QA submission; never a new logical message."""
import copy
import json
from datetime import timedelta

from tools.daily_research.runner import Refusal, canonical, digest, instant

SCOPE = "same-session-same-key-same-message-503-qa-retry-no-new-research"
AUTHORITY = "Sentinel_c2046c5f146c81918921eba1ed7f6caa"
DELEGATION = "01a0ef70-d046-74f6-9434-a19e5456b0ef"


def transient(receipt):
    return (isinstance(receipt, dict) and receipt.get("stage") == "provider_submission"
            and receipt.get("http_status") == 503
            and receipt.get("code") in {"service_unavailable_error", "server_is_overloaded"})


def retry_deadline(row, original_deadline):
    value = row.get("qa_retry_continuation")
    if not value:
        return original_deadline
    request, previous = value.get("request", {}), value.get("previous_qa", {})
    qa = row.get("qa", {})
    if (value.get("schema_version") != "blueprint.qa-submission-retry.v1"
            or value.get("duration_seconds") != 600 or value.get("max_attempts") != 2
            or request.get("scope") != SCOPE or request.get("authority_reference") != AUTHORITY
            or request.get("delegation_reference") != DELEGATION
            or request.get("baseline_id") != "baseline-20261002" or request.get("soft_total_usd") != 25
            or request.get("session_id") != row.get("session_id")
            or request.get("root_turn_id") != row.get("turn_id")
            or request.get("raw_output_sha256") != row.get("raw_output_digest")
            or request.get("packet_digest") != row.get("packet_digest")
            or request.get("idempotency_key") != row.get("run_key", "") + ":qa"
            or request.get("request_digest") != qa.get("request_digest")
            or request.get("previous_qa_digest") != digest(previous)
            or request.get("previous_continuation_digest") != digest(row.get("qa_continuation"))
            or previous.get("request_digest") != qa.get("request_digest")
            or previous.get("input_file") != qa.get("input_file")
            or previous.get("deadline_ms") != qa.get("deadline_ms")
            or previous.get("baseline_turn_ids") != qa.get("baseline_turn_ids")
            or not transient(previous.get("input_error_receipt"))
            or previous.get("state") != "qa_cancel_pending" or previous.get("cancel_attempted") is not True
            or previous.get("cancel_reply_unresolved") is True
            or previous.get("turn_id") or previous.get("artifact_digest")
            or original_deadline > instant(value["started_at"])
            or row.get("canary", {}).get("baseline", {}).get("baseline_id") != request["baseline_id"]
            or row.get("canary", {}).get("baseline", {}).get("soft_total_usd") != 25
            or row.get("canary", {}).get("baseline", {}).get("authority_reference") != AUTHORITY):
        raise Refusal("qa_retry_authority_or_binding_invalid")
    return instant(value["started_at"]) + timedelta(seconds=600)


def original_event(ledger, row):
    qa = row["qa"]
    if not qa.get("input_file") and isinstance(qa.get("event"), dict):
        if digest(qa["event"]) != qa["request_digest"]:
            raise Refusal("qa_retry_immutable_input_changed")
        return copy.deepcopy(qa["event"])
    if qa.get("input_file") != row["date"] + "-qa-input.json":
        raise Refusal("qa_retry_immutable_input_missing")
    raw = ledger.read_bytes(qa["input_file"])
    event = json.loads(raw)
    if digest(event) != qa["request_digest"] or raw != (canonical(event) + "\n").encode():
        raise Refusal("qa_retry_immutable_input_changed")
    return event


def reconcile(api, ledger, row):
    """Full GETs. Return False when a message/turn/effect may already be accepted.

    An idle session alone never admits replay. No outcome of the previous cancel
    is inferred: its receipt remains in previous_qa. Only the same idempotent
    input can be retried, and any accepted/cancelled QA turn is observation-only.
    """
    from tools.daily_research.consumer import QA_PATH, Consumer
    session = api.get("session", row["session_id"])
    Consumer.check_session(row, session)
    turns = api.listing("turns", row["session_id"])
    items = api.listing("items", row["session_id"])
    artifacts = api.listing("artifacts", row["session_id"])
    baseline = set(row["qa"]["baseline_turn_ids"])
    saved = json.loads(ledger.read_bytes(row["date"] + "-evidence.json"))
    if digest(saved) != row.get("evidence_digest"):
        raise Refusal("qa_retry_original_evidence_changed")
    binding = row["qa"].get("submission_binding")
    if not binding and baseline != {row["turn_id"]}:
        raise Refusal("qa_retry_baseline_scope_unsupported")
    idle = (session.get("status") == "idle" and not session.get("required_actions")
            and not session.get("error"))
    unchanged = ({t["id"] for t in turns} == baseline
                 and all(t.get("status") == "completed" and not t.get("subagent_id") for t in turns)
                 and digest(items) == (binding["items_digest"] if binding else digest(saved))
                 and digest([i for i in items if i.get("turn_id") == row["turn_id"]]) == digest(saved)
                 and all(i.get("turn_id") in baseline for i in items)
                 and (not binding or digest(artifacts) == binding["artifacts_digest"])
                 and not any(a.get("path") == QA_PATH or a.get("turn_id") not in baseline for a in artifacts))
    return {"clear": bool(idle and unchanged), "session_status": session.get("status"),
            "required_action_count": len(session.get("required_actions") or []),
            "turn_count": len(turns), "item_count": len(items), "artifact_count": len(artifacts),
            "turns_digest": digest(turns), "items_digest": digest(items), "artifacts_digest": digest(artifacts),
            "previous_cancel_outcome": "not_inferred"}


def check_submission_binding(row, deadline_ms):
    """Ordinary QA replays use the original workflow authority and deadline."""
    qa, binding = row["qa"], row["qa"].get("submission_binding", {})
    expected = {"schema_version": "blueprint.qa-submission.v1", "session_id": row["session_id"],
                "root_turn_id": row["turn_id"], "packet_digest": row["packet_digest"],
                "raw_output_sha256": row["raw_output_digest"], "request_digest": qa["request_digest"],
                "idempotency_key": row["run_key"] + ":qa", "deadline_ms": qa["deadline_ms"],
                "baseline_turn_ids": qa["baseline_turn_ids"]}
    if (any(binding.get(key) != value for key, value in expected.items())
            or binding.get("deadline_ms") != deadline_ms
            or not isinstance(binding.get("authority_reference"), str) or not binding["authority_reference"].strip()
            or binding["authority_reference"].startswith("PENDING")
            or any(not isinstance(binding.get(key), str) or len(binding[key]) != 64
                   for key in ("items_digest", "artifacts_digest"))):
        raise Refusal("qa_retry_authority_or_binding_invalid")
    return binding


def authorize(row, reconciliation, now):
    """Once-only explicit phase. Preserve every byte of the old QA receipt."""
    if row.get("qa_retry_continuation"):
        return row
    previous = copy.deepcopy(row.get("qa", {}))
    if (row.get("state") != "awaiting_review" or previous.get("state") != "qa_cancel_pending"
            or previous.get("cancel_attempted") is not True or previous.get("cancel_reply_unresolved") is True
            or previous.get("turn_id") or previous.get("artifact_digest")
            or not transient(previous.get("input_error_receipt")) or reconciliation.get("clear") is not True):
        raise Refusal("qa_retry_original_submission_not_admitted")
    request = {"scope": SCOPE, "authority_reference": AUTHORITY, "delegation_reference": DELEGATION,
               "baseline_id": "baseline-20261002", "soft_total_usd": 25,
               "session_id": row["session_id"], "root_turn_id": row["turn_id"],
               "raw_output_sha256": row["raw_output_digest"], "packet_digest": row["packet_digest"],
               "source_row_digest": digest(row), "request_digest": previous["request_digest"],
               "idempotency_key": row["run_key"] + ":qa", "previous_qa_digest": digest(previous),
               "previous_continuation_digest": digest(row.get("qa_continuation"))}
    row["qa_retry_continuation"] = {"schema_version": "blueprint.qa-submission-retry.v1", "request": request,
                                    "started_at": now.isoformat(), "duration_seconds": 600, "max_attempts": 2,
                                    "previous_qa": previous, "reconciliation": reconciliation}
    # Current observation belongs to the new explicit phase. Old cancellation
    # fields and deadline remain losslessly preserved above; input/key unchanged.
    row["qa"].update(state="qa_input_unresolved", cancel_attempted=False, input_retries=[])
    row["qa"].pop("cancel_reply_unresolved", None)
    row["qa"].pop("error", None)
    return row


def submit(consumer, row, deadline):
    """Two durable slots, no invisible SDK retries; unknown attempts observe only."""
    qa = row["qa"]
    attempts = qa.get("input_retries", [])
    if (qa.get("state") != "qa_input_unresolved" or qa.get("turn_id") or len(attempts) >= 2
            or consumer.stopped() or consumer.clock() >= deadline):
        return
    initial_error_at = (row.get("qa_retry_continuation", {}).get("started_at") or qa.get("input_error_at"))
    previous = attempts[-1] if attempts else {"error_receipt": qa.get("input_error_receipt"),
                                            "finished_at": initial_error_at}
    if not transient(previous.get("error_receipt")):
        return  # A crash/lost reply/change of error never consumes another slot.
    if not initial_error_at or (not row.get("qa_retry_continuation") and not qa.get("submission_binding")):
        return  # Legacy uncertain intents cannot be upgraded to replay authority.
    if not row.get("qa_retry_continuation"):
        check_submission_binding(row, int(deadline.timestamp() * 1000))
    delay = max(5 if not attempts else 15, previous["error_receipt"].get("retry_after_seconds") or 0)
    not_before = instant(previous["finished_at"]) + timedelta(seconds=delay)
    if not_before >= deadline:
        qa["retry_suppressed_reason"] = "qa_retry_after_exceeds_deadline"
        consumer.ledger.put(row)
        return
    if consumer.clock() < not_before:
        return
    event = original_event(consumer.ledger, row)
    observed = reconcile(consumer.api, consumer.ledger, row)
    if not observed["clear"]:
        return  # An accepted message, active or terminal turn is GET-only.
    attempt = {"number": len(attempts) + 1, "state": "input_unresolved",
               "started_at": consumer.clock().isoformat(), "deadline_ms": int(deadline.timestamp() * 1000),
               "request_digest": qa["request_digest"], "idempotency_key": row["run_key"] + ":qa",
               "not_before": not_before.isoformat(), "reconciliation": observed}
    qa.setdefault("input_retries", []).append(attempt)
    consumer.ledger.put(row)  # Durable intent before the one-use transactional claim and POST.
    consumer.api.qa_input_phase = "preconditions"
    try:
        consumer.api.qa_retry_input(row["session_id"], event, attempt["idempotency_key"], row["date"],
                                    qa["request_digest"], attempt["deadline_ms"], attempt["number"])
        attempt.update(state="reply_received", finished_at=consumer.clock().isoformat())
        qa["state"] = "qa_running"
    except Exception as error:  # noqa: BLE001 - safe typed receipt, unknown means observation-only
        from tools.daily_research.recovery import repair_error_receipt
        attempt.update(error_receipt=repair_error_receipt(error, consumer.api.qa_input_phase),
                       finished_at=consumer.clock().isoformat())
    try:
        consumer.ledger.put(row)
    except Exception as error:  # noqa: BLE001 - accepted reply persistence failure is GET-only recovery
        from tools.daily_research.recovery import repair_error_receipt
        attempt.update(state="input_unresolved", error_receipt=repair_error_receipt(error, "reply_persistence"))
        qa["state"] = "qa_input_unresolved"
        try:
            consumer.ledger.put(row)
        except Exception:  # noqa: BLE001 - never infer that a lost durable receipt permits replay
            qa["input_error_persistence_failed"] = True
        return "reply_persistence_unresolved"
