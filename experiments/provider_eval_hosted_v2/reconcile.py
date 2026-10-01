"""GET-only observation of an existing stopped pilot; never inference or cleanup."""

from pathlib import Path
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
import re
import time

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.journal import AgentJournal, TERMINAL_STATES
from experiments.provider_eval_recovery.harness import Ledger, digest, exclusive, read_json, write_once
from experiments.provider_eval_recovery.live_http import PROJECT
from .soft_pilot import MODES, SoftMonitor, make_runtime

CLEANUP_APPROVAL = ("Sentinel_d6eb85df8ad08191a630d071b244c607", "Sentinel_61c9e31a09708191b1bc4cbbd41d1ac5")
TRACE_SHA = "78c382fa8c1e3c508336bff6c8ca29a7711e63e3aabc57d9ebe60d6fb4feb637"


def retained_cleanup(journal, path, task_id, soft_sha, task_sha, started):
    proof = journal.event("hosted_cleanup_" + task_id)
    if proof is None:
        return None
    saved = path / task_id / "approved_cleanup.json"
    if (not saved.is_file() or digest(read_json(saved)) != proof.get("semantic_sha256")
            or proof.get("task_digest") != task_sha or proof.get("soft_receipt") != soft_sha
            or proof.get("task_id") != task_id or type(proof.get("deleted_at")) not in {int, float}
            or not math.isfinite(proof["deleted_at"]) or proof["deleted_at"] < started):
        raise AgentExecutionError("approved_cleanup_receipt_integrity_failure")
    return proof


class RetainedSession:
    """Offline receipt observations only. Every unknown route fails closed."""
    project_id = PROJECT
    def __init__(self, receipts):
        self.receipts = receipts
    def request(self, method, path, *, body=None, query=None):
        rows = [r for r in self.receipts if r.get("method") == "GET" and r.get("path") == path
                and r.get("phase") == "before_approved_deletion" and r.get("http_status") == 200]
        if method != "GET" or body is not None or len(rows) != 1:
            raise AgentExecutionError("offline_cleanup_retained_endpoint_missing_or_ambiguous")
        return rows[0]["response"]


def reconcile_cleanup(root, receipt, cleanup_path, expected_raw_sha, *, clock=time.time):
    """Adopt only the exact approved cleanup, without rebinding the old task."""
    blob = Path(cleanup_path).read_bytes()
    raw_sha = hashlib.sha256(blob).hexdigest()
    if raw_sha != expected_raw_sha:
        raise AgentExecutionError("exact_cleanup_receipt_hash_required")
    cleanup = json.loads(blob)
    required = {"schema": "approved_hosted_session_cleanup.v1", "delete_http_status": 200,
        "session_get_after_delete_http_status": 404, "environment_get_after_delete_http_status": 404,
        "deletion_approved_request": CLEANUP_APPROVAL[0], "deletion_approved_reply": CLEANUP_APPROVAL[1],
        "research_resume_authorized": False, "exactly_one_delete_dispatch": True,
        "no_other_resource_deleted": True, "new_inference_or_search_or_session_create_dispatches": 0,
        "trace_gzip_sha256": TRACE_SHA, "trace_library_file_id": "libfile_dfb2926095f88191b8c4108e84900211",
        "trace_unchanged": True}
    if not isinstance(cleanup, dict) or any(cleanup.get(k) != v for k, v in required.items()):
        raise AgentExecutionError("exact_approved_cleanup_evidence_required")
    session_id, env_id = cleanup.get("session_id"), cleanup.get("environment_id")
    if any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", value)
           for value in (session_id, env_id)):
        raise AgentExecutionError("agents_api_resource_id_invalid")
    rows = cleanup.get("owned_endpoint_receipts")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise AgentExecutionError("cleanup_endpoint_receipts_required")
    intents = [r for r in rows if r.get("method") == "DELETE" and r.get("approval_answer") == "yes"]
    acknowledgements = [r for r in rows if r.get("method") == "DELETE" and r.get("http_status") == 200]
    if (len(intents) != 1 or len(acknowledgements) != 1 or intents[0].get("session_id") != session_id
            or intents[0].get("environment_id") != env_id or intents[0].get("only_this_session") is not True
            or (intents[0].get("approval_request"), intents[0].get("approval_reply")) != CLEANUP_APPROVAL
            or intents[0].get("endpoint") != "https://api.openai.com/v1/agents/sessions/" + session_id
            or acknowledgements[0].get("session_id") != session_id
            or acknowledgements[0].get("response") != cleanup.get("delete_response")
            or acknowledgements[0].get("response", {}).get("id") != session_id
            or acknowledgements[0].get("response", {}).get("deleted") is not True
            or acknowledgements[0].get("uncertain") is not False):
        raise AgentExecutionError("cleanup_exact_delete_intent_acknowledgement_required")
    deleted_at = acknowledgements[0].get("at")
    absent_session = [r for r in rows if r.get("method") == "GET" and r.get("http_status") == 404
                      and r.get("path") == "/agents/sessions/" + session_id]
    absent_env = [r for r in rows if r.get("method") == "GET" and r.get("http_status") == 404
                  and r.get("path") == "/agents/environments/" + env_id]
    if (type(deleted_at) not in {int, float} or not math.isfinite(deleted_at)
            or not absent_session or len(absent_env) < 2
            or any(type(r.get("at")) not in {int, float} or not math.isfinite(r["at"]) or r["at"] < deleted_at
                   for r in absent_session + absent_env)
            or datetime.fromisoformat(cleanup["deleted_at_utc"]).timestamp() != deleted_at):
        raise AgentExecutionError("cleanup_post_delete_absence_evidence_required")
    root = Path(root).resolve()
    monitor = SoftMonitor(root, receipt, clock=clock, notify=lambda _: None)
    if not (monitor.path / "stop.json").is_file():
        raise AgentExecutionError("cleanup_requires_existing_stopped_pilot")
    with exclusive(monitor.path):
        accounting = cleanup.get("accounting", {})
        ledger = Ledger(root / "live_journal.jsonl", "10.00")
        n = accounting.get("ledger_events")
        if (type(n) is not int or not 1 <= n <= len(ledger.events)
                or ledger.events[n - 1]["sha256"] != accounting.get("ledger_head")
                or accounting.get("reservations_released_usd") != "0"):
            raise AgentExecutionError("cleanup_original_ledger_checkpoint_required")
        states = {e["attempt_id"]: e["kind"] for e in ledger.events[:n]}
        held = sum((Decimal(e["amount_usd"]) for e in ledger.events[:n] if e["kind"] == "reserved"
                    and states[e["attempt_id"]] != "not_accepted"), Decimal(0))
        if held != Decimal(accounting.get("aggregate_reserved_usd", "-1")):
            raise AgentExecutionError("cleanup_original_reserves_not_preserved")
        retained = RetainedSession(rows)
        session = retained.request("GET", "/agents/sessions/" + session_id)
        task_id = (session.get("metadata") or {}).get("blueprint_task_id")
        frozen = {monitor.task_id(mode): mode for mode in MODES}
        if task_id not in frozen:
            raise AgentExecutionError("cleanup_owned_frozen_task_required")
        runtime, task = make_runtime(root, receipt, monitor, frozen[task_id], transport=retained)
        if runtime._validate_session(task, session) != session_id or session["environment"].get("id") != env_id:
            raise AgentExecutionError("cleanup_owned_session_environment_mismatch")
        roots = runtime._root_turns(session_id)
        if (not roots or any(t.get("status") != "cancelled" for t in roots)
                or session.get("required_actions") not in (None, [])):
            raise AgentExecutionError("cleanup_retained_cancelled_turn_required")
        environment = retained.request("GET", "/agents/environments/" + env_id)
        if environment.get("id") != env_id or environment.get("type") != "openai_hosted":
            raise AgentExecutionError("cleanup_retained_environment_identity_required")
        journal = runtime.journal
        before = journal.task(task_id)
        anchor = journal.event("hosted_creation_" + task_id)
        if (anchor is None or anchor.get("task_digest") != task.task_digest
                or anchor.get("soft_receipt") != monitor.sha or deleted_at < anchor["at"]
                or read_json(monitor.path / task_id / "started.json") != anchor
                or not any(o["task_id"] == task_id and o.get("session_id") == session_id for o in monitor.observations())):
            raise AgentExecutionError("cleanup_retained_creation_observation_required")
        write_once(monitor.path / task_id / "approved_cleanup.json", cleanup)
        proof = {"task_id": task_id, "task_digest": task.task_digest, "soft_receipt": monitor.sha,
                 "session_id": session_id, "environment_id": env_id, "deleted_at": deleted_at,
                 "raw_sha256": raw_sha, "semantic_sha256": digest(cleanup)}
        journal.record_event("hosted_cleanup_" + task_id, proof)
        journal.cleanup_state(task_id, "deleted")
        report = {"session_id": session_id, "environment_id": env_id, "cleanup": "approved_deleted_api_absence_observed",
            "cleanup_receipt_sha256": raw_sha, "historical_task_state": before["state"],
            "historical_task_state_not_rebound_or_reset": True, "cancellation_settled_from_retained_turn": True,
            "new_paid_calls": 0, "http_calls": 0, "no_recreation": True, "no_reservations_released": True,
            "provider_async_teardown_and_final_billing_uncertain": True, "budget": monitor.report()}
        write_once(monitor.path / "readonly_reconciliation" / raw_sha / "cleanup_report.json", report)
        return report


class ReadOnlySession:
    def __init__(self, transport, session_id):
        self.transport, self.session_id = transport, session_id
        self.project_id = transport.project_id

    def request(self, method, path, *, body=None, query=None):
        base = "/agents/sessions/" + self.session_id
        if method != "GET" or body is not None or path not in {base, base + "/turns", base + "/items"}:
            raise AgentExecutionError("reconciliation_all_mutations_and_other_sessions_refused")
        return self.transport.request(method, path, query=query)


def reconcile_session(root, receipt, session_id, *, transport, clock=time.time):
    """Bind an authenticated GET to the old task and preserve raw/failure receipts.

    Remote idle is not terminal proof. Only validated root turn status can settle
    cancellation; unsettled operations remain unresolved. All paid gates stay
    stopped and no old receipt is adopted into the patched execution scope.
    """
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", session_id):
        raise AgentExecutionError("agents_api_resource_id_invalid")
    root = Path(root).resolve()
    monitor = SoftMonitor(root, receipt, clock=clock, notify=lambda _: None)
    if not (monitor.path / "stop.json").is_file():
        raise AgentExecutionError("reconciliation_requires_existing_stopped_pilot")
    with exclusive(monitor.path):
        journal = AgentJournal(monitor.path / "agent_journal")
        frozen = {monitor.task_id(mode): mode for mode in MODES}
        candidates = [s for s in journal.tasks(active_only=False, limit=10)
                      if s["task_id"] in frozen and s["state"] != "queued"]
        # Reject unknown IDs before HTTP when a retained usage/creation proof
        # already names the accepted session, including the original failure.
        observations = monitor.observations()
        owned = [s for s in candidates if s.get("session_id") == session_id or any(
            o.get("session_id") == session_id and o["task_id"] == s["task_id"] for o in observations)]
        if len(owned) != 1:
            raise AgentExecutionError("exact_retained_owned_session_required")
        before = owned[0]
        task_id, mode = before["task_id"], frozen[before["task_id"]]
        anchor = journal.event("hosted_creation_" + task_id)
        started = monitor.path / task_id / "started.json"
        if (anchor is None or not started.is_file() or read_json(started) != anchor
                or anchor.get("task_digest") != before["task_digest"]
                or anchor.get("soft_receipt") != monitor.sha):
            raise AgentExecutionError("durable_creation_proof_missing_or_changed")
        readonly = ReadOnlySession(transport, session_id)
        runtime, task = make_runtime(root, receipt, monitor, mode, transport=readonly)
        runtime.max_pages = 3
        raw = readonly.request("GET", "/agents/sessions/" + session_id)
        # Store the unmodified response/failure before semantic normalization.
        snapshot = {"soft_receipt_sha256": monitor.sha, "at": clock(),
                    "original_state": before, "session": raw}
        snapshot_sha = digest(snapshot)
        folder = monitor.path / "readonly_reconciliation" / snapshot_sha
        write_once(folder / "snapshot.json", snapshot)
        journal.record_event("hosted_readonly_" + snapshot_sha,
            {"snapshot_sha256": snapshot_sha, "task_id": task_id, "session_id": session_id,
             "soft_receipt_sha256": monitor.sha})
        if runtime._validate_session(task, raw) != session_id:
            raise AgentExecutionError("agents_api_session_identity_changed")
        roots = runtime._root_turns(session_id)
        items = runtime._list("/agents/sessions/" + session_id + "/items")
        write_once(folder / "turns.json", roots)
        write_once(folder / "items.json", items)
        # Retain complete evidence with journal-bound hashes, not a quality score.
        journal.record_event("hosted_readonly_evidence_" + snapshot_sha,
            {"turns_sha256": digest(roots), "items_sha256": digest(items)})
        statuses = [turn.get("status") for turn in roots]
        if any(status not in {"queued", "in_progress", "waiting", "completed", "failed", "cancelled"}
               for status in statuses):
            raise AgentExecutionError("agents_api_turn_status_invalid")
        actions = raw.get("required_actions")
        if actions is not None and not isinstance(actions, list):
            raise AgentExecutionError("agents_api_required_actions_invalid")
        unsettled_remote = any(status in {"queued", "in_progress", "waiting"} for status in statuses) or bool(actions)
        if before["state"] not in TERMINAL_STATES:
            journal.request_cancel(task_id, "readonly_reconciliation_no_paid_resume")
            journal.bind_session(task_id, session_id)
            if unsettled_remote:
                journal.set_state(task_id, "reconciling", error_code="agents_api_remote_work_not_terminal")
            else:
                runtime._collect_terminal(task, session_id, raw, roots)
        journal.record_usage(task_id, raw.get("usage"))
        monitor.observe(task_id, raw)
        state = journal.task(task_id)
        report = {"session_id": session_id, "task_id": task_id, "local_state": state["state"],
            "root_turn_statuses": [t.get("status") for t in roots],
            "cancellation_settled": state["state"] == "cancelled" and bool(roots) and not unsettled_remote,
            "snapshot_sha256": snapshot_sha, "raw_evidence_directory": str(folder),
            "new_paid_calls": 0, "no_mutating_http": True, "no_session_deleted": True,
            "original_receipt_unchanged": True, "remaining_arms_not_authorized": True,
            "budget": monitor.report()}
        write_once(folder / "report.json", report)
        return report
