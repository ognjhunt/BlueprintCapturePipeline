"""GET-only observation of an existing stopped pilot; never inference or cleanup."""

from pathlib import Path
import re
import time

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.journal import AgentJournal, TERMINAL_STATES
from experiments.provider_eval_recovery.harness import digest, exclusive, read_json, write_once
from .soft_pilot import MODES, SoftMonitor, make_runtime


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
