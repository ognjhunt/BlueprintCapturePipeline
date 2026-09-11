"""Durable ADP task and side-effect records shared by reasoning runtimes.

An agent's conversation is not an operation ledger. Intent is committed before
external work; results are committed before replying to the provider. A crash
between the two requires an authoritative reconciliation, not another attempt.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterator, Mapping

from .contracts import AgentExecutionError, AgentTask, AgentTool, canonical_json, digest


TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
TASK_STATES = frozenset({
    "queued", "creating", "creation_unresolved", "running", "reconciling",
    "cancelling", "continuing", "completed", "failed", "cancelled",
})


class AgentJournal:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.locks = self.root / "locks"
        self.locks.mkdir(exist_ok=True, mode=0o700)
        self.path = self.root / "agent_operations.sqlite3"
        if self.path.is_symlink():
            raise AgentExecutionError("agent_journal_symlink")
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_digest TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    session_id TEXT,
                    parent_task_id TEXT UNIQUE REFERENCES tasks(task_id),
                    turn_id TEXT,
                    result_json TEXT,
                    error_code TEXT,
                    usage_json TEXT,
                    cleanup_state TEXT NOT NULL DEFAULT 'not_requested',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    cancel_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    outcome_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calls (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    turn_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    call_digest TEXT NOT NULL,
                    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
                    delivery_state TEXT NOT NULL DEFAULT 'pending',
                    PRIMARY KEY (task_id, turn_id, call_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS continuation_inputs (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
                    payload_json TEXT NOT NULL,
                    previous_turns_json TEXT NOT NULL,
                    delivery_state TEXT NOT NULL DEFAULT 'pending',
                    turn_id TEXT
                );
                CREATE TABLE IF NOT EXISTS wakeups (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(task_id),
                    due_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def own_task(self, task_id: str) -> Iterator[None]:
        """The OS releases ownership on process death; no stale lease takeover."""

        lock_path = self.locks / (digest(task_id).removeprefix("sha256:") + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AgentExecutionError("agent_task_owned_by_another_worker") from exc
            yield
        finally:
            os.close(fd)

    @contextmanager
    def own_operation(self, operation_id: str) -> Iterator[None]:
        # Separate ownership survives a cancelled SDK coroutine while its
        # synchronous tool is still completing in another thread.
        with self.own_task("operation:" + operation_id):
            yield

    def register(self, task: AgentTask) -> dict[str, Any]:
        task = task.snapshot()
        now = time.time()
        document = canonical_json(task.model_dump(mode="json"))
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task.task_id,)
            ).fetchone()
            if row is not None:
                if row["task_digest"] != task.task_digest or row["task_json"] != document:
                    raise AgentExecutionError("agent_task_identity_conflict")
            else:
                connection.execute(
                    "INSERT INTO tasks(task_id,parent_task_id,task_digest,task_json,state,created_at,updated_at) "
                    "VALUES (?,?,?,?,'queued',?,?)",
                    (task.task_id, task.parent_task_id, task.task_digest, document, now, now),
                )
        return self.task(task.task_id)

    def task(self, task_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise AgentExecutionError("agent_task_missing")
        value = dict(row)
        for name in ("task", "result", "usage"):
            raw = value.pop(name + "_json")
            value[name] = json.loads(raw) if raw is not None else None
        if digest(value["task"]) != value["task_digest"]:
            raise AgentExecutionError("agent_task_journal_integrity_failure")
        return value

    def tasks(self, *, active_only: bool = True, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("agent_task_list_limit_invalid")
        where = "WHERE state NOT IN ('completed','failed','cancelled')" if active_only else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT task_id FROM tasks {where} ORDER BY updated_at,task_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.task(row["task_id"]) for row in rows]

    def set_state(
        self,
        task_id: str,
        state: str,
        *,
        error_code: str | None = None,
        result: Mapping[str, Any] | None = None,
        turn_id: str | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if state not in TASK_STATES:
            raise ValueError("agent_task_state_invalid")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state,result_json,cancel_requested,cancel_reason,task_json FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise AgentExecutionError("agent_task_missing")
            result_json = canonical_json(result) if result is not None else None
            if row["state"] in TERMINAL_STATES:
                if row["state"] != state or row["result_json"] != result_json:
                    raise AgentExecutionError("agent_terminal_state_immutable")
                return
            if state == "completed":
                deadline = json.loads(row["task_json"])["deadline"]
                completed_at = (result or {}).get("remote_completed_at")
                before_deadline = (type(completed_at) in (int, float) and math.isfinite(completed_at)
                                   and 0 < completed_at <= deadline)
                if row["cancel_requested"] and not (
                    row["cancel_reason"] == "agent_task_deadline" and before_deadline
                ):
                    raise AgentExecutionError("agent_completion_after_cancellation")
                if clock is not None and clock() >= deadline and not before_deadline:
                    raise AgentExecutionError("agent_completion_after_deadline")
            connection.execute(
                "UPDATE tasks SET state=?,error_code=?,result_json=?,"
                "turn_id=COALESCE(?,turn_id),updated_at=? WHERE task_id=?",
                (state, error_code, result_json, turn_id, time.time(), task_id),
            )

    def bind_session(self, task_id: str, session_id: str) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT session_id,parent_task_id,state,cancel_requested FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise AgentExecutionError("agent_task_missing")
            if row["session_id"] is not None and row["session_id"] != session_id:
                raise AgentExecutionError("agent_session_identity_conflict")
            if row["state"] in TERMINAL_STATES:
                raise AgentExecutionError("agent_terminal_state_immutable")
            owners = connection.execute(
                "SELECT task_id,state FROM tasks WHERE session_id=? AND task_id<>?",
                (session_id, task_id),
            ).fetchall()
            if owners and (
                row["parent_task_id"] not in {owner["task_id"] for owner in owners}
                or any(owner["state"] not in TERMINAL_STATES for owner in owners)
            ):
                raise AgentExecutionError("agent_session_already_owned")
            try:
                connection.execute(
                    "UPDATE tasks SET session_id=?,state=?,updated_at=? WHERE task_id=?",
                    (session_id, "cancelling" if row["cancel_requested"] else "running",
                     time.time(), task_id),
                )
            except sqlite3.IntegrityError as exc:
                raise AgentExecutionError("agent_session_already_owned") from exc

    def session_owner(self, task_id: str) -> dict[str, Any]:
        seen = set()
        state = self.task(task_id)
        while state["parent_task_id"] is not None:
            if state["task_id"] in seen or len(seen) >= 1000:
                raise AgentExecutionError("agent_session_lineage_invalid")
            seen.add(state["task_id"])
            state = self.task(state["parent_task_id"])
        return state

    def session_tasks(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id FROM tasks WHERE session_id=? ORDER BY created_at", (session_id,),
            ).fetchall()
        return [self.task(row["task_id"]) for row in rows]

    def lineage_tasks(self, task_id: str) -> list[dict[str, Any]]:
        owner_id = self.session_owner(task_id)["task_id"]
        with self._connect() as connection:
            rows = connection.execute(
                "WITH RECURSIVE lineage(task_id) AS (SELECT ? UNION ALL "
                "SELECT tasks.task_id FROM tasks JOIN lineage ON tasks.parent_task_id=lineage.task_id) "
                "SELECT task_id FROM lineage", (owner_id,),
            ).fetchall()
        return [self.task(row["task_id"]) for row in rows]

    def successor(self, task_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT task_id FROM tasks WHERE parent_task_id=?", (task_id,),
            ).fetchone()
        return row["task_id"] if row is not None else None

    def prepare_continuation(self, task_id: str, payload: Mapping[str, Any], turns: list[str]) -> None:
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT payload_json,previous_turns_json FROM continuation_inputs WHERE task_id=?", (task_id,),
            ).fetchone()
            values = (canonical_json(payload), canonical_json(turns))
            if existing is not None:
                if tuple(existing) != values:
                    raise AgentExecutionError("agent_continuation_intent_conflict")
                return
            connection.execute(
                "INSERT INTO continuation_inputs(task_id,payload_json,previous_turns_json) VALUES (?,?,?)",
                (task_id, *values),
            )

    def continuation(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM continuation_inputs WHERE task_id=?", (task_id,),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["payload"] = json.loads(value.pop("payload_json"))
        value["previous_turns"] = json.loads(value.pop("previous_turns_json"))
        return value

    def continuation_delivery(
        self, task_id: str, state: str, *, turn_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if state not in {"sent_unknown", "acknowledged", "bound", "rejected"}:
            raise ValueError("agent_continuation_delivery_state_invalid")
        with self._transaction() as connection:
            if state == "sent_unknown":
                task_row = connection.execute(
                    "SELECT task_json,cancel_requested,state FROM tasks WHERE task_id=?", (task_id,),
                ).fetchone()
                if task_row is None or task_row["cancel_requested"] or task_row["state"] in TERMINAL_STATES:
                    raise AgentExecutionError("agent_task_cancel_requested")
                if clock() >= json.loads(task_row["task_json"])["deadline"]:
                    raise AgentExecutionError("agent_task_authority_expired")
            connection.execute(
                "UPDATE continuation_inputs SET delivery_state=?,turn_id=COALESCE(?,turn_id) WHERE task_id=?",
                (state, turn_id, task_id),
            )

    def request_cancel(self, task_id: str, reason: str) -> None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state,session_id,cancel_reason FROM tasks WHERE task_id=?", (task_id,),
            ).fetchone()
            if row is None:
                raise AgentExecutionError("agent_task_missing")
            if row["state"] in TERMINAL_STATES:
                return
            # Revocation or an explicit cancellation supersedes a deadline-only
            # request; those must not accept a remotely completed result later.
            selected_reason = row["cancel_reason"]
            if selected_reason is None or selected_reason == "agent_task_deadline":
                selected_reason = reason
            state = "cancelling" if row["session_id"] is not None else row["state"]
            connection.execute(
                "UPDATE tasks SET cancel_requested=1,cancel_reason=?,state=?,updated_at=? "
                "WHERE task_id=?", (selected_reason, state, time.time(), task_id),
            )

    def record_usage(self, task_id: str, usage: Mapping[str, Any] | None) -> None:
        # Null stays unknown. This is observational usage, never a settled bill.
        with self._transaction() as connection:
            connection.execute(
                "UPDATE tasks SET usage_json=?,updated_at=? WHERE task_id=?",
                (canonical_json(usage) if usage is not None else None, time.time(), task_id),
            )

    def record_event(self, event_id: str, payload: Mapping[str, Any]) -> bool:
        text = canonical_json(payload)
        if not event_id or len(event_id) > 256 or len(text.encode("utf-8")) > 2_000_000:
            raise AgentExecutionError("agent_event_invalid")
        with self._transaction() as connection:
            old = connection.execute(
                "SELECT event_digest FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if old is not None:
                if old["event_digest"] != digest(payload):
                    raise AgentExecutionError("agent_event_identity_conflict")
                return False
            connection.execute(
                "INSERT INTO events VALUES (?,?,?,?)",
                (event_id, digest(payload), text, time.time()),
            )
            return True

    def event(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        if digest(payload) != row["event_digest"]:
            raise AgentExecutionError("agent_event_journal_integrity_failure")
        return payload

    def wake(self, task_id: str, *, due_at: float) -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO wakeups(task_id,due_at) VALUES (?,?) ON CONFLICT(task_id) "
                "DO UPDATE SET due_at=MIN(wakeups.due_at,excluded.due_at)", (task_id, due_at),
            )

    def recover_wakeups(self, *, now: float) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO wakeups(task_id,due_at) SELECT task_id,? FROM tasks "
                "WHERE state NOT IN ('completed','failed','cancelled') OR cleanup_state='pending' "
                "ON CONFLICT(task_id) DO UPDATE SET due_at=MIN(wakeups.due_at,excluded.due_at)", (now,),
            )
            return cursor.rowcount

    def claim_wakeup(self, *, now: float, visibility_timeout: float) -> dict[str, Any] | None:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT wakeups.task_id,wakeups.attempts FROM wakeups JOIN tasks USING(task_id) "
                "WHERE due_at<=? AND (tasks.state NOT IN ('completed','failed','cancelled') "
                "OR tasks.cleanup_state='pending') ORDER BY due_at,wakeups.task_id LIMIT 1", (now,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE wakeups SET due_at=?,attempts=attempts+1 WHERE task_id=?",
                (now + visibility_timeout, row["task_id"]),
            )
            return {"task_id": row["task_id"], "attempt": row["attempts"] + 1}

    def schedule_next(
        self, task_id: str, *, due_at: float, attempt: int, error_code: str | None = None,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE wakeups SET due_at=MIN(due_at,?),error_code=? WHERE task_id=? AND attempts=?",
                (due_at, error_code, task_id, attempt),
            )

    def wakeup(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM wakeups WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row is not None else None

    def prepare_call(
        self,
        task: AgentTask,
        tool: AgentTool,
        arguments: Mapping[str, Any],
        *,
        turn_id: str,
        call_id: str,
    ) -> dict[str, Any]:
        if not turn_id or not call_id or max(len(turn_id), len(call_id)) > 256:
            raise AgentExecutionError("agent_call_identity_invalid")
        request = {
            "task_id": task.task_id,
            "run_id": task.run_id,
            "authority_digest": task.admission.authority_digest,
            "context_revision": task.context_revision,
            "tool_id": tool.tool_id,
            "tool_digest": tool.tool_digest,
            "arguments": dict(arguments),
        }
        if tool.effect == "read_only":
            # A new read call may intentionally request fresh state.
            request.update(turn_id=turn_id, call_id=call_id)
        operation_id = digest(request)
        call_digest = digest({"request": request, "turn_id": turn_id, "call_id": call_id})
        with self._transaction() as connection:
            known = connection.execute(
                "SELECT call_digest FROM calls WHERE task_id=? AND turn_id=? AND call_id=?",
                (task.task_id, turn_id, call_id),
            ).fetchone()
            if known is not None:
                if known["call_digest"] != call_digest:
                    raise AgentExecutionError("agent_call_identity_conflict")
            else:
                count = connection.execute(
                    "SELECT COUNT(*) FROM calls WHERE task_id=?", (task.task_id,)
                ).fetchone()[0]
                if count >= task.max_tool_calls:
                    raise AgentExecutionError("agent_tool_call_budget_exhausted")
                old = connection.execute(
                    "SELECT request_digest FROM operations WHERE operation_id=?", (operation_id,)
                ).fetchone()
                if old is not None and old["request_digest"] != digest(request):
                    raise AgentExecutionError("agent_operation_identity_conflict")
                if old is None:
                    now = time.time()
                    connection.execute(
                        "INSERT INTO operations VALUES (?,?,?,'prepared',NULL,?,?)",
                        (operation_id, digest(request), canonical_json(request), now, now),
                    )
                connection.execute(
                    "INSERT INTO calls(task_id,turn_id,call_id,call_digest,operation_id) "
                    "VALUES (?,?,?,?,?)",
                    (task.task_id, turn_id, call_id, call_digest, operation_id),
                )
        return self.operation(operation_id)

    def operation(self, operation_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
        if row is None:
            raise AgentExecutionError("agent_operation_missing")
        value = dict(row)
        value["request"] = json.loads(value.pop("request_json"))
        raw = value.pop("outcome_json")
        value["outcome"] = json.loads(raw) if raw is not None else None
        if digest(value["request"]) != value["request_digest"]:
            raise AgentExecutionError("agent_operation_journal_integrity_failure")
        return value

    def mark_executing(
        self, operation_id: str, *, task: AgentTask, clock: Callable[[], float] = time.time,
        reconciled_not_started: bool = False,
    ) -> None:
        with self._transaction() as connection:
            owner = connection.execute(
                "SELECT task_digest,cancel_requested,state FROM tasks WHERE task_id=?", (task.task_id,),
            ).fetchone()
            if owner is None or owner["task_digest"] != task.task_digest:
                raise AgentExecutionError("agent_operation_task_binding_invalid")
            if owner["cancel_requested"] or owner["state"] in TERMINAL_STATES:
                raise AgentExecutionError("agent_task_cancel_requested")
            if clock() >= min(task.deadline, task.admission.expires_at):
                raise AgentExecutionError("agent_task_authority_expired")
            row = connection.execute(
                "SELECT state FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise AgentExecutionError("agent_operation_missing")
            if row["state"] != "prepared" and not (
                reconciled_not_started and row["state"] == "executing"
            ):
                raise AgentExecutionError("agent_operation_requires_reconciliation")
            connection.execute(
                "UPDATE operations SET state='executing',updated_at=? WHERE operation_id=?",
                (time.time(), operation_id),
            )

    def complete_operation(self, operation_id: str, outcome: Mapping[str, Any]) -> None:
        text = canonical_json(outcome)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state,outcome_json FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if row is None:
                raise AgentExecutionError("agent_operation_missing")
            if row["state"] == "completed":
                if row["outcome_json"] != text:
                    raise AgentExecutionError("agent_operation_result_immutable")
                return
            connection.execute(
                "UPDATE operations SET state='completed',outcome_json=?,updated_at=? "
                "WHERE operation_id=?",
                (text, time.time(), operation_id),
            )

    def record_delivery(self, task_id: str, turn_id: str, call_id: str, *, acknowledged: bool) -> None:
        with self._transaction() as connection:
            connection.execute(
                "UPDATE calls SET delivery_state=? WHERE task_id=? AND turn_id=? AND call_id=?",
                ("acknowledged" if acknowledged else "submitted", task_id, turn_id, call_id),
            )

    def unsettled_operations(self, task_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT o.operation_id FROM operations o JOIN calls c "
                "ON c.operation_id=o.operation_id WHERE c.task_id=? AND o.state='executing'",
                (task_id,),
            ).fetchall()
        return [row["operation_id"] for row in rows]

    def cleanup_state(self, task_id: str, state: str) -> None:
        if state not in {"not_requested", "pending", "deleted", "failed"}:
            raise ValueError("agent_cleanup_state_invalid")
        with self._transaction() as connection:
            connection.execute(
                "UPDATE tasks SET cleanup_state=?,updated_at=? WHERE task_id=?",
                (state, time.time(), task_id),
            )
