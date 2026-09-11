"""Restartable bridge from ADP task contracts to the managed Codex harness.

Each step performs bounded application work and returns. A durable worker can
call it after a webhook, timer tick or process restart. It never interprets an
idle session, a lost stream, or a completed tool as product/scientific success.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Callable, Mapping

from .contracts import AgentExecutionError, AgentTask, RUNTIME_API, canonical_json, digest
from .journal import AgentJournal, TERMINAL_STATES
from .openai_transport import AgentsAPITransport, AgentTransportError
from .operations import AgentOperations, OperationPending


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,256}", value) is None:
        raise AgentExecutionError("agents_api_resource_id_invalid")
    return value


def _input_messages(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for message in value:
        if set(message) - {"role", "content", "type"} or message.get("role") != "user":
            raise AgentExecutionError("agents_api_input_message_invalid")
        if "type" in message and message["type"] != "message":
            raise AgentExecutionError("agents_api_input_message_invalid")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "input_text", "text": content}]
        if not isinstance(content, list) or not content:
            raise AgentExecutionError("agents_api_input_content_invalid")
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"input_text", "input_image"}:
                raise AgentExecutionError("agents_api_input_content_invalid")
            key = "text" if part["type"] == "input_text" else "image_url"
            # In particular do not silently discard an SDK image-detail setting
            # that this endpoint does not support.
            if set(part) != {"type", key} or not isinstance(part[key], str):
                raise AgentExecutionError("agents_api_input_content_unsupported")
        messages.append({"role": "user", "content": content})
    return messages


class OpenAIAgentsRuntime:
    runtime_id = RUNTIME_API
    runtime_version = "agents=v1"

    def __init__(
        self,
        *,
        transport: AgentsAPITransport,
        project_id: str,
        journal: AgentJournal,
        operations: AgentOperations,
        validate_admission: Callable[[AgentTask], None],
        clock: Callable[[], float] = time.time,
        max_pages: int = 100,
    ) -> None:
        if not project_id or not 1 <= max_pages <= 1000:
            raise ValueError("agents_api_runtime_configuration_invalid")
        if transport.project_id != project_id:
            raise AgentExecutionError("agents_api_transport_project_mismatch")
        self.transport = transport
        self.project_id = project_id
        self.journal = journal
        self.operations = operations
        self.validate_admission = validate_admission
        self.clock = clock
        self.max_pages = max_pages

    def _request(self, method: str, path: str, **kwargs) -> Mapping[str, Any]:
        return self.transport.request(method, path, **kwargs)

    def _list(self, path: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        after: str | None = None
        seen: set[str] = set()
        for _ in range(self.max_pages):
            query: dict[str, str | int] = {"order": "asc", "limit": 100}
            if after is not None:
                query["after"] = after
            page = self._request("GET", path, query=query)
            rows = page.get("data")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise AgentExecutionError("agents_api_list_response_invalid")
            if not isinstance(page.get("has_more"), bool):
                raise AgentExecutionError("agents_api_pagination_missing")
            result.extend(rows)
            if page["has_more"] is False:
                return result
            after = _identifier(page.get("last_id"))
            if not rows or after in seen:
                raise AgentExecutionError("agents_api_pagination_did_not_advance")
            seen.add(after)
        raise AgentExecutionError("agents_api_pagination_limit_exceeded")

    def _validate_task(self, task: AgentTask) -> None:
        task.snapshot()
        if (task.admission.runtime != RUNTIME_API
                or task.admission.project_id != self.project_id
                or task.admission.project_id != self.transport.project_id):
            raise AgentExecutionError("agents_api_runtime_authority_mismatch")
        self.operations.validate_tools(task)

    def _validate_session(self, task: AgentTask, value: Mapping[str, Any]) -> str:
        if task.parent_task_id is not None:
            task = AgentTask.model_validate(self.journal.session_owner(task.parent_task_id)["task"])
        session_id = _identifier(value.get("id"))
        metadata = value.get("metadata")
        if not isinstance(metadata, dict) or (
            metadata.get("blueprint_task_digest") != task.task_digest
            or metadata.get("blueprint_task_id") != task.task_id
            or metadata.get("blueprint_run_id") != task.run_id
        ):
            raise AgentExecutionError("agents_api_session_binding_mismatch")
        agent = value.get("agent")
        if not isinstance(agent, dict) or agent.get("model") != task.model:
            raise AgentExecutionError("agents_api_model_identity_mismatch")
        multi = agent.get("multi_agent")
        if isinstance(multi, dict) and multi.get("enabled") is True:
            raise AgentExecutionError("agents_api_delegation_not_admitted")
        environment = value.get("environment")
        if environment is not None and (
            not isinstance(environment, dict) or environment.get("type") != "none"
        ):
            raise AgentExecutionError("agents_api_environment_not_admitted")
        return session_id

    def _create_payload(self, task: AgentTask) -> dict[str, Any]:
        tools = []
        for name in task.tool_ids:
            tool = self.operations.tools[name]
            tools.append({
                "type": "function", "name": tool.tool_id,
                "description": tool.description, "parameters": dict(tool.input_schema),
            })
        return {
            "agent": {
                "model": task.model,
                "instructions": task.instructions,
                "reasoning": {"effort": task.reasoning_effort},
                "service_tier": "default",
                "text": {
                    "format": {"type": "json_schema", "schema": task.output_schema},
                    "verbosity": "low",
                },
                "multi_agent": {"enabled": False},
                "tools": tools,
            },
            "environment": {"type": "none"},
            "input": _input_messages(task.input),
            "metadata": {
                "blueprint_task_id": task.task_id,
                "blueprint_task_digest": task.task_digest,
                "blueprint_run_id": task.run_id,
                "blueprint_source_commit": task.source_commit,
            },
            "stream": False,
        }

    def start(self, task: AgentTask) -> dict[str, Any]:
        task = task.snapshot()
        if task.parent_task_id is not None:
            return self.continue_task(task)
        self._validate_task(task)
        with self.journal.own_task(task.task_id):
            state = self.journal.register(task)
            if state["state"] != "queued":
                return state
            if state["cancel_requested"]:
                self.journal.set_state(task.task_id, "cancelled", error_code=state["cancel_reason"])
                return self.journal.task(task.task_id)
            if self.clock() >= task.deadline:
                self.journal.set_state(task.task_id, "failed", error_code="agent_task_expired")
                return self.journal.task(task.task_id)
            self.validate_admission(task)
            task = task.snapshot()
            # Validate the whole wire payload before committing creation intent.
            payload = self._create_payload(task)
            canonical_json(payload)
            self.journal.set_state(task.task_id, "creating")
            try:
                session = self._request("POST", "/agents/sessions", body=payload)
                session_id = self._validate_session(task, session)
                self.journal.bind_session(task.task_id, session_id)
            except AgentTransportError as exc:
                rejected = exc.definitively_rejected
                if exc.diagnostics:
                    self.journal.record_event("api_creation_rejection_" + task.task_id, {
                        "task_id": task.task_id, "task_digest": task.task_digest, **exc.diagnostics})
                self.journal.set_state(
                    task.task_id, "failed" if rejected else "creation_unresolved",
                    error_code=exc.code,
                )
            except AgentExecutionError:
                self.journal.set_state(
                    task.task_id, "creation_unresolved", error_code="agents_api_creation_unresolved",
                )
                raise
            return self.journal.task(task.task_id)

    def admit(self, task: AgentTask) -> dict[str, Any]:
        """Provider-free validation for the durable service intake."""
        task = task.snapshot()
        if task.parent_task_id is not None:
            from .continuation import admit_continuation

            return admit_continuation(self, task)
        self._validate_task(task)
        self.validate_admission(task)
        with self.journal.own_task(task.task_id):
            return self.journal.register(task)

    def _recover_creation(self, task: AgentTask) -> None:
        matches = []
        for session in self._list("/agents/sessions"):
            metadata = session.get("metadata")
            if isinstance(metadata, dict) and metadata.get("blueprint_task_id") == task.task_id:
                matches.append(self._validate_session(task, session))
        if len(matches) > 1:
            raise AgentExecutionError("agents_api_duplicate_owned_sessions")
        if matches:
            self.journal.bind_session(task.task_id, matches[0])
        else:
            # Absence in a list does not prove an interrupted create did not run.
            self.journal.set_state(
                task.task_id, "creation_unresolved", error_code="agents_api_creation_unresolved",
            )

    def inspect(self, task_id: str) -> dict[str, Any]:
        return self.journal.task(task_id)

    def continue_task(self, task: AgentTask) -> dict[str, Any]:
        from .continuation import continue_task

        return continue_task(self, task)

    def step(self, task_id: str) -> dict[str, Any]:
        queued = self.journal.task(task_id)
        intent = self.journal.continuation(task_id) if queued["parent_task_id"] is not None else None
        if intent is not None and intent["delivery_state"] == "pending" and queued["state"] not in TERMINAL_STATES:
            return self.continue_task(AgentTask.model_validate(queued["task"]))
        if queued["state"] == "queued":
            return self.start(AgentTask.model_validate(queued["task"]))
        with self.journal.own_task(task_id):
            state = self.journal.task(task_id)
            task = AgentTask.model_validate(state["task"])
            self._validate_task(task)
            if state["state"] in TERMINAL_STATES:
                return state
            if self.clock() >= task.deadline:
                self.journal.request_cancel(task_id, "agent_task_deadline")
            try:
                self.validate_admission(task)
                task = task.snapshot()
            except Exception:
                # Current admission may be revoked. Observation, cancellation
                # and cleanup remain available without admitting new work.
                self.journal.request_cancel(task_id, "agent_admission_revoked")
            state = self.journal.task(task_id)
            if state["state"] in {"creating", "creation_unresolved"}:
                self._recover_creation(task)
                state = self.journal.task(task_id)
            session_id = state["session_id"]
            if session_id is None:
                return state
            if state["cancel_requested"]:
                self._send_cancel(session_id)
            session = self._request("GET", f"/agents/sessions/{_identifier(session_id)}")
            if self._validate_session(task, session) != session_id:
                raise AgentExecutionError("agents_api_session_identity_changed")
            usage = session.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise AgentExecutionError("agents_api_usage_invalid")
            self.journal.record_usage(task_id, usage)
            if state["cancel_requested"]:
                try:
                    self.operations.reconcile_cancelled(task)
                except OperationPending as exc:
                    self.journal.set_state(task_id, "reconciling", error_code=str(exc))
            actions = session.get("required_actions") or []
            if not isinstance(actions, list):
                raise AgentExecutionError("agents_api_required_actions_invalid")
            roots = self._root_turns(session_id)
            if task.parent_task_id is not None:
                from .continuation import task_turns

                roots = task_turns(self, task, session_id, roots)
                if not roots:
                    return self.journal.task(task_id)
            if not state["cancel_requested"]:
                self._validate_actions(actions, roots)
                for action in actions:
                    try:
                        self._handle_action(task, session_id, action)
                    except OperationPending as exc:
                        self.journal.set_state(task_id, "reconciling", error_code=str(exc))
                        return self.journal.task(task_id)
                if actions:
                    self.journal.set_state(task_id, "running")
                    return self.journal.task(task_id)
            self._collect_terminal(task, session_id, session, roots)
            return self.journal.task(task_id)

    def _handle_action(self, task: AgentTask, session_id: str, action: Any) -> None:
        if not isinstance(action, dict) or action.get("type") != "function_call":
            raise AgentExecutionError("agents_api_required_action_not_admitted")
        turn_id = _identifier(action.get("turn_id"))
        call_id = _identifier(action.get("call_id"))
        name = action.get("name")
        if not isinstance(name, str):
            raise AgentExecutionError("agents_api_function_name_invalid")
        # The provider's function arguments are an object, not a Responses API
        # JSON string. Do not accept a different wire shape without qualification.
        outcome = self.operations.execute(
            task, turn_id=turn_id, call_id=call_id, name=name, arguments=action.get("arguments"),
        )
        if self.clock() >= task.deadline:
            self.journal.request_cancel(task.task_id, "agent_task_deadline")
        if self.journal.task(task.task_id)["cancel_requested"]:
            self._send_cancel(session_id)
            raise OperationPending("agent_task_cancellation_pending")
        event: dict[str, Any] = {
            "type": "agent.session.input.tool_result", "turn_id": turn_id, "call_id": call_id,
            "success": outcome["success"],
        }
        if outcome["success"]:
            output = outcome["output"]
            event["output"] = output if isinstance(output, list) else canonical_json(output)
        else:
            event["error"] = outcome["error"]
        self.journal.record_delivery(task.task_id, turn_id, call_id, acknowledged=False)
        self._request("POST", f"/agents/sessions/{session_id}/events", body={"events": [event]})
        self.journal.record_delivery(task.task_id, turn_id, call_id, acknowledged=True)

    def _root_turns(self, session_id: str) -> list[dict[str, Any]]:
        turns = self._list(f"/agents/sessions/{session_id}/turns")
        roots = []
        for turn in turns:
            if turn.get("session_id") != session_id or "subagent_id" not in turn:
                raise AgentExecutionError("agents_api_turn_binding_invalid")
            if turn["subagent_id"] is None:
                roots.append(turn)
        return roots

    @staticmethod
    def _validate_actions(actions: list[Any], roots: list[dict[str, Any]]) -> None:
        if not actions:
            return
        active = [turn for turn in roots if turn.get("status") in {"queued", "in_progress", "waiting"}]
        if len(active) != 1:
            raise AgentExecutionError("agents_api_active_turn_missing_or_ambiguous")
        expected_turn_id = _identifier(active[0].get("id"))
        seen: set[str] = set()
        for action in actions:
            if not isinstance(action, dict) or action.get("turn_id") != expected_turn_id:
                raise AgentExecutionError("agents_api_action_turn_mismatch")
            call_id = _identifier(action.get("call_id"))
            if call_id in seen:
                raise AgentExecutionError("agents_api_duplicate_required_action")
            seen.add(call_id)

    def _collect_terminal(
        self, task: AgentTask, session_id: str, session: Mapping[str, Any],
        roots: list[dict[str, Any]],
    ) -> None:
        if not roots:
            if session.get("status") == "failed":
                self.journal.set_state(task.task_id, "failed", error_code="agents_api_session_failed")
            return
        turn = roots[-1]
        turn_id = _identifier(turn.get("id"))
        status = turn.get("status")
        if status in {"queued", "in_progress", "waiting"}:
            return
        if status not in {"completed", "failed", "cancelled"}:
            raise AgentExecutionError("agents_api_turn_status_invalid")
        if self.journal.unsettled_operations(task.task_id):
            self.journal.set_state(
                task.task_id, "reconciling", error_code="agent_tool_outcome_unresolved",
            )
            return
        if status != "completed":
            self.journal.set_state(
                task.task_id, status, error_code="agents_api_turn_" + status, turn_id=turn_id,
            )
            return
        items = self._list(f"/agents/sessions/{session_id}/items")
        finals = [
            item for item in items
            if item.get("turn_id") == turn_id and item.get("type") == "message"
            and item.get("role") == "assistant" and item.get("phase") == "final_answer"
            and item.get("status") == "completed"
        ]
        state = self.journal.task(task.task_id)
        if state["cancel_requested"]:
            completed_at = turn.get("completed_at")
            completed_before_deadline = (
                type(completed_at) in (int, float) and math.isfinite(completed_at)
                and 0 < completed_at <= task.deadline
            )
            if state["cancel_reason"] != "agent_task_deadline" or not completed_before_deadline:
                observation = {
                    "task_id": task.task_id, "session_id": session_id,
                    "turn": turn, "final_messages": finals,
                    "accepted_as_task_completion": False,
                }
                self.journal.record_event("cancelled_completion_" + digest(observation)[7:], observation)
                self.journal.set_state(
                    task.task_id, "cancelled", error_code=state["cancel_reason"], turn_id=turn_id,
                )
                return
        if len(finals) != 1:
            raise AgentExecutionError("agents_api_final_message_missing_or_ambiguous")
        content = finals[0].get("content")
        if not isinstance(content, list) or not content or any(
            not isinstance(part, dict) or part.get("type") != "output_text"
            or not isinstance(part.get("text"), str) for part in content
        ):
            raise AgentExecutionError("agents_api_final_content_invalid")
        text = "".join(part["text"] for part in content)
        try:
            output = task.validate_output(json.loads(text))
        except (ValueError, TypeError) as exc:
            raise AgentExecutionError("agents_api_final_output_invalid") from exc
        result = {
            "schema_version": "blueprint_agent_task_result.v1",
            "task_id": task.task_id, "task_digest": task.task_digest,
            "run_id": task.run_id, "source_commit": task.source_commit,
            "runtime": self.runtime_id, "runtime_version": self.runtime_version,
            "model": task.model, "session_id": session_id, "turn_id": turn_id,
            "remote_completed_at": turn.get("completed_at"),
            "output": output, "output_digest": digest(output),
            "usage": session.get("usage"), "cost_status": "official_reconciliation_required",
            "scope": "agent_execution_only", "scientific_acceptance_granted": False,
        }
        result["result_digest"] = digest(result)
        self.journal.set_state(task.task_id, "completed", result=result, turn_id=turn_id, clock=self.clock)

    def _send_cancel(self, session_id: str) -> None:
        try:
            self._request(
                "POST", f"/agents/sessions/{_identifier(session_id)}/events",
                body={"events": [{"type": "agent.session.input.cancel"}]},
            )
        except AgentTransportError as exc:
            if exc.status != 409:
                raise

    def cancel(self, task_id: str) -> dict[str, Any]:
        self.journal.request_cancel(task_id, "agent_cancel_requested")
        try:
            with self.journal.own_task(task_id):
                state = self.journal.task(task_id)
                if state["state"] in TERMINAL_STATES:
                    return state
                if state["state"] == "queued":
                    self.journal.set_state(task_id, "cancelled", error_code="cancelled_before_creation")
                elif state["session_id"] is not None:
                    self._send_cancel(state["session_id"])
        except AgentExecutionError as exc:
            if str(exc) != "agent_task_owned_by_another_worker":
                raise
        # The active worker observes the durable intent before any next action.
        return self.journal.task(task_id)

    def cleanup(self, task_id: str) -> dict[str, Any]:
        from contextlib import ExitStack

        owner_id = self.journal.session_owner(task_id)["task_id"]
        with ExitStack() as ownership:
            ownership.enter_context(self.journal.own_task(owner_id))
            if owner_id != task_id:
                ownership.enter_context(self.journal.own_task(task_id))
            state = self.journal.task(task_id)
            if state["state"] not in TERMINAL_STATES or self.journal.unsettled_operations(task_id):
                raise AgentExecutionError("agents_api_cleanup_before_reconciliation")
            if state["cleanup_state"] == "deleted":
                return state
            # Only the last task may retire a retained conversation, and every
            # predecessor must be terminal with its effects reconciled.
            if self.journal.successor(task_id) is not None:
                raise AgentExecutionError("agents_api_session_has_successor")
            session_id = state["session_id"] or self.journal.task(owner_id)["session_id"]
            members = self.journal.lineage_tasks(task_id)
            if any(member["state"] not in TERMINAL_STATES
                   or self.journal.unsettled_operations(member["task_id"]) for member in members):
                raise AgentExecutionError("agents_api_cleanup_before_reconciliation")
            if any(member["session_id"] not in {None, session_id} for member in members):
                raise AgentExecutionError("agents_api_cleanup_lineage_session_mismatch")
            if session_id is None:
                for member in members:
                    self.journal.cleanup_state(member["task_id"], "deleted")
                return self.journal.task(task_id)
            self.journal.cleanup_state(task_id, "pending")
            try:
                self._request("DELETE", f"/agents/sessions/{_identifier(session_id)}")
                self._request("GET", f"/agents/sessions/{session_id}")
            except AgentTransportError as exc:
                if exc.status == 404:
                    for member in members:
                        self.journal.cleanup_state(member["task_id"], "deleted")
                elif exc.status != 409:
                    raise
            # A successful DELETE response alone is not an observed absence.
            return self.journal.task(task_id)
