"""Execute admitted tools once and reconcile interrupted work across runtimes."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .contracts import (
    AgentExecutionError,
    AgentTask,
    AgentTool,
    ToolContext,
    canonical_json,
)
from .journal import AgentJournal


class OperationPending(AgentExecutionError):
    """The original side effect needs observation, not another dispatch."""


class ToolRefused(AgentExecutionError):
    """A trusted tool refused before performing a side effect."""


class AgentOperations:
    def __init__(
        self,
        journal: AgentJournal,
        tools: tuple[AgentTool, ...],
        *,
        authorize: Callable[[AgentTask, AgentTool, Mapping[str, Any]], None],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.journal = journal
        self.tools = {tool.tool_id: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("agent_duplicate_registered_tools")
        self.authorize = authorize
        self.clock = clock

    def validate_tools(self, task: AgentTask) -> None:
        for name in task.tool_ids:
            tool = self.tools.get(name)
            if tool is None or tool.tool_digest != task.tool_digests[name]:
                raise AgentExecutionError("agent_tool_definition_changed")

    def execute(
        self,
        task: AgentTask,
        *,
        turn_id: str,
        call_id: str,
        name: str,
        arguments: Any,
    ) -> dict[str, Any]:
        """Caller holds the task lock; operation CAS protects shared mutations."""

        task = task.snapshot()
        if self.journal.task(task.task_id)["cancel_requested"]:
            raise AgentExecutionError("agent_task_cancel_requested")
        if self.clock() >= min(task.deadline, task.admission.expires_at):
            raise AgentExecutionError("agent_task_authority_expired")
        self.validate_tools(task)
        if name not in task.tool_ids:
            raise AgentExecutionError("agent_tool_not_admitted")
        tool = self.tools[name]
        arguments = tool.validate_arguments(arguments)
        # Revalidate current authority even when an old result will be reused.
        self.authorize(task, tool, arguments)
        task = task.snapshot()
        operation = self.journal.prepare_call(
            task,
            tool,
            arguments,
            turn_id=turn_id,
            call_id=call_id,
        )
        try:
            with self.journal.own_operation(operation["operation_id"]):
                return self._execute_prepared(task, tool, arguments, operation["operation_id"])
        except AgentExecutionError as exc:
            if str(exc) == "agent_task_owned_by_another_worker":
                raise OperationPending("agent_operation_still_running") from exc
            raise

    def _execute_prepared(self, task, tool, arguments, operation_id):
        operation = self.journal.operation(operation_id)
        if operation["state"] == "completed":
            return operation["outcome"]
        context = ToolContext(
            task.run_id,
            task.task_id,
            task.context_revision,
            operation["operation_id"],
            task.admission.authority_digest,
            task.deadline,
            task.admission.allowed_input_digests,
        )
        reconciled_not_started = False
        if operation["state"] == "executing":
            if tool.effect == "read_only":
                reconciled_not_started = True
            elif tool.reconcile is None:
                raise OperationPending("agent_tool_outcome_unresolved")
            else:
                observation = tool.reconcile(arguments, context)
                if observation.status == "completed":
                    if observation.output is None:
                        raise AgentExecutionError("agent_tool_reconciliation_result_missing")
                    return self._complete(task, operation["operation_id"], observation.output)
                if observation.status != "not_started":
                    raise OperationPending("agent_tool_outcome_unresolved")
                reconciled_not_started = True
        # Authorization can involve slow controller reads. Revalidate after
        # acquiring operation ownership, then let the atomic journal transition
        # linearize start against any concurrently committed cancellation.
        self.authorize(task, tool, arguments)
        task = task.snapshot()
        self.journal.mark_executing(
            operation["operation_id"],
            task=task, clock=self.clock,
            reconciled_not_started=reconciled_not_started,
        )
        try:
            output = tool.invoke(arguments, context)
        except OperationPending:
            raise
        except ToolRefused:
            # Never disclose an arbitrary exception string to the model.
            outcome = {"success": False, "error": "agent_tool_refused_before_execution"}
            self.journal.complete_operation(operation["operation_id"], outcome)
            return outcome
        except Exception as exc:
            if tool.effect != "read_only":
                raise OperationPending("agent_tool_outcome_unresolved") from exc
            outcome = {"success": False, "error": "agent_read_tool_failed"}
            self.journal.complete_operation(operation["operation_id"], outcome)
            return outcome
        return self._complete(task, operation["operation_id"], output)

    def reconcile_cancelled(self, task: AgentTask) -> None:
        """Observe unresolved effects after cancellation without resubmitting.

        Reconciliation is allowed after execution authority expires. Its own
        short deadline bounds observation, and never extends execution rights.
        The caller holds task ownership, just as for execute().
        """

        task = task.snapshot()
        for operation_id in self.journal.unsettled_operations(task.task_id):
            try:
                with self.journal.own_operation(operation_id):
                    self._reconcile_cancelled_operation(task, operation_id)
            except AgentExecutionError as exc:
                if str(exc) == "agent_task_owned_by_another_worker":
                    raise OperationPending("agent_operation_still_running") from exc
                raise

    def _reconcile_cancelled_operation(self, task, operation_id):
        operation = self.journal.operation(operation_id)
        if operation["state"] == "completed":
            return
        request = operation["request"]
        tool = self.tools.get(request["tool_id"])
        if tool is None or tool.tool_digest != request["tool_digest"]:
            raise OperationPending("agent_reconciler_definition_unavailable")
        if tool.effect == "read_only":
            self.journal.complete_operation(
                operation_id,
                {"success": False, "error": "agent_read_abandoned_on_cancel"},
            )
            return
        if tool.reconcile is None:
            raise OperationPending("agent_tool_outcome_unresolved")
        context = ToolContext(
            task.run_id,
            task.task_id,
            task.context_revision,
            operation_id,
            task.admission.authority_digest,
            self.clock() + 20,
            task.admission.allowed_input_digests,
            reconciliation_only=True,
        )
        observation = tool.reconcile(request["arguments"], context)
        if observation.status == "completed" and observation.output is not None:
            self._complete(task, operation_id, observation.output)
        elif observation.status == "not_started":
            self.journal.complete_operation(
                operation_id,
                {"success": False, "error": "agent_operation_abandoned_on_cancel"},
            )
        else:
            raise OperationPending("agent_tool_outcome_unresolved")

    def _complete(self, task: AgentTask, operation_id: str, output: Any) -> dict[str, Any]:
        if not isinstance(output, (dict, list)):
            raise AgentExecutionError("agent_tool_output_invalid")
        serialized = canonical_json(output)
        if len(serialized.encode("utf-8")) > task.max_tool_output_bytes:
            # The side effect may have happened; retain its unsettled operation.
            raise OperationPending("agent_tool_output_limit_exceeded")
        if isinstance(output, list):
            for part in output:
                if not isinstance(part, dict) or part.get("type") not in {
                    "input_text",
                    "input_image",
                }:
                    raise AgentExecutionError("agent_tool_content_part_invalid")
                key = "text" if part["type"] == "input_text" else "image_url"
                if set(part) != {"type", key} or not isinstance(part[key], str):
                    raise AgentExecutionError("agent_tool_content_part_invalid")
        outcome = {"success": True, "output": output}
        self.journal.complete_operation(operation_id, outcome)
        return outcome
