"""Project-scoped SDK fallback with durable reservations and cancellation.

Hermetic tests replace only HTTP, preserving the real SDK and reservation loop.
Interrupted calls retain their reservation and are never silently restarted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Mapping

from pydantic import BaseModel

from ..task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec, OpenAIAgentsSDKConfig, OpenAIAgentsSDKInvoker,
)
from ..task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from ..task_evaluation_supervisor.tools import RegisteredToolBinding
from .contracts import AgentExecutionError, AgentTask, RUNTIME_SDK, canonical_json, digest
from .journal import AgentJournal, TERMINAL_STATES
from .operations import AgentOperations, OperationPending


@dataclass(frozen=True)
class SDKCredential:
    project_id: str
    credential_id: str
    api_key: str = field(repr=False)

    def __post_init__(self):
        if not self.project_id or not self.credential_id or not self.api_key:
            raise ValueError("agents_sdk_credential_incomplete")


async def _tool_thread(function):
    """Keep cancellation responsive while a synchronous tool owns its operation.

    A cancelled await never implies the tool stopped; its durable operation and
    OS lock prevent concurrent retries/reconciliation until this thread exits.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def deliver(value, error):
        if not future.done():
            future.set_exception(error) if error else future.set_result(value)

    def run():
        try:
            value, error = function(), None
        except BaseException as exc:
            value, error = None, exc
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:
            pass  # Closed loop: durable operation remains authoritative.

    threading.Thread(target=run, daemon=True, name="blueprint-agent-tool").start()
    return await future


class OpenAIAgentsSDKRuntime:
    runtime_id = RUNTIME_SDK

    def __init__(
        self, *, journal: AgentJournal, operations: AgentOperations,
        output_models: Mapping[str, type[BaseModel]],
        validate_admission: Callable[[AgentTask], None],
        resolve_credential: Callable[[AgentTask], SDKCredential],
        clock: Callable[[], float] = time.time,
        hermetic_http_transport: Any = None,
    ) -> None:
        if hermetic_http_transport is not None:
            import httpx
            if not isinstance(hermetic_http_transport, httpx.MockTransport):
                raise ValueError("agents_sdk_hermetic_transport_must_be_mock")
        self.journal, self.operations = journal, operations
        self.output_models = dict(output_models)
        self.validate_admission, self.resolve_credential = validate_admission, resolve_credential
        self.clock, self._hermetic_transport = clock, hermetic_http_transport

    def _check_task(self, task: AgentTask) -> type[BaseModel]:
        if task.admission.runtime != RUNTIME_SDK or task.admission.budget_policy != "strict_per_call":
            raise AgentExecutionError("agents_sdk_runtime_authority_mismatch")
        self.operations.validate_tools(task)
        output_model = self.output_models.get(task.capability)
        if output_model is None or digest(output_model.model_json_schema()) != digest(task.output_schema):
            raise AgentExecutionError("agents_sdk_output_contract_not_registered")
        return output_model

    def start(self, task: AgentTask) -> dict[str, Any]:
        return self.admit(task)

    def admit(self, task: AgentTask) -> dict[str, Any]:
        task = task.snapshot()
        if task.parent_task_id is not None:
            raise AgentExecutionError("agents_sdk_managed_session_continuation_not_supported")
        self._check_task(task)
        self.validate_admission(task)
        with self.journal.own_task(task.task_id):
            return self.journal.register(task)

    def inspect(self, task_id: str) -> dict[str, Any]:
        return self.journal.task(task_id)

    def _ensure_active(self, task: AgentTask) -> None:
        if self.journal.task(task.task_id)["cancel_requested"]:
            raise AgentExecutionError("agent_task_cancel_requested")
        if self.clock() >= task.deadline:
            self.journal.request_cancel(task.task_id, "agent_task_deadline")
            raise AgentExecutionError("agent_task_deadline")
        try:
            self.validate_admission(task.snapshot())
        except Exception as exc:
            self.journal.request_cancel(task.task_id, "agent_admission_revoked")
            raise AgentExecutionError("agent_admission_revoked") from exc

    def _audit(self, task):
        return InferenceReservationAudit(
            run_root=self.journal.root / "sdk" / digest(task.task_id).removeprefix("sha256:"),
            run_id=task.run_id,
        )

    def step(self, task_id: str) -> dict[str, Any]:
        with self.journal.own_task(task_id):
            state = self.journal.task(task_id)
            task = AgentTask.model_validate(state["task"])
            output_model = self._check_task(task)
            if state["state"] in TERMINAL_STATES:
                return state
            if state["state"] != "queued":
                self.journal.set_state(task_id, "reconciling",
                                       error_code="agents_sdk_invocation_outcome_unresolved")
                if self.clock() >= task.deadline and not state["cancel_requested"]:
                    self.journal.request_cancel(task_id, "agent_task_deadline")
                    state = self.journal.task(task_id)
                if state["cancel_requested"]:
                    self.operations.reconcile_cancelled(task)
                    if self.journal.unsettled_operations(task_id):
                        raise OperationPending("agent_tool_outcome_unresolved")
                    if state["session_id"] is not None:
                        raise AgentExecutionError("agents_sdk_unexpected_remote_session")
                    if self.journal.event("sdk_cost_boundary_failed_" + task.task_digest[7:]) is not None:
                        return self.journal.task(task_id)
                    event_id = "sdk_local_cancellation_" + task.task_digest[7:]
                    receipt = self.journal.event(event_id)
                    if receipt is None:
                        manifest = self._audit(task).manifest()
                        receipt = {"task_id": task_id, "task_digest": task.task_digest,
                            "local_execution_stopped": True, "tool_operations_reconciled": True,
                            "reservation_manifest": manifest,
                            "unknown_cost_reserved": bool(manifest["in_flight_unknown_count"]),
                            "cancellation_released_inference_reservation": False,
                            "provider_response_completion_claimed": False,
                            "provider_resource_release_claimed": False}
                        self.journal.record_event(event_id, receipt)
                    self.journal.set_state(task_id, "cancelled", error_code=(
                        "agents_sdk_cancelled_reservation_held" if receipt["unknown_cost_reserved"]
                        else "agents_sdk_local_execution_cancelled"))
                return self.journal.task(task_id)
            try:
                self._ensure_active(task)
            except AgentExecutionError:
                self.journal.set_state(task_id, "cancelled", error_code="cancelled_before_invocation")
                return self.journal.task(task_id)
            credential = self.resolve_credential(task.snapshot())
            if type(credential) is not SDKCredential or credential.project_id != task.admission.project_id:
                raise AgentExecutionError("agents_sdk_credential_project_mismatch")
            has_images = any(
                isinstance(part, dict) and part.get("type") == "input_image"
                for message in task.input
                for part in (message.get("content") if isinstance(message.get("content"), list) else [])
            )
            if has_images and (task.tool_ids or task.max_model_turns != 1):
                raise AgentExecutionError("agents_sdk_multimodal_loop_not_qualified")
            adaptive_images = task.capability in {"episode_investigation", "visual_evidence_investigation"}
            # Hold a conservative portion for fixed prompt/schema/tool context.
            # The canonical invoker verifies the complete fixed context again.
            image_context_limit = (task.max_input_tokens - 40_000
                - (task.max_model_turns - 1) * (task.max_output_tokens + 4096)) if adaptive_images else 0
            if adaptive_images and image_context_limit <= 0:
                raise AgentExecutionError("agents_sdk_specialist_context_budget_insufficient")
            audit = self._audit(task)
            from agents import OpenAIProvider
            from openai import AsyncOpenAI
            import httpx

            client = AsyncOpenAI(
                api_key=credential.api_key, project=credential.project_id, organization="",
                base_url="https://api.openai.com/v1", max_retries=0,
                timeout=max(0.01, task.deadline - self.clock()),
                http_client=httpx.AsyncClient(transport=self._hermetic_transport, trust_env=False,
                                               follow_redirects=False),
            )

            def runner(*args, **kwargs):
                return asyncio.run(self._run(task, client, *args, **kwargs))

            invoker = OpenAIAgentsSDKInvoker(
                OpenAIAgentsSDKConfig(
                    model=task.model, max_turns=task.max_model_turns,
                    max_output_tokens=task.max_output_tokens, max_input_tokens=task.max_input_tokens,
                    max_tool_output_bytes=min(task.max_tool_output_bytes, 1_000_000),
                    max_inference_cost_usd=task.admission.inference_budget_usd,
                    tracing_disabled=True, allow_live_invocation=True,
                ), model_provider=OpenAIProvider(openai_client=client, use_responses=True,
                                                 use_responses_websocket=False),
                run_agent=runner, strict_context_accounting=True,
            )
            invoker.configure_reservation_audit(
                record_reservation=audit.record_reservation,
                record_completion=audit.record_completion,
                restored_reserved_cost_usd=float(audit.manifest()["reserved_max_cost_usd"]),
            )
            self.journal.set_state(task_id, "running")
            try:
                invocation = invoker.invoke(
                    AgentsSDKAgentSpec(
                        run_id=task.run_id, capability=task.capability, name=task.capability,
                        instructions=task.instructions, model=task.model,
                        max_turns=task.max_model_turns, max_input_tokens=task.max_input_tokens,
                        max_output_tokens=task.max_output_tokens,
                        max_tool_output_bytes=task.max_tool_output_bytes if adaptive_images else min(task.max_tool_output_bytes, 1_000_000),
                        max_tool_context_tokens=image_context_limit,
                        reasoning_effort=task.reasoning_effort,
                        tool_bindings=self._tool_bindings(task), output_type=output_model,
                        privacy_scope=task.admission.disclosure_scope,
                    ), task.input if has_images else canonical_json(task.input),
                )
                self._ensure_active(task)
                output = task.validate_output(invocation.output.model_dump(mode="json"))
                manifest = audit.manifest()
                if (manifest["reservation_count"] != 1 or manifest["in_flight_unknown_count"]
                        or manifest["reserved_max_cost_usd"] > task.admission.inference_budget_usd
                        or invocation.provider != "openai" or invocation.model != task.model):
                    raise AgentExecutionError("agents_sdk_result_or_budget_receipt_invalid")
                if invocation.cost_usd is not None and (
                    not math.isfinite(invocation.cost_usd)
                    or not 0 <= invocation.cost_usd <= task.admission.inference_budget_usd
                ):
                    raise AgentExecutionError("agents_sdk_result_cost_invalid")
                if self.journal.unsettled_operations(task_id):
                    raise OperationPending("agent_tool_outcome_unresolved")
                result = {
                    "schema_version": "blueprint_agent_task_result.v1",
                    "task_id": task.task_id, "task_digest": task.task_digest,
                    "run_id": task.run_id, "source_commit": task.source_commit,
                    "runtime": self.runtime_id, "runtime_version": invocation.sdk_version,
                    "model": task.model, "session_id": None, "turn_id": "sdk_turn",
                    "project_id": credential.project_id, "credential_id": credential.credential_id,
                    "hermetic": self._hermetic_transport is not None,
                    "output": output, "output_digest": digest(output),
                    "usage": dict(invocation.usage), "cost_status": invocation.cost_status,
                    "cost_usd": invocation.cost_usd, "scope": "agent_execution_only",
                    "reservation_manifest_digest": manifest["inference_reservation_manifest_digest"],
                    "scientific_acceptance_granted": False,
                }
                result["result_digest"] = digest(result)
                self.journal.record_usage(task_id, invocation.usage)
                self.journal.set_state(task_id, "completed", result=result, turn_id="sdk_turn", clock=self.clock)
            except BaseException as exc:
                if str(exc) in {"agents_sdk_actual_cost_exceeds_reserved_maximum",
                        "inference_completion_reconciled_cost_exceeds_reservation",
                        "agents_sdk_result_cost_invalid", "agents_sdk_result_or_budget_receipt_invalid"}:
                    self.journal.record_event("sdk_cost_boundary_failed_" + task.task_digest[7:],
                        {"task_id": task_id, "task_digest": task.task_digest, "cost_bound_unproven": True})
                never_dispatched = audit.manifest()["reservation_count"] == 0
                self.journal.set_state(
                    task_id, "failed" if never_dispatched else "reconciling",
                    error_code=("agents_sdk_refused_before_invocation" if never_dispatched
                                else "agents_sdk_invocation_outcome_unresolved"),
                )
                raise
            finally:
                if not client.is_closed():
                    asyncio.run(client.close())
                audit.write_manifest()
            return self.journal.task(task_id)

    async def _run(self, task, client, *args, **kwargs):
        from agents import Runner

        async def watch():
            while True:
                self._ensure_active(task)
                await asyncio.sleep(min(0.1, max(0.001, task.deadline - self.clock())))

        self._ensure_active(task)
        invocation = asyncio.create_task(Runner.run(*args, **kwargs))
        watcher = asyncio.create_task(watch())
        try:
            done, _ = await asyncio.wait({invocation, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                await watcher
            result = await invocation
            self._ensure_active(task)
            return result
        finally:
            for running in (invocation, watcher):
                running.cancel()
            await asyncio.gather(invocation, watcher, return_exceptions=True)
            await client.close()

    def _tool_bindings(self, task: AgentTask) -> tuple[RegisteredToolBinding, ...]:
        ordinal = 0
        bindings = []
        for name in task.tool_ids:
            tool = self.operations.tools[name]

            async def invoke(arguments, selected=name):
                nonlocal ordinal
                self._ensure_active(task)
                call_id = f"sdk_call_{ordinal}"
                ordinal += 1
                while True:
                    self._ensure_active(task)
                    try:
                        outcome = await _tool_thread(lambda: self.operations.execute(
                            task, turn_id="sdk_turn", call_id=call_id, name=selected, arguments=arguments,
                        ))
                        break
                    except OperationPending:
                        # A queued deterministic worker is still producing this
                        # tool's result. Keep the same call/operation identity;
                        # do not make another model call or abandon the run.
                        await asyncio.sleep(0.2)
                self._ensure_active(task)
                if (outcome["success"] and isinstance(outcome["output"], list)
                        and task.capability not in {"episode_investigation", "visual_evidence_investigation"}):
                    raise AgentExecutionError("agents_sdk_multimodal_tool_output_not_qualified")
                return outcome["output"] if outcome["success"] else outcome

            bindings.append(RegisteredToolBinding(
                tool_id=tool.tool_id, description=tool.description, input_schema=tool.input_schema,
                timeout_seconds=max(1, task.deadline - self.clock()), invoke=invoke,
            ))
        return tuple(bindings)

    def cancel(self, task_id: str) -> dict[str, Any]:
        self.journal.request_cancel(task_id, "agent_task_cancel_requested")
        return self.journal.task(task_id)

    def cleanup(self, task_id: str) -> dict[str, Any]:
        with self.journal.own_task(task_id):
            state = self.journal.task(task_id)
            if state["state"] not in TERMINAL_STATES or self.journal.unsettled_operations(task_id):
                raise AgentExecutionError("agents_sdk_cleanup_before_reconciliation")
            self.journal.cleanup_state(task_id, "deleted")
            return self.journal.task(task_id)
