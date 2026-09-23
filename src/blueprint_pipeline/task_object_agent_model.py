"""Per-request spend admission for a persistent SDK authoring loop.

Reuse the existing stage invoker and durable inference ledger. The SDK owns the
conversation and tool loop; each actual model request is separately reserved.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agents.models.interface import Model
from pydantic import BaseModel, ConfigDict

from .task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec
from .task_evaluation_supervisor.sdk_image_tools import encode_tool_output
from .task_object_astra_authoring import AssetAuthoringError


class _ResponseReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_id: str | None


def context_ceiling(value):
    """Count text conservatively and image patches independently of base64 size."""
    images = 0
    def normalized(item):
        nonlocal images
        if hasattr(item, "model_dump"):
            item = item.model_dump(mode="json")
        if isinstance(item, dict):
            if item.get("type") == "input_image":
                _, count = encode_tool_output([{"type": "input_image", "image_url": item.get("image_url")}], model="gpt-6-sol")
                images += count
                return {"type": "input_image", "image_tokens_reserved": count}
            return {key: normalized(part) for key, part in item.items()}
        if isinstance(item, list):
            return [normalized(part) for part in item]
        return item
    text = json.dumps(normalized(value), ensure_ascii=True, allow_nan=False)
    return len(text.encode()) + images + 4096


class BudgetedAuthoringModel(Model):
    def __init__(self, *, delegate, invoker, run_id, object_id):
        self.delegate, self.invoker = delegate, invoker
        self.run_id, self.object_id = run_id, object_id
        self.calls = 0
        self._lock = asyncio.Lock()

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, *, previous_response_id,
                           conversation_id, prompt):
        # Local SDK history is the entire costed context. Server-side hidden
        # history, handoffs, hosted tools and transport retries are not admitted.
        if previous_response_id or conversation_id or prompt or handoffs:
            raise AssetAuthoringError("authoring_uncounted_context_forbidden")
        from agents import FunctionTool
        if any(not isinstance(tool, FunctionTool) for tool in tools):
            raise AssetAuthoringError("authoring_only_local_function_tools_allowed")
        schema = output_schema.json_schema() if output_schema else None
        envelope = {"instructions": system_instructions, "input": input, "output_schema": schema,
            "tools": [{"name": t.name, "description": t.description, "parameters": t.params_json_schema} for t in tools]}
        ceiling = context_ceiling(envelope)
        if ceiling > 80_000:
            from .task_object_agent_context import compact_authoring_history
            input = compact_authoring_history(input)
            envelope["input"] = input
            ceiling = context_ceiling(envelope)
        if ceiling > 80_000:
            raise AssetAuthoringError("authoring_session_context_ceiling_exceeded")
        if not model_settings.max_tokens or model_settings.max_tokens > 12000:
            raise AssetAuthoringError("authoring_output_limit_required")
        async with self._lock:
            # The existing invoker executes a supplied runner callback after it
            # reserves this exact single request. It then records the SDK's
            # real response/usage using the unchanged cost accounting contract.
            self.calls += 1
            spec = AgentsSDKAgentSpec(run_id=self.run_id, capability=f"{self.object_id}_author_turn_{self.calls:03d}",
                name="Persistent asset author", instructions="Record this SDK response receipt.",
                model="gpt-6-sol", max_turns=1, max_input_tokens=ceiling,
                max_output_tokens=model_settings.max_tokens, reasoning_effort="medium", output_type=_ResponseReceipt)
            responses = []
            loop = asyncio.get_running_loop()
            def invoke_reserved():
                base = getattr(self.invoker, "invoker", self.invoker)
                original = base._run_agent
                def run_one(*_args, **_kwargs):
                    response = asyncio.run_coroutine_threadsafe(self.delegate.get_response(
                        system_instructions=system_instructions, input=input, model_settings=model_settings,
                        tools=tools, output_schema=output_schema, handoffs=[], tracing=tracing,
                        previous_response_id=None, conversation_id=None, prompt=None), loop).result()
                    responses.append(response)
                    return SimpleNamespace(final_output={"response_id": response.response_id},
                        raw_responses=[response], context_wrapper=SimpleNamespace(usage=response.usage))
                base._run_agent = run_one
                try:
                    self.invoker.invoke(spec, [{"role": "user", "content": [{"type": "input_text",
                        "text": json.dumps(envelope, default=lambda v: v.model_dump(mode="json"))}]}])
                finally:
                    base._run_agent = original
            await asyncio.to_thread(invoke_reserved)
            if len(responses) != 1:
                raise AssetAuthoringError("authoring_model_response_missing")
            return responses[0]

    def stream_response(self, *args, **kwargs):
        raise AssetAuthoringError("authoring_stream_requires_bounded_nonstreaming_runner")
