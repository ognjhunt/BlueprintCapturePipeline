"""Native Claude provider for local Agents SDK Luna replacements.

The SDK keeps ownership of local tools and Pydantic validation. Each model turn
uses the existing private receipt ledger and fixed, nonretrying Messages transport.
OpenAI hosted tools, remote media and server conversation state are unsupported.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV, env_truthy
from .haiku_vision_judge import MODEL, anthropic_key, bounded_message, replacement_model
from .task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec,
    AgentsSDKInvocationBlocked,
    AgentsSDKInvocationResult,
)
from .task_object_claude_model import ClaudeMessagesModel


def sdk_model(model: str) -> str:
    return replacement_model(model)


def sdk_credentials_present(model: str, *, openai_api_key: str | None = None,
                            anthropic_api_key: str | None = None) -> bool:
    import os
    if sdk_model(model) == MODEL:
        base = os.getenv("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
        if base and base != "https://api.anthropic.com":
            return False
        return bool(anthropic_api_key if anthropic_api_key is not None else anthropic_key()[0])
    return bool(openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY", "").strip())


def _object(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _object(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_object(item) for item in value]
    return value


class _BudgetedMessages:
    def __init__(self, *, audit_root: Path, capability: str, api_key: str,
                 input_limit: int, maximum_cost_usd: float):
        self.audit_root, self.capability, self.api_key = audit_root, capability, api_key
        self.input_limit, self.maximum_cost_usd = input_limit, maximum_cost_usd
        self.receipts: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **payload):
        response, receipt = await asyncio.to_thread(bounded_message, payload=payload,
            api_key=self.api_key, audit_root=self.audit_root, capability=self.capability,
            input_limit=self.input_limit, maximum_cost_usd=self.maximum_cost_usd)
        self.receipts.append(receipt)
        if not isinstance(response.get("content"), list) or not isinstance(response.get("id"), str) or not response["id"]:
            raise RuntimeError("haiku_sdk_response_invalid")
        # Preserve tool arguments as dictionaries; opaque thinking fields stay unchanged.
        result = _object(dict(response))
        for raw, block in zip(response["content"], result.content, strict=True):
            if isinstance(raw, dict) and raw.get("type") == "tool_use":
                block.input = raw.get("input")
        return result


def budgeted_sdk_model(*, audit_root: Path, capability: str, effort: str,
                       input_limit: int = 100_000, maximum_cost_usd: float = 5.0,
                       api_key: str | None = None) -> tuple[ClaudeMessagesModel, _BudgetedMessages]:
    if not sdk_credentials_present(MODEL, anthropic_api_key=api_key):
        raise AgentsSDKInvocationBlocked("missing_native_anthropic_api_key")
    key = api_key if api_key is not None else anthropic_key()[0]
    client = _BudgetedMessages(audit_root=audit_root, capability=capability, api_key=key,
                              input_limit=input_limit, maximum_cost_usd=maximum_cost_usd)
    return ClaudeMessagesModel(client=client, model=MODEL,
                               effort="low" if effort == "minimal" else effort), client


@dataclass
class HaikuAgentsSDKInvoker:
    """Explicit native invoker for admitted local structured proposal agents."""

    audit_root: Path
    allow_live_invocation: bool = False
    maximum_cost_usd: float = 0.0

    def invoke(self, spec: AgentsSDKAgentSpec,
               input_value: str | list[dict[str, Any]]) -> AgentsSDKInvocationResult:
        if not self.allow_live_invocation or not env_truthy(LIVE_AGENTS_SDK_ENV):
            raise AgentsSDKInvocationBlocked("live_agents_sdk_invocation_not_authorized")
        if (sdk_model(spec.model) != MODEL or not 1 <= spec.max_turns <= 12
                or not 256 <= spec.max_output_tokens <= 16000
                or not 0 < self.maximum_cost_usd <= 5
                or spec.processing_region != "default"
                or spec.hosted_tools or spec.cache_policy is not None
                or spec.stable_developer_prefix or spec.scene_static_prefix):
            raise AgentsSDKInvocationBlocked("haiku_sdk_spec_unsupported")
        if spec.tool_bindings and spec.max_tool_output_bytes <= 0:
            raise AgentsSDKInvocationBlocked("haiku_sdk_tool_output_ceiling_required")
        from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner

        model, client = budgeted_sdk_model(audit_root=self.audit_root,
            capability=f"{spec.run_id}:{spec.capability}", effort=spec.reasoning_effort or "medium",
            input_limit=spec.max_input_tokens or 100_000, maximum_cost_usd=self.maximum_cost_usd)
        observations = []
        output_bytes = 0
        tools = []
        for binding in spec.tool_bindings:
            try:
                from jsonschema import Draft202012Validator
            except ImportError as exc:
                raise AgentsSDKInvocationBlocked("haiku_sdk_tool_schema_validator_missing") from exc
            Draft202012Validator.check_schema(dict(binding.input_schema))
            validator = Draft202012Validator(dict(binding.input_schema))
            async def invoke_tool(_context, input_json, *, selected=binding, schema_validator=validator):
                try:
                    arguments = json.loads(input_json)
                except ValueError:
                    arguments = None
                if not isinstance(arguments, dict) or not schema_validator.is_valid(arguments):
                    observed = {"error": "haiku_sdk_tool_arguments_invalid",
                                "repair": "Use this registered tool's complete original argument schema."}
                else:
                    observed = selected.invoke(arguments)
                    if inspect.isawaitable(observed):
                        observed = await observed
                encoded = json.dumps(observed, ensure_ascii=True, allow_nan=False)
                nonlocal output_bytes
                output_bytes += len(encoded.encode())
                if output_bytes > spec.max_tool_output_bytes * max(1, spec.max_turns - 1):
                    raise ValueError("haiku_sdk_tool_output_ceiling_exceeded")
                observations.append(observed)
                return encoded
            tools.append(FunctionTool(name=binding.tool_id, description=binding.description,
                params_json_schema=dict(binding.input_schema), on_invoke_tool=invoke_tool,
                strict_json_schema=True, needs_approval=False,
                timeout_seconds=binding.timeout_seconds, timeout_behavior="raise_exception"))
        agent = Agent(name=spec.name, instructions=spec.instructions, model=model,
            model_settings=ModelSettings(store=False, max_tokens=spec.max_output_tokens,
                                         parallel_tool_calls=False),
            tools=tools, output_type=spec.output_type)
        started = time.monotonic()
        result = Runner.run_sync(agent, input_value, max_turns=spec.max_turns,
                                 run_config=RunConfig(tracing_disabled=True,
                                                      trace_include_sensitive_data=False))
        output = spec.output_type.model_validate(result.final_output)
        receipts = client.receipts
        return AgentsSDKInvocationResult(output=output, provider="anthropic", model=MODEL,
            sdk_version=metadata.version("openai-agents"), latency_seconds=time.monotonic() - started,
            usage={"provider_calls": receipts},
            cost_usd=sum(receipt["usage"]["estimated_total_cost_usd"] for receipt in receipts),
            cost_status="model_pricing_estimate_not_official_billing",
            tool_observations=tuple(observations))
