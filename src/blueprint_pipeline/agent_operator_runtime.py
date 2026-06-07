"""Shared gated runtime helpers for Blueprint SDK operator adapters."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from .common import utc_now_iso


LIVE_AGENTS_SDK_ENV = "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS"
LIVE_CODEX_SDK_ENV = "BLUEPRINT_ALLOW_LIVE_CODEX_SDK_OPERATORS"
AGENT_EXTERNAL_ACTIONS_ENV = "BLUEPRINT_ALLOW_AGENT_EXTERNAL_ACTIONS"
AGENT_SPEND_ACTIONS_ENV = "BLUEPRINT_ALLOW_AGENT_SPEND_ACTIONS"

OperatorExecutor = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class OperatorRunConfig:
    adapter: str
    model: str
    prompt: str
    plan_context: Mapping[str, Any]
    executor: OperatorExecutor | None = None
    sandbox: str | None = None


def env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def string(value: Any) -> str:
    return str(value or "").strip()


def module_available(candidates: Sequence[str]) -> bool:
    return any(importlib.util.find_spec(candidate) is not None for candidate in candidates)


def external_action_gates() -> Dict[str, Any]:
    return {
        "external_actions_env": AGENT_EXTERNAL_ACTIONS_ENV,
        "external_actions_allowed": env_truthy(AGENT_EXTERNAL_ACTIONS_ENV),
        "spend_actions_env": AGENT_SPEND_ACTIONS_ENV,
        "spend_actions_allowed": env_truthy(AGENT_SPEND_ACTIONS_ENV),
    }


def proof_effect(
    *,
    summary: str = "no_direct_proof_boolean_changes",
    deterministic_artifacts_required: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "summary": summary,
        "proof_booleans_mutable_by_agent": False,
        "direct_proof_booleans_set_true": [],
        "requires_deterministic_accepted_artifacts": True,
        "deterministic_artifacts_required": list(deterministic_artifacts_required),
    }


def blocked_operator_ledger(
    *,
    adapter: str,
    blockers: Sequence[str],
    command_chosen: str | None,
    proof_artifacts_required: Sequence[str],
) -> Dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "operator_mode": "live_operator_blocked",
        "decisions": [
            {
                "decision": "live_operator_execution_blocked",
                "owned_by": adapter,
                "reason": blocker,
                "proof_effect": "none",
            }
            for blocker in blockers
        ],
        "tool_call_summaries": [],
        "commands_chosen": [command_chosen] if command_chosen else [],
        "refusals": [
            {
                "reason": blocker,
                "action": "refused_live_operator_execution",
            }
            for blocker in blockers
        ],
        "blockers": list(blockers),
        "proof_effect": proof_effect(
            deterministic_artifacts_required=proof_artifacts_required
        ),
    }


def normalize_operator_output(output: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if isinstance(output, Mapping):
        final_output = output.get("final_output") or output.get("final_response") or output.get(
            "summary"
        )
        return {
            "final_output": string(final_output),
            "decisions": list(output.get("decisions") or []),
            "tool_call_summaries": list(output.get("tool_call_summaries") or []),
            "commands_chosen": list(output.get("commands_chosen") or []),
            "refusals": list(output.get("refusals") or []),
            "blockers": list(output.get("blockers") or []),
            "raw_result_type": string(output.get("raw_result_type")) or type(output).__name__,
        }
    return {
        "final_output": string(output),
        "decisions": [],
        "tool_call_summaries": [],
        "commands_chosen": [],
        "refusals": [],
        "blockers": [],
        "raw_result_type": type(output).__name__,
    }


def _summarize_result_items(result: Any) -> list[Dict[str, Any]]:
    summaries: list[Dict[str, Any]] = []
    for index, item in enumerate(getattr(result, "new_items", []) or []):
        item_type = string(getattr(item, "type", None) or getattr(item, "item_type", None))
        raw_item = getattr(item, "raw_item", None)
        tool_name = string(
            getattr(raw_item, "name", None)
            or getattr(raw_item, "tool_name", None)
            or getattr(item, "name", None)
        )
        if item_type or tool_name:
            summaries.append(
                {
                    "index": index,
                    "item_type": item_type or type(item).__name__,
                    "tool_name": tool_name or None,
                }
            )
    return summaries


def run_agents_sdk_operator(config: OperatorRunConfig) -> Dict[str, Any]:
    if config.executor is not None:
        return normalize_operator_output(config.executor(config.prompt, config.plan_context))
    try:
        from agents import Agent, Runner
    except ImportError as exc:
        raise RuntimeError("missing_openai_agents_sdk") from exc

    agent = Agent(
        name=config.adapter,
        model=config.model,
        instructions=(
            "You are a Blueprint pipeline operator. Inspect the provided manifest context, "
            "choose safe next deterministic commands, summarize blockers, and never claim "
            "proof booleans are true unless deterministic accepted artifacts already say so."
        ),
    )

    async def _run() -> Any:
        return await Runner.run(agent, config.prompt)

    result = asyncio.run(_run())
    return {
        "final_output": string(getattr(result, "final_output", result)),
        "decisions": [],
        "tool_call_summaries": _summarize_result_items(result),
        "commands_chosen": [],
        "refusals": [],
        "blockers": [],
        "raw_result_type": type(result).__name__,
    }


def run_codex_sdk_operator(config: OperatorRunConfig) -> Dict[str, Any]:
    if config.executor is not None:
        return normalize_operator_output(config.executor(config.prompt, config.plan_context))
    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise RuntimeError("missing_codex_sdk") from exc

    sandbox_name = config.sandbox or "read-only"
    sandbox_attr = "workspace_write" if sandbox_name == "workspace-write" else "read_only"
    sandbox_value = getattr(Sandbox, sandbox_attr, None)
    with Codex() as codex:
        thread_kwargs: Dict[str, Any] = {"model": config.model}
        if sandbox_value is not None:
            thread_kwargs["sandbox"] = sandbox_value
        thread = codex.thread_start(**thread_kwargs)
        result = thread.run(config.prompt)
    return {
        "final_output": string(getattr(result, "final_response", result)),
        "decisions": [],
        "tool_call_summaries": _summarize_result_items(result),
        "commands_chosen": [],
        "refusals": [],
        "blockers": [],
        "raw_result_type": type(result).__name__,
    }


def completed_operator_ledger(
    *,
    adapter: str,
    output: Mapping[str, Any],
    default_command: str,
    proof_artifacts_required: Sequence[str],
) -> Dict[str, Any]:
    commands = list(output.get("commands_chosen") or [])
    if default_command and default_command not in commands:
        commands.insert(0, default_command)
    decisions = list(output.get("decisions") or [])
    if not decisions:
        decisions = [
            {
                "decision": "live_operator_completed",
                "owned_by": adapter,
                "summary": string(output.get("final_output")),
                "proof_effect": "none_direct",
            }
        ]
    return {
        "generated_at": utc_now_iso(),
        "operator_mode": "live_operator",
        "decisions": decisions,
        "tool_call_summaries": list(output.get("tool_call_summaries") or []),
        "commands_chosen": commands,
        "refusals": list(output.get("refusals") or []),
        "blockers": list(output.get("blockers") or []),
        "final_output": string(output.get("final_output")),
        "raw_result_type": string(output.get("raw_result_type")),
        "proof_effect": proof_effect(
            deterministic_artifacts_required=proof_artifacts_required
        ),
    }
