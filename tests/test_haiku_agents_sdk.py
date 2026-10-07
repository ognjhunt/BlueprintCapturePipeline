from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from blueprint_pipeline import haiku_vision_judge as transport
from blueprint_pipeline.agent_operator_runtime import OperatorRunConfig, run_agents_sdk_operator
from blueprint_pipeline.exact_workcell_variation_inputs import AgentsSDKVariationProposalAgent
from blueprint_pipeline.haiku_agents_sdk import HaikuAgentsSDKInvoker
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec
from blueprint_pipeline.task_evaluation_supervisor.tools import RegisteredToolBinding


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=200)


def reply(*blocks, stop_reason="end_turn", **overrides):
    return {"model": transport.MODEL, "id": "msg_fixture", "type": "message", "role": "assistant",
            "stop_reason": stop_reason, "content": list(blocks),
            "usage": {"input_tokens": 100, "output_tokens": 50, "inference_geo": "us"}, **overrides}


@pytest.fixture(autouse=True)
def native_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-native-fixture")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_real_sdk_tool_loop_native_schema_thinking_usage_and_reservations(monkeypatch, tmp_path):
    calls = []
    responses = [reply({"type": "thinking", "thinking": "", "signature": "opaque-signed"},
        {"type": "tool_use", "id": "toolu_fixture", "name": "inspect_manifest", "input": {}}, stop_reason="tool_use"),
        reply({"type": "text", "text": '{"summary":"Review remains required"}'})]
    def send(payload, _key):
        calls.append(payload)
        assert len(list((tmp_path / "haiku_vision_budget/inference_reservations/reserved").glob("*.json"))) == len(calls)
        return responses.pop(0)
    monkeypatch.setattr(transport, "_post_message", send)
    observations = []
    binding = RegisteredToolBinding(tool_id="inspect_manifest", description="Read admitted manifest",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False}, timeout_seconds=1,
        invoke=lambda args: observations.append(args) or {"proof_complete": False})
    invoker = HaikuAgentsSDKInvoker(audit_root=tmp_path, allow_live_invocation=True, maximum_cost_usd=1)
    spec = AgentsSDKAgentSpec(run_id="fixture", capability="inspect", name="Inspector", instructions="Keep proof unchanged.",
        model="gpt-6-luna", reasoning_effort="max", max_turns=2, max_output_tokens=4000,
        tool_bindings=(binding,), max_tool_output_bytes=1000, output_type=Output)
    result = invoker.invoke(spec, "Inspect this development-only fixture")
    assert result.provider == "anthropic" and result.model == "claude-haiku-5-5"
    assert result.output.summary == "Review remains required" and observations == [{}]
    assert len(result.usage["provider_calls"]) == 2
    assert result.cost_usd == pytest.approx(0.000077)
    assert calls[0]["output_config"]["effort"] == "max"
    assert "minLength" not in json.dumps(calls[0]["output_config"]["format"]["schema"])
    assert "minLength" in calls[0]["system"]
    assert calls[1]["messages"][1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "opaque-signed"}
    assert calls[1]["messages"][2]["content"][0]["tool_use_id"] == "toolu_fixture"
    assert "proof_complete" in calls[1]["messages"][2]["content"][0]["content"]
    assert not {"reasoning", "prompt_cache_options", "previous_response_id", "temperature"} & calls[0].keys()


def test_real_operator_uses_native_model_without_openai_key(monkeypatch, tmp_path):
    monkeypatch.setattr(transport, "_post_message", lambda *_: reply({"type": "text", "text": "Await deterministic evidence"}))
    result = run_agents_sdk_operator(OperatorRunConfig(adapter="fixture-operator", model="gpt-6-luna",
        reasoning_effort="xhigh", prompt="Plan safe next commands", plan_context={"capture_root": str(tmp_path)}))
    assert result["provider"] == "anthropic" and result["model"] == transport.MODEL
    assert result["final_output"] == "Await deterministic evidence"
    assert result["provider_calls"][0]["usage"]["output_tokens"] == 50


def test_variation_proposer_uses_native_structured_invoker(monkeypatch, tmp_path):
    monkeypatch.setattr(transport, "_post_message", lambda *_: reply({"type": "text", "text":
        '{"dimension_priorities":[],"targeted_interactions":[],"object_cousins":[]}'}))
    proposer = AgentsSDKVariationProposalAgent(run_id="variation-fixture",
        invoker=HaikuAgentsSDKInvoker(audit_root=tmp_path, allow_live_invocation=True, maximum_cost_usd=1))
    assert proposer.propose(brief={"development_only": True})["object_cousins"] == []
    assert proposer.model_identity.startswith("anthropic-agents-sdk:claude-haiku-5-5:reasoning=max")


def test_unknown_transport_is_reserved_and_not_retried(monkeypatch, tmp_path):
    calls = []
    def send(*_):
        calls.append(1)
        raise TimeoutError("offline transport failure")
    monkeypatch.setattr(transport, "_post_message", send)
    config = OperatorRunConfig(adapter="fixture", model=transport.MODEL, prompt="Inspect", plan_context={"capture_root": str(tmp_path)})
    with pytest.raises(TimeoutError):
        run_agents_sdk_operator(config)
    with pytest.raises(ValueError, match="prior_inference_reservation"):
        run_agents_sdk_operator(config)
    assert calls == [1]
    assert len(list((tmp_path / "haiku_vision_budget/inference_reservations/reserved").glob("*.json"))) == 1
    assert not list((tmp_path / "haiku_vision_budget/inference_reservations/completed").glob("*.json"))


def test_missing_provider_and_durable_context_fail_before_call(monkeypatch, tmp_path):
    monkeypatch.setattr(transport, "_post_message", lambda *_: pytest.fail("must not call provider"))
    config = OperatorRunConfig(adapter="fixture", model=transport.MODEL, prompt="Inspect", plan_context={})
    with pytest.raises(RuntimeError, match="audit_root"):
        run_agents_sdk_operator(config)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    with pytest.raises(RuntimeError, match="native_anthropic"):
        run_agents_sdk_operator(OperatorRunConfig(adapter="fixture", model=transport.MODEL,
            prompt="Inspect", plan_context={"capture_root": str(tmp_path)}))


def test_constrained_tool_arguments_are_repaired_before_local_invocation(monkeypatch, tmp_path):
    calls, invoked = [], []
    replies = [reply({"type": "tool_use", "id": "bad", "name": "inspect_count", "input": {"count": 0}}, stop_reason="tool_use"),
               reply({"type": "tool_use", "id": "good", "name": "inspect_count", "input": {"count": 2}}, stop_reason="tool_use"),
               reply({"type": "text", "text": '{"summary":"Reviewed"}'})]
    def send(payload, _key):
        calls.append(payload)
        return replies.pop(0)
    monkeypatch.setattr(transport, "_post_message", send)
    binding = RegisteredToolBinding(tool_id="inspect_count", description="Read a bounded count",
        input_schema={"type": "object", "properties": {"count": {"type": "integer", "minimum": 1, "maximum": 3}},
                      "required": ["count"], "additionalProperties": False}, timeout_seconds=1,
        invoke=lambda args: invoked.append(args) or {"proof_complete": False})
    invoker = HaikuAgentsSDKInvoker(audit_root=tmp_path, allow_live_invocation=True, maximum_cost_usd=1)
    spec = AgentsSDKAgentSpec(run_id="constrained-fixture", capability="inspect", name="Inspector",
        instructions="Use admitted read-only tools.", model=transport.MODEL, max_turns=3, max_output_tokens=4000,
        tool_bindings=(binding,), max_tool_output_bytes=1000, output_type=Output)
    assert invoker.invoke(spec, "Inspect").output.summary == "Reviewed"
    assert invoked == [{"count": 2}]
    assert "minimum" not in json.dumps(calls[0]["tools"][0]["input_schema"])
    assert "minimum" in calls[0]["tools"][0]["description"]
    assert "haiku_sdk_tool_arguments_invalid" in calls[1]["messages"][2]["content"][0]["content"]
