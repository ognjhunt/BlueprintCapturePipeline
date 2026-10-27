"""ADP-009D: retain observed usage before the real SDK parses a refused answer."""
from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

import pytest
from agents import ModelProvider
from agents.exceptions import ModelBehaviorError
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from pydantic import BaseModel, ConfigDict

from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKAgentSpec, OpenAIAgentsSDKConfig, OpenAIAgentsSDKInvoker,
)
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class CutOffModel(Model):
    def __init__(self, known_usage=True):
        self.known_usage = known_usage

    async def get_response(self, **kwargs):
        return ModelResponse(output=[ResponseOutputMessage(id="msg-cut-off", role="assistant", status="incomplete",
            type="message", content=[ResponseOutputText(type="output_text",
                text='{"answer":"DO_NOT_LOG_MODEL_TEXT', annotations=[])])],
            usage=Usage(requests=1, input_tokens=8 if self.known_usage else 0,
                output_tokens=4000 if self.known_usage else 0, total_tokens=4008 if self.known_usage else 0),
            response_id="resp-cut-off", request_id="req-cut-off")

    def stream_response(self, *args, **kwargs):
        raise AssertionError("no streaming")


class Provider(ModelProvider):
    def __init__(self, known_usage=True):
        self.known_usage = known_usage

    def get_model(self, model_name):
        return CutOffModel(self.known_usage)


@pytest.mark.parametrize("known_usage", [True, False])
def test_real_sdk_parse_failure_keeps_the_completed_response_usage(tmp_path, monkeypatch, known_usage):
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="sdk-cut-off")
    invoker = OpenAIAgentsSDKInvoker(OpenAIAgentsSDKConfig(allow_live_invocation=True,
        tracing_disabled=True, max_inference_cost_usd=1.), model_provider=Provider(known_usage))
    invoker.configure_reservation_audit(record_reservation=audit.record_reservation,
        record_completion=audit.record_completion, restored_reserved_cost_usd=0.)
    spec = AgentsSDKAgentSpec(run_id="sdk-cut-off", capability="atlas_scene_planner", name="Scene planner",
        instructions="Return the typed answer", model="gpt-6.1-sol", max_turns=1, max_output_tokens=4000,
        output_type=Answer)

    with pytest.raises(ModelBehaviorError) as failed:
        invoker.invoke(spec, "DO_NOT_LOG_PROMPT_TEXT")
    # SDK 0.19.1 has already counted the returned usage but has not appended
    # this response when its structured-output parser raises.
    assert failed.value.run_data.raw_responses == []
    assert failed.value.run_data.context_wrapper.usage.output_tokens == (4000 if known_usage else 0)
    assert audit.manifest()["in_flight_unknown_count"] == 0
    completion = json.loads(next(audit.completed_root.glob("*.json")).read_text())
    assert completion["status"] == "invalid_structured_output"
    assert completion["usage"]["cost_is_actual"] is known_usage
    assert completion["usage"]["provider_response_id"] == "resp-cut-off"
    if known_usage:
        assert completion["usage"]["output_tokens"] == 4000
        assert completion["observed_actual_cost_usd"] == pytest.approx(.040016)
    else:
        assert completion["observed_actual_cost_usd"] is None
        assert completion["released_reservation_usd"] == 0
        assert completion["reconciled_actual_cost_usd"] == completion["projected_max_cost_usd"]
    assert "DO_NOT_LOG" not in json.dumps(completion)


def test_an_unanswered_later_call_cannot_be_settled_from_prior_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    context = SimpleNamespace(usage=Usage(requests=1, input_tokens=8, output_tokens=4000, total_tokens=4008))

    def interrupted(agent, inputs, *, hooks, **kwargs):
        async def record():
            await hooks.on_llm_start(context, agent, "", [])
            response = await CutOffModel().get_response()
            await hooks.on_llm_end(context, agent, response)
            await hooks.on_llm_start(context, agent, "", [])
            return response
        response = asyncio.run(record())
        error = ModelBehaviorError("later call ended without a response")
        error.run_data = SimpleNamespace(raw_responses=[response], context_wrapper=context)
        raise error

    audit = InferenceReservationAudit(run_root=tmp_path, run_id="sdk-unanswered")
    invoker = OpenAIAgentsSDKInvoker(OpenAIAgentsSDKConfig(allow_live_invocation=True,
        tracing_disabled=True, max_inference_cost_usd=1.), model_provider=Provider(), run_agent=interrupted)
    invoker.configure_reservation_audit(record_reservation=audit.record_reservation,
        record_completion=audit.record_completion, restored_reserved_cost_usd=0.)
    spec = AgentsSDKAgentSpec(run_id="sdk-unanswered", capability="atlas_scene_planner", name="Scene planner",
        instructions="Typed answer", model="gpt-6.1-sol", max_turns=2, max_output_tokens=4000,
        max_input_tokens=100_000, output_type=Answer)
    with pytest.raises(ModelBehaviorError):
        invoker.invoke(spec, "fixture")
    assert audit.manifest()["in_flight_unknown_count"] == 1
    assert not list(audit.completed_root.glob("*.json"))


def test_real_sdk_cutoff_reaches_the_planners_larger_allowance_without_spending(tmp_path, monkeypatch):
    from blueprint_pipeline import atlas_scene_planner as planner
    from tests.test_atlas_scene_planner import cut_off_review, fixture

    options, _ = cut_off_review(tmp_path, monkeypatch, cut_offs=0)
    _, _, proposal, _ = fixture(tmp_path)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS", "true")
    allowances = []

    class ReviewModel(CutOffModel):
        async def get_response(self, **kwargs):
            allowances.append(kwargs["model_settings"].max_tokens)
            if len(allowances) == 1:
                return await super().get_response(**kwargs)
            return ModelResponse(output=[ResponseOutputMessage(id="msg-reviewed", role="assistant", status="completed",
                type="message", content=[ResponseOutputText(type="output_text", text=json.dumps(proposal), annotations=[])])],
                usage=Usage(requests=1, input_tokens=8, output_tokens=300, total_tokens=308),
                response_id="resp-reviewed", request_id="req-reviewed")

    class ReviewProvider(ModelProvider):
        def get_model(self, model_name):
            return ReviewModel()

    invoker = OpenAIAgentsSDKInvoker(OpenAIAgentsSDKConfig(allow_live_invocation=True,
        tracing_disabled=True, max_inference_cost_usd=1.), model_provider=ReviewProvider())
    result = planner.review_survey(**{**options, "invoker": invoker})
    assert result["status"] == "completed"
    assert allowances == [4000, 8000]
    assert result["binding"]["max_output_tokens"] == 8000
    assert len(list((tmp_path / "plans").glob("*.settlement.json"))) == 2
