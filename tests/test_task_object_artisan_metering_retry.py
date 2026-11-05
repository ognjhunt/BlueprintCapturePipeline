"""A transient OpenAI failure no longer strands an ARTISAN stage.

With the SDK's retries off, a 429, 5xx or dropped connection left its call
``in_flight`` and unpriced, and every later call (and a resume) refused with
``artisan_unknown_charge_requires_reconciliation``. The meter now settles the
failed call (4xx: $0; otherwise the request's upper bound, else unknown) and
retries with the SDK's own policy, metering each attempt. These run the real
SDK resource surface over a local HTTP cassette; no provider call.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import types

import httpx
import pytest

from blueprint_pipeline import task_object_artisan_metering as metering
from blueprint_pipeline.task_object_artisan_metering import Meter, install_metering

_OK = {"id": "resp_fixture", "object": "response", "created_at": 0, "status": "completed",
       "model": "fixture-model", "output": [], "usage": {
           "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
           "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
# 100 input tokens at the highest input rate (12.5/M) + 100 output tokens at 50/M.
_BOUND = (100 * 12.5 + 100 * 50) / 1_000_000
_COST = (100 * 10 + 10 * 50) / 1_000_000


@pytest.fixture
def metered(tmp_path, monkeypatch):
    from openai.resources.responses.responses import AsyncResponses, Responses

    for name in ("pipeline", "pipeline.agents"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    cost = types.ModuleType("pipeline.cost")
    cost.pricing_for_provider_model = lambda *_: {"input_uncached": 10, "input_cached": 1,
                                                  "input_cache_write": 12.5, "output": 50}
    cost.calculate_cost = lambda usage, _: types.SimpleNamespace(
        total_cost=(usage["prompt_tokens"] * 10 + usage["candidates_tokens"] * 50) / 1_000_000)
    monkeypatch.setitem(sys.modules, "pipeline.cost", cost)
    codec = types.ModuleType("pipeline.llm.openai_codec")
    codec.extract_usage = lambda response: {"prompt_tokens": response.usage.input_tokens,
                                            "candidates_tokens": response.usage.output_tokens}
    monkeypatch.setitem(sys.modules, "pipeline.llm.openai_codec", codec)
    design = types.ModuleType("pipeline.agents.design")
    design._generate_image = lambda *_a, **_k: (True, {"accounting": {"budget_cost_usd": 0.1}})
    monkeypatch.setitem(sys.modules, "pipeline.agents.design", design)
    for resource in (Responses, AsyncResponses):
        for name in ("create", "compact"):
            monkeypatch.setattr(resource, name, getattr(resource, name))
    sleeps = []
    monkeypatch.setattr(metering.time, "sleep", sleeps.append)
    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=1.0, maximum_requests=20,
                  deadline_epoch=time.time() + 600)
    install_metering(meter)
    return meter, sleeps


def _transport(replies):
    """Serve token counts, then the scripted replies to each create/compact."""

    seen = []

    def handle(request):
        if request.url.path.endswith("/input_tokens"):
            return httpx.Response(200, json={"object": "response.input_tokens", "input_tokens": 100})
        seen.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return handle, seen


def _client(handle):
    from openai import OpenAI

    return OpenAI(api_key="synthetic-never-sent", http_client=httpx.Client(transport=httpx.MockTransport(handle)))


def test_rate_limit_is_unbilled_and_retried_with_retry_after(metered):
    meter, sleeps = metered
    handle, seen = _transport([httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "slow"}}),
                               httpx.Response(200, json=_OK)])
    with _client(handle) as client:
        assert client.responses.create(model="fixture-model", input="x", max_output_tokens=100).id == "resp_fixture"
    calls = meter.read()["calls"]
    assert [(c["status"], c["cost_usd"], c["cost_origin"]) for c in calls] == [
        ("priced", 0.0, "upstream_rejected_request"), ("priced", pytest.approx(_COST), "upstream_priced_usage")]
    assert calls[0]["failure"]["http_status"] == 429
    assert sleeps == [3.0] and len(seen) == 2
    assert meter.remaining() == pytest.approx(1.0 - _COST)


def test_server_errors_are_priced_at_their_bound_and_stop_after_the_sdk_retry_limit(metered):
    import openai

    meter, sleeps = metered
    handle, seen = _transport([httpx.Response(500, json={"error": {"message": "boom"}}) for _ in range(5)])
    with _client(handle) as client, pytest.raises(openai.InternalServerError):
        client.responses.create(model="fixture-model", input="x", max_output_tokens=100)
    calls = meter.read()["calls"]
    assert len(calls) == len(seen) == 1 + openai.DEFAULT_MAX_RETRIES
    assert all(c["status"] == "priced" and c["cost_usd"] == pytest.approx(_BOUND)
               and c["cost_origin"] == "request_upper_bound_no_usage" for c in calls)
    assert sleeps == [0.5, 1.0]
    # Nothing is left unknown, so the stage (or a resume) can keep calling.
    assert meter.before("next", "fixture-model")


def test_lost_connection_is_priced_at_its_bound_then_retried(metered):
    meter, _sleeps = metered
    handle, _seen = _transport([httpx.ConnectError("reset by peer"), httpx.Response(200, json=_OK)])
    with _client(handle) as client:
        client.responses.create(model="fixture-model", input="x", max_output_tokens=100)
    first, second = meter.read()["calls"]
    assert first["failure"]["error_type"] == "APIConnectionError" and first["failure"]["http_status"] is None
    assert first["cost_usd"] == pytest.approx(_BOUND) and second["cost_usd"] == pytest.approx(_COST)


def test_bad_request_is_unbilled_and_not_retried(metered):
    import openai

    meter, sleeps = metered
    handle, seen = _transport([httpx.Response(400, json={"error": {"message": "bad"}})])
    with _client(handle) as client, pytest.raises(openai.BadRequestError):
        client.responses.create(model="fixture-model", input="x", max_output_tokens=100)
    assert [(c["status"], c["cost_usd"]) for c in meter.read()["calls"]] == [("priced", 0.0)]
    assert sleeps == [] and len(seen) == 1


def test_unbounded_compaction_server_error_still_requires_reconciliation(metered):
    import openai

    meter, sleeps = metered
    handle, seen = _transport([httpx.Response(503, json={"error": {"message": "busy"}})])
    with _client(handle) as client, pytest.raises(openai.InternalServerError):
        client.responses.compact(model="fixture-model", input="x")
    assert meter.read()["calls"][0]["status"] == "reconciliation_required"
    assert sleeps == [] and len(seen) == 1
    with pytest.raises(RuntimeError, match="requires_reconciliation"):
        meter.before("next", "fixture-model")


def test_async_client_retries_the_same_way(metered, monkeypatch):
    from openai import AsyncOpenAI

    meter, _sleeps = metered
    waits = []

    async def no_wait(seconds):
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    handle, _seen = _transport([httpx.Response(429, json={"error": {"message": "slow"}}), httpx.Response(200, json=_OK)])

    async def run():
        async with AsyncOpenAI(api_key="synthetic-never-sent",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            return await client.responses.create(model="fixture-model", input="x", max_output_tokens=100)

    assert asyncio.run(run()).id == "resp_fixture"
    assert [c["cost_usd"] for c in meter.read()["calls"]] == [0.0, pytest.approx(_COST)]
    assert waits == [0.5]


def test_a_retry_that_would_pass_the_deadline_is_not_attempted(tmp_path):
    import openai

    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=1.0, maximum_requests=5, deadline_epoch=time.time() + 2)
    identity = meter.before("create", "fixture-model")
    request = httpx.Request("POST", "https://api.invalid/v1/responses")
    error = openai.RateLimitError("slow", response=httpx.Response(429, headers={"retry-after": "30"}, request=request),
                                  body=None)
    with pytest.raises(openai.RateLimitError):
        metering._next_attempt(meter, identity, error, 0.5, 0)
    assert meter.read()["calls"][0]["cost_usd"] == 0.0
