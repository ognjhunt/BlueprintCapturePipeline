"""ARTISAN's text roles on Anthropic: the meter, the bridge and the driver (stand-ins; no provider call, no key).

Reviewed in Group E and its re-checks (S/rv-pipe-e2); the official Anthropic gate is not part of this cut.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_object_artisan_metering as metering
from blueprint_pipeline.task_object_artisan_metering import Meter, install_metering
from blueprint_pipeline.task_object_authoring_backend import ARTISAN_MODELS

ARTISAN_CLAUDE_MODELS = {"provider": "anthropic", "model": "claude-opus-5-5", "thinking": "max",
                         "image_model": "gpt-6.1-sol", "image_tool_model": "gpt-image-2.5-sunburst"}
RATES = {"input_uncached": 4.0, "input_cached": 0.2, "input_cache_write": 5.0, "input_cache_write_1h": 8.0,
         "output": 20.0}


# -- the meter ------------------------------------------------------------------------------------------------

class _Count:
    def __init__(self, tokens, seen):
        self.tokens, self.seen = tokens, seen

    async def count_tokens(self, **kwargs):
        self.seen.append(("count", kwargs))
        return SimpleNamespace(input_tokens=self.tokens)


class _SDKError(Exception):
    def __init__(self, status, error_type, code=None, message="refused"):
        super().__init__(message)
        self.status_code = status
        self.body = {"type": "error", "error": {"type": error_type, "message": message,
                                                **({"code": code} if code else {})}}


class _UpstreamError(RuntimeError):
    """Upstream's AnthropicAPIError shape: status, type and request id, the SDK error as cause."""

    def __init__(self, status):
        super().__init__("Anthropic API error")
        self.status_code, self.request_id = status, "req_fixture"


def _response(model="claude-opus-5-5", iterations=None, stop="end_turn", **usage_fields):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=100, iterations=iterations, **usage_fields)
    return SimpleNamespace(id="msg_fixture", model=model, usage=usage, stop_reason=stop)


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    from openai.resources.responses.responses import AsyncResponses, Responses

    for name in ("pipeline", "pipeline.agents", "pipeline.llm"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    cost = types.ModuleType("pipeline.cost")
    cost.pricing_for_provider_model = lambda provider, model: (
        RATES if (provider, model) == ("anthropic", "claude-opus-5-5")
        else {"input_uncached": 10, "input_cached": 1, "input_cache_write": 12.5, "output": 50}
        if provider == "openai" else None)
    cost.calculate_cost = lambda usage, rates: SimpleNamespace(
        total_cost=(usage["prompt_tokens"] * rates["input_uncached"] + usage["candidates_tokens"] * rates["output"])
        / 1_000_000)
    monkeypatch.setitem(sys.modules, "pipeline.cost", cost)
    codec = types.ModuleType("pipeline.llm.openai_codec")
    codec.extract_usage = lambda response: None
    monkeypatch.setitem(sys.modules, "pipeline.llm.openai_codec", codec)
    design = types.ModuleType("pipeline.agents.design")
    design.outcome = (True, {"accounting": {"budget_cost_usd": 0.1}})
    design._generate_image = lambda *_a, **_k: design.outcome
    monkeypatch.setitem(sys.modules, "pipeline.agents.design", design)
    seen = []
    llm = types.ModuleType("pipeline.llm.anthropic")

    class AnthropicLLM:
        replies: list = []

        def __init__(self, betas=("thinking-binding-controls-2026-08-01",), tokens=1000):
            self._betas = list(betas)
            self._client = SimpleNamespace(messages=_Count(tokens, seen), beta=SimpleNamespace(messages=_Count(tokens, seen)))

        def _request_betas(self):
            return list(self._betas)

        async def _create_message(self, payload):
            seen.append(("create", payload))
            reply = AnthropicLLM.replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply

    llm.AnthropicLLM = AnthropicLLM
    llm._extract_usage = lambda response: {"prompt_tokens": response.usage.input_tokens,
                                           "candidates_tokens": response.usage.output_tokens}
    monkeypatch.setitem(sys.modules, "pipeline.llm.anthropic", llm)
    for resource in (Responses, AsyncResponses):
        for name in ("create", "compact"):
            monkeypatch.setattr(resource, name, getattr(resource, name))
    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=1.0, maximum_requests=20, deadline_epoch=time.time() + 600)
    return SimpleNamespace(meter=meter, llm=AnthropicLLM, seen=seen, design=design)


PAYLOAD = {"model": "claude-opus-5-5", "max_tokens": 64000, "messages": [{"role": "user", "content": "x"}],
           "system": "s", "tools": [], "thinking": {"type": "adaptive"}, "output_config": {"effort": "max"},
           "cache_control": {"type": "ephemeral"}}


def test_the_default_install_leaves_upstream_s_anthropic_adapter_alone(upstream):
    original = upstream.llm._create_message
    install_metering(upstream.meter)
    assert upstream.llm._create_message is original
    with pytest.raises(ValueError, match="artisan_meter_provider_unknown"):
        install_metering(upstream.meter, "google")


def test_a_messages_call_is_bounded_counted_and_priced_at_the_served_model(upstream, tmp_path):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    upstream.llm.replies = [_response()]  # its usage names no region
    response = asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    assert response.id == "msg_fixture"
    (count_kind, counted), (create_kind, sent) = upstream.seen
    assert count_kind == "count" and create_kind == "create"
    assert counted == {**{k: v for k, v in PAYLOAD.items() if k != "max_tokens"},
                       "betas": ["thinking-binding-controls-2026-08-01"]}
    # $1 less 1000 input tokens at the dearer rate that can apply (the 5-minute cache write, $5/M), at $20/M
    # output, both with the 1.1x US uplift, is 45,204 tokens. Pinned to US-only inference; standard
    # capacity only, as a Priority Tier answer would be priced above the bound.
    assert sent == {**PAYLOAD, "inference_geo": "us", "max_tokens": 45_204, "service_tier": "standard_only"}
    [row] = upstream.meter.read()["calls"]
    assert row["status"] == "priced" and row["kind"] == "messages" and row["model"] == "claude-opus-5-5"
    # What was asked decides the price: a response naming no region is priced as the US request it was.
    cost = (1000 * 4 + 100 * 20) / 1_000_000 * 1.1
    assert row["cost_usd"] == pytest.approx(cost) and upstream.meter.remaining() == pytest.approx(1.0 - cost)
    assert row["cost_origin"] == "upstream_priced_usage" and "failure" not in row
    # ... and the month settles at it.
    from blueprint_pipeline import artisan_anthropic_month as month
    from tests.test_artisan_anthropic_month import _archive
    archive = _archive(tmp_path / "out.zip", upstream.meter.read()["calls"])
    assert month.metered_anthropic_subtotal(archive) == pytest.approx(cost, abs=1e-6)


def test_the_request_s_own_max_tokens_is_kept_when_it_fits(upstream):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    upstream.llm.replies = [_response()]
    asyncio.run(upstream.llm(betas=())._create_message({**PAYLOAD, "max_tokens": 4096}))
    assert upstream.seen[0][1].keys() == {k for k in PAYLOAD if k != "max_tokens"}
    assert upstream.seen[1][1]["max_tokens"] == 4096


@pytest.mark.parametrize("response", [_response(model="claude-unpriced"),
                                      _response(iterations=[SimpleNamespace(type="compaction")])])
def test_an_unpriced_model_or_an_itemized_server_step_needs_reconciliation(upstream, response):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    upstream.llm.replies = [response]
    with pytest.raises(RuntimeError, match="artisan_unknown_charge_requires_reconciliation"):
        asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    assert upstream.meter.read()["calls"][0]["status"] == "reconciliation_required"


_BOUND = (1000 * 5 * 1.1 + 45_204 * 20 * 1.1) / 1_000_000
_UNPINNED_BOUND = (1000 * 5 + 49_750 * 20) / 1_000_000


@pytest.mark.parametrize("response", [
    _response(stop=None),                    # the stream ended before its final usage
    _response(inference_geo="eu"),           # a region the price does not know
    _response(speed="fast"),
    _response(service_tier="priority"),
    _response(inference_geo="global"),       # contradicts the US pin
])
def test_e_m2_s3_an_unpriceable_response_is_counted_at_its_bound_for_reconciliation(upstream, response):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    upstream.llm.replies = [response]
    asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    [row] = upstream.meter.read()["calls"]
    assert row["status"] == "priced" and row["cost_usd"] == pytest.approx(_BOUND)
    assert row["cost_origin"] == "request_upper_bound_unpriceable_response"
    assert row["failure"]["reconciliation"] == "official_cost_required"


def test_e_s3_the_us_region_is_priced_with_the_frozen_uplift(upstream):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    upstream.llm.replies = [_response(inference_geo="us"), _response()]  # named "us", or no region named
    for _ in range(2):
        asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    for row in upstream.meter.read()["calls"]:
        assert row["cost_usd"] == pytest.approx((1000 * 4 + 100 * 20) / 1_000_000 * 1.1)
        assert row["cost_origin"] == "upstream_priced_usage"


@pytest.mark.parametrize("multiplier", [None, True, "1.1", 0.9, float("nan"), float("inf")])
def test_a_us_pinned_meter_without_its_multiplier_is_refused_before_any_call(upstream, multiplier):
    original = upstream.llm._create_message
    with pytest.raises(RuntimeError, match="artisan_anthropic_us_multiplier_unavailable"):
        install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=multiplier)
    assert upstream.llm._create_message is original and upstream.seen == [] and upstream.meter.read()["calls"] == []


def test_server_compaction_is_refused_before_any_request(upstream):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    with pytest.raises(RuntimeError, match="artisan_anthropic_compaction_unmetered"):
        asyncio.run(upstream.llm()._create_message({**PAYLOAD, "context_management": {"edits": []}}))
    assert upstream.seen == [] and upstream.meter.read()["calls"] == []


def test_a_refusal_for_lack_of_funds_is_no_charge_and_not_retried(upstream):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    error = _UpstreamError(402)
    error.__cause__ = _SDKError(402, "billing_error")
    upstream.llm.replies = [error]
    with pytest.raises(metering.ProviderOutOfFunds) as raised:
        asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    assert raised.value.__cause__ is error
    # Upstream's retry rule reads the message text; nothing in it asks for a retry.
    assert not any(word in str(raised.value).lower() for word in ("timeout", "rate limit", "overloaded", "connection"))
    # The provider's own error stays on the cause chain.
    assert raised.value.__cause__.__cause__ is error.__cause__
    [row] = upstream.meter.read()["calls"]
    # Marked as the refusal for lack of funds, so the attempt settles at its true metered cost.
    assert (row["status"], row["cost_usd"], row["cost_origin"]) == ("priced", 0.0, "upstream_rejected_request_no_funds")


def test_other_failures_settle_as_before_and_reach_upstream_s_retry_unchanged(upstream):
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    rate_limited = _UpstreamError(429)
    rate_limited.__cause__ = _SDKError(429, "rate_limit_error")
    server = _UpstreamError(529)
    server.__cause__ = _SDKError(529, "overloaded_error")
    upstream.llm.replies = [rate_limited, server]
    for error in (rate_limited, server):
        with pytest.raises(_UpstreamError) as raised:
            asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
        assert raised.value is error
    first, second = upstream.meter.read()["calls"]
    assert (first["cost_usd"], first["cost_origin"]) == (0.0, "upstream_rejected_request")
    assert second["cost_origin"] == "request_upper_bound_no_usage"
    assert second["cost_usd"] == pytest.approx(_BOUND)


@pytest.mark.parametrize("detail", [
    {"summary": "status 429: b'...'", "http_status": 429, "error_type": "rate_limit_error", "error_code": "rate_limit_exceeded"},
    {"summary": "status 500: b'...'", "http_status": 500},
    {"summary": "status 400: b'not json'", "http_status": 400},
    {"summary": "status 429: b'...'"},  # upstream as pinned: no typed fields
    # A board refused for lack of funds, too: an OpenAI job behaves exactly as on main.
    {"summary": "status 429: b'...'", "http_status": 429, "error_type": "insufficient_quota", "error_code": "insufficient_quota"},
])
def test_every_image_failure_keeps_today_s_treatment(upstream, detail):
    install_metering(upstream.meter)
    upstream.design.outcome = (False, detail)
    with pytest.raises(RuntimeError, match="artisan_unknown_charge_requires_reconciliation"):
        upstream.design._generate_image("p", Path("o.png"))
    [row] = upstream.meter.read()["calls"]
    assert row == {"id": row["id"], "kind": "generated_design_board", "model": "configured_upstream_image_model",
                   "status": "reconciliation_required", "reserved_usd": 1.0, "cost_usd": None, "usage": {},
                   "response_id": None, "cost_origin": "upstream_image_budget_estimate"}


def test_a_successful_image_writes_today_s_record(upstream):
    install_metering(upstream.meter)
    assert upstream.design._generate_image("p", Path("o.png"))[0] is True
    [row] = upstream.meter.read()["calls"]
    assert row == {"id": row["id"], "kind": "generated_design_board", "model": "configured_upstream_image_model",
                   "status": "priced", "reserved_usd": 1.0, "cost_usd": 0.1, "usage": {"budget_cost_usd": 0.1},
                   "response_id": None, "cost_origin": "upstream_image_budget_estimate"}


# -- the bridge -----------------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    from blueprint_pipeline import task_object_artisan_bridge as module

    # The bridge's main() writes the scoped keys into this process's environment, and monkeypatch.delenv on an absent
    # name records nothing to undo, so the environment is put back whole afterwards: a later test in the same process
    # saw the leaked OPENAI_API_KEY (CI shard 3, 2026-11-23).
    environment = dict(os.environ)

    for name in ("artisan", "pipeline", "pipeline.llm"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    seen = {"retrieved": [], "installed": []}
    config = types.ModuleType("artisan.config")
    config.apply_runtime_config = lambda configuration, _: seen.update(configuration=dict(configuration))
    monkeypatch.setitem(sys.modules, "artisan.config", config)
    registry = types.ModuleType("pipeline.llm.model_registry")

    def validate_runtime_configuration(require_credentials):
        claude = seen["configuration"]["provider"] == "anthropic"
        model = seen["configuration"]["model"]
        return {"text_models": {"design": model, "coding": model, "review": model},
                "compaction_model": None if claude else seen["configuration"]["compaction_model"],
                "image_model": seen["configuration"]["image_model"],
                "image_tool_model": seen["configuration"]["image_tool_model"]}
    registry.validate_runtime_configuration = validate_runtime_configuration
    monkeypatch.setitem(sys.modules, "pipeline.llm.model_registry", registry)

    def client(provider):
        class Client:
            def __init__(self, **kwargs):
                seen.setdefault("clients", []).append((provider, kwargs))
                self.models = SimpleNamespace(retrieve=lambda model: seen["retrieved"].append((provider, model))
                                              or SimpleNamespace(id=model))

            def __enter__(self): return self
            def __exit__(self, *_): return False
        return Client
    import openai
    monkeypatch.setattr(openai, "OpenAI", client("openai"))
    cli = types.ModuleType("artisan.cli")

    def stop(_argv):
        raise _Stop()
    cli.main = stop
    monkeypatch.setitem(sys.modules, "artisan.cli", cli)
    monkeypatch.setattr(module, "install_metering",
                        lambda meter, *args, **kwargs: seen["installed"].append((*args, kwargs) if kwargs else args))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_CONTEXT_COMPACTION", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)

    def run(models, **extra):
        keys = {}
        for name in ("openai", "anthropic"):
            keys[name] = tmp_path / f"{name}-key"
            keys[name].write_text(f"synthetic-{name}-key-never-sent\n")
            keys[name].chmod(0o600)
        context = {"sandbox": {"blender": "/b"}, "api_key_file": str(keys["openai"]), "workspace": str(tmp_path),
                   "object_kind": "rigid_body", "model_configuration": models, "usage_file": str(tmp_path / "u.json"),
                   "maximum_cost_usd": 1.0, "maximum_requests": 3, "deadline_epoch": time.time() + 60, **extra}
        if extra.get("anthropic_api_key_file") == "issued":
            context["anthropic_api_key_file"] = str(keys["anthropic"])
        path = tmp_path / "context.json"
        path.write_text(json.dumps(context))
        monkeypatch.setenv("BLUEPRINT_ARTISAN_BRIDGE_CONTEXT_FILE", str(path))
        with pytest.raises(_Stop):
            module.main()
    yield SimpleNamespace(run=run, seen=seen)
    os.environ.clear()
    os.environ.update(environment)


def test_the_openai_bridge_checks_and_meters_exactly_as_today(bridge):
    import os

    bridge.run(dict(ARTISAN_MODELS))
    assert os.environ["OPENAI_API_KEY"] == "synthetic-openai-key-never-sent"
    assert "ANTHROPIC_API_KEY" not in os.environ and "ANTHROPIC_CONTEXT_COMPACTION" not in os.environ
    assert sorted(bridge.seen["retrieved"]) == [("openai", "gpt-6.1-sol"), ("openai", "gpt-image-2.5-sunburst")]
    assert bridge.seen["clients"] == [("openai", {"max_retries": 0, "timeout": 30})]
    assert bridge.seen["installed"] == [("openai",)]


def test_the_claude_bridge_reads_its_own_key_and_checks_each_provider_s_models(bridge):
    import os

    bridge.run(dict(ARTISAN_CLAUDE_MODELS), anthropic_api_key_file="issued",
               anthropic_us_inference_geo_multiplier=1.1)
    assert os.environ["ANTHROPIC_API_KEY"] == "synthetic-anthropic-key-never-sent"
    assert os.environ["ANTHROPIC_CONTEXT_COMPACTION"] == "0"
    # The text model was read on the host; here only the board's OpenAI models, by their explicit host.
    assert sorted(bridge.seen["retrieved"]) == [("openai", "gpt-6.1-sol"), ("openai", "gpt-image-2.5-sunburst")]
    assert bridge.seen["clients"] == [("openai", {"max_retries": 0, "timeout": 30, "base_url": "https://api.openai.com/v1"})]
    assert bridge.seen["installed"] == [("anthropic", {"us_inference_geo_multiplier": 1.1})]


def test_the_bridge_refuses_a_key_without_its_provider_or_an_unknown_provider(bridge):
    with pytest.raises(ValueError, match="artisan_text_provider_invalid"):
        bridge.run(dict(ARTISAN_MODELS), anthropic_api_key_file="issued")
    with pytest.raises(ValueError, match="artisan_text_provider_invalid"):
        bridge.run({**ARTISAN_MODELS, "provider": "google"})
    with pytest.raises(KeyError):
        bridge.run(dict(ARTISAN_CLAUDE_MODELS))


# -- the driver -----------------------------------------------------------------------------------------------

from tests import test_task_evaluation_scene_configuration_artisan_driver as driver_tests  # noqa: E402

FixtureGate = driver_tests.FixtureGate
stage_fixture = driver_tests.stage_fixture


def _claude_stage(fixture, tmp_path):
    from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver

    stage_path = Path(fixture.environment[driver.INPUT])
    stage = json.loads(stage_path.read_text())
    stage["configuration"]["authoring_models"] = dict(ARTISAN_CLAUDE_MODELS)
    stage_path.write_text(json.dumps(stage))
    key = tmp_path / "anthropic-key"
    key.write_text("synthetic-anthropic-key-never-sent")
    key.chmod(0o600)
    return {**fixture.environment, driver.ANTHROPIC_KEY_FILE_ENV: str(key)}, key


def test_a_claude_attempt_without_its_key_or_sdk_stops_before_any_reservation(stage_fixture, tmp_path):
    from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver

    environment, _ = _claude_stage(stage_fixture, tmp_path)
    with pytest.raises(Exception, match="artisan_anthropic_credential_missing"):
        driver.execute_artisan_component(environment={k: v for k, v in environment.items()
                                                      if k != driver.ANTHROPIC_KEY_FILE_ENV},
                                         runner=stage_fixture.runner, cost_gate_factory=stage_fixture.gate)
    def no_sdk(argv, **kwargs):
        if argv[1:3] == ["-S", "-c"]:
            return SimpleNamespace(returncode=1)
        pytest.fail("no controller run without the SDK")
    with pytest.raises(Exception, match="artisan_anthropic_controller_import_failed"):
        driver.execute_artisan_component(environment=environment, runner=no_sdk, cost_gate_factory=stage_fixture.gate)
    assert "reserve_original_pool" not in stage_fixture.events




def test_the_openai_driver_attempt_is_today_s(stage_fixture):
    from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver

    driver.execute_artisan_component(environment=stage_fixture.environment, runner=stage_fixture.runner,
                                     cost_gate_factory=stage_fixture.gate)
    [(argv, kwargs)] = stage_fixture.calls
    assert argv[argv.index("--provider") + 1] == "openai"
    context = json.loads((stage_fixture.output / "artisan_runtime" / "bridge_context.v1.json").read_text())
    assert "anthropic_api_key_file" not in context and context["model_configuration"] == ARTISAN_MODELS
    assert stage_fixture.events.count("reserve_original_pool") == 1


def test_a_claude_attempt_probes_the_sdk_holds_one_reservation_and_redacts_its_key(stage_fixture, tmp_path):
    from blueprint_pipeline import task_evaluation_scene_configuration_artisan_driver as driver

    environment, key = _claude_stage(stage_fixture, tmp_path)
    probes = []

    def runner(argv, **kwargs):
        if argv[1:3] == ["-S", "-c"]:
            probes.append(argv[-1])
            return SimpleNamespace(returncode=0)
        finished = stage_fixture.runner(argv, **kwargs)
        stage_fixture.calls.append((argv, kwargs))
        return SimpleNamespace(returncode=finished.returncode, stdout="echo " + key.read_text(), stderr="")
    driver.execute_artisan_component(environment=environment, runner=runner, cost_gate_factory=stage_fixture.gate)
    assert probes == [driver.ANTHROPIC_CONTROLLER_IMPORT_PROBE]
    argv = stage_fixture.calls[-1][0]
    assert argv[argv.index("--provider") + 1] == "anthropic"
    runtime = stage_fixture.output / "artisan_runtime"
    context = json.loads((runtime / "bridge_context.v1.json").read_text())
    assert context["anthropic_api_key_file"] == str(key.resolve())
    assert context["anthropic_us_inference_geo_multiplier"] == 1.1
    assert stage_fixture.events.count("reserve_original_pool") == 1  # the one (OpenAI) reservation
    assert not (runtime / "official_anthropic_cost").exists()
    [log] = runtime.glob("controller-*.log")
    assert key.read_text() not in log.read_text() and "[redacted]" in log.read_text()


def test_one_meter_holds_both_providers_to_their_sum_under_the_one_cap(tmp_path):
    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=1.0, maximum_requests=10, deadline_epoch=time.time() + 600)
    meter.after(meter.before("messages", "claude-opus-5-5"), 0.6)
    meter.after(meter.before("images", "gpt-image-2.5-sunburst"), 0.3)
    assert meter.remaining() == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="artisan_reported_cost_exceeded_stage_authority"):
        meter.after(meter.before("messages", "claude-opus-5-5"), 0.2)
    with pytest.raises(RuntimeError, match="artisan_shared_stage_authority_exhausted"):
        meter.before("images", "gpt-image-2.5-sunburst")


def test_an_anthropic_refusal_for_lack_of_funds_is_marked_on_its_row(tmp_path):
    meter = Meter(tmp_path / "usage.json", maximum_cost_usd=1.0, maximum_requests=10, deadline_epoch=time.time() + 600)
    priced = meter.before("messages", "claude-opus-5-5")
    meter.after(priced, 0.2)
    meter.mark_no_funds(priced)  # a charged row is never relabelled
    refused = meter.before("messages", "claude-opus-5-5")
    meter.failed(refused, SimpleNamespace(status_code=402))
    meter.mark_no_funds(refused)
    rows = meter.read()["calls"]
    assert [row["cost_origin"] for row in rows] == ["upstream_priced_usage", "upstream_rejected_request_no_funds"]


def test_a_us_served_call_that_uses_its_whole_bound_stays_inside_the_cap(upstream):
    # The bound carries the uplift pricing applies, so a full-output call served in the US region
    # is charged no more than the cap (the reviewer's probe recorded $1.0956 against $1 without it).
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    allowed, bound = metering._anthropic_bound(upstream.meter, RATES, 1000, 64000, 1.1)
    assert bound <= upstream.meter.cap
    upstream.llm.replies = [SimpleNamespace(id="m", model="claude-opus-5-5", stop_reason="end_turn", usage=SimpleNamespace(
        input_tokens=1000, output_tokens=allowed, iterations=None, inference_geo="us"))]
    asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    [(_, sent)] = [row for row in upstream.seen if row[0] == "create"]
    [row] = upstream.meter.read()["calls"]
    assert sent["max_tokens"] == allowed and row["cost_usd"] <= upstream.meter.cap
    # Without the uplift the same call could be charged above the cap.
    plain = metering._anthropic_bound(
        Meter(upstream.meter.path.with_name("other.json"), maximum_cost_usd=1.0, maximum_requests=2,
              deadline_epoch=time.time() + 60), RATES, 1000, 64000)[0]
    assert (1000 * 4 + plain * 20) * 1.1 / 1_000_000 > 1.0


def test_every_no_funds_answer_is_typed_including_a_billing_error_without_a_402():
    for fields in ((402, None, None, None), (400, "billing_error", None, None),
                   (429, "rate_limit_error", "enforced_spend_limit_reached", None),
                   (400, "invalid_request_error", None, "You have reached your specified API usage limits. x")):
        assert metering.anthropic_no_funds(*fields), fields
    for fields in ((429, "rate_limit_error", None, None), (400, "invalid_request_error", None, "max_tokens"),
                   (529, "overloaded_error", None, None)):
        assert not metering.anthropic_no_funds(*fields), fields


def test_the_input_side_of_the_bound_carries_the_uplift_too(upstream):
    # 190,000 input tokens fit a $1 cap at list price ($0.95 at the 5-minute write rate) but not with
    # the US uplift ($1.045): the call is refused before it is sent.
    with pytest.raises(RuntimeError, match="artisan_shared_stage_authority_exhausted"):
        metering._anthropic_bound(upstream.meter, RATES, 190_000, 64000, 1.1)
    assert metering._anthropic_bound(upstream.meter, RATES, 190_000, 64000) == (2_500, pytest.approx(1.0))


@pytest.mark.parametrize("geo, allowed, cost", [
    ("us", 45_204, (1000 * 4 + 100 * 20) / 1_000_000 * 1.1),  # pinned: bounded and priced with the uplift
    (None, 49_750, _UNPINNED_BOUND),  # not sent: no uplift in the bound, and a US answer is counted at that bound
])
def test_every_messages_request_is_pinned_to_the_us_and_only_the_pin_brings_the_uplift(
        upstream, monkeypatch, geo, allowed, cost):
    monkeypatch.setattr(metering, "INFERENCE_GEO", geo)
    install_metering(upstream.meter, "anthropic", us_inference_geo_multiplier=1.1)
    upstream.llm.replies = [_response(inference_geo="us")]
    asyncio.run(upstream.llm()._create_message(dict(PAYLOAD)))
    (_, counted), (_, sent) = upstream.seen
    assert "inference_geo" not in counted and sent.get("inference_geo") == geo and sent["max_tokens"] == allowed
    [row] = upstream.meter.read()["calls"]
    assert row["cost_usd"] == pytest.approx(cost)
