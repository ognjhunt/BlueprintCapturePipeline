"""No-network contract tests for the optional local Claude authoring model."""
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import threading

from PIL import Image
from pydantic import BaseModel
import pytest

from blueprint_pipeline.claude_opus_authoring_invoker import (
    ClaudeAuthoringBlocked, ClaudeAuthoringConfig, ClaudeOpusAuthoringInvoker, MODEL,
)
from blueprint_pipeline import astra_cad_skill_runtime as cad_runtime
from blueprint_pipeline import task_object_astra_authoring as author
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


SHA = "sha256:" + "a" * 64


class Output(BaseModel):
    content: str


def _spec(*, max_input_tokens=80_000, max_output_tokens=12_000):
    return AgentsSDKAgentSpec(run_id="future-scene", capability="cad_brief",
        name="Claude CAD", instructions="Create a CAD brief.", model=MODEL,
        max_turns=1, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens,
        reasoning_effort="high", output_type=Output)


def _authority(run_id, _input_digest):
    return {"run_id": run_id, "allowed_providers": ["anthropic"],
        "private_provider_processing_allowed": True, "provider_training_allowed": False,
        "authority_digest": SHA, "provider_terms_digest": SHA}


def _image():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "brown").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def _key(monkeypatch, tmp_path):
    path = tmp_path / "anthropic-key"
    path.write_text("test-only-placeholder", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(path))


def test_live_call_requires_authority_before_key_or_network(tmp_path):
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=4), audit=audit,
        send=lambda *_: pytest.fail("network called"))
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_live_authority_missing"):
        invoker.invoke(_spec(), "Draft a brief")
    assert audit.manifest()["reservation_count"] == 0


def test_signed_permission_without_scoped_key_refuses_before_network(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY_FILE", raising=False)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=4,
        allow_live_invocation=True), audit=audit,
        verify_authority=_authority,
        send=lambda *_: pytest.fail("network called"))
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_key_file_missing"):
        invoker.invoke(_spec(), "Draft a brief")
    assert audit.manifest()["reservation_count"] == 0


def test_bounded_opus_call_retains_provider_receipts(monkeypatch, tmp_path):
    _key(monkeypatch, tmp_path)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    observed = []

    def send(payload, key):
        assert key == "test-only-placeholder"
        observed.append(payload)
        return {"id": "msg_test", "model": MODEL, "stop_reason": "end_turn",
            "content": [{"type": "text", "text": json.dumps({"content": "drawer brief"})}],
            "usage": {"input_tokens": 1000, "output_tokens": 100}}

    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=4,
        allow_live_invocation=True), audit=audit, verify_authority=_authority, send=send)
    result = invoker.invoke(_spec(), [{"role": "user", "content": [
        {"type": "input_text", "text": "Describe this pedestal"},
        {"type": "input_image", "image_url": _image()}]}])
    assert result.output.content == "drawer brief"
    assert (result.provider, result.model) == ("anthropic", MODEL)
    assert observed[0]["max_tokens"] == 12_000
    assert "cache_control" not in json.dumps(observed[0])
    assert observed[0]["messages"][0]["content"][1]["type"] == "image"
    manifest = audit.manifest()
    assert manifest["reservation_count"] == 1
    assert manifest["in_flight_unknown_count"] == 0
    assert manifest["reservations"][0]["status"] == "completed"
    # The admission reserves the full published context window, even though
    # this particular image and prompt use far fewer tokens.
    reserved = json.loads(next((tmp_path / "inference_reservations/reserved").glob("*.json")).read_text())
    assert reserved["input_token_ceiling"] == 1_000_000
    assert reserved["projected_max_cost_usd"] == pytest.approx(
        (1_000_000 * 4 + 12_000 * 20) * 1.1 / 1_000_000)
    assert result.cost_usd == pytest.approx((1000 * 4 + 100 * 20) * 1.1 / 1_000_000)


def test_cap_refuses_before_dispatch_and_reservation(monkeypatch, tmp_path):
    _key(monkeypatch, tmp_path)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=0.01, maximum_calls=1,
        allow_live_invocation=True), audit=audit, verify_authority=_authority,
        send=lambda *_: pytest.fail("network called"))
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_spend_cap_exhausted"):
        invoker.invoke(_spec(), "Draft a brief")
    assert audit.manifest()["reservation_count"] == 0


def test_unsigned_provider_permission_refused(monkeypatch, tmp_path):
    _key(monkeypatch, tmp_path)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=1,
        allow_live_invocation=True), audit=audit,
        verify_authority=lambda run, digest: {**_authority(run, digest),
            "allowed_providers": ["vast", "openai"]},
        send=lambda *_: pytest.fail("network called"))
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_provider_authority_invalid"):
        invoker.invoke(_spec(), "Draft a brief")
    assert audit.manifest()["reservation_count"] == 0


def test_unknown_transport_holds_reservation_and_cannot_replay(monkeypatch, tmp_path):
    _key(monkeypatch, tmp_path)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    calls = []

    def send(*_args):
        calls.append(1)
        raise TimeoutError("simulated")

    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=2,
        allow_live_invocation=True), audit=audit, verify_authority=_authority, send=send)
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_provider_outcome_unknown"):
        invoker.invoke(_spec(), "Draft a brief")
    assert audit.manifest()["in_flight_unknown_count"] == 1
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_spend_cap_exhausted"):
        invoker.invoke(_spec(), "Draft a brief")
    assert len(calls) == 1


def test_parallel_calls_cannot_oversubscribe_attempt_cap(monkeypatch, tmp_path):
    _key(monkeypatch, tmp_path)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    entered, release = threading.Event(), threading.Event()
    dispatched = []

    def send(_payload, _key_value):
        dispatched.append(1)
        entered.set()
        assert release.wait(5)
        return {"id": "msg_parallel", "model": MODEL, "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"content":"done"}'}],
            "usage": {"input_tokens": 100, "output_tokens": 20}}

    def invoker(capability):
        return ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
            run_id="future-scene", maximum_cost_usd=7, maximum_calls=3,
            allow_live_invocation=True), audit=audit, verify_authority=_authority,
            send=send).invoke(replace(_spec(), capability=capability), "Draft")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(invoker, "cad_first")
        assert entered.wait(5)
        second = pool.submit(invoker, "cad_second")
        with pytest.raises(ClaudeAuthoringBlocked, match="claude_spend_cap_exhausted"):
            second.result(timeout=5)
        release.set()
        assert first.result(timeout=5).output.content == "done"
    assert dispatched == [1]
    assert audit.manifest()["reservation_count"] == 1


def test_invalid_json_retains_usage_receipt(monkeypatch, tmp_path):
    _key(monkeypatch, tmp_path)
    audit = InferenceReservationAudit(run_root=tmp_path, run_id="future-scene")
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=1,
        allow_live_invocation=True), audit=audit, verify_authority=_authority,
        send=lambda *_: {"id": "msg_bad", "model": MODEL, "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "not json"}],
            "usage": {"input_tokens": 20, "output_tokens": 10}})
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_output_invalid"):
        invoker.invoke(_spec(), "Draft a brief")
    assert audit.manifest()["in_flight_unknown_count"] == 0


def test_existing_asset_authoring_vision_uses_selected_claude_model(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "brown").save(image)
    frame = author.SourceFrame(path=str(image), sha256=author.file_record(image)["sha256"],
                               role="observed_source", description="Cabinet front")
    request = type("Request", (), {"run_id": "future-scene", "object_id": "drawer",
        "request_digest": SHA, "generated_specification": None})()
    brief = author.VisualBrief(object_identity="drawer", observed_parts=["front"],
        appearance_requirements=["wood"], unknown_regions=["interior"],
        cad_brief_markdown="Exact nominal dimensions", proposed_material="wood",
        proposed_appearance="opaque")

    class FakeClaude:
        model = MODEL

        def invoke(self, spec, _input):
            assert spec.model == MODEL
            assert spec.cache_policy is None
            return type("Result", (), {"output": brief, "model": MODEL,
                "provider": "anthropic", "usage": {}, "cost_usd": 0.1,
                "cost_status": "fixture"})()

    got = author.invoke_vision(FakeClaude(), request, capability="source_analysis",
        prompt="Inspect", output_type=author.VisualBrief, frames=[frame], root=tmp_path,
        cache_prefix="CAD skill instructions")
    assert got == brief
    retained = json.loads((tmp_path / "source_analysis.json").read_text())
    assert retained["provider"] == "anthropic" and retained["model"] == MODEL


def test_existing_cad_bridge_uses_claude_without_openai_cache(tmp_path):
    class FakeClaude:
        model = MODEL

        def invoke(self, spec, _input):
            assert spec.model == MODEL and spec.cache_policy is None
            return type("Result", (), {"output": cad_runtime._TextOutput(content="complete CAD"),
                "provider": "anthropic", "model": MODEL, "usage": {},
                "cost_usd": 0.1, "cost_status": "fixture", "trace_id": "msg_fixture",
                "sdk_version": "fixture"})()

    bridge = cad_runtime._SDKChatBridge(FakeClaude(), tmp_path, "middle drawer",
        "future-scene", 80_000, 12_000, 1)
    result = bridge.create(messages=[{"role": "user", "content": "plan CAD"}])
    assert result.choices[0].message.content == "complete CAD"
    calls = json.loads((tmp_path / "invocations.json").read_text())
    assert calls[0]["provider"] == "anthropic" and calls[0]["model"] == MODEL
