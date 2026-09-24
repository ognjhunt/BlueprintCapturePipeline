"""No-network SDK authoring with signed Anthropic per-request admission."""
import base64
import io
import json

from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner, SQLiteSession, StopAtTools
from PIL import Image
from pydantic import BaseModel, ConfigDict
import pytest

from blueprint_pipeline.claude_opus_authoring_invoker import (
    ClaudeAuthoringBlocked, ClaudeAuthoringConfig, ClaudeOpusAuthoringInvoker,
    _MODEL_INPUT_CONTEXT_TOKENS, _OUTPUT_RATE, _INPUT_RATE, _US_GEO_MULTIPLIER,
    _MAX_IMAGE_TOKENS, _payload,
)
from blueprint_pipeline.claude_opus_sdk_bridge import ClaudeSDKMessageClient
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_claude_model import ClaudeMessagesModel
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import AgentsSDKAgentSpec
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


SHA = "sha256:" + "a" * 64


class CandidateReady(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: str


def _source_image():
    output = io.BytesIO()
    Image.new("RGB", (1920, 1080), "tan").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def _authority(run_id, _digest):
    return {"run_id": run_id, "allowed_providers": ["anthropic"],
            "private_provider_processing_allowed": True, "provider_training_allowed": False,
            "authority_digest": SHA, "provider_terms_digest": SHA}


def _tool(name, calls):
    async def invoke(context, arguments):
        calls.append((name, context.tool_call_id, json.loads(arguments)))
        return '{"status":"completed"}'
    return FunctionTool(name=name, description="Local asset operation: " + name,
        params_json_schema={"type": "object", "properties": {"part": {"type": "string"}},
                            "required": ["part"], "additionalProperties": False},
        on_invoke_tool=invoke, needs_approval=False)


def _invoker(monkeypatch, tmp_path, send):
    key_file = tmp_path / "scoped-anthropic-key"
    key_file.write_text("test-only-placeholder")
    key_file.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key_file))
    audit = InferenceReservationAudit(run_root=tmp_path / "ledger", run_id="future-scene")
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=5,
        allow_live_invocation=True), audit=audit, verify_authority=_authority, send=send)
    return invoker, audit


def test_exact_sdk_tool_image_sequence_and_reviews_fit_after_completion(monkeypatch, tmp_path):
    sent, local_calls = [], []
    names = ["observe_object", "build_cad", "render_candidate"]

    def send(payload, key):
        assert key == "test-only-placeholder"
        sent.append(payload)
        index = len(sent)
        if index <= 3:
            assert payload["thinking"] == {"type": "adaptive", "display": "omitted"}
            assert payload["output_config"]["format"]["schema"]["title"] == "CandidateReady"
            assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
            assert payload["messages"][0]["content"][1]["type"] == "image"
            if index > 1:
                assert payload["messages"][1]["content"][0]["signature"] == "signed-1"
            return {"id": f"msg_{index}", "model": "claude-opus-5-5", "stop_reason": "tool_use",
                "content": [{"type": "thinking", "thinking": "", "signature": f"signed-{index}"},
                            {"type": "tool_use", "id": f"tool_{index}", "name": names[index - 1],
                             "input": {"part": "middle_drawer"}}],
                "usage": {"input_tokens": 1000 * index, "output_tokens": 100}}
        assert payload["messages"][0]["content"][1]["type"] == "image"
        return {"id": f"msg_{index}", "model": "claude-opus-5-5", "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"decision":"accepted"}'}],
            "usage": {"input_tokens": 1500, "output_tokens": 100}}

    invoker, audit = _invoker(monkeypatch, tmp_path, send)
    client = ClaudeSDKMessageClient(invoker=invoker, object_id="middle_drawer")
    agent = Agent(name="Future-scene Claude drawer author", model=ClaudeMessagesModel(client=client),
        tools=[_tool(name, local_calls) for name in names], output_type=CandidateReady,
        tool_use_behavior=StopAtTools(stop_at_tool_names=["render_candidate"]),
        model_settings=ModelSettings(max_tokens=12_000, store=False, parallel_tool_calls=False))
    db = tmp_path / "conversation.sqlite"
    session = SQLiteSession("middle_drawer", db_path=db)
    image = _source_image()
    try:
        Runner.run_sync(agent, [{"role": "user", "content": [
            {"type": "input_text", "text": "Observe, build and render the middle drawer."},
            {"type": "input_image", "image_url": image}]}], session=session,
            run_config=RunConfig(tracing_disabled=True), max_turns=5)
    finally:
        session.close()
    assert [name for name, _, _ in local_calls] == names
    assert db.is_file()

    # The independent physical and visual reviews still use the admitted
    # provider-specific invoker, sharing the *same* $7 ledger.
    for capability in ("physical_property_review", "appearance_review"):
        spec = AgentsSDKAgentSpec(run_id="future-scene", capability=capability,
            name=capability, instructions="Review independently.", model="claude-opus-5-5",
            max_turns=1, max_input_tokens=80_000, max_output_tokens=12_000,
            reasoning_effort="medium", output_type=Review)
        review_input = [{"role": "user", "content": [
            {"type": "input_text", "text": "Review the candidate against original frame."},
            {"type": "input_image", "image_url": image}]}]
        _, submitted_bound = _payload(spec, review_input)
        assert submitted_bound < 80_000
        assert invoker.invoke(spec, review_input).output.decision == "accepted"

    assert len(sent) == 5
    manifest = audit.manifest()
    assert manifest["reservation_count"] == 5
    assert manifest["in_flight_unknown_count"] == 0
    assert all(row["status"] == "completed" for row in manifest["reservations"])
    reservations = [json.loads(path.read_text()) for path in
                    (tmp_path / "ledger/inference_reservations/reserved").glob("*.json")]
    assert len(reservations) == 5
    assert all(row["provider"] == "anthropic" and row["model"] == "claude-opus-5-5"
               and row["input_token_ceiling"] == _MODEL_INPUT_CONTEXT_TOKENS
               and row["authority_digest"] == SHA and row["provider_terms_digest"] == SHA
               and row["inference_reservation_digest"] == canonical_digest(
                   row, digest_field="inference_reservation_digest")
               for row in reservations)
    full_window_quote = (_MODEL_INPUT_CONTEXT_TOKENS * _INPUT_RATE + 12_000 * _OUTPUT_RATE) * _US_GEO_MULTIPLIER / 1_000_000
    assert all(row["projected_max_cost_usd"] == pytest.approx(full_window_quote) for row in reservations)
    assert manifest["reserved_max_cost_usd"] < 7
    assert full_window_quote * 5 > 7  # Simultaneous or unresolved turns cannot fit.

    # Inspect the exact translated wire requests: source image and all tool
    # schemas are charged, but the base64 transport expansion is not tokenized
    # as text. The provider's full 1M window is still the actual reservation.
    for payload in sent[:3]:
        image_count = 0
        clone = json.loads(json.dumps(payload))
        for message in clone["messages"]:
            for block in message["content"]:
                if block["type"] == "image":
                    image_count += 1
                    block["source"]["data"] = ""
                if block["type"] == "tool_result" and isinstance(block["content"], list):
                    for part in block["content"]:
                        if part["type"] == "image":
                            image_count += 1
                            part["source"]["data"] = ""
        text_bytes = len(json.dumps(clone, ensure_ascii=True).encode())
        submitted_estimate = text_bytes + image_count * _MAX_IMAGE_TOKENS + 4096
        assert image_count == 1 and submitted_estimate < 80_000


def test_unknown_first_turn_keeps_full_quote_and_blocks_next(monkeypatch, tmp_path):
    invoker, audit = _invoker(monkeypatch, tmp_path,
        lambda *_: (_ for _ in ()).throw(TimeoutError("uncertain provider outcome")))
    payload = {"model": "claude-opus-5-5", "max_tokens": 12_000, "inference_geo": "us",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Build drawer"}]}],
        "tools": [], "output_config": {"effort": "medium"}}
    with pytest.raises(ClaudeAuthoringBlocked, match="provider_outcome_unknown"):
        invoker.invoke_tool_turn(capability="middle_drawer_author_turn_001", payload=payload)
    assert audit.manifest()["in_flight_unknown_count"] == 1
    with pytest.raises(ClaudeAuthoringBlocked, match="spend_cap_exhausted"):
        invoker.invoke_tool_turn(capability="middle_drawer_author_turn_002", payload=payload)
    assert audit.manifest()["reservation_count"] == 1
