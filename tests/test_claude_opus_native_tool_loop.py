"""Native Opus tool history preserves signed thinking and never replays uncertainty."""
import json

from agents import FunctionTool
from pydantic import BaseModel
import pytest

from blueprint_pipeline.claude_opus_authoring_invoker import (
    ClaudeAuthoringBlocked, ClaudeAuthoringConfig, ClaudeOpusAuthoringInvoker, MODEL,
)
from blueprint_pipeline.claude_opus_native_tool_loop import ClaudeNativeToolLoop
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit


SHA = "sha256:" + "a" * 64


class Final(BaseModel):
    summary: str


def _authority(run_id, _digest):
    return {"run_id": run_id, "allowed_providers": ["anthropic"],
        "private_provider_processing_allowed": True, "provider_training_allowed": False,
        "authority_digest": SHA, "provider_terms_digest": SHA}


def _invoker(monkeypatch, tmp_path, send):
    key = tmp_path / "key"
    key.write_text("test-only-placeholder")
    key.chmod(0o600)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(key))
    return ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id="future-scene", maximum_cost_usd=7, maximum_calls=4,
        allow_live_invocation=True),
        audit=InferenceReservationAudit(run_root=tmp_path / "ledger", run_id="future-scene"),
        verify_authority=_authority, send=send)


def _tool(handler):
    return FunctionTool(name="build_cad", description="Build a bounded CAD program",
        params_json_schema={"type": "object", "properties": {"program": {"type": "string"}},
                            "required": ["program"], "additionalProperties": False},
        on_invoke_tool=handler, needs_approval=False)


def test_signed_thinking_replayed_exactly_after_persisted_tool_result(monkeypatch, tmp_path):
    sent, tool_calls = [], []
    thinking = {"type": "thinking", "thinking": "", "signature": "signed-provider-bytes"}

    def send(payload, _key):
        sent.append(payload)
        if len(sent) == 1:
            return {"id": "msg_one", "model": MODEL, "stop_reason": "tool_use",
                "content": [thinking, {"type": "tool_use", "id": "tool_one",
                                       "name": "build_cad", "input": {"program": "box()"}}],
                "usage": {"input_tokens": 200, "output_tokens": 100}}
        assert payload["messages"][1]["content"][0] == thinking
        assert payload["messages"][2]["content"][0]["tool_use_id"] == "tool_one"
        return {"id": "msg_two", "model": MODEL, "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"summary":"candidate built"}'}],
            "usage": {"input_tokens": 300, "output_tokens": 60}}

    async def handler(context, arguments):
        tool_calls.append((context.tool_call_id, json.loads(arguments)))
        return '{"status":"built"}'

    invoker = _invoker(monkeypatch, tmp_path, send)
    loop = ClaudeNativeToolLoop(root=tmp_path / "transcript", run_id="future-scene",
        object_id="middle_drawer", invoker=invoker, system="Build the CAD asset.",
        tools=[_tool(handler)])
    stopped = loop.run(initial_content="Build one drawer", final_type=Final,
                       stop_after_tool="build_cad")
    assert stopped["status"] == "stopped_after_tool"
    assert len(sent) == 1 and len(tool_calls) == 1
    response_path = tmp_path / "transcript/turn-00-response.json"
    stored = json.loads(response_path.read_text())
    tampered = json.loads(response_path.read_text())
    tampered["content"][0]["signature"] = "altered"
    response_path.write_text(json.dumps(tampered))
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_transcript_receipt_missing"):
        loop.run(initial_content="Build one drawer", final_type=Final)
    response_path.write_text(json.dumps(stored))
    resumed = ClaudeNativeToolLoop(root=tmp_path / "transcript", run_id="future-scene",
        object_id="middle_drawer", invoker=invoker, system="Build the CAD asset.",
        tools=[_tool(handler)])
    final = resumed.run(initial_content="Build one drawer", final_type=Final)
    assert final.summary == "candidate built"
    assert len(sent) == 2 and len(tool_calls) == 1
    assert stored["content"][0] == thinking
    assert invoker.audit.manifest()["reservation_count"] == 2


def test_interrupted_tool_is_not_reexecuted(monkeypatch, tmp_path):
    sends = []

    def send(payload, _key):
        sends.append(payload)
        return {"id": "msg_one", "model": MODEL, "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "", "signature": "signed"},
                        {"type": "tool_use", "id": "tool_one", "name": "build_cad",
                         "input": {"program": "box()"}}],
            "usage": {"input_tokens": 100, "output_tokens": 50}}

    async def handler(_context, _arguments):
        raise TimeoutError("uncertain local tool outcome")

    loop = ClaudeNativeToolLoop(root=tmp_path / "transcript", run_id="future-scene",
        object_id="middle_drawer", invoker=_invoker(monkeypatch, tmp_path, send),
        system="Build the CAD asset.", tools=[_tool(handler)])
    with pytest.raises(TimeoutError):
        loop.run(initial_content="Build one drawer", final_type=Final)
    with pytest.raises(ClaudeAuthoringBlocked, match="claude_tool_outcome_unknown"):
        loop.run(initial_content="Build one drawer", final_type=Final)
    assert len(sends) == 1


def test_unstarted_tool_can_resume_from_paid_response(monkeypatch, tmp_path):
    sends, executions = [], []

    def send(payload, _key):
        sends.append(payload)
        if len(sends) == 1:
            return {"id": "msg_one", "model": MODEL, "stop_reason": "tool_use",
                "content": [{"type": "thinking", "thinking": "", "signature": "signed"},
                            {"type": "tool_use", "id": "tool_one", "name": "build_cad",
                             "input": {"program": "box()"}}],
                "usage": {"input_tokens": 100, "output_tokens": 50}}
        return {"id": "msg_two", "model": MODEL, "stop_reason": "end_turn",
            "content": [{"type": "text", "text": '{"summary":"done"}'}],
            "usage": {"input_tokens": 120, "output_tokens": 20}}

    async def handler(_context, _arguments):
        executions.append(1)
        return '{"status":"built"}'

    invoker = _invoker(monkeypatch, tmp_path, send)
    first = ClaudeNativeToolLoop(root=tmp_path / "transcript", run_id="future-scene",
        object_id="middle_drawer", invoker=invoker, system="Build.", tools=[_tool(handler)])
    original = first._tool_result

    def interrupt_before_tool(*_args, **_kwargs):
        raise TimeoutError("before local tool")

    first._tool_result = interrupt_before_tool
    with pytest.raises(TimeoutError):
        first.run(initial_content="Build", final_type=Final)
    second = ClaudeNativeToolLoop(root=tmp_path / "transcript", run_id="future-scene",
        object_id="middle_drawer", invoker=invoker, system="Build.", tools=[_tool(handler)])
    assert second.verify_existing("Build") is True
    assert second.run(initial_content="Build", final_type=Final).summary == "done"
    assert len(sends) == 2 and executions == [1]
    first._tool_result = original
