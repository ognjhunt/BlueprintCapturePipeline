"""Hermetic contract tests for the opt-in Claude Messages SDK model adapter."""
import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from agents import Agent, FunctionTool, ModelSettings, RunConfig, Runner, SQLiteSession
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem
from pydantic import BaseModel, ConfigDict

from blueprint_pipeline.task_object_claude_model import ClaudeMessagesModel, ClaudeModelBoundaryError


def value(**kwargs):
    return SimpleNamespace(**kwargs)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def response(*blocks, stop_reason="end_turn"):
    return value(id="msg_fixture", content=list(blocks), stop_reason=stop_reason,
                 usage=value(input_tokens=100, cache_creation_input_tokens=10,
                             cache_read_input_tokens=20, output_tokens=30))


def call(model, input, *, tools=(), output_schema=None):
    return asyncio.run(model.get_response(
        system_instructions="Keep observed geometry distinct from estimates.", input=input,
        model_settings=ModelSettings(max_tokens=1200, parallel_tool_calls=False, store=False),
        tools=list(tools), output_schema=output_schema, handoffs=[], tracing=None,
        previous_response_id=None, conversation_id=None, prompt=None))


def test_images_and_structured_output_are_preserved_without_external_fetch():
    schema = value(json_schema=lambda: {"type": "object", "properties": {"summary": {"type": "string"}},
                                        "required": ["summary"], "additionalProperties": False})
    messages = FakeMessages(response(value(type="text", text='{"summary":"ready"}')))
    model = ClaudeMessagesModel(client=value(messages=messages))
    png = base64.b64encode(b"fixture-png").decode()
    result = call(model, [{"role": "user", "content": [
        {"type": "input_text", "text": "Original frame"},
        {"type": "input_image", "image_url": "data:image/png;base64," + png}]}], output_schema=schema)
    sent = messages.calls[0]
    assert sent["model"] == "claude-opus-5-5" and sent["max_tokens"] == 1200
    assert sent["messages"][0]["content"][1]["source"] == {
        "type": "base64", "media_type": "image/png", "data": png}
    assert sent["output_config"]["format"]["schema"] == schema.json_schema()
    assert result.output[0].content[0].text == '{"summary":"ready"}'
    assert result.usage.input_tokens == 130  # Conservative full-price cache accounting.
    assert result.usage.output_tokens == 30


def test_tool_cycle_preserves_call_identity_and_json_arguments():
    messages = FakeMessages(response(value(type="thinking", thinking="", signature="signed-opaque"),
                                     value(type="tool_use", id="toolu_fixture", name="build_cad",
                                          input={"program": "make_box()"}), stop_reason="tool_use"))
    model = ClaudeMessagesModel(client=value(messages=messages))
    tool = value(name="build_cad", description="Build CAD", params_json_schema={"type": "object"})
    first = call(model, "Build the drawer", tools=[tool])
    assert isinstance(first.output[0], ResponseReasoningItem)
    item = first.output[1]
    assert isinstance(item, ResponseFunctionToolCall)
    assert item.call_id == "toolu_fixture" and json.loads(item.arguments) == {"program": "make_box()"}
    assert messages.calls[0]["tools"][0]["strict"] is True
    second = call(model, [
        {"role": "user", "content": "Build the drawer"},
        first.output[0].model_dump(mode="json"),
        item.model_dump(mode="json"),
        {"type": "function_call_output", "call_id": item.call_id, "output": "CAD compiled"},
        {"role": "user", "content": "Review the result"},
    ], tools=[tool])
    assert second.response_id == "msg_fixture"
    sent = messages.calls[1]["messages"]
    assert sent[1]["role"] == "assistant" and sent[1]["content"][0] == {
        "type": "thinking", "thinking": "", "signature": "signed-opaque"}
    assert sent[1]["content"][1]["id"] == "toolu_fixture"
    assert sent[2]["role"] == "user" and sent[2]["content"][0] == {
        "type": "tool_result", "tool_use_id": "toolu_fixture", "content": "CAD compiled"}
    assert sent[3]["role"] == "user" and sent[3]["content"] == [{"type": "text", "text": "Review the result"}]
    assert messages.calls[0]["thinking"] == {"type": "adaptive", "display": "omitted"}


@pytest.mark.parametrize("bad", [
    [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/frame.png"}]}],
    [{"type": "function_call_output", "call_id": "missing", "output": "orphan"}],
    [{"type": "reasoning", "encrypted_content": "opaque"}],
])
def test_unsupported_or_uncounted_context_fails_before_provider_call(bad):
    messages = FakeMessages(response(value(type="text", text="unused")))
    model = ClaudeMessagesModel(client=value(messages=messages))
    with pytest.raises(ClaudeModelBoundaryError):
        call(model, bad)
    assert messages.calls == []


def test_server_continuation_and_truncation_fail_closed():
    messages = FakeMessages(response(value(type="text", text="partial"), stop_reason="max_tokens"))
    model = ClaudeMessagesModel(client=value(messages=messages))
    with pytest.raises(ClaudeModelBoundaryError, match="incomplete"):
        call(model, "Build")
    assert len(messages.calls) == 1
    async def continuation():
        return await model.get_response(
            system_instructions=None, input="Build", model_settings=ModelSettings(max_tokens=100),
            tools=[], output_schema=None, handoffs=[], tracing=None,
            previous_response_id="server-id", conversation_id=None, prompt=None)
    with pytest.raises(ClaudeModelBoundaryError, match="server_context"):
        asyncio.run(continuation())
    assert len(messages.calls) == 1


def test_message_response_maps_to_agents_sdk_output():
    messages = FakeMessages(response(value(type="text", text="Ready")))
    result = call(ClaudeMessagesModel(client=value(messages=messages)), "Task")
    assert isinstance(result.output[0], ResponseOutputMessage)
    assert result.output[0].content[0].text == "Ready"


def test_thinking_only_answer_is_not_counted_as_completed_candidate():
    messages = FakeMessages(response(value(type="thinking", thinking="", signature="opaque")))
    with pytest.raises(ClaudeModelBoundaryError, match="response_shape_invalid"):
        call(ClaudeMessagesModel(client=value(messages=messages)), "Task")


def test_real_agents_sdk_round_trips_signed_thinking_tools_and_sqlite(tmp_path):
    class CandidateReady(BaseModel):
        model_config = ConfigDict(extra="forbid")
        summary: str

    messages = FakeMessages(response(value(type="thinking", thinking="", signature="signed-first"),
                                     value(type="tool_use", id="toolu_1", name="build_cad",
                                           input={"program": "fixture"}), stop_reason="tool_use"))
    replies = [messages.response,
               response(value(type="thinking", thinking="", signature="signed-second"),
                        value(type="text", text='{"summary":"CAD candidate ready"}'))]
    async def next_response(**kwargs):
        messages.calls.append(kwargs)
        return replies.pop(0)
    messages.create = next_response
    invoked = []
    async def build(context, arguments):
        invoked.append((context.tool_call_id, json.loads(arguments)))
        return "CAD compiled"
    tool = FunctionTool(name="build_cad", description="Build candidate CAD",
                        params_json_schema={"type": "object", "properties": {"program": {"type": "string"}},
                                            "required": ["program"], "additionalProperties": False},
                        on_invoke_tool=build, needs_approval=False)
    agent = Agent(name="Claude fixture author", model=ClaudeMessagesModel(client=value(messages=messages)),
                  tools=[tool], output_type=CandidateReady,
                  model_settings=ModelSettings(max_tokens=1200, store=False,
                                                            parallel_tool_calls=False))
    db = tmp_path / "conversation.sqlite"
    session = SQLiteSession("fixture", db_path=db)
    try:
        result = Runner.run_sync(agent, "Build a CAD candidate", session=session,
                                 run_config=RunConfig(tracing_disabled=True), max_turns=3)
        assert result.final_output.summary == "CAD candidate ready"
    finally:
        session.close()
    assert invoked == [("toolu_1", {"program": "fixture"})]
    assert messages.calls[1]["messages"][1]["content"][0] == {
        "type": "thinking", "thinking": "", "signature": "signed-first"}
    assert messages.calls[0]["output_config"]["format"]["schema"]["title"] == "CandidateReady"
    assert db.is_file()
