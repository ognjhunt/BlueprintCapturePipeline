"""Opt-in Claude Messages transport for the local Agents SDK authoring loop.

This is only a Model adapter. It is deliberately not selected by the paid
authoring worker until provider-specific reservation, pricing, receipt and
independent-review contracts are wired and qualified for a future scene.
"""
from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText, ResponseReasoningItem,
)


MODEL = "claude-opus-5-5"
_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
_THINKING_PREFIX = "anthropic-thinking-v1:"


class ClaudeModelBoundaryError(ValueError):
    """An SDK item or provider response cannot be translated without loss."""


def _dict(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        raise ClaudeModelBoundaryError("claude_sdk_item_invalid")
    return value


def _image(url: Any) -> dict:
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ClaudeModelBoundaryError("claude_remote_or_missing_image_forbidden")
    header, separator, encoded = url.partition(",")
    media_type = header.removeprefix("data:").removesuffix(";base64")
    if separator != "," or not header.endswith(";base64") or media_type not in _IMAGE_TYPES:
        raise ClaudeModelBoundaryError("claude_image_type_invalid")
    if len(encoded) > 10_000_000:
        raise ClaudeModelBoundaryError("claude_image_byte_limit")
    try:
        base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ClaudeModelBoundaryError("claude_image_base64_invalid") from exc
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}}


def _content(value: Any) -> list[dict]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise ClaudeModelBoundaryError("claude_message_content_invalid")
    result = []
    for part in value:
        item = _dict(part)
        kind = item.get("type")
        if kind in {"input_text", "output_text", "text"} and isinstance(item.get("text"), str):
            result.append({"type": "text", "text": item["text"]})
        elif kind == "input_image":
            result.append(_image(item.get("image_url")))
        else:
            raise ClaudeModelBoundaryError("claude_unsupported_message_part")
    if not result:
        raise ClaudeModelBoundaryError("claude_empty_message_content")
    return result


def _messages(value: str | list) -> list[dict]:
    if isinstance(value, str):
        return [{"role": "user", "content": _content(value)}]
    if not isinstance(value, list) or not value:
        raise ClaudeModelBoundaryError("claude_empty_history")
    messages: list[dict] = []
    pending_calls: set[str] = set()
    for raw in value:
        item = _dict(raw)
        kind = item.get("type")
        role = item.get("role")
        if kind == "reasoning":
            encoded = item.get("encrypted_content")
            if not isinstance(encoded, str) or not encoded.startswith(_THINKING_PREFIX):
                raise ClaudeModelBoundaryError("claude_foreign_reasoning_forbidden")
            try:
                block = json.loads(base64.b64decode(encoded[len(_THINKING_PREFIX):], validate=True))
            except (binascii.Error, ValueError) as exc:
                raise ClaudeModelBoundaryError("claude_thinking_envelope_invalid") from exc
            if not isinstance(block, dict) or block.get("type") not in {"thinking", "redacted_thinking"}:
                raise ClaudeModelBoundaryError("claude_thinking_envelope_invalid")
            if (block["type"] == "thinking"
                    and not all(isinstance(block.get(key), str) for key in ("thinking", "signature"))):
                raise ClaudeModelBoundaryError("claude_thinking_envelope_invalid")
            if block["type"] == "redacted_thinking" and not isinstance(block.get("data"), str):
                raise ClaudeModelBoundaryError("claude_thinking_envelope_invalid")
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "assistant", "content": [block]})
        elif kind == "function_call":
            call_id, name = item.get("call_id"), item.get("name")
            try:
                arguments = json.loads(item["arguments"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ClaudeModelBoundaryError("claude_tool_arguments_invalid") from exc
            if (not isinstance(call_id, str) or not call_id or call_id in pending_calls
                    or not isinstance(name, str) or not name or not isinstance(arguments, dict)):
                raise ClaudeModelBoundaryError("claude_tool_call_invalid")
            pending_calls.add(call_id)
            part = {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"].append(part)
            else:
                messages.append({"role": "assistant", "content": [part]})
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if call_id not in pending_calls:
                raise ClaudeModelBoundaryError("claude_orphan_tool_result")
            pending_calls.remove(call_id)
            result = item.get("output")
            content = _content(result)
            # Anthropic accepts text or multimodal blocks inside a tool result.
            part = {"type": "tool_result", "tool_use_id": call_id,
                    "content": content[0]["text"] if len(content) == 1 and content[0]["type"] == "text" else content}
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"].append(part)
            else:
                messages.append({"role": "user", "content": [part]})
        elif role in {"user", "assistant"} and kind in {None, "message"}:
            content = _content(item.get("content"))
            if (messages and messages[-1]["role"] == role
                    and not any(p["type"] == "tool_result" for p in messages[-1]["content"])):
                messages[-1]["content"].extend(content)
            else:
                messages.append({"role": role, "content": content})
        else:
            # No silent loss of OpenAI reasoning, hosted calls, or hidden state.
            raise ClaudeModelBoundaryError("claude_unsupported_history_item")
    if pending_calls:
        raise ClaudeModelBoundaryError("claude_unanswered_tool_call")
    if messages[-1]["role"] != "user":
        raise ClaudeModelBoundaryError("claude_history_must_end_with_user")
    return messages


def _usage(value: Any) -> Usage:
    def nonnegative(name: str) -> int:
        number = getattr(value, name, None)
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise ClaudeModelBoundaryError("claude_usage_invalid")
        return number
    # Count cache reads and writes at the full input rate until the durable
    # provider ledger supports exact cache tiers. This is conservative.
    input_tokens = sum(nonnegative(name) for name in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    output_tokens = nonnegative("output_tokens")
    return Usage(requests=1, input_tokens=input_tokens, output_tokens=output_tokens,
                 total_tokens=input_tokens + output_tokens)


class ClaudeMessagesModel(Model):
    """Translate one stateless Claude Messages call into an Agents SDK response."""

    def __init__(self, *, client):
        if client is None:
            raise ClaudeModelBoundaryError("claude_scoped_client_required")
        # The future worker must construct this client from its admitted _FILE
        # secret. Disable provider SDK retries even if its caller forgot to.
        self.client = client.with_options(max_retries=0) if hasattr(client, "with_options") else client

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, *, previous_response_id,
                           conversation_id, prompt):
        if previous_response_id or conversation_id or prompt or handoffs:
            raise ClaudeModelBoundaryError("claude_server_context_forbidden")
        if model_settings.store is not False or model_settings.max_tokens is None or not 0 < model_settings.max_tokens <= 12000:
            raise ClaudeModelBoundaryError("claude_bounded_local_request_required")
        if model_settings.temperature is not None or model_settings.top_p is not None:
            raise ClaudeModelBoundaryError("claude_unqualified_sampling_setting")
        if model_settings.tool_choice not in (None, "auto"):
            raise ClaudeModelBoundaryError("claude_tool_choice_unsupported")
        messages = _messages(input)
        kwargs: dict[str, Any] = {"model": MODEL, "max_tokens": model_settings.max_tokens,
                                  "messages": messages}
        if system_instructions:
            kwargs["system"] = system_instructions
        if tools:
            names = set()
            definitions = []
            for tool in tools:
                name, schema = getattr(tool, "name", None), getattr(tool, "params_json_schema", None)
                if (not isinstance(name, str) or not name or name in names
                        or not isinstance(schema, dict)):
                    raise ClaudeModelBoundaryError("claude_local_tool_invalid")
                names.add(name)
                definitions.append({"name": name, "description": tool.description,
                                    "input_schema": schema, "strict": True})
            kwargs["tools"] = definitions
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        # Opus 5.5 has adaptive thinking. Signed blocks must round-trip intact
        # on each tool continuation; omit readable thinking from local receipts.
        kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
        if output_schema is not None:
            schema = output_schema.json_schema()
            if not isinstance(schema, dict):
                raise ClaudeModelBoundaryError("claude_output_schema_invalid")
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        result = await self.client.messages.create(**kwargs)
        if result.stop_reason not in {"end_turn", "tool_use"}:
            raise ClaudeModelBoundaryError("claude_response_incomplete_or_refused")
        usage = _usage(result.usage)
        output = []
        tool_calls = 0
        for block in result.content:
            if block.type in {"thinking", "redacted_thinking"}:
                if block.type == "thinking":
                    opaque = {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
                else:
                    opaque = {"type": "redacted_thinking", "data": block.data}
                encoded = _THINKING_PREFIX + base64.b64encode(json.dumps(
                    opaque, ensure_ascii=False, separators=(",", ":")).encode()).decode()
                output.append(ResponseReasoningItem(id=result.id + f"_thinking_{len(output)}",
                    summary=[], type="reasoning", encrypted_content=encoded, status="completed"))
            elif block.type == "text":
                output.append(ResponseOutputMessage(id=result.id + f"_text_{len(output)}", role="assistant",
                    status="completed", type="message", content=[ResponseOutputText(
                        type="output_text", text=block.text, annotations=[])]))
            elif block.type == "tool_use":
                if not isinstance(block.input, dict):
                    raise ClaudeModelBoundaryError("claude_tool_input_invalid")
                tool_calls += 1
                if tool_calls > 1 or block.name not in {entry["name"] for entry in kwargs.get("tools", [])}:
                    raise ClaudeModelBoundaryError("claude_unadmitted_tool_call")
                output.append(ResponseFunctionToolCall(id=block.id, call_id=block.id, type="function_call",
                    name=block.name, arguments=json.dumps(block.input, ensure_ascii=False, allow_nan=False),
                    status="completed"))
            else:
                raise ClaudeModelBoundaryError("claude_unsupported_response_block")
        if (not any(isinstance(item, ResponseOutputMessage) for item in output) and not tool_calls
                or (result.stop_reason == "tool_use") != bool(tool_calls)):
            raise ClaudeModelBoundaryError("claude_response_shape_invalid")
        return ModelResponse(output=output, usage=usage, response_id=result.id)

    def stream_response(self, *args, **kwargs):
        raise ClaudeModelBoundaryError("claude_stream_requires_bounded_nonstreaming_runner")
