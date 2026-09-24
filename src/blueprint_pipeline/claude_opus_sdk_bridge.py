"""No-network-retry bridge from the local Agents SDK Model to Claude admission.

The bridge is opt-in and is not selected by the paid scene driver. It lets the
existing signed Anthropic reservation/settlement seam own each Messages call.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from .claude_opus_authoring_invoker import ClaudeOpusAuthoringInvoker


class ClaudeSDKMessageClient:
    def __init__(self, *, invoker: ClaudeOpusAuthoringInvoker, object_id: str):
        if not object_id or not isinstance(invoker, ClaudeOpusAuthoringInvoker):
            raise ValueError("claude_sdk_bridge_configuration_invalid")
        self.invoker = invoker
        self.object_id = object_id
        self.messages = self
        self._turn = 0
        self._lock = asyncio.Lock()

    async def create(self, **payload):
        async with self._lock:
            self._turn += 1
            capability = f"{self.object_id}_author_turn_{self._turn:03d}"
            # The synchronous invoker reserves before dispatch, preserves an
            # uncertain reservation on transport error, and receipts usage.
            value = await asyncio.to_thread(self.invoker.invoke_tool_turn,
                                            capability=capability, payload=payload)
            return SimpleNamespace(id=value["id"], stop_reason=value["stop_reason"],
                content=[SimpleNamespace(**block) for block in value["content"]],
                usage=SimpleNamespace(**value["usage"]))
