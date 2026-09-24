"""Durable, receipt-bound Messages transport for the local Agents SDK.

The exact translated request is written before dispatch. The full response,
including signed thinking, is written before Runner may execute a tool. A
request with no response is an unknown provider outcome and is never resent.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .claude_opus_authoring_invoker import (
    ClaudeAuthoringBlocked, ClaudeOpusAuthoringInvoker, MODEL,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_supervisor.sdk_image_tools import encode_tool_output
from .task_object_claude_model import _messages


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ClaudeAuthoringBlocked("claude_sdk_journal_path_invalid")
    data = (canonical_json(value) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ClaudeAuthoringBlocked("claude_sdk_journal_record_invalid")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClaudeAuthoringBlocked("claude_sdk_journal_record_invalid") from exc


def _response(value: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(id=value["id"], stop_reason=value["stop_reason"],
        content=[SimpleNamespace(**block) for block in value["content"]],
        usage=SimpleNamespace(**value["usage"]))


class ClaudeSDKMessageClient:
    """One journal and one ordered SDK conversation for a single asset."""

    def __init__(self, *, invoker: ClaudeOpusAuthoringInvoker, object_id: str,
                 journal_root: Path, session_db: Path | None = None,
                 tool_root: Path | None = None):
        if not object_id or not isinstance(invoker, ClaudeOpusAuthoringInvoker):
            raise ValueError("claude_sdk_bridge_configuration_invalid")
        self.invoker, self.object_id = invoker, object_id
        self.root = Path(journal_root)
        self.session_db = Path(session_db) if session_db is not None else None
        self.tool_root = Path(tool_root) if tool_root is not None else None
        self.messages = self
        self._lock = asyncio.Lock()
        self._initial_call = True
        binding = {"schema_version": "claude_sdk_messages_journal.v1",
                   "run_id": invoker.config.run_id, "object_id": object_id, "model": MODEL}
        path = self.root / "binding.json"
        if path.exists() or path.is_symlink():
            if _read(path) != binding:
                raise ClaudeAuthoringBlocked("claude_sdk_journal_binding_changed")
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ClaudeAuthoringBlocked("claude_sdk_journal_binding_missing")
            _write_once(path, binding)
        self._records = self.inspect_journal()

    def _paths(self, turn: int) -> tuple[Path, Path]:
        prefix = self.root / f"turn-{turn:03d}"
        return prefix.with_name(prefix.name + "-request.json"), prefix.with_name(prefix.name + "-response.json")

    def _verify_receipt(self, turn: int, payload: Mapping[str, Any], response: Mapping[str, Any]) -> None:
        capability = f"{self.object_id}_author_turn_{turn:03d}"
        request_digest = canonical_digest({"request": payload})
        output_digest = canonical_digest({"content": response.get("content")})
        matches = []
        for path in self.invoker.audit.completed_root.glob("*.json"):
            completion = _read(path)
            if completion.get("capability") != capability:
                continue
            reservation = _read(self.invoker.audit._reservation_path(completion["reservation_id"]))
            if (completion.get("inference_completion_digest") != canonical_digest(
                    completion, digest_field="inference_completion_digest")
                    or reservation.get("inference_reservation_digest") != canonical_digest(
                        reservation, digest_field="inference_reservation_digest")
                    or reservation.get("run_id") != self.invoker.config.run_id
                    or reservation.get("input_digest") != request_digest
                    or completion.get("reservation_id") != reservation.get("reservation_id")
                    or completion.get("model") != MODEL or completion.get("provider") != "anthropic"
                    or completion.get("status") != "completed"
                    or completion.get("provider_response_id") != response.get("id")
                    or completion.get("response_output_digest") != output_digest
                    or completion.get("provider_outcome") != response.get("stop_reason")
                    or completion.get("usage") != response.get("usage")):
                raise ClaudeAuthoringBlocked("claude_sdk_journal_receipt_changed")
            matches.append(completion)
        if len(matches) != 1:
            raise ClaudeAuthoringBlocked("claude_sdk_journal_receipt_missing")

    def inspect_journal(self) -> list[tuple[dict, dict]]:
        """Read-only verification of every retained provider turn and receipt."""
        records: list[tuple[dict, dict]] = []
        requests = sorted(self.root.glob("turn-*-request.json"))
        responses = sorted(self.root.glob("turn-*-response.json"))
        expected_requests = [self._paths(n)[0] for n in range(1, len(requests) + 1)]
        if requests != expected_requests or any(path.is_symlink() for path in requests + responses):
            raise ClaudeAuthoringBlocked("claude_sdk_journal_sequence_invalid")
        for turn, request_path in enumerate(requests, 1):
            response_path = self._paths(turn)[1]
            if not response_path.exists():
                raise ClaudeAuthoringBlocked("claude_provider_outcome_unknown")
            payload, response = _read(request_path), _read(response_path)
            if not isinstance(payload, dict) or not isinstance(response, dict):
                raise ClaudeAuthoringBlocked("claude_sdk_journal_record_invalid")
            self._verify_receipt(turn, payload, response)
            records.append((payload, response))
        if responses != [self._paths(n)[1] for n in range(1, len(requests) + 1)]:
            raise ClaudeAuthoringBlocked("claude_sdk_journal_sequence_invalid")
        return records

    def _last_response_checkpointed(self, response: Mapping[str, Any]) -> bool:
        """Inspect the closed SQLite checkpoint without writing into it."""
        if self.session_db is None or not self.session_db.is_file():
            raise ClaudeAuthoringBlocked("claude_sdk_replay_requires_session_inspection")
        wal = self.session_db.with_name(self.session_db.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ClaudeAuthoringBlocked("claude_sdk_conversation_not_checkpointed")
        calls = {block.get("id") for block in response["content"]
                 if block.get("type") == "tool_use"}
        if not calls or None in calls:
            raise ClaudeAuthoringBlocked("claude_sdk_replay_ambiguous")
        with closing(sqlite3.connect(self.session_db.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            items = [json.loads(row[0]) for row in db.execute(
                "SELECT message_data FROM agent_messages WHERE session_id=? ORDER BY id",
                (self.object_id,))]
        seen = {item.get("call_id") for item in items
                if item.get("type") == "function_call"}
        done = {item.get("call_id") for item in items
                if item.get("type") == "function_call_output"}
        if calls & seen and not calls <= done:
            raise ClaudeAuthoringBlocked("claude_sdk_replay_partial_checkpoint")
        return calls <= seen and calls <= done

    def completed_tool_history(self, tool_root: Path | None = None) -> list[tuple[str, dict, Any]]:
        """Read only SQLite, signed provider turns and local tool outcomes."""
        records = self.inspect_journal()
        tool_root = Path(tool_root) if tool_root is not None else self.tool_root
        if tool_root is None:
            raise ClaudeAuthoringBlocked("claude_sdk_tool_ledger_missing")
        if self.session_db is None or not self.session_db.is_file():
            raise ClaudeAuthoringBlocked("claude_sdk_conversation_missing")
        wal = self.session_db.with_name(self.session_db.name + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ClaudeAuthoringBlocked("claude_sdk_conversation_not_checkpointed")
        with closing(sqlite3.connect(self.session_db.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            items = [json.loads(row[0]) for row in db.execute(
                "SELECT message_data FROM agent_messages WHERE session_id=? ORDER BY id",
                (self.object_id,))]
        calls = [item for item in items if item.get("type") == "function_call"]
        outputs = [item for item in items if item.get("type") == "function_call_output"]
        expected = [block for _, response in records for block in response["content"]
                    if block.get("type") == "tool_use"]
        if (len(calls) != len(expected) or len(outputs) != len(expected)
                or [item.get("call_id") for item in calls] != [block.get("id") for block in expected]
                or {item.get("call_id") for item in outputs}
                    != {block.get("id") for block in expected}):
            raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed")
        for payload, response in records:
            blocks = [block for block in response["content"] if block.get("type") == "tool_use"]
            if len(blocks) != 1:
                raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed")
            call_id = blocks[0]["id"]
            matches = [index for index, item in enumerate(items)
                       if item.get("type") == "function_call" and item.get("call_id") == call_id]
            if len(matches) != 1:
                raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed")
            start = matches[0]
            while start and items[start - 1].get("type") == "reasoning":
                start -= 1
            if _messages(items[:start]) != payload.get("messages"):
                raise ClaudeAuthoringBlocked("claude_sdk_conversation_request_changed")
        history = []
        outputs_by_id = {item["call_id"]: item for item in outputs}
        for item, block in zip(calls, expected, strict=True):
            call_id, name, arguments = block.get("id"), block.get("name"), block.get("input")
            if (not isinstance(call_id, str) or not call_id or item.get("name") != name
                    or not isinstance(arguments, dict)):
                raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed")
            try:
                recorded_arguments = json.loads(item["arguments"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed") from exc
            if recorded_arguments != arguments:
                raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed")
            token = canonical_digest({"call_id": call_id})[7:]
            started = _read(Path(tool_root) / (token + ".started.json"))
            result = _read(Path(tool_root) / (token + ".result.json"))
            if started != {"tool": name, "call_id": call_id, "arguments": arguments}:
                raise ClaudeAuthoringBlocked("claude_sdk_tool_history_changed")
            encoded, _ = encode_tool_output(result, model="gpt-6-astra")
            if outputs_by_id[call_id].get("output") != (
                    encoded if encoded is not None else canonical_json(result)):
                raise ClaudeAuthoringBlocked("claude_sdk_tool_output_changed")
            history.append((name, arguments, result))
        return history

    async def create(self, **payload):
        async with self._lock:
            # A restarted Runner may ask for the last response again after a
            # crash before SQLite checkpoint. Never buy it twice.
            if self._initial_call and self._records:
                checkpointed = self._last_response_checkpointed(self._records[-1][1])
                if payload == self._records[-1][0]:
                    self._initial_call = False
                    if checkpointed:
                        raise ClaudeAuthoringBlocked("claude_sdk_replay_already_checkpointed")
                    return _response(self._records[-1][1])
                if not checkpointed:
                    raise ClaudeAuthoringBlocked("claude_sdk_prior_response_not_checkpointed")
            self._initial_call = False
            turn = len(self._records) + 1
            request_path, response_path = self._paths(turn)
            _write_once(request_path, payload)
            capability = f"{self.object_id}_author_turn_{turn:03d}"
            value = await asyncio.to_thread(self.invoker.invoke_tool_turn,
                                            capability=capability, payload=payload)
            _write_once(response_path, value)
            self._verify_receipt(turn, payload, value)
            self._records.append((payload, value))
            return _response(value)
