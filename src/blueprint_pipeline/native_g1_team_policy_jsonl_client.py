"""Bounded JSONL adapter for a separately sandboxed team policy process.

This module owns only the observation/action wire exchange. The caller must
verify the artifact or image, launch it in an isolated runtime, and retain its
process and teardown receipts before a scene observation is sent.
"""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import time
from collections.abc import Sequence
from typing import Any

from .native_g1_humanoidarena_policy_client import (
    build_semantic_v3_infer_request,
    validate_semantic_v3_infer_response,
)


PROTOCOL = "jsonl_observation_action_v1"
_PROFILE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class NativeG1TeamPolicyJsonlClient:
    """Speak one request at a time with a qualified policy subprocess."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        timeout_seconds: float = 30.0,
        profile_digest: str | None = None,
    ) -> None:
        if (
            process.poll() is not None
            or process.stdin is None
            or process.stdout is None
            or not 0 < timeout_seconds <= 120
            or (
                profile_digest is not None
                and (
                    not isinstance(profile_digest, str)
                    or _PROFILE_DIGEST.fullmatch(profile_digest) is None
                )
            )
        ):
            raise ValueError("g1_team_policy_process_invalid")
        self.process = process
        self.profile_digest = profile_digest
        self.timeout_seconds = timeout_seconds
        self._buffer = bytearray()
        self._request_index = 0
        self.candidate_policy_queried = False

    def _read_response(self, request_id: int) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + self.timeout_seconds
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("g1_team_policy_response_timeout")
                if not selector.select(remaining):
                    raise TimeoutError("g1_team_policy_response_timeout")
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise ValueError("g1_team_policy_process_exited_before_response")
                self._buffer.extend(block)
                if len(self._buffer) > MAX_RESPONSE_BYTES + 1:
                    raise ValueError("g1_team_policy_response_oversized")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("g1_team_policy_response_not_json") from exc
        if (
            not isinstance(value, dict)
            or value.get("protocol") != PROTOCOL
            or type(value.get("request_id")) is not int
            or value["request_id"] != request_id
            or (
                self.profile_digest is not None
                and value.get("profile_digest") != self.profile_digest
            )
        ):
            raise ValueError("g1_team_policy_response_identity_invalid")
        return value

    def _exchange(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise ValueError("g1_team_policy_process_exited")
        request_id = self._request_index
        self._request_index += 1
        request = {"protocol": PROTOCOL, "request_id": request_id, "kind": kind, **payload}
        if self.profile_digest is not None:
            request["profile_digest"] = self.profile_digest
        body = (json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("g1_team_policy_request_oversized")
        assert self.process.stdin is not None
        descriptor = self.process.stdin.fileno()
        deadline = time.monotonic() + self.timeout_seconds
        try:
            os.set_blocking(descriptor, False)
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_WRITE)
                remaining_body = memoryview(body)
                while remaining_body:
                    remaining_time = deadline - time.monotonic()
                    if remaining_time <= 0 or not selector.select(remaining_time):
                        raise TimeoutError("g1_team_policy_request_timeout")
                    try:
                        written = os.write(descriptor, remaining_body)
                    except BlockingIOError:
                        continue
                    if written == 0:
                        raise ValueError("g1_team_policy_process_write_failed")
                    remaining_body = remaining_body[written:]
        except TimeoutError:
            raise
        except (BrokenPipeError, OSError) as exc:
            raise ValueError("g1_team_policy_process_write_failed") from exc
        return self._read_response(request_id)

    def reset(self, *, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("g1_team_policy_seed_invalid")
        response = self._exchange("reset", {"seed": seed})
        if response.get("ok") is not True:
            raise ValueError("g1_team_policy_reset_ack_invalid")
        self.candidate_policy_queried = False

    def infer_chunk(
        self, *, front_rgb: Any, observation_state: Sequence[float], task: str
    ) -> list[list[float]]:
        observation = build_semantic_v3_infer_request(
            front_rgb=front_rgb,
            observation_state=observation_state,
            task=task,
        )
        response = self._exchange("infer", observation)
        self.candidate_policy_queried = True
        return validate_semantic_v3_infer_response(response)
