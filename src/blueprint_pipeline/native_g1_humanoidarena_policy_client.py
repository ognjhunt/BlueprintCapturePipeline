"""Strict inference client for the pinned HumanoidArena semantic-v3 server.

The upstream /infer protocol carries one RGB front image, a 64-value state,
robot_type and task. This client validates returned 40-value actions. A server
response does not attest the checkpoint; a worker must independently bind the
server process and materialized checkpoint inventory before admitting a run.
"""

from __future__ import annotations

import base64
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .native_g1_humanoidarena_interface import (
    STATE_WIDTH,
    parse_semantic_v3_action,
)


ROBOT_TYPE = "unitree_g1_refpose_v3_1"
IMAGE_SHAPE = (480, 640, 3)
MAX_ACTION_CHUNK = 64
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def build_semantic_v3_infer_request(
    *, front_rgb: Any, observation_state: Sequence[float], task: str
) -> dict[str, Any]:
    image = np.asarray(front_rgb)
    if image.shape != IMAGE_SHAPE or image.dtype != np.uint8:
        raise ValueError("g1_policy_front_rgb_shape_or_dtype_invalid")
    try:
        if any(isinstance(item, bool) for item in observation_state):
            raise ValueError("g1_policy_state_invalid")
        state = [float(item) for item in observation_state]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("g1_policy_state_invalid") from exc
    if len(state) != STATE_WIDTH or not all(math.isfinite(item) for item in state):
        raise ValueError("g1_policy_state_invalid")
    prompt = str(task).strip()
    if not prompt or len(prompt) > 512:
        raise ValueError("g1_policy_task_prompt_invalid")
    return {
        "observation": {
            "images": {"front": {
                "shape": list(IMAGE_SHAPE),
                "dtype": "uint8",
                "data_b64": base64.b64encode(image.tobytes(order="C")).decode("ascii"),
            }},
            "state": state,
        },
        "robot_type": ROBOT_TYPE,
        "return_chunk": True,
        "task": prompt,
    }


def validate_semantic_v3_infer_response(value: Any) -> list[list[float]]:
    if not isinstance(value, Mapping) or "action_chunk" not in value:
        raise ValueError("g1_policy_action_chunk_missing")
    chunk = value["action_chunk"]
    if (
        isinstance(chunk, (str, bytes, bytearray))
        or not isinstance(chunk, Sequence)
        or not 1 <= len(chunk) <= MAX_ACTION_CHUNK
    ):
        raise ValueError("g1_policy_action_chunk_invalid")
    actions: list[list[float]] = []
    for row in chunk:
        parse_semantic_v3_action(row)
        actions.append([float(item) for item in row])
    return actions


class NativeG1HumanoidArenaPolicyClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        transport: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment
            or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ValueError("g1_policy_endpoint_invalid")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > 120:
            raise ValueError("g1_policy_timeout_invalid")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post_json
        self.candidate_policy_queried = False

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("g1_policy_response_oversized")
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("g1_policy_response_invalid")
        return value

    def reset(self, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("g1_policy_seed_invalid")
        response = self._transport("/reset", {"seed": seed})
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise ValueError("g1_policy_reset_ack_invalid")
        self.candidate_policy_queried = False

    def infer_chunk(
        self, *, front_rgb: Any, observation_state: Sequence[float], task: str
    ) -> list[list[float]]:
        payload = build_semantic_v3_infer_request(
            front_rgb=front_rgb, observation_state=observation_state, task=task
        )
        response = self._transport("/infer", payload)
        self.candidate_policy_queried = True
        return validate_semantic_v3_infer_response(response)
