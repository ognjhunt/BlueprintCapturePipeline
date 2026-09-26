"""A team policy process must answer the G1 wire protocol, not just stay alive."""

import subprocess
import sys

import numpy as np
import pytest

from blueprint_pipeline.native_g1_team_policy_jsonl_client import (
    NativeG1TeamPolicyJsonlClient,
)


def _process(response: str) -> subprocess.Popen[bytes]:
    script = "\n".join(
        [
            "import json, sys",
            "for line in sys.stdin:",
            "    request = json.loads(line)",
            f"    value = {response}",
            "    value['protocol'] = request['protocol']",
            "    value.setdefault('request_id', request['request_id'])",
            "    print(json.dumps(value), flush=True)",
        ]
    )
    return subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def _close(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=2)


def test_jsonl_client_exchanges_reset_and_exact_g1_action() -> None:
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    process = _process("{'ok': True, 'action_chunk': " + repr([action]) + "}")
    try:
        client = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=2)
        client.reset(seed=9)
        assert client.infer_chunk(
            front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            observation_state=[0.0] * 64,
            task="Place the book.",
        ) == [action]
        assert client.candidate_policy_queried
    finally:
        _close(process)


def test_jsonl_client_rejects_wrong_response_identity() -> None:
    process = _process("{'ok': True, 'request_id': 999}")
    try:
        client = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=2)
        with pytest.raises(ValueError, match="response_identity_invalid"):
            client.reset(seed=9)
    finally:
        _close(process)


def test_jsonl_client_rejects_unacknowledged_reset_and_invalid_action() -> None:
    process = _process("{'ok': False, 'action_chunk': [[0.0] * 40]}")
    try:
        client = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=2)
        with pytest.raises(ValueError, match="reset_ack_invalid"):
            client.reset(seed=9)
        with pytest.raises(ValueError, match="semantic_action_rotation_invalid"):
            client.infer_chunk(
                front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                observation_state=[0.0] * 64,
                task="Place the book.",
            )
    finally:
        _close(process)


def test_jsonl_client_times_out_when_policy_stalls() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        client = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=0.1)
        with pytest.raises(TimeoutError, match="response_timeout"):
            client.reset(seed=9)
        with pytest.raises(TimeoutError, match="request_timeout"):
            client.infer_chunk(
                front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                observation_state=[0.0] * 64,
                task="Place the book.",
            )
    finally:
        _close(process)
