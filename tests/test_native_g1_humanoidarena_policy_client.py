"""G1 LeRobot inference accepts only the publisher's semantic action format."""

import base64
import json
import math

import numpy as np
import pytest

from blueprint_pipeline.native_g1_humanoidarena_policy_client import (
    NativeG1HumanoidArenaPolicyClient,
    build_semantic_v3_infer_request,
    validate_semantic_v3_infer_response,
)


def _action() -> list[float]:
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    return action


def test_exact_upstream_request_and_action_chunk() -> None:
    requests = []

    def transport(path, payload):
        requests.append((path, payload))
        return {"ok": True} if path == "/reset" else {"action_chunk": [_action(), _action()]}

    client = NativeG1HumanoidArenaPolicyClient(
        base_url="http://127.0.0.1:18080", transport=transport
    )
    client.reset(seed=7)
    actions = client.infer_chunk(
        front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        observation_state=[0.0] * 64,
        task="Place the box on the shelf.",
    )
    assert actions == [_action(), _action()]
    assert client.candidate_policy_queried is True
    assert requests[0] == ("/reset", {"seed": 7})
    path, payload = requests[1]
    assert path == "/infer"
    assert payload["robot_type"] == "unitree_g1_refpose_v3_1"
    assert payload["return_chunk"] is True
    assert payload["task"] == "Place the box on the shelf."
    assert payload["observation"]["state"] == [0.0] * 64
    image = payload["observation"]["images"]["front"]
    assert image["shape"] == [480, 640, 3]
    assert image["dtype"] == "uint8"
    assert len(base64.b64decode(image["data_b64"])) == 480 * 640 * 3


def test_inference_refuses_latents_bad_rows_and_unbound_input() -> None:
    for response in (
        {"latent64_chunk": [[0.0] * 64]},
        {"action_chunk": []},
        {"action_chunk": [[0.0] * 78]},
        {"action_chunk": [[math.nan] * 40]},
        {"action_chunk": [[True] * 40]},
    ):
        with pytest.raises(ValueError):
            validate_semantic_v3_infer_response(response)
    with pytest.raises(ValueError, match="front_rgb"):
        build_semantic_v3_infer_request(
            front_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            observation_state=[0.0] * 64,
            task="Place the box.",
        )
    with pytest.raises(ValueError, match="state_invalid"):
        build_semantic_v3_infer_request(
            front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            observation_state=[0.0] * 63,
            task="Place the box.",
        )
    with pytest.raises(ValueError, match="endpoint_invalid"):
        NativeG1HumanoidArenaPolicyClient(base_url="http://untrusted.example:18080")


def test_reset_requires_official_server_ack() -> None:
    client = NativeG1HumanoidArenaPolicyClient(
        base_url="http://127.0.0.1:18080", transport=lambda path, payload: {"error": "failed"}
    )
    with pytest.raises(ValueError, match="reset_ack_invalid"):
        client.reset(seed=7)


def test_real_transport_has_separate_first_steady_and_reset_deadlines(monkeypatch):
    from blueprint_pipeline import native_g1_humanoidarena_policy_client as module
    observed = []
    class Response:
        def __init__(self, value):
            self.value = value
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, count):
            return json.dumps(self.value).encode()
    class Opener:
        def open(self, request, *, timeout):
            observed.append((request.full_url.rsplit("/", 1)[-1], timeout))
            return Response({"ok": True} if request.full_url.endswith("/reset") else {"action_chunk": [_action()]})
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    client = NativeG1HumanoidArenaPolicyClient(
        base_url="http://127.0.0.1:18080", first_inference_timeout_seconds=600,
    )
    arguments = {"front_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                 "observation_state": [0.0] * 64, "task": "Place the book."}
    client.reset(seed=1)
    client.infer_chunk(**arguments)
    assert client.inference_timing_receipt["phase"] == "first_inference"
    assert client.inference_timing_receipt["status"] == "returned_valid_action"
    client.reset(seed=2)
    client.infer_chunk(**arguments)
    assert observed == [("reset", 30), ("infer", 600), ("reset", 30), ("infer", 30)]
    assert client.inference_timing_receipt["phase"] == "steady_inference"


@pytest.mark.parametrize("value", [True, 0, -1, math.inf, math.nan, 601])
def test_first_inference_deadline_is_bounded(value):
    with pytest.raises(ValueError, match="timeout_invalid"):
        NativeG1HumanoidArenaPolicyClient(
            base_url="http://127.0.0.1:18080", first_inference_timeout_seconds=value,
        )


def test_timeout_is_typed_retained_and_never_retried_or_counted():
    calls = []
    def transport(path, payload):
        calls.append(path)
        raise TimeoutError("private endpoint response")
    client = NativeG1HumanoidArenaPolicyClient(
        base_url="http://127.0.0.1:18080", transport=transport,
        first_inference_timeout_seconds=600,
    )
    with pytest.raises(TimeoutError, match="g1_policy_first_inference_timeout") as error:
        client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                           observation_state=[0.0] * 64, task="Place the book.")
    assert "private endpoint" not in str(error.value)
    assert calls == ["/infer"]
    assert client.candidate_policy_queried is False
    assert client.inference_timing_receipt["status"] == "timeout"
    assert client.inference_timing_receipt["timeout_seconds"] == 600
    assert client.inference_timing_receipt["elapsed_seconds"] >= 0


def test_invalid_action_does_not_count_as_completed_inference():
    client = NativeG1HumanoidArenaPolicyClient(
        base_url="http://127.0.0.1:18080", transport=lambda *args: {"action_chunk": []},
    )
    with pytest.raises(ValueError, match="action_chunk_invalid"):
        client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                           observation_state=[0.0] * 64, task="Place the book.")
    assert client.candidate_policy_queried is False
    assert client.inference_timing_receipt["status"] == "failed"
