"""G1 LeRobot inference accepts only the publisher's semantic action format."""

import base64
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
        return {} if path == "/reset" else {"action_chunk": [_action(), _action()]}

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
