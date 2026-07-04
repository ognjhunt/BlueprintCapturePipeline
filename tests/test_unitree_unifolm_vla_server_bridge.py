from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from blueprint_pipeline import unitree_unifolm_vla_server_bridge as bridge

import pytest

pytestmark = pytest.mark.slow


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfeA\x81\xb3\x1c\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def test_unitree_unifolm_bridge_posts_unitree_act_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(PNG_1X1)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        bridge,
        "_json_numpy_dumps",
        lambda payload: json.dumps(_jsonable(payload)),
    )
    monkeypatch.setattr(bridge, "_json_numpy_loads", json.loads)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(json.dumps([[0.1] * 23])).encode("utf-8")

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(bridge.urllib.request, "urlopen", fake_urlopen)

    payload = {
        "observation": {
            "task_id": "contact_or_push_light_object",
            "task_prompt": "Stack the block.",
            "visual_observation": {"camera_frame_path": str(frame)},
            "object_state": {"object_id": "red_block", "position": [0.2, -0.1, 0.3]},
        }
    }

    response, exit_code = bridge.run_bridge_policy(
        payload=payload,
        server_url="http://127.0.0.1:8777/act",
        timeout_seconds=3,
        task_name="g1_stack_block",
    )

    sent = json.loads(captured["body"]["encoded"])
    observation = sent["observations"][0]
    assert exit_code == 0
    assert captured["url"] == "http://127.0.0.1:8777/act"
    assert captured["timeout"] == 3
    assert observation["instruction"] == "Stack the block."
    assert observation["task_name"] == "g1_stack_block"
    assert len(observation["state"]) == 23
    assert observation["full_image"] == [[[255, 255, 255]]]
    assert response["status"] == "completed"
    assert response["model_ran"] is True
    assert response["unitree_unifolm_policy_action_command_ran"] is True
    assert response["action"]["action_type"] == "manipulation_contact"
    assert response["action"]["target_object_id"] == "red_block"
    assert response["action"]["unitree_unifolm_action_chunk_present"] is True
    assert response["claim_boundary"]["unitree_hand_manipulation_policy_used"] is True


def test_unitree_unifolm_bridge_blocks_missing_frame() -> None:
    response, exit_code = bridge.run_bridge_policy(
        payload={"observation": {"task_prompt": "Stack the block."}},
        server_url="http://127.0.0.1:8777/act",
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["model_ran"] is False
    assert "blocked_missing_policy_visual_observation_frame" in response["blockers"]


def test_unitree_unifolm_bridge_preserves_json_numpy_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.png"
    frame.write_bytes(PNG_1X1)
    monkeypatch.setattr(
        bridge,
        "_json_numpy_dumps",
        lambda _payload: (_ for _ in ()).throw(
            RuntimeError("blocked_missing_json_numpy_for_unitree_unifolm_bridge")
        ),
    )

    response, exit_code = bridge.run_bridge_policy(
        payload={
            "observation": {
                "task_prompt": "Stack the block.",
                "visual_observation": {"camera_frame_path": str(frame)},
            }
        },
        server_url="http://127.0.0.1:8777/act",
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["model_ran"] is False
    assert response["blockers"] == [
        "blocked_missing_json_numpy_for_unitree_unifolm_bridge"
    ]
