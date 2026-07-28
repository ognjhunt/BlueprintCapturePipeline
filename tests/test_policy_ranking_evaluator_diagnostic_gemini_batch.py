from __future__ import annotations

import json
import types

from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_batch import (
    _build_inline_request,
)


def test_gemini_batch_request_uses_two_videos_without_policy_identity() -> None:
    pair = {
        "pair_id": "pair-1",
        "task_instruction": "move the cup",
        "episode_a": {"policy_id_internal_only": "secret-a"},
        "episode_b": {"policy_id_internal_only": "secret-b"},
    }
    fake_types = types.SimpleNamespace(
        InlinedRequest=lambda **kwargs: kwargs,
        GenerateContentConfig=lambda **kwargs: kwargs,
        ThinkingConfig=lambda **kwargs: kwargs,
    )
    request = _build_inline_request(
        pair, "video-a", "video-b", types_module=fake_types
    )

    assert request["contents"][2] == "video-a"
    assert request["contents"][4] == "video-b"
    assert request["metadata"] == {"pair_id": "pair-1"}
    encoded = json.dumps(request)
    assert "secret-a" not in encoded
    assert "secret-b" not in encoded
    assert request["config"]["response_mime_type"] == "application/json"
