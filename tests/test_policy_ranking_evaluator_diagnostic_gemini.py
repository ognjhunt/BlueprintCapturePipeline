from __future__ import annotations

import json
import types

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import PAIR_OUTPUT_SCHEMA
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini import score_canary


def _payload() -> dict:
    value = {
        "preferred_episode": "tie",
        "episode_a_progress_0_to_5": 2,
        "episode_b_progress_0_to_5": 2,
        "stable_success_a": False,
        "stable_success_b": False,
        "comparison_confidence": 0.4,
        "uncertainty": 0.6,
        "decisive_evidence": [],
        "artifact_flags_a": [],
        "artifact_flags_b": [],
        "abstention_factors": [],
    }
    assert set(value) == set(PAIR_OUTPUT_SCHEMA["required"])
    return value


def test_gemini_canary_uses_native_videos_and_records_usage() -> None:
    class Models:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "gemini-3.6-flash"
            assert kwargs["contents"][2] == "video-a"
            assert kwargs["contents"][4] == "video-b"
            return types.SimpleNamespace(
                text=json.dumps(_payload()),
                response_id="gemini-response",
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=2000,
                    candidates_token_count=100,
                    thoughts_token_count=300,
                    cached_content_token_count=0,
                    total_token_count=2400,
                ),
            )

    pair = {
        "pair_id": "pair-1",
        "task_instruction": "move the cup",
        "episode_a": {"policy_id_internal_only": "secret-a"},
        "episode_b": {"policy_id_internal_only": "secret-b"},
    }
    result = score_canary(
        types.SimpleNamespace(models=Models()),
        pair,
        video_a="video-a",
        video_b="video-b",
        types_module=types.SimpleNamespace(
            GenerateContentConfig=lambda **kwargs: kwargs,
            ThinkingConfig=lambda **kwargs: kwargs,
        ),
    )
    assert result["usage"]["standard_cost_usd"] > 0
    assert result["policy_identity_sent_to_provider"] is False
    assert result["physical_ground_truth_pixels_sent_to_provider"] is False
