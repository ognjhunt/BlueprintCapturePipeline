from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import (
    GPT54_MINI_MODEL,
    PAIR_OUTPUT_SCHEMA,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_openai import (
    build_response_body,
    score_canary,
)


def _pair(tmp_path: Path) -> dict:
    def episode(prefix: str) -> dict:
        frames = []
        for index in range(32):
            path = tmp_path / f"{prefix}-{index}.jpg"
            path.write_bytes(f"{prefix}-{index}".encode())
            frames.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        return {"frames": frames, "policy_id_internal_only": f"secret-{prefix}"}

    return {
        "pair_id": "pair-1",
        "task_instruction": "move the cup",
        "episode_a": episode("a"),
        "episode_b": episode("b"),
    }


def _payload() -> dict:
    value = {
        "preferred_episode": "A",
        "episode_a_progress_0_to_5": 4,
        "episode_b_progress_0_to_5": 2,
        "stable_success_a": False,
        "stable_success_b": False,
        "comparison_confidence": 0.8,
        "uncertainty": 0.2,
        "decisive_evidence": ["A progressed farther"],
        "artifact_flags_a": [],
        "artifact_flags_b": [],
        "abstention_factors": [],
    }
    assert set(value) == set(PAIR_OUTPUT_SCHEMA["required"])
    return value


def test_body_sends_64_images_but_no_policy_identity(tmp_path: Path) -> None:
    body = build_response_body(_pair(tmp_path), model=GPT54_MINI_MODEL)
    content = body["input"][0]["content"]
    assert len([row for row in content if row["type"] == "input_image"]) == 64
    text = json.dumps(body)
    assert "secret-a" not in text
    assert "secret-b" not in text
    assert body["reasoning"] == {"effort": "high"}
    assert body["store"] is False


def test_canary_records_usage_and_redaction(tmp_path: Path) -> None:
    class Responses:
        def create(self, **kwargs):
            assert kwargs["extra_headers"]["Idempotency-Key"].startswith("diag-canary-")
            return types.SimpleNamespace(
                id="resp-test",
                status="completed",
                output_text=json.dumps(_payload()),
                usage=types.SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=200,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                ),
            )

    result = score_canary(
        types.SimpleNamespace(responses=Responses()),
        _pair(tmp_path),
        model=GPT54_MINI_MODEL,
    )
    assert result["usage"]["standard_cost_usd"] > 0
    assert result["policy_identity_sent_to_provider"] is False
    assert result["physical_ground_truth_pixels_sent_to_provider"] is False
