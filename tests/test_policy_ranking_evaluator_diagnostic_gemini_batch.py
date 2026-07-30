from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_batch import (
    BATCH_RESPONSE_SCHEMA,
    GeminiBatchDiagnosticError,
    _build_inline_request,
    collect_pilot,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def _valid_payload() -> dict:
    return {
        "preferred_episode": "B",
        "episode_a_progress_0_to_5": 2,
        "episode_b_progress_0_to_5": 4,
        "stable_success_a": False,
        "stable_success_b": True,
        "comparison_confidence": 0.8,
        "uncertainty": 0.2,
        "decisive_evidence": ["B completes more of the task"],
        "artifact_flags_a": [],
        "artifact_flags_b": [],
        "abstention_factors": [],
    }


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
    assert request["config"]["response_schema"] == BATCH_RESPONSE_SCHEMA
    assert "additionalProperties" not in request["config"]["response_schema"]
    assert "response_json_schema" not in request["config"]


def test_terminal_malformed_outputs_are_retained_and_uploads_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    deleted: list[str] = []
    response = types.SimpleNamespace(
        text='{"preference":"A"}',
        response_id="response-1",
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=10,
            thoughts_token_count=20,
            cached_content_token_count=0,
            total_token_count=130,
        ),
    )
    job = types.SimpleNamespace(
        state=types.SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
        dest=types.SimpleNamespace(
            inlined_responses=[types.SimpleNamespace(response=response, error=None)]
        ),
    )

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.batches = types.SimpleNamespace(get=lambda name: job)
            self.files = types.SimpleNamespace(delete=lambda name: deleted.append(name))

    monkeypatch.setitem(
        sys.modules,
        "google",
        types.SimpleNamespace(genai=types.SimpleNamespace(Client=FakeClient)),
    )
    key = tmp_path / "key"
    key.write_text("test-key")
    key.chmod(0o600)
    receipt = {
        "batch_name": "batches/test",
        "arm_id": "gemini36_flash_native_video",
        "pair_ids": ["pair-1"],
        "unique_video_count": 1,
        "uploads": [{"provider_file_name": "files/test"}],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    report = collect_pilot(
        receipt,
        api_key_file=key,
        output_path=tmp_path / "collection.json",
    )

    assert report["status"] == "failed"
    assert report["result_count"] == 0
    assert report["error_count"] == 1
    assert report["errors"][0]["raw_response_text"] == '{"preference":"A"}'
    assert report["estimated_batch_cost_usd"] > 0
    assert report["temporary_video_files_deleted"] is True
    assert deleted == ["files/test"]


def test_complete_graph_arm_is_propagated_to_collection_and_results(
    tmp_path: Path, monkeypatch
) -> None:
    deleted: list[str] = []
    response = types.SimpleNamespace(
        text=json.dumps(_valid_payload()),
        response_id="response-1",
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=10,
            thoughts_token_count=20,
            cached_content_token_count=0,
            total_token_count=130,
        ),
    )
    job = types.SimpleNamespace(
        state=types.SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
        dest=types.SimpleNamespace(
            inlined_responses=[types.SimpleNamespace(response=response, error=None)]
        ),
    )

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.batches = types.SimpleNamespace(get=lambda name: job)
            self.files = types.SimpleNamespace(delete=lambda name: deleted.append(name))

    monkeypatch.setitem(
        sys.modules,
        "google",
        types.SimpleNamespace(genai=types.SimpleNamespace(Client=FakeClient)),
    )
    key = tmp_path / "key"
    key.write_text("test-key")
    key.chmod(0o600)
    receipt = {
        "batch_name": "batches/test",
        "arm_id": "gemini36_flash_complete_graph",
        "pair_ids": ["pair-1"],
        "unique_video_count": 1,
        "uploads": [{"provider_file_name": "files/test"}],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)

    report = collect_pilot(
        receipt,
        api_key_file=key,
        output_path=tmp_path / "collection.json",
    )

    assert report["status"] == "completed"
    assert report["arm_id"] == "gemini36_flash_complete_graph"
    assert report["results"][0]["arm_id"] == "gemini36_flash_complete_graph"
    assert deleted == ["files/test"]


def test_collection_rejects_unregistered_receipt_arm_before_provider_call(
    tmp_path: Path,
) -> None:
    receipt = {
        "batch_name": "batches/test",
        "arm_id": "unregistered_fused_arm",
        "pair_ids": ["pair-1"],
        "unique_video_count": 0,
        "uploads": [],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)

    with pytest.raises(GeminiBatchDiagnosticError, match="batch_receipt_arm_id_invalid"):
        collect_pilot(
            receipt,
            api_key_file=tmp_path / "unused-key",
            output_path=tmp_path / "collection.json",
        )
