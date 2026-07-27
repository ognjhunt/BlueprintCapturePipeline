from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_thesis import build_preregistration, file_sha256
from blueprint_pipeline.policy_ranking_thesis_judge_openai import (
    GATE_ENV,
    JudgeResponseError,
    _score_one,
    build_request_inventory,
    evaluator_digest,
    run_inventory,
    sample_generated_half,
)


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 32))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :32] = (0, 0, 20 + index)
        frame[:, 32:] = (0, 255, 0)
        writer.write(frame)
    writer.release()


def test_crop_excludes_physical_right_half(tmp_path: Path) -> None:
    path = tmp_path / "paired.mp4"
    _video(path)
    frames, attestation = sample_generated_half(path, frame_limit=2)
    assert len(frames) == 2
    assert attestation["source_width"] == 64
    assert attestation["crop_x_pixels"] == [0, 32]
    assert attestation["third_party_physical_pixels_encoded"] is False
    encoded = frames[0]["image_url"].split(",", 1)[1]
    image = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(encoded), np.uint8), 1)
    assert float(image[:, :, 1].mean()) < 20.0


def test_inventory_blocks_lfs_pointers_and_binds_materialized_video(tmp_path: Path) -> None:
    sessions = [f"s{index:02d}" for index in range(63)]
    protocol = build_preregistration(sessions)
    session_id = protocol["partitions"]["pilot"][0]
    path = tmp_path / session_id / "paligemma_binning_droid" / "left" / "compare_overlay_vs_gt.mp4"
    path.parent.mkdir(parents=True)
    _video(path)
    index = {
        "rows": [
            {
                "session_id": session_id,
                "policy_id": "paligemma_binning_droid",
                "relative_path": path.relative_to(tmp_path).as_posix(),
                "sha256": file_sha256(path),
                "language_instruction": "put the can in the tray",
            }
        ]
    }
    result = build_request_inventory(index, protocol, rollout_root=tmp_path, partition="pilot")
    assert result["status"] == "blocked"
    assert result["request_count"] == 2
    assert result["requests"][0]["benchmark_labels_included"] is False
    assert result["requests"][0]["third_party_physical_pixels_included"] is False
    original_request_id = result["requests"][0]["request_id"]
    moved_root = tmp_path / "moved"
    moved_path = (
        moved_root / session_id / "paligemma_binning_droid" / "left" / "compare_overlay_vs_gt.mp4"
    )
    moved_path.parent.mkdir(parents=True)
    moved_path.write_bytes(path.read_bytes())
    moved = build_request_inventory(index, protocol, rollout_root=moved_root, partition="pilot")
    assert moved["requests"][0]["request_id"] == original_request_id
    assert result["precall_cost_bound"]["image_tokens"] == 2380
    assert result["precall_cost_bound"]["estimated_total_usd_upper_bound"] > 0
    assert result["sampling_contract"]["max_output_tokens_including_reasoning"] == 8192
    assert result["sampling_contract"]["temperature"] == "not_requested_model_default"
    assert "request_count_expected_98_got_2" in result["blockers"]


def test_run_requires_explicit_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(GATE_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    result = run_inventory({"requests": []}, output_path=tmp_path / "out.json")
    assert result["status"] == "blocked"
    assert result["provider_called"] is False
    assert result["data_uploaded"] is False


def test_missing_credential_preserves_matching_resumable_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(GATE_ENV, "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "run.json"
    previous = {
        "inventory_sha256": "a" * 64,
        "status": "running",
        "judgments": [{"request_id": "already-paid"}],
    }
    output.write_text(json.dumps(previous))

    result = run_inventory({"inventory_sha256": "a" * 64, "requests": []}, output_path=output)

    assert result["status"] == "blocked"
    assert result["existing_checkpoint_preserved"] is True
    assert result["existing_judgment_count"] == 1
    assert json.loads(output.read_text()) == previous


def test_score_emits_label_blind_schema(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "paired.mp4"
    _video(path)
    protocol = build_preregistration([f"s{index:02d}" for index in range(63)])

    class Responses:
        def create(self, **kwargs):
            content = kwargs["input"][0]["content"]
            assert len([row for row in content if row["type"] == "input_image"]) == 2
            return types.SimpleNamespace(
                id="resp_test",
                status="completed",
                usage=types.SimpleNamespace(
                    input_tokens=250,
                    output_tokens=100,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                ),
                output_text=json.dumps(
                    {
                        "success_probability": 0.7,
                        "progress_score_0_to_5": 4,
                        "judge_confidence": 0.8,
                        "action_following_confidence": 0.9,
                        "temporal_coherence_confidence": 0.85,
                        "critical_contradiction": False,
                        "abstain": False,
                        "rationale": "Visible progress.",
                    }
                ),
            )

    client = types.SimpleNamespace(responses=Responses())
    judgment, crop = _score_one(
        client,
        {
            "video_path": str(path),
            "frame_count": 2,
            "task_instruction": "put the can in the tray",
            "method": protocol["evaluator"]["cheap_baseline_method"],
            "session_id": "s00",
            "policy_id": "p",
            "evaluator_digest": evaluator_digest(protocol),
        },
    )
    assert judgment["success_probability"] == 0.7
    assert judgment["benchmark_labels_seen"] is False
    assert judgment["third_party_physical_pixels_seen"] is False
    assert judgment["usage"]["input_tokens"] == 250
    assert judgment["usage"]["estimated_cost_usd_conservative"] > 0
    assert crop["third_party_physical_pixels_encoded"] is False


def test_unparseable_provider_response_preserves_safe_usage(tmp_path: Path) -> None:
    path = tmp_path / "paired.mp4"
    _video(path)
    protocol = build_preregistration([f"s{index:02d}" for index in range(63)])

    class Responses:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                id="resp_incomplete",
                status="incomplete",
                incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
                usage=types.SimpleNamespace(
                    input_tokens=200,
                    output_tokens=900,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                ),
                output_text="",
            )

    client = types.SimpleNamespace(responses=Responses())
    with pytest.raises(JudgeResponseError) as raised:
        _score_one(
            client,
            {
                "video_path": str(path),
                "frame_count": 2,
                "task_instruction": "put the can in the tray",
                "method": protocol["evaluator"]["cheap_baseline_method"],
                "session_id": "s00",
                "policy_id": "p",
                "evaluator_digest": evaluator_digest(protocol),
            },
        )
    assert raised.value.safe_details["incomplete_reason"] == "max_output_tokens"
    assert raised.value.safe_details["usage"]["output_tokens"] == 900
    assert raised.value.safe_details["raw_response_persisted"] is False


def test_run_inventory_concurrent_checkpoint_is_request_ordered(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(GATE_ENV, "1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-sent")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=lambda **kwargs: object()),
    )

    def fake_score(client, request):
        return (
            {
                "request_id": request["request_id"],
                "usage": {"estimated_cost_usd_conservative": 0.001},
            },
            {},
        )

    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_thesis_judge_openai._score_one", fake_score
    )
    requests = [{"request_id": str(index)} for index in range(6)]
    result = run_inventory(
        {"inventory_sha256": "a" * 64, "requests": requests},
        output_path=tmp_path / "run.json",
        max_workers=3,
        max_estimated_cost_usd=1.0,
    )
    assert result["status"] == "completed"
    assert result["max_workers"] == 3
    assert result["max_attempts_per_request"] == 2
    assert [row["request_id"] for row in result["judgments"]] == [str(i) for i in range(6)]


def test_retry_cap_counts_failed_usage_and_exhausts_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(GATE_ENV, "1")
    monkeypatch.setenv("OPENAI_API_KEY", "not-sent")
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=lambda **kwargs: object()),
    )
    request = {"request_id": "r1"}
    previous = {
        "inventory_sha256": "b" * 64,
        "judgments": [],
        "failed_requests": [
            {
                "request_id": "r1",
                "usage": {"estimated_cost_usd_conservative": 0.04},
            },
            {
                "request_id": "r1",
                "usage": {"estimated_cost_usd_conservative": 0.05},
            },
        ],
    }
    output = tmp_path / "run.json"
    output.write_text(json.dumps(previous))
    result = run_inventory(
        {"inventory_sha256": "b" * 64, "requests": [request]},
        output_path=output,
        max_estimated_cost_usd=1.0,
        max_attempts_per_request=2,
    )
    assert result["status"] == "blocked"
    assert result["estimated_cost_usd_conservative"] == pytest.approx(0.09)
    assert result["blockers"] == ["retry_exhausted:r1"]
