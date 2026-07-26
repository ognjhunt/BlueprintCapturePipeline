from __future__ import annotations

import json
import types
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.policy_ranking_thesis import build_preregistration, file_sha256
from blueprint_pipeline.policy_ranking_thesis_judge_openai import (
    GATE_ENV,
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
    assert result["precall_cost_bound"]["image_tokens"] == 2380
    assert result["precall_cost_bound"]["estimated_total_usd_upper_bound"] > 0
    assert "request_count_expected_98_got_2" in result["blockers"]


def test_run_requires_explicit_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(GATE_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    result = run_inventory({"requests": []}, output_path=tmp_path / "out.json")
    assert result["status"] == "blocked"
    assert result["provider_called"] is False
    assert result["data_uploaded"] is False


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
                )
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
