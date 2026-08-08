from __future__ import annotations

import json

import numpy as np
import pytest

from blueprint_pipeline.adp009d_review_video_repair import rederive_review_video
from blueprint_pipeline.episode_visual_evidence import (
    finalize_visual_evidence,
    persist_observation_frame,
)


def _sealed_episode(tmp_path):
    episode_id = "pi05_droid-episode-000"
    rows = [
        persist_observation_frame(
            np.full((32, 64, 3), value, dtype=np.uint8),
            output_dir=tmp_path,
            episode_id=episode_id,
            frame_index=index,
            kind="policy-input" if index < 2 else "terminal-observation",
        )
        for index, value in enumerate((20, 80, 140))
    ]
    finalize_visual_evidence(
        output_dir=tmp_path,
        episode_id=episode_id,
        identity={"policy_id": "pi05_droid"},
        policy_input_frames=rows[:2],
        terminal_observation=rows[2],
        frames_per_second=4.0,
    )
    return tmp_path / "media" / episode_id / "frame_manifest.json"


def test_repair_binds_lossless_manifest_and_emits_h264_without_overwrite(
    tmp_path,
) -> None:
    manifest = _sealed_episode(tmp_path)
    output = tmp_path / "review-video-repair"

    receipt = rederive_review_video(
        frame_manifest_path=manifest,
        output_dir=output,
    )

    assert receipt["status"] == "completed"
    assert receipt["source_frame_count"] == 3
    assert receipt["output_video"]["codec"] == "h264"
    assert receipt["output_video"]["fourcc"] == "avc1"
    assert receipt["output_video"]["decode_round_trip_passed"] is True
    assert (output / "pi05_droid-episode-000.mp4").is_file()
    persisted = json.loads(
        (output / "pi05_droid-episode-000.review_video_repair.json").read_text()
    )
    assert persisted["receipt_digest"] == receipt["receipt_digest"]

    with pytest.raises(FileExistsError, match="overwrite_forbidden"):
        rederive_review_video(
            frame_manifest_path=manifest,
            output_dir=output,
        )


def test_repair_rejects_changed_lossless_frame(tmp_path) -> None:
    manifest = _sealed_episode(tmp_path)
    payload = json.loads(manifest.read_text())
    changed = tmp_path / payload["policy_input_frames"][0]["relative_path"]
    changed.write_bytes(b"changed")

    with pytest.raises(ValueError, match="source_png_digest_mismatch"):
        rederive_review_video(
            frame_manifest_path=manifest,
            output_dir=tmp_path / "repair",
        )
