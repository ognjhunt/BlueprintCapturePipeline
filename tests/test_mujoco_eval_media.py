from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blueprint_pipeline import mujoco_eval_media as media


def test_sampling_and_timing_contracts_preserve_simulator_time() -> None:
    assert media.episode_frame_steps(
        steps_per_episode=10,
        render_frame_count=0,
        video_frame_stride_steps=4,
    ) == ([0, 4, 8, 9], "full_episode_stride", 4)
    assert media.video_output_fps(requested_fps=0, timestep=0.002, stride_steps=20) == 25

    timing = media.video_timing_contract(
        requested_fps=10,
        encoded_fps=10,
        timestep=0.002,
        stride_steps=1,
        physics_frame_count=100,
        encoded_frame_count=100,
    )
    assert timing["video_playback_may_look_slow_motion"] is True
    assert timing["physics_duration_s"] == 0.2
    assert timing["encoded_duration_estimate_s"] == 10.0

    review = media.review_video_sampling_contract(
        fps=media.DEFAULT_REVIEW_VIDEO_FPS,
        timestep=0.002,
        video_frame_stride_steps=media.DEFAULT_VIDEO_FRAME_STRIDE_STEPS,
        render_frame_count=0,
        extend_terminal_frame_for_review=False,
    )
    assert review["sampling_mode"] == "nominal_realtime_stride_review"
    assert review["review_video_stops_at_terminal_failure_by_default"] is True


def test_ffprobe_and_grouped_counts_are_provider_independent(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    video = tmp_path / "review.mp4"
    video.write_bytes(b"review")
    monkeypatch.setattr(media.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {"streams": [{"nb_frames": "3", "duration": "0.5", "width": 640, "height": 360}]}
            ),
            "",
        ),
    )

    inspection = media.ffprobe_video(video)
    assert inspection["status"] == "complete"
    assert inspection["frame_count"] == 3
    assert media.counts_by_key(
        [
            {"lane": "fixture", "success": True},
            {"lane": "fixture", "success": False},
            {"lane": "live", "status": "blocked"},
        ],
        "lane",
    ) == [
        {"id": "fixture", "attempted": 2, "passed": 1, "failed": 1, "blocked": 0},
        {"id": "live", "attempted": 1, "passed": 0, "failed": 0, "blocked": 1},
    ]
