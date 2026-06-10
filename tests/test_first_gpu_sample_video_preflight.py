from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.first_gpu_sample_video_preflight import (
    FIRST_GPU_SAMPLE_VIDEO_PREFLIGHT_SCHEMA_VERSION,
    build_first_gpu_sample_video_preflight,
    main,
)


def test_sample_video_preflight_marks_short_video_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    def fake_probe(path: Path) -> dict:
        assert path == source_video.resolve()
        return {
            "status": "ready",
            "duration_seconds": 12.5,
            "width": 1920,
            "height": 1080,
            "blockers": [],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.first_gpu_sample_video_preflight._ffprobe_media_metadata",
        fake_probe,
    )

    result = build_first_gpu_sample_video_preflight(source_videos=[source_video])

    assert result["schema_version"] == FIRST_GPU_SAMPLE_VIDEO_PREFLIGHT_SCHEMA_VERSION
    assert result["status"] == "ready"
    assert result["ready_for_capture_staging_count"] == 1
    assert result["ready_for_worldlabs_first_clip_count"] == 1
    assert result["claim_boundary"]["gpu_provisioning_performed"] is False
    candidate = result["candidates"][0]
    assert candidate["ready_for_capture_staging"] is True
    assert candidate["ready_for_worldlabs_first_clip"] is True
    assert candidate["next_commands"]["stage_sample"].startswith(
        "blueprint-stage-first-gpu-sample-video --source-video"
    )


def test_sample_video_preflight_blocks_long_worldlabs_clip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video = tmp_path / "long-sample.mov"
    source_video.write_bytes(b"fake-video")

    monkeypatch.setattr(
        "blueprint_pipeline.first_gpu_sample_video_preflight._ffprobe_media_metadata",
        lambda path: {
            "status": "ready",
            "duration_seconds": 31.0,
            "width": 1280,
            "height": 720,
            "blockers": [],
        },
    )

    result = build_first_gpu_sample_video_preflight(source_videos=[source_video])

    assert result["status"] == "blocked"
    assert result["blockers"] == ["no_source_videos_ready_for_worldlabs_first_clip"]
    assert result["ready_for_capture_staging_count"] == 1
    assert result["ready_for_worldlabs_first_clip_count"] == 0
    assert result["candidates"][0]["staging_blockers"] == []
    assert result["candidates"][0]["worldlabs_blockers"] == [
        "source_video_exceeds_worldlabs_duration_limit"
    ]


def test_sample_video_preflight_blocks_missing_or_unsupported_source(
    tmp_path: Path,
) -> None:
    missing_video = tmp_path / "missing.mp4"
    unsupported_source = tmp_path / "notes.txt"
    unsupported_source.write_text("not video", encoding="utf-8")

    result = build_first_gpu_sample_video_preflight(
        source_videos=[missing_video, unsupported_source]
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["no_source_videos_ready_for_capture_staging"]
    assert result["ready_for_capture_staging_count"] == 0
    assert result["candidates"][0]["staging_blockers"] == ["source_video_missing"]
    assert result["candidates"][1]["staging_blockers"] == ["unsupported_video_suffix"]


def test_sample_video_preflight_cli_search_root_writes_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video = tmp_path / "search-root" / "sample.m4v"
    source_video.parent.mkdir()
    source_video.write_bytes(b"fake-video")
    output_path = tmp_path / "preflight.json"

    monkeypatch.setattr(
        "blueprint_pipeline.first_gpu_sample_video_preflight._ffprobe_media_metadata",
        lambda path: {
            "status": "ready",
            "duration_seconds": 8.0,
            "width": 640,
            "height": 480,
            "blockers": [],
        },
    )

    exit_code = main(
        [
            "--search-root",
            str(source_video.parent),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ready"
    assert payload["source_video_count"] == 1
    assert payload["candidates"][0]["path"] == str(source_video.resolve())


def test_sample_video_preflight_require_probe_blocks_probe_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_video = tmp_path / "sample.mp4"
    source_video.write_bytes(b"fake-video")

    monkeypatch.setattr(
        "blueprint_pipeline.first_gpu_sample_video_preflight._ffprobe_media_metadata",
        lambda path: {
            "status": "failed",
            "blockers": ["ffprobe_failed"],
        },
    )

    result = build_first_gpu_sample_video_preflight(
        source_videos=[source_video],
        require_probe=True,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["no_source_videos_ready_for_worldlabs_first_clip"]
    assert result["candidates"][0]["worldlabs_blockers"] == ["media_probe_failed"]
