from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from blueprint_pipeline.wam_generated_video_review import (
    assess_source_policy_observation_visual_qa,
    write_persistent_wam_visual_quality_artifacts,
)


def _write_good_frame(path: Path, *, size: tuple[int, int] = (320, 256)) -> Path:
    width, height = size
    x_gradient = np.tile(np.linspace(48, 210, width, dtype=np.uint8), (height, 1))
    y_gradient = np.tile(np.linspace(32, 120, height, dtype=np.uint8), (width, 1)).T
    frame = np.dstack((x_gradient, np.roll(x_gradient, 32, axis=1), y_gradient))
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 2 - 42, height // 2 - 30, width // 2 + 42, height // 2 + 30), outline=(255, 255, 255), width=4)
    draw.ellipse((width // 2 - 14, height // 2 - 14, width // 2 + 14, height // 2 + 14), fill=(230, 70, 55))
    for x in range(0, width, 24):
        draw.line((x, 0, x, height), fill=(20, 20, 20), width=1)
    for y in range(0, height, 24):
        draw.line((0, y, width, y), fill=(240, 240, 240), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_dark_frame(path: Path, *, size: tuple[int, int] = (320, 240)) -> Path:
    image = Image.new("RGB", size, (8, 8, 8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(20, 22, 18))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _video_status(*, width: int, height: int, fps: str, frames: int) -> dict[str, object]:
    return {
        "status": "completed",
        "ffprobe_metadata": {
            "streams": [
                {
                    "width": width,
                    "height": height,
                    "avg_frame_rate": fps,
                    "r_frame_rate": fps,
                    "nb_frames": str(frames),
                    "duration": "6.0",
                }
            ],
            "format": {"duration": "6.0", "size": "1000"},
        },
    }


def test_source_policy_observation_visual_qa_good_frame_passes(tmp_path: Path) -> None:
    frame = _write_good_frame(tmp_path / "good.jpg")

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "passed_visual_quality_gate"
    assert qa["visual_success"] is True
    assert qa["target_visibility_status"] == "passed_visual_proxy"
    assert qa["blockers"] == []


def test_source_policy_observation_visual_qa_dark_flat_occluded_frame_fails(
    tmp_path: Path,
) -> None:
    frame = _write_dark_frame(tmp_path / "dark.jpg")

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "failed_visual_quality_gate"
    assert qa["visual_success"] is False
    assert "source_policy_observation_too_dark_for_review" in qa["blockers"]
    assert "target_object_visibility_failed_visual_proxy" in qa["blockers"]


def test_review_quality_profile_rejects_128px_media_while_smoke_marks_smoke_only(
    tmp_path: Path,
) -> None:
    source = _write_good_frame(tmp_path / "source.jpg")
    generated = _write_good_frame(tmp_path / "generated.jpg", size=(128, 128))

    review_report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "review-job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[generated],
        review_video_path=tmp_path / "review.mp4",
        video_status=_video_status(width=128, height=128, fps="4/1", frames=9),
        visual_profile="review_quality",
        requested_settings={"width": 128, "height": 128, "fps": 4, "num_frames": 9},
        provider_status="completed",
        live_wam_generation_success_count=1,
        learned_wam_model_success_count=1,
        target_object_id="Sink054_handle",
        task_id="turn_on_sink_handle",
    )
    smoke_report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "smoke-job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[generated],
        review_video_path=tmp_path / "smoke.mp4",
        video_status=_video_status(width=128, height=128, fps="4/1", frames=9),
        visual_profile="smoke",
        requested_settings={"width": 128, "height": 128, "fps": 4, "num_frames": 9},
        provider_status="completed",
        live_wam_generation_success_count=1,
        learned_wam_model_success_count=1,
    )

    assert review_report["visual_success"] is False
    assert "review_quality_profile_media_below_minimum" in review_report["blockers"]
    assert review_report["provider_completed_visual_quality_failed"] is True
    assert smoke_report["profile_contract"]["smoke_only"] is True
    assert "review_quality_profile_media_below_minimum" not in smoke_report["blockers"]
    assert Path(str(smoke_report["contact_sheet_path"])).is_file()


def test_provider_completed_but_visual_quality_fails_on_dark_generated_frame(
    tmp_path: Path,
) -> None:
    source = _write_good_frame(tmp_path / "source.jpg")
    dark_generated = _write_dark_frame(tmp_path / "generated-dark.jpg")

    report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[dark_generated],
        review_video_path=tmp_path / "review.mp4",
        video_status=_video_status(width=640, height=480, fps="15/1", frames=24),
        visual_profile="review_quality",
        requested_settings={"width": 640, "height": 480, "fps": 15, "num_frames": 24},
        provider_status="completed",
        live_wam_generation_success_count=1,
        learned_wam_model_success_count=1,
    )

    assert report["provider_completed"] is True
    assert report["live_wam_generation_success"] is True
    assert report["visual_success"] is False
    assert report["provider_completed_visual_quality_failed"] is True
    assert "wam_generated_frame_too_dark_for_review" in report["blockers"]


def test_generated_frame_drift_marks_visual_success_false(tmp_path: Path) -> None:
    source = _write_good_frame(tmp_path / "source.jpg")
    first = _write_good_frame(tmp_path / "generated-1.jpg")
    second = _write_dark_frame(tmp_path / "generated-2.jpg")

    report = write_persistent_wam_visual_quality_artifacts(
        job_dir=tmp_path / "job",
        generated_at="now",
        source_frame_path=source,
        generated_frame_paths=[first, second],
        review_video_path=tmp_path / "review.mp4",
        video_status=_video_status(width=640, height=480, fps="15/1", frames=24),
        visual_profile="review_quality",
        requested_settings={"width": 640, "height": 480, "fps": 15, "num_frames": 24},
        provider_status="completed",
        live_wam_generation_success_count=2,
        learned_wam_model_success_count=2,
    )

    assert report["visual_success"] is False
    assert "wam_generated_frame_darkening_drift" in report["blockers"]
    assert (tmp_path / "job" / "wam_rollout_frame_stats.jsonl").is_file()
    rows = [
        json.loads(line)
        for line in (tmp_path / "job" / "wam_rollout_frame_stats.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 2
