"""Website clip normalization before the first preparation stage (E7: one 4K clip cannot exhaust a host).

Synthetic clips only, made with ffmpeg's test source in a temporary site directory.
"""
# Covers: src/blueprint_pipeline/website_source_video.py
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import selected_atlas_preview_recovery as recovery
from blueprint_pipeline import website_capture_withdrawal as withdrawal
from blueprint_pipeline import website_source_video as source_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not on PATH")


def _clip(path: Path, *, width: int, height: int, rate: int, seconds: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={width}x{height}:rate={rate}:duration={seconds}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=300)
    return path


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def capture(tmp_path) -> Path:
    root = tmp_path / "scenes/site-fixture/captures/walkthrough-fixture"
    (root / "raw").mkdir(parents=True)
    return root


def test_4k_clip_is_normalized_once_and_the_original_is_kept(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=3840, height=2160, rate=25, seconds=2)
    before = _sha(raw)
    chosen = source_video.preparation_source_video(capture_root=capture, raw_video=raw)
    assert chosen == capture / "pipeline/source_video/normalized.mp4"
    probe = source_video.probe_video(chosen)
    assert (probe["width"], probe["height"]) == (1920, 1080) and probe["fps"] <= 30
    assert probe["frame_count"] == 50 and probe["duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert _sha(raw) == before
    record = json.loads((capture / "pipeline/source_video/normalization.json").read_text())
    assert record["normalization_ran"] is True and record["outcome"] == "normalized"
    assert {"path": "raw/walkthrough.mp4", "width": 3840, "height": 2160, "fps": 25.0, "frame_count": 50,
            "sha256": before}.items() <= record["original"].items()
    assert record["normalized"]["sha256"] == _sha(chosen)
    assert record["preparation_video"] == "pipeline/source_video/normalized.mp4"
    # Every later caller, including one that finds derived artifacts, reads the same file.
    (capture / "pipeline/atlas").mkdir()
    mtime = chosen.stat().st_mtime_ns
    assert source_video.preparation_source_video(capture_root=capture, raw_video=raw) == chosen
    assert chosen.stat().st_mtime_ns == mtime


def test_clip_inside_the_bounds_is_not_reencoded(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=1920, height=1080, rate=30)
    assert source_video.preparation_source_video(capture_root=capture, raw_video=raw) == raw
    record = source_video.read_record(capture)
    assert record["normalization_ran"] is False and record["outcome"] == "within_bounds"
    assert record["normalized"] is None and record["preparation_video"] == "raw/walkthrough.mp4"
    assert not (capture / "pipeline/source_video/normalized.mp4").exists()


def test_portrait_clip_keeps_its_aspect(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=2160, height=3840, rate=25, seconds=0.5)
    probe = source_video.probe_video(source_video.preparation_source_video(capture_root=capture, raw_video=raw))
    assert (probe["width"], probe["height"]) == (1080, 1920)


def test_60_fps_clip_is_capped_at_30_without_cutting_duration(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=1280, height=720, rate=60, seconds=2)
    probe = source_video.probe_video(source_video.preparation_source_video(capture_root=capture, raw_video=raw))
    assert (probe["width"], probe["height"]) == (1280, 720)
    assert probe["fps"] == pytest.approx(30.0) and probe["frame_count"] == 60
    assert probe["duration_seconds"] == pytest.approx(2.0, abs=0.05)


def test_target_dimensions_never_upscale_and_stay_even():
    assert source_video.target_dimensions(1280, 720) == (1280, 720)
    assert source_video.target_dimensions(3840, 2160) == (1920, 1080)
    assert source_video.target_dimensions(4000, 3000) == (1920, 1440)
    assert source_video.target_dimensions(1080, 2340) == (886, 1920)


def test_a_job_that_predates_normalization_keeps_its_original(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=3840, height=2160, rate=25, seconds=0.2)
    (capture / "pipeline/clean_plate").mkdir(parents=True)
    assert source_video.preparation_source_video(capture_root=capture, raw_video=raw) == raw
    assert not (capture / "pipeline/source_video").exists()


def test_unreadable_clip_is_returned_as_is_to_fail_where_it_fails_today(capture):
    raw = capture / "raw/walkthrough.mp4"
    raw.write_bytes(b"not a video")
    assert source_video.preparation_source_video(capture_root=capture, raw_video=raw) == raw
    assert source_video.read_record(capture) is None


def test_withdrawal_plans_list_the_normalized_copy(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=3840, height=2160, rate=25, seconds=0.2)
    source_video.preparation_source_video(capture_root=capture, raw_video=raw)
    withdrawal.acknowledge_withdrawal(capture_root=capture, command={
        "schema_version": "website_capture_withdrawal.v1", "request_id": "fixture", "scene_id": "site-fixture",
        "capture_id": "walkthrough-fixture", "withdrawal_id": "w-1", "requested_at_iso": "2026-11-09T00:00:00Z"})
    # The local cleanup plan is the file list the deletion plan's ``scene_files`` digest seals.
    paths = {row["path"] for row in withdrawal.plan_local_cleanup(capture_root=capture)["files"]}
    assert "captures/walkthrough-fixture/pipeline/source_video/normalized.mp4" in paths
    assert "captures/walkthrough-fixture/raw/walkthrough.mp4" in paths
    # After withdrawal no new copy is made.
    shutil.rmtree(capture / "pipeline/source_video")
    assert source_video.preparation_source_video(capture_root=capture, raw_video=raw) == raw
    assert not (capture / "pipeline/source_video").exists()


def test_atlas_recovery_binds_a_normalized_job_to_its_copy(capture):
    raw = _clip(capture / "raw/walkthrough.mp4", width=3840, height=2160, rate=25, seconds=0.2)
    chosen = source_video.preparation_source_video(capture_root=capture, raw_video=raw)
    row = {"relative_path": "raw/walkthrough.mp4", "sha256": _sha(raw)}
    job = {"source_video": "pipeline/source_video/normalized.mp4"}
    storage = capture.parents[3]
    assert recovery._normalized_source(job, row, capture, storage) == _sha(chosen)
    with pytest.raises(ValueError, match="source_binding_invalid"):
        recovery._normalized_source(job, {**row, "sha256": "sha256:" + "0" * 64}, capture, storage)
    with pytest.raises(ValueError, match="source_binding_invalid"):
        recovery._normalized_source({"source_video": "raw/other.mp4"}, row, capture, storage)
