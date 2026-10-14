"""Capture audio minimization: video-only derivatives, truthful receipts.

Website free-beta privacy minimization (ADP-010 site-submitted evidence,
ADP-021 capture privacy contract). The hermetic tests pin the exact encoder
command and the receipt logic with fake media tools; the one real-ffmpeg test
is in the slow lane.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import capture_audio_minimization as cam
from blueprint_pipeline.common import sha256_file
from tests.capture_audio_fakes import VIDEO_ONLY_MARKER, FakeMediaTools


def _source(tmp_path: Path, name: str = "walkthrough.mov", payload: bytes = b"raw-with-audio") -> Path:
    path = tmp_path / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_video_only_copy_drops_audio_and_never_touches_the_source(monkeypatch, tmp_path: Path) -> None:
    tools = FakeMediaTools().install(monkeypatch)
    source = _source(tmp_path)
    destination = tmp_path / "privacy" / "final_walkthrough.mov"

    receipt = cam.write_audio_free_video(source, destination)

    assert source.read_bytes() == b"raw-with-audio"
    assert destination.read_bytes() == VIDEO_ONLY_MARKER + b"raw-with-audio"
    [command] = tools.ffmpeg_commands()
    assert command[command.index("-i") + 1] == str(source)
    assert command[-1] == str(destination)
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-c:v") + 1] == "copy"
    assert "-an" in command
    assert not any(item.startswith("-c:a") or item.endswith(":a") or ":a?" in item for item in command)
    assert receipt["schema_version"] == "capture_audio_removal.v1"
    assert receipt["status"] == "removed"
    assert receipt["audio_removed"] is True
    assert receipt["evidence"] == "ffprobe_output_has_no_audio_stream"
    assert receipt["output_audio_stream_count"] == 0
    assert receipt["output_sha256"] == sha256_file(destination)
    # Receipts carry the command shape, not scratch paths.
    assert receipt["command"][0] == "ffmpeg"
    assert "{source}" in receipt["command"] and "{output}" in receipt["command"]
    assert str(tmp_path) not in " ".join(receipt["command"])


def test_without_ffprobe_the_receipt_rests_on_the_exact_command(monkeypatch, tmp_path: Path) -> None:
    FakeMediaTools(ffprobe=False).install(monkeypatch)
    receipt = cam.write_audio_free_video(_source(tmp_path), tmp_path / "out.mov")

    assert receipt["audio_removed"] is True
    assert receipt["evidence"] == "encoder_command_disables_audio"
    assert receipt["output_audio_stream_count"] is None


def test_audio_found_after_encode_is_never_reported_removed(monkeypatch, tmp_path: Path) -> None:
    FakeMediaTools(output_audio_streams=1).install(monkeypatch)
    destination = tmp_path / "out.mov"

    receipt = cam.write_audio_free_video(_source(tmp_path), destination)

    assert receipt["audio_removed"] is False
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "audio_stream_present_after_encode"
    assert receipt["output_audio_stream_count"] == 1
    assert not destination.exists()


@pytest.mark.parametrize(
    ("tools", "reason"),
    [
        (FakeMediaTools(ffmpeg=False), "ffmpeg_not_found"),
        (FakeMediaTools(ffmpeg_returncode=1), "audio_removal_encoder_failed:1"),
    ],
)
def test_failed_removal_leaves_no_copy_behind(monkeypatch, tmp_path: Path, tools, reason) -> None:
    tools.install(monkeypatch)
    source = _source(tmp_path)
    destination = tmp_path / "out.mov"
    destination.write_bytes(b"stale copy that may carry audio")

    receipt = cam.write_audio_free_video(source, destination)

    assert receipt["audio_removed"] is False
    assert receipt["reason"] == reason
    assert not destination.exists()
    assert source.read_bytes() == b"raw-with-audio"


def test_a_command_that_keeps_audio_is_not_a_removal(monkeypatch, tmp_path: Path) -> None:
    FakeMediaTools(ffprobe=False).install(monkeypatch)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"encoded")

    receipt = cam.audio_removal_receipt(
        command=["ffmpeg", "-i", "in.mov", "-c:a", "aac", str(output)],
        returncode=0,
        output=output,
        method="test",
    )

    assert receipt["audio_removed"] is False
    assert receipt["evidence"] == "encoder_command_keeps_audio"


def test_video_only_encoder_receipt_uses_the_probe_when_present(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "frames.mov"
    output.write_bytes(b"frames")

    FakeMediaTools(ffprobe=False).install(monkeypatch)
    assert cam.video_only_encoder_receipt(output, encoder="opencv_video_writer")["evidence"] == (
        "encoder_has_no_audio_input"
    )
    FakeMediaTools(output_audio_streams=1).install(monkeypatch)
    assert cam.video_only_encoder_receipt(output, encoder="opencv_video_writer")["audio_removed"] is False
    assert cam.video_only_encoder_receipt(tmp_path / "missing.mov", encoder="opencv_video_writer")[
        "reason"
    ] == "video_only_output_missing"


def test_upload_copy_is_temporary_video_only_and_keeps_the_name(monkeypatch, tmp_path: Path) -> None:
    FakeMediaTools().install(monkeypatch)
    source = _source(tmp_path, "walkthrough.mp4")

    with cam.audio_free_upload_copy(source) as (copy, receipt):
        assert copy != source
        assert copy.name == "walkthrough.mp4"
        assert copy.read_bytes() == VIDEO_ONLY_MARKER + b"raw-with-audio"
        assert receipt["audio_removed"] is True
    assert not copy.exists()

    FakeMediaTools(ffmpeg=False).install(monkeypatch)
    with pytest.raises(cam.AudioRemovalError, match="ffmpeg_not_found") as error:
        with cam.audio_free_upload_copy(source):
            raise AssertionError("nothing may be uploaded without a video-only copy")
    assert error.value.receipt["audio_removed"] is False


def test_in_memory_bytes_are_replaced_by_video_only_bytes(monkeypatch, tmp_path: Path) -> None:
    FakeMediaTools().install(monkeypatch)
    data, receipt = cam.audio_free_video_bytes(b"bytes-with-audio", suffix="mov")
    assert data == VIDEO_ONLY_MARKER + b"bytes-with-audio"
    assert receipt["audio_removed"] is True


@pytest.mark.slow
def test_real_ffmpeg_strips_audio_and_preserves_the_raw_capture(tmp_path: Path) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe not installed")
    source = tmp_path / "raw" / "walkthrough.mov"
    source.parent.mkdir()
    made = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=96x64:rate=5",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
         "-t", "1", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", str(source)],
        capture_output=True, text=True, check=False,
    )
    if made.returncode != 0:
        pytest.skip(f"ffmpeg test source failed: {made.stderr[-200:]}")
    assert cam.probe_audio_stream_count(source) == 1
    source_digest = sha256_file(source)

    destination = tmp_path / "privacy" / "final_walkthrough.mov"
    receipt = cam.write_audio_free_video(source, destination)

    assert receipt["audio_removed"] is True
    assert receipt["evidence"] == "ffprobe_output_has_no_audio_stream"
    assert cam.probe_audio_stream_count(destination) == 0
    video_streams = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(destination)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert len(video_streams) == 1
    assert sha256_file(source) == source_digest
    assert cam.probe_audio_stream_count(source) == 1
