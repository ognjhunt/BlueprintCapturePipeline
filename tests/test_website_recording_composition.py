"""Short-view method input: exact originals, timing, audio and retained reuse."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from blueprint_pipeline import website_recording_composition as composition
from blueprint_pipeline.local_reconstruction_adapters import _probe_video, _sha256_file


def _run(command):
    return subprocess.run(command, check=True, capture_output=True, timeout=30)


def _row(path, capture):
    return {"capture_id": capture, "source_video": path, "source_video_digest": _sha256_file(path),
            "source_identity": {"object_name": f"scenes/site-req1/captures/{capture}/raw/walkthrough.mp4",
                                "generation": "123", "size_bytes": path.stat().st_size, "crc32c": "synthetic"}}


@pytest.fixture
def media(tmp_path):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("real FFmpeg media lane is not installed")
    first, second, rotated = [tmp_path / name for name in ("original.mp4", "extra.mp4", "rotated.mp4")]
    _run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=5:duration=1",
          "-f", "lavfi", "-i", "sine=frequency=400:duration=1", "-c:v", "libx264rgb", "-crf", "0",
          "-c:a", "aac", "-shortest", str(first)])
    _run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=80x48:rate=7:duration=1",
          "-c:v", "libx264rgb", "-crf", "0", str(second)])
    _run([ffmpeg, "-v", "error", "-y", "-display_rotation:v:0", "90", "-i", str(second), "-c", "copy", str(rotated)])
    assert _probe_video(rotated, ffprobe)["stream"]["display_rotation_degrees"] == 90
    return ffmpeg, ffprobe, [_row(first, "walkthrough-req1"), _row(rotated, "supplement-" + "a" * 40)]


@pytest.mark.slow
def test_real_composition_keeps_every_frame_native_timing_pixels_and_originals(tmp_path, media, monkeypatch):
    ffmpeg, ffprobe, rows = media
    original_hashes = [_sha256_file(row["source_video"]) for row in rows]
    value = composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")
    path = Path(value["source_video"])
    output = _probe_video(path, ffprobe)
    assert output["frame_count"] == 12
    assert (value["width"], value["height"]) == (96, 80)
    assert value["audio_stream_count"] == 0
    assert value["coordinate_frames_independent"] is True
    assert value["geometry_registration_performed"] is False
    assert [_sha256_file(row["source_video"]) for row in rows] == original_hashes
    first, second = value["sources"]
    assert (first["frame_count"], second["frame_count"]) == (5, 7)
    assert second["output_first_frame_index"] == 5
    assert second["display_rotation_degrees"] == 90
    assert output["presentation_times_seconds"][5] == second["output_start_seconds"]
    # Compare decoded RGB pixels to the declared rotation/padding of each raw
    # source, including the join. This catches re-encoding loss or dropped views.
    layout = [{key: row[key] for key in ("display_rotation_degrees",)} for row in value["sources"]]
    command = composition.composition_command(ffmpeg, [row["source_video"] for row in rows], layout, 96, 80, path)
    prefix = command[:command.index("-c:v")]
    expected = _run(prefix + ["-c:v", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-f", "framemd5", "-"]).stdout
    actual = _run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "rawvideo", "-pix_fmt", "rgb24",
                   "-fps_mode", "passthrough", "-f", "framemd5", "-"]).stdout
    def hashes(data):
        return [line.split(b",")[-1].strip() for line in data.splitlines() if not line.startswith(b"#")]
    assert hashes(expected) == hashes(actual)
    # Completed inputs reopen without encoding; their bytes and provenance stay
    # stable even if another operation is temporarily unavailable.
    monkeypatch.setattr(composition, "_probe_video", lambda *_: pytest.fail("completed composition was decoded again"))
    assert composition.compose_recordings(recordings=rows, output_root=tmp_path / "method") == value
    stored = json.loads(Path(value["manifest_path"]).read_text())
    assert "source_video" not in stored and "manifest_path" not in stored


@pytest.mark.slow
@pytest.mark.parametrize("fault", ["changed_original", "changed_method", "changed_receipt"])
def test_changed_inputs_or_retained_derivative_never_reopen_as_the_old_input(tmp_path, media, fault):
    _, _, rows = media
    value = composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")
    if fault == "changed_original":
        rows[0]["source_video"].write_bytes(b"changed source")
    elif fault == "changed_method":
        Path(value["source_video"]).write_bytes(b"changed method input")
    else:
        Path(value["manifest_path"]).write_text("{}")
    with pytest.raises(ValueError, match="website_recording_(source_changed|composition_changed)"):
        composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")


def test_bad_source_refuses_before_media_tools_or_output_creation(tmp_path, monkeypatch):
    path = tmp_path / "raw.mov"
    path.write_bytes(b"original")
    row = _row(path, "walkthrough-req1")
    row["source_video_digest"] = "sha256:" + "0" * 64
    monkeypatch.setattr(composition, "_tool_identity", lambda: pytest.fail("media tool invoked for changed source"))
    with pytest.raises(ValueError, match="source_changed"):
        composition.compose_recordings(recordings=[row, row], output_root=tmp_path / "method")
    assert not (tmp_path / "method").exists()


def test_source_symlink_cannot_become_a_method_input(tmp_path):
    raw = tmp_path / "raw.mov"
    raw.write_bytes(b"raw")
    link = tmp_path / "source.mov"
    link.symlink_to(raw)
    with pytest.raises(ValueError, match="source_invalid"):
        composition.compose_recordings(recordings=[_row(link, "walkthrough-req1")], output_root=tmp_path / "method")


@pytest.mark.slow
def test_an_interrupted_receipt_write_reuses_the_same_derivative_without_overwriting_it(tmp_path, media):
    _, _, rows = media
    original = composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")
    video = Path(original["source_video"])
    before = video.stat()
    Path(original["manifest_path"]).unlink()
    resumed = composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")
    assert resumed == original
    assert (video.stat().st_ino, video.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
