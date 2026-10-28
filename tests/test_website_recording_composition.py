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
@pytest.mark.parametrize("rotated_first", [False, True])
def test_real_composition_keeps_every_frame_native_timing_pixels_and_originals(tmp_path, media, monkeypatch, rotated_first):
    ffmpeg, ffprobe, rows = media
    if rotated_first:
        rows = [_row(rows[1]["source_video"], "walkthrough-req1"), _row(rows[0]["source_video"], "supplement-" + "a" * 40)]
    original_hashes = [_sha256_file(row["source_video"]) for row in rows]
    value = composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")
    path = Path(value["source_video"])
    output = _probe_video(path, ffprobe)
    assert output["frame_count"] == 12
    assert not output["stream"]["display_rotation_degrees"]
    assert (value["width"], value["height"]) == (96, 80)
    assert value["audio_stream_count"] == 0
    assert value["coordinate_frames_independent"] is True
    assert value["geometry_registration_performed"] is False
    assert [_sha256_file(row["source_video"]) for row in rows] == original_hashes
    first, second = value["sources"]
    counts = (7, 5) if rotated_first else (5, 7)
    assert (first["frame_count"], second["frame_count"]) == counts
    assert second["output_first_frame_index"] == counts[0]
    assert second["display_rotation_degrees"] == (0 if rotated_first else 90)
    assert output["presentation_times_seconds"][counts[0]] == second["output_start_seconds"]
    # Independently rotate/pad each decoded original with Pillow. This checks
    # the declared transform against actual pixels, not the encoder command.
    from PIL import Image
    expected = []
    for row, segment in zip(rows, value["sources"], strict=True):
        decoded = _run([ffmpeg, "-v", "error", "-noautorotate", "-i", str(row["source_video"]), "-map", "0:v:0",
                        "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-f", "rawvideo", "-"]).stdout
        size = (segment["source_width"], segment["source_height"])
        stride = size[0] * size[1] * 3
        assert len(decoded) == stride * segment["frame_count"]
        for offset in range(0, len(decoded), stride):
            image = Image.frombytes("RGB", size, decoded[offset:offset + stride]).rotate(segment["display_rotation_degrees"], expand=True)
            canvas = Image.new("RGB", (96, 80), "black")
            canvas.paste(image, (segment["padding_left"], segment["padding_top"]))
            expected.append(canvas.tobytes())
    actual = _run([ffmpeg, "-v", "error", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-pix_fmt", "rgb24",
                   "-fps_mode", "passthrough", "-f", "rawvideo", "-"]).stdout
    assert actual == b"".join(expected)
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


@pytest.mark.slow
def test_mixed_orientation_views_are_not_rejected_for_the_carriers_black_padding(tmp_path, media, monkeypatch):
    import cv2
    import numpy as np
    from PIL import Image
    from blueprint_pipeline import atlas_video_inputs as video, atlas_video_extension as extension
    _, _, rows = media
    result = composition.compose_recordings(recordings=rows, output_root=tmp_path / "method")
    source = Path(result["source_video"])
    survey_root = tmp_path / "survey"
    survey = video.survey_video(source_video=source, output_root=survey_root, capture_id="current", sample_count=4)
    assert {row["recording_capture_id"] for row in survey["frames"]} == {row["capture_id"] for row in rows}
    for frame in survey["frames"]:
        with Image.open(frame["source_image_path"]) as image:
            content = np.asarray(image.crop(frame["source_content_bounds"]).convert("L"))
            expected = round(float(np.mean((content < 5) | (content > 250))), 4)
            assert frame["clipped_fraction"] == expected
    portrait = next(frame for frame in survey["frames"] if frame["recording_capture_id"] == rows[1]["capture_id"])
    with Image.open(portrait["source_image_path"]) as image:
        full = np.asarray(image.convert("L"))
        assert float(np.mean((full < 5) | (full > 250))) > portrait["clipped_fraction"]

    masks = []
    class Features:
        def detectAndCompute(self, _image, mask):
            masks.append(mask.copy())
            return None, None
    monkeypatch.setattr(cv2, "SIFT_create", lambda **_: Features())
    video._features(portrait, 0)
    left, top, right, bottom = portrait["source_content_bounds"]
    assert np.count_nonzero(masks[0]) == (right - left) * (bottom - top)
    assert np.all(masks[0][top:bottom, left:right] == 255)

    index = next((survey_root / "source").rglob("decoded_observation_index.json"))
    review = {"schema_version": video.REVIEW_SCHEMA, "survey_digest": survey["digest"], "moving_people_excluded": True,
              "same_scene_state": True, "static_intervals_seconds": [[0, 2]]}
    extended = extension.extend_survey(source_video=source, base_survey=survey_root / "atlas_video_survey.json",
        decoded_index=index, review=review, output_root=tmp_path / "extended", step_seconds=.1, gap_threshold_seconds=.15)
    assert all(frame["quality_basis"] == "original_content_excluding_carrier_padding" for frame in extended["frames"])
    assert extended["heldout_pixels_included"] is False
