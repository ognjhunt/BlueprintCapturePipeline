"""The one video file website preparation stages read for a capture.

Exit criterion E7 of the invited-beta plan: one 4K clip cannot exhaust an
8 GB host. A 3840x2160 walkthrough was killed for memory in SAM (about 4 GB)
and in the Atlas preview (4.9-6.5 GB) while the intake unit is capped at 3 GB.
Each stage was fixed on its own; nothing bounded the clip itself.

Every new website intake is therefore normalized once, before the first
preparation stage, to a long edge of at most ``MAX_LONG_EDGE`` pixels and at
most ``MAX_FPS`` frames a second:

- downscale only: never upscale, keep the aspect ratio (each side rounded to an
  even pixel count, which H.264 needs), keep the whole duration and the frame
  timestamps; only frames above 30 a second are dropped;
- a clip already inside both bounds is not re-encoded: it is read as uploaded;
- the original is never touched. ``normalization.json`` beside the copy records
  the original's and the normalized clip's width, height, fps, frame count,
  duration and digest, and that normalization ran;
- the copy lives in the capture's own ``pipeline/source_video/``, inside
  ``scenes/site-<id>/``, so a consent withdrawal deletes it with the rest of the
  site, and a retired workspace archives and restores it like every other
  pipeline output.

``preparation_source_video`` is the only answer to "which file do website
preparation stages read": the orchestrator resolves it once and hands the same
path to the Atlas preview job and to clean-plate preparation, which passes it on
to every later stage. A job that already holds derived artifacts (an Atlas
preview job or clean-plate outputs) and no record keeps reading the file it
started with; nothing in flight changes its input.

There is no size gate. A clip ffprobe cannot read is returned unchanged, so it
fails in the first stage exactly as an unreadable clip fails today. A readable
clip whose encode fails raises, so the run retries instead of handing a 4K clip
to the stages this exists to protect.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from .core.handoff_lease import guarded_handoff_write_json, handoff_effect, handoff_lease_checkpoint

SCHEMA = "website_source_video_normalization.v1"
MAX_LONG_EDGE = 1920
MAX_FPS = 30
#: 30000/1001 and rounding noise in an average rate are inside the bound.
_FPS_TOLERANCE = 0.01
DIRECTORY = "source_video"
RECORD_FILE = "normalization.json"
NORMALIZED_FILE = "normalized.mp4"
#: Outputs of the first stages that read the video. Their presence without a
#: record means the job started before normalization existed.
DERIVED_ARTIFACT_DIRS = ("atlas", "clean_plate")
ENCODER = {"codec": "libx264", "preset": "veryfast", "crf": 18, "pix_fmt": "yuv420p", "threads": 2,
           "scale_flags": "lanczos", "audio": "aac_128k"}
_ENCODE_TIMEOUT_SECONDS = 3600


def normalization_root(capture_root: Path) -> Path:
    return Path(capture_root) / "pipeline" / DIRECTORY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _rate(value: Any) -> float | None:
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def probe_video(path: Path) -> dict[str, Any]:
    """Displayed width and height, average fps, frame count and duration of the first video stream."""

    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", ("stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets,duration"
                           ":stream_tags=rotate:stream_side_data=rotation:format=duration"),
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=600)
    value = json.loads(completed.stdout or "{}")
    streams = value.get("streams") or []
    if not streams:
        raise ValueError("website_source_video_stream_missing")
    stream = streams[0]
    width, height = stream.get("width"), stream.get("height")
    if not all(isinstance(side, int) and side > 0 for side in (width, height)):
        raise ValueError("website_source_video_dimensions_unreadable")
    rotation = _number((stream.get("tags") or {}).get("rotate"))
    for side_data in stream.get("side_data_list") or []:
        if rotation is None and isinstance(side_data, dict):
            rotation = _number(side_data.get("rotation"))
    rotation = rotation or 0.0
    if round(rotation) % 180 == 90:
        # A phone that recorded portrait stores landscape pixels plus a display rotation.
        width, height = height, width
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    if fps is None:
        raise ValueError("website_source_video_frame_rate_unreadable")
    duration = _number(stream.get("duration")) or _number((value.get("format") or {}).get("duration"))
    frames = stream.get("nb_read_packets")
    return {"width": width, "height": height, "fps": round(fps, 6),
            "frame_count": int(frames) if str(frames or "").isdigit() else None,
            "duration_seconds": round(duration, 6) if duration is not None else None,
            "display_rotation_degrees": rotation}


def target_dimensions(width: int, height: int) -> tuple[int, int]:
    """The downscaled size: long edge at most ``MAX_LONG_EDGE``, same aspect, even sides; never larger."""

    long_edge = max(width, height)
    if long_edge <= MAX_LONG_EDGE:
        return width, height
    scale = MAX_LONG_EDGE / long_edge

    def even(side: int) -> int:
        return max(2, round(side * scale / 2) * 2)

    return (MAX_LONG_EDGE, even(height)) if width >= height else (even(width), MAX_LONG_EDGE)


def within_bounds(probe: dict[str, Any]) -> bool:
    return max(probe["width"], probe["height"]) <= MAX_LONG_EDGE and probe["fps"] <= MAX_FPS + _FPS_TOLERANCE


def normalize_video(source: Path, target: Path, probe: dict[str, Any] | None = None) -> dict[str, Any]:
    """Encode ``source`` within the bounds to ``target``; ``source`` is only read.

    ffmpeg applies the display rotation to the pixels (its default), so the copy
    is upright with no rotation tag and every stage decodes it the same way.
    """

    probe = probe or probe_video(source)
    width, height = target_dimensions(probe["width"], probe["height"])
    filters = []
    if (width, height) != (probe["width"], probe["height"]):
        filters.append(f"scale={width}:{height}:flags={ENCODER['scale_flags']}")
    capped = probe["fps"] > MAX_FPS + _FPS_TOLERANCE
    if capped:
        filters.append(f"fps={MAX_FPS}")
    command = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", str(ENCODER["threads"]),
               "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "0",
               "-filter_threads", "1"]
    if filters:
        command += ["-vf", ",".join(filters)]
    # Without a frame-rate cap every source frame keeps its own timestamp.
    command += ["-fps_mode", "cfr" if capped else "passthrough",
                "-c:v", ENCODER["codec"], "-preset", ENCODER["preset"], "-crf", str(ENCODER["crf"]),
                "-pix_fmt", ENCODER["pix_fmt"], "-threads", str(ENCODER["threads"]),
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True, capture_output=True, timeout=_ENCODE_TIMEOUT_SECONDS)
    return probe_video(target)


def _relative(path: Path, capture_root: Path) -> str:
    resolved, root = Path(path).resolve(), Path(capture_root).resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def _describe(path: Path, probe: dict[str, Any], capture_root: Path) -> dict[str, Any]:
    return {"path": _relative(path, capture_root), "size_bytes": Path(path).stat().st_size,
            "sha256": _sha256(path), **probe}


def read_record(capture_root: Path) -> dict[str, Any] | None:
    path = normalization_root(capture_root) / RECORD_FILE
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get("schema_version") != SCHEMA:
        raise ValueError("website_source_video_record_invalid")
    return value


def _recorded_video(record: dict[str, Any], capture_root: Path, raw_video: Path) -> Path:
    original = record.get("original") or {}
    if original.get("path") != _relative(raw_video, capture_root) or \
            original.get("size_bytes") != raw_video.stat().st_size:
        raise ValueError("website_source_video_original_changed")
    chosen = Path(record["preparation_video"])
    chosen = chosen if chosen.is_absolute() else Path(capture_root) / chosen
    if not chosen.is_file():
        raise ValueError("website_source_video_normalized_missing")
    return chosen


def preparation_source_video(*, capture_root: Path, raw_video: Path | None) -> Path | None:
    """The file every website preparation stage reads for this capture (see the module docstring)."""

    if raw_video is None:
        return None
    capture_root, raw_video = Path(capture_root), Path(raw_video)
    if not raw_video.is_file():
        return raw_video
    record = read_record(capture_root)
    if record is not None:
        return _recorded_video(record, capture_root, raw_video)
    pipeline = capture_root / "pipeline"
    if any((pipeline / name).exists() for name in DERIVED_ARTIFACT_DIRS):
        return raw_video
    from .website_atlas_preview_store import capture_withdrawn
    if capture_withdrawn(capture_root):
        # No new copy of a withdrawn site's footage; the stages refuse as before.
        return raw_video
    root = normalization_root(capture_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        record = read_record(capture_root)
        if record is not None:
            return _recorded_video(record, capture_root, raw_video)
        try:
            original_probe = probe_video(raw_video)
        except (subprocess.SubprocessError, OSError, ValueError):
            # Unreadable: the first stage fails on it exactly as before.
            return raw_video
        original = _describe(raw_video, original_probe, capture_root)
        record = {"schema_version": SCHEMA, "bounds": {"max_long_edge": MAX_LONG_EDGE, "max_fps": MAX_FPS},
                  "original": original, "recorded_at_epoch": int(time.time())}
        if within_bounds(original_probe):
            record.update(normalization_ran=False, outcome="within_bounds", normalized=None, encoder=None,
                          preparation_video=original["path"])
            guarded_handoff_write_json(root / RECORD_FILE, record)
            return raw_video
        handoff_lease_checkpoint("website_source_video_before_normalize")
        target = root / NORMALIZED_FILE
        started = time.monotonic()
        with tempfile.TemporaryDirectory(dir=root, prefix=".normalizing-") as scratch:
            staged = Path(scratch) / NORMALIZED_FILE
            try:
                normalized_probe = normalize_video(raw_video, staged, original_probe)
            except subprocess.CalledProcessError as error:
                raise ValueError("website_source_video_normalization_failed") from error
            with handoff_effect("website_source_video_normalized"):
                os.replace(staged, target)
        if _sha256(raw_video) != original["sha256"]:
            raise ValueError("website_source_video_original_changed")
        record.update(normalization_ran=True, outcome="normalized",
                      normalized=_describe(target, normalized_probe, capture_root),
                      encoder=dict(ENCODER), encode_seconds=round(time.monotonic() - started, 3),
                      preparation_video=_relative(target, capture_root))
        guarded_handoff_write_json(root / RECORD_FILE, record)
        return target
