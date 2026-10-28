"""A video-only method input from already admitted, immutable site recordings.

This local adapter grants no upload, provider, spending or geometry authority.
Callers must resolve the current same-job source set and its rights first. The
original files are read only; the derivative explicitly retains each original
identity, frame range, timing offset and pixel transform. Independent clips do
not acquire a shared camera frame merely by appearing in one method input.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .capture_audio_minimization import probe_audio_stream_count
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _probe_video, _sha256_file, _tool_identity
from .website_scene_geometry import source_to_view_pixels

SCHEMA = "website_recording_composition.v1"


def _sources(recordings: Sequence[Mapping[str, Any]]) -> tuple[list[Path], list[dict[str, Any]]]:
    paths, sources = [], []
    for row in recordings:
        path = Path(row["source_video"])
        identity, capture = row.get("source_identity"), row.get("capture_id")
        if (path.is_symlink() or not path.is_file() or not isinstance(identity, Mapping) or not identity
                or not isinstance(capture, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,180}", capture)
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(row.get("source_video_digest", "")))):
            raise ValueError("website_recording_source_invalid")
        if _sha256_file(path) != row["source_video_digest"]:
            raise ValueError("website_recording_source_changed")
        paths.append(path.resolve())
        sources.append({"capture_id": capture, "source_identity": dict(identity),
                        "source_video_digest": row["source_video_digest"]})
    if len(paths) < 2 or len({row["capture_id"] for row in sources}) != len(sources) or len(set(paths)) != len(paths):
        raise ValueError("website_recording_sources_invalid")
    return paths, sources


def _layout(probes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    rows = []
    for probe in probes:
        stream = probe["stream"]
        width, height = stream.get("width"), stream.get("height")
        rotation = float(stream.get("display_rotation_degrees") or 0)
        if (type(width) is not int or type(height) is not int or min(width, height) <= 0
                or not math.isfinite(rotation) or rotation % 90):
            raise ValueError("website_recording_dimensions_invalid")
        upright = (height, width) if int(rotation) % 180 else (width, height)
        rows.append({"source_width": width, "source_height": height,
                     "display_rotation_degrees": rotation, "upright_width": upright[0], "upright_height": upright[1]})
    return rows, max(row["upright_width"] for row in rows), max(row["upright_height"] for row in rows)


def composition_command(ffmpeg: str, paths: Sequence[Path], layout: Sequence[Mapping[str, Any]],
                        width: int, height: int, destination: Path) -> list[str]:
    """Keep native pixels and frame cadence; only rotate and pad to one canvas."""
    command = [ffmpeg, "-nostdin", "-y", "-v", "error"]
    for path in paths:
        command.extend(["-noautorotate", "-i", str(path)])
    transforms = {0: [], 90: ["transpose=cclock"], 180: ["hflip", "vflip"], 270: ["transpose=clock"]}
    filters = []
    for index, row in enumerate(layout):
        operations = ["settb=AVTB", "setpts=PTS-STARTPTS", "format=rgb24",
                      *transforms[int(row["display_rotation_degrees"]) % 360],
                      f"pad={width}:{height}:0:0:color=black", "setsar=1"]
        filters.append(f"[{index}:v:0]" + ",".join(operations) + f"[v{index}]")
    filters.append("".join(f"[v{index}]" for index in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0[out]")
    return command + ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-sn", "-dn",
                      "-map_metadata", "-1", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
                      "-threads", "1", "-pix_fmt", "rgb24", "-fps_mode", "passthrough",
                      "-enc_time_base", "1:1000000", "-video_track_timescale", "1000000",
                      "-movflags", "+faststart", str(destination)]


def _segments(probes: Sequence[Mapping[str, Any]], layout: Sequence[Mapping[str, Any]],
              output: Mapping[str, Any]) -> list[dict[str, Any]]:
    times = output["presentation_times_seconds"]
    if len(times) != sum(probe["frame_count"] for probe in probes):
        raise ValueError("website_recording_frame_count_changed")
    cursor, result = 0, []
    for probe, row in zip(probes, layout, strict=True):
        count, start = probe["frame_count"], times[cursor]
        if any(abs((actual - start) - expected) > .0000021 for actual, expected in
               zip(times[cursor:cursor + count], probe["presentation_times_seconds"], strict=True)):
            raise ValueError("website_recording_frame_timing_changed")
        result.append({**row, "output_first_frame_index": cursor, "frame_count": count,
                       "output_start_seconds": start, "source_first_pts_seconds": probe["first_source_pts_seconds"],
                       "source_to_output_pixels": source_to_view_pixels(width=row["source_width"], height=row["source_height"],
                           rotation=row["display_rotation_degrees"], size=(row["upright_width"], row["upright_height"]))})
        cursor += count
    return result


def compose_recordings(*, recordings: Sequence[Mapping[str, Any]], output_root: Path) -> dict[str, Any]:
    """Retain one reusable method input; never rewrite an original or completed derivative.

    ``source_identity`` is caller-supplied provenance, not verified authority.
    This adapter verifies the actual content digests supplied by the caller.
    The caller must separately prove those bytes belong to the current job.
    """
    paths, sources = _sources(recordings)
    ffprobe, ffmpeg, _, version_digest = _tool_identity()
    runtime = {"version_digest": version_digest, "ffmpeg_sha256": _sha256_file(Path(ffmpeg).resolve())}
    binding = {"schema_version": SCHEMA, "sources": sources, "runtime": runtime,
               "method": "native_cadence_lossless_rgb_rotate_pad_concat_v1"}
    key = canonical_digest(binding)
    if output_root.is_symlink():
        raise ValueError("website_recording_output_symlink")
    root = output_root / key[7:]
    if root.is_symlink():
        raise ValueError("website_recording_output_symlink")
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if any(path.is_relative_to(root) for path in paths):
        raise ValueError("website_recording_output_overlaps_source")
    lock = root / "composition.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(fd, "a") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("website_recording_lock_invalid")
        fcntl.flock(handle, fcntl.LOCK_EX)
        manifest, video = root / "composition.json", root / "walkthrough.mp4"
        if manifest.is_symlink() or video.is_symlink():
            raise ValueError("website_recording_output_symlink")
        if manifest.exists():
            value = json.loads(manifest.read_text())
            if (value.get("digest") != canonical_digest(value, digest_field="digest")
                    or value.get("binding") != binding or value.get("input_digest") != key
                    or not video.is_file() or _sha256_file(video) != value.get("source_video_digest")):
                raise ValueError("website_recording_composition_changed")
            return {**value, "source_video": str(video), "manifest_path": str(manifest)}
        probes = [_probe_video(path, ffprobe) for path in paths]
        layout, width, height = _layout(probes)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".composition-", suffix=".mp4", dir=root)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            command = composition_command(ffmpeg, paths, layout, width, height, temporary)
            completed = subprocess.run(command, check=False, capture_output=True, timeout=600)
            if completed.returncode:
                raise ValueError("website_recording_composition_failed")
            output = _probe_video(temporary, ffprobe)
            segments = _segments(probes, layout, output)
            if (output["stream"]["width"] != width or output["stream"]["height"] != height
                    or probe_audio_stream_count(temporary) != 0):
                raise ValueError("website_recording_output_invalid")
            # Recheck after reading/encoding so a concurrent local change never
            # becomes an input for later upload under the earlier source digest.
            if any(_sha256_file(path) != source["source_video_digest"] for path, source in zip(paths, sources, strict=True)):
                raise ValueError("website_recording_source_changed")
            digest = _sha256_file(temporary)
            if video.exists():
                if _sha256_file(video) != digest:
                    raise ValueError("website_recording_composition_incomplete")
            else:
                os.link(temporary, video)
            value = {"schema_version": SCHEMA, "input_digest": key, "binding": binding,
                     "source_video_digest": digest, "width": width, "height": height,
                     "frame_count": output["frame_count"], "audio_stream_count": 0,
                     "sources": [{**source, **segment} for source, segment in zip(sources, segments, strict=True)],
                     "coordinate_frames_independent": True, "geometry_registration_performed": False,
                     "raw_recordings_unchanged": True, "claim_ceiling": "development_only"}
            value["digest"] = canonical_digest(value, digest_field="digest")
            receipt_fd, receipt_name = tempfile.mkstemp(prefix=".receipt-", suffix=".json", dir=root)
            os.close(receipt_fd)
            receipt = Path(receipt_name)
            try:
                write_json(receipt, value)
                receipt.replace(manifest)
            finally:
                receipt.unlink(missing_ok=True)
            return {**value, "source_video": str(video), "manifest_path": str(manifest)}
        finally:
            temporary.unlink(missing_ok=True)
