"""Capture-orientation normalization and ffprobe inspection."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import try_parse_float

def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _normalize_rotation_degrees(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        degrees = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    normalized = degrees % 360
    for candidate in (0, 90, 180, 270):
        if abs(normalized - candidate) <= 1:
            return candidate
    return normalized


def _size_payload(width: Optional[int], height: Optional[int]) -> Dict[str, int]:
    if width is None or height is None or width <= 0 or height <= 0:
        return {}
    return {"width": int(width), "height": int(height)}


def _infer_display_orientation(width: Optional[int], height: Optional[int]) -> str:
    if width is None or height is None or width <= 0 or height <= 0:
        return "unknown"
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _display_size_from_rotation(
    encoded_width: Optional[int],
    encoded_height: Optional[int],
    rotation_degrees: Optional[int],
) -> Dict[str, int]:
    if encoded_width is None or encoded_height is None:
        return {}
    if rotation_degrees in {90, 270}:
        return _size_payload(encoded_height, encoded_width)
    return _size_payload(encoded_width, encoded_height)


def _declared_capture_dimensions(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[Optional[int], Optional[int]]:
    width = _first_int(
        manifest.get("declared_capture_width"),
        manifest.get("declaredCaptureWidth"),
        manifest.get("capture_width"),
        manifest.get("captureWidth"),
        context.get("declaredCaptureWidth"),
        context.get("declared_capture_width"),
        context.get("captureWidth"),
        context.get("capture_width"),
        context.get("displayWidth"),
        context.get("display_width"),
    )
    height = _first_int(
        manifest.get("declared_capture_height"),
        manifest.get("declaredCaptureHeight"),
        manifest.get("capture_height"),
        manifest.get("captureHeight"),
        context.get("declaredCaptureHeight"),
        context.get("declared_capture_height"),
        context.get("captureHeight"),
        context.get("capture_height"),
        context.get("displayHeight"),
        context.get("display_height"),
    )
    return width, height


def _orientation_payload(
    *,
    encoded_width: Optional[int],
    encoded_height: Optional[int],
    declared_capture_width: Optional[int],
    declared_capture_height: Optional[int],
    display_rotation_degrees: int,
    display_orientation: str,
    normalization_applied: bool,
    source: str,
    probe_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    encoded_size = _size_payload(encoded_width, encoded_height)
    display_size = _display_size_from_rotation(
        encoded_width,
        encoded_height,
        display_rotation_degrees,
    )
    if declared_capture_width and declared_capture_height:
        display_size = (
            _size_payload(declared_capture_width, declared_capture_height) or display_size
        )
    if not display_orientation:
        display_orientation = _infer_display_orientation(
            display_size.get("width"),
            display_size.get("height"),
        )
    payload: Dict[str, Any] = {
        "encoded_width": int(encoded_width or 0),
        "encoded_height": int(encoded_height or 0),
        "declared_capture_width": int(declared_capture_width or 0),
        "declared_capture_height": int(declared_capture_height or 0),
        "display_rotation_degrees": int(display_rotation_degrees),
        "display_orientation": display_orientation or "unknown",
        "normalization_applied": bool(normalization_applied),
        "source": source,
        "rotation_degrees": int(display_rotation_degrees),
        "display_size": display_size,
        "encoded_size": encoded_size,
        "preserve_original_display_orientation": True,
    }
    if probe_details:
        payload["probe_details"] = dict(probe_details)
    return payload


def _raw_orientation_mapping(raw_value: Any) -> Dict[str, Any]:
    if not isinstance(raw_value, Mapping):
        return {}
    return dict(raw_value)


def _capture_orientation_from_metadata(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = [
        (
            "capture_context.captureOrientation",
            _raw_orientation_mapping(context.get("captureOrientation")),
        ),
        (
            "capture_context.capture_orientation",
            _raw_orientation_mapping(context.get("capture_orientation")),
        ),
        (
            "raw_manifest.capture_orientation",
            _raw_orientation_mapping(manifest.get("capture_orientation")),
        ),
        (
            "raw_manifest.captureOrientation",
            _raw_orientation_mapping(manifest.get("captureOrientation")),
        ),
    ]
    encoded_width = _first_int(
        manifest.get("width"),
        context.get("width"),
        context.get("sourceVideoWidth"),
        context.get("videoWidth"),
    )
    encoded_height = _first_int(
        manifest.get("height"),
        context.get("height"),
        context.get("sourceVideoHeight"),
        context.get("videoHeight"),
    )
    declared_capture_width, declared_capture_height = _declared_capture_dimensions(
        manifest=manifest,
        context=context,
    )
    for source, raw in candidates:
        if not raw:
            continue
        display_orientation = (
            str(raw.get("display_orientation") or raw.get("displayOrientation") or "")
            .strip()
            .lower()
        )
        rotation_degrees = _normalize_rotation_degrees(
            raw.get("rotation_degrees") or raw.get("rotationDegrees")
        )
        display_width = _first_int(
            ((raw.get("display_size") or raw.get("displaySize")) or {}).get("width")
            if isinstance(raw.get("display_size") or raw.get("displaySize"), Mapping)
            else None,
            raw.get("display_width"),
            raw.get("displayWidth"),
        )
        display_height = _first_int(
            ((raw.get("display_size") or raw.get("displaySize")) or {}).get("height")
            if isinstance(raw.get("display_size") or raw.get("displaySize"), Mapping)
            else None,
            raw.get("display_height"),
            raw.get("displayHeight"),
        )
        if not display_orientation and display_width and display_height:
            display_orientation = _infer_display_orientation(display_width, display_height)
        if not display_orientation and rotation_degrees is not None:
            display_orientation = _infer_display_orientation(
                *(
                    (
                        encoded_height,
                        encoded_width,
                    )
                    if rotation_degrees in {90, 270}
                    else (encoded_width, encoded_height)
                )
            )
        if (
            not display_orientation
            and rotation_degrees is None
            and not display_width
            and not display_height
        ):
            continue
        display_size = _size_payload(display_width, display_height) or _display_size_from_rotation(
            encoded_width, encoded_height, rotation_degrees
        )
        if not display_orientation:
            display_orientation = _infer_display_orientation(
                display_size.get("width"),
                display_size.get("height"),
            )
        metadata_source = "capture_context" if source.startswith("capture_context") else "manifest"
        return _orientation_payload(
            encoded_width=encoded_width,
            encoded_height=encoded_height,
            declared_capture_width=declared_capture_width or display_size.get("width"),
            declared_capture_height=declared_capture_height or display_size.get("height"),
            display_rotation_degrees=rotation_degrees if rotation_degrees is not None else 0,
            display_orientation=display_orientation or "unknown",
            normalization_applied=bool(rotation_degrees if rotation_degrees is not None else 0),
            source=metadata_source,
        )
    return {}


def _stream_rotation_degrees(stream: Mapping[str, Any]) -> int:
    tags = stream.get("tags") if isinstance(stream.get("tags"), Mapping) else {}
    side_data_list = (
        stream.get("side_data_list") if isinstance(stream.get("side_data_list"), list) else []
    )
    rotation_candidates: List[Any] = [tags.get("rotate") if isinstance(tags, Mapping) else None]
    for item in side_data_list:
        if isinstance(item, Mapping):
            rotation_candidates.append(item.get("rotation"))
    return next(
        (
            normalized
            for normalized in (
                _normalize_rotation_degrees(candidate) for candidate in rotation_candidates
            )
            if normalized is not None
        ),
        0,
    )


def _positive_number(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            number = float(numerator) / float(denominator)
        else:
            number = float(text)
    except (ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _ffprobe_video_metadata(video_path: Path) -> Dict[str, float]:
    """Width and height as displayed, frame rate and duration, read off the file.

    For a manifest that cannot say: a browser upload whose browser could not
    play the video records these as null rather than guessing. Returns only the
    values ffprobe reported, and ``{}`` when it is missing or rejects the file.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(video_path),
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    stream = next(
        (
            item
            for item in (streams if isinstance(streams, list) else [])
            if isinstance(item, Mapping) and item.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(stream, Mapping):
        return {}
    format_info = payload.get("format") if isinstance(payload.get("format"), Mapping) else {}
    display = _display_size_from_rotation(
        _first_int(stream.get("width")),
        _first_int(stream.get("height")),
        _stream_rotation_degrees(stream),
    )
    measured = {
        "width": _positive_number(display.get("width")),
        "height": _positive_number(display.get("height")),
        "fps_source": _positive_number(stream.get("avg_frame_rate"))
        or _positive_number(stream.get("r_frame_rate")),
        "duration_seconds": _positive_number(format_info.get("duration"))
        or _positive_number(stream.get("duration")),
    }
    return {key: value for key, value in measured.items() if value is not None}


def _video_metadata(
    raw_root: Path, manifest: Mapping[str, Any], video_candidates: List[str]
) -> Dict[str, Any]:
    """The manifest's video values, measured off the file where it has none.

    A browser upload may legitimately record width, height or frame rate as
    null: the browser that sent it could not play the video. The file itself is
    the authority then. ``value_sources`` names where each value came from, and
    a value nobody could read stays ``0.0`` with source ``None``.
    """
    metadata: Dict[str, Any] = {
        key: try_parse_float(manifest.get(key), 0.0)
        for key in ("width", "height", "fps_source", "capture_start_epoch_ms")
    }
    sources: Dict[str, Optional[str]] = {
        key: "raw_manifest" if value > 0 else None for key, value in metadata.items()
    }
    missing = [key for key in ("width", "height", "fps_source") if not sources[key]]
    if missing and video_candidates:
        measured = _ffprobe_video_metadata(raw_root / video_candidates[0])
        for key in missing:
            if key in measured:
                metadata[key] = measured[key]
                sources[key] = "ffprobe_video_file"
    metadata["value_sources"] = sources
    return metadata


def _ffprobe_capture_orientation(video_path: Path) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(video_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return {}
    stream = next(
        (
            item
            for item in streams
            if isinstance(item, Mapping) and item.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(stream, Mapping):
        return {}
    encoded_width = _first_int(stream.get("width"))
    encoded_height = _first_int(stream.get("height"))
    rotation_degrees = _stream_rotation_degrees(stream)
    display_size = _display_size_from_rotation(encoded_width, encoded_height, rotation_degrees)
    return _orientation_payload(
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        declared_capture_width=display_size.get("width"),
        declared_capture_height=display_size.get("height"),
        display_rotation_degrees=rotation_degrees,
        display_orientation=_infer_display_orientation(
            display_size.get("width"),
            display_size.get("height"),
        ),
        normalization_applied=bool(rotation_degrees),
        source="video_metadata",
        probe_details={
            "tool": "ffprobe",
            "stream_rotation_degrees": rotation_degrees,
            "video_path": str(video_path),
        },
    )


def _capture_orientation_from_dimensions(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    encoded_width = _first_int(
        manifest.get("width"),
        context.get("width"),
        context.get("sourceVideoWidth"),
        context.get("videoWidth"),
    )
    encoded_height = _first_int(
        manifest.get("height"),
        context.get("height"),
        context.get("sourceVideoHeight"),
        context.get("videoHeight"),
    )
    declared_capture_width, declared_capture_height = _declared_capture_dimensions(
        manifest=manifest,
        context=context,
    )
    encoded_orientation = _infer_display_orientation(encoded_width, encoded_height)
    declared_orientation = _infer_display_orientation(
        declared_capture_width,
        declared_capture_height,
    )
    normalization_applied = (
        declared_orientation not in {"unknown", encoded_orientation}
        and encoded_orientation != "unknown"
    )
    display_rotation_degrees = 90 if normalization_applied else 0
    display_orientation = (
        declared_orientation if declared_orientation != "unknown" else encoded_orientation
    )
    return _orientation_payload(
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        declared_capture_width=declared_capture_width or encoded_width,
        declared_capture_height=declared_capture_height or encoded_height,
        display_rotation_degrees=display_rotation_degrees,
        display_orientation=display_orientation,
        normalization_applied=normalization_applied,
        source="inferred",
    )


def _resolve_capture_orientation(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    raw_root: Path,
) -> Dict[str, Any]:
    from_metadata = _capture_orientation_from_metadata(manifest=manifest, context=context)
    if from_metadata:
        return from_metadata
    video_candidates = _orientation_video_candidates(raw_root)
    if video_candidates:
        video_path = raw_root / video_candidates[0]
        if video_path.is_file():
            from_probe = _ffprobe_capture_orientation(video_path)
            if from_probe:
                return from_probe
    return _capture_orientation_from_dimensions(manifest=manifest, context=context)


def _orientation_video_candidates(raw_root: Path) -> List[str]:
    names = ("walkthrough.mov", "walkthrough.mp4", "recording.mov", "recording.mp4")
    return [name for name in names if (raw_root / name).is_file()]
