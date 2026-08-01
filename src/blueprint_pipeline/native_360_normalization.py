"""Deterministic native-360 source and rig normalization.

The normalizer consumes immutable source files plus a digest-bound probe
receipt. It never stitches, estimates poses, invents calibration, or establishes
metric scale. Native bytes remain authoritative in the admitted capture root.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


NATIVE_360_NORMALIZATION_SCHEMA_VERSION = "native_360_capture_normalization.v1"
DUAL_FISHEYE_BINDING_SCHEMA_VERSION = "dual_fisheye_stream_binding.v1"
CAMERA_360_RIG_SCHEMA_VERSION = "camera_360_rig_declaration.v1"
NATIVE_360_PROBE_SCHEMA_VERSION = "native_360_probe_receipt.v1"
_LENS_IDS = {"front", "rear"}
_MAX_NATIVE_SOURCE_BYTES = 100 * 1024 * 1024 * 1024
_MAX_CALIBRATION_MASK_BYTES = 64 * 1024 * 1024
_MAX_PROBE_OUTPUT_BYTES = 128 * 1024 * 1024
_PROBE_TIMEOUT_SECONDS = 600.0
_REQUIRED_LOCAL_AUTHORITY = {
    "source_capture_rights_valid": True,
    "consent_valid": True,
    "privacy_review_valid": True,
    "retention_authorized": True,
    "local_processing_authorized": True,
    "provider_upload_authorized": False,
    "paid_compute_authorized": False,
}

ProbeRunner = Callable[
    [Sequence[str], float, int],
    subprocess.CompletedProcess[Any],
]


class Native360NormalizationError(ValueError):
    """Stable fail-closed native-360 normalization failure."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _validate_local_authority(authority_used: Mapping[str, Any]) -> None:
    if any(
        authority_used.get(key) is not expected
        for key, expected in _REQUIRED_LOCAL_AUTHORITY.items()
    ):
        raise Native360NormalizationError(["native_360_authority_invalid"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _bounded_probe_command(
    argv: Sequence[str], timeout_seconds: float, maximum_output_bytes: int
) -> subprocess.CompletedProcess[bytes]:
    """Run a probe without a shell while bounding time and captured bytes."""

    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    try:
        process = subprocess.Popen(  # noqa: S603 - exact executable is digest-bound
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        if process.stdout is None or process.stderr is None:
            raise OSError("probe pipes unavailable")
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(list(argv), timeout_seconds)
            events = selector.select(min(remaining, 0.1))
            for key, _mask in events:
                chunk = os.read(key.fd, 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = key.data
                target.extend(chunk)
                if len(stdout) + len(stderr) > maximum_output_bytes:
                    raise Native360NormalizationError(["native_360_probe_output_oversized"])
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        return subprocess.CompletedProcess(list(argv), returncode, bytes(stdout), bytes(stderr))
    finally:
        selector.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def _probe_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return b""


def _run_probe(
    runner: ProbeRunner,
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    maximum_output_bytes: int,
) -> bytes:
    try:
        completed = runner(argv, timeout_seconds, maximum_output_bytes)
    except Native360NormalizationError:
        raise
    except subprocess.TimeoutExpired as exc:
        raise Native360NormalizationError(["native_360_probe_timeout"]) from exc
    except OSError as exc:
        raise Native360NormalizationError(["native_360_probe_execution_failed"]) from exc
    stdout = _probe_bytes(completed.stdout)
    stderr = _probe_bytes(completed.stderr)
    if len(stdout) + len(stderr) > maximum_output_bytes:
        raise Native360NormalizationError(["native_360_probe_output_oversized"])
    if completed.returncode != 0:
        raise Native360NormalizationError(["native_360_probe_media_rejected"])
    return stdout


def _strict_probe_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Native360NormalizationError(
                    [f"native_360_probe_duplicate_json_key:{label}:{key}"]
                )
            value[key] = item
        return value

    def reject_nonfinite_constant(value: str) -> Any:
        raise Native360NormalizationError([f"native_360_probe_json_nonfinite:{label}:{value}"])

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except Native360NormalizationError:
        raise
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Native360NormalizationError([f"native_360_probe_json_invalid:{label}"]) from exc
    if not isinstance(decoded, Mapping):
        raise Native360NormalizationError([f"native_360_probe_json_invalid:{label}"])
    return dict(decoded)


def _probe_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Native360NormalizationError([label])
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Native360NormalizationError([label]) from exc
    if not math.isfinite(number):
        raise Native360NormalizationError([label])
    return number


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _safe_relative(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Native360NormalizationError(["native_360_source_relative_path_unsafe"])
    return path.as_posix()


def _safe_source(root: Path, relative_path: str) -> Path:
    lexical = root.joinpath(*PurePosixPath(relative_path).parts)
    if lexical.is_symlink():
        raise Native360NormalizationError(["native_360_source_symlink_forbidden"])
    resolved = lexical.resolve()
    if root != resolved and root not in resolved.parents:
        raise Native360NormalizationError(["native_360_source_path_escape"])
    if resolved.is_symlink() or not resolved.is_file():
        raise Native360NormalizationError(["native_360_source_missing"])
    return resolved


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Native360NormalizationError(["native_360_immutable_artifact_invalid"]) from exc
        if canonical_json(existing) != canonical_json(normalized):
            raise Native360NormalizationError(["native_360_immutable_artifact_conflict"])
        return dict(existing)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise Native360NormalizationError(
                    ["native_360_immutable_artifact_invalid"]
                ) from exc
            if canonical_json(existing) != canonical_json(normalized):
                raise Native360NormalizationError(["native_360_immutable_artifact_conflict"])
            return dict(existing)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _copy_immutable_file(source: Path, destination: Path, expected_digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or _sha256_file(destination) != expected_digest:
            raise Native360NormalizationError(["native_360_immutable_mask_artifact_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_stream, os.fdopen(descriptor, "wb") as target:
            shutil.copyfileobj(source_stream, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if _sha256_file(temporary) != expected_digest:
            raise Native360NormalizationError(["native_360_calibration_mask_digest_mismatch"])
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or _sha256_file(destination) != expected_digest:
                raise Native360NormalizationError(["native_360_immutable_mask_artifact_conflict"])
    finally:
        temporary.unlink(missing_ok=True)


def _matrix4(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        values: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            number = float(item)
            if not math.isfinite(number):
                return None
            values.append(number)
        rows.append(values)
    return rows


def _rigid_transform4(value: Any) -> list[list[float]] | None:
    matrix = _matrix4(value)
    if matrix is None or any(
        not math.isclose(matrix[3][index], expected, abs_tol=1e-8)
        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
    ):
        return None
    rotation = [row[:3] for row in matrix[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            if not math.isclose(dot, 1.0 if left == right else 0.0, abs_tol=1e-5):
                return None
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, abs_tol=1e-5):
        return None
    baseline = math.sqrt(sum(matrix[row][3] ** 2 for row in range(3)))
    return matrix if baseline > 1e-8 else None


def _declared_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Native360NormalizationError(["native_360_timestamp_invalid"]) from exc
    if parsed.tzinfo is None:
        raise Native360NormalizationError(["native_360_timestamp_invalid"])
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_pts(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise Native360NormalizationError([f"native_360_pts_missing:{label}"])
    rows: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Native360NormalizationError([f"native_360_pts_invalid:{label}"])
        number = round(float(item), 9)
        if not math.isfinite(number) or (rows and number <= rows[-1]):
            raise Native360NormalizationError([f"native_360_pts_not_strictly_increasing:{label}"])
        rows.append(number)
    return rows


def build_native_360_probe_receipt(
    *,
    source_file_digest: str,
    runtime_identity: str,
    runtime_digest: str,
    streams: Sequence[Mapping[str, Any]],
    format_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable output expected from a bounded native-media probe."""

    if not _is_digest(source_file_digest) or not _is_digest(runtime_digest):
        raise Native360NormalizationError(["native_360_probe_digest_binding_invalid"])
    normalized_streams: list[dict[str, Any]] = []
    indexes: set[int] = set()
    for raw in streams:
        index = raw.get("stream_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in indexes:
            raise Native360NormalizationError(["native_360_probe_stream_index_invalid"])
        indexes.add(index)
        media_type = str(raw.get("media_type") or "")
        row = {
            "stream_index": index,
            "media_type": media_type,
            "codec_name": str(raw.get("codec_name") or "unknown"),
            "width": raw.get("width"),
            "height": raw.get("height"),
            "time_base": str(raw.get("time_base") or "unknown"),
            "pts_seconds": (
                _normalized_pts(raw.get("pts_seconds"), label=f"stream_{index}")
                if media_type == "video"
                else []
            ),
            "metadata": dict(raw.get("metadata") or {}),
        }
        if media_type == "video" and (
            isinstance(row["width"], bool)
            or not isinstance(row["width"], int)
            or row["width"] <= 0
            or isinstance(row["height"], bool)
            or not isinstance(row["height"], int)
            or row["height"] <= 0
        ):
            raise Native360NormalizationError(["native_360_probe_video_dimensions_invalid"])
        normalized_streams.append(row)
    if not normalized_streams or not str(runtime_identity).strip():
        raise Native360NormalizationError(["native_360_probe_receipt_incomplete"])
    receipt = {
        "schema_version": NATIVE_360_PROBE_SCHEMA_VERSION,
        "probe_status": "decodable",
        "source_file_digest": source_file_digest,
        "runtime_identity": runtime_identity,
        "runtime_digest": runtime_digest,
        "format_metadata": dict(format_metadata),
        "streams": sorted(normalized_streams, key=lambda row: row["stream_index"]),
    }
    receipt["probe_receipt_digest"] = canonical_digest(receipt, digest_field="probe_receipt_digest")
    return receipt


def probe_native_360_source(
    *,
    capture_root: str | Path,
    source_relative_path: str,
    ffprobe_executable: str | Path | None = None,
    timeout_seconds: float = _PROBE_TIMEOUT_SECONDS,
    maximum_source_bytes: int = _MAX_NATIVE_SOURCE_BYTES,
    maximum_output_bytes: int = _MAX_PROBE_OUTPUT_BYTES,
    runner: ProbeRunner | None = None,
) -> dict[str, Any]:
    """Probe exact native bytes into a digest-bound, claim-limited receipt.

    This executor observes container streams and decoded frame timestamps only.
    It does not assign lens identity, establish calibration, infer sensor streams,
    stitch pixels, or establish trajectory or metric scale.
    """

    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or maximum_source_bytes <= 0
        or maximum_output_bytes <= 0
    ):
        raise Native360NormalizationError(["native_360_probe_limit_invalid"])
    root = Path(capture_root).expanduser().resolve()
    if not root.is_dir():
        raise Native360NormalizationError(["native_360_capture_root_missing"])
    relative_path = _safe_relative(source_relative_path)
    if Path(relative_path).suffix.lower() != ".insv":
        raise Native360NormalizationError(["native_360_original_must_be_insv"])
    source = _safe_source(root, relative_path)
    size = source.stat().st_size
    if size <= 0 or size > maximum_source_bytes:
        raise Native360NormalizationError(["native_360_source_oversized"])
    source_digest = _sha256_file(source)

    executable_value = (
        str(ffprobe_executable) if ffprobe_executable is not None else shutil.which("ffprobe")
    )
    if not executable_value:
        raise Native360NormalizationError(["native_360_probe_runtime_unavailable"])
    executable = Path(executable_value).expanduser().resolve()
    if not executable.is_file():
        raise Native360NormalizationError(["native_360_probe_runtime_unavailable"])
    runtime_digest = _sha256_file(executable)
    execute = runner or _bounded_probe_command

    version_output = _run_probe(
        execute,
        [str(executable), "-version"],
        timeout_seconds=min(timeout_seconds, 30.0),
        maximum_output_bytes=min(maximum_output_bytes, 1024 * 1024),
    )
    try:
        version_line = version_output.decode("utf-8").splitlines()[0].strip()
    except (IndexError, UnicodeDecodeError) as exc:
        raise Native360NormalizationError(["native_360_probe_runtime_identity_invalid"]) from exc
    if not version_line or len(version_line) > 512:
        raise Native360NormalizationError(["native_360_probe_runtime_identity_invalid"])

    metadata_output = _run_probe(
        execute,
        [
            str(executable),
            "-v",
            "error",
            "-hide_banner",
            "-show_entries",
            (
                "format=format_name,format_long_name,start_time,duration,size,"
                "bit_rate,tags:stream=index,codec_type,codec_name,profile,width,"
                "height,pix_fmt,color_range,color_space,color_transfer,"
                "color_primaries,time_base,start_time,duration,nb_frames,"
                "avg_frame_rate,r_frame_rate,disposition,tags,side_data_list"
                ":stream_side_data"
            ),
            "-of",
            "json",
            str(source),
        ],
        timeout_seconds=timeout_seconds,
        maximum_output_bytes=maximum_output_bytes,
    )
    timing_output = _run_probe(
        execute,
        [
            str(executable),
            "-v",
            "error",
            "-hide_banner",
            "-threads",
            "1",
            "-show_frames",
            "-show_entries",
            (
                "frame=stream_index,media_type,pts_time,pkt_dts_time,"
                "best_effort_timestamp_time,pkt_duration_time,key_frame"
            ),
            "-of",
            "json",
            str(source),
        ],
        timeout_seconds=timeout_seconds,
        maximum_output_bytes=maximum_output_bytes,
    )
    metadata_payload = _strict_probe_json(metadata_output, label="metadata")
    timing_payload = _strict_probe_json(timing_output, label="timing")
    raw_streams = metadata_payload.get("streams")
    raw_format = metadata_payload.get("format")
    raw_frames = timing_payload.get("frames")
    if (
        not isinstance(raw_streams, list)
        or not isinstance(raw_format, Mapping)
        or not isinstance(raw_frames, list)
    ):
        raise Native360NormalizationError(["native_360_probe_payload_incomplete"])

    streams_by_index: dict[int, Mapping[str, Any]] = {}
    video_indexes: set[int] = set()
    for raw_stream in raw_streams:
        if not isinstance(raw_stream, Mapping):
            raise Native360NormalizationError(["native_360_probe_stream_invalid"])
        index = raw_stream.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index in streams_by_index
        ):
            raise Native360NormalizationError(["native_360_probe_stream_index_invalid"])
        streams_by_index[index] = raw_stream
        if raw_stream.get("codec_type") == "video":
            video_indexes.add(index)
    if not video_indexes:
        raise Native360NormalizationError(["native_360_probe_video_stream_missing"])

    timing_by_stream: dict[int, list[dict[str, Any]]] = {index: [] for index in video_indexes}
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, Mapping):
            raise Native360NormalizationError(["native_360_probe_frame_invalid"])
        index = raw_frame.get("stream_index")
        if index not in video_indexes:
            continue
        if raw_frame.get("media_type") not in {None, "video"}:
            raise Native360NormalizationError(["native_360_probe_frame_type_invalid"])
        if "pts_time" not in raw_frame:
            raise Native360NormalizationError(
                [f"native_360_probe_frame_pts_missing:stream_{index}"]
            )
        pts = _probe_number(
            raw_frame["pts_time"],
            label=f"native_360_probe_frame_pts_invalid:stream_{index}",
        )
        dts_raw = raw_frame.get("pkt_dts_time")
        dts = (
            _probe_number(
                dts_raw,
                label=f"native_360_probe_frame_dts_invalid:stream_{index}",
            )
            if dts_raw is not None
            else None
        )
        duration_raw = raw_frame.get("pkt_duration_time")
        duration = (
            _probe_number(
                duration_raw,
                label=f"native_360_probe_frame_duration_invalid:stream_{index}",
            )
            if duration_raw is not None
            else None
        )
        if duration is not None and duration <= 0:
            raise Native360NormalizationError(
                [f"native_360_probe_frame_duration_invalid:stream_{index}"]
            )
        key_frame_raw = raw_frame.get("key_frame")
        if key_frame_raw is not None and key_frame_raw not in {0, 1, False, True}:
            raise Native360NormalizationError(
                [f"native_360_probe_frame_key_frame_invalid:stream_{index}"]
            )
        timing_by_stream[index].append(
            {
                "pts_seconds": pts,
                "dts_seconds": dts,
                "duration_seconds": duration,
                "key_frame": (bool(key_frame_raw) if key_frame_raw is not None else None),
                "best_effort_timestamp_time": raw_frame.get("best_effort_timestamp_time"),
            }
        )

    receipt_streams: list[dict[str, Any]] = []
    stream_metadata_keys = (
        "profile",
        "pix_fmt",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "start_time",
        "duration",
        "nb_frames",
        "avg_frame_rate",
        "r_frame_rate",
        "disposition",
        "tags",
        "side_data_list",
    )
    for index, raw_stream in sorted(streams_by_index.items()):
        media_type = str(raw_stream.get("codec_type") or "unknown")
        timing = timing_by_stream.get(index, [])
        if media_type == "video" and not timing:
            raise Native360NormalizationError(
                [f"native_360_probe_video_frames_missing:stream_{index}"]
            )
        metadata = {key: raw_stream[key] for key in stream_metadata_keys if key in raw_stream}
        if media_type == "video":
            metadata.update(
                {
                    "decoded_timestamp_field": "pts_time",
                    "decoded_frame_timing_digest": canonical_digest({"frames": timing}),
                    "decoded_frame_timing": timing,
                    "all_decoded_dts_observed": all(
                        row["dts_seconds"] is not None for row in timing
                    ),
                    "lens_identity_inferred": False,
                }
            )
        receipt_streams.append(
            {
                "stream_index": index,
                "media_type": media_type,
                "codec_name": str(raw_stream.get("codec_name") or "unknown"),
                "width": raw_stream.get("width"),
                "height": raw_stream.get("height"),
                "time_base": str(raw_stream.get("time_base") or "unknown"),
                "pts_seconds": [row["pts_seconds"] for row in timing],
                "metadata": metadata,
            }
        )

    format_keys = (
        "format_name",
        "format_long_name",
        "start_time",
        "duration",
        "size",
        "bit_rate",
        "tags",
    )
    format_metadata = {key: raw_format[key] for key in format_keys if key in raw_format}
    video_streams = [row for row in receipt_streams if row["media_type"] == "video"]
    stitched_projection_streams: list[dict[str, Any]] = []
    for stream in video_streams:
        side_data = stream["metadata"].get("side_data_list")
        if not isinstance(side_data, list):
            continue
        for side_data_row in side_data:
            if not isinstance(side_data_row, Mapping):
                continue
            projection = str(side_data_row.get("projection") or "").strip().lower()
            if projection in {"equirectangular", "cubemap", "cylindrical"}:
                stitched_projection_streams.append(
                    {"stream_index": stream["stream_index"], "projection": projection}
                )
    if len(video_streams) == 1 and len(stitched_projection_streams) == 1:
        compatible_processing_lane = "camera_360_equirectangular"
    elif len(video_streams) >= 2 and not stitched_projection_streams:
        compatible_processing_lane = "camera_360_native_candidate_requires_calibration"
    else:
        compatible_processing_lane = "unsupported_or_ambiguous_360_topology"
    format_metadata.update(
        {
            "source_relative_path": relative_path,
            "source_size_bytes": size,
            "ffprobe_version_output_digest": "sha256:" + hashlib.sha256(version_output).hexdigest(),
            "ffprobe_metadata_output_digest": "sha256:"
            + hashlib.sha256(metadata_output).hexdigest(),
            "ffprobe_timing_output_digest": "sha256:" + hashlib.sha256(timing_output).hexdigest(),
            "observed_video_stream_count": len(video_streams),
            "observed_stitched_projection_streams": stitched_projection_streams,
            "compatible_processing_lane": compatible_processing_lane,
            "processing_lane_claim_ceiling": "container_stream_topology_only",
            "capture_profile_fully_validated": False,
            "probe_behavior": {
                "shell_used": False,
                "decoded_frames_observed": True,
                "lens_identity_inferred": False,
                "calibration_inferred": False,
                "imu_inferred": False,
                "gyro_inferred": False,
                "camera_trajectory_inferred": False,
                "metric_scale_inferred": False,
            },
        }
    )
    if source.stat().st_size != size or _sha256_file(source) != source_digest:
        raise Native360NormalizationError(["native_360_probe_source_changed"])
    if _sha256_file(executable) != runtime_digest:
        raise Native360NormalizationError(["native_360_probe_runtime_changed"])
    return build_native_360_probe_receipt(
        source_file_digest=source_digest,
        runtime_identity=version_line,
        runtime_digest=runtime_digest,
        streams=receipt_streams,
        format_metadata=format_metadata,
    )


def _validated_probe(value: Mapping[str, Any], *, source_digest: str) -> dict[str, Any]:
    receipt = dict(value)
    if (
        receipt.get("schema_version") != NATIVE_360_PROBE_SCHEMA_VERSION
        or receipt.get("probe_status") != "decodable"
        or receipt.get("source_file_digest") != source_digest
        or not str(receipt.get("runtime_identity") or "").strip()
        or not _is_digest(receipt.get("runtime_digest"))
        or receipt.get("probe_receipt_digest")
        != canonical_digest(receipt, digest_field="probe_receipt_digest")
        or not isinstance(receipt.get("streams"), list)
    ):
        raise Native360NormalizationError(["native_360_probe_receipt_invalid"])
    return receipt


def _stitched_projection(stream: Mapping[str, Any]) -> str | None:
    """Return an observed non-lens projection without treating metadata as calibration."""

    metadata = stream.get("metadata")
    side_data = metadata.get("side_data_list") if isinstance(metadata, Mapping) else None
    if not isinstance(side_data, list):
        return None
    for row in side_data:
        if not isinstance(row, Mapping):
            continue
        projection = str(row.get("projection") or "").strip().lower()
        if projection in {"equirectangular", "cubemap", "cylindrical"}:
            return projection
    return None


def _calibrated_rig(
    camera_metadata: Mapping[str, Any],
    *,
    capture_digest: str,
    capture_root: Path,
    maximum_mask_bytes: int,
) -> tuple[dict[str, Any], list[str], dict[str, Path]]:
    blockers: list[str] = []
    calibrations = camera_metadata.get("lens_calibrations")
    calibrations = calibrations if isinstance(calibrations, list) else []
    by_lens: dict[str, dict[str, Any]] = {}
    mask_sources: dict[str, Path] = {}
    for raw in calibrations:
        if not isinstance(raw, Mapping):
            continue
        lens_id = str(raw.get("lens_id") or "")
        intrinsics = raw.get("intrinsics")
        distortion = raw.get("distortion")
        numeric_intrinsics = {
            key: intrinsics.get(key) if isinstance(intrinsics, Mapping) else None
            for key in ("fx", "fy", "cx", "cy", "width", "height")
        }
        coefficients = distortion.get("coefficients") if isinstance(distortion, Mapping) else None
        calibration_source = str(raw.get("calibration_source") or "")
        mask_relative_path: str | None = None
        mask_source: Path | None = None
        try:
            mask_relative_path = _safe_relative(raw.get("valid_pixel_mask_relative_path"))
            mask_source = _safe_source(capture_root, mask_relative_path)
        except Native360NormalizationError:
            mask_relative_path = None
            mask_source = None
        mask_digest = raw.get("valid_pixel_mask_digest")
        mask_valid = bool(
            mask_source is not None
            and mask_source.stat().st_size > 0
            and mask_source.stat().st_size <= maximum_mask_bytes
            and _is_digest(mask_digest)
            and _sha256_file(mask_source) == mask_digest
        )
        valid_dimensions = (
            isinstance(numeric_intrinsics["width"], int)
            and not isinstance(numeric_intrinsics["width"], bool)
            and numeric_intrinsics["width"] > 0
            and isinstance(numeric_intrinsics["height"], int)
            and not isinstance(numeric_intrinsics["height"], bool)
            and numeric_intrinsics["height"] > 0
        )
        finite_intrinsics = all(
            isinstance(numeric_intrinsics[key], (int, float))
            and not isinstance(numeric_intrinsics[key], bool)
            and math.isfinite(float(numeric_intrinsics[key]))
            for key in ("fx", "fy", "cx", "cy")
        )
        plausible_intrinsics = bool(
            valid_dimensions
            and finite_intrinsics
            and float(numeric_intrinsics["fx"]) > 0
            and float(numeric_intrinsics["fy"]) > 0
            and 0 <= float(numeric_intrinsics["cx"]) <= int(numeric_intrinsics["width"])
            and 0 <= float(numeric_intrinsics["cy"]) <= int(numeric_intrinsics["height"])
        )
        if (
            lens_id not in _LENS_IDS
            or lens_id in by_lens
            or not isinstance(intrinsics, Mapping)
            or not plausible_intrinsics
            or not isinstance(distortion, Mapping)
            or not str(distortion.get("model") or "")
            or not isinstance(coefficients, list)
            or not coefficients
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in coefficients
            )
            or not mask_valid
            or calibration_source
            not in {
                "embedded_camera_metadata",
                "official_sdk_sidecar",
                "qualified_external_calibration",
            }
            or not _is_digest(raw.get("calibration_source_digest"))
        ):
            blockers.append(f"native_360_lens_calibration_invalid:{lens_id or 'unknown'}")
            continue
        by_lens[lens_id] = {
            "lens_id": lens_id,
            "intrinsics": dict(intrinsics),
            "distortion": dict(distortion),
            "valid_pixel_mask_relative_path": mask_relative_path,
            "valid_pixel_mask_digest": mask_digest,
            "calibration_source": calibration_source,
            "calibration_source_digest": raw["calibration_source_digest"],
        }
        assert mask_source is not None
        mask_sources[lens_id] = mask_source
    extrinsics = camera_metadata.get("rig_extrinsics")
    transform = _rigid_transform4(
        extrinsics.get("T_front_rear") if isinstance(extrinsics, Mapping) else None
    )
    extrinsics_source = (
        str(extrinsics.get("calibration_source") or "") if isinstance(extrinsics, Mapping) else ""
    )
    extrinsics_source_digest = (
        extrinsics.get("calibration_source_digest") if isinstance(extrinsics, Mapping) else None
    )
    transform_semantics = (
        str(extrinsics.get("transform_semantics") or "") if isinstance(extrinsics, Mapping) else ""
    )
    translation_units = (
        str(extrinsics.get("translation_units") or "") if isinstance(extrinsics, Mapping) else ""
    )
    if set(by_lens) != _LENS_IDS:
        blockers.append("native_360_complete_lens_calibration_missing")
    if transform is None:
        blockers.append("native_360_fixed_rig_extrinsics_missing")
    if extrinsics_source not in {
        "embedded_camera_metadata",
        "official_sdk_sidecar",
        "qualified_external_calibration",
    } or not _is_digest(extrinsics_source_digest):
        blockers.append("native_360_rig_extrinsics_provenance_missing")
    if transform_semantics not in {
        "rear_camera_from_front_rig",
        "front_rig_from_rear_camera",
    }:
        blockers.append("native_360_rig_transform_semantics_missing")
    if translation_units != "meters":
        blockers.append("native_360_rig_translation_units_invalid")
    rig = {
        "schema_version": CAMERA_360_RIG_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "camera_model": str(camera_metadata.get("camera_model") or ""),
        "capture_mode": str(camera_metadata.get("capture_mode") or ""),
        "firmware_version": str(camera_metadata.get("firmware_version") or "unknown"),
        "lens_calibrations": [by_lens[lens_id] for lens_id in sorted(by_lens)],
        "rig_extrinsics": {
            "T_front_rear": transform,
            "transform_semantics": transform_semantics,
            "translation_units": translation_units,
            "calibration_source": extrinsics_source,
            "calibration_source_digest": extrinsics_source_digest,
        },
        "rig_is_fixed": transform is not None,
        "calibration_status": "valid" if not blockers else "invalid",
        "metric_scale_status": "not_established",
        "agent_may_alter_calibration": False,
        "blockers": sorted(set(blockers)),
    }
    rig["rig_declaration_digest"] = canonical_digest(rig, digest_field="rig_declaration_digest")
    return rig, sorted(set(blockers)), mask_sources


def normalize_native_360_capture(
    *,
    capture_root: str | Path,
    output_root: str | Path,
    intake_id: str,
    capture_digest: str,
    camera_metadata: Mapping[str, Any],
    probe_receipts_by_path: Mapping[str, Mapping[str, Any]],
    source_commit_sha: str,
    implementation_digest: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
    parent_artifact_or_event: Mapping[str, Any] | None = None,
    synchronization_tolerance_seconds: float = 0.0005,
    maximum_source_bytes: int = _MAX_NATIVE_SOURCE_BYTES,
    maximum_mask_bytes: int = _MAX_CALIBRATION_MASK_BYTES,
) -> dict[str, Any]:
    """Normalize declared native 360 segments without modifying source truth."""

    if (
        not str(intake_id).strip()
        or not _is_digest(capture_digest)
        or not _is_digest(implementation_digest)
        or len(source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_commit_sha)
    ):
        raise Native360NormalizationError(["native_360_source_binding_invalid"])
    if (
        not math.isfinite(synchronization_tolerance_seconds)
        or synchronization_tolerance_seconds < 0
        or maximum_source_bytes <= 0
        or maximum_mask_bytes <= 0
    ):
        raise Native360NormalizationError(["native_360_normalization_limit_invalid"])
    _validate_local_authority(authority_used)
    root = Path(capture_root).expanduser().resolve()
    if not root.is_dir():
        raise Native360NormalizationError(["native_360_capture_root_missing"])
    if camera_metadata.get("schema_version") != "native_360_camera_metadata.v1":
        raise Native360NormalizationError(["native_360_camera_metadata_invalid"])
    if camera_metadata.get("source_capture_digest") != capture_digest:
        raise Native360NormalizationError(["native_360_camera_metadata_capture_mismatch"])
    if not str(camera_metadata.get("camera_model") or "") or not str(
        camera_metadata.get("capture_mode") or ""
    ):
        raise Native360NormalizationError(["native_360_camera_identity_missing"])
    coordinate_frame = camera_metadata.get("coordinate_frame_declaration")
    if (
        not isinstance(coordinate_frame, Mapping)
        or coordinate_frame.get("units") != "meters"
        or coordinate_frame.get("handedness") not in {"right_handed", "left_handed"}
        or not str(coordinate_frame.get("camera_axes") or "")
        or not str(coordinate_frame.get("rig_frame") or "")
    ):
        raise Native360NormalizationError(["native_360_coordinate_frame_invalid"])
    compiled_at = _declared_timestamp(timestamp)
    declared_segments = camera_metadata.get("segments")
    if not isinstance(declared_segments, list) or not declared_segments:
        raise Native360NormalizationError(["native_360_segments_missing"])
    segments = sorted(
        [dict(row) for row in declared_segments if isinstance(row, Mapping)],
        key=lambda row: row.get("sequence_index", -1),
    )
    if len(segments) != len(declared_segments) or [
        row.get("sequence_index") for row in segments
    ] != list(range(len(segments))):
        raise Native360NormalizationError(["native_360_segment_sequence_invalid"])
    segment_ids = [str(row.get("segment_id") or "") for row in segments]
    if any(not item for item in segment_ids) or len(set(segment_ids)) != len(segment_ids):
        raise Native360NormalizationError(["native_360_segment_identity_invalid"])
    rig, blockers, mask_sources = _calibrated_rig(
        camera_metadata,
        capture_digest=capture_digest,
        capture_root=root,
        maximum_mask_bytes=maximum_mask_bytes,
    )
    segment_timeline_starts: dict[int, float | None] = {}
    segment_timeline_sources: dict[int, str] = {}
    for segment in segments:
        sequence_index = int(segment["sequence_index"])
        raw_start = segment.get("capture_timeline_start_seconds")
        if raw_start is None and len(segments) == 1:
            segment_timeline_starts[sequence_index] = 0.0
            segment_timeline_sources[sequence_index] = "single_segment_relative_origin"
        elif (
            isinstance(raw_start, bool)
            or not isinstance(raw_start, (int, float))
            or not math.isfinite(float(raw_start))
            or float(raw_start) < 0
        ):
            segment_timeline_starts[sequence_index] = None
            segment_timeline_sources[sequence_index] = "missing_or_invalid"
            blockers.append(f"native_360_segment_capture_timeline_missing:{sequence_index}")
        else:
            segment_timeline_starts[sequence_index] = round(float(raw_start), 9)
            segment_timeline_sources[sequence_index] = "declared_capture_timeline"
    calibration_by_lens = {
        str(row["lens_id"]): row
        for row in rig["lens_calibrations"]
        if isinstance(row, Mapping) and row.get("lens_id") in _LENS_IDS
    }
    normalized_segments: list[dict[str, Any]] = []
    total_source_bytes = 0
    runtime_digests: set[str] = set()
    source_file_references: list[dict[str, Any]] = []
    validated_probes: dict[str, dict[str, Any]] = {}
    source_paths_seen: set[str] = set()
    stitched_projection_seen = False
    for segment in segments:
        files = segment.get("files")
        if not isinstance(files, list) or not files:
            raise Native360NormalizationError(["native_360_segment_files_missing"])
        lens_streams: dict[str, dict[str, Any]] = {}
        file_references: list[dict[str, Any]] = []
        for raw_file in files:
            if not isinstance(raw_file, Mapping):
                raise Native360NormalizationError(["native_360_segment_file_invalid"])
            relative_path = _safe_relative(raw_file.get("relative_path"))
            if relative_path in source_paths_seen:
                raise Native360NormalizationError(["native_360_source_path_reused"])
            source_paths_seen.add(relative_path)
            if Path(relative_path).suffix.lower() != ".insv":
                raise Native360NormalizationError(["native_360_original_must_be_insv"])
            if (
                str(raw_file.get("original_filename") or "") != Path(relative_path).name
                or isinstance(raw_file.get("size_bytes"), bool)
                or not isinstance(raw_file.get("size_bytes"), int)
                or raw_file.get("size_bytes") <= 0
            ):
                raise Native360NormalizationError(["native_360_source_declaration_invalid"])
            source = _safe_source(root, relative_path)
            size = source.stat().st_size
            total_source_bytes += size
            if size <= 0 or total_source_bytes > maximum_source_bytes:
                raise Native360NormalizationError(["native_360_source_oversized"])
            if size != raw_file["size_bytes"]:
                raise Native360NormalizationError(["native_360_source_size_mismatch"])
            source_digest = _sha256_file(source)
            if source_digest != raw_file.get("digest"):
                raise Native360NormalizationError(["native_360_source_digest_mismatch"])
            probe_value = probe_receipts_by_path.get(relative_path)
            if not isinstance(probe_value, Mapping):
                raise Native360NormalizationError(["native_360_probe_receipt_missing"])
            probe = _validated_probe(probe_value, source_digest=source_digest)
            validated_probes[relative_path] = probe
            runtime_digests.add(str(probe["runtime_digest"]))
            streams = {
                row.get("stream_index"): row for row in probe["streams"] if isinstance(row, Mapping)
            }
            bindings = raw_file.get("lens_streams")
            if not isinstance(bindings, list) or not bindings:
                raise Native360NormalizationError(["native_360_lens_stream_binding_missing"])
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    raise Native360NormalizationError(["native_360_lens_stream_binding_invalid"])
                lens_id = str(binding.get("lens_id") or "")
                stream_index = binding.get("stream_index")
                stream = streams.get(stream_index)
                if (
                    lens_id not in _LENS_IDS
                    or lens_id in lens_streams
                    or not isinstance(stream, Mapping)
                    or stream.get("media_type") != "video"
                ):
                    raise Native360NormalizationError(["native_360_lens_stream_binding_invalid"])
                stitched_projection = _stitched_projection(stream)
                if stitched_projection is not None:
                    stitched_projection_seen = True
                    blockers.append(
                        "native_360_lens_stream_is_stitched_projection:"
                        f"{segment['sequence_index']}:{lens_id}:{stitched_projection}"
                    )
                pts = _normalized_pts(
                    stream.get("pts_seconds"),
                    label=f"segment_{segment['sequence_index']}_{lens_id}",
                )
                lens_streams[lens_id] = {
                    "lens_id": lens_id,
                    "source_relative_path": relative_path,
                    "source_digest": source_digest,
                    "stream_index": stream_index,
                    "codec_name": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "time_base": stream.get("time_base"),
                    "frame_count": len(pts),
                    "first_pts_seconds": pts[0],
                    "last_pts_seconds": pts[-1],
                    "pts_digest": canonical_digest({"pts_seconds": pts}),
                    "observed_source_projection": stitched_projection,
                    "_pts": pts,
                }
            file_reference = {
                "relative_path": relative_path,
                "digest": source_digest,
                "size_bytes": size,
                "probe_receipt_digest": probe["probe_receipt_digest"],
            }
            file_references.append(file_reference)
            if file_reference not in source_file_references:
                source_file_references.append(file_reference)
        frame_pairs: list[dict[str, Any]] = []
        if set(lens_streams) != _LENS_IDS:
            blockers.append(f"native_360_dual_lens_streams_incomplete:{segment['sequence_index']}")
            maximum_residual = None
            synchronized = False
        else:
            front = lens_streams["front"]
            rear = lens_streams["rear"]
            front_pts = front.pop("_pts")
            rear_pts = rear.pop("_pts")
            same_shape = (
                front["frame_count"] == rear["frame_count"]
                and front["width"] == rear["width"]
                and front["height"] == rear["height"]
            )
            residuals = (
                [abs(left - right) for left, right in zip(front_pts, rear_pts, strict=True)]
                if same_shape
                else []
            )
            maximum_residual = max(residuals) if residuals else None
            frame_pairs = (
                [
                    {
                        "pair_index": index,
                        "front_pts_seconds": left,
                        "rear_pts_seconds": right,
                        "absolute_residual_seconds": residuals[index],
                    }
                    for index, (left, right) in enumerate(zip(front_pts, rear_pts, strict=True))
                ]
                if same_shape
                else []
            )
            synchronized = bool(
                same_shape
                and maximum_residual is not None
                and maximum_residual <= synchronization_tolerance_seconds
            )
            if not same_shape:
                blockers.append(
                    f"native_360_lens_dimensions_or_counts_mismatch:{segment['sequence_index']}"
                )
            elif not synchronized:
                blockers.append(
                    f"native_360_lens_streams_unsynchronized:{segment['sequence_index']}"
                )
            for lens_id, stream in (("front", front), ("rear", rear)):
                calibration = calibration_by_lens.get(lens_id)
                intrinsics = (
                    calibration.get("intrinsics") if isinstance(calibration, Mapping) else None
                )
                if not isinstance(intrinsics, Mapping) or (
                    intrinsics.get("width") != stream["width"]
                    or intrinsics.get("height") != stream["height"]
                ):
                    blockers.append(
                        "native_360_calibration_stream_dimensions_mismatch:"
                        f"{segment['sequence_index']}:{lens_id}"
                    )
        for stream in lens_streams.values():
            stream.pop("_pts", None)
        normalized_segments.append(
            {
                "sequence_index": segment["sequence_index"],
                "segment_id": str(
                    segment.get("segment_id") or f"segment-{segment['sequence_index']:04d}"
                ),
                "capture_timeline_start_seconds": segment_timeline_starts[
                    int(segment["sequence_index"])
                ],
                "capture_timeline_start_source": segment_timeline_sources[
                    int(segment["sequence_index"])
                ],
                "files": sorted(file_references, key=lambda row: row["relative_path"]),
                "lens_streams": [lens_streams[lens_id] for lens_id in sorted(lens_streams)],
                "frame_pairs": frame_pairs,
                "frame_pair_digest": canonical_digest({"frame_pairs": frame_pairs}),
                "maximum_lens_pts_residual_seconds": maximum_residual,
                "synchronized": synchronized,
            }
        )
    if set(probe_receipts_by_path) != source_paths_seen:
        raise Native360NormalizationError(["native_360_unbound_probe_receipt"])
    if len(runtime_digests) != 1:
        blockers.append("native_360_probe_runtime_inconsistent")
    for sensor in ("imu", "gyro"):
        declaration = camera_metadata.get(sensor)
        if not isinstance(declaration, Mapping) or declaration.get("status") not in {
            "available",
            "unavailable",
        }:
            blockers.append(f"native_360_{sensor}_declaration_missing")
        elif declaration.get("status") == "available" and not _is_digest(declaration.get("digest")):
            blockers.append(f"native_360_{sensor}_digest_missing")
    previous_capture_end: float | None = None
    for segment in normalized_segments:
        start = segment["capture_timeline_start_seconds"]
        pairs = segment["frame_pairs"]
        if start is None or not pairs:
            segment["capture_timeline_end_seconds"] = None
            continue
        duration = float(pairs[-1]["front_pts_seconds"]) - float(pairs[0]["front_pts_seconds"])
        end = round(float(start) + duration, 9)
        segment["capture_timeline_end_seconds"] = end
        if previous_capture_end is not None and float(start) <= previous_capture_end:
            blockers.append(
                f"native_360_segment_capture_timeline_overlap:{segment['sequence_index']}"
            )
        previous_capture_end = end
    timeline_valid = all(
        row["capture_timeline_start_seconds"] is not None
        and row["capture_timeline_end_seconds"] is not None
        for row in normalized_segments
    ) and not any("capture_timeline_overlap" in blocker for blocker in blockers)
    blockers = sorted(set(blockers))
    binding = {
        "schema_version": DUAL_FISHEYE_BINDING_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "camera_model": camera_metadata["camera_model"],
        "capture_mode": camera_metadata["capture_mode"],
        "segments": normalized_segments,
        "synchronization_tolerance_seconds": synchronization_tolerance_seconds,
        "all_segments_synchronized": all(row["synchronized"] for row in normalized_segments),
        "capture_timeline_valid": timeline_valid,
        "original_distorted_pixels_preserved": not stitched_projection_seen,
        "source_pixels_unmodified": True,
        "agent_may_rebind_lens_streams": False,
        "blockers": blockers,
    }
    binding["dual_fisheye_binding_digest"] = canonical_digest(
        binding, digest_field="dual_fisheye_binding_digest"
    )
    configuration_digest = canonical_digest(
        {
            "capture_digest": capture_digest,
            "camera_metadata_digest": canonical_digest(camera_metadata),
            "source_file_digests": sorted(row["digest"] for row in source_file_references),
            "probe_receipt_digests": sorted(
                row["probe_receipt_digest"] for row in source_file_references
            ),
            "valid_pixel_mask_source_digests": sorted(
                str(row["valid_pixel_mask_digest"]) for row in rig["lens_calibrations"]
            ),
            "rig_declaration_digest": rig["rig_declaration_digest"],
            "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
            "implementation_digest": implementation_digest,
            "source_commit_sha": source_commit_sha,
            "parent_artifact_digest": canonical_digest(dict(parent_artifact_or_event or {})),
        }
    )
    artifact_root = (
        Path(output_root).expanduser().resolve()
        / f"native_360_normalization_{configuration_digest[7:23]}"
    )
    rig = _write_immutable(artifact_root / "camera_360_rig_declaration.json", rig)
    binding = _write_immutable(artifact_root / "dual_fisheye_stream_binding.json", binding)
    valid_pixel_mask_references: dict[str, dict[str, str]] = {}
    for lens_id in sorted(mask_sources):
        calibration = calibration_by_lens[lens_id]
        mask_relative_path = f"calibration_masks/{lens_id}.png"
        mask_path = artifact_root / mask_relative_path
        _copy_immutable_file(
            mask_sources[lens_id],
            mask_path,
            str(calibration["valid_pixel_mask_digest"]),
        )
        valid_pixel_mask_references[lens_id] = {
            "relative_path": mask_relative_path,
            "digest": _sha256_file(mask_path),
        }
    rig_reference = {
        "relative_path": "camera_360_rig_declaration.json",
        "digest": _sha256_file(artifact_root / "camera_360_rig_declaration.json"),
    }
    binding_reference = {
        "relative_path": "dual_fisheye_stream_binding.json",
        "digest": _sha256_file(artifact_root / "dual_fisheye_stream_binding.json"),
    }
    probe_receipt_references: list[dict[str, Any]] = []
    for ordinal, relative_path in enumerate(sorted(validated_probes)):
        receipt_relative_path = f"probe_receipts/probe_{ordinal:04d}.json"
        receipt_path = artifact_root / receipt_relative_path
        receipt = _write_immutable(receipt_path, validated_probes[relative_path])
        probe_receipt_references.append(
            {
                "source_relative_path": relative_path,
                "relative_path": receipt_relative_path,
                "digest": _sha256_file(receipt_path),
                "probe_receipt_digest": receipt["probe_receipt_digest"],
            }
        )
    result_path = artifact_root / "native_360_capture_normalization.json"
    persisted_timestamp = compiled_at
    if result_path.exists():
        try:
            existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Native360NormalizationError(["native_360_immutable_artifact_invalid"]) from exc
        if not isinstance(existing_result, Mapping):
            raise Native360NormalizationError(["native_360_immutable_artifact_invalid"])
        persisted_timestamp = _declared_timestamp(existing_result.get("timestamp"))
    result = {
        "schema_version": NATIVE_360_NORMALIZATION_SCHEMA_VERSION,
        "stable_run_identity": f"native-360-{configuration_digest[7:31]}",
        "source_capture_identity": intake_id,
        "source_capture_digest": capture_digest,
        "producing_method": "deterministic_native_360_normalizer.v1",
        "source_commit_sha": source_commit_sha,
        "implementation_version": implementation_digest,
        "container_image_digest": None,
        "deterministic_configuration_digest": configuration_digest,
        "original_file_references": sorted(
            source_file_references, key=lambda row: row["relative_path"]
        ),
        "probe_receipt_references": probe_receipt_references,
        "camera_metadata_digest": canonical_digest(camera_metadata),
        "probe_runtime_digest": next(iter(runtime_digests)) if len(runtime_digests) == 1 else None,
        "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
        "rig_declaration_digest": rig["rig_declaration_digest"],
        "input_digests": {
            "camera_metadata_digest": canonical_digest(camera_metadata),
            "authority_digest": canonical_digest(authority_used),
            "source_file_digests": sorted(row["digest"] for row in source_file_references),
            "probe_receipt_digests": sorted(
                row["probe_receipt_digest"] for row in source_file_references
            ),
        },
        "output_digests": {
            "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
            "rig_declaration_digest": rig["rig_declaration_digest"],
            "probe_receipt_artifact_digests": [row["digest"] for row in probe_receipt_references],
            "valid_pixel_mask_artifact_digests": sorted(
                row["digest"] for row in valid_pixel_mask_references.values()
            ),
        },
        "artifact_references": {
            "camera_360_rig_declaration": rig_reference,
            "dual_fisheye_stream_binding": binding_reference,
        },
        "valid_pixel_mask_references": valid_pixel_mask_references,
        "train_heldout_split_digest": None,
        "camera_calibration_binding": rig["rig_declaration_digest"],
        "coordinate_frame_declaration": dict(coordinate_frame),
        "units": "meters_and_source_stream_seconds",
        "status": "normalized" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": ["native_pixels_not_stitched_or_rectified"],
        "metric_scale_status": "not_established",
        "camera_trajectory_status": "not_established",
        "raw_native_bytes_remain_authoritative": True,
        "original_native_bytes_modified": False,
        "provider_runtime_identity": {
            "provider": "local",
            "runtime": "recorded_probe",
            "runtime_digest": (next(iter(runtime_digests)) if len(runtime_digests) == 1 else None),
        },
        "authority_used": dict(authority_used),
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "proof_effect": "calibrated_native_360_rig_only" if not blockers else "none",
        "claim_ceiling": "calibrated_camera_rig" if not blockers else "decoded_native_container",
        "parent_artifact_or_event": dict(parent_artifact_or_event or {}),
        "timestamp": persisted_timestamp,
        "legal_next_actions": (
            [
                "compile_frozen_frame_dataset",
                "run_rig_constrained_pose_estimation",
                "request_metric_scale_anchor",
            ]
            if not blockers
            else ["preserve_evidence_and_stop", "request_corrected_native_360_metadata"]
        ),
        "agent_selected_camera_model": False,
        "agent_altered_calibration": False,
        "appearance_reconstruction_proven": False,
        "metric_geometry_proven": False,
        "collision_geometry_proven": False,
        "isaac_compatibility_proven": False,
    }
    result["native_360_normalization_digest"] = canonical_digest(
        result, digest_field="native_360_normalization_digest"
    )
    return _write_immutable(result_path, result)


def probe_and_normalize_native_360_capture(
    *,
    capture_root: str | Path,
    output_root: str | Path,
    intake_id: str,
    capture_digest: str,
    camera_metadata: Mapping[str, Any],
    source_commit_sha: str,
    implementation_digest: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
    parent_artifact_or_event: Mapping[str, Any] | None = None,
    synchronization_tolerance_seconds: float = 0.0005,
    maximum_source_bytes: int = _MAX_NATIVE_SOURCE_BYTES,
    ffprobe_executable: str | Path | None = None,
    probe_timeout_seconds: float = _PROBE_TIMEOUT_SECONDS,
    maximum_probe_output_bytes: int = _MAX_PROBE_OUTPUT_BYTES,
    probe_runner: ProbeRunner | None = None,
) -> dict[str, Any]:
    """Execute bounded source probes and normalize their exact receipts."""

    _validate_local_authority(authority_used)
    if maximum_source_bytes <= 0:
        raise Native360NormalizationError(["native_360_probe_limit_invalid"])
    if (
        camera_metadata.get("schema_version") != "native_360_camera_metadata.v1"
        or camera_metadata.get("source_capture_digest") != capture_digest
    ):
        raise Native360NormalizationError(["native_360_camera_metadata_invalid"])
    segments = camera_metadata.get("segments")
    if not isinstance(segments, list) or not segments:
        raise Native360NormalizationError(["native_360_segments_missing"])
    source_paths: list[str] = []
    for segment in segments:
        files = segment.get("files") if isinstance(segment, Mapping) else None
        if not isinstance(files, list) or not files:
            raise Native360NormalizationError(["native_360_segment_files_missing"])
        for raw_file in files:
            if not isinstance(raw_file, Mapping):
                raise Native360NormalizationError(["native_360_segment_file_invalid"])
            relative_path = _safe_relative(raw_file.get("relative_path"))
            if relative_path in source_paths:
                raise Native360NormalizationError(["native_360_source_path_reused"])
            source_paths.append(relative_path)
    if not source_paths:
        raise Native360NormalizationError(["native_360_segment_files_missing"])

    total_source_bytes = 0
    for relative_path in source_paths:
        total_source_bytes += (
            _safe_source(Path(capture_root).expanduser().resolve(), relative_path).stat().st_size
        )
        if total_source_bytes > maximum_source_bytes:
            raise Native360NormalizationError(["native_360_source_oversized"])

    per_source_limit = maximum_source_bytes
    receipts = {
        relative_path: probe_native_360_source(
            capture_root=capture_root,
            source_relative_path=relative_path,
            ffprobe_executable=ffprobe_executable,
            timeout_seconds=probe_timeout_seconds,
            maximum_source_bytes=per_source_limit,
            maximum_output_bytes=maximum_probe_output_bytes,
            runner=probe_runner,
        )
        for relative_path in source_paths
    }
    return normalize_native_360_capture(
        capture_root=capture_root,
        output_root=output_root,
        intake_id=intake_id,
        capture_digest=capture_digest,
        camera_metadata=camera_metadata,
        probe_receipts_by_path=receipts,
        source_commit_sha=source_commit_sha,
        implementation_digest=implementation_digest,
        authority_used=authority_used,
        timestamp=timestamp,
        parent_artifact_or_event=parent_artifact_or_event,
        synchronization_tolerance_seconds=synchronization_tolerance_seconds,
        maximum_source_bytes=maximum_source_bytes,
    )


__all__ = [
    "CAMERA_360_RIG_SCHEMA_VERSION",
    "DUAL_FISHEYE_BINDING_SCHEMA_VERSION",
    "NATIVE_360_NORMALIZATION_SCHEMA_VERSION",
    "NATIVE_360_PROBE_SCHEMA_VERSION",
    "Native360NormalizationError",
    "build_native_360_probe_receipt",
    "normalize_native_360_capture",
    "probe_and_normalize_native_360_capture",
    "probe_native_360_source",
]
