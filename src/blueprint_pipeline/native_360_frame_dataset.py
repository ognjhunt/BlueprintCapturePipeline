"""Bind decoded dual-fisheye observations to one frozen grouped dataset.

This adapter is downstream of native container probing and pixel decoding. It
does not infer lens identity or synchronization: both must already be present in
the validated native-360 normalization artifacts. Synchronized front/rear
observations receive one immutable split assignment so a hidden counterpart can
never leak into a candidate method's dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .native_360_normalization import (
    Native360NormalizationError,
    ProbeRunner,
    _bounded_probe_command,
)
from .reconstruction_frame_dataset import compile_frozen_frame_dataset


NATIVE_360_GROUPED_DATASET_ADAPTER_VERSION = (
    "native_360_grouped_frame_dataset_adapter.v1"
)
NATIVE_360_LENS_DECODE_SCHEMA_VERSION = "native_360_lens_decode_manifest.v1"
_LENS_IDS = {"front", "rear"}
_REQUIRED_LOCAL_AUTHORITY = {
    "source_capture_rights_valid": True,
    "consent_valid": True,
    "privacy_review_valid": True,
    "retention_authorized": True,
    "local_processing_authorized": True,
    "provider_upload_authorized": False,
    "paid_compute_authorized": False,
}


class Native360FrameDatasetError(ValueError):
    """Stable fail-closed error for native paired-observation compilation."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _finite(value: Any, *, code: str) -> float:
    if isinstance(value, bool):
        raise Native360FrameDatasetError([code])
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Native360FrameDatasetError([code]) from exc
    if not math.isfinite(number):
        raise Native360FrameDatasetError([code])
    return number


def _optional_finite(value: Any, *, code: str, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    number = _finite(value, code=code)
    if nonnegative and number < 0:
        raise Native360FrameDatasetError([code])
    return number


def _normalized_timestamp(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_timestamp_invalid"]
        ) from exc
    if parsed.tzinfo is None:
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_timestamp_invalid"]
        )
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_immutable(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Native360FrameDatasetError(
                ["native_360_lens_decode_immutable_manifest_invalid"]
            ) from exc
        if canonical_json(existing) != canonical_json(normalized):
            raise Native360FrameDatasetError(
                ["native_360_lens_decode_immutable_manifest_conflict"]
            )
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
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(existing) != canonical_json(normalized):
                raise Native360FrameDatasetError(
                    ["native_360_lens_decode_immutable_manifest_conflict"]
                )
            return dict(existing)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _run_media_command(
    runner: ProbeRunner,
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    maximum_output_bytes: int,
) -> bytes:
    try:
        completed = runner(argv, timeout_seconds, maximum_output_bytes)
    except Native360NormalizationError as exc:
        raise Native360FrameDatasetError(
            [f"native_360_lens_decode_{code.removeprefix('native_360_probe_')}" for code in exc.codes]
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise Native360FrameDatasetError(["native_360_lens_decode_timeout"]) from exc
    except OSError as exc:
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_execution_failed"]
        ) from exc
    stdout = completed.stdout.encode() if isinstance(completed.stdout, str) else completed.stdout
    stderr = completed.stderr.encode() if isinstance(completed.stderr, str) else completed.stderr
    stdout = stdout if isinstance(stdout, bytes) else b""
    stderr = stderr if isinstance(stderr, bytes) else b""
    if len(stdout) + len(stderr) > maximum_output_bytes:
        raise Native360FrameDatasetError(["native_360_lens_decode_output_oversized"])
    if completed.returncode != 0:
        raise Native360FrameDatasetError(["native_360_lens_decode_media_rejected"])
    return stdout


def _safe_bound_source(root: Path, relative_path: str, expected_digest: str) -> Path:
    text = str(relative_path or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Native360FrameDatasetError(["native_360_lens_decode_source_path_unsafe"])
    lexical = root / Path(*relative.parts)
    if lexical.is_symlink():
        raise Native360FrameDatasetError(["native_360_lens_decode_source_invalid"])
    source = lexical.resolve()
    if (
        (root != source and root not in source.parents)
        or not source.is_file()
        or not _is_digest(expected_digest)
        or _sha256_file(source) != expected_digest
    ):
        raise Native360FrameDatasetError(["native_360_lens_decode_source_invalid"])
    return source


def _validated_existing_decode_manifest(
    manifest_path: Path, *, artifact_root: Path, configuration_digest: str
) -> dict[str, Any]:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_existing_manifest_invalid"]
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != NATIVE_360_LENS_DECODE_SCHEMA_VERSION
        or value.get("deterministic_configuration_digest") != configuration_digest
        or value.get("lens_decode_manifest_digest")
        != canonical_digest(value, digest_field="lens_decode_manifest_digest")
        or not isinstance(value.get("frames"), list)
    ):
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_existing_manifest_invalid"]
        )
    for ordinal, row in enumerate(value["frames"]):
        if not isinstance(row, Mapping):
            raise Native360FrameDatasetError(
                ["native_360_lens_decode_existing_manifest_invalid"]
            )
        text = str(row.get("artifact_relative_path") or "").replace("\\", "/")
        relative = PurePosixPath(text)
        if (
            not text
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise Native360FrameDatasetError(
                [f"native_360_lens_decode_replay_frame_invalid:{ordinal}"]
            )
        lexical = artifact_root / Path(*relative.parts)
        if lexical.is_symlink():
            raise Native360FrameDatasetError(
                [f"native_360_lens_decode_replay_frame_invalid:{ordinal}"]
            )
        path = lexical.resolve()
        if (
            (artifact_root != path and artifact_root not in path.parents)
            or path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != row.get("digest")
        ):
            raise Native360FrameDatasetError(
                [f"native_360_lens_decode_replay_frame_invalid:{ordinal}"]
            )
    return dict(value)


def _validated_image(
    *, root: Path, raw: Mapping[str, Any], declared: Mapping[str, Any], ordinal: int
) -> tuple[str, dict[str, Any]]:
    text = str(raw.get("artifact_relative_path") or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_path_unsafe:{ordinal}"]
        )
    lexical = root / Path(*relative.parts)
    if lexical.is_symlink():
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_artifact_invalid:{ordinal}"]
        )
    path = lexical.resolve()
    if (
        (root != path and root not in path.parents)
        or path.is_symlink()
        or not path.is_file()
        or not _is_digest(raw.get("digest"))
        or _sha256_file(path) != raw.get("digest")
    ):
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_artifact_invalid:{ordinal}"]
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            actual_size = image.size
    except (OSError, SyntaxError) as exc:
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_image_invalid:{ordinal}"]
        ) from exc
    image_metadata = dict(raw.get("image_metadata") or {})
    declared_size = (declared.get("width"), declared.get("height"))
    if (
        actual_size != declared_size
        or image_metadata.get("width") != declared_size[0]
        or image_metadata.get("height") != declared_size[1]
    ):
        raise Native360FrameDatasetError(
            [f"native_360_grouped_dataset_frame_dimensions_invalid:{ordinal}"]
        )
    return relative.as_posix(), image_metadata


def _validated_parent_artifacts(
    *,
    capture_digest: str,
    normalization_result: Mapping[str, Any],
    rig_declaration: Mapping[str, Any],
    dual_fisheye_binding: Mapping[str, Any],
) -> None:
    normalization_digest = normalization_result.get(
        "native_360_normalization_digest"
    )
    if (
        normalization_result.get("schema_version")
        != "native_360_capture_normalization.v1"
        or normalization_result.get("source_capture_digest") != capture_digest
        or normalization_result.get("status") != "normalized"
        or normalization_result.get("blockers") != []
        or normalization_result.get("raw_native_bytes_remain_authoritative")
        is not True
        or normalization_result.get("original_native_bytes_modified") is not False
        or not _is_digest(normalization_digest)
        or normalization_digest
        != canonical_digest(
            normalization_result, digest_field="native_360_normalization_digest"
        )
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_normalization_invalid"]
        )
    if (
        rig_declaration.get("schema_version") != "camera_360_rig_declaration.v1"
        or rig_declaration.get("capture_digest") != capture_digest
        or rig_declaration.get("calibration_status") != "valid"
        or rig_declaration.get("blockers") != []
        or rig_declaration.get("rig_declaration_digest")
        != canonical_digest(rig_declaration, digest_field="rig_declaration_digest")
        or normalization_result.get("rig_declaration_digest")
        != rig_declaration.get("rig_declaration_digest")
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_rig_invalid"]
        )
    if (
        dual_fisheye_binding.get("schema_version")
        != "dual_fisheye_stream_binding.v1"
        or dual_fisheye_binding.get("capture_digest") != capture_digest
        or dual_fisheye_binding.get("all_segments_synchronized") is not True
        or dual_fisheye_binding.get("capture_timeline_valid") is not True
        or dual_fisheye_binding.get("blockers") != []
        or dual_fisheye_binding.get("dual_fisheye_binding_digest")
        != canonical_digest(
            dual_fisheye_binding, digest_field="dual_fisheye_binding_digest"
        )
        or normalization_result.get("dual_fisheye_binding_digest")
        != dual_fisheye_binding.get("dual_fisheye_binding_digest")
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_stream_binding_invalid"]
        )


def decode_native_360_lens_observations(
    *,
    capture_root: str | Path,
    artifact_root: str | Path,
    capture_digest: str,
    normalization_result: Mapping[str, Any],
    rig_declaration: Mapping[str, Any],
    dual_fisheye_binding: Mapping[str, Any],
    implementation_digest: str,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
    ffmpeg_executable: str | Path | None = None,
    timeout_seconds_per_frame: float = 120.0,
    maximum_command_output_bytes: int = 4 * 1024 * 1024,
    runner: ProbeRunner | None = None,
) -> dict[str, Any]:
    """Decode every declared synchronized lens observation without inference."""

    if any(
        authority_used.get(key) is not expected
        for key, expected in _REQUIRED_LOCAL_AUTHORITY.items()
    ):
        raise Native360FrameDatasetError(["native_360_lens_decode_authority_invalid"])
    if (
        not math.isfinite(timeout_seconds_per_frame)
        or timeout_seconds_per_frame <= 0
        or maximum_command_output_bytes <= 0
        or not _is_digest(implementation_digest)
        or len(source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_commit_sha)
    ):
        raise Native360FrameDatasetError(["native_360_lens_decode_request_invalid"])
    _validated_parent_artifacts(
        capture_digest=capture_digest,
        normalization_result=normalization_result,
        rig_declaration=rig_declaration,
        dual_fisheye_binding=dual_fisheye_binding,
    )
    segments = dual_fisheye_binding.get("segments")
    if not isinstance(segments, list) or not segments:
        raise Native360FrameDatasetError(["native_360_lens_decode_segments_invalid"])
    normalized_source_references = {
        (str(row.get("relative_path") or ""), row.get("digest"))
        for row in normalization_result.get("original_file_references", [])
        if isinstance(row, Mapping)
    }
    source_root = Path(capture_root).expanduser().resolve()
    if not source_root.is_dir():
        raise Native360FrameDatasetError(["native_360_lens_decode_capture_root_missing"])
    segment_contexts: list[dict[str, Any]] = []
    source_references: list[dict[str, str]] = []
    for sequence_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or segment.get("sequence_index") != sequence_index:
            raise Native360FrameDatasetError(["native_360_lens_decode_segment_invalid"])
        files = segment.get("files")
        lens_streams = segment.get("lens_streams")
        frame_pairs = segment.get("frame_pairs")
        capture_timeline_start = _optional_finite(
            segment.get("capture_timeline_start_seconds"),
            code="native_360_lens_decode_capture_timeline_invalid",
            nonnegative=True,
        )
        if (
            capture_timeline_start is None
            or not isinstance(files, list)
            or len(files) != 1
            or not isinstance(lens_streams, list)
            or not isinstance(frame_pairs, list)
            or not frame_pairs
        ):
            raise Native360FrameDatasetError(["native_360_lens_decode_segment_invalid"])
        source_reference = files[0]
        if not isinstance(source_reference, Mapping):
            raise Native360FrameDatasetError(["native_360_lens_decode_source_invalid"])
        source_relative_path = str(source_reference.get("relative_path") or "")
        source_digest = str(source_reference.get("digest") or "")
        if (source_relative_path, source_digest) not in normalized_source_references:
            raise Native360FrameDatasetError(["native_360_lens_decode_source_invalid"])
        source = _safe_bound_source(source_root, source_relative_path, source_digest)
        streams = {
            str(row.get("lens_id") or ""): row
            for row in lens_streams
            if isinstance(row, Mapping)
        }
        if set(streams) != _LENS_IDS or any(
            row.get("source_relative_path") != source_relative_path
            or row.get("source_digest") != source_digest
            or isinstance(row.get("stream_index"), bool)
            or not isinstance(row.get("stream_index"), int)
            for row in streams.values()
        ):
            raise Native360FrameDatasetError(["native_360_lens_decode_streams_invalid"])
        segment_contexts.append(
            {
                "sequence_index": sequence_index,
                "segment": segment,
                "frame_pairs": frame_pairs,
                "source": source,
                "source_relative_path": source_relative_path,
                "source_digest": source_digest,
                "streams": streams,
                "capture_timeline_start_seconds": capture_timeline_start,
            }
        )
        source_reference_row = {
            "relative_path": source_relative_path,
            "digest": source_digest,
        }
        if source_reference_row in source_references:
            raise Native360FrameDatasetError(
                ["native_360_lens_decode_source_segment_duplicate"]
            )
        source_references.append(source_reference_row)

    executable_value = (
        str(ffmpeg_executable)
        if ffmpeg_executable is not None
        else shutil.which("ffmpeg")
    )
    if not executable_value:
        raise Native360FrameDatasetError(["native_360_lens_decode_runtime_unavailable"])
    executable = Path(executable_value).expanduser().resolve()
    if not executable.is_file():
        raise Native360FrameDatasetError(["native_360_lens_decode_runtime_unavailable"])
    runtime_digest = _sha256_file(executable)
    compiled_at = _normalized_timestamp(timestamp)
    execute = runner or _bounded_probe_command
    version_output = _run_media_command(
        execute,
        [str(executable), "-version"],
        timeout_seconds=min(timeout_seconds_per_frame, 30.0),
        maximum_output_bytes=min(maximum_command_output_bytes, 1024 * 1024),
    )
    try:
        runtime_identity = version_output.decode("utf-8").splitlines()[0].strip()
    except (IndexError, UnicodeDecodeError) as exc:
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_runtime_identity_invalid"]
        ) from exc
    if not runtime_identity or len(runtime_identity) > 512:
        raise Native360FrameDatasetError(
            ["native_360_lens_decode_runtime_identity_invalid"]
        )

    configuration = {
        "decoder_version": "native_360_ffmpeg_lens_decoder.v1",
        "source_capture_digest": capture_digest,
        "source_references": source_references,
        "source_reference_set_digest": canonical_digest(
            {"references": source_references}
        ),
        "native_360_normalization_digest": normalization_result[
            "native_360_normalization_digest"
        ],
        "rig_declaration_digest": rig_declaration["rig_declaration_digest"],
        "dual_fisheye_binding_digest": dual_fisheye_binding[
            "dual_fisheye_binding_digest"
        ],
        "segments": [
            {
                "sequence_index": context["sequence_index"],
                "capture_timeline_start_seconds": context[
                    "capture_timeline_start_seconds"
                ],
                "frame_pair_digest": context["segment"].get("frame_pair_digest"),
                "source_relative_path": context["source_relative_path"],
                "source_digest": context["source_digest"],
            }
            for context in segment_contexts
        ],
        "runtime_digest": runtime_digest,
        "implementation_digest": implementation_digest,
        "source_commit_sha": source_commit_sha,
        "authority_digest": canonical_digest(authority_used),
        "pixel_output": "png_native_distorted_pixels_no_autorotate",
        "threading": "ffmpeg_default_decode_single_frame_requests",
    }
    configuration_digest = canonical_digest(configuration)
    root = Path(artifact_root).expanduser().resolve()
    decode_root = root / f"native_360_lens_decode_{configuration_digest[7:23]}"
    manifest_path = decode_root / "native_360_lens_decode_manifest.json"
    if manifest_path.is_file():
        return _validated_existing_decode_manifest(
            manifest_path,
            artifact_root=root,
            configuration_digest=configuration_digest,
        )

    decoded_frames: list[dict[str, Any]] = []
    for context in segment_contexts:
        sequence_index = context["sequence_index"]
        decode_requests = [
            (pair, lens_id)
            for pair in context["frame_pairs"]
            for lens_id in sorted(_LENS_IDS)
        ]
        for pair, lens_id in decode_requests:
            if not isinstance(pair, Mapping):
                raise Native360FrameDatasetError(
                    ["native_360_lens_decode_frame_pair_invalid"]
                )
            pair_index = pair.get("pair_index")
            if isinstance(pair_index, bool) or not isinstance(pair_index, int):
                raise Native360FrameDatasetError(
                    ["native_360_lens_decode_frame_pair_invalid"]
                )
            stream = context["streams"][lens_id]
            source = context["source"]
            source_relative_path = context["source_relative_path"]
            source_digest = context["source_digest"]
            target = (
                decode_root
                / "frames"
                / f"segment-{sequence_index:04d}"
                / lens_id
                / f"pair-{pair_index:09d}.png"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=".png", dir=target.parent
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                _run_media_command(
                    execute,
                    [
                        str(executable),
                        "-v",
                        "error",
                        "-hide_banner",
                        "-nostdin",
                        "-noautorotate",
                        "-i",
                        str(source),
                        "-map",
                        f"0:{stream['stream_index']}",
                        "-vf",
                        f"select=eq(n\\,{pair_index})",
                        "-frames:v",
                        "1",
                        "-fps_mode",
                        "passthrough",
                        "-c:v",
                        "png",
                        "-f",
                        "image2",
                        "-y",
                        str(temporary),
                    ],
                    timeout_seconds=timeout_seconds_per_frame,
                    maximum_output_bytes=maximum_command_output_bytes,
                )
                if (
                    temporary.is_symlink()
                    or not temporary.is_file()
                    or temporary.stat().st_size <= 0
                ):
                    raise Native360FrameDatasetError(
                        [f"native_360_lens_decode_frame_missing:{pair_index}:{lens_id}"]
                    )
                try:
                    with Image.open(temporary) as image:
                        image.verify()
                    with Image.open(temporary) as image:
                        actual_size = image.size
                        gray = np.asarray(image.convert("L"), dtype=np.float32)
                except (OSError, SyntaxError) as exc:
                    raise Native360FrameDatasetError(
                        [f"native_360_lens_decode_frame_invalid:{pair_index}:{lens_id}"]
                    ) from exc
                declared_size = (stream.get("width"), stream.get("height"))
                if actual_size != declared_size:
                    raise Native360FrameDatasetError(
                        [
                            "native_360_lens_decode_frame_dimensions_invalid:"
                            f"{pair_index}:{lens_id}"
                        ]
                    )
                digest = _sha256_file(temporary)
                if target.is_symlink():
                    raise Native360FrameDatasetError(
                        ["native_360_lens_decode_target_symlink_forbidden"]
                    )
                if target.exists():
                    if _sha256_file(target) != digest:
                        raise Native360FrameDatasetError(
                            ["native_360_lens_decode_immutable_frame_conflict"]
                        )
                else:
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        if _sha256_file(target) != digest:
                            raise Native360FrameDatasetError(
                                ["native_360_lens_decode_immutable_frame_conflict"]
                            )
                horizontal = (
                    np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
                )
                vertical = (
                    np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
                )
                pts = _finite(
                    pair.get(f"{lens_id}_pts_seconds"),
                    code="native_360_lens_decode_frame_pair_invalid",
                )
                decoded_frames.append(
                    {
                        "segment_sequence_index": sequence_index,
                        "pair_index": pair_index,
                        "lens_id": lens_id,
                        "source_relative_path": source_relative_path,
                        "source_digest": source_digest,
                        "stream_index": stream["stream_index"],
                        "decoded_frame_index": pair_index,
                        "source_pts_seconds": pts,
                        "source_dts_seconds": pair.get(f"{lens_id}_dts_seconds"),
                        "duration_seconds": pair.get(f"{lens_id}_duration_seconds"),
                        "key_frame": pair.get(f"{lens_id}_key_frame"),
                        "artifact_relative_path": target.relative_to(root).as_posix(),
                        "digest": digest,
                        "image_metadata": {
                            "width": actual_size[0],
                            "height": actual_size[1],
                            "pixel_orientation": "native_distorted_pixels_no_autorotate",
                            "source_camera_identity": lens_id,
                        },
                        "quality_signals": {
                            "mean_luma_0_255": round(float(np.mean(gray)), 6),
                            "gradient_energy": round(
                                float(
                                    np.mean(horizontal * horizontal)
                                    + np.mean(vertical * vertical)
                                ),
                                6,
                            ),
                            "exposure_metadata": {},
                            "excessive_blur_deterministically_established": False,
                        },
                    }
                )
            finally:
                temporary.unlink(missing_ok=True)

    if any(
        _sha256_file(context["source"]) != context["source_digest"]
        for context in segment_contexts
    ):
        raise Native360FrameDatasetError(["native_360_lens_decode_source_changed"])
    if _sha256_file(executable) != runtime_digest:
        raise Native360FrameDatasetError(["native_360_lens_decode_runtime_changed"])
    manifest = {
        "schema_version": NATIVE_360_LENS_DECODE_SCHEMA_VERSION,
        "stable_run_identity": f"native-360-decode-{configuration_digest[7:31]}",
        "source_capture_identity": normalization_result["source_capture_identity"],
        "source_capture_digest": capture_digest,
        "original_file_references": source_references,
        "producing_method": "native_360_ffmpeg_lens_decoder.v1",
        "implementation_version": implementation_digest,
        "container_image_digest": None,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration": configuration,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": {
            "source_reference_set_digest": configuration[
                "source_reference_set_digest"
            ],
            "native_360_normalization_digest": normalization_result[
                "native_360_normalization_digest"
            ],
            "rig_declaration_digest": rig_declaration["rig_declaration_digest"],
            "dual_fisheye_binding_digest": dual_fisheye_binding[
                "dual_fisheye_binding_digest"
            ],
            "authority_digest": canonical_digest(authority_used),
        },
        "output_digests": {
            "decoded_frame_digests": [row["digest"] for row in decoded_frames]
        },
        "runtime_identity": runtime_identity,
        "runtime_digest": runtime_digest,
        "frames": decoded_frames,
        "decoded_frame_count": len(decoded_frames),
        "complete_retained_native_source_preserved": True,
        "original_distorted_pixels_preserved": True,
        "lens_identity_inferred": False,
        "calibration_inferred": False,
        "candidate_method_access_allowed": False,
        "access_scope": "trusted_dataset_compiler_only",
        "authority_used": dict(authority_used),
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "warnings": sorted(
            {
                warning
                for row in decoded_frames
                for warning, missing in (
                    ("decoded_dts_not_established_for_some_frames", row["source_dts_seconds"] is None),
                    ("decoded_duration_not_established_for_some_frames", row["duration_seconds"] is None),
                    ("decoded_keyframe_status_not_established_for_some_frames", row["key_frame"] is None),
                    ("decoded_exposure_metadata_not_established", True),
                )
                if missing
            }
        ),
        "blockers": [],
        "proof_effect": "decoded_native_lens_observation_availability_only",
        "claim_ceiling": "decoded_observation_availability",
        "parent_artifact_or_event": {
            "native_360_normalization_digest": normalization_result[
                "native_360_normalization_digest"
            ],
            "dual_fisheye_binding_digest": dual_fisheye_binding[
                "dual_fisheye_binding_digest"
            ],
        },
        "timestamp": compiled_at,
    }
    manifest["lens_decode_manifest_digest"] = canonical_digest(
        manifest, digest_field="lens_decode_manifest_digest"
    )
    return _write_immutable(manifest_path, manifest)


def compile_native_360_grouped_frame_dataset(
    *,
    artifact_root: str | Path,
    intake_id: str,
    capture_digest: str,
    normalization_result: Mapping[str, Any],
    rig_declaration: Mapping[str, Any],
    dual_fisheye_binding: Mapping[str, Any],
    lens_decode_manifest: Mapping[str, Any],
    decoded_lens_frames: Sequence[Mapping[str, Any]],
    runtime_identity: str,
    runtime_digest: str,
    implementation_digest: str,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
    parent_artifact_or_event: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile validated dual-fisheye segments into one frozen grouped timeline."""

    if not _is_digest(capture_digest):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_capture_digest_invalid"]
        )
    if any(
        authority_used.get(key) is not expected
        for key, expected in _REQUIRED_LOCAL_AUTHORITY.items()
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_authority_invalid"]
        )
    _validated_parent_artifacts(
        capture_digest=capture_digest,
        normalization_result=normalization_result,
        rig_declaration=rig_declaration,
        dual_fisheye_binding=dual_fisheye_binding,
    )
    lens_decode_manifest_digest = lens_decode_manifest.get(
        "lens_decode_manifest_digest"
    )
    decode_parent = lens_decode_manifest.get("parent_artifact_or_event")
    if (
        lens_decode_manifest.get("schema_version")
        != NATIVE_360_LENS_DECODE_SCHEMA_VERSION
        or lens_decode_manifest.get("source_capture_digest") != capture_digest
        or not _is_digest(lens_decode_manifest_digest)
        or lens_decode_manifest_digest
        != canonical_digest(
            lens_decode_manifest, digest_field="lens_decode_manifest_digest"
        )
        or lens_decode_manifest.get("runtime_identity") != runtime_identity
        or lens_decode_manifest.get("runtime_digest") != runtime_digest
        or lens_decode_manifest.get("candidate_method_access_allowed") is not False
        or lens_decode_manifest.get("blockers") != []
        or not isinstance(decode_parent, Mapping)
        or decode_parent.get("native_360_normalization_digest")
        != normalization_result.get("native_360_normalization_digest")
        or decode_parent.get("dual_fisheye_binding_digest")
        != dual_fisheye_binding.get("dual_fisheye_binding_digest")
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decode_manifest_invalid"]
        )
    if normalization_result.get("source_capture_identity") != intake_id:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_intake_identity_mismatch"]
        )
    segments = dual_fisheye_binding.get("segments")
    if not isinstance(segments, list) or not segments:
        raise Native360FrameDatasetError(["native_360_grouped_dataset_segments_invalid"])
    normalized_source_references = {
        (str(row.get("relative_path") or ""), row.get("digest"))
        for row in normalization_result.get("original_file_references", [])
        if isinstance(row, Mapping)
    }
    source_references: list[dict[str, str]] = []
    expected: dict[tuple[int, int, str], dict[str, Any]] = {}
    global_pair_index = 0
    for sequence_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or segment.get("sequence_index") != sequence_index:
            raise Native360FrameDatasetError(["native_360_grouped_dataset_segment_invalid"])
        files = segment.get("files")
        lens_streams = segment.get("lens_streams")
        frame_pairs = segment.get("frame_pairs")
        capture_timeline_start = _optional_finite(
            segment.get("capture_timeline_start_seconds"),
            code="native_360_grouped_dataset_capture_timeline_invalid",
            nonnegative=True,
        )
        if (
            capture_timeline_start is None
            or not isinstance(files, list)
            or len(files) != 1
            or not isinstance(lens_streams, list)
            or not isinstance(frame_pairs, list)
            or not frame_pairs
        ):
            raise Native360FrameDatasetError(["native_360_grouped_dataset_segment_invalid"])
        source_reference = files[0]
        if not isinstance(source_reference, Mapping):
            raise Native360FrameDatasetError(["native_360_grouped_dataset_source_invalid"])
        source_relative_path = str(source_reference.get("relative_path") or "")
        source_digest = str(source_reference.get("digest") or "")
        streams = {
            str(row.get("lens_id") or ""): row
            for row in lens_streams
            if isinstance(row, Mapping)
        }
        if (
            set(streams) != _LENS_IDS
            or not _is_digest(source_digest)
            or (source_relative_path, source_digest) not in normalized_source_references
        ):
            raise Native360FrameDatasetError(["native_360_grouped_dataset_source_invalid"])
        first_pair_pts = _finite(
            frame_pairs[0].get("front_pts_seconds"),
            code="native_360_grouped_dataset_frame_pts_invalid",
        )
        source_reference_row = {
            "relative_path": source_relative_path,
            "digest": source_digest,
        }
        if source_reference_row in source_references:
            raise Native360FrameDatasetError(
                ["native_360_grouped_dataset_source_segment_duplicate"]
            )
        source_references.append(source_reference_row)
        for pair in frame_pairs:
            if not isinstance(pair, Mapping):
                raise Native360FrameDatasetError(
                    ["native_360_grouped_dataset_frame_pair_invalid"]
                )
            pair_index = pair.get("pair_index")
            if isinstance(pair_index, bool) or not isinstance(pair_index, int):
                raise Native360FrameDatasetError(
                    ["native_360_grouped_dataset_frame_pair_invalid"]
                )
            group_reference_pts = _finite(
                pair.get("front_pts_seconds"),
                code="native_360_grouped_dataset_frame_pair_invalid",
            )
            capture_timeline_seconds = round(
                capture_timeline_start + group_reference_pts - first_pair_pts, 9
            )
            for lens_id in sorted(_LENS_IDS):
                stream = streams[lens_id]
                expected[(sequence_index, pair_index, lens_id)] = {
                    "source_relative_path": stream.get("source_relative_path"),
                    "source_digest": stream.get("source_digest"),
                    "stream_index": stream.get("stream_index"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "source_pts_seconds": pair.get(f"{lens_id}_pts_seconds"),
                    "source_dts_seconds": pair.get(f"{lens_id}_dts_seconds"),
                    "duration_seconds": pair.get(f"{lens_id}_duration_seconds"),
                    "key_frame": pair.get(f"{lens_id}_key_frame"),
                    "capture_timeline_seconds": capture_timeline_seconds,
                    "global_pair_index": global_pair_index,
                }
            global_pair_index += 1
    if len(expected) != global_pair_index * 2:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_frame_pair_duplicate"]
        )
    if lens_decode_manifest.get("original_file_references") != source_references:
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decode_source_set_mismatch"]
        )
    selected_frames: list[dict[str, Any]] = []
    observed: set[tuple[int, int, str]] = set()
    for ordinal, raw in enumerate(decoded_lens_frames):
        if not isinstance(raw, Mapping):
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_frame_invalid:{ordinal}"]
            )
        pair_index = raw.get("pair_index")
        lens_id = str(raw.get("lens_id") or "")
        segment_sequence_index = raw.get("segment_sequence_index")
        key = (segment_sequence_index, pair_index, lens_id)
        declared = expected.get(key)
        if declared is None or key in observed:
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_binding_invalid:{ordinal}"]
            )
        observed.add(key)
        pts = _finite(
            raw.get("source_pts_seconds"),
            code=f"native_360_grouped_dataset_frame_pts_invalid:{ordinal}",
        )
        declared_pts = _finite(
            declared["source_pts_seconds"],
            code="native_360_grouped_dataset_frame_pair_invalid",
        )
        if (
            raw.get("source_relative_path") != declared["source_relative_path"]
            or raw.get("source_digest") != declared["source_digest"]
            or raw.get("stream_index") != declared["stream_index"]
            or raw.get("decoded_frame_index") != pair_index
            or not math.isclose(pts, declared_pts, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_binding_invalid:{ordinal}"]
            )
        artifact_relative_path, image_metadata = _validated_image(
            root=Path(artifact_root).expanduser().resolve(),
            raw=raw,
            declared=declared,
            ordinal=ordinal,
        )
        source_dts_seconds = _optional_finite(
            raw.get("source_dts_seconds"),
            code=f"native_360_grouped_dataset_frame_dts_invalid:{ordinal}",
        )
        duration_seconds = _optional_finite(
            raw.get("duration_seconds"),
            code=f"native_360_grouped_dataset_frame_duration_invalid:{ordinal}",
            nonnegative=True,
        )
        declared_dts_seconds = _optional_finite(
            declared.get("source_dts_seconds"),
            code="native_360_grouped_dataset_frame_pair_invalid",
        )
        declared_duration_seconds = _optional_finite(
            declared.get("duration_seconds"),
            code="native_360_grouped_dataset_frame_pair_invalid",
            nonnegative=True,
        )
        key_frame = raw.get("key_frame")
        declared_key_frame = declared.get("key_frame")
        if (
            (source_dts_seconds is None) != (declared_dts_seconds is None)
            or (
                source_dts_seconds is not None
                and declared_dts_seconds is not None
                and not math.isclose(
                    source_dts_seconds, declared_dts_seconds, rel_tol=0.0, abs_tol=1e-9
                )
            )
            or (duration_seconds is None) != (declared_duration_seconds is None)
            or (
                duration_seconds is not None
                and declared_duration_seconds is not None
                and not math.isclose(
                    duration_seconds,
                    declared_duration_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            or (key_frame is not None and not isinstance(key_frame, bool))
            or key_frame is not declared_key_frame
        ):
            raise Native360FrameDatasetError(
                [f"native_360_grouped_dataset_decoded_timing_mismatch:{ordinal}"]
            )
        group_id = (
            f"segment-{segment_sequence_index:04d}-pair-{pair_index:09d}"
        )
        selected_frames.append(
            {
                "frame_id": f"{group_id}-{lens_id}",
                "decoded_frame_index": declared["global_pair_index"],
                "t_video_sec": declared["capture_timeline_seconds"],
                "source_pts_seconds": pts,
                "source_dts_seconds": source_dts_seconds,
                "duration_seconds": duration_seconds,
                "key_frame": key_frame,
                "artifact_relative_path": artifact_relative_path,
                "digest": raw.get("digest"),
                "image_metadata": image_metadata,
                "quality_signals": dict(raw.get("quality_signals") or {}),
                "source_camera_identity": lens_id,
                "observation_group_id": group_id,
            }
        )
    if observed != set(expected):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decoded_observations_incomplete"]
        )
    if canonical_digest({"frames": lens_decode_manifest.get("frames")}) != canonical_digest(
        {"frames": [dict(row) for row in decoded_lens_frames]}
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decode_manifest_frames_mismatch"]
        )

    decode_configuration_digest = lens_decode_manifest.get(
        "deterministic_configuration_digest"
    )
    if not _is_digest(decode_configuration_digest):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decode_manifest_invalid"]
        )
    decode_manifest_relative_path = (
        "native_360_lens_decode_"
        f"{str(decode_configuration_digest)[7:23]}/"
        "native_360_lens_decode_manifest.json"
    )
    decode_manifest_path = (
        Path(artifact_root).expanduser().resolve() / decode_manifest_relative_path
    )
    if (
        decode_manifest_path.is_symlink()
        or not decode_manifest_path.is_file()
    ):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_decode_manifest_artifact_missing"]
        )
    decode_manifest_artifact_digest = _sha256_file(decode_manifest_path)

    parent = dict(parent_artifact_or_event or {})
    native_parent = {
        "native_360_normalization_digest": normalization_result[
            "native_360_normalization_digest"
        ],
        "dual_fisheye_binding_digest": dual_fisheye_binding[
            "dual_fisheye_binding_digest"
        ],
        "lens_decode_manifest_digest": lens_decode_manifest_digest,
    }
    if any(key in parent and parent[key] != value for key, value in native_parent.items()):
        raise Native360FrameDatasetError(
            ["native_360_grouped_dataset_parent_binding_conflict"]
        )
    parent.update(native_parent)

    return compile_frozen_frame_dataset(
        artifact_root=artifact_root,
        intake_id=intake_id,
        capture_digest=capture_digest,
        capture_authority_profile="camera_360_native",
        source_video_relative_path=source_references[0]["relative_path"],
        source_video_digest=source_references[0]["digest"],
        decoded_frame_count=len(selected_frames),
        selected_frames=selected_frames,
        stream_metadata={
            "camera_representation": "calibrated_dual_fisheye_rig",
            "source_camera_identities": ["front", "rear"],
            "shared_physical_observation_groups": True,
            "group_timestamp_reference": "front_lens_decoded_pts",
            "capture_timeline_source": "declared_segment_start_plus_relative_front_pts",
            "native_segment_count": len(segments),
            "native_360_normalization_digest": normalization_result[
                "native_360_normalization_digest"
            ],
            "dual_fisheye_binding_digest": dual_fisheye_binding[
                "dual_fisheye_binding_digest"
            ],
            "lens_decode_manifest_digest": lens_decode_manifest_digest,
            "group_adapter_version": NATIVE_360_GROUPED_DATASET_ADAPTER_VERSION,
        },
        runtime_identity=runtime_identity,
        runtime_digest=runtime_digest,
        implementation_digest=implementation_digest,
        source_commit_sha=source_commit_sha,
        rights_and_retention=authority_used,
        selection_rule="evenly_spaced_actual_decoded_pts_with_endpoints_v1",
        parent_artifact=parent,
        timestamp=timestamp,
        camera_calibration_binding={
            "camera_360_rig_declaration_digest": rig_declaration[
                "rig_declaration_digest"
            ]
        },
        coordinate_frame_declaration=normalization_result[
            "coordinate_frame_declaration"
        ],
        supporting_artifact_references=[
            {
                "relative_path": decode_manifest_relative_path,
                "digest": decode_manifest_artifact_digest,
                "artifact_type": NATIVE_360_LENS_DECODE_SCHEMA_VERSION,
            }
        ],
        source_video_references=source_references,
    )


def build_native_360_dataset_compiler_service(
    *,
    capture_root: str | Path,
    intake_id: str,
    capture_digest: str,
    capture_build_digest: str,
    capture_reconstruction_route_digest: str,
    normalization_result: Mapping[str, Any],
    rig_declaration: Mapping[str, Any],
    dual_fisheye_binding: Mapping[str, Any],
    implementation_digest: str,
    source_commit_sha: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
    ffmpeg_executable: str | Path | None = None,
    timeout_seconds_per_frame: float = 120.0,
    maximum_command_output_bytes: int = 4 * 1024 * 1024,
    runner: ProbeRunner | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a trusted callable for the registered frozen-dataset tool."""

    if not _is_digest(capture_build_digest) or not _is_digest(
        capture_reconstruction_route_digest
    ):
        raise Native360FrameDatasetError(
            ["native_360_dataset_service_route_binding_invalid"]
        )

    def compiler(*, request: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
        if (
            request.get("capture_build_digest") != capture_build_digest
            or request.get("capture_reconstruction_route_digest")
            != capture_reconstruction_route_digest
            or request.get("capture_authority_profile") != "camera_360_native"
            or not isinstance(request.get("requested_claim_types"), list)
        ):
            raise Native360FrameDatasetError(
                ["native_360_dataset_service_request_binding_mismatch"]
            )
        decode_manifest = decode_native_360_lens_observations(
            capture_root=capture_root,
            artifact_root=output_root,
            capture_digest=capture_digest,
            normalization_result=normalization_result,
            rig_declaration=rig_declaration,
            dual_fisheye_binding=dual_fisheye_binding,
            implementation_digest=implementation_digest,
            source_commit_sha=source_commit_sha,
            authority_used=authority_used,
            timestamp=timestamp,
            ffmpeg_executable=ffmpeg_executable,
            timeout_seconds_per_frame=timeout_seconds_per_frame,
            maximum_command_output_bytes=maximum_command_output_bytes,
            runner=runner,
        )
        return compile_native_360_grouped_frame_dataset(
            artifact_root=output_root,
            intake_id=intake_id,
            capture_digest=capture_digest,
            normalization_result=normalization_result,
            rig_declaration=rig_declaration,
            dual_fisheye_binding=dual_fisheye_binding,
            lens_decode_manifest=decode_manifest,
            decoded_lens_frames=decode_manifest["frames"],
            runtime_identity=decode_manifest["runtime_identity"],
            runtime_digest=decode_manifest["runtime_digest"],
            implementation_digest=implementation_digest,
            source_commit_sha=source_commit_sha,
            authority_used=authority_used,
            timestamp=timestamp,
            parent_artifact_or_event={
                "capture_build_digest": capture_build_digest,
                "capture_reconstruction_route_digest": (
                    capture_reconstruction_route_digest
                ),
            },
        )

    return compiler


__all__ = [
    "NATIVE_360_GROUPED_DATASET_ADAPTER_VERSION",
    "NATIVE_360_LENS_DECODE_SCHEMA_VERSION",
    "Native360FrameDatasetError",
    "build_native_360_dataset_compiler_service",
    "compile_native_360_grouped_frame_dataset",
    "decode_native_360_lens_observations",
]
