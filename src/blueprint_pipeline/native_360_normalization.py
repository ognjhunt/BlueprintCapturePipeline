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
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


NATIVE_360_NORMALIZATION_SCHEMA_VERSION = "native_360_capture_normalization.v1"
DUAL_FISHEYE_BINDING_SCHEMA_VERSION = "dual_fisheye_stream_binding.v1"
CAMERA_360_RIG_SCHEMA_VERSION = "camera_360_rig_declaration.v1"
NATIVE_360_PROBE_SCHEMA_VERSION = "native_360_probe_receipt.v1"
_LENS_IDS = {"front", "rear"}
_MAX_NATIVE_SOURCE_BYTES = 100 * 1024 * 1024 * 1024


class Native360NormalizationError(ValueError):
    """Stable fail-closed native-360 normalization failure."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
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
            raise Native360NormalizationError(
                ["native_360_immutable_artifact_invalid"]
            ) from exc
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
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
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
    receipt["probe_receipt_digest"] = canonical_digest(
        receipt, digest_field="probe_receipt_digest"
    )
    return receipt


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


def _calibrated_rig(
    camera_metadata: Mapping[str, Any], *, capture_digest: str
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    calibrations = camera_metadata.get("lens_calibrations")
    calibrations = calibrations if isinstance(calibrations, list) else []
    by_lens: dict[str, dict[str, Any]] = {}
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
            or not _is_digest(raw.get("valid_pixel_mask_digest"))
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
            "valid_pixel_mask_digest": raw["valid_pixel_mask_digest"],
            "calibration_source": calibration_source,
            "calibration_source_digest": raw["calibration_source_digest"],
        }
    extrinsics = camera_metadata.get("rig_extrinsics")
    transform = _rigid_transform4(
        extrinsics.get("T_front_rear") if isinstance(extrinsics, Mapping) else None
    )
    extrinsics_source = (
        str(extrinsics.get("calibration_source") or "")
        if isinstance(extrinsics, Mapping)
        else ""
    )
    extrinsics_source_digest = (
        extrinsics.get("calibration_source_digest")
        if isinstance(extrinsics, Mapping)
        else None
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
    rig = {
        "schema_version": CAMERA_360_RIG_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "camera_model": str(camera_metadata.get("camera_model") or ""),
        "capture_mode": str(camera_metadata.get("capture_mode") or ""),
        "firmware_version": str(camera_metadata.get("firmware_version") or "unknown"),
        "lens_calibrations": [by_lens[lens_id] for lens_id in sorted(by_lens)],
        "rig_extrinsics": {
            "T_front_rear": transform,
            "calibration_source": extrinsics_source,
            "calibration_source_digest": extrinsics_source_digest,
        },
        "rig_is_fixed": transform is not None,
        "calibration_status": "valid" if not blockers else "invalid",
        "metric_scale_status": "not_established",
        "agent_may_alter_calibration": False,
        "blockers": sorted(set(blockers)),
    }
    rig["rig_declaration_digest"] = canonical_digest(
        rig, digest_field="rig_declaration_digest"
    )
    return rig, sorted(set(blockers))


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
    ):
        raise Native360NormalizationError(["native_360_normalization_limit_invalid"])
    required_authority = {
        "source_capture_rights_valid": True,
        "consent_valid": True,
        "privacy_review_valid": True,
        "retention_authorized": True,
        "local_processing_authorized": True,
        "provider_upload_authorized": False,
        "paid_compute_authorized": False,
    }
    if any(authority_used.get(key) is not expected for key, expected in required_authority.items()):
        raise Native360NormalizationError(["native_360_authority_invalid"])
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
    rig, blockers = _calibrated_rig(camera_metadata, capture_digest=capture_digest)
    calibration_by_lens = {
        str(row["lens_id"]): row
        for row in rig["lens_calibrations"]
        if isinstance(row, Mapping) and row.get("lens_id") in _LENS_IDS
    }
    normalized_segments: list[dict[str, Any]] = []
    total_source_bytes = 0
    runtime_digests: set[str] = set()
    source_file_references: list[dict[str, Any]] = []
    source_paths_seen: set[str] = set()
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
            runtime_digests.add(str(probe["runtime_digest"]))
            streams = {
                row.get("stream_index"): row
                for row in probe["streams"]
                if isinstance(row, Mapping)
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
            frame_pairs = [
                {
                    "pair_index": index,
                    "front_pts_seconds": left,
                    "rear_pts_seconds": right,
                    "absolute_residual_seconds": residuals[index],
                }
                for index, (left, right) in enumerate(
                    zip(front_pts, rear_pts, strict=True)
                )
            ] if same_shape else []
            synchronized = bool(
                same_shape
                and maximum_residual is not None
                and maximum_residual <= synchronization_tolerance_seconds
            )
            if not same_shape:
                blockers.append(f"native_360_lens_dimensions_or_counts_mismatch:{segment['sequence_index']}")
            elif not synchronized:
                blockers.append(f"native_360_lens_streams_unsynchronized:{segment['sequence_index']}")
            for lens_id, stream in (("front", front), ("rear", rear)):
                calibration = calibration_by_lens.get(lens_id)
                intrinsics = (
                    calibration.get("intrinsics")
                    if isinstance(calibration, Mapping)
                    else None
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
                "segment_id": str(segment.get("segment_id") or f"segment-{segment['sequence_index']:04d}"),
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
        elif declaration.get("status") == "available" and not _is_digest(
            declaration.get("digest")
        ):
            blockers.append(f"native_360_{sensor}_digest_missing")
    blockers = sorted(set(blockers))
    binding = {
        "schema_version": DUAL_FISHEYE_BINDING_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "camera_model": camera_metadata["camera_model"],
        "capture_mode": camera_metadata["capture_mode"],
        "segments": normalized_segments,
        "synchronization_tolerance_seconds": synchronization_tolerance_seconds,
        "all_segments_synchronized": all(row["synchronized"] for row in normalized_segments),
        "original_distorted_pixels_preserved": True,
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
            "rig_declaration_digest": rig["rig_declaration_digest"],
            "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
            "implementation_digest": implementation_digest,
            "source_commit_sha": source_commit_sha,
            "parent_artifact_digest": canonical_digest(
                dict(parent_artifact_or_event or {})
            ),
        }
    )
    artifact_root = (
        Path(output_root).expanduser().resolve()
        / f"native_360_normalization_{configuration_digest[7:23]}"
    )
    rig = _write_immutable(artifact_root / "camera_360_rig_declaration.json", rig)
    binding = _write_immutable(artifact_root / "dual_fisheye_stream_binding.json", binding)
    rig_reference = {
        "relative_path": "camera_360_rig_declaration.json",
        "digest": _sha256_file(artifact_root / "camera_360_rig_declaration.json"),
    }
    binding_reference = {
        "relative_path": "dual_fisheye_stream_binding.json",
        "digest": _sha256_file(artifact_root / "dual_fisheye_stream_binding.json"),
    }
    result_path = artifact_root / "native_360_capture_normalization.json"
    persisted_timestamp = compiled_at
    if result_path.exists():
        try:
            existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Native360NormalizationError(
                ["native_360_immutable_artifact_invalid"]
            ) from exc
        if not isinstance(existing_result, Mapping):
            raise Native360NormalizationError(
                ["native_360_immutable_artifact_invalid"]
            )
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
        "camera_metadata_digest": canonical_digest(camera_metadata),
        "probe_runtime_digest": next(iter(runtime_digests)) if len(runtime_digests) == 1 else None,
        "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
        "rig_declaration_digest": rig["rig_declaration_digest"],
        "input_digests": {
            "camera_metadata_digest": canonical_digest(camera_metadata),
            "authority_digest": canonical_digest(authority_used),
            "source_file_digests": sorted(
                row["digest"] for row in source_file_references
            ),
            "probe_receipt_digests": sorted(
                row["probe_receipt_digest"] for row in source_file_references
            ),
        },
        "output_digests": {
            "dual_fisheye_binding_digest": binding["dual_fisheye_binding_digest"],
            "rig_declaration_digest": rig["rig_declaration_digest"],
        },
        "artifact_references": {
            "camera_360_rig_declaration": rig_reference,
            "dual_fisheye_stream_binding": binding_reference,
        },
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
            "runtime_digest": (
                next(iter(runtime_digests)) if len(runtime_digests) == 1 else None
            ),
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


__all__ = [
    "CAMERA_360_RIG_SCHEMA_VERSION",
    "DUAL_FISHEYE_BINDING_SCHEMA_VERSION",
    "NATIVE_360_NORMALIZATION_SCHEMA_VERSION",
    "NATIVE_360_PROBE_SCHEMA_VERSION",
    "Native360NormalizationError",
    "build_native_360_probe_receipt",
    "normalize_native_360_capture",
]
