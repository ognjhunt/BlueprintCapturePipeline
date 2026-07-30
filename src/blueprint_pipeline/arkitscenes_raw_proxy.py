"""Compile one ARKitScenes Raw scene into a claim-limited reconstruction proxy.

ARKitScenes is useful because its MOV retains a timed metadata track that binds
decoded video PTS to original ARKit timestamps and camera intrinsics.  It is not
a Blueprint Raw Contract 3.2 bundle: it was captured on iPad, has no Blueprint
encoder-attempt ledger, and does not publish tracking/relocalization state.
This module preserves that boundary while exercising the real media, timing,
pose, intrinsics, depth-confidence, and frozen-split kernels locally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .arkit_depth_surface_compiler import compile_arkit_depth_surface
from .decision_evidence_contracts import canonical_digest, canonical_json
from .local_reconstruction_adapters import _probe_video, _tool_identity
from .reconstruction_frame_dataset import compile_frozen_frame_dataset


ARKITSCENES_PROXY_SCHEMA_VERSION = "arkitscenes_raw_proxy_compilation.v1"
ARKITSCENES_SCAFFOLD_SCHEMA_VERSION = "arkitscenes_metric_scaffold_proxy.v1"
ARKITSCENES_OBSERVATIONS_SCHEMA_VERSION = "arkitscenes_camera_observations_proxy.v1"
ARKITSCENES_PROXY_COMPILER_VERSION = "arkitscenes_raw_proxy_compiler.v1"
ARKITSCENES_OFFICIAL_HELPER_COMMIT = "7283761bf26c27570ec59a5dc0f8686fbff07726"
_REQUIRED_SOURCE_FILES = (
    "{video_id}.mov",
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
)


class ArkitScenesProxyError(ValueError):
    """Stable fail-closed error for public ARKitScenes proxy compilation."""

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


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(dict(value)))
    payload = (canonical_json(normalized) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArkitScenesProxyError(["arkitscenes_immutable_artifact_invalid"]) from exc
        if canonical_json(existing) != canonical_json(normalized):
            raise ArkitScenesProxyError(["arkitscenes_immutable_artifact_conflict"])
        return dict(existing)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(normalized):
            raise ArkitScenesProxyError(["arkitscenes_immutable_artifact_conflict"])
        return dict(existing)
    finally:
        temporary.unlink(missing_ok=True)
    return normalized


def _safe_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ArkitScenesProxyError([f"arkitscenes_source_symlink_forbidden:{relative}"])
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise ArkitScenesProxyError([f"arkitscenes_source_path_escape:{relative}"])
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ArkitScenesProxyError([f"arkitscenes_source_missing:{relative}"])
    return resolved


def _normalized_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArkitScenesProxyError(["arkitscenes_timestamp_invalid"]) from exc
    if parsed.tzinfo is None:
        raise ArkitScenesProxyError(["arkitscenes_timestamp_invalid"])
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _probe_packet_pts(ffprobe: str, video: Path, stream_index: int) -> list[float]:
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                str(stream_index),
                "-show_packets",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "json",
                str(video),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArkitScenesProxyError(["arkitscenes_metadata_pts_probe_failed"]) from exc
    if completed.returncode != 0:
        raise ArkitScenesProxyError(["arkitscenes_metadata_pts_probe_failed"])
    try:
        payload = json.loads(completed.stdout)
        values = [round(float(row["pts_time"]), 6) for row in payload["packets"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArkitScenesProxyError(["arkitscenes_metadata_pts_invalid"]) from exc
    if not values or any(right <= left for left, right in zip(values, values[1:])):
        raise ArkitScenesProxyError(["arkitscenes_metadata_pts_not_monotonic"])
    return values


def _extract_timed_metadata(ffmpeg: str, video: Path, stream_index: int) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(video),
                "-map",
                f"0:{stream_index}",
                "-c",
                "copy",
                "-f",
                "data",
                "pipe:1",
            ],
            check=False,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ArkitScenesProxyError(["arkitscenes_timed_metadata_extract_failed"]) from exc
    if completed.returncode != 0 or not completed.stdout:
        raise ArkitScenesProxyError(["arkitscenes_timed_metadata_extract_failed"])
    data = completed.stdout
    offset = 0
    records: list[dict[str, Any]] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ArkitScenesProxyError(["arkitscenes_timed_metadata_truncated"])
        record_size = int.from_bytes(data[offset : offset + 4], "big")
        record_type = int.from_bytes(data[offset + 4 : offset + 8], "big")
        if record_size < 9 or offset + record_size > len(data) or record_type != 1:
            raise ArkitScenesProxyError(["arkitscenes_timed_metadata_framing_invalid"])
        try:
            record = json.loads(data[offset + 8 : offset + record_size])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArkitScenesProxyError(["arkitscenes_timed_metadata_json_invalid"]) from exc
        if not isinstance(record, Mapping):
            raise ArkitScenesProxyError(["arkitscenes_timed_metadata_record_invalid"])
        records.append(dict(record))
        offset += record_size
    return records


def _capture_timestamp(record: Mapping[str, Any]) -> float:
    declaration = record.get("OriginalTimestampWhenWrittenToFile")
    if not isinstance(declaration, Mapping):
        raise ArkitScenesProxyError(["arkitscenes_original_timestamp_missing"])
    value = declaration.get("value")
    scale = declaration.get("timescale")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or isinstance(scale, bool)
        or not isinstance(scale, int)
        or scale <= 0
    ):
        raise ArkitScenesProxyError(["arkitscenes_original_timestamp_invalid"])
    return value / scale


def _intrinsics(record: Mapping[str, Any], *, width: int, height: int) -> dict[str, Any]:
    values = record.get("CameraIntrinsicMatrix")
    if not isinstance(values, list) or len(values) != 9:
        raise ArkitScenesProxyError(["arkitscenes_video_intrinsics_missing"])
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ArkitScenesProxyError(["arkitscenes_video_intrinsics_invalid"]) from exc
    if not all(math.isfinite(value) for value in numbers) or numbers[0] <= 0 or numbers[4] <= 0:
        raise ArkitScenesProxyError(["arkitscenes_video_intrinsics_invalid"])
    return {
        "fx": numbers[0],
        "fy": numbers[4],
        "cx": numbers[6],
        "cy": numbers[7],
        "width": width,
        "height": height,
        "matrix_storage": "column_major_as_recorded",
    }


def _axis_angle_camera_to_world(values: Sequence[float]) -> list[list[float]]:
    vector = np.asarray(values[:3], dtype=np.float64)
    translation = np.asarray(values[3:], dtype=np.float64)
    if (
        vector.shape != (3,)
        or translation.shape != (3,)
        or not np.isfinite(vector).all()
        or not np.isfinite(translation).all()
    ):
        raise ArkitScenesProxyError(["arkitscenes_trajectory_pose_invalid"])
    angle = float(np.linalg.norm(vector))
    if angle == 0.0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        axis = vector / angle
        cross = np.asarray(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        rotation = (
            np.eye(3)
            + math.sin(angle) * cross
            + (1.0 - math.cos(angle)) * (cross @ cross)
        )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation.T
    camera_to_world[:3, 3] = -(rotation.T @ translation)
    return [[round(float(item), 12) for item in row] for row in camera_to_world]


def _trajectory(path: Path) -> dict[float, dict[str, Any]]:
    result: dict[float, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArkitScenesProxyError(["arkitscenes_trajectory_unreadable"]) from exc
    for index, line in enumerate(lines):
        try:
            values = [float(value) for value in line.split()]
        except ValueError as exc:
            raise ArkitScenesProxyError([f"arkitscenes_trajectory_invalid:{index}"]) from exc
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ArkitScenesProxyError([f"arkitscenes_trajectory_invalid:{index}"])
        timestamp_ms = round(values[0], 3)
        if timestamp_ms in result:
            raise ArkitScenesProxyError(["arkitscenes_trajectory_timestamp_duplicate"])
        result[timestamp_ms] = {
            "source_timestamp_seconds": values[0],
            "T_world_camera": _axis_angle_camera_to_world(values[1:]),
        }
    if not result:
        raise ArkitScenesProxyError(["arkitscenes_trajectory_missing"])
    return result


def _timestamped_files(root: Path, *, suffix: str) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for path in sorted(root.glob(f"*{suffix}")):
        if path.is_symlink() or not path.is_file():
            raise ArkitScenesProxyError(["arkitscenes_observation_symlink_forbidden"])
        try:
            timestamp = float(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ArkitScenesProxyError(["arkitscenes_observation_filename_invalid"]) from exc
        if timestamp in result:
            raise ArkitScenesProxyError(["arkitscenes_observation_timestamp_duplicate"])
        result[timestamp] = path
    if not result:
        raise ArkitScenesProxyError(["arkitscenes_observations_missing"])
    return result


def _select_evenly(values: Sequence[float], maximum: int) -> list[float]:
    if maximum < 3:
        raise ArkitScenesProxyError(["arkitscenes_maximum_frames_invalid"])
    if len(values) <= maximum:
        return list(values)
    indexes = {
        round(ordinal * (len(values) - 1) / (maximum - 1))
        for ordinal in range(maximum)
    }
    if len(indexes) != maximum:
        raise ArkitScenesProxyError(["arkitscenes_frame_selection_ambiguous"])
    return [values[index] for index in sorted(indexes)]


def _extract_selected_frames(
    *,
    ffmpeg: str,
    video: Path,
    selected: Sequence[tuple[int, str]],
    destination: Path,
) -> dict[int, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {index: destination / f"{frame_id}.png" for index, frame_id in selected}
    if all(path.is_file() and not path.is_symlink() for path in expected.values()):
        return expected
    expression = "+".join(f"eq(n\\,{index})" for index, _ in selected)
    with tempfile.TemporaryDirectory(prefix="arkitscenes-decode-", dir=destination) as temp_name:
        pattern = Path(temp_name) / "selected-%06d.png"
        try:
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-noautorotate",
                    "-i",
                    str(video),
                    "-map",
                    "0:v:0",
                    "-vf",
                    f"select={expression}",
                    "-vsync",
                    "0",
                    "-start_number",
                    "0",
                    "-y",
                    str(pattern),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ArkitScenesProxyError(["arkitscenes_selected_frame_decode_failed"]) from exc
        outputs = sorted(Path(temp_name).glob("selected-*.png"))
        if completed.returncode != 0 or len(outputs) != len(selected):
            raise ArkitScenesProxyError(["arkitscenes_selected_frame_decode_failed"])
        for output, (index, _) in zip(outputs, selected, strict=True):
            target = expected[index]
            if target.exists():
                if target.is_symlink() or _sha256_file(target) != _sha256_file(output):
                    raise ArkitScenesProxyError(["arkitscenes_decoded_frame_conflict"])
            else:
                output.replace(target)
    return expected


def _load_artifact(root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(reference.get("relative_path") or "")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArkitScenesProxyError(["arkitscenes_dataset_artifact_invalid"]) from exc
    if not isinstance(value, Mapping):
        raise ArkitScenesProxyError(["arkitscenes_dataset_artifact_invalid"])
    return dict(value)


def _scaffold_artifact(
    *,
    capture_digest: str,
    dataset_manifest_digest: str,
    split_digest: str,
    access_scope: str,
    camera_frames: Sequence[Mapping[str, Any]],
    depth_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    depth_by_frame = {str(row["frame_id"]): row for row in depth_pairs}
    scoped_pairs = [depth_by_frame[str(row["frame_id"])] for row in camera_frames]
    artifact: dict[str, Any] = {
        "schema_version": ARKITSCENES_SCAFFOLD_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "dataset_manifest_digest": dataset_manifest_digest,
        "split_digest": split_digest,
        "access_scope": access_scope,
        "coordinate_frame": {
            "declaration": "arkitscenes_official_loader_camera_to_world",
            "units": "meters",
            "handedness": "not_explicitly_declared_by_dataset",
            "gravity_alignment": "not_independently_validated",
        },
        "camera_frames": list(camera_frames),
        "depth_confidence_pairs": scoped_pairs,
        "confidence_filter": "confidence_equals_2_and_positive_depth",
        "total_depth_pixels": sum(192 * 256 for _ in scoped_pairs),
        "positive_source_depth_pixels": sum(
            int(row["positive_source_depth_pixel_count"]) for row in scoped_pairs
        ),
        "accepted_high_confidence_depth_pixels": sum(
            int(row["accepted_pixel_count"]) for row in scoped_pairs
        ),
        "unseen_or_rejected_depth_filled": False,
        "metric_scale_status": "dataset_declared_not_independently_validated",
        "collision_geometry_status": "not_compiled",
    }
    artifact["metric_scaffold_digest"] = canonical_digest(
        artifact, digest_field="metric_scaffold_digest"
    )
    return artifact


def compile_arkitscenes_raw_proxy(
    *,
    scene_root: str | Path,
    output_root: str | Path,
    video_id: str,
    split: str,
    maximum_selected_frames: int,
    source_commit_sha: str,
    implementation_digest: str,
    authority_used: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Compile a real ARKitScenes scene without promoting it to Blueprint raw truth."""

    compiled_at = _normalized_timestamp(timestamp)
    if (
        not video_id.isdigit()
        or split not in {"Training", "Validation"}
        or len(source_commit_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_commit_sha)
        or not _is_digest(implementation_digest)
    ):
        raise ArkitScenesProxyError(["arkitscenes_source_binding_invalid"])
    if (
        authority_used.get("arkitscenes_license_accepted") is not True
        or authority_used.get("local_processing_authorized") is not True
        or authority_used.get("provider_upload_authorized") is not False
        or authority_used.get("paid_compute_authorized") is not False
    ):
        raise ArkitScenesProxyError(["arkitscenes_authority_invalid"])
    root = Path(scene_root).expanduser().resolve()
    source = root / "source"
    extracted = root / "extracted"
    if not source.is_dir() or not extracted.is_dir():
        raise ArkitScenesProxyError(["arkitscenes_scene_layout_invalid"])
    source_references: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    for template in _REQUIRED_SOURCE_FILES:
        name = template.format(video_id=video_id)
        path = _safe_file(source, name)
        source_paths[name] = path
        source_references.append(
            {"relative_path": f"source/{name}", "digest": _sha256_file(path), "size_bytes": path.stat().st_size}
        )
    capture_digest = canonical_digest(
        {"dataset": "ARKitScenes Raw", "video_id": video_id, "split": split, "source_files": source_references}
    )
    ffprobe, ffmpeg, runtime_identity, runtime_digest = _tool_identity()
    probe = _probe_video(source_paths[f"{video_id}.mov"], ffprobe)
    stream = probe["stream"]
    if stream.get("width") != 1920 or stream.get("height") != 1440:
        raise ArkitScenesProxyError(["arkitscenes_video_dimensions_unexpected"])
    metadata_records = _extract_timed_metadata(ffmpeg, source_paths[f"{video_id}.mov"], 1)
    metadata_pts = _probe_packet_pts(ffprobe, source_paths[f"{video_id}.mov"], 1)
    if len(metadata_records) != len(metadata_pts):
        raise ArkitScenesProxyError(["arkitscenes_metadata_packet_count_mismatch"])
    metadata_by_timestamp: dict[float, dict[str, Any]] = {}
    metadata_by_pts: dict[float, dict[str, Any]] = {}
    for pts, record in zip(metadata_pts, metadata_records, strict=True):
        capture_timestamp = _capture_timestamp(record)
        rounded_timestamp = round(capture_timestamp, 3)
        if rounded_timestamp in metadata_by_timestamp or pts in metadata_by_pts:
            raise ArkitScenesProxyError(["arkitscenes_metadata_binding_duplicate"])
        binding = {"capture_timestamp_seconds": capture_timestamp, "video_pts_seconds": pts, "metadata": record}
        metadata_by_timestamp[rounded_timestamp] = binding
        metadata_by_pts[pts] = binding
    decoded_source_pts = [
        round(float(row["source_pts_seconds"]), 6) for row in probe["frames"]
    ]
    decoded_index_by_source_pts = {
        pts: index for index, pts in enumerate(decoded_source_pts)
    }
    if len(decoded_index_by_source_pts) != len(decoded_source_pts):
        raise ArkitScenesProxyError(["arkitscenes_decoded_pts_duplicate"])
    decoded_without_metadata = sorted(set(decoded_source_pts) - set(metadata_by_pts))
    metadata_without_decoded = sorted(set(metadata_by_pts) - set(decoded_source_pts))
    if decoded_without_metadata:
        raise ArkitScenesProxyError(["arkitscenes_decoded_frame_metadata_missing"])
    rgb = _timestamped_files(extracted / "lowres_wide", suffix=".png")
    depth = _timestamped_files(extracted / "lowres_depth", suffix=".png")
    confidence = _timestamped_files(extracted / "confidence", suffix=".png")
    intrinsics = _timestamped_files(extracted / "lowres_wide_intrinsics", suffix=".pincam")
    if set(rgb) != set(metadata_by_timestamp):
        raise ArkitScenesProxyError(["arkitscenes_rgb_metadata_timestamp_mismatch"])
    trajectory = _trajectory(source_paths["lowres_wide.traj"])
    eligible = sorted(set(trajectory) & set(rgb) & set(depth) & set(confidence) & set(intrinsics))
    eligible = [
        value
        for value in eligible
        if metadata_by_timestamp[value]["video_pts_seconds"]
        in decoded_index_by_source_pts
    ]
    if len(eligible) != len(trajectory):
        raise ArkitScenesProxyError(["arkitscenes_trajectory_observation_binding_incomplete"])
    selected_timestamps = _select_evenly(eligible, maximum_selected_frames)
    configuration = {
        "compiler_version": ARKITSCENES_PROXY_COMPILER_VERSION,
        "capture_digest": capture_digest,
        "maximum_selected_frames": maximum_selected_frames,
        "selection_rule": "arkitscenes_exact_trajectory_timestamp_even_coverage_v1",
        "confidence_rule": "retain_only_arkit_confidence_2_and_positive_depth_v1",
        "implementation_digest": implementation_digest,
        "source_commit_sha": source_commit_sha,
        "runtime_digest": runtime_digest,
    }
    configuration_digest = canonical_digest(configuration)
    artifact_root = Path(output_root).expanduser().resolve() / f"arkitscenes_proxy_{configuration_digest[7:23]}"
    selected_bindings: list[dict[str, Any]] = []
    decode_requests: list[tuple[int, str]] = []
    for rounded_timestamp in selected_timestamps:
        metadata_binding = metadata_by_timestamp[rounded_timestamp]
        pts = metadata_binding["video_pts_seconds"]
        decoded_index = decoded_index_by_source_pts[pts]
        t_video_sec = float(probe["presentation_times_seconds"][decoded_index])
        frame_id = f"arkitscenes-{video_id}-{int(round(rounded_timestamp * 1000)):010d}"
        selected_bindings.append(
            {
                "frame_id": frame_id,
                "rounded_capture_timestamp_seconds": rounded_timestamp,
                "capture_timestamp_seconds": metadata_binding["capture_timestamp_seconds"],
                "source_video_pts_seconds": pts,
                "t_video_sec": t_video_sec,
                "decoded_frame_index": decoded_index,
                "metadata": metadata_binding["metadata"],
                "trajectory": trajectory[rounded_timestamp],
            }
        )
        decode_requests.append((decoded_index, frame_id))
    decoded_paths = _extract_selected_frames(
        ffmpeg=ffmpeg,
        video=source_paths[f"{video_id}.mov"],
        selected=decode_requests,
        destination=artifact_root / "decoded_observations",
    )
    selected_frames: list[dict[str, Any]] = []
    camera_frames: list[dict[str, Any]] = []
    depth_pairs: list[dict[str, Any]] = []
    high_confidence_pixels = 0
    positive_depth_pixels = 0
    total_depth_pixels = 0
    for binding in selected_bindings:
        decoded_path = decoded_paths[binding["decoded_frame_index"]]
        with Image.open(decoded_path) as image:
            if image.size != (1920, 1440):
                raise ArkitScenesProxyError(["arkitscenes_decoded_frame_dimensions_invalid"])
            gray = np.asarray(image.convert("L"), dtype=np.float32)
        horizontal = np.diff(gray, axis=1)
        vertical = np.diff(gray, axis=0)
        frame_digest = _sha256_file(decoded_path)
        selected_frames.append(
            {
                "frame_id": binding["frame_id"],
                "decoded_frame_index": binding["decoded_frame_index"],
                "t_video_sec": binding["t_video_sec"],
                "source_pts_seconds": binding["source_video_pts_seconds"],
                "digest": frame_digest,
                "artifact_relative_path": decoded_path.relative_to(artifact_root).as_posix(),
                "image_metadata": {"width": 1920, "height": 1440, "pixel_orientation": "encoded_source_no_autorotate"},
                "quality_signals": {
                    "mean_luma_0_255": round(float(np.mean(gray)), 6),
                    "gradient_energy": round(
                        float(
                            np.mean(horizontal * horizontal)
                            + np.mean(vertical * vertical)
                        ),
                        6,
                    ),
                    "excessive_blur_deterministically_established": False,
                    "exposure_metadata": dict(binding["metadata"].get("MetadataDictionary") or {}),
                },
            }
        )
        rounded_timestamp = binding["rounded_capture_timestamp_seconds"]
        try:
            lowres_values = [float(value) for value in intrinsics[rounded_timestamp].read_text(encoding="utf-8").split()]
        except (OSError, ValueError) as exc:
            raise ArkitScenesProxyError(["arkitscenes_lowres_intrinsics_invalid"]) from exc
        if len(lowres_values) != 6 or lowres_values[0:2] != [256.0, 192.0]:
            raise ArkitScenesProxyError(["arkitscenes_lowres_intrinsics_invalid"])
        rgb_intrinsics = _intrinsics(binding["metadata"], width=1920, height=1440)
        lowres_intrinsics = {
            "width": 256,
            "height": 192,
            "fx": lowres_values[2],
            "fy": lowres_values[3],
            "cx": lowres_values[4],
            "cy": lowres_values[5],
        }
        with Image.open(depth[rounded_timestamp]) as depth_image:
            depth_array = np.asarray(depth_image, dtype=np.uint16)
        with Image.open(confidence[rounded_timestamp]) as confidence_image:
            confidence_array = np.asarray(confidence_image, dtype=np.uint8)
        if depth_array.shape != (192, 256) or confidence_array.shape != depth_array.shape:
            raise ArkitScenesProxyError(["arkitscenes_depth_confidence_dimensions_invalid"])
        if not set(np.unique(confidence_array).tolist()) <= {0, 1, 2}:
            raise ArkitScenesProxyError(["arkitscenes_depth_confidence_values_invalid"])
        accepted = (confidence_array == 2) & (depth_array > 0)
        filtered = np.where(accepted, depth_array, 0).astype(np.uint16)
        filtered_path = (
            artifact_root
            / "metric_scaffold"
            / "high_confidence_depth"
            / f"{binding['frame_id']}.png"
        )
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        if filtered_path.exists():
            with Image.open(filtered_path) as existing_image:
                if not np.array_equal(np.asarray(existing_image, dtype=np.uint16), filtered):
                    raise ArkitScenesProxyError(["arkitscenes_filtered_depth_conflict"])
        else:
            descriptor, temporary_name = tempfile.mkstemp(suffix=".png", dir=filtered_path.parent)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                Image.fromarray(filtered).save(temporary, format="PNG")
                temporary.replace(filtered_path)
            finally:
                temporary.unlink(missing_ok=True)
        accepted_count = int(np.count_nonzero(accepted))
        positive_count = int(np.count_nonzero(depth_array > 0))
        total_depth_pixels += int(depth_array.size)
        positive_depth_pixels += positive_count
        high_confidence_pixels += accepted_count
        camera_frames.append(
            {
                "frame_id": binding["frame_id"],
                "decoded_frame_index": binding["decoded_frame_index"],
                "t_video_sec": binding["t_video_sec"],
                "source_pts_seconds": binding["source_video_pts_seconds"],
                "t_capture_sec": binding["capture_timestamp_seconds"],
                "T_world_camera": binding["trajectory"]["T_world_camera"],
                "rgb_intrinsics": rgb_intrinsics,
                "depth_intrinsics": lowres_intrinsics,
                "pose_source": "arkitscenes_official_trajectory_exact_timestamp",
            }
        )
        depth_pairs.append(
            {
                "frame_id": binding["frame_id"],
                "source_depth_relative_path": depth[rounded_timestamp].relative_to(root).as_posix(),
                "source_depth_digest": _sha256_file(depth[rounded_timestamp]),
                "source_confidence_relative_path": confidence[rounded_timestamp]
                .relative_to(root)
                .as_posix(),
                "source_confidence_digest": _sha256_file(confidence[rounded_timestamp]),
                "filtered_depth_relative_path": filtered_path.relative_to(artifact_root).as_posix(),
                "filtered_depth_digest": _sha256_file(filtered_path),
                "accepted_pixel_count": accepted_count,
                "positive_source_depth_pixel_count": positive_count,
                "missing_or_rejected_pixels_preserved_as_zero": True,
            }
        )
    dataset = compile_frozen_frame_dataset(
        artifact_root=artifact_root,
        intake_id=f"arkitscenes-{video_id}",
        capture_digest=capture_digest,
        capture_authority_profile="public_dataset_arkitscenes_proxy",
        source_video_relative_path=f"source/{video_id}.mov",
        source_video_digest=source_references[0]["digest"],
        decoded_frame_count=probe["frame_count"],
        selected_frames=selected_frames,
        stream_metadata={
            **dict(stream),
            "declared_video_sample_count": len(metadata_pts),
            "decoded_frame_count": probe["frame_count"],
            "metadata_without_decoded_pts": metadata_without_decoded,
            "exact_timestamp_eligible_pose_count": len(eligible),
        },
        runtime_identity=runtime_identity,
        runtime_digest=runtime_digest,
        implementation_digest=implementation_digest,
        source_commit_sha=source_commit_sha,
        rights_and_retention=authority_used,
        selection_rule="arkitscenes_exact_trajectory_timestamp_even_coverage_v1",
        parent_artifact={"dataset": "ARKitScenes Raw", "video_id": video_id, "split": split},
        timestamp=compiled_at,
    )
    dataset_root = artifact_root
    split_manifest = _load_artifact(dataset_root, dataset["artifact_references"]["frozen_split_manifest"])
    candidate_manifest = _load_artifact(dataset_root, dataset["artifact_references"]["candidate_dataset_manifest"])
    heldout_manifest = _load_artifact(
        dataset_root,
        dataset["artifact_references"]["hidden_heldout_evaluator_manifest"],
    )
    dataset_directory = Path(
        str(dataset["artifact_references"]["candidate_dataset_manifest"]["relative_path"])
    ).parent
    camera_by_index = {row["decoded_frame_index"]: row for row in camera_frames}
    depth_by_frame = {row["frame_id"]: row for row in depth_pairs}
    observations = {
        "schema_version": ARKITSCENES_OBSERVATIONS_SCHEMA_VERSION,
        "capture_digest": capture_digest,
        "dataset_manifest_digest": dataset["dataset_manifest_digest"],
        "split_digest": split_manifest["split_digest"],
        "candidate_splits_only": True,
        "hidden_heldout_pixels_included": False,
        "observations": [
            {
                "observation_id": row["frame_id"],
                "decoded_frame_index": row["decoded_frame_index"],
                "t_video_sec": row["t_video_sec"],
                "split": row["split"],
                "image_relative_path": (
                    dataset_directory / row["candidate_relative_path"]
                ).as_posix(),
                "image_digest": row["frame_digest"],
                "camera": camera_by_index[row["decoded_frame_index"]],
                "depth_confidence_binding": depth_by_frame[row["frame_id"]],
            }
            for row in candidate_manifest["frames"]
        ],
        "candidate_may_access_hidden_heldout": False,
        "candidate_may_modify_poses_or_calibration": False,
    }
    observations["camera_observation_digest"] = canonical_digest(
        observations, digest_field="camera_observation_digest"
    )
    observations = _write_immutable_json(
        artifact_root / "camera_observations_proxy.json", observations
    )
    candidate_ids = {str(row["frame_id"]) for row in candidate_manifest["frames"]}
    heldout_ids = {str(row["frame_id"]) for row in heldout_manifest["frames"]}
    if candidate_ids & heldout_ids or candidate_ids | heldout_ids != {
        str(row["frame_id"]) for row in camera_frames
    }:
        raise ArkitScenesProxyError(["arkitscenes_split_scope_binding_invalid"])
    candidate_scaffold = _scaffold_artifact(
        capture_digest=capture_digest,
        dataset_manifest_digest=dataset["dataset_manifest_digest"],
        split_digest=split_manifest["split_digest"],
        access_scope="candidate_training_and_validation_only",
        camera_frames=[
            row for row in camera_frames if str(row["frame_id"]) in candidate_ids
        ],
        depth_pairs=depth_pairs,
    )
    candidate_scaffold = _write_immutable_json(
        artifact_root / "candidate_metric_scaffold_proxy.json", candidate_scaffold
    )
    evaluator_scaffold = _scaffold_artifact(
        capture_digest=capture_digest,
        dataset_manifest_digest=dataset["dataset_manifest_digest"],
        split_digest=split_manifest["split_digest"],
        access_scope="independent_evaluator_only",
        camera_frames=[
            row for row in camera_frames if str(row["frame_id"]) in heldout_ids
        ],
        depth_pairs=depth_pairs,
    )
    evaluator_scaffold = _write_immutable_json(
        artifact_root / "evaluator_hidden" / "metric_scaffold_proxy.json",
        evaluator_scaffold,
    )
    scaffold_summary = {
        "coordinate_frame": candidate_scaffold["coordinate_frame"],
        "metric_scale_status": candidate_scaffold["metric_scale_status"],
        "selected_frame_count": len(camera_frames),
        "candidate_frame_count": len(candidate_ids),
        "heldout_frame_count": len(heldout_ids),
        "total_depth_pixels": total_depth_pixels,
        "positive_source_depth_pixels": positive_depth_pixels,
        "accepted_high_confidence_depth_pixels": high_confidence_pixels,
        "candidate_and_evaluator_scaffolds_disjoint": True,
    }
    output_digests = {
        "dataset_manifest_digest": dataset["dataset_manifest_digest"],
        "split_digest": split_manifest["split_digest"],
        "camera_observation_digest": observations["camera_observation_digest"],
        "candidate_metric_scaffold_digest": candidate_scaffold[
            "metric_scaffold_digest"
        ],
        "evaluator_metric_scaffold_digest": evaluator_scaffold[
            "metric_scaffold_digest"
        ],
    }
    report = {
        "schema_version": ARKITSCENES_PROXY_SCHEMA_VERSION,
        "stable_run_identity": f"arkitscenes-proxy-{configuration_digest[7:31]}",
        "source_capture_identity": f"arkitscenes-{video_id}",
        "source_capture_digest": capture_digest,
        "original_file_references": source_references,
        "producing_method": ARKITSCENES_PROXY_COMPILER_VERSION,
        "implementation_version": implementation_digest,
        "container_image_digest": None,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": {"source_capture_digest": capture_digest, "runtime_digest": runtime_digest},
        "output_digests": output_digests,
        "train_heldout_split_digest": split_manifest["split_digest"],
        "camera_calibration_binding": observations["camera_observation_digest"],
        "coordinate_frame_declaration": scaffold_summary["coordinate_frame"],
        "units": "rgb_pixels_depth_millimeters_pose_meters",
        "metric_scale_status": scaffold_summary["metric_scale_status"],
        "provider_runtime_identity": {"provider": "local", "runtime_identity": runtime_identity, "runtime_digest": runtime_digest},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": dict(authority_used),
        "warnings": [
            "arkitscenes_was_captured_on_ipad_not_iphone",
            "arkitscenes_coordinate_handedness_not_explicitly_declared",
            "local_compilation_duration_not_measured",
        ],
        "blockers": [
            "blueprint_encoder_attempt_and_retention_ledger_missing",
            "arkit_tracking_and_relocalization_state_missing",
            "metric_scale_not_independently_validated",
            "collider_not_compiled_or_qualified",
            "isaac_not_verified",
        ],
        "proof_effect": "public_dataset_proxy_pipeline_evidence_only",
        "claim_ceiling": "public_dataset_calibrated_observation_proxy",
        "parent_artifact_or_event": {"dataset": "ARKitScenes Raw", "video_id": video_id, "split": split},
        "timestamp": dataset["timestamp"],
        "status": "partial",
        "decoded_frame_count": probe["frame_count"],
        "declared_video_sample_count": len(metadata_pts),
        "metadata_samples_without_decoded_frames": metadata_without_decoded,
        "exact_timestamp_eligible_pose_count": len(eligible),
        "selected_frame_count": len(selected_frames),
        "hidden_heldout_pixels_exposed_to_candidate": False,
        "raw_contract_3_2_proven": False,
        "iphone_route_proven": False,
        "metric_geometry_proven": False,
        "collision_or_physics_proven": False,
        "isaac_compatibility_proven": False,
    }
    report["arkitscenes_proxy_compilation_digest"] = canonical_digest(report, digest_field="arkitscenes_proxy_compilation_digest")
    return _write_immutable_json(artifact_root / "arkitscenes_raw_proxy_compilation.json", report)


def compile_arkitscenes_depth_surface_proxy(
    *, scene_root: str | Path, proxy_artifact_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Back-project candidate-only public iPad depth using Apple's helper convention."""

    scene = Path(scene_root).expanduser().resolve(strict=True)
    proxy_root = Path(proxy_artifact_root).expanduser().resolve(strict=True)
    try:
        report = json.loads(
            _safe_file(proxy_root, "arkitscenes_raw_proxy_compilation.json").read_text()
        )
        observations = json.loads(
            _safe_file(proxy_root, "camera_observations_proxy.json").read_text()
        )
        scaffold = json.loads(
            _safe_file(proxy_root, "candidate_metric_scaffold_proxy.json").read_text()
        )
    except json.JSONDecodeError as exc:
        raise ArkitScenesProxyError(["arkitscenes_proxy_artifact_invalid_json"]) from exc
    if (
        report.get("schema_version") != ARKITSCENES_PROXY_SCHEMA_VERSION
        or report.get("arkitscenes_proxy_compilation_digest")
        != canonical_digest(report, digest_field="arkitscenes_proxy_compilation_digest")
        or observations.get("schema_version") != ARKITSCENES_OBSERVATIONS_SCHEMA_VERSION
        or observations.get("camera_observation_digest")
        != canonical_digest(observations, digest_field="camera_observation_digest")
        or scaffold.get("schema_version") != ARKITSCENES_SCAFFOLD_SCHEMA_VERSION
        or scaffold.get("metric_scaffold_digest")
        != canonical_digest(scaffold, digest_field="metric_scaffold_digest")
    ):
        raise ArkitScenesProxyError(["arkitscenes_proxy_artifact_digest_invalid"])
    if (
        observations.get("candidate_splits_only") is not True
        or observations.get("hidden_heldout_pixels_included") is not False
        or observations.get("candidate_may_access_hidden_heldout") is not False
        or scaffold.get("access_scope") != "candidate_training_and_validation_only"
        or scaffold.get("unseen_or_rejected_depth_filled") is not False
    ):
        raise ArkitScenesProxyError(["arkitscenes_proxy_candidate_scope_invalid"])
    capture_digest = report["source_capture_digest"]
    if (
        observations.get("capture_digest") != capture_digest
        or scaffold.get("capture_digest") != capture_digest
        or observations.get("split_digest") != scaffold.get("split_digest")
    ):
        raise ArkitScenesProxyError(["arkitscenes_proxy_source_binding_mismatch"])
    original_references: dict[tuple[str, str], dict[str, Any]] = {}
    for index, reference in enumerate(report.get("original_file_references", [])):
        if not isinstance(reference, Mapping):
            raise ArkitScenesProxyError(["arkitscenes_proxy_original_reference_invalid"])
        relative = str(reference.get("relative_path") or "")
        digest = str(reference.get("digest") or "")
        if _sha256_file(_safe_file(scene, relative)) != digest:
            raise ArkitScenesProxyError(["arkitscenes_proxy_original_digest_mismatch"])
        original_references[(relative, digest)] = {
            "artifact_id": f"arkitscenes-source-{index:02d}",
            "relative_path": relative,
            "digest": digest,
        }
    frames: list[dict[str, Any]] = []
    for observation in observations.get("observations", []):
        if not isinstance(observation, Mapping) or observation.get("split") not in {
            "training",
            "validation",
        }:
            raise ArkitScenesProxyError(["arkitscenes_proxy_hidden_or_invalid_split"])
        camera = observation.get("camera")
        binding = observation.get("depth_confidence_binding")
        if not isinstance(camera, Mapping) or not isinstance(binding, Mapping):
            raise ArkitScenesProxyError(["arkitscenes_proxy_depth_camera_binding_missing"])
        frame_id = str(camera.get("frame_id") or "")
        if frame_id != binding.get("frame_id") or frame_id != observation.get(
            "observation_id"
        ):
            raise ArkitScenesProxyError(["arkitscenes_proxy_frame_binding_mismatch"])
        depth_relative = str(binding.get("source_depth_relative_path") or "")
        confidence_relative = str(binding.get("source_confidence_relative_path") or "")
        depth_digest = _sha256_file(_safe_file(scene, depth_relative))
        confidence_digest = _sha256_file(_safe_file(scene, confidence_relative))
        if (
            depth_digest != binding.get("source_depth_digest")
            or confidence_digest != binding.get("source_confidence_digest")
        ):
            raise ArkitScenesProxyError(["arkitscenes_proxy_depth_digest_mismatch"])
        for label, relative, digest in (
            ("depth", depth_relative, depth_digest),
            ("confidence", confidence_relative, confidence_digest),
        ):
            original_references[(relative, digest)] = {
                "artifact_id": f"arkitscenes-{label}-{frame_id}",
                "relative_path": relative,
                "digest": digest,
            }
        frames.append(
            {
                "frame_id": frame_id,
                "split": observation["split"],
                "region_id": "arkitscenes-observed-frusta",
                "depth_asset": {
                    "relative_path": depth_relative,
                    "digest": depth_digest,
                    "encoding": "uint16_png",
                    "scale_to_meters": 0.001,
                },
                "confidence_asset": {
                    "relative_path": confidence_relative,
                    "digest": confidence_digest,
                    "encoding": "uint8_png",
                },
                "depth_intrinsics": camera.get("depth_intrinsics"),
                "T_world_camera": camera.get("T_world_camera"),
            }
        )
    configuration = {
        "adapter": "arkitscenes_depth_surface_proxy.v1",
        "arkitscenes_proxy_compilation_digest": report[
            "arkitscenes_proxy_compilation_digest"
        ],
        "camera_observation_digest": observations["camera_observation_digest"],
        "metric_scaffold_digest": scaffold["metric_scaffold_digest"],
        "official_helper_commit": ARKITSCENES_OFFICIAL_HELPER_COMMIT,
        "camera_ray_convention": "opencv_x_right_y_down_z_forward",
        "depth_scale_to_meters": 0.001,
        "pixel_stride": 4,
        "maximum_edge_length_m": 0.25,
        "maximum_depth_discontinuity_m": 0.1,
    }
    request = {
        "schema_version": "arkit_depth_surface_compilation_request.v1",
        "stable_run_identity": f"arkitscenes-surface-{capture_digest[7:31]}",
        "source_capture_identity": report["source_capture_identity"],
        "source_capture_digest": capture_digest,
        "original_file_references": sorted(
            original_references.values(), key=lambda row: row["artifact_id"]
        ),
        "source_commit_sha": report["source_commit_sha"],
        "deterministic_configuration_digest": canonical_digest(configuration),
        "train_heldout_split_digest": observations["split_digest"],
        "camera_calibration_binding": {
            "camera_observation_digest": observations["camera_observation_digest"],
            "metric_scaffold_digest": scaffold["metric_scaffold_digest"],
            "official_helper_commit": ARKITSCENES_OFFICIAL_HELPER_COMMIT,
        },
        "coordinate_frame_declaration": {
            "frame": "arkitscenes_official_loader_world",
            "units": "meters",
            "up_axis": "not_independently_validated",
            "handedness": "not_explicitly_declared_by_dataset",
            "gravity_aligned": False,
        },
        "authority_used": report["authority_used"],
        "timestamp": report["timestamp"],
        "capture_profile": "public_dataset_arkitscenes_proxy",
        "camera_ray_convention": "opencv_x_right_y_down_z_forward",
        "metric_scale_status": "sensor_metric_unvalidated",
        "pixel_stride": 4,
        "accepted_confidence_values": [2],
        "maximum_edge_length_m": 0.25,
        "maximum_depth_discontinuity_m": 0.1,
        "declared_region_ids": [
            "arkitscenes-observed-frusta",
            "arkitscenes-unobserved-regions",
        ],
        "unsupported_region_ids": ["arkitscenes-unobserved-regions"],
        "generated_fill_used": False,
        "candidate_may_read_hidden_heldout": False,
        "warnings": [
            "public_ipad_dataset_not_blueprint_raw_contract_3_2",
            "world_handedness_up_axis_and_gravity_not_independently_validated",
        ],
        "frames": sorted(frames, key=lambda row: row["frame_id"]),
    }
    request["arkit_depth_surface_compilation_request_digest"] = canonical_digest(
        request, digest_field="arkit_depth_surface_compilation_request_digest"
    )
    output = Path(output_root).expanduser().resolve()
    _write_immutable_json(output / "arkit_depth_surface_proxy_request.json", request)
    result = compile_arkit_depth_surface(
        source_artifact=request, artifact_root=scene, output_root=output
    )
    return _write_immutable_json(
        output / "arkit_depth_surface_proxy_result.json", result
    )


__all__ = [
    "ARKITSCENES_OBSERVATIONS_SCHEMA_VERSION",
    "ARKITSCENES_PROXY_SCHEMA_VERSION",
    "ARKITSCENES_SCAFFOLD_SCHEMA_VERSION",
    "ARKITSCENES_OFFICIAL_HELPER_COMMIT",
    "ArkitScenesProxyError",
    "compile_arkitscenes_raw_proxy",
    "compile_arkitscenes_depth_surface_proxy",
]
