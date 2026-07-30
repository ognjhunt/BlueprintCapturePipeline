"""Explicitly authorized local reconstruction adapters with strict claim ceilings.

The decoded-observation lane indexes frames that actually exist in a retained
video.  It does not infer calibration, scale, geometry, or physics.  The ARKit
metric-scaffold lane is deliberately narrower: it accepts only Capture Raw
Contract V3.2 LiDAR bundles whose decoded PTS, encoder-retention map, AR poses,
intrinsics, coordinate semantics, and depth/confidence pairs agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .arkit_reconstruction_dataset import (
    ArkitReconstructionDatasetError,
    compile_arkit_reconstruction_dataset,
)
from .decision_evidence_contracts import canonical_digest
from .reconstruction_frame_dataset import (
    ReconstructionFrameDatasetError,
    compile_frozen_frame_dataset,
)
from .reconstruction_capability import (
    ReconstructionContractError,
    build_reconstruction_method_profile,
    normalize_reconstruction_result,
)


LOCAL_DECODED_OBSERVATION_ADAPTER = "local://decoded-observation-index-v1"
LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER = "local://arkit-metric-scaffold-v1"
LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER = "local://external-reconstruction-import-v1"
_PTS_TOLERANCE_SECONDS = 0.0001
_MAX_RETAINED_VIDEO_BYTES = 64 * 1024 * 1024 * 1024
_MAX_PLY_HEADER_BYTES = 1024 * 1024
_VIDEO_PROFILES = {
    "iphone_arkit_lidar",
    "iphone_arkit_non_lidar",
    "camera_360_equirectangular",
    "camera_360_native",
    "monocular_video",
}


class LocalReconstructionAdapterError(ReconstructionContractError):
    """Stable fail-closed errors for local reconstruction execution."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = _text(value)
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _safe_child(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path.replace("\\", "/"))
    if (
        not relative_path
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise LocalReconstructionAdapterError(["artifact_relative_path:unsafe"])
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / Path(*relative.parts)).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise LocalReconstructionAdapterError(["artifact_relative_path:escapes_capture_root"])
    return candidate


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalReconstructionAdapterError([f"{label}:missing_or_invalid_json"]) from exc
    if not isinstance(value, Mapping):
        raise LocalReconstructionAdapterError([f"{label}:not_object"])
    return dict(value)


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LocalReconstructionAdapterError([f"{label}:missing_or_unreadable"]) from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LocalReconstructionAdapterError([f"{label}:invalid_jsonl:{index}"]) from exc
        if not isinstance(value, Mapping):
            raise LocalReconstructionAdapterError([f"{label}:row_not_object:{index}"])
        rows.append(dict(value))
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(value)
    digest = _sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise LocalReconstructionAdapterError(["derived_artifact:immutable_conflict"])
        return digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _implementation_digest() -> str:
    return canonical_digest(
        {
            "local_reconstruction_adapters": _sha256_file(Path(__file__).resolve()),
            "reconstruction_frame_dataset": _sha256_file(
                Path(__file__).with_name("reconstruction_frame_dataset.py").resolve()
            ),
            "arkit_reconstruction_dataset": _sha256_file(
                Path(__file__).with_name("arkit_reconstruction_dataset.py").resolve()
            ),
        }
    )


def _source_commit_sha() -> str:
    configured = _text(os.getenv("BLUEPRINT_SOURCE_COMMIT"))
    if configured:
        if len(configured) == 40 and all(character in "0123456789abcdef" for character in configured):
            return configured
        raise LocalReconstructionAdapterError(["local_source_commit_sha_invalid"])
    repository_root = Path(__file__).resolve().parents[2]
    git_entry = repository_root / ".git"
    try:
        if git_entry.is_file():
            line = git_entry.read_text(encoding="utf-8").strip()
            if not line.startswith("gitdir:"):
                raise LocalReconstructionAdapterError(["local_gitdir_pointer_invalid"])
            git_root = Path(line.split(":", 1)[1].strip())
            if not git_root.is_absolute():
                git_root = (repository_root / git_root).resolve()
        else:
            git_root = git_entry.resolve()
        common_git_root = git_root
        common_dir = git_root / "commondir"
        if common_dir.is_file():
            configured_common_dir = Path(common_dir.read_text(encoding="utf-8").strip())
            common_git_root = (
                configured_common_dir
                if configured_common_dir.is_absolute()
                else (git_root / configured_common_dir).resolve()
            )
        head = (git_root / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            reference = head.split(":", 1)[1].strip()
            value = ""
            for reference_root in dict.fromkeys((git_root, common_git_root)):
                reference_path = reference_root / reference
                if reference_path.is_file():
                    value = reference_path.read_text(encoding="utf-8").strip()
                    break
                packed = reference_root / "packed-refs"
                if packed.is_file():
                    for packed_line in packed.read_text(encoding="utf-8").splitlines():
                        if packed_line.endswith(f" {reference}"):
                            value = packed_line.split(" ", 1)[0]
                            break
                if value:
                    break
        else:
            value = head
    except OSError as exc:
        raise LocalReconstructionAdapterError(["local_source_commit_sha_unavailable"]) from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise LocalReconstructionAdapterError(["local_source_commit_sha_invalid"])
    return value


def _tool_identity() -> tuple[str, str, str, str]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise LocalReconstructionAdapterError(["local_media_tools:ffprobe_or_ffmpeg_missing"])
    versions: dict[str, str] = {}
    for name, command in (("ffprobe", ffprobe), ("ffmpeg", ffmpeg)):
        try:
            completed = subprocess.run(
                [command, "-version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LocalReconstructionAdapterError(
                [f"local_media_tools:{name}_version_failed"]
            ) from exc
        if completed.returncode != 0 or not completed.stdout.strip():
            raise LocalReconstructionAdapterError([f"local_media_tools:{name}_version_failed"])
        versions[name] = completed.stdout.splitlines()[0].strip()
    runtime = {"ffmpeg": versions["ffmpeg"], "ffprobe": versions["ffprobe"]}
    return ffprobe, ffmpeg, "ffmpeg_ffprobe_local", canonical_digest(runtime)


def _probe_video(video_path: Path, ffprobe: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=index,codec_name,width,height,avg_frame_rate,time_base,pix_fmt,color_range,color_space,color_transfer,color_primaries,field_order:stream_tags=rotate:stream_side_data=rotation:frame=best_effort_timestamp,best_effort_timestamp_time,pkt_dts_time,pkt_duration_time,key_frame,pict_type,color_range,color_space,color_transfer,color_primaries:frame_tags",
                "-show_streams",
                "-show_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LocalReconstructionAdapterError(["decoded_observation:ffprobe_failed"]) from exc
    if completed.returncode != 0:
        raise LocalReconstructionAdapterError(["decoded_observation:media_not_decodable"])
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LocalReconstructionAdapterError(
            ["decoded_observation:ffprobe_output_invalid"]
        ) from exc
    frames = payload.get("frames") if isinstance(payload, Mapping) else None
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    if not isinstance(frames, list) or not frames or not isinstance(streams, list) or not streams:
        raise LocalReconstructionAdapterError(["decoded_observation:no_decoded_video_frames"])
    presentation_times: list[float] = []
    decoded_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise LocalReconstructionAdapterError([f"decoded_observation:frame_invalid:{index}"])
        pts = _number(frame.get("best_effort_timestamp_time"))
        if pts is None:
            raise LocalReconstructionAdapterError([f"decoded_observation:pts_missing:{index}"])
        if presentation_times and pts <= presentation_times[-1]:
            blocker = (
                "decoded_observation:duplicate_pts"
                if pts == presentation_times[-1]
                else "decoded_observation:pts_not_monotonic"
            )
            raise LocalReconstructionAdapterError([blocker])
        presentation_times.append(pts)
        decoded_frames.append(
            {
                "decoded_frame_index": index,
                "source_pts_seconds": pts,
                "source_dts_seconds": _number(frame.get("pkt_dts_time")),
                "duration_seconds": _number(frame.get("pkt_duration_time")),
                "key_frame": bool(_integer(frame.get("key_frame"))),
                "picture_type": _text(frame.get("pict_type")) or None,
                "color_metadata": {
                    key: frame.get(key)
                    for key in ("color_range", "color_space", "color_transfer", "color_primaries")
                    if frame.get(key) not in (None, "")
                },
                "exposure_metadata": dict(frame.get("tags") or {}),
            }
        )
    first = presentation_times[0]
    normalized = [round(value - first, 9) for value in presentation_times]
    for index, row in enumerate(decoded_frames):
        row["t_video_sec"] = normalized[index]
    stream = dict(streams[0]) if isinstance(streams[0], Mapping) else {}
    rotation = None
    tags = stream.get("tags") if isinstance(stream.get("tags"), Mapping) else {}
    if tags:
        rotation = _number(tags.get("rotate"))
    side_data = stream.get("side_data_list")
    if rotation is None and isinstance(side_data, list):
        for value in side_data:
            if isinstance(value, Mapping) and _number(value.get("rotation")) is not None:
                rotation = _number(value.get("rotation"))
                break
    return {
        "stream": {
            key: stream.get(key)
            for key in (
                "index",
                "codec_name",
                "width",
                "height",
                "avg_frame_rate",
                "time_base",
                "pix_fmt",
                "color_range",
                "color_space",
                "color_transfer",
                "color_primaries",
                "field_order",
            )
        }
        | {"display_rotation_degrees": rotation},
        "frame_count": len(frames),
        "presentation_times_seconds": normalized,
        "first_source_pts_seconds": first,
        "frames": decoded_frames,
    }


def _sample_indexes(presentation_times: Sequence[float], maximum_frames: int) -> list[int]:
    if maximum_frames <= 0:
        raise LocalReconstructionAdapterError(["decoded_observation:maximum_frames_invalid"])
    frame_count = len(presentation_times)
    if frame_count <= 0:
        raise LocalReconstructionAdapterError(["decoded_observation:no_decoded_video_frames"])
    if frame_count <= maximum_frames:
        return list(range(frame_count))
    if maximum_frames == 1:
        return [0]
    start = float(presentation_times[0])
    stop = float(presentation_times[-1])
    if stop <= start:
        raise LocalReconstructionAdapterError(["decoded_observation:pts_not_monotonic"])
    selected = {0, frame_count - 1}
    for ordinal in range(1, maximum_frames - 1):
        target = start + ((stop - start) * ordinal / (maximum_frames - 1))
        index = min(
            range(frame_count),
            key=lambda candidate: (abs(float(presentation_times[candidate]) - target), candidate),
        )
        selected.add(index)
    # Extremely uneven timing can map multiple targets to one frame. Fill the
    # bounded remainder by maximizing temporal distance from selected frames.
    while len(selected) < maximum_frames:
        remaining = [index for index in range(frame_count) if index not in selected]
        chosen = max(
            remaining,
            key=lambda candidate: (
                min(
                    abs(
                        float(presentation_times[candidate])
                        - float(presentation_times[existing])
                    )
                    for existing in selected
                ),
                -candidate,
            ),
        )
        selected.add(chosen)
    return sorted(selected)


def _extract_frames(
    *,
    video_path: Path,
    ffmpeg: str,
    indexes: Sequence[int],
    presentation_times: Sequence[float],
    decoded_frame_metadata: Sequence[Mapping[str, Any]],
    frame_root: Path,
) -> list[dict[str, Any]]:
    frame_root.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for index in indexes:
        frame_id = f"decoded-{index:09d}"
        target = frame_root / f"{frame_id}.png"
        descriptor, temporary_name = tempfile.mkstemp(suffix=".png", dir=frame_root)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-noautorotate",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"select=eq(n\\,{index})",
                    "-frames:v",
                    "1",
                    "-vsync",
                    "0",
                    "-y",
                    str(temporary),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if (
                completed.returncode != 0
                or not temporary.is_file()
                or temporary.stat().st_size == 0
            ):
                raise LocalReconstructionAdapterError(
                    [f"decoded_observation:frame_extract_failed:{index}"]
                )
            digest = _sha256_file(temporary)
            if target.is_symlink():
                raise LocalReconstructionAdapterError(["derived_artifact:symlink_forbidden"])
            if target.exists():
                if _sha256_file(target) != digest:
                    raise LocalReconstructionAdapterError(["derived_artifact:immutable_conflict"])
            else:
                temporary.replace(target)
            with Image.open(target if target.exists() else temporary) as image:
                gray = np.asarray(image.convert("L"), dtype=np.float32)
                image_width, image_height = image.size
            horizontal = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
            vertical = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
            gradient_energy = float(np.mean(horizontal * horizontal) + np.mean(vertical * vertical))
            metadata = dict(decoded_frame_metadata[index])
            frames.append(
                {
                    "frame_id": frame_id,
                    "decoded_frame_index": index,
                    "t_video_sec": presentation_times[index],
                    "source_pts_seconds": metadata.get("source_pts_seconds"),
                    "source_dts_seconds": metadata.get("source_dts_seconds"),
                    "duration_seconds": metadata.get("duration_seconds"),
                    "key_frame": bool(metadata.get("key_frame")),
                    "uri": f"local-reconstruction-frame://{digest[7:]}",
                    "digest": digest,
                    "artifact_relative_path": f"frames/{frame_id}.png",
                    "image_metadata": {
                        "width": image_width,
                        "height": image_height,
                        "pixel_orientation": "encoded_source_no_autorotate",
                    },
                    "quality_signals": {
                        "mean_luma_0_255": round(float(np.mean(gray)), 6),
                        "gradient_energy": round(gradient_energy, 6),
                        "excessive_blur_deterministically_established": False,
                        "exposure_metadata": metadata.get("exposure_metadata", {}),
                    },
                }
            )
        finally:
            if temporary.exists():
                temporary.unlink()
    return frames


def decoded_observation_method_profile(*, execution_authorized: bool = False) -> dict[str, Any]:
    return build_reconstruction_method_profile(
        {
            "method_id": "local_decoded_observation_index",
            "version": "1",
            "implementation_digest": _implementation_digest(),
            "method_kind": "decoded_observation_index",
            "provider_identity": "local",
            "execution_mode": "hermetic_local",
            "adapter_reference": LOCAL_DECODED_OBSERVATION_ADAPTER,
            "outputs": ["decoded_observation_frames"],
            "required_capture_authority_profiles": sorted(_VIDEO_PROFILES),
            "required_claim_ceiling_flags": [],
            "qualified_claim_types": [
                "appearance_review",
                "perception_visibility",
                "task_discovery",
            ],
            "execution_authorized": execution_authorized,
            "qualification_status": "qualified",
            "expected_cost_usd": 0.0,
            "provider_constraints": {"external_processing": False},
            "rights_constraints": {"requires_local_processing_allowed": True},
            "failure_modes": [
                "media_not_decodable",
                "decoded_pts_unavailable",
                "frame_extract_failed",
            ],
        }
    )


def arkit_metric_scaffold_method_profile(*, execution_authorized: bool = False) -> dict[str, Any]:
    return build_reconstruction_method_profile(
        {
            "method_id": "local_arkit_metric_scaffold",
            "version": "1",
            "implementation_digest": _implementation_digest(),
            "method_kind": "lidar_depth_fusion",
            "provider_identity": "local",
            "execution_mode": "hermetic_local",
            "adapter_reference": LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER,
            "outputs": [
                "calibrated_frames",
                "decoded_observation_frames",
                "metric_reference_layer",
            ],
            "required_capture_authority_profiles": ["iphone_arkit_lidar"],
            "required_claim_ceiling_flags": [
                "calibrated_camera_poses",
                "decoded_video_pts",
                "metric_geometry",
            ],
            "qualified_claim_types": ["perception_visibility", "reachability", "robot_placement"],
            "execution_authorized": execution_authorized,
            "qualification_status": "qualified",
            "expected_cost_usd": 0.0,
            "provider_constraints": {"external_processing": False},
            "rights_constraints": {"requires_local_processing_allowed": True},
            "failure_modes": [
                "raw_contract_3_2_required",
                "decoded_pts_mismatch",
                "pose_or_intrinsics_mismatch",
                "depth_confidence_pair_missing",
            ],
        }
    )


def external_reconstruction_import_method_profile(
    *, execution_authorized: bool = False
) -> dict[str, Any]:
    return build_reconstruction_method_profile(
        {
            "method_id": "local_external_reconstruction_import",
            "version": "1",
            "implementation_digest": _implementation_digest(),
            "method_kind": "precomputed_external_reconstruction_import",
            "provider_identity": "local",
            "execution_mode": "hermetic_local",
            "adapter_reference": LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER,
            "outputs": ["appearance_layer"],
            "required_capture_authority_profiles": ["precomputed_external_reconstruction"],
            "required_claim_ceiling_flags": [],
            "qualified_claim_types": ["appearance_review"],
            "execution_authorized": execution_authorized,
            "qualification_status": "debug_only",
            "expected_cost_usd": 0.0,
            "provider_constraints": {"external_processing": False},
            "rights_constraints": {"requires_local_processing_allowed": True},
            "failure_modes": [
                "source_capture_binding_missing",
                "asset_digest_mismatch",
                "unsupported_external_reconstruction_format",
                "ply_header_invalid_or_too_large",
            ],
        }
    )


@dataclass(frozen=True)
class LocalDecodedObservationAdapter:
    adapter_reference: str = LOCAL_DECODED_OBSERVATION_ADAPTER

    def execute(
        self,
        *,
        intake_id: str,
        capture_digest: str,
        capture_authority_profile: str,
        capture_root: Path,
        video_relative_path: str,
        output_root: Path,
        rights_and_retention: Mapping[str, Any],
        maximum_frames: int = 12,
        maximum_source_bytes: int = _MAX_RETAINED_VIDEO_BYTES,
    ) -> dict[str, Any]:
        if not _text(intake_id) or not _is_digest(capture_digest):
            raise LocalReconstructionAdapterError(["decoded_observation:source_binding_invalid"])
        if capture_authority_profile not in _VIDEO_PROFILES:
            raise LocalReconstructionAdapterError(
                ["decoded_observation:capture_profile_not_supported"]
            )
        relative_video = PurePosixPath(video_relative_path.replace("\\", "/"))
        lexical_video = capture_root.expanduser().resolve() / Path(*relative_video.parts)
        if lexical_video.is_symlink():
            raise LocalReconstructionAdapterError(["decoded_observation:video_symlink_forbidden"])
        video_path = _safe_child(capture_root, video_relative_path)
        if video_path.is_symlink():
            raise LocalReconstructionAdapterError(["decoded_observation:video_symlink_forbidden"])
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            raise LocalReconstructionAdapterError(["decoded_observation:video_missing"])
        if maximum_source_bytes <= 0 or video_path.stat().st_size > maximum_source_bytes:
            raise LocalReconstructionAdapterError(["decoded_observation:video_oversized"])
        ffprobe, ffmpeg, runtime_identity, runtime_digest = _tool_identity()
        probe = _probe_video(video_path, ffprobe)
        indexes = _sample_indexes(probe["presentation_times_seconds"], maximum_frames)
        method_profile = decoded_observation_method_profile(execution_authorized=True)
        artifact_root = (
            output_root.expanduser().resolve()
            / capture_digest[7:]
            / "local_decoded_observation_index_v1"
            / method_profile["implementation_digest"][7:23]
            / f"maximum_frames_{maximum_frames}"
        )
        frames = _extract_frames(
            video_path=video_path,
            ffmpeg=ffmpeg,
            indexes=indexes,
            presentation_times=probe["presentation_times_seconds"],
            decoded_frame_metadata=probe["frames"],
            frame_root=artifact_root / "frames",
        )
        try:
            dataset = compile_frozen_frame_dataset(
                artifact_root=artifact_root,
                intake_id=intake_id,
                capture_digest=capture_digest,
                capture_authority_profile=capture_authority_profile,
                source_video_relative_path=video_relative_path,
                source_video_digest=_sha256_file(video_path),
                decoded_frame_count=probe["frame_count"],
                selected_frames=frames,
                stream_metadata=probe["stream"],
                runtime_identity=runtime_identity,
                runtime_digest=runtime_digest,
                implementation_digest=method_profile["implementation_digest"],
                source_commit_sha=_source_commit_sha(),
                rights_and_retention=rights_and_retention,
            )
        except ReconstructionFrameDatasetError as exc:
            raise LocalReconstructionAdapterError(
                [f"frame_dataset:{code}" for code in exc.codes]
            ) from exc
        split_reference = dataset["artifact_references"]["frozen_split_manifest"]
        split_manifest = _load_json(
            artifact_root / split_reference["relative_path"], label="frozen_split_manifest"
        )
        heldout_ids = {
            _text(row.get("frame_id"))
            for row in split_manifest.get("assignments", [])
            if isinstance(row, Mapping) and row.get("split") == "held_out"
        }
        candidate_visible_frames = [
            {key: value for key, value in frame.items() if key != "artifact_relative_path"}
            for frame in frames
            if frame["frame_id"] not in heldout_ids
        ]
        index = {
            "schema_version": "decoded_observation_index.v1",
            "capture_digest": capture_digest,
            "video_relative_path": video_relative_path,
            "video_digest": _sha256_file(video_path),
            "decoded_frame_count": probe["frame_count"],
            "decoded_presentation_times_seconds": probe["presentation_times_seconds"],
            "sampled_frames": candidate_visible_frames,
            "selected_frame_count": len(frames),
            "selection_method": "evenly_spaced_actual_decoded_pts_with_endpoints_v1",
            "stream_metadata": probe["stream"],
            "reconstruction_dataset_manifest_digest": dataset["dataset_manifest_digest"],
            "frozen_split_digest": dataset["train_heldout_split_digest"],
            "hidden_heldout_frame_count": len(heldout_ids),
        }
        index_path = artifact_root / "decoded_observation_index.json"
        index_digest = _write_immutable_json(index_path, index)
        dataset_path = (
            artifact_root
            / f"frozen_dataset_{dataset['deterministic_configuration_digest'][7:23]}"
            / "reconstruction_dataset_manifest.json"
        )
        candidate_manifest_path = (
            artifact_root
            / dataset["artifact_references"]["candidate_dataset_manifest"]["relative_path"]
        )
        result = {
            "result_id": f"decoded-observation-{index_digest[7:23]}",
            "intake_id": intake_id,
            "capture_digest": capture_digest,
            "method_id": method_profile["method_id"],
            "method_version": method_profile["version"],
            "method_profile_digest": method_profile["method_profile_digest"],
            "implementation_digest": method_profile["implementation_digest"],
            "provider_identity": "local",
            "runtime_identity": runtime_identity,
            "runtime_digest": runtime_digest,
            "outputs": ["decoded_observation_frames"],
            "source_frames": {
                "decoded_frame_count": probe["frame_count"],
                "sampled_frames": candidate_visible_frames,
                "hidden_heldout_frame_count": len(heldout_ids),
                "hidden_heldout_pixels_exposed_to_candidate": False,
            },
            "camera_solution": {"status": "not_available", "calibrated": False},
            "coordinate_system": {
                "scale_status": "not_authoritative",
                "camera_pose_status": "not_available",
            },
            "asset_references": {
                "decoded_observation_index": {
                    "uri": f"local-reconstruction://{index_digest[7:]}",
                    "digest": index_digest,
                    "relative_path": index_path.relative_to(
                        output_root.expanduser().resolve()
                    ).as_posix(),
                },
                "reconstruction_dataset_manifest": {
                    "uri": f"local-reconstruction://{dataset['dataset_manifest_digest'][7:]}",
                    "digest": dataset["dataset_manifest_digest"],
                    "relative_path": dataset_path.relative_to(
                        output_root.expanduser().resolve()
                    ).as_posix(),
                },
                "candidate_dataset_manifest": {
                    "uri": f"local-reconstruction://{dataset['output_digests']['candidate_dataset_digest'][7:]}",
                    "digest": dataset["output_digests"]["candidate_dataset_digest"],
                    "relative_path": candidate_manifest_path.relative_to(
                        output_root.expanduser().resolve()
                    ).as_posix(),
                },
            },
            "coverage_map": {
                "timeline_sample_fraction": round(len(frames) / probe["frame_count"], 9),
                "spatial_coverage_status": "not_established",
                "hidden_heldout_frame_count": len(heldout_ids),
            },
            "observed_regions": [{"region_id": "retained_video_timeline"}],
            "generated_regions": [],
            "uncertainty_map": {"spatial_uncertainty_status": "not_estimated"},
            "invalid_regions": [],
            "validation_metrics": {
                "decoded_pts_monotonic": True,
                "decoded_frame_count": probe["frame_count"],
                "source_video_digest": index["video_digest"],
                "frozen_split_digest": dataset["train_heldout_split_digest"],
                "candidate_can_change_split": False,
                "candidate_can_read_hidden_heldout_pixels": False,
                "dataset_blockers": dataset["blockers"],
            },
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "provider_receipt": None,
            "rights_and_retention": dict(rights_and_retention),
            "deletion_evidence": None,
            "claim_ceiling": {
                "captured_observation": True,
                "task_discovery": True,
                "calibrated_camera_poses": False,
                "metric_geometry": False,
                "collision_geometry": False,
                "physics": False,
                "physical_task_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }
        return normalize_reconstruction_result(result)


def _matrix4(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        values = [_number(item) for item in row]
        if any(item is None for item in values):
            return None
        matrix.append([float(item) for item in values if item is not None])
    return matrix


def _intrinsics(value: Mapping[str, Any]) -> dict[str, Any] | None:
    nested = value.get("intrinsics")
    source = nested if isinstance(nested, Mapping) else value
    numbers = {key: _number(source.get(key)) for key in ("fx", "fy", "cx", "cy")}
    width = _integer(source.get("width"))
    height = _integer(source.get("height"))
    if (
        any(number is None for number in numbers.values())
        or numbers["fx"] <= 0  # type: ignore[operator]
        or numbers["fy"] <= 0  # type: ignore[operator]
        or width is None
        or height is None
        or width <= 0
        or height <= 0
    ):
        return None
    return {**numbers, "width": width, "height": height}


def _verified_depth_pairs(capture_root: Path) -> list[dict[str, Any]]:
    depth = _load_json(capture_root / "arkit/depth_manifest.json", label="depth_manifest")
    confidence = _load_json(
        capture_root / "arkit/confidence_manifest.json", label="confidence_manifest"
    )
    depth_rows = {
        _text(row.get("frame_id")): dict(row)
        for row in depth.get("frames", [])
        if isinstance(row, Mapping) and _text(row.get("frame_id"))
    }
    confidence_rows = {
        _text(row.get("frame_id")): dict(row)
        for row in confidence.get("frames", [])
        if isinstance(row, Mapping) and _text(row.get("frame_id"))
    }
    if not depth_rows or set(depth_rows) != set(confidence_rows):
        raise LocalReconstructionAdapterError(["metric_scaffold:depth_confidence_pairing_invalid"])
    pairs: list[dict[str, Any]] = []
    for frame_id in sorted(depth_rows):
        depth_path_text = _text(depth_rows[frame_id].get("depth_path"))
        confidence_path_text = _text(confidence_rows[frame_id].get("confidence_path"))
        if (
            _text(depth_rows[frame_id].get("paired_confidence_path")) != confidence_path_text
            or _text(confidence_rows[frame_id].get("paired_depth_path")) != depth_path_text
        ):
            raise LocalReconstructionAdapterError(
                ["metric_scaffold:depth_confidence_pairing_invalid"]
            )
        depth_path = _safe_child(capture_root, depth_path_text)
        confidence_path = _safe_child(capture_root, confidence_path_text)
        if not depth_path.is_file() or not confidence_path.is_file():
            raise LocalReconstructionAdapterError(
                ["metric_scaffold:depth_confidence_artifact_missing"]
            )
        pairs.append(
            {
                "frame_id": frame_id,
                "depth_relative_path": depth_path_text,
                "depth_digest": _sha256_file(depth_path),
                "confidence_relative_path": confidence_path_text,
                "confidence_digest": _sha256_file(confidence_path),
            }
        )
    return pairs


@dataclass(frozen=True)
class LocalArkitMetricScaffoldAdapter:
    adapter_reference: str = LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER

    def execute(
        self,
        *,
        intake_id: str,
        capture_digest: str,
        capture_root: Path,
        output_root: Path,
        rights_and_retention: Mapping[str, Any],
        maximum_frames: int = 12,
    ) -> dict[str, Any]:
        root = capture_root.expanduser().resolve()
        if not _text(intake_id) or not _is_digest(capture_digest):
            raise LocalReconstructionAdapterError(["metric_scaffold:source_binding_invalid"])
        manifest = _load_json(root / "manifest.json", label="manifest")
        if manifest.get("capture_schema_version") != "3.2.0":
            raise LocalReconstructionAdapterError(["metric_scaffold:raw_contract_3_2_required"])
        if manifest.get("capture_profile_id") != "iphone_arkit_lidar":
            raise LocalReconstructionAdapterError(["metric_scaffold:iphone_arkit_lidar_required"])
        capabilities = manifest.get("capture_capabilities")
        if not isinstance(capabilities, Mapping) or any(
            capabilities.get(key) is not True
            for key in (
                "camera_pose",
                "camera_intrinsics",
                "depth",
                "depth_confidence",
                "tracking_state",
            )
        ):
            raise LocalReconstructionAdapterError(
                ["metric_scaffold:required_capture_capabilities_missing"]
            )
        video_track = _load_json(root / "video_track.json", label="video_track")
        video_relative_path = _text(video_track.get("video_file")) or "walkthrough.mov"
        video_path = _safe_child(root, video_relative_path)
        ffprobe, _, runtime_identity, runtime_digest = _tool_identity()
        probe = _probe_video(video_path, ffprobe)
        sync_rows = _load_jsonl(root / "sync_map.jsonl", label="sync_map")
        retention_rows = _load_jsonl(
            root / "video_frame_retention.jsonl", label="video_frame_retention"
        )
        poses = _load_jsonl(root / "arkit/poses.jsonl", label="arkit_poses")
        ar_frames = _load_jsonl(root / "arkit/frames.jsonl", label="arkit_frames")
        decoded_count = probe["frame_count"]
        retained = [row for row in retention_rows if row.get("retention_status") == "retained"]
        dropped = [row for row in retention_rows if row.get("retention_status") != "retained"]
        errors: list[str] = []
        if (
            video_track.get("decoded_pts_verified") is not True
            or video_track.get("frame_count_source") != "decoded_sample_presentation_timestamps"
        ):
            errors.append("metric_scaffold:decoded_pts_not_verified")
        declared = {
            "frame_count": decoded_count,
            "write_attempt_count": len(retention_rows),
            "retained_frame_count": len(retained),
            "dropped_frame_count": len(dropped),
        }
        for key, expected in declared.items():
            if _integer(video_track.get(key)) != expected:
                errors.append(f"metric_scaffold:video_track_count_mismatch:{key}")
        if len(sync_rows) != decoded_count or len(retained) != decoded_count:
            errors.append("metric_scaffold:decoded_retained_sync_count_mismatch")
        pose_by_id = {_text(row.get("frame_id")): row for row in poses}
        frame_ids = {_text(row.get("frame_id")) for row in ar_frames}
        coordinate_frame_session_id = _text(manifest.get("coordinate_frame_session_id"))
        tracking_state_rows = _integer(capabilities.get("tracking_state_rows"))
        if tracking_state_rows is None or tracking_state_rows < len(sync_rows):
            errors.append("metric_scaffold:tracking_state_rows_missing")
        for index, row in enumerate(sync_rows):
            frame_id = _text(row.get("frame_id"))
            if (
                row.get("sync_status") != "encoded_decoded_pts_match"
                or _integer(row.get("encoded_frame_index")) != index
                or _number(row.get("t_video_sec")) is None
                or abs(float(row["t_video_sec"]) - probe["presentation_times_seconds"][index])
                > _PTS_TOLERANCE_SECONDS
            ):
                errors.append(f"metric_scaffold:decoded_pts_mismatch:{index}")
            if index >= len(retained) or _text(retained[index].get("frame_id")) != frame_id:
                errors.append(f"metric_scaffold:retention_binding_mismatch:{index}")
            elif _integer(row.get("write_attempt_index")) != _integer(
                retained[index].get("write_attempt_index")
            ):
                errors.append(f"metric_scaffold:retention_attempt_binding_mismatch:{index}")
            if _text(row.get("pose_frame_id")) != frame_id:
                errors.append(f"metric_scaffold:pose_frame_binding_mismatch:{index}")
            pose = pose_by_id.get(frame_id)
            if (
                pose is None
                or _matrix4(pose.get("T_world_camera")) is None
                or _text(pose.get("coordinate_frame_session_id")) != coordinate_frame_session_id
                or frame_id not in frame_ids
            ):
                errors.append(f"metric_scaffold:pose_or_frame_binding_invalid:{frame_id or index}")
        for index, row in enumerate(retention_rows):
            if _integer(row.get("write_attempt_index")) != index:
                errors.append("metric_scaffold:retention_attempt_order_invalid")
                break
            if row.get("retention_status") == "retained":
                if (
                    _integer(row.get("encoded_frame_index")) is None
                    or _number(row.get("t_video_sec")) is None
                ):
                    errors.append(f"metric_scaffold:retained_binding_missing:{index}")
            elif (
                row.get("retention_status") != "dropped_backpressure"
                or not _text(row.get("drop_reason"))
                or row.get("encoded_frame_index") is not None
                or row.get("t_video_sec") is not None
            ):
                errors.append(f"metric_scaffold:dropped_attempt_invalid:{index}")
        recording = _load_json(root / "recording_session.json", label="recording_session")
        if (
            _text(recording.get("coordinate_frame_session_id")) != coordinate_frame_session_id
            or recording.get("units") != "meters"
            or recording.get("handedness") != "right_handed"
            or recording.get("gravity_aligned") is not True
            or _integer(recording.get("session_reset_count")) != 0
        ):
            errors.append("metric_scaffold:coordinate_frame_semantics_not_supported")
        intrinsics_document = _load_json(
            root / "arkit/session_intrinsics.json", label="arkit_session_intrinsics"
        )
        normalized_intrinsics = _intrinsics(intrinsics_document)
        if (
            normalized_intrinsics is None
            or _text(intrinsics_document.get("coordinate_frame_session_id"))
            != coordinate_frame_session_id
        ):
            errors.append("metric_scaffold:intrinsics_invalid")
        if sync_rows:
            first_video_time = _number(sync_rows[0].get("t_video_sec"))
            first_capture_time = _number(sync_rows[0].get("t_capture_sec"))
            if (
                first_video_time is None
                or first_capture_time is None
                or abs(first_video_time) > _PTS_TOLERANCE_SECONDS
                or abs(first_capture_time) > _PTS_TOLERANCE_SECONDS
            ):
                errors.append("metric_scaffold:first_retained_frame_not_capture_origin")
        if errors:
            raise LocalReconstructionAdapterError(errors)
        depth_pairs = _verified_depth_pairs(root)
        if any(pair["frame_id"] not in pose_by_id for pair in depth_pairs):
            raise LocalReconstructionAdapterError(
                ["metric_scaffold:depth_pair_pose_binding_missing"]
            )
        decoded_result = LocalDecodedObservationAdapter().execute(
            intake_id=intake_id,
            capture_digest=capture_digest,
            capture_authority_profile="iphone_arkit_lidar",
            capture_root=root,
            video_relative_path=video_relative_path,
            output_root=output_root,
            rights_and_retention=rights_and_retention,
            maximum_frames=maximum_frames,
        )
        method_profile = arkit_metric_scaffold_method_profile(execution_authorized=True)
        scaffold = {
            "schema_version": "arkit_metric_scaffold.v1",
            "capture_digest": capture_digest,
            "capture_schema_version": "3.2.0",
            "coordinate_frame_session_id": coordinate_frame_session_id,
            "coordinate_system": {
                "world_frame_definition": recording.get("world_frame_definition"),
                "units": "meters",
                "handedness": "right_handed",
                "gravity_aligned": True,
            },
            "intrinsics": normalized_intrinsics,
            "camera_frames": [
                {
                    "frame_id": _text(row.get("frame_id")),
                    "encoded_frame_index": _integer(row.get("encoded_frame_index")),
                    "t_video_sec": _number(row.get("t_video_sec")),
                    "t_capture_sec": _number(row.get("t_capture_sec")),
                    "T_world_camera": pose_by_id[_text(row.get("frame_id"))]["T_world_camera"],
                }
                for row in sync_rows
            ],
            "depth_confidence_pairs": depth_pairs,
            "source_artifact_digests": {
                relative: _sha256_file(_safe_child(root, relative))
                for relative in (
                    video_relative_path,
                    "manifest.json",
                    "video_track.json",
                    "video_frame_retention.jsonl",
                    "sync_map.jsonl",
                    "arkit/poses.jsonl",
                    "arkit/frames.jsonl",
                    "arkit/session_intrinsics.json",
                    "arkit/depth_manifest.json",
                    "arkit/confidence_manifest.json",
                    "recording_session.json",
                )
            },
        }
        artifact_root = (
            output_root.expanduser().resolve()
            / capture_digest[7:]
            / "local_arkit_metric_scaffold_v1"
        )
        scaffold_digest = _write_immutable_json(
            artifact_root / "arkit_metric_scaffold.json", scaffold
        )
        decoded_dataset_reference = decoded_result["asset_references"][
            "reconstruction_dataset_manifest"
        ]
        decoded_dataset_path = _safe_child(
            output_root.expanduser().resolve(),
            _text(decoded_dataset_reference.get("relative_path")),
        )
        dataset_manifest = _load_json(decoded_dataset_path, label="reconstruction_dataset")
        decoded_artifact_root = decoded_dataset_path.parents[1]
        split_manifest = _load_json(
            _safe_child(
                decoded_artifact_root,
                dataset_manifest["artifact_references"]["frozen_split_manifest"][
                    "relative_path"
                ],
            ),
            label="frozen_reconstruction_split",
        )
        candidate_manifest = _load_json(
            _safe_child(
                decoded_artifact_root,
                dataset_manifest["artifact_references"]["candidate_dataset_manifest"][
                    "relative_path"
                ],
            ),
            label="candidate_reconstruction_dataset",
        )
        try:
            arkit_export = compile_arkit_reconstruction_dataset(
                output_root=artifact_root / "reconstruction_dataset_export",
                intake_id=intake_id,
                capture_digest=capture_digest,
                dataset_manifest=dataset_manifest,
                split_manifest=split_manifest,
                candidate_manifest=candidate_manifest,
                metric_scaffold=scaffold,
                metric_scaffold_digest=scaffold_digest,
                implementation_digest=method_profile["implementation_digest"],
                source_commit_sha=_source_commit_sha(),
                authority_used=rights_and_retention,
            )
        except ArkitReconstructionDatasetError as exc:
            raise LocalReconstructionAdapterError(
                [f"arkit_reconstruction_export:{code}" for code in exc.codes]
            ) from exc
        coverage = round(len(depth_pairs) / len(sync_rows), 9)
        result_binding_digest = canonical_digest(
            {
                "metric_scaffold_digest": scaffold_digest,
                "decoded_observation_digest": decoded_result["asset_references"][
                    "decoded_observation_index"
                ]["digest"],
            }
        )
        return normalize_reconstruction_result(
            {
                "result_id": f"arkit-metric-scaffold-{result_binding_digest[7:23]}",
                "intake_id": intake_id,
                "capture_digest": capture_digest,
                "method_id": method_profile["method_id"],
                "method_version": method_profile["version"],
                "method_profile_digest": method_profile["method_profile_digest"],
                "implementation_digest": method_profile["implementation_digest"],
                "provider_identity": "local",
                "runtime_identity": runtime_identity,
                "runtime_digest": runtime_digest,
                "outputs": [
                    "calibrated_frames",
                    "decoded_observation_frames",
                    "metric_reference_layer",
                ],
                "source_frames": decoded_result["source_frames"],
                "camera_solution": {
                    "status": "raw_contract_3_2_verified",
                    "calibrated": True,
                    "pose_count": len(sync_rows),
                },
                "coordinate_system": scaffold["coordinate_system"],
                "asset_references": {
                    "decoded_observation_index": decoded_result["asset_references"][
                        "decoded_observation_index"
                    ],
                    "metric_scaffold": {
                        "uri": f"local-reconstruction://{scaffold_digest[7:]}",
                        "digest": scaffold_digest,
                    },
                    "arkit_reconstruction_dataset_export": {
                        "uri": (
                            "local-reconstruction://"
                            f"{arkit_export['arkit_reconstruction_dataset_export_digest'][7:]}"
                        ),
                        "digest": arkit_export[
                            "arkit_reconstruction_dataset_export_digest"
                        ],
                    },
                },
                "coverage_map": {
                    "calibrated_frame_fraction": 1.0,
                    "depth_confidence_frame_fraction": coverage,
                },
                "observed_regions": [{"region_id": "arkit_observed_frusta"}],
                "generated_regions": [],
                "uncertainty_map": {
                    "unobserved_surfaces_remain_unsupported": True,
                    "depth_confidence_is_raw_sensor_evidence": True,
                    "confidence_filtering_status": "not_executed",
                    "rgb_depth_alignment_status": "not_independently_validated",
                    "metric_scale_validation_status": "not_executed",
                },
                "invalid_regions": [],
                "validation_metrics": {
                    "decoded_pts_verified": True,
                    "retained_sync_pose_count": len(sync_rows),
                    "depth_confidence_pair_count": len(depth_pairs),
                    "tracking_reset_count": 0,
                    "sensor_declared_units": "meters",
                    "independent_metric_scale_validation_passed": False,
                    "arkit_reconstruction_dataset_export_digest": arkit_export[
                        "arkit_reconstruction_dataset_export_digest"
                    ],
                    "pose_refinement_executed": False,
                },
                "cost_usd": 0.0,
                "duration_seconds": 0.0,
                "provider_receipt": None,
                "rights_and_retention": dict(rights_and_retention),
                "deletion_evidence": None,
                "claim_ceiling": {
                    "captured_observation": True,
                    "calibrated_camera_poses": True,
                    "sensor_declared_metric_scale": True,
                    "metric_scale": False,
                    "metric_reference_layer": False,
                    "complete_geometry": False,
                    "collision_geometry": False,
                    "physics": False,
                    "physical_task_success": False,
                    "deployment_readiness": False,
                    "safety_certification": False,
                },
            }
        )


def _parse_ply_header(asset_path: Path) -> dict[str, Any]:
    try:
        with asset_path.open("rb") as stream:
            prefix = stream.read(_MAX_PLY_HEADER_BYTES + 1)
    except OSError as exc:
        raise LocalReconstructionAdapterError(["external_reconstruction:asset_unreadable"]) from exc
    marker = b"end_header\n"
    marker_offset = prefix.find(marker)
    if marker_offset < 0:
        marker = b"end_header\r\n"
        marker_offset = prefix.find(marker)
    if marker_offset < 0:
        code = (
            "external_reconstruction:ply_header_too_large"
            if len(prefix) > _MAX_PLY_HEADER_BYTES
            else "external_reconstruction:ply_header_invalid"
        )
        raise LocalReconstructionAdapterError([code])
    header_bytes = prefix[: marker_offset + len(marker)]
    try:
        lines = header_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise LocalReconstructionAdapterError(
            ["external_reconstruction:ply_header_not_ascii"]
        ) from exc
    if not lines or lines[0] != "ply":
        raise LocalReconstructionAdapterError(["external_reconstruction:ply_magic_invalid"])
    format_rows = [line.split() for line in lines if line.startswith("format ")]
    if len(format_rows) != 1 or len(format_rows[0]) != 3:
        raise LocalReconstructionAdapterError(["external_reconstruction:ply_format_invalid"])
    ply_format, version = format_rows[0][1:]
    if ply_format not in {"ascii", "binary_little_endian", "binary_big_endian"} or version != "1.0":
        raise LocalReconstructionAdapterError(["external_reconstruction:ply_format_unsupported"])
    elements: dict[str, int] = {}
    element_properties: dict[str, list[str]] = {}
    current_element = ""
    property_count = 0
    for line in lines:
        if line.startswith("element "):
            parts = line.split()
            if len(parts) != 3 or parts[1] in elements:
                raise LocalReconstructionAdapterError(
                    ["external_reconstruction:ply_element_invalid"]
                )
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise LocalReconstructionAdapterError(
                    ["external_reconstruction:ply_element_count_invalid"]
                ) from exc
            if count < 0 or count > 1_000_000_000:
                raise LocalReconstructionAdapterError(
                    ["external_reconstruction:ply_element_count_invalid"]
                )
            elements[parts[1]] = count
            current_element = parts[1]
            element_properties[current_element] = []
        elif line.startswith("property "):
            parts = line.split()
            if not current_element or len(parts) not in {3, 5}:
                raise LocalReconstructionAdapterError(
                    ["external_reconstruction:ply_property_invalid"]
                )
            element_properties[current_element].append(parts[-1])
            property_count += 1
    if elements.get("vertex", 0) <= 0 or property_count <= 0:
        raise LocalReconstructionAdapterError(
            ["external_reconstruction:ply_vertex_metadata_missing"]
        )
    vertex_properties = set(element_properties.get("vertex", []))
    chunk_properties = set(element_properties.get("chunk", []))
    position_properties = {"x", "y", "z"}
    color_properties = {"red", "green", "blue"}
    standard_3dgs_properties = {
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    }
    compressed_vertex_properties = {
        "packed_position",
        "packed_rotation",
        "packed_scale",
        "packed_color",
    }
    compressed_chunk_properties = {
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z",
    }
    if position_properties.issubset(vertex_properties) and color_properties.issubset(
        vertex_properties
    ):
        representation_profile = "colored_point_cloud"
    elif position_properties.issubset(vertex_properties) and standard_3dgs_properties.issubset(
        vertex_properties
    ):
        representation_profile = "standard_3dgs"
    elif (
        elements.get("chunk", 0) > 0
        and compressed_vertex_properties.issubset(vertex_properties)
        and compressed_chunk_properties.issubset(chunk_properties)
    ):
        representation_profile = "supersplat_compressed_3dgs"
    elif not position_properties.issubset(vertex_properties):
        raise LocalReconstructionAdapterError(
            ["external_reconstruction:ply_vertex_position_missing"]
        )
    else:
        raise LocalReconstructionAdapterError(
            ["external_reconstruction:ply_vertex_color_missing"]
        )
    if asset_path.stat().st_size <= len(header_bytes):
        raise LocalReconstructionAdapterError(
            ["external_reconstruction:ply_vertex_payload_missing"]
        )
    return {
        "format": ply_format,
        "version": version,
        "header_size_bytes": len(header_bytes),
        "header_digest": _sha256_bytes(header_bytes),
        "representation_profile": representation_profile,
        "elements": dict(sorted(elements.items())),
        "element_properties": {
            key: sorted(value) for key, value in sorted(element_properties.items())
        },
        "property_count": property_count,
    }


@dataclass(frozen=True)
class LocalExternalReconstructionImportAdapter:
    adapter_reference: str = LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER

    def execute(
        self,
        *,
        intake_id: str,
        capture_digest: str,
        source_capture_binding: Mapping[str, Any],
        capture_root: Path,
        asset_relative_path: str,
        original_filename: str,
        output_root: Path,
        rights_and_retention: Mapping[str, Any],
        coordinate_frame_declaration: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_capture_digest = _text(source_capture_binding.get("source_capture_digest"))
        if (
            not _text(intake_id)
            or not _is_digest(capture_digest)
            or not _is_digest(source_capture_digest)
        ):
            raise LocalReconstructionAdapterError(
                ["external_reconstruction:source_binding_invalid"]
            )
        if not original_filename or Path(original_filename).name != original_filename:
            raise LocalReconstructionAdapterError(
                ["external_reconstruction:original_filename_invalid"]
            )
        if Path(original_filename).suffix.lower() != ".ply":
            raise LocalReconstructionAdapterError(["external_reconstruction:format_not_supported"])
        asset_path = _safe_child(capture_root, asset_relative_path)
        if not asset_path.is_file() or asset_path.stat().st_size <= 0:
            raise LocalReconstructionAdapterError(["external_reconstruction:asset_missing"])
        if _sha256_file(asset_path) != capture_digest:
            raise LocalReconstructionAdapterError(["external_reconstruction:asset_digest_mismatch"])
        ply = _parse_ply_header(asset_path)
        method_profile = external_reconstruction_import_method_profile(execution_authorized=True)
        runtime_identity = "blueprint-ply-header-importer-v1"
        runtime_digest = _sha256_bytes(runtime_identity.encode("utf-8"))
        artifact_root = (
            output_root.expanduser().resolve()
            / capture_digest[7:]
            / "local_external_reconstruction_import_v1"
        )
        manifest = {
            "schema_version": "external_reconstruction_import_manifest.v1",
            "intake_id": intake_id,
            "imported_asset": {
                "digest": capture_digest,
                "size_bytes": asset_path.stat().st_size,
                "original_filename": original_filename,
                "format": "ply",
                "ply_header": ply,
            },
            "source_capture_binding": dict(source_capture_binding),
            "coordinate_frame_declaration": dict(coordinate_frame_declaration),
            "authority": {
                "raw_capture_authority": False,
                "metric_authority": False,
                "collision_or_physics_authority": False,
                "observed_vs_generated_regions_verified": False,
            },
        }
        manifest_digest = _write_immutable_json(
            artifact_root / "external_reconstruction_import_manifest.json", manifest
        )
        result = {
            "result_id": f"external-reconstruction-{manifest_digest[7:23]}",
            "intake_id": intake_id,
            "capture_digest": capture_digest,
            "source_capture_binding": dict(source_capture_binding),
            "method_id": method_profile["method_id"],
            "method_version": method_profile["version"],
            "method_profile_digest": method_profile["method_profile_digest"],
            "implementation_digest": method_profile["implementation_digest"],
            "provider_identity": "local",
            "runtime_identity": runtime_identity,
            "runtime_digest": runtime_digest,
            "outputs": ["appearance_layer"],
            "source_frames": {
                "status": "not_included_in_import",
                "source_capture_digest": source_capture_digest,
            },
            "camera_solution": {
                "status": "not_verified_from_import",
                "calibrated": False,
            },
            "coordinate_system": {
                "declaration": dict(coordinate_frame_declaration),
                "scale_status": "not_authoritative",
                "transform_status": "not_verified",
            },
            "asset_references": {
                "imported_external_reconstruction": {
                    "uri": f"content-addressed-capture://sha256/{capture_digest[7:]}",
                    "digest": capture_digest,
                },
                "normalization_manifest": {
                    "uri": f"local-reconstruction://{manifest_digest[7:]}",
                    "digest": manifest_digest,
                },
            },
            "coverage_map": {
                "status": "not_verified_from_import",
                "source_views_available": False,
            },
            "observed_regions": [],
            "generated_regions": [],
            "uncertainty_map": {
                "hidden_surfaces": "unknown",
                "generated_region_status": "unknown",
                "metric_scale_status": "not_authoritative",
            },
            "invalid_regions": [
                {
                    "region_id": "unverified_import_extent",
                    "reason": "coverage_and_observed_vs_generated_regions_not_verified",
                }
            ],
            "validation_metrics": {
                "asset_digest_verified": True,
                "source_capture_digest_bound": True,
                "format": "ply",
                "ply_header": ply,
            },
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "provider_receipt": None,
            "rights_and_retention": dict(rights_and_retention),
            "deletion_evidence": None,
            "claim_ceiling": {
                "appearance_review": True,
                "external_reconstruction_imported": True,
                "raw_capture_authority": False,
                "captured_observation": False,
                "task_discovery": False,
                "calibrated_camera_poses": False,
                "metric_geometry": False,
                "metric_scale": False,
                "collision_geometry": False,
                "physics": False,
                "physical_task_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
                "comparative_policy_ranking_verdict": "thesis_not_supported",
            },
        }
        return normalize_reconstruction_result(result)


def authorized_local_reconstruction_adapter_registry(
    authorized_references: Sequence[str],
) -> dict[
    str,
    LocalDecodedObservationAdapter
    | LocalArkitMetricScaffoldAdapter
    | LocalExternalReconstructionImportAdapter,
]:
    available: dict[
        str,
        LocalDecodedObservationAdapter
        | LocalArkitMetricScaffoldAdapter
        | LocalExternalReconstructionImportAdapter,
    ] = {
        LOCAL_DECODED_OBSERVATION_ADAPTER: LocalDecodedObservationAdapter(),
        LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER: LocalArkitMetricScaffoldAdapter(),
        LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER: (LocalExternalReconstructionImportAdapter()),
    }
    requested = sorted({_text(item) for item in authorized_references if _text(item)})
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise LocalReconstructionAdapterError(
            [f"local_reconstruction_adapter_not_registered:{','.join(unknown)}"]
        )
    return {reference: available[reference] for reference in requested}


__all__ = [
    "LOCAL_ARKIT_METRIC_SCAFFOLD_ADAPTER",
    "LOCAL_DECODED_OBSERVATION_ADAPTER",
    "LOCAL_EXTERNAL_RECONSTRUCTION_IMPORT_ADAPTER",
    "LocalArkitMetricScaffoldAdapter",
    "LocalDecodedObservationAdapter",
    "LocalExternalReconstructionImportAdapter",
    "LocalReconstructionAdapterError",
    "arkit_metric_scaffold_method_profile",
    "authorized_local_reconstruction_adapter_registry",
    "decoded_observation_method_profile",
    "external_reconstruction_import_method_profile",
]
