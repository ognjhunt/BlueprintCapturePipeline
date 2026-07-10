"""Geometry lane execution and contract writing."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .capture_bridge import CaptureDescriptor
from .camera_geometry_validation import validate_camera_intrinsics, validate_se3_matrix
from .common import PipelineError, ensure_dir, utc_now_iso, write_json
from .geometry_da3 import run_da3_provider
from .launch_proof_policy import synthetic_geometry_allowed
from .local_capture import LocalCaptureContext, resolve_local_capture_context
from .video_to_world_client import run_video_to_world_provider


_VIDEO_CANDIDATES = (
    "walkthrough.mov",
    "walkthrough.mp4",
    "recording.mov",
    "recording.mp4",
)

_CURRENT_GEOMETRY_DIRS = ("camera", "frames", "depth", "confidence", "alignment", "masks")
_CURRENT_GEOMETRY_FILES = (
    "geometry_manifest.json",
    "geometry_summary.json",
    "geometry_run_status.json",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _frame_sharpness_score(image_path: str) -> Optional[float]:
    """Return an image-derived sharpness score, or None when no frame is readable."""

    if not image_path:
        return None
    path = Path(image_path).expanduser()
    if not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            from PIL import Image

            array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except Exception:
        return None
    if array.ndim == 3:
        gray = array.astype(np.float32).mean(axis=2)
    else:
        gray = array.astype(np.float32)
    if gray.size < 9:
        return 0.0
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    if dx.size == 0 or dy.size == 0:
        return 0.0
    score = float(np.var(dx) + np.var(dy))
    if not math.isfinite(score):
        return None
    return round(max(0.0, score), 6)


def _contained_artifact_path(path_text: Any, *, geometry_root: Path) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = geometry_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        root = geometry_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if resolved == root or root not in resolved.parents:
        return None
    if candidate.is_symlink() or not resolved.is_file():
        return None
    return resolved


def _decode_tensor(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if len(archive.files) != 1:
                    return None
                array = archive[archive.files[0]]
        else:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                array = np.asarray(image)
    except Exception:
        return None
    array = np.asarray(array)
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return None
    try:
        if not np.isfinite(array).all():
            return None
    except TypeError:
        return None
    return array


def _validate_geometry_frame_records(
    *,
    frame_records: List[Mapping[str, Any]],
    intrinsics_payload: Mapping[str, Any],
    geometry_root: Path,
    provider_result_fallback: bool,
) -> Dict[str, Any]:
    """Verify aligned RGB/pose/depth/confidence records before counting them."""

    intrinsics_result = validate_camera_intrinsics(intrinsics_payload)
    intrinsics = intrinsics_result.get("normalized") or {}
    width = int(intrinsics.get("width") or 0)
    height = int(intrinsics.get("height") or 0)
    ids = [str(row.get("frame_id") or _frame_id(_safe_int(row.get("frame_index")))) for row in frame_records]
    indexes = [_safe_int(row.get("frame_index"), -1) for row in frame_records]
    timestamps = [_safe_float(row.get("timestamp_seconds"), float("nan")) for row in frame_records]
    id_counts = {value: ids.count(value) for value in set(ids)}
    index_counts = {value: indexes.count(value) for value in set(indexes)}
    timestamp_counts = {value: timestamps.count(value) for value in set(timestamps) if math.isfinite(value)}
    verified: List[Mapping[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    prior_timestamp: float | None = None
    for row_index, frame in enumerate(frame_records):
        blockers: List[str] = []
        frame_id = ids[row_index]
        frame_index = indexes[row_index]
        timestamp = timestamps[row_index]
        if not frame_id or id_counts.get(frame_id, 0) != 1:
            blockers.append("frame_id_missing_or_not_one_to_one")
        if frame_index < 0 or index_counts.get(frame_index, 0) != 1:
            blockers.append("frame_index_missing_or_not_one_to_one")
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp_counts.get(timestamp, 0) != 1:
            blockers.append("timestamp_missing_nonfinite_negative_or_duplicate")
        if prior_timestamp is not None and math.isfinite(timestamp) and timestamp <= prior_timestamp:
            blockers.append("timestamp_not_strictly_monotonic")
        if math.isfinite(timestamp):
            prior_timestamp = timestamp

        world_result = validate_se3_matrix(frame.get("world_from_camera"), field="world_from_camera")
        camera_result = validate_se3_matrix(frame.get("camera_from_world"), field="camera_from_world")
        blockers.extend(world_result["blockers"])
        blockers.extend(camera_result["blockers"])
        if world_result["valid"] and camera_result["valid"]:
            world = np.asarray(world_result["matrix"], dtype=np.float64)
            camera = np.asarray(camera_result["matrix"], dtype=np.float64)
            if float(np.max(np.abs(world @ camera - np.eye(4)))) > 1e-5:
                blockers.append("world_camera_pose_inverse_mismatch")

        image_path = _contained_artifact_path(frame.get("image_path"), geometry_root=geometry_root)
        depth_path = _contained_artifact_path(frame.get("depth_path"), geometry_root=geometry_root)
        confidence_path = _contained_artifact_path(frame.get("confidence_path"), geometry_root=geometry_root)
        image = _decode_tensor(image_path) if image_path else None
        depth = _decode_tensor(depth_path) if depth_path else None
        confidence = _decode_tensor(confidence_path) if confidence_path else None
        if image is None or image.ndim not in {2, 3} or tuple(image.shape[:2]) != (height, width):
            blockers.append("rgb_tensor_missing_corrupt_or_shape_mismatch")
        if depth is None or depth.ndim != 2 or tuple(depth.shape) != (height, width):
            blockers.append("depth_tensor_missing_corrupt_or_shape_mismatch")
        elif float(np.min(depth)) <= 0.0 or float(np.max(depth)) > 10_000.0:
            blockers.append("depth_tensor_range_invalid")
        if confidence is None or confidence.ndim != 2 or tuple(confidence.shape) != (height, width):
            blockers.append("confidence_tensor_missing_corrupt_or_shape_mismatch")
        elif float(np.min(confidence)) < 0.0 or float(np.max(confidence)) > 1.0:
            blockers.append("confidence_tensor_range_invalid")

        depth_unit = str(frame.get("depth_unit") or "").strip().lower()
        metric_depth_truth = frame.get("metric_depth_truth") is True
        depth_source = str(frame.get("depth_measurement_source") or "").strip().lower()
        if depth_unit not in {"m", "meter", "meters", "metre", "metres"}:
            blockers.append("depth_unit_missing_or_not_meters")
        if not provider_result_fallback and (
            not metric_depth_truth
            or depth_source not in {"sensor_depth", "validated_sfm", "provider_metric_reconstruction"}
        ):
            blockers.append("metric_depth_truth_not_explicitly_proven")
        confidence_range = frame.get("confidence_range")
        if not (
            isinstance(confidence_range, list)
            and len(confidence_range) == 2
            and _safe_float(confidence_range[0], -1.0) == 0.0
            and _safe_float(confidence_range[1], -1.0) == 1.0
        ):
            blockers.append("confidence_unit_range_missing_or_invalid")
        pose_confidence = _safe_float(frame.get("pose_confidence"), float("nan"))
        if not math.isfinite(pose_confidence) or not (0.0 <= pose_confidence <= 1.0):
            blockers.append("pose_confidence_missing_or_out_of_range")

        if not blockers:
            normalized = dict(frame)
            normalized.update(
                {
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "confidence_path": str(confidence_path),
                    "width": width,
                    "height": height,
                    "depth_unit": "meters",
                    "metric_depth_truth": metric_depth_truth and not provider_result_fallback,
                    "geometry_record_verified": True,
                }
            )
            verified.append(normalized)
        else:
            rejections.append(
                {
                    "row_index": row_index,
                    "frame_id": frame_id or None,
                    "frame_index": frame_index,
                    "blockers": list(dict.fromkeys(blockers)),
                }
            )
    return {
        "schema_version": "geometry_record_validation.v1",
        "intrinsics": intrinsics_result,
        "input_record_count": len(frame_records),
        "verified_record_count": len(verified),
        "all_records_verified": bool(frame_records) and len(verified) == len(frame_records),
        "verified_records": verified,
        "rejections": rejections,
    }


def _frame_id(frame_index: int) -> str:
    return str(int(frame_index)).zfill(6)


def _start_immutable_geometry_run(geometry_root: Path) -> Dict[str, Any]:
    """Archive the prior canonical run before exposing a new current run."""

    lineage: Dict[str, Any] = {
        "schema_version": "geometry_previous_run_lineage.v1",
        "previous_run_present": False,
        "previous_run_archive_path": None,
        "previous_status": None,
        "previous_synthetic_geometry": False,
    }
    if not geometry_root.exists():
        return lineage
    if geometry_root.is_symlink() or not geometry_root.is_dir():
        raise PipelineError(f"Geometry root must be a real directory: {geometry_root}")
    previous_summary = _optional_json(geometry_root / "geometry_summary.json")
    previous_status = _optional_json(geometry_root / "geometry_run_status.json")
    history_root = geometry_root.parent / "geometry_runs"
    ensure_dir(history_root)
    base_name = utc_now_iso().replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")
    archive_path = history_root / base_name
    suffix = 1
    while archive_path.exists():
        archive_path = history_root / f"{base_name}_{suffix}"
        suffix += 1
    geometry_root.replace(archive_path)
    lineage.update(
        {
            "previous_run_present": True,
            "previous_run_archive_path": str(archive_path),
            "previous_status": previous_summary.get("status") or previous_status.get("status"),
            "previous_synthetic_geometry": bool(
                previous_summary.get("synthetic_geometry")
                or previous_summary.get("synthetic_geometry_used")
                or previous_status.get("synthetic_geometry")
            ),
        }
    )
    return lineage


def _remove_current_geometry_tensors(geometry_root: Path) -> List[str]:
    removed: List[str] = []
    for name in _CURRENT_GEOMETRY_DIRS:
        path = geometry_root / name
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
            removed.append(name)
        elif path.is_dir():
            shutil.rmtree(path)
            removed.append(name)
    return removed


def _summary_capture_source(descriptor: CaptureDescriptor) -> str:
    media_metadata = (
        descriptor.metadata.get("media_metadata")
        if isinstance(descriptor.metadata.get("media_metadata"), Mapping)
        else {}
    )
    source_device = str(
        descriptor.metadata.get("source_device")
        or media_metadata.get("source_device")
        or ""
    ).strip()
    capture_source = str(descriptor.capture_source or "unknown").strip() or "unknown"
    if capture_source == "glasses" and source_device == "meta_glasses":
        return "meta_glasses"
    if capture_source == "glasses":
        return "non_arkit_video"
    return capture_source


@dataclass(frozen=True)
class GeometryStageResult:
    capture_root: Path
    geometry_root: Path
    manifest_path: Path
    summary_path: Path
    status_path: Path
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "v1",
            "capture_root": str(self.capture_root),
            "geometry_root": str(self.geometry_root),
            "manifest_path": str(self.manifest_path),
            "summary_path": str(self.summary_path),
            "status_path": str(self.status_path),
            "status": self.status,
        }


def _optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_video_path(context: LocalCaptureContext) -> Path:
    for name in _VIDEO_CANDIDATES:
        candidate = context.raw_root / name
        if candidate.is_file():
            return candidate
    raise PipelineError(
        "No walkthrough video found in raw bundle. Expected one of: "
        + ", ".join(_VIDEO_CANDIDATES)
    )


def _probe_video(video_path: Path) -> Dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {
            "probe_status": "unavailable",
            "video_path": str(video_path),
            "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
        }

    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, Mapping) and str(stream.get("codec_type") or "") == "video"
        ),
        {},
    )
    format_payload = payload.get("format") if isinstance(payload.get("format"), Mapping) else {}
    return {
        "probe_status": "ok",
        "video_path": str(video_path),
        "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
        "codec_name": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "pix_fmt": video_stream.get("pix_fmt"),
        "avg_frame_rate": video_stream.get("avg_frame_rate"),
        "duration_seconds": format_payload.get("duration"),
        "bit_rate": format_payload.get("bit_rate"),
    }


def _json_pointer(path: Path, *, context: LocalCaptureContext) -> Dict[str, Any]:
    relative = path.relative_to(context.capture_root)
    gs_uri = f"gs://{context.bucket}/{context.capture_prefix}/{relative.as_posix()}"
    return {
        "path": str(path),
        "relative_path": relative.as_posix(),
        "gs_uri": gs_uri,
        "present": path.exists(),
    }


def _load_descriptor(context: LocalCaptureContext) -> CaptureDescriptor:
    return CaptureDescriptor.from_file(context.descriptor_path)


def assess_geometry_scale(descriptor: CaptureDescriptor) -> Dict[str, Any]:
    scaffolding_validation = (
        descriptor.scaffolding_validation
        if isinstance(descriptor.scaffolding_validation, Mapping)
        else {}
    )
    validated_bundle = bool(scaffolding_validation.get("validated_metric_bundle"))
    if descriptor.evidence_tier == "qualified_metric_capture":
        return {
            "status": "metric_trusted",
            "metric_trusted": True,
            "trusted_for_measurement": True,
            "reason": "qualified_metric_capture",
            "capture_modality": descriptor.capture_modality,
            "evidence_tier": descriptor.evidence_tier,
        }
    if descriptor.evidence_tier == "video_with_validated_scaffolding" and validated_bundle:
        return {
            "status": "metric_trusted",
            "metric_trusted": True,
            "trusted_for_measurement": True,
            "reason": "validated_video_scaffolding",
            "capture_modality": descriptor.capture_modality,
            "evidence_tier": descriptor.evidence_tier,
        }
    if (
        descriptor.capture_source in {"glasses", "android"}
        or descriptor.capture_modality.startswith("glasses")
        or descriptor.capture_modality in {
            "android_video_only",
            "android_plus_scaffolding",
            "android_arcore_depth",
            "android_arcore_pose_only",
        }
    ):
        if descriptor.capture_modality in {"glasses_plus_scaffolding", "android_plus_scaffolding"}:
            reason = "video_geometry_without_validated_scaffolding"
        elif descriptor.capture_modality in {"android_arcore_depth", "android_arcore_pose_only"}:
            reason = "raw_tracking_without_validated_scale"
        else:
            reason = "video_conditioning_only"
        return {
            "status": "conditioning_only",
            "metric_trusted": False,
            "trusted_for_measurement": False,
            "reason": reason,
            "capture_modality": descriptor.capture_modality,
            "evidence_tier": descriptor.evidence_tier,
        }
    return {
        "status": "estimated_scale",
        "metric_trusted": False,
        "trusted_for_measurement": False,
        "reason": "no_validated_metric_scaffolding",
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
    }


def _frame_lines(frames: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(item), sort_keys=True) + "\n" for item in frames)


def _track_length_m(poses: List[Mapping[str, Any]]) -> float:
    total = 0.0
    previous: Optional[List[List[float]]] = None
    for pose in poses:
        matrix = pose.get("world_from_camera")
        if not isinstance(matrix, list) or len(matrix) < 4:
            continue
        if previous is not None:
            dx = float(matrix[0][3]) - float(previous[0][3])
            dy = float(matrix[1][3]) - float(previous[1][3])
            dz = float(matrix[2][3]) - float(previous[2][3])
            total += math.sqrt(dx * dx + dy * dy + dz * dz)
        previous = matrix
    return round(total, 6)


def _confidence_summary(frames: List[Mapping[str, Any]]) -> Dict[str, Any]:
    pose_values = [
        float(frame.get("pose_confidence") or 0.0)
        for frame in frames
        if frame.get("pose_confidence") is not None
    ]
    if not pose_values:
        return {"mean_pose_confidence": 0.0, "min_pose_confidence": 0.0}
    return {
        "mean_pose_confidence": round(sum(pose_values) / len(pose_values), 6),
        "min_pose_confidence": round(min(pose_values), 6),
    }


def _build_status_payload(
    *,
    provider: str,
    model: str,
    execution_mode: str,
    status: str,
    ready_for_world_model: bool,
    blocking_issues: List[str],
    geometry_source: str = "pending",
    fallback_used: bool = False,
    fallback_kind: Optional[str] = None,
    synthetic_geometry: bool = False,
    provider_native_result: Optional[bool] = None,
    contract_ready_for_world_model: bool = False,
    internal_fallback_ready: bool = False,
    geometry_live_ready: bool = False,
    external_market_ready: bool = False,
    site_faithful_market_ready: bool = False,
    launch_blockers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "stage": "geometry",
        "status": status,
        "geometry_source": geometry_source,
        "fallback_used": bool(fallback_used),
        "fallback_kind": fallback_kind,
        "synthetic_geometry": bool(synthetic_geometry),
        "provider_native_result": provider_native_result,
        "ready_for_world_model": ready_for_world_model,
        "contract_ready_for_world_model": bool(contract_ready_for_world_model),
        "internal_fallback_ready": bool(internal_fallback_ready),
        "geometry_live_ready": bool(geometry_live_ready),
        "external_market_ready": bool(external_market_ready),
        "site_faithful_market_ready": bool(site_faithful_market_ready),
        "launch_blockers": list(launch_blockers or []),
        "provider": provider,
        "model": model,
        "execution_mode": execution_mode,
        "blocking_issues": list(blocking_issues),
    }


def _build_manifest_payload(
    *,
    context: LocalCaptureContext,
    provider: str,
    model: str,
    execution_mode: str,
    status: str,
    summary_path: Path,
    status_path: Path,
    inputs_path: Path,
    provider_request_path: Path,
    provider_result_path: Path,
    intrinsics_path: Path,
    poses_path: Path,
    trajectory_summary_path: Path,
    keyframes_path: Path,
    frame_index_path: Path,
    depth_manifest_path: Path,
    confidence_manifest_path: Path,
    implementation_notes_path: Path,
    alignment_manifest_path: Optional[Path] = None,
    canonical_pointcloud_path: Optional[Path] = None,
    dynamic_mask_manifest_path: Optional[Path] = None,
    geometry_source: str = "pending",
    fallback_used: bool = False,
    fallback_kind: Optional[str] = None,
    provider_native_result: Optional[bool] = None,
    contract_ready_for_world_model: bool = False,
    internal_fallback_ready: bool = False,
    geometry_live_ready: bool = False,
    external_market_ready: bool = False,
    site_faithful_market_ready: bool = False,
    launch_blockers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "manifest_type": "geometry_manifest",
        "stage": "geometry",
        "status": status,
        "capture_identity": {
            "bucket": context.bucket,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "capture_prefix": context.capture_prefix,
        },
        "provider": {
            "name": provider,
            "model": model,
            "execution_mode": execution_mode,
            "geometry_source": geometry_source,
            "provider_native_result": provider_native_result,
            "fallback_used": bool(fallback_used),
            "fallback_kind": fallback_kind,
        },
        "artifacts": {
            "geometry_summary": _json_pointer(summary_path, context=context),
            "geometry_run_status": _json_pointer(status_path, context=context),
            "geometry_inputs": _json_pointer(inputs_path, context=context),
            "provider_request": _json_pointer(provider_request_path, context=context),
            "provider_result": _json_pointer(provider_result_path, context=context),
            "camera_intrinsics": _json_pointer(intrinsics_path, context=context),
            "camera_poses": _json_pointer(poses_path, context=context),
            "trajectory_summary": _json_pointer(trajectory_summary_path, context=context),
            "keyframes": _json_pointer(keyframes_path, context=context),
            "frame_index": _json_pointer(frame_index_path, context=context),
            "depth_manifest": _json_pointer(depth_manifest_path, context=context),
            "confidence_manifest": _json_pointer(confidence_manifest_path, context=context),
            "implementation_notes": _json_pointer(implementation_notes_path, context=context),
            "alignment_manifest": _json_pointer(alignment_manifest_path, context=context) if alignment_manifest_path else None,
            "canonical_pointcloud": _json_pointer(canonical_pointcloud_path, context=context) if canonical_pointcloud_path else None,
            "dynamic_mask_manifest": _json_pointer(dynamic_mask_manifest_path, context=context) if dynamic_mask_manifest_path else None,
        },
        "world_model_contract": {
            "authoritative": False,
            "raw_capture_authoritative": True,
            "truth_label": (
                "synthetic_diagnostic_not_capture_truth"
                if fallback_used
                else "video_to_world_derived_geometry"
                if geometry_live_ready
                else "pending_or_incomplete_geometry"
            ),
            "geometry_source": geometry_source,
            "provider_native_result": provider_native_result,
            "fallback_used": bool(fallback_used),
            "fallback_kind": fallback_kind,
            "ready_for_world_model": bool(geometry_live_ready),
            "contract_ready_for_world_model": bool(contract_ready_for_world_model),
            "internal_fallback_ready": bool(internal_fallback_ready),
            "geometry_live_ready": bool(geometry_live_ready),
            "external_market_ready": bool(external_market_ready),
            "site_faithful_market_ready": bool(site_faithful_market_ready),
            "intended_use": [
                "retrieval grounding",
                "scene_memory conditioning",
                "world-model API input",
                "pose-conditioned semantic processing",
            ],
            "blocked_until": list(launch_blockers or []),
        },
    }


def _build_inputs_payload(
    *,
    context: LocalCaptureContext,
    descriptor: CaptureDescriptor,
    video_path: Path,
    video_probe: Mapping[str, Any],
    provider: str,
    model: str,
    execution_mode: str,
) -> Dict[str, Any]:
    raw_manifest = _optional_json(context.raw_root / "manifest.json")
    capture_context = _optional_json(context.raw_root / "capture_context.json")
    descriptor_payload = descriptor.to_dict()
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "capture_identity": {
            "bucket": context.bucket,
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "capture_prefix": context.capture_prefix,
        },
        "source": {
            "video": _json_pointer(video_path, context=context),
            "raw_manifest": _json_pointer(context.raw_root / "manifest.json", context=context),
            "capture_context": _json_pointer(context.raw_root / "capture_context.json", context=context),
            "capture_descriptor": _json_pointer(context.descriptor_path, context=context),
        },
        "video_probe": dict(video_probe),
        "raw_manifest_hints": {
            "capture_modality": raw_manifest.get("capture_modality") or descriptor.capture_modality,
            "orientation": raw_manifest.get("capture_orientation") or capture_context.get("captureOrientation"),
        },
        "descriptor_hints": {
            "capture_source": descriptor.capture_source,
            "capture_modality": descriptor.capture_modality,
            "evidence_tier": descriptor.evidence_tier,
            "scaffolding_validation": dict(descriptor.scaffolding_validation),
            "quality": dict(descriptor.quality),
            "requested_outputs": list(descriptor.requested_outputs),
            "raw_descriptor": descriptor_payload,
        },
        "provider_config": {
            "provider": provider,
            "model": model,
            "execution_mode": execution_mode,
        },
    }


def _build_provider_request_payload(
    *,
    video_path: Path,
    video_uri: str,
    geometry_root: Path,
    dynamic_mask_manifest_path: Path,
    dynamic_mask_manifest_uri: str,
    provider: str,
    model: str,
    execution_mode: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "request_type": "geometry_reconstruction",
        "provider": provider,
        "model": model,
        "execution_mode": execution_mode,
        "input_video_path": str(video_path),
        "input_video_uri": video_uri,
        "geometry_root": str(geometry_root),
        "dynamic_mask_manifest_path": str(dynamic_mask_manifest_path),
        "dynamic_mask_manifest_uri": dynamic_mask_manifest_uri,
    }


def _video_to_world_provider_blocker() -> Optional[Dict[str, Any]]:
    missing = [
        name
        for name in ("VIDEO_TO_WORLD_URL", "VIDEO_TO_WORLD_RUNNER_TOKEN")
        if not str(os.getenv(name) or "").strip()
    ]
    if not missing:
        return None
    return {
        "id": "provider_native_geometry_missing",
        "reason": "video_to_world_runner_not_configured",
        "required_env": ["VIDEO_TO_WORLD_URL", "VIDEO_TO_WORLD_RUNNER_TOKEN"],
        "missing_env": missing,
        "command": (
            "VIDEO_TO_WORLD_URL=https://<video-to-world-runner> "
            "VIDEO_TO_WORLD_RUNNER_TOKEN=<secret> "
            "python3 scripts/run_geometry_lane.py --capture-root <capture-root> "
            "--provider video_to_world --model video_to_world-default"
        ),
    }


class SyntheticGeometryDisallowedError(PipelineError):
    """Raised when a synthetic-geometry path is requested but disallowed.

    Production launch-proof mode never fabricates geometry; the stage writes a
    blocked artifact instead of synthetic tensors.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"synthetic_geometry_disallowed:{reason}")
        self.reason = reason


def _run_geometry_provider(
    *,
    video_path: Path,
    video_uri: str,
    geometry_root: Path,
    dynamic_mask_manifest_path: Path,
    dynamic_mask_manifest_uri: str,
    provider: str,
    model: str,
    execution_mode: str,
    video_probe: Mapping[str, Any],
) -> Dict[str, Any]:
    provider_key = str(provider or "").strip().lower()
    if provider_key in {"local_sfm", "sfm", "offline_sfm", "non_arkit_local"}:
        if not synthetic_geometry_allowed():
            raise SyntheticGeometryDisallowedError("local_sfm_synthetic_dev_requested")
        return _build_local_sfm_provider_result(
            video_path=video_path,
            geometry_root=geometry_root,
            video_probe=video_probe,
            provider_blocker=None,
        )
    if provider_key in {"da3", "local_da3", "depth_anything_3"}:
        result = run_da3_provider(
            video_path=video_path,
            geometry_root=geometry_root,
            video_probe=video_probe,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
        )
        metrics = dict(result.get("provider_metrics") or {})
        if bool(metrics.get("fallback_used")):
            result["fallback_used"] = True
            result["fallback_kind"] = "local_da3_synthetic_depth"
            warnings = list(result.get("provider_warnings") or [])
            result["provider_warnings"] = list(dict.fromkeys([*warnings, "local_da3_synthetic_depth_used"]))
        return result
    provider_blocker = _video_to_world_provider_blocker()
    if provider_blocker is not None:
        if not synthetic_geometry_allowed():
            raise SyntheticGeometryDisallowedError(
                str(provider_blocker.get("reason") or "video_to_world_runner_not_configured")
            )
        return _build_local_sfm_provider_result(
            video_path=video_path,
            geometry_root=geometry_root,
            video_probe=video_probe,
            provider_blocker=provider_blocker,
        )
    return run_video_to_world_provider(
        video_path=video_path,
        video_uri=video_uri,
        geometry_root=geometry_root,
        dynamic_mask_manifest_path=dynamic_mask_manifest_path,
        dynamic_mask_manifest_uri=dynamic_mask_manifest_uri,
        provider=provider,
        model=model,
        execution_mode=execution_mode,
        video_probe=video_probe,
    )


def _provider_result_payload(
    *,
    provider: str,
    model: str,
    execution_mode: str,
    status: str,
    geometry_source: str,
    provider_native_result: bool,
    fallback_used: bool,
    fallback_kind: Optional[str],
    frame_count: int,
    depth_count: int,
    confidence_count: int,
    metrics: Mapping[str, Any],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "provider": provider,
        "model": model,
        "execution_mode": execution_mode,
        "geometry_source": geometry_source,
        "provider_native_result": bool(provider_native_result),
        "fallback_used": bool(fallback_used),
        "fallback_kind": fallback_kind,
        "artifacts_written": [
            f"frames:{frame_count}",
            f"depth:{depth_count}",
            f"confidence:{confidence_count}",
        ],
        "metrics": dict(metrics),
        "warnings": list(warnings),
        "errors": list(errors),
    }


def _write_ascii_pointcloud(path: Path, poses: List[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    points: List[tuple[float, float, float]] = []
    for pose in poses:
        matrix = pose.get("world_from_camera") or pose.get("T_world_camera")
        if not isinstance(matrix, list) or len(matrix) < 4:
            continue
        try:
            points.append(
                (
                    float(matrix[0][3]),
                    float(matrix[1][3]),
                    float(matrix[2][3]),
                )
            )
        except (TypeError, ValueError, IndexError):
            continue
    # An empty trajectory yields an honest 0-vertex pointcloud, never a
    # fabricated origin point.
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_fallback_provider_result(
    *,
    video_path: Path,
    geometry_root: Path,
    video_probe: Mapping[str, Any],
    provider_error: Exception,
) -> Dict[str, Any]:
    frames_dir = geometry_root / "frames" / "images"
    depth_dir = geometry_root / "depth"
    confidence_dir = geometry_root / "confidence"
    ensure_dir(frames_dir)
    ensure_dir(depth_dir)
    ensure_dir(confidence_dir)

    width = _safe_int(video_probe.get("width"), 640)
    height = _safe_int(video_probe.get("height"), 480)
    duration = _safe_float(video_probe.get("duration_seconds"), 1.0)
    diagnostic_height = max(height // 16, 24)
    diagnostic_width = max(width // 16, 32)
    sample_count = max(3, min(8, int(round(duration)) + 2))
    timestamps = [round(duration * idx / float(max(sample_count - 1, 1)), 3) for idx in range(sample_count)]
    frames: List[Dict[str, Any]] = []

    for idx, timestamp_seconds in enumerate(timestamps):
        image_path = frames_dir / f"frame_{idx:06d}.npy"
        depth_path = depth_dir / f"depth_{idx:06d}.npy"
        confidence_path = confidence_dir / f"confidence_{idx:06d}.npy"
        rgb = np.full((diagnostic_height, diagnostic_width, 3), 80 + idx * 12, dtype=np.float32)
        depth = np.full(rgb.shape[:2], 1.5 + idx * 0.05, dtype=np.float32)
        confidence = np.full(rgb.shape[:2], 0.75, dtype=np.float32)
        np.save(image_path, rgb)
        np.save(depth_path, depth)
        np.save(confidence_path, confidence)
        tx = idx * 0.18
        frames.append(
            {
                "frame_index": idx,
                "frame_id": _frame_id(idx),
                "timestamp_seconds": timestamp_seconds,
                "image_path": str(image_path),
                "is_keyframe": idx == 0 or idx == sample_count - 1 or idx % 2 == 0,
                "blur_score": 0.0,
                "overlap_hint": max(0.1, 0.95 - idx * 0.08),
                "world_from_camera": [
                    [1.0, 0.0, 0.0, tx],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.2],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "camera_from_world": [
                    [1.0, 0.0, 0.0, -tx],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.2],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "pose_confidence": 0.55,
                "depth_path": str(depth_path),
                "depth_format": "npy",
                "confidence_path": str(confidence_path),
                "confidence_format": "npy",
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "min_depth_m": float(depth.min()),
                "max_depth_m": float(depth.max()),
                "depth_unit": "meters",
                "metric_depth_truth": False,
                "depth_measurement_source": "synthetic_diagnostic",
                "confidence_range": [0.0, 1.0],
            }
        )

    return {
        "intrinsics": {
            "camera_model": "pinhole",
            "image_width": int(diagnostic_width),
            "image_height": int(diagnostic_height),
            "fx": float(max(diagnostic_width, diagnostic_height)),
            "fy": float(max(diagnostic_width, diagnostic_height)),
            "cx": float(diagnostic_width / 2.0),
            "cy": float(diagnostic_height / 2.0),
            "distortion": {"model": "none", "coefficients": []},
        },
        "frames": frames,
        "provider_metrics": {"fallback_reason": str(provider_error)},
        "provider_warnings": [f"provider_failed:{provider_error.__class__.__name__}", "fallback_geometry_used"],
        "provider_errors": [str(provider_error)],
        "loop_closure_detected": False,
        "geometry_source": "fallback_geometry",
        "provider_native_result": False,
        "fallback_used": True,
        "fallback_kind": "internal_synthetic_geometry",
        "synthetic_geometry": True,
        "synthetic_geometry_used": True,
        "synthetic_artifact_truth": {
            "poses_are_capture_truth": False,
            "intrinsics_are_calibrated_capture_truth": False,
            "depth_is_sensor_or_sfm_truth": False,
            "intended_use": "diagnostic_shape_only",
        },
    }


def _build_local_sfm_provider_result(
    *,
    video_path: Path,
    geometry_root: Path,
    video_probe: Mapping[str, Any],
    provider_blocker: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    local = _build_fallback_provider_result(
        video_path=video_path,
        geometry_root=geometry_root,
        video_probe=video_probe,
        provider_error=RuntimeError(
            str(provider_blocker.get("reason"))
            if isinstance(provider_blocker, Mapping)
            else "local_sfm_relative_geometry"
        ),
    )
    warnings = [
        warning
        for warning in list(local.get("provider_warnings") or [])
        if not str(warning).startswith("provider_failed:")
        and warning != "fallback_geometry_used"
    ]
    warnings.extend(
        [
            "local_sfm_relative_geometry_only",
            "synthetic_geometry_used",
            "local_sfm_real_runner_not_implemented",
            "local_sfm_uses_synthetic_diagnostics_only",
            "scale_not_proven",
            "site_frame_not_proven",
        ]
    )
    if provider_blocker is not None:
        warnings.append(str(provider_blocker.get("reason") or "video_to_world_runner_not_configured"))
    local["geometry_source"] = "fallback_geometry"
    local["requested_geometry_source"] = "local_sfm"
    local["provider_native_result"] = False
    # Truth flags are append-only: this path reuses the synthetic fabricator,
    # so it must stay labeled as fallback/synthetic. No real SfM ran here.
    local["fallback_used"] = True
    local["fallback_kind"] = "internal_synthetic_geometry"
    local["synthetic_geometry"] = True
    local["synthetic_geometry_used"] = True
    local["provider_metrics"] = {
        **dict(local.get("provider_metrics") or {}),
        "backend": "local_sfm_offline",
        "requested_backend": "local_sfm_offline",
        "real_sfm_runner_executed": False,
        "provider_native_result": False,
        "scale_resolved": False,
        "site_frame_available": False,
        "synthetic_geometry": True,
        "provider_blocker": dict(provider_blocker) if isinstance(provider_blocker, Mapping) else None,
    }
    local["provider_warnings"] = list(dict.fromkeys(warnings))
    local["provider_errors"] = [] if provider_blocker is not None else ["local_sfm_real_runner_not_implemented"]
    local["provider_blocker"] = dict(provider_blocker) if isinstance(provider_blocker, Mapping) else None
    local["site_frame_available"] = False
    local["scale_resolved"] = False
    local["pose_match_rate"] = 0.0
    local["p95_pose_delta_sec"] = None
    return local


def _build_dynamic_mask_manifest(
    *,
    context: LocalCaptureContext,
    geometry_root: Path,
) -> Path:
    masks_root = geometry_root / "masks"
    ensure_dir(masks_root)
    manifest_path = masks_root / "dynamic_mask_manifest.json"
    privacy_manifest = context.pipeline_root / "privacy_processing_manifest.json"
    privacy_masks_root = context.capture_root / "privacy" / "masks"
    artifacts: List[Dict[str, Any]] = []
    if privacy_masks_root.is_dir():
        for candidate in sorted(privacy_masks_root.rglob("*")):
            if not candidate.is_file():
                continue
            artifacts.append(
                {
                    "path": str(candidate),
                    "relative_path": str(candidate.relative_to(context.capture_root)),
                    "kind": "privacy_mask",
                }
            )
    payload: Dict[str, Any] = {
        "schema_version": "v2",
        "generated_at": utc_now_iso(),
        "mask_source": "privacy_processing" if privacy_manifest.is_file() else "none",
        "privacy_manifest_path": str(privacy_manifest) if privacy_manifest.is_file() else None,
        "policies": {
            "exclude_from_retrieval": True,
            "exclude_from_static_fusion": True,
            "fallback_dynamic_regions": ["people", "unknown_motion"],
        },
        "static_scene_priors": {
            "assume_walls_static": True,
            "assume_floor_static": True,
            "assume_vehicles_dynamic": True,
            "assume_people_dynamic": True,
        },
        "artifacts": artifacts,
    }
    write_json(manifest_path, payload)
    return manifest_path


def _build_canonical_geometry_artifacts(
    *,
    context: LocalCaptureContext,
    geometry_root: Path,
    pose_records: List[Mapping[str, Any]],
    geometry_source: str,
    fallback_used: bool,
    coordinate_frame_session_id: str,
    canonical_pointcloud_source_path: Optional[str] = None,
) -> Dict[str, Path]:
    alignment_root = geometry_root / "alignment"
    ensure_dir(alignment_root)

    canonical_pointcloud_path = alignment_root / "canonical_pointcloud.ply"
    source_path = (
        _contained_artifact_path(canonical_pointcloud_source_path, geometry_root=geometry_root)
        if canonical_pointcloud_source_path
        else None
    )
    if source_path and source_path.is_file():
        ensure_dir(canonical_pointcloud_path.parent)
        canonical_pointcloud_path.write_bytes(source_path.read_bytes())
    else:
        _write_ascii_pointcloud(canonical_pointcloud_path, pose_records)

    canonical_frame_id = str((pose_records[0].get("frame_index") if pose_records else 0))
    alignment_manifest_path = alignment_root / "alignment_manifest.json"
    write_json(
        alignment_manifest_path,
        {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "geometry_source": geometry_source,
            "fallback_used": fallback_used,
            "coordinate_frame_session_id": coordinate_frame_session_id,
            "canonical_frame_id": _frame_id(_safe_int(canonical_frame_id)),
            "canonical_pointcloud_path": str(canonical_pointcloud_path),
        },
    )
    return {
        "alignment_manifest_path": alignment_manifest_path,
        "canonical_pointcloud_path": canonical_pointcloud_path,
    }


def _geometry_gs_uri(*, context: LocalCaptureContext, path: Path) -> str:
    relative = path.relative_to(context.capture_root)
    return f"gs://{context.bucket}/{context.capture_prefix}/{relative.as_posix()}"


def _patch_descriptor_with_geometry(
    *,
    context: LocalCaptureContext,
    descriptor: CaptureDescriptor,
    geometry_source: str,
    ready_for_world_model: bool,
    contract_ready_for_world_model: bool,
    internal_fallback_ready: bool,
    geometry_live_ready: bool,
    external_market_ready: bool,
    site_faithful_market_ready: bool,
    provider_native_result: bool,
    fallback_used: bool,
    fallback_kind: Optional[str],
    launch_blockers: List[str],
    coordinate_frame_session_id: str,
    summary_path: Path,
    manifest_path: Path,
) -> None:
    descriptor_payload = descriptor.to_dict()
    quality = dict(descriptor_payload.get("quality") or {})
    metadata = dict(descriptor_payload.get("metadata") or {})
    scene_memory_capture = (
        dict(metadata.get("scene_memory_capture"))
        if isinstance(metadata.get("scene_memory_capture"), Mapping)
        else {}
    )
    rights = (
        dict(metadata.get("capture_rights"))
        if isinstance(metadata.get("capture_rights"), Mapping)
        else {}
    )
    derived_allowed = bool(rights.get("derived_scene_generation_allowed", False))
    capture_mode = (
        dict(metadata.get("capture_mode"))
        if isinstance(metadata.get("capture_mode"), Mapping)
        else {}
    )
    requested_mode = str(capture_mode.get("requested_mode") or "site_world_candidate")
    candidate = bool(ready_for_world_model and derived_allowed and requested_mode == "site_world_candidate")
    reasoning = [
        f"capture_mode_site_world_candidate:{requested_mode == 'site_world_candidate'}",
        f"geometry_ready:{ready_for_world_model}",
        f"geometry_live_ready:{geometry_live_ready}",
        f"geometry_source:{geometry_source}",
        f"fallback_used:{fallback_used}",
        f"derived_scene_generation_allowed:{derived_allowed}",
    ]

    descriptor_payload["geometry_source"] = geometry_source
    descriptor_payload["geometry_ready"] = ready_for_world_model
    descriptor_payload["geometry_live_ready"] = geometry_live_ready
    descriptor_payload["coordinate_frame_session_id"] = coordinate_frame_session_id
    descriptor_payload["world_model_candidate"] = candidate
    quality["geometry_source"] = geometry_source
    quality["geometry_ready"] = ready_for_world_model
    quality["contract_ready_for_world_model"] = contract_ready_for_world_model
    quality["internal_fallback_ready"] = internal_fallback_ready
    quality["geometry_live_ready"] = geometry_live_ready
    quality["external_market_ready"] = external_market_ready
    quality["site_faithful_market_ready"] = site_faithful_market_ready
    quality["provider_native_result"] = provider_native_result
    quality["fallback_used"] = fallback_used
    quality["fallback_kind"] = fallback_kind
    quality["world_model_candidate"] = candidate
    descriptor_payload["quality"] = quality

    scene_memory_capture["geometry_source"] = geometry_source
    scene_memory_capture["geometry_ready"] = ready_for_world_model
    scene_memory_capture["contract_ready_for_world_model"] = contract_ready_for_world_model
    scene_memory_capture["internal_fallback_ready"] = internal_fallback_ready
    scene_memory_capture["geometry_live_ready"] = geometry_live_ready
    scene_memory_capture["provider_native_result"] = provider_native_result
    scene_memory_capture["fallback_used"] = fallback_used
    scene_memory_capture["fallback_kind"] = fallback_kind
    scene_memory_capture["world_model_candidate"] = candidate
    scene_memory_capture["world_model_candidate_reasoning"] = reasoning
    metadata["scene_memory_capture"] = scene_memory_capture

    metadata["geometry"] = {
        "geometry_source": geometry_source,
        "ready_for_world_model": ready_for_world_model,
        "geometry_ready": ready_for_world_model,
        "contract_ready_for_world_model": contract_ready_for_world_model,
        "internal_fallback_ready": internal_fallback_ready,
        "geometry_live_ready": geometry_live_ready,
        "external_market_ready": external_market_ready,
        "site_faithful_market_ready": site_faithful_market_ready,
        "provider_native_result": provider_native_result,
        "fallback_used": fallback_used,
        "fallback_kind": fallback_kind,
        "launch_blockers": list(launch_blockers),
        "geometry_summary_uri": _geometry_gs_uri(context=context, path=summary_path),
        "geometry_manifest_uri": _geometry_gs_uri(context=context, path=manifest_path),
        "coordinate_frame_session_id": coordinate_frame_session_id,
    }
    metadata["capture_mode"] = {
        "requested_mode": requested_mode,
        "resolved_mode": "site_world_candidate" if candidate else "qualification_only",
        "downgrade_reason": None
        if candidate
        else "fallback_geometry_not_live_video_to_world"
        if fallback_used
        else "provider_native_geometry_missing"
        if geometry_source == "local_sfm"
        else "geometry_not_ready",
    }
    topology = (
        dict(metadata.get("capture_topology"))
        if isinstance(metadata.get("capture_topology"), Mapping)
        else {}
    )
    topology["capture_session_id"] = coordinate_frame_session_id
    metadata["capture_topology"] = topology
    descriptor_payload["metadata"] = metadata
    write_json(context.descriptor_path, descriptor_payload)


def _write_blocked_geometry_artifacts(
    *,
    context: LocalCaptureContext,
    provider: str,
    model: str,
    execution_mode: str,
    summary_path: Path,
    status_path: Path,
    manifest_path: Path,
    inputs_path: Path,
    provider_request_path: Path,
    provider_result_path: Path,
    intrinsics_path: Path,
    poses_path: Path,
    trajectory_summary_path: Path,
    keyframes_path: Path,
    frame_index_path: Path,
    depth_manifest_path: Path,
    confidence_manifest_path: Path,
    implementation_notes_path: Path,
    dynamic_mask_manifest_path: Path,
    geometry_root: Path,
    reason: str,
) -> GeometryStageResult:
    """Record a blocked geometry run without fabricating any tensors.

    No depth/pose/intrinsics artifacts are written; the run is marked
    blocked so downstream stages and gates fail closed instead of consuming
    synthetic geometry.
    """
    removed_tensor_directories = _remove_current_geometry_tensors(geometry_root)
    status_label = "blocked_geometry_unavailable"
    launch_blockers = [
        "geometry_provider_unavailable",
        "synthetic_geometry_disallowed",
        reason,
    ]
    blockers = list(dict.fromkeys(launch_blockers))
    summary_payload = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "stage": "geometry",
        "status": status_label,
        "geometry_source": "unavailable",
        "fallback_used": False,
        "fallback_kind": None,
        "synthetic_geometry": False,
        "provider_native_result": False,
        "ready_for_world_model": False,
        "contract_ready_for_world_model": False,
        "internal_fallback_ready": False,
        "geometry_live_ready": False,
        "external_market_ready": False,
        "site_faithful_market_ready": False,
        "launch_blockers": blockers,
        "blockers": blockers,
        "blocked_reason": reason,
        "intrinsics_available": False,
        "site_frame_available": False,
        "scale_resolved": False,
        "current_usable_artifacts": [],
        "current_usable_tensor_count": 0,
        "removed_partial_tensor_directories": removed_tensor_directories,
        "previous_run_lineage": _optional_json(geometry_root / "previous_run_lineage.json"),
    }
    write_json(summary_path, summary_payload)
    status_payload = _build_status_payload(
        provider=provider,
        model=model,
        execution_mode=execution_mode,
        status=status_label,
        ready_for_world_model=False,
        geometry_source="unavailable",
        launch_blockers=blockers,
        blocking_issues=blockers,
    )
    status_payload.update(
        {
            "current_usable_artifacts": [],
            "current_usable_tensor_count": 0,
        }
    )
    write_json(
        status_path,
        status_payload,
    )
    write_json(
        manifest_path,
        _build_manifest_payload(
            context=context,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            status=status_label,
            summary_path=summary_path,
            status_path=status_path,
            inputs_path=inputs_path,
            provider_request_path=provider_request_path,
            provider_result_path=provider_result_path,
            intrinsics_path=intrinsics_path,
            poses_path=poses_path,
            trajectory_summary_path=trajectory_summary_path,
            keyframes_path=keyframes_path,
            frame_index_path=frame_index_path,
            depth_manifest_path=depth_manifest_path,
            confidence_manifest_path=confidence_manifest_path,
            implementation_notes_path=implementation_notes_path,
            dynamic_mask_manifest_path=dynamic_mask_manifest_path,
            geometry_source="unavailable",
            launch_blockers=blockers,
        ),
    )
    manifest_payload = _optional_json(manifest_path)
    manifest_payload.update(
        {
            "current_usable_artifacts": [],
            "current_usable_tensor_count": 0,
            "previous_run_lineage_path": str(geometry_root / "previous_run_lineage.json"),
        }
    )
    write_json(manifest_path, manifest_payload)
    descriptor = _load_descriptor(context)
    topology = (
        descriptor.metadata.get("capture_topology")
        if isinstance(descriptor.metadata.get("capture_topology"), Mapping)
        else {}
    )
    coordinate_frame_session_id = str(
        descriptor.coordinate_frame_session_id
        or topology.get("capture_session_id")
        or topology.get("captureSessionId")
        or context.capture_id
    )
    _patch_descriptor_with_geometry(
        context=context,
        descriptor=descriptor,
        geometry_source="unavailable",
        ready_for_world_model=False,
        contract_ready_for_world_model=False,
        internal_fallback_ready=False,
        geometry_live_ready=False,
        external_market_ready=False,
        site_faithful_market_ready=False,
        provider_native_result=False,
        fallback_used=False,
        fallback_kind=None,
        launch_blockers=blockers,
        coordinate_frame_session_id=coordinate_frame_session_id,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )
    return GeometryStageResult(
        capture_root=context.capture_root,
        geometry_root=geometry_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        status_path=status_path,
        status=status_label,
    )


def build_geometry_stage_contract(
    capture_root: str | Path,
    *,
    provider: str = "video_to_world",
    model: str = "video_to_world-default",
    execution_mode: str = "standard",
) -> GeometryStageResult:
    context = resolve_local_capture_context(capture_root)
    descriptor = _load_descriptor(context)
    video_path = _resolve_video_path(context)
    video_probe = _probe_video(video_path)

    geometry_root = context.pipeline_root / "geometry"
    previous_run_lineage = _start_immutable_geometry_run(geometry_root)
    camera_root = geometry_root / "camera"
    frames_root = geometry_root / "frames"
    depth_root = geometry_root / "depth"
    confidence_root = geometry_root / "confidence"
    logs_root = geometry_root / "logs"
    for path in (geometry_root, camera_root, frames_root, depth_root, confidence_root, logs_root):
        ensure_dir(path)
    write_json(geometry_root / "previous_run_lineage.json", previous_run_lineage)

    manifest_path = geometry_root / "geometry_manifest.json"
    summary_path = geometry_root / "geometry_summary.json"
    status_path = geometry_root / "geometry_run_status.json"
    inputs_path = geometry_root / "geometry_inputs.json"
    provider_request_path = logs_root / "provider_request.json"
    provider_result_path = logs_root / "provider_result.json"
    intrinsics_path = camera_root / "intrinsics.json"
    poses_path = camera_root / "poses.jsonl"
    trajectory_summary_path = camera_root / "trajectory_summary.json"
    keyframes_path = frames_root / "keyframes.json"
    frame_index_path = frames_root / "frame_index.jsonl"
    depth_manifest_path = depth_root / "depth_manifest.json"
    confidence_manifest_path = confidence_root / "confidence_manifest.json"
    implementation_notes_path = geometry_root / "IMPLEMENTATION_NOTES.md"
    alignment_manifest_path = geometry_root / "alignment" / "alignment_manifest.json"
    canonical_pointcloud_path = geometry_root / "alignment" / "canonical_pointcloud.ply"
    dynamic_mask_manifest_path = geometry_root / "masks" / "dynamic_mask_manifest.json"
    dynamic_mask_manifest_path = _build_dynamic_mask_manifest(context=context, geometry_root=geometry_root)

    initial_status = _build_status_payload(
        provider=provider,
        model=model,
        execution_mode=execution_mode,
        status="running",
        ready_for_world_model=False,
        blocking_issues=["provider_execution_in_progress"],
    )
    inputs_payload = _build_inputs_payload(
        context=context,
        descriptor=descriptor,
        video_path=video_path,
        video_probe=video_probe,
        provider=provider,
        model=model,
        execution_mode=execution_mode,
    )
    provider_request = _build_provider_request_payload(
        video_path=video_path,
        video_uri=descriptor.raw_video_uri or "",
        geometry_root=geometry_root,
        dynamic_mask_manifest_path=dynamic_mask_manifest_path,
        dynamic_mask_manifest_uri=f"gs://{context.bucket}/{context.capture_prefix}/pipeline/geometry/masks/dynamic_mask_manifest.json",
        provider=provider,
        model=model,
        execution_mode=execution_mode,
    )
    write_json(status_path, initial_status)
    write_json(inputs_path, inputs_payload)
    write_json(provider_request_path, provider_request)
    write_json(
        manifest_path,
        _build_manifest_payload(
            context=context,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            status="running",
            summary_path=summary_path,
            status_path=status_path,
            inputs_path=inputs_path,
            provider_request_path=provider_request_path,
            provider_result_path=provider_result_path,
            intrinsics_path=intrinsics_path,
            poses_path=poses_path,
            trajectory_summary_path=trajectory_summary_path,
            keyframes_path=keyframes_path,
            frame_index_path=frame_index_path,
            depth_manifest_path=depth_manifest_path,
            confidence_manifest_path=confidence_manifest_path,
            implementation_notes_path=implementation_notes_path,
            alignment_manifest_path=alignment_manifest_path,
            canonical_pointcloud_path=canonical_pointcloud_path,
            dynamic_mask_manifest_path=dynamic_mask_manifest_path,
        ),
    )

    scale_assessment = assess_geometry_scale(descriptor)
    implementation_notes = """# Geometry Lane Notes

This folder contains derived canonical geometry for downstream SWM/Cosmos-style retrieval and world-model paths.

- The geometry lane is non-authoritative.
- Metric trust follows capture evidence tier and validated scaffolding policy.
- Meta/glasses geometry without validated scaffolding remains conditioning-only.
- Canonical poses, depth, and reference frames are readiness-critical.
- GS/splat render assets are not part of the public alpha contract.
"""
    implementation_notes_path.write_text(implementation_notes, encoding="utf-8")

    try:
        provider_result = _run_geometry_provider(
            video_path=video_path,
            video_uri=descriptor.raw_video_uri or "",
            geometry_root=geometry_root,
            dynamic_mask_manifest_path=dynamic_mask_manifest_path,
            dynamic_mask_manifest_uri=f"gs://{context.bucket}/{context.capture_prefix}/pipeline/geometry/masks/dynamic_mask_manifest.json",
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            video_probe=video_probe,
        )
    except SyntheticGeometryDisallowedError as exc:
        return _write_blocked_geometry_artifacts(
            context=context,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            summary_path=summary_path,
            status_path=status_path,
            manifest_path=manifest_path,
            inputs_path=inputs_path,
            provider_request_path=provider_request_path,
            provider_result_path=provider_result_path,
            intrinsics_path=intrinsics_path,
            poses_path=poses_path,
            trajectory_summary_path=trajectory_summary_path,
            keyframes_path=keyframes_path,
            frame_index_path=frame_index_path,
            depth_manifest_path=depth_manifest_path,
            confidence_manifest_path=confidence_manifest_path,
            implementation_notes_path=implementation_notes_path,
            dynamic_mask_manifest_path=dynamic_mask_manifest_path,
            geometry_root=geometry_root,
            reason=exc.reason,
        )
    except Exception as exc:
        if not synthetic_geometry_allowed():
            return _write_blocked_geometry_artifacts(
                context=context,
                provider=provider,
                model=model,
                execution_mode=execution_mode,
                summary_path=summary_path,
                status_path=status_path,
                manifest_path=manifest_path,
                inputs_path=inputs_path,
                provider_request_path=provider_request_path,
                provider_result_path=provider_result_path,
                intrinsics_path=intrinsics_path,
                poses_path=poses_path,
                trajectory_summary_path=trajectory_summary_path,
                keyframes_path=keyframes_path,
                frame_index_path=frame_index_path,
                depth_manifest_path=depth_manifest_path,
                confidence_manifest_path=confidence_manifest_path,
                implementation_notes_path=implementation_notes_path,
                dynamic_mask_manifest_path=dynamic_mask_manifest_path,
                geometry_root=geometry_root,
                reason=f"provider_failed:{exc.__class__.__name__}",
            )
        provider_result = _build_fallback_provider_result(
            video_path=video_path,
            geometry_root=geometry_root,
            video_probe=video_probe,
            provider_error=exc,
        )

    frame_records = list(provider_result.get("frames") or [])
    if not frame_records:
        return _write_blocked_geometry_artifacts(
            context=context,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            summary_path=summary_path,
            status_path=status_path,
            manifest_path=manifest_path,
            inputs_path=inputs_path,
            provider_request_path=provider_request_path,
            provider_result_path=provider_result_path,
            intrinsics_path=intrinsics_path,
            poses_path=poses_path,
            trajectory_summary_path=trajectory_summary_path,
            keyframes_path=keyframes_path,
            frame_index_path=frame_index_path,
            depth_manifest_path=depth_manifest_path,
            confidence_manifest_path=confidence_manifest_path,
            implementation_notes_path=implementation_notes_path,
            dynamic_mask_manifest_path=dynamic_mask_manifest_path,
            geometry_root=geometry_root,
            reason="provider_returned_zero_frame_records",
        )
    provider_key = str(provider or "").strip().lower()
    local_da3_provider = provider_key in {"da3", "local_da3", "depth_anything_3"}
    provider_result_fallback = bool(provider_result.get("fallback_used"))
    provider_result_source = str(provider_result.get("geometry_source") or "").strip()
    local_sfm_provider = provider_result_source == "local_sfm" or provider_key in {
        "local_sfm",
        "sfm",
        "offline_sfm",
        "non_arkit_local",
    }
    frame_geometry_source = (
        "fallback_geometry"
        if provider_result_fallback
        else "local_sfm"
        if local_sfm_provider
        else "local_da3"
        if local_da3_provider
        else "video_to_world"
    )

    intrinsics_payload = {
        "schema_version": "v1",
        **dict(provider_result.get("intrinsics") or {}),
        "source": {
            "producer": provider,
            "model": model,
            "execution_mode": execution_mode,
            "calibration_truth": "synthetic_diagnostic_not_calibrated"
            if provider_result_fallback
            else "provider_output",
            "capture_truth": not provider_result_fallback,
        },
    }
    validation_report = _validate_geometry_frame_records(
        frame_records=[row for row in frame_records if isinstance(row, Mapping)],
        intrinsics_payload=intrinsics_payload,
        geometry_root=geometry_root,
        provider_result_fallback=provider_result_fallback,
    )
    verified_frame_records = list(validation_report.pop("verified_records"))
    validation_report_path = geometry_root / "geometry_validation_report.json"
    write_json(validation_report_path, validation_report)
    write_json(intrinsics_path, intrinsics_payload)

    pose_records: List[Dict[str, Any]] = []
    keyframe_items: List[Dict[str, Any]] = []
    frame_index_records: List[Dict[str, Any]] = []
    depth_artifacts: List[Dict[str, Any]] = []
    confidence_artifacts: List[Dict[str, Any]] = []

    for frame in verified_frame_records:
        frame_index = _safe_int(frame.get("frame_index"))
        timestamp_seconds = _safe_float(frame.get("timestamp_seconds"))
        frame_id = str(frame.get("frame_id") or _frame_id(frame_index))
        pose_confidence = _safe_float(frame.get("pose_confidence"), 1.0)
        image_path = str(frame.get("image_path") or "")
        depth_path = str(frame.get("depth_path") or "")
        confidence_path = str(frame.get("confidence_path") or "")
        sharpness_score = _frame_sharpness_score(image_path)
        if sharpness_score is None and frame.get("sharpness_score") is not None:
            sharpness_score = _safe_float(frame.get("sharpness_score"))
        pose_records.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "image_path": image_path,
                "world_from_camera": frame.get("world_from_camera"),
                "camera_from_world": frame.get("camera_from_world"),
                "pose_confidence": pose_confidence,
                "is_keyframe": bool(frame.get("is_keyframe")),
                "pose_truth_source": "synthetic_diagnostic"
                if provider_result_fallback
                else frame_geometry_source,
            }
        )
        frame_index_records.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "image_path": image_path,
                "depth_path": depth_path,
                "confidence_path": confidence_path,
                "pose_present": bool(frame.get("world_from_camera")) and not provider_result_fallback,
                "intrinsics_present": bool(provider_result.get("intrinsics")) and not provider_result_fallback,
                "pose_confidence": pose_confidence,
                # Only carry a sharpness value that was actually measured
                # (image gradient variance) or provider-reported; a missing
                # measurement stays visible as missing, never a stamped constant.
                "sharpness_score": sharpness_score,
                "sharpness_score_source": "image_gradient_variance"
                if sharpness_score is not None
                else "missing",
                "sharpness_measured": sharpness_score is not None,
                "geometry_source": frame_geometry_source,
                "synthetic_geometry_used": provider_result_fallback,
                "capture_truth": not provider_result_fallback,
            }
        )
        depth_artifacts.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "path": depth_path,
                "format": str(frame.get("depth_format") or "npy"),
                "width": _safe_int(frame.get("width") or intrinsics_payload.get("image_width")),
                "height": _safe_int(frame.get("height") or intrinsics_payload.get("image_height")),
                "min_depth_m": _safe_float(frame.get("min_depth_m")),
                "max_depth_m": _safe_float(frame.get("max_depth_m")),
                "metric_depth_truth": frame.get("metric_depth_truth") is True,
                "depth_source": "synthetic_diagnostic"
                if provider_result_fallback
                else frame_geometry_source,
            }
        )
        confidence_artifacts.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "path": confidence_path,
                "format": str(frame.get("confidence_format") or "npy"),
                "width": _safe_int(frame.get("width") or intrinsics_payload.get("image_width")),
                "height": _safe_int(frame.get("height") or intrinsics_payload.get("image_height")),
                "value_range": list(frame.get("confidence_range") or [0.0, 1.0]),
            }
        )
        if bool(frame.get("is_keyframe")):
            keyframe_items.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "image_path": image_path,
                    "blur_score": _safe_float(frame.get("blur_score")),
                    "overlap_hint": _safe_float(frame.get("overlap_hint"), 1.0),
                }
            )

    poses_path.write_text(_frame_lines(pose_records), encoding="utf-8")
    frame_index_path.write_text(_frame_lines(frame_index_records), encoding="utf-8")
    write_json(
        keyframes_path,
        {
            "schema_version": "v1",
            "sampling_strategy": f"{provider}:{execution_mode}",
            "frames": keyframe_items,
        },
    )
    write_json(
        depth_manifest_path,
        {
            "schema_version": "v1",
            "representation": "per_frame_depth_map",
            "unit": "meters",
            "frame_count": len(depth_artifacts),
            "artifacts": depth_artifacts,
        },
    )
    write_json(
        confidence_manifest_path,
        {
            "schema_version": "v1",
            "representation": "per_frame_confidence_map",
            "frame_count": len(confidence_artifacts),
            "artifacts": confidence_artifacts,
        },
    )
    trajectory_summary = {
        "schema_version": "v1",
        "pose_count": len(pose_records),
        "keyframe_count": len(keyframe_items),
        "track_length_m": _track_length_m(pose_records),
        "loop_closure_detected": bool(provider_result.get("loop_closure_detected")),
        "scale_status": str(scale_assessment.get("status") or "conditioning_only"),
        "confidence_summary": _confidence_summary(pose_records),
    }
    write_json(trajectory_summary_path, trajectory_summary)

    provider_key = str(provider or "").strip().lower()
    local_da3_provider = provider_key in {"da3", "local_da3", "depth_anything_3"}
    local_sfm_provider = str(provider_result.get("geometry_source") or "").strip() == "local_sfm" or provider_key in {
        "local_sfm",
        "sfm",
        "offline_sfm",
        "non_arkit_local",
    }
    fallback_used = bool(provider_result.get("fallback_used"))
    synthetic_geometry = bool(provider_result.get("synthetic_geometry"))
    fallback_kind = (
        str(provider_result.get("fallback_kind") or "internal_synthetic_geometry")
        if fallback_used
        else None
    )
    geometry_source = (
        "fallback_geometry"
        if fallback_used
        else "local_sfm"
        if local_sfm_provider
        else "local_da3"
        if local_da3_provider
        else "video_to_world"
    )
    provider_native_result = bool(
        provider_result.get(
            "provider_native_result",
            (not fallback_used and geometry_source == "video_to_world"),
        )
    )
    launch_blockers = (
        [
            "fallback_geometry_not_launchable",
            "fallback_geometry_not_live_video_to_world",
            "synthetic_geometry_not_capture_truth",
            "synthetic_intrinsics_not_calibrated",
            "synthetic_depth_not_sensor_depth_or_sfm",
        ]
        if fallback_used
        else []
    )
    provider_blocker = (
        dict(provider_result.get("provider_blocker"))
        if isinstance(provider_result.get("provider_blocker"), Mapping)
        else None
    )
    if provider_blocker is not None:
        launch_blockers.append(str(provider_blocker.get("reason") or "video_to_world_runner_not_configured"))
        launch_blockers.append("provider_native_geometry_missing")
    metadata_topology = (
        descriptor.metadata.get("capture_topology")
        if isinstance(descriptor.metadata.get("capture_topology"), Mapping)
        else {}
    )
    coordinate_frame_session_id = str(
        descriptor.coordinate_frame_session_id
        or metadata_topology.get("capture_session_id")
        or metadata_topology.get("captureSessionId")
        or context.capture_id
    )
    if pose_records:
        canonical_artifacts = _build_canonical_geometry_artifacts(
            context=context,
            geometry_root=geometry_root,
            pose_records=pose_records,
            geometry_source=geometry_source,
            fallback_used=fallback_used,
            coordinate_frame_session_id=coordinate_frame_session_id,
            canonical_pointcloud_source_path=str(provider_result.get("canonical_pointcloud_source_path") or ""),
        )
    else:
        canonical_artifacts = {
            "alignment_manifest_path": alignment_manifest_path,
            "canonical_pointcloud_path": canonical_pointcloud_path,
        }

    provider_result_payload = _provider_result_payload(
        provider=provider,
        model=model,
        execution_mode=execution_mode,
        status="provider_failed_synthetic_diagnostics_written" if fallback_used else "succeeded",
        geometry_source=geometry_source,
        provider_native_result=provider_native_result,
        fallback_used=fallback_used,
        fallback_kind=fallback_kind,
        frame_count=len(frame_records),
        depth_count=len(depth_artifacts),
        confidence_count=len(confidence_artifacts),
        metrics=provider_result.get("provider_metrics") if isinstance(provider_result.get("provider_metrics"), Mapping) else {},
        warnings=list(provider_result.get("provider_warnings") or []),
        errors=list(provider_result.get("provider_errors") or []),
    )
    write_json(provider_result_path, provider_result_payload)

    input_frame_count = int(validation_report.get("input_record_count") or len(frame_records))
    pose_coverage = round(len(pose_records) / float(input_frame_count or 1), 6)
    confidence_coverage = round(len(confidence_artifacts) / float(input_frame_count or 1), 6)
    depth_coverage = round(len(depth_artifacts) / float(input_frame_count or 1), 6)
    intrinsics_available = bool(
        (validation_report.get("intrinsics") or {}).get("valid")
    )
    pose_track_count = len(pose_records)
    pose_match_rate = min(
        pose_coverage,
        _safe_float(provider_result.get("pose_match_rate"), pose_coverage),
    )
    p95_pose_delta_raw = provider_result.get("p95_pose_delta_sec")
    p95_pose_delta_sec = (
        _safe_float(p95_pose_delta_raw)
        if p95_pose_delta_raw is not None
        else None
    )
    site_frame_available = bool(provider_result.get("site_frame_available"))
    scale_resolved = bool(provider_result.get("scale_resolved")) or bool(
        scale_assessment.get("metric_trusted")
    )
    all_records_verified = bool(validation_report.get("all_records_verified"))
    diagnostic_artifacts_shape_ready = bool(
        all_records_verified
        and pose_records
        and depth_artifacts
        and confidence_artifacts
        and intrinsics_available
    )
    contract_ready_for_world_model = bool(
        diagnostic_artifacts_shape_ready
        and not fallback_used
        and provider_native_result
    )
    internal_fallback_ready = False
    geometry_live_ready = bool(
        contract_ready_for_world_model
        and provider_native_result
        and geometry_source == "video_to_world"
        and site_frame_available
        and scale_resolved
        and pose_match_rate >= 0.65
        and (p95_pose_delta_sec is not None and p95_pose_delta_sec <= 0.2)
    )
    ready_for_world_model = geometry_live_ready
    if local_sfm_provider:
        launch_blockers.extend(
            [
                "provider_native_geometry_missing",
                "scale_not_proven",
                "site_frame_not_proven",
            ]
        )
    elif not geometry_live_ready and not fallback_used:
        launch_blockers.append(
            "local_da3_not_live_video_to_world"
            if local_da3_provider
            else "video_to_world_geometry_incomplete"
        )
    if not intrinsics_available:
        launch_blockers.append("intrinsics_missing")
    if not all_records_verified:
        launch_blockers.append("geometry_records_failed_verification")
    if not site_frame_available:
        launch_blockers.append("site_frame_not_proven")
    if not scale_resolved:
        launch_blockers.append("scale_not_proven")
    if not provider_native_result:
        launch_blockers.append("provider_native_geometry_missing")
    if p95_pose_delta_sec is None:
        launch_blockers.append("pose_timing_not_proven")
    external_market_ready = bool(geometry_live_ready and not launch_blockers)
    site_faithful_market_ready = bool(
        external_market_ready
        and str(scale_assessment.get("status") or "") in {"metric_trusted", "estimated_scale"}
    )

    status_label = (
        "completed_with_fallback"
        if fallback_used
        else "completed"
        if ready_for_world_model
        else "completed_degraded"
    )
    blockers = list(dict.fromkeys(launch_blockers))
    summary_payload = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "stage": "geometry",
        "status": status_label,
        "geometry_source": geometry_source,
        "capture_source": _summary_capture_source(descriptor),
        "canonical_frame_id": pose_records[0]["frame_id"] if pose_records else None,
        "fallback_used": fallback_used,
        "fallback_kind": fallback_kind,
        "synthetic_geometry": synthetic_geometry,
        "provider_native_result": provider_native_result,
        "ready_for_world_model": ready_for_world_model,
        "contract_ready_for_world_model": contract_ready_for_world_model,
        "internal_fallback_ready": internal_fallback_ready,
        "diagnostic_artifacts_shape_ready": diagnostic_artifacts_shape_ready,
        "geometry_records_all_verified": all_records_verified,
        "input_geometry_record_count": input_frame_count,
        "verified_geometry_record_count": len(pose_records),
        "rejected_geometry_record_count": len(validation_report.get("rejections") or []),
        "geometry_validation_report": _json_pointer(validation_report_path, context=context),
        "synthetic_geometry_used": fallback_used,
        "synthetic_artifacts_are_capture_truth": not fallback_used,
        "geometry_live_ready": geometry_live_ready,
        "external_market_ready": external_market_ready,
        "site_faithful_market_ready": site_faithful_market_ready,
        "launch_blockers": blockers,
        "blockers": blockers,
        "provider_blocker": provider_blocker,
        "pose_track_count": pose_track_count,
        "pose_match_rate": round(float(pose_match_rate), 6),
        "p95_pose_delta_sec": p95_pose_delta_sec,
        "intrinsics_available": intrinsics_available,
        "site_frame_available": site_frame_available,
        "scale_resolved": scale_resolved,
        "scale_trust_classification": str(scale_assessment.get("status") or "conditioning_only"),
        "source_video": {
            "path": str(video_path),
            "duration_seconds": video_probe.get("duration_seconds"),
            "width": video_probe.get("width"),
            "height": video_probe.get("height"),
        },
        "provider": {
            "name": provider,
            "model": model,
            "execution_mode": execution_mode,
            "provider_native_result": provider_native_result,
            "provider_blocker": provider_blocker,
            "non_fallback_pose_count": len(pose_records) if not fallback_used else 0,
            "non_fallback_depth_count": len(depth_artifacts) if not fallback_used else 0,
            "warnings": list(provider_result_payload.get("warnings") or []),
        },
        "scale_assessment": {
            **scale_assessment,
            "pose_coverage": pose_coverage,
            "confidence_coverage": confidence_coverage,
            "depth_coverage": depth_coverage,
        },
        "canonical_scene_assets": {
            "alignment_manifest": _json_pointer(canonical_artifacts["alignment_manifest_path"], context=context),
            "canonical_pointcloud": _json_pointer(canonical_artifacts["canonical_pointcloud_path"], context=context),
        },
        "deliverables": {
            "manifest": _json_pointer(manifest_path, context=context),
            "status": _json_pointer(status_path, context=context),
            "intrinsics": _json_pointer(intrinsics_path, context=context),
            "poses": _json_pointer(poses_path, context=context),
            "trajectory_summary": _json_pointer(trajectory_summary_path, context=context),
            "keyframes": _json_pointer(keyframes_path, context=context),
            "frame_index": _json_pointer(frame_index_path, context=context),
            "depth_manifest": _json_pointer(depth_manifest_path, context=context),
            "confidence_manifest": _json_pointer(confidence_manifest_path, context=context),
            "dynamic_mask_manifest": _json_pointer(dynamic_mask_manifest_path, context=context),
            "pose_count": len(pose_records),
            "keyframe_count": len(keyframe_items),
            "depth_frame_count": len(depth_artifacts),
            "confidence_frame_count": len(confidence_artifacts),
            "verified_frame_count": len(pose_records),
            "rejected_frame_count": len(validation_report.get("rejections") or []),
            "pose_coverage": pose_coverage,
            "depth_coverage": depth_coverage,
            "confidence_coverage": confidence_coverage,
        },
    }
    write_json(summary_path, summary_payload)
    status_payload = _build_status_payload(
        provider=provider,
        model=model,
        execution_mode=execution_mode,
        status=status_label,
        ready_for_world_model=ready_for_world_model,
        geometry_source=geometry_source,
        fallback_used=fallback_used,
        fallback_kind=fallback_kind,
        synthetic_geometry=synthetic_geometry,
        provider_native_result=provider_native_result,
        contract_ready_for_world_model=contract_ready_for_world_model,
        internal_fallback_ready=internal_fallback_ready,
        geometry_live_ready=geometry_live_ready,
        external_market_ready=external_market_ready,
        site_faithful_market_ready=site_faithful_market_ready,
        launch_blockers=list(dict.fromkeys(launch_blockers)),
        blocking_issues=list(dict.fromkeys([*list(provider_result_payload.get("errors") or []), *launch_blockers])),
    )
    status_payload.update(
        {
            "input_geometry_record_count": input_frame_count,
            "verified_geometry_record_count": len(pose_records),
            "rejected_geometry_record_count": len(validation_report.get("rejections") or []),
            "current_usable_tensor_count": len(pose_records) * 3 if contract_ready_for_world_model else 0,
        }
    )
    write_json(
        status_path,
        status_payload,
    )
    write_json(
        manifest_path,
        _build_manifest_payload(
            context=context,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            status=status_label,
            summary_path=summary_path,
            status_path=status_path,
            inputs_path=inputs_path,
            provider_request_path=provider_request_path,
            provider_result_path=provider_result_path,
            intrinsics_path=intrinsics_path,
            poses_path=poses_path,
            trajectory_summary_path=trajectory_summary_path,
            keyframes_path=keyframes_path,
            frame_index_path=frame_index_path,
            depth_manifest_path=depth_manifest_path,
            confidence_manifest_path=confidence_manifest_path,
            implementation_notes_path=implementation_notes_path,
            alignment_manifest_path=canonical_artifacts["alignment_manifest_path"],
            canonical_pointcloud_path=canonical_artifacts["canonical_pointcloud_path"],
            dynamic_mask_manifest_path=dynamic_mask_manifest_path,
            geometry_source=geometry_source,
            fallback_used=fallback_used,
            fallback_kind=fallback_kind,
            provider_native_result=provider_native_result,
            contract_ready_for_world_model=contract_ready_for_world_model,
            internal_fallback_ready=internal_fallback_ready,
            geometry_live_ready=geometry_live_ready,
            external_market_ready=external_market_ready,
            site_faithful_market_ready=site_faithful_market_ready,
            launch_blockers=list(dict.fromkeys(launch_blockers)),
        ),
    )
    manifest_payload = _optional_json(manifest_path)
    manifest_payload.update(
        {
            "geometry_validation_report": _json_pointer(validation_report_path, context=context),
            "verified_geometry_record_count": len(pose_records),
            "current_usable_tensor_count": len(pose_records) * 3 if contract_ready_for_world_model else 0,
            "previous_run_lineage_path": str(geometry_root / "previous_run_lineage.json"),
        }
    )
    write_json(manifest_path, manifest_payload)
    _patch_descriptor_with_geometry(
        context=context,
        descriptor=descriptor,
        geometry_source=geometry_source,
        ready_for_world_model=ready_for_world_model,
        contract_ready_for_world_model=contract_ready_for_world_model,
        internal_fallback_ready=internal_fallback_ready,
        geometry_live_ready=geometry_live_ready,
        external_market_ready=external_market_ready,
        site_faithful_market_ready=site_faithful_market_ready,
        provider_native_result=provider_native_result,
        fallback_used=fallback_used,
        fallback_kind=fallback_kind,
        launch_blockers=list(dict.fromkeys(launch_blockers)),
        coordinate_frame_session_id=coordinate_frame_session_id,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )
    return GeometryStageResult(
        capture_root=context.capture_root,
        geometry_root=geometry_root,
        manifest_path=manifest_path,
        summary_path=summary_path,
        status_path=status_path,
        status=status_label,
    )
