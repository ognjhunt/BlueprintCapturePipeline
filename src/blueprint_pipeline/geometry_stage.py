"""Geometry lane execution and contract writing."""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .capture_bridge import CaptureDescriptor
from .common import PipelineError, ensure_dir, utc_now_iso, write_json
from .geometry_da3 import run_da3_provider
from .local_capture import LocalCaptureContext, resolve_local_capture_context
from .video_to_world_client import run_video_to_world_provider


_VIDEO_CANDIDATES = (
    "walkthrough.mov",
    "walkthrough.mp4",
    "recording.mov",
    "recording.mp4",
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


def _frame_id(frame_index: int) -> str:
    return str(int(frame_index)).zfill(6)


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
                "internal_fallback_not_site_faithful"
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
    if not points:
        points = [(0.0, 0.0, 0.0)]

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
    sample_count = max(3, min(8, int(round(duration)) + 2))
    timestamps = [round(duration * idx / float(max(sample_count - 1, 1)), 3) for idx in range(sample_count)]
    frames: List[Dict[str, Any]] = []

    for idx, timestamp_seconds in enumerate(timestamps):
        image_path = frames_dir / f"frame_{idx:06d}.npy"
        depth_path = depth_dir / f"depth_{idx:06d}.npy"
        confidence_path = confidence_dir / f"confidence_{idx:06d}.npy"
        rgb = np.full((max(height // 16, 24), max(width // 16, 32), 3), 80 + idx * 12, dtype=np.float32)
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
                "confidence_range": [0.0, 1.0],
            }
        )

    return {
        "intrinsics": {
            "camera_model": "pinhole",
            "image_width": int(width),
            "image_height": int(height),
            "fx": float(max(width, height)),
            "fy": float(max(width, height)),
            "cx": float(width / 2.0),
            "cy": float(height / 2.0),
            "distortion": {"model": "none", "coefficients": []},
        },
        "frames": frames,
        "provider_metrics": {"fallback_reason": str(provider_error)},
        "provider_warnings": [f"provider_failed:{provider_error.__class__.__name__}", "fallback_geometry_used"],
        "provider_errors": [str(provider_error)],
        "loop_closure_detected": False,
        "fallback_used": True,
        "fallback_kind": "internal_synthetic_geometry",
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
            "scale_not_proven",
            "site_frame_not_proven",
        ]
    )
    if provider_blocker is not None:
        warnings.append(str(provider_blocker.get("reason") or "video_to_world_runner_not_configured"))
    local["geometry_source"] = "local_sfm"
    local["provider_native_result"] = False
    local["fallback_used"] = False
    local["fallback_kind"] = None
    local["provider_metrics"] = {
        **dict(local.get("provider_metrics") or {}),
        "backend": "local_sfm_offline",
        "provider_native_result": False,
        "scale_resolved": False,
        "site_frame_available": False,
        "provider_blocker": dict(provider_blocker) if isinstance(provider_blocker, Mapping) else None,
    }
    local["provider_warnings"] = list(dict.fromkeys(warnings))
    local["provider_errors"] = []
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
    source_path = Path(canonical_pointcloud_source_path) if canonical_pointcloud_source_path else None
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
    camera_root = geometry_root / "camera"
    frames_root = geometry_root / "frames"
    depth_root = geometry_root / "depth"
    confidence_root = geometry_root / "confidence"
    logs_root = geometry_root / "logs"
    for path in (geometry_root, camera_root, frames_root, depth_root, confidence_root, logs_root):
        ensure_dir(path)

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
    except Exception as exc:
        provider_result = _build_fallback_provider_result(
            video_path=video_path,
            geometry_root=geometry_root,
            video_probe=video_probe,
            provider_error=exc,
        )

    frame_records = list(provider_result.get("frames") or [])
    if not frame_records:
        raise PipelineError("Geometry stage produced no frame records.")
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
        },
    }
    write_json(intrinsics_path, intrinsics_payload)

    pose_records: List[Dict[str, Any]] = []
    keyframe_items: List[Dict[str, Any]] = []
    frame_index_records: List[Dict[str, Any]] = []
    depth_artifacts: List[Dict[str, Any]] = []
    confidence_artifacts: List[Dict[str, Any]] = []

    for frame in frame_records:
        frame_index = _safe_int(frame.get("frame_index"))
        timestamp_seconds = _safe_float(frame.get("timestamp_seconds"))
        frame_id = str(frame.get("frame_id") or _frame_id(frame_index))
        pose_confidence = _safe_float(frame.get("pose_confidence"), 1.0)
        image_path = str(frame.get("image_path") or "")
        depth_path = str(frame.get("depth_path") or "")
        confidence_path = str(frame.get("confidence_path") or "")
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
                "pose_present": True,
                "intrinsics_present": True,
                "pose_confidence": pose_confidence,
                "sharpness_score": 100.0,
                "geometry_source": frame_geometry_source,
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
        ["fallback_geometry_not_launchable", "fallback_geometry_not_live_video_to_world"]
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
    canonical_artifacts = _build_canonical_geometry_artifacts(
        context=context,
        geometry_root=geometry_root,
        pose_records=pose_records,
        geometry_source=geometry_source,
        fallback_used=fallback_used,
        coordinate_frame_session_id=coordinate_frame_session_id,
        canonical_pointcloud_source_path=str(provider_result.get("canonical_pointcloud_source_path") or ""),
    )

    provider_result_payload = _provider_result_payload(
        provider=provider,
        model=model,
        execution_mode=execution_mode,
        status="provider_failed_fallback_generated" if fallback_used else "succeeded",
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

    pose_coverage = round(len(pose_records) / float(len(frame_records) or 1), 6)
    confidence_coverage = round(len(confidence_artifacts) / float(len(frame_records) or 1), 6)
    depth_coverage = round(len(depth_artifacts) / float(len(frame_records) or 1), 6)
    intrinsics_available = bool(
        intrinsics_payload.get("fx")
        and intrinsics_payload.get("fy")
        and intrinsics_payload.get("image_width")
        and intrinsics_payload.get("image_height")
    )
    pose_track_count = len(pose_records)
    pose_match_rate = _safe_float(provider_result.get("pose_match_rate"), pose_coverage)
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
    contract_ready_for_world_model = bool(
        pose_records and depth_artifacts and confidence_artifacts and intrinsics_available
    )
    internal_fallback_ready = bool(contract_ready_for_world_model and fallback_used)
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
        "provider_native_result": provider_native_result,
        "ready_for_world_model": ready_for_world_model,
        "contract_ready_for_world_model": contract_ready_for_world_model,
        "internal_fallback_ready": internal_fallback_ready,
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
            "pose_coverage": pose_coverage,
            "depth_coverage": depth_coverage,
            "confidence_coverage": confidence_coverage,
        },
    }
    write_json(summary_path, summary_payload)
    write_json(
        status_path,
        _build_status_payload(
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            status=status_label,
            ready_for_world_model=ready_for_world_model,
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
            blocking_issues=list(dict.fromkeys([*list(provider_result_payload.get("errors") or []), *launch_blockers])),
        ),
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
