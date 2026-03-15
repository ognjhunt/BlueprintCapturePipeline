"""Geometry lane execution and contract writing."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import PipelineError, ensure_dir, utc_now_iso, write_json
from .geometry_da3 import run_da3_provider
from .local_capture import LocalCaptureContext, resolve_local_capture_context


_VIDEO_CANDIDATES = (
    "walkthrough.mov",
    "walkthrough.mp4",
    "recording.mov",
    "recording.mp4",
)


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
    if descriptor.evidence_tier == "glasses_with_validated_scaffolding" and validated_bundle:
        return {
            "status": "metric_trusted",
            "metric_trusted": True,
            "trusted_for_measurement": True,
            "reason": "validated_glasses_scaffolding",
            "capture_modality": descriptor.capture_modality,
            "evidence_tier": descriptor.evidence_tier,
        }
    if descriptor.capture_source == "glasses" or descriptor.capture_modality.startswith("glasses"):
        reason = (
            "glasses_geometry_without_validated_scaffolding"
            if descriptor.capture_modality == "glasses_plus_scaffolding"
            else "glasses_video_conditioning_only"
        )
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
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "stage": "geometry",
        "status": status,
        "ready_for_world_model": ready_for_world_model,
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
        },
        "world_model_contract": {
            "authoritative": False,
            "intended_use": [
                "scene_memory conditioning",
                "world-model API input",
                "pose-conditioned semantic processing",
            ],
            "blocked_until": [],
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
    geometry_root: Path,
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
        "geometry_root": str(geometry_root),
    }


def _provider_result_payload(
    *,
    provider: str,
    model: str,
    execution_mode: str,
    status: str,
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
        "artifacts_written": [
            f"frames:{frame_count}",
            f"depth:{depth_count}",
            f"confidence:{confidence_count}",
        ],
        "metrics": dict(metrics),
        "warnings": list(warnings),
        "errors": list(errors),
    }


def build_geometry_stage_contract(
    capture_root: str | Path,
    *,
    provider: str = "da3",
    model: str = "DA3Nested-Giant-Large-1.1",
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
        geometry_root=geometry_root,
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
        ),
    )

    scale_assessment = assess_geometry_scale(descriptor)
    implementation_notes = """# Geometry Lane Notes

This folder contains derived geometry conditioning for downstream scene-memory and world-model paths.

- The geometry lane is non-authoritative.
- Metric trust follows capture evidence tier and validated scaffolding policy.
- Meta/glasses geometry without validated scaffolding remains conditioning-only.
"""
    implementation_notes_path.write_text(implementation_notes, encoding="utf-8")

    try:
        provider_result = run_da3_provider(
            video_path=video_path,
            geometry_root=geometry_root,
            video_probe=video_probe,
            provider=provider,
            model=model,
            execution_mode=execution_mode,
        )
        frame_records = list(provider_result.get("frames") or [])
        if not frame_records:
            raise PipelineError("Geometry provider returned no frame records.")

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
            frame_index = int(frame.get("frame_index") or 0)
            timestamp_seconds = float(frame.get("timestamp_seconds") or 0.0)
            pose_records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "image_path": str(frame.get("image_path") or ""),
                    "world_from_camera": frame.get("world_from_camera"),
                    "camera_from_world": frame.get("camera_from_world"),
                    "pose_confidence": float(frame.get("pose_confidence") or 0.0),
                    "is_keyframe": bool(frame.get("is_keyframe")),
                }
            )
            frame_index_records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "image_path": str(frame.get("image_path") or ""),
                    "depth_path": str(frame.get("depth_path") or ""),
                    "confidence_path": str(frame.get("confidence_path") or ""),
                    "pose_present": True,
                    "intrinsics_present": True,
                }
            )
            depth_artifacts.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "path": str(frame.get("depth_path") or ""),
                    "format": str(frame.get("depth_format") or "npy"),
                    "width": int(frame.get("width") or intrinsics_payload.get("image_width") or 0),
                    "height": int(frame.get("height") or intrinsics_payload.get("image_height") or 0),
                    "min_depth_m": float(frame.get("min_depth_m") or 0.0),
                    "max_depth_m": float(frame.get("max_depth_m") or 0.0),
                }
            )
            confidence_artifacts.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "path": str(frame.get("confidence_path") or ""),
                    "format": str(frame.get("confidence_format") or "npy"),
                    "width": int(frame.get("width") or intrinsics_payload.get("image_width") or 0),
                    "height": int(frame.get("height") or intrinsics_payload.get("image_height") or 0),
                    "value_range": list(frame.get("confidence_range") or [0.0, 1.0]),
                }
            )
            if bool(frame.get("is_keyframe")):
                keyframe_items.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp_seconds,
                        "image_path": str(frame.get("image_path") or ""),
                        "blur_score": float(frame.get("blur_score") or 0.0),
                        "overlap_hint": float(frame.get("overlap_hint") or 0.0),
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

        provider_result_payload = _provider_result_payload(
            provider=provider,
            model=model,
            execution_mode=execution_mode,
            status="completed",
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
        ready_for_world_model = bool(
            pose_records
            and depth_artifacts
            and confidence_artifacts
            and not provider_result_payload["errors"]
        )
        summary_payload = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "stage": "geometry",
            "status": "completed",
            "ready_for_world_model": ready_for_world_model,
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
                "warnings": list(provider_result_payload.get("warnings") or []),
            },
            "scale_assessment": {
                **scale_assessment,
                "pose_coverage": pose_coverage,
                "confidence_coverage": confidence_coverage,
                "depth_coverage": depth_coverage,
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
                status="completed",
                ready_for_world_model=ready_for_world_model,
                blocking_issues=[],
            ),
        )
        write_json(
            manifest_path,
            _build_manifest_payload(
                context=context,
                provider=provider,
                model=model,
                execution_mode=execution_mode,
                status="completed",
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
            ),
        )
        return GeometryStageResult(
            capture_root=context.capture_root,
            geometry_root=geometry_root,
            manifest_path=manifest_path,
            summary_path=summary_path,
            status_path=status_path,
            status="completed",
        )
    except Exception as exc:
        errors = [str(exc)]
        write_json(
            provider_result_path,
            _provider_result_payload(
                provider=provider,
                model=model,
                execution_mode=execution_mode,
                status="failed",
                frame_count=0,
                depth_count=0,
                confidence_count=0,
                metrics={},
                warnings=[],
                errors=errors,
            ),
        )
        write_json(
            summary_path,
            {
                "schema_version": "v1",
                "generated_at": utc_now_iso(),
                "stage": "geometry",
                "status": "failed",
                "ready_for_world_model": False,
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
                },
                "scale_assessment": scale_assessment,
                "deliverables": {
                    "manifest": _json_pointer(manifest_path, context=context),
                    "status": _json_pointer(status_path, context=context),
                    "provider_result": _json_pointer(provider_result_path, context=context),
                },
                "errors": errors,
            },
        )
        write_json(
            status_path,
            _build_status_payload(
                provider=provider,
                model=model,
                execution_mode=execution_mode,
                status="failed",
                ready_for_world_model=False,
                blocking_issues=errors,
            ),
        )
        write_json(
            manifest_path,
            _build_manifest_payload(
                context=context,
                provider=provider,
                model=model,
                execution_mode=execution_mode,
                status="failed",
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
            ),
        )
        return GeometryStageResult(
            capture_root=context.capture_root,
            geometry_root=geometry_root,
            manifest_path=manifest_path,
            summary_path=summary_path,
            status_path=status_path,
            status="failed",
        )
