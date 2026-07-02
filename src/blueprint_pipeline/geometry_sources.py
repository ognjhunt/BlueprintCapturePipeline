"""Source-neutral geometry helpers for ARKit and pipeline/geometry consumers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import read_json
from .local_capture import LocalCaptureContext


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _zero_pad_frame_id(value: Any, fallback: int = 0) -> str:
    if value is None or value == "":
        return str(int(fallback)).zfill(6)
    try:
        return str(int(value)).zfill(6)
    except (TypeError, ValueError):
        text = str(value).strip()
        if text.isdigit():
            return text.zfill(6)
        return text or str(int(fallback)).zfill(6)


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


def _first_existing_frame_path(*roots: Path, frame_id: str) -> Optional[str]:
    suffixes = (".jpg", ".jpeg", ".png", ".npy")
    folders = ("frames", "images", "frames/images")
    for root in roots:
        for folder in folders:
            base = root / folder
            for suffix in suffixes:
                candidate = base / f"{frame_id}{suffix}"
                if candidate.is_file():
                    return str(candidate)
    return None


def _parse_intrinsics_from_arkit_row(row: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    raw = row.get("intrinsics")
    res = row.get("imageResolution")
    if isinstance(raw, Mapping):
        try:
            return {
                "fx": float(raw["fx"]),
                "fy": float(raw["fy"]),
                "cx": float(raw["cx"]),
                "cy": float(raw["cy"]),
                "width": int(raw.get("width") or 0),
                "height": int(raw.get("height") or 0),
            }
        except (KeyError, TypeError, ValueError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 9:
        return None
    width = int(res[0]) if isinstance(res, (list, tuple)) and len(res) >= 2 else 0
    height = int(res[1]) if isinstance(res, (list, tuple)) and len(res) >= 2 else 0
    return {
        "fx": float(raw[0]),
        "fy": float(raw[4]),
        "cx": float(raw[6]),
        "cy": float(raw[7]),
        "width": width,
        "height": height,
    }


_SYNTHETIC_FALLBACK_KINDS = {
    "internal_synthetic_geometry",
    "local_sfm_synthetic_dev",
    "local_da3_synthetic_depth",
}


class SyntheticGeometryExportError(RuntimeError):
    """Raised when synthetic geometry reaches an export path that forbids it."""


def reconcile_geometry_truth_flags(summary: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (synthetic_geometry, fallback_used) with append-only semantics.

    Truth flags may only strengthen: if any synthetic indicator is present
    (explicit flag, a known synthetic fallback kind, or a synthetic warning),
    both flags are treated as set even if a later stage rewrote
    ``fallback_used`` to false. A relabel can never launder synthetic
    geometry back into provider truth.
    """
    if not isinstance(summary, Mapping):
        return (False, False)
    fallback_kind = str(summary.get("fallback_kind") or "").strip()
    warnings = {str(w) for w in (summary.get("warnings") or []) if w}
    provider = summary.get("provider") if isinstance(summary.get("provider"), Mapping) else {}
    provider_warnings = {str(w) for w in (provider.get("warnings") or []) if w}
    synthetic = bool(
        summary.get("synthetic_geometry")
        or fallback_kind in _SYNTHETIC_FALLBACK_KINDS
        or "synthetic_geometry_used" in warnings
        or "synthetic_geometry_used" in provider_warnings
        or "fallback_geometry_used" in warnings
        or "fallback_geometry_used" in provider_warnings
    )
    fallback_used = bool(summary.get("fallback_used")) or synthetic
    return (synthetic, fallback_used)


def geometry_export_gate(
    summary: Mapping[str, Any],
    *,
    export_name: str = "export",
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Gate package/eval exports on geometry truth flags.

    Raises ``SyntheticGeometryExportError`` when the geometry is synthetic or
    fallback-derived and synthetic geometry is disallowed (always the case in
    production launch-proof mode). When allowed (dev), returns a provenance
    stamp that MUST be attached to the export manifest.
    """
    from .launch_proof_policy import synthetic_geometry_allowed

    synthetic, fallback_used = reconcile_geometry_truth_flags(summary)
    if (synthetic or fallback_used) and not synthetic_geometry_allowed(env):
        raise SyntheticGeometryExportError(
            f"{export_name}_refused:synthetic_or_fallback_geometry_disallowed"
        )
    return {
        "synthetic_geometry": synthetic,
        "fallback_used": fallback_used,
        "fallback_kind": summary.get("fallback_kind") if isinstance(summary, Mapping) else None,
        "export_allowed_by": (
            "provider_geometry"
            if not (synthetic or fallback_used)
            else "synthetic_geometry_dev_allowance"
        ),
    }


def resolve_geometry_source(
    *,
    context: LocalCaptureContext,
    descriptor: Mapping[str, Any],
) -> str:
    geometry_dir = context.pipeline_root / "geometry"
    geometry_pose_path = geometry_dir / "camera" / "poses.jsonl"
    geometry_summary = read_json(geometry_dir / "geometry_summary.json") if (geometry_dir / "geometry_summary.json").is_file() else {}
    summary_source = (
        str(geometry_summary.get("geometry_source") or "").strip()
        if isinstance(geometry_summary, Mapping)
        else ""
    )
    arkit_pose_path = context.raw_root / "arkit" / "poses.jsonl"
    arcore_pose_path = context.raw_root / "arcore" / "poses.jsonl"

    top_level = str(descriptor.get("geometry_source") or "").strip()
    quality = descriptor.get("quality") if isinstance(descriptor.get("quality"), Mapping) else {}
    quality_source = str(quality.get("geometry_source") or "").strip()
    if geometry_pose_path.is_file():
        return summary_source or top_level or quality_source or "video_to_world"
    if arkit_pose_path.is_file():
        return "arkit"
    if arcore_pose_path.is_file():
        return top_level or quality_source or "arcore"
    return top_level or quality_source or "unknown"


def load_capture_geometry(
    *,
    context: LocalCaptureContext,
    descriptor: Mapping[str, Any],
) -> Dict[str, Any]:
    source = resolve_geometry_source(context=context, descriptor=descriptor)
    if source == "arkit":
        return _load_arkit_geometry(context=context, descriptor=descriptor)
    if source == "arcore":
        return _load_arcore_geometry(context=context, descriptor=descriptor)
    return _load_pipeline_geometry(context=context, descriptor=descriptor, source=source)


def _load_arkit_geometry(
    *,
    context: LocalCaptureContext,
    descriptor: Mapping[str, Any],
) -> Dict[str, Any]:
    arkit_root = context.raw_root / "arkit"
    poses_raw = _load_jsonl(arkit_root / "poses.jsonl")
    frames_raw = _load_jsonl(arkit_root / "frames.jsonl")
    frame_meta: Dict[str, Dict[str, Any]] = {}

    for row in frames_raw:
        frame_index = row.get("frameIndex")
        frame_id = _zero_pad_frame_id(frame_index)
        depth_path = arkit_root / "depth" / f"{frame_id}.png"
        confidence_path = arkit_root / "confidence" / f"{frame_id}.png"
        source_image_path = _first_existing_frame_path(
            arkit_root,
            context.raw_root,
            frame_id=frame_id,
        )
        frame_meta[frame_id] = {
            "frame_index": _safe_int(frame_index),
            "timestamp_seconds": _safe_float(row.get("timestamp")),
            "intrinsics_payload": _parse_intrinsics_from_arkit_row(row),
            "trackingState": row.get("trackingState", "normal"),
            "sharpnessScore": row.get("sharpnessScore"),
            "relocalizationEvent": bool(row.get("relocalizationEvent", False)),
            "worldMappingStatus": row.get("worldMappingStatus"),
            "anchorObservations": list(row.get("anchorObservations") or []),
            "source_image_path": source_image_path,
            "depth_path": str(depth_path) if depth_path.is_file() else None,
            "confidence_path": str(confidence_path) if confidence_path.is_file() else None,
            "pose_confidence": 1.0,
        }

    poses: List[Dict[str, Any]] = []
    for idx, row in enumerate(poses_raw):
        frame_index = row.get("frameIndex", idx)
        frame_id = _zero_pad_frame_id(row.get("frame_id"), fallback=_safe_int(frame_index))
        poses.append(
            {
                "frame_id": frame_id,
                "frame_index": _safe_int(frame_index, idx),
                "timestamp": _safe_float(row.get("timestamp"), _safe_float(row.get("t_device_sec"))),
                "T_world_camera": row.get("T_world_camera") or row.get("transform"),
            }
        )

    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    topology = metadata.get("capture_topology") if isinstance(metadata.get("capture_topology"), Mapping) else {}
    coordinate_frame_session_id = str(
        descriptor.get("coordinate_frame_session_id")
        or topology.get("capture_session_id")
        or topology.get("captureSessionId")
        or context.capture_id
    )
    return {
        "source": "arkit",
        "poses": poses,
        "frame_meta": frame_meta,
        "intrinsics": read_json(arkit_root / "intrinsics.json") if (arkit_root / "intrinsics.json").is_file() else {},
        "coordinate_frame_session_id": coordinate_frame_session_id,
        "ready_for_world_model": bool(descriptor.get("geometry_ready") or descriptor.get("quality", {}).get("world_model_candidate")),
    }


def _load_pipeline_geometry(
    *,
    context: LocalCaptureContext,
    descriptor: Mapping[str, Any],
    source: str,
) -> Dict[str, Any]:
    geometry_root = context.pipeline_root / "geometry"
    geometry_summary = read_json(geometry_root / "geometry_summary.json") if (geometry_root / "geometry_summary.json").is_file() else {}
    poses_raw = _load_jsonl(geometry_root / "camera" / "poses.jsonl")
    frames_raw = _load_jsonl(geometry_root / "frames" / "frame_index.jsonl")
    intrinsics = read_json(geometry_root / "camera" / "intrinsics.json") if (geometry_root / "camera" / "intrinsics.json").is_file() else {}

    meta_by_index: Dict[int, Dict[str, Any]] = {}
    for row in frames_raw:
        frame_index = _safe_int(row.get("frame_index"))
        meta_by_index[frame_index] = {
            "frame_index": frame_index,
            "timestamp_seconds": _safe_float(row.get("timestamp_seconds"), _safe_float(row.get("timestamp"))),
            "intrinsics_payload": dict(intrinsics) if isinstance(intrinsics, Mapping) else {},
            "trackingState": "normal",
            # None means "not measured" — downstream gates must not treat a
            # missing measurement as perfectly sharp.
            "sharpnessScore": row.get("sharpness_score"),
            "relocalizationEvent": False,
            "worldMappingStatus": row.get("world_mapping_status", "mapped"),
            "anchorObservations": list(row.get("anchor_observations") or []),
            "source_image_path": row.get("image_path"),
            "depth_path": row.get("depth_path"),
            "confidence_path": row.get("confidence_path"),
            "pose_confidence": _safe_float(row.get("pose_confidence"), 1.0),
        }

    poses: List[Dict[str, Any]] = []
    for idx, row in enumerate(poses_raw):
        frame_index = _safe_int(row.get("frame_index"), idx)
        frame_id = _zero_pad_frame_id(row.get("frame_id"), fallback=frame_index)
        pose_meta = meta_by_index.get(frame_index, {})
        if frame_id not in {"", "000000"} and pose_meta:
            pose_meta.setdefault("frame_id", frame_id)
        poses.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp": _safe_float(row.get("timestamp_seconds"), _safe_float(row.get("timestamp"))),
                "T_world_camera": row.get("world_from_camera") or row.get("T_world_camera"),
            }
        )
        meta_by_index.setdefault(frame_index, {})
        meta_by_index[frame_index].setdefault("frame_id", frame_id)

    frame_meta: Dict[str, Dict[str, Any]] = {}
    for frame_index, row in meta_by_index.items():
        frame_id = _zero_pad_frame_id(row.get("frame_id"), fallback=frame_index)
        row["frame_id"] = frame_id
        frame_meta[frame_id] = row

    coordinate_frame_session_id = str(
        descriptor.get("coordinate_frame_session_id")
        or context.capture_id
    )
    quality = descriptor.get("quality") if isinstance(descriptor.get("quality"), Mapping) else {}
    summary = geometry_summary if isinstance(geometry_summary, Mapping) else {}
    synthetic_geometry, fallback_used = reconcile_geometry_truth_flags(summary)
    return {
        "source": source or "video_to_world",
        "fallback_used": fallback_used,
        "fallback_kind": summary.get("fallback_kind"),
        "synthetic_geometry": synthetic_geometry,
        "provider_native_result": bool(summary.get("provider_native_result")),
        "contract_ready_for_world_model": bool(summary.get("contract_ready_for_world_model")),
        "internal_fallback_ready": bool(summary.get("internal_fallback_ready")),
        "geometry_live_ready": bool(summary.get("geometry_live_ready")),
        "external_market_ready": bool(summary.get("external_market_ready")),
        "site_faithful_market_ready": bool(summary.get("site_faithful_market_ready")),
        "poses": poses,
        "frame_meta": frame_meta,
        "intrinsics": dict(intrinsics) if isinstance(intrinsics, Mapping) else {},
        "coordinate_frame_session_id": coordinate_frame_session_id,
        "ready_for_world_model": bool(
            summary.get("ready_for_world_model")
            if summary
            else descriptor.get("geometry_ready")
            or quality.get("geometry_ready")
            or quality.get("world_model_candidate")
        ),
    }


def _load_arcore_geometry(
    *,
    context: LocalCaptureContext,
    descriptor: Mapping[str, Any],
) -> Dict[str, Any]:
    arcore_root = context.raw_root / "arcore"
    poses_raw = _load_jsonl(arcore_root / "poses.jsonl")
    frames_raw = _load_jsonl(arcore_root / "frames.jsonl")
    intrinsics = read_json(arcore_root / "session_intrinsics.json") if (arcore_root / "session_intrinsics.json").is_file() else {}

    frame_meta: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(frames_raw):
        frame_index = _safe_int(row.get("frame_index"), idx)
        frame_id = _zero_pad_frame_id(row.get("frame_id"), fallback=frame_index)
        frame_meta[frame_id] = {
            "frame_index": frame_index,
            "timestamp_seconds": _safe_float(row.get("t_capture_sec"), _safe_float(row.get("timestamp_seconds"))),
            "intrinsics_payload": dict(intrinsics) if isinstance(intrinsics, Mapping) else {},
            "trackingState": row.get("tracking_state", "unknown"),
            "sharpnessScore": None,
            "relocalizationEvent": False,
            "worldMappingStatus": None,
            "anchorObservations": [],
            "source_image_path": None,
            "depth_path": _manifest_relative_path(arcore_root / "depth_manifest.json", "depth_path", frame_id),
            "confidence_path": _manifest_relative_path(arcore_root / "confidence_manifest.json", "confidence_path", frame_id),
            "pose_confidence": 1.0 if str(row.get("tracking_state") or "").upper() == "TRACKING" else 0.0,
        }

    poses: List[Dict[str, Any]] = []
    for idx, row in enumerate(poses_raw):
        frame_index = _safe_int(row.get("frame_index"), idx)
        frame_id = _zero_pad_frame_id(row.get("frame_id"), fallback=frame_index)
        poses.append(
            {
                "frame_id": frame_id,
                "frame_index": frame_index,
                "timestamp": _safe_float(row.get("t_capture_sec"), _safe_float(row.get("timestamp_seconds"))),
                "T_world_camera": row.get("T_world_camera"),
            }
        )

    coordinate_frame_session_id = str(
        descriptor.get("coordinate_frame_session_id")
        or context.capture_id
    )
    quality = descriptor.get("quality") if isinstance(descriptor.get("quality"), Mapping) else {}
    return {
        "source": "arcore",
        "poses": poses,
        "frame_meta": frame_meta,
        "intrinsics": dict(intrinsics) if isinstance(intrinsics, Mapping) else {},
        "coordinate_frame_session_id": coordinate_frame_session_id,
        "ready_for_world_model": bool(
            descriptor.get("geometry_ready")
            or quality.get("geometry_ready")
            or quality.get("world_model_candidate")
        ),
    }


def _manifest_relative_path(manifest_path: Path, key: str, frame_id: str) -> Optional[str]:
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    rows = manifest.get("frames") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("frame_id") or "") == frame_id:
            raw_path = row.get(key)
            if raw_path:
                return str(raw_path)
    return None
