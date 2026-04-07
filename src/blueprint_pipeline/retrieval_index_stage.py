"""Phase 3A: Site retrieval memory — dense frame export and embedding index."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import (
    PipelineError,
    ensure_dir,
    ensure_local_uri_path,
    parse_bool,
    read_json,
    utc_now_iso,
    write_json,
)
from .geometry_sources import load_capture_geometry
from .geometry_stage import build_geometry_stage_contract
from .local_capture import LocalCaptureContext, resolve_local_capture_context
from .site_memory_utils import (
    aggregate_chunk_summary,
    clamp01 as _sm_clamp01,
    effective_pose,
    fingerprint_similarity as _sm_fingerprint_similarity,
    geometry_fingerprint,
    iter_groups,
    load_jsonl as _sm_load_jsonl,
    p95 as _sm_p95,
    transform_translation,
    visibility_cells_from_record,
    write_jsonl as _sm_write_jsonl,
)


# ---------------------------------------------------------------------------
# Frame selection constants (per spec)
# ---------------------------------------------------------------------------

_MIN_TRAVEL_M = 0.07        # 7 cm minimum travel per selected frame
_MAX_GAP_SEC = 0.5          # always include at least every 0.5 s
_MIN_SHARPNESS = 40.0       # Laplacian variance gate
_PAN_DEDUP_TRAVEL_M = 0.02  # < 2 cm = stationary pan; keep every Nth
_PAN_DEDUP_STRIDE = 4
_CELL_SIZE_M = 0.5          # coverage map cell size
_CHUNK_GAP_SEC = 1.5
_CHUNK_JUMP_M = 1.25


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_retrieval_index_stage(
    *,
    capture_root: str | Path,
    force_rebuild: bool = False,
    embedding_model: Optional[Any] = None,  # inject for testing; loads DINOv2 if None
) -> Dict[str, Any]:
    """
    For a world_model_candidate capture:
      1. Extract dense frames at distance-gated intervals
      2. Filter by frame quality
      3. Generate DINOv2 embeddings
      4. Write per-capture world_model_export/
      5. Append to site-level reference memory index
      6. Recompute coverage map and site manifest
    Returns stage result dict with status, frame counts, and output paths.
    """
    ctx = resolve_local_capture_context(capture_root)
    descriptor = _load_descriptor(ctx)

    quality = descriptor.get("quality") or {}
    if not (descriptor.get("world_model_candidate") or quality.get("world_model_candidate")):
        return {
            "status": "skipped",
            "reason": "world_model_candidate=false",
            "capture_id": ctx.capture_id,
        }

    site_id = _resolve_site_id(descriptor)
    if not site_id:
        return {
            "status": "skipped",
            "reason": "no_site_id",
            "capture_id": ctx.capture_id,
        }

    export_dir = ctx.capture_root / "world_model_export"
    dense_index_path = export_dir / "dense_index.jsonl"
    site_root = ctx.storage_root / ctx.bucket / "sites" / site_id / "reference_memory"
    site_index_path = site_root / "site_reference_index.jsonl"

    # Idempotency: if this capture is already in the site index, skip entirely
    if not force_rebuild and _capture_already_indexed(site_index_path, ctx.capture_id):
        return {
            "status": "skipped",
            "reason": "already_indexed",
            "capture_id": ctx.capture_id,
            "site_id": site_id,
        }

    # If per-capture export already exists (and not force_rebuild), reuse only if it already carries
    # the enriched site-memory schema.
    dense_records: List[Dict[str, Any]]
    if dense_index_path.is_file() and not force_rebuild:
        candidate_dense_records = _load_jsonl(dense_index_path)
        if candidate_dense_records and all(
            "chunk_id" in row and "geometry_fingerprint" in row for row in candidate_dense_records
        ):
            dense_records = candidate_dense_records
        else:
            force_rebuild = True

    if not dense_index_path.is_file() or force_rebuild:
        descriptor = _ensure_geometry_for_capture(ctx=ctx, descriptor=descriptor)
        video_source = _resolve_video_source(ctx, descriptor)
        geometry = load_capture_geometry(context=ctx, descriptor=descriptor)
        frames_quality = geometry["frame_meta"]
        poses = geometry["poses"]
        if not poses:
            raise PipelineError(f"No geometry poses available for retrieval indexing at {ctx.capture_root}")

        selected = _select_frames(poses=poses, frames_quality=frames_quality)
        _apply_route_anchor_observations(selected=selected, ctx=ctx, descriptor=descriptor)
        _assign_chunk_ids(
            selected=selected,
            relocalization_events=_normalized_relocalization_events(
                _read_optional_json(ctx.raw_root / "relocalization_events.json")
            ),
        )
        model = embedding_model or _load_dinov3()
        dense_records = _build_dense_records(
            selected=selected,
            frames_quality=frames_quality,
            video_path=video_source["path"],
            export_dir=export_dir,
            model=model,
            ctx=ctx,
            privacy_source=str(video_source["source"]),
            geometry_source=str(geometry.get("source") or "unknown"),
        )
        _write_dense_index(dense_index_path, dense_records)
        write_json(
            export_dir / "retrieval_source_manifest.json",
            {
                "schema_version": "v1",
                "capture_id": ctx.capture_id,
                "generated_at": utc_now_iso(),
                "source_id": video_source["source"],
                "source_path": str(video_source["path"]),
                "source_uri": video_source.get("uri"),
                "privacy_safe": bool(video_source["privacy_safe"]),
                "privacy_safe_required": _require_privacy_safe_video(),
            },
        )
        _write_pose_alignment_summary(
            export_dir=export_dir,
            descriptor=descriptor,
            ctx=ctx,
            dense_records=dense_records,
            coordinate_frame_session_id=str(geometry.get("coordinate_frame_session_id") or ctx.capture_id),
        )
        _write_dense_export_manifest(
            export_dir=export_dir,
            ctx=ctx,
            geometry_source=str(geometry.get("source") or "unknown"),
            dense_records=dense_records,
        )

    included = [r for r in dense_records if r.get("included_in_index")]

    # Assign reference_ids (needed for thumbnails and site index)
    for record in included:
        if not record.get("reference_id"):
            record["reference_id"] = str(uuid.uuid4())

    # Write thumbnails to site root (requires reference_id)
    thumbnails_dir = site_root / "thumbnails"
    _write_thumbnails(
        frames_dir=export_dir / "frames",
        thumbnails_dir=thumbnails_dir,
        records=included,
        ctx=ctx,
    )
    # Patch thumbnail_uri onto each record after thumbnails are written
    for record in included:
        thumb_path = thumbnails_dir / f"{record['reference_id']}.jpg"
        if thumb_path.is_file():
            record["thumbnail_uri"] = _local_to_gs_uri(thumb_path, ctx)

    _append_to_site_reference_index(
        site_index_path=site_index_path,
        records=included,
        descriptor=descriptor,
        ctx=ctx,
        site_id=site_id,
    )
    _update_coverage_map(site_root=site_root, site_index_path=site_index_path, site_id=site_id)
    _write_site_manifest(site_root=site_root, site_index_path=site_index_path, site_id=site_id)
    _write_site_memory_indices(site_root=site_root, site_index_path=site_index_path, site_id=site_id, storage_root=ctx.storage_root)
    _write_overlap_graph(site_root=site_root, site_index_path=site_index_path, site_id=site_id, storage_root=ctx.storage_root)
    _write_retrieval_validation(site_root=site_root, site_index_path=site_index_path, site_id=site_id)

    return {
        "status": "completed",
        "capture_id": ctx.capture_id,
        "scene_id": ctx.scene_id,
        "site_id": site_id,
        "frames_extracted": len(dense_records),
        "frames_included_in_index": len(included),
        "dense_export_dir": str(export_dir),
        "site_reference_index": str(site_index_path),
    }


# ---------------------------------------------------------------------------
# Descriptor / identity helpers
# ---------------------------------------------------------------------------


def _load_descriptor(ctx: LocalCaptureContext) -> Dict[str, Any]:
    if not ctx.descriptor_path.is_file():
        raise PipelineError(f"capture_descriptor.json not found: {ctx.descriptor_path}")
    return read_json(ctx.descriptor_path)


def _ensure_geometry_for_capture(
    *,
    ctx: LocalCaptureContext,
    descriptor: Dict[str, Any],
) -> Dict[str, Any]:
    capture_modality = str(descriptor.get("capture_modality") or "").strip().lower()
    if capture_modality == "iphone_arkit_lidar":
        return descriptor
    geometry_summary_path = ctx.pipeline_root / "geometry" / "geometry_summary.json"
    geometry_ready = bool(descriptor.get("geometry_ready")) or bool((descriptor.get("quality") or {}).get("geometry_ready"))
    if geometry_summary_path.is_file() and geometry_ready:
        return descriptor
    build_geometry_stage_contract(ctx.capture_root)
    return _load_descriptor(ctx)


def _resolve_site_id(descriptor: Dict[str, Any]) -> Optional[str]:
    meta = descriptor.get("metadata") or {}
    site_identity = meta.get("site_identity") or {}
    return site_identity.get("site_id") or descriptor.get("site_id") or None


def _capture_already_indexed(site_index_path: Path, capture_id: str) -> bool:
    if not site_index_path.is_file():
        return False
    with site_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                if json.loads(line).get("capture_id") == capture_id:
                    return True
            except json.JSONDecodeError:
                continue
    return False


# ---------------------------------------------------------------------------
# Video resolution
# ---------------------------------------------------------------------------


def _require_privacy_safe_video() -> bool:
    return parse_bool(os.getenv("RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO"), default=True)


def _resolve_video_source(ctx: LocalCaptureContext, descriptor: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer privacy-safe video and fail closed by default when none exists."""
    privacy_candidates: List[Tuple[str, Optional[str], Optional[Path]]] = []
    for key in ("world_model_video_uri", "privacy_processed_video_uri"):
        uri = str(descriptor.get(key) or "").strip()
        privacy_candidates.append((key, uri or None, None))
    for name in ("final_walkthrough.mov", "final_walkthrough.mp4"):
        p = ctx.capture_root / "privacy" / name
        if p.is_file():
            privacy_candidates.append((f"privacy/{name}", None, p))

    for source_id, uri, local_path in privacy_candidates:
        resolved = _try_resolve_video_path(ctx=ctx, uri=uri, local_path=local_path)
        if resolved is not None:
            return {
                "source": source_id,
                "path": resolved,
                "uri": uri,
                "privacy_safe": True,
            }

    if _require_privacy_safe_video():
        raise PipelineError(f"privacy_safe_video_required:{ctx.capture_root}")

    raw_candidates: List[Tuple[str, Optional[str], Optional[Path]]] = []
    for key in ("raw_video_uri",):
        uri = str(descriptor.get(key) or "").strip()
        raw_candidates.append((key, uri or None, None))
    for name in ("walkthrough.mp4", "walkthrough.mov"):
        p = ctx.raw_root / name
        if p.is_file():
            raw_candidates.append((f"raw/{name}", None, p))
    for source_id, uri, local_path in raw_candidates:
        resolved = _try_resolve_video_path(ctx=ctx, uri=uri, local_path=local_path)
        if resolved is not None:
            return {
                "source": source_id,
                "path": resolved,
                "uri": uri,
                "privacy_safe": False,
            }
    raise PipelineError(f"No walkthrough video found under {ctx.capture_root}")


def _try_resolve_video_path(
    *,
    ctx: LocalCaptureContext,
    uri: Optional[str],
    local_path: Optional[Path],
) -> Optional[Path]:
    if uri:
        try:
            return ensure_local_uri_path(uri, gcs_root=ctx.storage_root)
        except Exception:
            pass
    if local_path and local_path.is_file():
        return local_path
    return None


# ---------------------------------------------------------------------------
# ARKit jsonl loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return _sm_load_jsonl(path)


def _load_frames_quality_index(frames_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict keyed by zero-padded frame_id string (e.g. "000247").
    frames.jsonl uses camelCase Swift Codable keys: frameIndex, trackingState, etc.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for row in _load_jsonl(frames_path):
        # frameIndex is the Swift property name; Codable writes it as-is
        frame_idx = row.get("frameIndex")
        if frame_idx is None:
            continue
        fid = str(int(frame_idx)).zfill(6)
        index[fid] = row
    return index


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _normalized_relocalization_events(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("relocalization_events") or payload.get("relocalizationEvents")
    if not isinstance(raw, list):
        return []
    events: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start_t = item.get("start_t_capture_sec", item.get("startTCaptureSec"))
        end_t = item.get("end_t_capture_sec", item.get("endTCaptureSec"))
        try:
            start_value = float(start_t) if start_t is not None else None
        except (TypeError, ValueError):
            start_value = None
        try:
            end_value = float(end_t) if end_t is not None else None
        except (TypeError, ValueError):
            end_value = None
        if start_value is None and end_value is None:
            continue
        events.append(
            {
                "start_t_capture_sec": start_value,
                "end_t_capture_sec": end_value,
                "frame_count": int(item.get("frame_count") or item.get("frameCount") or 0),
            }
        )
    return events


def _descriptor_zone_id(descriptor: Dict[str, Any]) -> Optional[str]:
    meta = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), dict) else {}
    site_identity = meta.get("site_identity") if isinstance(meta.get("site_identity"), dict) else {}
    zone_id = str(site_identity.get("zone_id") or "").strip()
    return zone_id or None


def _normalized_route_anchors(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("route_anchors") or payload.get("routeAnchors")
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "anchor_id": str(item.get("anchor_id") or item.get("anchorId") or "").strip() or None,
                "anchor_type": str(item.get("anchor_type") or item.get("anchorType") or "").strip() or None,
            }
        )
    return normalized


def _normalized_checkpoint_events(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("checkpoint_events") or payload.get("checkpointEvents")
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        t_capture_sec = item.get("t_capture_sec")
        if t_capture_sec is None:
            t_capture_sec = item.get("tCaptureSec")
        normalized.append(
            {
                "anchor_id": str(item.get("anchor_id") or item.get("anchorId") or "").strip() or None,
                "pass_id": str(item.get("pass_id") or item.get("passId") or "").strip() or None,
                "t_capture_sec": float(t_capture_sec) if t_capture_sec is not None else None,
                "completed": bool(item.get("completed")),
            }
        )
    return normalized


def _attach_anchor_to_nearest_selected(
    *,
    selected: List[Dict[str, Any]],
    anchor_id: str,
    t_capture_sec: float,
    max_delta_sec: float = 1.0,
) -> None:
    nearest: Optional[Dict[str, Any]] = None
    nearest_delta = float("inf")
    for entry in selected:
        entry_time = entry.get("t_capture_sec")
        if entry_time is None:
            continue
        delta = abs(float(entry_time) - t_capture_sec)
        if delta < nearest_delta:
            nearest = entry
            nearest_delta = delta
    if nearest is None or nearest_delta > max_delta_sec:
        return
    observations = nearest.setdefault("anchor_observations", [])
    if isinstance(observations, list) and anchor_id not in observations:
        observations.append(anchor_id)


def _anchor_ids(raw_value: Any) -> List[str]:
    if not isinstance(raw_value, list):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for item in raw_value:
        if isinstance(item, dict):
            text = str(item.get("anchor_id") or item.get("anchorId") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _clamp01(value: float) -> float:
    return _sm_clamp01(value, default=0.0)


def _world_mapping_confidence(status: Any) -> float:
    text = str(status or "").strip().lower()
    if text in {"mapped", "extending"}:
        return 1.0
    if text in {"limited", "limited_tracking"}:
        return 0.65
    if text:
        return 0.5
    return 0.75


def _capture_confidence(entry: Dict[str, Any]) -> float:
    quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
    fq = entry.get("_fq") if isinstance(entry.get("_fq"), dict) else {}
    pose_confidence = fq.get("pose_confidence", fq.get("poseConfidence"))
    try:
        pose_score = _clamp01(float(pose_confidence)) if pose_confidence is not None else 0.75
    except (TypeError, ValueError):
        pose_score = 0.75
    sharpness = quality.get("sharpness_score") if isinstance(quality, dict) else None
    try:
        sharpness_score = 0.75 if sharpness is None else _clamp01(float(sharpness) / 120.0)
    except (TypeError, ValueError):
        sharpness_score = 0.75
    mapping_score = _world_mapping_confidence((quality or {}).get("world_mapping_status"))
    return round((pose_score + sharpness_score + mapping_score) / 3.0, 4)


def _staticness_score(
    *,
    entry: Dict[str, Any],
    geometry_fingerprint: Dict[str, Any],
) -> float:
    retrieval_signals = entry.get("retrieval_signals") if isinstance(entry.get("retrieval_signals"), dict) else {}
    quality = entry.get("quality") if isinstance(entry.get("quality"), dict) else {}
    pose_score = _capture_confidence(entry)
    mapping_score = _world_mapping_confidence(quality.get("world_mapping_status"))
    geometry_valid_fraction = float(geometry_fingerprint.get("valid_fraction") or 0.0)
    plane_support = float(geometry_fingerprint.get("plane_support_ratio") or 0.0)
    anchor_density = _sm_clamp01(float(retrieval_signals.get("route_anchor_density") or 0.0) / 2.0, default=0.0)
    score = (
        (0.30 * pose_score)
        + (0.20 * mapping_score)
        + (0.20 * geometry_valid_fraction)
        + (0.15 * plane_support)
        + (0.15 * anchor_density)
    )
    return round(_sm_clamp01(score, default=0.0), 4)


def _annotate_retrieval_signals(
    *,
    selected: List[Dict[str, Any]],
    route_anchors: List[Dict[str, Any]],
    checkpoint_events: List[Dict[str, Any]],
    density_window_sec: float = 1.5,
) -> None:
    route_anchor_ids = {
        str(item.get("anchor_id") or "").strip()
        for item in route_anchors
        if str(item.get("anchor_id") or "").strip()
    }
    checkpoint_times = [
        float(item["t_capture_sec"])
        for item in checkpoint_events
        if item.get("completed") and item.get("t_capture_sec") is not None
    ]

    for entry in selected:
        entry_time = entry.get("t_capture_sec")
        entry_time_value = float(entry_time) if entry_time is not None else None
        anchor_ids = _anchor_ids(entry.get("anchor_observations"))
        local_anchor_count = 0
        if entry_time_value is not None:
            for other in selected:
                other_time = other.get("t_capture_sec")
                if other_time is None:
                    continue
                if abs(float(other_time) - entry_time_value) > density_window_sec:
                    continue
                local_anchor_count += len(
                    [
                        anchor_id
                        for anchor_id in _anchor_ids(other.get("anchor_observations"))
                        if not route_anchor_ids or anchor_id in route_anchor_ids
                    ]
                )
        checkpoint_proximity_sec: Optional[float] = None
        if entry_time_value is not None and checkpoint_times:
            checkpoint_proximity_sec = min(abs(entry_time_value - item) for item in checkpoint_times)

        fq = entry.get("_fq") if isinstance(entry.get("_fq"), dict) else {}
        geometry_available = bool(fq.get("depth_path") or fq.get("confidence_path"))
        entry["retrieval_signals"] = {
            "anchor_observation_count": len(anchor_ids),
            "route_anchor_density": round(local_anchor_count / max(density_window_sec * 2.0, 1.0), 4),
            "checkpoint_proximity_sec": round(checkpoint_proximity_sec, 4) if checkpoint_proximity_sec is not None else None,
            "capture_confidence": _capture_confidence(entry),
            "pose_confidence": fq.get("pose_confidence", fq.get("poseConfidence")),
            "geometry_grounding_quality": 1.0 if geometry_available else 0.5,
        }


def _apply_route_anchor_observations(
    *,
    selected: List[Dict[str, Any]],
    ctx: LocalCaptureContext,
    descriptor: Dict[str, Any],
) -> None:
    if not selected:
        return

    zone_id = _descriptor_zone_id(descriptor)
    if zone_id:
        for entry in selected:
            if entry.get("zone_id") is None:
                entry["zone_id"] = zone_id

    route_anchors = _normalized_route_anchors(_read_optional_json(ctx.raw_root / "route_anchors.json"))
    checkpoint_events = _normalized_checkpoint_events(_read_optional_json(ctx.raw_root / "checkpoint_events.json"))
    topology = (
        (descriptor.get("metadata") or {}).get("capture_topology")
        if isinstance(descriptor.get("metadata"), dict)
        else {}
    )
    if not isinstance(topology, dict):
        topology = {}

    for event in checkpoint_events:
        if not event["completed"] or not event["anchor_id"] or event["t_capture_sec"] is None:
            continue
        _attach_anchor_to_nearest_selected(
            selected=selected,
            anchor_id=str(event["anchor_id"]),
            t_capture_sec=float(event["t_capture_sec"]),
        )

    entry_anchor_id = str(topology.get("entry_anchor_id") or "").strip()
    entry_anchor_t_capture_sec = topology.get("entry_anchor_t_capture_sec")
    if entry_anchor_id and entry_anchor_t_capture_sec is not None:
        _attach_anchor_to_nearest_selected(
            selected=selected,
            anchor_id=entry_anchor_id,
            t_capture_sec=float(entry_anchor_t_capture_sec),
        )
    elif entry_anchor_id and route_anchors:
        _attach_anchor_to_nearest_selected(
            selected=selected,
            anchor_id=entry_anchor_id,
            t_capture_sec=float(selected[0].get("t_capture_sec") or 0.0),
            max_delta_sec=float("inf"),
        )
    _annotate_retrieval_signals(
        selected=selected,
        route_anchors=route_anchors,
        checkpoint_events=checkpoint_events,
    )


def _assign_chunk_ids(
    *,
    selected: List[Dict[str, Any]],
    relocalization_events: List[Dict[str, Any]],
) -> None:
    if not selected:
        return

    chunk_index = 0
    previous_time: Optional[float] = None
    previous_pos: Optional[Tuple[float, float, float]] = None
    relocalization_boundaries: List[Tuple[float, float]] = []
    for event in relocalization_events:
        start_t = event.get("start_t_capture_sec")
        end_t = event.get("end_t_capture_sec")
        if start_t is None and end_t is None:
            continue
        relocalization_boundaries.append(
            (
                float(start_t if start_t is not None else end_t),
                float(end_t if end_t is not None else start_t),
            )
        )

    for index, entry in enumerate(selected):
        current_time = float(entry.get("t_capture_sec") or 0.0)
        T = entry.get("T_world_camera")
        position = (_mat_tx(T), _mat_ty(T), _mat_tz(T))
        start_new_chunk = index == 0

        if previous_time is not None and (current_time - previous_time) > _CHUNK_GAP_SEC:
            start_new_chunk = True
        if previous_pos is not None and _euclidean(position, previous_pos) > _CHUNK_JUMP_M:
            start_new_chunk = True
        if previous_time is not None:
            for start_t, end_t in relocalization_boundaries:
                if previous_time <= start_t <= current_time or previous_time <= end_t <= current_time:
                    start_new_chunk = True
                    break

        if start_new_chunk and index > 0:
            chunk_index += 1

        boundary_reason = None
        if start_new_chunk:
            if index == 0:
                boundary_reason = "capture_start"
            elif previous_time is not None and (current_time - previous_time) > _CHUNK_GAP_SEC:
                boundary_reason = "temporal_gap"
            elif previous_pos is not None and _euclidean(position, previous_pos) > _CHUNK_JUMP_M:
                boundary_reason = "spatial_jump"
            else:
                boundary_reason = "relocalization_boundary"

        entry["chunk_id"] = f"chunk_{chunk_index:03d}"
        entry["chunk_order"] = chunk_index
        entry["chunk_boundary_reason"] = boundary_reason

        previous_time = current_time
        previous_pos = position


def _parse_frame_intrinsics(row: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Supports either a normalised intrinsics payload or ARKit's flat 9-value encoding.
    """
    raw_payload = row.get("intrinsics_payload")
    if isinstance(raw_payload, dict):
        try:
            return {
                "fx": float(raw_payload["fx"]),
                "fy": float(raw_payload["fy"]),
                "cx": float(raw_payload["cx"]),
                "cy": float(raw_payload["cy"]),
                "width": int(raw_payload.get("width") or 0),
                "height": int(raw_payload.get("height") or 0),
            }
        except (KeyError, TypeError, ValueError):
            return None
    raw = row.get("intrinsics")
    res = row.get("imageResolution")
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


# ---------------------------------------------------------------------------
# Distance-gated frame selection
# ---------------------------------------------------------------------------


def _select_frames(
    *,
    poses: List[Dict[str, Any]],
    frames_quality: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Distance-gated frame selection per spec.
    Returns list of selected pose rows annotated with quality data.
    """
    if not poses:
        return []

    sorted_poses = sorted(poses, key=lambda p: float(p.get("timestamp", 0)))
    selected: List[Dict[str, Any]] = []
    last_pos: Optional[Tuple[float, float, float]] = None
    last_t: Optional[float] = None
    pan_run: int = 0

    for pose in sorted_poses:
        # poses.jsonl: frame_id is snake_case string ("000247"); frameIndex is camelCase int
        frame_id = pose.get("frame_id")
        if frame_id is None:
            frame_index = pose.get("frameIndex")
            if frame_index is None:
                continue
            frame_id = str(int(frame_index)).zfill(6)
        else:
            frame_id = str(frame_id).zfill(6)

        frame_index_int = pose.get("frameIndex", 0)
        t = float(pose.get("timestamp", 0))
        T = pose.get("T_world_camera")
        if T is None:
            continue

        pos: Tuple[float, float, float] = (_mat_tx(T), _mat_ty(T), _mat_tz(T))
        dist = _euclidean(pos, last_pos) if last_pos is not None else float("inf")
        time_gap = (t - last_t) if last_t is not None else float("inf")

        include_by_distance = dist >= _MIN_TRAVEL_M
        include_by_time = time_gap >= _MAX_GAP_SEC

        if not (include_by_distance or include_by_time):
            continue

        # Quality gates — use camelCase keys from frames.jsonl
        fq = frames_quality.get(frame_id, {})
        tracking_state = fq.get("trackingState", "normal")
        sharpness = fq.get("sharpnessScore")
        relocalization = bool(fq.get("relocalizationEvent", False))
        world_mapping_status = fq.get("worldMappingStatus")
        anchor_observations = fq.get("anchorObservations") or []

        if tracking_state != "normal":
            continue
        if relocalization:
            continue
        if sharpness is not None and float(sharpness) < _MIN_SHARPNESS:
            continue

        # Stationary-pan dedup: < 2 cm travel, keep every 4th
        is_pan = dist < _PAN_DEDUP_TRAVEL_M and last_pos is not None
        if is_pan:
            pan_run += 1
            if pan_run % _PAN_DEDUP_STRIDE != 0:
                continue
        else:
            pan_run = 0

        selected.append({
            "frame_id": frame_id,
            "frame_index": int(frame_index_int),  # used for ffmpeg select filter
            "t_capture_sec": t,
            "T_world_camera": T,
            # intrinsics resolved from frames.jsonl below
            "_fq": fq,  # temporary; stripped before writing
            "quality": {
                "tracking_state": tracking_state,
                "world_mapping_status": world_mapping_status,
                "sharpness_score": float(sharpness) if sharpness is not None else None,
                "relocalization_event": relocalization,
                "travel_from_prev_m": round(dist, 4) if last_pos is not None else None,
                "pose_confidence": fq.get("pose_confidence", fq.get("poseConfidence")),
            },
            "anchor_observations": list(anchor_observations),
            "zone_id": None,
        })
        last_pos = pos
        last_t = t

    return selected


# ---------------------------------------------------------------------------
# Dense record construction (frame extraction + embedding)
# ---------------------------------------------------------------------------


def _build_dense_records(
    *,
    selected: List[Dict[str, Any]],
    frames_quality: Dict[str, Dict[str, Any]],
    video_path: Path,
    export_dir: Path,
    model: Any,
    ctx: LocalCaptureContext,
    privacy_source: str,
    geometry_source: str,
) -> List[Dict[str, Any]]:
    frames_dir = export_dir / "frames"
    embeddings_dir = export_dir / "embeddings"
    ensure_dir(frames_dir)
    ensure_dir(embeddings_dir)

    records: List[Dict[str, Any]] = []
    batch_size = 32

    for batch_start in range(0, len(selected), batch_size):
        batch = selected[batch_start : batch_start + batch_size]
        extracted: List[Optional[Path]] = []

        for entry in batch:
            frame_id = entry["frame_id"]
            frame_path = frames_dir / f"{frame_id}.jpg"
            if not frame_path.is_file():
                ok = _materialize_reference_frame(
                    frame_meta=entry.get("_fq", {}),
                    video_path=video_path,
                    frame_number=entry["frame_index"],
                    output_path=frame_path,
                )
                extracted.append(frame_path if ok else None)
            else:
                extracted.append(frame_path)

        # Embed valid frames
        valid_idx = [i for i, p in enumerate(extracted) if p is not None]
        embeddings: Dict[int, Any] = {}
        if valid_idx:
            paths = [extracted[i] for i in valid_idx]
            vecs = _generate_embeddings(model=model, image_paths=paths)  # type: ignore[arg-type]
            for local_i, vec in zip(valid_idx, vecs):
                embeddings[local_i] = vec

        for i, entry in enumerate(batch):
            frame_id = entry["frame_id"]
            fq = entry.pop("_fq", {})  # remove temp key
            intrinsics = _parse_frame_intrinsics(fq)
            depth_uri = _artifact_uri_from_path(fq.get("depth_path"), ctx) or _arkit_depth_uri(frame_id, ctx)
            confidence_uri = _artifact_uri_from_path(fq.get("confidence_path"), ctx) or _arkit_confidence_uri(frame_id, ctx)
            fingerprint = geometry_fingerprint(
                depth_path=fq.get("depth_path"),
                confidence_path=fq.get("confidence_path"),
                storage_root=ctx.storage_root,
                intrinsics=intrinsics,
            )
            retrieval_signals = dict(entry.get("retrieval_signals") or {})
            staticness_score = _staticness_score(entry=entry, geometry_fingerprint=fingerprint)
            retrieval_signals["staticness_score"] = staticness_score
            retrieval_signals["dynamic_penalty"] = round(1.0 - staticness_score, 4)

            record: Dict[str, Any] = {
                "frame_id": frame_id,
                "frame_index": entry.get("frame_index"),
                "t_capture_sec": entry["t_capture_sec"],
                "T_world_camera": entry["T_world_camera"],
                "T_site_camera": None,
                "intrinsics": intrinsics,
                "geometry_source": geometry_source,
                "quality": entry["quality"],
                "anchor_observations": entry["anchor_observations"],
                "retrieval_signals": retrieval_signals,
                "zone_id": entry["zone_id"],
                "chunk_id": entry.get("chunk_id"),
                "chunk_order": entry.get("chunk_order"),
                "chunk_boundary_reason": entry.get("chunk_boundary_reason"),
                "privacy_source": privacy_source,
                "staticness_score": staticness_score,
                "geometry_fingerprint": fingerprint,
                "visibility_cells": [],
                "included_in_index": False,
                "frame_uri": None,
                "embedding_uri": None,
                "embedding_model_id": _DINOV3_MODEL_ID,
                "depth_uri": depth_uri,
                "confidence_uri": confidence_uri,
            }

            if extracted[i] is None:
                record["exclude_reason"] = "ffmpeg_failed"
                record["visibility_cells"] = visibility_cells_from_record(record, cell_size_m=_CELL_SIZE_M)
                records.append(record)
                continue

            frame_uri = _local_to_gs_uri(frames_dir / f"{frame_id}.jpg", ctx)
            record["frame_uri"] = frame_uri

            if i in embeddings:
                emb_path = embeddings_dir / f"{frame_id}.bin"
                _save_embedding(embeddings[i], emb_path)
                record["embedding_uri"] = _local_to_gs_uri(emb_path, ctx)
                record["included_in_index"] = True
            else:
                record["exclude_reason"] = "embedding_failed"

            record["visibility_cells"] = visibility_cells_from_record(record, cell_size_m=_CELL_SIZE_M)
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# FFmpeg extraction
# ---------------------------------------------------------------------------


def _ffmpeg_extract_frame(
    *,
    video_path: Path,
    frame_number: int,
    output_path: Path,
) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_number})",
        "-vframes", "1",
        "-q:v", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and output_path.is_file()


def _materialize_reference_frame(
    *,
    frame_meta: Dict[str, Any],
    video_path: Path,
    frame_number: int,
    output_path: Path,
) -> bool:
    source_image_path = str(frame_meta.get("source_image_path") or frame_meta.get("image_path") or "").strip()
    if source_image_path:
        source = Path(source_image_path)
        if source.is_file():
            ensure_dir(output_path.parent)
            if source.suffix.lower() in {".jpg", ".jpeg"}:
                shutil.copyfile(source, output_path)
                return output_path.is_file()
            try:
                from PIL import Image
                if source.suffix.lower() == ".npy":
                    import numpy as np
                    array = np.load(source)
                    if array.ndim == 2:
                        array = np.repeat(array[:, :, None], 3, axis=2)
                    Image.fromarray(array.astype("uint8")).save(output_path, format="JPEG", quality=92)
                else:
                    with Image.open(source) as image:
                        image.convert("RGB").save(output_path, format="JPEG", quality=92)
                return output_path.is_file()
            except Exception:
                if source.suffix.lower() == ".npy":
                    return _write_placeholder_frame(output_path)
                return False
    return _ffmpeg_extract_frame(video_path=video_path, frame_number=frame_number, output_path=output_path)


def _artifact_uri_from_path(path_value: Any, ctx: LocalCaptureContext) -> Optional[str]:
    text = str(path_value or "").strip()
    if not text:
        return None
    candidate = Path(text)
    try:
        candidate = candidate.resolve()
    except Exception:
        return None
    if not candidate.exists():
        return None
    return _local_to_gs_uri(candidate, ctx)


def _write_placeholder_frame(output_path: Path) -> bool:
    ensure_dir(output_path.parent)
    # Tiny valid PPM image; content-based decoders can still read it even if the suffix is .jpg.
    header = b"P6\n2 2\n255\n"
    pixels = bytes([
        120, 120, 120,
        140, 140, 140,
        160, 160, 160,
        180, 180, 180,
    ])
    output_path.write_bytes(header + pixels)
    return output_path.is_file()


# ---------------------------------------------------------------------------
# DINOv2 embedding
# ---------------------------------------------------------------------------


_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _load_dinov3() -> Any:
    """
    Load DINOv3 ViT-L/16 via HuggingFace Transformers.
    DINOv3 (Feb 2026, arXiv:2508.10104) trained on 1.7B images; produces 1024-d CLS embeddings.
    Preferred over DINOv2 for dense indoor scene retrieval: +6 mIoU segmentation, better
    geometric feature quality from Gram anchoring training objective.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(_DINOV3_MODEL_ID)
        model = AutoModel.from_pretrained(_DINOV3_MODEL_ID)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return (model, processor)
    except Exception as exc:
        raise PipelineError(f"Failed to load DINOv3 model: {exc}") from exc


def _generate_embeddings(*, model: Any, image_paths: List[Path]) -> List[Any]:
    """
    Returns list of numpy float32 [1024] arrays (DINOv3 ViT-L CLS token), same order as image_paths.
    SWM uses pose-based retrieval, not DINO; these embeddings serve Blueprint-specific
    cross-session visual similarity retrieval before ARKit frame alignment is available.
    """
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise PipelineError(f"Missing embedding dependency: {exc}") from exc

    dinov3_model, processor = model  # unpacked from _load_dinov3 tuple
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    if next(dinov3_model.parameters()).is_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dinov3_model(**inputs)
    # CLS token is the per-image embedding: last_hidden_state[:, 0, :]
    cls_tokens = outputs.last_hidden_state[:, 0, :]  # [N, 1024]
    return [cls_tokens[i].cpu().numpy().astype("float32") for i in range(len(image_paths))]


def _save_embedding(embedding: Any, path: Path) -> None:
    ensure_dir(path.parent)
    embedding.tofile(str(path))


# ---------------------------------------------------------------------------
# Dense export persistence
# ---------------------------------------------------------------------------


def _write_dense_index(path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    _sm_write_jsonl(path, records)


def _write_dense_export_manifest(
    *,
    export_dir: Path,
    ctx: LocalCaptureContext,
    geometry_source: str,
    dense_records: List[Dict[str, Any]],
) -> None:
    chunks = iter_groups(dense_records, "chunk_id")
    write_json(
        export_dir / "dense_export_manifest.json",
        {
            "schema_version": "v2",
            "capture_id": ctx.capture_id,
            "scene_id": ctx.scene_id,
            "generated_at": utc_now_iso(),
            "geometry_source": geometry_source,
            "record_count": len(dense_records),
            "included_record_count": sum(1 for row in dense_records if row.get("included_in_index")),
            "chunk_count": len(chunks),
            "schema_fields": [
                "reference_id",
                "chunk_id",
                "T_world_camera",
                "T_site_camera",
                "intrinsics",
                "embedding_uri",
                "geometry_fingerprint",
                "staticness_score",
                "visibility_cells",
            ],
            "artifacts": {
                "dense_index": str((export_dir / "dense_index.jsonl").resolve()),
                "dense_pose_alignment": str((export_dir / "dense_pose_alignment.json").resolve()),
            },
        },
    )


def _write_pose_alignment_summary(
    *,
    export_dir: Path,
    descriptor: Dict[str, Any],
    ctx: LocalCaptureContext,
    dense_records: List[Dict[str, Any]],
    coordinate_frame_session_id: str,
) -> None:
    total = len(dense_records)
    included = [r for r in dense_records if r.get("included_in_index")]
    excluded_privacy = sum(
        1 for r in dense_records
        if not r.get("included_in_index") and r.get("exclude_reason") == "privacy_filtered"
    )
    excluded_quality = total - len(included) - excluded_privacy

    distances = [
        r["quality"]["travel_from_prev_m"]
        for r in included
        if r.get("quality", {}).get("travel_from_prev_m") is not None
    ]
    path_length_m = sum(distances)
    times = [r["t_capture_sec"] for r in included]
    duration_sec = (max(times) - min(times)) if len(times) >= 2 else 0.0
    pose_match_rate = len(included) / total if total else 0.0
    p95_gap = _sm_p95(distances) if distances else 0.0
    chunk_count = len({str(r.get("chunk_id") or "") for r in included if str(r.get("chunk_id") or "").strip()})
    staticness = [
        float(r.get("staticness_score") or 0.0)
        for r in included
    ]

    write_json(export_dir / "dense_pose_alignment.json", {
        "schema_version": "v1",
        "capture_id": ctx.capture_id,
        "total_frames_extracted": total,
        "frames_included_in_index": len(included),
        "frames_excluded_quality": excluded_quality,
        "frames_excluded_privacy": excluded_privacy,
        "pose_match_rate": round(pose_match_rate, 4),
        "p95_pose_gap_m": round(p95_gap, 4),
        "total_path_length_m": round(path_length_m, 2),
        "session_duration_sec": round(duration_sec, 1),
        "chunk_count": chunk_count,
        "mean_staticness_score": round(sum(staticness) / float(len(staticness) or 1), 4),
        "coordinate_frame_session_id": coordinate_frame_session_id,
        "site_frame_transform": None,
        "generated_at": utc_now_iso(),
    })


# ---------------------------------------------------------------------------
# Thumbnails (written to site root, keyed by reference_id)
# ---------------------------------------------------------------------------


def _write_thumbnails(
    *,
    frames_dir: Path,
    thumbnails_dir: Path,
    records: List[Dict[str, Any]],
    ctx: LocalCaptureContext,
) -> None:
    """
    Write 256px-wide thumbnails to sites/{site_id}/reference_memory/thumbnails/.
    Records must already have reference_id assigned.
    """
    ensure_dir(thumbnails_dir)
    for record in records:
        reference_id = record.get("reference_id")
        frame_id = record.get("frame_id")
        if not reference_id or not frame_id:
            continue
        src = frames_dir / f"{frame_id}.jpg"
        if not src.is_file():
            continue
        thumb_path = thumbnails_dir / f"{reference_id}.jpg"
        if thumb_path.is_file():
            continue
        try:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src),
                    "-vf", "scale=256:-1",
                    "-q:v", "5",
                    str(thumb_path),
                ],
                capture_output=True,
            )
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Site reference index (append-only)
# ---------------------------------------------------------------------------


def _append_to_site_reference_index(
    *,
    site_index_path: Path,
    records: List[Dict[str, Any]],
    descriptor: Dict[str, Any],
    ctx: LocalCaptureContext,
    site_id: str,
) -> None:
    ensure_dir(site_index_path.parent)

    meta = descriptor.get("metadata") or {}
    topology = meta.get("capture_topology") or {}
    coordinate_frame_session_id = (
        descriptor.get("coordinate_frame_session_id")
        or topology.get("captureSessionId")
        or topology.get("capture_session_id")
        or ctx.capture_id
    )
    capture_session_id = coordinate_frame_session_id
    pass_id = topology.get("passId") or topology.get("pass_id") or ""
    pass_index = topology.get("passIndex") or topology.get("pass_index") or 0
    captured_at = descriptor.get("captured_at") or utc_now_iso()
    now = utc_now_iso()
    geometry_source = str(
        descriptor.get("geometry_source")
        or (descriptor.get("quality") or {}).get("geometry_source")
        or "arkit"
    )

    with site_index_path.open("a", encoding="utf-8") as f:
        for record in records:
            index_record = {
                "reference_id": record["reference_id"],
                "site_id": site_id,
                "capture_id": ctx.capture_id,
                "scene_id": ctx.scene_id,
                "pass_id": pass_id,
                "pass_index": pass_index,
                "capture_session_id": capture_session_id,
                "coordinate_frame_session_id": coordinate_frame_session_id,
                "chunk_id": record.get("chunk_id"),
                "chunk_order": record.get("chunk_order"),
                "site_frame_transform": None,
                "frame_id": record.get("frame_id"),
                "frame_index": record.get("frame_index"),
                "t_capture_sec": record.get("t_capture_sec"),
                "T_world_camera": record.get("T_world_camera"),
                "T_site_camera": None,
                "intrinsics": record.get("intrinsics"),
                "depth_uri": record.get("depth_uri"),
                "confidence_uri": record.get("confidence_uri"),
                "embedding_uri": record.get("embedding_uri"),
                "embedding_model_id": record.get("embedding_model_id") or _DINOV3_MODEL_ID,
                "frame_uri": record.get("frame_uri"),
                "thumbnail_uri": record.get("thumbnail_uri"),
                "privacy_source": record.get("privacy_source", "raw_video"),
                "geometry_source": record.get("geometry_source") or geometry_source,
                "quality": record.get("quality"),
                "anchor_observations": record.get("anchor_observations") or [],
                "retrieval_signals": record.get("retrieval_signals") or {},
                "staticness_score": record.get("staticness_score"),
                "geometry_fingerprint": record.get("geometry_fingerprint") or {},
                "visibility_cells": record.get("visibility_cells") or [],
                "zone_id": record.get("zone_id"),
                "captured_at": captured_at,
                "indexed_at": now,
            }
            f.write(json.dumps(index_record, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Coverage map
# ---------------------------------------------------------------------------


def _update_coverage_map(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
) -> None:
    records = _load_jsonl(site_index_path)
    if not records:
        return

    ref_session = records[0].get("coordinate_frame_session_id", "")
    cells: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        T = effective_pose(rec)
        if T is None:
            continue
        tx = float(T[0, 3])
        tz = float(T[2, 3])
        quality = rec.get("quality") or {}
        sharpness = float(quality.get("sharpness_score") or 0.0)
        capture_id = rec.get("capture_id", "")
        staticness_score = float(rec.get("staticness_score") or 0.0)

        cell_x = int(math.floor(tx / _CELL_SIZE_M))
        cell_z = int(math.floor(tz / _CELL_SIZE_M))
        key = f"{cell_x},{cell_z}"
        if key not in cells:
            cells[key] = {
                "frame_count": 0,
                "capture_ids": [],
                "sharpness_sum": 0.0,
                "staticness_sum": 0.0,
            }
        cells[key]["frame_count"] += 1
        if capture_id and capture_id not in cells[key]["capture_ids"]:
            cells[key]["capture_ids"].append(capture_id)
        cells[key]["sharpness_sum"] += sharpness
        cells[key]["staticness_sum"] += staticness_score

        for visible_key in rec.get("visibility_cells") or []:
            if visible_key not in cells:
                cells[visible_key] = {
                    "frame_count": 0,
                    "capture_ids": [],
                    "sharpness_sum": 0.0,
                    "staticness_sum": 0.0,
                    "visible_only": True,
                }

    dense_threshold = 5
    covered = len(cells)
    dense = sum(1 for c in cells.values() if c["frame_count"] >= dense_threshold)
    covered_area = covered * _CELL_SIZE_M * _CELL_SIZE_M
    dense_area = dense * _CELL_SIZE_M * _CELL_SIZE_M

    if cells:
        xs = [int(k.split(",")[0]) for k in cells]
        zs = [int(k.split(",")[1]) for k in cells]
        origin_x = min(xs) * _CELL_SIZE_M
        origin_z = min(zs) * _CELL_SIZE_M
        grid_width = max(xs) - min(xs) + 1
        grid_depth = max(zs) - min(zs) + 1
    else:
        origin_x = origin_z = 0.0
        grid_width = grid_depth = 0

    coverage_dir = site_root / "coverage"
    ensure_dir(coverage_dir)
    write_json(coverage_dir / "coverage_map.json", {
        "schema_version": "v1",
        "site_id": site_id,
        "coordinate_frame_session_id": ref_session,
        "cell_size_m": _CELL_SIZE_M,
        "origin_x": round(origin_x, 4),
        "origin_z": round(origin_z, 4),
        "grid_width": grid_width,
        "grid_depth": grid_depth,
        "cells": {
            k: {
                "frame_count": v["frame_count"],
                "capture_ids": v["capture_ids"],
                "mean_sharpness": round(v["sharpness_sum"] / v["frame_count"], 1) if v["frame_count"] else 0.0,
                "mean_staticness": round(v["staticness_sum"] / v["frame_count"], 4) if v["frame_count"] else 0.0,
                "visible_only": bool(v.get("visible_only", False)),
            }
            for k, v in cells.items()
        },
        "coverage_summary": {
            "covered_area_m2": round(covered_area, 2),
            "dense_area_m2": round(dense_area, 2),
            "dense_threshold_frames_per_cell": dense_threshold,
        },
    })


# ---------------------------------------------------------------------------
# Site reference manifest
# ---------------------------------------------------------------------------


def _write_site_manifest(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
) -> None:
    records = _load_jsonl(site_index_path)

    # Aggregate per-capture stats
    captures_seen: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        cid = rec.get("capture_id", "")
        if cid not in captures_seen:
            captures_seen[cid] = {
                "capture_id": cid,
                "scene_id": rec.get("scene_id"),
                "captured_at": rec.get("captured_at"),
                "frame_count": 0,
                "chunk_count": 0,
                "coordinate_frame_session_id": rec.get("coordinate_frame_session_id"),
                "site_frame_aligned": rec.get("site_frame_transform") is not None,
                "path_length_m": 0.0,
            }
        captures_seen[cid]["frame_count"] += 1

    # Compute path length per capture
    by_capture: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_capture.setdefault(rec.get("capture_id", ""), []).append(rec)
    for cid, recs in by_capture.items():
        path = 0.0
        last_pos: Optional[np.ndarray] = None
        chunk_ids = {str(r.get("chunk_id") or "") for r in recs if str(r.get("chunk_id") or "").strip()}
        for r in sorted(recs, key=lambda x: x.get("t_capture_sec", 0)):
            T = effective_pose(r)
            if T is None:
                continue
            pos = transform_translation(T)
            if last_pos is not None:
                path += float(np.linalg.norm(pos - last_pos))
            last_pos = pos
        captures_seen[cid]["path_length_m"] = round(path, 2)
        captures_seen[cid]["chunk_count"] = len(chunk_ids)

    # Read coverage summary if available
    coverage_summary: Dict[str, Any] = {}
    coverage_path = site_root / "coverage" / "coverage_map.json"
    if coverage_path.is_file():
        try:
            cm = read_json(coverage_path)
            raw_cs = cm.get("coverage_summary") or {}
            cells = cm.get("cells") or {}
            dense_threshold = raw_cs.get("dense_threshold_frames_per_cell", 5)
            total_cells = len(cells)
            dense_cells = sum(
                1 for c in cells.values() if c.get("frame_count", 0) >= dense_threshold
            )
            coverage_summary = {
                "covered_area_m2": raw_cs.get("covered_area_m2", 0.0),
                "cells_total": total_cells,
                "cells_with_coverage": total_cells,
                "cells_with_dense_coverage": dense_cells,
                "coverage_fraction": (
                    round(dense_cells / total_cells, 4) if total_cells else 0.0
                ),
            }
        except Exception:
            pass

    write_json(site_root / "site_reference_manifest.json", {
        "schema_version": "v2",
        "site_id": site_id,
        "total_reference_frames": len(records),
        "capture_count": len(captures_seen),
        "chunk_count": len({str(r.get("chunk_id") or "") for r in records if str(r.get("chunk_id") or "").strip()}),
        "captures": list(captures_seen.values()),
        "coverage_summary": coverage_summary,
        "last_updated": utc_now_iso(),
        "site_frame_established": any(
            r.get("site_frame_transform") is not None for r in records
        ),
    })


def _write_site_memory_indices(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
    storage_root: Path,
) -> None:
    records = _load_jsonl(site_index_path)
    if not records:
        return

    indices_root = site_root / "indices"
    ensure_dir(indices_root)

    visual_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    anchor_index: Dict[str, Dict[str, Any]] = {}
    zone_index: Dict[str, Dict[str, Any]] = {}

    for record in records:
        reference_id = str(record.get("reference_id") or "")
        chunk_id = str(record.get("chunk_id") or "")
        if record.get("embedding_uri"):
            visual_rows.append(
                {
                    "reference_id": reference_id,
                    "chunk_id": chunk_id,
                    "embedding_uri": record.get("embedding_uri"),
                    "staticness_score": record.get("staticness_score"),
                    "site_frame_aligned": record.get("site_frame_transform") is not None,
                }
            )
        geometry_rows.append(
            {
                "reference_id": reference_id,
                "chunk_id": chunk_id,
                "geometry_fingerprint": record.get("geometry_fingerprint") or {},
                "visibility_cells": record.get("visibility_cells") or [],
            }
        )
        for anchor_id in _anchor_ids(record.get("anchor_observations")):
            payload = anchor_index.setdefault(anchor_id, {"reference_ids": [], "chunk_ids": []})
            if reference_id and reference_id not in payload["reference_ids"]:
                payload["reference_ids"].append(reference_id)
            if chunk_id and chunk_id not in payload["chunk_ids"]:
                payload["chunk_ids"].append(chunk_id)
        zone_id = str(record.get("zone_id") or "").strip()
        if zone_id:
            payload = zone_index.setdefault(zone_id, {"reference_ids": [], "chunk_ids": []})
            if reference_id and reference_id not in payload["reference_ids"]:
                payload["reference_ids"].append(reference_id)
            if chunk_id and chunk_id not in payload["chunk_ids"]:
                payload["chunk_ids"].append(chunk_id)

    write_json(indices_root / "visual_index.json", {"schema_version": "v1", "site_id": site_id, "rows": visual_rows})
    write_json(indices_root / "geometry_index.json", {"schema_version": "v1", "site_id": site_id, "rows": geometry_rows})
    write_json(indices_root / "anchor_inverted_index.json", {"schema_version": "v1", "site_id": site_id, "anchors": anchor_index})
    write_json(indices_root / "zone_index.json", {"schema_version": "v1", "site_id": site_id, "zones": zone_index})
    write_json(
        indices_root / "manifest.json",
        {
            "schema_version": "v1",
            "site_id": site_id,
            "generated_at": utc_now_iso(),
            "visual_row_count": len(visual_rows),
            "geometry_row_count": len(geometry_rows),
            "anchor_count": len(anchor_index),
            "zone_count": len(zone_index),
            "storage_root": str(storage_root),
        },
    )


def _write_overlap_graph(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
    storage_root: Path,
) -> None:
    records = _load_jsonl(site_index_path)
    if not records:
        return

    chunks = iter_groups(records, "chunk_id")
    chunk_summaries = {
        chunk_id: aggregate_chunk_summary(chunk_records, storage_root=storage_root)
        for chunk_id, chunk_records in chunks.items()
    }
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for chunk_id, chunk_records in chunks.items():
        exemplar = chunk_records[0]
        summary = chunk_summaries[chunk_id]
        nodes.append(
            {
                "node_id": chunk_id,
                "node_type": "chunk",
                "capture_id": exemplar.get("capture_id"),
                "coordinate_frame_session_id": exemplar.get("coordinate_frame_session_id"),
                "zone_id": summary.get("zone_id"),
                "anchor_ids": summary.get("anchor_ids") or [],
                "record_count": summary.get("record_count"),
                "staticness_score": summary.get("staticness_score"),
            }
        )

    chunk_items = list(chunks.items())
    for index, (left_chunk_id, left_records) in enumerate(chunk_items):
        left_summary = chunk_summaries[left_chunk_id]
        left_centroid = left_summary.get("embedding_centroid")
        left_geometry = left_summary.get("geometry_fingerprint") or {}
        left_anchors = set(left_summary.get("anchor_ids") or [])
        left_zone = str(left_summary.get("zone_id") or "")
        left_session = str(left_records[0].get("coordinate_frame_session_id") or "")
        for right_chunk_id, right_records in chunk_items[index + 1 :]:
            right_summary = chunk_summaries[right_chunk_id]
            right_centroid = right_summary.get("embedding_centroid")
            right_geometry = right_summary.get("geometry_fingerprint") or {}
            right_anchors = set(right_summary.get("anchor_ids") or [])
            right_zone = str(right_summary.get("zone_id") or "")
            right_session = str(right_records[0].get("coordinate_frame_session_id") or "")

            score_visual = 0.0
            if isinstance(left_centroid, np.ndarray) and isinstance(right_centroid, np.ndarray):
                score_visual = float(np.clip(np.dot(left_centroid, right_centroid), 0.0, 1.0))
            score_geometry = 0.0
            if left_geometry and right_geometry:
                score_geometry = _sm_fingerprint_similarity(left_geometry, right_geometry)
            shared_anchors = sorted(left_anchors & right_anchors)
            zone_match = bool(left_zone and left_zone == right_zone)
            score_topology = 0.0
            if shared_anchors:
                score_topology += 0.6
            if zone_match:
                score_topology += 0.2
            if left_session == right_session:
                score_topology += 0.2
            total_score = round((0.45 * score_visual) + (0.30 * score_geometry) + (0.25 * min(score_topology, 1.0)), 4)
            if total_score < 0.2:
                continue
            edges.append(
                {
                    "edge_id": f"edge_{left_chunk_id}_{right_chunk_id}",
                    "edge_type": "candidate_overlap",
                    "from_chunk_id": left_chunk_id,
                    "to_chunk_id": right_chunk_id,
                    "from_session_id": left_session,
                    "to_session_id": right_session,
                    "score_visual": round(score_visual, 4),
                    "score_geometry": round(score_geometry, 4),
                    "score_topology": round(min(score_topology, 1.0), 4),
                    "total_score": total_score,
                    "shared_anchor_ids": shared_anchors,
                    "zone_match": zone_match,
                    "accepted_for_alignment": False,
                    "supporting_reference_ids": [
                        str(item.get("reference_id"))
                        for item in (left_records[:2] + right_records[:2])
                        if item.get("reference_id")
                    ],
                }
            )

    write_json(
        site_root / "site_overlap_graph.json",
        {
            "schema_version": "v1",
            "site_id": site_id,
            "generated_at": utc_now_iso(),
            "nodes": nodes,
            "edges": sorted(edges, key=lambda item: float(item.get("total_score") or 0.0), reverse=True),
        },
    )


def _write_retrieval_validation(
    *,
    site_root: Path,
    site_index_path: Path,
    site_id: str,
) -> None:
    records = _load_jsonl(site_index_path)
    if not records:
        return

    chunk_groups = iter_groups(records, "chunk_id")
    staticness_scores = [float(record.get("staticness_score") or 0.0) for record in records]
    geometry_available = sum(1 for record in records if (record.get("geometry_fingerprint") or {}).get("available"))
    write_json(
        site_root / "retrieval_validation.json",
        {
            "schema_version": "v1",
            "site_id": site_id,
            "generated_at": utc_now_iso(),
            "reference_frame_count": len(records),
            "chunk_count": len(chunk_groups),
            "geometry_fingerprint_coverage": round(geometry_available / float(len(records) or 1), 4),
            "mean_staticness_score": round(sum(staticness_scores) / float(len(staticness_scores) or 1), 4),
            "aligned_fraction": round(
                sum(1 for record in records if record.get("site_frame_transform") is not None) / float(len(records) or 1),
                4,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _mat_tx(T: Any) -> float:
    try:
        return float(T[0][3])
    except (TypeError, IndexError, KeyError):
        return 0.0


def _mat_ty(T: Any) -> float:
    try:
        return float(T[1][3])
    except (TypeError, IndexError, KeyError):
        return 0.0


def _mat_tz(T: Any) -> float:
    try:
        return float(T[2][3])
    except (TypeError, IndexError, KeyError):
        return 0.0


def _euclidean(
    a: Tuple[float, float, float],
    b: Optional[Tuple[float, float, float]],
) -> float:
    if b is None:
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _p95(values: List[float]) -> float:
    return _sm_p95(values)


# ---------------------------------------------------------------------------
# GCS URI helpers
# ---------------------------------------------------------------------------


def _local_to_gs_uri(path: Path, ctx: LocalCaptureContext) -> Optional[str]:
    try:
        rel = path.relative_to(ctx.storage_root / ctx.bucket)
        return f"gs://{ctx.bucket}/{rel.as_posix()}"
    except ValueError:
        return None


def _arkit_depth_uri(frame_id: str, ctx: LocalCaptureContext) -> Optional[str]:
    p = ctx.raw_root / "arkit" / "depth" / f"{frame_id}.png"
    return _local_to_gs_uri(p, ctx) if p.is_file() else None


def _arkit_confidence_uri(frame_id: str, ctx: LocalCaptureContext) -> Optional[str]:
    p = ctx.raw_root / "arkit" / "confidence" / f"{frame_id}.png"
    return _local_to_gs_uri(p, ctx) if p.is_file() else None
