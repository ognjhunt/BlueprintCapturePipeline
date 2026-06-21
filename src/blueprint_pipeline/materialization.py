"""Materialize qualification inputs from raw capture uploads."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .capture_bridge import _normalize_requested_lanes
from .common import (
    ensure_dir,
    join_gs_uri,
    parse_bool,
    read_json,
    resolve_gs_uri_to_path,
    try_parse_float,
    utc_now_iso,
    write_json,
)

_IPHONE_POSE_MATCH_RATE_MIN = 0.65
_IPHONE_P95_POSE_DELTA_MAX = 0.2
_DEFAULT_REQUESTED_OUTPUTS = ["qualification", "preview_simulation"]


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _merge_manifest_with_sidecars(manifest: Mapping[str, Any], raw_root: Path) -> Dict[str, Any]:
    """Merge raw sidecar contract files into old/minimal manifests without inventing values."""
    merged: Dict[str, Any] = dict(manifest)
    sidecar_files = {
        "site_identity": "site_identity.json",
        "capture_topology": "capture_topology.json",
        "capture_mode": "capture_mode.json",
        "route_anchors": "route_anchors.json",
        "checkpoint_events": "checkpoint_events.json",
        "relocalization_events": "relocalization_events.json",
    }
    for key, filename in sidecar_files.items():
        if isinstance(merged.get(key), Mapping):
            continue
        payload = _read_optional_json(raw_root / filename)
        if payload:
            merged[key] = payload
    return merged


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    out: List[str] = []
    for item in values:
        text = item.strip()
        if text and text not in out:
            out.append(text)
    return out


def _default_preview_disabled(manifest: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), Mapping) else {}
    for raw in (
        context.get("disable_default_preview"),
        context.get("disableDefaultPreview"),
        manifest.get("disable_default_preview"),
        manifest.get("disableDefaultPreview"),
        metadata.get("disable_default_preview") if isinstance(metadata, Mapping) else None,
        metadata.get("disableDefaultPreview") if isinstance(metadata, Mapping) else None,
    ):
        if raw is not None:
            return parse_bool(raw, default=False)
    return False


def _normalized_requested_outputs(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> List[str]:
    requested_outputs = _string_list(
        manifest.get("requested_outputs")
        or manifest.get("requestedOutputs")
        or context.get("requested_outputs")
        or context.get("requestedOutputs")
    )
    if _default_preview_disabled(manifest, context):
        return requested_outputs
    normalized = [str(value).strip().lower() for value in requested_outputs if str(value).strip()]
    if not normalized or normalized == ["qualification"]:
        return list(_DEFAULT_REQUESTED_OUTPUTS)
    return requested_outputs


def _dict_float(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, float] = {}
    for key, item in value.items():
        text = str(key).strip()
        if not text:
            continue
        try:
            out[text] = float(item)
        except (TypeError, ValueError):
            continue
    return out


def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    except OSError:
        return []
    return rows


def _normalized_frame_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if text:
        return text
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return str(max(0, index) + 1).zfill(6)


def _time_value(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("t_device_sec", "tCaptureSec", "timestamp"):
        if row.get(key) is None:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(len(ordered) - 1, low + 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _nearest_pose_time(ordered_pose_times: List[float], target_time: float) -> Optional[float]:
    if not ordered_pose_times:
        return None
    if len(ordered_pose_times) == 1:
        return ordered_pose_times[0]
    low = 0
    high = len(ordered_pose_times) - 1
    while low < high:
        mid = (low + high) // 2
        if ordered_pose_times[mid] < target_time:
            low = mid + 1
        else:
            high = mid
    best = ordered_pose_times[low]
    if low > 0:
        previous = ordered_pose_times[low - 1]
        if abs(previous - target_time) <= abs(best - target_time):
            best = previous
    return best


def _inspect_pose_alignment(raw_root: Path) -> Dict[str, Optional[float]]:
    frames_rows = _read_json_lines(raw_root / "arkit" / "frames.jsonl")
    pose_rows = _read_json_lines(raw_root / "arkit" / "poses.jsonl")
    if not frames_rows or not pose_rows:
        return {
            "pose_match_rate": None,
            "p95_pose_delta_sec": None,
            "matched_pose_count": None,
            "frame_count": float(len(frames_rows)) if frames_rows else None,
        }

    poses_by_frame_id: Dict[str, float] = {}
    pose_times: List[float] = []
    for row in pose_rows:
        pose_time = _time_value(row)
        if pose_time is None:
            continue
        pose_times.append(pose_time)
        frame_id = _normalized_frame_id(
            row.get("frame_id") or row.get("frameIndex") or row.get("frame_index")
        )
        if frame_id:
            poses_by_frame_id[frame_id] = pose_time
    pose_times.sort()

    matched = 0
    deltas: List[float] = []
    for row in frames_rows:
        frame_time = _time_value(row)
        frame_id = _normalized_frame_id(
            row.get("frame_id") or row.get("frameIndex") or row.get("frame_index")
        )
        matched_pose_time: Optional[float] = None
        if frame_id and frame_id in poses_by_frame_id:
            matched_pose_time = poses_by_frame_id[frame_id]
        elif frame_time is not None:
            matched_pose_time = _nearest_pose_time(pose_times, frame_time)
        if matched_pose_time is None:
            continue
        matched += 1
        if frame_time is not None:
            deltas.append(abs(frame_time - matched_pose_time))

    frame_count = len(frames_rows)
    pose_match_rate = float(matched) / float(frame_count) if frame_count else None
    p95_pose_delta_sec = _percentile(deltas, 95.0)
    return {
        "pose_match_rate": pose_match_rate,
        "p95_pose_delta_sec": p95_pose_delta_sec,
        "matched_pose_count": float(matched),
        "frame_count": float(frame_count),
    }


def _iphone_pose_alignment_ok(
    pose_match_rate: Optional[float],
    p95_pose_delta_sec: Optional[float],
) -> bool:
    return (
        pose_match_rate is not None
        and p95_pose_delta_sec is not None
        and pose_match_rate >= _IPHONE_POSE_MATCH_RATE_MIN
        and p95_pose_delta_sec <= _IPHONE_P95_POSE_DELTA_MAX
    )


def _stable_site_id_present(manifest: Mapping[str, Any]) -> bool:
    raw = manifest.get("site_identity")
    if not isinstance(raw, Mapping):
        return False
    return bool(str(raw.get("site_id") or "").strip())


def _canonical_world_model_candidate(
    manifest: Mapping[str, Any],
    arkit_poses_uri: Optional[str],
    arkit_intrinsics_uri: Optional[str],
    arkit_depth_prefix_uri: Optional[str],
    intake_complete: bool,
    evidence_tier: str,
    capture_source: str = "iphone",
    pose_match_rate: Optional[float] = None,
    p95_pose_delta_sec: Optional[float] = None,
    geometry_ready: bool = False,
    geometry_source: Optional[str] = None,
) -> bool:
    """Canonical world_model_candidate rule — shared across iOS finalizer, cloud bridge,
    and local pipeline. Must be kept in sync with bridge/index.ts canonicalWorldModelCandidate()
    and CaptureBundleContext.worldModelCandidate() in iOS.

    If capture_mode is absent (old captures), falls back to evidence_tier heuristic for
    backwards compatibility.
    """
    site_id_present = _stable_site_id_present(manifest)
    capture_mode = manifest.get("capture_mode")
    if not isinstance(capture_mode, Mapping):
        # Backwards compatibility: captures predating capture_mode field.
        return site_id_present and evidence_tier != "pre_screen_video"
    resolved_mode = str(capture_mode.get("resolved_mode") or "qualification_only")
    rights_block = manifest.get("capture_rights") if isinstance(manifest.get("capture_rights"), Mapping) else {}
    arkit_ready = (
        arkit_poses_uri is not None
        and arkit_intrinsics_uri is not None
        and arkit_depth_prefix_uri is not None
    )
    if capture_source == "iphone":
        spatial_conditioning_ready = arkit_ready and _iphone_pose_alignment_ok(
            pose_match_rate,
            p95_pose_delta_sec,
        )
    else:
        spatial_conditioning_ready = arkit_ready or geometry_ready
    return (
        site_id_present
        and resolved_mode == "site_world_candidate"
        and spatial_conditioning_ready
        and intake_complete
        and bool(rights_block.get("derived_scene_generation_allowed", False))
    )


def _world_model_candidate_reasoning(
    manifest: Mapping[str, Any],
    arkit_poses_uri: Optional[str],
    arkit_intrinsics_uri: Optional[str],
    arkit_depth_prefix_uri: Optional[str],
    intake_complete: bool,
    capture_source: str = "iphone",
    pose_match_rate: Optional[float] = None,
    p95_pose_delta_sec: Optional[float] = None,
    geometry_ready: bool = False,
    geometry_source: Optional[str] = None,
) -> list:
    capture_mode = manifest.get("capture_mode")
    resolved_mode = str((capture_mode or {}).get("resolved_mode") or "qualification_only") if isinstance(capture_mode, Mapping) else "qualification_only"
    rights_block = manifest.get("capture_rights") if isinstance(manifest.get("capture_rights"), Mapping) else {}
    return [
        f"capture_mode_site_world_candidate:{resolved_mode == 'site_world_candidate'}",
        f"site_id_present:{_stable_site_id_present(manifest)}",
        f"capture_source:{capture_source or 'unknown'}",
        f"arkit_poses_valid:{arkit_poses_uri is not None}",
        f"arkit_intrinsics_valid:{arkit_intrinsics_uri is not None}",
        f"depth_coverage_ok:{arkit_depth_prefix_uri is not None}",
        f"pose_alignment_ok:{capture_source != 'iphone' or _iphone_pose_alignment_ok(pose_match_rate, p95_pose_delta_sec)}",
        f"pose_match_rate:{round(pose_match_rate, 4) if pose_match_rate is not None else 'missing'}",
        f"p95_pose_delta_sec:{round(p95_pose_delta_sec, 4) if p95_pose_delta_sec is not None else 'missing'}",
        f"geometry_ready:{geometry_ready}",
        f"geometry_source:{geometry_source or 'none'}",
        f"intake_complete:{intake_complete}",
        f"derived_scene_generation_allowed:{bool(rights_block.get('derived_scene_generation_allowed', False))}",
    ]


def _normalized_site_identity(manifest: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = manifest.get("site_identity")
    if not isinstance(raw, Mapping):
        return None
    geo_raw = raw.get("geo")
    geo = None
    if isinstance(geo_raw, Mapping):
        geo = {
            "latitude": geo_raw.get("latitude"),
            "longitude": geo_raw.get("longitude"),
            "accuracy_m": geo_raw.get("accuracy_m"),
        }
    return {
        "site_id": str(raw.get("site_id") or "").strip() or None,
        "site_id_source": str(raw.get("site_id_source") or "unknown"),
        "place_id": str(raw.get("place_id") or "").strip() or None,
        "site_name": str(raw.get("site_name") or "").strip() or None,
        "address_full": str(raw.get("address_full") or "").strip() or None,
        "geo": geo,
        "building_id": str(raw.get("building_id") or "").strip() or None,
        "floor_id": str(raw.get("floor_id") or "").strip() or None,
        "room_id": str(raw.get("room_id") or "").strip() or None,
        "zone_id": str(raw.get("zone_id") or "").strip() or None,
    }


def _normalized_capture_topology(manifest: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = manifest.get("capture_topology")
    if not isinstance(raw, Mapping):
        return None
    return {
        "capture_session_id": str(raw.get("capture_session_id") or "").strip() or None,
        "route_id": str(raw.get("route_id") or "").strip() or None,
        "pass_id": str(raw.get("pass_id") or "").strip() or None,
        "pass_index": int(raw["pass_index"]) if isinstance(raw.get("pass_index"), (int, float)) else None,
        "intended_pass_role": str(raw.get("intended_pass_role") or "primary"),
        "entry_anchor_id": str(raw.get("entry_anchor_id") or "").strip() or None,
        "return_anchor_id": str(raw.get("return_anchor_id") or "").strip() or None,
        "entry_anchor_t_capture_sec": (
            try_parse_float(raw.get("entry_anchor_t_capture_sec"), 0.0)
            if raw.get("entry_anchor_t_capture_sec") is not None
            else None
        ),
        "entry_anchor_hold_duration_sec": (
            try_parse_float(raw.get("entry_anchor_hold_duration_sec"), 0.0)
            if raw.get("entry_anchor_hold_duration_sec") is not None
            else None
        ),
        "site_visit_id": str(raw.get("site_visit_id") or "").strip() or None,
        "coordinate_frame_session_id": str(raw.get("coordinate_frame_session_id") or "").strip() or None,
        "arkit_session_id": str(raw.get("arkit_session_id") or "").strip() or None,
    }


def _normalized_route_anchors(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    route_anchors_raw = raw.get("route_anchors") or raw.get("routeAnchors")
    route_anchors: List[Dict[str, Any]] = []
    if isinstance(route_anchors_raw, list):
        for item in route_anchors_raw:
            if not isinstance(item, Mapping):
                continue
            route_anchors.append(
                {
                    "anchor_id": str(item.get("anchor_id") or item.get("anchorId") or "").strip() or None,
                    "anchor_type": str(item.get("anchor_type") or item.get("anchorType") or "").strip() or None,
                    "label": str(item.get("label") or "").strip() or None,
                    "expected_observation": str(
                        item.get("expected_observation") or item.get("expectedObservation") or ""
                    ).strip()
                    or None,
                    "required_in_primary_pass": bool(
                        item.get("required_in_primary_pass")
                        if item.get("required_in_primary_pass") is not None
                        else item.get("requiredInPrimaryPass")
                    ),
                    "required_in_revisit_pass": bool(
                        item.get("required_in_revisit_pass")
                        if item.get("required_in_revisit_pass") is not None
                        else item.get("requiredInRevisitPass")
                    ),
                }
            )
    return {
        "schema_version": str(raw.get("schema_version") or raw.get("schemaVersion") or "v1"),
        "route_anchors": route_anchors,
    }


def _normalized_checkpoint_events(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    checkpoint_events_raw = raw.get("checkpoint_events") or raw.get("checkpointEvents")
    checkpoint_events: List[Dict[str, Any]] = []
    if isinstance(checkpoint_events_raw, list):
        for item in checkpoint_events_raw:
            if not isinstance(item, Mapping):
                continue
            checkpoint_events.append(
                {
                    "anchor_id": str(item.get("anchor_id") or item.get("anchorId") or "").strip() or None,
                    "pass_id": str(item.get("pass_id") or item.get("passId") or "").strip() or None,
                    "t_capture_sec": (
                        try_parse_float(item.get("t_capture_sec") or item.get("tCaptureSec"), 0.0)
                        if (item.get("t_capture_sec") is not None or item.get("tCaptureSec") is not None)
                        else None
                    ),
                    "hold_duration_sec": (
                        try_parse_float(item.get("hold_duration_sec") or item.get("holdDurationSec"), 0.0)
                        if (item.get("hold_duration_sec") is not None or item.get("holdDurationSec") is not None)
                        else None
                    ),
                    "completed": bool(item.get("completed")),
                }
            )
    return {
        "schema_version": str(raw.get("schema_version") or raw.get("schemaVersion") or "v1"),
        "checkpoint_events": checkpoint_events,
    }


def _normalized_relocalization_events(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    events_raw = raw.get("relocalization_events") or raw.get("relocalizationEvents")
    events: List[Dict[str, Any]] = []
    if isinstance(events_raw, list):
        for item in events_raw:
            if not isinstance(item, Mapping):
                continue
            events.append(
                {
                    "event_id": str(item.get("event_id") or item.get("eventId") or "").strip() or None,
                    "pass_id": str(item.get("pass_id") or item.get("passId") or "").strip() or None,
                    "route_id": str(item.get("route_id") or item.get("routeId") or "").strip() or None,
                    "t_capture_sec": (
                        try_parse_float(item.get("t_capture_sec") or item.get("tCaptureSec"), 0.0)
                        if (item.get("t_capture_sec") is not None or item.get("tCaptureSec") is not None)
                        else None
                    ),
                    "status": str(item.get("status") or "").strip() or None,
                    "anchor_id": str(item.get("anchor_id") or item.get("anchorId") or "").strip() or None,
                    "coordinate_frame_session_id": str(
                        item.get("coordinate_frame_session_id") or item.get("coordinateFrameSessionId") or ""
                    ).strip()
                    or None,
                }
            )
    return {
        "schema_version": str(raw.get("schema_version") or raw.get("schemaVersion") or "v1"),
        "relocalization_events": events,
    }


def _world_model_candidate_downgrade_reason(
    *,
    manifest: Mapping[str, Any],
    arkit_poses_uri: Optional[str],
    arkit_intrinsics_uri: Optional[str],
    arkit_depth_prefix_uri: Optional[str],
    intake_complete: bool,
    capture_source: str,
    pose_match_rate: Optional[float],
    p95_pose_delta_sec: Optional[float],
    geometry_ready: bool,
) -> str:
    if not _stable_site_id_present(manifest):
        return "missing_site_id"
    if capture_source == "iphone":
        if arkit_poses_uri is None:
            return "missing_arkit_poses"
        if arkit_intrinsics_uri is None:
            return "missing_arkit_intrinsics"
        if arkit_depth_prefix_uri is None:
            return "missing_lidar_depth"
        if not _iphone_pose_alignment_ok(pose_match_rate, p95_pose_delta_sec):
            return "insufficient_spatial_evidence"
    elif not (arkit_poses_uri is not None and arkit_intrinsics_uri is not None and arkit_depth_prefix_uri is not None) and not geometry_ready:
        return "awaiting_geometry_stage"
    if not intake_complete:
        return "missing_complete_intake"
    rights_block = manifest.get("capture_rights") if isinstance(manifest.get("capture_rights"), Mapping) else {}
    if not bool(rights_block.get("derived_scene_generation_allowed", False)):
        return "derived_scene_generation_not_allowed"
    return "site_world_candidate_gates_not_met"


def _normalized_capture_mode(
    manifest: Mapping[str, Any],
    arkit_poses_uri: Optional[str],
    arkit_intrinsics_uri: Optional[str],
    arkit_depth_prefix_uri: Optional[str],
    intake_complete: bool,
    evidence_tier: str,
    capture_source: str = "iphone",
    pose_match_rate: Optional[float] = None,
    p95_pose_delta_sec: Optional[float] = None,
    geometry_ready: bool = False,
    geometry_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    raw = manifest.get("capture_mode")
    if not isinstance(raw, Mapping):
        return None
    requested_mode = str(raw.get("requested_mode") or "qualification_only")
    candidate = _canonical_world_model_candidate(
        manifest=manifest,
        arkit_poses_uri=arkit_poses_uri,
        arkit_intrinsics_uri=arkit_intrinsics_uri,
        arkit_depth_prefix_uri=arkit_depth_prefix_uri,
        intake_complete=intake_complete,
        evidence_tier=evidence_tier,
        capture_source=capture_source,
        pose_match_rate=pose_match_rate,
        p95_pose_delta_sec=p95_pose_delta_sec,
        geometry_ready=geometry_ready,
        geometry_source=geometry_source,
    )
    resolved_mode = "site_world_candidate" if candidate else "qualification_only"
    downgrade_reason: Optional[str] = None
    if requested_mode == "site_world_candidate" and resolved_mode == "qualification_only":
        downgrade_reason = _world_model_candidate_downgrade_reason(
            manifest=manifest,
            arkit_poses_uri=arkit_poses_uri,
            arkit_intrinsics_uri=arkit_intrinsics_uri,
            arkit_depth_prefix_uri=arkit_depth_prefix_uri,
            intake_complete=intake_complete,
            capture_source=capture_source,
            pose_match_rate=pose_match_rate,
            p95_pose_delta_sec=p95_pose_delta_sec,
            geometry_ready=geometry_ready,
        )
    return {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "downgrade_reason": downgrade_reason,
    }


def _capture_rights_block(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    raw = manifest.get("capture_rights") if isinstance(manifest.get("capture_rights"), Mapping) else {}
    return {
        "derived_scene_generation_allowed": bool(raw.get("derived_scene_generation_allowed", False)),
        "data_licensing_allowed": bool(raw.get("data_licensing_allowed", False)),
        "capture_contributor_payout_eligible": bool(raw.get("capture_contributor_payout_eligible", False)),
        "consent_status": str(raw.get("consent_status") or "unknown"),
        "permission_document_uri": str(raw.get("permission_document_uri") or "").strip() or None,
        "consent_scope": _string_list(raw.get("consent_scope")),
        "consent_notes": _string_list(raw.get("consent_notes")),
    }


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = int(round(float(value)))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _normalize_rotation_degrees(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        degrees = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    normalized = degrees % 360
    for candidate in (0, 90, 180, 270):
        if abs(normalized - candidate) <= 1:
            return candidate
    return normalized


def _size_payload(width: Optional[int], height: Optional[int]) -> Dict[str, int]:
    if width is None or height is None or width <= 0 or height <= 0:
        return {}
    return {"width": int(width), "height": int(height)}


def _infer_display_orientation(width: Optional[int], height: Optional[int]) -> str:
    if width is None or height is None or width <= 0 or height <= 0:
        return "unknown"
    if width == height:
        return "square"
    return "portrait" if height > width else "landscape"


def _display_size_from_rotation(
    encoded_width: Optional[int],
    encoded_height: Optional[int],
    rotation_degrees: Optional[int],
) -> Dict[str, int]:
    if encoded_width is None or encoded_height is None:
        return {}
    if rotation_degrees in {90, 270}:
        return _size_payload(encoded_height, encoded_width)
    return _size_payload(encoded_width, encoded_height)


def _declared_capture_dimensions(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[Optional[int], Optional[int]]:
    width = _first_int(
        manifest.get("declared_capture_width"),
        manifest.get("declaredCaptureWidth"),
        manifest.get("capture_width"),
        manifest.get("captureWidth"),
        context.get("declaredCaptureWidth"),
        context.get("declared_capture_width"),
        context.get("captureWidth"),
        context.get("capture_width"),
        context.get("displayWidth"),
        context.get("display_width"),
    )
    height = _first_int(
        manifest.get("declared_capture_height"),
        manifest.get("declaredCaptureHeight"),
        manifest.get("capture_height"),
        manifest.get("captureHeight"),
        context.get("declaredCaptureHeight"),
        context.get("declared_capture_height"),
        context.get("captureHeight"),
        context.get("capture_height"),
        context.get("displayHeight"),
        context.get("display_height"),
    )
    return width, height


def _orientation_payload(
    *,
    encoded_width: Optional[int],
    encoded_height: Optional[int],
    declared_capture_width: Optional[int],
    declared_capture_height: Optional[int],
    display_rotation_degrees: int,
    display_orientation: str,
    normalization_applied: bool,
    source: str,
    probe_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    encoded_size = _size_payload(encoded_width, encoded_height)
    display_size = _display_size_from_rotation(
        encoded_width,
        encoded_height,
        display_rotation_degrees,
    )
    if declared_capture_width and declared_capture_height:
        display_size = _size_payload(declared_capture_width, declared_capture_height) or display_size
    if not display_orientation:
        display_orientation = _infer_display_orientation(
            display_size.get("width"),
            display_size.get("height"),
        )
    payload: Dict[str, Any] = {
        "encoded_width": int(encoded_width or 0),
        "encoded_height": int(encoded_height or 0),
        "declared_capture_width": int(declared_capture_width or 0),
        "declared_capture_height": int(declared_capture_height or 0),
        "display_rotation_degrees": int(display_rotation_degrees),
        "display_orientation": display_orientation or "unknown",
        "normalization_applied": bool(normalization_applied),
        "source": source,
        "rotation_degrees": int(display_rotation_degrees),
        "display_size": display_size,
        "encoded_size": encoded_size,
        "preserve_original_display_orientation": True,
    }
    if probe_details:
        payload["probe_details"] = dict(probe_details)
    return payload


def _raw_orientation_mapping(raw_value: Any) -> Dict[str, Any]:
    if not isinstance(raw_value, Mapping):
        return {}
    return dict(raw_value)


def _capture_orientation_from_metadata(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    candidates = [
        ("capture_context.captureOrientation", _raw_orientation_mapping(context.get("captureOrientation"))),
        ("capture_context.capture_orientation", _raw_orientation_mapping(context.get("capture_orientation"))),
        ("raw_manifest.capture_orientation", _raw_orientation_mapping(manifest.get("capture_orientation"))),
        ("raw_manifest.captureOrientation", _raw_orientation_mapping(manifest.get("captureOrientation"))),
    ]
    encoded_width = _first_int(
        manifest.get("width"),
        context.get("width"),
        context.get("sourceVideoWidth"),
        context.get("videoWidth"),
    )
    encoded_height = _first_int(
        manifest.get("height"),
        context.get("height"),
        context.get("sourceVideoHeight"),
        context.get("videoHeight"),
    )
    declared_capture_width, declared_capture_height = _declared_capture_dimensions(
        manifest=manifest,
        context=context,
    )
    for source, raw in candidates:
        if not raw:
            continue
        display_orientation = str(
            raw.get("display_orientation")
            or raw.get("displayOrientation")
            or ""
        ).strip().lower()
        rotation_degrees = _normalize_rotation_degrees(
            raw.get("rotation_degrees") or raw.get("rotationDegrees")
        )
        display_width = _first_int(
            ((raw.get("display_size") or raw.get("displaySize")) or {}).get("width")
            if isinstance(raw.get("display_size") or raw.get("displaySize"), Mapping)
            else None,
            raw.get("display_width"),
            raw.get("displayWidth"),
        )
        display_height = _first_int(
            ((raw.get("display_size") or raw.get("displaySize")) or {}).get("height")
            if isinstance(raw.get("display_size") or raw.get("displaySize"), Mapping)
            else None,
            raw.get("display_height"),
            raw.get("displayHeight"),
        )
        if not display_orientation and display_width and display_height:
            display_orientation = _infer_display_orientation(display_width, display_height)
        if not display_orientation and rotation_degrees is not None:
            display_orientation = _infer_display_orientation(
                *(
                    (
                        encoded_height,
                        encoded_width,
                    )
                    if rotation_degrees in {90, 270}
                    else (encoded_width, encoded_height)
                )
            )
        if not display_orientation and rotation_degrees is None and not display_width and not display_height:
            continue
        display_size = _size_payload(display_width, display_height) or _display_size_from_rotation(
            encoded_width, encoded_height, rotation_degrees
        )
        if not display_orientation:
            display_orientation = _infer_display_orientation(
                display_size.get("width"),
                display_size.get("height"),
            )
        metadata_source = "capture_context" if source.startswith("capture_context") else "manifest"
        return _orientation_payload(
            encoded_width=encoded_width,
            encoded_height=encoded_height,
            declared_capture_width=declared_capture_width or display_size.get("width"),
            declared_capture_height=declared_capture_height or display_size.get("height"),
            display_rotation_degrees=rotation_degrees if rotation_degrees is not None else 0,
            display_orientation=display_orientation or "unknown",
            normalization_applied=bool(rotation_degrees if rotation_degrees is not None else 0),
            source=metadata_source,
        )
    return {}


def _ffprobe_capture_orientation(video_path: Path) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                str(video_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return {}
    if result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return {}
    stream = next((item for item in streams if isinstance(item, Mapping) and item.get("codec_type") == "video"), None)
    if not isinstance(stream, Mapping):
        return {}
    encoded_width = _first_int(stream.get("width"))
    encoded_height = _first_int(stream.get("height"))
    tags = stream.get("tags") if isinstance(stream.get("tags"), Mapping) else {}
    side_data_list = stream.get("side_data_list") if isinstance(stream.get("side_data_list"), list) else []
    rotation_candidates: List[Any] = [tags.get("rotate") if isinstance(tags, Mapping) else None]
    for item in side_data_list:
        if isinstance(item, Mapping):
            rotation_candidates.append(item.get("rotation"))
    rotation_degrees = next(
        (
            normalized
            for normalized in (_normalize_rotation_degrees(candidate) for candidate in rotation_candidates)
            if normalized is not None
        ),
        0,
    )
    display_size = _display_size_from_rotation(encoded_width, encoded_height, rotation_degrees)
    return _orientation_payload(
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        declared_capture_width=display_size.get("width"),
        declared_capture_height=display_size.get("height"),
        display_rotation_degrees=rotation_degrees,
        display_orientation=_infer_display_orientation(
            display_size.get("width"),
            display_size.get("height"),
        ),
        normalization_applied=bool(rotation_degrees),
        source="video_metadata",
        probe_details={
            "tool": "ffprobe",
            "stream_rotation_degrees": rotation_degrees,
            "video_path": str(video_path),
        },
    )


def _capture_orientation_from_dimensions(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    encoded_width = _first_int(
        manifest.get("width"),
        context.get("width"),
        context.get("sourceVideoWidth"),
        context.get("videoWidth"),
    )
    encoded_height = _first_int(
        manifest.get("height"),
        context.get("height"),
        context.get("sourceVideoHeight"),
        context.get("videoHeight"),
    )
    declared_capture_width, declared_capture_height = _declared_capture_dimensions(
        manifest=manifest,
        context=context,
    )
    encoded_orientation = _infer_display_orientation(encoded_width, encoded_height)
    declared_orientation = _infer_display_orientation(
        declared_capture_width,
        declared_capture_height,
    )
    normalization_applied = (
        declared_orientation not in {"unknown", encoded_orientation}
        and encoded_orientation != "unknown"
    )
    display_rotation_degrees = 90 if normalization_applied else 0
    display_orientation = declared_orientation if declared_orientation != "unknown" else encoded_orientation
    return _orientation_payload(
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        declared_capture_width=declared_capture_width or encoded_width,
        declared_capture_height=declared_capture_height or encoded_height,
        display_rotation_degrees=display_rotation_degrees,
        display_orientation=display_orientation,
        normalization_applied=normalization_applied,
        source="inferred",
    )


def _resolve_capture_orientation(
    *,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    raw_root: Path,
) -> Dict[str, Any]:
    from_metadata = _capture_orientation_from_metadata(manifest=manifest, context=context)
    if from_metadata:
        return from_metadata
    video_candidates = _raw_video_candidates(raw_root)
    if video_candidates:
        video_path = raw_root / video_candidates[0]
        if video_path.is_file():
            from_probe = _ffprobe_capture_orientation(video_path)
            if from_probe:
                return from_probe
    return _capture_orientation_from_dimensions(manifest=manifest, context=context)


def _default_requested_lanes(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> List[str]:
    raw_capture_mode = manifest.get("capture_mode") if isinstance(manifest.get("capture_mode"), Mapping) else {}
    capture_mode = (
        _normalized_capture_mode(
            manifest=manifest,
            arkit_poses_uri=None,
            arkit_intrinsics_uri=None,
            arkit_depth_prefix_uri=None,
            intake_complete=bool(context.get("intake_complete") or False),
            evidence_tier=str(context.get("evidence_tier") or ""),
            capture_source=str(context.get("capture_source") or manifest.get("capture_source") or "iphone"),
            pose_match_rate=try_parse_float(context.get("pose_match_rate")),
            p95_pose_delta_sec=try_parse_float(context.get("p95_pose_delta_sec")),
            geometry_ready=bool(context.get("geometry_ready") or False),
            geometry_source=context.get("geometry_source"),
        )
        if isinstance(manifest, Mapping)
        else {}
    )
    scene_memory_capture = (
        manifest.get("scene_memory_capture")
        if isinstance(manifest.get("scene_memory_capture"), Mapping)
        else {}
    )
    native_default_candidate = (
        str((raw_capture_mode or {}).get("resolved_mode") or (capture_mode or {}).get("resolved_mode") or "").strip().lower() == "site_world_candidate"
        and bool(scene_memory_capture.get("world_model_candidate"))
    )
    current_default_lanes = ["qualification", "evaluation_prep", "simulation_automation"]
    legacy_scene_memory_lanes = ["qualification", "scene_memory"]
    requested_outputs = _normalized_requested_outputs(manifest, context)
    if not requested_outputs:
        beta_default = str(
            context.get("sim_only_beta_default_task_eval")
            or manifest.get("sim_only_beta_default_task_eval")
            or os.getenv("BLUEPRINT_SIM_ONLY_BETA_DEFAULT_TASK_EVAL")
            or os.getenv("BLUEPRINT_SIM_ONLY_BETA_AUTONOMY")
            or ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        return current_default_lanes if native_default_candidate or beta_default else ["qualification"]

    lanes: List[str] = []

    def append_lanes(values: List[str]) -> None:
        for lane in values:
            if lane not in lanes:
                lanes.append(lane)

    for output in requested_outputs:
        lowered = str(output).strip().lower()
        if lowered == "qualification":
            if "qualification" not in lanes:
                lanes.append("qualification")
        elif lowered in {"preview", "preview_simulation", "managed_tuning", "data_licensing"}:
            append_lanes(current_default_lanes)
        elif lowered == "robot_eval_dataset":
            append_lanes(["qualification", "evaluation_prep"])
        elif lowered == "task_evaluation_run":
            append_lanes(current_default_lanes)
        elif lowered == "scene_memory":
            append_lanes(legacy_scene_memory_lanes)
        elif lowered in {"deeper_evaluation", "evaluation_prep"}:
            append_lanes(current_default_lanes)
        elif lowered == "review_intake":
            if "qualification" not in lanes:
                lanes.append("qualification")
    return lanes or ["qualification"]


def _requested_lanes_override(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
) -> List[str]:
    for raw in (
        context.get("requestedLanes"),
        context.get("requested_lanes"),
        manifest.get("requested_lanes"),
        manifest.get("requestedLanes"),
    ):
        normalized = _normalize_requested_lanes(raw)
        if normalized != ["qualification"] or raw is not None:
            return normalized
    return []


def _raw_video_candidates(raw_root: Path) -> List[str]:
    names = [
        "walkthrough.mov",
        "walkthrough.mp4",
        "recording.mov",
        "recording.mp4",
    ]
    out: List[str] = []
    for name in names:
        path = raw_root / name
        if path.is_file():
            out.append(name)
    return out


def capture_materialization_readiness(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    gcs_root: Path,
    raw_prefix_uri: Optional[str] = None,
) -> Dict[str, Any]:
    raw_prefix_uri = raw_prefix_uri or f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw"
    raw_root = resolve_gs_uri_to_path(raw_prefix_uri, gcs_root)
    manifest_path = raw_root / "manifest.json"
    manifest = _merge_manifest_with_sidecars(_read_optional_json(manifest_path), raw_root)
    requested_outputs = _normalized_requested_outputs(manifest, {})
    walkthrough_path = raw_root / "walkthrough.mov"
    video_candidates = _raw_video_candidates(raw_root)
    selected_video_path = raw_root / video_candidates[0] if video_candidates else None
    issues: List[str] = []
    if not manifest_path.is_file():
        issues.append("missing_manifest")
    elif not manifest:
        issues.append("invalid_manifest")
    if not video_candidates:
        issues.append("missing_raw_video")
    return {
        "ready": not issues,
        "issues": issues,
        "raw_root": str(raw_root),
        "manifest_path": str(manifest_path),
        "walkthrough_path": str(walkthrough_path),
        "selected_video_path": str(selected_video_path) if selected_video_path else None,
        "requested_outputs": requested_outputs,
        "video_candidates": video_candidates,
    }


def assert_capture_materialization_ready(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    gcs_root: Path,
    raw_prefix_uri: Optional[str] = None,
) -> Dict[str, Any]:
    readiness = capture_materialization_readiness(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=gcs_root,
        raw_prefix_uri=raw_prefix_uri,
    )
    if readiness["ready"]:
        return readiness
    raise RuntimeError(
        "capture_not_ready:" + ",".join(str(item) for item in readiness["issues"])
    )


def _capture_source(manifest: Mapping[str, Any], context: Mapping[str, Any]) -> str:
    profile = str(manifest.get("capture_profile_id") or context.get("captureProfileId") or "").strip().lower()
    if profile.startswith("android_"):
        return "android"
    if profile.startswith("glasses_"):
        return "glasses"
    if profile.startswith("iphone_"):
        return "iphone"
    for candidate in (
        str(manifest.get("capture_source") or "").strip().lower(),
        str(context.get("captureSource") or "").strip().lower(),
    ):
        if candidate in {"meta_glasses", "metaglasses", "rayban_meta", "ray-ban_meta"}:
            return "glasses"
        if candidate in {"iphone", "glasses", "android"}:
            return candidate
        if candidate == "android_phone":
            return "android"
        if candidate == "iphonevideo":
            return "iphone"
        if candidate == "metaglasses":  # pragma: no cover - kept for readability; handled by alias set above.
            return "glasses"
    return "unknown"


def _source_device(manifest: Mapping[str, Any], context: Mapping[str, Any], source: str) -> str:
    for value in (
        manifest.get("source_device"),
        manifest.get("device_source"),
        context.get("sourceDevice"),
        context.get("source_device"),
        manifest.get("capture_source"),
        context.get("captureSource"),
    ):
        text = str(value or "").strip().lower()
        if text in {"meta_glasses", "metaglasses", "rayban_meta", "ray-ban_meta"}:
            return "meta_glasses"
        if text:
            return text
    if source == "glasses":
        return "non_arkit_video"
    return source or "unknown"


def _capture_tier(source: str, manifest: Mapping[str, Any]) -> str:
    tier = str(manifest.get("capture_tier_hint") or "").strip()
    if tier.lower() == "tier2_android_phone":
        return "tier2_android"
    if tier:
        return tier
    if source == "glasses":
        return "tier2_glasses"
    if source == "android":
        return "tier2_android"
    return "tier1_iphone"


def _capture_modality(
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    source: str,
    scaffolding_used: List[str],
    has_metric_arkit_bundle: bool,
) -> str:
    explicit = str(context.get("captureModality") or manifest.get("capture_modality") or "").strip().lower()
    explicit_profile = str(manifest.get("capture_profile_id") or "").strip().lower()
    if explicit == "glasses_video_only" and explicit_profile == "glasses_pov":
        return explicit_profile
    if explicit in {
        "iphone_arkit_lidar",
        "iphone_arkit_non_lidar",
        "iphone_video_only",
        "android_arcore_depth",
        "android_arcore_pose_only",
        "glasses_video_only",
        "glasses_pov",
        "glasses_pov_companion_phone",
        "glasses_plus_scaffolding",
        "android_video_only",
        "android_plus_scaffolding",
    }:
        return explicit
    if explicit_profile in {
        "iphone_arkit_lidar",
        "iphone_arkit_non_lidar",
        "android_arcore_depth",
        "android_arcore_pose_only",
        "android_camera_only",
        "glasses_pov",
        "glasses_pov_companion_phone",
    }:
        return explicit_profile
    if source == "iphone":
        if has_metric_arkit_bundle or parse_bool(manifest.get("has_lidar"), default=False):
            return "iphone_arkit_lidar"
        return "iphone_video_only"
    if source == "glasses" and scaffolding_used:
        return "glasses_plus_scaffolding"
    if source == "glasses":
        return "glasses_video_only"
    if source == "android" and scaffolding_used:
        return "android_plus_scaffolding"
    if source == "android":
        return "android_video_only"
    return "android_video_only" if source == "android" else "iphone_video_only"


def _has_minimum_intake(intake: Mapping[str, Any]) -> bool:
    return bool(
        str(intake.get("workflowName") or "").strip()
        and _string_list(intake.get("taskSteps"))
        and (
            str(intake.get("zone") or "").strip()
            or str(intake.get("owner") or "").strip()
        )
    )


def _evidence_tier(
    *,
    source: str,
    modality: str,
    intake_complete: bool,
    calibration_assets: List[str],
    scaffolding_validation: Mapping[str, Any],
) -> str:
    if source in {"glasses", "android"}:
        if (
            modality in {"glasses_plus_scaffolding", "android_plus_scaffolding"}
            and intake_complete
            and calibration_assets
            and parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False)
        ):
            return "video_with_validated_scaffolding"
        return "pre_screen_video"
    if modality == "iphone_arkit_lidar" and intake_complete:
        return "qualified_metric_capture"
    return "pre_screen_video"


def materialize_capture_bundle(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    gcs_root: Path,
    raw_prefix_uri: Optional[str] = None,
) -> Dict[str, Any]:
    result = build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=gcs_root,
        raw_prefix_uri=raw_prefix_uri,
    )
    capture_root = resolve_gs_uri_to_path(str(result["descriptor_uri"]), gcs_root).parent
    write_json(capture_root / "capture_descriptor.json", result["descriptor"])
    write_json(capture_root / "qa_report.json", result["qa_report"])
    return result


def build_capture_bundle_records(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    gcs_root: Path,
    raw_prefix_uri: Optional[str] = None,
    write_frames_index: bool = True,
) -> Dict[str, Any]:
    raw_prefix_uri = raw_prefix_uri or f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw"
    assert_capture_materialization_ready(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=gcs_root,
        raw_prefix_uri=raw_prefix_uri,
    )
    raw_root = resolve_gs_uri_to_path(raw_prefix_uri, gcs_root)
    capture_root = raw_root.parent

    manifest_path = raw_root / "manifest.json"
    intake_path = raw_root / "intake_packet.json"
    task_hypothesis_path = raw_root / "task_hypothesis.json"
    context_path = raw_root / "capture_context.json"

    manifest = _merge_manifest_with_sidecars(_read_optional_json(manifest_path), raw_root)
    intake = _read_optional_json(intake_path)
    task_hypothesis = _read_optional_json(task_hypothesis_path)
    context = _read_optional_json(context_path)
    capture_orientation = _resolve_capture_orientation(
        manifest=manifest,
        context=context,
        raw_root=raw_root,
    )

    source = _capture_source(manifest, context)
    source_device = _source_device(manifest, context, source)
    tier = _capture_tier(source, manifest)
    scaffolding_used = _string_list(context.get("scaffoldingUsed") or manifest.get("scaffolding_used"))
    coverage_plan = _string_list(context.get("coveragePlan") or manifest.get("coverage_plan"))
    calibration_assets = _string_list(context.get("calibrationAssets") or manifest.get("calibration_assets"))
    uncertainty_priors = _dict_float(context.get("uncertaintyPriors") or manifest.get("uncertainty_priors"))

    arkit_root = raw_root / "arkit"
    arcore_root = raw_root / "arcore"
    companion_phone_root = raw_root / "companion_phone"
    has_metric_arkit_bundle = bool(
        (arkit_root / "poses.jsonl").is_file()
        and (arkit_root / "intrinsics.json").is_file()
        and (arkit_root / "depth").is_dir()
    )
    modality = _capture_modality(
        manifest,
        context,
        source,
        scaffolding_used,
        has_metric_arkit_bundle,
    )
    arkit_poses_uri = None
    arkit_intrinsics_uri = None
    arkit_frames_uri = None
    arkit_depth_prefix_uri = None
    arkit_confidence_prefix_uri = None
    arcore_poses_uri = None
    arcore_intrinsics_uri = None
    arcore_frames_uri = None
    arcore_depth_manifest_uri = None
    arcore_confidence_manifest_uri = None
    arcore_depth_prefix_uri = None
    arcore_confidence_prefix_uri = None
    arcore_point_cloud_uri = None
    arcore_planes_uri = None
    arcore_tracking_state_uri = None
    arcore_light_estimates_uri = None
    companion_phone_poses_uri = None
    companion_phone_intrinsics_uri = None
    companion_phone_calibration_uri = None
    object_index_uri = None
    motion_log_uri = None
    if (arkit_root / "poses.jsonl").is_file():
        arkit_poses_uri = join_gs_uri(raw_prefix_uri, "arkit/poses.jsonl")
    if (arkit_root / "intrinsics.json").is_file():
        arkit_intrinsics_uri = join_gs_uri(raw_prefix_uri, "arkit/intrinsics.json")
    if (arkit_root / "frames.jsonl").is_file():
        arkit_frames_uri = join_gs_uri(raw_prefix_uri, "arkit/frames.jsonl")
    if (arkit_root / "depth").is_dir():
        arkit_depth_prefix_uri = join_gs_uri(raw_prefix_uri, "arkit/depth")
    if (arkit_root / "confidence").is_dir():
        arkit_confidence_prefix_uri = join_gs_uri(raw_prefix_uri, "arkit/confidence")
    if (arcore_root / "poses.jsonl").is_file():
        arcore_poses_uri = join_gs_uri(raw_prefix_uri, "arcore/poses.jsonl")
    if (arcore_root / "session_intrinsics.json").is_file():
        arcore_intrinsics_uri = join_gs_uri(raw_prefix_uri, "arcore/session_intrinsics.json")
    if (arcore_root / "frames.jsonl").is_file():
        arcore_frames_uri = join_gs_uri(raw_prefix_uri, "arcore/frames.jsonl")
    if (arcore_root / "depth_manifest.json").is_file():
        arcore_depth_manifest_uri = join_gs_uri(raw_prefix_uri, "arcore/depth_manifest.json")
    if (arcore_root / "confidence_manifest.json").is_file():
        arcore_confidence_manifest_uri = join_gs_uri(raw_prefix_uri, "arcore/confidence_manifest.json")
    if (arcore_root / "depth").is_dir():
        arcore_depth_prefix_uri = join_gs_uri(raw_prefix_uri, "arcore/depth")
    if (arcore_root / "confidence").is_dir():
        arcore_confidence_prefix_uri = join_gs_uri(raw_prefix_uri, "arcore/confidence")
    if (arcore_root / "point_cloud.jsonl").is_file():
        arcore_point_cloud_uri = join_gs_uri(raw_prefix_uri, "arcore/point_cloud.jsonl")
    if (arcore_root / "planes.jsonl").is_file():
        arcore_planes_uri = join_gs_uri(raw_prefix_uri, "arcore/planes.jsonl")
    if (arcore_root / "tracking_state.jsonl").is_file():
        arcore_tracking_state_uri = join_gs_uri(raw_prefix_uri, "arcore/tracking_state.jsonl")
    if (arcore_root / "light_estimates.jsonl").is_file():
        arcore_light_estimates_uri = join_gs_uri(raw_prefix_uri, "arcore/light_estimates.jsonl")
    if (companion_phone_root / "poses.jsonl").is_file():
        companion_phone_poses_uri = join_gs_uri(raw_prefix_uri, "companion_phone/poses.jsonl")
    if (companion_phone_root / "session_intrinsics.json").is_file():
        companion_phone_intrinsics_uri = join_gs_uri(raw_prefix_uri, "companion_phone/session_intrinsics.json")
    if (companion_phone_root / "calibration.json").is_file():
        companion_phone_calibration_uri = join_gs_uri(raw_prefix_uri, "companion_phone/calibration.json")
    if (raw_root / "object_index.json").is_file():
        object_index_uri = join_gs_uri(raw_prefix_uri, "object_index.json")
    if (raw_root / "motion.jsonl").is_file():
        motion_log_uri = join_gs_uri(raw_prefix_uri, "motion.jsonl")
    video_candidates = _raw_video_candidates(raw_root)
    raw_video_uri = (
        join_gs_uri(raw_prefix_uri, video_candidates[0])
        if video_candidates
        else str(manifest.get("video_uri") or "").strip()
        or None
    )
    frame_timestamps_uri = (
        join_gs_uri(raw_prefix_uri, "glasses/frame_timestamps.jsonl")
        if (raw_root / "glasses" / "frame_timestamps.jsonl").is_file()
        else None
    )
    stream_metadata_uri = (
        join_gs_uri(raw_prefix_uri, "glasses/stream_metadata.json")
        if (raw_root / "glasses" / "stream_metadata.json").is_file()
        else None
    )
    media_metadata = {
        "source_device": source_device,
        "capture_source": source,
        "original_video_uri": raw_video_uri,
        "original_video_path": str(raw_root / video_candidates[0]) if video_candidates else None,
        "frame_timestamps_uri": frame_timestamps_uri,
        "stream_metadata_uri": stream_metadata_uri,
        "video_metadata": {
            "width": try_parse_float(manifest.get("width"), 0.0),
            "height": try_parse_float(manifest.get("height"), 0.0),
            "fps_source": try_parse_float(manifest.get("fps_source"), 0.0),
            "capture_start_epoch_ms": try_parse_float(manifest.get("capture_start_epoch_ms"), 0.0),
        },
    }
    arkit_geometry_ready = bool(
        arkit_poses_uri is not None
        and arkit_intrinsics_uri is not None
        and arkit_depth_prefix_uri is not None
    )
    arcore_geometry_present = bool(arcore_poses_uri is not None and arcore_intrinsics_uri is not None)
    geometry_source = "arkit" if arkit_geometry_ready else "arcore" if arcore_geometry_present else None
    pose_alignment = _inspect_pose_alignment(raw_root)
    pose_match_rate = try_parse_float(
        manifest.get("pose_match_rate"),
        pose_alignment.get("pose_match_rate"),
    )
    p95_pose_delta_sec = try_parse_float(
        manifest.get("p95_pose_delta_sec"),
        pose_alignment.get("p95_pose_delta_sec"),
    )
    pose_alignment_ok = source != "iphone" or _iphone_pose_alignment_ok(
        pose_match_rate,
        p95_pose_delta_sec,
    )

    frames_index_uri = f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/frames/index.jsonl"
    frames_dir = capture_root / "frames"
    frames_path = frames_dir / "index.jsonl"
    frame_index_payload = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "raw_prefix_uri": raw_prefix_uri,
        "video_candidates": _raw_video_candidates(raw_root),
        "generated_at": utc_now_iso(),
    }
    if write_frames_index:
        ensure_dir(frames_dir)
        frames_path.write_text(json.dumps(frame_index_payload) + "\n", encoding="utf-8")

    intake_packet_uri = join_gs_uri(raw_prefix_uri, "intake_packet.json") if intake_path.is_file() else None
    task_hypothesis_uri = join_gs_uri(raw_prefix_uri, "task_hypothesis.json") if task_hypothesis_path.is_file() else None
    intake_complete = _has_minimum_intake(intake)
    validated_scale_raw = context.get("validatedScaleMeters") or manifest.get("validated_scale_m")
    validated_scale_m = None
    if validated_scale_raw is not None:
        validated_scale_m = try_parse_float(validated_scale_raw, 0.0)
    validated_pose_coverage = try_parse_float(
        context.get("validatedPoseCoverage") or manifest.get("validated_pose_coverage"),
        0.0,
    )
    hidden_zone_bound = try_parse_float(
        context.get("hiddenZoneBound") or manifest.get("hidden_zone_bound"),
        1.0,
    )
    scale_anchor_count = len(_string_list(context.get("scaleAnchorAssets") or manifest.get("scale_anchor_assets")))
    checkpoint_count = len(_string_list(context.get("checkpointAssets") or manifest.get("checkpoint_assets")))
    scaffolding_validation = {
        "scale_anchor_count": scale_anchor_count,
        "checkpoint_count": checkpoint_count,
        "validated_scale_m": validated_scale_m,
        "validated_pose_coverage": round(float(validated_pose_coverage or 0.0), 4),
        "hidden_zone_bound": round(float(hidden_zone_bound or 1.0), 4),
        "validated_metric_bundle": bool(
            modality in {"glasses_plus_scaffolding", "android_plus_scaffolding"}
            and calibration_assets
            and validated_scale_m is not None
            and float(validated_pose_coverage or 0.0) >= 0.7
            and float(hidden_zone_bound or 1.0) <= 0.35
            and scale_anchor_count > 0
            and checkpoint_count > 0
        ),
    }
    evidence_tier = _evidence_tier(
        source=source,
        modality=modality,
        intake_complete=intake_complete,
        calibration_assets=calibration_assets,
        scaffolding_validation=scaffolding_validation,
    )
    normalized_site_identity = _normalized_site_identity(manifest)
    normalized_capture_topology = _normalized_capture_topology(manifest)
    normalized_route_anchors = _normalized_route_anchors(manifest.get("route_anchors"))
    normalized_checkpoint_events = _normalized_checkpoint_events(manifest.get("checkpoint_events"))
    normalized_relocalization_events = _normalized_relocalization_events(manifest.get("relocalization_events"))

    capture_capabilities = manifest.get("capture_capabilities") if isinstance(manifest.get("capture_capabilities"), Mapping) else {}
    upstream_handoff = manifest.get("upstream_handoff") if isinstance(manifest.get("upstream_handoff"), Mapping) else {}
    site_submission_id = str(
        manifest.get("site_submission_id")
        or upstream_handoff.get("site_submission_id")
        or ""
    ).strip() or None
    buyer_request_id = str(
        manifest.get("buyer_request_id")
        or upstream_handoff.get("buyer_request_id")
        or ""
    ).strip() or None
    capture_job_id = str(
        manifest.get("capture_job_id")
        or upstream_handoff.get("capture_job_id")
        or ""
    ).strip() or None
    upstream_link_blockers = [
        blocker
        for blocker, value in (
            ("missing_site_submission_id", site_submission_id),
            ("missing_buyer_request_id", buyer_request_id),
            ("missing_capture_job_id", capture_job_id),
        )
        if not value
    ]
    metadata: Dict[str, Any] = {
        "site_submission_id": site_submission_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
        "upstream_link_truth_state": "verified" if not upstream_link_blockers else "blocked_missing_upstream_ids",
        "upstream_link_blockers": upstream_link_blockers,
        "opportunity_id": scene_id,
        "task_statement": str(intake.get("workflowName") or manifest.get("scene_id") or scene_id),
        "workflow_context": " | ".join(_string_list(intake.get("taskSteps"))),
        "success_criteria": [str(intake.get("targetKPI") or "").strip()] if str(intake.get("targetKPI") or "").strip() else [],
        "task_zone": {"label": str(intake.get("zone") or "").strip()} if str(intake.get("zone") or "").strip() else {},
        "operating_constraints": [value for value in [str(intake.get("shift") or "").strip()] if value],
        "privacy_restrictions": _string_list(intake.get("privacySecurityLimits")),
        "security_restrictions": _string_list(intake.get("captureRestrictions")),
        "known_blockers": _string_list(intake.get("knownBlockers")),
        "owner": str(intake.get("owner") or "").strip() or None,
        "adjacent_systems": _string_list(intake.get("adjacentSystems")),
        "non_routine_modes": _string_list(intake.get("nonRoutineModes")),
        "people_traffic_notes": _string_list(intake.get("peopleTrafficNotes")),
        "capture_restrictions": _string_list(intake.get("captureRestrictions")),
        "capture_modality": modality,
        "evidence_tier": evidence_tier,
        "scaffolding_used": scaffolding_used,
        "coverage_plan": coverage_plan,
        "calibration_assets": calibration_assets,
        "uncertainty_priors": uncertainty_priors,
        "scaffolding_validation": scaffolding_validation,
        "task_hypothesis": task_hypothesis if task_hypothesis else None,
        "capture_rights": _capture_rights_block(manifest),
        "privacy_lineage": (
            dict(manifest.get("privacy_lineage"))
            if isinstance(manifest.get("privacy_lineage"), Mapping)
            else {"status": "unknown", "source": "raw_capture"}
        ),
        "provenance_lineage": (
            dict(manifest.get("provenance_lineage"))
            if isinstance(manifest.get("provenance_lineage"), Mapping)
            else {"source": "raw_capture_bundle"}
        ),
        "media_metadata": media_metadata,
        "capture_profile_id": str(manifest.get("capture_profile_id") or "").strip() or None,
        "capture_capabilities": dict(capture_capabilities) if isinstance(capture_capabilities, Mapping) else {},
        "capture_orientation": capture_orientation,
        "scene_memory_capture": {
            "continuity_score": 0.9 if raw_video_uri else 0.0,
            "lighting_consistency": "unknown",
            "dynamic_object_density": "unknown",
            "sensor_availability": {
                "arkit_poses": arkit_poses_uri is not None,
                "arkit_intrinsics": arkit_intrinsics_uri is not None,
                "arkit_depth": arkit_depth_prefix_uri is not None,
                "arkit_confidence": arkit_confidence_prefix_uri is not None,
                "depth_conditioning": arkit_depth_prefix_uri is not None,
                "camera_pose": arcore_poses_uri is not None,
                "camera_intrinsics": arcore_intrinsics_uri is not None,
                "depth": arcore_depth_manifest_uri is not None or arcore_depth_prefix_uri is not None,
                "depth_confidence": arcore_confidence_manifest_uri is not None or arcore_confidence_prefix_uri is not None,
                "point_cloud": arcore_point_cloud_uri is not None,
                "planes": arcore_planes_uri is not None,
                "tracking_state": arcore_tracking_state_uri is not None,
                "light_estimate": arcore_light_estimates_uri is not None,
                "motion": motion_log_uri is not None,
                "companion_phone_pose": companion_phone_poses_uri is not None,
                "companion_phone_intrinsics": companion_phone_intrinsics_uri is not None,
                "companion_phone_calibration": companion_phone_calibration_uri is not None,
            },
            "operator_notes": [],
            "world_model_candidate": _canonical_world_model_candidate(
                manifest=manifest,
                arkit_poses_uri=arkit_poses_uri,
                arkit_intrinsics_uri=arkit_intrinsics_uri,
                arkit_depth_prefix_uri=arkit_depth_prefix_uri,
                intake_complete=intake_complete,
                evidence_tier=evidence_tier,
                capture_source=source,
                pose_match_rate=pose_match_rate,
                p95_pose_delta_sec=p95_pose_delta_sec,
                geometry_ready=arkit_geometry_ready,
                geometry_source=geometry_source,
            ),
            "world_model_candidate_reasoning": _world_model_candidate_reasoning(
                manifest=manifest,
                arkit_poses_uri=arkit_poses_uri,
                arkit_intrinsics_uri=arkit_intrinsics_uri,
                arkit_depth_prefix_uri=arkit_depth_prefix_uri,
                intake_complete=intake_complete,
                capture_source=source,
                pose_match_rate=pose_match_rate,
                p95_pose_delta_sec=p95_pose_delta_sec,
                geometry_ready=arkit_geometry_ready,
                geometry_source=geometry_source,
            ),
            "geometry_source": geometry_source,
            "geometry_ready": arkit_geometry_ready,
        },
        "site_identity": normalized_site_identity,
        "capture_topology": normalized_capture_topology,
        "route_anchors": normalized_route_anchors,
        "checkpoint_events": normalized_checkpoint_events,
        "relocalization_events": normalized_relocalization_events,
        "capture_mode": _normalized_capture_mode(
            manifest=manifest,
            arkit_poses_uri=arkit_poses_uri,
            arkit_intrinsics_uri=arkit_intrinsics_uri,
            arkit_depth_prefix_uri=arkit_depth_prefix_uri,
            intake_complete=intake_complete,
            evidence_tier=evidence_tier,
            capture_source=source,
            pose_match_rate=pose_match_rate,
            p95_pose_delta_sec=p95_pose_delta_sec,
            geometry_ready=arkit_geometry_ready,
            geometry_source=geometry_source,
        ),
    }

    descriptor = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_source": source,
        "source_device": source_device,
        "capture_profile_id": str(manifest.get("capture_profile_id") or "").strip() or None,
        "capture_capabilities": dict(capture_capabilities) if isinstance(capture_capabilities, Mapping) else {},
        "capture_tier": tier,
        "capture_modality": modality,
        "evidence_tier": evidence_tier,
        "raw_prefix_uri": raw_prefix_uri,
        "frames_index_uri": frames_index_uri,
        "raw_video_uri": raw_video_uri,
        "media_metadata": media_metadata,
        "privacy_processed_video_uri": None,
        "world_model_video_uri": None,
        "privacy_status": "not_run",
        "privacy_mode": "none",
        "privacy_manifest_uri": None,
        "arkit_poses_uri": arkit_poses_uri,
        "arkit_intrinsics_uri": arkit_intrinsics_uri,
        "arkit_frames_uri": arkit_frames_uri,
        "arkit_depth_prefix_uri": arkit_depth_prefix_uri,
        "arkit_confidence_prefix_uri": arkit_confidence_prefix_uri,
        "arcore_poses_uri": arcore_poses_uri,
        "arcore_intrinsics_uri": arcore_intrinsics_uri,
        "arcore_frames_uri": arcore_frames_uri,
        "arcore_depth_manifest_uri": arcore_depth_manifest_uri,
        "arcore_confidence_manifest_uri": arcore_confidence_manifest_uri,
        "arcore_depth_prefix_uri": arcore_depth_prefix_uri,
        "arcore_confidence_prefix_uri": arcore_confidence_prefix_uri,
        "arcore_point_cloud_uri": arcore_point_cloud_uri,
        "arcore_planes_uri": arcore_planes_uri,
        "arcore_tracking_state_uri": arcore_tracking_state_uri,
        "arcore_light_estimates_uri": arcore_light_estimates_uri,
        "companion_phone_poses_uri": companion_phone_poses_uri,
        "companion_phone_intrinsics_uri": companion_phone_intrinsics_uri,
        "companion_phone_calibration_uri": companion_phone_calibration_uri,
        "depth_conditioning": (
            {
                "status": "available",
                "source": "arkit",
                "provider": "arkit",
                "depth_prefix_uri": arkit_depth_prefix_uri,
                "confidence_prefix_uri": arkit_confidence_prefix_uri,
                "depth_manifest_uri": None,
                "confidence_manifest_uri": None,
            }
            if arkit_depth_prefix_uri
            else {
                "status": "available",
                "source": "arcore",
                "provider": "arcore",
                "depth_prefix_uri": arcore_depth_prefix_uri,
                "confidence_prefix_uri": arcore_confidence_prefix_uri,
                "depth_manifest_uri": arcore_depth_manifest_uri,
                "confidence_manifest_uri": arcore_confidence_manifest_uri,
            }
            if arcore_depth_manifest_uri or arcore_depth_prefix_uri
            else {}
        ),
        "geometry_source": geometry_source,
        "geometry_ready": arkit_geometry_ready,
        "coordinate_frame_session_id": (
            (normalized_capture_topology or {}).get("capture_session_id")
            if isinstance(normalized_capture_topology, Mapping)
            else None
        ),
        "object_index_uri": object_index_uri,
        "motion_log_uri": motion_log_uri,
        "qa_report_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/qa_report.json",
        "qa_status": None,
        "intended_space_type": str(manifest.get("intended_space_type") or "default"),
        "scaffolding_used": scaffolding_used,
        "intake_packet_uri": intake_packet_uri,
        "task_hypothesis_uri": task_hypothesis_uri,
        "coverage_plan": coverage_plan,
        "calibration_assets": calibration_assets,
        "scaffolding_validation": scaffolding_validation,
        "uncertainty_priors": uncertainty_priors,
        "capture_orientation": capture_orientation,
        "requested_lanes": (
            _requested_lanes_override(manifest, context)
            or _default_requested_lanes(manifest, context)
        ),
        "requested_outputs": _normalized_requested_outputs(manifest, context),
        "site_identity": normalized_site_identity,
        "capture_topology": normalized_capture_topology,
        "route_anchors": normalized_route_anchors,
        "checkpoint_events": normalized_checkpoint_events,
        "relocalization_events": normalized_relocalization_events,
        "site_submission_id": site_submission_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
        "upstream_link_truth_state": metadata["upstream_link_truth_state"],
        "upstream_link_blockers": upstream_link_blockers,
        "quality": {
            "pose_match_rate": pose_match_rate,
            "p95_pose_delta_sec": p95_pose_delta_sec,
            "pose_alignment_ok": pose_alignment_ok,
            "has_metric_geometry": evidence_tier in {"qualified_metric_capture", "video_with_validated_scaffolding"},
            "intake_complete": intake_complete,
            "world_model_candidate": _canonical_world_model_candidate(
                manifest=manifest,
                arkit_poses_uri=arkit_poses_uri,
                arkit_intrinsics_uri=arkit_intrinsics_uri,
                arkit_depth_prefix_uri=arkit_depth_prefix_uri,
                intake_complete=intake_complete,
                evidence_tier=evidence_tier,
                capture_source=source,
                pose_match_rate=pose_match_rate,
                p95_pose_delta_sec=p95_pose_delta_sec,
                geometry_ready=arkit_geometry_ready,
                geometry_source=geometry_source,
            ),
            "geometry_source": geometry_source,
            "geometry_ready": arkit_geometry_ready,
        },
        "metadata": metadata,
    }

    hidden_zone_score = min(
        1.0,
        0.2 * len(_string_list(intake.get("captureRestrictions")))
        + 0.15 * len(_string_list(intake.get("privacySecurityLimits"))),
    )
    uncertainty_score = 0.15 if modality == "iphone_arkit_lidar" else 0.45
    if modality in {"glasses_video_only", "android_video_only"}:
        uncertainty_score = 0.78
    if not intake_complete:
        uncertainty_score = min(1.0, uncertainty_score + 0.15)
    if not raw_video_uri:
        uncertainty_score = min(1.0, uncertainty_score + 0.25)
    if modality in {"glasses_plus_scaffolding", "android_plus_scaffolding"} and not parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False):
        uncertainty_score = min(1.0, uncertainty_score + 0.2)
    if hidden_zone_bound is not None:
        uncertainty_score = min(1.0, uncertainty_score + max(0.0, float(hidden_zone_bound) - 0.2) * 0.4)

    checks = [
        {
            "name": "raw_manifest_present",
            "passed": manifest_path.is_file(),
            "detail": "raw manifest present" if manifest_path.is_file() else "raw manifest missing",
        },
        {
            "name": "raw_video_present",
            "passed": bool(raw_video_uri),
            "detail": raw_video_uri or "raw video missing",
        },
        {
            "name": "intake_present",
            "passed": intake_path.is_file(),
            "detail": "intake packet present" if intake_path.is_file() else "intake packet missing",
        },
        {
            "name": "intake_complete",
            "passed": intake_complete,
            "detail": "intake has workflow, steps, and zone/owner" if intake_complete else "intake missing workflow, steps, or zone/owner",
        },
        {
            "name": "metric_geometry_present",
            "passed": evidence_tier in {"qualified_metric_capture", "video_with_validated_scaffolding"},
            "detail": "validated metric evidence present" if evidence_tier in {"qualified_metric_capture", "video_with_validated_scaffolding"} else "metric geometry not present",
        },
        {
            "name": "scaffolding_validated",
            "passed": modality not in {"glasses_plus_scaffolding", "android_plus_scaffolding"} or parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False),
            "detail": "scaffolding validated for metric checks" if modality not in {"glasses_plus_scaffolding", "android_plus_scaffolding"} or parse_bool(scaffolding_validation.get("validated_metric_bundle"), default=False) else "video scaffolding lacks validated scale/pose coverage",
        },
    ]

    if evidence_tier == "qualified_metric_capture":
        status = "passed" if manifest_path.is_file() and raw_video_uri and intake_complete else "degraded"
    elif evidence_tier == "video_with_validated_scaffolding":
        status = "passed" if manifest_path.is_file() and raw_video_uri and intake_complete else "degraded"
    else:
        status = "degraded"

    recommended_lane = (
        "current"
        if "simulation_automation" in descriptor["requested_lanes"]
        else "scene_memory"
        if "scene_memory" in descriptor["requested_lanes"]
        else "qualification"
    )

    qa_report = {
        "schema_version": "v1",
        "scene_id": scene_id,
        "capture_id": capture_id,
        "generated_at": utc_now_iso(),
        "status": status,
        "capture_modality": modality,
        "evidence_tier": evidence_tier,
        "uncertainty_score": round(uncertainty_score, 4),
        "hidden_zone_score": round(hidden_zone_score, 4),
        "hidden_zone_bound": round(float(hidden_zone_bound or 1.0), 4),
        "scaffolding_validation": scaffolding_validation,
        "checks": checks,
        "escalation_recommendation": {
            "recommended_lane": recommended_lane if status == "passed" else "qualification",
            "human_review_required": evidence_tier != "qualified_metric_capture" or uncertainty_score >= 0.3,
            "reason": (
                "validated metric capture supports scene-memory derivation and explicit geometry conditioning"
                if evidence_tier in {"qualified_metric_capture", "video_with_validated_scaffolding"}
                else "capture remains pre-screen only because metric evidence is incomplete"
            ),
        },
        "scene_memory_readiness": {
            "world_model_candidate": _canonical_world_model_candidate(
                manifest=manifest,
                arkit_poses_uri=arkit_poses_uri,
                arkit_intrinsics_uri=arkit_intrinsics_uri,
                arkit_depth_prefix_uri=arkit_depth_prefix_uri,
                intake_complete=intake_complete,
                evidence_tier=evidence_tier,
            ),
            "recommended_lane": recommended_lane,
            "derived_only": True,
        },
    }

    return {
        "descriptor_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/capture_descriptor.json",
        "qa_report_uri": f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/qa_report.json",
        "descriptor": descriptor,
        "qa_report": qa_report,
    }


def preview_capture_bundle(
    *,
    bucket: str,
    scene_id: str,
    capture_id: str,
    gcs_root: Path,
    raw_prefix_uri: Optional[str] = None,
) -> Dict[str, Any]:
    return build_capture_bundle_records(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        gcs_root=gcs_root,
        raw_prefix_uri=raw_prefix_uri,
        write_frames_index=False,
    )
