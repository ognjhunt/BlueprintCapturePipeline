"""Capture descriptor contracts for site-world orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_ALLOWED_SWAP_FOCUS = {
    "default",
    "bedroom",
    "kitchen",
    "warehouse",
    "industrial_unknown",
    "fulfillment",
    "manufacturing",
    "brownfield_site",
}
_ALLOWED_ENVIRONMENT_HINTS = set(_ALLOWED_SWAP_FOCUS)
_ALLOWED_REQUESTED_LANES = {
    "qualification",
    "scene_memory",
    "retrieval_index",
    "frame_alignment",
    "evaluation_prep",
    "synthesis_coverage_validation",
}
_ALLOWED_CAPTURE_MODALITIES = {
    "iphone_arkit_lidar",
    "iphone_video_only",
    "glasses_video_only",
    "glasses_plus_scaffolding",
    "android_video_only",
    "android_plus_scaffolding",
    "android_xr_video_only",
}
_ALLOWED_EVIDENCE_TIERS = {
    "pre_screen_video",
    "qualified_metric_capture",
    "video_with_validated_scaffolding",
}
_ALLOWED_DISPLAY_ORIENTATIONS = {"portrait", "landscape", "square", "unknown"}
_ANDROID_XR_VIDEO_ONLY_PROFILE = "android_xr_glasses"
_ANDROID_XR_VIDEO_ONLY_MODALITY = "android_xr_video_only"
_ANDROID_XR_FALSE_CAPABILITY_KEYS = (
    "camera_pose",
    "camera_intrinsics",
    "depth",
    "depth_confidence",
    "point_cloud",
    "planes",
    "tracking_state",
    "light_estimate",
    "geospatial",
    "motion_authoritative",
    "geometry_expected_downstream",
    "world_model_ready",
    "provider_ready",
    "hosted_session_ready",
    "payout_ready",
)
_ANDROID_XR_ZERO_CAPABILITY_KEYS = (
    "pose_rows",
    "depth_frames",
    "confidence_frames",
    "point_cloud_samples",
    "plane_rows",
    "tracking_state_rows",
    "light_estimate_rows",
    "geospatial_rows",
)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _normalize_mapping(raw_value: Any) -> Dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _is_android_xr_video_only(*, capture_profile_id: Any, capture_modality: Any) -> bool:
    profile = str(capture_profile_id or "").strip().lower()
    modality = str(capture_modality or "").strip().lower()
    return profile.startswith("android_xr") or modality == _ANDROID_XR_VIDEO_ONLY_MODALITY


def _sanitize_android_xr_video_only_capabilities(raw_value: Any) -> Dict[str, Any]:
    capabilities = _normalize_mapping(raw_value)
    for key in _ANDROID_XR_FALSE_CAPABILITY_KEYS:
        capabilities[key] = False
    for key in _ANDROID_XR_ZERO_CAPABILITY_KEYS:
        capabilities[key] = 0
    return capabilities


def _sanitize_android_xr_video_only_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized = dict(metadata)
    scene_memory = _normalize_mapping(sanitized.get("scene_memory_capture"))
    scene_memory["world_model_candidate"] = False
    scene_memory["geometry_expected_downstream"] = False
    scene_memory["geometry_ready"] = False
    scene_memory["geometry_source"] = None
    sanitized["scene_memory_capture"] = scene_memory

    capture_rights = _normalize_mapping(sanitized.get("capture_rights"))
    capture_rights["capture_contributor_payout_eligible"] = False
    sanitized["capture_rights"] = capture_rights
    return sanitized


def _sanitize_android_xr_video_only_quality(quality: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized = dict(quality)
    sanitized["world_model_candidate"] = False
    sanitized["geometry_ready"] = False
    sanitized["provider_ready"] = False
    sanitized["hosted_session_ready"] = False
    sanitized["payout_ready"] = False
    sanitized["world_model_ready"] = False
    sanitized["geometry_source"] = None
    return sanitized


def _normalize_environment_hint(raw_environment: Any) -> Optional[str]:
    text = _optional_str(raw_environment)
    if text is None:
        return None
    lowered = text.strip().lower()
    aliases = {
        "auto": "default",
        "bed room": "bedroom",
        "bed-room": "bedroom",
        "livingroom": "default",
        "living_room": "default",
        "residential": "default",
        "home": "default",
        "industrial": "industrial_unknown",
        "factory": "manufacturing",
        "plant": "manufacturing",
        "warehouse_floor": "warehouse",
    }
    lowered = aliases.get(lowered, lowered)
    if lowered in _ALLOWED_ENVIRONMENT_HINTS:
        return lowered
    return lowered


def _normalize_swap_focus(raw_swap_focus: Any) -> List[str]:
    if raw_swap_focus is None:
        return []
    if isinstance(raw_swap_focus, str):
        values = [raw_swap_focus]
    elif isinstance(raw_swap_focus, (list, tuple, set)):
        values = [str(v) for v in raw_swap_focus]
    else:
        values = [str(raw_swap_focus)]

    normalized: List[str] = []
    for value in values:
        lowered = value.strip().lower()
        if lowered in _ALLOWED_SWAP_FOCUS and lowered not in normalized:
            normalized.append(lowered)
    return normalized


def _normalize_requested_lanes(raw_requested_lanes: Any) -> List[str]:
    if raw_requested_lanes is None:
        return ["qualification"]
    if isinstance(raw_requested_lanes, str):
        values = [raw_requested_lanes]
    elif isinstance(raw_requested_lanes, (list, tuple, set)):
        values = [str(v) for v in raw_requested_lanes]
    else:
        values = [str(raw_requested_lanes)]

    normalized: List[str] = []
    for value in values:
        lowered = value.strip().lower()
        if not lowered:
            continue
        if lowered == "all":
            for lane in (
                "qualification",
                "scene_memory",
                "retrieval_index",
                "frame_alignment",
                "evaluation_prep",
                "synthesis_coverage_validation",
            ):
                if lane not in normalized:
                    normalized.append(lane)
            continue
        if lowered in _ALLOWED_REQUESTED_LANES and lowered not in normalized:
            normalized.append(lowered)
            if lowered in {"retrieval_index", "frame_alignment", "evaluation_prep"} and "qualification" not in normalized:
                normalized.append("qualification")
    if (
        {"retrieval_index", "frame_alignment", "evaluation_prep"} & set(normalized)
        and "qualification" not in normalized
    ):
        normalized.append("qualification")
    ordered: List[str] = []
    for lane in (
        "qualification",
        "scene_memory",
        "retrieval_index",
        "frame_alignment",
        "evaluation_prep",
        "synthesis_coverage_validation",
    ):
        if lane in normalized and lane not in ordered:
            ordered.append(lane)
    return ordered or ["qualification"]


def _normalize_capture_tier(raw_capture_tier: Any) -> str:
    tier = str(raw_capture_tier or "").strip()
    if not tier:
        return "tier2_glasses"
    lowered = tier.lower()
    if lowered == "tier2_android_phone":
        return "tier2_android"
    return tier


def _infer_capture_source(raw_source: str, capture_tier: str) -> str:
    source = raw_source.strip().lower()
    if source == "android_phone":
        return "android"
    if source in {"iphone", "glasses", "android"}:
        return source
    tier = capture_tier.strip().lower()
    if "glasses" in tier:
        return "glasses"
    if "android" in tier:
        return "android"
    return "iphone"


def _normalize_string_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = [str(v) for v in raw_value]
    else:
        values = [str(raw_value)]

    normalized: List[str] = []
    for value in values:
        text = value.strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_uncertainty_priors(raw_value: Any) -> Dict[str, float]:
    if not isinstance(raw_value, Mapping):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw_value.items():
        text = str(key).strip()
        if not text:
            continue
        try:
            out[text] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _normalize_scaffolding_validation(raw_value: Any) -> Dict[str, Any]:
    if not isinstance(raw_value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "scale_anchor_count",
        "checkpoint_count",
        "validated_scale_m",
        "validated_pose_coverage",
        "hidden_zone_bound",
    ):
        if raw_value.get(key) is None:
            continue
        try:
            out[key] = float(raw_value[key])
        except (TypeError, ValueError):
            continue
    if "validated_metric_bundle" in raw_value:
        out["validated_metric_bundle"] = bool(raw_value.get("validated_metric_bundle"))
    return out


def _normalize_orientation_size(raw_value: Any) -> Dict[str, int]:
    if not isinstance(raw_value, Mapping):
        return {}
    out: Dict[str, int] = {}
    for source_key, target_key in (
        ("width", "width"),
        ("w", "width"),
        ("height", "height"),
        ("h", "height"),
    ):
        if raw_value.get(source_key) is None:
            continue
        try:
            value = int(raw_value.get(source_key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            out[target_key] = value
    return out if {"width", "height"}.issubset(out) else {}


def _normalize_rotation_degrees(raw_value: Any) -> Optional[int]:
    if raw_value is None:
        return None
    try:
        value = int(round(float(raw_value)))
    except (TypeError, ValueError):
        return None
    normalized = value % 360
    for candidate in (0, 90, 180, 270):
        if abs(normalized - candidate) <= 1:
            return candidate
    return normalized


def _normalize_capture_orientation(raw_value: Any) -> Dict[str, Any]:
    if not isinstance(raw_value, Mapping):
        return {}
    out: Dict[str, Any] = {}
    encoded_width = _first_nonzero_int(
        raw_value.get("encoded_width"),
        ((raw_value.get("encoded_size") or raw_value.get("encodedSize")) or {}).get("width")
        if isinstance(raw_value.get("encoded_size") or raw_value.get("encodedSize"), Mapping)
        else None,
    )
    encoded_height = _first_nonzero_int(
        raw_value.get("encoded_height"),
        ((raw_value.get("encoded_size") or raw_value.get("encodedSize")) or {}).get("height")
        if isinstance(raw_value.get("encoded_size") or raw_value.get("encodedSize"), Mapping)
        else None,
    )
    declared_capture_width = _first_nonzero_int(
        raw_value.get("declared_capture_width"),
        raw_value.get("declaredCaptureWidth"),
        ((raw_value.get("display_size") or raw_value.get("displaySize")) or {}).get("width")
        if isinstance(raw_value.get("display_size") or raw_value.get("displaySize"), Mapping)
        else None,
    )
    declared_capture_height = _first_nonzero_int(
        raw_value.get("declared_capture_height"),
        raw_value.get("declaredCaptureHeight"),
        ((raw_value.get("display_size") or raw_value.get("displaySize")) or {}).get("height")
        if isinstance(raw_value.get("display_size") or raw_value.get("displaySize"), Mapping)
        else None,
    )
    display_orientation = _optional_str(
        raw_value.get("display_orientation") or raw_value.get("displayOrientation")
    )
    if display_orientation:
        lowered = display_orientation.lower()
        if lowered in _ALLOWED_DISPLAY_ORIENTATIONS:
            out["display_orientation"] = lowered
    rotation_degrees = _normalize_rotation_degrees(
        raw_value.get("display_rotation_degrees")
        or raw_value.get("displayRotationDegrees")
        or raw_value.get("rotation_degrees")
        or raw_value.get("rotationDegrees")
    )
    if rotation_degrees is not None:
        out["display_rotation_degrees"] = rotation_degrees
        out["rotation_degrees"] = rotation_degrees
    if encoded_width is not None:
        out["encoded_width"] = encoded_width
    if encoded_height is not None:
        out["encoded_height"] = encoded_height
    if declared_capture_width is not None:
        out["declared_capture_width"] = declared_capture_width
    if declared_capture_height is not None:
        out["declared_capture_height"] = declared_capture_height
    if "normalization_applied" in raw_value:
        out["normalization_applied"] = bool(raw_value.get("normalization_applied"))
    elif "normalizationApplied" in raw_value:
        out["normalization_applied"] = bool(raw_value.get("normalizationApplied"))
    source = _optional_str(raw_value.get("source"))
    if source:
        out["source"] = source
    display_size = _normalize_orientation_size(
        raw_value.get("display_size") or raw_value.get("displaySize")
    )
    if not display_size and declared_capture_width is not None and declared_capture_height is not None:
        display_size = {"width": declared_capture_width, "height": declared_capture_height}
    if display_size:
        out["display_size"] = display_size
    encoded_size = _normalize_orientation_size(
        raw_value.get("encoded_size") or raw_value.get("encodedSize")
    )
    if not encoded_size and encoded_width is not None and encoded_height is not None:
        encoded_size = {"width": encoded_width, "height": encoded_height}
    if encoded_size:
        out["encoded_size"] = encoded_size
    if "preserve_original_display_orientation" in raw_value:
        out["preserve_original_display_orientation"] = bool(
            raw_value.get("preserve_original_display_orientation")
        )
    elif "preserveOriginalDisplayOrientation" in raw_value:
        out["preserve_original_display_orientation"] = bool(
            raw_value.get("preserveOriginalDisplayOrientation")
        )
    if isinstance(raw_value.get("probe_details"), Mapping):
        out["probe_details"] = dict(raw_value.get("probe_details") or {})
    return out


def _first_nonzero_int(*values: Any) -> Optional[int]:
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


def _resolve_evidence_tier(raw_value: Any, capture_modality: str, quality: Mapping[str, Any]) -> str:
    explicit = _optional_str(raw_value)
    if explicit:
        lowered = explicit.lower()
        if lowered == "glasses_with_validated_scaffolding":
            return "video_with_validated_scaffolding"
        if lowered in _ALLOWED_EVIDENCE_TIERS:
            return lowered
    if capture_modality == "iphone_arkit_lidar":
        return "qualified_metric_capture"
    if capture_modality in {"glasses_plus_scaffolding", "android_plus_scaffolding"}:
        return "video_with_validated_scaffolding"
    return "pre_screen_video"


def _resolve_capture_modality(
    *,
    raw_modality: Any,
    capture_source: str,
    quality: Mapping[str, Any],
    scaffolding_used: List[str],
    has_metric_arkit_bundle: bool,
    evidence_tier_hint: Optional[str] = None,
) -> str:
    explicit = _optional_str(raw_modality)
    if explicit:
        lowered = explicit.lower()
        if lowered in _ALLOWED_CAPTURE_MODALITIES:
            return lowered
    evidence_tier = str(evidence_tier_hint or "").strip().lower()
    if capture_source == "iphone":
        if has_metric_arkit_bundle or evidence_tier == "qualified_metric_capture":
            return "iphone_arkit_lidar"
        return "iphone_video_only"
    if capture_source == "glasses" and scaffolding_used:
        return "glasses_plus_scaffolding"
    if capture_source == "glasses":
        return "glasses_video_only"
    if capture_source == "android" and scaffolding_used:
        return "android_plus_scaffolding"
    if capture_source == "android":
        return "android_video_only"
    return "iphone_arkit_lidar"


@dataclass(frozen=True)
class CaptureDescriptor:
    """Canonical descriptor produced by BlueprintCapture ``extract-frames``."""

    schema_version: str
    scene_id: str
    capture_id: str
    capture_source: str
    capture_tier: str
    raw_prefix_uri: str
    frames_index_uri: str
    quality: Dict[str, Any] = field(default_factory=dict)
    raw_video_uri: Optional[str] = None
    privacy_processed_video_uri: Optional[str] = None
    world_model_video_uri: Optional[str] = None
    privacy_status: Optional[str] = None
    privacy_mode: Optional[str] = None
    privacy_manifest_uri: Optional[str] = None
    keyframe_uri: Optional[str] = None
    arkit_poses_uri: Optional[str] = None
    arkit_intrinsics_uri: Optional[str] = None
    arkit_depth_prefix_uri: Optional[str] = None
    arkit_confidence_prefix_uri: Optional[str] = None
    depth_conditioning: Dict[str, Any] = field(default_factory=dict)
    geometry_source: Optional[str] = None
    geometry_ready: bool = False
    coordinate_frame_session_id: Optional[str] = None
    qa_report_uri: Optional[str] = None
    qa_status: Optional[str] = None
    object_index_uri: Optional[str] = None
    motion_log_uri: Optional[str] = None
    arkit_frames_uri: Optional[str] = None
    environment_type_hint: Optional[str] = None
    capture_profile_id: Optional[str] = None
    capture_capabilities: Dict[str, Any] = field(default_factory=dict)
    capture_modality: str = "iphone_arkit_lidar"
    evidence_tier: str = "pre_screen_video"
    scaffolding_used: List[str] = field(default_factory=list)
    intake_packet_uri: Optional[str] = None
    task_hypothesis_uri: Optional[str] = None
    coverage_plan: List[str] = field(default_factory=list)
    calibration_assets: List[str] = field(default_factory=list)
    scaffolding_validation: Dict[str, Any] = field(default_factory=dict)
    uncertainty_priors: Dict[str, float] = field(default_factory=dict)
    capture_orientation: Dict[str, Any] = field(default_factory=dict)
    requested_lanes: List[str] = field(default_factory=lambda: ["qualification"])
    site_submission_id: Optional[str] = None
    buyer_request_id: Optional[str] = None
    capture_job_id: Optional[str] = None
    region_id: Optional[str] = None
    special_task_type: Optional[str] = None
    priority_weight: Optional[float] = None
    quoted_payout_cents: Optional[int] = None
    rights_profile: Optional[str] = None
    requested_outputs: List[str] = field(default_factory=list)
    swap_focus: List[str] = field(default_factory=list)
    manipulation_candidates: List[Dict[str, Any]] = field(default_factory=list)
    articulation_hints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaptureDescriptor":
        schema_version = str(data.get("schema_version", "")).strip()
        if schema_version != "v1":
            raise ValueError(f"Unsupported capture descriptor schema_version: {schema_version!r}")

        scene_id = str(data.get("scene_id", "")).strip()
        capture_id = str(data.get("capture_id", "")).strip()
        raw_prefix_uri = str(data.get("raw_prefix_uri", "")).strip()
        frames_index_uri = str(data.get("frames_index_uri", "")).strip()
        capture_tier = _normalize_capture_tier(data.get("capture_tier"))
        capture_source = _infer_capture_source(str(data.get("capture_source", "")), capture_tier)

        if not scene_id:
            raise ValueError("capture_descriptor.scene_id is required")
        if not capture_id:
            raise ValueError("capture_descriptor.capture_id is required")
        if not raw_prefix_uri:
            raise ValueError("capture_descriptor.raw_prefix_uri is required")
        if not frames_index_uri:
            raise ValueError("capture_descriptor.frames_index_uri is required")

        quality = data.get("quality") if isinstance(data.get("quality"), Mapping) else {}
        raw_metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        metadata = dict(raw_metadata)
        for key in (
            "scene_memory_capture",
            "capture_rights",
            "site_identity",
            "capture_topology",
            "capture_mode",
            "route_anchors",
            "checkpoint_events",
        ):
            if key in metadata:
                continue
            value = data.get(key)
            if isinstance(value, Mapping):
                metadata[key] = dict(value)
        capture_bundle = (
            data.get("capture_bundle") if isinstance(data.get("capture_bundle"), Mapping) else {}
        )
        capture_profile_id = _optional_str(
            data.get("capture_profile_id")
            or capture_bundle.get("capture_profile_id")
            or metadata.get("capture_profile_id")
        )
        capture_capabilities = _normalize_mapping(
            data.get("capture_capabilities")
            or capture_bundle.get("capture_capabilities")
            or metadata.get("capture_capabilities")
        )
        scaffolding_used = _normalize_string_list(
            data.get("scaffolding_used") or capture_bundle.get("scaffolding_used")
        )
        arkit_poses_uri = (
            _optional_str(data.get("arkit_poses_uri"))
            or _optional_str(capture_bundle.get("arkit_poses_uri"))
        )
        arkit_intrinsics_uri = (
            _optional_str(data.get("arkit_intrinsics_uri"))
            or _optional_str(capture_bundle.get("arkit_intrinsics_uri"))
        )
        arkit_depth_prefix_uri = (
            _optional_str(data.get("arkit_depth_prefix_uri"))
            or _optional_str(capture_bundle.get("arkit_depth_prefix_uri"))
        )
        arkit_confidence_prefix_uri = (
            _optional_str(data.get("arkit_confidence_prefix_uri"))
            or _optional_str(capture_bundle.get("arkit_confidence_prefix_uri"))
        )
        evidence_tier_hint = _optional_str(
            data.get("evidence_tier") or capture_bundle.get("evidence_tier")
        )
        has_metric_arkit_bundle = bool(
            arkit_poses_uri and arkit_intrinsics_uri and arkit_depth_prefix_uri
        )

        swap_focus = _normalize_swap_focus(data.get("swap_focus"))
        if not swap_focus:
            swap_focus = _normalize_swap_focus(metadata.get("swap_focus"))

        environment_type_hint = _normalize_environment_hint(
            _optional_str(data.get("environment_type_hint"))
            or _optional_str(data.get("intended_space_type"))
        )
        raw_capture_modality = data.get("capture_modality") or capture_bundle.get("capture_modality")
        if _optional_str(raw_capture_modality) is None and capture_profile_id == _ANDROID_XR_VIDEO_ONLY_PROFILE:
            raw_capture_modality = _ANDROID_XR_VIDEO_ONLY_MODALITY

        capture_modality = _resolve_capture_modality(
            raw_modality=raw_capture_modality,
            capture_source=capture_source,
            quality=quality,
            scaffolding_used=scaffolding_used,
            has_metric_arkit_bundle=has_metric_arkit_bundle,
            evidence_tier_hint=evidence_tier_hint,
        )
        android_xr_video_only = _is_android_xr_video_only(
            capture_profile_id=capture_profile_id,
            capture_modality=capture_modality,
        )
        if android_xr_video_only:
            capture_profile_id = capture_profile_id or _ANDROID_XR_VIDEO_ONLY_PROFILE
            capture_modality = _ANDROID_XR_VIDEO_ONLY_MODALITY
            capture_capabilities = _sanitize_android_xr_video_only_capabilities(capture_capabilities)
            quality = _sanitize_android_xr_video_only_quality(quality)
            metadata = _sanitize_android_xr_video_only_metadata(metadata)
            metadata["capture_profile_id"] = capture_profile_id
            metadata["capture_capabilities"] = dict(capture_capabilities)
            scaffolding_used = []
        elif capture_profile_id:
            metadata.setdefault("capture_profile_id", capture_profile_id)
            if capture_capabilities:
                metadata.setdefault("capture_capabilities", dict(capture_capabilities))
        evidence_tier = _resolve_evidence_tier(
            evidence_tier_hint,
            capture_modality,
            quality,
        )
        if android_xr_video_only:
            evidence_tier = "pre_screen_video"

        raw_geometry_source = _optional_str(
            data.get("geometry_source")
            or quality.get("geometry_source")
            or ((metadata.get("scene_memory_capture") or {}) if isinstance(metadata.get("scene_memory_capture"), Mapping) else {}).get("geometry_source")
        )
        raw_geometry_ready = bool(
            data.get("geometry_ready")
            or quality.get("geometry_ready")
            or ((metadata.get("scene_memory_capture") or {}) if isinstance(metadata.get("scene_memory_capture"), Mapping) else {}).get("geometry_ready")
        )
        raw_quoted_payout_cents = (
            int(data.get("quoted_payout_cents") or capture_bundle.get("quoted_payout_cents"))
            if (data.get("quoted_payout_cents") or capture_bundle.get("quoted_payout_cents")) is not None
            else None
        )

        return cls(
            schema_version=schema_version,
            scene_id=scene_id,
            capture_id=capture_id,
            capture_source=capture_source,
            capture_tier=capture_tier,
            raw_prefix_uri=raw_prefix_uri,
            frames_index_uri=frames_index_uri,
            quality=dict(quality),
            raw_video_uri=_optional_str(data.get("raw_video_uri")),
            privacy_processed_video_uri=_optional_str(data.get("privacy_processed_video_uri")),
            world_model_video_uri=_optional_str(data.get("world_model_video_uri")),
            privacy_status=_optional_str(data.get("privacy_status")),
            privacy_mode=_optional_str(data.get("privacy_mode")),
            privacy_manifest_uri=_optional_str(data.get("privacy_manifest_uri")),
            keyframe_uri=_optional_str(data.get("keyframe_uri")),
            arkit_poses_uri=arkit_poses_uri,
            arkit_intrinsics_uri=arkit_intrinsics_uri,
            arkit_depth_prefix_uri=arkit_depth_prefix_uri,
            arkit_confidence_prefix_uri=arkit_confidence_prefix_uri,
            depth_conditioning=(
                dict(data.get("depth_conditioning"))
                if isinstance(data.get("depth_conditioning"), Mapping)
                else {}
            ),
            geometry_source=None if android_xr_video_only else raw_geometry_source,
            geometry_ready=False if android_xr_video_only else raw_geometry_ready,
            coordinate_frame_session_id=_optional_str(
                data.get("coordinate_frame_session_id")
                or ((metadata.get("capture_topology") or {}) if isinstance(metadata.get("capture_topology"), Mapping) else {}).get("capture_session_id")
                or ((metadata.get("capture_topology") or {}) if isinstance(metadata.get("capture_topology"), Mapping) else {}).get("captureSessionId")
            ),
            qa_report_uri=_optional_str(data.get("qa_report_uri")),
            qa_status=_optional_str(data.get("qa_status")),
            object_index_uri=_optional_str(data.get("object_index_uri")),
            motion_log_uri=_optional_str(data.get("motion_log_uri")),
            arkit_frames_uri=_optional_str(data.get("arkit_frames_uri")),
            environment_type_hint=environment_type_hint,
            capture_profile_id=capture_profile_id,
            capture_capabilities=dict(capture_capabilities),
            capture_modality=capture_modality,
            evidence_tier=evidence_tier,
            scaffolding_used=scaffolding_used,
            intake_packet_uri=(
                _optional_str(data.get("intake_packet_uri"))
                or _optional_str(capture_bundle.get("intake_packet_uri"))
            ),
            task_hypothesis_uri=(
                _optional_str(data.get("task_hypothesis_uri"))
                or _optional_str(capture_bundle.get("task_hypothesis_uri"))
            ),
            coverage_plan=_normalize_string_list(
                data.get("coverage_plan") or capture_bundle.get("coverage_plan")
            ),
            calibration_assets=_normalize_string_list(
                data.get("calibration_assets") or capture_bundle.get("calibration_assets")
            ),
            scaffolding_validation=_normalize_scaffolding_validation(
                data.get("scaffolding_validation")
                or capture_bundle.get("scaffolding_validation")
                or metadata.get("scaffolding_validation")
            ),
            uncertainty_priors=_normalize_uncertainty_priors(
                data.get("uncertainty_priors") or capture_bundle.get("uncertainty_priors")
            ),
            capture_orientation=_normalize_capture_orientation(
                data.get("capture_orientation")
                or capture_bundle.get("capture_orientation")
                or metadata.get("capture_orientation")
            ),
            requested_lanes=_normalize_requested_lanes(data.get("requested_lanes")),
            site_submission_id=_optional_str(data.get("site_submission_id") or capture_bundle.get("site_submission_id")),
            buyer_request_id=_optional_str(data.get("buyer_request_id") or capture_bundle.get("buyer_request_id")),
            capture_job_id=_optional_str(data.get("capture_job_id") or capture_bundle.get("capture_job_id")),
            region_id=_optional_str(data.get("region_id") or capture_bundle.get("region_id")),
            special_task_type=_optional_str(data.get("special_task_type") or capture_bundle.get("special_task_type")),
            priority_weight=(
                float(data.get("priority_weight") or capture_bundle.get("priority_weight"))
                if (data.get("priority_weight") or capture_bundle.get("priority_weight")) is not None
                else None
            ),
            quoted_payout_cents=None if android_xr_video_only else raw_quoted_payout_cents,
            rights_profile=_optional_str(data.get("rights_profile") or capture_bundle.get("rights_profile")),
            requested_outputs=_normalize_string_list(
                data.get("requested_outputs") or capture_bundle.get("requested_outputs")
            ),
            swap_focus=swap_focus,
            manipulation_candidates=_dict_list(data.get("manipulation_candidates")),
            articulation_hints=_dict_list(data.get("articulation_hints")),
            metadata=dict(metadata),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CaptureDescriptor":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_file(cls, path: str | Path) -> "CaptureDescriptor":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "capture_id": self.capture_id,
            "capture_source": self.capture_source,
            "capture_tier": self.capture_tier,
            "raw_prefix_uri": self.raw_prefix_uri,
            "frames_index_uri": self.frames_index_uri,
            "quality": dict(self.quality),
            "requested_lanes": list(self.requested_lanes),
            "site_submission_id": self.site_submission_id,
            "buyer_request_id": self.buyer_request_id,
            "capture_job_id": self.capture_job_id,
            "region_id": self.region_id,
            "special_task_type": self.special_task_type,
            "priority_weight": self.priority_weight,
            "quoted_payout_cents": self.quoted_payout_cents,
            "rights_profile": self.rights_profile,
            "requested_outputs": list(self.requested_outputs),
            "swap_focus": list(self.swap_focus),
            "manipulation_candidates": list(self.manipulation_candidates),
            "articulation_hints": list(self.articulation_hints),
            "metadata": dict(self.metadata),
        }
        optional = {
            "raw_video_uri": self.raw_video_uri,
            "privacy_processed_video_uri": self.privacy_processed_video_uri,
            "world_model_video_uri": self.world_model_video_uri,
            "privacy_status": self.privacy_status,
            "privacy_mode": self.privacy_mode,
            "privacy_manifest_uri": self.privacy_manifest_uri,
            "keyframe_uri": self.keyframe_uri,
            "arkit_poses_uri": self.arkit_poses_uri,
            "arkit_intrinsics_uri": self.arkit_intrinsics_uri,
            "arkit_depth_prefix_uri": self.arkit_depth_prefix_uri,
            "arkit_confidence_prefix_uri": self.arkit_confidence_prefix_uri,
            "geometry_source": self.geometry_source,
            "geometry_ready": self.geometry_ready if self.geometry_ready else None,
            "coordinate_frame_session_id": self.coordinate_frame_session_id,
            "qa_report_uri": self.qa_report_uri,
            "qa_status": self.qa_status,
            "object_index_uri": self.object_index_uri,
            "motion_log_uri": self.motion_log_uri,
            "arkit_frames_uri": self.arkit_frames_uri,
            "environment_type_hint": self.environment_type_hint,
            "capture_profile_id": self.capture_profile_id,
            "capture_modality": self.capture_modality,
            "evidence_tier": self.evidence_tier,
            "intake_packet_uri": self.intake_packet_uri,
            "task_hypothesis_uri": self.task_hypothesis_uri,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
        payload["scaffolding_used"] = list(self.scaffolding_used)
        payload["coverage_plan"] = list(self.coverage_plan)
        payload["calibration_assets"] = list(self.calibration_assets)
        payload["scaffolding_validation"] = dict(self.scaffolding_validation)
        payload["uncertainty_priors"] = dict(self.uncertainty_priors)
        if self.capture_capabilities:
            payload["capture_capabilities"] = dict(self.capture_capabilities)
        if self.depth_conditioning:
            payload["depth_conditioning"] = dict(self.depth_conditioning)
        if self.capture_orientation:
            payload["capture_orientation"] = dict(self.capture_orientation)
        return payload

    @property
    def preferred_world_model_video_uri(self) -> Optional[str]:
        return (
            self.world_model_video_uri
            or self.privacy_processed_video_uri
        )


def build_capture_bundle_constraints(
    descriptor: CaptureDescriptor,
    *,
    descriptor_uri: Optional[str] = None,
    qa_report_uri: Optional[str] = None,
    qa_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``constraints.capture_bundle`` object."""

    bundle: Dict[str, Any] = {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "capture_source": descriptor.capture_source,
        "capture_tier": descriptor.capture_tier,
        "raw_prefix_uri": descriptor.raw_prefix_uri,
        "raw_video_uri": descriptor.raw_video_uri,
        "privacy_processed_video_uri": descriptor.privacy_processed_video_uri,
        "world_model_video_uri": descriptor.world_model_video_uri,
        "privacy_status": descriptor.privacy_status,
        "privacy_mode": descriptor.privacy_mode,
        "privacy_manifest_uri": descriptor.privacy_manifest_uri,
        "frames_index_uri": descriptor.frames_index_uri,
        "keyframe_uri": descriptor.keyframe_uri,
        "requested_lanes": list(descriptor.requested_lanes),
        "site_submission_id": descriptor.site_submission_id,
        "buyer_request_id": descriptor.buyer_request_id,
        "capture_job_id": descriptor.capture_job_id,
        "region_id": descriptor.region_id,
        "special_task_type": descriptor.special_task_type,
        "priority_weight": descriptor.priority_weight,
        "quoted_payout_cents": descriptor.quoted_payout_cents,
        "rights_profile": descriptor.rights_profile,
        "requested_outputs": list(descriptor.requested_outputs),
        "swap_focus": list(descriptor.swap_focus),
        "quality": dict(descriptor.quality),
        "environment_type_hint": descriptor.environment_type_hint,
        "capture_profile_id": descriptor.capture_profile_id,
        "capture_modality": descriptor.capture_modality,
        "evidence_tier": descriptor.evidence_tier,
        "scaffolding_used": list(descriptor.scaffolding_used),
        "intake_packet_uri": descriptor.intake_packet_uri,
        "coverage_plan": list(descriptor.coverage_plan),
        "calibration_assets": list(descriptor.calibration_assets),
        "scaffolding_validation": dict(descriptor.scaffolding_validation),
        "uncertainty_priors": dict(descriptor.uncertainty_priors),
        "capture_orientation": dict(descriptor.capture_orientation),
        "descriptor_uri": descriptor_uri,
        "qa_report_uri": qa_report_uri or descriptor.qa_report_uri,
        "object_index_uri": descriptor.object_index_uri,
        "motion_log_uri": descriptor.motion_log_uri,
        "arkit_frames_uri": descriptor.arkit_frames_uri,
        "depth_conditioning": dict(descriptor.depth_conditioning),
    }

    arkit = {
        "arkit_poses_uri": descriptor.arkit_poses_uri,
        "arkit_intrinsics_uri": descriptor.arkit_intrinsics_uri,
        "arkit_depth_prefix_uri": descriptor.arkit_depth_prefix_uri,
        "arkit_confidence_prefix_uri": descriptor.arkit_confidence_prefix_uri,
    }
    arkit = {k: v for k, v in arkit.items() if v is not None}
    if arkit:
        bundle["capture_bundle"] = arkit

    if qa_report is not None:
        bundle["qa"] = dict(qa_report)
    if descriptor.capture_capabilities:
        bundle["capture_capabilities"] = dict(descriptor.capture_capabilities)

    return {k: v for k, v in bundle.items() if v is not None}


def build_scene_request_from_descriptor(
    descriptor: CaptureDescriptor,
    *,
    keyframe_uri: Optional[str] = None,
    descriptor_uri: Optional[str] = None,
    qa_report_uri: Optional[str] = None,
    qa_report: Optional[Mapping[str, Any]] = None,
    quality_tier: str = "standard",
    provider_policy: str = "openai_primary",
    allow_image_fallback: bool = False,
) -> Dict[str, Any]:
    """Build a source-orchestrator style ``scene_request_v1`` payload."""

    keyframe = keyframe_uri or descriptor.keyframe_uri
    if not keyframe:
        raise ValueError("A keyframe URI is required to build scene_request payload")

    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "source_mode": "image",
        "quality_tier": quality_tier,
        "image": {"gcs_uri": keyframe},
        "constraints": {
            "capture_bundle": build_capture_bundle_constraints(
                descriptor,
                descriptor_uri=descriptor_uri,
                qa_report_uri=qa_report_uri,
                qa_report=qa_report,
            )
        },
        "provider_policy": provider_policy,
        "fallback": {"allow_image_fallback": allow_image_fallback},
    }


def _normalize_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(candidate)
    normalized["must_be_separate_asset"] = True
    return normalized


def build_scene_manifest_seed(
    descriptor: CaptureDescriptor,
    *,
    manipulation_candidates: Optional[List[Mapping[str, Any]]] = None,
    articulation_hints: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a scene-manifest seed from capture descriptor metadata."""

    merged_candidates = list(descriptor.manipulation_candidates)
    if manipulation_candidates:
        merged_candidates.extend(dict(item) for item in manipulation_candidates)

    merged_hints = list(descriptor.articulation_hints)
    if articulation_hints:
        merged_hints.extend(dict(item) for item in articulation_hints)

    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "environment": {
            "type_hint": descriptor.environment_type_hint or "unknown",
            "requested_lanes": list(descriptor.requested_lanes),
            "swap_focus": list(descriptor.swap_focus),
        },
        "scene_shell": {
            "raw_prefix_uri": descriptor.raw_prefix_uri,
            "frames_index_uri": descriptor.frames_index_uri,
            "default_role": "static_background",
            "proxy_collision_enabled": True,
        },
        "manipulation_candidates": [_normalize_candidate(c) for c in merged_candidates],
        "articulation_hints": [dict(item) for item in merged_hints],
        "quality": dict(descriptor.quality),
    }
