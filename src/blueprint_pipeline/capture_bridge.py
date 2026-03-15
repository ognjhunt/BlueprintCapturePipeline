"""Capture descriptor contracts for site-world orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_ALLOWED_NUREC_MODES = {"mono_pose_assisted", "mono_slam"}
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
_ALLOWED_REQUESTED_LANES = {"qualification", "scene_memory", "evaluation_prep"}
_ALLOWED_CAPTURE_MODALITIES = {
    "iphone_arkit_lidar",
    "glasses_video_only",
    "glasses_plus_scaffolding",
}
_ALLOWED_EVIDENCE_TIERS = {
    "pre_screen_video",
    "qualified_metric_capture",
    "glasses_with_validated_scaffolding",
}
_ALLOWED_DISPLAY_ORIENTATIONS = {"portrait", "landscape", "square", "unknown"}


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


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


def _resolve_nurec_mode(
    *,
    explicit_mode: Optional[str],
    capture_source: str,
    quality: Mapping[str, Any],
) -> str:
    if explicit_mode in _ALLOWED_NUREC_MODES:
        return explicit_mode
    pose_match_rate = float(quality.get("pose_match_rate", 0.0) or 0.0)
    if capture_source == "iphone" and pose_match_rate >= 0.9:
        return "mono_pose_assisted"
    return "mono_slam"


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
            for lane in ("qualification", "scene_memory", "evaluation_prep"):
                if lane not in normalized:
                    normalized.append(lane)
            continue
        if lowered in _ALLOWED_REQUESTED_LANES and lowered not in normalized:
            normalized.append(lowered)
            if lowered == "evaluation_prep" and "qualification" not in normalized:
                normalized.append("qualification")
    if "evaluation_prep" in normalized and "qualification" not in normalized:
        normalized.append("qualification")
    ordered: List[str] = []
    for lane in ("qualification", "scene_memory", "evaluation_prep"):
        if lane in normalized and lane not in ordered:
            ordered.append(lane)
    return ordered or ["qualification"]


def _infer_capture_source(raw_source: str, capture_tier: str) -> str:
    source = raw_source.strip().lower()
    if source in {"iphone", "glasses"}:
        return source
    tier = capture_tier.strip().lower()
    if "glasses" in tier:
        return "glasses"
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
        if lowered in _ALLOWED_EVIDENCE_TIERS:
            return lowered
    if capture_modality == "iphone_arkit_lidar":
        return "qualified_metric_capture"
    if capture_modality == "glasses_plus_scaffolding":
        return "glasses_with_validated_scaffolding"
    return "pre_screen_video"


def _resolve_capture_modality(
    *,
    raw_modality: Any,
    capture_source: str,
    quality: Mapping[str, Any],
    scaffolding_used: List[str],
) -> str:
    explicit = _optional_str(raw_modality)
    if explicit:
        lowered = explicit.lower()
        if lowered in _ALLOWED_CAPTURE_MODALITIES:
            return lowered
    if capture_source == "iphone" and float(quality.get("pose_match_rate", 0.0) or 0.0) >= 0.9:
        return "iphone_arkit_lidar"
    if capture_source == "glasses" and scaffolding_used:
        return "glasses_plus_scaffolding"
    if capture_source == "glasses":
        return "glasses_video_only"
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
    nurec_mode: str
    quality: Dict[str, Any] = field(default_factory=dict)
    raw_video_uri: Optional[str] = None
    keyframe_uri: Optional[str] = None
    arkit_poses_uri: Optional[str] = None
    arkit_intrinsics_uri: Optional[str] = None
    arkit_depth_prefix_uri: Optional[str] = None
    arkit_confidence_prefix_uri: Optional[str] = None
    qa_report_uri: Optional[str] = None
    qa_status: Optional[str] = None
    object_index_uri: Optional[str] = None
    motion_log_uri: Optional[str] = None
    arkit_frames_uri: Optional[str] = None
    environment_type_hint: Optional[str] = None
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
        capture_tier = str(data.get("capture_tier", "")).strip() or "tier2_glasses"
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
        metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        capture_bundle = (
            data.get("capture_bundle") if isinstance(data.get("capture_bundle"), Mapping) else {}
        )
        scaffolding_used = _normalize_string_list(
            data.get("scaffolding_used") or capture_bundle.get("scaffolding_used")
        )

        swap_focus = _normalize_swap_focus(data.get("swap_focus"))
        if not swap_focus:
            swap_focus = _normalize_swap_focus(metadata.get("swap_focus"))

        environment_type_hint = _normalize_environment_hint(
            _optional_str(data.get("environment_type_hint"))
            or _optional_str(data.get("intended_space_type"))
        )
        capture_modality = _resolve_capture_modality(
            raw_modality=data.get("capture_modality") or capture_bundle.get("capture_modality"),
            capture_source=capture_source,
            quality=quality,
            scaffolding_used=scaffolding_used,
        )
        evidence_tier = _resolve_evidence_tier(
            data.get("evidence_tier") or capture_bundle.get("evidence_tier"),
            capture_modality,
            quality,
        )

        return cls(
            schema_version=schema_version,
            scene_id=scene_id,
            capture_id=capture_id,
            capture_source=capture_source,
            capture_tier=capture_tier,
            raw_prefix_uri=raw_prefix_uri,
            frames_index_uri=frames_index_uri,
            nurec_mode=_resolve_nurec_mode(
                explicit_mode=_optional_str(data.get("nurec_mode")),
                capture_source=capture_source,
                quality=quality,
            ),
            quality=dict(quality),
            raw_video_uri=_optional_str(data.get("raw_video_uri")),
            keyframe_uri=_optional_str(data.get("keyframe_uri")),
            arkit_poses_uri=(
                _optional_str(data.get("arkit_poses_uri"))
                or _optional_str(capture_bundle.get("arkit_poses_uri"))
            ),
            arkit_intrinsics_uri=(
                _optional_str(data.get("arkit_intrinsics_uri"))
                or _optional_str(capture_bundle.get("arkit_intrinsics_uri"))
            ),
            arkit_depth_prefix_uri=(
                _optional_str(data.get("arkit_depth_prefix_uri"))
                or _optional_str(capture_bundle.get("arkit_depth_prefix_uri"))
            ),
            arkit_confidence_prefix_uri=(
                _optional_str(data.get("arkit_confidence_prefix_uri"))
                or _optional_str(capture_bundle.get("arkit_confidence_prefix_uri"))
            ),
            qa_report_uri=_optional_str(data.get("qa_report_uri")),
            qa_status=_optional_str(data.get("qa_status")),
            object_index_uri=_optional_str(data.get("object_index_uri")),
            motion_log_uri=_optional_str(data.get("motion_log_uri")),
            arkit_frames_uri=_optional_str(data.get("arkit_frames_uri")),
            environment_type_hint=environment_type_hint,
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
            quoted_payout_cents=(
                int(data.get("quoted_payout_cents") or capture_bundle.get("quoted_payout_cents"))
                if (data.get("quoted_payout_cents") or capture_bundle.get("quoted_payout_cents")) is not None
                else None
            ),
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
            "nurec_mode": self.nurec_mode,
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
            "keyframe_uri": self.keyframe_uri,
            "arkit_poses_uri": self.arkit_poses_uri,
            "arkit_intrinsics_uri": self.arkit_intrinsics_uri,
            "arkit_depth_prefix_uri": self.arkit_depth_prefix_uri,
            "arkit_confidence_prefix_uri": self.arkit_confidence_prefix_uri,
            "qa_report_uri": self.qa_report_uri,
            "qa_status": self.qa_status,
            "object_index_uri": self.object_index_uri,
            "motion_log_uri": self.motion_log_uri,
            "arkit_frames_uri": self.arkit_frames_uri,
            "environment_type_hint": self.environment_type_hint,
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
        if self.capture_orientation:
            payload["capture_orientation"] = dict(self.capture_orientation)
        return payload


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
        "frames_index_uri": descriptor.frames_index_uri,
        "keyframe_uri": descriptor.keyframe_uri,
        "nurec_mode": descriptor.nurec_mode,
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
            "nurec_mode": descriptor.nurec_mode,
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
