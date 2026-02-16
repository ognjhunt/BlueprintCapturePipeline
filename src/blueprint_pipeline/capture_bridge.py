"""Capture descriptor contracts for NuRec-first swap orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_ALLOWED_NUREC_MODES = {"mono_pose_assisted", "mono_slam"}
_ALLOWED_SWAP_FOCUS = {"kitchen", "warehouse"}


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


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


def _infer_capture_source(raw_source: str, capture_tier: str) -> str:
    source = raw_source.strip().lower()
    if source in {"iphone", "glasses"}:
        return source
    tier = capture_tier.strip().lower()
    if "glasses" in tier:
        return "glasses"
    return "iphone"


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
    environment_type_hint: Optional[str] = None
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

        swap_focus = _normalize_swap_focus(data.get("swap_focus"))
        if not swap_focus:
            swap_focus = _normalize_swap_focus(metadata.get("swap_focus"))

        environment_type_hint = (
            _optional_str(data.get("environment_type_hint"))
            or _optional_str(data.get("intended_space_type"))
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
            environment_type_hint=environment_type_hint,
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
            "environment_type_hint": self.environment_type_hint,
        }
        for key, value in optional.items():
            if value is not None:
                payload[key] = value
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
        "swap_focus": list(descriptor.swap_focus),
        "quality": dict(descriptor.quality),
        "environment_type_hint": descriptor.environment_type_hint,
        "descriptor_uri": descriptor_uri,
        "qa_report_uri": qa_report_uri or descriptor.qa_report_uri,
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
