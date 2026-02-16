"""BlueprintCapture iOS manifest helpers for NuRec swap orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .common import join_gs_uri, read_json, read_json_any, resolve_gs_uri_to_path


@dataclass(frozen=True)
class IOSManifest:
    """Parsed ``raw/manifest.json`` from BlueprintCapture."""

    scene_id: str
    video_uri: str
    device_model: str
    os_version: str
    fps_source: float
    width: int
    height: int
    capture_start_epoch_ms: int
    has_lidar: bool
    scale_hint_m_per_unit: float
    intended_space_type: str
    exposure_samples: List[Dict[str, Any]] = field(default_factory=list)
    object_point_cloud_index: Optional[str] = None
    object_point_cloud_count: int = 0
    capture_schema_version: Optional[str] = None
    capture_source: Optional[str] = None
    capture_tier_hint: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IOSManifest":
        return cls(
            scene_id=str(data.get("scene_id", "")),
            video_uri=str(data.get("video_uri", "")),
            device_model=str(data.get("device_model", "iPhone")),
            os_version=str(data.get("os_version", "unknown")),
            fps_source=float(data.get("fps_source", 30.0) or 30.0),
            width=int(data.get("width", 1920) or 1920),
            height=int(data.get("height", 1080) or 1080),
            capture_start_epoch_ms=int(data.get("capture_start_epoch_ms", 0) or 0),
            has_lidar=bool(data.get("has_lidar", False)),
            scale_hint_m_per_unit=float(data.get("scale_hint_m_per_unit", 1.0) or 1.0),
            intended_space_type=str(data.get("intended_space_type", "unknown") or "unknown"),
            exposure_samples=[
                dict(item) for item in data.get("exposure_samples", []) if isinstance(item, Mapping)
            ],
            object_point_cloud_index=(
                str(data.get("object_point_cloud_index")).strip()
                if data.get("object_point_cloud_index") is not None
                else None
            ),
            object_point_cloud_count=int(data.get("object_point_cloud_count", 0) or 0),
            capture_schema_version=(
                str(data.get("capture_schema_version")).strip()
                if data.get("capture_schema_version") is not None
                else None
            ),
            capture_source=(
                str(data.get("capture_source")).strip().lower()
                if data.get("capture_source") is not None
                else None
            ),
            capture_tier_hint=(
                str(data.get("capture_tier_hint")).strip()
                if data.get("capture_tier_hint") is not None
                else None
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "IOSManifest":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_path(cls, path: Path) -> "IOSManifest":
        return cls.from_dict(read_json(path))


def load_ios_manifest_from_uri(raw_manifest_uri: str, *, gcs_root: Path) -> IOSManifest:
    path = resolve_gs_uri_to_path(raw_manifest_uri, gcs_root)
    return IOSManifest.from_path(path)


def resolve_object_index_uri(raw_prefix_uri: str, manifest: IOSManifest | Mapping[str, Any]) -> Optional[str]:
    """Resolve manifest ``object_point_cloud_index`` to a fully-qualified URI."""

    if isinstance(manifest, IOSManifest):
        rel = manifest.object_point_cloud_index
    else:
        rel = str(manifest.get("object_point_cloud_index", "")).strip() or None

    if not rel:
        return None
    if rel.startswith("gs://"):
        return rel
    return join_gs_uri(raw_prefix_uri, rel)


def load_object_index(index_uri: str, *, gcs_root: Path) -> list[dict[str, Any]]:
    """Load object index as a normalized list of object entries."""

    path = resolve_gs_uri_to_path(index_uri, gcs_root)
    payload = read_json_any(path)

    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for field in ("objects", "items", "summaries"):
            value = payload.get(field)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
    raise ValueError(f"Unsupported object index payload at {path}")


def load_raw_manifest(raw_prefix_uri: str, *, gcs_root: Path) -> IOSManifest:
    manifest_uri = join_gs_uri(raw_prefix_uri, "manifest.json")
    manifest_path = resolve_gs_uri_to_path(manifest_uri, gcs_root)
    return IOSManifest.from_path(manifest_path)


def object_index_path(raw_prefix_path: Path, manifest: IOSManifest) -> Optional[Path]:
    rel = manifest.object_point_cloud_index
    if not rel:
        return None
    rel_clean = rel.lstrip("/")
    return raw_prefix_path / rel_clean
