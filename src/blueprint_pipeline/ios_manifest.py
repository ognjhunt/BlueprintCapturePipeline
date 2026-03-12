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
    object_index_uri: Optional[str] = None
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
            object_index_uri=(
                str(data.get("object_index_uri")).strip()
                if data.get("object_index_uri") is not None
                else None
            ),
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
        rel = manifest.object_index_uri or manifest.object_point_cloud_index
    else:
        rel = (
            str(manifest.get("object_index_uri", "")).strip()
            or str(manifest.get("object_point_cloud_index", "")).strip()
            or None
        )

    if not rel:
        return None
    if rel.startswith("gs://"):
        return rel
    return join_gs_uri(raw_prefix_uri, rel)


def _is_uri(path_text: str) -> bool:
    return "://" in path_text


def _normalize_crop_paths(
    entries: list[dict[str, Any]],
    *,
    index_dir: Path,
) -> list[dict[str, Any]]:
    """Resolve relative crop paths against the object index location."""
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)

        reference_crop = item.get("reference_crop")
        if isinstance(reference_crop, str):
            ref_text = reference_crop.strip()
            if ref_text and not _is_uri(ref_text):
                ref_path = Path(ref_text)
                if not ref_path.is_absolute():
                    item["reference_crop"] = str((index_dir / ref_path).resolve())

        all_crops = item.get("all_crops")
        if isinstance(all_crops, list):
            resolved_crops: list[str] = []
            for value in all_crops:
                crop_text = str(value).strip()
                if not crop_text:
                    continue
                if _is_uri(crop_text):
                    resolved_crops.append(crop_text)
                    continue
                crop_path = Path(crop_text)
                if not crop_path.is_absolute():
                    crop_path = (index_dir / crop_path).resolve()
                resolved_crops.append(str(crop_path))
            item["all_crops"] = resolved_crops

        normalized.append(item)

    return normalized


def load_object_index(index_uri: str, *, gcs_root: Path) -> list[dict[str, Any]]:
    """Load object index as a normalized list of object entries."""

    path = resolve_gs_uri_to_path(index_uri, gcs_root)
    payload = read_json_any(path)

    entries: list[dict[str, Any]]
    if isinstance(payload, list):
        entries = [dict(item) for item in payload if isinstance(item, Mapping)]
    elif isinstance(payload, Mapping):
        for field in ("objects", "items", "summaries"):
            value = payload.get(field)
            if isinstance(value, list):
                entries = [dict(item) for item in value if isinstance(item, Mapping)]
                break
        else:
            raise ValueError(f"Unsupported object index payload at {path}")
    else:
        raise ValueError(f"Unsupported object index payload at {path}")

    return _normalize_crop_paths(entries, index_dir=path.parent)


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
