"""BlueprintCapture iOS manifest helpers for capture orchestration."""

from __future__ import annotations

import json
from hashlib import sha256
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
    approx_floor_area_m2: Optional[float] = None
    ceiling_height_m: Optional[float] = None
    floor_count: Optional[int] = None
    dominant_aisle_width_m: Optional[float] = None
    site_scale_class: Optional[str] = None
    site_extent_status: Optional[str] = None
    site_extent_source: Optional[str] = None
    site_levels: List[Dict[str, Any]] = field(default_factory=list)
    coverage_by_level: List[Dict[str, Any]] = field(default_factory=list)
    vertical_structure_notes: List[str] = field(default_factory=list)
    site_operating_conditions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IOSManifest":
        site_extent = data.get("site_extent") if isinstance(data.get("site_extent"), Mapping) else {}
        site_operating_conditions = (
            data.get("site_operating_conditions")
            if isinstance(data.get("site_operating_conditions"), Mapping)
            else data.get("operating_conditions")
            if isinstance(data.get("operating_conditions"), Mapping)
            else data.get("environmental_conditions")
            if isinstance(data.get("environmental_conditions"), Mapping)
            else {}
        )
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
                str(data.get("capture_schema_version") or data.get("schema_version")).strip()
                if data.get("capture_schema_version") is not None or data.get("schema_version") is not None
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
            approx_floor_area_m2=_positive_float(
                data.get("approx_floor_area_m2") or site_extent.get("approx_floor_area_m2")
            ),
            ceiling_height_m=_positive_float(
                data.get("ceiling_height_m") or site_extent.get("ceiling_height_m")
            ),
            floor_count=_positive_int(data.get("floor_count") or site_extent.get("floor_count")),
            dominant_aisle_width_m=_positive_float(
                data.get("dominant_aisle_width_m") or site_extent.get("dominant_aisle_width_m")
            ),
            site_scale_class=_optional_text(
                data.get("site_scale_class") or site_extent.get("site_scale_class")
            ),
            site_extent_status=_optional_text(
                data.get("site_extent_status") or site_extent.get("status")
            ),
            site_extent_source=_optional_text(
                data.get("site_extent_source") or site_extent.get("source")
            ),
            site_levels=_mapping_list(
                data.get("site_levels") or site_extent.get("site_levels") or site_extent.get("levels")
            ),
            coverage_by_level=_mapping_list(
                data.get("coverage_by_level") or site_extent.get("coverage_by_level")
            ),
            vertical_structure_notes=_string_list(
                data.get("vertical_structure_notes")
                or site_extent.get("vertical_structure_notes")
                or site_extent.get("multi_level_notes")
            ),
            site_operating_conditions=dict(site_operating_conditions),
        )

    @classmethod
    def from_json(cls, payload: str) -> "IOSManifest":
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_path(cls, path: Path) -> "IOSManifest":
        return cls.from_dict(read_json(path))


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _mapping_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {str(key): item_value for key, item_value in item.items() if item_value not in (None, "")}
        for item in value
        if isinstance(item, Mapping)
    ]


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_hash(artifact_hashes: Mapping[str, str]) -> str:
    canonical = "\n".join(
        f"{path}:{artifact_hashes[path]}" for path in sorted(artifact_hashes)
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _regular_file_relative_paths(raw_prefix_path: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(raw_prefix_path.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name == "hashes.json":
            continue
        paths.append(path.relative_to(raw_prefix_path).as_posix())
    return paths


def verify_raw_bundle_hashes_path(raw_prefix_path: Path) -> Dict[str, Any]:
    """Recompute and compare the BlueprintCapture v3 ``hashes.json`` manifest."""

    hashes_path = raw_prefix_path / "hashes.json"
    errors: list[str] = []
    if not hashes_path.is_file():
        errors.append("missing_hash_manifest")
        return {
            "schema_version": "raw_bundle_hash_verification.v1",
            "valid": False,
            "status": "failed",
            "errors": errors,
            "artifact_count": 0,
            "bundle_sha256_expected": None,
            "bundle_sha256_actual": None,
            "bundle_sha256_matches": False,
            "claim_boundary": "hash_verification_checks_local_raw_bundle_bytes_not_capture_semantic_quality",
        }

    payload = read_json_any(hashes_path)
    artifacts_raw = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if not isinstance(artifacts_raw, Mapping):
        errors.append("missing_hash_manifest")
        artifacts: Dict[str, str] = {}
    else:
        artifacts = {
            str(path).strip(): str(digest).strip().lower()
            for path, digest in artifacts_raw.items()
            if str(path).strip() and str(digest).strip()
        }

    actual_hashes: Dict[str, str] = {}
    for relative_path, expected_hash in sorted(artifacts.items()):
        target = raw_prefix_path / relative_path
        if not target.is_file():
            errors.append(f"hash_target_missing:{relative_path}")
            continue
        actual_hash = _sha256_file(target)
        actual_hashes[relative_path] = actual_hash
        if actual_hash != expected_hash:
            errors.append(f"hash_mismatch:{relative_path}")

    for relative_path in _regular_file_relative_paths(raw_prefix_path):
        if relative_path not in artifacts:
            errors.append(f"hash_coverage_missing:{relative_path}")
            actual_hashes[relative_path] = _sha256_file(raw_prefix_path / relative_path)

    expected_bundle_hash = (
        str(payload.get("bundle_sha256")).strip().lower()
        if isinstance(payload, Mapping) and payload.get("bundle_sha256") is not None
        else None
    )
    actual_bundle_hash = _bundle_hash(actual_hashes) if actual_hashes else None
    bundle_matches = bool(expected_bundle_hash and actual_bundle_hash == expected_bundle_hash)
    if expected_bundle_hash and actual_bundle_hash and not bundle_matches:
        errors.append("bundle_sha256_mismatch")
    elif not expected_bundle_hash:
        errors.append("bundle_sha256_missing")

    return {
        "schema_version": "raw_bundle_hash_verification.v1",
        "valid": not errors,
        "status": "verified" if not errors else "failed",
        "errors": errors,
        "artifact_count": len(artifacts),
        "bundle_sha256_expected": expected_bundle_hash,
        "bundle_sha256_actual": actual_bundle_hash,
        "bundle_sha256_matches": bundle_matches,
        "claim_boundary": "hash_verification_checks_local_raw_bundle_bytes_not_capture_semantic_quality",
    }


def verify_raw_bundle_hashes(raw_prefix_uri: str, *, gcs_root: Path) -> Dict[str, Any]:
    raw_prefix_path = resolve_gs_uri_to_path(raw_prefix_uri, gcs_root)
    return verify_raw_bundle_hashes_path(raw_prefix_path)


def load_raw_manifest(
    raw_prefix_uri: str,
    *,
    gcs_root: Path,
    verify_hashes: bool | None = None,
) -> IOSManifest:
    manifest_uri = join_gs_uri(raw_prefix_uri, "manifest.json")
    manifest_path = resolve_gs_uri_to_path(manifest_uri, gcs_root)
    manifest = IOSManifest.from_path(manifest_path)
    should_verify_hashes = (
        str(manifest.capture_schema_version or "").strip().lower() == "v3"
        if verify_hashes is None
        else bool(verify_hashes)
    )
    if should_verify_hashes:
        report = verify_raw_bundle_hashes(raw_prefix_uri, gcs_root=gcs_root)
        if not report["valid"]:
            raise ValueError(
                "raw_bundle_hash_verification_failed:"
                + ",".join(str(error) for error in report.get("errors", []))
            )
    return manifest


def object_index_path(raw_prefix_path: Path, manifest: IOSManifest) -> Optional[Path]:
    rel = manifest.object_point_cloud_index
    if not rel:
        return None
    rel_clean = rel.lstrip("/")
    return raw_prefix_path / rel_clean
