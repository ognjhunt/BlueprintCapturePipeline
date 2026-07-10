"""BlueprintCapture iOS manifest helpers for capture orchestration."""

from __future__ import annotations

import json
import math
import os
import re
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from .common import join_gs_uri, parse_gs_uri, read_json, read_json_any, resolve_gs_uri_to_path


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
            device_model=str(data.get("device_model") or data.get("device_model_marketing") or "unknown"),
            os_version=str(data.get("os_version") or data.get("ios_version") or "unknown"),
            fps_source=float(data.get("fps_source", 30.0) or 30.0),
            width=int(data.get("width", 1920) or 1920),
            height=int(data.get("height", 1080) or 1080),
            capture_start_epoch_ms=int(data.get("capture_start_epoch_ms", 0) or 0),
            has_lidar=bool(data.get("has_lidar", False)),
            scale_hint_m_per_unit=float(data.get("scale_hint_m_per_unit", 0.0) or 0.0),
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


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_SIDECARS = (
    "capture_context.json",
    "provenance.json",
    "rights_consent.json",
    "recording_session.json",
    "capture_upload_complete.json",
)
_UPLOAD_COMPLETE_STATUS_VALUES = {"complete", "completed", "uploaded", "finalized", "ready"}
_FORBIDDEN_PIPELINE_DERIVATIVES_IN_RAW = (
    "object_index_build_report.json",
    "object_index_keyframes.json",
    "object_grounding_hints.json",
    "object_index_artifacts",
)


def _is_current_v3_manifest(payload: Mapping[str, Any]) -> bool:
    schema_version = str(payload.get("schema_version") or "").strip().lower()
    capture_schema_version = str(payload.get("capture_schema_version") or "").strip().lower()
    return schema_version == "v3" or capture_schema_version == "v3" or capture_schema_version.startswith("3.")


def _safe_raw_relative_path(raw_prefix_path: Path, raw_relative_path: str) -> tuple[Optional[Path], Optional[str]]:
    """Resolve one raw-manifest member without permitting path or symlink escape."""

    text = str(raw_relative_path or "").strip()
    if not text:
        return None, "empty_path"
    if "\\" in text or ":" in PurePosixPath(text).parts[0]:
        return None, "non_posix_path"
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or relative.as_posix() != text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None, "path_escape"

    current = raw_prefix_path
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None, "symlink"

    root = raw_prefix_path.resolve()
    candidate = raw_prefix_path.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None, "path_escape"

    return candidate, None


def _raw_tree_entries(raw_prefix_path: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    if raw_prefix_path.is_symlink():
        return [], ["raw_root_symlink"]
    if not raw_prefix_path.is_dir():
        return [], ["missing_raw_root"]
    for directory, dirnames, filenames in os.walk(raw_prefix_path, followlinks=False):
        directory_path = Path(directory)
        for name in sorted([*dirnames, *filenames]):
            path = directory_path / name
            relative = path.relative_to(raw_prefix_path).as_posix()
            if path.is_symlink():
                errors.append(f"raw_symlink_forbidden:{relative}")
        for filename in sorted(filenames):
            path = directory_path / filename
            if path.is_symlink():
                continue
            if not path.is_file():
                errors.append(f"raw_non_regular_member:{path.relative_to(raw_prefix_path).as_posix()}")
                continue
            files.append(path)
    return sorted(files), errors


def _regular_file_relative_paths(raw_prefix_path: Path) -> list[str]:
    files, _errors = _raw_tree_entries(raw_prefix_path)
    paths: list[str] = []
    for path in files:
        if path.name == "hashes.json":
            continue
        paths.append(path.relative_to(raw_prefix_path).as_posix())
    return paths


def verify_raw_bundle_hashes_path(raw_prefix_path: Path) -> Dict[str, Any]:
    """Recompute and compare the BlueprintCapture v3 ``hashes.json`` manifest."""

    hashes_path = raw_prefix_path / "hashes.json"
    _tree_files, errors = _raw_tree_entries(raw_prefix_path)
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

    try:
        payload = read_json_any(hashes_path)
    except Exception as exc:
        errors.append(f"invalid_json:hashes.json:{type(exc).__name__}")
        payload = {}
    artifacts_raw = payload.get("artifacts") if isinstance(payload, Mapping) else None
    if not isinstance(artifacts_raw, Mapping):
        errors.append("missing_hash_manifest")
        artifacts: Dict[str, str] = {}
    else:
        artifacts = {}
        for raw_path, raw_digest in artifacts_raw.items():
            relative_path = str(raw_path).strip()
            if not relative_path:
                errors.append("invalid_hash_path:empty_path")
                continue
            if relative_path == "hashes.json":
                errors.append("invalid_hash_path:hashes.json:self_reference")
                continue
            digest = (
                str(raw_digest.get("sha256") or "").strip().lower()
                if isinstance(raw_digest, Mapping)
                else str(raw_digest).strip().lower()
            )
            if not _SHA256_PATTERN.fullmatch(digest):
                errors.append(f"invalid_hash_digest:{relative_path}")
                continue
            if relative_path in artifacts:
                errors.append(f"duplicate_hash_path:{relative_path}")
                continue
            artifacts[relative_path] = digest

    actual_hashes: Dict[str, str] = {}
    artifact_sizes_bytes: Dict[str, int] = {}
    for relative_path, expected_hash in sorted(artifacts.items()):
        target, path_error = _safe_raw_relative_path(raw_prefix_path, relative_path)
        if path_error:
            errors.append(f"invalid_hash_path:{relative_path}:{path_error}")
            continue
        assert target is not None
        if not target.is_file():
            errors.append(f"hash_target_missing:{relative_path}")
            continue
        actual_hash = _sha256_file(target)
        actual_hashes[relative_path] = actual_hash
        artifact_sizes_bytes[relative_path] = target.stat().st_size
        if actual_hash != expected_hash:
            errors.append(f"hash_mismatch:{relative_path}")

    for relative_path in _regular_file_relative_paths(raw_prefix_path):
        if relative_path not in artifacts:
            errors.append(f"hash_coverage_missing:{relative_path}")
            target = raw_prefix_path / relative_path
            actual_hashes[relative_path] = _sha256_file(target)
            artifact_sizes_bytes[relative_path] = target.stat().st_size

    expected_bundle_hash = (
        str(payload.get("bundle_sha256")).strip().lower()
        if isinstance(payload, Mapping) and payload.get("bundle_sha256") is not None
        else None
    )
    actual_bundle_hash = _bundle_hash(actual_hashes) if actual_hashes else None
    bundle_matches = bool(expected_bundle_hash and actual_bundle_hash == expected_bundle_hash)
    if expected_bundle_hash and not _SHA256_PATTERN.fullmatch(expected_bundle_hash):
        errors.append("bundle_sha256_invalid")
    elif expected_bundle_hash and actual_bundle_hash and not bundle_matches:
        errors.append("bundle_sha256_mismatch")
    elif not expected_bundle_hash:
        errors.append("bundle_sha256_missing")

    return {
        "schema_version": "raw_bundle_hash_verification.v1",
        "valid": not errors,
        "status": "verified" if not errors else "failed",
        "errors": errors,
        "artifact_count": len(artifacts),
        "artifact_sizes_bytes": artifact_sizes_bytes,
        "total_size_bytes": sum(artifact_sizes_bytes.values()),
        "bundle_sha256_expected": expected_bundle_hash,
        "bundle_sha256_actual": actual_bundle_hash,
        "bundle_sha256_matches": bundle_matches,
        "claim_boundary": "hash_verification_checks_local_raw_bundle_bytes_not_capture_semantic_quality",
    }


def _read_json_object_for_verification(path: Path, errors: list[str]) -> Dict[str, Any]:
    try:
        payload = read_json_any(path)
    except Exception as exc:
        errors.append(f"invalid_json:{path.name}:{type(exc).__name__}")
        return {}
    if not isinstance(payload, Mapping):
        errors.append(f"invalid_json_object:{path.name}")
        return {}
    return dict(payload)


def _validate_all_json_sidecars(raw_prefix_path: Path, errors: list[str]) -> None:
    files, tree_errors = _raw_tree_entries(raw_prefix_path)
    errors.extend(tree_errors)
    for path in files:
        relative = path.relative_to(raw_prefix_path).as_posix()
        if path.suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, Mapping):
                    errors.append(f"malformed_sidecar:{relative}:not_object")
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"malformed_sidecar:{relative}:{type(exc).__name__}")
        elif path.suffix == ".jsonl":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if not isinstance(value, Mapping):
                            errors.append(f"malformed_sidecar:{relative}:line_{line_number}:not_object")
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"malformed_sidecar:{relative}:{type(exc).__name__}")


def _identity_value(payload: Mapping[str, Any], snake: str, camel: str) -> str:
    return str(payload.get(snake) or payload.get(camel) or "").strip()


def _validate_current_manifest_fields(manifest: Mapping[str, Any], errors: list[str]) -> None:
    required_text_fields = (
        "capture_source",
        "capture_tier_hint",
        "capture_profile_id",
        "coordinate_frame_session_id",
        "video_uri",
        "app_version",
        "app_build",
        "ios_version",
        "ios_build",
        "hardware_model_identifier",
        "device_model_marketing",
        "rights_profile",
    )
    for field_name in required_text_fields:
        if not str(manifest.get(field_name) or "").strip():
            errors.append(f"manifest_missing_field:{field_name}")
    if not isinstance(manifest.get("capture_capabilities"), Mapping):
        errors.append("manifest_invalid_field:capture_capabilities")
    for field_name in ("has_lidar", "depth_supported"):
        if not isinstance(manifest.get(field_name), bool):
            errors.append(f"manifest_invalid_field:{field_name}")
    for field_name in ("capture_start_epoch_ms", "width", "height"):
        value = manifest.get(field_name)
        valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        if not valid:
            errors.append(f"manifest_invalid_field:{field_name}")
    fps_source = manifest.get("fps_source")
    fps_valid = bool(
        isinstance(fps_source, (int, float))
        and not isinstance(fps_source, bool)
        and math.isfinite(float(fps_source))
        and float(fps_source) > 0.0
    )
    if not fps_valid:
        errors.append("manifest_invalid_field:fps_source")
    requested_outputs = manifest.get("requested_outputs")
    if not isinstance(requested_outputs, list) or any(
        not isinstance(item, str) or not item.strip() for item in requested_outputs
    ):
        errors.append("manifest_invalid_field:requested_outputs")


def verify_canonical_raw_bundle_path(
    raw_prefix_path: Path,
    *,
    expected_bucket: Optional[str] = None,
    expected_scene_id: Optional[str] = None,
    expected_capture_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify canonical V3 intake before any product artifact is derived.

    Legacy bundles are classified explicitly as degraded. A bundle that declares
    any V3 schema token is treated as current and must pass every check.
    """

    errors: list[str] = []
    manifest_path = raw_prefix_path / "manifest.json"
    if not manifest_path.is_file():
        errors.append("missing_required_file:manifest.json")
        manifest: Dict[str, Any] = {}
    else:
        manifest = _read_json_object_for_verification(manifest_path, errors)

    current_v3 = _is_current_v3_manifest(manifest)
    for relative in _FORBIDDEN_PIPELINE_DERIVATIVES_IN_RAW:
        if (raw_prefix_path / relative).exists() or (raw_prefix_path / relative).is_symlink():
            errors.append(f"pipeline_derivative_forbidden_in_raw:{relative}")
    if not current_v3:
        legacy_digest = None
        files, tree_errors = _raw_tree_entries(raw_prefix_path)
        errors.extend(tree_errors)
        if files and not tree_errors:
            hashes = {
                path.relative_to(raw_prefix_path).as_posix(): _sha256_file(path)
                for path in files
                if path.name != "hashes.json"
            }
            legacy_digest = _bundle_hash(hashes) if hashes else None
        return {
            "schema_version": "raw_bundle_intake_verification.v1",
            "status": "quarantined" if errors or not manifest else "legacy_degraded",
            "valid_for_derivation": bool(manifest and not errors),
            "current_schema": False,
            "errors": sorted(set(errors)),
            "quarantine_reasons": sorted(set(errors)) if errors else [],
            "intake_digest": legacy_digest,
            "hash_verification": None,
            "claim_boundary": "legacy_bundle_has_no_current_v3_integrity_claim_and_is_not_public_launch_proof",
        }

    if str(manifest.get("schema_version") or "").strip().lower() != "v3":
        errors.append("invalid_manifest_schema_version")
    capture_schema = str(manifest.get("capture_schema_version") or "").strip().lower()
    if not capture_schema.startswith("3."):
        errors.append("invalid_capture_schema_version")
    _validate_current_manifest_fields(manifest, errors)

    for required in ("hashes.json", "capture_upload_complete.json"):
        if not (raw_prefix_path / required).is_file():
            errors.append(f"missing_required_file:{required}")

    _validate_all_json_sidecars(raw_prefix_path, errors)
    hash_report = verify_raw_bundle_hashes_path(raw_prefix_path)
    errors.extend(str(error) for error in hash_report.get("errors", []))
    hashes_payload = _read_json_object_for_verification(raw_prefix_path / "hashes.json", errors)
    if hashes_payload and str(hashes_payload.get("schema_version") or "").strip().lower() != "v1":
        errors.append("invalid_hash_manifest_schema_version")

    scene_id = _identity_value(manifest, "scene_id", "sceneId")
    capture_id = _identity_value(manifest, "capture_id", "captureId")
    if not scene_id:
        errors.append("missing_identity:manifest.json:scene_id")
    if not capture_id:
        errors.append("missing_identity:manifest.json:capture_id")
    if expected_scene_id and scene_id and scene_id != expected_scene_id:
        errors.append("path_identity_mismatch:manifest.json:scene_id")
    if expected_capture_id and capture_id and capture_id != expected_capture_id:
        errors.append("path_identity_mismatch:manifest.json:capture_id")

    for filename in _IDENTITY_SIDECARS:
        path = raw_prefix_path / filename
        if not path.is_file():
            continue
        payload = _read_json_object_for_verification(path, errors)
        sidecar_scene_id = _identity_value(payload, "scene_id", "sceneId")
        sidecar_capture_id = _identity_value(payload, "capture_id", "captureId")
        if not sidecar_scene_id:
            errors.append(f"missing_identity:{filename}:scene_id")
        elif scene_id and sidecar_scene_id != scene_id:
            errors.append(f"identity_mismatch:{filename}:scene_id")
        if not sidecar_capture_id:
            errors.append(f"missing_identity:{filename}:capture_id")
        elif capture_id and sidecar_capture_id != capture_id:
            errors.append(f"identity_mismatch:{filename}:capture_id")

    completion_path = raw_prefix_path / "capture_upload_complete.json"
    completion = (
        _read_json_object_for_verification(completion_path, errors)
        if completion_path.is_file()
        else {}
    )
    if completion:
        if str(completion.get("schema_version") or "").strip().lower() != "v1":
            errors.append("invalid_upload_complete_schema_version")
        status = str(completion.get("status") or "").strip().lower()
        if status and status not in _UPLOAD_COMPLETE_STATUS_VALUES:
            errors.append("upload_not_complete")
        if not str(completion.get("completed_at") or completion.get("completedAt") or "").strip():
            errors.append("upload_completion_timestamp_missing")
        expected_raw_prefix = (
            f"scenes/{expected_scene_id}/captures/{expected_capture_id}/raw"
            if expected_scene_id and expected_capture_id
            else None
        )
        completion_raw_prefix = str(completion.get("raw_prefix") or completion.get("rawPrefix") or "").strip()
        if not completion_raw_prefix:
            errors.append("upload_raw_prefix_missing")
        elif expected_raw_prefix and completion_raw_prefix != expected_raw_prefix:
            errors.append("upload_raw_prefix_mismatch")

    video_uri = str(manifest.get("video_uri") or "").strip()
    if not video_uri:
        errors.append("missing_manifest_video_uri")
    elif "://" not in video_uri:
        video_path, video_path_error = _safe_raw_relative_path(raw_prefix_path, video_uri)
        if video_path_error:
            errors.append(f"invalid_video_path:{video_path_error}")
        elif video_path is None or not video_path.is_file() or video_path.stat().st_size <= 0:
            errors.append("missing_or_empty_raw_video")
    elif video_uri.startswith("gs://"):
        _scheme, _separator, remainder = video_uri.partition("://")
        _bucket, _slash, object_name = remainder.partition("/")
        expected_prefix = (
            f"scenes/{expected_scene_id}/captures/{expected_capture_id}/raw/"
            if expected_scene_id and expected_capture_id
            else ""
        )
        if (
            (expected_bucket and _bucket != expected_bucket)
            or not expected_prefix
            or not object_name.startswith(expected_prefix)
        ):
            errors.append("raw_video_uri_outside_capture")
        else:
            relative_video = object_name[len(expected_prefix) :]
            video_path, video_path_error = _safe_raw_relative_path(raw_prefix_path, relative_video)
            if video_path_error:
                errors.append(f"invalid_video_path:{video_path_error}")
            elif video_path is None or not video_path.is_file() or video_path.stat().st_size <= 0:
                errors.append("missing_or_empty_raw_video")
    else:
        errors.append("raw_video_uri_unsupported_scheme")

    intake_digest = hash_report.get("bundle_sha256_actual")
    unique_errors = sorted(set(errors))
    valid = bool(not unique_errors and hash_report.get("valid") is True and intake_digest)
    return {
        "schema_version": "raw_bundle_intake_verification.v1",
        "status": "verified" if valid else "quarantined",
        "valid_for_derivation": valid,
        "current_schema": True,
        "errors": unique_errors,
        "quarantine_reasons": unique_errors,
        "intake_digest": intake_digest,
        "hash_verification": hash_report,
        "identity": {
            "scene_id": scene_id or None,
            "capture_id": capture_id or None,
            "expected_scene_id": expected_scene_id,
            "expected_capture_id": expected_capture_id,
        },
        "claim_boundary": "verification_proves_local_raw_bundle_integrity_and_contract_shape_not_capture_semantic_quality",
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
    raw_manifest_payload = read_json(manifest_path)
    current_v3 = _is_current_v3_manifest(raw_manifest_payload)
    if current_v3:
        parsed = parse_gs_uri(raw_prefix_uri)
        key_parts = PurePosixPath(parsed.key).parts
        expected_scene_id = None
        expected_capture_id = None
        if len(key_parts) >= 5 and key_parts[-5] == "scenes" and key_parts[-3] == "captures" and key_parts[-1] == "raw":
            expected_scene_id = key_parts[-4]
            expected_capture_id = key_parts[-2]
        report = verify_canonical_raw_bundle_path(
            manifest_path.parent,
            expected_bucket=parsed.bucket,
            expected_scene_id=expected_scene_id,
            expected_capture_id=expected_capture_id,
        )
        if not report["valid_for_derivation"]:
            raise ValueError(
                "raw_bundle_intake_verification_failed:"
                + ",".join(str(error) for error in report.get("errors", []))
            )
    elif verify_hashes is True:
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
