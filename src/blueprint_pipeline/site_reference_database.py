"""Site Reference Database v1 contract helpers.

The site reference database is a derived support layer. These helpers keep the
local contract executable without calling provider or live WebApp services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .common import read_json, utc_now_iso, write_json

SITE_REFERENCE_DATABASE_SCHEMA_VERSION = "site_reference_database.v1"
WEBAPP_PROJECTION_SCHEMA_VERSION = "site_reference_webapp_projection.v1"

REFERENCE_RECORD_REQUIRED_FIELDS = (
    "reference_id",
    "site_id",
    "scene_id",
    "capture_id",
    "authority_level",
    "storage_class",
    "capture_session_id",
    "coordinate_frame_session_id",
    "frame_id",
    "frame_index",
    "t_capture_sec",
    "T_world_camera",
    "T_site_camera",
    "intrinsics",
    "depth_uri",
    "confidence_uri",
    "embedding_uri",
    "frame_uri",
    "thumbnail_uri",
    "privacy_source",
    "geometry_source",
    "provenance_lineage",
    "privacy_lineage",
    "rights_lineage",
    "quality",
    "retrieval_signals",
    "visibility_cells",
    "anchor_observations",
    "captured_at",
    "indexed_at",
)

MANIFEST_REQUIRED_FIELDS = (
    "schema_version",
    "site_id",
    "authority_level",
    "storage_class",
    "raw_capture_authority",
    "total_reference_frames",
    "capture_count",
    "chunk_count",
    "captures",
    "coverage_summary",
    "readiness",
    "artifact_uris",
    "last_updated",
)

DENSE_RECORD_FIELD_KEYS = frozenset(
    {
        "reference_records",
        "records",
        "references",
        "T_world_camera",
        "T_site_camera",
        "intrinsics",
        "depth_uri",
        "confidence_uri",
        "embedding_uri",
        "frame_uri",
        "thumbnail_uri",
        "splat_uri",
        "plucker_map_uri",
        "visibility_cells",
        "geometry_fingerprint",
    }
)


class SiteReferenceContractError(ValueError):
    """Raised when a site-reference artifact violates the local v1 contract."""


def validate_site_reference_record(record: Mapping[str, Any]) -> None:
    """Validate the shape of one `site_reference_index.jsonl` row."""
    missing = [field for field in REFERENCE_RECORD_REQUIRED_FIELDS if field not in record]
    if missing:
        raise SiteReferenceContractError(
            "site_reference_record_missing_fields:" + ",".join(missing)
        )
    if record.get("authority_level") != "derived_reference_record":
        raise SiteReferenceContractError("site_reference_record_authority_level_invalid")
    if record.get("storage_class") != "jsonl_reference_record":
        raise SiteReferenceContractError("site_reference_record_storage_class_invalid")
    _validate_matrix_or_null(record.get("T_world_camera"), field="T_world_camera", allow_null=False)
    _validate_matrix_or_null(record.get("T_site_camera"), field="T_site_camera", allow_null=True)
    if not isinstance(record.get("intrinsics"), Mapping) or not record.get("intrinsics"):
        raise SiteReferenceContractError("site_reference_record_intrinsics_missing")
    for lineage_field in ("provenance_lineage", "privacy_lineage", "rights_lineage"):
        if not isinstance(record.get(lineage_field), Mapping):
            raise SiteReferenceContractError(f"site_reference_record_{lineage_field}_invalid")


def validate_site_reference_manifest(payload: Mapping[str, Any]) -> None:
    """Validate the site-reference manifest summary shape."""
    missing = [field for field in MANIFEST_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise SiteReferenceContractError(
            "site_reference_manifest_missing_fields:" + ",".join(missing)
        )
    if payload.get("schema_version") != SITE_REFERENCE_DATABASE_SCHEMA_VERSION:
        raise SiteReferenceContractError("site_reference_manifest_schema_version_invalid")
    if payload.get("authority_level") != "derived_site_reference_manifest":
        raise SiteReferenceContractError("site_reference_manifest_authority_level_invalid")
    if payload.get("storage_class") != "object_storage_manifest":
        raise SiteReferenceContractError("site_reference_manifest_storage_class_invalid")
    if not isinstance(payload.get("artifact_uris"), Mapping):
        raise SiteReferenceContractError("site_reference_manifest_artifact_uris_invalid")
    if not isinstance(payload.get("readiness"), Mapping):
        raise SiteReferenceContractError("site_reference_manifest_readiness_invalid")


def build_reference_record_lineage(
    *,
    capture_prefix_uri: Optional[str],
    descriptor_uri: Optional[str],
    geometry_source: str,
    privacy_source: str,
    descriptor: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Build conservative provenance/privacy/rights lineage for an index record."""
    raw_manifest_uri = f"{capture_prefix_uri}/raw/manifest.json" if capture_prefix_uri else None
    raw_rights_uri = f"{capture_prefix_uri}/raw/rights_consent.json" if capture_prefix_uri else None
    rights = _rights_payload_from_descriptor(descriptor)
    derived_generation_allowed = _first_present_bool(
        rights,
        (
            "derived_scene_generation_allowed",
            "derived_generation_allowed",
            "world_model_generation_allowed",
            "commercialization_allowed",
        ),
    )
    rights_status = str(
        rights.get("rights_status")
        or rights.get("consent_status")
        or rights.get("rights_profile")
        or "unknown"
    ).strip()
    return {
        "provenance_lineage": {
            "raw_capture_prefix_uri": capture_prefix_uri,
            "raw_manifest_uri": raw_manifest_uri,
            "capture_descriptor_uri": descriptor_uri,
            "derived_from": [
                "raw_capture",
                "capture_descriptor",
                "privacy_safe_video",
                "geometry_reference",
            ],
            "geometry_source": geometry_source,
        },
        "privacy_lineage": {
            "privacy_source": privacy_source,
            "privacy_safe_required": True,
            "privacy_status": "privacy_safe_source" if privacy_source.startswith("privacy/") else "raw_or_unknown_source",
        },
        "rights_lineage": {
            "rights_source_uri": raw_rights_uri,
            "rights_status": rights_status or "unknown",
            "derived_scene_generation_allowed": derived_generation_allowed,
            "claim_policy": "do_not_infer_rights_clearance",
        },
    }


def build_site_reference_manifest_payload(
    *,
    site_id: str,
    total_reference_frames: int,
    capture_count: int,
    chunk_count: int,
    captures: Iterable[Mapping[str, Any]],
    coverage_summary: Mapping[str, Any],
    artifact_uris: Mapping[str, Any],
    readiness: Mapping[str, Any],
    site_frame_established: bool,
    last_updated: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": SITE_REFERENCE_DATABASE_SCHEMA_VERSION,
        "site_id": site_id,
        "authority_level": "derived_site_reference_manifest",
        "storage_class": "object_storage_manifest",
        "raw_capture_authority": {
            "authority": "BlueprintCapture raw bundle",
            "rule": "Raw capture, provenance, rights, privacy, timestamps, poses, and device metadata remain authoritative.",
        },
        "total_reference_frames": int(total_reference_frames),
        "capture_count": int(capture_count),
        "chunk_count": int(chunk_count),
        "captures": [dict(item) for item in captures],
        "coverage_summary": dict(coverage_summary),
        "readiness": dict(readiness),
        "artifact_uris": {str(key): value for key, value in artifact_uris.items() if value},
        "last_updated": last_updated or utc_now_iso(),
        "site_frame_established": bool(site_frame_established),
    }
    validate_site_reference_manifest(payload)
    return payload


def build_site_reference_summary_projection(
    *,
    site_id: str,
    site_root: Path,
    site_index_path: Path,
    storage_root: Path,
    manifest_payload: Optional[Mapping[str, Any]] = None,
    validation_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a WebApp/Firestore-safe summary projection from local artifacts."""
    manifest = dict(manifest_payload or _read_optional_json(site_root / "site_reference_manifest.json"))
    validation = dict(validation_payload or _read_optional_json(site_root / "retrieval_validation.json"))
    counts = {
        "total_reference_frames": int(manifest.get("total_reference_frames") or 0),
        "capture_count": int(manifest.get("capture_count") or 0),
        "chunk_count": int(manifest.get("chunk_count") or validation.get("chunk_count") or 0),
    }
    coverage_summary = (
        dict(manifest.get("coverage_summary") or {})
        if isinstance(manifest.get("coverage_summary"), Mapping)
        else {}
    )
    readiness = _site_reference_readiness(
        manifest=manifest,
        validation=validation,
        counts=counts,
    )
    payload: Dict[str, Any] = {
        "schema_version": WEBAPP_PROJECTION_SCHEMA_VERSION,
        "site_id": site_id,
        "authority_level": "derived_summary_projection",
        "storage_class": "firestore_summary_safe",
        "artifact_uris": {
            "site_reference_manifest_uri": _path_to_gs_uri(
                site_root / "site_reference_manifest.json",
                storage_root=storage_root,
            ),
            "site_reference_index_uri": _path_to_gs_uri(site_index_path, storage_root=storage_root),
            "site_reference_summary_projection_uri": _path_to_gs_uri(
                site_root / "site_reference_summary_projection.json",
                storage_root=storage_root,
            ),
            "retrieval_validation_uri": _path_to_gs_uri(
                site_root / "retrieval_validation.json",
                storage_root=storage_root,
            ),
            "coverage_map_uri": _path_to_gs_uri(
                site_root / "coverage" / "coverage_map.json",
                storage_root=storage_root,
            ),
            "indices_manifest_uri": _path_to_gs_uri(
                site_root / "indices" / "manifest.json",
                storage_root=storage_root,
            ),
            "site_overlap_graph_uri": _path_to_gs_uri(
                site_root / "site_overlap_graph.json",
                storage_root=storage_root,
            ),
        },
        "readiness": readiness,
        "counts": counts,
        "scores": {
            "coverage_fraction": coverage_summary.get("coverage_fraction"),
            "geometry_fingerprint_coverage": validation.get("geometry_fingerprint_coverage"),
            "mean_staticness_score": validation.get("mean_staticness_score"),
            "aligned_fraction": validation.get("aligned_fraction"),
        },
        "blockers": readiness["blockers"],
        "last_updated": str(manifest.get("last_updated") or validation.get("generated_at") or utc_now_iso()),
    }
    assert_summary_projection_safe(payload)
    return payload


def write_site_reference_summary_projection(
    *,
    site_id: str,
    site_root: Path,
    site_index_path: Path,
    storage_root: Path,
    manifest_payload: Optional[Mapping[str, Any]] = None,
    validation_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = build_site_reference_summary_projection(
        site_id=site_id,
        site_root=site_root,
        site_index_path=site_index_path,
        storage_root=storage_root,
        manifest_payload=manifest_payload,
        validation_payload=validation_payload,
    )
    write_json(site_root / "site_reference_summary_projection.json", payload)
    return payload


def assert_summary_projection_safe(payload: Mapping[str, Any]) -> None:
    """Reject dense per-record fields from a WebApp/Firestore summary payload."""
    if payload.get("schema_version") != WEBAPP_PROJECTION_SCHEMA_VERSION:
        raise SiteReferenceContractError("site_reference_projection_schema_version_invalid")
    if payload.get("storage_class") != "firestore_summary_safe":
        raise SiteReferenceContractError("site_reference_projection_storage_class_invalid")
    violations = sorted(_find_dense_field_violations(payload))
    if violations:
        raise SiteReferenceContractError(
            "site_reference_projection_contains_dense_fields:" + ",".join(violations)
        )


def _find_dense_field_violations(value: Any, *, path: str = "$") -> set[str]:
    violations: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in DENSE_RECORD_FIELD_KEYS:
                violations.add(child_path)
            violations.update(_find_dense_field_violations(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.update(_find_dense_field_violations(child, path=f"{path}[{index}]"))
    return violations


def _site_reference_readiness(
    *,
    manifest: Mapping[str, Any],
    validation: Mapping[str, Any],
    counts: Mapping[str, int],
) -> Dict[str, Any]:
    blockers: list[str] = []
    if int(counts.get("total_reference_frames") or 0) <= 0:
        blockers.append("no_reference_frames")
    if int(counts.get("capture_count") or 0) <= 0:
        blockers.append("no_captures_indexed")
    geometry_coverage = _optional_float(validation.get("geometry_fingerprint_coverage"))
    if geometry_coverage is not None and geometry_coverage < 0.5:
        blockers.append("low_geometry_fingerprint_coverage")
    if not bool(manifest.get("site_frame_established")):
        blockers.append("site_frame_not_established")

    if not blockers:
        state = "ready"
    elif int(counts.get("total_reference_frames") or 0) > 0:
        state = "degraded"
    else:
        state = "blocked"
    return {
        "state": state,
        "blockers": blockers,
        "operational_launch_ready": False,
        "claim_policy": "local_site_reference_readiness_only",
    }


def _rights_payload_from_descriptor(descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = [
        descriptor.get("capture_rights"),
        descriptor.get("rights"),
        descriptor.get("rights_consent"),
    ]
    metadata = descriptor.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.extend(
            [
                metadata.get("capture_rights"),
                metadata.get("rights"),
                metadata.get("rights_consent"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _first_present_bool(payload: Mapping[str, Any], keys: Iterable[str]) -> Optional[bool]:
    for key in keys:
        if key in payload:
            value = payload.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "yes", "1", "allowed"}:
                    return True
                if normalized in {"false", "no", "0", "blocked", "denied"}:
                    return False
    return None


def _read_optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def _path_to_gs_uri(path: Path, *, storage_root: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(storage_root.resolve())
    except ValueError:
        return str(path)
    parts = rel.parts
    if len(parts) < 2:
        return None
    bucket = parts[0]
    key = "/".join(parts[1:])
    return f"gs://{bucket}/{key}"


def _validate_matrix_or_null(value: Any, *, field: str, allow_null: bool) -> None:
    if value is None and allow_null:
        return
    if not isinstance(value, list) or len(value) != 4:
        raise SiteReferenceContractError(f"site_reference_record_{field}_invalid")
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise SiteReferenceContractError(f"site_reference_record_{field}_invalid")


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
