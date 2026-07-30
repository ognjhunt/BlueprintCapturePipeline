"""Deterministic, bounded projection of a completed capture build for agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..decision_evidence_contracts import canonical_digest


CAPTURE_BUILD_INGRESS_SCHEMA_VERSION = "task_evaluation_capture_build_ingress.v1"
_MAX_JSON_BYTES = 2_000_000
_KNOWN_ARTIFACTS = (
    "pipeline_handoff.json",
    "capture_descriptor.json",
    "raw/capture_upload_complete.json",
    "raw/manifest.json",
    "raw/capture_context.json",
    "evaluation_prep/site_package_manifest.json",
    "pipeline/evaluation_prep/site_package_manifest.json",
    "evaluation_prep/task_anchor_manifest.json",
    "pipeline/evaluation_prep/task_anchor_manifest.json",
    "evaluation_prep/rights_review.json",
    "pipeline/evaluation_prep/rights_review.json",
)
_SAFE_FIELDS = {
    "schema_version",
    "scene_id",
    "capture_id",
    "site_submission_id",
    "buyer_request_id",
    "capture_job_id",
    "site_slug",
    "site_type",
    "site_type_source",
    "intended_space_type",
    "capture_source",
    "capture_mode",
    "has_lidar",
    "scale_hint_m_per_unit",
    "requested_outputs",
    "requested_lanes",
    "robot_eval_dataset_requested",
    "status",
    "task_id",
    "task_name",
    "task_family",
    "task_description",
    "task_intent",
    "task_steps",
    "zone_id",
    "zone_name",
    "rights",
    "consent",
    "privacy",
    "rights_status",
    "consent_status",
    "privacy_status",
    "blockers",
}


class CaptureBuildIngressError(ValueError):
    """Raised when a capture build cannot be safely projected."""


def _safe_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projection: dict[str, Any] = {}
    for key in sorted(_SAFE_FIELDS & set(value)):
        field = value[key]
        if isinstance(field, (str, int, float, bool)) or field is None:
            projection[key] = field
        elif isinstance(field, list):
            projection[key] = [
                item
                for item in field[:100]
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        elif isinstance(field, Mapping):
            projection[key] = _safe_projection(field)
    return projection


def _load_artifact(path: Path, *, relative_path: str) -> dict[str, Any]:
    if path.is_symlink():
        raise CaptureBuildIngressError(f"capture_build_symlink_not_allowed:{relative_path}")
    payload = path.read_bytes()
    if len(payload) > _MAX_JSON_BYTES:
        raise CaptureBuildIngressError(f"capture_build_artifact_too_large:{relative_path}")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureBuildIngressError(
            f"capture_build_artifact_invalid_json:{relative_path}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise CaptureBuildIngressError(f"capture_build_artifact_not_object:{relative_path}")
    return {
        "relative_path": relative_path,
        "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "size_bytes": len(payload),
        "schema_version": str(decoded.get("schema_version") or "unknown"),
        "top_level_keys": sorted(str(key) for key in decoded)[:200],
        "approved_projection": _safe_projection(decoded),
    }


def load_capture_build_ingress(path: str | Path) -> dict[str, Any]:
    """Load only known JSON manifests, never arbitrary files from a capture build."""

    source = Path(path).expanduser()
    artifacts: list[dict[str, Any]] = []
    source_kind: str
    if source.is_file():
        source_kind = "manifest"
        artifacts.append(_load_artifact(source, relative_path=source.name))
    elif source.is_dir():
        source_kind = "capture_root"
        resolved_root = source.resolve()
        for relative in _KNOWN_ARTIFACTS:
            candidate = source / relative
            if not candidate.is_file():
                continue
            resolved_candidate = candidate.resolve()
            if resolved_root not in resolved_candidate.parents:
                raise CaptureBuildIngressError(f"capture_build_path_escape:{relative}")
            artifacts.append(_load_artifact(candidate, relative_path=relative))
    else:
        raise CaptureBuildIngressError("capture_build_not_found")
    if not artifacts:
        raise CaptureBuildIngressError("capture_build_has_no_known_manifests")
    value: dict[str, Any] = {
        "schema_version": CAPTURE_BUILD_INGRESS_SCHEMA_VERSION,
        "source_kind": source_kind,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "raw_media_included": False,
        "arbitrary_files_read": False,
        "projection_is_authoritative_evidence": False,
        "requires_deterministic_capture_validation": True,
    }
    value["capture_build_digest"] = canonical_digest(value, digest_field="capture_build_digest")
    return value


__all__ = [
    "CAPTURE_BUILD_INGRESS_SCHEMA_VERSION",
    "CaptureBuildIngressError",
    "load_capture_build_ingress",
]
