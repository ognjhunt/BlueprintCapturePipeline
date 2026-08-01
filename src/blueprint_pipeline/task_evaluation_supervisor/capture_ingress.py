"""Deterministic, bounded projection of a completed capture build for agents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..decision_evidence_contracts import canonical_digest


CAPTURE_BUILD_INGRESS_SCHEMA_VERSION = "task_evaluation_capture_build_ingress.v1"
_MAX_JSON_BYTES = 2_000_000
_STANDALONE_MANIFEST_LABEL = "submitted_manifest.json"
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
    "evaluation_prep/capture_profile_validation.json",
    "pipeline/evaluation_prep/capture_profile_validation.json",
)
_SAFE_FIELDS = {
    "schema_version",
    "scene_id",
    "capture_id",
    "capture_session_id",
    "intake_id",
    "site_submission_id",
    "buyer_request_id",
    "capture_job_id",
    "site_slug",
    "site_type",
    "site_type_source",
    "intended_space_type",
    "capture_source",
    "capture_mode",
    "capture_modality",
    "capture_authority_profile",
    "capture_tier",
    "capture_digest",
    "envelope_digest",
    "qa_report_digest",
    "object_manifest_digest",
    "source_capture_digest",
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
    "declared_capture_authority_profile",
    "compatible_capture_authority_profile",
    "validation_status",
    "probe_receipt_digests",
    "probe_source_file_digests",
    "observed_processing_lanes",
    "native_normalization_digest",
    "agent_selected_capture_profile",
    "agent_may_change_capture_profile",
    "proof_effect",
    "claim_ceiling",
    "legal_next_actions",
    "capture_profile_routing_binding_digest",
    "capture_profile_validation_digest",
}


class CaptureBuildIngressError(ValueError):
    """Raised when a capture build cannot be safely projected."""


def capture_build_source_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the unique non-secret source identity declared by known manifests.

    The capture-build digest binds the bounded projection itself.  This helper
    additionally exposes raw-capture and intake digests so downstream control
    artifacts can prove that they refer to the same admitted source.  Missing
    values remain missing and conflicting declarations fail closed.
    """

    capture_build = validate_capture_build_ingress(value)
    fields = (
        "capture_session_id",
        "intake_id",
        "capture_digest",
        "envelope_digest",
        "qa_report_digest",
        "object_manifest_digest",
    )
    candidates: dict[str, set[str]] = {field: set() for field in fields}
    for artifact in capture_build["artifacts"]:
        projection = artifact["approved_projection"]
        for field in fields:
            candidate = str(projection.get(field) or "").strip()
            if candidate:
                candidates[field].add(candidate)
    conflicts = sorted(field for field, values in candidates.items() if len(values) > 1)
    if conflicts:
        raise CaptureBuildIngressError(
            f"capture_build_source_binding_conflict:{','.join(conflicts)}"
        )
    invalid_digests = sorted(
        field
        for field in (
            "capture_digest",
            "envelope_digest",
            "qa_report_digest",
            "object_manifest_digest",
        )
        for candidate in candidates[field]
        if re.fullmatch(r"sha256:[0-9a-f]{64}", candidate) is None
    )
    invalid_identifiers = sorted(
        field
        for field in ("capture_session_id", "intake_id")
        for candidate in candidates[field]
        if len(candidate) > 192 or any(ord(character) < 32 for character in candidate)
    )
    if invalid_digests or invalid_identifiers:
        invalid = sorted(set(invalid_digests + invalid_identifiers))
        raise CaptureBuildIngressError(f"capture_build_source_binding_invalid:{','.join(invalid)}")
    binding = {
        "schema_version": "task_evaluation_capture_source_binding.v1",
        "capture_build_digest": capture_build["capture_build_digest"],
        **{field: next(iter(values)) if values else None for field, values in candidates.items()},
        "raw_media_included": False,
        "source_binding_is_proof_upgrade": False,
    }
    binding["source_binding_digest"] = canonical_digest(
        binding,
        digest_field="source_binding_digest",
    )
    return binding


def validate_capture_build_ingress(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete bounded projection accepted by the supervisor."""

    required_fields = {
        "schema_version",
        "source_kind",
        "artifact_count",
        "artifacts",
        "raw_media_included",
        "arbitrary_files_read",
        "projection_is_authoritative_evidence",
        "requires_deterministic_capture_validation",
        "capture_build_digest",
    }
    if set(value) != required_fields:
        raise CaptureBuildIngressError("capture_build_ingress_fields_invalid")
    artifacts = value.get("artifacts")
    artifact_count = value.get("artifact_count")
    source_kind = value.get("source_kind")
    if (
        value.get("schema_version") != CAPTURE_BUILD_INGRESS_SCHEMA_VERSION
        or source_kind not in {"capture_root", "manifest"}
        or isinstance(artifact_count, bool)
        or not isinstance(artifact_count, int)
        or artifact_count < 1
        or not isinstance(artifacts, list)
        or len(artifacts) != artifact_count
        or artifact_count > len(_KNOWN_ARTIFACTS)
        or (source_kind == "manifest" and artifact_count != 1)
        or value.get("raw_media_included") is not False
        or value.get("arbitrary_files_read") is not False
        or value.get("projection_is_authoritative_evidence") is not False
        or value.get("requires_deterministic_capture_validation") is not True
    ):
        raise CaptureBuildIngressError("capture_build_ingress_contract_invalid")
    seen_paths: set[str] = set()
    required_artifact_fields = {
        "relative_path",
        "sha256",
        "size_bytes",
        "schema_version",
        "top_level_keys",
        "approved_projection",
    }
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != required_artifact_fields:
            raise CaptureBuildIngressError("capture_build_ingress_artifact_fields_invalid")
        relative_path = str(artifact.get("relative_path") or "")
        normalized_path = relative_path.replace("\\", "/")
        if (
            not relative_path
            or normalized_path.startswith("/")
            or ".." in normalized_path.split("/")
            or relative_path in seen_paths
        ):
            raise CaptureBuildIngressError("capture_build_ingress_artifact_path_invalid")
        if source_kind == "capture_root" and relative_path not in _KNOWN_ARTIFACTS:
            raise CaptureBuildIngressError("capture_build_ingress_artifact_unregistered")
        if source_kind == "manifest" and relative_path != _STANDALONE_MANIFEST_LABEL:
            raise CaptureBuildIngressError("capture_build_ingress_manifest_name_invalid")
        seen_paths.add(relative_path)
        sha256 = str(artifact.get("sha256") or "")
        size_bytes = artifact.get("size_bytes")
        top_level_keys = artifact.get("top_level_keys")
        projection = artifact.get("approved_projection")
        if (
            not isinstance(top_level_keys, list)
            or len(top_level_keys) > 200
            or any(not isinstance(key, str) for key in top_level_keys)
        ):
            raise CaptureBuildIngressError("capture_build_ingress_artifact_keys_invalid")
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", sha256)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or size_bytes > _MAX_JSON_BYTES
            or not str(artifact.get("schema_version") or "").strip()
            or top_level_keys != sorted(set(top_level_keys))
            or not isinstance(projection, Mapping)
            or _safe_projection(projection) != dict(projection)
        ):
            raise CaptureBuildIngressError("capture_build_ingress_artifact_contract_invalid")
    expected = canonical_digest(value, digest_field="capture_build_digest")
    if value.get("capture_build_digest") != expected:
        raise CaptureBuildIngressError("capture_build_digest_mismatch")
    return dict(value)


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
        artifacts.append(_load_artifact(source, relative_path=_STANDALONE_MANIFEST_LABEL))
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
    return validate_capture_build_ingress(value)


__all__ = [
    "CAPTURE_BUILD_INGRESS_SCHEMA_VERSION",
    "CaptureBuildIngressError",
    "capture_build_source_binding",
    "load_capture_build_ingress",
    "validate_capture_build_ingress",
]
