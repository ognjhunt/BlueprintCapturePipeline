"""Immutable Pipeline-owned catalog for Task Evaluation Run planning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import (
    EvidenceMethodProfile,
    QualificationRecord,
    canonical_digest,
)


TASK_EVALUATION_METHOD_CATALOG_PATH_ENV = "BLUEPRINT_TASK_EVALUATION_METHOD_CATALOG_PATH"
MAX_CATALOG_BYTES = 5 * 1024 * 1024


class TaskEvaluationMethodCatalogError(ValueError):
    pass


def _secret_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        found: list[str] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            if (
                lowered in {
                    "authorization",
                    "credential",
                    "credentials",
                    "password",
                    "private_key",
                    "secret",
                    "token",
                }
                or lowered.endswith(("_credential", "_password", "_secret", "_token"))
            ) and child not in (None, "", False, [], {}):
                found.append(path)
            found.extend(_secret_paths(child, path))
        return found
    if isinstance(value, list):
        return [
            path
            for index, child in enumerate(value)
            for path in _secret_paths(child, f"{prefix}[{index}]")
        ]
    return []


def validate_task_evaluation_method_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    catalog = json.loads(json.dumps(dict(value)))
    if catalog.get("schema_version") != "task_evaluation_method_catalog.v1":
        raise TaskEvaluationMethodCatalogError("method_catalog_schema_version_invalid")
    raw_profiles = catalog.get("method_profiles")
    raw_qualifications = catalog.get("qualifications")
    if not isinstance(raw_profiles, list) or not isinstance(raw_qualifications, list):
        raise TaskEvaluationMethodCatalogError("method_catalog_entries_invalid")
    if _secret_paths(catalog):
        raise TaskEvaluationMethodCatalogError("method_catalog_secret_value_forbidden")
    profiles = [
        EvidenceMethodProfile.from_mapping(row).to_mapping()
        for row in raw_profiles
        if isinstance(row, Mapping)
    ]
    qualifications = [
        QualificationRecord.from_mapping(row).to_mapping()
        for row in raw_qualifications
        if isinstance(row, Mapping)
    ]
    if len(profiles) != len(raw_profiles) or len(qualifications) != len(raw_qualifications):
        raise TaskEvaluationMethodCatalogError("method_catalog_entries_invalid")
    profile_by_digest = {row["method_profile_digest"]: row for row in profiles}
    if len(profile_by_digest) != len(profiles):
        raise TaskEvaluationMethodCatalogError("method_catalog_duplicate_profile_digest")
    qualification_ids: set[str] = set()
    for qualification in qualifications:
        qualification_id = qualification["qualification_id"]
        if qualification_id in qualification_ids:
            raise TaskEvaluationMethodCatalogError("method_catalog_duplicate_qualification_id")
        qualification_ids.add(qualification_id)
        profile = profile_by_digest.get(qualification["method_profile_digest"])
        if profile is None:
            raise TaskEvaluationMethodCatalogError("method_catalog_qualification_profile_missing")
        if any(
            qualification[field] != profile[profile_field]
            for field, profile_field in (
                ("method_id", "method_id"),
                ("method_version", "version"),
                ("implementation_digest", "implementation_digest"),
            )
        ):
            raise TaskEvaluationMethodCatalogError("method_catalog_qualification_profile_mismatch")
        if qualification["claim_type"] not in profile["supported_claim_types"]:
            raise TaskEvaluationMethodCatalogError("method_catalog_qualification_claim_mismatch")
    normalized = {
        "schema_version": "task_evaluation_method_catalog.v1",
        "catalog_id": str(catalog.get("catalog_id") or "").strip(),
        "version": str(catalog.get("version") or "").strip(),
        "method_profiles": sorted(
            profiles, key=lambda row: (row["method_id"], row["version"], row["method_profile_digest"])
        ),
        "qualifications": sorted(
            qualifications,
            key=lambda row: (
                row["method_id"],
                row["claim_type"],
                row["qualification_id"],
            ),
        ),
        "proof_boundary": {
            "catalog_entry_is_execution_authorization": False,
            "provider_availability_is_qualification": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
    }
    if not normalized["catalog_id"] or not normalized["version"]:
        raise TaskEvaluationMethodCatalogError("method_catalog_identity_missing")
    expected_digest = canonical_digest(normalized, digest_field="catalog_digest")
    supplied_digest = catalog.get("catalog_digest")
    if supplied_digest is not None and supplied_digest != expected_digest:
        raise TaskEvaluationMethodCatalogError("method_catalog_digest_mismatch")
    normalized["catalog_digest"] = expected_digest
    return normalized


def load_task_evaluation_method_catalog(path: str | Path | None = None) -> dict[str, Any]:
    configured = str(path or os.getenv(TASK_EVALUATION_METHOD_CATALOG_PATH_ENV) or "").strip()
    if not configured:
        raise TaskEvaluationMethodCatalogError("method_catalog_not_configured")
    resolved = Path(configured).expanduser().resolve()
    if not resolved.is_file():
        raise TaskEvaluationMethodCatalogError("method_catalog_not_found")
    if resolved.stat().st_size > MAX_CATALOG_BYTES:
        raise TaskEvaluationMethodCatalogError("method_catalog_too_large")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationMethodCatalogError("method_catalog_unreadable") from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationMethodCatalogError("method_catalog_not_object")
    return validate_task_evaluation_method_catalog(value)


__all__ = [
    "TASK_EVALUATION_METHOD_CATALOG_PATH_ENV",
    "TaskEvaluationMethodCatalogError",
    "load_task_evaluation_method_catalog",
    "validate_task_evaluation_method_catalog",
]
