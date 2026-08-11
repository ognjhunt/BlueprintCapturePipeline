"""Validation and byte verification for the immutable 840313 runtime bundle."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009d_franka_runtime_bundle.v1"
RUNTIME_BUNDLE_ID = "adp009d-840313-franka-runtime-v1"
SOURCE_BUNDLE_ID = "adp009d-840313-interiorgs-sage-v1"
SOURCE_BUNDLE_DIGEST = (
    "sha256:4cbf6781cd43cdf02353e0417aefd9ee4df1a65a99e7dbb2ef69a0a0170f22ba"
)
MATERIALIZED_ROLES = {
    "aura_construction_result",
    "aura_final_surfels",
    "task_volume_exclusion_receipt",
    "task_volume_excluded_surfels",
    "nurec_authoring_receipt",
    "aura_nurec_appearance",
}
REPOSITORY_ROLES = {
    "approved_simready_can",
    "franka_evaluation_harness",
    "canonical_scenario_instance",
    "task_destination",
}
SOURCE_ROLES = {"static_collision_geometry"}
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class RuntimeBundleError(ValueError):
    """Raised when a production runtime input is not the frozen byte."""


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def validate_runtime_bundle_manifest(value: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    source = value.get("source_bundle")
    source_row = dict(source) if isinstance(source, Mapping) else {}
    if value.get("schema_version") != SCHEMA_VERSION:
        blockers.append("runtime_bundle_schema_invalid")
    if value.get("runtime_bundle_id") != RUNTIME_BUNDLE_ID:
        blockers.append("runtime_bundle_id_invalid")
    if value.get("status") != "admitted_development_only":
        blockers.append("runtime_bundle_status_invalid")
    if source_row != {
        "bundle_id": SOURCE_BUNDLE_ID,
        "bundle_digest": SOURCE_BUNDLE_DIGEST,
    }:
        blockers.append("runtime_bundle_source_binding_invalid")
    for field, expected in (
        ("materialized_artifacts", MATERIALIZED_ROLES),
        ("repository_inputs", REPOSITORY_ROLES),
        ("source_bundle_inputs", SOURCE_ROLES),
    ):
        rows = _rows(value.get(field))
        roles = {str(row.get("role") or "") for row in rows}
        if roles != expected or len(rows) != len(expected):
            blockers.append(f"runtime_bundle_{field}_roles_invalid")
        for row in rows:
            if (
                not _SHA256.fullmatch(str(row.get("sha256") or ""))
                or not isinstance(row.get("size_bytes"), int)
                or int(row["size_bytes"]) <= 0
            ):
                blockers.append(
                    f"runtime_bundle_artifact_contract_invalid:{row.get('role') or 'unknown'}"
                )
    appearance = next(
        (
            row
            for row in _rows(value.get("materialized_artifacts"))
            if row.get("role") == "aura_nurec_appearance"
        ),
        {},
    )
    if appearance.get("visual_only") is not True or appearance.get("collision_authority") is not False:
        blockers.append("runtime_bundle_appearance_authority_invalid")
    expected_digest = canonical_digest(value, digest_field="runtime_bundle_digest")
    if value.get("runtime_bundle_digest") != expected_digest:
        blockers.append("runtime_bundle_digest_invalid")
    return sorted(set(blockers))


def _verify_file(path: Path, row: Mapping[str, Any]) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != row.get("size_bytes")
        or _file_digest(path) != row.get("sha256")
    ):
        raise RuntimeBundleError(
            f"runtime_bundle_artifact_invalid:{row.get('role') or 'unknown'}"
        )


def verify_materialized_runtime_inputs(
    value: Mapping[str, Any],
    *,
    runtime_input_root: str | Path,
    source_input_root: str | Path,
    repo_root: str | Path,
) -> list[dict[str, str]]:
    blockers = validate_runtime_bundle_manifest(value)
    if blockers:
        raise RuntimeBundleError(",".join(blockers))
    runtime_root = Path(runtime_input_root).expanduser().resolve()
    source_root = Path(source_input_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    verified: list[dict[str, str]] = []
    for row in _rows(value.get("materialized_artifacts")):
        declared = Path(str(row.get("production_path") or ""))
        if declared.parent.name != RUNTIME_BUNDLE_ID:
            raise RuntimeBundleError(
                f"runtime_bundle_production_path_invalid:{row.get('role') or 'unknown'}"
            )
        path = runtime_root / declared.name
        _verify_file(path, row)
        verified.append({"name": str(row["role"]), "path": str(path), "digest": str(row["sha256"])})
    for row in _rows(value.get("repository_inputs")):
        relative = Path(str(row.get("path") or ""))
        path = (repo / relative).resolve()
        if repo not in path.parents:
            raise RuntimeBundleError(
                f"runtime_bundle_repository_path_invalid:{row.get('role') or 'unknown'}"
            )
        _verify_file(path, row)
        verified.append({"name": str(row["role"]), "path": str(path), "digest": str(row["sha256"])})
    for row in _rows(value.get("source_bundle_inputs")):
        filename = Path(str(row.get("filename") or ""))
        if filename.name != str(filename) or not filename.name:
            raise RuntimeBundleError(
                f"runtime_bundle_source_path_invalid:{row.get('role') or 'unknown'}"
            )
        path = source_root / filename.name
        _verify_file(path, row)
        verified.append({"name": str(row["role"]), "path": str(path), "digest": str(row["sha256"])})
    return verified


__all__ = [
    "RUNTIME_BUNDLE_ID",
    "RuntimeBundleError",
    "validate_runtime_bundle_manifest",
    "verify_materialized_runtime_inputs",
]
