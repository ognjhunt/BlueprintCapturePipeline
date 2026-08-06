"""Exact, fail-closed ADP-009 public-suite index admission.

The index composes separately admitted component manifests into one immutable
development-only matrix.  It deliberately binds exact project identities:
another public dataset, a Blueprint-authored look-alike, or a different
inpainting implementation cannot silently satisfy a required role.

A JSON-shaped index always yields a deterministic ``matrix_complete`` or
``blocked`` receipt.  A blocked component is valid evidence of an unresolved
gap, but the matrix is complete only when every exact role appears once and is
admitted.  This module does not open artifacts or create physical, partner, or
deployment evidence.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


INDEX_SCHEMA_VERSION = "public_scene_suite_index.v1"
RECEIPT_SCHEMA_VERSION = "public_scene_suite_index_receipt.v1"
PROGRAM_ID = "arm-decision-proof-v1"
ADP_ITEM = "ADP-009"
GATE_ID = "exact_public_scene_suite_index"
CLAIM_CEILING = "development_only"

REQUIRED_ROLE_PROJECTS: dict[str, str] = {
    "inpaint360_author_smoke": "Inpaint360GS",
    "infusion_primary_adapter": "InFusion",
    "aurafusion360_quality_challenger": "AuraFusion360",
    "interiorgs_appearance_scene": "InteriorGS",
    "sage3d_collision_companion": "SAGE-3D",
    "controlled_background_truth": "Blueprint-controlled",
    "exact_simready_object": "Blueprint-controlled",
    "usd_content_agents_candidate": "NVIDIA-Omniverse/usd-content-agents",
    "physics_positive_control": "Blueprint-controlled",
    "scannetpp_real_transfer": "ScanNet++",
}

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "program_id",
    "adp_item",
    "index_id",
    "components",
    "claim_ceiling",
    "claim_boundaries",
    "index_digest",
}
_COMPONENT_FIELDS = {
    "role",
    "source_project_id",
    "component_manifest_digest",
    "component_admission_receipt_digest",
    "exact_revision",
    "exact_artifact_digest",
    "status",
    "blockers",
}
_REVISION_FIELDS = {"kind", "value"}
_BOUNDARY_EXPECTATIONS: dict[str, bool] = {
    "exact_public_suite_binding": True,
    "public_scene_software_qualified": False,
    "metric_geometry_qualified": False,
    "task_physics_qualified": False,
    "partner_capture_qualified": False,
    "prospective_validation": False,
    "physical_evidence": False,
    "digital_twin": False,
    "deployment_readiness": False,
    "physical_safety": False,
    "customer_value": False,
    "general_sim_to_real_fidelity": False,
}

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,191}$")
_RELEASE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,190}$")
_BLOCKER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}$")
_MOVING_REVISIONS = {
    "current",
    "head",
    "latest",
    "main",
    "master",
    "tip",
    "trunk",
}


class PublicSceneSuiteIndexError(ValueError):
    """The index or evaluation date could not be represented deterministically."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise PublicSceneSuiteIndexError(["index:not_json_serializable"]) from exc
    if not isinstance(cloned, dict):
        raise PublicSceneSuiteIndexError(["index:not_mapping"])
    return cloned


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256.fullmatch(_string(value)))


def _reject_unknown(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    path: str,
    blockers: list[str],
) -> None:
    for key in sorted(set(value) - allowed):
        blockers.append(f"{path}.{key}:unknown_property" if path else f"{key}:unknown_property")


def _parse_date(value: dt.date | str) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return None
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(_string(value))
    except ValueError:
        return None


def _validate_revision(
    value: Any, *, path: str, blockers: list[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        blockers.append(f"{path}:must_be_object")
        return {}
    revision = dict(value)
    _reject_unknown(
        revision,
        allowed=_REVISION_FIELDS,
        path=path,
        blockers=blockers,
    )
    for field in sorted(_REVISION_FIELDS - set(revision)):
        blockers.append(f"{path}.{field}:missing")

    kind = _string(revision.get("kind"))
    revision_value = _string(revision.get("value"))
    if kind == "git_commit":
        if not _GIT_COMMIT.fullmatch(revision_value):
            blockers.append(f"{path}.value:not_exact_git_commit")
    elif kind == "content_digest":
        if not _is_sha256(revision_value):
            blockers.append(f"{path}.value:not_exact_content_digest")
    elif kind == "release_tag":
        if (
            not _RELEASE_TAG.fullmatch(revision_value)
            or revision_value.lower() in _MOVING_REVISIONS
        ):
            blockers.append(f"{path}.value:not_exact_release_tag")
    else:
        blockers.append(f"{path}.kind:invalid")
    return revision


def _validate_component(
    value: Any, *, index: int, blockers: list[str]
) -> dict[str, Any]:
    path = f"components[{index}]"
    if not isinstance(value, Mapping):
        blockers.append(f"{path}:must_be_object")
        return {}
    component = dict(value)
    _reject_unknown(
        component,
        allowed=_COMPONENT_FIELDS,
        path=path,
        blockers=blockers,
    )
    for field in sorted(_COMPONENT_FIELDS - set(component)):
        blockers.append(f"{path}.{field}:missing")

    role = _string(component.get("role"))
    source_project_id = _string(component.get("source_project_id"))
    expected_project = REQUIRED_ROLE_PROJECTS.get(role)
    if expected_project is None:
        blockers.append(f"{path}.role:unknown")
    elif source_project_id != expected_project:
        blockers.append(f"{path}.source_project_id:must_be:{expected_project}")

    if not _is_sha256(component.get("component_manifest_digest")):
        blockers.append(f"{path}.component_manifest_digest:invalid")
    if not _is_sha256(component.get("component_admission_receipt_digest")):
        blockers.append(f"{path}.component_admission_receipt_digest:invalid")
    _validate_revision(
        component.get("exact_revision"),
        path=f"{path}.exact_revision",
        blockers=blockers,
    )
    if not _is_sha256(component.get("exact_artifact_digest")):
        blockers.append(f"{path}.exact_artifact_digest:invalid")

    status = _string(component.get("status"))
    component_blockers = component.get("blockers")
    if status not in {"admitted", "blocked"}:
        blockers.append(f"{path}.status:invalid")
    elif status == "blocked":
        blockers.append(f"{path}.status:blocked")
    if not isinstance(component_blockers, list):
        blockers.append(f"{path}.blockers:must_be_array")
    else:
        normalized_blockers = [_string(item) for item in component_blockers]
        for blocker_index, blocker in enumerate(normalized_blockers):
            if not _BLOCKER.fullmatch(blocker):
                blockers.append(f"{path}.blockers[{blocker_index}]:invalid")
        if len(normalized_blockers) != len(set(normalized_blockers)):
            blockers.append(f"{path}.blockers:duplicate")
        if status == "admitted" and normalized_blockers:
            blockers.append(f"{path}.blockers:admitted_must_be_empty")
        elif status == "blocked" and not normalized_blockers:
            blockers.append(f"{path}.blockers:blocked_must_be_nonempty")
    return component


def _validate_index(value: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    _reject_unknown(value, allowed=_TOP_LEVEL_FIELDS, path="", blockers=blockers)
    for field in sorted(_TOP_LEVEL_FIELDS - set(value)):
        blockers.append(f"{field}:missing")

    if value.get("schema_version") != INDEX_SCHEMA_VERSION:
        blockers.append("schema_version:invalid")
    if value.get("program_id") != PROGRAM_ID:
        blockers.append("program_id:invalid")
    if value.get("adp_item") != ADP_ITEM:
        blockers.append("adp_item:invalid")
    if not _IDENTIFIER.fullmatch(_string(value.get("index_id"))):
        blockers.append("index_id:invalid")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        blockers.append("claim_ceiling:must_be:development_only")

    raw_components = value.get("components")
    if not isinstance(raw_components, list):
        blockers.append("components:must_be_array")
        components: list[dict[str, Any]] = []
    else:
        components = [
            _validate_component(component, index=index, blockers=blockers)
            for index, component in enumerate(raw_components)
        ]
        if len(raw_components) != len(REQUIRED_ROLE_PROJECTS):
            blockers.append(f"components:must_have_exactly:{len(REQUIRED_ROLE_PROJECTS)}")

    roles = [_string(component.get("role")) for component in components]
    for role in REQUIRED_ROLE_PROJECTS:
        count = roles.count(role)
        if count == 0:
            blockers.append(f"components:missing_role:{role}")
        elif count > 1:
            blockers.append(f"components:duplicate_role:{role}")

    boundaries = value.get("claim_boundaries")
    if not isinstance(boundaries, Mapping):
        blockers.append("claim_boundaries:must_be_object")
    else:
        boundary_map = dict(boundaries)
        _reject_unknown(
            boundary_map,
            allowed=set(_BOUNDARY_EXPECTATIONS),
            path="claim_boundaries",
            blockers=blockers,
        )
        for claim, expected in _BOUNDARY_EXPECTATIONS.items():
            if boundary_map.get(claim) is not expected:
                blockers.append(
                    f"claim_boundaries.{claim}:must_be:{str(expected).lower()}"
                )

    supplied_digest = _string(value.get("index_digest"))
    expected_digest = canonical_digest(value, digest_field="index_digest")
    if not supplied_digest:
        blockers.append("index_digest:missing")
    elif not _is_sha256(supplied_digest):
        blockers.append("index_digest:invalid")
    elif supplied_digest != expected_digest:
        blockers.append("index_digest:mismatch")
    return sorted(set(blockers))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path, blocker: str, blockers: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        blockers.append(blocker)
        return {}
    if not isinstance(value, dict):
        blockers.append(blocker)
        return {}
    return value


def _verify_component_files(
    *,
    components: Sequence[Mapping[str, Any]],
    component_root: Path,
    artifact_roots: Sequence[Path],
) -> tuple[list[str], int]:
    blockers: list[str] = []
    artifact_count = 0
    rows = {_string(row.get("role")): dict(row) for row in components}
    for role, expected_project in REQUIRED_ROLE_PROJECTS.items():
        row = rows.get(role) or {}
        manifest_path = component_root / f"{role}.component_manifest.json"
        receipt_path = component_root / f"{role}.component_receipt.json"
        manifest = _load_object(
            manifest_path, f"component_files:{role}:manifest_missing_or_invalid", blockers
        )
        receipt = _load_object(
            receipt_path, f"component_files:{role}:receipt_missing_or_invalid", blockers
        )
        if not manifest or not receipt:
            continue
        manifest_digest = canonical_digest(manifest, digest_field="manifest_digest")
        receipt_digest = canonical_digest(receipt, digest_field="receipt_digest")
        if manifest.get("manifest_digest") != manifest_digest:
            blockers.append(f"component_files:{role}:manifest_digest_mismatch")
        if receipt.get("receipt_digest") != receipt_digest:
            blockers.append(f"component_files:{role}:receipt_digest_mismatch")
        if (
            manifest.get("role") != role
            or receipt.get("role") != role
            or manifest.get("component_id") != receipt.get("component_id")
        ):
            blockers.append(f"component_files:{role}:component_identity_mismatch")
        if manifest.get("source_project_id") != expected_project:
            blockers.append(f"component_files:{role}:source_project_id_mismatch")
        if (
            row.get("source_project_id") != manifest.get("source_project_id")
            or row.get("component_manifest_digest") != manifest.get("manifest_digest")
            or receipt.get("component_manifest_digest") != manifest.get("manifest_digest")
            or row.get("component_admission_receipt_digest") != receipt.get("receipt_digest")
            or row.get("status") != receipt.get("status")
            or row.get("blockers") != receipt.get("blockers")
        ):
            blockers.append(f"component_files:{role}:index_binding_mismatch")
        artifacts = manifest.get("materialized_artifacts")
        if not isinstance(artifacts, list):
            blockers.append(f"component_files:{role}:artifacts_invalid")
            continue
        if receipt.get("status") == "admitted" and not artifacts:
            blockers.append(f"component_files:{role}:admitted_artifacts_missing")
        expected_artifact_digest = canonical_digest({"artifacts": artifacts})
        if row.get("exact_artifact_digest") != expected_artifact_digest:
            blockers.append(f"component_files:{role}:artifact_digest_mismatch")
        for artifact_index, artifact in enumerate(artifacts):
            artifact_count += 1
            if not isinstance(artifact, Mapping):
                blockers.append(
                    f"component_files:{role}:artifact_{artifact_index}:invalid"
                )
                continue
            relative = _string(artifact.get("external_relative_path"))
            candidates: list[Path] = []
            for root in artifact_roots:
                candidate = (root / relative).resolve()
                if relative and (candidate == root or root in candidate.parents) and candidate.is_file():
                    candidates.append(candidate)
            matching = [
                path
                for path in candidates
                if path.stat().st_size == artifact.get("size_bytes")
                and _sha256(path) == artifact.get("sha256")
            ]
            if len(matching) != 1:
                blockers.append(
                    f"component_files:{role}:artifact_{artifact_index}:bytes_missing_or_changed"
                )
    return sorted(set(blockers)), artifact_count


def build_public_scene_suite_index_receipt(
    value: Mapping[str, Any],
    *,
    evaluated_on: dt.date | str,
    component_root: str | Path | None = None,
    artifact_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Return a deterministic exact-suite index receipt.

    ``matrix_complete`` proves only that all exact component declarations are
    admitted and digest-bound.  It does not prove artifact bytes, method
    behavior, geometry, physics, a partner capture, or a physical outcome.
    """

    if not isinstance(value, Mapping):
        raise PublicSceneSuiteIndexError(["index:not_mapping"])
    evaluation_date = _parse_date(evaluated_on)
    if evaluation_date is None:
        raise PublicSceneSuiteIndexError(["evaluation_date:invalid"])

    normalized = _clone(value)
    blockers = _validate_index(normalized)
    raw_components = normalized.get("components")
    components = (
        [dict(row) for row in raw_components if isinstance(row, Mapping)]
        if isinstance(raw_components, list)
        else []
    )
    rows_by_role: dict[str, list[dict[str, Any]]] = {
        role: [row for row in components if _string(row.get("role")) == role]
        for role in REQUIRED_ROLE_PROJECTS
    }
    role_bindings: list[dict[str, Any]] = []
    for role, expected_project in REQUIRED_ROLE_PROJECTS.items():
        matching = rows_by_role[role]
        row = matching[0] if len(matching) == 1 else {}
        role_bindings.append(
            {
                "role": role,
                "expected_project_id": expected_project,
                "source_project_id": _string(row.get("source_project_id")) or None,
                "component_manifest_digest": (
                    _string(row.get("component_manifest_digest")) or None
                ),
                "component_admission_receipt_digest": (
                    _string(row.get("component_admission_receipt_digest")) or None
                ),
                "exact_revision": (
                    dict(row["exact_revision"])
                    if isinstance(row.get("exact_revision"), Mapping)
                    else None
                ),
                "exact_artifact_digest": (
                    _string(row.get("exact_artifact_digest")) or None
                ),
                "declared_status": _string(row.get("status")) or None,
                "component_blockers": sorted(
                    _string(item)
                    for item in row.get("blockers", [])
                    if _string(item)
                )
                if isinstance(row.get("blockers"), list)
                else [],
            }
        )

    file_backed = component_root is not None and bool(artifact_roots)
    artifact_count = 0
    if file_backed:
        resolved_component_root = Path(component_root).expanduser().resolve()
        resolved_artifact_roots = tuple(
            Path(root).expanduser().resolve() for root in artifact_roots
        )
        if not resolved_component_root.is_dir() or any(
            not root.is_dir() for root in resolved_artifact_roots
        ):
            blockers.append("component_files:allowlisted_root_missing")
        else:
            file_blockers, artifact_count = _verify_component_files(
                components=components,
                component_root=resolved_component_root,
                artifact_roots=resolved_artifact_roots,
            )
            blockers.extend(file_blockers)
    else:
        blockers.append("component_files:not_verified")
    blockers = sorted(set(blockers))
    matrix_complete = not blockers
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "adp_item": ADP_ITEM,
        "gate_id": GATE_ID,
        "index_id": _string(normalized.get("index_id")) or None,
        "index_digest": canonical_digest(normalized, digest_field="index_digest"),
        "supplied_index_digest": _string(normalized.get("index_digest")) or None,
        "status": "matrix_complete" if matrix_complete else "blocked",
        "blockers": blockers,
        "evaluated_on": evaluation_date.isoformat(),
        "required_role_count": len(REQUIRED_ROLE_PROJECTS),
        "declared_component_count": len(components),
        "admitted_role_count": sum(
            1
            for binding in role_bindings
            if binding["declared_status"] == "admitted"
            and binding["source_project_id"] == binding["expected_project_id"]
        ),
        "blocked_roles": sorted(
            binding["role"]
            for binding in role_bindings
            if binding["declared_status"] != "admitted"
        ),
        "role_bindings": role_bindings,
        "adp009_matrix_complete": matrix_complete,
        "claim_ceiling": CLAIM_CEILING,
        "artifact_bytes_opened": file_backed and artifact_count > 0,
        "artifact_bytes_verified": file_backed and artifact_count > 0 and not any(
            blocker.startswith("component_files:") for blocker in blockers
        ),
        "public_scene_software_qualified": False,
        "metric_geometry_qualified": False,
        "task_physics_qualified": False,
        "partner_capture_qualified": False,
        "prospective_validation": False,
        "physical_evidence_created": False,
        "deployment_readiness": False,
        "customer_value": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "ADP_ITEM",
    "CLAIM_CEILING",
    "GATE_ID",
    "INDEX_SCHEMA_VERSION",
    "PublicSceneSuiteIndexError",
    "REQUIRED_ROLE_PROJECTS",
    "build_public_scene_suite_index_receipt",
]
