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
import json
import re
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


def build_public_scene_suite_index_receipt(
    value: Mapping[str, Any], *, evaluated_on: dt.date | str
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
        "artifact_bytes_opened": False,
        "artifact_bytes_verified": False,
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
