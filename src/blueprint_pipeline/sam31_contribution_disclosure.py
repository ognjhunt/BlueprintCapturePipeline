"""Prevent frame-disclosure permission from authorizing a full source-splat upload."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import _parse_ply_header
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha

AUTHORITY_SCHEMA = "public_scene_full_source_provider_disclosure_authority.v1"
PROOF_SCHEMA = "sam31_full_source_provider_disclosure_readback.v1"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("sam31_contribution_disclosure_" + code)


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def validate_full_source_disclosure(
    *, task_authority: Mapping[str, Any], conversion_path: Path,
    standard_splat_path: Path, original_source_path: Path,
    expected_source_commit: str, publisher_scene_id: str,
    approved_roots: Sequence[Path],
    purpose: str = "released_code_segment_contribution_sweep",
) -> dict[str, Any]:
    """Reopen exact conversion/source bytes, then require separate explicit authority.

    A local conversion receipt with upload=false remains local-only evidence.
    Neither changed encoding, frame permission nor general compute spend consent
    supplies authority for disclosing all of its original scene content.
    """
    _require(purpose in {"released_code_segment_contribution_sweep", "exact_source_calibration_gpu_render",
                        "configured_scene_partitioned_source_processing"},
             "purpose_invalid")
    conversion = read(conversion_path, digest_field="receipt_digest")
    _require(conversion.get("schema_version") == "standard_splat_conversion_receipt.v1"
             and conversion.get("status") == "standard_splat_conversion_materialized"
             and conversion.get("repository", {}).get("commit") == expected_source_commit
             and conversion.get("claim_ceiling") == "local_format_conversion_only",
             "conversion_identity_invalid")
    source, output, rights = (conversion.get(name, {}) for name in ("source", "output", "rights"))
    _require(isinstance(source, Mapping) and isinstance(output, Mapping)
             and isinstance(rights, Mapping), "conversion_shape_invalid")
    checked_file(original_source_path, dict(source))
    checked_file(standard_splat_path, dict(output))
    with standard_splat_path.open("rb") as stream:
        _format, actual_count, _properties, _offset = _parse_ply_header(stream)
    _require(type(actual_count) is int and actual_count > 0
             and type(source.get("source_gaussian_count")) is int
             and type(output.get("gaussian_count")) is int
             and source.get("source_bytes_unchanged") is True
             and source.get("source_gaussian_count") == actual_count
             and output.get("gaussian_count") == actual_count
             and output.get("gaussian_count_preserved") is True
             and output.get("standard_3dgs_schema_validated") is True,
             "full_source_count_mismatch")
    _require(rights.get("conversion_execution_location") == "local_only"
             and rights.get("raw_private_upload_authorized") is False
             and rights.get("training_authorized") is False,
             "conversion_rights_invalid")
    _require(bool(str(source.get("dataset") or "").strip())
             and re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision") or "")) is not None
             and re.fullmatch(r"sha256:[0-9a-f]{64}", str(rights.get("terms_digest") or "")) is not None,
             "source_identity_invalid")
    binding = {
        "publisher_scene_id": publisher_scene_id,
        "dataset": source.get("dataset"), "publisher_revision": source.get("revision"),
        "original_source_sha256": source["sha256"], "original_source_size_bytes": source["size_bytes"],
        "standard_splat_sha256": output["sha256"], "standard_splat_size_bytes": output["size_bytes"],
        "retained_gaussian_count": actual_count, "source_gaussian_count": source["source_gaussian_count"],
        "publisher_terms_digest": rights.get("terms_digest"),
    }
    scopes = task_authority.get("full_source_provider_disclosure_authorities")
    reference = (scopes.get(purpose) if isinstance(scopes, Mapping) else None)
    if reference is None:
        reference = task_authority.get("full_source_provider_disclosure_authority")
    _require(isinstance(reference, Mapping), "explicit_full_source_authority_required")
    path = Path(str(reference.get("path") or ""))
    _require(path.is_absolute() and any(path.resolve().is_relative_to(root.resolve())
             for root in approved_roots), "authority_path_invalid")
    checked_file(path, dict(reference))
    authority = read(path, digest_field="authorization_digest")
    _require(authority.get("schema_version") == AUTHORITY_SCHEMA
             and authority.get("status") == "authorized"
             and authority.get("authority_kind") == "explicit_human_full_source_provider_processing"
             and authority.get("authorized_by") == task_authority.get("accepted_by")
             and bool(str(authority.get("authority_reference") or "").strip())
             and bool(str(authority.get("authorized_on") or "").strip())
             and authority.get("agent_accepted_terms") is False
             and authority.get("source_commit") == expected_source_commit
             and authority.get("provider_id") == "vast"
             and authority.get("purpose") == purpose
             and authority.get("source_binding") == binding,
             "explicit_full_source_authority_invalid")
    _require(all(authority.get(key) is True for key in (
        "full_source_scene_content_upload_authorized", "private_provider_processing_authorized",
        "publisher_rights_permit_private_full_source_processing", "provider_retention_terms_accepted",
        "provider_training_terms_accepted", "format_conversion_does_not_reduce_disclosure_scope",
    )) and all(authority.get(key) is False for key in (
        "public_redistribution_authorized", "provider_training_authorized",
    )), "explicit_full_source_scope_invalid")
    basis = authority.get("publisher_rights_basis")
    _require(isinstance(basis, Mapping)
             and basis.get("kind") in {"publisher_license_private_processing", "explicit_publisher_permission"}
             and bool(str(basis.get("scope_explanation") or "").strip()),
             "publisher_rights_basis_missing")
    for name in ("publisher_terms_evidence", "private_processing_permission_evidence"):
        ref = basis.get(name)
        _require(isinstance(ref, Mapping), "publisher_rights_basis_missing")
        evidence = Path(str(ref.get("path") or ""))
        _require(evidence.is_absolute() and any(evidence.resolve().is_relative_to(root.resolve())
                 for root in approved_roots), "publisher_rights_basis_path_invalid")
        checked_file(evidence, dict(ref))
        if name == "publisher_terms_evidence":
            _require(sha(evidence) == rights["terms_digest"], "publisher_terms_mismatch")
    proof = {
        "schema_version": PROOF_SCHEMA, "status": "explicit_full_source_disclosure_verified",
        "payload_kind": "full_source_scene_reencoded_standard_splat", "purpose": purpose,
        "source_binding": binding, "conversion_receipt": _record(conversion_path),
        "conversion_receipt_digest": conversion["receipt_digest"],
        "conversion_rights": dict(rights), "publisher_rights_basis": dict(basis),
        "disclosure_authority": _record(path),
        "disclosure_authority_digest": authority["authorization_digest"],
        "frame_permission_used_as_full_source_authority": False,
        "proof_digest": "",
    }
    proof["proof_digest"] = canonical_digest(proof, digest_field="proof_digest")
    return proof
