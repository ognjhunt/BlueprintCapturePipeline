"""Admit AuraFusion360 only for exact-mask private-derived residual editing.

AuraFusion360 is a released, retraining-based multi-view editor rather than an
in-place publisher-PLY edit.  This gate turns its already-pinned source and
checkpoint rights evidence into a *narrow* backend admission for one through
five residual packets.  It deliberately rejects the historical broad-mask
configuration: every dilation parameter must be zero, and a later execution
must still prove byte-exact outside-mask preservation before any result can be
qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_residual_inpainting_packet import BACKEND_ADMISSION_SCHEMA


REQUEST_SCHEMA = "public_scene_aura_residual_backend_admission_request.v1"
RECEIPT_SCHEMA = BACKEND_ADMISSION_SCHEMA
ABSTENTION_SCHEMA = "public_scene_released_code_inpainting_abstention.v1"
PACKET_ABSTENTION_SCHEMA = "public_scene_residual_inpainting_execution_abstention.v1"
AURA_REPOSITORY = "https://github.com/kkennethwu/AuraFusion360_official"
AURA_COMMIT = "f23b26c44ba84608306ba952510533ebf4c7877d"
AURA_TREE = "cc8447c66448b29bb4d39fec29c031df63d4b179"
AURA_LICENSE_SHA256 = "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
NESTED_COMPONENT_LICENSES = {
    "submodules/diff-surfel-rasterization/LICENSE.md": (
        "sha256:cd5c95b3cfff3acc1bd412420c770f88809331c3db6872df11a970147aa8e81f"
    ),
    "submodules/simple-knn/LICENSE.md": (
        "sha256:c5ba70a2194af2aefe85dfe3da68608dcb3abd21a3aa53b55aa297c2f0b60eb3"
    ),
}
NONCOMMERCIAL_ATTESTATION_SCHEMA = (
    "third_scene_released_code_noncommercial_use_attestation.v1"
)
REQUIRED_CHECKPOINTS = frozenset(
    {
        "aurafusion360_sam2_hiera_large",
        "aurafusion360_marigold_depth_v1_0",
        "aurafusion360_marigold_agdd_v1_0",
        "aurafusion360_sd2_inpainting_exact_checkpoint",
    }
)
REQUIRED_AURA_ARCHIVE_MEMBERS = frozenset(
    {"LICENSE", "inpaint.py", "arguments/__init__.py", *NESTED_COMPONENT_LICENSES}
)


class AuraResidualBackendAdmissionError(ValueError):
    """Stable, typed failures for the exact-mask Aura admission boundary."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraResidualBackendAdmissionError([code]) from exc
    if not isinstance(value, dict):
        raise AuraResidualBackendAdmissionError([code])
    return value


def _file(path: str | Path, *, code: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.is_symlink():
        raise AuraResidualBackendAdmissionError([code])
    return value


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _validated_policy(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AuraResidualBackendAdmissionError([code])
    policy = dict(value)
    retention = policy.get("maximum_retention_days")
    if (
        policy.get("raw_dataset_bytes_upload") is not False
        or policy.get("private_derived_upload") is not True
        or policy.get("provider_training") is not False
        or policy.get("publication") is not False
        or isinstance(retention, bool)
        or not isinstance(retention, int)
        or not 1 <= retention <= 30
    ):
        raise AuraResidualBackendAdmissionError([code])
    return policy


def _validate_execution_authority(path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    authority = _read_object(path, code="aura_residual_execution_authority_unreadable")
    paid = authority.get("paid_compute")
    if (
        authority.get("schema_version") != "third_scene_dual_task_execution_authority.v1"
        or authority.get("program_id") != "arm-decision-proof-v1"
        or authority.get("publisher_scene_id") != "840920"
        or authority.get("authority_kind") != "explicit_user_direction_in_current_goal"
        or not isinstance(authority.get("authorized_by"), str)
        or not authority["authorized_by"].strip()
        or authority.get("private_rights_admitted_scene_derived_uploads_authorized")
        is not True
        or authority.get("raw_interiorgs_upload_authorized") is not False
        or authority.get("training_authorized") is not False
        or authority.get("public_dataset_bytes_publication_authorized") is not False
        or authority.get("retention") != "bounded_to_goal_then_provider_zero"
        or not isinstance(paid, Mapping)
        or paid.get("provider") != "vast"
        or paid.get("zero_retry") is not True
        or paid.get("provider_zero_required_for_lane") is not True
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_execution_authority_invalid"])
    if policy["maximum_retention_days"] > 30:
        raise AuraResidualBackendAdmissionError(["aura_residual_retention_policy_invalid"])
    return authority


def _validate_prerequisite(path: Path) -> dict[str, Any]:
    receipt = _read_object(path, code="aura_residual_prerequisite_unreadable")
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        raise AuraResidualBackendAdmissionError(["aura_residual_prerequisite_digest_invalid"])
    method = (receipt.get("methods") or {}).get("aurafusion360_quality_challenger")
    if (
        receipt.get("schema_version") != "public_scene_method_prerequisite_receipt.v1"
        or not isinstance(method, Mapping)
        or method.get("author_data_rights_established") is not True
        or method.get("checkpoint_rights_established") is not True
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_prerequisite_rights_invalid"])
    observed = {
        str(row.get("artifact_id") or "")
        for row in method.get("remote_snapshots") or []
        if isinstance(row, Mapping) and row.get("rights_established") is True
    }
    if not REQUIRED_CHECKPOINTS.issubset(observed):
        raise AuraResidualBackendAdmissionError(["aura_residual_checkpoint_rights_incomplete"])
    return receipt


def _validate_source_identity_spec(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    spec = _read_object(path, code="aura_residual_source_identity_spec_unreadable")
    if (
        spec.get("schema_version") != "adp_aura_interiorgs_spec.v1"
        or spec.get("source_repository") != AURA_REPOSITORY
        or spec.get("source_commit") != AURA_COMMIT
        or spec.get("source_tree") != AURA_TREE
        or not isinstance(spec.get("source_files"), list)
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_source_identity_spec_invalid"])
    expected: dict[str, dict[str, Any]] = {}
    for row in spec["source_files"]:
        if not isinstance(row, Mapping):
            raise AuraResidualBackendAdmissionError(["aura_residual_source_identity_spec_invalid"])
        name = str(row.get("path") or "")
        if (
            not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or name in expected
            or isinstance(row.get("size_bytes"), bool)
            or not isinstance(row.get("size_bytes"), int)
            or row["size_bytes"] < 0
            or not _digest(row.get("sha256"))
        ):
            raise AuraResidualBackendAdmissionError(["aura_residual_source_identity_spec_invalid"])
        expected[name] = {
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
    required_members = {
        "LICENSE", "inpaint.py", "arguments/__init__.py", *NESTED_COMPONENT_LICENSES
    }
    if not required_members.issubset(expected):
        raise AuraResidualBackendAdmissionError(["aura_residual_source_identity_spec_invalid"])
    return spec, expected


def _validate_source_archive(path: Path, *, expected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (
                len(names) != len(set(names))
                or set(names) != set(expected)
                or any(
                    not name
                    or name.startswith("/")
                    or ".." in Path(name).parts
                    or name.endswith("/")
                    for name in names
                )
            ):
                raise AuraResidualBackendAdmissionError(
                    ["aura_residual_source_archive_manifest_mismatch"]
                )
            license_bytes = archive.read("LICENSE")
            if "Apache License" not in license_bytes.decode("utf-8", errors="ignore"):
                raise AuraResidualBackendAdmissionError(["aura_residual_source_license_invalid"])
            license_sha256 = "sha256:" + hashlib.sha256(license_bytes).hexdigest()
            nested_license_records = []
            for name, expected_sha256 in sorted(NESTED_COMPONENT_LICENSES.items()):
                bytes_value = archive.read(name)
                observed_sha256 = "sha256:" + hashlib.sha256(bytes_value).hexdigest()
                if (
                    observed_sha256 != expected_sha256
                    or "Gaussian-Splatting License"
                    not in bytes_value.decode("utf-8", errors="ignore")
                    or "non-commercially"
                    not in bytes_value.decode("utf-8", errors="ignore")
                ):
                    raise AuraResidualBackendAdmissionError(
                        ["aura_residual_nested_component_license_invalid"]
                    )
                nested_license_records.append(
                    {
                        "path": name,
                        "sha256": observed_sha256,
                        "license": "Gaussian-Splatting-noncommercial-research-evaluation",
                    }
                )
            for member in members:
                row = expected[member.filename]
                if member.file_size != row["size_bytes"]:
                    raise AuraResidualBackendAdmissionError(
                        ["aura_residual_source_archive_manifest_mismatch"]
                    )
                with archive.open(member) as stream:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if "sha256:" + digest.hexdigest() != row["sha256"]:
                    raise AuraResidualBackendAdmissionError(
                        ["aura_residual_source_archive_manifest_mismatch"]
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise AuraResidualBackendAdmissionError(["aura_residual_source_archive_invalid"]) from exc
    if license_sha256 != AURA_LICENSE_SHA256:
        raise AuraResidualBackendAdmissionError(["aura_residual_source_license_digest_mismatch"])
    return {
        **_record(path),
        "top_level_license": "Apache-2.0",
        "top_level_license_sha256": license_sha256,
        "nested_component_licenses": nested_license_records,
        "noncommercial_research_evaluation_attestation_required": True,
    }


def _validate_noncommercial_attestation(
    *,
    path: Path,
    execution_authority: Mapping[str, Any],
    execution_authority_path: Path,
    source_archive: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    source_identity_path: Path,
) -> dict[str, Any]:
    attestation = _read_object(path, code="aura_residual_noncommercial_attestation_unreadable")
    expected_nested = source_archive["nested_component_licenses"]
    if (
        attestation.get("schema_version") != NONCOMMERCIAL_ATTESTATION_SCHEMA
        or attestation.get("program_id") != "arm-decision-proof-v1"
        or attestation.get("publisher_scene_id") != "840920"
        or attestation.get("reviewer_role") != "authorized_rights_holder"
        or attestation.get("source_repository") != AURA_REPOSITORY
        or attestation.get("source_revision") != AURA_COMMIT
        or attestation.get("source_tree") != AURA_TREE
        or attestation.get("source_archive_sha256") != source_archive["sha256"]
        or attestation.get("source_identity_spec_sha256") != _sha256(source_identity_path)
        or attestation.get("source_identity_spec_source_file_count")
        != len(source_identity["source_files"])
        or attestation.get("nested_component_licenses") != expected_nested
        # The attestation records the user's bounded internal-use direction;
        # it cannot be reused with a different scene/spend/upload authority.
        or attestation.get("authorization_kind")
        != "explicit_user_direction_in_current_goal"
        or attestation.get("authorized_by") != execution_authority["authorized_by"]
        or attestation.get("execution_authority_sha256")
        != _sha256(execution_authority_path)
        or attestation.get("execution_authority_digest")
        != execution_authority["authority_digest"]
        or attestation.get("internal_noncommercial_use_only") is not True
        or attestation.get("private_derived_upload_authorized") is not True
        or attestation.get("raw_dataset_bytes_upload_authorized") is not False
        or attestation.get("provider_training_authorized") is not False
        or attestation.get("noncommercial_research_evaluation_use_authorized") is not True
        or attestation.get("commercial_use_authorized") is not False
        or attestation.get("redistribution_authorized") is not False
        or attestation.get("publication_authorized") is not False
        or attestation.get("attestation_digest")
        != canonical_digest(attestation, digest_field="attestation_digest")
    ):
        raise AuraResidualBackendAdmissionError(
            ["aura_residual_noncommercial_attestation_invalid"]
        )
    return attestation


def materialize_aura_residual_noncommercial_attestation(
    *,
    execution_authority_path: str | Path,
    source_archive_path: str | Path,
    source_identity_spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal the user's bounded internal-use direction against inspected Aura bytes.

    This is deliberately not a generic license waiver.  It only transcribes an
    existing, digest-valid third-scene authority that already rules out raw
    upload, training, commercial use, and publication; it then binds that
    authority to the exact Aura archive and nested component licenses.
    """

    policy = {
        "raw_dataset_bytes_upload": False,
        "private_derived_upload": True,
        "maximum_retention_days": 7,
        "provider_training": False,
        "publication": False,
    }
    authority_path = _file(
        execution_authority_path, code="aura_residual_execution_authority_missing"
    )
    authority = _validate_execution_authority(authority_path, policy)
    terms = authority.get("terms")
    if (
        not isinstance(terms, Mapping)
        or terms.get("interiorgs_commercial_use_authorized") is not False
        or terms.get("interiorgs_redistribution_authorized") is not False
    ):
        raise AuraResidualBackendAdmissionError(
            ["aura_residual_execution_authority_internal_use_invalid"]
        )
    source_path = _file(source_archive_path, code="aura_residual_source_archive_missing")
    source_identity_path = _file(
        source_identity_spec_path, code="aura_residual_source_identity_spec_missing"
    )
    source_identity, expected_source_members = _validate_source_identity_spec(
        source_identity_path
    )
    source = _validate_source_archive(source_path, expected=expected_source_members)
    attestation: dict[str, Any] = {
        "schema_version": NONCOMMERCIAL_ATTESTATION_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": "840920",
        "reviewer_role": "authorized_rights_holder",
        "authorization_kind": authority["authority_kind"],
        "authorized_by": authority["authorized_by"],
        "execution_authority_sha256": _sha256(authority_path),
        "execution_authority_digest": authority["authority_digest"],
        "source_repository": AURA_REPOSITORY,
        "source_revision": AURA_COMMIT,
        "source_tree": AURA_TREE,
        "source_archive_sha256": source["sha256"],
        "source_identity_spec_sha256": _sha256(source_identity_path),
        "source_identity_spec_source_file_count": len(source_identity["source_files"]),
        "nested_component_licenses": source["nested_component_licenses"],
        "internal_noncommercial_use_only": True,
        "private_derived_upload_authorized": True,
        "raw_dataset_bytes_upload_authorized": False,
        "provider_training_authorized": False,
        "noncommercial_research_evaluation_use_authorized": True,
        "commercial_use_authorized": False,
        "redistribution_authorized": False,
        "publication_authorized": False,
        "claim_boundary": (
            "records_authorized_bounded_internal_noncommercial_use_only;"
            "does_not_modify_third_party_license_terms"
        ),
        "attestation_digest": "",
    }
    attestation["attestation_digest"] = canonical_digest(
        attestation, digest_field="attestation_digest"
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(attestation) + "\n", encoding="utf-8")
    return attestation


def _validate_environment_lock(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip() or "torch==" not in text:
        raise AuraResidualBackendAdmissionError(["aura_residual_environment_lock_invalid"])
    return _record(path)


def build_aura_residual_backend_admission_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise AuraResidualBackendAdmissionError(["aura_residual_request_not_json"]) from exc
    if not isinstance(request, dict):
        raise AuraResidualBackendAdmissionError(["aura_residual_request_not_json"])
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("aura_residual_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1" or request.get("adp_item") != "ADP-009D":
        errors.append("aura_residual_request_program_invalid")
    if request.get("frozen_before_inpainting_execution") is not True:
        errors.append("aura_residual_request_not_frozen")
    if request.get("learned_policy_outcomes_accessed") is not False:
        errors.append("aura_residual_request_policy_outcome_leakage")
    if request.get("strict_exact_residual_masks_required") is not True:
        errors.append("aura_residual_exact_mask_requirement_missing")
    if request.get("outside_mask_pixel_delta_required") != 0:
        errors.append("aura_residual_outside_mask_delta_requirement_invalid")
    if request.get("multi_view_consistency_required") is not True:
        errors.append("aura_residual_multiview_requirement_missing")
    if any(key in request for key in ("status", "rights_admitted", "execution_result")):
        errors.append("aura_residual_request_caller_outcome_forbidden")
    for key in (
        "execution_authority_path",
        "prerequisite_receipt_path",
        "source_archive_path",
        "source_identity_spec_path",
        "noncommercial_attestation_path",
        "environment_lock_path",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"aura_residual_request_{key}_missing")
    try:
        _validated_policy(request.get("private_derived_upload_policy"), code="aura_residual_request_private_upload_policy_invalid")
    except AuraResidualBackendAdmissionError as exc:
        errors.extend(exc.codes)
    if errors:
        raise AuraResidualBackendAdmissionError(errors)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != request["request_digest"]:
        raise AuraResidualBackendAdmissionError(["aura_residual_request_digest_mismatch"])
    return request


def materialize_aura_residual_backend_admission_request(
    *, value: Mapping[str, Any], output_path: str | Path
) -> dict[str, Any]:
    """Validate and seal a caller's intended backend admission request."""

    request = build_aura_residual_backend_admission_request(value)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(request) + "\n", encoding="utf-8")
    return request


def materialize_aura_residual_backend_admission(
    *, request_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Create a digest-bound private-derived Aura backend admission receipt."""

    request_file = _file(request_path, code="aura_residual_request_missing")
    request = build_aura_residual_backend_admission_request(
        _read_object(request_file, code="aura_residual_request_unreadable")
    )
    policy = _validated_policy(
        request["private_derived_upload_policy"], code="aura_residual_request_private_upload_policy_invalid"
    )
    authority_path = _file(
        request["execution_authority_path"], code="aura_residual_execution_authority_missing"
    )
    prerequisite_path = _file(
        request["prerequisite_receipt_path"], code="aura_residual_prerequisite_missing"
    )
    source_path = _file(request["source_archive_path"], code="aura_residual_source_archive_missing")
    source_identity_path = _file(
        request["source_identity_spec_path"], code="aura_residual_source_identity_spec_missing"
    )
    attestation_path = _file(
        request["noncommercial_attestation_path"],
        code="aura_residual_noncommercial_attestation_missing",
    )
    lock_path = _file(request["environment_lock_path"], code="aura_residual_environment_lock_missing")
    authority = _validate_execution_authority(authority_path, policy)
    prerequisite = _validate_prerequisite(prerequisite_path)
    source_identity, expected_source_members = _validate_source_identity_spec(source_identity_path)
    source = _validate_source_archive(source_path, expected=expected_source_members)
    noncommercial_attestation = _validate_noncommercial_attestation(
        path=attestation_path,
        execution_authority=authority,
        execution_authority_path=authority_path,
        source_archive=source,
        source_identity=source_identity,
        source_identity_path=source_identity_path,
    )
    environment = _validate_environment_lock(lock_path)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "rights_admitted_for_private_derived_inpainting",
        "backend_id": "aurafusion360_exact_residual_multiview",
        "source_repository": AURA_REPOSITORY,
        "source_revision": AURA_COMMIT,
        "source_tree": AURA_TREE,
        "source_archive_sha256": source["sha256"],
        "environment_lock_sha256": environment["sha256"],
        "model_identity": "AuraFusion360_SD2_inpainting_exact_checkpoint",
        "private_derived_upload_policy": policy,
        "request": {**_record(request_file), "request_digest": request["request_digest"]},
        "execution_authority": {
            **_record(authority_path),
            "authority_digest": authority["authority_digest"],
        },
        "method_prerequisite": {
            **_record(prerequisite_path),
            "receipt_digest": prerequisite["receipt_digest"],
        },
        "source_archive": source,
        "source_identity_provenance": {
            **_record(source_identity_path),
            "source_repository": source_identity["source_repository"],
            "source_commit": source_identity["source_commit"],
            "source_tree": source_identity["source_tree"],
            "source_member_count": len(expected_source_members),
        },
        "noncommercial_research_evaluation_attestation": {
            **_record(attestation_path),
            "attestation_digest": noncommercial_attestation["attestation_digest"],
            "reviewer_role": noncommercial_attestation["reviewer_role"],
        },
        "environment_lock": environment,
        "strict_exact_residual_masks_required": True,
        "mask_dilation_pixels": 0,
        "outside_mask_pixel_delta_required": 0,
        "multi_view_consistency_required": True,
        "execution_authorized": False,
        "inpainting_result_qualified": False,
        "claim_boundary": {
            "raw_dataset_bytes_upload_authorized": False,
            "provider_training_authorized": False,
            "publication_authorized": False,
            "publisher_ply_edited_in_place": False,
            "inpainting_result_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_aura_residual_backend_abstention(
    *, request_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Seal the smallest rights blocker without pretending the backend is admitted."""

    request_file = _file(request_path, code="aura_residual_request_missing")
    request = build_aura_residual_backend_admission_request(
        _read_object(request_file, code="aura_residual_request_unreadable")
    )
    source_path = _file(request["source_archive_path"], code="aura_residual_source_archive_missing")
    source_identity_path = _file(
        request["source_identity_spec_path"], code="aura_residual_source_identity_spec_missing"
    )
    source_identity, expected_source_members = _validate_source_identity_spec(source_identity_path)
    source = _validate_source_archive(source_path, expected=expected_source_members)
    attestation = Path(request["noncommercial_attestation_path"]).expanduser().resolve()
    try:
        _validate_noncommercial_attestation(
            path=_file(
                attestation,
                code="aura_residual_noncommercial_attestation_missing",
            ),
            execution_authority=_validate_execution_authority(
                _file(
                    request["execution_authority_path"],
                    code="aura_residual_execution_authority_missing",
                ),
                _validated_policy(
                    request["private_derived_upload_policy"],
                    code="aura_residual_request_private_upload_policy_invalid",
                ),
            ),
            execution_authority_path=_file(
                request["execution_authority_path"],
                code="aura_residual_execution_authority_missing",
            ),
            source_archive=source,
            source_identity=source_identity,
            source_identity_path=source_identity_path,
        )
    except AuraResidualBackendAdmissionError as exc:
        if not set(exc.codes).issubset(
            {
                "aura_residual_noncommercial_attestation_missing",
                "aura_residual_noncommercial_attestation_unreadable",
                "aura_residual_noncommercial_attestation_invalid",
            }
        ):
            raise
        blocker = "aura_nested_gaussian_splatting_noncommercial_use_attestation_missing"
    else:
        raise AuraResidualBackendAdmissionError(["aura_residual_abstention_no_longer_applicable"])
    receipt: dict[str, Any] = {
        "schema_version": ABSTENTION_SCHEMA,
        "status": "abstained_rights_admission_missing",
        "backend_id": "aurafusion360_exact_residual_multiview",
        "request": {**_record(request_file), "request_digest": request["request_digest"]},
        "source_archive": source,
        "source_identity_provenance": {
            **_record(source_identity_path),
            "source_repository": source_identity["source_repository"],
            "source_commit": source_identity["source_commit"],
            "source_tree": source_identity["source_tree"],
            "source_member_count": len(expected_source_members),
        },
        "smallest_missing_capability": (
            "authorized_rights_holder_attestation_that_the_pinned_AuraFusion360_"
            "Gaussian-Splatting_nested_components_are_used_only_for_the_bounded_"
            "noncommercial_research_evaluation_rehearsal"
        ),
        "blocked_operation": "private_derived_inpainting_upload_and_execution",
        "inpainting_executed": False,
        "provider_mutations_performed": 0,
        "claim_boundary": {
            "top_level_apache_license_alone_not_sufficient": True,
            "raw_dataset_bytes_upload_authorized": False,
            "private_derived_upload_performed": False,
            "provider_training_authorized": False,
            "publication_authorized": False,
            "inpainting_result_qualified": False,
        },
        "blockers": [blocker],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_aura_residual_packet_rights_abstention(
    *,
    input_packet_path: str | Path,
    backend_abstention_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Prohibit execution from a prior input packet when Aura rights are unresolved.

    Input packets are immutable and may correctly have been prepared before a
    later source-license inspection.  This receipt never alters those inputs;
    it establishes the current execution prohibition by reopening both files.
    """

    packet_path = _file(input_packet_path, code="aura_residual_input_packet_missing")
    packet = _read_object(packet_path, code="aura_residual_input_packet_unreadable")
    if (
        packet.get("schema_version") != "public_scene_residual_inpainting_input_packet.v1"
        or packet.get("status")
        != "exact_mask_contained_inpainting_input_packet_materialized"
        or packet.get("packet_digest")
        != canonical_digest(packet, digest_field="packet_digest")
        or packet.get("replacement_object_count") not in range(1, 6)
        or (packet.get("claim_boundary") or {}).get("released_code_inpainting_executed")
        is not False
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_input_packet_invalid"])
    prior_backend = packet.get("backend_admission")
    if not isinstance(prior_backend, Mapping):
        raise AuraResidualBackendAdmissionError(["aura_residual_input_packet_backend_missing"])
    prior_backend_path = _file(
        str(prior_backend.get("path") or ""),
        code="aura_residual_input_packet_backend_missing",
    )
    if (
        prior_backend.get("size_bytes") != prior_backend_path.stat().st_size
        or prior_backend.get("sha256") != _sha256(prior_backend_path)
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_input_packet_backend_changed"])
    backend = _read_object(
        prior_backend_path, code="aura_residual_input_packet_backend_unreadable"
    )
    if (
        backend.get("schema_version") != RECEIPT_SCHEMA
        or backend.get("receipt_digest")
        != canonical_digest(backend, digest_field="receipt_digest")
        or backend.get("backend_id") != "aurafusion360_exact_residual_multiview"
        or prior_backend.get("receipt_digest") != backend["receipt_digest"]
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_input_packet_backend_invalid"])
    abstention_path = _file(
        backend_abstention_path, code="aura_residual_backend_abstention_missing"
    )
    abstention = _read_object(
        abstention_path, code="aura_residual_backend_abstention_unreadable"
    )
    if (
        abstention.get("schema_version") != ABSTENTION_SCHEMA
        or abstention.get("status") != "abstained_rights_admission_missing"
        or abstention.get("backend_id") != backend["backend_id"]
        or abstention.get("receipt_digest")
        != canonical_digest(abstention, digest_field="receipt_digest")
        or abstention.get("source_archive", {}).get("sha256")
        != backend.get("source_archive_sha256")
        or abstention.get("provider_mutations_performed") != 0
        or not abstention.get("blockers")
    ):
        raise AuraResidualBackendAdmissionError(["aura_residual_backend_abstention_invalid"])
    receipt: dict[str, Any] = {
        "schema_version": PACKET_ABSTENTION_SCHEMA,
        "status": "inpainting_execution_prohibited_rights_admission_missing",
        "input_packet": {**_record(packet_path), "packet_digest": packet["packet_digest"]},
        "prior_backend_admission": {
            **_record(prior_backend_path),
            "receipt_digest": backend["receipt_digest"],
        },
        "rights_abstention": {
            **_record(abstention_path),
            "receipt_digest": abstention["receipt_digest"],
        },
        "operation_prohibited": "private_derived_inpainting_upload_and_execution",
        "inpainting_executed": False,
        "provider_mutations_performed": 0,
        "replacement_object_count": packet["replacement_object_count"],
        "claim_boundary": {
            "input_packet_remains_immutable": True,
            "input_packet_is_not_execution_authority": True,
            "private_derived_upload_performed": False,
            "inpainting_result_qualified": False,
        },
        "blockers": list(abstention["blockers"]),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_aura_residual_backend_admission(
        request_path=args.request, output_path=args.output
    )
    print(canonical_json(receipt))
    return 0


__all__ = [
    "AURA_COMMIT",
    "AURA_REPOSITORY",
    "AURA_TREE",
    "ABSTENTION_SCHEMA",
    "AuraResidualBackendAdmissionError",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "build_aura_residual_backend_admission_request",
    "materialize_aura_residual_noncommercial_attestation",
    "materialize_aura_residual_backend_admission_request",
    "materialize_aura_residual_backend_admission",
    "materialize_aura_residual_backend_abstention",
    "materialize_aura_residual_packet_rights_abstention",
]
