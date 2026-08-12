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
AURA_REPOSITORY = "https://github.com/kkennethwu/AuraFusion360_official"
AURA_COMMIT = "f23b26c44ba84608306ba952510533ebf4c7877d"
AURA_TREE = "cc8447c66448b29bb4d39fec29c031df63d4b179"
AURA_LICENSE_SHA256 = "sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1"
REQUIRED_CHECKPOINTS = frozenset(
    {
        "aurafusion360_sam2_hiera_large",
        "aurafusion360_marigold_depth_v1_0",
        "aurafusion360_marigold_agdd_v1_0",
        "aurafusion360_sd2_inpainting_exact_checkpoint",
    }
)
REQUIRED_AURA_ARCHIVE_MEMBERS = frozenset({"LICENSE", "inpaint.py", "arguments/__init__.py"})


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
    if not REQUIRED_AURA_ARCHIVE_MEMBERS.issubset(expected):
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
    return {**_record(path), "license": "Apache-2.0", "license_sha256": license_sha256}


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
    lock_path = _file(request["environment_lock_path"], code="aura_residual_environment_lock_missing")
    authority = _validate_execution_authority(authority_path, policy)
    prerequisite = _validate_prerequisite(prerequisite_path)
    source_identity, expected_source_members = _validate_source_identity_spec(source_identity_path)
    source = _validate_source_archive(source_path, expected=expected_source_members)
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
    "AuraResidualBackendAdmissionError",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "build_aura_residual_backend_admission_request",
    "materialize_aura_residual_backend_admission",
]
