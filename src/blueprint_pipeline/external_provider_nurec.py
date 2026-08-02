"""Strict admission and CPU qualification for provider-authored NuRec USDZ assets.

This is the versioned, no-raw-capture lane for public provider samples and
user-managed provider exports.  It deliberately does not reuse Blueprint's
capture-bound packaging contract: the exact provider package is retained, and
CPU observations remain candidates for a later live Isaac verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest


ACQUISITION_RECEIPT_SCHEMA = "external_provider_acquisition_receipt.v1"
IMPORT_REQUEST_SCHEMA = "external_reconstruction_import_request.v2"
RIGHTS_RECEIPT_SCHEMA = "external_provider_provenance_rights_receipt.v2"
IMPORT_RECEIPT_SCHEMA = "external_reconstruction_import_receipt.v2"
QUALIFICATION_SCHEMA = "provider_nurec_usdz_qualification.v1"
ISAAC_REQUEST_SCHEMA = "provider_nurec_isaac_verification_request.v1"
ISAAC_RUNTIME_SCHEMA = "provider_nurec_isaac_runtime_result.v1"
ISAAC_VERIFICATION_RESULT_SCHEMA = "provider_nurec_isaac_verification_result.v1"

SOURCE_PROFILES = {"public_provider_sample", "user_managed_provider_export"}
SUPPORTED_PROVIDERS = {"scaniverse"}
SUPPORTED_SUFFIXES = {".usdz", ".usd", ".usda", ".usdc", ".ply", ".spz", ".glb"}
MAX_ASSET_BYTES = 2_000_000_000
MAX_USDZ_MEMBER_BYTES = 2_000_000_000
MAX_USDZ_TOTAL_BYTES = 4_000_000_000
MAX_USDZ_MEMBERS = 20_000

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class ExternalProviderNuRecError(ValueError):
    """Deterministic failure carrying stable blocker codes."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalProviderNuRecError(["artifact_not_json_serializable"]) from exc
    if not isinstance(result, dict):
        raise ExternalProviderNuRecError(["artifact_not_object"])
    return result


def _finalize(value: Mapping[str, Any], *, schema: str, digest_field: str) -> dict[str, Any]:
    artifact = _clone(value)
    supplied = artifact.pop(digest_field, None)
    artifact["schema_version"] = schema
    expected = canonical_digest(artifact, digest_field=digest_field)
    if supplied is not None and supplied != expected:
        raise ExternalProviderNuRecError([f"{digest_field}_mismatch"])
    artifact[digest_field] = expected
    return artifact


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_relative(value: Any, code: str) -> PurePosixPath:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ExternalProviderNuRecError([code])
    return path


def build_acquisition_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    for key in (
        "source_page_url",
        "final_download_url",
        "original_filename",
        "acquisition_timestamp",
        "supplier_identity",
        "declared_sample_scene",
        "rights_terms_source",
        "rights_review_status",
        "tool_version",
    ):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            errors.append(f"acquisition_{key}_missing")
    if receipt.get("source_class") != "public_provider_sample":
        errors.append("acquisition_source_class_invalid")
    if not _is_digest(receipt.get("asset_digest")):
        errors.append("acquisition_asset_digest_invalid")
    if not isinstance(receipt.get("byte_size"), int) or receipt.get("byte_size", 0) < 1:
        errors.append("acquisition_byte_size_invalid")
    if _COMMIT.fullmatch(str(receipt.get("source_commit_sha") or "")) is None:
        errors.append("acquisition_source_commit_invalid")
    if not isinstance(receipt.get("http_metadata"), Mapping):
        errors.append("acquisition_http_metadata_invalid")
    expected = {
        "confidential": False,
        "blueprint_raw_capture": False,
        "provider_reconstruction": True,
        "capture_hardware": "unknown",
        "capture_modality": "unknown",
        "remote_provider_login_performed": False,
        "proof_effect": "immutable_external_asset_acquisition_only",
        "claim_ceiling": "file_identity_and_declared_source_only",
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            errors.append(f"acquisition_boundary_invalid:{key}")
    if errors:
        raise ExternalProviderNuRecError(errors)
    return _finalize(
        receipt,
        schema=ACQUISITION_RECEIPT_SCHEMA,
        digest_field="acquisition_receipt_digest",
    )


def _validate_rights_scope(
    rights: Mapping[str, Any], source_profile: str, errors: list[str]
) -> None:
    for key in (
        "terms_version",
        "rights_terms_source",
        "ownership_or_license_status",
        "commercial_use_status",
        "consent_privacy_status",
        "retention_status",
        "deletion_status",
        "model_training_status",
        "benchmarking_status",
    ):
        if not isinstance(rights.get(key), str) or not rights[key]:
            errors.append(f"external_source_rights_{key}_missing")
    uses = rights.get("allowed_uses")
    if (
        not isinstance(uses, list)
        or not uses
        or any(not isinstance(item, str) or not item for item in uses)
    ):
        errors.append("external_source_rights_allowed_uses_invalid")
    if "local_engineering_inspection" not in (uses or []):
        errors.append("external_source_local_inspection_not_authorized")
    confidential = rights.get("confidential")
    public_reporting = rights.get("public_reporting_allowed")
    if source_profile == "public_provider_sample":
        if confidential is not False or public_reporting is not True:
            errors.append("public_provider_sample_confidentiality_invalid")
    elif confidential is not True or public_reporting is not False:
        errors.append("private_provider_export_confidentiality_invalid")
    if rights.get("remote_upload_authorized") is not False:
        errors.append("external_source_remote_upload_forbidden")


def build_external_source_import_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build v2 external-source admission without inventing raw-capture lineage."""

    request = _clone(value)
    errors: list[str] = []
    if request.get("schema_version") not in {None, IMPORT_REQUEST_SCHEMA}:
        errors.append("external_source_request_schema_invalid")
    source_profile = str(request.get("source_profile") or "")
    if source_profile not in SOURCE_PROFILES:
        errors.append("external_source_profile_unsupported")
    if not _IDENTITY.fullmatch(str(request.get("stable_run_identity") or "")):
        errors.append("external_source_stable_run_identity_invalid")
    if _COMMIT.fullmatch(str(request.get("source_commit_sha") or "")) is None:
        errors.append("external_source_commit_invalid")
    if not _is_digest(request.get("acquisition_or_export_receipt_digest")):
        errors.append("external_source_acquisition_receipt_digest_invalid")
    if "source_capture_identity" in request or "source_capture_digest" in request:
        errors.append("external_source_fabricated_raw_capture_identity_forbidden")

    identity = request.get("external_source_identity")
    if not isinstance(identity, Mapping):
        errors.append("external_source_identity_missing")
        identity = {}
    provider = str(identity.get("provider") or "")
    if provider not in SUPPORTED_PROVIDERS:
        errors.append("external_source_provider_unsupported")
    if identity.get("source_relationship_to_blueprint_raw_capture") != "none":
        errors.append("external_source_raw_capture_relationship_must_be_none")
    if "source_capture_identity" in identity or "source_capture_digest" in identity:
        errors.append("external_source_fabricated_raw_capture_identity_forbidden")
    if not _is_digest(identity.get("local_asset_digest")):
        errors.append("external_source_asset_digest_invalid")
    for key in ("acquisition_or_export_time", "terms_version"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            errors.append(f"external_source_identity_{key}_missing")
    if not _IDENTITY.fullmatch(str(identity.get("operator_reference") or "")):
        errors.append("external_source_operator_reference_invalid")
    identifiers = identity.get("provider_asset_identifiers")
    if not isinstance(identifiers, Mapping):
        errors.append("external_source_provider_identifiers_invalid")
    modality = identity.get("capture_modality")
    if not isinstance(modality, Mapping) or modality.get("status") not in {"unknown", "verified"}:
        errors.append("external_source_capture_modality_invalid")
    elif modality.get("status") == "unknown" and modality.get("value") not in {None, "unknown"}:
        errors.append("external_source_unverified_capture_modality_forbidden")

    binding = request.get("asset_binding")
    if not isinstance(binding, Mapping):
        errors.append("external_source_asset_binding_missing")
        binding = {}
    try:
        relative = _safe_relative(binding.get("relative_path"), "external_source_asset_path_unsafe")
        if relative.suffix.lower() not in SUPPORTED_SUFFIXES:
            errors.append("external_source_asset_format_unsupported")
    except ExternalProviderNuRecError as exc:
        errors.extend(exc.codes)
    if not _is_digest(binding.get("digest")):
        errors.append("external_source_asset_digest_invalid")
    if binding.get("digest") != identity.get("local_asset_digest"):
        errors.append("external_source_asset_identity_digest_mismatch")

    rights = request.get("rights_scope")
    if not isinstance(rights, Mapping):
        errors.append("external_source_rights_scope_missing")
        rights = {}
    _validate_rights_scope(rights, source_profile, errors)
    if rights.get("terms_version") != identity.get("terms_version"):
        errors.append("external_source_rights_terms_mismatch")
    expected = {
        "remote_calls_authorized": False,
        "remote_calls_performed": False,
        "external_derived_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "proof_effect": "external_import_request_only",
        "claim_ceiling": "none",
    }
    for key, expected_value in expected.items():
        if request.get(key) != expected_value:
            errors.append(f"external_source_request_boundary_invalid:{key}")
    if errors:
        raise ExternalProviderNuRecError(errors)
    return _finalize(
        request,
        schema=IMPORT_REQUEST_SCHEMA,
        digest_field="external_import_request_digest",
    )


def build_provider_rights_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    source_profile = str(receipt.get("source_profile") or "")
    if source_profile not in SOURCE_PROFILES:
        errors.append("provider_rights_source_profile_invalid")
    for key in ("external_import_request_digest", "asset_digest"):
        if not _is_digest(receipt.get(key)):
            errors.append(f"provider_rights_{key}_invalid")
    rights = receipt.get("rights_scope")
    if not isinstance(rights, Mapping):
        errors.append("provider_rights_scope_missing")
        rights = {}
    _validate_rights_scope(rights, source_profile, errors)
    expected = {
        "status": "accepted_for_declared_local_import_only",
        "remote_upload_authorized_by_receipt": False,
        "provider_success_is_blueprint_qualification": False,
        "external_derived_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "proof_effect": "provenance_and_rights_for_local_import_only",
        "claim_ceiling": "external_reconstruction_import",
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            errors.append(f"provider_rights_boundary_invalid:{key}")
    if errors:
        raise ExternalProviderNuRecError(errors)
    return _finalize(
        receipt,
        schema=RIGHTS_RECEIPT_SCHEMA,
        digest_field="provenance_rights_receipt_digest",
    )


def build_external_source_import_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    for key in (
        "external_import_request_digest",
        "provenance_rights_receipt_digest",
        "asset_digest",
    ):
        if not _is_digest(receipt.get(key)):
            errors.append(f"external_source_receipt_{key}_invalid")
    try:
        _safe_relative(receipt.get("asset_reference"), "external_source_receipt_asset_path_unsafe")
    except ExternalProviderNuRecError as exc:
        errors.extend(exc.codes)
    if (
        not isinstance(receipt.get("asset_size_bytes"), int)
        or receipt.get("asset_size_bytes", 0) < 1
    ):
        errors.append("external_source_receipt_asset_size_invalid")
    expected = {
        "status": "imported_derived_support_only",
        "source_capture_binding": "none",
        "external_derived_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "metric_scale_proven": False,
        "collision_geometry_validated": False,
        "isaac_compatibility_proven": False,
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "remote_calls_performed": False,
        "proof_effect": "external_reconstruction_derived_support_only",
        "claim_ceiling": "external_reconstruction_import",
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            errors.append(f"external_source_receipt_boundary_invalid:{key}")
    if errors:
        raise ExternalProviderNuRecError(errors)
    return _finalize(
        receipt,
        schema=IMPORT_RECEIPT_SCHEMA,
        digest_field="external_import_receipt_digest",
    )


def _bound_source(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = _safe_relative(binding.get("relative_path"), "external_source_asset_path_unsafe")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ExternalProviderNuRecError(["external_source_asset_symlink_forbidden"])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExternalProviderNuRecError(["external_source_asset_missing"]) from exc
    if root != resolved and root not in resolved.parents:
        raise ExternalProviderNuRecError(["external_source_asset_path_escape"])
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise ExternalProviderNuRecError(["external_source_asset_invalid"])
    if resolved.stat().st_size > MAX_ASSET_BYTES:
        raise ExternalProviderNuRecError(["external_source_asset_oversized"])
    if sha256_file(resolved) != binding.get("digest"):
        raise ExternalProviderNuRecError(["external_source_asset_digest_mismatch"])
    return resolved


def import_external_source(
    *,
    source_artifact: Mapping[str, Any],
    artifact_root: str | Path,
    output_root: str | Path,
    admit_in_place: bool = False,
) -> dict[str, Any]:
    """Admit exact local bytes and emit v2 rights/import receipts without raw lineage."""

    started = time.monotonic()
    request = build_external_source_import_request(source_artifact)
    root = Path(artifact_root)
    if root.is_symlink() or not root.is_dir():
        raise ExternalProviderNuRecError(["external_source_artifact_root_invalid"])
    root = root.resolve()
    source = _bound_source(root, request["asset_binding"])
    output = Path(output_root)
    if output.is_symlink():
        raise ExternalProviderNuRecError(["external_source_output_root_symlink_forbidden"])
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    content_id = request["external_import_request_digest"][7:]
    final = output / content_id
    receipt_path = final / "external_reconstruction_import_receipt.v2.json"
    if final.exists():
        try:
            replay = build_external_source_import_receipt(json.loads(receipt_path.read_text()))
            replay_rights = build_provider_rights_receipt(
                json.loads(
                    (final / "external_provider_provenance_rights_receipt.v2.json").read_text()
                )
            )
        except (OSError, json.JSONDecodeError, ExternalProviderNuRecError) as exc:
            raise ExternalProviderNuRecError(["external_source_import_replay_tampered"]) from exc
        if (
            replay["external_import_request_digest"] != request["external_import_request_digest"]
            or replay["provenance_rights_receipt_digest"]
            != replay_rights["provenance_rights_receipt_digest"]
            or replay_rights["external_import_request_digest"]
            != request["external_import_request_digest"]
            or replay_rights["asset_digest"] != request["asset_binding"]["digest"]
        ):
            raise ExternalProviderNuRecError(["external_source_import_replay_binding_mismatch"])
        replay_asset = Path(replay["asset_absolute_path"])
        if (
            replay_asset.is_symlink()
            or not replay_asset.is_file()
            or sha256_file(replay_asset) != replay["asset_digest"]
        ):
            raise ExternalProviderNuRecError(["external_source_import_replay_asset_tampered"])
        return replay

    temporary = Path(tempfile.mkdtemp(prefix=".external-source-v2-", dir=output))
    try:
        if admit_in_place:
            admitted = source
            materialization = "admitted_in_place_content_addressed"
        else:
            assets = temporary / "assets"
            assets.mkdir()
            admitted = assets / f"asset{source.suffix.lower()}"
            shutil.copy2(source, admitted)
            materialization = "copied_into_content_addressed_intake"
        if sha256_file(admitted) != request["asset_binding"]["digest"]:
            raise ExternalProviderNuRecError(["external_source_materialization_digest_mismatch"])
        rights = build_provider_rights_receipt(
            {
                "source_profile": request["source_profile"],
                "provider": request["external_source_identity"]["provider"],
                "external_import_request_digest": request["external_import_request_digest"],
                "asset_digest": request["asset_binding"]["digest"],
                "rights_scope": request["rights_scope"],
                "status": "accepted_for_declared_local_import_only",
                "remote_upload_authorized_by_receipt": False,
                "provider_success_is_blueprint_qualification": False,
                "external_derived_support_asset": True,
                "blueprint_raw_capture_truth": False,
                "proof_effect": "provenance_and_rights_for_local_import_only",
                "claim_ceiling": "external_reconstruction_import",
            }
        )
        (temporary / "external_provider_provenance_rights_receipt.v2.json").write_text(
            json.dumps(rights, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        asset_reference = (
            str(source.relative_to(root)).replace(os.sep, "/")
            if admit_in_place
            else f"{content_id}/assets/{admitted.name}"
        )
        final_asset_path = source.resolve() if admit_in_place else final / "assets" / admitted.name
        receipt = build_external_source_import_receipt(
            {
                "stable_run_identity": request["stable_run_identity"],
                "source_profile": request["source_profile"],
                "provider": request["external_source_identity"]["provider"],
                "external_import_request_digest": request["external_import_request_digest"],
                "provenance_rights_receipt_digest": rights["provenance_rights_receipt_digest"],
                "asset_digest": request["asset_binding"]["digest"],
                "asset_size_bytes": admitted.stat().st_size,
                "asset_reference": asset_reference,
                "asset_absolute_path": str(final_asset_path),
                "materialization": materialization,
                "duration_seconds": round(time.monotonic() - started, 6),
                "cost_usd": 0.0,
                "status": "imported_derived_support_only",
                "source_capture_binding": "none",
                "external_derived_support_asset": True,
                "blueprint_raw_capture_truth": False,
                "metric_scale_proven": False,
                "collision_geometry_validated": False,
                "isaac_compatibility_proven": False,
                "simulator_task_success_proven": False,
                "physical_success_proven": False,
                "remote_calls_performed": False,
                "proof_effect": "external_reconstruction_derived_support_only",
                "claim_ceiling": "external_reconstruction_import",
            }
        )
        (temporary / "external_reconstruction_import_request.v2.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "external_reconstruction_import_receipt.v2.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _zip_member_is_regular(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    # Some standards-compliant USDZ writers preserve permissions but omit the
    # Unix file-type bits.  Treat that representation as a regular file; still
    # reject explicit directory, symlink, device, FIFO, and socket types.
    return stat.S_IFMT(mode) in {0, stat.S_IFREG}


def _archive_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExternalProviderNuRecError(["provider_nurec_usdz_signature_invalid"]) from exc
    with archive:
        members = archive.infolist()
        errors: list[str] = []
        names = [member.filename for member in members]
        if not members:
            errors.append("provider_nurec_usdz_empty")
        if len(members) > MAX_USDZ_MEMBERS:
            errors.append("provider_nurec_usdz_member_count_exceeded")
        if len(names) != len(set(names)):
            errors.append("provider_nurec_usdz_duplicate_member")
        if sum(member.file_size for member in members) > MAX_USDZ_TOTAL_BYTES:
            errors.append("provider_nurec_usdz_total_size_exceeded")
        inventory: list[dict[str, Any]] = []
        for member in members:
            try:
                relative = _safe_relative(member.filename, "provider_nurec_usdz_member_path_unsafe")
            except ExternalProviderNuRecError as exc:
                errors.extend(exc.codes)
                continue
            if member.flag_bits & 0x1:
                errors.append("provider_nurec_usdz_encrypted_member")
            if member.is_dir() or not _zip_member_is_regular(member):
                errors.append("provider_nurec_usdz_nonregular_member")
            if member.file_size > MAX_USDZ_MEMBER_BYTES:
                errors.append("provider_nurec_usdz_member_size_exceeded")
            digest = hashlib.sha256()
            try:
                with archive.open(member, "r") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            except (OSError, zipfile.BadZipFile) as exc:
                raise ExternalProviderNuRecError(["provider_nurec_usdz_crc_invalid"]) from exc
            inventory.append(
                {
                    "member": str(relative),
                    "size_bytes": member.file_size,
                    "compressed_size_bytes": member.compress_size,
                    "compression_method": member.compress_type,
                    "crc32": f"{member.CRC:08x}",
                    "digest": f"sha256:{digest.hexdigest()}",
                    "encrypted": False,
                    "regular_file": True,
                }
            )
        if errors:
            raise ExternalProviderNuRecError(errors)
        return inventory, names[0]


def _vec3(value: Any) -> list[float]:
    return [round(float(value[index]), 9) for index in range(3)]


def _inspect_stage(path: Path) -> dict[str, Any]:
    try:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    except ImportError as exc:
        raise ExternalProviderNuRecError(["provider_nurec_openusd_runtime_unavailable"]) from exc
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExternalProviderNuRecError(["provider_nurec_stage_open_failed"]) from exc
    if stage is None:
        raise ExternalProviderNuRecError(["provider_nurec_stage_open_failed"])
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ExternalProviderNuRecError(["provider_nurec_dependency_inspection_failed"]) from exc

    visual_prims: list[dict[str, Any]] = []
    collision_prims: list[dict[str, Any]] = []
    mesh_prims: list[dict[str, Any]] = []
    unknown_schemas: set[str] = set()
    ground_candidates: list[dict[str, Any]] = []
    all_bounds: list[list[float]] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        type_name = str(prim.GetTypeName())
        attrs = {str(attr.GetName()): attr.Get() for attr in prim.GetAttributes()}
        nurec_markers = sorted(name for name in attrs if name.startswith("omni:nurec:"))
        if type_name in {"OmniNuRecFieldAsset", "ParticleField3DGaussianSplat"} or nurec_markers:
            visual_prims.append(
                {
                    "prim_path": prim_path,
                    "type_name": type_name,
                    "nurec_attribute_names": nurec_markers,
                }
            )
        if type_name.startswith("Omni"):
            unknown_schemas.add(type_name)
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if has_collision:
            collision_prims.append(
                {
                    "prim_path": prim_path,
                    "type_name": type_name,
                    "physics_collision_api": True,
                    "physics_mesh_collision_api": prim.HasAPI(UsdPhysics.MeshCollisionAPI),
                }
            )
        if type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
            mesh_row: dict[str, Any] = {
                "prim_path": prim_path,
                "point_count": int(len(points)),
                "face_count": int(len(counts)),
                "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
                "physics_collision_api": has_collision,
            }
            if len(points):
                transform = np.asarray(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                    dtype=float,
                )
                homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                world = homogeneous @ transform
                world = world[:, :3] / world[:, 3, None]
                bounds = [world.min(axis=0).tolist(), world.max(axis=0).tolist()]
                mesh_row["world_bounds"] = [_vec3(bounds[0]), _vec3(bounds[1])]
                all_bounds.extend(bounds)
                if (
                    has_collision
                    and len(counts)
                    and np.all(counts == 3)
                    and len(indices) == len(counts) * 3
                ):
                    triangles = indices.reshape((-1, 3))
                    a, b, c = world[triangles[:, 0]], world[triangles[:, 1]], world[triangles[:, 2]]
                    normals = np.cross(b - a, c - a)
                    double_area = np.linalg.norm(normals, axis=1)
                    vertical = np.divide(
                        np.abs(normals[:, 2]),
                        double_area,
                        out=np.zeros_like(double_area),
                        where=double_area > 0,
                    )
                    candidates = np.where((vertical >= 0.98) & (double_area >= 0.02))[0]
                    if len(candidates):
                        selected = int(candidates[np.argmax(double_area[candidates])])
                        centroid = (a[selected] + b[selected] + c[selected]) / 3.0
                        ground_candidates.append(
                            {
                                "collision_prim_path": prim_path,
                                "probe_xy_m": _vec3(centroid)[:2],
                                "candidate_ground_height_m": round(float(centroid[2]), 9),
                                "triangle_area_m2": round(float(double_area[selected] / 2.0), 9),
                                "selection_method": "largest_near_horizontal_collision_triangle",
                                "status": "cpu_geometry_candidate_unverified_in_isaac",
                            }
                        )
            mesh_prims.append(mesh_row)

    def dependency_name(value: Any) -> str:
        text = str(getattr(value, "realPath", "") or getattr(value, "identifier", "") or value)
        if "[" in text and text.endswith("]"):
            return text.rsplit("[", 1)[1][:-1]
        return text

    bounds_value = None
    if all_bounds:
        array = np.asarray(all_bounds, dtype=float)
        bounds_value = [_vec3(array.min(axis=0)), _vec3(array.max(axis=0))]
    return {
        "stage_opened": True,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "dependency_graph": {
            "layers": sorted(set(dependency_name(item) for item in layers)),
            "assets": sorted(set(dependency_name(item) for item in assets)),
            "unresolved": sorted(str(item) for item in unresolved),
        },
        "visual_prims": visual_prims,
        "mesh_prims": mesh_prims,
        "collision_prims": collision_prims,
        "estimated_world_bounds": bounds_value,
        "unknown_or_provider_schemas": sorted(unknown_schemas),
        "ground_probe_candidates": sorted(
            ground_candidates,
            key=lambda row: (-row["triangle_area_m2"], row["collision_prim_path"]),
        ),
    }


def build_provider_nurec_qualification(value: Mapping[str, Any]) -> dict[str, Any]:
    report = _clone(value)
    errors: list[str] = []
    for key in ("package_digest", "external_import_receipt_digest"):
        if not _is_digest(report.get(key)):
            errors.append(f"provider_nurec_qualification_{key}_invalid")
    if not isinstance(report.get("archive_inventory"), list) or not report["archive_inventory"]:
        errors.append("provider_nurec_qualification_archive_inventory_invalid")
    if not isinstance(report.get("dependency_graph"), Mapping):
        errors.append("provider_nurec_qualification_dependency_graph_invalid")
    if report.get("cpu_qualification_verdict") not in {
        "qualified_for_live_isaac_verification",
        "abstention",
    }:
        errors.append("provider_nurec_cpu_verdict_invalid")
    blockers = report.get("blockers")
    if not isinstance(blockers, list):
        errors.append("provider_nurec_blockers_invalid")
    elif bool(blockers) == (
        report.get("cpu_qualification_verdict") == "qualified_for_live_isaac_verification"
    ):
        errors.append("provider_nurec_verdict_blocker_inconsistent")
    expected = {
        "exact_provider_package_preserved": True,
        "provider_authored_package": True,
        "blueprint_authored_package": False,
        "isaac_rendering_proven": False,
        "collision_correctness_proven": False,
        "task_execution_proven": False,
        "physical_success_proven": False,
        "proof_effect": "cpu_package_qualification_only",
        "claim_ceiling": "candidate_for_live_isaac_verification",
    }
    for key, expected_value in expected.items():
        if report.get(key) != expected_value:
            errors.append(f"provider_nurec_qualification_boundary_invalid:{key}")
    if errors:
        raise ExternalProviderNuRecError(errors)
    return _finalize(
        report,
        schema=QUALIFICATION_SCHEMA,
        digest_field="qualification_report_digest",
    )


def qualify_provider_nurec_usdz(
    *,
    package_path: str | Path,
    expected_digest: str,
    external_import_receipt_digest: str,
) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(package_path)
    if path.is_symlink():
        raise ExternalProviderNuRecError(["provider_nurec_package_symlink_forbidden"])
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExternalProviderNuRecError(["provider_nurec_package_missing"]) from exc
    if path.suffix.lower() != ".usdz" or not path.is_file():
        raise ExternalProviderNuRecError(["provider_nurec_package_format_invalid"])
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        raise ExternalProviderNuRecError(["provider_nurec_package_digest_mismatch"])
    inventory, root_member = _archive_inventory(path)
    stage = _inspect_stage(path)
    blockers: list[str] = []
    inventory_names = {row["member"] for row in inventory}
    for required in ("default.usda", "gauss.usda", "sim.nurec", "mesh.usd"):
        if required not in inventory_names:
            blockers.append(f"provider_nurec_required_member_missing:{required}")
    if root_member != "default.usda":
        blockers.append("provider_nurec_root_layer_unexpected")
    if stage["meters_per_unit"] != 1.0:
        blockers.append("provider_nurec_stage_units_not_meters")
    if stage["up_axis"] != "Z":
        blockers.append("provider_nurec_stage_up_axis_not_z")
    if stage["dependency_graph"]["unresolved"]:
        blockers.append("provider_nurec_unresolved_dependencies")
    if not stage["visual_prims"]:
        blockers.append("provider_nurec_visual_layer_not_detected")
    if not stage["collision_prims"]:
        blockers.append("provider_nurec_collision_api_not_detected")
    if not any(row.get("visibility") == "invisible" for row in stage["mesh_prims"]):
        blockers.append("provider_nurec_invisible_mesh_not_detected")
    report = {
        "package_digest": observed_digest,
        "package_size_bytes": path.stat().st_size,
        "package_absolute_path": str(path),
        "external_import_receipt_digest": external_import_receipt_digest,
        "archive_inventory": sorted(inventory, key=lambda row: row["member"]),
        "root_layer": root_member,
        **stage,
        "target_renderer_requirement": "NVIDIA Isaac Sim with RTX and OmniNuRec support",
        "cpu_qualification_verdict": (
            "abstention" if blockers else "qualified_for_live_isaac_verification"
        ),
        "blockers": sorted(blockers),
        "warnings": [
            "provider schema support was observed through OpenUSD composition, not rendered",
            "ground probe coordinates are mesh-derived candidates, not collision proof",
            "provider metric and alignment claims were not independently verified",
        ],
        "duration_seconds": round(time.monotonic() - started, 6),
        "exact_provider_package_preserved": True,
        "provider_authored_package": True,
        "blueprint_authored_package": False,
        "isaac_rendering_proven": False,
        "collision_correctness_proven": False,
        "task_execution_proven": False,
        "physical_success_proven": False,
        "proof_effect": "cpu_package_qualification_only",
        "claim_ceiling": "candidate_for_live_isaac_verification",
    }
    return build_provider_nurec_qualification(report)


def build_provider_nurec_isaac_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(value)
    errors: list[str] = []
    for key in (
        "package_digest",
        "external_import_receipt_digest",
        "qualification_report_digest",
        "fixed_camera_spec_digest",
        "runtime_implementation_digest",
    ):
        if not _is_digest(request.get(key)):
            errors.append(f"provider_isaac_request_{key}_invalid")
    if request.get("render_options_digest") is not None and not _is_digest(
        request.get("render_options_digest")
    ):
        errors.append("provider_isaac_request_render_options_digest_invalid")
    if _COMMIT.fullmatch(str(request.get("source_commit_sha") or "")) is None:
        errors.append("provider_isaac_request_source_commit_invalid")
    if _IMAGE.fullmatch(str(request.get("runtime_container_image_digest") or "")) is None:
        errors.append("provider_isaac_request_runtime_image_invalid")
    try:
        reference = _safe_relative(
            request.get("package_artifact_reference"), "provider_isaac_package_reference_unsafe"
        )
        if reference.suffix.lower() != ".usdz":
            errors.append("provider_isaac_package_format_invalid")
    except ExternalProviderNuRecError as exc:
        errors.extend(exc.codes)
    paths = request.get("expected_prim_paths")
    if (
        not isinstance(paths, Mapping)
        or not str(paths.get("appearance") or "").startswith("/")
        or not str(paths.get("collision") or "").startswith("/")
    ):
        errors.append("provider_isaac_expected_prim_paths_invalid")
    camera_ids = request.get("fixed_camera_ids")
    if (
        not isinstance(camera_ids, list)
        or not camera_ids
        or len(camera_ids) != len(set(camera_ids))
    ):
        errors.append("provider_isaac_camera_ids_invalid")
    probe = request.get("physics_probe_request")
    if not isinstance(probe, Mapping):
        errors.append("provider_isaac_physics_probe_missing")
    else:
        if probe.get("ground_collider_prim") != (paths or {}).get("collision"):
            errors.append("provider_isaac_probe_collision_prim_mismatch")
        if not isinstance(probe.get("ground_height_m"), (int, float)) or isinstance(
            probe.get("ground_height_m"), bool
        ):
            errors.append("provider_isaac_ground_height_invalid")
        xy = probe.get("probe_xy_m")
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in (xy or [])
            )
        ):
            errors.append("provider_isaac_probe_xy_invalid")
        if probe.get("selection_status") != "cpu_geometry_candidate_unverified_in_isaac":
            errors.append("provider_isaac_probe_status_invalid")
        if (
            probe.get("manufacture_ground_plane") is not False
            or probe.get("require_contact_event") is not True
        ):
            errors.append("provider_isaac_probe_boundary_invalid")
        if not isinstance(probe.get("steps"), int) or probe.get("steps", 0) < 2:
            errors.append("provider_isaac_probe_steps_invalid")
    timeout = request.get("timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 60 <= timeout <= 14_400:
        errors.append("provider_isaac_timeout_invalid")
    spend = request.get("spend_controls")
    if not isinstance(spend, Mapping) or spend.get("authorized") is not False:
        errors.append("provider_isaac_paid_authority_must_be_false")
    elif (
        not isinstance(spend.get("estimated_max_spend_usd"), (int, float))
        or spend.get("estimated_max_spend_usd", 0) <= 0
        or spend.get("hard_ttl_seconds") != timeout
        or spend.get("teardown_required") is not True
        or spend.get("provider_zero_required_before_and_after") is not True
    ):
        errors.append("provider_isaac_spend_controls_invalid")
    expected = {
        "provider_authored_package": True,
        "exact_package_required": True,
        "headless": True,
        "display_attached": False,
        "execution_status": "awaiting_explicit_paid_runtime_authorization",
        "provider_allocation_performed": False,
        "expected_runtime_schema": ISAAC_RUNTIME_SCHEMA,
        "proof_effect": "none",
        "claim_ceiling": "request_only",
    }
    for key, expected_value in expected.items():
        if request.get(key) != expected_value:
            errors.append(f"provider_isaac_request_boundary_invalid:{key}")
    if errors:
        raise ExternalProviderNuRecError(errors)
    return _finalize(
        request,
        schema=ISAAC_REQUEST_SCHEMA,
        digest_field="isaac_verification_request_digest",
    )


def build_provider_nurec_isaac_request_from_checkout(
    value: Mapping[str, Any], *, source_checkout: str | Path
) -> dict[str, Any]:
    """Bind request construction to an existing clean checkout at its real HEAD.

    The portable contract validator intentionally cannot prove that a syntactically
    valid commit exists. Bundle preparation is local, however, so it must close
    that gap before copying a multi-gigabyte package or presenting a request to the
    paid-resource allocator.
    """

    checkout = Path(source_checkout)
    if checkout.is_symlink():
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_symlink_forbidden"])
    try:
        checkout = checkout.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_missing"]) from exc
    if not checkout.is_dir():
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_invalid"])

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(checkout), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExternalProviderNuRecError(
                ["provider_isaac_source_checkout_git_unavailable"]
            ) from exc
        return completed.stdout.strip()

    try:
        top_level = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_invalid"]) from exc
    if top_level != checkout:
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_root_mismatch"])

    observed_commit = git("rev-parse", "HEAD")
    if _COMMIT.fullmatch(observed_commit) is None:
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_head_invalid"])
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_not_clean"])

    request = _clone(value)
    supplied_commit = request.get("source_commit_sha")
    if supplied_commit not in {None, observed_commit}:
        raise ExternalProviderNuRecError(["provider_isaac_source_checkout_commit_mismatch"])
    request["source_commit_sha"] = observed_commit
    return build_provider_nurec_isaac_request(request)


def build_provider_nurec_isaac_runtime_result(
    value: Mapping[str, Any], *, verification_request: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a provider-package runtime result against the exact frozen request."""

    request = build_provider_nurec_isaac_request(verification_request)
    result = _clone(value)
    supplied = result.pop("isaac_runtime_result_digest", None)
    errors: list[str] = []
    if result.get("schema_version") != ISAAC_RUNTIME_SCHEMA:
        errors.append("provider_isaac_runtime_schema_invalid")
    if result.get("status") not in {"running", "completed", "blocked"}:
        errors.append("provider_isaac_runtime_status_invalid")
    for key in (
        "isaac_verification_request_digest",
        "package_digest",
        "fixed_camera_spec_digest",
        "runtime_container_image_digest",
        "runtime_implementation_digest",
    ):
        if result.get(key) != request.get(key):
            errors.append(f"provider_isaac_runtime_request_binding_mismatch:{key}")
    if result.get("raw_secret_values_recorded") is not False:
        errors.append("provider_isaac_runtime_secret_boundary_invalid")
    if result.get("status") == "completed":
        stage = result.get("stage")
        if not isinstance(stage, Mapping):
            errors.append("provider_isaac_runtime_stage_missing")
            stage = {}
        required_stage = {
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "transforms_valid": True,
            "dependency_inspection_available": True,
            "missing_asset_count": 0,
            "obvious_scale_mismatch_detected": False,
        }
        for key, expected_value in required_stage.items():
            if stage.get(key) != expected_value:
                errors.append(f"provider_isaac_runtime_stage_invalid:{key}")
        if stage.get("expected_prim_paths") != request["expected_prim_paths"]:
            errors.append("provider_isaac_runtime_expected_prim_binding_mismatch")
        for key in ("particlefield_prim_count", "active_collision_prim_count"):
            if (
                not isinstance(stage.get(key), int)
                or isinstance(stage.get(key), bool)
                or stage.get(key, 0) < 1
            ):
                errors.append(f"provider_isaac_runtime_stage_invalid:{key}")
        physics = result.get("physics_probe")
        if not isinstance(physics, Mapping):
            errors.append("provider_isaac_runtime_physics_probe_missing")
            physics = {}
        if (
            physics.get("ground_contact_surface_present") is not True
            or physics.get("live_rigid_body_pose_observed") is not True
            or physics.get("test_body_fell_through_floor") is not False
            or not isinstance(physics.get("contact_event_count"), int)
            or physics.get("contact_event_count", 0) < 1
            or not isinstance(physics.get("steps_executed"), int)
            or physics.get("steps_executed", 0) < request["physics_probe_request"]["steps"]
        ):
            errors.append("provider_isaac_runtime_physics_probe_incomplete")
        cameras = result.get("cameras")
        if (
            not isinstance(cameras, list)
            or [row.get("id") for row in cameras if isinstance(row, Mapping)]
            != request["fixed_camera_ids"]
            or any(
                row.get("nonblank") is not True
                or not _is_digest(row.get("digest"))
                or not isinstance(row.get("pixel_std"), (int, float))
                or isinstance(row.get("pixel_std"), bool)
                or float(row.get("pixel_std", 0)) <= 3.0
                for row in cameras
                if isinstance(row, Mapping)
            )
        ):
            errors.append("provider_isaac_runtime_camera_evidence_invalid")
        boundary = result.get("proof_boundary")
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("isaac_load_render_physics_presence_compatibility") is not True
            or any(
                boundary.get(key) is not False
                for key in (
                    "simulator_task_success_proven",
                    "physics_navigation_control_proven",
                    "physical_success_proven",
                    "physical_robot_readiness_proven",
                    "deployment_readiness_proven",
                )
            )
        ):
            errors.append("provider_isaac_runtime_claim_boundary_invalid")
        trace_pair = result.get("articulated_policy_trace_pair")
        if trace_pair is not None:
            if (
                not isinstance(trace_pair, Mapping)
                or trace_pair.get("articulated_policy_trace_pair_digest")
                != canonical_digest(trace_pair, digest_field="articulated_policy_trace_pair_digest")
                or trace_pair.get("status") not in {"completed", "blocked", "not_requested"}
                or trace_pair.get("physical_success_claimed") not in {False, None}
            ):
                errors.append("provider_isaac_runtime_policy_trace_pair_invalid")
            if isinstance(boundary, Mapping) and (
                boundary.get("articulated_policy_execution_observed")
                is not (isinstance(trace_pair, Mapping) and trace_pair.get("status") == "completed")
                or boundary.get("comparative_policy_ranking_proven") is not False
            ):
                errors.append("provider_isaac_runtime_policy_trace_boundary_invalid")
    expected = canonical_digest(result, digest_field="isaac_runtime_result_digest")
    if supplied is not None and supplied != expected:
        errors.append("provider_isaac_runtime_result_digest_mismatch")
    if errors:
        raise ExternalProviderNuRecError(errors)
    result["isaac_runtime_result_digest"] = expected
    return result


def normalize_provider_nurec_isaac_verification(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    package_artifact_root: str | Path,
    runtime_artifact_root: str | Path,
) -> dict[str, Any]:
    """Independently rehash a completed provider-package Isaac result."""

    request = build_provider_nurec_isaac_request(verification_request)
    runtime = build_provider_nurec_isaac_runtime_result(
        runtime_result, verification_request=request
    )
    errors: list[str] = []
    if runtime.get("status") != "completed":
        errors.append("provider_isaac_runtime_not_completed")

    package_root = Path(package_artifact_root)
    runtime_root = Path(runtime_artifact_root)
    for root, code in (
        (package_root, "provider_isaac_package_root_invalid"),
        (runtime_root, "provider_isaac_runtime_root_invalid"),
    ):
        if root.is_symlink() or not root.is_dir():
            errors.append(code)
    if errors:
        raise ExternalProviderNuRecError(errors)
    package_root = package_root.resolve()
    runtime_root = runtime_root.resolve()

    try:
        package_reference = _safe_relative(
            request["package_artifact_reference"],
            "provider_isaac_package_reference_unsafe",
        )
        package = package_root.joinpath(*package_reference.parts)
        if (
            package.is_symlink()
            or not package.is_file()
            or package_root not in package.resolve().parents
            or sha256_file(package) != request["package_digest"]
        ):
            errors.append("provider_isaac_package_rehash_mismatch")
    except (OSError, RuntimeError, ExternalProviderNuRecError) as exc:
        if isinstance(exc, ExternalProviderNuRecError):
            errors.extend(exc.codes)
        else:
            errors.append("provider_isaac_package_rehash_failed")

    camera_rows: list[dict[str, Any]] = []
    for camera in runtime.get("cameras") or []:
        if not isinstance(camera, Mapping):
            errors.append("provider_isaac_camera_artifact_invalid")
            continue
        try:
            reference = _safe_relative(
                camera.get("artifact_reference"),
                "provider_isaac_camera_reference_unsafe",
            )
            if reference.suffix.lower() != ".png":
                raise ExternalProviderNuRecError(["provider_isaac_camera_reference_unsafe"])
            artifact = runtime_root.joinpath(*reference.parts)
            observed = sha256_file(artifact)
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or runtime_root not in artifact.resolve().parents
                or observed != camera.get("digest")
            ):
                errors.append("provider_isaac_camera_artifact_digest_mismatch")
                continue
            camera_rows.append(
                {
                    "id": camera.get("id"),
                    "artifact_reference": reference.as_posix(),
                    "digest": observed,
                    "nonblank": camera.get("nonblank"),
                    "pixel_std": camera.get("pixel_std"),
                }
            )
        except (OSError, RuntimeError, ExternalProviderNuRecError) as exc:
            if isinstance(exc, ExternalProviderNuRecError):
                errors.extend(exc.codes)
            else:
                errors.append("provider_isaac_camera_artifact_invalid")
    if [row["id"] for row in camera_rows] != request["fixed_camera_ids"]:
        errors.append("provider_isaac_camera_artifact_inventory_mismatch")
    if errors:
        raise ExternalProviderNuRecError(errors)

    result = {
        "schema_version": ISAAC_VERIFICATION_RESULT_SCHEMA,
        "status": "verified_compatibility_only",
        "isaac_verification_request_digest": request["isaac_verification_request_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "package_digest": request["package_digest"],
        "camera_artifacts": camera_rows,
        "collision_prim_path": request["expected_prim_paths"]["collision"],
        "appearance_prim_path": request["expected_prim_paths"]["appearance"],
        "simulator_task_success_proven": False,
        "physics_navigation_control_proven": False,
        "physical_success_proven": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": "isaac_load_render_physics_presence_only",
        "claim_ceiling": "isaac_load_render_compatibility",
    }
    result["provider_isaac_verification_result_digest"] = canonical_digest(
        result, digest_field="provider_isaac_verification_result_digest"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Admit and CPU-qualify an exact provider-authored NuRec USDZ"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--qualification-output", required=True)
    parser.add_argument("--admit-in-place", action="store_true")
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        receipt = import_external_source(
            source_artifact=request,
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            admit_in_place=args.admit_in_place,
        )
        qualification = qualify_provider_nurec_usdz(
            package_path=receipt["asset_absolute_path"],
            expected_digest=receipt["asset_digest"],
            external_import_receipt_digest=receipt["external_import_receipt_digest"],
        )
        destination = Path(args.qualification_output)
        if destination.is_symlink():
            raise ExternalProviderNuRecError(
                ["provider_nurec_qualification_output_symlink_forbidden"]
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, json.JSONDecodeError, ExternalProviderNuRecError) as exc:
        codes = (
            list(exc.codes)
            if isinstance(exc, ExternalProviderNuRecError)
            else [f"provider_nurec_cli_input_error:{type(exc).__name__}"]
        )
        print(json.dumps({"status": "abstention", "blockers": sorted(codes)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": qualification["cpu_qualification_verdict"],
                "external_import_receipt_digest": receipt["external_import_receipt_digest"],
                "qualification_report_digest": qualification["qualification_report_digest"],
                "blockers": qualification["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0 if not qualification["blockers"] else 2


__all__ = [
    "ACQUISITION_RECEIPT_SCHEMA",
    "IMPORT_RECEIPT_SCHEMA",
    "IMPORT_REQUEST_SCHEMA",
    "ISAAC_REQUEST_SCHEMA",
    "ISAAC_RUNTIME_SCHEMA",
    "ISAAC_VERIFICATION_RESULT_SCHEMA",
    "QUALIFICATION_SCHEMA",
    "RIGHTS_RECEIPT_SCHEMA",
    "ExternalProviderNuRecError",
    "build_acquisition_receipt",
    "build_external_source_import_receipt",
    "build_external_source_import_request",
    "build_provider_nurec_isaac_request",
    "build_provider_nurec_isaac_request_from_checkout",
    "build_provider_nurec_isaac_runtime_result",
    "build_provider_nurec_qualification",
    "build_provider_rights_receipt",
    "import_external_source",
    "normalize_provider_nurec_isaac_verification",
    "qualify_provider_nurec_usdz",
    "sha256_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
