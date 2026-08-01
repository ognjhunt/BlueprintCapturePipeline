"""Strict local import contracts for provider-generated reconstruction assets.

This lane copies already-exported files only. It performs no network access and
cannot turn provider output into raw capture, metric, collision, Isaac, task,
physical, or deployment proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .nurec_openusd_packaging import (
    NuRecOpenUSDPackagingError,
    validate_safe_usdz_archive,
)


IMPORT_REQUEST_SCHEMA = "external_reconstruction_import_request.v1"
RIGHTS_RECEIPT_SCHEMA = "niantic_scaniverse_provenance_rights_receipt.v1"
GENERIC_RIGHTS_RECEIPT_SCHEMA = "external_reconstruction_provenance_rights_receipt.v1"
IMPORT_RECEIPT_SCHEMA = "external_reconstruction_import_receipt.v1"
SUPPORTED_SUFFIXES = {".usdz", ".usd", ".usda", ".usdc", ".ply", ".spz", ".glb"}
SUPPORTED_LOCAL_PROVIDERS = {"scaniverse", "polycam"}
MAX_ASSET_COUNT = 16
MAX_ASSET_BYTES = 2_000_000_000
MAX_TOTAL_BYTES = 4_000_000_000
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RIGHTS_DECLARATION_KEYS = (
    "provider_identity",
    "product_tier",
    "terms_version",
    "provider_scan_or_job_identity",
    "export_created_at",
    "export_performed_by",
    "source_capture_identity",
    "source_capture_digest",
    "ownership_or_license_confirmed",
    "commercial_use_status",
    "intended_uses",
    "consent_status",
    "privacy_status",
    "confidentiality_terms_status",
    "retention_status",
    "deletion_status",
    "model_training_terms_status",
    "competitive_use_status",
    "resale_status",
    "benchmarking_status",
    "user_managed_provider_processing_attested",
    "blueprint_remote_upload_performed",
    "declaration_digest",
)


class ExternalReconstructionImportError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Any) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ExternalReconstructionImportError(["artifact_not_json_serializable"]) from exc
    if not isinstance(result, dict):
        raise ExternalReconstructionImportError(["artifact_not_object"])
    return result


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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
        raise ExternalReconstructionImportError([code])
    return path


def _validate_lineage(value: Mapping[str, Any], errors: list[str]) -> None:
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "producing_method",
        "implementation_version",
        "timestamp",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            errors.append(f"{key}_missing")
    for key in (
        "source_capture_digest",
        "deterministic_configuration_digest",
        "train_heldout_split_digest",
    ):
        if not _is_digest(value.get(key)):
            errors.append(f"{key}_invalid")
    if _COMMIT.fullmatch(str(value.get("source_commit_sha") or "")) is None:
        errors.append("source_commit_sha_invalid")
    if value.get("units") != "meters":
        errors.append("units_must_be_meters")
    for key in ("original_file_references", "input_digests", "output_digests", "warnings", "blockers"):
        if not isinstance(value.get(key), list):
            errors.append(f"{key}_invalid")
    for key in (
        "camera_calibration_binding",
        "coordinate_frame_declaration",
        "provider_runtime_identity",
        "authority_used",
        "parent_artifact_or_event",
    ):
        if not isinstance(value.get(key), Mapping):
            errors.append(f"{key}_invalid")
    for key in ("cost_usd", "duration_seconds"):
        number = value.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or number < 0
        ):
            errors.append(f"{key}_invalid")


def _finalize(value: Mapping[str, Any], schema: str, digest_field: str) -> dict[str, Any]:
    artifact = _clone(value)
    supplied = artifact.pop(digest_field, None)
    artifact["schema_version"] = schema
    expected = canonical_digest(artifact, digest_field=digest_field)
    if supplied is not None and supplied != expected:
        raise ExternalReconstructionImportError([f"{digest_field}_mismatch"])
    artifact[digest_field] = expected
    return artifact


def _validate_rights_declaration(
    value: Mapping[str, Any],
    *,
    provider_identity: str,
    source_capture_identity: str,
    source_capture_digest: str,
) -> dict[str, Any]:
    declaration = _clone(value)
    errors: list[str] = []
    supplied = declaration.get("declaration_digest")
    expected = canonical_digest(declaration, digest_field="declaration_digest")
    if supplied != expected:
        errors.append("rights_declaration_digest_invalid")
    if (
        provider_identity not in SUPPORTED_LOCAL_PROVIDERS
        or declaration.get("provider_identity") != provider_identity
    ):
        errors.append("rights_provider_identity_invalid")
    for key in (
        "product_tier",
        "terms_version",
        "provider_scan_or_job_identity",
        "export_created_at",
        "export_performed_by",
        "confidentiality_terms_status",
        "retention_status",
        "deletion_status",
        "model_training_terms_status",
        "competitive_use_status",
        "resale_status",
        "benchmarking_status",
    ):
        if not isinstance(declaration.get(key), str) or not declaration[key]:
            errors.append(f"rights_{key}_missing")
    if declaration.get("source_capture_identity") != source_capture_identity or declaration.get(
        "source_capture_digest"
    ) != source_capture_digest:
        errors.append("rights_source_capture_binding_mismatch")
    if declaration.get("ownership_or_license_confirmed") is not True:
        errors.append("rights_ownership_or_license_not_confirmed")
    if declaration.get("commercial_use_status") not in {"permitted", "not_requested"}:
        errors.append("rights_commercial_use_status_invalid")
    if declaration.get("consent_status") not in {"accepted", "not_required"}:
        errors.append("rights_consent_not_accepted")
    if declaration.get("privacy_status") not in {"cleared", "restricted_local_only"}:
        errors.append("rights_privacy_not_cleared")
    if declaration.get("user_managed_provider_processing_attested") is not True:
        errors.append("rights_user_managed_provider_processing_not_attested")
    if declaration.get("blueprint_remote_upload_performed") is not False:
        errors.append("rights_blueprint_remote_upload_must_be_false")
    uses = declaration.get("intended_uses")
    if not isinstance(uses, list) or not uses or any(not isinstance(item, str) or not item for item in uses):
        errors.append("rights_intended_uses_invalid")
    if errors:
        raise ExternalReconstructionImportError(errors)
    declaration["declaration_digest"] = expected
    return declaration


def build_external_reconstruction_import_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(value)
    errors: list[str] = []
    _validate_lineage(request, errors)
    provider_identity = str(request.get("provider_identity") or "")
    if provider_identity not in SUPPORTED_LOCAL_PROVIDERS or request.get(
        "import_lane"
    ) != "local_external_import":
        errors.append("external_import_provider_or_lane_invalid")
    if request.get("remote_calls_authorized") is not False or request.get(
        "remote_calls_performed"
    ) is not False:
        errors.append("external_import_must_be_local_only")
    provider_runtime = request.get("provider_runtime_identity")
    if (
        not isinstance(provider_runtime, Mapping)
        or provider_runtime.get("provider") != "local"
        or provider_runtime.get("source_provider") != provider_identity
    ):
        errors.append("external_import_provider_runtime_binding_invalid")
    if request.get("output_digests") != []:
        errors.append("external_import_request_cannot_predeclare_outputs")
    bindings = request.get("asset_bindings")
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= MAX_ASSET_COUNT:
        errors.append("external_import_asset_bindings_invalid")
        bindings = []
    seen_ids: set[str] = set()
    original_digests = {
        item.get("digest") for item in request.get("original_file_references") or [] if isinstance(item, Mapping)
    }
    input_digests = {
        item.get("digest") for item in request.get("input_digests") or [] if isinstance(item, Mapping)
    }
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            errors.append(f"external_import_asset_binding_not_object:{index}")
            continue
        asset_id = str(binding.get("asset_id") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", asset_id) or asset_id in seen_ids:
            errors.append(f"external_import_asset_id_invalid:{index}")
        seen_ids.add(asset_id)
        try:
            relative = _safe_relative(binding.get("relative_path"), "external_import_asset_path_unsafe")
        except ExternalReconstructionImportError as exc:
            errors.extend(exc.codes)
            continue
        if relative.suffix.lower() not in SUPPORTED_SUFFIXES:
            errors.append(f"external_import_asset_format_unsupported:{index}")
        digest = binding.get("digest")
        if not _is_digest(digest):
            errors.append(f"external_import_asset_digest_invalid:{index}")
        elif digest not in original_digests or digest not in input_digests:
            errors.append(f"external_import_asset_provenance_binding_missing:{index}")
    try:
        request["provenance_rights_declaration"] = _validate_rights_declaration(
            request.get("provenance_rights_declaration") or {},
            provider_identity=provider_identity,
            source_capture_identity=str(request.get("source_capture_identity") or ""),
            source_capture_digest=str(request.get("source_capture_digest") or ""),
        )
    except ExternalReconstructionImportError as exc:
        errors.extend(exc.codes)
    if request.get("proof_effect") != "external_import_request_only" or request.get(
        "claim_ceiling"
    ) != "none":
        errors.append("external_import_request_claim_boundary_invalid")
    if errors:
        raise ExternalReconstructionImportError(errors)
    return _finalize(request, IMPORT_REQUEST_SCHEMA, "external_import_request_digest")


def build_scaniverse_provenance_rights_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("provider_identity") != "scaniverse":
        raise ExternalReconstructionImportError(["rights_receipt_provider_identity_invalid"])
    return _build_external_reconstruction_provenance_rights_receipt(
        value,
        schema=RIGHTS_RECEIPT_SCHEMA,
    )


def build_external_reconstruction_provenance_rights_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a provider-neutral local-import rights receipt.

    This receipt never authorizes a remote upload. Scaniverse callers retain
    their historical provider-specific schema; Polycam uses this generic
    schema so the import contract does not mislabel the source provider.
    """

    return _build_external_reconstruction_provenance_rights_receipt(
        value,
        schema=GENERIC_RIGHTS_RECEIPT_SCHEMA,
    )


def _build_external_reconstruction_provenance_rights_receipt(
    value: Mapping[str, Any], *, schema: str
) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "source_capture_digest",
        "declaration_digest",
    ):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            errors.append(f"rights_receipt_{key}_missing")
    if not _is_digest(receipt.get("source_capture_digest")) or not _is_digest(
        receipt.get("declaration_digest")
    ):
        errors.append("rights_receipt_binding_invalid")
    try:
        _validate_rights_declaration(
            {key: receipt.get(key) for key in _RIGHTS_DECLARATION_KEYS},
            provider_identity=str(receipt.get("provider_identity") or ""),
            source_capture_identity=str(receipt.get("source_capture_identity") or ""),
            source_capture_digest=str(receipt.get("source_capture_digest") or ""),
        )
    except ExternalReconstructionImportError as exc:
        errors.extend(f"rights_receipt_{code}" for code in exc.codes)
    if receipt.get("status") != "accepted_for_declared_local_import_only":
        errors.append("rights_receipt_status_invalid")
    if receipt.get("blueprint_remote_upload_performed") is not False or receipt.get(
        "remote_upload_authorized_by_receipt"
    ) is not False:
        errors.append("rights_receipt_cannot_authorize_remote_upload")
    if receipt.get("provider_success_is_blueprint_qualification") is not False:
        errors.append("rights_receipt_provider_success_promotion_forbidden")
    if receipt.get("proof_effect") != "provenance_and_rights_for_local_import_only" or receipt.get(
        "claim_ceiling"
    ) != "external_reconstruction_import":
        errors.append("rights_receipt_claim_boundary_invalid")
    if errors:
        raise ExternalReconstructionImportError(errors)
    return _finalize(receipt, schema, "provenance_rights_receipt_digest")


def build_external_reconstruction_import_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _clone(value)
    errors: list[str] = []
    _validate_lineage(receipt, errors)
    for key in ("external_import_request_digest", "provenance_rights_receipt_digest"):
        if not _is_digest(receipt.get(key)):
            errors.append(f"{key}_invalid")
    if receipt.get("provenance_rights_receipt_schema_version") not in {
        RIGHTS_RECEIPT_SCHEMA,
        GENERIC_RIGHTS_RECEIPT_SCHEMA,
    }:
        errors.append("provenance_rights_receipt_schema_version_invalid")
    if receipt.get("status") != "imported_derived_support_only":
        errors.append("external_import_receipt_status_invalid")
    assets = receipt.get("imported_assets")
    if not isinstance(assets, list) or not assets:
        errors.append("external_import_receipt_assets_missing")
        assets = []
    asset_digests = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            errors.append(f"external_import_receipt_asset_not_object:{index}")
            continue
        if not _is_digest(asset.get("digest")):
            errors.append(f"external_import_receipt_asset_digest_invalid:{index}")
        else:
            asset_digests.add(asset["digest"])
        size = asset.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append(f"external_import_receipt_asset_size_invalid:{index}")
        try:
            _safe_relative(asset.get("relative_path"), "external_import_receipt_asset_path_unsafe")
        except ExternalReconstructionImportError as exc:
            errors.extend(exc.codes)
        if asset.get("metadata_treated_as_untrusted") is not True:
            errors.append(f"external_import_receipt_untrusted_metadata_marker_missing:{index}")
    output_digests = {
        item.get("digest")
        for item in receipt.get("output_digests") or []
        if isinstance(item, Mapping)
    }
    if asset_digests != output_digests:
        errors.append("external_import_receipt_output_digest_binding_mismatch")
    if receipt.get("source_capture_binding_verified") is not True or receipt.get(
        "rights_and_provenance_verified"
    ) is not True:
        errors.append("external_import_receipt_binding_unverified")
    for key in (
        "raw_capture_truth",
        "metric_scale_proven",
        "collision_geometry_validated",
        "isaac_compatibility_proven",
        "simulator_task_success_proven",
        "physical_success_proven",
        "deployment_readiness_proven",
        "remote_calls_performed",
    ):
        if receipt.get(key) is not False:
            errors.append(f"external_import_forbidden_claim:{key}")
    if receipt.get("proof_effect") != "external_reconstruction_derived_support_only" or receipt.get(
        "claim_ceiling"
    ) != "external_reconstruction_import":
        errors.append("external_import_receipt_claim_boundary_invalid")
    if errors:
        raise ExternalReconstructionImportError(errors)
    return _finalize(receipt, IMPORT_RECEIPT_SCHEMA, "external_import_receipt_digest")


def _source_path(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = _safe_relative(binding.get("relative_path"), "external_import_asset_path_unsafe")
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ExternalReconstructionImportError(["external_import_asset_symlink_forbidden"])
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ExternalReconstructionImportError(["external_import_asset_path_escape"])
    if resolved.is_symlink() or not resolved.is_file():
        raise ExternalReconstructionImportError(["external_import_asset_missing"])
    size = resolved.stat().st_size
    if size > MAX_ASSET_BYTES:
        raise ExternalReconstructionImportError(["external_import_asset_oversized"])
    if _sha256_file(resolved) != binding.get("digest"):
        raise ExternalReconstructionImportError(["external_import_asset_digest_mismatch"])
    if resolved.suffix.lower() == ".usdz":
        try:
            validate_safe_usdz_archive(resolved, "external_import_asset")
        except NuRecOpenUSDPackagingError as exc:
            raise ExternalReconstructionImportError(exc.codes) from exc
    return resolved


def import_external_reconstruction(
    *, source_artifact: Mapping[str, Any], artifact_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    started = time.monotonic()
    request = build_external_reconstruction_import_request(source_artifact)
    root = Path(artifact_root).resolve()
    if Path(artifact_root).is_symlink() or not root.is_dir():
        raise ExternalReconstructionImportError(["external_import_artifact_root_invalid"])
    output = Path(output_root)
    if output.is_symlink():
        raise ExternalReconstructionImportError(["external_import_output_root_symlink_forbidden"])
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    content_id = request["external_import_request_digest"][7:]
    final_dir = output / content_id
    receipt_path = final_dir / "external_reconstruction_import_receipt.v1.json"
    if receipt_path.is_file():
        receipt = build_external_reconstruction_import_receipt(
            json.loads(receipt_path.read_text(encoding="utf-8"))
        )
        for asset in receipt["imported_assets"]:
            path = output / asset["relative_path"]
            if not path.is_file() or _sha256_file(path) != asset["digest"]:
                raise ExternalReconstructionImportError(["external_import_replay_asset_tampered"])
        return receipt
    if final_dir.exists() or final_dir.is_symlink():
        raise ExternalReconstructionImportError(["external_import_existing_output_incomplete"])
    sources = [(binding, _source_path(root, binding)) for binding in request["asset_bindings"]]
    if sum(path.stat().st_size for _binding, path in sources) > MAX_TOTAL_BYTES:
        raise ExternalReconstructionImportError(["external_import_total_size_oversized"])
    temporary = Path(tempfile.mkdtemp(prefix=".external-import-", dir=output))
    try:
        assets_dir = temporary / "assets"
        assets_dir.mkdir()
        imported_assets = []
        for index, (binding, source) in enumerate(sources):
            destination = assets_dir / f"{index:02d}-{binding['asset_id']}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            imported_assets.append(
                {
                    "asset_id": binding["asset_id"],
                    "format": source.suffix.lower(),
                    "digest": _sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                    "relative_path": f"{content_id}/assets/{destination.name}",
                    "untrusted_source_filename": source.name,
                    "metadata_treated_as_untrusted": True,
                }
            )
        declaration = request["provenance_rights_declaration"]
        rights_value = {
            **declaration,
            "stable_run_identity": request["stable_run_identity"],
            "status": "accepted_for_declared_local_import_only",
            "remote_upload_authorized_by_receipt": False,
            "provider_success_is_blueprint_qualification": False,
            "proof_effect": "provenance_and_rights_for_local_import_only",
            "claim_ceiling": "external_reconstruction_import",
        }
        if request["provider_identity"] == "scaniverse":
            rights = build_scaniverse_provenance_rights_receipt(rights_value)
            rights_filename = "niantic_scaniverse_provenance_rights_receipt.v1.json"
        else:
            rights = build_external_reconstruction_provenance_rights_receipt(rights_value)
            rights_filename = "external_reconstruction_provenance_rights_receipt.v1.json"
        (temporary / rights_filename).write_text(
            json.dumps(rights, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        value = dict(request)
        value.pop("schema_version", None)
        value.pop("asset_bindings", None)
        value.pop("provenance_rights_declaration", None)
        value.pop("external_import_request_digest", None)
        value.update(
            {
                "producing_method": "strict_local_external_reconstruction_import",
                "implementation_version": "1",
                "output_digests": [
                    {"artifact_id": item["asset_id"], "digest": item["digest"]}
                    for item in imported_assets
                ],
                "provider_runtime_identity": {
                    "provider": "local",
                    "source_provider": request["provider_identity"],
                },
                "cost_usd": 0.0,
                "duration_seconds": round(time.monotonic() - started, 6),
                "parent_artifact_or_event": {"digest": request["external_import_request_digest"]},
                "external_import_request_digest": request["external_import_request_digest"],
                "provenance_rights_receipt_digest": rights["provenance_rights_receipt_digest"],
                "provenance_rights_receipt_schema_version": rights["schema_version"],
                "imported_assets": imported_assets,
                "source_capture_binding_verified": True,
                "rights_and_provenance_verified": True,
                "status": "imported_derived_support_only",
                "raw_capture_truth": False,
                "metric_scale_proven": False,
                "collision_geometry_validated": False,
                "isaac_compatibility_proven": False,
                "simulator_task_success_proven": False,
                "physical_success_proven": False,
                "deployment_readiness_proven": False,
                "remote_calls_performed": False,
                "proof_effect": "external_reconstruction_derived_support_only",
                "claim_ceiling": "external_reconstruction_import",
            }
        )
        value.pop("provider_identity", None)
        value.pop("import_lane", None)
        value.pop("remote_calls_authorized", None)
        receipt = build_external_reconstruction_import_receipt(value)
        (temporary / "external_reconstruction_import_receipt.v1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final_dir)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "ExternalReconstructionImportError",
    "IMPORT_RECEIPT_SCHEMA",
    "IMPORT_REQUEST_SCHEMA",
    "GENERIC_RIGHTS_RECEIPT_SCHEMA",
    "RIGHTS_RECEIPT_SCHEMA",
    "SUPPORTED_LOCAL_PROVIDERS",
    "build_external_reconstruction_provenance_rights_receipt",
    "build_external_reconstruction_import_receipt",
    "build_external_reconstruction_import_request",
    "build_scaniverse_provenance_rights_receipt",
    "import_external_reconstruction",
]
