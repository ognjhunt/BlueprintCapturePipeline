"""Compile deterministic candidate-only pose and trainer GPU input bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_gpu_admission import (
    EXPECTED_RUNTIME_RESULT_SCHEMAS,
    build_reconstruction_gpu_canary_request,
)
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_pose_estimation_request,
    build_training_request,
)


SCHEMA_VERSION = "reconstruction_gpu_operation_bundle.v1"
MAX_MEMBER_BYTES = 8 * 1024**3
MAX_TOTAL_BYTES = 120 * 1024**3
MAX_MEMBER_COUNT = 20_000
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_PARTS = {
    "evaluator_hidden",
    "held_out",
    "heldout",
    "hidden_heldout",
}
_SECRET_MARKERS = {"credential", "credentials", "secret", "token", "password"}
_SECRET_SUFFIXES = {".env", ".key", ".pem", ".p12", ".pfx"}
_ALLOWED_ROLES = {
    "pose_canary": {
        "pose_execution_plan",
        "candidate_observation",
        "camera_rig",
        "calibration",
        "retention_manifest",
        "source_metadata",
    },
    "trainer_canary": {
        "dataset_export",
        "candidate_dataset_member",
        "initialization_geometry",
        "pose_result",
        "calibration",
        "retention_manifest",
    },
}
_REQUIRED_ROLES = {
    "pose_canary": {"pose_execution_plan", "candidate_observation"},
    "trainer_canary": {"dataset_export", "candidate_dataset_member"},
}


class ReconstructionGpuOperationBundleError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _portable_relative(value: Any) -> PurePosixPath | None:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        return None
    return path


def _contains_forbidden_name(path: PurePosixPath) -> bool:
    normalized = [part.casefold().replace("-", "_") for part in path.parts]
    if any(part in _FORBIDDEN_PARTS for part in normalized):
        return True
    if path.suffix.casefold() in _SECRET_SUFFIXES:
        return True
    return any(marker in part for part in normalized for marker in _SECRET_MARKERS)


def _accepted_request(
    operation: str, value: Mapping[str, Any]
) -> tuple[dict[str, Any], str, str]:
    try:
        if operation == "pose_canary":
            request = build_pose_estimation_request(value)
            digest_field = "pose_estimation_request_digest"
        elif operation == "trainer_canary":
            request = build_training_request(value)
            digest_field = "reconstruction_training_request_digest"
        else:
            raise ReconstructionGpuOperationBundleError(
                ["reconstruction_operation_bundle_operation_unsupported"]
            )
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionGpuOperationBundleError(
            [f"reconstruction_operation_request_invalid:{code}" for code in exc.codes]
        ) from exc
    return request, digest_field, EXPECTED_RUNTIME_RESULT_SCHEMAS[operation]


def _bound_source(root: Path, row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifact_id = row.get("artifact_id")
    relative = _portable_relative(row.get("relative_path"))
    role = str(row.get("role") or "")
    errors: list[str] = []
    if not isinstance(artifact_id, str) or _ARTIFACT_ID.fullmatch(artifact_id) is None:
        errors.append("reconstruction_operation_bundle_artifact_id_invalid")
    if relative is None:
        errors.append("reconstruction_operation_bundle_relative_path_invalid")
    elif _contains_forbidden_name(relative):
        errors.append("reconstruction_operation_bundle_forbidden_member_name")
    if row.get("contains_hidden_heldout_pixels") is not False:
        errors.append("reconstruction_operation_bundle_hidden_heldout_forbidden")
    expected_digest = str(row.get("digest") or "")
    if _DIGEST.fullmatch(expected_digest) is None:
        errors.append("reconstruction_operation_bundle_artifact_digest_invalid")
    if errors:
        raise ReconstructionGpuOperationBundleError(errors)
    assert relative is not None and isinstance(artifact_id, str)
    lexical = root.joinpath(*relative.parts)
    if lexical.is_symlink():
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_symlink_forbidden"]
        )
    try:
        source = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_artifact_missing"]
        ) from exc
    if (
        (source != root and root not in source.parents)
        or not source.is_file()
        or source.is_symlink()
    ):
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_artifact_escape_or_type_invalid"]
        )
    size = source.stat().st_size
    if size > MAX_MEMBER_BYTES:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_member_oversized"]
        )
    if _sha256(source) != expected_digest:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_artifact_digest_mismatch"]
        )
    suffix = "".join(source.suffixes)[-32:]
    archive_path = f"inputs/{artifact_id}{suffix}"
    return source, {
        "artifact_id": artifact_id,
        "role": role,
        "digest": expected_digest,
        "bytes": size,
        "archive_path": archive_path,
        "contains_hidden_heldout_pixels": False,
    }


def _write_member(
    archive: zipfile.ZipFile, name: str, source: Path | bytes
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    if isinstance(source, bytes):
        archive.writestr(info, source)
        return
    with source.open("rb") as input_stream, archive.open(
        info, "w", force_zip64=True
    ) as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def _validate_existing(
    final: Path, *, expected_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_path = final / "reconstruction_gpu_operation_bundle.v1.json"
    archive_path = final / "reconstruction_gpu_operation_bundle.zip"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_existing_receipt_invalid"]
        ) from exc
    if (
        receipt.get("bundle_manifest_digest")
        != expected_manifest.get("bundle_manifest_digest")
        or receipt.get("operation_input_bundle_digest") != _sha256(archive_path)
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_existing_output_tampered"]
        )
    return receipt


def compile_reconstruction_gpu_operation_bundle(
    *,
    operation: str,
    operation_request: Mapping[str, Any],
    artifact_root: str | Path,
    artifact_bindings: Sequence[Mapping[str, Any]],
    output_root: str | Path,
) -> dict[str, Any]:
    """Compile an immutable candidate-only bundle without spending or uploading."""

    request, digest_field, expected_result_schema = _accepted_request(
        operation, operation_request
    )
    lexical_root = Path(artifact_root)
    if lexical_root.is_symlink():
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_artifact_root_symlink_forbidden"]
        )
    try:
        root = lexical_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_artifact_root_invalid"]
        ) from exc
    if not root.is_dir():
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_artifact_root_invalid"]
        )
    if not artifact_bindings or len(artifact_bindings) > MAX_MEMBER_COUNT:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_member_count_invalid"]
        )
    sources: list[tuple[Path, dict[str, Any]]] = []
    for raw in artifact_bindings:
        if not isinstance(raw, Mapping):
            raise ReconstructionGpuOperationBundleError(
                ["reconstruction_operation_bundle_binding_invalid"]
            )
        source, row = _bound_source(root, raw)
        if row["role"] not in _ALLOWED_ROLES[operation]:
            raise ReconstructionGpuOperationBundleError(
                ["reconstruction_operation_bundle_role_invalid"]
            )
        sources.append((source, row))
    ids = [row["artifact_id"] for _, row in sources]
    archive_names = [row["archive_path"] for _, row in sources]
    roles = {row["role"] for _, row in sources}
    if len(ids) != len(set(ids)) or len(archive_names) != len(set(archive_names)):
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_duplicate_member"]
        )
    if not _REQUIRED_ROLES[operation] <= roles:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_required_role_missing"]
        )
    total_bytes = sum(row["bytes"] for _, row in sources)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_total_oversized"]
        )
    request_digest = request[digest_field]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "operation_request_schema": request["schema_version"],
        "operation_request_digest": request_digest,
        "expected_runtime_result_schema": expected_result_schema,
        "source_capture_digest": request["source_capture_digest"],
        "source_commit_sha": request["source_commit_sha"],
        "worker_image_digest": request["container_image_digest"],
        "reconstruction_dataset_digest": request["reconstruction_dataset_digest"],
        "frozen_split_digest": request["train_heldout_split_digest"],
        "calibration_digest": request.get("calibration_digest"),
        "artifact_members": [row for _, row in sorted(sources, key=lambda item: item[1]["artifact_id"])],
        "artifact_member_count": len(sources),
        "artifact_total_bytes": total_bytes,
        "request_archive_path": "operation_request.json",
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "raw_secret_values_included": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "candidate_operation_input_only",
    }
    manifest["bundle_manifest_digest"] = canonical_digest(
        manifest, digest_field="bundle_manifest_digest"
    )
    destination = Path(output_root)
    if destination.is_symlink():
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_output_root_symlink_forbidden"]
        )
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    final = destination / request_digest.removeprefix("sha256:")
    if final.exists():
        return _validate_existing(final, expected_manifest=manifest)

    temporary = Path(tempfile.mkdtemp(prefix=".reconstruction-op-", dir=destination))
    try:
        archive_path = temporary / "reconstruction_gpu_operation_bundle.zip"
        request_bytes = (
            json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            _write_member(archive, "bundle_manifest.json", manifest_bytes)
            _write_member(archive, "operation_request.json", request_bytes)
            for source, row in sorted(sources, key=lambda item: item[1]["archive_path"]):
                _write_member(archive, row["archive_path"], source)
        bundle_digest = _sha256(archive_path)
        receipt = {
            **manifest,
            "status": "compiled",
            "operation_input_bundle_digest": bundle_digest,
            "bundle_artifact_reference": (
                f"{request_digest.removeprefix('sha256:')}/"
                "reconstruction_gpu_operation_bundle.zip"
            ),
            "bundle_bytes": archive_path.stat().st_size,
            "cost_usd": 0.0,
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        write_json(temporary / "reconstruction_gpu_operation_bundle.v1.json", receipt)
        os.replace(temporary, final)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_canary_request_from_operation_bundle(
    *,
    request_fields: Mapping[str, Any],
    operation_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind paid admission inputs to one verified, non-authorizing bundle receipt."""

    receipt = json.loads(json.dumps(dict(operation_bundle)))
    errors: list[str] = []
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("status") != "compiled":
        errors.append("reconstruction_operation_bundle_receipt_schema_or_status_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("reconstruction_operation_bundle_receipt_digest_mismatch")
    if receipt.get("candidate_may_read_hidden_heldout") is not False:
        errors.append("reconstruction_operation_bundle_receipt_hidden_access_forbidden")
    if receipt.get("trainer_may_grade_heldout") is not False:
        errors.append("reconstruction_operation_bundle_receipt_self_grading_forbidden")
    if receipt.get("raw_secret_values_included") is not False:
        errors.append("reconstruction_operation_bundle_receipt_secret_values_forbidden")
    if receipt.get("provider_allocation_performed") is not False:
        errors.append("reconstruction_operation_bundle_receipt_provider_mutation_invalid")
    if receipt.get("paid_execution_authorized_by_bundle") is not False:
        errors.append("reconstruction_operation_bundle_receipt_authority_invalid")
    if receipt.get("proof_effect") != "none":
        errors.append("reconstruction_operation_bundle_receipt_proof_effect_invalid")
    for key in (
        "operation_request_digest",
        "operation_input_bundle_digest",
        "bundle_manifest_digest",
    ):
        if _DIGEST.fullmatch(str(receipt.get(key) or "")) is None:
            errors.append(f"reconstruction_operation_bundle_receipt_{key}_invalid")
    fields = json.loads(json.dumps(dict(request_fields)))
    expected_bindings = {
        "operation": receipt.get("operation"),
        "source_commit_sha": receipt.get("source_commit_sha"),
        "worker_image_digest": receipt.get("worker_image_digest"),
        "reconstruction_dataset_digest": receipt.get("reconstruction_dataset_digest"),
        "frozen_split_digest": receipt.get("frozen_split_digest"),
        "calibration_digest": receipt.get("calibration_digest"),
        "expected_runtime_result_schema": receipt.get("expected_runtime_result_schema"),
    }
    for key, expected in expected_bindings.items():
        supplied = fields.get(key)
        if supplied is not None and supplied != expected:
            errors.append(f"reconstruction_operation_bundle_canary_{key}_mismatch")
        fields[key] = expected
    fields["operation_request_digest"] = receipt.get("operation_request_digest")
    fields["operation_input_bundle_digest"] = receipt.get(
        "operation_input_bundle_digest"
    )
    if errors:
        raise ReconstructionGpuOperationBundleError(errors)
    try:
        return build_reconstruction_gpu_canary_request(fields)
    except ValueError as exc:
        raise ReconstructionGpuOperationBundleError(
            [f"reconstruction_operation_bundle_canary_request_invalid:{exc}"]
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=sorted(_ALLOWED_ROLES), required=True)
    parser.add_argument("--operation-request", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-bindings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    bindings = json.loads(args.artifact_bindings.read_text(encoding="utf-8"))
    if not isinstance(bindings, list):
        raise ReconstructionGpuOperationBundleError(
            ["reconstruction_operation_bundle_bindings_not_array"]
        )
    receipt = compile_reconstruction_gpu_operation_bundle(
        operation=args.operation,
        operation_request=json.loads(
            args.operation_request.read_text(encoding="utf-8")
        ),
        artifact_root=args.artifact_root,
        artifact_bindings=bindings,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "operation": receipt["operation"],
                "operation_request_digest": receipt["operation_request_digest"],
                "operation_input_bundle_digest": receipt[
                    "operation_input_bundle_digest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ReconstructionGpuOperationBundleError",
    "SCHEMA_VERSION",
    "build_canary_request_from_operation_bundle",
    "compile_reconstruction_gpu_operation_bundle",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
