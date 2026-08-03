"""Deterministic output transport for the canonical Splatfacto Vast adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping
import zipfile

from .gaussian_splat_decode import read_standard_3dgs_ply
from .decision_evidence_contracts import canonical_digest, canonical_json
from .reconstruction_gpu_operation_output import ReconstructionGpuOperationOutputError


SCHEMA_VERSION = "canonical_3dgs_vast_output_bundle.v1"
MAX_MEMBER_BYTES = 16 * 1024**3
MAX_TOTAL_BYTES = 96 * 1024**3
MAX_MEMBER_COUNT = 20_000
MANIFEST_MEMBER = "canonical_3dgs_vast_output_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _portable(value: Any) -> PurePosixPath | None:
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


def _member(archive: zipfile.ZipFile, name: str, source: Path | bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    if isinstance(source, bytes):
        archive.writestr(info, source)
    else:
        with source.open("rb") as input_stream, archive.open(
            info, "w", force_zip64=True
        ) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def compile_canonical_3dgs_vast_output_bundle(
    *,
    result_root: str | Path,
    worker_receipt: Mapping[str, Any],
    output_path: str | Path,
    worker_image_digest: str,
    source_commit_sha: str,
) -> dict[str, Any]:
    """Package one successful, externally admitted arm without grading it."""

    root = Path(result_root).resolve()
    receipt = json.loads(canonical_json(dict(worker_receipt)))
    errors: list[str] = []
    if (
        receipt.get("exit_code") != 0
        or receipt.get("canonical_3dgs_worker_receipt_digest")
        != canonical_digest(receipt, digest_field="canonical_3dgs_worker_receipt_digest")
        or receipt.get("runtime_identity", {}).get("worker_image_digest")
        != worker_image_digest
        or receipt.get("runtime_identity", {}).get("source_commit_sha_bound_by_plan")
        != source_commit_sha
    ):
        errors.append("canonical_vast_output_worker_receipt_invalid")
    artifacts = receipt.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    splats = [
        row
        for row in artifacts
        if isinstance(row, Mapping) and row.get("kind") == "standard_3dgs_ply"
    ]
    if len(splats) != 1:
        errors.append("canonical_vast_output_standard_ply_missing")
    if errors:
        raise ReconstructionGpuOperationOutputError(errors)
    files: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReconstructionGpuOperationOutputError(
                ["canonical_vast_output_symlink_forbidden"]
            )
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_MEMBER_BYTES:
            raise ReconstructionGpuOperationOutputError(
                ["canonical_vast_output_member_oversized"]
            )
        files.append(
            (
                path,
                {
                    "relative_path": relative,
                    "archive_path": "results/" + relative,
                    "digest": _sha256(path),
                    "bytes": path.stat().st_size,
                },
            )
        )
    if not files or len(files) > MAX_MEMBER_COUNT:
        raise ReconstructionGpuOperationOutputError(
            ["canonical_vast_output_member_count_invalid"]
        )
    if sum(row["bytes"] for _, row in files) > MAX_TOTAL_BYTES:
        raise ReconstructionGpuOperationOutputError(
            ["canonical_vast_output_total_oversized"]
        )
    splat_path = root.joinpath(*PurePosixPath(str(splats[0]["relative_path"])).parts)
    if (
        not splat_path.is_file()
        or _sha256(splat_path) != splats[0].get("digest")
    ):
        raise ReconstructionGpuOperationOutputError(
            ["canonical_vast_output_standard_ply_digest_mismatch"]
        )
    try:
        decoded = read_standard_3dgs_ply(splat_path)
    except (OSError, TypeError, ValueError) as exc:
        raise ReconstructionGpuOperationOutputError(
            ["canonical_vast_output_standard_ply_invalid"]
        ) from exc
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_generated_not_graded",
        "operation": "trainer_canary",
        "arm_id": "splatfacto-comparison",
        "canonical_3dgs_execution_plan_digest": receipt[
            "canonical_3dgs_execution_plan_digest"
        ],
        "transport_bundle_digest": receipt["transport_bundle_digest"],
        "worker_receipt_digest": receipt["canonical_3dgs_worker_receipt_digest"],
        "worker_image_digest": worker_image_digest,
        "source_commit_sha": source_commit_sha,
        "standard_3dgs_ply_digest": splats[0]["digest"],
        "gaussian_count": int(decoded.count),
        "members": [row for _, row in files],
        "member_count": len(files),
        "hidden_heldout_pixels_included": False,
        "candidate_self_graded": False,
        "quality_winner": None,
        "proof_effect": "appearance_asset_candidate_only",
        "claim_ceiling": "appearance_reconstruction",
    }
    manifest["output_bundle_receipt_digest"] = canonical_digest(
        manifest, digest_field="output_bundle_receipt_digest"
    )
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".canonical-vast-output-", dir=destination.parent))
    try:
        archive_path = temporary / "output.zip"
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            _member(archive, MANIFEST_MEMBER, (canonical_json(manifest) + "\n").encode())
            for source, row in files:
                _member(archive, row["archive_path"], source)
        if destination.exists():
            if _sha256(destination) != _sha256(archive_path):
                raise ReconstructionGpuOperationOutputError(
                    ["canonical_vast_output_immutable_conflict"]
                )
        else:
            shutil.copyfile(archive_path, destination)
        return {
            **manifest,
            "operation_output_bundle_digest": _sha256(destination),
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def validate_canonical_3dgs_vast_output_bundle(
    *,
    bundle_path: str | Path,
    expected_operation: str,
    expected_operation_request_digest: str,
    expected_worker_image_digest: str,
    expected_source_commit_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently decode every returned byte before accepting transport."""

    bundle = Path(bundle_path).resolve()
    errors: list[str] = []
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or MANIFEST_MEMBER not in names:
                raise ValueError("member_set")
            manifest = json.loads(archive.read(MANIFEST_MEMBER))
            expected_names = {MANIFEST_MEMBER} | {
                str(row["archive_path"]) for row in manifest.get("members") or []
            }
            if set(names) != expected_names:
                raise ValueError("member_set")
            for info in infos:
                if (
                    _portable(info.filename) is None
                    or info.is_dir()
                    or ((info.external_attr >> 16) & 0o170000) == stat.S_IFLNK
                    or info.file_size > MAX_MEMBER_BYTES
                    or info.compress_type != zipfile.ZIP_STORED
                ):
                    raise ValueError("unsafe_member")
            for row in manifest.get("members") or []:
                payload = archive.read(str(row["archive_path"]))
                if (
                    len(payload) != row.get("bytes")
                    or "sha256:" + hashlib.sha256(payload).hexdigest() != row.get("digest")
                ):
                    raise ValueError("digest")
            splat_rows = [
                row
                for row in manifest.get("members") or []
                if row.get("digest") == manifest.get("standard_3dgs_ply_digest")
            ]
            if len(splat_rows) != 1:
                raise ValueError("splat")
            with tempfile.TemporaryDirectory(prefix="canonical-vast-ply-") as tmp:
                splat_path = Path(tmp) / "candidate.ply"
                splat_path.write_bytes(archive.read(str(splat_rows[0]["archive_path"])))
                decoded_count = read_standard_3dgs_ply(splat_path).count
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ReconstructionGpuOperationOutputError(
            ["canonical_vast_output_bundle_invalid"]
        ) from exc
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "candidate_generated_not_graded"
        or manifest.get("operation") != expected_operation
        or manifest.get("canonical_3dgs_execution_plan_digest")
        != expected_operation_request_digest
        or manifest.get("worker_image_digest") != expected_worker_image_digest
        or manifest.get("source_commit_sha") != expected_source_commit_sha
        or manifest.get("output_bundle_receipt_digest")
        != canonical_digest(manifest, digest_field="output_bundle_receipt_digest")
        or manifest.get("gaussian_count") != decoded_count
        or decoded_count < 1
        or manifest.get("hidden_heldout_pixels_included") is not False
        or manifest.get("candidate_self_graded") is not False
        or manifest.get("quality_winner") is not None
    ):
        errors.append("canonical_vast_output_binding_invalid")
    if errors:
        raise ReconstructionGpuOperationOutputError(errors)
    receipt = {
        **manifest,
        "operation_output_bundle_digest": _sha256(bundle),
    }
    runtime = {
        "schema_version": "canonical_3dgs_vast_runtime_result.v1",
        "status": "succeeded",
        "canonical_3dgs_worker_receipt_digest": manifest["worker_receipt_digest"],
        "reconstruction_training_result_digest": canonical_digest(
            {
                "worker_receipt_digest": manifest["worker_receipt_digest"],
                "standard_3dgs_ply_digest": manifest["standard_3dgs_ply_digest"],
            }
        ),
        "hidden_heldout_pixels_included": False,
        "candidate_self_graded": False,
        "proof_effect": "appearance_asset_candidate_only",
    }
    return receipt, runtime


__all__ = [
    "SCHEMA_VERSION",
    "compile_canonical_3dgs_vast_output_bundle",
    "validate_canonical_3dgs_vast_output_bundle",
]
