"""Package and independently validate reconstruction GPU operation outputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    POSE_REQUEST_SCHEMA_VERSION,
    POSE_RESULT_SCHEMA_VERSION,
    TRAINING_REQUEST_SCHEMA_VERSION,
    TRAINING_RESULT_SCHEMA_VERSION,
    build_pose_estimation_request,
    build_pose_estimation_result,
    build_training_request,
    build_training_result,
)


SCHEMA_VERSION = "reconstruction_gpu_operation_output_bundle.v1"
MAX_MEMBER_BYTES = 16 * 1024**3
MAX_TOTAL_BYTES = 160 * 1024**3
MAX_MEMBER_COUNT = 50_000
MAX_METADATA_BYTES = 16 * 1024**2
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 128 * 1024**2
STREAM_CHUNK_BYTES = 1024 * 1024


class ReconstructionGpuOperationOutputError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(STREAM_CHUNK_BYTES), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _accepted(
    operation: str,
    operation_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    try:
        if operation == "pose_canary":
            request = build_pose_estimation_request(operation_request)
            result = build_pose_estimation_result(runtime_result)
            request_field = "pose_estimation_request_digest"
            result_field = "pose_estimation_result_digest"
        elif operation == "trainer_canary":
            request = build_training_request(operation_request)
            result = build_training_result(runtime_result)
            request_field = "reconstruction_training_request_digest"
            result_field = "reconstruction_training_result_digest"
        else:
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_operation_unsupported"]
            )
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionGpuOperationOutputError(
            [f"reconstruction_operation_output_contract_invalid:{code}" for code in exc.codes]
        ) from exc
    if result.get(request_field) != request.get(request_field):
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_request_binding_mismatch"]
        )
    return request, result, request_field, result_field


def _result_root(
    operation: str,
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    output_root: Path,
) -> Path:
    if operation == "pose_canary":
        plan_digest = str(result.get("native_360_colmap_execution_plan_digest") or "")
        if not plan_digest.startswith("sha256:"):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_pose_plan_binding_missing"]
            )
        relative = "native_colmap_execution_" + plan_digest[7:23]
    else:
        relative = str(request["reconstruction_training_request_digest"])[7:23]
    root = output_root / relative
    if root.is_symlink() or not root.is_dir() or output_root not in root.resolve().parents:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_result_root_invalid"]
        )
    return root.resolve()


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


def _artifact_rows(result: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    source_rows = [
        *(result.get("output_digests") or []),
        *(result.get("checkpoint_references") or []),
    ]
    if len(source_rows) > MAX_MEMBER_COUNT:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_member_count_invalid"]
        )
    rows: list[dict[str, Any]] = []
    for raw in source_rows:
        if not isinstance(raw, Mapping):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_reference_invalid"]
            )
        portable = _portable(raw.get("artifact_id"))
        if portable is None:
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_reference_invalid"]
            )
        source = root.joinpath(*portable.parts)
        if (
            source.is_symlink()
            or not source.is_file()
            or root not in source.resolve().parents
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_artifact_invalid"]
            )
        size = source.stat().st_size
        digest = _sha256(source)
        if size > MAX_MEMBER_BYTES or digest != raw.get("digest"):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_artifact_binding_invalid"]
            )
        rows.append(
            {
                "artifact_id": portable.as_posix(),
                "archive_path": f"artifacts/{portable.as_posix()}",
                "digest": digest,
                "bytes": size,
                "source": source,
            }
        )
    ids = [row["artifact_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_duplicate_artifact"]
        )
    if sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_total_oversized"]
        )
    return sorted(rows, key=lambda row: row["artifact_id"])


def _write_member(archive: zipfile.ZipFile, name: str, source: Path | bytes) -> None:
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
        while True:
            chunk = input_stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            output_stream.write(chunk)


def compile_reconstruction_gpu_operation_output_bundle(
    *,
    operation: str,
    operation_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    operation_output_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a deterministic transport bundle without accepting its scientific claims."""

    request, result, request_field, result_field = _accepted(
        operation, operation_request, runtime_result
    )
    lexical_root = Path(operation_output_root)
    if lexical_root.is_symlink():
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_root_symlink_forbidden"]
        )
    output_root = lexical_root.resolve()
    root = _result_root(operation, request, result, output_root)
    rows = _artifact_rows(result, root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "operation_request_schema": request["schema_version"],
        "operation_request_digest": request[request_field],
        "runtime_result_schema": result["schema_version"],
        "runtime_result_digest": result[result_field],
        "source_commit_sha": request["source_commit_sha"],
        "worker_image_digest": request["container_image_digest"],
        "artifact_members": [
            {key: row[key] for key in ("artifact_id", "archive_path", "digest", "bytes")}
            for row in rows
        ],
        "artifact_member_count": len(rows),
        "artifact_total_bytes": sum(row["bytes"] for row in rows),
        "runtime_result_archive_path": "runtime_result.json",
        "heldout_labels_included": False,
        "candidate_self_graded": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "unaccepted_candidate_result_transport_only",
    }
    manifest["output_manifest_digest"] = canonical_digest(
        manifest, digest_field="output_manifest_digest"
    )
    destination = Path(output_path)
    if destination.is_symlink():
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_bundle_symlink_forbidden"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    result_bytes = (
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as archive:
            _write_member(archive, "output_manifest.json", manifest_bytes)
            _write_member(archive, "runtime_result.json", result_bytes)
            for row in rows:
                _write_member(archive, row["archive_path"], row["source"])
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        **manifest,
        "status": "compiled",
        "operation_output_bundle_digest": _sha256(destination),
        "operation_output_bundle_bytes": destination.stat().st_size,
        "cost_usd": 0.0,
    }


def _read_member_capped(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, cap: int
) -> bytes:
    if member.file_size > cap:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_metadata_oversized"]
        )
    payload = bytearray()
    with archive.open(member, "r") as stream:
        while True:
            chunk = stream.read(min(STREAM_CHUNK_BYTES, cap + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > cap:
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_metadata_oversized"]
                )
    return bytes(payload)


def _hash_member_streamed(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, cap: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as stream:
        while True:
            chunk = stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > cap:
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_member_oversized"]
                )
            digest.update(chunk)
    return size, "sha256:" + digest.hexdigest()


def _result_artifact_bindings(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows = [
        *(result.get("output_digests") or []),
        *(result.get("checkpoint_references") or []),
    ]
    bindings: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_result_reference_invalid"]
            )
        portable = _portable(row.get("artifact_id"))
        digest = str(row.get("digest") or "")
        if portable is None or not digest.startswith("sha256:"):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_result_reference_invalid"]
            )
        bindings.append((portable.as_posix(), digest))
    if len(bindings) != len(set(bindings)):
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_duplicate_artifact"]
        )
    return sorted(bindings)


def validate_reconstruction_gpu_operation_output_bundle(
    *,
    bundle_path: str | Path,
    expected_operation: str,
    expected_operation_request_digest: str,
    expected_worker_image_digest: str,
    expected_source_commit_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a retrieved bundle independently before provider teardown."""

    if expected_operation not in {"pose_canary", "trainer_canary"}:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_operation_unsupported"]
        )

    source = Path(bundle_path)
    if source.is_symlink():
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_bundle_symlink_forbidden"]
        )
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_bundle_missing"]
        ) from exc
    if not source.is_file() or source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_bundle_invalid"]
        )
    try:
        archive_context = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReconstructionGpuOperationOutputError(
            ["reconstruction_operation_output_bundle_invalid"]
        ) from exc
    with archive_context as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        errors: list[str] = []
        if len(names) != len(set(names)):
            errors.append("reconstruction_operation_output_inventory_duplicate")
        total = 0
        for member in members:
            portable = _portable(member.filename)
            if (
                portable is None
                or member.is_dir()
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.compress_type != zipfile.ZIP_STORED
                or member.file_size > MAX_MEMBER_BYTES
            ):
                errors.append("reconstruction_operation_output_member_unsafe")
            total += member.file_size
        if total > MAX_ARCHIVE_BYTES:
            errors.append("reconstruction_operation_output_uncompressed_oversized")
        if errors:
            raise ReconstructionGpuOperationOutputError(errors)
        try:
            manifest_member = archive.getinfo("output_manifest.json")
            result_member = archive.getinfo("runtime_result.json")
            manifest_value = json.loads(
                _read_member_capped(archive, manifest_member, cap=MAX_METADATA_BYTES)
            )
            result_value = json.loads(
                _read_member_capped(archive, result_member, cap=MAX_METADATA_BYTES)
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_metadata_invalid"]
            ) from exc
        if not isinstance(manifest_value, Mapping) or not isinstance(
            result_value, Mapping
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_metadata_invalid"]
            )
        manifest = dict(manifest_value)
        result = dict(result_value)
        if manifest.get("output_manifest_digest") != canonical_digest(
            manifest, digest_field="output_manifest_digest"
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_manifest_digest_mismatch"]
            )
        expected_request_schema = (
            POSE_REQUEST_SCHEMA_VERSION
            if expected_operation == "pose_canary"
            else TRAINING_REQUEST_SCHEMA_VERSION
        )
        expected_result_schema = (
            POSE_RESULT_SCHEMA_VERSION
            if expected_operation == "pose_canary"
            else TRAINING_RESULT_SCHEMA_VERSION
        )
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("operation") != expected_operation
            or manifest.get("operation_request_schema") != expected_request_schema
            or manifest.get("runtime_result_schema") != expected_result_schema
            or manifest.get("operation_request_digest")
            != expected_operation_request_digest
            or manifest.get("worker_image_digest") != expected_worker_image_digest
            or manifest.get("source_commit_sha") != expected_source_commit_sha
            or manifest.get("heldout_labels_included") is not False
            or manifest.get("candidate_self_graded") is not False
            or manifest.get("scientific_qualification_inferred") is not False
            or manifest.get("proof_effect") != "none"
            or manifest.get("claim_ceiling")
            != "unaccepted_candidate_result_transport_only"
            or manifest.get("runtime_result_archive_path") != "runtime_result.json"
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_expected_binding_mismatch"]
            )
        if expected_operation == "pose_canary":
            try:
                result = build_pose_estimation_result(result)
            except ReconstructionWorkerContractError as exc:
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_pose_result_invalid"]
                ) from exc
            result_digest = result["pose_estimation_result_digest"]
        elif expected_operation == "trainer_canary":
            try:
                result = build_training_result(result)
            except ReconstructionWorkerContractError as exc:
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_training_result_invalid"]
                ) from exc
            result_digest = result["reconstruction_training_result_digest"]
        else:
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_operation_unsupported"]
            )
        rows = manifest.get("artifact_members")
        if not isinstance(rows, list) or len(rows) > MAX_MEMBER_COUNT:
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_inventory_invalid"]
            )
        expected_names = {
            "output_manifest.json",
            "runtime_result.json",
            *(
                str(row.get("archive_path") or "")
                for row in rows
                if isinstance(row, Mapping)
            ),
        }
        if set(names) != expected_names or manifest.get("artifact_member_count") != len(
            rows
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_inventory_invalid"]
            )
        manifest_bindings: list[tuple[str, str]] = []
        artifact_total_bytes = 0
        for row in rows:
            if not isinstance(row, Mapping):
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_manifest_member_invalid"]
                )
            artifact_id = _portable(row.get("artifact_id"))
            archive_path = _portable(row.get("archive_path"))
            if (
                artifact_id is None
                or archive_path is None
                or archive_path.as_posix() != f"artifacts/{artifact_id.as_posix()}"
                or not isinstance(row.get("bytes"), int)
                or isinstance(row.get("bytes"), bool)
                or row.get("bytes", -1) < 0
            ):
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_manifest_member_invalid"]
                )
            try:
                member = archive.getinfo(archive_path.as_posix())
                size, digest = _hash_member_streamed(
                    archive, member, cap=MAX_MEMBER_BYTES
                )
            except (KeyError, zipfile.BadZipFile) as exc:
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_artifact_invalid"]
                ) from exc
            if (
                size != row.get("bytes")
                or digest != row.get("digest")
            ):
                raise ReconstructionGpuOperationOutputError(
                    ["reconstruction_operation_output_artifact_digest_mismatch"]
                )
            artifact_total_bytes += size
            manifest_bindings.append((artifact_id.as_posix(), digest))
        if (
            sorted(manifest_bindings) != _result_artifact_bindings(result)
            or manifest.get("artifact_total_bytes") != artifact_total_bytes
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_result_artifacts_mismatch"]
            )
        if (
            manifest.get("runtime_result_schema") != result["schema_version"]
            or manifest.get("runtime_result_digest") != result_digest
        ):
            raise ReconstructionGpuOperationOutputError(
                ["reconstruction_operation_output_result_manifest_mismatch"]
            )
    receipt = {
        **manifest,
        "status": "validated",
        "operation_output_bundle_digest": _sha256(source),
        "operation_output_bundle_bytes": source.stat().st_size,
        "cost_usd": 0.0,
    }
    receipt["output_bundle_receipt_digest"] = canonical_digest(
        receipt, digest_field="output_bundle_receipt_digest"
    )
    return receipt, result


__all__ = [
    "ReconstructionGpuOperationOutputError",
    "SCHEMA_VERSION",
    "compile_reconstruction_gpu_operation_output_bundle",
    "validate_reconstruction_gpu_operation_output_bundle",
]
