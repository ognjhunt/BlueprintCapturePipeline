"""Package and validate complete Isaac reconstruction verification outputs."""

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
from .external_provider_nurec import (
    ISAAC_RUNTIME_SCHEMA as PROVIDER_ISAAC_RUNTIME_SCHEMA,
    ExternalProviderNuRecError,
    build_provider_nurec_isaac_runtime_result,
)
from .isaac_reconstruction_verification import (
    ISAAC_RUNTIME_RESULT_SCHEMA,
    IsaacReconstructionVerificationError,
    build_isaac_runtime_result_v3,
)
from .reconstruction_isaac_worker_bundle import (
    PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA,
    validate_isaac_verification_worker_bundle_receipt,
)


SCHEMA_VERSION = "isaac_verification_output_bundle.v1"
MAX_MEMBER_BYTES = 4_000_000_000
MAX_TOTAL_BYTES = 5_000_000_000
MAX_METADATA_BYTES = 16 * 1024**2
STREAM_CHUNK_BYTES = 1024 * 1024


class IsaacVerificationOutputBundleError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(STREAM_CHUNK_BYTES), b""):
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


def _load_runtime_result(
    path: Path,
    *,
    receipt: Mapping[str, Any],
    verification_request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("runtime result is not an object")
        if receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA:
            if verification_request is None:
                raise TypeError("provider verification request is missing")
            return build_provider_nurec_isaac_runtime_result(
                value, verification_request=verification_request
            )
        return build_isaac_runtime_result_v3(value)
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ExternalProviderNuRecError,
        IsaacReconstructionVerificationError,
    ) as exc:
        raise IsaacVerificationOutputBundleError(["isaac_output_runtime_result_invalid"]) from exc


def _result_bindings(*, result: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected_schema = (
        PROVIDER_ISAAC_RUNTIME_SCHEMA
        if receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
        else ISAAC_RUNTIME_RESULT_SCHEMA
    )
    if (
        result.get("schema_version") != expected_schema
        or result.get("isaac_verification_request_digest")
        != receipt.get("isaac_verification_request_digest")
        or result.get("package_digest") != receipt.get("package_digest")
        or result.get("fixed_camera_spec_digest") != receipt.get("fixed_camera_spec_digest")
        or result.get("runtime_container_image_digest")
        != receipt.get("runtime_container_image_digest")
        or result.get("runtime_implementation_digest")
        != receipt.get("runtime_implementation_digest")
        or result.get("raw_secret_values_recorded") is not False
    ):
        raise IsaacVerificationOutputBundleError(["isaac_output_runtime_binding_mismatch"])


def _result_artifact_bindings(result: Mapping[str, Any]) -> list[tuple[PurePosixPath, str]]:
    cameras = result.get("cameras")
    cameras = cameras if isinstance(cameras, list) else []
    bindings: list[tuple[PurePosixPath, str]] = []
    for camera in cameras:
        if not isinstance(camera, Mapping):
            raise IsaacVerificationOutputBundleError(["isaac_output_camera_reference_invalid"])
        reference = _portable(camera.get("artifact_reference"))
        if reference is None or reference.suffix.lower() != ".png":
            raise IsaacVerificationOutputBundleError(["isaac_output_camera_reference_invalid"])
        bindings.append((reference, str(camera.get("digest") or "")))

    robot = result.get("robot")
    robot = robot if isinstance(robot, Mapping) else {}
    robot_only = robot.get("robot_only_pass")
    robot_only = robot_only if isinstance(robot_only, list) else []
    for evidence in robot_only:
        if not isinstance(evidence, Mapping):
            raise IsaacVerificationOutputBundleError(["isaac_output_robot_reference_invalid"])
        for reference_key, digest_key, suffix in (
            ("rgb_artifact_reference", "rgb_digest", ".png"),
            ("distance_artifact_reference", "distance_digest", ".npy"),
        ):
            reference_value = evidence.get(reference_key)
            digest_value = evidence.get(digest_key)
            if reference_value is None and digest_value is None:
                continue
            reference = _portable(reference_value)
            if reference is None or reference.suffix.lower() != suffix or not digest_value:
                raise IsaacVerificationOutputBundleError(["isaac_output_robot_reference_invalid"])
            bindings.append((reference, str(digest_value)))

    trace_pair = result.get("articulated_policy_trace_pair")
    trace_pair = trace_pair if isinstance(trace_pair, Mapping) else {}
    candidate_traces = trace_pair.get("candidate_traces")
    candidate_traces = candidate_traces if isinstance(candidate_traces, list) else []
    for trace in candidate_traces:
        if not isinstance(trace, Mapping):
            raise IsaacVerificationOutputBundleError(
                ["isaac_output_policy_trace_reference_invalid"]
            )
        observation = trace.get("egocentric_observation")
        if observation is None:
            continue
        if not isinstance(observation, Mapping):
            raise IsaacVerificationOutputBundleError(
                ["isaac_output_policy_trace_reference_invalid"]
            )
        reference = _portable(observation.get("artifact_reference"))
        digest = str(observation.get("digest") or "")
        if reference is None or reference.suffix.lower() != ".png" or not digest:
            raise IsaacVerificationOutputBundleError(
                ["isaac_output_policy_trace_reference_invalid"]
            )
        bindings.append((reference, digest))

    references = [reference.as_posix() for reference, _digest in bindings]
    if len(references) != len(set(references)):
        raise IsaacVerificationOutputBundleError(["isaac_output_artifact_inventory_invalid"])
    return sorted(bindings, key=lambda row: row[0].as_posix())


def _artifact_rows(result: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference, expected_digest in _result_artifact_bindings(result):
        source = root.joinpath(*reference.parts)
        if source.is_symlink() or not source.is_file() or root not in source.resolve().parents:
            raise IsaacVerificationOutputBundleError(["isaac_output_artifact_invalid"])
        size = source.stat().st_size
        digest = _sha256(source)
        if size < 1 or size > MAX_MEMBER_BYTES or digest != expected_digest:
            raise IsaacVerificationOutputBundleError(["isaac_output_artifact_binding_invalid"])
        rows.append(
            {
                "artifact_id": reference.as_posix(),
                "archive_path": f"artifacts/{reference.as_posix()}",
                "digest": digest,
                "bytes": size,
                "source": source.resolve(),
            }
        )
    ids = [row["artifact_id"] for row in rows]
    if len(ids) != len(set(ids)) or sum(row["bytes"] for row in rows) > MAX_TOTAL_BYTES:
        raise IsaacVerificationOutputBundleError(["isaac_output_artifact_inventory_invalid"])
    return sorted(rows, key=lambda row: row["artifact_id"])


def _write_member(archive: zipfile.ZipFile, name: str, source: Path | bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    if isinstance(source, bytes):
        archive.writestr(info, source)
        return
    with (
        source.open("rb") as input_stream,
        archive.open(info, "w", force_zip64=True) as output_stream,
    ):
        for chunk in iter(lambda: input_stream.read(STREAM_CHUNK_BYTES), b""):
            output_stream.write(chunk)


def compile_isaac_verification_output_bundle(
    *,
    bundle_receipt: Mapping[str, Any],
    runtime_output_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Atomically compile runtime evidence without qualifying compatibility."""

    receipt = validate_isaac_verification_worker_bundle_receipt(bundle_receipt)
    lexical_root = Path(runtime_output_root)
    if lexical_root.is_symlink():
        raise IsaacVerificationOutputBundleError(["isaac_output_runtime_root_symlink_forbidden"])
    try:
        root = lexical_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacVerificationOutputBundleError(["isaac_output_runtime_root_missing"]) from exc
    provider_bundle = receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
    request_member = str(receipt.get("verification_request_member") or "")
    request_path: Path | None = None
    verification_request: Mapping[str, Any] | None = None
    if provider_bundle:
        request_path = root.parent / "bundle" / request_member
        try:
            request_value = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IsaacVerificationOutputBundleError(
                ["isaac_output_verification_request_missing"]
            ) from exc
        if not isinstance(request_value, Mapping):
            raise IsaacVerificationOutputBundleError(["isaac_output_verification_request_invalid"])
        verification_request = request_value
    result = _load_runtime_result(
        root / "isaac_runtime_result.json",
        receipt=receipt,
        verification_request=verification_request,
    )
    _result_bindings(result=result, receipt=receipt)
    rows = _artifact_rows(result, root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "compiled",
        "isaac_verification_request_digest": receipt["isaac_verification_request_digest"],
        "input_bundle_digest": receipt["bundle_digest"],
        "package_digest": receipt["package_digest"],
        "runtime_container_image_digest": receipt["runtime_container_image_digest"],
        "source_commit_sha": receipt["source_commit_sha"],
        "runtime_result_schema": result["schema_version"],
        "runtime_result_digest": result["isaac_runtime_result_digest"],
        "runtime_result_archive_path": "isaac_runtime_result.json",
        "verification_request_archive_path": request_member if provider_bundle else None,
        "artifact_members": [
            {key: row[key] for key in ("artifact_id", "archive_path", "digest", "bytes")}
            for row in rows
        ],
        "artifact_member_count": len(rows),
        "artifact_total_bytes": sum(row["bytes"] for row in rows),
        "raw_secret_values_included": False,
        "scientific_qualification_inferred": False,
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": "none",
        "claim_ceiling": "unaccepted_isaac_runtime_transport_only",
    }
    manifest["output_manifest_digest"] = canonical_digest(
        manifest, digest_field="output_manifest_digest"
    )
    destination = Path(output_path)
    if destination.is_symlink():
        raise IsaacVerificationOutputBundleError(["isaac_output_bundle_symlink_forbidden"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    result_bytes = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            _write_member(archive, "output_manifest.json", manifest_bytes)
            _write_member(archive, "isaac_runtime_result.json", result_bytes)
            if provider_bundle and request_path is not None:
                _write_member(archive, request_member, request_path)
            for row in rows:
                _write_member(archive, row["archive_path"], row["source"])
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        **manifest,
        "output_bundle_digest": _sha256(destination),
        "output_bundle_bytes": destination.stat().st_size,
        "cost_usd": 0.0,
    }


def _read_capped(archive: zipfile.ZipFile, member: zipfile.ZipInfo, *, cap: int) -> bytes:
    if member.file_size > cap:
        raise IsaacVerificationOutputBundleError(["isaac_output_metadata_oversized"])
    payload = bytearray()
    with archive.open(member, "r") as stream:
        while True:
            chunk = stream.read(min(STREAM_CHUNK_BYTES, cap + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > cap:
                raise IsaacVerificationOutputBundleError(["isaac_output_metadata_oversized"])
    return bytes(payload)


def _extract_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(member, "r") as input_stream, destination.open("wb") as output:
        while True:
            chunk = input_stream.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_MEMBER_BYTES:
                raise IsaacVerificationOutputBundleError(["isaac_output_member_oversized"])
            digest.update(chunk)
            output.write(chunk)
    return size, "sha256:" + digest.hexdigest()


def validate_and_extract_isaac_verification_output_bundle(
    *,
    bundle_path: str | Path,
    expected_input_receipt: Mapping[str, Any],
    expected_source_commit_sha: str,
    output_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Independently validate and materialize a retrieved Isaac output bundle."""

    input_receipt = validate_isaac_verification_worker_bundle_receipt(expected_input_receipt)
    source = Path(bundle_path)
    if source.is_symlink():
        raise IsaacVerificationOutputBundleError(["isaac_output_bundle_symlink_forbidden"])
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacVerificationOutputBundleError(["isaac_output_bundle_missing"]) from exc
    if not source.is_file() or source.stat().st_size > MAX_TOTAL_BYTES + 64 * 1024**2:
        raise IsaacVerificationOutputBundleError(["isaac_output_bundle_invalid"])
    bundle_digest = _sha256(source)
    destination = Path(output_root)
    if destination.is_symlink():
        raise IsaacVerificationOutputBundleError(["isaac_output_extraction_root_symlink_forbidden"])
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    final = destination / bundle_digest[7:]
    if final.exists():
        raise IsaacVerificationOutputBundleError(
            ["isaac_output_extraction_existing_requires_recorded_replay"]
        )
    temporary = Path(tempfile.mkdtemp(prefix=".isaac-output-", dir=destination))
    try:
        try:
            archive_context = zipfile.ZipFile(source, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise IsaacVerificationOutputBundleError(["isaac_output_bundle_invalid"]) from exc
        with archive_context as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            errors: list[str] = []
            if len(names) != len(set(names)):
                errors.append("isaac_output_inventory_duplicate")
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
                    errors.append("isaac_output_member_unsafe")
                total += member.file_size
            if total > MAX_TOTAL_BYTES + 2 * MAX_METADATA_BYTES:
                errors.append("isaac_output_total_oversized")
            if errors:
                raise IsaacVerificationOutputBundleError(errors)
            try:
                manifest_value = json.loads(
                    _read_capped(
                        archive,
                        archive.getinfo("output_manifest.json"),
                        cap=MAX_METADATA_BYTES,
                    )
                )
                result_value = json.loads(
                    _read_capped(
                        archive,
                        archive.getinfo("isaac_runtime_result.json"),
                        cap=MAX_METADATA_BYTES,
                    )
                )
            except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise IsaacVerificationOutputBundleError(["isaac_output_metadata_invalid"]) from exc
            if not isinstance(manifest_value, Mapping) or not isinstance(result_value, Mapping):
                raise IsaacVerificationOutputBundleError(["isaac_output_metadata_invalid"])
            manifest = dict(manifest_value)
            provider_bundle = (
                input_receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
            )
            request_member = str(input_receipt.get("verification_request_member") or "")
            request_value: Mapping[str, Any] | None = None
            if provider_bundle:
                try:
                    loaded_request = json.loads(
                        _read_capped(
                            archive,
                            archive.getinfo(request_member),
                            cap=MAX_METADATA_BYTES,
                        )
                    )
                except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise IsaacVerificationOutputBundleError(
                        ["isaac_output_verification_request_invalid"]
                    ) from exc
                if not isinstance(loaded_request, Mapping):
                    raise IsaacVerificationOutputBundleError(
                        ["isaac_output_verification_request_invalid"]
                    )
                request_value = loaded_request
            try:
                if provider_bundle:
                    assert request_value is not None
                    result = build_provider_nurec_isaac_runtime_result(
                        result_value, verification_request=request_value
                    )
                else:
                    result = build_isaac_runtime_result_v3(result_value)
            except (ExternalProviderNuRecError, IsaacReconstructionVerificationError) as exc:
                raise IsaacVerificationOutputBundleError(
                    ["isaac_output_runtime_result_invalid"]
                ) from exc
            _result_bindings(result=result, receipt=input_receipt)
            if (
                manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("status") != "compiled"
                or manifest.get("output_manifest_digest")
                != canonical_digest(manifest, digest_field="output_manifest_digest")
                or manifest.get("isaac_verification_request_digest")
                != input_receipt["isaac_verification_request_digest"]
                or manifest.get("input_bundle_digest") != input_receipt["bundle_digest"]
                or manifest.get("package_digest") != input_receipt["package_digest"]
                or manifest.get("runtime_container_image_digest")
                != input_receipt["runtime_container_image_digest"]
                or manifest.get("source_commit_sha") != expected_source_commit_sha
                or manifest.get("runtime_result_digest") != result["isaac_runtime_result_digest"]
                or manifest.get("verification_request_archive_path")
                != (request_member if provider_bundle else None)
                or manifest.get("raw_secret_values_included") is not False
                or manifest.get("scientific_qualification_inferred") is not False
                or manifest.get("simulator_task_success_proven") is not False
                or manifest.get("physical_success_proven") is not False
                or manifest.get("deployment_readiness_proven") is not False
                or manifest.get("proof_effect") != "none"
            ):
                raise IsaacVerificationOutputBundleError(["isaac_output_manifest_binding_invalid"])
            rows = manifest.get("artifact_members")
            if not isinstance(rows, list):
                raise IsaacVerificationOutputBundleError(["isaac_output_inventory_invalid"])
            expected_names = {
                "output_manifest.json",
                "isaac_runtime_result.json",
                *(str(row.get("archive_path") or "") for row in rows if isinstance(row, Mapping)),
            }
            if provider_bundle:
                expected_names.add(request_member)
            if set(names) != expected_names or manifest.get("artifact_member_count") != len(rows):
                raise IsaacVerificationOutputBundleError(["isaac_output_inventory_invalid"])
            result_bindings = [
                (reference.as_posix(), digest)
                for reference, digest in _result_artifact_bindings(result)
            ]
            manifest_bindings: list[tuple[str, str]] = []
            total_artifacts = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    raise IsaacVerificationOutputBundleError(
                        ["isaac_output_manifest_member_invalid"]
                    )
                artifact_id = _portable(row.get("artifact_id"))
                archive_path = _portable(row.get("archive_path"))
                if (
                    artifact_id is None
                    or archive_path is None
                    or archive_path.as_posix() != f"artifacts/{artifact_id.as_posix()}"
                ):
                    raise IsaacVerificationOutputBundleError(
                        ["isaac_output_manifest_member_invalid"]
                    )
                target = temporary.joinpath(*artifact_id.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                size, digest = _extract_member(
                    archive, archive.getinfo(archive_path.as_posix()), target
                )
                if size != row.get("bytes") or digest != row.get("digest"):
                    raise IsaacVerificationOutputBundleError(
                        ["isaac_output_artifact_digest_mismatch"]
                    )
                total_artifacts += size
                manifest_bindings.append((artifact_id.as_posix(), digest))
            if (
                sorted(manifest_bindings) != result_bindings
                or manifest.get("artifact_total_bytes") != total_artifacts
            ):
                raise IsaacVerificationOutputBundleError(["isaac_output_result_artifacts_mismatch"])
        write_result = temporary / "isaac_runtime_result.json"
        write_result.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        receipt = {
            **manifest,
            "status": "validated",
            "output_bundle_digest": bundle_digest,
            "output_bundle_bytes": source.stat().st_size,
            "cost_usd": 0.0,
        }
        receipt["output_bundle_receipt_digest"] = canonical_digest(
            receipt, digest_field="output_bundle_receipt_digest"
        )
        (temporary / "validated_output_bundle_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, final)
        return receipt, result, final
    except Exception:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "IsaacVerificationOutputBundleError",
    "SCHEMA_VERSION",
    "compile_isaac_verification_output_bundle",
    "validate_and_extract_isaac_verification_output_bundle",
]
