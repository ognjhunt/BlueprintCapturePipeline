"""Typed archive adapter from customer inputs to verified native Arena roots.

The website contract carries no production host paths.  It carries two
immutable, digest-bound archives instead: the construction packet and the
released runtime-source packet.  This adapter verifies their identity manifest,
extracts them without following paths or links, reopens every byte, and applies
the existing native-Arena validators.  It performs no profile publication,
standing authorization, allocator call, or provider mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import verify_native_task_arena_packet
from .native_task_runtime_source_packet import (
    verify_native_task_runtime_source_packet,
)
from .task_evaluation_launch_preparation_contract import (
    validate_launch_preparation_request,
)
from .task_evaluation_launch_preparation_queue import (
    write_launch_preparation_record_exclusive,
)


ADAPTER_KIND = "native_task_arena"
ADAPTER_VERSION = "v1"
MANIFEST_SCHEMA_VERSION = "task_evaluation_adapter_bundle_manifest.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_native_arena_adapter_result.v1"
MANIFEST_NAME = "task_evaluation_adapter_bundle_manifest.v1.json"
PAYLOAD_PREFIX = "payload/"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024 * 1024
_ROLES = frozenset({"construction_packet", "runtime_source"})


class TaskEvaluationNativeArenaAdapterError(RuntimeError):
    """A customer adapter archive could not be admitted without mutation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity_bindings(request: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene": dict(request["scene"]["identity"]),
        "robot": dict(request["robot"]["identity"]),
        "controller": dict(request["controller"]["identity"]),
        "task": dict(request["task"]["identity"]),
        "runtime": dict(request["runtime"]["identity"]),
    }


def _validated_relative_path(value: Any) -> PurePosixPath:
    text = str(value or "")
    relative = PurePosixPath(text)
    if (
        not text.startswith(PAYLOAD_PREFIX)
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.name in {"", ".", ".."}
    ):
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_member_path_invalid"
        )
    return relative


def _manifest_from_archive(
    archive: zipfile.ZipFile,
    *,
    request: Mapping[str, Any],
    expected_role: str,
) -> dict[str, Any]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)) or MANIFEST_NAME not in names:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_member_set_invalid"
        )
    for info in infos:
        mode = info.external_attr >> 16
        if (
            info.is_dir()
            or info.flag_bits & 0x1
            or stat.S_ISLNK(mode)
            or info.file_size < 0
        ):
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_bundle_member_unsafe"
            )
    manifest_info = archive.getinfo(MANIFEST_NAME)
    if manifest_info.file_size > MAX_MANIFEST_BYTES:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_manifest_too_large"
        )
    try:
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_manifest_invalid"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_manifest_invalid"
        )
    value = dict(manifest)
    if (
        value.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or value.get("adapter_kind") != ADAPTER_KIND
        or value.get("adapter_version") != ADAPTER_VERSION
        or value.get("bundle_role") != expected_role
        or value.get("identity_bindings") != _identity_bindings(request)
        or value.get("manifest_digest")
        != canonical_digest(value, digest_field="manifest_digest")
    ):
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_manifest_invalid"
        )
    rows = value.get("entries")
    if not isinstance(rows, list) or not rows:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_manifest_invalid"
        )
    expected_names = {MANIFEST_NAME}
    total_size = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_bundle_manifest_invalid"
            )
        relative = _validated_relative_path(row.get("relative_path"))
        name = relative.as_posix()
        if name in expected_names:
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_bundle_member_set_invalid"
            )
        expected_names.add(name)
        try:
            declared_size = int(row["size_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_bundle_manifest_invalid"
            ) from exc
        info = archive.getinfo(name)
        if (
            declared_size <= 0
            or declared_size != info.file_size
            or not isinstance(row.get("sha256"), str)
            or not str(row["sha256"]).startswith("sha256:")
        ):
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_bundle_member_identity_invalid"
            )
        total_size += declared_size
    disk_bound = min(
        MAX_UNCOMPRESSED_BYTES,
        int(float(request["runtime"]["requirements"]["disk_gib"]) * 1024**3),
    )
    if expected_names != set(names) or total_size > disk_bound:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_member_set_invalid"
        )
    return value


def _extract_verified_bundle(
    *,
    bundle_path: Path,
    request: Mapping[str, Any],
    expected_reference: Mapping[str, Any],
    role: str,
    destination: Path,
) -> tuple[dict[str, Any], Path]:
    if role not in _ROLES:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_role_invalid"
        )
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_source_invalid"
        )
    if (
        bundle_path.stat().st_size != expected_reference.get("size_bytes")
        or _sha256_file(bundle_path) != expected_reference.get("digest")
    ):
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_source_identity_mismatch"
        )
    try:
        archive = zipfile.ZipFile(bundle_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_archive_invalid"
        ) from exc
    with archive:
        manifest = _manifest_from_archive(
            archive, request=request, expected_role=role
        )
        destination.mkdir(parents=True, exist_ok=False, mode=0o750)
        try:
            for row in manifest["entries"]:
                relative = _validated_relative_path(row["relative_path"])
                target = destination.joinpath(*relative.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
                descriptor = os.open(
                    target,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o440,
                )
                try:
                    digest = hashlib.sha256()
                    size = 0
                    with archive.open(relative.as_posix(), "r") as source:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            view = memoryview(chunk)
                            while view:
                                written = os.write(descriptor, view)
                                if written <= 0:
                                    raise OSError("short adapter bundle write")
                                view = view[written:]
                            digest.update(chunk)
                            size += len(chunk)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                if (
                    size != row["size_bytes"]
                    or "sha256:" + digest.hexdigest() != row["sha256"]
                ):
                    raise TaskEvaluationNativeArenaAdapterError(
                        "task_evaluation_adapter_bundle_member_readback_mismatch"
                    )
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    return manifest, destination


def materialize_native_arena_adapter(
    *,
    request: Mapping[str, Any],
    construction_bundle_path: str | Path,
    runtime_source_bundle_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Extract and independently verify both native-Arena adapter bundles."""

    validated = validate_launch_preparation_request(request)
    adapter = validated["execution_adapter"]
    if (adapter["kind"], adapter["version"]) != (ADAPTER_KIND, ADAPTER_VERSION):
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_execution_adapter_unavailable"
        )
    root = Path(output_root).expanduser()
    if root.exists() or root.is_symlink():
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_output_exists"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o750)
    try:
        construction_manifest, packet_root = _extract_verified_bundle(
            bundle_path=Path(construction_bundle_path).expanduser(),
            request=validated,
            expected_reference=adapter["construction_packet_bundle"],
            role="construction_packet",
            destination=root / "construction-packet",
        )
        runtime_manifest, runtime_root = _extract_verified_bundle(
            bundle_path=Path(runtime_source_bundle_path).expanduser(),
            request=validated,
            expected_reference=adapter["runtime_source_bundle"],
            role="runtime_source",
            destination=root / "runtime-source",
        )
        _packet_path, packet_receipt, _packet_rows = (
            verify_native_task_arena_packet(packet_root)
        )
        if (
            packet_receipt.get("scene_id") != validated["scene"]["identity"]["id"]
            or packet_receipt.get("task_id")
            != validated["task"]["identity"]["id"]
        ):
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_packet_identity_mismatch"
            )
        runtime_receipt_path = (
            runtime_root / "native_task_runtime_source_packet.v1.json"
        )
        try:
            runtime_receipt = json.loads(
                runtime_receipt_path.read_text(encoding="utf-8")
            )
            runtime_packet_name = Path(
                str(runtime_receipt.get("packet_path") or "")
            ).name
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_runtime_source_invalid"
            ) from exc
        if not runtime_packet_name:
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_runtime_source_invalid"
            )
        verified_runtime = verify_native_task_runtime_source_packet(
            runtime_receipt_path,
            packet_path_override=runtime_root / runtime_packet_name,
        )
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "native_arena_adapter_materialized",
            "preparation_id": validated["preparation_id"],
            "source_commit": validated["expected_production_commit"],
            "adapter_kind": ADAPTER_KIND,
            "adapter_version": ADAPTER_VERSION,
            "construction_manifest_digest": construction_manifest[
                "manifest_digest"
            ],
            "runtime_source_manifest_digest": runtime_manifest["manifest_digest"],
            "packet_receipt_digest": packet_receipt["receipt_digest"],
            "runtime_source_receipt_digest": verified_runtime["receipt_digest"],
            "packet_root": str(packet_root),
            "runtime_source_receipt": str(runtime_receipt_path),
            "provider_mutation_performed": False,
            "catalog_mutation_performed": False,
            "paid_execution_requested": False,
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(
            result, digest_field="result_digest"
        )
        write_launch_preparation_record_exclusive(
            root / "task_evaluation_native_arena_adapter_result.v1.json", result
        )
        return result
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def build_task_evaluation_adapter_bundle(
    *,
    source_root: str | Path,
    output_path: str | Path,
    request: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    """Build deterministic customer-uploadable bytes for one adapter role."""

    validated = validate_launch_preparation_request(request)
    if role not in _ROLES:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_role_invalid"
        )
    raw_source = Path(source_root).expanduser()
    output = Path(output_path).expanduser()
    if raw_source.is_symlink() or output.is_symlink():
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_build_input_invalid"
        )
    source = raw_source.resolve()
    if not source.is_dir() or output.exists():
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_build_input_invalid"
        )
    rows: list[dict[str, Any]] = []
    sources: list[tuple[str, Path]] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise TaskEvaluationNativeArenaAdapterError(
                "task_evaluation_adapter_bundle_build_input_invalid"
            )
        relative = path.relative_to(source).as_posix()
        archive_path = f"{PAYLOAD_PREFIX}{relative}"
        rows.append(
            {
                "relative_path": archive_path,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        sources.append((archive_path, path))
    if not rows:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_build_input_invalid"
        )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "adapter_kind": ADAPTER_KIND,
        "adapter_version": ADAPTER_VERSION,
        "bundle_role": role,
        "identity_bindings": _identity_bindings(validated),
        "entries": rows,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            manifest_bytes = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            manifest_info = zipfile.ZipInfo(MANIFEST_NAME, (1980, 1, 1, 0, 0, 0))
            manifest_info.external_attr = 0o100440 << 16
            manifest_info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(manifest_info, manifest_bytes)
            for archive_path, path in sources:
                info = zipfile.ZipInfo(archive_path, (1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100440 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                with archive.open(info, "w") as destination, path.open(
                    "rb"
                ) as source_stream:
                    shutil.copyfileobj(
                        source_stream, destination, length=1024 * 1024
                    )
        os.chmod(temporary, 0o440)
        os.link(temporary, output, follow_symlinks=False)
        directory = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise TaskEvaluationNativeArenaAdapterError(
            "task_evaluation_adapter_bundle_output_conflict"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": "task_evaluation_adapter_bundle_build_receipt.v1",
        "status": "built",
        "role": role,
        "path": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256_file(output),
        "manifest_digest": manifest["manifest_digest"],
    }


__all__ = [
    "ADAPTER_KIND",
    "ADAPTER_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "TaskEvaluationNativeArenaAdapterError",
    "build_task_evaluation_adapter_bundle",
    "materialize_native_arena_adapter",
]
