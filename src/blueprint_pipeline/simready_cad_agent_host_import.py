"""Rebase digest-bound CAD-agent candidates onto a host-resident evidence root.

The importer copies every bound source byte before changing only the absolute
file records needed by the existing CAD contracts.  STEP and generator bytes
are never regenerated or rewritten.  JSON receipt digests are recomputed only
by explicit schema adapters; arbitrary JSON is never traversed or rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any, Callable

from .cad_agent_review_media import (
    SCHEMA_VERSION as CAD_REVIEW_MEDIA_SCHEMA_VERSION,
    VISUAL_REVIEW_SCHEMA_VERSION,
    CadAgentReviewMediaError,
    materialize_cad_agent_visual_comparison,
    seal_cad_agent_visual_reference_review,
    validate_cad_agent_visual_reference_review,
)
from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError,
    validate_task_freeze,
)
from .simready_cad_agent_contract import (
    ADMITTED_BACKENDS,
    CLAIM_BOUNDARY,
    EXECUTION_SCHEMA_VERSION,
    INSPECTION_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    REFERENCE_MANIFEST_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    SimReadyCadAgentContractError,
    validate_cad_agent_execution_receipt,
    validate_cad_agent_output,
    validate_cad_agent_reference_manifest,
    validate_cad_agent_request,
    validate_step_inspection_receipt,
    seal_cad_agent_matrix,
)


IMPORT_SCHEMA_VERSION = "simready_cad_agent_host_import.v1"
IMPORT_DIGEST_FIELD = "import_digest"
REVIEW_IMPORT_SCHEMA_VERSION = "simready_cad_visual_review_host_rematerialization.v1"
REVIEW_IMPORT_DIGEST_FIELD = "rematerialization_digest"
MAX_BOUND_FILES = 256
MAX_BOUND_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BOUND_BYTES = 512 * 1024 * 1024
HOST_FILE_MODE = 0o640
HOST_DIRECTORY_MODE = 0o750
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class SimReadyCadAgentHostImportError(ValueError):
    """One source binding or imported receipt failed closed."""


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SimReadyCadAgentHostImportError(code) from exc
    if not isinstance(result, dict):
        raise SimReadyCadAgentHostImportError(code)
    return result


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(value).encode("utf-8") + b"\n"


def _record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SimReadyCadAgentHostImportError("cad_host_import_output_file_invalid")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _ownership_policy(owner_uid: int, owner_gid: int) -> tuple[int, int]:
    if (
        not isinstance(owner_uid, int)
        or isinstance(owner_uid, bool)
        or owner_uid < 0
        or not isinstance(owner_gid, int)
        or isinstance(owner_gid, bool)
        or owner_gid < 0
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_ownership_policy_invalid"
        )
    return owner_uid, owner_gid


def _directories_for_files(files: Sequence[Path], roots: Sequence[Path]) -> set[Path]:
    directories = {root for root in roots}
    for path in files:
        for root in roots:
            if _is_inside(path, root):
                cursor = path.parent
                while _is_inside(cursor, root):
                    directories.add(cursor)
                    if cursor == root:
                        break
                    cursor = cursor.parent
                break
    return directories


def _seal_ownership_and_readback(
    *,
    files: Sequence[Path],
    roots: Sequence[Path],
    owner_uid: int,
    owner_gid: int,
) -> tuple[int, int]:
    owner_uid, owner_gid = _ownership_policy(owner_uid, owner_gid)
    unique_files = sorted(set(files), key=str)
    directories = sorted(_directories_for_files(unique_files, roots), key=str)
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_ownership_directory_invalid"
            )
        os.chmod(directory, HOST_DIRECTORY_MODE)
        metadata = directory.stat()
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            try:
                os.chown(directory, owner_uid, owner_gid)
            except PermissionError as exc:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_ownership_chown_failed"
                ) from exc
    for path in unique_files:
        if path.is_symlink() or not path.is_file():
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_ownership_file_invalid"
            )
        os.chmod(path, HOST_FILE_MODE)
        metadata = path.stat()
        if metadata.st_uid != owner_uid or metadata.st_gid != owner_gid:
            try:
                os.chown(path, owner_uid, owner_gid)
            except PermissionError as exc:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_ownership_chown_failed"
                ) from exc
    if os.geteuid() == owner_uid:
        try:
            for path in unique_files:
                with path.open("rb") as stream:
                    stream.read(1)
        except OSError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_service_account_readback_failed"
            ) from exc
    elif os.geteuid() == 0 and hasattr(os, "fork"):
        child = os.fork()
        if child == 0:  # pragma: no cover - exercised only by root host install
            try:
                os.setgroups([owner_gid])
                os.setgid(owner_gid)
                os.setuid(owner_uid)
                for path in unique_files:
                    with path.open("rb") as stream:
                        stream.read(1)
            except Exception:
                os._exit(1)
            os._exit(0)
        _pid, status = os.waitpid(child, 0)
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_service_account_readback_failed"
            )
    else:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_service_account_readback_unavailable"
        )
    return len(unique_files), len(directories)


def _structural_record(value: Any, *, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SimReadyCadAgentHostImportError(code)
    path_text = str(value.get("path") or "")
    size = value.get("size_bytes")
    digest = str(value.get("sha256") or "")
    if (
        not path_text
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or _DIGEST.fullmatch(digest) is None
        or "\x00" in path_text
    ):
        raise SimReadyCadAgentHostImportError(code)
    path = PurePath(path_text)
    if not path.is_absolute() and ".." in path.parts:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_path_traversal"
        )
    if ".." in path.parts:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_path_traversal"
        )
    return dict(value)


def _safe_basename(path_text: str, *, fallback: str) -> str:
    name = PurePath(path_text).name or fallback
    if (
        name in {".", ".."}
        or "/" in name
        or "\\" in name
        or not name.isprintable()
        or len(name.encode("utf-8")) > 240
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_basename_invalid"
        )
    return name


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _contains_symlink_below_root(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


class _ImportPlan:
    """Plan source-bound copies fully in memory before mutating the destination."""

    def __init__(
        self,
        *,
        destination_root: Path,
        source_prefix_mappings: Sequence[tuple[PurePath, Path]],
        source_overrides: Mapping[str, Path],
    ) -> None:
        self.destination_root = destination_root
        self.source_prefix_mappings = tuple(source_prefix_mappings)
        self.source_overrides = dict(source_overrides)
        self.files: dict[Path, bytes] = {}
        self.bindings: list[dict[str, Any]] = []
        self.total_source_bytes = 0
        self.source_record_paths: set[str] = set()
        self._leaf_cache: dict[tuple[str, int, str], dict[str, Any]] = {}
        self._json_cache: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}

    def _candidate_path(self, record: Mapping[str, Any], *, role: str) -> Path:
        path_text = str(record["path"])
        declared = PurePath(path_text)
        candidates: list[Path] = []
        if declared.is_absolute():
            matches = [
                (prefix, root)
                for prefix, root in self.source_prefix_mappings
                if declared == prefix or prefix in declared.parents
            ]
            if len(matches) != 1:
                raise SimReadyCadAgentHostImportError(
                    f"cad_host_import_source_prefix_unmapped_or_ambiguous:{role}"
                )
            prefix, root = matches[0]
            relative = declared.relative_to(prefix)
            candidate = root.joinpath(*relative.parts)
            resolved = candidate.resolve(strict=False)
            if not _is_inside(resolved, root):
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_source_path_traversal"
                )
            if _contains_symlink_below_root(candidate, root):
                raise SimReadyCadAgentHostImportError(
                    f"cad_host_import_source_symlink_or_missing:{role}"
                )
            if candidate.exists() or candidate.is_symlink():
                candidates.append(candidate)
        else:
            for _prefix, root in self.source_prefix_mappings:
                candidate = root.joinpath(*declared.parts)
                resolved = candidate.resolve(strict=False)
                if not _is_inside(resolved, root):
                    raise SimReadyCadAgentHostImportError(
                        "cad_host_import_source_path_traversal"
                    )
                if _contains_symlink_below_root(candidate, root):
                    raise SimReadyCadAgentHostImportError(
                        f"cad_host_import_source_symlink_or_missing:{role}"
                    )
                if candidate.exists() or candidate.is_symlink():
                    candidates.append(candidate)
        override = self.source_overrides.get(str(record["sha256"]))
        if not candidates and override is not None:
            override_roots = [
                root
                for _prefix, root in self.source_prefix_mappings
                if _is_inside(override, root)
            ]
            if (
                len(override_roots) != 1
                or _contains_symlink_below_root(override, override_roots[0])
            ):
                raise SimReadyCadAgentHostImportError(
                    f"cad_host_import_source_override_outside_mapping:{role}"
                )
            candidates.append(override)
        if len(candidates) != 1:
            raise SimReadyCadAgentHostImportError(
                f"cad_host_import_source_unavailable:{role}"
            )
        path = candidates[0]
        if path.is_symlink() or not path.is_file():
            raise SimReadyCadAgentHostImportError(
                f"cad_host_import_source_symlink_or_missing:{role}"
            )
        return path

    def source_bytes(self, value: Any, *, role: str) -> tuple[dict[str, Any], bytes]:
        record = _structural_record(
            value, code=f"cad_host_import_source_record_invalid:{role}"
        )
        self.source_record_paths.add(str(record["path"]))
        path = self._candidate_path(record, role=role)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise SimReadyCadAgentHostImportError(
                f"cad_host_import_source_unreadable:{role}"
            ) from exc
        if (
            len(payload) != record["size_bytes"]
            or _sha256_bytes(payload) != record["sha256"]
        ):
            raise SimReadyCadAgentHostImportError(
                f"cad_host_import_source_digest_drift:{role}"
            )
        if len(payload) > MAX_BOUND_FILE_BYTES:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_source_file_too_large"
            )
        return record, payload

    def _reserve(self, *, relative: Path, payload: bytes) -> Path:
        destination = self.destination_root / relative
        if not _is_inside(destination, self.destination_root):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_destination_path_traversal"
            )
        previous = self.files.get(destination)
        if previous is not None and previous != payload:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_destination_collision"
            )
        if previous is None:
            if len(self.files) >= MAX_BOUND_FILES:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_file_count_exceeded"
                )
            self.total_source_bytes += len(payload)
            if self.total_source_bytes > MAX_TOTAL_BOUND_BYTES:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_total_bytes_exceeded"
                )
            self.files[destination] = payload
        return destination

    def leaf(self, value: Any, *, role: str) -> dict[str, Any]:
        source, payload = self.source_bytes(value, role=role)
        basename = _safe_basename(str(source["path"]), fallback="artifact.bin")
        digest_hex = str(source["sha256"]).split(":", 1)[1]
        cache_key = (str(source["sha256"]), int(source["size_bytes"]), basename)
        cached = self._leaf_cache.get(cache_key)
        if cached is None:
            destination = self._reserve(
                relative=Path("objects") / "sha256" / digest_hex[:2] / digest_hex / basename,
                payload=payload,
            )
            cached = {
                **{
                    key: item
                    for key, item in source.items()
                    if key not in {"path", "sha256", "size_bytes"}
                },
                "path": str(destination),
                "sha256": source["sha256"],
                "size_bytes": source["size_bytes"],
            }
            self._leaf_cache[cache_key] = cached
        self.bindings.append(
            {
                "role": role,
                "source_sha256": source["sha256"],
                "source_size_bytes": source["size_bytes"],
                "output": dict(cached),
                "source_bytes_preserved_exactly": True,
                "metadata_paths_rewritten_only": False,
            }
        )
        return dict(cached)

    def json_receipt(
        self,
        value: Any,
        *,
        role: str,
        schema_version: str,
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source, payload = self.source_bytes(value, role=role)
        cache_key = (str(source["sha256"]), role)
        cached = self._json_cache.get(cache_key)
        if cached is not None:
            record, transformed = cached
            return dict(record), _clone(transformed, code="cad_host_import_cache_invalid")
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SimReadyCadAgentHostImportError(
                f"cad_host_import_json_invalid:{role}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("schema_version") != schema_version:
            raise SimReadyCadAgentHostImportError(
                f"cad_host_import_json_schema_invalid:{role}"
            )
        transformed = transform(_clone(raw, code=f"cad_host_import_json_invalid:{role}"))
        output_payload = _json_bytes(transformed)
        source_hex = str(source["sha256"]).split(":", 1)[1]
        destination = self._reserve(
            relative=Path("receipts") / schema_version / f"{source_hex}.json",
            payload=output_payload,
        )
        record = {
            "path": str(destination),
            "sha256": _sha256_bytes(output_payload),
            "size_bytes": len(output_payload),
        }
        self.bindings.append(
            {
                "role": role,
                "source_sha256": source["sha256"],
                "source_size_bytes": source["size_bytes"],
                "output": dict(record),
                "source_bytes_preserved_exactly": output_payload == payload,
                "metadata_paths_rewritten_only": output_payload != payload,
                "source_schema_version": schema_version,
            }
        )
        self._json_cache[cache_key] = (dict(record), _clone(transformed, code="cad_host_import_cache_invalid"))
        return record, transformed


def _task_freeze(plan: _ImportPlan, record: Any, *, role: str) -> dict[str, Any]:
    source, payload = plan.source_bytes(record, role=role)
    try:
        freeze = json.loads(payload.decode("utf-8"))
        validate_task_freeze(freeze)
    except (UnicodeDecodeError, json.JSONDecodeError, DualTaskRehearsalContractError) as exc:
        raise SimReadyCadAgentHostImportError(
            f"cad_host_import_task_freeze_invalid:{role}"
        ) from exc
    return plan.leaf(source, role=role)


def _reference_manifest(
    plan: _ImportPlan, record: Any, *, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    def transform(manifest: dict[str, Any]) -> dict[str, Any]:
        source_manifest = _clone(
            manifest, code="cad_host_import_reference_manifest_invalid"
        )
        try:
            validate_cad_agent_reference_manifest(manifest, verify_files=False)
        except SimReadyCadAgentContractError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_reference_manifest_invalid"
            ) from exc
        objects = manifest.get("objects")
        if not isinstance(objects, list):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_reference_manifest_invalid"
            )
        for index, row in enumerate(objects):
            if not isinstance(row, dict):
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_reference_manifest_invalid"
                )
            row["task_freeze"] = _task_freeze(
                plan,
                row.get("task_freeze"),
                role=f"{role}.objects[{index}].task_freeze",
            )
            references = row.get("reference_images")
            if not isinstance(references, list):
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_reference_manifest_invalid"
                )
            row["reference_images"] = [
                plan.leaf(
                    item,
                    role=f"{role}.objects[{index}].reference_images[{ref_index}]",
                )
                for ref_index, item in enumerate(references)
            ]
        manifest["manifest_digest"] = canonical_digest(
            manifest, digest_field="manifest_digest"
        )
        _assert_all_absolute_paths_were_explicitly_bound(source_manifest, plan)
        try:
            return validate_cad_agent_reference_manifest(
                manifest, verify_files=False
            )
        except SimReadyCadAgentContractError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_reference_manifest_invalid"
            ) from exc

    return plan.json_receipt(
        record,
        role=role,
        schema_version=REFERENCE_MANIFEST_SCHEMA_VERSION,
        transform=transform,
    )


def _selected_reference_object(
    manifest: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("objects") or []
        if isinstance(row, Mapping)
        and row.get("replacement_slot") == request.get("replacement_slot")
        and row.get("task_id") == request.get("task_id")
        and row.get("asset_id") == request.get("asset_id")
        and manifest.get("scene_id") == request.get("scene_id")
    ]
    if len(matches) != 1:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_reference_manifest_join_invalid"
        )
    return dict(matches[0])


def _inspection_receipt(
    plan: _ImportPlan, record: Any, *, role: str
) -> dict[str, Any]:
    def transform(receipt: dict[str, Any]) -> dict[str, Any]:
        source_receipt = _clone(
            receipt, code="cad_host_import_inspection_receipt_invalid"
        )
        try:
            validate_step_inspection_receipt(receipt, verify_files=False)
        except SimReadyCadAgentContractError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_inspection_receipt_invalid"
            ) from exc
        receipt["step"] = plan.leaf(receipt.get("step"), role=f"{role}.step")
        inspector = receipt.get("inspector")
        if not isinstance(inspector, dict):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_inspection_receipt_invalid"
            )
        inspector["module_source"] = plan.leaf(
            inspector.get("module_source"), role=f"{role}.inspector.module_source"
        )
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        _assert_all_absolute_paths_were_explicitly_bound(source_receipt, plan)
        try:
            return validate_step_inspection_receipt(receipt, verify_files=False)
        except SimReadyCadAgentContractError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_inspection_receipt_invalid"
            ) from exc

    imported_record, _ = plan.json_receipt(
        record,
        role=role,
        schema_version=INSPECTION_SCHEMA_VERSION,
        transform=transform,
    )
    return imported_record


def _execution_receipt(
    plan: _ImportPlan,
    record: Any,
    *,
    role: str,
    source_request_digest: str,
    imported_request_digest: str,
) -> dict[str, Any]:
    def transform(receipt: dict[str, Any]) -> dict[str, Any]:
        source_receipt = _clone(
            receipt, code="cad_host_import_execution_receipt_invalid"
        )
        try:
            validate_cad_agent_execution_receipt(receipt, verify_files=False)
        except SimReadyCadAgentContractError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_execution_receipt_invalid"
            ) from exc
        if receipt.get("request_digest") != source_request_digest:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_execution_request_join_invalid"
            )
        for field in ("generator_source", "cad_brief", "output_step"):
            receipt[field] = plan.leaf(
                receipt.get(field), role=f"{role}.{field}"
            )
        receipt["request_digest"] = imported_request_digest
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        _assert_all_absolute_paths_were_explicitly_bound(source_receipt, plan)
        try:
            return validate_cad_agent_execution_receipt(
                receipt, verify_files=False
            )
        except SimReadyCadAgentContractError as exc:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_execution_receipt_invalid"
            ) from exc

    imported_record, _ = plan.json_receipt(
        record,
        role=role,
        schema_version=EXECUTION_SCHEMA_VERSION,
        transform=transform,
    )
    return imported_record


def _import_request(plan: _ImportPlan, source: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(source, code="cad_host_import_request_invalid")
    if (
        request.get("schema_version") != REQUEST_SCHEMA_VERSION
        or request.get("claim_boundary") != CLAIM_BOUNDARY
        or request.get("request_digest")
        != canonical_digest(request, digest_field="request_digest")
    ):
        raise SimReadyCadAgentHostImportError("cad_host_import_request_invalid")
    backend = request.get("backend")
    inputs = request.get("inputs")
    if not isinstance(backend, dict) or not isinstance(inputs, dict):
        raise SimReadyCadAgentHostImportError("cad_host_import_request_invalid")
    backend["source_archive"] = plan.leaf(
        backend.get("source_archive"), role="request.backend.source_archive"
    )
    inputs["cad_brief"] = plan.leaf(
        inputs.get("cad_brief"), role="request.inputs.cad_brief"
    )
    inputs["task_freeze"] = _task_freeze(
        plan, inputs.get("task_freeze"), role="request.inputs.task_freeze"
    )
    references = inputs.get("reference_images")
    if not isinstance(references, list):
        raise SimReadyCadAgentHostImportError("cad_host_import_request_invalid")
    inputs["reference_images"] = [
        plan.leaf(item, role=f"request.inputs.reference_images[{index}]")
        for index, item in enumerate(references)
    ]
    imported_manifest_record, imported_manifest = _reference_manifest(
        plan,
        inputs.get("reference_manifest"),
        role="request.inputs.reference_manifest",
    )
    inputs["reference_manifest"] = imported_manifest_record
    selected = _selected_reference_object(imported_manifest, request)
    inputs["reference_manifest_object_digest"] = canonical_digest(selected)
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    try:
        return validate_cad_agent_request(request, verify_files=False)
    except SimReadyCadAgentContractError as exc:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_request_invalid"
        ) from exc


def _source_receipt_record(payload: bytes, receipt_digest: str) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "receipt_digest": receipt_digest,
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "source_path_recorded": False,
    }


def _record_identity(value: Any) -> tuple[str, int, str] | None:
    if not isinstance(value, Mapping):
        return None
    path = str(value.get("path") or "")
    size = value.get("size_bytes")
    digest = str(value.get("sha256") or "")
    if (
        not path
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or _DIGEST.fullmatch(digest) is None
    ):
        return None
    return (path, size, digest)


def _read_bound_json(record: Any, *, code: str) -> dict[str, Any]:
    identity = _record_identity(record)
    if identity is None:
        raise SimReadyCadAgentHostImportError(code)
    path = Path(identity[0])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyCadAgentHostImportError(code) from exc
    if not isinstance(value, dict):
        raise SimReadyCadAgentHostImportError(code)
    return value


def _expected_import_bindings(imported: Mapping[str, Any]) -> dict[str, Any]:
    request = imported.get("request")
    artifacts = imported.get("artifacts")
    execution = imported.get("execution")
    if not all(isinstance(item, Mapping) for item in (request, artifacts, execution)):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_binding_coverage_invalid"
        )
    assert isinstance(request, Mapping)
    assert isinstance(artifacts, Mapping)
    assert isinstance(execution, Mapping)
    backend = request.get("backend")
    inputs = request.get("inputs")
    if not isinstance(backend, Mapping) or not isinstance(inputs, Mapping):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_binding_coverage_invalid"
        )
    expected: dict[str, Any] = {
        "request.backend.source_archive": backend.get("source_archive"),
        "request.inputs.cad_brief": inputs.get("cad_brief"),
        "request.inputs.task_freeze": inputs.get("task_freeze"),
        "request.inputs.reference_manifest": inputs.get("reference_manifest"),
        "artifacts.generator_source": artifacts.get("generator_source"),
        "artifacts.step": artifacts.get("step"),
        "artifacts.inspection_receipt": artifacts.get("inspection_receipt"),
        "execution.execution_receipt": execution.get("execution_receipt"),
    }
    references = inputs.get("reference_images")
    snapshots = artifacts.get("snapshots")
    if not isinstance(references, list) or not isinstance(snapshots, list):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_binding_coverage_invalid"
        )
    expected.update(
        {
            f"request.inputs.reference_images[{index}]": row
            for index, row in enumerate(references)
        }
    )
    expected.update(
        {
            f"artifacts.snapshots[{index}]": row
            for index, row in enumerate(snapshots)
        }
    )
    manifest = _read_bound_json(
        inputs.get("reference_manifest"),
        code="cad_host_import_binding_coverage_invalid",
    )
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_binding_coverage_invalid"
        )
    for index, row in enumerate(objects):
        if not isinstance(row, Mapping) or not isinstance(
            row.get("reference_images"), list
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_binding_coverage_invalid"
            )
        expected[f"request.inputs.reference_manifest.objects[{index}].task_freeze"] = (
            row.get("task_freeze")
        )
        expected.update(
            {
                (
                    f"request.inputs.reference_manifest.objects[{index}]"
                    f".reference_images[{ref_index}]"
                ): reference
                for ref_index, reference in enumerate(row["reference_images"])
            }
        )
    inspection = _read_bound_json(
        artifacts.get("inspection_receipt"),
        code="cad_host_import_binding_coverage_invalid",
    )
    inspector = inspection.get("inspector")
    if not isinstance(inspector, Mapping):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_binding_coverage_invalid"
        )
    expected["artifacts.inspection_receipt.step"] = inspection.get("step")
    expected["artifacts.inspection_receipt.inspector.module_source"] = inspector.get(
        "module_source"
    )
    execution_receipt = _read_bound_json(
        execution.get("execution_receipt"),
        code="cad_host_import_binding_coverage_invalid",
    )
    for field in ("generator_source", "cad_brief", "output_step"):
        expected[f"execution.execution_receipt.{field}"] = execution_receipt.get(field)
    if any(_record_identity(record) is None for record in expected.values()):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_binding_coverage_invalid"
        )
    return expected


def _write_plan(plan: _ImportPlan) -> None:
    plan.destination_root.mkdir(mode=0o750, parents=True, exist_ok=True)
    for path, payload in sorted(plan.files.items(), key=lambda item: str(item[0])):
        path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_import_existing_artifact_mismatch"
                )
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o640)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.read_bytes() != payload
                ):
                    raise SimReadyCadAgentHostImportError(
                        "cad_host_import_existing_artifact_mismatch"
                    )
        finally:
            temporary_path.unlink(missing_ok=True)


def _normalize_source_prefix_mappings(
    values: Sequence[tuple[str, str | Path]],
) -> list[tuple[PurePath, Path]]:
    mappings: list[tuple[PurePath, Path]] = []
    for prefix_value, root_value in values:
        prefix = PurePath(str(prefix_value))
        unresolved_root = Path(root_value).expanduser()
        root = unresolved_root.resolve()
        if (
            not prefix.is_absolute()
            or ".." in prefix.parts
            or unresolved_root.is_symlink()
            or not root.is_dir()
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_source_prefix_mapping_invalid"
            )
        mappings.append((prefix, root))
    if not mappings:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_prefix_mapping_missing"
        )
    prefixes = [prefix for prefix, _root in mappings]
    if len(prefixes) != len(set(prefixes)) or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(prefixes)
        for right in prefixes[index + 1 :]
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_prefix_mapping_ambiguous"
        )
    return mappings


def _normalize_source_overrides(
    values: Mapping[str, str | Path] | None,
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for digest, path_value in (values or {}).items():
        if _DIGEST.fullmatch(str(digest)) is None:
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_source_override_invalid"
            )
        path = Path(path_value).expanduser()
        if path.is_symlink() or not path.is_file():
            raise SimReadyCadAgentHostImportError(
                "cad_host_import_source_override_invalid"
            )
        overrides[str(digest)] = path
    return overrides


def _assert_source_prefixes_absent(
    value: Mapping[str, Any], mappings: Sequence[tuple[PurePath, Path]]
) -> None:
    serialized = canonical_json(value)
    if any(str(prefix) in serialized for prefix, _root in mappings):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_prefix_retained"
        )


def _absolute_path_strings(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for item in value.values():
            result.update(_absolute_path_strings(item))
        return result
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(_absolute_path_strings(item))
        return result
    if isinstance(value, str) and (
        value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        return {value}
    return set()


def _assert_all_absolute_paths_were_explicitly_bound(
    value: Mapping[str, Any], plan: _ImportPlan
) -> None:
    unbound = _absolute_path_strings(value) - plan.source_record_paths
    if unbound:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_unadapted_absolute_path"
        )


def validate_simready_cad_agent_host_import(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    receipt = _clone(value, code="cad_host_import_receipt_invalid")
    errors: list[str] = []
    destination = Path(str(receipt.get("destination_root") or "")).expanduser()
    artifact_root = Path(str(receipt.get("artifact_root") or "")).expanduser()
    ownership_value = receipt.get("ownership")
    ownership = ownership_value if isinstance(ownership_value, Mapping) else {}
    imported_record = receipt.get("imported_cad_agent_output")
    source_record = receipt.get("source_cad_agent_output")
    bindings = receipt.get("bindings")
    expected_bindings: dict[str, Any] = {}
    if (
        receipt.get("schema_version") != IMPORT_SCHEMA_VERSION
        or receipt.get("status") != "host_resident_import_completed"
        or not destination.is_absolute()
        or not artifact_root.is_absolute()
        or not isinstance(ownership_value, Mapping)
        or not isinstance(ownership.get("owner_uid"), int)
        or isinstance(ownership.get("owner_uid"), bool)
        or ownership.get("owner_uid", -1) < 0
        or not isinstance(ownership.get("owner_gid"), int)
        or isinstance(ownership.get("owner_gid"), bool)
        or ownership.get("owner_gid", -1) < 0
        or ownership.get("file_mode") != "0640"
        or ownership.get("directory_mode") != "0750"
        or ownership.get("service_account_readback_passed") is not True
        or not isinstance(ownership.get("readback_file_count"), int)
        or ownership.get("readback_file_count", 0) < 1
        or not isinstance(source_record, Mapping)
        or source_record.get("schema_version") != OUTPUT_SCHEMA_VERSION
        or source_record.get("receipt_digest") != receipt.get("source_receipt_digest")
        or _DIGEST.fullmatch(str(source_record.get("receipt_digest") or "")) is None
        or _DIGEST.fullmatch(str(source_record.get("sha256") or "")) is None
        or not isinstance(source_record.get("size_bytes"), int)
        or isinstance(source_record.get("size_bytes"), bool)
        or source_record.get("size_bytes", 0) <= 0
        or source_record.get("source_path_recorded") is not False
        or not isinstance(bindings, list)
        or not bindings
        or receipt.get("binding_count") != len(bindings)
        or receipt.get("source_paths_retained") is not False
        or receipt.get("source_prefixes_retained") is not False
        or not isinstance(receipt.get("source_prefix_mapping_count"), int)
        or isinstance(receipt.get("source_prefix_mapping_count"), bool)
        or receipt.get("source_prefix_mapping_count", 0) < 1
        or receipt.get("geometry_bytes_modified") is not False
        or receipt.get("geometry_generated") is not False
        or receipt.get("source_step_sha256") != receipt.get("imported_step_sha256")
        or _DIGEST.fullmatch(str(receipt.get("source_step_sha256") or "")) is None
        or receipt.get("claim_boundary")
        != {
            "host_resident_copy_only": True,
            "agent_authored_cad_candidate": True,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        }
    ):
        errors.append("cad_host_import_receipt_invalid")
    if verify_files and any(
        root.is_symlink()
        or not root.is_dir()
        or root.stat().st_uid != ownership.get("owner_uid")
        or root.stat().st_gid != ownership.get("owner_gid")
        or stat.S_IMODE(root.stat().st_mode) != HOST_DIRECTORY_MODE
        for root in (destination, artifact_root)
    ):
        errors.append("cad_host_import_ownership_directory_invalid")
    if verify_files:
        receipt_path = destination / f"{IMPORT_SCHEMA_VERSION}.json"
        try:
            persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("cad_host_import_persisted_receipt_invalid")
        else:
            if (
                persisted_receipt != receipt
                or receipt_path.is_symlink()
                or receipt_path.stat().st_uid != ownership.get("owner_uid")
                or receipt_path.stat().st_gid != ownership.get("owner_gid")
                or stat.S_IMODE(receipt_path.stat().st_mode) != HOST_FILE_MODE
            ):
                errors.append("cad_host_import_persisted_receipt_invalid")
    if not isinstance(imported_record, Mapping):
        errors.append("cad_host_import_output_record_invalid")
    else:
        output_path = Path(str(imported_record.get("path") or ""))
        if (
            not output_path.is_absolute()
            or not _is_inside(output_path, destination)
            or (verify_files and (
                output_path.is_symlink()
                or not output_path.is_file()
                or output_path.stat().st_size != imported_record.get("size_bytes")
                or _sha256(output_path) != imported_record.get("sha256")
                or output_path.stat().st_uid != ownership.get("owner_uid")
                or output_path.stat().st_gid != ownership.get("owner_gid")
                or stat.S_IMODE(output_path.stat().st_mode) != HOST_FILE_MODE
            ))
        ):
            errors.append("cad_host_import_output_record_invalid")
        elif verify_files:
            try:
                imported = json.loads(output_path.read_text(encoding="utf-8"))
                validated = validate_cad_agent_output(imported, verify_files=True)
            except (
                OSError,
                json.JSONDecodeError,
                SimReadyCadAgentContractError,
            ):
                errors.append("cad_host_import_output_invalid")
            else:
                if (
                    validated.get("receipt_digest")
                    != receipt.get("imported_receipt_digest")
                    or (validated.get("artifacts") or {}).get("step", {}).get("sha256")
                    != receipt.get("source_step_sha256")
                ):
                    errors.append("cad_host_import_output_join_invalid")
                try:
                    expected_bindings = _expected_import_bindings(validated)
                except SimReadyCadAgentHostImportError:
                    errors.append("cad_host_import_binding_coverage_invalid")
    if isinstance(bindings, list):
        roles: set[str] = set()
        observed_bindings: dict[str, tuple[str, int, str]] = {}
        for row in bindings:
            if not isinstance(row, Mapping):
                errors.append("cad_host_import_binding_invalid")
                continue
            role = str(row.get("role") or "")
            source_sha = str(row.get("source_sha256") or "")
            source_size = row.get("source_size_bytes")
            output = row.get("output")
            if (
                not role
                or role in roles
                or _DIGEST.fullmatch(source_sha) is None
                or not isinstance(source_size, int)
                or isinstance(source_size, bool)
                or source_size <= 0
                or not isinstance(row.get("source_bytes_preserved_exactly"), bool)
                or not isinstance(row.get("metadata_paths_rewritten_only"), bool)
                or row.get("source_bytes_preserved_exactly")
                == row.get("metadata_paths_rewritten_only")
                or not isinstance(output, Mapping)
            ):
                errors.append("cad_host_import_binding_invalid")
                continue
            roles.add(role)
            identity = _record_identity(output)
            if identity is not None:
                observed_bindings[role] = identity
            output_path = Path(str(output.get("path") or ""))
            if (
                not output_path.is_absolute()
                or not _is_inside(output_path, artifact_root)
                or _DIGEST.fullmatch(str(output.get("sha256") or "")) is None
                or not isinstance(output.get("size_bytes"), int)
                or isinstance(output.get("size_bytes"), bool)
                or output.get("size_bytes", 0) <= 0
                or (
                    verify_files
                    and (
                        output_path.is_symlink()
                        or not output_path.is_file()
                        or output_path.stat().st_size != output.get("size_bytes")
                        or _sha256(output_path) != output.get("sha256")
                        or output_path.stat().st_uid != ownership.get("owner_uid")
                        or output_path.stat().st_gid != ownership.get("owner_gid")
                        or stat.S_IMODE(output_path.stat().st_mode) != HOST_FILE_MODE
                    )
                )
                or (
                    row.get("source_bytes_preserved_exactly") is True
                    and (
                        source_sha != output.get("sha256")
                        or source_size != output.get("size_bytes")
                    )
                )
            ):
                errors.append("cad_host_import_binding_invalid")
        expected_identities = {
            role: _record_identity(record)
            for role, record in expected_bindings.items()
        }
        if not expected_bindings or observed_bindings != expected_identities:
            errors.append("cad_host_import_binding_coverage_invalid")
    if receipt.get(IMPORT_DIGEST_FIELD) != canonical_digest(
        receipt, digest_field=IMPORT_DIGEST_FIELD
    ):
        errors.append("cad_host_import_digest_invalid")
    if errors:
        raise SimReadyCadAgentHostImportError(";".join(sorted(set(errors))))
    return receipt


def materialize_simready_cad_agent_host_import(
    *,
    source_receipt_path: str | Path,
    destination_root: str | Path,
    source_prefix_mappings: Sequence[tuple[str, str | Path]],
    owner_uid: int,
    owner_gid: int,
    source_overrides: Mapping[str, str | Path] | None = None,
    artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Import one authored CAD candidate without changing any geometry bytes."""

    owner_uid, owner_gid = _ownership_policy(owner_uid, owner_gid)
    source_path = Path(source_receipt_path).expanduser()
    unresolved_destination = Path(destination_root).expanduser()
    destination = unresolved_destination.resolve()
    unresolved_artifact_root = (
        Path(artifact_root).expanduser()
        if artifact_root is not None
        else unresolved_destination / "artifacts"
    )
    artifact_root_path = (
        unresolved_artifact_root.resolve()
    )
    if source_path.is_symlink() or not source_path.is_file():
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_receipt_invalid"
        )
    if destination.exists() or unresolved_destination.is_symlink():
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_destination_exists"
        )
    if unresolved_artifact_root.is_symlink() or (
        artifact_root_path.exists() and not artifact_root_path.is_dir()
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_artifact_root_invalid"
        )
    mappings = _normalize_source_prefix_mappings(source_prefix_mappings)
    overrides = _normalize_source_overrides(source_overrides)
    try:
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_receipt_invalid"
        ) from exc
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != OUTPUT_SCHEMA_VERSION
        or source.get("status") != "candidate_authored"
        or source.get("claim_boundary") != CLAIM_BOUNDARY
        or source.get("receipt_digest")
        != canonical_digest(source, digest_field="receipt_digest")
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_receipt_invalid"
        )
    plan = _ImportPlan(
        destination_root=artifact_root_path,
        source_prefix_mappings=mappings,
        source_overrides=overrides,
    )
    source_request = source.get("request")
    if not isinstance(source_request, Mapping):
        raise SimReadyCadAgentHostImportError("cad_host_import_request_invalid")
    imported_request = _import_request(plan, source_request)
    source_artifacts = source.get("artifacts")
    execution = source.get("execution")
    if not isinstance(source_artifacts, Mapping) or not isinstance(execution, Mapping):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_source_receipt_invalid"
        )
    imported = _clone(source, code="cad_host_import_source_receipt_invalid")
    imported["request"] = imported_request
    imported["request_digest"] = imported_request["request_digest"]
    imported_artifacts = imported["artifacts"]
    imported_artifacts["generator_source"] = plan.leaf(
        source_artifacts.get("generator_source"), role="artifacts.generator_source"
    )
    imported_artifacts["step"] = plan.leaf(
        source_artifacts.get("step"), role="artifacts.step"
    )
    imported_artifacts["inspection_receipt"] = _inspection_receipt(
        plan,
        source_artifacts.get("inspection_receipt"),
        role="artifacts.inspection_receipt",
    )
    snapshots = source_artifacts.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_snapshots_invalid"
        )
    imported_artifacts["snapshots"] = [
        plan.leaf(item, role=f"artifacts.snapshots[{index}]")
        for index, item in enumerate(snapshots)
    ]
    imported_execution = imported["execution"]
    imported_execution["execution_receipt"] = _execution_receipt(
        plan,
        execution.get("execution_receipt"),
        role="execution.execution_receipt",
        source_request_digest=str(source_request.get("request_digest") or ""),
        imported_request_digest=imported_request["request_digest"],
    )
    _assert_all_absolute_paths_were_explicitly_bound(source, plan)
    imported["receipt_digest"] = canonical_digest(
        imported, digest_field="receipt_digest"
    )
    _assert_source_prefixes_absent(imported, mappings)
    output_payload = _json_bytes(imported)
    _write_plan(plan)
    destination.mkdir(mode=0o750, parents=True, exist_ok=True)
    output_path = destination / "cad_agent_output.v1.json"
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(output_payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        validated_output = validate_cad_agent_output(imported, verify_files=True)
    except SimReadyCadAgentContractError as exc:
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_output_validation_failed:" + ";".join(exc.codes)
        ) from exc
    source_step = source_artifacts.get("step") or {}
    receipt: dict[str, Any] = {
        "schema_version": IMPORT_SCHEMA_VERSION,
        "status": "host_resident_import_completed",
        "destination_root": str(destination),
        "artifact_root": str(artifact_root_path),
        "source_cad_agent_output": _source_receipt_record(
            source_bytes, str(source["receipt_digest"])
        ),
        "source_receipt_digest": source["receipt_digest"],
        "imported_cad_agent_output": _record(output_path),
        "imported_receipt_digest": validated_output["receipt_digest"],
        "source_step_sha256": source_step.get("sha256"),
        "imported_step_sha256": imported_artifacts["step"]["sha256"],
        "binding_count": len(plan.bindings),
        "bindings": sorted(plan.bindings, key=lambda row: row["role"]),
        "total_planned_bytes": (
            sum(len(payload) for payload in plan.files.values()) + len(output_payload)
        ),
        "source_paths_retained": False,
        "source_prefixes_retained": False,
        "source_prefix_mapping_count": len(mappings),
        "geometry_bytes_modified": False,
        "geometry_generated": False,
        "ownership": {},
        "claim_boundary": {
            "host_resident_copy_only": True,
            "agent_authored_cad_candidate": True,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        IMPORT_DIGEST_FIELD: "",
    }
    receipt_path = destination / f"{IMPORT_SCHEMA_VERSION}.json"
    sealed_files = [*plan.files, output_path, receipt_path]
    sealed_directories = _directories_for_files(
        sealed_files, (artifact_root_path, destination)
    )
    receipt["ownership"] = {
        "owner_uid": owner_uid,
        "owner_gid": owner_gid,
        "file_mode": "0640",
        "directory_mode": "0750",
        "service_account_readback_passed": True,
        "readback_file_count": len(sealed_files),
        "sealed_directory_count": len(sealed_directories),
    }
    receipt[IMPORT_DIGEST_FIELD] = canonical_digest(
        receipt, digest_field=IMPORT_DIGEST_FIELD
    )
    receipt_path.write_bytes(_json_bytes(receipt))
    observed_file_count, observed_directory_count = _seal_ownership_and_readback(
        files=sealed_files,
        roots=(artifact_root_path, destination),
        owner_uid=owner_uid,
        owner_gid=owner_gid,
    )
    if (
        observed_file_count != receipt["ownership"]["readback_file_count"]
        or observed_directory_count
        != receipt["ownership"]["sealed_directory_count"]
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_import_ownership_inventory_mismatch"
        )
    return validate_simready_cad_agent_host_import(receipt, verify_files=True)


def _load_json_file(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise SimReadyCadAgentHostImportError(code)
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimReadyCadAgentHostImportError(code) from exc
    if not isinstance(value, dict):
        raise SimReadyCadAgentHostImportError(code)
    return value, payload


def _source_review_catalog(
    *,
    plan: _ImportPlan,
    review: Mapping[str, Any],
) -> tuple[dict[tuple[int, str, str, str], dict[str, Any]], dict[str, Any]]:
    media_record = review.get("review_media")
    _source_record, media_bytes = plan.source_bytes(
        media_record, role="source_visual_review.review_media"
    )
    try:
        media = json.loads(media_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_media_invalid"
        ) from exc
    if (
        not isinstance(media, dict)
        or media.get("schema_version") != CAD_REVIEW_MEDIA_SCHEMA_VERSION
        or media.get("status") != "review_media_materialized"
        or media.get("receipt_digest")
        != canonical_digest(media, digest_field="receipt_digest")
        or review.get("review_media_digest") != media.get("receipt_digest")
        or review.get("scene_id") != media.get("scene_id")
        or media.get("claim_boundary")
        != {
            "human_review_media_only": True,
            "agent_authored_cad_candidate": True,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "appearance_qualified": False,
            "physics_qualified": False,
            "physical_equivalence": False,
        }
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_media_invalid"
        )
    for field in ("cad_matrix", "contact_sheet", "html"):
        plan.source_bytes(media.get(field), role=f"source_review_media.{field}")
    catalog: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    rows = media.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_media_invalid"
        )
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_source_media_invalid"
            )
        references = row.get("reference_images")
        candidates = row.get("candidates")
        if not isinstance(references, list) or not references or not isinstance(
            candidates, list
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_source_media_invalid"
            )
        for field in ("reference_manifest", "reference_thumbnail"):
            plan.source_bytes(
                row.get(field), role=f"source_review_media.rows[{row_index}].{field}"
            )
        verified_references: list[dict[str, Any]] = []
        for ref_index, reference in enumerate(references):
            source_record, _payload = plan.source_bytes(
                reference,
                role=f"source_review_media.rows[{row_index}].references[{ref_index}]",
            )
            verified_references.append(source_record)
        reference_signature = canonical_digest(
            {
                "reference_images": [
                    {
                        "sha256": item["sha256"],
                        "size_bytes": item["size_bytes"],
                    }
                    for item in verified_references
                ]
            }
        )
        if row.get("reference_signature") != reference_signature:
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_source_media_invalid"
            )
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise SimReadyCadAgentHostImportError(
                    "cad_host_review_source_media_invalid"
                )
            plan.source_bytes(
                candidate.get("snapshot"),
                role=(
                    f"source_review_media.rows[{row_index}]"
                    f".candidates[{candidate_index}].snapshot"
                ),
            )
            plan.source_bytes(
                candidate.get("step"),
                role=(
                    f"source_review_media.rows[{row_index}]"
                    f".candidates[{candidate_index}].step"
                ),
            )
            slot = row.get("replacement_slot")
            key = (
                slot if isinstance(slot, int) and not isinstance(slot, bool) else 0,
                str(row.get("task_id") or ""),
                str(row.get("asset_id") or ""),
                str(candidate.get("backend_id") or ""),
            )
            if key[0] < 1 or any(not item for item in key[1:]) or key in catalog:
                raise SimReadyCadAgentHostImportError(
                    "cad_host_review_source_media_invalid"
                )
            catalog[key] = {
                "source_output_receipt_digest": str(
                    candidate.get("output_receipt_digest") or ""
                ),
                "reference_signature": reference_signature,
                "reference_image_digests": [
                    item["sha256"] for item in verified_references
                ],
            }
    return catalog, media


def _read_imported_candidate(
    path_value: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = Path(path_value).expanduser().resolve()
    receipt, _payload = _load_json_file(
        path, code="cad_host_review_import_receipt_invalid"
    )
    validated_import = validate_simready_cad_agent_host_import(
        receipt, verify_files=True
    )
    output_path = Path(validated_import["imported_cad_agent_output"]["path"])
    output, _output_payload = _load_json_file(
        output_path, code="cad_host_review_imported_output_invalid"
    )
    try:
        validated_output = validate_cad_agent_output(output, verify_files=True)
    except SimReadyCadAgentContractError as exc:
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_imported_output_invalid"
        ) from exc
    return validated_import, validated_output, path


def _normalize_expected_candidates(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, str, str, str]] = set()
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {
            "replacement_slot",
            "task_id",
            "asset_id",
            "backend_id",
            "source_receipt_digest",
        }:
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_expected_candidate_invalid"
            )
        slot = value.get("replacement_slot")
        key = (
            slot if isinstance(slot, int) and not isinstance(slot, bool) else 0,
            str(value.get("task_id") or ""),
            str(value.get("asset_id") or ""),
            str(value.get("backend_id") or ""),
        )
        digest = str(value.get("source_receipt_digest") or "")
        if (
            key[0] < 1
            or any(not item for item in key[1:])
            or key in keys
            or _DIGEST.fullmatch(digest) is None
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_expected_candidate_invalid"
            )
        keys.add(key)
        rows.append(
            {
                "replacement_slot": key[0],
                "task_id": key[1],
                "asset_id": key[2],
                "backend_id": key[3],
                "source_receipt_digest": digest,
            }
        )
    object_backends: dict[tuple[int, str, str], set[str]] = {}
    for row in rows:
        object_key = (
            int(row["replacement_slot"]),
            str(row["task_id"]),
            str(row["asset_id"]),
        )
        object_backends.setdefault(object_key, set()).add(str(row["backend_id"]))
    if (
        len(rows) != 4
        or len(object_backends) != 2
        or any(backends != set(ADMITTED_BACKENDS) for backends in object_backends.values())
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_expected_four_candidate_set_invalid"
        )
    return sorted(
        rows,
        key=lambda row: (
            row["replacement_slot"],
            row["task_id"],
            row["asset_id"],
            row["backend_id"],
        ),
    )


def validate_cad_visual_review_host_rematerialization(
    value: Mapping[str, Any],
    *,
    expected_source_visual_review_digest: str,
    expected_source_visual_review_sha256: str,
    expected_source_visual_review_size_bytes: int,
    verify_files: bool = True,
) -> dict[str, Any]:
    receipt = _clone(value, code="cad_host_review_receipt_invalid")
    errors: list[str] = []
    destination = Path(str(receipt.get("destination_root") or ""))
    records = receipt.get("outputs")
    source_review_record = receipt.get("source_visual_review")
    import_receipt_records = receipt.get("cad_host_import_receipts")
    rebindings = receipt.get("candidate_digest_rebindings")
    ownership_value = receipt.get("ownership")
    ownership = ownership_value if isinstance(ownership_value, Mapping) else {}
    try:
        expected_candidates = _normalize_expected_candidates(
            receipt.get("expected_candidates") or []
        )
    except SimReadyCadAgentHostImportError:
        expected_candidates = []
    if (
        receipt.get("schema_version") != REVIEW_IMPORT_SCHEMA_VERSION
        or receipt.get("status") != "exhaustive_visual_review_rematerialized"
        or not destination.is_absolute()
        or not isinstance(ownership_value, Mapping)
        or not isinstance(ownership.get("owner_uid"), int)
        or isinstance(ownership.get("owner_uid"), bool)
        or ownership.get("owner_uid", -1) < 0
        or not isinstance(ownership.get("owner_gid"), int)
        or isinstance(ownership.get("owner_gid"), bool)
        or ownership.get("owner_gid", -1) < 0
        or ownership.get("file_mode") != "0640"
        or ownership.get("directory_mode") != "0750"
        or ownership.get("service_account_readback_passed") is not True
        or not isinstance(source_review_record, Mapping)
        or source_review_record.get("schema_version")
        != VISUAL_REVIEW_SCHEMA_VERSION
        or source_review_record.get("review_digest")
        != expected_source_visual_review_digest
        or source_review_record.get("sha256")
        != expected_source_visual_review_sha256
        or source_review_record.get("size_bytes")
        != expected_source_visual_review_size_bytes
        or _DIGEST.fullmatch(
            str(source_review_record.get("review_digest") or "")
        )
        is None
        or _DIGEST.fullmatch(str(source_review_record.get("sha256") or ""))
        is None
        or not isinstance(source_review_record.get("size_bytes"), int)
        or isinstance(source_review_record.get("size_bytes"), bool)
        or source_review_record.get("size_bytes", 0) <= 0
        or source_review_record.get("source_path_recorded") is not False
        or not isinstance(records, Mapping)
        or receipt.get("all_source_candidates_imported") is not True
        or receipt.get("selected_candidates_alone_sufficient") is not False
        or receipt.get("geometry_bytes_modified") is not False
        or receipt.get("geometry_generated") is not False
        or receipt.get("source_paths_retained") is not False
        or receipt.get("can_feed_seal_agent_cad_visual_binding") is not True
        or not isinstance(import_receipt_records, list)
        or not import_receipt_records
        or not isinstance(rebindings, list)
        or receipt.get("candidate_count") != len(rebindings)
        or receipt.get("candidate_count") != len(expected_candidates)
        or len(import_receipt_records) != len(rebindings)
        or receipt.get("claim_boundary")
        != {
            "prior_visual_decisions_rebound_to_exact_imported_candidates": True,
            "appearance_materially_qualified": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        }
    ):
        errors.append("cad_host_review_receipt_invalid")
    if verify_files and (
        destination.is_symlink()
        or not destination.is_dir()
        or destination.stat().st_uid != ownership.get("owner_uid")
        or destination.stat().st_gid != ownership.get("owner_gid")
        or stat.S_IMODE(destination.stat().st_mode) != HOST_DIRECTORY_MODE
    ):
        errors.append("cad_host_review_ownership_invalid")
    if verify_files:
        receipt_path = destination / f"{REVIEW_IMPORT_SCHEMA_VERSION}.json"
        try:
            persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("cad_host_review_persisted_receipt_invalid")
        else:
            if (
                persisted_receipt != receipt
                or receipt_path.is_symlink()
                or receipt_path.stat().st_uid != ownership.get("owner_uid")
                or receipt_path.stat().st_gid != ownership.get("owner_gid")
                or stat.S_IMODE(receipt_path.stat().st_mode) != HOST_FILE_MODE
            ):
                errors.append("cad_host_review_persisted_receipt_invalid")
    imported_digest_pairs: set[tuple[str, str]] = set()
    if isinstance(import_receipt_records, list):
        for record in import_receipt_records:
            if not isinstance(record, Mapping):
                errors.append("cad_host_review_import_receipt_invalid")
                continue
            path = Path(str(record.get("path") or ""))
            try:
                if (
                    not path.is_absolute()
                    or path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != record.get("size_bytes")
                    or _sha256(path) != record.get("sha256")
                    or path.stat().st_uid != ownership.get("owner_uid")
                    or path.stat().st_gid != ownership.get("owner_gid")
                    or stat.S_IMODE(path.stat().st_mode) != HOST_FILE_MODE
                ):
                    raise SimReadyCadAgentHostImportError("invalid")
                imported_receipt = json.loads(path.read_text(encoding="utf-8"))
                imported_receipt = validate_simready_cad_agent_host_import(
                    imported_receipt, verify_files=verify_files
                )
            except (
                OSError,
                json.JSONDecodeError,
                SimReadyCadAgentHostImportError,
            ):
                errors.append("cad_host_review_import_receipt_invalid")
            else:
                imported_digest_pairs.add(
                    (
                        imported_receipt["source_receipt_digest"],
                        imported_receipt["imported_receipt_digest"],
                    )
                )
    rebinding_pairs: set[tuple[str, str]] = set()
    rebinding_keys: set[tuple[int, str, str, str]] = set()
    rebinding_sources: dict[tuple[int, str, str, str], str] = {}
    if isinstance(rebindings, list):
        for row in rebindings:
            if not isinstance(row, Mapping):
                errors.append("cad_host_review_rebinding_invalid")
                continue
            key = (
                row.get("replacement_slot")
                if isinstance(row.get("replacement_slot"), int)
                and not isinstance(row.get("replacement_slot"), bool)
                else 0,
                str(row.get("task_id") or ""),
                str(row.get("asset_id") or ""),
                str(row.get("backend_id") or ""),
            )
            pair = (
                str(row.get("source_receipt_digest") or ""),
                str(row.get("imported_receipt_digest") or ""),
            )
            if (
                key[0] < 1
                or any(not item for item in key[1:])
                or key in rebinding_keys
                or _DIGEST.fullmatch(pair[0]) is None
                or _DIGEST.fullmatch(pair[1]) is None
                or pair in rebinding_pairs
            ):
                errors.append("cad_host_review_rebinding_invalid")
                continue
            rebinding_keys.add(key)
            rebinding_pairs.add(pair)
            rebinding_sources[key] = pair[0]
        if rebinding_pairs != imported_digest_pairs:
            errors.append("cad_host_review_rebinding_join_invalid")
        expected_sources = {
            (
                int(row["replacement_slot"]),
                str(row["task_id"]),
                str(row["asset_id"]),
                str(row["backend_id"]),
            ): str(row["source_receipt_digest"])
            for row in expected_candidates
        }
        if rebinding_sources != expected_sources:
            errors.append("cad_host_review_expected_candidate_join_invalid")
    if isinstance(records, Mapping):
        for role in ("cad_matrix", "review_media", "visual_review"):
            record = records.get(role)
            if not isinstance(record, Mapping):
                errors.append("cad_host_review_output_invalid")
                continue
            path = Path(str(record.get("path") or ""))
            if (
                not path.is_absolute()
                or not _is_inside(path, destination)
                or (
                    verify_files
                    and (
                        path.is_symlink()
                        or not path.is_file()
                        or path.stat().st_size != record.get("size_bytes")
                        or _sha256(path) != record.get("sha256")
                        or path.stat().st_uid != ownership.get("owner_uid")
                        or path.stat().st_gid != ownership.get("owner_gid")
                        or stat.S_IMODE(path.stat().st_mode) != HOST_FILE_MODE
                    )
                )
            ):
                errors.append("cad_host_review_output_invalid")
        if verify_files and isinstance(records.get("visual_review"), Mapping):
            review_path = Path(str(records["visual_review"].get("path") or ""))
            try:
                review = json.loads(review_path.read_text(encoding="utf-8"))
                admitted = validate_cad_agent_visual_reference_review(
                    review, verify_files=True
                )
            except (OSError, json.JSONDecodeError, CadAgentReviewMediaError):
                errors.append("cad_host_review_output_invalid")
            else:
                if admitted.get("review_digest") != receipt.get(
                    "visual_review_digest"
                ):
                    errors.append("cad_host_review_output_join_invalid")
                decision_rows = {
                    (
                        int(row["replacement_slot"]),
                        str(row["task_id"]),
                        str(row["asset_id"]),
                        str(row["backend_id"]),
                    ): str(row["cad_agent_output_receipt_digest"])
                    for row in admitted.get("candidate_decisions") or []
                }
                if set(decision_rows) != rebinding_keys or any(
                    decision_rows[key]
                    != next(
                        str(row["imported_receipt_digest"])
                        for row in rebindings
                        if (
                            int(row["replacement_slot"]),
                            str(row["task_id"]),
                            str(row["asset_id"]),
                            str(row["backend_id"]),
                        )
                        == key
                    )
                    for key in decision_rows
                ):
                    errors.append("cad_host_review_output_join_invalid")
        if verify_files:
            try:
                matrix = json.loads(
                    Path(str((records.get("cad_matrix") or {}).get("path") or ""))
                    .read_text(encoding="utf-8")
                )
                media = json.loads(
                    Path(str((records.get("review_media") or {}).get("path") or ""))
                    .read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                errors.append("cad_host_review_output_invalid")
            else:
                if (
                    matrix.get("matrix_digest") != receipt.get("cad_matrix_digest")
                    or media.get("receipt_digest")
                    != receipt.get("review_media_digest")
                ):
                    errors.append("cad_host_review_output_join_invalid")
    if receipt.get(REVIEW_IMPORT_DIGEST_FIELD) != canonical_digest(
        receipt, digest_field=REVIEW_IMPORT_DIGEST_FIELD
    ):
        errors.append("cad_host_review_digest_invalid")
    if errors:
        raise SimReadyCadAgentHostImportError(";".join(sorted(set(errors))))
    return receipt


def materialize_cad_visual_review_host_rematerialization(
    *,
    cad_host_import_receipt_paths: Sequence[str | Path],
    source_visual_review_path: str | Path,
    destination_root: str | Path,
    source_prefix_mappings: Sequence[tuple[str, str | Path]],
    expected_candidates: Sequence[Mapping[str, Any]],
    expected_source_visual_review_digest: str,
    expected_source_visual_review_sha256: str,
    expected_source_visual_review_size_bytes: int,
    owner_uid: int,
    owner_gid: int,
    source_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Rebuild exhaustive review media for an exact complete imported candidate set."""

    owner_uid, owner_gid = _ownership_policy(owner_uid, owner_gid)
    unresolved_destination = Path(destination_root).expanduser()
    destination = unresolved_destination.resolve()
    if destination.exists() or unresolved_destination.is_symlink():
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_destination_exists"
        )
    mappings = _normalize_source_prefix_mappings(source_prefix_mappings)
    normalized_expected = _normalize_expected_candidates(expected_candidates)
    expected_by_key = {
        (
            int(row["replacement_slot"]),
            str(row["task_id"]),
            str(row["asset_id"]),
            str(row["backend_id"]),
        ): str(row["source_receipt_digest"])
        for row in normalized_expected
    }
    overrides = _normalize_source_overrides(source_overrides)
    source_review_path = Path(source_visual_review_path).expanduser()
    source_review, source_review_bytes = _load_json_file(
        source_review_path, code="cad_host_review_source_review_invalid"
    )
    if (
        source_review.get("schema_version") != VISUAL_REVIEW_SCHEMA_VERSION
        or source_review.get("status") != "all_candidates_visually_reviewed"
        or source_review.get("review_digest")
        != canonical_digest(source_review, digest_field="review_digest")
        or source_review.get("claim_boundary")
        != {
            "all_manifest_bound_reference_images_reviewed": True,
            "candidate_visual_similarity_automatically_proven": False,
            "appearance_materially_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        }
        or source_review.get("review_digest")
        != expected_source_visual_review_digest
        or _sha256_bytes(source_review_bytes)
        != expected_source_visual_review_sha256
        or len(source_review_bytes) != expected_source_visual_review_size_bytes
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_review_invalid"
        )
    source_plan = _ImportPlan(
        destination_root=destination,
        source_prefix_mappings=mappings,
        source_overrides=overrides,
    )
    source_catalog, source_media = _source_review_catalog(
        plan=source_plan, review=source_review
    )
    _assert_all_absolute_paths_were_explicitly_bound(source_review, source_plan)
    _assert_all_absolute_paths_were_explicitly_bound(source_media, source_plan)
    source_catalog_digests = {
        key: str(row["source_output_receipt_digest"])
        for key, row in source_catalog.items()
    }
    if source_catalog_digests != expected_by_key:
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_expected_candidate_set_mismatch"
        )
    reference_sets: dict[tuple[int, str, str], list[str]] = {}
    for key, row in source_catalog.items():
        object_key = key[:3]
        references = list(row["reference_image_digests"])
        if object_key in reference_sets and reference_sets[object_key] != references:
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_source_media_invalid"
            )
        reference_sets[object_key] = references
    expected_reference_count = sum(len(row) for row in reference_sets.values())
    if (
        source_review.get("candidate_count") != len(source_catalog)
        or source_review.get("reviewed_reference_image_count")
        != expected_reference_count
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_review_invalid"
        )
    decisions = source_review.get("candidate_decisions")
    if not isinstance(decisions, list) or not decisions:
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_review_invalid"
        )
    source_decisions: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_source_review_invalid"
            )
        slot = decision.get("replacement_slot")
        key = (
            slot if isinstance(slot, int) and not isinstance(slot, bool) else 0,
            str(decision.get("task_id") or ""),
            str(decision.get("asset_id") or ""),
            str(decision.get("backend_id") or ""),
        )
        expected = source_catalog.get(key)
        if (
            expected is None
            or key in source_decisions
            or decision.get("cad_agent_output_receipt_digest")
            != expected["source_output_receipt_digest"]
            or decision.get("reference_signature")
            != expected["reference_signature"]
            or decision.get("reviewed_reference_image_digests")
            != expected["reference_image_digests"]
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_source_decision_invalid"
            )
        source_decisions[key] = dict(decision)
    if set(source_decisions) != set(source_catalog):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_source_decisions_incomplete"
        )

    imported_by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    imported_receipts: list[tuple[dict[str, Any], Path]] = []
    for path in cad_host_import_receipt_paths:
        imported_receipt, output, receipt_path = _read_imported_candidate(path)
        metadata = receipt_path.stat()
        if (
            metadata.st_uid != owner_uid
            or metadata.st_gid != owner_gid
            or stat.S_IMODE(metadata.st_mode) != HOST_FILE_MODE
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_import_receipt_ownership_invalid"
            )
        request = output["request"]
        key = (
            int(request["replacement_slot"]),
            str(request["task_id"]),
            str(request["asset_id"]),
            str(request["backend"]["backend_id"]),
        )
        if key in imported_by_key:
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_duplicate_imported_candidate"
            )
        if (
            key not in source_catalog
            or imported_receipt["source_receipt_digest"]
            != source_catalog[key]["source_output_receipt_digest"]
        ):
            raise SimReadyCadAgentHostImportError(
                "cad_host_review_imported_candidate_mismatch"
            )
        imported_by_key[key] = output
        imported_receipts.append((imported_receipt, receipt_path))
    if set(imported_by_key) != set(source_catalog):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_all_source_candidates_required"
        )

    objects: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for key, output in imported_by_key.items():
        object_key = key[:3]
        objects.setdefault(object_key, []).append(output)
    matrix = seal_cad_agent_matrix(
        objects=[
            {
                "replacement_slot": slot,
                "task_id": task_id,
                "asset_id": asset_id,
                "candidates": sorted(
                    candidates,
                    key=lambda item: item["request"]["backend"]["backend_id"],
                ),
            }
            for (slot, task_id, asset_id), candidates in sorted(objects.items())
        ]
    )
    destination.mkdir(mode=0o750, parents=True)
    matrix_path = destination / "cad_agent_matrix.v1.json"
    matrix_path.write_bytes(_json_bytes(matrix))
    media = materialize_cad_agent_visual_comparison(
        matrix_path=matrix_path,
        output_dir=destination / "review_media",
        title="Host-resident CAD-agent visual comparison",
    )
    remapped_decisions = []
    for key, decision in source_decisions.items():
        row = dict(decision)
        row["cad_agent_output_receipt_digest"] = imported_by_key[key][
            "receipt_digest"
        ]
        remapped_decisions.append(row)
    review_path = destination / "cad_agent_visual_reference_review.v1.json"
    review = seal_cad_agent_visual_reference_review(
        review_media_receipt_path=(
            destination / "review_media" / "cad_agent_visual_comparison.v1.json"
        ),
        reviewer=source_review["reviewer"],
        candidate_decisions=remapped_decisions,
        output_path=review_path,
    )
    _assert_source_prefixes_absent(matrix, mappings)
    _assert_source_prefixes_absent(media, mappings)
    _assert_source_prefixes_absent(review, mappings)
    receipt: dict[str, Any] = {
        "schema_version": REVIEW_IMPORT_SCHEMA_VERSION,
        "status": "exhaustive_visual_review_rematerialized",
        "destination_root": str(destination),
        "source_visual_review": {
            "schema_version": VISUAL_REVIEW_SCHEMA_VERSION,
            "review_digest": source_review["review_digest"],
            "sha256": _sha256_bytes(source_review_bytes),
            "size_bytes": len(source_review_bytes),
            "source_path_recorded": False,
        },
        "candidate_count": len(imported_by_key),
        "expected_candidates": normalized_expected,
        "candidate_digest_rebindings": [
            {
                "replacement_slot": key[0],
                "task_id": key[1],
                "asset_id": key[2],
                "backend_id": key[3],
                "source_receipt_digest": source_catalog[key][
                    "source_output_receipt_digest"
                ],
                "imported_receipt_digest": imported_by_key[key]["receipt_digest"],
            }
            for key in sorted(imported_by_key)
        ],
        "cad_host_import_receipts": [
            _record(path) for _receipt, path in sorted(imported_receipts, key=lambda item: str(item[1]))
        ],
        "outputs": {
            "cad_matrix": _record(matrix_path),
            "review_media": _record(
                destination / "review_media" / "cad_agent_visual_comparison.v1.json"
            ),
            "visual_review": _record(review_path),
        },
        "cad_matrix_digest": matrix["matrix_digest"],
        "review_media_digest": media["receipt_digest"],
        "visual_review_digest": review["review_digest"],
        "all_source_candidates_imported": True,
        "selected_candidates_alone_sufficient": False,
        "can_feed_seal_agent_cad_visual_binding": True,
        "source_paths_retained": False,
        "geometry_bytes_modified": False,
        "geometry_generated": False,
        "ownership": {},
        "claim_boundary": {
            "prior_visual_decisions_rebound_to_exact_imported_candidates": True,
            "appearance_materially_qualified": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "physical_equivalence": False,
        },
        REVIEW_IMPORT_DIGEST_FIELD: "",
    }
    receipt_path = destination / f"{REVIEW_IMPORT_SCHEMA_VERSION}.json"
    sealed_files = [
        path
        for path in destination.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    sealed_files.append(receipt_path)
    sealed_directories = _directories_for_files(sealed_files, (destination,))
    receipt["ownership"] = {
        "owner_uid": owner_uid,
        "owner_gid": owner_gid,
        "file_mode": "0640",
        "directory_mode": "0750",
        "service_account_readback_passed": True,
        "readback_file_count": len(sealed_files),
        "sealed_directory_count": len(sealed_directories),
    }
    receipt[REVIEW_IMPORT_DIGEST_FIELD] = canonical_digest(
        receipt, digest_field=REVIEW_IMPORT_DIGEST_FIELD
    )
    _assert_source_prefixes_absent(receipt, mappings)
    receipt_path.write_bytes(_json_bytes(receipt))
    observed_file_count, observed_directory_count = _seal_ownership_and_readback(
        files=sealed_files,
        roots=(destination,),
        owner_uid=owner_uid,
        owner_gid=owner_gid,
    )
    if (
        observed_file_count != receipt["ownership"]["readback_file_count"]
        or observed_directory_count
        != receipt["ownership"]["sealed_directory_count"]
    ):
        raise SimReadyCadAgentHostImportError(
            "cad_host_review_ownership_inventory_mismatch"
        )
    return validate_cad_visual_review_host_rematerialization(
        receipt,
        expected_source_visual_review_digest=expected_source_visual_review_digest,
        expected_source_visual_review_sha256=expected_source_visual_review_sha256,
        expected_source_visual_review_size_bytes=expected_source_visual_review_size_bytes,
        verify_files=True,
    )


__all__ = [
    "IMPORT_DIGEST_FIELD",
    "IMPORT_SCHEMA_VERSION",
    "REVIEW_IMPORT_DIGEST_FIELD",
    "REVIEW_IMPORT_SCHEMA_VERSION",
    "SimReadyCadAgentHostImportError",
    "materialize_cad_visual_review_host_rematerialization",
    "materialize_simready_cad_agent_host_import",
    "validate_cad_visual_review_host_rematerialization",
    "validate_simready_cad_agent_host_import",
]
