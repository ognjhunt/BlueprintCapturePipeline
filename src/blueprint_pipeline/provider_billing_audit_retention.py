"""Content-address and safely deduplicate retained provider billing responses.

Provider billing source receipts bind an absolute ``retained_path`` as well as
the exact response digest and size.  Those receipt bytes are evidence and must
not be rewritten.  This module therefore keeps every receipt-local path while
making equal response paths hard links to one digest-addressed object.

Historical conversion is deliberately two phase: a dry run records every
receipt and response inode, and apply requires that exact plan, an explicit
acknowledgement, an exclusive audit-root lock, and a byte-for-byte rescan before
the first relink.  Ambiguity blocks the entire operation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


BILLING_SOURCE_SCHEMA_VERSION = "blueprint.provider_billing_source_receipt.v1"
PLAN_SCHEMA_VERSION = "blueprint.provider_billing_audit_retention_plan.v1"
APPLY_SCHEMA_VERSION = "blueprint.provider_billing_audit_retention_apply.v1"
APPLY_ACKNOWLEDGEMENT = "deduplicate-provider-billing-audit"
OBJECT_DIRECTORY = "objects"
OBJECT_MODE = 0o600
DIRECTORY_MODE = 0o700

_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})")
_AUDIT_DIRECTORY_RE = re.compile(r"[0-9]{8}T[0-9]{6}\.[0-9]{6}Z")
_RESPONSE_NAME_RE = re.compile(r"response-[0-9]{3}-[a-z0-9_-]+\.json")


class ProviderBillingAuditRetentionError(RuntimeError):
    """The billing response retention boundary could not be proven safe."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    body = dict(value)
    body.pop(digest_field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(body)).hexdigest()


def _payload_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _absolute_root(value: str | Path) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_root_must_be_absolute"
        )
    return root


def _assert_directory(path: Path, *, blocker: str) -> os.stat_result:
    if path.is_symlink() or not path.is_dir():
        raise ProviderBillingAuditRetentionError(blocker)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ProviderBillingAuditRetentionError(blocker)
    return info


@contextmanager
def audit_root_lock(
    audit_root: str | Path, *, create: bool = False
) -> Iterator[tuple[Path, os.stat_result]]:
    """Hold an exclusive, file-free lock on the audit-root directory inode."""

    root = _absolute_root(audit_root)
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=DIRECTORY_MODE)
    root_info = _assert_directory(
        root, blocker="provider_billing_audit_root_invalid"
    )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as exc:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_root_lock_unavailable"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (root_info.st_dev, root_info.st_ino):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_root_changed_before_lock"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        current = root.stat()
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_root_changed_after_lock"
            )
        yield root.resolve(), current
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _ensure_owned_directory(
    path: Path, *, audit_device: int, owner_uid: int, owner_gid: int
) -> None:
    created = not os.path.lexists(path)
    if created:
        path.mkdir(mode=DIRECTORY_MODE)
        if os.geteuid() == 0:
            os.chown(path, owner_uid, owner_gid)
        os.chmod(path, DIRECTORY_MODE)
    info = _assert_directory(
        path, blocker="provider_billing_audit_object_directory_invalid"
    )
    if (
        info.st_dev != audit_device
        or info.st_uid != owner_uid
        or info.st_gid != owner_gid
        or stat.S_IMODE(info.st_mode) != DIRECTORY_MODE
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_object_directory_metadata_invalid"
        )


def _validate_regular_file(
    path: Path,
    *,
    expected_digest: str,
    expected_size: int,
    audit_device: int,
    owner_uid: int,
    owner_gid: int,
    expected_mode: int = OBJECT_MODE,
) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_file_invalid"
        )
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_dev != audit_device
        or info.st_uid != owner_uid
        or info.st_gid != owner_gid
        or stat.S_IMODE(info.st_mode) != expected_mode
        or info.st_size != expected_size
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_metadata_invalid"
        )
    try:
        observed = _payload_digest(path.read_bytes())
    except OSError as exc:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_unreadable"
        ) from exc
    if observed != expected_digest:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_digest_mismatch"
        )
    return info


def _validate_retained_file(
    path: Path,
    *,
    expected_digest: str,
    expected_size: int,
    audit_device: int,
    audit_uid: int,
    audit_gid: int,
) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_file_invalid"
        )
    info = path.stat()
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_dev != audit_device
        or info.st_uid not in {0, audit_uid}
        or info.st_gid not in {0, audit_gid}
        or mode not in {0o440, 0o600}
        or mode & (stat.S_IWGRP | stat.S_IWOTH)
        or info.st_size != expected_size
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_metadata_invalid"
        )
    try:
        observed = _payload_digest(path.read_bytes())
    except OSError as exc:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_unreadable"
        ) from exc
    if observed != expected_digest:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_digest_mismatch"
        )
    return info


def _object_path(audit_root: Path, digest: str) -> Path:
    match = _DIGEST_RE.fullmatch(digest)
    if match is None:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_digest_invalid"
        )
    hexadecimal = match.group(1)
    return audit_root / OBJECT_DIRECTORY / "sha256" / hexadecimal[:2] / hexadecimal


def _publish_object(
    *, audit_root: Path, root_info: os.stat_result, payload: bytes
) -> tuple[Path, str]:
    digest = _payload_digest(payload)
    object_path = _object_path(audit_root, digest)
    current = audit_root
    for component in (OBJECT_DIRECTORY, "sha256", object_path.parent.name):
        current = current / component
        _ensure_owned_directory(
            current,
            audit_device=root_info.st_dev,
            owner_uid=root_info.st_uid,
            owner_gid=root_info.st_gid,
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{object_path.name}.", dir=object_path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, OBJECT_MODE)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, object_path, follow_symlinks=False)
        except FileExistsError:
            pass
        _validate_regular_file(
            object_path,
            expected_digest=digest,
            expected_size=len(payload),
            audit_device=root_info.st_dev,
            owner_uid=root_info.st_uid,
            owner_gid=root_info.st_gid,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    return object_path, digest


def retain_billing_response(
    *,
    audit_root: Path,
    audit_directory: Path,
    response_name: str,
    payload: bytes,
    root_info: os.stat_result,
) -> Path:
    """Publish one receipt-local hard link to an exact digest object.

    The caller must hold :func:`audit_root_lock` for ``audit_root``.
    """

    if not _RESPONSE_NAME_RE.fullmatch(response_name):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_name_invalid"
        )
    resolved_root = audit_root.resolve()
    if audit_directory.parent.resolve() != resolved_root:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_directory_outside_root"
        )
    directory_info = _assert_directory(
        audit_directory, blocker="provider_billing_audit_directory_invalid"
    )
    if (
        directory_info.st_dev != root_info.st_dev
        or directory_info.st_uid != root_info.st_uid
        or directory_info.st_gid != root_info.st_gid
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_directory_metadata_invalid"
        )
    response_path = audit_directory / response_name
    if os.path.lexists(response_path):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_response_path_exists"
        )
    object_path, digest = _publish_object(
        audit_root=resolved_root, root_info=root_info, payload=payload
    )
    os.link(object_path, response_path, follow_symlinks=False)
    _validate_regular_file(
        response_path,
        expected_digest=digest,
        expected_size=len(payload),
        audit_device=root_info.st_dev,
        owner_uid=root_info.st_uid,
        owner_gid=root_info.st_gid,
    )
    return response_path


def _read_json(path: Path, *, blocker: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ProviderBillingAuditRetentionError(blocker)
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderBillingAuditRetentionError(blocker) from exc
    if not isinstance(value, dict):
        raise ProviderBillingAuditRetentionError(blocker)
    return value, payload


def _snapshot(path: Path, *, digest: str) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path),
        "digest": digest,
        "size_bytes": int(info.st_size),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "mtime_ns": int(info.st_mtime_ns),
    }


def _directory_snapshot(path: Path) -> dict[str, Any]:
    info = path.stat()
    return {
        "path": str(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "mtime_ns": int(info.st_mtime_ns),
    }


def _scan_objects(audit_root: Path, root_info: os.stat_result) -> list[dict[str, Any]]:
    objects_root = audit_root / OBJECT_DIRECTORY
    if not os.path.lexists(objects_root):
        return []
    objects_info = _assert_directory(
        objects_root, blocker="provider_billing_audit_object_directory_invalid"
    )
    if (
        objects_info.st_dev != root_info.st_dev
        or objects_info.st_uid != root_info.st_uid
        or objects_info.st_gid != root_info.st_gid
        or stat.S_IMODE(objects_info.st_mode) != DIRECTORY_MODE
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_object_directory_metadata_invalid"
        )
    sha_root = objects_root / "sha256"
    if set(path.name for path in objects_root.iterdir()) != {"sha256"}:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_object_directory_contents_invalid"
        )
    sha_info = _assert_directory(
        sha_root, blocker="provider_billing_audit_object_directory_invalid"
    )
    if (
        sha_info.st_dev != root_info.st_dev
        or sha_info.st_uid != root_info.st_uid
        or sha_info.st_gid != root_info.st_gid
        or stat.S_IMODE(sha_info.st_mode) != DIRECTORY_MODE
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_object_directory_metadata_invalid"
        )
    snapshots: list[dict[str, Any]] = []
    for prefix in sorted(sha_root.iterdir(), key=lambda item: item.name):
        if not re.fullmatch(r"[0-9a-f]{2}", prefix.name):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_object_prefix_invalid"
            )
        prefix_info = _assert_directory(
            prefix, blocker="provider_billing_audit_object_directory_invalid"
        )
        if (
            prefix_info.st_dev != root_info.st_dev
            or prefix_info.st_uid != root_info.st_uid
            or prefix_info.st_gid != root_info.st_gid
            or stat.S_IMODE(prefix_info.st_mode) != DIRECTORY_MODE
        ):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_object_directory_metadata_invalid"
            )
        for object_path in sorted(prefix.iterdir(), key=lambda item: item.name):
            if not re.fullmatch(r"[0-9a-f]{64}", object_path.name) or not object_path.name.startswith(
                prefix.name
            ):
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_object_name_invalid"
                )
            digest = "sha256:" + object_path.name
            if object_path.is_symlink() or not object_path.is_file():
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_object_file_invalid"
                )
            size = object_path.stat().st_size
            _validate_regular_file(
                object_path,
                expected_digest=digest,
                expected_size=size,
                audit_device=root_info.st_dev,
                owner_uid=root_info.st_uid,
                owner_gid=root_info.st_gid,
            )
            snapshots.append(_snapshot(object_path, digest=digest))
    return snapshots


def _scan_locked(audit_root: Path, root_info: os.stat_result) -> dict[str, Any]:
    if os.geteuid() not in {0, root_info.st_uid}:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_executor_identity_mismatch"
        )
    objects = _scan_objects(audit_root, root_info)
    responses_by_digest: dict[str, list[dict[str, Any]]] = {}
    receipts: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    directory_repairs: list[dict[str, Any]] = []
    incomplete_transactions: list[dict[str, Any]] = []
    audit_directories = 0
    for directory in sorted(audit_root.iterdir(), key=lambda item: item.name):
        if directory.name == OBJECT_DIRECTORY:
            _assert_directory(
                directory, blocker="provider_billing_audit_object_directory_invalid"
            )
            continue
        if not _AUDIT_DIRECTORY_RE.fullmatch(directory.name):
            raise ProviderBillingAuditRetentionError(
                f"provider_billing_audit_unknown_child:{directory.name}"
            )
        directory_info = _assert_directory(
            directory, blocker="provider_billing_audit_directory_invalid"
        )
        if directory_info.st_dev != root_info.st_dev:
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_directory_metadata_invalid"
            )
        directory_snapshot = _directory_snapshot(directory)
        directories.append(directory_snapshot)
        receipt_path = directory / "provider_billing_source_receipt.json"
        if not os.path.lexists(receipt_path):
            orphan_responses: list[dict[str, Any]] = []
            retained_bytes = 0
            for path in sorted(directory.iterdir(), key=lambda item: item.name):
                if not _RESPONSE_NAME_RE.fullmatch(path.name):
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_incomplete_transaction_child_invalid"
                    )
                if path.is_symlink() or not path.is_file():
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_incomplete_transaction_response_invalid"
                    )
                try:
                    payload = path.read_bytes()
                except OSError as exc:
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_incomplete_transaction_response_unreadable"
                    ) from exc
                digest = _payload_digest(payload)
                _validate_retained_file(
                    path,
                    expected_digest=digest,
                    expected_size=len(payload),
                    audit_device=root_info.st_dev,
                    audit_uid=root_info.st_uid,
                    audit_gid=root_info.st_gid,
                )
                orphan_responses.append(_snapshot(path, digest=digest))
                retained_bytes += len(payload)
            incomplete_transactions.append(
                {
                    "directory": directory_snapshot,
                    "response_files": orphan_responses,
                    "retained_bytes": retained_bytes,
                    "retention_reason": "source_receipt_missing",
                    "mutation_eligible": False,
                }
            )
            audit_directories += 1
            continue
        if (
            directory_info.st_uid != root_info.st_uid
            or directory_info.st_gid != root_info.st_gid
            or stat.S_IMODE(directory_info.st_mode) != DIRECTORY_MODE
        ):
            directory_repairs.append(
                {
                    **directory_snapshot,
                    "target_mode": DIRECTORY_MODE,
                    "target_uid": int(root_info.st_uid),
                    "target_gid": int(root_info.st_gid),
                }
            )
        receipt, receipt_bytes = _read_json(
            receipt_path, blocker="provider_billing_audit_receipt_invalid"
        )
        receipt_info = receipt_path.stat()
        if (
            receipt_info.st_dev != root_info.st_dev
            or receipt_info.st_uid not in {0, root_info.st_uid}
            or receipt_info.st_gid not in {0, root_info.st_gid}
            or stat.S_IMODE(receipt_info.st_mode) not in {0o440, 0o600}
            or stat.S_IMODE(receipt_info.st_mode) & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_receipt_metadata_invalid"
            )
        if (
            receipt.get("schema_version") != BILLING_SOURCE_SCHEMA_VERSION
            or receipt.get("status") != "reconciled"
            or receipt.get("receipt_digest")
            != _canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("provider_mutation_performed") is not False
            or receipt.get("raw_secret_values_recorded") is not False
            or not isinstance(receipt.get("sources"), list)
        ):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_receipt_contract_invalid"
            )
        referenced: set[Path] = set()
        for row in receipt["sources"]:
            if not isinstance(row, Mapping):
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_source_row_invalid"
                )
            retained = row.get("retained_path")
            digest = row.get("response_digest")
            size = row.get("response_size_bytes")
            if (
                not isinstance(retained, str)
                or not Path(retained).is_absolute()
                or _DIGEST_RE.fullmatch(str(digest or "")) is None
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
            ):
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_source_binding_invalid"
                )
            path = Path(retained)
            if path.parent != directory.resolve() or path in referenced:
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_source_path_invalid"
                )
            _validate_retained_file(
                path,
                expected_digest=str(digest),
                expected_size=size,
                audit_device=root_info.st_dev,
                audit_uid=root_info.st_uid,
                audit_gid=root_info.st_gid,
            )
            referenced.add(path)
            responses_by_digest.setdefault(str(digest), []).append(
                _snapshot(path, digest=str(digest))
            )
        actual_responses = {
            path.resolve(strict=False)
            for path in directory.iterdir()
            if path.name != receipt_path.name
        }
        if actual_responses != {path.resolve(strict=False) for path in referenced}:
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_directory_contents_invalid"
            )
        receipts.append(
            _snapshot(receipt_path, digest=_payload_digest(receipt_bytes))
        )
        audit_directories += 1

    groups: list[dict[str, Any]] = []
    duplicate_bytes = 0
    excluded_paths: list[dict[str, Any]] = []
    for digest, paths in sorted(responses_by_digest.items()):
        all_ordered = sorted(paths, key=lambda item: item["path"])
        ordered = [
            item
            for item in all_ordered
            if item["uid"] == root_info.st_uid
            and item["gid"] == root_info.st_gid
            and item["mode"] == OBJECT_MODE
        ]
        excluded_paths.extend(
            item
            for item in all_ordered
            if item not in ordered
        )
        if not ordered:
            continue
        size = ordered[0]["size_bytes"]
        if any(item["size_bytes"] != size for item in ordered):
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_digest_size_conflict"
            )
        unique_inodes = {(item["device"], item["inode"]) for item in ordered}
        reclaim = size * max(0, len(unique_inodes) - 1)
        duplicate_bytes += reclaim
        groups.append(
            {
                "response_digest": digest,
                "response_size_bytes": size,
                "paths": ordered,
                "duplicate_path_count": max(0, len(ordered) - 1),
                "unique_inode_count": len(unique_inodes),
                "predicted_relinked_bytes": reclaim,
            }
        )
    return {
        "audit_directory_count": audit_directories,
        "receipt_count": len(receipts),
        "response_path_count": sum(
            len(paths) for paths in responses_by_digest.values()
        ),
        "receipts": sorted(receipts, key=lambda item: item["path"]),
        "audit_directories": sorted(directories, key=lambda item: item["path"]),
        "directory_repairs": sorted(directory_repairs, key=lambda item: item["path"]),
        "unreconciled_incomplete_transactions": sorted(
            incomplete_transactions, key=lambda item: item["directory"]["path"]
        ),
        "unreconciled_incomplete_transaction_count": len(incomplete_transactions),
        "unreconciled_retained_bytes": sum(
            item["retained_bytes"] for item in incomplete_transactions
        ),
        "objects": objects,
        "response_groups": groups,
        "metadata_excluded_response_paths": sorted(
            excluded_paths, key=lambda item: item["path"]
        ),
        "predicted_relinked_bytes": duplicate_bytes,
    }


def _build_plan_locked(root: Path, root_info: os.stat_result) -> dict[str, Any]:
    scan = _scan_locked(root, root_info)
    result: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "dry_run",
        "audit_root": str(root),
        "audit_root_snapshot": {
            "device": int(root_info.st_dev),
            "inode": int(root_info.st_ino),
            "uid": int(root_info.st_uid),
            "gid": int(root_info.st_gid),
        },
        **scan,
        "production_artifact_mutation_performed": False,
        "provider_mutation_performed": False,
        "receipt_bytes_changed": False,
        "retained_paths_changed": False,
    }
    result["plan_digest"] = _canonical_digest(result, digest_field="plan_digest")
    return result


def build_provider_billing_audit_retention_plan(
    *, audit_root: str | Path
) -> dict[str, Any]:
    with audit_root_lock(audit_root) as (root, root_info):
        return _build_plan_locked(root, root_info)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_receipt_output_unavailable"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _reserve_apply_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_receipt_output_unavailable"
        ) from exc
    os.close(descriptor)


def _replace_reserved_output(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_snapshot(snapshot: Mapping[str, Any]) -> Path:
    path = Path(str(snapshot.get("path") or ""))
    if path.is_symlink() or not path.is_file():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_snapshot_changed"
        )
    info = path.stat()
    fields = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mtime_ns": info.st_mtime_ns,
    }
    if any(snapshot.get(key) != value for key, value in fields.items()):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_snapshot_changed"
        )
    if _payload_digest(path.read_bytes()) != snapshot.get("digest"):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_snapshot_changed"
        )
    return path


def _assert_directory_snapshot(snapshot: Mapping[str, Any]) -> Path:
    path = Path(str(snapshot.get("path") or ""))
    if path.is_symlink() or not path.is_dir():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_directory_snapshot_changed"
        )
    info = path.stat()
    fields = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mtime_ns": info.st_mtime_ns,
    }
    if any(snapshot.get(key) != value for key, value in fields.items()):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_directory_snapshot_changed"
        )
    return path


def _repair_directory(
    snapshot: Mapping[str, Any], *, target_uid: int, target_gid: int
) -> Path:
    path = _assert_directory_snapshot(snapshot)
    info = path.stat()
    if (info.st_uid, info.st_gid) != (target_uid, target_gid):
        os.chown(path, target_uid, target_gid)
    if stat.S_IMODE(path.stat().st_mode) != DIRECTORY_MODE:
        os.chmod(path, DIRECTORY_MODE)
    current = path.stat()
    if (
        current.st_uid != target_uid
        or current.st_gid != target_gid
        or stat.S_IMODE(current.st_mode) != DIRECTORY_MODE
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_directory_repair_failed"
        )
    return path


def _link_object_from_snapshot(
    *,
    root: Path,
    root_info: os.stat_result,
    digest: str,
    source: Path,
) -> Path:
    object_path = _object_path(root, digest)
    current = root
    for component in (OBJECT_DIRECTORY, "sha256", object_path.parent.name):
        current = current / component
        _ensure_owned_directory(
            current,
            audit_device=root_info.st_dev,
            owner_uid=root_info.st_uid,
            owner_gid=root_info.st_gid,
        )
    try:
        os.link(source, object_path, follow_symlinks=False)
    except FileExistsError:
        pass
    _validate_regular_file(
        object_path,
        expected_digest=digest,
        expected_size=source.stat().st_size,
        audit_device=root_info.st_dev,
        owner_uid=root_info.st_uid,
        owner_gid=root_info.st_gid,
    )
    return object_path


def _atomic_relink(path: Path, object_path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        os.link(object_path, temporary, follow_symlinks=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def apply_provider_billing_audit_retention_plan(
    *,
    dry_run_plan_path: str | Path,
    acknowledgement: str,
    receipt_out: str | Path,
) -> dict[str, Any]:
    if acknowledgement != APPLY_ACKNOWLEDGEMENT:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_apply_acknowledgement_missing"
        )
    plan_path = Path(dry_run_plan_path).expanduser()
    output_path = Path(receipt_out).expanduser()
    if not plan_path.is_absolute() or not output_path.is_absolute():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_apply_path_must_be_absolute"
        )
    plan, _payload = _read_json(
        plan_path, blocker="provider_billing_audit_dry_run_plan_invalid"
    )
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "dry_run"
        or plan.get("plan_digest")
        != _canonical_digest(plan, digest_field="plan_digest")
    ):
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_dry_run_plan_invalid"
        )
    root = Path(str(plan.get("audit_root") or ""))
    if plan_path.resolve() == output_path.resolve():
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_apply_receipt_overlaps_plan"
        )
    if root == plan_path or root in plan_path.parents or root == output_path or root in output_path.parents:
        raise ProviderBillingAuditRetentionError(
            "provider_billing_audit_receipt_inside_audit_root"
        )
    _reserve_apply_output(output_path)
    try:
        with audit_root_lock(root) as (locked_root, root_info):
            current = _build_plan_locked(locked_root, root_info)
            if current.get("plan_digest") != plan.get("plan_digest"):
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_plan_changed_since_dry_run"
                )
            all_snapshots: list[Mapping[str, Any]] = [
                *list(plan.get("receipts") or []),
                *list(plan.get("objects") or []),
                *list(plan.get("metadata_excluded_response_paths") or []),
            ]
            for transaction in plan.get("unreconciled_incomplete_transactions") or []:
                if not isinstance(transaction, Mapping):
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_dry_run_plan_invalid"
                    )
                all_snapshots.extend(transaction.get("response_files") or [])
            for group in plan.get("response_groups") or []:
                if not isinstance(group, Mapping):
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_dry_run_plan_invalid"
                    )
                all_snapshots.extend(group.get("paths") or [])
            directory_snapshots = list(plan.get("audit_directories") or [])
            directory_repairs = list(plan.get("directory_repairs") or [])
            for snapshot in directory_snapshots:
                if not isinstance(snapshot, Mapping):
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_dry_run_plan_invalid"
                    )
                _assert_directory_snapshot(snapshot)
            for snapshot in all_snapshots:
                if not isinstance(snapshot, Mapping):
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_dry_run_plan_invalid"
                    )
                _assert_snapshot(snapshot)

            for repair in directory_repairs:
                if not isinstance(repair, Mapping):
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_dry_run_plan_invalid"
                    )
                if os.geteuid() != 0 and repair.get("uid") != os.geteuid():
                    raise ProviderBillingAuditRetentionError(
                        "provider_billing_audit_directory_repair_requires_owner"
                    )

            repaired_directories: list[str] = []
            for repair in directory_repairs:
                repaired = _repair_directory(
                    repair,
                    target_uid=int(root_info.st_uid),
                    target_gid=int(root_info.st_gid),
                )
                repaired_directories.append(str(repaired))

            relinked_paths = 0
            relinked_bytes = 0
            for group in plan.get("response_groups") or []:
                paths = list(group.get("paths") or [])
                if len(paths) < 2:
                    continue
                digest = str(group.get("response_digest") or "")
                source = _assert_snapshot(paths[0])
                object_path = _link_object_from_snapshot(
                    root=locked_root,
                    root_info=root_info,
                    digest=digest,
                    source=source,
                )
                object_inode = object_path.stat().st_ino
                for snapshot in paths:
                    path = _assert_snapshot(snapshot)
                    if path.stat().st_ino == object_inode:
                        continue
                    _atomic_relink(path, object_path)
                    _validate_regular_file(
                        path,
                        expected_digest=digest,
                        expected_size=int(group["response_size_bytes"]),
                        audit_device=root_info.st_dev,
                        owner_uid=root_info.st_uid,
                        owner_gid=root_info.st_gid,
                    )
                    relinked_paths += 1
                    relinked_bytes += int(group["response_size_bytes"])

            result: dict[str, Any] = {
                "schema_version": APPLY_SCHEMA_VERSION,
                "status": "applied",
                "dry_run_plan_path": str(plan_path),
                "dry_run_plan_digest": plan["plan_digest"],
                "audit_root": str(locked_root),
                "relinked_path_count": relinked_paths,
                "relinked_bytes": relinked_bytes,
                "repaired_directories": repaired_directories,
                "predicted_relinked_bytes": plan["predicted_relinked_bytes"],
                "receipt_bytes_changed": False,
                "retained_paths_changed": False,
                "provider_mutation_performed": False,
                "production_artifact_mutation_performed": bool(
                    relinked_paths or repaired_directories
                ),
            }
            result["receipt_digest"] = _canonical_digest(
                result, digest_field="receipt_digest"
            )
        _replace_reserved_output(output_path, result)
    except BaseException:
        if output_path.is_file() and output_path.stat().st_size == 0:
            output_path.unlink()
        raise
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root")
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run-plan")
    parser.add_argument("--ack")
    args = parser.parse_args(argv)
    try:
        output = Path(args.receipt_out).expanduser()
        if not output.is_absolute():
            raise ProviderBillingAuditRetentionError(
                "provider_billing_audit_receipt_out_must_be_absolute"
            )
        if args.apply:
            if args.audit_root or not args.dry_run_plan:
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_apply_parameters_invalid"
                )
            result = apply_provider_billing_audit_retention_plan(
                dry_run_plan_path=args.dry_run_plan,
                acknowledgement=str(args.ack or ""),
                receipt_out=output,
            )
        else:
            if not args.audit_root or args.dry_run_plan or args.ack:
                raise ProviderBillingAuditRetentionError(
                    "provider_billing_audit_dry_run_parameters_invalid"
                )
            result = build_provider_billing_audit_retention_plan(
                audit_root=args.audit_root
            )
            _write_exclusive(output, result)
    except (OSError, ProviderBillingAuditRetentionError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "blueprint.provider_billing_audit_retention_run.v1",
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "APPLY_ACKNOWLEDGEMENT",
    "ProviderBillingAuditRetentionError",
    "apply_provider_billing_audit_retention_plan",
    "audit_root_lock",
    "build_provider_billing_audit_retention_plan",
    "retain_billing_response",
]
