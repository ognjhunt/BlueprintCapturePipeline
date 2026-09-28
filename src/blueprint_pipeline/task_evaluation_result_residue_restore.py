"""Bring a sealed result run's offloaded residue back, member by member.

``task_evaluation_result_residue_offload`` moved the residue behind
``<run>.residue.v1.json``. Restore streams that archive back, verifies every
member's digest and size, never overwrites a different file, and records a
receipt beside the pointer as ``<run>.residue-restore.v1.json``. The pointer
stays, so the next storage GC tick does not offload the restored files again.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from . import completed_replay_cache_retention as held_files
from . import control_plane_evidence_offload as evidence
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_result_artifact_store import _sealed_registry
from .task_evaluation_result_residue_offload import (
    RESTORE_RECEIPT_SUFFIX,
    RESTORE_SCHEMA_VERSION,
    _adopt_owner,
    _read_pointer,
    _write_json,
)
from .task_evaluation_result_residue_scan import ResultResidueOffloadError

_MIB = 1024 * 1024


def _same_file(directory: int, name: str, entry: os.stat_result, member: Mapping[str, Any]) -> bool:
    if not stat.S_ISREG(entry.st_mode) or entry.st_size != member["size_bytes"]:
        return False
    try:
        return held_files._held_sha(directory, name, entry) == member["sha256"]
    except held_files._CrossDevice:
        return False


def _restore_member(root_fd: int, archive: tarfile.TarFile, info: tarfile.TarInfo, member, owner) -> str:
    """Put one member back through descriptors from the run root: ``restored``, ``already_present`` or ``conflict``."""

    parts = PurePosixPath(member["relative_path"]).parts
    opened: list[int] = []
    directory = root_fd
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o750, dir_fd=directory)
                created = True
            except FileExistsError:
                created = False
            # O_NOFOLLOW: a directory swapped for a link is never entered.
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            opened.append(directory)
            if created:
                _adopt_owner(directory, owner, mode=0o750)
        name = parts[-1]
        try:
            entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            entry = None
        if entry is not None:
            return "already_present" if _same_file(directory, name, entry, member) else "conflict"
        temporary = f".{name}.residue-restore-{secrets.token_hex(6)}"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=directory)
        try:
            digest, size = hashlib.sha256(), 0
            with os.fdopen(descriptor, "wb") as stream:
                source = archive.extractfile(info)
                if source is None:
                    raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
                for chunk in iter(lambda: source.read(_MIB), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    stream.write(chunk)
                stream.flush()
                if ("sha256:" + digest.hexdigest(), size) != (member["sha256"], member["size_bytes"]):
                    raise ResultResidueOffloadError("result_residue_restore_member_mismatch")
                os.utime(stream.fileno(), (info.mtime, info.mtime))
                _adopt_owner(stream.fileno(), owner, mode=int(member["mode"]) & 0o7777)
                os.fsync(stream.fileno())
            try:
                # A link never replaces an entry that appeared meanwhile.
                os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            except FileExistsError:
                entry = os.stat(name, dir_fd=directory, follow_symlinks=False)
                return "already_present" if _same_file(directory, name, entry, member) else "conflict"
            return "restored"
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def restore_result_residue(
    *,
    run_root: str | Path,
    materializer: Callable[..., Any] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Bring a run's offloaded residue back, verifying every member; never overwrite a different file.

    Members the pointer lists as ``kept`` never left and are not touched. A
    member already in place with the same bytes is ``already_present``; a
    different file at its path is a ``conflict`` and stays. The receipt is
    written beside the pointer as ``<run>.residue-restore.v1.json`` and returned.
    """

    unresolved = Path(run_root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ResultResidueOffloadError("result_residue_restore_run_invalid")
    root = unresolved.resolve()
    pointer = _read_pointer(root)
    registry, _registry_path, _registry_bytes = _sealed_registry(root)
    if registry["registry_digest"] != pointer["registry_digest"]:
        raise ResultResidueOffloadError("result_residue_restore_registry_changed")
    archive_row = pointer["archive"]
    kept = {str(row.get("relative_path")) for row in pointer["kept"] if isinstance(row, Mapping)}
    expected = {member["relative_path"]: member for member in pointer["members"] if member["relative_path"] not in kept}
    reservation = evidence.reserve_control_plane_disk(
        "evidence_offload", target_root=root.parent,
        expected_bytes=_MIB + int(archive_row["size_bytes"]) + sum(int(m["size_bytes"]) for m in expected.values()),
        reservation_root=evidence.DEFAULT_RESERVATION_ROOT)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.residue-restore-", dir=root.parent))
    outcomes: dict[str, list[str]] = {"restored": [], "already_present": [], "conflict": []}
    try:
        archive_path = staging / "residue.tar"
        (materializer or evidence.materialize_configured_scene_artifact)(
            reference={
                "schema_version": "task_evaluation_scene_artifact_reference.v1",
                "status": "remote_verified",
                "artifact_kind": archive_row.get("artifact_kind", evidence.ARTIFACT_KIND),
                "uri": archive_row["uri"],
                "digest": archive_row["sha256"],
                "size_bytes": archive_row["size_bytes"],
                # A pointer is written only after a full remote readback.
                "remote_identity_verified": True,
                "full_byte_service_account_readback_passed": True,
                "raw_secret_values_recorded": False,
            },
            destination=archive_path,
            maximum_size_bytes=int(archive_row["size_bytes"]),
        )
        if (evidence._sha256(archive_path), archive_path.stat().st_size) != (
                archive_row["sha256"], archive_row["size_bytes"]):
            raise ResultResidueOffloadError("result_residue_restore_digest_mismatch")
        owner = os.stat(root)
        with tarfile.open(archive_path, mode="r:") as archive:
            infos: dict[str, tarfile.TarInfo] = {}
            for info in archive.getmembers():
                if info.name in infos:
                    raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
                infos[info.name] = info
            if set(infos) != {member["relative_path"] for member in pointer["members"]} or any(
                    not (infos[relative].isreg() or infos[relative].islnk()) for relative in expected):
                raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                for relative in sorted(expected):
                    outcome = _restore_member(root_fd, archive, infos[relative], expected[relative], owner)
                    outcomes[outcome].append(relative)
            finally:
                os.close(root_fd)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        reservation.release()
    receipt: dict[str, Any] = {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "status": "restored_with_conflicts" if outcomes["conflict"] else "restored",
        "run": root.name,
        "pointer_digest": pointer["pointer_digest"],
        "archive_sha256": archive_row["sha256"],
        "restored_count": len(outcomes["restored"]),
        "restored_bytes": sum(int(expected[relative]["size_bytes"]) for relative in outcomes["restored"]),
        "already_present_count": len(outcomes["already_present"]),
        "kept_in_place_count": len(kept),
        "conflicts": [{"relative_path": relative, "reason": "existing_file_differs"}
                      for relative in outcomes["conflict"]],
        "restored_at_epoch": float(now()),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(root.parent / f"{root.name}{RESTORE_RECEIPT_SUFFIX}", receipt)
    return receipt


__all__ = ["restore_result_residue"]
