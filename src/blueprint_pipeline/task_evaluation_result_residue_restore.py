"""Bring a sealed result run's offloaded residue back, member by member.

``task_evaluation_result_residue_offload`` moved the residue behind
``<run>.residue.v1.json``. Restore streams that archive back and verifies it,
then puts each member back through directory descriptors held from the run
root, verifying its digest and size, and records a receipt beside the pointer as
``<run>.residue-restore.v1.json``. It holds the run's offload lock
(``artifacts/result_delivery/.offload.lock``) for its whole pass, so no tick can
resume an eviction while members come back, and it refuses to start while a tick
holds it (``result_residue_restore_locked``). Once its pass is done it rewrites
the pointer ``restored``: the pointer stays, and no tick offloads the run again.

It never overwrites: a member whose path holds a different file, or whose place
cannot be reached (its directory became a file or a link), is a typed conflict
and the others are still restored. It needs no sealed or unchanged registry,
only the pointer's run: its directory name, and its run id when the registry
still names one. The names of one inode (a ``group`` in the pointer) come back
as hard links of one restored file, each only when the pointer gives it that
file's digest and size (``group_member_differs`` otherwise). Every directory it creates an entry in is
fsynced, and the receipt is written whatever happens once the pointer verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from . import completed_replay_cache_retention as held_files
from . import control_plane_evidence_offload as evidence
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_result_artifact_store import offload_failure
from .task_evaluation_result_residue_offload import (
    POINTER_SUFFIX,
    RESTORE_RECEIPT_SUFFIX,
    RESTORE_SCHEMA_VERSION,
    RESULT_DELIVERY,
    _adopt_owner,
    _hold_offload_lock,
    _read_pointer,
    _write_json,
    archive_reference,
    pointer_with_state,
)
from .task_evaluation_result_residue_scan import ResultResidueOffloadError

_MIB = 1024 * 1024
_MAX_REGISTRY_BYTES = 64 * _MIB
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class _Anchor(NamedTuple):
    """A group's member already in place with the archive's bytes: its siblings link to it.

    It carries the digest and size the pointer records for it; a sibling the
    pointer gives other bytes is never linked to it.
    """

    directory: int
    name: str
    identity: tuple[int, int]
    sha256: str
    size_bytes: int


def _registry_run_id(root: Path) -> str | None:
    """The run id the registry names, read without requiring it sealed or unchanged; None when it does not say."""

    path = root / RESULT_DELIVERY / "artifact_registry.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_REGISTRY_BYTES:
            return None
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    run_id = value.get("run_id") if isinstance(value, dict) else None
    return run_id if isinstance(run_id, str) else None


def _entry(directory: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _same_file(directory: int, name: str, entry: os.stat_result, member: Mapping[str, Any]) -> bool:
    if not stat.S_ISREG(entry.st_mode) or entry.st_size != member["size_bytes"]:
        return False
    try:
        return held_files._held_sha(directory, name, entry) == member["sha256"]
    except held_files._CrossDevice:
        return False


def _open_directory(root_fd: int, parts: tuple[str, ...], owner: os.stat_result, opened: list[int]) -> int:
    """The member's directory, opened one ``O_NOFOLLOW`` component at a time, created where missing."""

    directory = root_fd
    for part in parts:
        try:
            os.mkdir(part, 0o750, dir_fd=directory)
            created = True
        except FileExistsError:
            created = False
        # O_NOFOLLOW: a directory swapped for a link is never entered.
        child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory)
        opened.append(child)
        if created:
            _adopt_owner(child, owner, mode=0o750)
            os.fsync(directory)
        directory = child
    return directory


def _existing(directory: int, name: str, member: Mapping[str, Any]) -> tuple[str, _Anchor | None]:
    entry = _entry(directory, name)
    if entry is not None and _same_file(directory, name, entry, member):
        return "already_present", _Anchor(directory, name, (entry.st_dev, entry.st_ino), member["sha256"],
                                          member["size_bytes"])
    return "existing_file_differs", None


def _link_sibling(anchor: _Anchor, directory: int, name: str, member) -> tuple[str, _Anchor | None]:
    """Link ``name`` to the group's anchor, only while the anchor is still the file it placed.

    The pointer must give the sibling the anchor's digest and size; otherwise it
    is a ``group_member_differs`` conflict, and nothing is linked at its path.
    """

    if (member["sha256"], member["size_bytes"]) != (anchor.sha256, anchor.size_bytes):
        return "group_member_differs", None
    current = _entry(anchor.directory, anchor.name)
    if current is None or not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != anchor.identity:
        raise ResultResidueOffloadError("result_residue_restore_anchor_changed")
    try:
        os.link(anchor.name, name, src_dir_fd=anchor.directory, dst_dir_fd=directory, follow_symlinks=False)
    except FileExistsError:
        return _existing(directory, name, member)
    linked = _entry(directory, name)
    if linked is None or (linked.st_dev, linked.st_ino) != anchor.identity:
        # The anchor changed between the check and the link: take back the name just made.
        os.unlink(name, dir_fd=directory)
        raise ResultResidueOffloadError("result_residue_restore_anchor_changed")
    os.fsync(directory)
    return "restored", anchor


def _write_member(archive: tarfile.TarFile, info: tarfile.TarInfo, directory: int, name: str, member,
                  owner: os.stat_result) -> tuple[str, _Anchor | None]:
    """Write one member from the archive beside its place, verify it, then link it into place."""

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
            identity = os.fstat(stream.fileno())
        try:
            # A link never replaces an entry that appeared meanwhile.
            os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        except FileExistsError:
            return _existing(directory, name, member)
        os.fsync(directory)
        return "restored", _Anchor(directory, name, (identity.st_dev, identity.st_ino), member["sha256"],
                                   member["size_bytes"])
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def _restore_group(root_fd, archive, infos, members, owner, outcomes, conflicts) -> None:
    """Put one inode group back: its first placeable member from the archive, the others linked to it."""

    opened: list[int] = []
    anchor: _Anchor | None = None
    try:
        for member in members:
            relative = member["relative_path"]
            try:
                parts = PurePosixPath(relative).parts
                directory = _open_directory(root_fd, parts[:-1], owner, opened)
                if _entry(directory, parts[-1]) is not None:
                    outcome, placed = _existing(directory, parts[-1], member)
                elif anchor is not None:
                    outcome, placed = _link_sibling(anchor, directory, parts[-1], member)
                else:
                    outcome, placed = _write_member(archive, infos[relative], directory, parts[-1], member, owner)
            except (OSError, ResultResidueOffloadError) as exc:
                conflicts.append({"relative_path": relative, "reason": f"restore_failed:{type(exc).__name__}"})
                continue
            if outcome not in outcomes:
                conflicts.append({"relative_path": relative, "reason": outcome})
                continue
            outcomes[outcome].append(relative)
            anchor = anchor or placed
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
    different file at its path, or a place it cannot reach, is a typed
    ``conflict`` and the rest still come back. The run's offload lock is held
    throughout, and a pass that finishes rewrites the pointer ``restored``. The
    receipt is written beside the pointer whatever happens once the pointer
    verified; a failure (the archive cannot be fetched or does not verify, or
    the pointer cannot be rewritten) is recorded in it and then raised.
    """

    unresolved = Path(run_root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ResultResidueOffloadError("result_residue_restore_run_invalid")
    root = unresolved.resolve()
    with ExitStack() as stack:
        if not _hold_offload_lock(root, stack):
            # A tick is offloading or resuming this run: restoring now would race its eviction.
            raise ResultResidueOffloadError("result_residue_restore_locked")
        return _restore_locked(root, materializer, now)


def _restore_locked(root: Path, materializer: Callable[..., Any] | None, now: Callable[[], float]) -> dict[str, Any]:
    pointer = _read_pointer(root)
    run_id = _registry_run_id(root)
    if run_id is not None and run_id != pointer.get("run_id"):
        raise ResultResidueOffloadError("result_residue_restore_run_mismatch")
    archive_row = pointer["archive"]
    kept = {str(row.get("relative_path")) for row in pointer["kept"] if isinstance(row, Mapping)}
    expected = {member["relative_path"]: member for member in pointer["members"] if member["relative_path"] not in kept}
    outcomes: dict[str, list[str]] = {"restored": [], "already_present": []}
    conflicts: list[dict[str, str]] = []
    failure: BaseException | None = None
    reservation = staging = None
    try:
        reservation = evidence.reserve_control_plane_disk(
            "evidence_offload", target_root=root.parent,
            expected_bytes=_MIB + int(archive_row["size_bytes"]) + sum(int(m["size_bytes"]) for m in expected.values()),
            reservation_root=evidence.DEFAULT_RESERVATION_ROOT)
        staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.residue-restore-", dir=root.parent))
        archive_path = staging / "residue.tar"
        (materializer or evidence.materialize_configured_scene_artifact)(
            reference=archive_reference(archive_row),
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
            # Only what it restores must be a file entry; a kept member's is never read.
            if set(infos) != {member["relative_path"] for member in pointer["members"]} or any(
                    not (infos[relative].isreg() or infos[relative].islnk()) for relative in expected):
                raise ResultResidueOffloadError("result_residue_restore_archive_invalid")
            groups: dict[Any, list[Mapping[str, Any]]] = {}
            for relative in sorted(expected):
                member = expected[relative]
                groups.setdefault(member.get("group", ("alone", relative)), []).append(member)
            root_fd = os.open(root, _DIRECTORY_FLAGS)
            try:
                for members in groups.values():
                    _restore_group(root_fd, archive, infos, members, owner, outcomes, conflicts)
            finally:
                os.close(root_fd)
    except Exception as exc:  # noqa: BLE001 - recorded in the receipt, then raised
        failure = exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if reservation is not None:
            reservation.release()
    if failure is None and pointer.get("state") != "restored":
        try:
            # An operator brought the run back: no tick may evict it again, or resume an eviction.
            _write_json(root.parent / f"{root.name}{POINTER_SUFFIX}", pointer_with_state(pointer, "restored"))
        except Exception as exc:  # noqa: BLE001 - recorded in the receipt, then raised
            failure = exc
    receipt: dict[str, Any] = {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "status": "failed" if failure is not None else ("restored_with_conflicts" if conflicts else "restored"),
        "run": root.name,
        "pointer_digest": pointer["pointer_digest"],
        "archive_sha256": archive_row["sha256"],
        "restored_count": len(outcomes["restored"]),
        "restored_bytes": sum(int(expected[relative]["size_bytes"]) for relative in outcomes["restored"]),
        "already_present_count": len(outcomes["already_present"]),
        "kept_in_place_count": len(kept),
        "conflicts": conflicts,
        "restored_at_epoch": float(now()),
        "receipt_digest": "",
    }
    if failure is not None:
        receipt["failure"] = offload_failure(failure, "restore")
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(root.parent / f"{root.name}{RESTORE_RECEIPT_SUFFIX}", receipt)
    if failure is not None:
        raise failure
    return receipt


__all__ = ["restore_result_residue"]
