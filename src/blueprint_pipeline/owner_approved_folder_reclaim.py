"""Delete the internal folders the owner approved on the control-plane host, and nothing else.

The host's disk fills, and a paid run then refuses its pre-rental reservation. Some of the
bytes are internal folders nothing reads any more, which only the owner may approve deleting
(AGENTS.md: ask the owner first before deleting retained data). This module deletes exactly the
folders listed in a reviewed, versioned allowlist that ships with the release,
``config/owner_approved_reclaim_folders.v1.json``. Each entry names one absolute path and the
owner's approval (who, when, the quoted words). There is no path argument: a caller can choose
only between a plan and an apply.

A listed folder is refused, and nothing at all is deleted in that run, unless:

* it is a direct child directory of ``/var/lib/blueprint/pipeline-control-plane``, by a plain
  name (no leading dot, no ``..``);
* it is not, and does not contain, a root that ``control_plane_storage_roots.STORAGE_ROOTS``
  classifies (``evidence_hot``, ``evidence_cold``, ``ledger``, ``work``, ``release`` and every
  other class): the refusal names the root and its class;
* no component of its path is a symlink (every directory is opened from ``/`` one component at
  a time with ``O_NOFOLLOW``), and it is not a symlink itself;
* it is on its parent's filesystem, holds no other mount (by ``st_dev`` and the mount table)
  and no file in it was modified after the entry's ``refuse_if_modified_after``, the start of
  the day the owner approved it; and
* the user running it can remove every directory in it, so a deletion never stops half way
  for want of permission.

A plan (the default) reports for each folder whether it exists, its total bytes, file count,
newest mtime and any refusal, and writes nothing. An apply re-runs every check, then removes each
present folder with a recursive delete over directory descriptors that never follows a symlink
(a link is unlinked, its target untouched) and never leaves the folder's filesystem. It keeps a
receipt (paths, bytes and counts removed, the disk's free bytes before and after, UTC times) in
``storage-gc/owner-approved-reclaims/`` and prints the same document, which the operator door
keeps as its result. An absent folder is ``already_reclaimed``; a second apply changes nothing.
An apply that stops part way records what it removed and is finished by the next apply.

The operator door's ``reclaim-approved-folders`` request carries only ``apply``. Output is JSON
only. Nothing here reads a secret, uses the network, starts compute or moves money.
"""

from __future__ import annotations

import argparse
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .control_plane_storage_roots import STORAGE_ROOTS, StorageRoot

SCHEMA_VERSION = "owner_approved_folder_reclaim.v1"
ALLOWLIST_SCHEMA = "owner_approved_reclaim_folders.v1"
CONTROL_PLANE = PurePosixPath("/var/lib/blueprint/pipeline-control-plane")
ALLOWLIST = Path(__file__).resolve().parents[2] / "config" / "owner_approved_reclaim_folders.v1.json"
#: Kept receipts, under the storage reclaim reports (``storage-gc`` is ``evidence_hot``).
RECEIPTS_DIR = Path(str(CONTROL_PLANE)) / "storage-gc" / "owner-approved-reclaims"
RECEIPT_PREFIX = "reclaim-"
MAX_ALLOWLIST_BYTES = 64 * 1024
MAX_FOLDERS = 32
MAX_DEPTH = 128
MAX_OWNER_UIDS = 16
DIRECTORY_MODE = 0o755
FILE_MODE = 0o644
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_INSTANT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
OUTCOMES = ("planned", "reclaimed", "already_reclaimed", "refused")
REFUSALS = (
    "reclaim_approved_folders_allowlist_invalid",
    "reclaim_approved_folders_path_outside_parent",
    "reclaim_approved_folders_protected_storage_root",
    "reclaim_approved_folders_parent_unsafe",
    "reclaim_approved_folders_symlink",
    "reclaim_approved_folders_not_a_directory",
    "reclaim_approved_folders_mount_point",
    "reclaim_approved_folders_mount_point_inside",
    "reclaim_approved_folders_too_deep",
    "reclaim_approved_folders_unreadable",
    "reclaim_approved_folders_changed_while_checked",
    "reclaim_approved_folders_modified_after_approval",
    "reclaim_approved_folders_not_removable",
    "reclaim_approved_folders_busy",
    "reclaim_approved_folders_receipts_unavailable",
    "reclaim_approved_folders_delete_incomplete",
    "reclaim_approved_folders_failed",
)
_EXIT_ONE = frozenset({"reclaim_approved_folders_failed", "reclaim_approved_folders_delete_incomplete"})


@dataclass(frozen=True)
class Settings:
    """Where the allowlist, the folders and the receipts are. Tests point these at tmp folders;
    an entry's path is always checked against ``CONTROL_PLANE`` itself."""

    allowlist: Path
    parent: Path
    receipts_dir: Path


SETTINGS = Settings(allowlist=ALLOWLIST, parent=Path(str(CONTROL_PLANE)), receipts_dir=RECEIPTS_DIR)


class ReclaimRefused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _Tally:
    file_count: int = 0
    total_bytes: int = 0
    allocated_bytes: int = 0
    directory_count: int = 0
    symlink_count: int = 0
    other_count: int = 0
    multiply_linked_file_count: int = 0
    newest_mtime: float | None = None
    newest_file_mtime: float | None = None
    owner_uids: set[int] = field(default_factory=set)
    blocked_directories: int = 0
    _inodes: set[int] = field(default_factory=set)

    def add(self, info: os.stat_result) -> None:
        self.newest_mtime = info.st_mtime if self.newest_mtime is None else max(self.newest_mtime, info.st_mtime)
        if len(self.owner_uids) < MAX_OWNER_UIDS:
            self.owner_uids.add(info.st_uid)
        if stat.S_ISDIR(info.st_mode):
            self.directory_count += 1
            return
        self.newest_file_mtime = (info.st_mtime if self.newest_file_mtime is None
                                  else max(self.newest_file_mtime, info.st_mtime))
        if stat.S_ISLNK(info.st_mode):
            self.symlink_count += 1
        elif stat.S_ISREG(info.st_mode):
            self.file_count += 1
            self.total_bytes += info.st_size
            if info.st_nlink > 1:
                self.multiply_linked_file_count += 1
            if info.st_ino not in self._inodes:
                self._inodes.add(info.st_ino)
                self.allocated_bytes += info.st_blocks * 512
        else:
            self.other_count += 1

    def measured(self) -> dict[str, Any]:
        return {"total_bytes": self.total_bytes, "allocated_bytes": self.allocated_bytes,
                "file_count": self.file_count, "directory_count": self.directory_count,
                "symlink_count": self.symlink_count, "other_count": self.other_count,
                "multiply_linked_file_count": self.multiply_linked_file_count,
                "newest_mtime": _epoch_iso(self.newest_mtime),
                "newest_file_mtime": _epoch_iso(self.newest_file_mtime),
                "owner_uids": sorted(self.owner_uids)}

    def removed(self) -> dict[str, Any]:
        return {"total_bytes": self.total_bytes, "allocated_bytes": self.allocated_bytes,
                "file_count": self.file_count, "directory_count": self.directory_count,
                "symlink_count": self.symlink_count, "other_count": self.other_count}


# --- small helpers ---------------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _epoch_iso(value: float | None) -> str | None:
    if value is None:
        return None
    return _iso(datetime.fromtimestamp(int(value), timezone.utc))


def _instant(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _disk_free(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _stat(name: str, dir_fd: int) -> os.stat_result:
    return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)


def _removable(name: str, dir_fd: int) -> bool:
    """Whether this user may remove entries in the directory ``name`` (write and search on it). The
    name was just found to be a directory; a read-only filesystem answers no, so a sandbox that
    cannot write here refuses before anything is deleted."""

    return os.access(name, os.W_OK | os.X_OK, dir_fd=dir_fd)


def _open_directory(name: str, *, dir_fd: int | None = None) -> int:
    return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)


def _open_path(path: Path) -> int:
    """A descriptor for ``path``, opened from ``/`` one component at a time without following a link."""

    descriptor = _open_directory("/")
    try:
        for part in path.parts[1:]:
            child = _open_directory(part, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _mount_points() -> list[str] | None:
    """Every mount point this process sees, or None when the mount table cannot be read."""

    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape") as stream:
            lines = stream.read().splitlines()
    except OSError:
        return None
    points = []
    for line in lines:
        fields = line.split(" ")
        if len(fields) > 4:
            points.append(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), fields[4]))
    return points


# --- the allowlist ---------------------------------------------------------------------------------

def _text(value: Any, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and value.isprintable()


def _valid_approval(approval: Any) -> bool:
    if not isinstance(approval, dict) or not {"approved_by", "approved_on", "words", "note"} <= set(approval):
        return False
    words = approval["words"]
    return (_text(approval["approved_by"], 200) and isinstance(approval["approved_on"], str)
            and _DAY.fullmatch(approval["approved_on"]) is not None and _text(approval["note"], 2000)
            and isinstance(words, list) and 0 < len(words) <= 16 and all(_text(word, 500) for word in words)
            and all(_text(value, 2000) for key, value in approval.items() if key not in {"words"}))


def load_allowlist(settings: Settings) -> tuple[dict[str, Any], str]:
    """The reviewed allowlist and the sha256 of its exact bytes; any doubt refuses the whole run."""

    code = "reclaim_approved_folders_allowlist_invalid"
    try:
        fd = os.open(settings.allowlist, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ReclaimRefused(code)
            raw = stream.read(MAX_ALLOWLIST_BYTES + 1)
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ReclaimRefused(code) from error
    if len(raw) > MAX_ALLOWLIST_BYTES or not isinstance(document, dict):
        raise ReclaimRefused(code)
    if set(document) != {"schema_version", "parent", "folders"} or document["schema_version"] != ALLOWLIST_SCHEMA \
            or document["parent"] != str(CONTROL_PLANE):
        raise ReclaimRefused(code)
    folders = document["folders"]
    if not isinstance(folders, list) or not 0 < len(folders) <= MAX_FOLDERS:
        raise ReclaimRefused(code)
    for entry in folders:
        if not isinstance(entry, dict) or set(entry) - {"reviewed"} != {"path", "approval", "refuse_if_modified_after"}:
            raise ReclaimRefused(code)
        if not isinstance(entry["path"], str) or not _valid_approval(entry["approval"]):
            raise ReclaimRefused(code)
        bound = entry["refuse_if_modified_after"]
        if not isinstance(bound, str) or not _INSTANT.fullmatch(bound):
            raise ReclaimRefused(code)
        try:
            _instant(bound)
        except ValueError as error:
            raise ReclaimRefused(code) from error
        reviewed = entry.get("reviewed", {})
        if not isinstance(reviewed, dict) or not all(
                isinstance(value, (str, int)) and not isinstance(value, bool) for value in reviewed.values()):
            raise ReclaimRefused(code)
    if len({entry["path"] for entry in folders}) != len(folders):
        raise ReclaimRefused(code)
    return document, "sha256:" + hashlib.sha256(raw).hexdigest()


# --- the checks ------------------------------------------------------------------------------------

def _shape_refusal(raw: str) -> str | None:
    """A direct child of the control-plane state directory, by a plain name, or the refusal."""

    path = PurePosixPath(raw)
    if not path.is_absolute() or str(path) != raw or path.parent != CONTROL_PLANE or not _NAME.fullmatch(path.name):
        return "reclaim_approved_folders_path_outside_parent"
    return None


def _matches(pattern_parts: Sequence[str], parts: Sequence[str]) -> bool:
    return all(fnmatch.fnmatchcase(part, pattern) if "*" in pattern else part == pattern
               for pattern, part in zip(pattern_parts, parts))


def protected_storage_root(raw: str) -> StorageRoot | None:
    """The classified root this path is, lies in, or contains; None for an unclassified folder.

    A ``container`` root only refuses a path it is, or that contains it: every folder sits in one."""

    parts = PurePosixPath(raw).parts
    found = []
    for root in STORAGE_ROOTS:
        root_parts = PurePosixPath(root.path).parts
        if len(root_parts) < len(parts):
            if root.storage_class != "container" and _matches(root_parts, parts[:len(root_parts)]):
                found.append(root)
        elif _matches(root_parts[:len(parts)], parts):
            found.append(root)
    if not found:
        return None
    return next((root for root in found if root.path == raw), found[0])


def _walk(fd: int, device: int, depth: int, tally: _Tally) -> None:
    """Measure the tree under ``fd`` without following a link or leaving its filesystem."""

    try:
        names = sorted(os.listdir(fd))
    except OSError as error:
        raise ReclaimRefused("reclaim_approved_folders_unreadable") from error
    for name in names:
        try:
            info = _stat(name, fd)
        except OSError as error:
            raise ReclaimRefused("reclaim_approved_folders_unreadable") from error
        tally.add(info)
        if not stat.S_ISDIR(info.st_mode):
            continue
        if info.st_dev != device:
            raise ReclaimRefused("reclaim_approved_folders_mount_point_inside")
        if depth >= MAX_DEPTH:
            raise ReclaimRefused("reclaim_approved_folders_too_deep")
        if not _removable(name, fd):
            tally.blocked_directories += 1
        try:
            child = _open_directory(name, dir_fd=fd)
        except OSError as error:
            raise ReclaimRefused("reclaim_approved_folders_unreadable") from error
        try:
            opened = os.fstat(child)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ReclaimRefused("reclaim_approved_folders_changed_while_checked")
            _walk(child, device, depth + 1, tally)
        finally:
            os.close(child)


def _row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": entry["path"], "approval": entry["approval"],
            "refuse_if_modified_after": entry["refuse_if_modified_after"], "exists": None, "state": None,
            "refusal": None, "storage_root": None, "measured": None, "outcome": None}


def check_folder(entry: Mapping[str, Any], settings: Settings, mounts: list[str] | None) -> dict[str, Any]:
    """Every check for one listed folder; the row carries the refusal instead of raising it."""

    row = _row(entry)
    try:
        _check(entry, settings, mounts, row)
    except ReclaimRefused as refusal:
        row["refusal"] = refusal.code
    if row["refusal"] is not None:
        row["outcome"] = "refused"
    return row


def _check(entry: Mapping[str, Any], settings: Settings, mounts: list[str] | None, row: dict[str, Any]) -> None:
    raw = entry["path"]
    shape = _shape_refusal(raw)
    if shape is not None:
        raise ReclaimRefused(shape)
    root = protected_storage_root(raw)
    if root is not None:
        row["storage_root"] = {"path": root.path, "storage_class": root.storage_class, "note": root.note}
        raise ReclaimRefused("reclaim_approved_folders_protected_storage_root")
    name = PurePosixPath(raw).name
    try:
        parent_fd = _open_path(settings.parent)
    except OSError as error:
        raise ReclaimRefused("reclaim_approved_folders_parent_unsafe") from error
    try:
        parent_info = os.fstat(parent_fd)
        try:
            info = _stat(name, parent_fd)
        except FileNotFoundError:
            row.update(exists=False, state="absent", outcome="already_reclaimed")
            return
        except OSError as error:
            raise ReclaimRefused("reclaim_approved_folders_unreadable") from error
        row.update(exists=True, state="present")
        if stat.S_ISLNK(info.st_mode):
            raise ReclaimRefused("reclaim_approved_folders_symlink")
        if not stat.S_ISDIR(info.st_mode):
            raise ReclaimRefused("reclaim_approved_folders_not_a_directory")
        if info.st_dev != parent_info.st_dev:
            raise ReclaimRefused("reclaim_approved_folders_mount_point")
        folder = PurePosixPath(str(settings.parent)) / name
        if mounts is not None and any(PurePosixPath(point) == folder or folder in PurePosixPath(point).parents
                                      for point in mounts):
            raise ReclaimRefused("reclaim_approved_folders_mount_point_inside")
        row["mount_table_read"] = mounts is not None
        tally = _Tally()
        tally.add(info)
        if not _removable(".", parent_fd) or not _removable(name, parent_fd):
            tally.blocked_directories += 1
        try:
            top = _open_directory(name, dir_fd=parent_fd)
        except OSError as error:
            raise ReclaimRefused("reclaim_approved_folders_unreadable") from error
        try:
            opened = os.fstat(top)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ReclaimRefused("reclaim_approved_folders_changed_while_checked")
            _walk(top, info.st_dev, 1, tally)
        finally:
            os.close(top)
        row["measured"] = {**tally.measured(), "dev": info.st_dev, "ino": info.st_ino,
                           "blocked_directories": tally.blocked_directories}
        if tally.newest_file_mtime is not None and tally.newest_file_mtime > _instant(
                entry["refuse_if_modified_after"]):
            raise ReclaimRefused("reclaim_approved_folders_modified_after_approval")
        if tally.blocked_directories:
            raise ReclaimRefused("reclaim_approved_folders_not_removable")
        row["outcome"] = "would_reclaim"
    finally:
        os.close(parent_fd)


# --- deleting --------------------------------------------------------------------------------------

def _remove_tree(fd: int, device: int, tally: _Tally) -> None:
    """Remove everything under ``fd``: links are unlinked, never followed; no other filesystem is entered."""

    for name in sorted(os.listdir(fd)):
        info = _stat(name, fd)
        if stat.S_ISDIR(info.st_mode):
            if info.st_dev != device:
                raise ReclaimRefused("reclaim_approved_folders_mount_point_inside")
            child = _open_directory(name, dir_fd=fd)
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ReclaimRefused("reclaim_approved_folders_changed_while_checked")
                _remove_tree(child, device, tally)
            finally:
                os.close(child)
            os.rmdir(name, dir_fd=fd)
        else:
            os.unlink(name, dir_fd=fd)
        tally.add(info)


def _delete_folder(row: dict[str, Any], settings: Settings, tally: _Tally) -> None:
    name = PurePosixPath(row["path"]).name
    parent_fd = _open_path(settings.parent)
    try:
        info = _stat(name, parent_fd)
        measured = row["measured"]
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (measured["dev"], measured["ino"]):
            raise ReclaimRefused("reclaim_approved_folders_changed_while_checked")
        top = _open_directory(name, dir_fd=parent_fd)
        try:
            opened = os.fstat(top)
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ReclaimRefused("reclaim_approved_folders_changed_while_checked")
            _remove_tree(top, info.st_dev, tally)
        finally:
            os.close(top)
        os.rmdir(name, dir_fd=parent_fd)
        tally.add(info)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _receipts_fd(settings: Settings) -> int:
    """The kept receipts directory, made (0755) where missing below the parent; never through a link."""

    target = settings.receipts_dir
    base = settings.parent
    if base not in target.parents:
        raise ReclaimRefused("reclaim_approved_folders_receipts_unavailable")
    try:
        descriptor = _open_path(base)
    except OSError as error:
        raise ReclaimRefused("reclaim_approved_folders_receipts_unavailable") from error
    try:
        for part in target.relative_to(base).parts:
            try:
                child = _open_directory(part, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, DIRECTORY_MODE, dir_fd=descriptor)
                child = _open_directory(part, dir_fd=descriptor)
                os.fchmod(child, DIRECTORY_MODE)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ReclaimRefused("reclaim_approved_folders_receipts_unavailable") from error
    except BaseException:
        os.close(descriptor)
        raise


def _write_receipt(directory_fd: int, name: str, document: Mapping[str, Any]) -> None:
    raw = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    temporary = f".{name}.{secrets.token_hex(4)}.partial"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                 dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(fd, view):]
        os.fchmod(fd, FILE_MODE)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        # A link never replaces an existing name.
        os.link(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
    finally:
        os.unlink(temporary, dir_fd=directory_fd)
    os.fsync(directory_fd)


# --- plan and apply --------------------------------------------------------------------------------

def _check_all(allowlist: Mapping[str, Any], settings: Settings, report: dict[str, Any]) -> list[dict[str, Any]]:
    mounts = _mount_points()
    rows = [check_folder(entry, settings, mounts) for entry in allowlist["folders"]]
    report["folders"] = rows
    present = [row for row in rows if row["state"] == "present" and row["measured"]]
    report["totals"] = {
        "folders": len(rows),
        "present": sum(1 for row in rows if row["state"] == "present"),
        "absent": sum(1 for row in rows if row["state"] == "absent"),
        "refused": sum(1 for row in rows if row["refusal"]),
        "total_bytes": sum(row["measured"]["total_bytes"] for row in present),
        "allocated_bytes": sum(row["measured"]["allocated_bytes"] for row in present),
        "file_count": sum(row["measured"]["file_count"] for row in present),
    }
    refused = next((row["refusal"] for row in rows if row["refusal"]), None)
    if refused is not None:
        raise ReclaimRefused(refused)
    return rows


def _plan(allowlist: Mapping[str, Any], settings: Settings, report: dict[str, Any]) -> str:
    try:
        report["disk"] = {"free_bytes": _disk_free(settings.parent)}
    except OSError:
        report["disk"] = {"free_bytes": None}
    _check_all(allowlist, settings, report)
    return "planned"


def _apply(allowlist: Mapping[str, Any], settings: Settings, report: dict[str, Any]) -> str:
    receipts_fd = _receipts_fd(settings)
    try:
        try:
            fcntl.flock(receipts_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReclaimRefused("reclaim_approved_folders_busy") from error
        rows = _check_all(allowlist, settings, report)
        present = [row for row in rows if row["state"] == "present"]
        if not present:
            report["disk"] = {"free_bytes": _disk_free(settings.parent)}
            return "already_reclaimed"
        started = _now()
        report["started_at"] = _iso(started)
        before = _disk_free(settings.parent)
        report["disk"] = {"free_bytes_before": before, "free_bytes_after": None, "freed_bytes": None}
        failure: BaseException | None = None
        for row in present:
            tally = _Tally()
            try:
                _delete_folder(row, settings, tally)
                row.update(outcome="reclaimed", exists=False, removed_at=_iso(_now()))
            except BaseException as error:  # noqa: BLE001 - recorded in the receipt, then raised
                row.update(outcome="incomplete", error_type=type(error).__name__,
                           error_code=error.code if isinstance(error, ReclaimRefused) else None)
                failure = error
            finally:
                row["removed"] = tally.removed()
            if failure is not None:
                break
        for row in present:
            row.setdefault("removed", None)
            if row["outcome"] == "would_reclaim":
                row["outcome"] = "not_started"
        after = _disk_free(settings.parent)
        report["disk"].update(free_bytes_after=after, freed_bytes=after - before)
        report["removed"] = {key: sum((row["removed"] or {}).get(key, 0) for row in present)
                             for key in ("total_bytes", "allocated_bytes", "file_count", "directory_count",
                                         "symlink_count", "other_count")}
        report["finished_at"] = _iso(_now())
        outcome = "reclaimed" if failure is None else "refused"
        report["outcome"] = outcome
        if failure is not None:
            report["code"] = "reclaim_approved_folders_delete_incomplete"
        name = f"{RECEIPT_PREFIX}{_stamp(started)}-{secrets.token_hex(4)}.json"
        report["receipt_path"] = str(settings.receipts_dir / name)
        try:
            _write_receipt(receipts_fd, name, report)
        except OSError as error:
            # The deletion stands; the door's result is still the full record.
            report["receipt_path"] = None
            report["receipt_flag"] = {"written": False, "error_type": type(error).__name__}
        if failure is not None:
            if isinstance(failure, Exception):
                raise ReclaimRefused("reclaim_approved_folders_delete_incomplete") from failure
            raise failure
        return "reclaimed"
    finally:
        os.close(receipts_fd)


def _base_report(settings: Settings, apply: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(_now()),
        "mode": "apply" if apply else "plan",
        "outcome": None,
        "code": None,
        "allowlist": {"path": str(settings.allowlist), "sha256": None, "folder_count": None},
        "parent": str(CONTROL_PLANE),
        "folders": [],
        "totals": None,
        "disk": None,
        "receipts_dir": str(settings.receipts_dir),
        "receipt_path": None,
        "network_used": False,
        "provider_mutation_performed": False,
        "compute_started": False,
        "file_contents_recorded": False,
    }


def reclaim(*, apply: bool = False, settings: Settings | None = None) -> dict[str, Any]:
    """Plan (default) or apply the reclaim; always returns the result document."""

    settings = settings or SETTINGS
    report = _base_report(settings, apply)
    try:
        allowlist, digest = load_allowlist(settings)
        report["allowlist"].update(sha256=digest, folder_count=len(allowlist["folders"]))
        outcome = _apply(allowlist, settings, report) if apply else _plan(allowlist, settings, report)
    except ReclaimRefused as refusal:
        report.update(outcome="refused", code=refusal.code)
        return report
    except Exception as error:  # noqa: BLE001 - one fixed code, never the message
        report.update(outcome="refused", code="reclaim_approved_folders_failed", error_type=type(error).__name__)
        return report
    report["outcome"] = outcome
    return report


def _terminate(_signum: int, _frame: Any) -> None:
    # The door's RuntimeMaxSec sends SIGTERM: unwind so what was removed is still recorded.
    raise SystemExit(143)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="delete; without this only a plan is made")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    document = reclaim(apply=args.apply)
    print(json.dumps(document, sort_keys=True))
    if document["outcome"] != "refused":
        return 0
    return 1 if document["code"] in _EXIT_ONE else 2


if __name__ == "__main__":
    sys.exit(main())
