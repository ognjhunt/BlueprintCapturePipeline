"""Plan, or with ``--apply`` empty, one owner-approved old experiment folder on the host.

The owner approved (2026-11-18) permanently deleting old experiment folders that nothing
uses, to free disk for new jobs. Only the folders named in ``FOLDERS`` can be named: the
door's request schema and this module both check the name, and the path is built here,
never taken from the request. ``g1-841757-development`` was in the owner's list but is
not here: ``config/native_g1_shared_inputs.v1.json`` still points at it for the G1
humanoid trial, so it is in use.

The folder is opened once (no symlink anywhere on its path, the opened directory the one
checked), and everything after works through that handle and handles below it, never by
looking a path up again: renaming or replacing the folder's name mid-run cannot redirect
the deletion. A plan reports the folder's size and what would stop an apply. An apply
first surveys the whole tree (any part that cannot be read stops it), then proves that:

- no running process, in any mount namespace, has a file, working directory (any thread's)
  or mapping that is one of the tree's own files or directories, matched by device and
  inode as well as by path;
- no mount in any namespace other than this unit's own shows the folder or anything under
  it (a bind mount into a container included), and no host mount point is under it;
- no file under it was modified in the last ``QUIET_SECONDS`` (by modification time: a
  copy that kept old times, such as ``cp -a`` or ``rsync -a``, is not caught);
- everything under it is on the folder's own filesystem.

It then writes a receipt naming every top-level entry and its size, deletes without
following symlinks, keeps going past an entry it cannot delete, and writes the receipt
again with the space freed and anything left. The empty folder stays: the door's sandbox
makes it a mount point. Nothing here can be restored. Only the standard library is used,
so the door runs it with ``python3 -I -S`` from its own install, independent of any release.
"""
from __future__ import annotations

import argparse
import datetime
import errno
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "blueprint_operator_door_experiment_purge.v1"
PARENT = "/var/lib/blueprint/pipeline-control-plane"
#: The only folders this can empty. Each is a direct child of ``PARENT``.
FOLDERS = frozenset({"atlas-beta-dishwasher-20261021", "flora-episode-benchmarks"})
APPROVAL = ("owner, 2026-11-18, answering 'Build a door command that permanently deletes exactly "
            "these three old experiment folders on the host (after I verify nothing uses them): "
            "atlas-beta-dishwasher-20261021, flora-episode-benchmarks, g1-841757-development "
            "(~60 GB)?' with 'Yes, delete those 3 (Recommended)'; g1-841757-development is kept "
            "because the G1 trial configuration still uses it")
#: A file written in the last week means something may still use the folder.
QUIET_SECONDS = 7 * 86400
_MAX_REPORTED = 20
_OCTAL = re.compile(r"\\([0-7]{3})")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


class Refused(Exception):
    """A typed reason this folder cannot be emptied now."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _unescape(field: str) -> str:
    # Mount table paths escape space, tab, newline and backslash as octal.
    return _OCTAL.sub(lambda match: chr(int(match.group(1), 8)), field)


def _device_text(device: int) -> str:
    return f"{os.major(device)}:{os.minor(device)}"


def _errno_name(error: OSError) -> str:
    return errno.errorcode.get(error.errno or 0) or "unknown"


# ------------------------------------------------------------------------------- the folder
def _open_folder(folder: str, parent: str) -> tuple[str, int, os.stat_result]:
    """The folder's path and a handle to it, after proving no symlink lies on the way."""
    if folder not in FOLDERS:
        raise Refused("folder_not_approved")
    root = os.path.join(parent, folder)
    current = "/"
    for part in Path(root).parts[1:]:
        current = os.path.join(current, part)
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            raise Refused("folder_absent") from None
        if stat.S_ISLNK(info.st_mode):
            raise Refused("folder_path_has_symlink")
        if not stat.S_ISDIR(info.st_mode):
            raise Refused("folder_not_directory")
    try:
        fd = os.open(root, _DIRECTORY_FLAGS)
    except FileNotFoundError:
        raise Refused("folder_absent") from None
    except OSError as error:
        raise Refused("folder_open_failed:" + _errno_name(error)) from None
    opened = os.fstat(fd)
    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
        os.close(fd)
        raise Refused("folder_changed_while_checked")
    return root, fd, opened


def _survey(fd: int) -> dict[str, Any]:
    """Size, counts, newest file time and every inode under the handle; any unreadable part refuses."""
    device = os.fstat(fd).st_dev
    totals = {"files": 0, "directories": 0, "symlinks": 0, "other_entries": 0,
              "bytes_apparent": 0, "bytes_on_disk": 0}
    newest: float | None = None
    other_filesystem = False
    inodes: set[int] = set()
    entries: list[dict[str, Any]] = []

    def unreadable(error: OSError) -> None:
        raise Refused("survey_incomplete:" + _errno_name(error))

    def count(name: str, dir_fd: int) -> int:
        nonlocal other_filesystem, newest
        try:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return 0
        if info.st_dev != device:
            other_filesystem = True
        inodes.add(info.st_ino)
        if stat.S_ISDIR(info.st_mode):
            totals["directories"] += 1
            return 0
        kind = ("symlinks" if stat.S_ISLNK(info.st_mode) else
                "files" if stat.S_ISREG(info.st_mode) else "other_entries")
        totals[kind] += 1
        totals["bytes_apparent"] += info.st_size
        totals["bytes_on_disk"] += info.st_blocks * 512
        newest = info.st_mtime if newest is None else max(newest, info.st_mtime)
        return info.st_blocks * 512

    inodes.add(os.fstat(fd).st_ino)  # a working directory at the folder itself
    for name in sorted(os.listdir(fd)):
        entry_bytes = count(name, fd)
        try:
            top = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(top.st_mode):
            # fwalk lists a symlink to a directory among its directories but never descends it.
            for _path, dirnames, filenames, walk_fd in os.fwalk(name, dir_fd=fd, follow_symlinks=False,
                                                                  onerror=unreadable):
                for child in dirnames + filenames:
                    entry_bytes += count(child, walk_fd)
        entries.append({"name": name, "bytes_on_disk": entry_bytes})
    return {**totals, "newest_file_mtime": newest, "other_filesystem": other_filesystem,
            "top_level_entries": entries, "inodes": inodes, "device": device}


# ------------------------------------------------------------------------------- what uses it
def _tasks(base: str) -> list[str]:
    """A process's thread directories (each has its own working directory), or the process itself."""
    try:
        return [os.path.join(base, "task", tid) for tid in sorted(os.listdir(os.path.join(base, "task")))]
    except (FileNotFoundError, NotADirectoryError):
        return [base]


def _holds(base: str, root: str, device: int, inodes: set[int]) -> str | None:
    """What a process holds in the tree: a thread's cwd, an open file or a mapping, else None."""
    def ours(link: str) -> bool:
        try:
            if _under(os.readlink(link), root):
                return True
            info = os.stat(link)  # the open file itself, in whatever namespace it was opened
        except (FileNotFoundError, ProcessLookupError):
            return False
        return info.st_dev == device and info.st_ino in inodes

    tasks = _tasks(base)
    for task in tasks:
        if ours(os.path.join(task, "cwd")):
            return "cwd"
    major_minor = f"{os.major(device):02x}:{os.minor(device):02x}"
    # Threads share their process's open files and memory: read the first thread that answers.
    for task in tasks:
        try:
            names = os.listdir(os.path.join(task, "fd"))
            with open(os.path.join(task, "maps"), encoding="utf-8", errors="replace") as stream:
                maps = stream.read()
        except (FileNotFoundError, ProcessLookupError):
            continue
        for name in names:
            if ours(os.path.join(task, "fd", name)):
                return "fd"
        for line in maps.splitlines():
            parts = line.split(None, 5)
            if len(parts) < 5:
                continue
            if parts[3] == major_minor and parts[4].isdigit() and int(parts[4]) in inodes:
                return "map"
            if len(parts) == 6 and _under(parts[5].removesuffix(" (deleted)"), root):
                return "map"
        return None
    return None


def _mount_root_of(root: str, host_lines: list[list[str]], device: int) -> str | None:
    """The folder's path inside its filesystem, from the host mount that holds it."""
    best: tuple[int, str] | None = None
    for fields in host_lines:
        point = _unescape(fields[4])
        if fields[2] == _device_text(device) and _under(root, point) and (best is None or len(point) > best[0]):
            relative = root[len(point.rstrip("/")):] if point != "/" else root
            best = (len(point), _unescape(fields[3]).rstrip("/") + relative)
    return None if best is None else best[1]


def _checks(root: str, survey: dict[str, Any], proc_root: str) -> dict[str, Any]:
    """Processes and mounts that use the tree, and how many could not be checked."""
    me = os.getpid()
    device, inodes = survey["device"], survey["inodes"]
    try:
        own_namespace = os.stat(os.path.join(proc_root, "self", "ns", "mnt")).st_ino
    except OSError:
        own_namespace = None
    try:
        with open(os.path.join(proc_root, "1", "mountinfo"), encoding="utf-8", errors="replace") as stream:
            host_lines = [fields for fields in (line.split() for line in stream) if len(fields) > 4]
    except OSError as error:
        raise Refused("mount_table_unreadable") from error
    folder_in_fs = _mount_root_of(root, host_lines, device)

    def binds_folder(fields: list[str]) -> bool:
        # The same filesystem mounted from the folder or below it: a bind mount of it.
        return (folder_in_fs is not None and fields[2] == _device_text(device)
                and _under(_unescape(fields[3]), folder_in_fs))

    mounts = [_unescape(fields[4]) for fields in host_lines
              if _under(_unescape(fields[4]), root) or binds_folder(fields)]
    try:
        pids = sorted((name for name in os.listdir(proc_root) if name.isdigit()), key=int)
    except OSError as error:
        raise Refused("process_table_unreadable") from error
    holders: list[dict[str, Any]] = []
    unreadable = 0
    seen_namespaces: set[Any] = set()
    for pid in pids:
        if int(pid) == me:
            continue
        base = os.path.join(proc_root, pid)
        try:
            try:
                namespace: Any = os.stat(os.path.join(base, "ns", "mnt")).st_ino
            except FileNotFoundError:
                namespace = ("pid", pid)
            if namespace not in seen_namespaces and namespace != own_namespace and pid != "1":
                seen_namespaces.add(namespace)
                try:
                    with open(os.path.join(base, "mountinfo"), encoding="utf-8", errors="replace") as stream:
                        for line in stream:
                            fields = line.split()
                            if len(fields) > 4 and binds_folder(fields):
                                mounts.append(f"pid {pid}: {_unescape(fields[4])}")
                except FileNotFoundError:
                    pass
            found = _holds(base, root, device, inodes)
        except (FileNotFoundError, ProcessLookupError):
            continue  # the process ended while it was read
        except OSError:  # PermissionError included: a process this could not check
            unreadable += 1
            continue
        if found is not None:
            try:
                with open(os.path.join(base, "comm"), encoding="utf-8", errors="replace") as stream:
                    comm = stream.read().strip()[:32]
            except OSError:
                comm = ""
            holders.append({"pid": int(pid), "process": comm, "holds": found})
    return {"mounts": mounts, "holders": holders, "unreadable": unreadable}


# ------------------------------------------------------------------------------- deletion
def _delete_entry(name: str, fd: int, failures: list[dict[str, str]]) -> None:
    """Delete one top-level entry through handles, bottom-up, never following a symlink."""
    def failed(where: str, error: OSError) -> None:
        if len(failures) < _MAX_REPORTED:
            failures.append({"entry": where, "error": _errno_name(error)})

    try:
        top = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        failed(name, error)
        return
    if not stat.S_ISDIR(top.st_mode):
        try:
            os.unlink(name, dir_fd=fd)
        except OSError as error:
            failed(name, error)
        return
    for path, dirnames, filenames, walk_fd in os.fwalk(name, dir_fd=fd, topdown=False, follow_symlinks=False,
                                                         onerror=lambda error: failed(name, error)):
        for child in filenames:
            try:
                os.unlink(child, dir_fd=walk_fd)
            except FileNotFoundError:
                continue
            except OSError as error:
                failed(os.path.join(path, child), error)
        for child in dirnames:
            try:
                if stat.S_ISLNK(os.stat(child, dir_fd=walk_fd, follow_symlinks=False).st_mode):
                    os.unlink(child, dir_fd=walk_fd)
                else:
                    os.rmdir(child, dir_fd=walk_fd)
            except FileNotFoundError:
                continue
            except OSError as error:
                failed(os.path.join(path, child), error)
    try:
        os.rmdir(name, dir_fd=fd)
    except OSError as error:
        failed(name, error)


# ------------------------------------------------------------------------------- the run
def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).isoformat(timespec="seconds")


def _write(result_out: str, document: dict[str, Any]) -> None:
    tmp = f"{result_out}.tmp.{os.getpid()}"
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, result_out)


def _free_bytes(fd: int) -> int:
    info = os.fstatvfs(fd)
    return info.f_bavail * info.f_frsize


def run(folder: str, apply: bool, result_out: str, request_id: str, *, parent: str = PARENT,
        proc_root: str = "/proc", now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    document: dict[str, Any] = {"schema": SCHEMA, "request_id": request_id, "folder": folder,
                                "mode": "apply" if apply else "plan", "approval": APPROVAL,
                                "checked_at": _iso(now)}
    fd = None
    try:
        root, fd, opened = _open_folder(folder, parent)
        document["path"] = root
        survey = _survey(fd)
        # After the survey, so they hold up to the moment the deletion starts.
        checks = _checks(root, survey, proc_root)
        newest = survey["newest_file_mtime"]
        stops = []
        if checks["mounts"]:
            stops.append("folder_has_mount_point")
        if checks["holders"]:
            stops.append("folder_in_use_by_process")
        if checks["unreadable"]:
            stops.append("process_check_incomplete")
        if survey["other_filesystem"]:
            stops.append("folder_spans_filesystems")
        if newest is not None and now - newest < QUIET_SECONDS:
            stops.append("folder_recently_modified")
        document.update({
            "mount_points_under": checks["mounts"][:_MAX_REPORTED],
            "processes_holding": checks["holders"][:_MAX_REPORTED],
            "processes_unreadable": checks["unreadable"],
            **{key: value for key, value in survey.items()
               if key not in {"newest_file_mtime", "inodes", "device"}},
            "newest_file_mtime": _iso(newest),
            "would_stop": stops,
            "free_bytes_before": _free_bytes(fd),
        })
        if not apply:
            document["outcome"] = "planned"
            _write(result_out, document)
            return document
        if stops:
            raise Refused(stops[0])
        if not survey["top_level_entries"]:
            document["outcome"] = "already_empty"
            _write(result_out, document)
            return document
        document["outcome"] = "emptying"
        _write(result_out, document)
        failures: list[dict[str, str]] = []
        for entry in survey["top_level_entries"]:
            _delete_entry(entry["name"], fd, failures)
        left = os.listdir(fd)
        document.update({"entries_left": len(left), "entries_left_names": sorted(left)[:_MAX_REPORTED],
                         "delete_failures": failures, "free_bytes_after": _free_bytes(fd),
                         "finished_at": _iso(time.time())})
        document["freed_bytes"] = document["free_bytes_after"] - document["free_bytes_before"]
        still = os.fstat(fd)
        if (still.st_dev, still.st_ino) != (opened.st_dev, opened.st_ino):
            document["code"] = "folder_handle_changed"
        if left or failures:
            document["outcome"] = "partly_emptied"
            document.setdefault("code", "delete_failed:" + (failures[0]["error"] if failures else "entries_left"))
        else:
            document["outcome"] = "emptied"
        _write(result_out, document)
        return document
    except Refused as refusal:
        document["outcome"] = "absent" if refusal.code == "folder_absent" else "refused"
        document["code"] = refusal.code
        _write(result_out, document)
        return document
    finally:
        if fd is not None:
            os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--result-out", required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args(argv)
    document = run(args.folder, args.apply, args.result_out, args.request_id)
    print(json.dumps({key: document.get(key) for key in ("outcome", "code", "folder", "bytes_on_disk",
                                                           "freed_bytes", "would_stop")}, sort_keys=True))
    return 0 if document["outcome"] in {"planned", "emptied", "already_empty", "absent"} else 1


if __name__ == "__main__":
    sys.exit(main())
