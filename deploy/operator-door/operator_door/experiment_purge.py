"""Plan, or with ``--apply`` empty, one owner-approved old experiment folder on the host.

The owner approved (2026-11-18) permanently deleting old experiment folders that nothing
uses, to free disk for new jobs. Only the folders named in ``FOLDERS`` can be named: the
door's request schema and this module both check the name, and the path is built here,
never taken from the request. ``g1-841757-development`` was in the owner's list but is
not here: ``config/native_g1_shared_inputs.v1.json`` still points at it for the G1
humanoid trial, so it is in use.

A plan reports the folder's size and what would stop an apply. An apply proves again that:

- the path is a real directory at its expected place (no symlink anywhere on the way);
- no mount point is at or under it (read from the host's mount table);
- no running process has a file, working directory or mapping under it;
- no file under it was modified in the last ``QUIET_SECONDS``;
- everything under it is on the same filesystem as the folder.

It then writes a receipt naming every top-level entry and its size, deletes the folder's
contents without following symlinks, and writes the receipt again with the space freed.
The empty folder itself stays: the door's sandbox makes it a mount point, which cannot be
removed from inside. Nothing here can be restored. Only the standard library is used, so
the door runs it with ``python3 -I -S`` from its own install, independent of any release.
"""
from __future__ import annotations

import argparse
import datetime
import errno
import json
import os
import re
import shutil
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
_MAX_HANDLES_REPORTED = 20
_OCTAL = re.compile(r"\\([0-7]{3})")


class Refused(Exception):
    """A typed reason this folder cannot be emptied now."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _mount_points(proc_root: str) -> list[str]:
    """Mount points in the host's namespace (pid 1), not this sandbox's own binds."""
    points = []
    try:
        with open(os.path.join(proc_root, "1", "mountinfo"), encoding="utf-8", errors="replace") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) > 4:
                    # Mount points escape space, tab, newline and backslash as octal.
                    points.append(_OCTAL.sub(lambda match: chr(int(match.group(1), 8)), fields[4]))
    except OSError as error:
        raise Refused("mount_table_unreadable") from error
    return points


def _open_handles(root: str, proc_root: str) -> tuple[list[dict[str, Any]], int]:
    """Processes holding something under ``root``, and how many processes could not be read."""
    holders: list[dict[str, Any]] = []
    unreadable = 0
    me = os.getpid()
    try:
        pids = [name for name in os.listdir(proc_root) if name.isdigit()]
    except OSError as error:
        raise Refused("process_table_unreadable") from error
    for pid in pids:
        if int(pid) == me:
            continue
        base = os.path.join(proc_root, pid)
        found = None
        try:
            if _under(os.readlink(os.path.join(base, "cwd")), root):
                found = "cwd"
            if found is None:
                fd_dir = os.path.join(base, "fd")
                for fd in os.listdir(fd_dir):
                    try:
                        if _under(os.readlink(os.path.join(fd_dir, fd)), root):
                            found = "fd"
                            break
                    except FileNotFoundError:
                        continue
            if found is None:
                with open(os.path.join(base, "maps"), encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        parts = line.split(None, 5)
                        if len(parts) == 6 and _under(parts[5].rstrip("\n").removesuffix(" (deleted)"), root):
                            found = "map"
                            break
        except FileNotFoundError:
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
    return holders, unreadable


def _survey(root: str) -> dict[str, Any]:
    """Size, counts and newest file time under ``root``, never following a symlink."""
    device = os.lstat(root).st_dev
    totals = {"files": 0, "directories": 0, "symlinks": 0, "other_entries": 0,
              "bytes_apparent": 0, "bytes_on_disk": 0}
    newest: list[float] = []
    other_filesystem = False
    entries: list[dict[str, Any]] = []

    def count(path: str) -> int:
        nonlocal other_filesystem
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return 0
        if info.st_dev != device:
            other_filesystem = True
        if stat.S_ISDIR(info.st_mode):
            totals["directories"] += 1
            return 0
        kind = ("symlinks" if stat.S_ISLNK(info.st_mode) else
                "files" if stat.S_ISREG(info.st_mode) else "other_entries")
        totals[kind] += 1
        totals["bytes_apparent"] += info.st_size
        totals["bytes_on_disk"] += info.st_blocks * 512
        newest[:] = [max(newest[0], info.st_mtime)] if newest else [info.st_mtime]
        return info.st_blocks * 512

    for name in sorted(os.listdir(root)):
        entry_path = os.path.join(root, name)
        entry_bytes = count(entry_path)
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            # os.walk lists a symlink to a directory among dirnames and does not descend it.
            for dirpath, dirnames, filenames in os.walk(entry_path, followlinks=False):
                for child in dirnames + filenames:
                    entry_bytes += count(os.path.join(dirpath, child))
        entries.append({"name": name, "bytes_on_disk": entry_bytes})
    return {**totals, "newest_file_mtime": newest[0] if newest else None,
            "other_filesystem": other_filesystem, "top_level_entries": entries}


def _iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).isoformat(timespec="seconds")


def _write(result_out: str, document: dict[str, Any]) -> None:
    tmp = f"{result_out}.tmp.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, result_out)


def _free_bytes(path: str) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def _check_path(folder: str, parent: str) -> str:
    if folder not in FOLDERS:
        raise Refused("folder_not_approved")
    root = os.path.join(parent, folder)
    # Every component from / to the folder must be a real directory: no symlink can redirect it.
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
    if os.path.realpath(root) != root:
        raise Refused("folder_path_not_canonical")
    return root


def run(folder: str, apply: bool, result_out: str, request_id: str, *, parent: str = PARENT,
        proc_root: str = "/proc", now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    document: dict[str, Any] = {"schema": SCHEMA, "request_id": request_id, "folder": folder,
                                "mode": "apply" if apply else "plan", "approval": APPROVAL,
                                "checked_at": _iso(now)}
    try:
        root = _check_path(folder, parent)
        document["path"] = root
        mounts = [point for point in _mount_points(proc_root) if _under(point, root)]
        holders, unreadable = _open_handles(root, proc_root)
        survey = _survey(root)
        newest = survey["newest_file_mtime"]
        stops = []
        if mounts:
            stops.append("folder_has_mount_point")
        if holders:
            stops.append("folder_in_use_by_process")
        if unreadable:
            stops.append("process_check_incomplete")
        if survey["other_filesystem"]:
            stops.append("folder_spans_filesystems")
        if newest is not None and now - newest < QUIET_SECONDS:
            stops.append("folder_recently_modified")
        document.update({
            "mount_points_under": mounts[:_MAX_HANDLES_REPORTED],
            "processes_holding": holders[:_MAX_HANDLES_REPORTED],
            "processes_unreadable": unreadable,
            **{key: value for key, value in survey.items() if key != "newest_file_mtime"},
            "newest_file_mtime": _iso(newest),
            "would_stop": stops,
            "free_bytes_before": _free_bytes(root),
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
        if not shutil.rmtree.avoids_symlink_attacks:
            raise Refused("rmtree_not_symlink_safe")
        try:
            for entry in survey["top_level_entries"]:
                entry_path = os.path.join(root, entry["name"])
                try:
                    info = os.lstat(entry_path)
                except FileNotFoundError:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(entry_path)
                else:
                    os.unlink(entry_path)
        except OSError as error:
            document["code"] = "delete_failed:" + (errno.errorcode.get(error.errno or 0) or "unknown")
        left = os.listdir(root)
        document["entries_left"] = len(left)
        document["free_bytes_after"] = _free_bytes(root)
        document["freed_bytes"] = document["free_bytes_after"] - document["free_bytes_before"]
        document["finished_at"] = _iso(time.time())
        document["outcome"] = "emptied" if not left else "partly_emptied"
        _write(result_out, document)
        return document
    except Refused as refusal:
        document["outcome"] = "absent" if refusal.code == "folder_absent" else "refused"
        document["code"] = refusal.code
        _write(result_out, document)
        return document


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
