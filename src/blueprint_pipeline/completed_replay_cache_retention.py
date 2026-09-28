"""Reclaim disposable binary copies after an offline replay has completed.

By default only a finished replay's writable binary working copies go (a known
binary suffix, at least 64 KiB, one link); reports, logs, source code, readonly
files and shared inodes remain untouched.

``reclaim_store_copies`` (storage GC's opt-in; the standalone unit never passes
it) also recognises, by place and name instead of suffix, a parent replay's copy
of a content-store blob at ``prepared-references/content-addressed/sha256/<digest>``,
with every other name it has in the replay's ``prepared-references`` (the
worker's materialized references are hard links to it). It keeps the store's
read-only mode and has no suffix, so the whole inode qualifies when all of its
links are there and its bytes match the digest it is named by; a link anywhere
else keeps it. Under this opt-in any finished parent replay counts, whatever its
status.

``reclaim_scratch_inputs`` (storage GC's alone as well) goes further for a
finished parent replay whose report says it ran in that root. The replay made
``prepared-references`` in its own temporary root, so every regular file there
whose links are all inside it and that is not newer than the report is scratch,
digest-named or not, and goes with every name; the directories left empty inside
the tree go after it, and the tree itself stays. It subsumes the store-copy rule
for that replay, so no inode is counted twice.

Both of those rules stay on the replay child's own filesystem: nothing whose
st_dev differs from the child's is planned, unlinked or pruned. A bind mount of
the same filesystem keeps its st_dev and cannot be told apart that way. Without
the opt-ins the rules are the ones this module always had.

Apply rechecks and unlinks every name, and hashes what a rule's digest rests on,
through directory descriptors held from the replay child down, never through a
path. Under every rule it skips an item whose directory moved or became a link
since it was opened, and one on another device than the child (cross_device),
whether its directory lists it there or it is opened there.

This module is also a standalone maintenance entrypoint; it never allocates a
provider or changes the scientific release used by a live run.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

SCHEMA = "completed_offline_replay_cache_retention.v1"
ACK = "reclaim-completed-offline-replay-caches"
_SCRATCH_INPUTS = "prepared-references"
_SCRATCH_STORE = (_SCRATCH_INPUTS, "content-addressed", "sha256")
_DIGEST_NAME = re.compile(r"[0-9a-f]{64}")
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".ply",
    ".bin",
    ".npy",
    ".npz",
    ".pt",
    ".zip",
    ".whl",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".nurec",
    ".so",
}


def digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def file_sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def scratch_store_copy(relative):
    """Whether ``relative`` (to a replay child) names a parent replay's copy of a store blob."""
    parts = Path(relative).parts
    return len(parts) == 4 and parts[:3] == _SCRATCH_STORE and bool(_DIGEST_NAME.fullmatch(parts[3]))


def _no_linked_parent(path, child):
    return not any(p.is_symlink() for p in path.parents if p != child.parent)


def _directory_on(path, device):
    """Whether ``path`` is a directory, not a link, on ``device``."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_dev == device


def _inode_groups(child):
    """Every regular file under the child's ``prepared-references`` with no linked parent, on the
    child's own filesystem, grouped by inode from names and metadata alone:
    ``{(dev, ino): (lstat, [names relative to the child])}``.

    The walk never leaves the child's device: a directory whose st_dev differs (a mount point)
    is not entered, and a file whose st_dev differs is skipped. A bind mount of the same
    filesystem keeps its st_dev, so it cannot be told apart this way. A child that is gone by
    the time it is looked at has no groups.
    """
    subtree = child / _SCRATCH_INPUTS
    groups = {}
    try:
        device = os.lstat(child).st_dev
    except FileNotFoundError:
        return groups
    if not _directory_on(subtree, device):
        return groups
    for directory, directories, names in os.walk(subtree):
        directories[:] = [name for name in directories if _directory_on(Path(directory) / name, device)]
        for name in names:
            path = Path(directory) / name
            info = os.lstat(path)
            if stat.S_ISREG(info.st_mode) and info.st_dev == device and _no_linked_parent(path, child):
                groups.setdefault((info.st_dev, info.st_ino), (info, []))[1].append(path.relative_to(child))
    return groups


def _group(info, names):
    """A plan's record of one inode and every name it has in the scratch inputs."""
    return {
        "relative_paths": sorted(str(name) for name in names),
        "inode": info.st_ino,
        "dev": info.st_dev,
        "nlink": info.st_nlink,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def _store_copies(child, report_mtime_ns):
    """Candidate copies of store blobs in the child's scratch inputs, each with every name it has there.

    A group of ``_inode_groups`` is a candidate only when all of its links are in that
    subtree, one of its names is a store name, and it is not newer than the report;
    ``_verified`` then requires its bytes to hash to that name. Also returns every inode
    that has a store name, so the single-file rules never plan one of its names.
    """
    copies, store_inodes = [], set()
    for key, (info, names) in _inode_groups(child).items():
        store_names = [name for name in names if scratch_store_copy(name)]
        if not store_names:
            continue
        store_inodes.add(key)
        if len(names) != info.st_nlink or info.st_mtime_ns > report_mtime_ns:
            continue
        copies.append(_group(info, names))
    return sorted(copies, key=lambda copy: copy["relative_paths"]), store_inodes


def _scratch_inputs(child, report_mtime_ns):
    """Every group of names a finished parent replay left in its scratch inputs, and every inode there.

    ``replay_parent`` made ``prepared-references`` inside its own temporary root and now
    releases it when it returns, so what an older replay left there is scratch whatever its
    name or bytes: a group of ``_inode_groups`` is a candidate when all of its links are in
    that subtree and it is not newer than the report. No digest is needed and none is read.
    A link anywhere else keeps the inode. Every inode seen there is returned too, so the
    single-file rules never plan one of its names.
    """
    groups = _inode_groups(child)
    inputs = [
        _group(info, names)
        for info, names in groups.values()
        if len(names) == info.st_nlink and info.st_mtime_ns <= report_mtime_ns
    ]
    return sorted(inputs, key=lambda group: group["relative_paths"]), set(groups)


def _verified(child, files, copies):
    """Hash each candidate: a file records its digest, and a store copy stays only when its
    bytes hash to the store name it carries."""
    files = [{**entry, "sha256": file_sha(child / entry["relative_path"])} for entry in files]
    verified = []
    for copy in copies:
        sha = file_sha(child / copy["relative_paths"][0])
        if any(scratch_store_copy(name) and sha == "sha256:" + Path(name).name for name in copy["relative_paths"]):
            verified.append({**copy, "sha256": sha})
    return files, verified


def _single_files(child, report_mtime_ns, store_inodes):
    files = []
    for path in child.rglob("*"):
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) in store_inodes
            or info.st_nlink != 1
            or not info.st_mode & 0o222
            or info.st_size < 64 * 1024
            or info.st_mtime_ns > report_mtime_ns
            or path.suffix.lower() not in BINARY_SUFFIXES
            or not _no_linked_parent(path, child)
        ):
            continue
        files.append(
            {
                "relative_path": str(path.relative_to(child)),
                "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns,
                "size_bytes": info.st_size,
            }
        )
    return files


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
# O_NONBLOCK: a FIFO swapped in after a leaf's recheck opens without waiting for a writer, and
# the identity check then refuses it. It changes nothing for a regular file.
_LEAF_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


class _HeldChild:
    """Directory descriptors held from a replay child down to the directories an item's names are in.

    Apply rechecks, hashes and unlinks through them and never through a path. Each one was
    opened a single O_NOFOLLOW component at a time from the replay root, so a directory
    swapped for a symlink while apply runs cannot redirect a removal outside the replay.
    Only the root and the child stay open for a whole row: ``item`` closes the directories
    below the child once each item is done, so a row across any number of directories holds
    no more descriptors than one item needs. ``device`` is the held child's own filesystem
    (fstat of its descriptor); nothing on any other is unlinked or pruned.
    """

    def __init__(self, base, name):
        self._name = name
        self._base = os.open(os.fspath(base), _DIRECTORY_FLAGS)
        self._held = {}
        try:
            self._held[()] = os.open(name, _DIRECTORY_FLAGS, dir_fd=self._base)
            self.device = os.fstat(self._held[()]).st_dev
        except OSError:
            self.close()
            raise

    def directory(self, parts):
        if parts not in self._held:
            parent = self.directory(parts[:-1])
            self._held[parts] = os.open(parts[-1], _DIRECTORY_FLAGS, dir_fd=parent)
        return self._held[parts]

    def item(self, remove, *args):
        """``remove(self, *args)`` for one item, then close every directory it opened below the child."""
        try:
            return remove(self, *args)
        finally:
            self._release()

    def _release(self):
        for parts in [parts for parts in self._held if parts]:
            os.close(self._held.pop(parts))

    def named_by(self, path):
        """Whether ``path``, through no link, still names the child held.

        The root and child are opened by path after apply checks that path, so an ancestor
        swapped for a link in between would re-root every descriptor held; one swapped back
        since then leaves the path naming a different directory than the one held.
        """
        try:
            entry, held = os.lstat(path), os.fstat(self._held[()])
        except OSError:
            return False
        return (
            stat.S_ISDIR(entry.st_mode)
            and (entry.st_dev, entry.st_ino) == (held.st_dev, held.st_ino)
            and not any(p.is_symlink() for p in (path, *path.parents))
        )

    def in_place(self, parts):
        """Whether each held directory from the child down to ``parts`` is still the entry
        its parent names: a directory moved or swapped for a link since it was opened is not."""
        chain = [(self._base, self._name, self._held[()])] + [
            (self._held[parts[:index]], parts[index], self._held[parts[:index + 1]])
            for index in range(len(parts))
        ]
        for parent, name, fd in chain:
            entry, held = os.stat(name, dir_fd=parent, follow_symlinks=False), os.fstat(fd)
            if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (held.st_dev, held.st_ino):
                return False
        return True

    def close(self):
        self._release()
        if () in self._held:
            os.close(self._held.pop(()))
        os.close(self._base)


def _leaf(directory, name):
    return os.stat(name, dir_fd=directory, follow_symlinks=False)


def _same_leaf(directory, name, expected):
    entry = _leaf(directory, name)
    return stat.S_ISREG(entry.st_mode) and (entry.st_dev, entry.st_ino) == (expected.st_dev, expected.st_ino)


class _CrossDevice(Exception):
    """What apply opened is on another filesystem than the held replay child: a ``cross_device`` skip."""


def _held_sha(directory, name, expected):
    """Hash ``name`` opened O_NOFOLLOW in its held directory, or None if it is not ``expected``.

    Every caller has already required ``expected`` (the leaf's lstat) to be on the held
    child's device, so an opened file on any other device than ``expected``'s raises
    ``_CrossDevice``: something from another filesystem took the leaf's place.
    """
    fd = os.open(name, _LEAF_FLAGS, dir_fd=directory)
    try:
        opened = os.fstat(fd)
        if opened.st_dev != expected.st_dev:
            raise _CrossDevice(name)
        if not stat.S_ISREG(opened.st_mode) or opened.st_ino != expected.st_ino:
            return None
        with open(fd, "rb", closefd=False) as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    finally:
        os.close(fd)


def _remove_file(held, item):
    """Recheck, hash and unlink one planned file through held descriptors; a skip reason or None.

    The file must be on the held child's own device, both as its directory lists it and as
    it is opened to be hashed; otherwise it is a ``cross_device`` skip.
    """
    relative = Path(item["relative_path"])
    try:
        directory = held.directory(relative.parts[:-1])
        info = _leaf(directory, relative.name)
        if info.st_dev != held.device:
            return "cross_device"
        if (
            not stat.S_ISREG(info.st_mode)
            or relative.suffix.lower() not in BINARY_SUFFIXES
            or info.st_size < 64 * 1024
            or info.st_nlink != 1
            or not info.st_mode & 0o222
            or info.st_ino != item["inode"]
            or info.st_mtime_ns != item["mtime_ns"]
            or info.st_size != item["size_bytes"]
            or _held_sha(directory, relative.name, info) != item["sha256"]
        ):
            return "file_changed"
        if not held.in_place(relative.parts[:-1]) or not _same_leaf(directory, relative.name, info):
            return "path_changed"
    except _CrossDevice:
        return "cross_device"
    except OSError as exc:
        return f"recheck_failed:{type(exc).__name__}"
    try:
        os.unlink(relative.name, dir_fd=directory)
    except OSError as exc:
        return f"unlink_failed:{type(exc).__name__}"
    return None


def _remove_store_copy(held, copy, names):
    """Recheck every name of a planned store copy, hash it once, then unlink every name: its
    bytes must still be the store digest one of its names carries."""
    if not any(scratch_store_copy(name) and copy["sha256"] == "sha256:" + name.name for name in names):
        return "copy_changed"
    return _remove_group(held, copy, names, changed="copy_changed", sha256=copy["sha256"])


def _remove_scratch_input(held, group, names):
    """Recheck and unlink every name of a planned scratch input. Its bytes are never read:
    nothing that makes it scratch depends on them."""
    return _remove_group(held, group, names, changed="scratch_input_changed")


# rmdir of a directory that still holds anything: POSIX allows either.
_NOT_EMPTY = (errno.ENOTEMPTY, errno.EEXIST)


def _remove_empty_directories(held, root):
    """Remove the directories left empty inside the child's scratch inputs, deepest first; typed skips.

    Each directory is opened O_NOFOLLOW from its parent's descriptor and removed by rmdir
    relative to it, so a link is never entered or removed and nothing outside
    ``prepared-references`` is reached; ``prepared-references`` itself stays. A directory
    whose lstat shows another device than the held child's (a mount point) is not entered,
    and what is opened must be what was looked at, same device and inode, or it is a
    ``prune_failed:changed`` skip. One descriptor per level is open at a time. A directory
    that still holds anything is simply kept. A directory that cannot be looked at, opened
    or listed is a ``prune_failed:<type>`` skip, and ``rmdir_failed:<type>`` is only rmdir's
    own failure; a ``prepared-references`` already gone is nothing to do.
    """
    skipped = []

    def failed(parts, reason):
        skipped.append({"path": str(root.joinpath(_SCRATCH_INPUTS, *parts)), "reason": reason})

    def entered(parent, name, parts):
        """``name`` in ``parent`` opened as a directory on the held child's device, or None."""
        try:
            info = _leaf(parent, name)
            if not stat.S_ISDIR(info.st_mode) or info.st_dev != held.device:
                return None
            directory = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
        except OSError as exc:
            if parts or exc.errno != errno.ENOENT:
                failed(parts, f"prune_failed:{type(exc).__name__}")
            return None
        try:
            opened = os.fstat(directory)
            if (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino):
                return directory
            failed(parts, "prune_failed:changed")
        except OSError as exc:
            failed(parts, f"prune_failed:{type(exc).__name__}")
        os.close(directory)
        return None

    def prune(directory, parts):
        try:
            names = sorted(os.listdir(directory))
        except OSError as exc:
            failed(parts, f"prune_failed:{type(exc).__name__}")
            return
        for name in names:
            inner = entered(directory, name, (*parts, name))
            if inner is None:
                continue
            try:
                prune(inner, (*parts, name))
            finally:
                os.close(inner)
            try:
                os.rmdir(name, dir_fd=directory)
            except OSError as exc:
                if exc.errno not in _NOT_EMPTY:
                    failed((*parts, name), f"rmdir_failed:{type(exc).__name__}")

    inputs = entered(held.directory(()), _SCRATCH_INPUTS, ())
    if inputs is not None:
        try:
            prune(inputs, ())
        finally:
            os.close(inputs)
    return skipped


def _remove_group(held, group, names, *, changed, sha256=None, unlinked=None):
    """Recheck every name of a planned inode group through held descriptors, then unlink every name.

    The group's planned device, each of its names' devices now, and the file opened to hash
    it must be the held child's own; otherwise it is a ``cross_device`` skip, so nothing on a
    filesystem mounted inside the replay is removed. Each name must still be the planned
    inode, those names all of its links, its size and mtime unchanged and, with ``sha256``,
    its bytes that digest; otherwise the group is a ``changed`` skip. One failed check keeps
    every name, and a store name goes last, so a removal cut short leaves a group the next
    plan still recognises. Each name it unlinks is appended to ``unlinked`` when given, so a
    caller knows exactly which names an ``unlink_failed`` removal already took.
    """
    if group.get("dev") != held.device:
        return "cross_device"
    if len(names) != group["nlink"]:
        return changed
    try:
        entries = []
        for name in names:
            directory = held.directory(name.parts[:-1])
            entries.append((name, directory, _leaf(directory, name.name)))
        if any(info.st_dev != held.device for _name, _directory, info in entries):
            return "cross_device"
        first_name, first_directory, first = entries[0]
        planned = (group["dev"], group["inode"])
        if any(
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != planned
            or info.st_nlink != group["nlink"]
            or info.st_size != group["size_bytes"]
            or info.st_mtime_ns != group["mtime_ns"]
            for _name, _directory, info in entries
        ) or (sha256 is not None and _held_sha(first_directory, first_name.name, first) != sha256):
            return changed
        if not all(
            held.in_place(name.parts[:-1]) and _same_leaf(directory, name.name, info)
            for name, directory, info in entries
        ):
            return "path_changed"
    except _CrossDevice:
        return "cross_device"
    except OSError as exc:
        return f"recheck_failed:{type(exc).__name__}"
    for name, directory, _info in sorted(entries, key=lambda entry: scratch_store_copy(entry[0])):
        try:
            os.unlink(name.name, dir_fd=directory)
        except OSError as exc:
            return f"unlink_failed:{type(exc).__name__}"
        if unlinked is not None:
            unlinked.append(name)
    return None


def completed_report(root, *, any_parent_status=False):
    """The report of a finished offline replay under ``root``, or None.

    A parent replay writes its report when it returns and its fetcher refuses every
    fetch, so with ``any_parent_status`` its report counts whatever its status or
    ``nothing_fetched``; without it (the standalone unit) only ``nothing_fetched`` does.
    """
    return _finished_report(root, any_parent_status=any_parent_status)[0]


def _finished_report(root, *, any_parent_status):
    """``completed_report``, and whether the parent replay that ran in ``root`` wrote it.

    ``(path, ran_here)``, or ``(None, False)``. ``replay_parent`` has recorded its scratch
    queue as ``<its own root>/launch-preparations`` since 2026-09-05, so a parent report
    with no scratch queue, or one in another root (a report copied from elsewhere), is not
    the word of the replay that made this root's scratch inputs.
    """
    for name in ("stage_replay_report.v1.json", "replay_report.json", "replay.json", "report.json"):
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024**2:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        parent = (
            value.get("schema_version") == "task_evaluation_parent_replay_report.v1"
            and (any_parent_status or value.get("nothing_fetched") is True)
            and value.get("paid_execution_requested") is False
            and value.get("provider_mutation_performed") is False
        )
        legacy = (
            value.get("status") == "repair_inputs_and_semantic_request_replay_passed"
            and all(
                value.get(k) is False
                for k in (
                    "model_inference_performed",
                    "network_fetch_performed",
                    "provider_mutation_performed",
                )
            )
        ) or (
            value.get("status") == "derived_method_inputs_materialized"
            and value.get("diagnostic_only") is True
            and value.get("provider_calls") is False
        )
        legacy = legacy or (
            value.get("status") == "offline_repair_targets_materialized"
            and value.get("new_gpu_allocations") == 0
            and value.get("new_model_calls") == 0
            and value.get("appearance_qualified") is False
        )
        if parent or legacy:
            queue = value.get("scratch_queue_root")
            return path, parent and isinstance(queue, str) and Path(queue).parent.name == root.name
    return None, False


PROCESS_REFERENCED = "referenced"
PROCESS_INVENTORY_UNREADABLE = "inventory_unreadable"
# The process exited between being listed and being read: it references nothing.
_EXITED = (FileNotFoundError, ProcessLookupError)


def process_reference(root, *, process_root=Path("/proc"), ignored_process_ids=()):
    """Why a live process may still read ``root``: ``referenced``, ``inventory_unreadable`` or None.

    Reads each process's command line, environment, working directory and open
    descriptors without ever returning or storing their values. An entry that
    cannot be read (``PermissionError``, or any ``OSError`` other than the process
    having exited) proves nothing about ``root``, so the answer is then
    ``inventory_unreadable`` unless another process is seen referencing it. A
    missing process root still raises.
    """
    if not process_root.is_dir():
        raise ValueError("replay_cache_process_inventory_unavailable")
    needle = str(root).encode()
    unreadable = False
    try:
        processes = list(process_root.iterdir())
    except OSError:
        return PROCESS_INVENTORY_UNREADABLE
    for process in processes:
        if not process.name.isdigit():
            continue
        if int(process.name) in ignored_process_ids:
            continue
        for name in ("cmdline", "environ"):
            try:
                if needle in (process / name).read_bytes():
                    return PROCESS_REFERENCED
            except _EXITED:
                continue
            except OSError:
                unreadable = True
        try:
            descriptors = list((process / "fd").iterdir())
        except _EXITED:
            continue
        except OSError:
            unreadable, descriptors = True, []
        for descriptor in (process / "cwd", *descriptors):
            try:
                target = os.readlink(descriptor)
            except _EXITED:
                continue
            except OSError:
                unreadable = True
                continue
            if target == str(root) or target.startswith(str(root) + "/"):
                return PROCESS_REFERENCED
    return PROCESS_INVENTORY_UNREADABLE if unreadable else None


def active_reference(root, *, process_root=Path("/proc"), ignored_process_ids=()):
    """Whether a live process may still read ``root``; an entry that cannot be read counts as one."""
    return process_reference(root, process_root=process_root, ignored_process_ids=ignored_process_ids) is not None


def process_reference_index(*, process_root=Path("/proc"), ignored_process_ids=()):
    """One sweep of the process table, for checking many roots: ``index(root)`` answers as ``active_reference``.

    It reads each process's command line, environment, working directory and
    open descriptors once, and never returns or stores anything but what it
    matches against. A root is referenced when one of them names it; when any
    entry could not be read, every root counts as referenced, as
    ``active_reference`` counts an unreadable inventory. A missing process root
    still raises.
    """
    if not process_root.is_dir():
        raise ValueError("replay_cache_process_inventory_unavailable")
    blobs, targets, unreadable = [], [], False
    try:
        processes = list(process_root.iterdir())
    except OSError:
        return lambda _root: True
    for process in processes:
        if not process.name.isdigit() or int(process.name) in ignored_process_ids:
            continue
        for name in ("cmdline", "environ"):
            try:
                blobs.append((process / name).read_bytes())
            except _EXITED:
                continue
            except OSError:
                unreadable = True
        try:
            descriptors = list((process / "fd").iterdir())
        except _EXITED:
            continue
        except OSError:
            unreadable, descriptors = True, []
        for descriptor in (process / "cwd", *descriptors):
            try:
                targets.append(os.readlink(descriptor))
            except _EXITED:
                continue
            except OSError:
                unreadable = True

    def referenced(root):
        needle = str(root)
        return (unreadable or any(needle.encode() in blob for blob in blobs)
                or any(target == needle or target.startswith(needle + "/") for target in targets))

    return referenced


def _scan(
    replay_root, minimum_closed_seconds, now, process_root, *, verify, reclaim_store_copies, single_files,
    reclaim_scratch_inputs,
):
    root = Path(replay_root)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("replay_cache_root_unsafe")
    if type(minimum_closed_seconds) is not int or minimum_closed_seconds < 0:
        raise ValueError("replay_cache_age_invalid")
    clock = time.time() if now is None else now
    # Either opt-in counts any finished parent replay, whatever its status.
    opted_in = reclaim_store_copies or reclaim_scratch_inputs
    rows, kept = [], []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        report, ran_here = _finished_report(child, any_parent_status=opted_in)
        if report is None or clock - report.stat().st_mtime < minimum_closed_seconds:
            continue
        # Without the opt-ins, the order this module always had: a live reader keeps the root
        # before anything in it is looked at.
        if verify and not opted_in and active_reference(child, process_root=process_root):
            kept.append({"root": str(child), "reason": "active_reference"})
            continue
        report_mtime_ns = report.stat().st_mtime_ns
        copies, inputs, seen = [], [], set()
        if reclaim_scratch_inputs and ran_here:
            # The whole subtree is scratch here, its store copies included: each inode is planned once.
            inputs, seen = _scratch_inputs(child, report_mtime_ns)
        elif reclaim_store_copies:
            copies, seen = _store_copies(child, report_mtime_ns)
        files = _single_files(child, report_mtime_ns, seen) if single_files else []
        if not files and not copies and not inputs:
            continue
        if not verify:
            rows.append({"files": files, "store_copies": copies, "scratch_inputs": inputs})
            continue
        # With them, only a root that has something to reclaim is worth a sweep of the process table.
        if opted_in and active_reference(child, process_root=process_root):
            kept.append({"root": str(child), "reason": "active_reference"})
            continue
        files, copies = _verified(child, files, copies)
        if files or copies or inputs:
            # Without the opt-ins a row is exactly the one this module always wrote.
            row = {"root": str(child), "report_path": str(report), "report_sha256": file_sha(report), "files": files}
            if reclaim_store_copies:
                row["store_copies"] = copies
            if reclaim_scratch_inputs:
                row["scratch_inputs"] = inputs
            rows.append(row)
    candidate_bytes = sum(
        entry["size_bytes"]
        for r in rows
        for entry in (*r["files"], *r.get("store_copies", ()), *r.get("scratch_inputs", ()))
    )
    return root, clock, rows, kept, candidate_bytes


def plan_replay_cache_retention(
    *, replay_root, minimum_closed_seconds=60, now=None, process_root=Path("/proc"),
    reclaim_store_copies=False, single_files=True, reclaim_scratch_inputs=False,
):
    root, clock, rows, kept, candidate_bytes = _scan(
        replay_root, minimum_closed_seconds, now, process_root, verify=True,
        reclaim_store_copies=reclaim_store_copies, single_files=single_files,
        reclaim_scratch_inputs=reclaim_scratch_inputs,
    )
    plan = {
        "schema_version": SCHEMA,
        "status": "dry_run",
        "observed_at_epoch": clock,
        "replay_root": str(root),
        "rows": rows,
        "kept": kept,
        "candidate_bytes": candidate_bytes,
        "reports_and_original_evidence_removed": False,
    }
    plan["plan_digest"] = digest(plan)
    return plan


def estimate_replay_cache_retention(
    *, replay_root, minimum_closed_seconds=60, now=None, reclaim_store_copies=False, single_files=True,
    reclaim_scratch_inputs=False,
):
    """What a plan could reclaim, from names, links, sizes and ages.

    It hashes nothing, reads no candidate's bytes and sweeps no process table, though it
    still parses each replay's report. A plan also requires every store copy's bytes to
    hash to its name and no live reader, and apply rechecks all of it, so this is an upper
    bound. It carries no rows and cannot be applied.
    """
    root, clock, _rows, _kept, candidate_bytes = _scan(
        replay_root, minimum_closed_seconds, now, None, verify=False,
        reclaim_store_copies=reclaim_store_copies, single_files=single_files,
        reclaim_scratch_inputs=reclaim_scratch_inputs,
    )
    return {
        "schema_version": SCHEMA,
        "status": "estimate",
        "observed_at_epoch": clock,
        "replay_root": str(root),
        "estimated_candidate_bytes": candidate_bytes,
        "digests_verified": False,
        "live_readers_checked": False,
        "reports_and_original_evidence_removed": False,
    }


def _group_names(group):
    """A planned group's names: distinct relative paths inside the child's scratch inputs."""
    names = [Path(name) for name in group["relative_paths"]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(name.is_absolute() or ".." in name.parts or name.parts[:1] != (_SCRATCH_INPUTS,) for name in names)
    ):
        raise ValueError("replay_cache_member_unsafe")
    return names


def apply_replay_cache_retention(
    plan, *, ack, process_root=Path("/proc"), reclaim_store_copies=False, single_files=True,
    reclaim_scratch_inputs=False,
):
    if ack != ACK or plan.get("plan_digest") != digest(
        {k: v for k, v in plan.items() if k != "plan_digest"}
    ):
        raise ValueError("replay_cache_plan_invalid")
    # A plan is applied only under the options it could have been made with.
    if not reclaim_store_copies and any(row.get("store_copies") for row in plan["rows"]):
        raise ValueError("replay_cache_store_copies_not_admitted")
    if not reclaim_scratch_inputs and any(row.get("scratch_inputs") for row in plan["rows"]):
        raise ValueError("replay_cache_scratch_inputs_not_admitted")
    if not single_files and any(row.get("files") for row in plan["rows"]):
        raise ValueError("replay_cache_single_files_not_admitted")
    opted_in = reclaim_store_copies or reclaim_scratch_inputs
    removed, skipped = [], []
    base = Path(plan["replay_root"])
    for row in plan["rows"]:
        root = Path(row["root"])
        if root.parent != base or any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("replay_cache_root_changed")
        report, ran_here = _finished_report(root, any_parent_status=opted_in)
        if (
            report is None
            or str(report) != row["report_path"]
            or file_sha(report) != row["report_sha256"]
        ):
            skipped.append({"root": str(root), "reason": "report_changed"})
            continue
        if row.get("scratch_inputs") and not ran_here:
            # Only the parent replay that ran in this root made its scratch inputs; however a plan
            # was sealed, nothing else's report makes files here scratch.
            skipped.append({"root": str(root), "reason": "report_not_parent_replay"})
            continue
        if active_reference(root, process_root=process_root):
            skipped.append({"root": str(root), "reason": "active_reference"})
            continue
        # Every member is checked before anything is opened or removed.
        for item in row["files"]:
            relative = Path(item["relative_path"])
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("replay_cache_member_unsafe")
        groups = [(copy, _group_names(copy), _remove_store_copy) for copy in row.get("store_copies", [])] + [
            (group, _group_names(group), _remove_scratch_input) for group in row.get("scratch_inputs", [])]
        try:
            held = _HeldChild(base, root.name)
        except OSError as exc:
            skipped.append({"root": str(root), "reason": f"root_unavailable:{type(exc).__name__}"})
            continue
        try:
            if not held.named_by(root):
                raise ValueError("replay_cache_root_changed")
            for item in row["files"]:
                path = str(root / item["relative_path"])
                reason = held.item(_remove_file, item)
                if reason:
                    skipped.append({"path": path, "reason": reason})
                else:
                    removed.append({"path": path, "sha256": item["sha256"], "size_bytes": item["size_bytes"]})
            for group, names, remove in groups:
                paths = [str(root / name) for name in names]
                reason = held.item(remove, group, names)
                if reason:
                    skipped.append({"paths": paths, "reason": reason})
                else:
                    # A store copy's digest was verified; a scratch input's bytes were never read.
                    verified = {"sha256": group["sha256"]} if "sha256" in group else {}
                    removed.append({"paths": paths, **verified, "size_bytes": group["size_bytes"]})
            if row.get("scratch_inputs"):
                skipped.extend(held.item(_remove_empty_directories, root))
        finally:
            held.close()
    return {
        "schema_version": SCHEMA,
        "status": "applied",
        "plan_digest": plan["plan_digest"],
        "removed_bytes": sum(r["size_bytes"] for r in removed),
        "removed": removed,
        "skipped": skipped,
        "reports_and_original_evidence_removed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--minimum-closed-seconds", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ack", default="")
    parser.add_argument("--reclaim-store-copies", action="store_true",
                        help="also reclaim parent replays' content-store copies (storage GC's opt-in)")
    args = parser.parse_args()
    options = {"reclaim_store_copies": args.reclaim_store_copies}
    plan = plan_replay_cache_retention(
        replay_root=args.replay_root, minimum_closed_seconds=args.minimum_closed_seconds, **options
    )
    output = Path(args.report_root)
    output.mkdir(parents=True, exist_ok=True)
    key = plan["plan_digest"][7:]
    (output / (key + "-plan.json")).write_text(json.dumps(plan, indent=2) + "\n")
    result = apply_replay_cache_retention(plan, ack=args.ack, **options) if args.apply else plan
    (output / (key + "-result.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in {
                    "status",
                    "candidate_bytes",
                    "removed_bytes",
                    "reports_and_original_evidence_removed",
                }
            }
        )
    )


if __name__ == "__main__":
    main()
