"""Reclaim disposable binary copies after an offline replay has completed.

Reports, logs, source code, readonly files and shared inodes remain untouched.
With ``reclaim_store_copies`` (storage GC's opt-in; the standalone unit never
passes it) one more kind of copy is recognised, by its place and name instead
of a suffix: a parent replay's copy of a content-store blob, at
``prepared-references/content-addressed/sha256/<digest>``, together with every
other name it has in the replay's ``prepared-references`` (the worker's
materialized references are hard links to it). It keeps the store's read-only
mode and has no suffix, so the whole inode qualifies when all of its links are
there and its bytes match the digest it is named by; a link anywhere else
keeps it. Under the same opt-in any finished parent replay counts, whatever its
status. Without it the rules are the ones this module always had.
Apply rechecks, hashes and unlinks every name through directory descriptors
held from the replay child down, never through a path, and skips an item whose
directory moved or became a link since it was opened.
This module is also a standalone maintenance entrypoint; it never allocates a
provider or changes the scientific release used by a live run.
"""

from __future__ import annotations

import argparse
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


def _store_copies(child, report_mtime_ns):
    """Candidate copies of store blobs in the child's scratch inputs, each with every name it has there.

    Files under ``prepared-references`` are grouped by inode, from names and metadata
    alone. A group is a candidate only when all of its links are in that subtree, one of
    its names is a store name, and it is not newer than the report; ``_verified`` then
    requires its bytes to hash to that name. Also returns every inode that has a store
    name, so the single-file rules never plan one of its names.
    """
    subtree = child / _SCRATCH_INPUTS
    groups = {}
    if subtree.is_dir() and not subtree.is_symlink():
        for directory, _directories, names in os.walk(subtree):
            for name in names:
                path = Path(directory) / name
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and _no_linked_parent(path, child):
                    groups.setdefault((info.st_dev, info.st_ino), (info, []))[1].append(path.relative_to(child))
    copies, store_inodes = [], set()
    for key, (info, names) in groups.items():
        store_names = [name for name in names if scratch_store_copy(name)]
        if not store_names:
            continue
        store_inodes.add(key)
        if len(names) != info.st_nlink or info.st_mtime_ns > report_mtime_ns:
            continue
        copies.append({
            "relative_paths": sorted(str(name) for name in names),
            "inode": info.st_ino,
            "nlink": info.st_nlink,
            "size_bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        })
    return sorted(copies, key=lambda copy: copy["relative_paths"]), store_inodes


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
    no more descriptors than one item needs.
    """

    def __init__(self, base, name):
        self._name = name
        self._base = os.open(os.fspath(base), _DIRECTORY_FLAGS)
        try:
            self._held = {(): os.open(name, _DIRECTORY_FLAGS, dir_fd=self._base)}
        except OSError:
            os.close(self._base)
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
        os.close(self._held.pop(()))
        os.close(self._base)


def _leaf(directory, name):
    return os.stat(name, dir_fd=directory, follow_symlinks=False)


def _same_leaf(directory, name, expected):
    entry = _leaf(directory, name)
    return stat.S_ISREG(entry.st_mode) and (entry.st_dev, entry.st_ino) == (expected.st_dev, expected.st_ino)


def _held_sha(directory, name, expected):
    """Hash ``name`` opened O_NOFOLLOW in its held directory, or None if it is not ``expected``."""
    fd = os.open(name, _LEAF_FLAGS, dir_fd=directory)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            return None
        with open(fd, "rb", closefd=False) as stream:
            return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    finally:
        os.close(fd)


def _remove_file(held, item):
    """Recheck, hash and unlink one planned file through held descriptors; a skip reason or None."""
    relative = Path(item["relative_path"])
    try:
        directory = held.directory(relative.parts[:-1])
        info = _leaf(directory, relative.name)
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
    except OSError as exc:
        return f"recheck_failed:{type(exc).__name__}"
    try:
        os.unlink(relative.name, dir_fd=directory)
    except OSError as exc:
        return f"unlink_failed:{type(exc).__name__}"
    return None


def _remove_store_copy(held, copy, names):
    """Recheck every name of a planned store copy, hash it once, then unlink every name.

    Each name must still be the planned inode, those names all of its links, and its bytes
    the store digest one of them carries. One failed check keeps every name, and the store
    name goes last, so a removal cut short leaves a group the next plan still recognises.
    """
    if len(names) != copy["nlink"] or not any(
        scratch_store_copy(name) and copy["sha256"] == "sha256:" + name.name for name in names
    ):
        return "copy_changed"
    try:
        entries = []
        for name in names:
            directory = held.directory(name.parts[:-1])
            entries.append((name, directory, _leaf(directory, name.name)))
        first_name, first_directory, first = entries[0]
        if any(
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != (first.st_dev, copy["inode"])
            or info.st_nlink != copy["nlink"]
            or info.st_size != copy["size_bytes"]
            or info.st_mtime_ns != copy["mtime_ns"]
            for _name, _directory, info in entries
        ) or _held_sha(first_directory, first_name.name, first) != copy["sha256"]:
            return "copy_changed"
        if not all(
            held.in_place(name.parts[:-1]) and _same_leaf(directory, name.name, info)
            for name, directory, info in entries
        ):
            return "path_changed"
    except OSError as exc:
        return f"recheck_failed:{type(exc).__name__}"
    for name, directory, _info in sorted(entries, key=lambda entry: scratch_store_copy(entry[0])):
        try:
            os.unlink(name.name, dir_fd=directory)
        except OSError as exc:
            return f"unlink_failed:{type(exc).__name__}"
    return None


def completed_report(root, *, any_parent_status=False):
    """The report of a finished offline replay under ``root``, or None.

    A parent replay writes its report when it returns and its fetcher refuses every
    fetch, so with ``any_parent_status`` its report counts whatever its status or
    ``nothing_fetched``; without it (the standalone unit) only ``nothing_fetched`` does.
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
            return path
    return None


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


def _scan(replay_root, minimum_closed_seconds, now, process_root, *, verify, reclaim_store_copies, single_files):
    root = Path(replay_root)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("replay_cache_root_unsafe")
    if type(minimum_closed_seconds) is not int or minimum_closed_seconds < 0:
        raise ValueError("replay_cache_age_invalid")
    clock = time.time() if now is None else now
    rows, kept = [], []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        report = completed_report(child, any_parent_status=reclaim_store_copies)
        if report is None or clock - report.stat().st_mtime < minimum_closed_seconds:
            continue
        # Without the opt-in, the order this module always had: a live reader keeps the root
        # before anything in it is looked at.
        if verify and not reclaim_store_copies and active_reference(child, process_root=process_root):
            kept.append({"root": str(child), "reason": "active_reference"})
            continue
        report_mtime_ns = report.stat().st_mtime_ns
        copies, store_inodes = _store_copies(child, report_mtime_ns) if reclaim_store_copies else ([], set())
        files = _single_files(child, report_mtime_ns, store_inodes) if single_files else []
        if not files and not copies:
            continue
        if not verify:
            rows.append({"files": files, "store_copies": copies})
            continue
        # With it, only a root that has something to reclaim is worth a sweep of the process table.
        if reclaim_store_copies and active_reference(child, process_root=process_root):
            kept.append({"root": str(child), "reason": "active_reference"})
            continue
        files, copies = _verified(child, files, copies)
        if files or copies:
            # Without store copies a row is exactly the one this module always wrote.
            row = {"root": str(child), "report_path": str(report), "report_sha256": file_sha(report), "files": files}
            if reclaim_store_copies:
                row["store_copies"] = copies
            rows.append(row)
    candidate_bytes = sum(
        entry["size_bytes"] for r in rows for entry in (*r["files"], *r.get("store_copies", ()))
    )
    return root, clock, rows, kept, candidate_bytes


def plan_replay_cache_retention(
    *, replay_root, minimum_closed_seconds=60, now=None, process_root=Path("/proc"),
    reclaim_store_copies=False, single_files=True,
):
    root, clock, rows, kept, candidate_bytes = _scan(
        replay_root, minimum_closed_seconds, now, process_root, verify=True,
        reclaim_store_copies=reclaim_store_copies, single_files=single_files,
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
    *, replay_root, minimum_closed_seconds=60, now=None, reclaim_store_copies=False, single_files=True
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


def apply_replay_cache_retention(
    plan, *, ack, process_root=Path("/proc"), reclaim_store_copies=False, single_files=True
):
    if ack != ACK or plan.get("plan_digest") != digest(
        {k: v for k, v in plan.items() if k != "plan_digest"}
    ):
        raise ValueError("replay_cache_plan_invalid")
    # A plan is applied only under the options it could have been made with.
    if not reclaim_store_copies and any(row.get("store_copies") for row in plan["rows"]):
        raise ValueError("replay_cache_store_copies_not_admitted")
    if not single_files and any(row.get("files") for row in plan["rows"]):
        raise ValueError("replay_cache_single_files_not_admitted")
    removed, skipped = [], []
    base = Path(plan["replay_root"])
    for row in plan["rows"]:
        root = Path(row["root"])
        if root.parent != base or any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError("replay_cache_root_changed")
        report = completed_report(root, any_parent_status=reclaim_store_copies)
        if (
            report is None
            or str(report) != row["report_path"]
            or file_sha(report) != row["report_sha256"]
        ):
            skipped.append({"root": str(root), "reason": "report_changed"})
            continue
        if active_reference(root, process_root=process_root):
            skipped.append({"root": str(root), "reason": "active_reference"})
            continue
        # Every member is checked before anything is opened or removed.
        for item in row["files"]:
            relative = Path(item["relative_path"])
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("replay_cache_member_unsafe")
        copies = []
        for copy in row.get("store_copies", []):
            names = [Path(name) for name in copy["relative_paths"]]
            if (
                not names
                or len(set(names)) != len(names)
                or any(
                    name.is_absolute() or ".." in name.parts or name.parts[:1] != (_SCRATCH_INPUTS,)
                    for name in names
                )
            ):
                raise ValueError("replay_cache_member_unsafe")
            copies.append((copy, names))
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
            for copy, names in copies:
                paths = [str(root / name) for name in names]
                reason = held.item(_remove_store_copy, copy, names)
                if reason:
                    skipped.append({"paths": paths, "reason": reason})
                else:
                    removed.append({"paths": paths, "sha256": copy["sha256"], "size_bytes": copy["size_bytes"]})
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
