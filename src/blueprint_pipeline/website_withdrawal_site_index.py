"""A per-site ownership index for the nine Pipeline storage roots that record no site.

Beta journey step: site withdrawal. ``website_capture_withdrawal_deletion`` scans run roots
by their own records, but nine roots hold derived copies with no per-site binding
(``UNCOVERED_ROOTS`` there), so no withdrawal could be declared complete. This index says,
for each top-level entry of those roots, which site identifiers its records name.

How ownership is determined, the same way for each of the nine roots: every JSON or JSONL
record under the entry is read (streamed, bounded) and the identifiers it names are kept:
website scene ids (``site-<id>``), capture ids (``walkthrough-<id>``) and scene intent ids
(``scene-<64 hex>``, which the deletion plan maps to a site through its source
registration). An entry is the site's when it names the site's ids and no other site's.

- ``completed-scene-preparation-inputs``, ``prepared-references``, ``compiled-episodes``:
  owner-scoped materialized directories name their scene intent; their shared
  content-addressed stores (``content-addressed``) hold blobs any site may share and name
  no site, so those entries stay ``owner_unknown`` and are left for the owner.
- ``sam31-preparations``, ``scene-object-discovery-outputs``,
  ``task-evaluation-terminal-results``, ``scene-configuration-diagnostics``,
  ``capture-reconstruction-derived``, ``result-artifact-cache``: each entry's own records
  (bindings, receipts, result indexes) name the scene, capture or intent it was made for;
  an entry whose records name none, or cannot be read, is ``owner_unknown``.

An entry naming two or more sites is ``shared``: never deleted, always reported. Nothing
here is guessed and nothing is deleted here; the index only reads.

The scan is $0 and read-only, runs in the hourly withdrawal unit
(``website_capture_withdrawal_automatic``), is bounded per run (entries, bytes, a deadline)
and resumes from a cursor. A root whose sweep never finished, or that holds an entry the
index has not read yet, is ``unknown`` for every site: never "nothing found". A finished
sweep starts again after ``RESWEEP_SECONDS`` so changed and new entries are read again.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import website_withdrawal_content_blobs as content_blobs
from .core.common import write_json

INDEX_SCHEMA = "website_withdrawal_site_index.v1"
RESWEEP_SECONDS = 24 * 3600
MAX_TOKENS_PER_ENTRY = 64
MAX_FILE_BYTES = 64 * 1024 * 1024
_TOKEN = re.compile(rb"(?<![A-Za-z0-9._-])(site-[A-Za-z0-9][A-Za-z0-9._-]{0,119}|walkthrough-[A-Za-z0-9]"
                    rb"[A-Za-z0-9._-]{0,119}|scene-[0-9a-f]{64})(?![A-Za-z0-9._-])")
_CHUNK, _OVERLAP = 1024 * 1024, 256
_SUFFIXES = (".json", ".jsonl")


class _Budget:
    def __init__(self, entries: int, read_bytes: int, deadline: float):
        self.entries, self.bytes, self.deadline = entries, read_bytes, deadline

    def spent(self) -> bool:
        return self.entries <= 0 or self.bytes <= 0 or time.monotonic() >= self.deadline


def _strip_site_suffix(token: str) -> str:
    # ``site-<id>.retired.v1.json`` names the site ``site-<id>``.
    for marker in (".retired", ".json", ".v1"):
        token = token.split(marker, 1)[0]
    return token


class _Special(OSError):
    pass


def _record_files(entry: Path) -> list[Path]:
    """The entry's JSON and JSONL records, in a fixed order; raises ``OSError`` when it cannot be listed."""

    info = os.lstat(entry)
    if stat.S_ISREG(info.st_mode):
        return [entry] if entry.name.endswith(_SUFFIXES) else []
    if not stat.S_ISDIR(info.st_mode):
        raise _Special("special_entry")
    files: list[Path] = []

    def unlistable(error: OSError) -> None:
        raise error

    for root, directories, names in os.walk(entry, onerror=unlistable):
        directories.sort()
        files.extend(Path(root) / name for name in sorted(names) if name.endswith(_SUFFIXES))
    return files


def fingerprint(entry: Path) -> str:
    """Every record's path, size and mtime under the entry: a nested record edited or added changes it.

    A directory's own mtime does not change when a file deep inside it does, so the index compares
    this, not the entry's mtime. Raises ``OSError`` when the entry cannot be listed.
    """

    rows = []
    for path in _record_files(entry):
        info = os.lstat(path)
        rows.append([str(path.relative_to(entry.parent)), info.st_size, info.st_mtime_ns])
    return "sha256:" + hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _scan_entry(entry: Path, budget: _Budget) -> dict[str, Any]:
    """The identifiers an entry's records name, or why they could not be read."""

    tokens: set[str] = set()
    try:
        try:
            files = _record_files(entry)
            stamp = fingerprint(entry)
        except _Special:
            return {"status": "unreadable", "reason": "special_entry", "tokens": []}
        for path in files:
            mode = os.lstat(path)
            if not stat.S_ISREG(mode.st_mode):
                continue
            if mode.st_size > MAX_FILE_BYTES:
                return {"status": "unreadable", "reason": "record_too_large", "tokens": []}
            budget.bytes -= mode.st_size
            if budget.bytes < 0 or time.monotonic() >= budget.deadline:
                return {"status": "unreadable", "reason": "scan_budget_spent", "tokens": []}
            tail = b""
            with path.open("rb") as stream:
                while chunk := stream.read(_CHUNK):
                    window = tail + chunk
                    tokens.update(_strip_site_suffix(match.group(1).decode()) for match in _TOKEN.finditer(window))
                    tail = window[-_OVERLAP:]
            if len(tokens) > MAX_TOKENS_PER_ENTRY:
                return {"status": "shared", "reason": "too_many_identifiers", "tokens": []}
    except OSError as error:
        return {"status": "unreadable", "reason": "unreadable:" + type(error).__name__, "tokens": []}
    try:
        # The hard links it holds: how the shared content stores' blobs are referred to.
        links = content_blobs.linked_inodes(entry)
    except OSError:
        links = None
    extra = {"fingerprint": stamp, "links": links or {}, "links_complete": links is not None}
    if not tokens:
        return {"status": "owner_unknown", "reason": "no_site_named", "tokens": [], **extra}
    return {"status": "named", "reason": None, "tokens": sorted(tokens), **extra}


def confirm_owned(entry: Path, *, seeds: Iterable[str], site_tokens: Iterable[str],
                  max_bytes: int = 512 * 1024 * 1024, max_seconds: float = 300.0) -> str | None:
    """Read the entry's records again from disk; None only if they name this site's ids and nothing else.

    The index is a cache for planning; this fresh read is the authority to delete. Otherwise the reason.
    """

    fresh = _scan_entry(entry, _Budget(1, max_bytes, time.monotonic() + max_seconds))
    tokens = set(fresh.get("tokens") or ())
    if fresh["status"] != "named":
        return "fresh_read_" + str(fresh.get("reason") or fresh["status"])
    if not tokens & set(seeds):
        return "fresh_read_names_no_identifier_of_this_site"
    if tokens - set(seeds) - set(site_tokens):
        return "fresh_read_names_another_owner"
    return None


def mark_unscanned(index_root: Path, root: Path, name: str) -> None:
    """Drop one entry from the index and queue it, so lookups read it as unknown until it is read again."""

    lock_path = Path(index_root) / (Path(root).name + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        index = read_index(index_root, root)
        if index is None:
            return
        index["entries"].pop(name, None)
        index["pending"] = sorted({*index.get("pending", []), name})
        write_json(_index_path(index_root, root), index)
    finally:
        os.close(descriptor)


def _index_path(index_root: Path, root: Path) -> Path:
    return Path(index_root) / (Path(root).name + ".json")


def read_index(index_root: Path, root: Path) -> dict[str, Any] | None:
    path = _index_path(index_root, root)
    try:
        if path.is_symlink():
            return None
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("schema_version") == INDEX_SCHEMA else None


def _listing(root: Path) -> list[os.DirEntry[str]]:
    with os.scandir(root) as entries:
        return sorted(entries, key=lambda entry: entry.name)


def advance(index_root: Path, roots: Sequence[Path], *, now: float, max_entries: int = 500,
            max_bytes: int = 512 * 1024 * 1024, max_seconds: float = 600.0) -> dict[str, Any]:
    """Read the next entries of each root into its index, within the budget; resumable.

    Each root's index is replaced atomically under its own lock after every entry, so a run
    killed part way keeps what it read and the next run continues from the cursor.
    """

    Path(index_root).mkdir(parents=True, exist_ok=True, mode=0o750)
    budget = _Budget(max_entries, max_bytes, time.monotonic() + max_seconds)
    report: dict[str, Any] = {}
    for root in map(Path, roots):
        if budget.spent():
            report[root.name] = "budget_spent"
            continue
        lock_path = Path(index_root) / (root.name + ".lock")
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report[root.name] = "locked"
                continue
            report[root.name] = _advance_root(index_root, root, now=now, budget=budget)
        finally:
            os.close(descriptor)
    return report


def _advance_root(index_root: Path, root: Path, *, now: float, budget: _Budget) -> str:
    index = read_index(index_root, root) or {
        "schema_version": INDEX_SCHEMA, "root": str(root), "entries": {},
        "sweep": {"cursor": None, "started_at_epoch": now, "completed_at_epoch": None, "ever_completed": False}}
    sweep = index["sweep"]
    rest = False
    if sweep.get("cursor") is None and sweep.get("completed_at_epoch") is not None:
        if now - float(sweep["completed_at_epoch"]) < RESWEEP_SECONDS:
            if not index.get("pending"):
                return "swept"
            rest = True  # read only the queued entries; the next full sweep is not due yet
        else:
            sweep.update(started_at_epoch=now, completed_at_epoch=None)
    if not root.exists():
        index["entries"], index["root_absent"] = {}, True
        sweep.update(cursor=None, completed_at_epoch=now, ever_completed=True)
        write_json(_index_path(index_root, root), index)
        return "absent"
    index["root_absent"] = False
    try:
        listing = _listing(root)
    except OSError:
        return "unlistable"
    present = {entry.name for entry in listing}
    # Entries a deletion found changed are read first, whatever the sweep's cursor.
    for name in list(index.get("pending") or []):
        if budget.spent():
            write_json(_index_path(index_root, root), index)
            return "in_progress"
        path = root / name
        if name in present:
            budget.entries -= 1
            try:
                mtime = os.lstat(path).st_mtime_ns
            except OSError:
                mtime = None
            index["entries"][name] = {**_scan_entry(path, budget), "mtime_ns": mtime, "scanned_at_epoch": now}
        index["pending"] = [item for item in index["pending"] if item != name]
        write_json(_index_path(index_root, root), index)
    if rest:
        return "swept"
    cursor = sweep.get("cursor")
    for entry in listing:
        if cursor is not None and entry.name <= cursor:
            continue
        if budget.spent():
            write_json(_index_path(index_root, root), index)
            return "in_progress"
        if entry.name == content_blobs.STORE_ENTRY and entry.is_dir(follow_symlinks=False):
            # A shared store: one stat per blob, in chunks; the sweep moves on only once it is read whole.
            record = index["entries"].get(entry.name)
            if sweep.get("store_started_at_epoch") != sweep.get("started_at_epoch"):
                record = {**(record or {}), "store_cursor": None, "scanning": {}, "in_progress": False}
                sweep["store_started_at_epoch"] = sweep.get("started_at_epoch")
            record = content_blobs.scan_store(Path(entry.path), record, budget=budget, now=now)
            index["entries"][entry.name] = record
            if record.get("in_progress"):
                write_json(_index_path(index_root, root), index)
                return "in_progress"
            sweep["cursor"] = entry.name
            write_json(_index_path(index_root, root), index)
            continue
        budget.entries -= 1
        try:
            mtime = os.lstat(entry.path).st_mtime_ns
        except OSError:
            continue
        index["entries"][entry.name] = {**_scan_entry(Path(entry.path), budget), "mtime_ns": mtime,
                                        "scanned_at_epoch": now}
        sweep["cursor"] = entry.name
        write_json(_index_path(index_root, root), index)
    index["entries"] = {name: row for name, row in index["entries"].items() if name in present}
    sweep.update(cursor=None, completed_at_epoch=now, ever_completed=True)
    write_json(_index_path(index_root, root), index)
    return "complete"


def lookup(index_root: Path, roots: Iterable[Path], *, seeds: Iterable[str], site_tokens: Iterable[str],
           acknowledged_at_epoch: float | None = None) -> list[dict[str, Any]]:
    """For each root, what the index says about one site: its entries, shared ones, and what is unknown.

    ``seeds`` are every identifier of the site (scene, capture, its scene intents);
    ``site_tokens`` the site's own scene and capture ids. An entry naming any other identifier
    (another site's scene or capture, or a scene intent not the site's) is shared. Status per root:
    ``absent`` (no such root on this host), ``unknown`` (its index never finished, or an entry
    there was never read, or it changed since), or ``indexed``.
    """

    seeds, mine = set(seeds), set(site_tokens)
    rows = []
    for root in map(Path, roots):
        row: dict[str, Any] = {"root": str(root), "index_root": str(index_root), "status": "unknown", "owned": [],
                               "shared": [], "owner_unknown": [], "unscanned": []}
        rows.append(row)
        if not root.exists():
            row["status"] = "absent"
            continue
        index = read_index(index_root, root)
        if index is None or not index["sweep"].get("ever_completed"):
            row["reason"] = "index_never_completed"
            continue
        try:
            listing = _listing(root)
        except OSError:
            row["reason"] = "root_unlistable"
            continue
        verdicts: dict[str, str] = {}
        store_present = False
        for entry in listing:
            record = index["entries"].get(entry.name)
            if entry.name == content_blobs.STORE_ENTRY and entry.is_dir(follow_symlinks=False):
                store_present = True
                continue
            try:
                mtime = os.lstat(entry.path).st_mtime_ns
                stamp = fingerprint(Path(entry.path))
            except OSError:
                row["unscanned"].append(entry.name)
                continue
            # A record nested anywhere below that changed since the read changes the fingerprint; an index
            # written before fingerprints existed is read again.
            if record is None or record.get("fingerprint") != stamp or entry.name in (index.get("pending") or ()):
                row["unscanned"].append(entry.name)
                verdicts[entry.name] = "unknown"
                continue
            tokens = set(record.get("tokens") or ())
            verdicts[entry.name] = "unknown" if not record.get("links_complete", False) else "other"
            if record.get("status") == "shared" and not tokens:
                row["shared"].append(entry.name)
            elif record.get("status") != "named":
                row["owner_unknown"].append({"name": entry.name, "reason": record.get("reason")})
                verdicts[entry.name] = "unknown"
            elif tokens & seeds:
                # Any identifier that is not this site's (another site, or a scene intent of another site).
                others = tokens - seeds - mine
                (row["shared"] if others else row["owned"]).append(
                    entry.name if others else {"name": entry.name, "path": entry.path, "mtime_ns": mtime})
                if not others and record.get("links_complete", False):
                    verdicts[entry.name] = "site"
        if store_present:
            row["content_store"] = content_blobs.attribute(
                index["entries"].get(content_blobs.STORE_ENTRY), index["entries"], verdicts=verdicts,
                store_path=root / content_blobs.STORE_ENTRY, acknowledged_at_epoch=acknowledged_at_epoch)
            if not row["content_store"]["complete"]:
                row["unscanned"].append(content_blobs.STORE_ENTRY)
        row["status"] = "unknown" if row["unscanned"] else "indexed"
        if row["unscanned"]:
            row["reason"] = "entries_not_yet_read"
    return rows


def blocking(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """What keeps a site's deletion from being complete in these roots, one row per root and reason."""

    found = []
    for row in rows:
        if row["status"] == "unknown":
            found.append({"holder": "pipeline_root_not_indexed", "path": row["root"],
                          "reason": row.get("reason") or "unknown", "unscanned": len(row["unscanned"])})
        if row["shared"]:
            found.append({"holder": "pipeline_root_shared_entries", "path": row["root"], "entries": row["shared"],
                          "reason": "these entries also name another site; they are never deleted for one site"})
        unreferenced = (row.get("content_store") or {}).get("unreferenced") or {}
        if unreferenced.get("possible_leftover_count"):
            found.append({"holder": "unreferenced_content_blobs", "path": row["content_store"]["store"],
                          "count": unreferenced["possible_leftover_count"],
                          "bytes": unreferenced["possible_leftover_bytes"],
                          "reason": "no entry refers to these blobs and they predate the withdrawal, so they may be "
                                    "this site's leftovers; a withdrawal never deletes them on a guess, and only "
                                    "garbage collection under its own approved policy class does"})
        if row["owner_unknown"]:
            found.append({"holder": "pipeline_root_owner_unknown", "path": row["root"],
                          "count": len(row["owner_unknown"]), "examples": row["owner_unknown"][:10],
                          "reason": "whose these entries are cannot be determined from their records; "
                                    "they are left for the owner, never deleted"})
    return found


__all__ = ["INDEX_SCHEMA", "RESWEEP_SECONDS", "advance", "blocking", "confirm_owned", "fingerprint", "lookup",
           "mark_unscanned", "read_index"]
