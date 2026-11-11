"""The shared content-addressed stores, attributed to sites through the directories that refer to each blob.

Beta journey step: site withdrawal. Three of the nine roots the ownership index reads
(``website_withdrawal_site_index``) keep a shared store of digest-named blobs:

- ``prepared-references/content-addressed/sha256/<digest>``,
- ``completed-scene-preparation-inputs/content-addressed/sha256/<digest>`` (the same launch
  preparation worker, run for scene preparation), and
- ``compiled-episodes/content-addressed/adapter-members/sha256/<digest>``.

A blob names no site, but its referrers do. Every store refers to blobs the same way: the
owner-scoped materialized directory beside the store holds a hard link to the blob
(``task_evaluation_launch_preparation_worker`` and ``task_evaluation_scene_retirement_cache.
project_content`` link it; the episode compiler links adapter members). So the referrers of a
blob are the entries holding a link to its inode, and the blob's link count says how many
links exist in all: links the index did not find are referrers outside the root, never
attributed to anyone.

- A blob whose every referrer is an entry of the withdrawing site is the site's. It is deleted
  only after a fresh check at that moment: its link count is exactly one plus the links found,
  freshly, in the site's entries (a referrer added anywhere since adds a link and stops it).
- A blob another site, an entry of unknown owner, or a link outside the root also refers to is
  kept and named on the receipt as shared, with the reason.
- A blob nobody refers to (one link) may be the leftover of a site withdrawn earlier. It is
  never deleted for a site. It is counted on every receipt and in every hourly summary, and a
  separate garbage-collection plan (``plan_gc``) deletes it only under the approval file's own
  ``garbage_collection_policy_class``.

Before any deletion, the site's blobs are written to its withdrawal journal
(``content_blobs.json``). A pass interrupted after its entries went but before a blob did
finds the blob there on the next pass, with one link and the same inode, and deletes it.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .core.common import write_json

STORE_ENTRY = "content-addressed"
JOURNAL_FILENAME = "content_blobs.json"
GC_POLICY_CLASS = "website_withdrawal_unreferenced_content_blob_gc.v1"
GC_GRACE_SECONDS = 72 * 3600
GC_MAX_BYTES_PER_PASS = 1024 * 1024 * 1024
MAX_LINKED_FILES = 200_000
BLOBS_PER_ENTRY_BUDGET = 100
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _key(info: os.stat_result) -> str:
    return f"{info.st_dev}:{info.st_ino}"


def linked_inodes(entry: Path) -> dict[str, int] | None:
    """How many of an entry's regular files link each shared inode (link count above one); None past the cap."""

    found: dict[str, int] = {}
    seen = 0
    paths = [entry] if not entry.is_dir() or entry.is_symlink() else None
    if paths is None:
        paths = (Path(root) / name for root, _dirs, names in os.walk(entry) for name in names)
    for path in paths:
        info = os.lstat(path)
        seen += 1
        if seen > MAX_LINKED_FILES:
            return None
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            found[_key(info)] = found.get(_key(info), 0) + 1
    return found


def _store_files(store: Path) -> list[str]:
    rows = []
    for root, directories, names in os.walk(store):
        directories.sort()
        rows.extend(str((Path(root) / name).relative_to(store)) for name in sorted(names) if _DIGEST.fullmatch(name))
    return rows


def scan_store(store: Path, record: Mapping[str, Any] | None, *, budget: Any, now: float) -> dict[str, Any]:
    """Read the store's blobs (a stat each) from the cursor on, within the budget; resumable."""

    record = dict(record or {})
    if record.get("status") != "store":
        record = {"status": "store", "blobs": {}, "complete": False}
    scanning = dict(record.get("scanning") or {})
    cursor = record.get("store_cursor")
    stats = 0
    for relative in _store_files(store):
        if cursor is not None and relative <= cursor:
            continue
        if budget.spent():
            record.update(scanning=scanning, store_cursor=cursor, in_progress=True)
            return record
        stats += 1
        if stats % BLOBS_PER_ENTRY_BUDGET == 0:  # a blob costs one stat, not a read
            budget.entries -= 1
        try:
            info = os.lstat(store / relative)
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            scanning[relative] = {"inode": _key(info), "nlink": info.st_nlink, "size": info.st_size,
                                  "mtime": info.st_mtime}
        cursor = relative
    record.update(blobs=scanning, scanning={}, store_cursor=None, complete=True, completed_at_epoch=now,
                  in_progress=False)
    return record


def attribute(store_record: Mapping[str, Any] | None, entries: Mapping[str, Mapping[str, Any]], *,
              verdicts: Mapping[str, str], store_path: Path, acknowledged_at_epoch: float | None) -> dict[str, Any]:
    """Which blobs are the site's, shared, or unreferenced, from the index (a plan; never a delete).

    ``verdicts`` maps each entry name to ``site`` (only the withdrawing site's), ``other`` or ``unknown``.
    """

    section: dict[str, Any] = {"store": str(store_path), "complete": bool(store_record and store_record.get("complete")),
                               "site_blobs": [], "shared": [], "unreferenced": {
                                   "count": 0, "bytes": 0, "possible_leftover_count": 0, "possible_leftover_bytes": 0}}
    if not section["complete"]:
        return section
    referrers: dict[str, dict[str, int]] = {}
    for name, record in entries.items():
        for inode, count in (record.get("links") or {}).items():
            referrers.setdefault(inode, {})[name] = count
    for relative, blob in sorted(store_record["blobs"].items()):
        if blob["nlink"] <= 1:
            unreferenced = section["unreferenced"]
            unreferenced["count"] += 1
            unreferenced["bytes"] += blob["size"]
            # A blob first written after the acknowledgement cannot be this site's leftover.
            if acknowledged_at_epoch is None or blob["mtime"] <= acknowledged_at_epoch:
                unreferenced["possible_leftover_count"] += 1
                unreferenced["possible_leftover_bytes"] += blob["size"]
            continue
        by = referrers.get(blob["inode"], {})
        verdict = {verdicts.get(name, "unknown") for name in by}
        if "site" not in verdict:
            continue
        outside = blob["nlink"] - 1 - sum(by.values())
        row = {"path": str(store_path / relative), "inode": blob["inode"], "size": blob["size"],
               "referrers": sorted(by)}
        if verdict == {"site"} and outside == 0:
            section["site_blobs"].append(row)
        else:
            row["reason"] = ("also referred to by another site's entry" if "other" in verdict
                             else "also referred to by an entry whose owner is unknown" if "unknown" in verdict
                             else "also linked from outside this root")
            section["shared"].append(row)
    return section


def confirm_site_blob(path: Path, inode: str, entries: Iterable[Path]) -> str | None:
    """Fresh, at the moment of deletion: None only if every link of the blob is in the site's ``entries``."""

    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return "blob_gone"
    if not stat.S_ISREG(info.st_mode) or _key(info) != inode:
        return "blob_replaced_since_plan"
    found = 0
    for entry in entries:
        try:
            found += (linked_inodes(entry) or {}).get(inode, 0)
        except OSError:
            return "entry_unreadable"
    if info.st_nlink != 1 + found:
        return "blob_has_another_referrer_now"
    return None


def record_decision(journal: Path, blobs: Sequence[Mapping[str, Any]]) -> None:
    """Write the site's blobs to its journal before anything is deleted; earlier decisions stay."""

    path = journal / JOURNAL_FILENAME
    ledger = _read(path)
    for blob in blobs:
        ledger["blobs"].setdefault(blob["path"], {"inode": blob["inode"], "status": "decided_site_only"})
    write_json(path, ledger)


def _read(path: Path) -> dict[str, Any]:
    import json
    try:
        value = json.loads(path.read_text()) if path.is_file() and not path.is_symlink() else None
    except (OSError, ValueError):
        value = None
    return value if isinstance(value, dict) and isinstance(value.get("blobs"), dict) else {"blobs": {}}


def pending_decisions(journal: Path) -> list[dict[str, Any]]:
    """Blobs an earlier pass decided were the site's and that are still there."""

    return [{"path": path, **row} for path, row in sorted(_read(journal / JOURNAL_FILENAME)["blobs"].items())
            if row.get("status") == "decided_site_only"]


def settle(journal: Path, path: str, status: str) -> None:
    ledger_path = journal / JOURNAL_FILENAME
    ledger = _read(ledger_path)
    if path in ledger["blobs"]:
        ledger["blobs"][path]["status"] = status
        write_json(ledger_path, ledger)


def apply_site_blobs(section_rows: Sequence[Mapping[str, Any]], *, journal: Path, confirmed_entries: Sequence[Path]
                     ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Delete the site's blobs, each after its fresh check; then the ones an earlier pass decided.

    Returns (deleted, refused, now_shared). Runs before the site's entries are removed.
    """

    deleted, refused, shared = [], [], []
    handled = {row["path"] for row in section_rows}
    for row in section_rows:
        reason = confirm_site_blob(Path(row["path"]), row["inode"], confirmed_entries)
        if reason == "blob_gone":
            settle(journal, row["path"], "gone")
            continue
        if reason is not None:
            refused.append({"path": row["path"], "reason": reason})
            continue
        os.unlink(row["path"])
        settle(journal, row["path"], "deleted")
        deleted.append(row["path"])
    for row in pending_decisions(journal):
        if row["path"] in handled:
            continue
        reason = confirm_site_blob(Path(row["path"]), row["inode"], confirmed_entries)
        if reason == "blob_gone":
            settle(journal, row["path"], "gone")
        elif reason == "blob_has_another_referrer_now":
            # Someone links it again: it is no longer only this site's, and stays.
            settle(journal, row["path"], "kept_now_shared")
            shared.append({"path": row["path"], "reason": "linked again by another referrer after this site's "
                                                          "entries were removed"})
        elif reason is None:
            os.unlink(row["path"])
            settle(journal, row["path"], "deleted")
            deleted.append(row["path"])
        else:
            settle(journal, row["path"], "kept_" + reason)
    return deleted, refused, shared


def plan_gc(index_root: Path, roots: Sequence[Path], *, now: float, grace_seconds: float = GC_GRACE_SECONDS
            ) -> dict[str, Any]:
    """Unreferenced blobs older than the grace period, from a fully read index of every store; deletes nothing."""

    from .website_withdrawal_site_index import read_index

    plan: dict[str, Any] = {"policy_class": GC_POLICY_CLASS, "grace_seconds": grace_seconds,
                            "complete": True, "candidates": [], "bytes": 0}
    for root in map(Path, roots):
        if not root.exists():
            continue
        index = read_index(index_root, root)
        store = (index or {}).get("entries", {}).get(STORE_ENTRY)
        if not (root / STORE_ENTRY).is_dir():
            continue
        if not index or not index["sweep"].get("ever_completed") or not (store and store.get("complete")):
            plan["complete"] = False
            continue
        for relative, blob in sorted(store["blobs"].items()):
            if blob["nlink"] <= 1 and now - blob["mtime"] >= grace_seconds:
                plan["candidates"].append({"path": str(root / STORE_ENTRY / relative), "inode": blob["inode"],
                                           "size": blob["size"]})
                plan["bytes"] += blob["size"]
    if not plan["complete"]:
        plan["candidates"], plan["bytes"] = [], 0
    return plan


def apply_gc(plan: Mapping[str, Any], *, now: float, grace_seconds: float = GC_GRACE_SECONDS,
             max_bytes: int = GC_MAX_BYTES_PER_PASS) -> dict[str, Any]:
    """Delete each candidate after a fresh check: one link, the same inode, old enough, bytes match its name."""

    result: dict[str, Any] = {"deleted": [], "kept": [], "bytes_deleted": 0}
    budget = max_bytes
    for row in plan["candidates"]:
        path = Path(row["path"])
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            continue
        reason = None
        if not stat.S_ISREG(info.st_mode) or _key(info) != row["inode"]:
            reason = "replaced_since_plan"
        elif info.st_nlink != 1:
            reason = "referred_to_now"
        elif now - info.st_mtime < grace_seconds:
            reason = "younger_than_grace"
        elif info.st_size > budget:
            reason = "pass_budget_spent"
        else:
            budget -= info.st_size
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != path.name:
                reason = "bytes_do_not_match_name"
        if reason:
            result["kept"].append({"path": str(path), "reason": reason})
            continue
        os.unlink(path)
        result["deleted"].append(str(path))
        result["bytes_deleted"] += info.st_size
    result["finished_at_epoch"] = time.time()
    return result


__all__ = ["GC_GRACE_SECONDS", "GC_POLICY_CLASS", "JOURNAL_FILENAME", "STORE_ENTRY", "apply_gc", "apply_site_blobs",
           "attribute", "confirm_site_blob", "linked_inodes", "plan_gc", "record_decision", "scan_store"]
