#!/usr/bin/env python3
"""Measure host folders for an owner-approved deletion manifest, through the operator door's read-only ``ls``.

For every direct child directory of ``--parent`` (or only the ``--only`` names) it walks the tree with
``operator_door.py ls`` and records what ``owner_approved_deletion`` compares: the newest mtime over the
folder and everything in it, the regular-file bytes and count, and how many names the door would not
list (``door_unlisted_entries``; the tool then accepts only a larger tree). It writes manifest entries
of ``--entry-class``; a folder it cannot measure in full (unreadable, a listing it cannot complete even
split by name prefix, over the call budget) is reported under ``unmeasured`` and never becomes an entry.
It changes nothing on the host.

    python3 scripts/measure_owner_approved_deletion_folders.py \\
        --parent /var/lib/blueprint/task-evaluation-inputs/<allowed parent> \\
        --entry-class task_evaluation_inputs_subfolder --output entries.json
"""

from __future__ import annotations

import argparse
import json
import string
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DOOR = Path(__file__).resolve().parent / "operator_door.py"
INPUTS_ROOT = "/var/lib/blueprint/task-evaluation-inputs"


#: First characters a listing too long for the door is split by (``ls --match '[c]*'``); the parts must add
#: up to the full listing's ``total`` or the folder is not measured.
_SPLIT = string.ascii_letters + string.digits + "._-+@,=~%#"


def door_ls(path: str, match: str | None = None) -> dict | None:
    command = [sys.executable, str(DOOR), "ls", *(["--match", match] if match else []), path]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    try:
        listing = json.loads(result.stdout)
    except ValueError:
        return None
    return listing if result.returncode == 0 and isinstance(listing, dict) else None


def door_entries(path: str, calls: list[int], prefix: str = "") -> list[dict] | None:
    """Every entry of ``path``, splitting a truncated listing by name prefix; ``None`` when incomplete."""

    calls[0] += 1
    listing = door_ls(path, "".join(f"[{c}]" for c in prefix) + "*" if prefix else None)
    if listing is None:
        return None
    if not listing.get("truncated"):
        return list(listing.get("entries") or [])
    if len(prefix) >= 16:
        return None
    entries: list[dict] = []
    for character in _SPLIT:
        part = door_entries(path, calls, prefix + character)
        if part is None:
            return None
        entries.extend(part)
    return entries if len(entries) == listing.get("total") else None


def measure(parent: str, top: dict, entry_class: str, budget: int, workers: int) -> dict:
    path = f"{parent}/{top['name']}"
    newest, apparent, files, unlisted, calls = int(top["mtime"]), 0, 0, 0, 0
    level = [path]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        while level:  # one directory level at a time, listed in parallel
            if calls >= budget:
                return {"path": path, "status": "over_budget", "calls": calls}
            counters = [[0] for _ in level]
            listings = list(pool.map(door_entries, level, counters))
            calls += sum(counter[0] for counter in counters)
            following = []
            for current, listing in zip(level, listings):
                if listing is None:
                    return {"path": path, "status": "unreadable_or_incomplete", "at": current}
                for entry in listing:
                    if "refused" in entry or "mtime" not in entry:
                        unlisted += 1
                        continue
                    newest = max(newest, int(entry["mtime"]))
                    if entry.get("type") == "file":
                        apparent += int(entry.get("size", 0))
                        files += 1
                    elif entry.get("type") == "dir":
                        following.append(f"{current}/{entry['name']}")
            level = following
    return {"class": entry_class, "path": path, "newest_mtime_epoch": newest, "apparent_bytes": apparent,
            "file_count": files, "door_unlisted_entries": unlisted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--entry-class", required=True)
    parser.add_argument("--only", action="append", default=[], help="measure only this child name (repeatable)")
    parser.add_argument("--budget", type=int, default=50000, help="door calls per folder")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    parent = args.parent.rstrip("/")
    if parent != INPUTS_ROOT and not parent.startswith(INPUTS_ROOT + "/"):
        parser.error("--parent must be or lie under " + INPUTS_ROOT)
    listing = door_ls(parent)
    if listing is None or listing.get("truncated"):
        print(json.dumps({"error": "parent_unlistable", "parent": parent}), file=sys.stderr)
        return 1
    tops = [entry for entry in listing["entries"] if entry.get("type") == "dir" and "mtime" in entry
            and (not args.only or entry["name"] in args.only)]
    rows = [measure(parent, top, args.entry_class, args.budget, args.workers) for top in tops]
    entries = sorted((row for row in rows if "status" not in row), key=lambda row: row["path"])
    unmeasured = [row for row in rows if "status" in row]
    args.output.write_text(json.dumps({"parent": parent, "entries": entries, "unmeasured": unmeasured}, indent=1)
                           + "\n", encoding="utf-8")
    print(json.dumps({"measured": len(entries), "unmeasured": len(unmeasured),
                      "apparent_bytes": sum(row["apparent_bytes"] for row in entries)}))
    return 0 if not unmeasured else 1


if __name__ == "__main__":
    sys.exit(main())
