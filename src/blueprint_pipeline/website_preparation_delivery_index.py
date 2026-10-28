"""Recoverable due hints for the existing preparation delivery records.

This private index carries no source or delivery authority. Dispatch always
reopens the canonical delivery file, ledger and current signed source. Losing
an index entry is repaired by the listener's separate bounded ledger sweep.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .common import write_json

DIRECTORY = ".website_preparation_pending"
SCHEMA_VERSION = "website_preparation_delivery_hint.v1"
METRICS_FILE = ".website_preparation_delivery_metrics.json"
MAX_HINT_SCAN = 20000


def _location(capture_root: Path) -> tuple[Path, str]:
    root = Path(capture_root).absolute()
    if (root.parent.name != "captures" or root.parents[2].name != "scenes"
            or not root.parents[1].name.startswith("site-")
            or not root.name.startswith(("walkthrough-", "supplement-"))
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError("preparation_delivery_index_root_invalid")
    storage = root.parents[4]
    return storage, root.relative_to(storage).as_posix()


@contextmanager
def _locked(storage_root: Path):
    directory = Path(storage_root) / DIRECTORY
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("preparation_delivery_index_unsafe")
    descriptor = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("preparation_delivery_index_unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield directory
    finally:
        os.close(descriptor)


def _number(value: Any, default: float) -> float:
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else default


def update(capture_root: Path, delivery: Mapping[str, Any]) -> None:
    storage, relative = _location(capture_root)
    key = hashlib.sha256(relative.encode()).hexdigest()
    with _locked(storage) as directory:
        path = directory / (key + ".json")
        if path.is_symlink():
            raise ValueError("preparation_delivery_index_unsafe")
        if delivery.get("state") != "pending":
            # A newer revision's hint must survive a late acknowledgement.
            try:
                old = json.loads(path.read_text())
            except (FileNotFoundError, ValueError):
                return
            if old.get("revision") == delivery.get("revision"):
                path.unlink()
            return
        write_json(path, {"schema_version": SCHEMA_VERSION, "capture_relative_path": relative,
                          "revision": delivery.get("revision"),
                          "next_attempt_at_epoch": _number(delivery.get("next_attempt_at_epoch"), 0.0),
                          "pending_since_epoch": _number(delivery.get("pending_since_epoch"), 0.0)})


def due(storage_root: Path, *, now: float, limit: int) -> tuple[list[Path], dict[str, Any]]:
    storage = Path(storage_root).absolute()
    rows = []
    invalid = scanned = 0
    with _locked(storage) as directory:
        for path in directory.iterdir():
            if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
                continue
            if scanned >= MAX_HINT_SCAN:
                break
            scanned += 1
            try:
                if path.is_symlink():
                    raise ValueError("unsafe")
                hint = json.loads(path.read_text())
                relative = hint["capture_relative_path"]
                root = storage / relative
                actual_storage, actual_relative = _location(root)
                if (hint.get("schema_version") != SCHEMA_VERSION or actual_storage != storage
                        or actual_relative != relative
                        or path.stem != hashlib.sha256(relative.encode()).hexdigest()):
                    raise ValueError("invalid")
                next_attempt = _number(hint.get("next_attempt_at_epoch"), 0.0)
                if next_attempt <= now:
                    rows.append((next_attempt, _number(hint.get("pending_since_epoch"), now), root))
            except (OSError, ValueError, KeyError, TypeError):
                invalid += 1
    rows.sort(key=lambda row: (row[0], row[1], str(row[2])))
    return [row[2] for row in rows[:limit]], {
        "due_backlog": len(rows), "oldest_due_age_seconds": max([0.0, *(now - row[1] for row in rows)]),
        "hint_count_scanned": scanned, "hint_scan_limit_reached": scanned >= MAX_HINT_SCAN,
        "invalid_hint_count": invalid,
        "overloaded": len(rows) > limit or scanned >= MAX_HINT_SCAN}
