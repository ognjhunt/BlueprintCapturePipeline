"""Withdrawal deletion of a site's unclaimed website preparation handoffs.

The preparation retry keeps the bytes of a handoff no run has claimed yet in
``<handoff storage root>/.website_preparation_unclaimed/``: one entry (a file or a
directory) per handoff, holding identifiers (bucket, scene, capture), not footage.
Its per-capture twin, ``.pipeline_preparation_retry_payload.json`` beside a
capture's job ledger, lies inside ``scenes/site-<id>/`` and goes with the site's
scene files (``website_capture_withdrawal.plan_local_cleanup``).

An entry is the site's when its JSON records name the site's scene or capture id and
no other site's; it is then deleted, after its inventory is checked unchanged. An
entry naming another site too is unresolved. An entry whose records cannot be read,
or that names no scene or capture at all, is not guessed at: it is reported under
``remaining`` and left in place.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

UNCLAIMED_DIRNAME = ".website_preparation_unclaimed"
RETRY_PAYLOAD_FILENAME = ".pipeline_preparation_retry_payload.json"
HANDOFF_ROOT_ENV = "BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT"
DEFAULT_ROOT = Path(os.environ.get(HANDOFF_ROOT_ENV) or "/var/lib/blueprint/pubsub-handoffs") / UNCLAIMED_DIRNAME
_SCENE_KEYS = frozenset({"scene_id", "sceneId"})
_CAPTURE_KEYS = frozenset({"capture_id", "captureId"})
_MAX_FILES, _MAX_BYTES = 256, 4 * 1024 * 1024


class _Unreadable(Exception):
    pass


def _files(entry: Path) -> list[Path]:
    info = os.lstat(entry)
    if stat.S_ISREG(info.st_mode):
        return [entry]
    if not stat.S_ISDIR(info.st_mode):
        raise _Unreadable("special_entry")
    def unlistable(error: OSError) -> None:
        raise _Unreadable("unlistable") from error

    files: list[Path] = []
    for root, directories, names in os.walk(entry, onerror=unlistable):
        for name in (*directories, *names):
            mode = os.lstat(Path(root) / name).st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise _Unreadable("special_entry")
            if stat.S_ISREG(mode):
                files.append(Path(root) / name)
    if len(files) > _MAX_FILES:
        raise _Unreadable("too_many_files")
    return sorted(files)


def _identifiers(value: Any) -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _SCENE_KEYS and isinstance(item, str):
                yield "scene", item
            elif key in _CAPTURE_KEYS and isinstance(item, str):
                yield "capture", item
            else:
                yield from _identifiers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _identifiers(item)


def _inventory(entry: Path) -> str:
    rows = []
    for path in _files(entry):
        info = os.lstat(path)
        rows.append([str(path.relative_to(entry.parent)), info.st_size, info.st_ino, info.st_mtime_ns])
    return "sha256:" + hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def plan_unclaimed(identity: Mapping[str, str], *, root: Path | None,
                   unresolved: list[dict[str, Any]]) -> dict[str, Any]:
    """The site's unclaimed handoff entries, and the entries that could not be read (reported, never deleted)."""

    section: dict[str, Any] = {"root": str(root) if root is not None else None, "root_present": False,
                               "entries": [], "unreadable": []}
    if root is None or not Path(root).exists():
        return section
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        unresolved.append({"path": str(root), "reason": "website_withdrawal_deletion_unclaimed_root_unsafe"})
        return section
    section["root_present"] = True
    try:
        entries = sorted(root.iterdir())
    except OSError:
        unresolved.append({"path": str(root), "reason": "website_withdrawal_deletion_root_unreadable"})
        return section
    for entry in entries:
        try:
            named: set[tuple[str, str]] = set()
            for path in _files(entry):
                if path.stat().st_size > _MAX_BYTES:
                    raise _Unreadable("record_too_large")
                try:
                    named.update(_identifiers(json.loads(path.read_bytes())))
                except ValueError:
                    continue  # not a JSON record; the entry is judged by its records
            if not named:
                raise _Unreadable("no_scene_or_capture_named")
        except (_Unreadable, OSError) as error:
            reason = str(error) if isinstance(error, _Unreadable) else "unreadable"
            section["unreadable"].append({"path": str(entry), "reason": reason})
            continue
        mine = {("scene", identity["scene_id"]), ("capture", identity["capture_id"])}
        if not named & mine:
            continue
        if named - mine:
            unresolved.append({"path": str(entry), "reason": "website_withdrawal_deletion_unclaimed_names_another_site"})
            continue
        section["entries"].append({"path": str(entry), "inventory_digest": _inventory(entry)})
    return section


def apply_unclaimed(section: Mapping[str, Any]) -> list[str]:
    """Delete the planned entries, each only while its inventory is unchanged since the plan."""

    from .b2_object_versions import WithdrawalDeletionError

    deleted = []
    for row in section.get("entries") or []:
        entry = Path(row["path"])
        if not os.path.lexists(entry):
            continue
        try:
            unchanged = _inventory(entry) == row["inventory_digest"]
        except (_Unreadable, OSError):
            unchanged = False
        if not unchanged:
            raise WithdrawalDeletionError("website_withdrawal_deletion_unclaimed_changed")
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        deleted.append(str(entry))
    return deleted


def remaining(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if section.get("root") is None:
        rows.append({"holder": "unclaimed_preparation_handoffs_not_scanned",
                     "reason": "no unclaimed handoff root was named for this deletion; entries there hold the "
                               "site's identifiers (bucket, scene, capture) until a run claims them"})
    if section.get("unreadable"):
        rows.append({"holder": "unclaimed_preparation_handoffs_unreadable", "entries": list(section["unreadable"]),
                     "reason": "these entries could not be read or name no scene or capture, so whether they are "
                               "this site's is unknown; they were left in place for a person to read"})
    return rows


__all__ = ["DEFAULT_ROOT", "RETRY_PAYLOAD_FILENAME", "UNCLAIMED_DIRNAME", "apply_unclaimed", "plan_unclaimed",
           "remaining"]
