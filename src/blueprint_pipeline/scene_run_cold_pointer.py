"""Read-only facts about a scene run in cold storage: its pointer, the bytes the pointer kept, and whether
its stored archive failed its integrity check.

A leaf: it imports nothing but the digest helper, so read-only evidence modules (the screening re-plan's
source check, the robot-free probe gate) can read an offloaded run without reaching any worker
(``tests/test_controls_read_evidence.py``). The worker side (``scene_run_cold_storage``,
``scene_run_restore``, ``control_plane_retained_receipt``) imports these from here.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

COLD_STORAGE = "scene_run_in_cold_storage"
POINTER_SUFFIX = ".offloaded.v1.json"
POINTER_SCHEMA = "control_plane_evidence_offload_pointer.v1"
#: The restore's own record beside the run (``scene_run_restore``).
RESTORE_STATE_SUFFIX = ".restore.v1.json"
#: The archive in the store does not match its pointer: tried again only for a changed pointer. Published as the
#: hold's detail under ``team_scene_packet_blocked`` (``scene_run_cold_storage.cold_storage_hold``).
ARCHIVE_INVALID = "scene_run_restore_archive_integrity_failed"


def offload_pointer(run_root: str | Path) -> dict[str, Any] | None:
    """The sealed offload pointer standing where ``run_root`` was, or None.

    None while the directory is on disk, and for anything that is not a valid,
    self-digest-intact pointer of exactly this directory: a missing run without
    a pointer stays the reader's own "unavailable" refusal.
    """

    run = Path(run_root)
    if not run.is_absolute() or run.is_dir():
        return None
    pointer = run.parent / (run.name + POINTER_SUFFIX)
    try:
        if pointer.is_symlink() or not pointer.is_file():
            return None
        value = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(value, dict) or value.get("schema_version") != POINTER_SCHEMA
            or value.get("status") != "offloaded" or value.get("directory") != run.name
            or value.get("pointer_digest") != canonical_digest(value, digest_field="pointer_digest")):
        return None
    return value


def retained_member(run_root: Path, relative: str) -> bytes | None:
    """The retained bytes of ``relative`` from the run's sealed offload pointer, checked; None when not kept."""
    pointer_path = run_root.parent / (run_root.name + POINTER_SUFFIX)
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (pointer_path.is_symlink() or not isinstance(pointer, dict) or pointer.get("status") != "offloaded"
            or pointer.get("directory") != run_root.name
            or pointer.get("pointer_digest") != canonical_digest(pointer, digest_field="pointer_digest")):
        return None
    encoded = (pointer.get("retained_receipt_bytes") or {}).get(relative)
    rows = [row for row in pointer.get("members", []) if row.get("relative_path") == relative]
    if not isinstance(encoded, str) or len(rows) != 1:
        return None
    raw = base64.b64decode(encoded, validate=True)
    if rows[0].get("size_bytes") != len(raw) or rows[0].get("sha256") != "sha256:" + hashlib.sha256(raw).hexdigest():
        return None
    return raw


def restore_state(run_root: str | Path) -> dict[str, Any]:
    """The restore's record beside the run, or ``{"attempts": 0}`` when there is none or it is unreadable."""
    run = Path(run_root)
    try:
        value = json.loads((run.parent / (run.name + RESTORE_STATE_SUFFIX)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"attempts": 0}
    return value if isinstance(value, dict) else {"attempts": 0}


def archive_invalid(run_root: str | Path) -> bool:
    """Whether the run's stored archive failed its integrity check for the pointer that stands now."""
    pointer = offload_pointer(run_root)
    state = restore_state(run_root)
    return (pointer is not None and state.get("status") == ARCHIVE_INVALID
            and state.get("pointer_digest") == pointer.get("pointer_digest"))


__all__ = ["ARCHIVE_INVALID", "COLD_STORAGE", "POINTER_SCHEMA", "POINTER_SUFFIX", "RESTORE_STATE_SUFFIX",
           "archive_invalid", "offload_pointer", "restore_state", "retained_member"]
