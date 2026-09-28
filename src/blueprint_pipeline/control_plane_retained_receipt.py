"""Read exact accounting receipt bytes after bulk run evidence is offloaded.

The pointer retains a small byte-for-byte copy, checked against its archive
member inventory. Reading never downloads evidence or changes billing authority.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest

RETAINED_RECEIPTS = frozenset({"launch_receipt.json"})
MAX_RECEIPT_BYTES = 64 * 1024


def read_receipt_bytes(path: Path) -> bytes:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unstarted_controls_evidence_unsafe")
    if path.is_file():
        return path.read_bytes()
    # A present but invalid path must never be hidden by archived evidence.
    if path.exists() or path.name not in RETAINED_RECEIPTS:
        raise ValueError("unstarted_controls_evidence_unsafe")
    pointer_path = path.parent.with_name(path.parent.name + ".offloaded.v1.json")
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise ValueError("unstarted_controls_evidence_unsafe")
    pointer = json.loads(pointer_path.read_text())
    if (not isinstance(pointer, dict)
            or pointer.get("schema_version") != "control_plane_evidence_offload_pointer.v1"
            or pointer.get("status") != "offloaded"
            or pointer.get("directory") != path.parent.name
            or pointer.get("pointer_digest") != canonical_digest(pointer, digest_field="pointer_digest")):
        raise ValueError("retained_accounting_pointer_invalid")
    encoded = (pointer.get("retained_receipt_bytes") or {}).get(path.name)
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_RECEIPT_BYTES + 2) // 3):
        raise ValueError("retained_accounting_receipt_missing")
    raw = base64.b64decode(encoded, validate=True)
    members = [row for row in pointer.get("members", []) if row.get("relative_path") == path.name]
    if (len(members) != 1 or len(raw) > MAX_RECEIPT_BYTES
            or members[0].get("size_bytes") != len(raw)
            or members[0].get("sha256") != "sha256:" + hashlib.sha256(raw).hexdigest()):
        raise ValueError("retained_accounting_receipt_changed")
    return raw
