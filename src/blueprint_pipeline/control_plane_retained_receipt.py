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
from .scene_run_cold_pointer import retained_member  # noqa: F401 - re-exported; the read-only leaf holds it

RETAINED_RECEIPTS = frozenset({"launch_receipt.json"})
#: Kept as well, so a website job's re-plan reads them without restoring the run (stage 8, 2026-11-12): the
#: website-sync and post-teardown zero receipts and the robot-free probe's receipt, plus the launch
#: receipt's own terminal result when it lies inside the run (``replan_members``). capture-447's: 1.6, 2.4,
#: 5.4 and 35.5 KB, each under ``MAX_RECEIPT_BYTES``. Read only through ``retained_member``.
REPLAN_RECORDS = frozenset({"webapp_sync_succeeded.json", "post_teardown_provider_zero_receipt.json",
                            "robot_free_readiness.v1.json"})
MAX_RECEIPT_BYTES = 64 * 1024


def replan_members(directory: Path, members: list[dict]) -> set[str]:
    """The members a re-plan reads, by relative path: the fixed records and the receipt's result when inside."""
    wanted = set(REPLAN_RECORDS)
    try:
        receipt = json.loads((directory / "launch_receipt.json").read_text(encoding="utf-8"))
        result = Path(str(((receipt.get("terminal_evidence") or {}).get("result") or {}).get("path") or ""))
        wanted.add(result.relative_to(directory).as_posix())
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return {row["relative_path"] for row in members if row["relative_path"] in wanted
            and int(row.get("size_bytes", MAX_RECEIPT_BYTES + 1)) <= MAX_RECEIPT_BYTES}


def read_receipt_bytes(path: Path) -> bytes:
    from .task_evaluation_scene_retirement_metadata import read_logical_metadata
    retained = read_logical_metadata(path)
    if retained is not None:
        return retained
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
