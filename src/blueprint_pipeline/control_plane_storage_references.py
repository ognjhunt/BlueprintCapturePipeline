"""What still references a run or a derived directory: queue messages and settlement records.

A leaf module shared by the storage GC and the retention reasons it reports,
so neither reaches into the other's private helpers. The GC re-exports these
names under their old ones for its existing callers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .control_plane_retained_receipt import RETAINED_RECEIPTS

QUEUE_STATES = ("pending", "processing")
MAX_QUEUE_MESSAGE_BYTES = 16 * 1024 * 1024
# A settled scene attempt keeps reopening the local ``launch_receipt.json`` of the
# launch it settled against.  Offloading that launch run leaves the accounting and
# controls readers permanently unable to validate the attempt, which strands the
# whole intent.  Retention therefore has to read the settlement records too, not
# just the queues.
SETTLEMENT_RECORD_GLOBS = (
    "*/attempts/*.json",
    "*/cancelled-unstarted-controls/*.json",
    "*/preparations/*.json",
)


def queue_reference_text(queue_roots: Sequence[str | Path]) -> str:
    """Concatenate every pending or processing queue message; a name in it is live."""

    chunks: list[str] = []
    for raw_root in queue_roots:
        root = Path(raw_root).expanduser()
        for state in QUEUE_STATES:
            directory = root / state
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    if path.is_symlink() or path.stat().st_size > MAX_QUEUE_MESSAGE_BYTES:
                        continue
                    chunks.append(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    continue
    return "\n".join(chunks)


def settlement_reopens_beyond_retained_receipts(name: str, settlement_text: str) -> bool:
    """Whether a settlement record reads something of ``name`` the pointer will not keep.

    Offload retains the accounting receipts in ``RETAINED_RECEIPTS`` byte-for-byte
    inside the pointer, and the settlement readers reopen them through
    ``read_receipt_bytes``, which falls back to that copy. A record that names
    the run only as an identifier, or reopens only retained receipts, therefore
    keeps working after the bulk evidence is archived. Any other path under the
    run is a reopen the archive would break, so the run stays.
    """

    for match in re.finditer(re.escape(name) + r"/([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*)", settlement_text):
        if match.group(1) not in RETAINED_RECEIPTS:
            return True
    return False


def settlement_reference_text(settlement_roots: Sequence[str | Path]) -> tuple[str, int]:
    """Concatenate every settlement record; a directory named in it is still read.

    The second element counts records that exist but could not be read.  A
    configured root that cannot be enumerated must never be silently treated as
    "nothing is referenced", so the caller protects all evidence for that tick.
    """

    chunks: list[str] = []
    unreadable = 0
    for raw_root in settlement_roots:
        root = Path(raw_root).expanduser()
        if root.is_symlink() or not root.is_dir():
            unreadable += 1
            continue
        for pattern in SETTLEMENT_RECORD_GLOBS:
            try:
                paths = sorted(root.glob(pattern))
            except OSError:
                unreadable += 1
                continue
            for path in paths:
                try:
                    if path.is_symlink() or path.stat().st_size > MAX_QUEUE_MESSAGE_BYTES:
                        # A record we decline to read is a record whose references
                        # we do not know.  Count it rather than skipping it, or a
                        # symlinked or oversized record silently unprotects its run.
                        unreadable += 1
                        continue
                    chunks.append(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    unreadable += 1
    return "\n".join(chunks), unreadable


__all__ = [
    "MAX_QUEUE_MESSAGE_BYTES",
    "QUEUE_STATES",
    "SETTLEMENT_RECORD_GLOBS",
    "queue_reference_text",
    "settlement_reference_text",
    "settlement_reopens_beyond_retained_receipts",
]
