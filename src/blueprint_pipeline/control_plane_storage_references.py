"""What still references a run or a derived directory: queue messages and settlement records.

A leaf module shared by the storage GC and the retention reasons it reports,
so neither reaches into the other's private helpers. The GC re-exports these
names under their old ones for its existing callers.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
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


class QueueReferenceUnreadable(ValueError):
    """A queue root, state directory or row that cannot be read proves nothing about what it names."""


def queue_reference_text(
    queue_roots: Sequence[str | Path],
    states: Sequence[str] | Mapping[str, Sequence[str]] | None = QUEUE_STATES,
    *,
    strict: bool = False,
) -> str:
    """Concatenate queue messages; a name in them is live.

    ``states`` are the state directories read under each root: one sequence for
    every root, a mapping from a root's directory name to its states (a root it
    does not name reads ``QUEUE_STATES``), or None for every directory the root
    holds. By default only pending and processing rows are read, and a linked,
    oversized or unreadable row, or a linked state directory, is skipped: every
    original caller reads that way. ``strict`` raises ``QueueReferenceUnreadable``
    for each of those instead, and for a linked queue root, since a row that
    cannot be read proves nothing about what it names. A missing root or state
    directory holds no rows either way.
    """

    chunks: list[str] = []
    for raw_root in queue_roots:
        root = Path(raw_root).expanduser()
        for state in _queue_states(root, states, strict=strict):
            directory = root / state
            if strict:
                chunks.extend(_strict_rows(directory))
                continue
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


def _queue_states(root: Path, states, *, strict: bool) -> list[str]:
    if strict and root.is_symlink():
        raise QueueReferenceUnreadable("queue_root_linked")
    if isinstance(states, Mapping):
        return list(states.get(root.name, QUEUE_STATES))
    if states is not None:
        return list(states)
    try:
        if root.is_symlink():
            return []
        if not root.is_dir():
            return []
        with os.scandir(root) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
            linked = [entry.name for entry in children if entry.is_symlink()]
            if linked and strict:
                raise QueueReferenceUnreadable("queue_state_linked")
            return [entry.name for entry in children if not entry.is_symlink() and entry.is_dir()]
    except OSError as exc:
        if strict:
            raise QueueReferenceUnreadable("queue_root_unreadable") from exc
        return []


def _strict_rows(directory: Path) -> list[str]:
    """Every ``*.json`` row of one state directory, or ``QueueReferenceUnreadable``."""

    try:
        if directory.is_symlink():
            raise QueueReferenceUnreadable("queue_state_linked")
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise QueueReferenceUnreadable("queue_state_not_a_directory")
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries if entry.name.endswith(".json"))
    except OSError as exc:
        raise QueueReferenceUnreadable("queue_state_unreadable") from exc
    rows: list[str] = []
    for name in names:
        path = directory / name
        try:
            if path.is_symlink() or not path.is_file():
                raise QueueReferenceUnreadable("queue_row_linked")
            if path.stat().st_size > MAX_QUEUE_MESSAGE_BYTES:
                raise QueueReferenceUnreadable("queue_row_oversized")
            rows.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise QueueReferenceUnreadable("queue_row_unreadable") from exc
    return rows


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
    "QueueReferenceUnreadable",
    "SETTLEMENT_RECORD_GLOBS",
    "queue_reference_text",
    "settlement_reference_text",
    "settlement_reopens_beyond_retained_receipts",
]
