"""Why the storage GC kept what it kept: typed retention reasons, with their bytes.

On 2026-09-27 the installed reclaim timer applied with evidence offload enabled
and reclaimed nothing, and its report could not say why: the evidence manifest
folded four different reasons into one ``active_or_unsealed`` counter. A GC
manifest now carries ``retained_by_reason``: every reason it kept an entry for,
as ``{reason: {"count": n, "bytes": b}}``. A reason is a typed string, never a
path, and ``bytes`` are logical bytes (a hardlinked file counts once per name).
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from . import completed_replay_cache_retention as retention
from .control_plane_storage_pins import live_pinned_paths, load_storage_pins

PROTECTED_UNREADABLE_SETTLEMENT = "protected_unreadable_settlement"
PROTECTED_PROCESS = "protected_process"
PROTECTED_PROCESS_INVENTORY_UNREADABLE = "protected_process_inventory_unreadable"
PROTECTED_PIN = "protected_pin"
PROTECTED_SETTLEMENT = "protected_settlement"
PROTECTED_QUEUE = "protected_queue"
#: In the order ``evidence_protection_reason`` checks them.
EVIDENCE_PROTECTION_REASONS = (
    PROTECTED_UNREADABLE_SETTLEMENT,
    PROTECTED_PROCESS,
    PROTECTED_PROCESS_INVENTORY_UNREADABLE,
    PROTECTED_PIN,
    PROTECTED_SETTLEMENT,
    PROTECTED_QUEUE,
)


def evidence_protection_reason(
    directory: Path,
    *,
    settlement_roots: Sequence[str | Path],
    pins_root: str | Path,
    queue_roots: Sequence[str | Path],
    now: Callable[[], float],
    ignored_process_ids: Sequence[int] = (),
    process_root: Path = Path("/proc"),
) -> str | None:
    """Why the evidence phase must leave ``directory`` in place, or None when nothing protects it.

    The checks run in the order the storage GC always ran them, and the first that
    holds names the reason:

    1. ``protected_unreadable_settlement``: a settlement root or record cannot be
       read, which proves nothing is unreferenced;
    2. ``protected_process``, or ``protected_process_inventory_unreadable`` when a
       process entry could not be read and none was seen holding the directory;
    3. ``protected_pin``: a live storage pin names it, lies inside it or contains it;
    4. ``protected_settlement``: a settlement record reopens a path under it that
       the offload pointer does not retain;
    5. ``protected_queue``: a pending or processing queue message names it.

    Settlement records and queue messages are re-read on every call.
    ``apply_evidence_offload`` re-checks protection immediately before it evicts
    each candidate, so a settlement written after the manifest was built must
    protect its launch run at that final check too.
    """

    from .control_plane_storage_gc import (
        _queue_reference_text,
        _settlement_reference_text,
        settlement_reopens_beyond_retained_receipts,
    )

    settlement_text, settlement_unreadable = _settlement_reference_text(settlement_roots)
    if settlement_unreadable:
        return PROTECTED_UNREADABLE_SETTLEMENT
    process = retention.process_reference(
        directory, process_root=process_root, ignored_process_ids=ignored_process_ids
    )
    if process == retention.PROCESS_INVENTORY_UNREADABLE:
        return PROTECTED_PROCESS_INVENTORY_UNREADABLE
    if process:
        return PROTECTED_PROCESS
    pinned = live_pinned_paths(pins_root, now=now)
    if any(Path(p) == directory or directory in Path(p).parents or Path(p) in directory.parents for p in pinned):
        return PROTECTED_PIN
    if settlement_reopens_beyond_retained_receipts(directory.name, settlement_text):
        return PROTECTED_SETTLEMENT
    if directory.name in _queue_reference_text(queue_roots):
        return PROTECTED_QUEUE
    return None


def count_retained(
    by_reason: MutableMapping[str, dict[str, Any]], reason: str, size_bytes: int, *, kind: str | None = None
) -> None:
    """Count one retained entry of ``size_bytes`` under ``reason``, and under ``kind`` within it when given."""

    row = by_reason.setdefault(reason, {"count": 0, "bytes": 0})
    row["count"] += 1
    row["bytes"] += int(size_bytes)
    if kind is not None:
        detail = row.setdefault("by_kind", {}).setdefault(kind, {"count": 0, "bytes": 0})
        detail["count"] += 1
        detail["bytes"] += int(size_bytes)


def walked_bytes(walk: Callable[[Path], tuple[Any, ...]], directory: Path) -> int:
    """The bytes ``walk`` (a manifest's ``_tree_snapshot``) finds in a kept directory.

    A directory that vanished since it was listed holds none; sizing what is
    kept never fails the phase that keeps it.
    """

    try:
        return int(walk(directory)[1])
    except OSError:
        return 0


def entry_bytes(entry: Path) -> int:
    """A link's or a stray file's own size: an unsafe entry is never followed."""

    try:
        return entry.lstat().st_size
    except OSError:
        return 0


def live_pin_kinds(pins_root: str | Path, *, now: Callable[[], float]) -> dict[str, str]:
    """Each live-pinned path with the kinds of the pins that name it, sorted and joined by ``+``.

    For reporting only: the paths are exactly ``live_pinned_paths``, which alone
    decides what is pinned.
    """

    kinds: dict[str, set[str]] = {}
    for pin in load_storage_pins(pins_root, now=now):
        if pin["status"] == "live":
            for path in pin.get("paths") or []:
                kinds.setdefault(str(path), set()).add(str(pin["kind"]))
    return {path: "+".join(sorted(names)) for path, names in kinds.items()}


__all__ = [
    "EVIDENCE_PROTECTION_REASONS",
    "PROTECTED_PIN",
    "PROTECTED_PROCESS",
    "PROTECTED_PROCESS_INVENTORY_UNREADABLE",
    "PROTECTED_QUEUE",
    "PROTECTED_SETTLEMENT",
    "PROTECTED_UNREADABLE_SETTLEMENT",
    "count_retained",
    "entry_bytes",
    "evidence_protection_reason",
    "live_pin_kinds",
    "walked_bytes",
]
