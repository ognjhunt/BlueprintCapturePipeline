"""Reclaim the lookahead scratch that several activation lookaheads share.

Until 2026-09-20 the content store sat on the root disk beside the activations, so a parent
replay (``task_evaluation_stage_replay.replay_parent``) hard-linked the store's own blobs into
its scratch inputs. The store then moved to a work volume and its root copies were deleted, so
each such inode now lives only in names spread across many lookahead replays. The per-replay
scratch rule (``completed_replay_cache_retention`` with ``reclaim_scratch_inputs``) plans an inode
only when one replay holds all of its links, so it never plans these. On 2026-09-28, 89
lookahead replays still held scratch, and 957 of their 964 blob digests (9.07 GB) sat in more
than one of them.

One plan per storage GC tick covers every lookahead the replay cache phase scans. A replay is
eligible exactly when the per-replay rule would take its scratch inputs: its finished parent
report says it ran there with no paid execution and no provider mutation, it closed at least
the phase's window ago, and no live process references it (the same process check, an
unreadable inventory counting as a reference). Each replay's ``prepared-references`` is walked
without following a link or entering a directory on another device, and every regular file is
grouped by inode. A group whose names all sit in one replay stays with the per-replay rule. A
group with names in two or more replays is a candidate when those names are all in eligible
replays and are every link it has, it is on the device of every replay holding it, and it is
not newer than any of their reports.

Apply first rechecks every replay holding a candidate (the same report, unchanged, no live
reader), then unlinks a candidate's names one replay at a time through directory descriptors
held from that replay down, rechecking each name before it goes: its device, inode, size and
mtime, and a link count equal to the names still to go. A group's bytes count only when its
last name goes. A recheck that fails stops the group; the names already unlinked were scratch
and stay unlinked, and the next tick plans the rest, whose names are again all of its links.
The directories left empty in each replay touched are then pruned as the per-replay rule
prunes them. No file's bytes are read.
"""

from __future__ import annotations

import errno
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import completed_replay_cache_retention as retention

#: Per-row detail kept in a report, as the terminal pin phase keeps it; every counter covers every group.
MAX_ROWS = 200
_TYPE_WORD = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _unsafe(root: Path) -> bool:
    return not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents))


def _walk(child: Path) -> tuple[int, list[tuple[os.stat_result, Path]]] | None:
    """The child's st_dev and every regular file under its ``prepared-references``, by name
    relative to the child, or None for a child that is gone.

    A link is never followed and a directory on another device than the child's is never
    entered; nothing reached through a linked directory counts, and an entry that vanishes
    while it is walked names nothing.
    """

    try:
        device = os.lstat(child).st_dev
    except OSError:
        return None
    subtree = child / retention._SCRATCH_INPUTS
    found: list[tuple[os.stat_result, Path]] = []
    if not retention._directory_on(subtree, device):
        return device, found
    for directory, directories, names in os.walk(subtree):
        here = Path(directory)
        directories[:] = [name for name in directories if retention._directory_on(here / name, device)]
        if any(p.is_symlink() for p in (here, *here.parents) if p != child.parent):
            continue
        for name in names:
            try:
                info = os.lstat(here / name)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                found.append((info, (here / name).relative_to(child)))
    return device, found


def _gate(child: Path, *, clock: float, minimum_closed_seconds: int) -> tuple[str | None, str | None, int | None]:
    """The report gate of the per-replay scratch rule ``child`` fails (None when it passes), with
    the report's path and mtime."""

    try:
        report, ran_here = retention._finished_report(child, any_parent_status=True)
        info = None if report is None else report.stat()
    except OSError:
        return "no_finished_report", None, None
    if report is None or info is None:
        return "no_finished_report", None, None
    if not ran_here:
        return "report_not_parent_replay", str(report), info.st_mtime_ns
    if clock - info.st_mtime < minimum_closed_seconds:
        return "closed_too_recently", str(report), info.st_mtime_ns
    return None, str(report), info.st_mtime_ns


def _kept_reason(group: dict[str, Any], holders: Sequence[dict[str, Any]]) -> str | None:
    holding = {index for index, _name in group["names"]}
    if any(holders[index]["device"] != group["dev"] for index in holding):
        return "cross_device"
    if group["nlink"] > len(group["names"]):
        # Every lookahead's names are counted, so a link beyond them is outside every lookahead.
        return "linked_outside_lookaheads"
    if any(holders[index]["gate"] for index in holding):
        return "holder_ineligible"
    if group["nlink"] < len(group["names"]):
        # Counted twice (a bind mount of the same filesystem keeps its st_dev): not to be trusted.
        return "more_names_than_links"
    if any(group["mtime_ns"] > holders[index]["report_mtime_ns"] for index in holding):
        return "newer_than_report"
    return None


def plan_shared_scratch(
    lookaheads: Sequence[str | Path], *, now: float, minimum_closed_seconds: int, check_readers: bool,
    process_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Group every scratch input under the lookaheads' replays by inode; plan the groups held by two
    or more replays that may go. Live readers are swept, once, only with ``check_readers``."""

    holders: list[dict[str, Any]] = []
    inodes: dict[tuple[int, int], tuple[os.stat_result, list[tuple[int, Path]]]] = {}
    for lookahead in lookaheads:
        root = Path(lookahead)
        try:
            if _unsafe(root):
                continue
            children = sorted(root.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_symlink() or not child.is_dir():
                continue
            walked = _walk(child)
            if walked is None or not walked[1]:
                continue
            index = len(holders)
            holders.append({"path": child, "device": walked[0], "gate": None, "report": None,
                            "report_mtime_ns": None})
            for info, name in walked[1]:
                inodes.setdefault((info.st_dev, info.st_ino), (info, []))[1].append((index, name))
    groups = [
        {"dev": info.st_dev, "inode": info.st_ino, "nlink": info.st_nlink, "size_bytes": info.st_size,
         "mtime_ns": info.st_mtime_ns, "names": sorted(names, key=lambda item: (item[0], str(item[1])))}
        for info, names in inodes.values()
        if len({index for index, _name in names}) > 1
    ]
    involved = sorted({index for group in groups for index, _name in group["names"]})
    for index in involved:
        gate, report, mtime_ns = _gate(holders[index]["path"], clock=now,
                                       minimum_closed_seconds=minimum_closed_seconds)
        holders[index].update(gate=gate, report=report, report_mtime_ns=mtime_ns)
    if check_readers and any(holders[index]["gate"] is None for index in involved):
        referenced = retention.process_reference_index(process_root=process_root)
        for index in involved:
            if holders[index]["gate"] is None and referenced(holders[index]["path"]):
                holders[index]["gate"] = "active_reference"
    candidates, kept = [], []
    for group in sorted(groups, key=lambda group: _first_path(group, holders)):
        reason = _kept_reason(group, holders)
        if reason:
            kept.append((group, reason))
        else:
            candidates.append(group)
    return {"clock": now, "minimum_closed_seconds": minimum_closed_seconds, "holders": holders,
            "involved": involved, "candidates": candidates, "kept": kept}


def _first_path(group: dict[str, Any], holders: Sequence[dict[str, Any]]) -> str:
    index, name = group["names"][0]
    return str(holders[index]["path"] / name)


def _row(group: dict[str, Any], holders: Sequence[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    """A report row for one group: its first name, how many names and holders it has, its links and size."""

    row = {"path": _first_path(group, holders), "name_count": len(group["names"]),
           "holder_count": len({index for index, _name in group["names"]}), "nlink": group["nlink"],
           "size_bytes": group["size_bytes"]}
    return {"reason": reason, **row} if reason else row


def _capped(block: dict[str, Any], key: str, rows: Sequence[dict[str, Any]]) -> None:
    block[key] = list(rows[:MAX_ROWS])
    block[f"omitted_{key}_count"] = max(0, len(rows) - MAX_ROWS)


def _by_holder(group: dict[str, Any]) -> list[tuple[int, list[Path]]]:
    names: dict[int, list[Path]] = {}
    for index, name in group["names"]:
        names.setdefault(index, []).append(name)
    return sorted(names.items())


def _failure(exc: OSError) -> str:
    """A typed reason for an error met while rechecking or unlinking."""

    if isinstance(exc, FileNotFoundError):
        return "vanished"
    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
        return "path_changed"
    return _TYPE_WORD.sub("_", type(exc).__name__).lower()


def _changed(info: os.stat_result, group: dict[str, Any], links: int, device: int) -> str | None:
    """Why the name looked at is no longer one of the planned group's ``links`` remaining links."""

    if info.st_dev != device:
        return "cross_device"
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != (group["dev"], group["inode"]):
        return "changed"
    if info.st_nlink > links:
        return "extra_link"
    if info.st_nlink < links:
        return "vanished"
    if (info.st_size, info.st_mtime_ns) != (group["size_bytes"], group["mtime_ns"]):
        return "changed"
    return None


def _unlink_names(held: Any, group: dict[str, Any], names: Sequence[Path], links: int) -> tuple[str | None, int]:
    """Recheck and unlink one holder's names of a group through its held descriptors, in order."""

    unlinked = 0
    for name in names:
        try:
            directory = held.directory(name.parts[:-1])
            info = retention._leaf(directory, name.name)
            why = _changed(info, group, links - unlinked, held.device)
            if why is None and not (
                held.in_place(name.parts[:-1]) and retention._same_leaf(directory, name.name, info)
            ):
                why = "path_changed"
            if why:
                return why, unlinked
            os.unlink(name.name, dir_fd=directory)
        except OSError as exc:
            return _failure(exc), unlinked
        unlinked += 1
    return None, unlinked


def _unlink_group(group: dict[str, Any], holders: Sequence[dict[str, Any]]) -> tuple[str | None, list[int]]:
    """Unlink every name of a planned group, one holder at a time: why it stopped (None when its last
    name went), and the holders a name was unlinked in."""

    links, touched = group["nlink"], []
    for index, names in _by_holder(group):
        child = holders[index]["path"]
        try:
            held = retention._HeldChild(child.parent, child.name)
        except OSError as exc:
            return _failure(exc), touched
        try:
            if not held.named_by(child):
                return "path_changed", touched
            if held.device != group["dev"]:
                return "cross_device", touched
            why, unlinked = held.item(_unlink_names, group, names, links)
        finally:
            held.close()
        links -= unlinked
        if unlinked:
            touched.append(index)
        if why:
            return why, touched
    return None, touched


def _prune(child: Path) -> list[dict[str, str]]:
    """Remove the directories left empty in the child's ``prepared-references``, as the per-replay
    rule does; typed skips."""

    try:
        held = retention._HeldChild(child.parent, child.name)
    except OSError as exc:
        return [{"path": str(child), "reason": f"prune_failed:{type(exc).__name__}"}]
    try:
        if not held.named_by(child):
            return [{"path": str(child), "reason": "prune_failed:changed"}]
        return held.item(retention._remove_empty_directories, child)
    finally:
        held.close()


def apply_shared_scratch(plan: dict[str, Any], *, process_root: Path = Path("/proc")) -> dict[str, Any]:
    """Remove the plan's candidates: ``removed`` groups, ``kept`` ``(group, reason)`` for each apply
    stopped, and the prune's typed skips."""

    holders = plan["holders"]
    removed: list[dict[str, Any]] = []
    kept: list[tuple[dict[str, Any], str]] = []
    touched: set[int] = set()
    if not plan["candidates"]:
        return {"removed": removed, "kept": kept, "prune_skipped": []}
    referenced = retention.process_reference_index(process_root=process_root)
    verdicts: dict[int, bool] = {}

    def still_eligible(index: int) -> bool:
        if index not in verdicts:
            holder = holders[index]
            gate, report, mtime_ns = _gate(holder["path"], clock=plan["clock"],
                                           minimum_closed_seconds=plan["minimum_closed_seconds"])
            verdicts[index] = (gate is None and (report, mtime_ns) == (holder["report"], holder["report_mtime_ns"])
                               and not referenced(holder["path"]))
        return verdicts[index]

    for group in plan["candidates"]:
        if not all(still_eligible(index) for index, _names in _by_holder(group)):
            kept.append((group, "recheck_failed:holder_ineligible"))
            continue
        why, unlinked_in = _unlink_group(group, holders)
        touched.update(unlinked_in)
        if why:
            kept.append((group, f"recheck_failed:{why}"))
        else:
            removed.append(group)
    pruned = [row for index in sorted(touched) for row in _prune(holders[index]["path"])]
    return {"removed": removed, "kept": kept, "prune_skipped": pruned}


def reclaim_shared_scratch(
    lookaheads: Sequence[str | Path], *, now: float, minimum_closed_seconds: int, enabled: bool, apply: bool,
    check_readers: bool, process_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Plan the tick's shared scratch across ``lookaheads``, remove it with ``apply``, and report both.

    ``enabled`` is whether the owner has enabled the removal (both switches); ``status`` is
    ``applied`` only for a tick that removed. ``live_readers_checked`` says whether the plan swept
    the process table, which a tick that does not apply never does: its candidates are an upper bound.
    """

    check_readers = check_readers or apply
    plan = plan_shared_scratch(lookaheads, now=now, minimum_closed_seconds=minimum_closed_seconds,
                               check_readers=check_readers, process_root=process_root)
    block: dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "applied" if apply else "dry_run",
        "live_readers_checked": bool(check_readers),
        "candidate_groups": len(plan["candidates"]),
        "candidate_bytes": sum(group["size_bytes"] for group in plan["candidates"]),
        "removed_groups": 0,
        "removed_bytes": 0,
    }
    kept = list(plan["kept"])
    if apply:
        result = apply_shared_scratch(plan, process_root=process_root)
        block["removed_groups"] = len(result["removed"])
        block["removed_bytes"] = sum(group["size_bytes"] for group in result["removed"])
    by_reason: dict[str, dict[str, int]] = {}
    for group, reason in kept:
        counted = by_reason.setdefault(reason, {"groups": 0, "bytes": 0})
        counted["groups"] += 1
        counted["bytes"] += group["size_bytes"]
    block["kept_by_reason"] = dict(sorted(by_reason.items()))
    # Every replay holding a name of a group two or more hold, by the gate it failed.
    gates: dict[str, int] = {}
    for index in plan["involved"]:
        gate = plan["holders"][index]["gate"] or "eligible"
        gates[gate] = gates.get(gate, 0) + 1
    block["holders_by_gate"] = dict(sorted(gates.items()))
    _capped(block, "candidates", [_row(group, plan["holders"]) for group in plan["candidates"]])
    _capped(block, "kept", [_row(group, plan["holders"], reason) for group, reason in kept])
    return block


__all__ = ["apply_shared_scratch", "plan_shared_scratch", "reclaim_shared_scratch"]
