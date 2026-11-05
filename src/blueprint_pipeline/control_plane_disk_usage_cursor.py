"""Bounded partition continuation for the usage survey: a cycle that completes across passes.

Each hourly pass walks some partitions of each surveyed mount and records
their measurements in a ledger carried by the cursor. The published survey is
the composite of the latest measurement of every current partition, so after
enough passes it covers the whole mount. The cursor selects observation work
and keeps measurements only; it conveys no deletion authority.

Three things kept the v2 cursor from ever completing on the production host
(2026-11-04: 0.11% of the root disk surveyed, ``stale_cursor_identity_refused``
every hour):

* every root's position was tied to a digest of the whole mount table, so any
  unrelated mount change (a transient unit's credentials, a container layer)
  reset all three roots to their first four partitions every pass;
* a partition that disappeared or was replaced (normal churn on a work
  volume) was reported as a refused cursor;
* a symlink beside the root's directories (``/bin`` on a merged-usr host)
  left the inventory incomplete forever, and a tree too large for one pass
  (``/opt`` with 93 release checkouts) could never finish.

Now a root's position depends only on that root's own identity; departed or
replaced partitions are counted as churn; a symlink is a leaf partition that
is never followed; and a partition whose walk truncates is split, on a later
pass, into its subdirectories plus a shallow partition for its own entries.
A measurement is used in the composite only while the partition still has the
identity it was measured with.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .control_plane_usage_paths import (
    UsageDirectory, UsagePathError, anchored_directory, descendant_directory,
)
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_usage_partition_cursor.v3"
_COMPATIBLE_SCHEMAS = frozenset({"control_plane_usage_partition_cursor.v2", SCHEMA_VERSION})
MAX_PARTITION_CANDIDATES = 20_000
MAX_PARTITIONS_PER_MOUNT = 4_096
#: How deep a truncated partition may be split, below its mount root.
MAX_SPLIT_DEPTH = 8
MAX_SPLITS_PER_ROOT = 4_000
#: The composite is ``complete`` only while every measurement is this fresh.
COMPOSITE_MAX_AGE_SECONDS = 24 * 60 * 60
SHALLOW = "."
_PRIORITY_NAMES = frozenset(("pubsub-handoffs", "task-evaluation-inputs", "pipeline-control-plane",
                             "production-gpu-artifacts", "lanes", "capture-reconstruction-runs",
                             "capture-reconstruction-derived", "beta-policy-screening", "team-scene-packets",
                             "rigid-robot-free-probes", "workspace"))


def _digest(name: str) -> str:
    # Persist no untrusted file names in the cursor. A digest locates an entry in
    # this tick's bounded inventory; it never authorizes a path open. 128 bits keep
    # nested keys short.
    return hashlib.sha256(os.fsencode(name)).hexdigest()[:32]


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)]


@dataclass
class Candidate:
    """One partition: a whole subtree, a leaf, or a split directory's own entries (``shallow``)."""

    key: str
    path: str
    info: os.stat_result
    names: tuple[str, ...]
    shallow: bool = False


@dataclass
class RootPlan:
    key: str
    path: str
    identity: dict[str, int]
    candidates: dict[str, Candidate] = field(default_factory=dict)
    queue: list[str] = field(default_factory=list)
    splits: dict[str, list[int]] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    complete: bool = True
    inventory_digest: str = ""


@dataclass
class PartitionPlan:
    roots: list[RootPlan]
    order: list[tuple[RootPlan, Candidate]]
    reasons: set[str]
    churn: dict[str, int]
    examined: int
    old_valid: bool
    mount_digest: str


def _sort_key(candidate: Candidate) -> tuple[Any, ...]:
    leading = candidate.names[0] if candidate.names else ""
    return (leading not in _PRIORITY_NAMES, candidate.names, candidate.shallow)


def _inventory(
    plan: RootPlan, anchor: UsageDirectory, directory: str, names: tuple[str, ...], key: str, *,
    device: int, mount_points: frozenset[str], deadline: float, clock: Callable[[], float],
    seen_splits: set[str], counter: list[int], reasons: set[str],
) -> None:
    """Add the partitions below one directory, expanding the ones a truncated walk split."""

    with os.scandir(anchor.fd) as entries:
        rows = []
        for entry in entries:
            if len(plan.candidates) + len(rows) >= MAX_PARTITION_CANDIDATES or clock() >= deadline:
                plan.complete = False
                reasons.add("partition_queue_limit_reached" if clock() < deadline else "partition_inventory_timed_out")
                break
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                plan.complete = False
                continue
            counter[0] += 1
            rows.append((entry.name, info))
    anchor.check()
    for name, info in sorted(rows):
        child_path = os.path.join(directory, name)
        if info.st_dev != device or child_path in mount_points:
            continue
        child_key = f"{key}/{_digest(name)}" if key else _digest(name)
        child_names = (*names, name)
        if names and not stat.S_ISDIR(info.st_mode):
            continue  # a split directory's files belong to its shallow partition
        split = plan.splits.get(child_key)
        if (split is not None and split == _identity(info) and stat.S_ISDIR(info.st_mode)
                and len(child_names) < MAX_SPLIT_DEPTH):
            seen_splits.add(child_key)
            shallow_key = f"{child_key}/{SHALLOW}"
            plan.candidates[shallow_key] = Candidate(shallow_key, child_path, info, child_names, True)
            try:
                with descendant_directory(anchor, (name,), info) as child_anchor:
                    _inventory(plan, child_anchor, child_path, child_names, child_key, device=device,
                               mount_points=mount_points, deadline=deadline, clock=clock,
                               seen_splits=seen_splits, counter=counter, reasons=reasons)
            except UsagePathError:
                plan.complete = False
                reasons.add("partition_identity_refused")
            except OSError:
                plan.complete = False
                reasons.add("partition_inventory_unreadable")
            continue
        # A symlink is a leaf: its own inode is measured, its target never followed.
        plan.candidates[child_key] = Candidate(child_key, child_path, info, child_names)


def plan_partitions(
    roots: Sequence[tuple[str, os.stat_result]], *, continuation: Mapping[str, Any] | None,
    partitions_per_mount: int, mount_points: frozenset[str], deadline: float,
    clock: Callable[[], float] = time.time,
) -> PartitionPlan:
    """Inventory every root's current partitions and order this pass's work.

    ``order`` lists every candidate, alternating between roots, each root's
    unvisited partitions first, at most ``partitions_per_mount`` from each;
    the caller walks it until its budget runs out.
    """

    if type(partitions_per_mount) is not int or not 1 <= partitions_per_mount <= MAX_PARTITIONS_PER_MOUNT:
        raise ValueError("usage_partition_limit_invalid")
    old = dict(continuation) if isinstance(continuation, Mapping) else {}
    old_valid = (old.get("schema_version") in _COMPATIBLE_SCHEMAS
                 and old.get("cursor_digest") == canonical_digest(old, digest_field="cursor_digest"))
    reasons: set[str] = set()
    if continuation is not None and not old_valid:
        reasons.add("invalid_cursor_refused")
    v3 = old_valid and old.get("schema_version") == SCHEMA_VERSION
    previous_roots = old.get("roots") if v3 and isinstance(old.get("roots"), Mapping) else {}
    churn = {"departed": 0, "replaced": 0}
    plans: list[RootPlan] = []
    started = clock()
    share = max(0.0, deadline - started) / max(1, len(roots))
    counter = [0]
    for path, root_info in roots:
        identity = {"device": root_info.st_dev, "inode": root_info.st_ino,
                    "mode": stat.S_IFMT(root_info.st_mode)}
        plan = RootPlan(hashlib.sha256(os.fsencode(path)).hexdigest(), path, identity)
        plans.append(plan)
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            reasons.add("root_not_regular_directory")
            plan.complete = False
            continue
        previous = previous_roots.get(plan.key)
        # Only this root's own identity decides whether its position still applies.
        if isinstance(previous, Mapping) and all(previous.get(k) == v for k, v in identity.items()):
            plan.splits = {k: v for k, v in (previous.get("splits") or {}).items()
                           if isinstance(v, list) and len(v) == 3}
            plan.ledger = dict(previous.get("ledger") or {})
            prior_queue = [k for k in previous.get("pending") or [] if isinstance(k, str)]
        else:
            if isinstance(previous, Mapping):
                reasons.add("stale_cursor_identity_refused")  # the root itself was replaced
            prior_queue = []
        seen_splits: set[str] = set()
        try:
            with anchored_directory(path, root_info) as anchor:
                _inventory(plan, anchor, path, (), "", device=root_info.st_dev, mount_points=mount_points,
                           deadline=min(deadline, clock() + share), clock=clock, seen_splits=seen_splits,
                           counter=counter, reasons=reasons)
        except UsagePathError:
            plan.candidates = {}
            plan.complete = False
            reasons.add("partition_identity_refused")
        except OSError:
            plan.complete = False
            reasons.add("partition_inventory_unreadable")
        if plan.complete:
            plan.splits = {k: v for k, v in plan.splits.items() if k in seen_splits}
            if not plan.candidates:
                # An empty root is measured as itself, never as a false empty claim.
                plan.candidates[SHALLOW] = Candidate(SHALLOW, path, root_info, (), True)
        plan.inventory_digest = canonical_digest({"entries": sorted(
            [key, *_identity(candidate.info)] for key, candidate in plan.candidates.items())})
        queued = set()
        for key in prior_queue:
            if key in queued:
                continue
            if key in plan.candidates or not plan.complete:
                # An incomplete inventory cannot prove an entry disappeared: it keeps
                # its place, but is never selected from an old observation.
                plan.queue.append(key)
                queued.add(key)
            elif key not in plan.splits:  # a split partition was replaced by its parts
                churn["departed"] += 1
        for key, measurement in plan.ledger.items():
            candidate = plan.candidates.get(key)
            if candidate is not None and isinstance(measurement, Mapping) and \
                    measurement.get("identity") != _identity(candidate.info):
                churn["replaced"] += 1
        # New partitions join the tail; existing unvisited ones keep their place.
        plan.queue.extend(candidate.key for candidate in sorted(
            (c for k, c in plan.candidates.items() if k not in queued), key=_sort_key))
    order: list[tuple[RootPlan, Candidate]] = []
    selectable = [[plan.candidates[key] for key in plan.queue if key in plan.candidates][:partitions_per_mount]
                  for plan in plans]
    for depth in range(max((len(rows) for rows in selectable), default=0)):
        for plan, rows in zip(plans, selectable):
            if depth < len(rows):
                order.append((plan, rows[depth]))
    mount_digest = hashlib.sha256("\n".join(sorted(mount_points)).encode()).hexdigest()
    return PartitionPlan(plans, order, reasons, churn, counter[0], old_valid, mount_digest)


def measurement_usable(measurement: Any, candidate: Candidate) -> bool:
    return isinstance(measurement, Mapping) and measurement.get("identity") == _identity(candidate.info)


def finish_partitions(
    plan: PartitionPlan, *, measurements: Mapping[tuple[str, str], Mapping[str, Any]],
    identity_refused: bool,
) -> dict[str, Any]:
    """The next cursor and this pass's partition metadata from what was walked.

    ``measurements`` maps a walked ``(root key, candidate key)`` to its
    measurement (with ``complete``). A truncated whole-subtree directory is split for the next
    pass; visited partitions rotate behind the unvisited ones.
    """

    cursor: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "mount_snapshot_digest": plan.mount_digest,
                              "roots": {}, "cursor_digest": ""}
    reasons = set(plan.reasons)
    inventory_complete = all(root.complete for root in plan.roots) and not identity_refused
    for root in plan.roots:
        ledger = {key: value for key, value in root.ledger.items()
                  if key in root.candidates or not root.complete}
        splits = dict(root.splits)
        visited = []
        for key in root.queue:
            measurement = measurements.get((root.key, key))
            if measurement is None:
                continue
            visited.append(key)
            candidate = root.candidates[key]
            ledger[key] = dict(measurement)
            if (not measurement.get("complete") and not candidate.shallow
                    and stat.S_ISDIR(candidate.info.st_mode) and len(candidate.names) < MAX_SPLIT_DEPTH):
                if len(splits) < MAX_SPLITS_PER_ROOT:
                    splits[key] = _identity(candidate.info)
                    reasons.add("partition_split")
                else:
                    reasons.add("partition_split_limit_reached")
        visited_set = set(visited)
        cursor["roots"][root.key] = {
            **root.identity, "inventory_complete": root.complete, "inventory_digest": root.inventory_digest,
            "pending": [key for key in root.queue if key not in visited_set] + visited,
            "splits": dict(sorted(splits.items())), "ledger": dict(sorted(ledger.items())),
        }
    cursor["cursor_digest"] = canonical_digest(cursor, digest_field="cursor_digest")
    if identity_refused:
        reasons.add("partition_identity_refused")
    status = sorted(reasons - {"partition_split"}) or ["continued" if plan.old_valid else "started"]
    if "partition_split" in reasons:
        status = sorted({*status, "partition_split"})
    return {"inventory_complete": inventory_complete,
            "candidate_partitions": sum(len(root.candidates) for root in plan.roots),
            "selected_partitions": len(measurements),
            "partition_entries_examined": plan.examined,
            "partition_churn": dict(plan.churn),
            "continuation_status": status,
            "continuation": cursor, "mutations": 0, "retirement_authority": False}


__all__ = [
    "COMPOSITE_MAX_AGE_SECONDS",
    "Candidate",
    "MAX_PARTITIONS_PER_MOUNT",
    "PartitionPlan",
    "RootPlan",
    "SCHEMA_VERSION",
    "finish_partitions",
    "measurement_usable",
    "plan_partitions",
]
