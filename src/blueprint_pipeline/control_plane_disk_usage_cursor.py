"""Bounded root-partition continuation for the existing usage survey.

Every result remains one partial snapshot. Bytes from successive passes must
not be added: hardlinks and changing files can occur across partitions. The
cursor selects observation work only and conveys no deletion authority.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_usage_partition_cursor.v1"
MAX_PARTITION_CANDIDATES = 20_000
_PRIORITY_NAMES = frozenset(("pubsub-handoffs", "task-evaluation-inputs", "pipeline-control-plane",
                             "production-gpu-artifacts", "lanes", "capture-reconstruction-runs",
                             "capture-reconstruction-derived", "beta-policy-screening", "team-scene-packets",
                             "rigid-robot-free-probes", "workspace"))


def select_partitions(
    roots: Sequence[tuple[str, os.stat_result]], *, continuation: Mapping[str, Any] | None,
    partitions_per_mount: int, mount_points: frozenset[str], deadline: float,
    clock=time.time,
) -> tuple[list[tuple[str, os.stat_result]], dict[str, Any]]:
    if type(partitions_per_mount) is not int or not 1 <= partitions_per_mount <= 32:
        raise ValueError("usage_partition_limit_invalid")
    old = dict(continuation) if isinstance(continuation, Mapping) else {}
    old_valid = old.get("schema_version") == SCHEMA_VERSION and old.get("cursor_digest") == canonical_digest(old, digest_field="cursor_digest")
    if not old_valid:
        old = {}
    mount_digest = hashlib.sha256("\n".join(sorted(mount_points)).encode()).hexdigest()
    previous_roots = old.get("roots") if isinstance(old.get("roots"), dict) else {}
    cursor: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "mount_snapshot_digest": mount_digest,
                              "roots": {}, "cursor_digest": ""}
    selected = []
    candidate_count = 0
    inventory_complete = True
    reasons = {"invalid_cursor_refused"} if continuation is not None and not old_valid else set()
    started = clock()
    inventory_share = max(0.0, deadline - started) / max(1, len(roots))
    examined = 0
    for path, root_info in roots:
        root_key = hashlib.sha256(os.fsencode(path)).hexdigest()
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            reasons.add("root_not_regular_directory")
            inventory_complete = False
            continue
        candidates = []
        complete = True
        root_deadline = min(deadline, clock() + inventory_share)
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if len(candidates) >= MAX_PARTITION_CANDIDATES or clock() >= root_deadline:
                        complete = False
                        break
                    try:
                        info = entry.stat(follow_symlinks=False)
                        examined += 1
                    except OSError:
                        complete = False
                        continue
                    if info.st_dev != root_info.st_dev or entry.path in mount_points:
                        continue
                    # Symlink bytes may be counted by a full survey, but a
                    # cursor must never turn a link into a traversable root.
                    if stat.S_ISLNK(info.st_mode):
                        complete = False
                        continue
                    candidates.append((entry.name, entry.path, info))
        except OSError:
            complete = False
            reasons.add("partition_inventory_unreadable")
        candidates.sort(key=lambda row: (row[0] not in _PRIORITY_NAMES, row[0]))
        identity = {"device": root_info.st_dev, "inode": root_info.st_ino,
                    "mode": stat.S_IFMT(root_info.st_mode), "inventory_complete": complete,
                    "inventory_digest": canonical_digest({"entries": [
                        [name, info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)]
                        for name, _path, info in candidates]})}
        previous = previous_roots.get(root_key)
        matches = (isinstance(previous, dict) and old.get("mount_snapshot_digest") == mount_digest
                   and all(previous.get(key) == value for key, value in identity.items()))
        if previous and not matches:
            reasons.add("stale_cursor_identity_refused")
        start = previous.get("next_index", 0) if matches else 0
        if type(start) is not int or not 0 <= start < max(1, len(candidates)):
            start = 0
            reasons.add("cursor_position_invalid")
        count = min(partitions_per_mount, len(candidates))
        chosen = [candidates[(start + offset) % len(candidates)] for offset in range(count)] if candidates else []
        selected.extend((child_path, info) for _name, child_path, info in chosen)
        # Empty directories still have measurable metadata, but never recurse
        # into an unreadable or incomplete inventory under a false empty claim.
        if complete and not candidates:
            selected.append((path, root_info))
        cursor["roots"][root_key] = {**identity, "next_index": (start + count) % len(candidates) if candidates else 0}
        candidate_count += len(candidates)
        inventory_complete &= complete
    cursor["cursor_digest"] = canonical_digest(cursor, digest_field="cursor_digest")
    return selected, {"scope": "bounded_partition_snapshot", "inventory_complete": inventory_complete,
                      "candidate_partitions": candidate_count, "selected_partitions": len(selected),
                      "partition_entries_examined": examined,
                      "continuation_status": sorted(reasons) or ["continued" if old_valid else "started"],
                      "continuation": cursor, "mutations": 0, "retirement_authority": False}
