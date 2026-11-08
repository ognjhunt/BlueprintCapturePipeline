"""Remove the sealed bundle of a screening attempt that provably reached no GPU provider.

Owner decision, 2026-11-08 about 18:50Z, relayed by the library-aisle session: "yes prune".
The question put to them: delete the bundle only from attempts that provably never reached a
GPU provider (refused before spawn, no qualifying offer, a preflight refusal); keep every
receipt, the bundle's sha256 included; attempts that ran keep their bundle as evidence. It
amends the "never evicted" note on ``beta-policy-screening-work`` for those attempts only.

Why: every attempt seals its own bundle (about 200 MB for G1, and a copy of the customer's
scene packet). A retry the market or a preflight refused left one more copy behind each time,
and on 2026-11-08 the work volume's headroom fell 0.35 GB short of a G1 run's collection
reservation.

An attempt's bundle is removed only when all of these hold:

- the ledger records it closed: its reservation ``released`` and ``provider_allocation_observed``
  false, and it is not the current attempt of an active row;
- its own records prove nothing was allocated: its launch intent was ``refused_before_spawn``,
  or its allocator result proves no provider allocation (``proves_no_provider_allocation``);
- its directory is the coordinator's own (``<root>/attempts/screen-<32 hex>``), with no symlink
  on the path, and the bundle directories hold only regular files and directories.

Only ``bundle/`` and ``bundle.interrupted-*`` go. ``bundle_receipt.json`` (with the bundle's
sha256 and size), ``prepared.json``, the launch intent, the allocator records and every other
receipt stay. A receipt, ``bundle_pruned.v1.json``, is written beside them before anything is
removed and names what goes. A spent attempt (committed reservation) is never touched. Withdrawal
still deletes the whole attempt directory; a prune between its plan and apply makes it refuse
``screening_attempt_changed`` and retry, never delete less. Runs outside the ledger lock; the
attempts it touches are terminal, so no coordinator step writes into them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: At most this many attempts are pruned per worker tick.
PER_TICK = 16
RECEIPT = "bundle_pruned.v1.json"
SCHEMA = "blueprint.beta_screening_unspent_bundle_prune.v1"
OWNER_DECISION = ('2026-11-08 about 18:50Z, relayed by the library-aisle session: "yes prune" '
                  "(bundles of attempts that provably reached no GPU provider; receipts kept)")
_ATTEMPT_ID = re.compile(r"screen-[0-9a-f]{32}")


def _read(path: Path) -> Mapping[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _proof(directory: Path) -> str | None:
    """How the attempt's own records prove it allocated nothing, or None."""
    from .task_evaluation_scene_execution_authority import proves_no_provider_allocation

    intent = _read(directory / "allocator_launch_intent.json")
    if intent is not None and intent.get("status") == "refused_before_spawn":
        return "refused_before_spawn"
    result = _read(directory / "allocator_result.json")
    if result is not None and proves_no_provider_allocation(result):
        return "allocator_proved_no_allocation"
    return None


def _directory(queue_root: Path, roots: set[Path], attempt: Mapping[str, Any]) -> Path | None:
    from . import beta_policy_screening_coordinator as coordinator

    if not _ATTEMPT_ID.fullmatch(str(attempt.get("attempt_id") or "")):
        return None
    try:
        directory = coordinator._attempt_path(queue_root, attempt)
    except (coordinator.ScreeningCoordinatorError, TypeError):
        return None
    if (directory.parent not in roots or directory.is_symlink() or not directory.is_dir()
            or any(path.is_symlink() for path in directory.parents)):
        return None
    return directory


def _inventory(target: Path) -> tuple[int, int] | None:
    """(files, bytes) under ``target``; None when anything is not a regular file or directory."""
    if target.is_symlink():
        return None
    if target.is_file():
        return 1, target.stat().st_size
    files = size = 0
    for directory, directories, names in os.walk(target, followlinks=False):
        for name in directories + names:
            path = Path(directory) / name
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                return None
            if path.is_file():
                files += 1
                size += path.stat().st_size
    return files, size


def _candidates(queue_root: Path) -> list[Mapping[str, Any]]:
    from . import beta_policy_screening_coordinator as coordinator
    from .beta_policy_screening_result import ACTIVE_STATUSES

    with coordinator._lock(queue_root):
        ledger = coordinator._ledger(queue_root)
    attempts = []
    for state in ledger.get("campaigns", {}).values():
        for row in state.get("candidates", {}).values():
            current = row.get("attempt")
            closed = [*row.get("attempt_history", []),
                      *([current] if current and row.get("status") not in ACTIVE_STATUSES else [])]
            attempts += [attempt for attempt in closed if isinstance(attempt, Mapping)
                         and attempt.get("reservation_status") == "released"
                         and attempt.get("provider_allocation_observed") is False]
    return attempts


def prune_unspent_bundles(*, queue_root: Path, per_tick: int = PER_TICK) -> dict[str, Any]:
    """Remove up to ``per_tick`` bundles of closed attempts that provably allocated nothing."""
    from . import beta_policy_screening_coordinator as coordinator

    root = Path(queue_root)
    work_root = coordinator._work_root()
    roots = {root / "attempts", *([work_root / "attempts"] if work_root else [])}
    pruned, refused = [], []
    for attempt in _candidates(root):
        if len(pruned) >= per_tick:
            break
        directory = _directory(root, roots, attempt)
        if directory is None:
            continue
        targets = [path for path in (directory / "bundle", *sorted(directory.glob("bundle.interrupted-*")))
                   if path.exists() or path.is_symlink()]
        if not targets:
            continue
        proof = _proof(directory)
        if proof is None:
            continue
        rows = []
        for target in targets:
            counted = _inventory(target)
            if counted is None:
                rows = None
                break
            rows.append({"path": target.name, "files": counted[0], "bytes": counted[1]})
        if rows is None:
            refused.append({"attempt_id": attempt["attempt_id"], "blocker": "beta_screening_bundle_prune_unsafe_entry"})
            continue
        bundle_receipt = directory / "bundle_receipt.json"
        receipt = {"schema_version": SCHEMA, "attempt_id": attempt["attempt_id"], "proof": proof,
                   "owner_decision": OWNER_DECISION, "removed": rows,
                   "removed_bytes": sum(row["bytes"] for row in rows),
                   "bundle_receipt_sha256": ("sha256:" + hashlib.sha256(bundle_receipt.read_bytes()).hexdigest()
                                             if bundle_receipt.is_file() and not bundle_receipt.is_symlink() else None),
                   "pruned_at": datetime.now(timezone.utc).isoformat()}
        temporary = directory / (RECEIPT + ".partial")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, directory / RECEIPT)
        for target in targets:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        pruned.append({"attempt_id": attempt["attempt_id"], "proof": proof, "removed_bytes": receipt["removed_bytes"]})
    return {"status": "processed", "pruned": pruned, "refused": refused,
            "removed_bytes": sum(row["removed_bytes"] for row in pruned)}


__all__ = ["OWNER_DECISION", "PER_TICK", "RECEIPT", "SCHEMA", "prune_unspent_bundles"]
