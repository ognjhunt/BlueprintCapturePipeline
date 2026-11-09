"""Remove the sealed bundle of a screening attempt that provably reached no GPU provider, or whose
spend is settled.

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

A spent attempt's bundle goes too (owner decision, 2026-11-09 20:50:12Z, relayed by the library-aisle
session: "go with option A for the disk cleanup"), when all of these hold:

- the ledger records it spent and settled: reservation ``committed``, ``provider_allocation_observed``
  true, and ``settled_cost_usd`` recorded, which ``beta_policy_screening_settlement`` writes only from
  the attempt's own completed teardown manifest with no continuing spend; an attempt held at its
  maximum (``settlement_held``) is never touched;
- it is closed (history, or the current attempt of a row that is no longer queued or running), and its
  campaign has no row queued, running or awaiting reconciliation, so nothing live can read it again;
- its bundle was sealed (``bundle_receipt.json``, kept) more than ``SPENT_MIN_AGE_SECONDS`` ago;
- the same directory and entry checks as above.

Why: on 2026-11-09 the work volume had 5.16 GB available against a G1 run's 8.8 GB collection
reservation, so the next G1 admission was refused at $0. Settled spent attempts held about 7.7 GB of
bundles older than two days (27 attempts, each a 202 MB zip and on older ones a 106 MB expanded base).

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
SPENT_OWNER_DECISION = ('2026-11-09 20:50:12Z, relayed by the library-aisle session: "go with option A for the '
                        'disk cleanup" (bundles of settled spent attempts older than two days; receipts and '
                        'outputs kept)')
#: A settled spent attempt keeps its bundle this long after sealing it: the last two days stay as evidence.
SPENT_MIN_AGE_SECONDS = 2 * 86400
#: At most this many spent attempts are pruned per worker tick (about 7.7 GB on the first, 2026-11-09).
SPENT_PER_TICK = 32
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


def _settled_spent(queue_root: Path) -> list[Mapping[str, Any]]:
    """Closed attempts whose spend is settled, in campaigns with nothing queued, running or reconciling."""
    from . import beta_policy_screening_coordinator as coordinator
    from .beta_policy_screening_result import ACTIVE_STATUSES

    with coordinator._lock(queue_root):
        ledger = coordinator._ledger(queue_root)
    attempts = []
    for state in ledger.get("campaigns", {}).values():
        rows = list(state.get("candidates", {}).values())
        if any(row.get("status") in ACTIVE_STATUSES for row in rows):
            continue
        for row in rows:
            current = row.get("attempt")
            closed = [*row.get("attempt_history", []), *([current] if current else [])]
            attempts += [attempt for attempt in closed if isinstance(attempt, Mapping)
                         and attempt.get("reservation_status") == "committed"
                         and attempt.get("provider_allocation_observed") is True
                         and attempt.get("settled_cost_usd") not in (None, "")
                         and coordinator.SETTLEMENT_HELD_FIELD not in attempt]
    return attempts


def _sealed_before(directory: Path, seconds: float) -> bool:
    receipt = directory / "bundle_receipt.json"
    try:
        return (not receipt.is_symlink() and receipt.is_file()
                and datetime.now(timezone.utc).timestamp() - receipt.stat().st_mtime > seconds)
    except OSError:
        return False


def _prune_one(directory: Path, attempt: Mapping[str, Any], *, proof: str, owner_decision: str,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Write the receipt, then remove the attempt's bundle; None when there is none, a refusal row if unsafe."""
    targets = [path for path in (directory / "bundle", *sorted(directory.glob("bundle.interrupted-*")))
               if path.exists() or path.is_symlink()]
    if not targets:
        return None
    rows = []
    for target in targets:
        counted = _inventory(target)
        if counted is None:
            return {"attempt_id": attempt["attempt_id"], "blocker": "beta_screening_bundle_prune_unsafe_entry"}
        rows.append({"path": target.name, "files": counted[0], "bytes": counted[1]})
    bundle_receipt = directory / "bundle_receipt.json"
    receipt = {"schema_version": SCHEMA, "attempt_id": attempt["attempt_id"], "proof": proof,
               "owner_decision": owner_decision, "removed": rows,
               "removed_bytes": sum(row["bytes"] for row in rows),
               "bundle_receipt_sha256": ("sha256:" + hashlib.sha256(bundle_receipt.read_bytes()).hexdigest()
                                         if bundle_receipt.is_file() and not bundle_receipt.is_symlink() else None),
               **dict(extra or {}), "pruned_at": datetime.now(timezone.utc).isoformat()}
    temporary = directory / (RECEIPT + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, directory / RECEIPT)
    for target in targets:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    return {"attempt_id": attempt["attempt_id"], "proof": proof, "removed_bytes": receipt["removed_bytes"]}


def prune_unspent_bundles(*, queue_root: Path, per_tick: int = PER_TICK,
                          spent_per_tick: int = SPENT_PER_TICK) -> dict[str, Any]:
    """Remove up to ``per_tick`` bundles of closed attempts that provably allocated nothing, and up to
    ``spent_per_tick`` of settled spent attempts sealed more than ``SPENT_MIN_AGE_SECONDS`` ago."""
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
        proof = _proof(directory)
        if proof is None:
            continue
        done = _prune_one(directory, attempt, proof=proof, owner_decision=OWNER_DECISION)
        if done is None:
            continue
        (refused if "blocker" in done else pruned).append(done)
    spent = []
    for attempt in _settled_spent(root):
        if len(spent) >= spent_per_tick:
            break
        directory = _directory(root, roots, attempt)
        if directory is None or not _sealed_before(directory, SPENT_MIN_AGE_SECONDS):
            continue
        done = _prune_one(directory, attempt, proof="settled_spend", owner_decision=SPENT_OWNER_DECISION,
                          extra={"settled_cost_usd": str(attempt["settled_cost_usd"])})
        if done is None:
            continue
        (refused if "blocker" in done else spent).append(done)
    return {"status": "processed", "pruned": pruned, "spent_pruned": spent, "refused": refused,
            "removed_bytes": sum(row["removed_bytes"] for row in [*pruned, *spent])}


__all__ = ["OWNER_DECISION", "PER_TICK", "RECEIPT", "SCHEMA", "SPENT_MIN_AGE_SECONDS", "SPENT_OWNER_DECISION",
           "SPENT_PER_TICK", "prune_unspent_bundles"]
