"""The beta policy screening copies of one withdrawn site, for withdrawal deletion.

Beta journey step: site withdrawal after the site's candidates were screened.
The screening coordinator keeps each attempt's records and provider run
directory in the attempt's own directory (``<work root>/attempts/<attempt id>``,
or ``<queue root>/attempts/<attempt id>`` for an attempt reserved without a
work root), and promotion copies each run's provider output archive (episode
MP4s and PNG frames of the site), and any paired witness, into the private
artifact store before settlement. The run roots the withdrawal deletion scans
cover neither.

``plan_screening`` binds attempts through the screening ledger, never by
scanning for names. A campaign is the site's when its ledger row or its plan
names the site's scene id, its plan names the site's walkthrough capture, or
its plan's scene intent is one the site's website source registration names.
Each of its attempts (the current one and its history) is then:

- deletable once its money is final: released, or committed and settled, held
  at its maximum, or never allocated. Its directory is inventoried, and every
  sealed promotion receipt (a superseded one too) and member index inside it
  names the archives to delete;
- deferred, with a typed reason, while its candidate is active, its
  reservation is open, or its settlement is still to be decided (settlement
  reads the attempt's directory). A later deletion run deletes it once it
  settles; the receipt names it until then.

The ledger is never written: its rows keep every attempt's spend and
settlement, which is what the site budget counts. Plans, planning inputs and
publications stay beside it. A ledger that cannot be read, an attempt
directory no ledger row names, an unsealed receipt or an invalid archive
reference fails closed into ``unresolved``. Records here carry identifiers,
paths and object keys only, never what a file holds.
"""
from __future__ import annotations

import fcntl
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import beta_policy_screening_coordinator as coordinator
from .beta_policy_screening_result import ACTIVE_STATUSES
from .decision_evidence_contracts import canonical_digest
from .provider_output_member_index import SCHEMA as INDEX_SCHEMA
from .provider_output_member_index import ProviderOutputMemberIndexError, durable_reference_facts
from .provider_output_promotion import INDEX_FILENAME
from .provider_output_promotion_records import RECEIPT_FILENAME, RECEIPT_SCHEMA

# The units' values (deploy/systemd/blueprint-beta-policy-screening.service), unless set here too.
DEFAULT_QUEUE_ROOT = Path(os.environ.get(coordinator.QUEUE_ROOT_ENV) or coordinator.DEFAULT_QUEUE_ROOT)
DEFAULT_WORK_ROOT = Path(os.environ.get(coordinator.WORK_ROOT_ENV)
                         or "/mnt/blueprint-work/beta-policy-screening-work")
_PREFIX = "website_withdrawal_deletion_"
ACTIVE = _PREFIX + "screening_attempt_active"
RESERVATION_OPEN = _PREFIX + "screening_reservation_open"
SETTLEMENT_PENDING = _PREFIX + "screening_settlement_pending"
_SUPERSEDED = RECEIPT_FILENAME + ".superseded-"


def _attempts(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [*row.get("attempt_history", []), *([row["attempt"]] if row.get("attempt") else [])]


def _money(status: str, attempt: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """(how its money ended, None), or (None, why it must wait)."""
    state = attempt.get("reservation_status")
    if status in ACTIVE_STATUSES:
        return None, ACTIVE
    if state == "reserved":
        return None, RESERVATION_OPEN
    if state == "released":
        return "released", None
    if "settled_cost_usd" in attempt:
        return "settled", None
    if coordinator.SETTLEMENT_HELD_FIELD in attempt:
        return "held_at_maximum", None
    if attempt.get("provider_allocation_observed") is not True:
        return "committed_at_maximum", None  # never settled: nothing reads its directory for money
    return None, SETTLEMENT_PENDING


def _plan_record(queue_root: Path, campaign_id: str, state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The campaign's plan when it is the one the ledger binds; else nothing (the ledger row decides)."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _json
    path = queue_root / "plans" / f"{campaign_id}.json"
    try:
        plan = _json(path) if path.exists() or path.is_symlink() else {}
    except WithdrawalDeletionError:
        return {}
    return plan if plan.get("plan_digest") == state.get("plan_digest") else {}


def _bound(identity: Mapping[str, str], intents: set[str], state: Mapping[str, Any],
           plan: Mapping[str, Any]) -> bool:
    source = plan.get("source") if isinstance(plan.get("source"), Mapping) else {}
    return (identity["scene_id"] in {state.get("site_id"), plan.get("site_id"), plan.get("scene_id")}
            or plan.get("capture_id") == identity["capture_id"] or source.get("scene_intent_id") in intents)


def _sealed(value: Mapping[str, Any], schema: str, field: str) -> bool:
    try:
        return value.get("schema_version") == schema and value.get(field) == canonical_digest(value, digest_field=field)
    except (TypeError, ValueError):
        return False


def _archive_uris(directory: Path, relative_paths: list[str]) -> list[str]:
    """Every durable archive a sealed promotion receipt or member index in the attempt names."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _json, _require
    uris = set()
    for relative in relative_paths:
        name = relative.rpartition("/")[2]
        if name != RECEIPT_FILENAME and not name.startswith(_SUPERSEDED) and name != INDEX_FILENAME:
            continue
        value = _json(directory / relative)
        try:
            if name == INDEX_FILENAME:
                _require(_sealed(value, INDEX_SCHEMA, "index_digest"), "screening_member_index_unsealed")
                references = [value["archive"].get("durable_reference")]
            else:
                _require(_sealed(value, RECEIPT_SCHEMA, "receipt_digest"), "screening_promotion_receipt_unsealed")
                references = [value.get("durable_reference"), *(
                    version.get("durable_reference") for section in (value.get("staged_objects") or {}).values()
                    for version in section.get("versions") or [])]
            uris.update(durable_reference_facts(reference)["uri"] for reference in references if reference is not None)
        except ProviderOutputMemberIndexError:
            raise WithdrawalDeletionError(_PREFIX + "screening_archive_reference_invalid") from None
        except (AttributeError, KeyError, TypeError):
            raise WithdrawalDeletionError(_PREFIX + "record_unreadable") from None
    return sorted(uris)


def _foreign_uris(directory: Path) -> set[str]:
    """Archives another site's attempt names, so a shared object is never this site's to delete.

    Best effort: a record that cannot be read here names nothing.
    """
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError
    if directory.is_symlink() or not directory.is_dir():
        return set()
    rows = [(Path(root) / name).relative_to(directory).as_posix()
            for root, _directories, files in os.walk(directory) for name in files]
    try:
        return set(_archive_uris(directory, rows))
    except (WithdrawalDeletionError, OSError):
        return set()


def _ledger(queue_root: Path, listed: list[Path], unresolved: list[dict[str, Any]]) -> dict[str, Any] | None:
    path = queue_root / coordinator.LEDGER_FILENAME
    if not (path.exists() or path.is_symlink()):
        # An attempt directory, or a pointer naming one on the work root, without the ledger that records it.
        attempts = queue_root / "attempts"
        if listed or (attempts.is_dir() and any(attempts.iterdir())):
            unresolved.append({"path": str(path), "reason": _PREFIX + "screening_ledger_missing"})
        return None
    try:
        return coordinator._ledger(queue_root)
    except (coordinator.ScreeningCoordinatorError, OSError, ValueError, KeyError, TypeError):
        unresolved.append({"path": str(path), "reason": _PREFIX + "screening_ledger_unreadable"})
        return None


def plan_screening(identity: Mapping[str, str], intents: list[str], *, queue_root: Path | None,
                   work_root: Path | None, unresolved: list[dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    """The site's screening attempts (deletable and deferred) and the archives other sites' attempts name."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _inventory
    section: dict[str, Any] = {"queue_root": str(queue_root) if queue_root else None,
                               "work_root": str(work_root) if work_root else None,
                               "campaigns": [], "attempts": [], "deferred": [], "retained_records": []}
    if queue_root is None or not Path(queue_root).is_dir():
        return section, set()
    queue_root = Path(queue_root)
    # Listed before the ledger is read: the coordinator records an attempt before it creates its
    # directory, so every directory listed here is named by the ledger read below if it is named at all.
    listed = [child for root in sorted({queue_root / "attempts", *([Path(work_root) / "attempts"] if work_root else [])})
              if root.is_dir() and not root.is_symlink() for child in sorted(root.iterdir())
              if child.is_symlink() or child.is_dir()]
    ledger = _ledger(queue_root, listed, unresolved)
    if ledger is None:
        return section, set()
    named, foreign = set(), set()
    for campaign_id, state in sorted(ledger["campaigns"].items()):
        bound = _bound(identity, set(intents), state, _plan_record(queue_root, campaign_id, state))
        if bound:
            section["campaigns"].append(campaign_id)
            section["retained_records"].extend(str(path) for path in (
                queue_root / "plans" / f"{campaign_id}.json", queue_root / "inputs" / campaign_id,
                queue_root / "publications" / f"{campaign_id}.json") if path.exists())
        for candidate_id, row in sorted(state["candidates"].items()):
            for attempt in _attempts(row):
                try:
                    directory = coordinator._attempt_path(queue_root, attempt)
                except (coordinator.ScreeningCoordinatorError, KeyError, TypeError):
                    unresolved.append({"campaign_id": campaign_id, "reason": _PREFIX + "screening_attempt_path_unsafe"})
                    continue
                named.add(directory)
                if not bound:
                    foreign |= _foreign_uris(directory)
                    continue
                ids = {"campaign_id": campaign_id, "candidate_id": candidate_id, "attempt_id": attempt["attempt_id"]}
                money, reason = _money(row["status"], attempt)
                if reason is not None:
                    section["deferred"].append({**ids, "reason": reason})
                    continue
                entry = {**ids, "money": money, "directory": None, "inventory_digest": None, "file_count": 0,
                         "object_uris": [], "staged_objects_absence_proven": money != "held_at_maximum"
                         and not (money == "released" and attempt.get("provider_allocation_observed") is True)}
                if directory.exists() or directory.is_symlink():
                    try:
                        if directory.is_symlink() or not directory.is_dir():
                            raise WithdrawalDeletionError(_PREFIX + "unsafe_entry")
                        inventory = _inventory(directory)
                        entry.update(directory=str(directory), file_count=len(inventory),
                                     inventory_digest=canonical_digest({"rows": inventory}),
                                     object_uris=_archive_uris(directory, [row[0] for row in inventory]))
                    except (WithdrawalDeletionError, OSError) as error:
                        reason = str(error) if isinstance(error, WithdrawalDeletionError) else _PREFIX + \
                            "record_unreadable"
                        unresolved.append({"path": str(directory), "reason": reason})
                        continue
                section["attempts"].append(entry)
    # Every attempt directory is the ledger's; one it does not name may be any site's.
    unresolved.extend({"path": str(child), "reason": _PREFIX + "screening_attempt_unbound"}
                      for child in listed if child not in named)
    return section, foreign


@contextmanager
def ledger_lock(queue_root: Path | None) -> Iterator[None]:
    """Hold the coordinator's ledger lock, so no attempt of the site changes state while it is deleted.

    The lock file is the coordinator's; it is never created here (an operator-owned
    file would lock the worker out). With no lock file nothing has ever reserved.
    """
    if queue_root is None:
        yield
        return
    try:
        descriptor = os.open(Path(queue_root) / ".ledger.lock", os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        yield
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def delete_attempt_directories(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Remove each reviewed attempt directory (its archives are already proven gone); never a ledger row."""
    from .website_capture_withdrawal_deletion import _inventory, _require
    removed = []
    for attempt in section["attempts"]:
        if attempt["directory"] is None:
            continue
        directory = Path(attempt["directory"])
        _require(canonical_digest({"rows": _inventory(directory)}) == attempt["inventory_digest"],
                 "screening_attempt_changed")
        shutil.rmtree(directory)
        removed.append(attempt)
    return removed


def receipt_section(section: Mapping[str, Any], removed: list[Mapping[str, Any]], store: Any,
                    objects: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and object keys of what this run removed; identifiers only."""
    versions = {row["uri"]: len(row["versions"]) for row in objects}
    attempts = [{key: attempt[key] for key in ("campaign_id", "candidate_id", "attempt_id", "money")}
                | {"directory": attempt["directory"], "files_deleted": attempt["file_count"] if attempt in removed else 0,
                   "archive_object_keys": [store.key(uri) for uri in attempt["object_uris"]]}
                for attempt in section["attempts"]]
    uris = sorted({uri for attempt in section["attempts"] for uri in attempt["object_uris"]})
    return {"campaigns": list(section["campaigns"]), "attempts": attempts,
            "attempt_directories_deleted": len(removed),
            "attempt_directories_already_absent": sum(attempt["directory"] is None for attempt in section["attempts"]),
            "attempt_files_deleted": sum(attempt["file_count"] for attempt in removed),
            "archive_objects_deleted": len(uris), "archive_object_versions_deleted": sum(versions[uri] for uri in uris),
            "archive_object_keys": [store.key(uri) for uri in uris],
            "deferred": [dict(row) for row in section["deferred"]], "ledger_rows_deleted": 0}


def remaining(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if section["queue_root"] is None:
        rows.append({"holder": "beta_screening_not_scanned",
                     "reason": "no screening queue root was named for this deletion"})
    if section["deferred"]:
        rows.append({"holder": "beta_screening_attempts_deferred", "attempts": [dict(row) for row in section["deferred"]],
                     "reason": "still active, or its money is not final; run the deletion again once it settles"})
    unproven = [attempt["attempt_id"] for attempt in section["attempts"] if not attempt["staged_objects_absence_proven"]]
    if unproven:
        rows.append({"holder": "beta_screening_staging_store", "attempt_ids": unproven,
                     "reason": "no settlement proved these attempts' staged provider objects absent; the staging "
                               "store may still hold them"})
    if section["campaigns"]:
        rows.append({"holder": "beta_screening_money_and_plan_records", "campaigns": list(section["campaigns"]),
                     "paths": [str(Path(section["queue_root"]) / coordinator.LEDGER_FILENAME),
                               *section["retained_records"]],
                     "reason": "the ledger keeps every attempt's spend and settlement so the site budget still "
                               "reconciles; plans, planning inputs and publications are JSON records the coordinator "
                               "reads, not capture media or renders"})
    return rows


__all__ = ["ACTIVE", "DEFAULT_QUEUE_ROOT", "DEFAULT_WORK_ROOT", "RESERVATION_OPEN", "SETTLEMENT_PENDING",
           "delete_attempt_directories", "ledger_lock", "plan_screening", "receipt_section", "remaining"]
