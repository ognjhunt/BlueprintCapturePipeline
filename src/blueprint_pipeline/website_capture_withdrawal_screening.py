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
scanning for names. A campaign is the site's when its ledger row or its sealed
plan names the site's scene id, its plan names the site's walkthrough capture,
or its plan's scene intent is one the site's website source registration
names. Each of its attempts (the current one and its history) is then:

- deletable once its money is final (released, or committed and settled, held
  at its maximum, or never allocated) and its staged provider objects are
  proven absent (by its settlement, or by each staging directory's absence
  proof). Its directory is inventoried, and every sealed promotion receipt (a
  superseded one too) and member index inside it names the output and witness
  archives to delete;
- deferred, with a typed reason, while its candidate is active, its
  reservation is open, its settlement is still to be decided (settlement reads
  the attempt's directory), or a staging directory has no absence proof (the
  directory is the only record of those staged keys). A later deletion run
  deletes it once that changes; the receipt names it until then.

``apply_screening`` deletes under the coordinator's ledger lock, held only for
these deletes: it resolves the site's attempts again and refuses unless they
are exactly the reviewed ones, then takes each staging directory's promotion
lock without waiting, deletes the attempt's archive versions, proves them
absent, and removes the directory.

The ledger is never written: its rows keep every attempt's spend and
settlement, which is what the site budget counts. Plans, planning inputs,
publications, selections, holds and attempt pointers stay beside it. A
ledger that cannot be read, an attempt directory no ledger row names, an
attempt id or directory outside the coordinator's layout, an attempts root
that is not there (an unmounted work volume), an unsealed or unreadable
record (another site's too), a campaign naming the site but booked to
another, or an archive reference outside the promotion layout fails closed
into ``unresolved``. Records here carry identifiers, paths and object keys
only, never what a file holds.
"""
from __future__ import annotations

import fcntl
import os
import re
import shutil
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from . import beta_policy_screening_coordinator as coordinator
from .beta_policy_screening_result import ACTIVE_STATUSES
from .decision_evidence_contracts import canonical_digest
from .provider_output_member_index import SCHEMA as INDEX_SCHEMA
from .provider_output_member_index import ProviderOutputMemberIndexError, durable_reference_facts
from .provider_output_promotion import INDEX_FILENAME, LOCK_FILENAME, OUTPUT_ARTIFACT_KIND, WITNESS_ARTIFACT_KIND
from .provider_output_promotion_records import (
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA,
    STAGING_MANIFEST_FILENAME,
    ProviderOutputPromotionRecordError,
    load_staged_object_absence_proof,
    staged_object_keys,
)
from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX

# The units' values (deploy/systemd/blueprint-beta-policy-screening.service), unless set here too.
DEFAULT_QUEUE_ROOT = Path(os.environ.get(coordinator.QUEUE_ROOT_ENV) or coordinator.DEFAULT_QUEUE_ROOT)
DEFAULT_WORK_ROOT = Path(os.environ.get(coordinator.WORK_ROOT_ENV)
                         or "/mnt/blueprint-work/beta-policy-screening-work")
_PREFIX = "website_withdrawal_deletion_"
ACTIVE = _PREFIX + "screening_attempt_active"
RESERVATION_OPEN = _PREFIX + "screening_reservation_open"
SETTLEMENT_PENDING = _PREFIX + "screening_settlement_pending"
STAGED_UNPROVEN = _PREFIX + "screening_staged_objects_unproven"
_SUPERSEDED = RECEIPT_FILENAME + ".superseded-"
_RECORDS = (RECEIPT_FILENAME, INDEX_FILENAME)
_ARCHIVE_KINDS = (OUTPUT_ARTIFACT_KIND, WITNESS_ARTIFACT_KIND)
# ``_new_attempt``'s ids: anything else is not a coordinator attempt and is never a path here.
_ATTEMPT_ID = re.compile(r"screen-[0-9a-f]{32}")


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


def _sealed(value: Mapping[str, Any], field: str, schema: str | None = None) -> bool:
    try:
        return ((schema is None or value.get("schema_version") == schema)
                and value.get(field) == canonical_digest(value, digest_field=field))
    except (TypeError, ValueError):
        return False


def _plan_record(queue_root: Path, campaign_id: str, state: Mapping[str, Any]) -> Mapping[str, Any]:
    """The campaign's sealed plan when it is the one the ledger binds; else nothing (the ledger row decides)."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _json
    path = queue_root / "plans" / f"{campaign_id}.json"
    try:
        plan = _json(path) if path.exists() or path.is_symlink() else {}
    except WithdrawalDeletionError:
        return {}
    return plan if _sealed(plan, "plan_digest") and plan.get("plan_digest") == state.get("plan_digest") else {}


def _binding(identity: Mapping[str, str], intents: set[str], state: Mapping[str, Any],
             plan: Mapping[str, Any]) -> tuple[bool, bool]:
    """(the campaign is the site's, its ledger books it to another site)."""
    source = plan.get("source") if isinstance(plan.get("source"), Mapping) else {}
    bound = (identity["scene_id"] in {state.get("site_id"), plan.get("site_id"), plan.get("scene_id")}
             or plan.get("capture_id") == identity["capture_id"] or source.get("scene_intent_id") in intents)
    return bound, bound and state.get("site_id") != identity["scene_id"]


def _record_names(path: Path, name: str) -> tuple[set[str], bool]:
    """The archives one sealed promotion receipt or member index names, and whether its witness step is pending."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _json, _require
    value = _json(path)
    try:
        if name == INDEX_FILENAME:
            _require(_sealed(value, "index_digest", INDEX_SCHEMA), "screening_member_index_unsealed")
            references, pending = [value["archive"].get("durable_reference")], False
        else:
            _require(_sealed(value, "receipt_digest", RECEIPT_SCHEMA), "screening_promotion_receipt_unsealed")
            sections = value.get("staged_objects") or {}
            references = [value.get("durable_reference"), *(version.get("durable_reference")
                          for section in sections.values() for version in section.get("versions") or [])]
            pending = (sections.get("paired_witness") or {}).get("state") == "pending"
        uris = set()
        for reference in references:
            if reference is None:
                continue
            facts = durable_reference_facts(reference)
            kind, key = facts["artifact_kind"], facts["uri"].removeprefix("s3://").partition("/")[2]
            # Only promotion's own layout: an output or witness under its content-addressed prefix.
            _require(kind in _ARCHIVE_KINDS and key.startswith(
                f"{LARGE_ARTIFACT_KEY_PREFIX}/{kind}/sha256/{facts['digest'].removeprefix('sha256:')}/"),
                "screening_archive_reference_invalid")
            uris.add(facts["uri"])
        return uris, pending
    except ProviderOutputMemberIndexError:
        raise WithdrawalDeletionError(_PREFIX + "screening_archive_reference_invalid") from None
    except (AttributeError, KeyError, TypeError):
        raise WithdrawalDeletionError(_PREFIX + "record_unreadable") from None


def _is_record(name: str) -> bool:
    return name in _RECORDS or name.startswith(_SUPERSEDED)


def _named_by(directory: Path, unresolved: list[dict[str, Any]]) -> set[str]:
    """Archives an attempt that is not deleted now names: each record on its own; one unreadable is unresolved."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError
    if not (directory.exists() or directory.is_symlink()):
        return set()
    if directory.is_symlink() or not directory.is_dir():
        unresolved.append({"path": str(directory), "reason": _PREFIX + "unsafe_entry"})
        return set()
    uris = set()
    for root, _directories, files in os.walk(directory):
        for name in filter(_is_record, files):
            try:
                uris |= _record_names(Path(root) / name, name)[0]
            except (WithdrawalDeletionError, OSError) as error:
                reason = str(error) if isinstance(error, WithdrawalDeletionError) else _PREFIX + "record_unreadable"
                unresolved.append({"path": str(Path(root) / name), "reason": reason})
    return uris


def _staged_unproven(directory: Path, relative_paths: list[str]) -> list[dict[str, Any]]:
    """Each staging directory without a valid absence proof, with the staged keys only its manifest records."""
    from .website_capture_withdrawal_deletion import _json
    rows = []
    for relative in relative_paths:
        if relative.rpartition("/")[2] != STAGING_MANIFEST_FILENAME:
            continue
        staging = directory / relative.rpartition("/")[0]
        try:
            proven = load_staged_object_absence_proof(staging) is not None
        except ProviderOutputPromotionRecordError:
            proven = False
        if not proven:
            keys = staged_object_keys(_json(staging / STAGING_MANIFEST_FILENAME))
            rows.append({"staging_dir": str(staging), "keys": [{"role": role, "key": key} for role, key in keys]})
    return rows


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


def _attempt_directory(queue_root: Path, roots: set[Path], attempt: Mapping[str, Any]) -> Path | None:
    """The attempt's directory when its id and place are the coordinator's own, else None."""
    if not isinstance(attempt, Mapping) or not _ATTEMPT_ID.fullmatch(str(attempt.get("attempt_id") or "")):
        return None
    try:
        directory = coordinator._attempt_path(queue_root, attempt)
    except (coordinator.ScreeningCoordinatorError, TypeError):
        return None
    parent = directory.parent
    if parent not in roots or any(path.is_symlink() for path in (parent, *parent.parents)):
        return None
    return directory


def _retained(queue_root: Path, campaign_id: str, plan: Mapping[str, Any], attempt_ids: list[str]) -> list[str]:
    from .beta_policy_screening_holds import HOLDS_DIRNAME, hold_id
    from .website_candidate_selection import SELECTION_DIRECTORY
    intent = (plan.get("source") or {}).get("scene_intent_id") if isinstance(plan.get("source"), Mapping) else None
    paths = [queue_root / "plans" / f"{campaign_id}.json", queue_root / "inputs" / campaign_id,
             queue_root / "publications" / f"{campaign_id}.json",
             queue_root / SELECTION_DIRECTORY / f"{campaign_id}.json",
             *([queue_root / HOLDS_DIRNAME / f"{hold_id(str(intent))}.json"] if intent else []),
             *(queue_root / "attempts" / f"{attempt_id}.attempt_dir.json" for attempt_id in attempt_ids)]
    return [str(path) for path in paths if path.exists()]


def plan_screening(identity: Mapping[str, str], intents: list[str], *, queue_root: Path | None,
                   work_root: Path | None, unresolved: list[dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    """The site's screening attempts (deletable and deferred), and the archives no deletion here may touch.

    The second value is every archive another site's attempt, or one of this
    site's deferred attempts, names.
    """
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError, _inventory
    section: dict[str, Any] = {"queue_root": str(queue_root) if queue_root else None,
                               "work_root": str(work_root) if work_root else None,
                               "queue_root_present": bool(queue_root) and Path(queue_root).is_dir(),
                               "campaigns": [], "attempts": [], "deferred": [], "retained_records": []}
    if not section["queue_root_present"]:
        return section, set()
    queue_root = Path(queue_root)
    roots = {queue_root / "attempts", *([Path(work_root) / "attempts"] if work_root else [])}
    # Listed before the ledger is read: the coordinator records an attempt before it creates its
    # directory, so every directory listed here is named by the ledger read below if it is named at all.
    listed = [child for root in sorted(roots) if root.is_dir() and not root.is_symlink()
              for child in sorted(root.iterdir()) if child.is_symlink() or child.is_dir()]
    ledger = _ledger(queue_root, listed, unresolved)
    if ledger is None:
        return section, set()
    named, protected, unavailable = set(), set(), set()
    for campaign_id, state in sorted(ledger["campaigns"].items()):
        plan = _plan_record(queue_root, campaign_id, state)
        bound, booked_elsewhere = _binding(identity, set(intents), state, plan)
        if booked_elsewhere:
            unresolved.append({"campaign_id": campaign_id, "reason": _PREFIX + "screening_campaign_names_another_site"})
        if bound:
            section["campaigns"].append(campaign_id)
            section["retained_records"].extend(_retained(queue_root, campaign_id, plan, [
                str(attempt.get("attempt_id")) for row in state["candidates"].values() for attempt in _attempts(row)
                if _ATTEMPT_ID.fullmatch(str(attempt.get("attempt_id") or ""))]))
        for candidate_id, row in sorted(state["candidates"].items()):
            for attempt in _attempts(row):
                directory = _attempt_directory(queue_root, roots, attempt)
                if directory is None:
                    unresolved.append({"campaign_id": campaign_id, "candidate_id": candidate_id,
                                       "reason": _PREFIX + "screening_attempt_path_unsafe"})
                    continue
                named.add(directory)
                if not (directory.parent.is_dir() and not directory.parent.is_symlink()):
                    # An unmounted work volume: nothing under it can be read or proven gone.
                    unavailable.add(directory.parent)
                    continue
                if not bound:
                    protected |= _named_by(directory, unresolved)
                    continue
                ids = {"campaign_id": campaign_id, "candidate_id": candidate_id, "attempt_id": attempt["attempt_id"]}
                money, reason = _money(row["status"], attempt)
                if reason is not None:
                    section["deferred"].append({**ids, "reason": reason})
                    protected |= _named_by(directory, unresolved)
                    continue
                entry = {**ids, "money": money, "allocated": attempt.get("provider_allocation_observed") is True,
                         "directory": None, "inventory_digest": None, "file_count": 0, "object_uris": [],
                         "staging_locks": [], "witness_pending": False}
                if directory.exists() or directory.is_symlink():
                    try:
                        if directory.is_symlink() or not directory.is_dir():
                            raise WithdrawalDeletionError(_PREFIX + "unsafe_entry")
                        inventory = _inventory(directory)
                        paths = [row[0] for row in inventory]
                        # A settlement proved every staged object absent; anything else needs each proof.
                        staged = [] if money == "settled" else _staged_unproven(directory, paths)
                        if staged:
                            section["deferred"].append({**ids, "reason": STAGED_UNPROVEN, "staged_objects": staged})
                            protected |= _named_by(directory, unresolved)
                            continue
                        uris, pending = set(), False
                        for relative in paths:
                            if _is_record(relative.rpartition("/")[2]):
                                names, waiting = _record_names(directory / relative, relative.rpartition("/")[2])
                                uris, pending = uris | names, pending or waiting
                        entry.update(directory=str(directory), file_count=len(inventory),
                                     inventory_digest=canonical_digest({"rows": inventory}), object_uris=sorted(uris),
                                     witness_pending=pending,
                                     staging_locks=[path for path in paths if path.rpartition("/")[2] == LOCK_FILENAME])
                    except (WithdrawalDeletionError, OSError) as error:
                        reason = str(error) if isinstance(error, WithdrawalDeletionError) else _PREFIX + \
                            "record_unreadable"
                        unresolved.append({"path": str(directory), "reason": reason})
                        continue
                section["attempts"].append(entry)
    unresolved.extend({"path": str(root), "reason": _PREFIX + "screening_attempt_root_unavailable"}
                      for root in sorted(unavailable))
    # Every attempt directory is the ledger's; one it does not name may be any site's.
    unresolved.extend({"path": str(child), "reason": _PREFIX + "screening_attempt_unbound"}
                      for child in listed if child not in named)
    return section, protected


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


def _promotion_locks(stack: ExitStack, directory: Path, relative_paths: list[str]) -> None:
    """Each staging directory's promotion lock, without waiting: a promoter at work refuses the deletion."""
    from .website_capture_withdrawal_deletion import WithdrawalDeletionError
    for relative in relative_paths:
        try:
            descriptor = os.open(directory / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            raise WithdrawalDeletionError(_PREFIX + "screening_attempt_changed") from None
        stack.callback(os.close, descriptor)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WithdrawalDeletionError(_PREFIX + "screening_promotion_busy") from None


def apply_screening(section: Mapping[str, Any], objects: list[Mapping[str, Any]], store: Any,
                    replan: Callable[[list[dict[str, Any]]], tuple[dict[str, Any], set[str]]]
                    ) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """Delete the reviewed attempts' archives, then their directories; never a ledger row.

    Under the ledger lock only: the site's attempts are resolved again and must
    be exactly the reviewed ones. Returns the removed attempts and the deleted
    object rows.
    """
    from .website_capture_withdrawal_deletion import _inventory, _require
    versions = {row["uri"]: row["versions"] for row in objects}
    removed, deleted, done = [], [], set()
    with ledger_lock(Path(section["queue_root"]) if section["queue_root"] else None):
        unresolved: list[dict[str, Any]] = []
        current, protected = replan(unresolved)
        uris = {uri for attempt in section["attempts"] for uri in attempt["object_uris"]}
        _require(not unresolved and current == section and not uris & protected, "screening_changed")
        for attempt in section["attempts"]:
            if attempt["directory"] is None:
                continue
            directory = Path(attempt["directory"])
            with ExitStack() as stack:
                _promotion_locks(stack, directory, attempt["staging_locks"])
                _require(canonical_digest({"rows": _inventory(directory)}) == attempt["inventory_digest"],
                         "screening_attempt_changed")
                for uri in attempt["object_uris"]:
                    if uri in done:
                        continue
                    for key, version in versions[uri]:
                        store.delete(key, version)
                    _require(store.versions(uri) == [], "object_versions_remaining")
                    deleted.append({"uri": uri, "versions_deleted": len(versions[uri])})
                    done.add(uri)
                # The archives are proven gone first, so a crash here leaves records naming only gone objects.
                shutil.rmtree(directory)
            removed.append(attempt)
    return removed, deleted


def receipt_section(section: Mapping[str, Any], removed: list[Mapping[str, Any]], store: Any,
                    objects: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts and object keys of what this run removed; identifiers only."""
    versions = {row["uri"]: row["versions_deleted"] for row in objects}
    removed_ids = {attempt["attempt_id"] for attempt in removed}
    attempts = [{key: attempt[key] for key in ("campaign_id", "candidate_id", "attempt_id", "money")}
                | {"directory": attempt["directory"],
                   "files_deleted": attempt["file_count"] if attempt["attempt_id"] in removed_ids else 0,
                   "archive_object_keys": [store.key(uri) for uri in attempt["object_uris"]]}
                for attempt in section["attempts"]]
    uris = sorted(versions)
    return {"campaigns": list(section["campaigns"]), "attempts": attempts,
            "attempt_directories_deleted": len(removed),
            "attempt_directories_already_absent": sum(attempt["directory"] is None for attempt in section["attempts"]),
            "attempt_files_deleted": sum(attempt["file_count"] for attempt in removed),
            "archive_objects_deleted": len(uris), "archive_object_versions_deleted": sum(versions.values()),
            "archive_object_keys": [store.key(uri) for uri in uris],
            "deferred": [dict(row) for row in section["deferred"]], "ledger_rows_deleted": 0}


def _absent_allocated(section: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [attempt for attempt in section["attempts"] if attempt["directory"] is None and attempt["allocated"]]


def incomplete(section: Mapping[str, Any]) -> bool:
    """Whether a screening copy may still exist after this run: deferred, an unrecorded witness, or unknown staging."""
    return bool(section["deferred"] or any(attempt["witness_pending"] for attempt in section["attempts"])
                or any(attempt["money"] != "settled" for attempt in _absent_allocated(section)))


def remaining(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if section["queue_root"] is None:
        rows.append({"holder": "beta_screening_not_scanned",
                     "reason": "no screening queue root was named for this deletion"})
    elif not section["queue_root_present"]:
        rows.append({"holder": "beta_screening_queue_root_absent", "path": section["queue_root"],
                     "reason": "the named screening queue root does not exist here, so no screening ledger was read"})
    for reason, holder, why in (
            (STAGED_UNPROVEN, "beta_screening_staging_store", "no settlement or absence proof shows these staged "
             "provider objects gone, and the attempt directory is the only record of their keys; run "
             "provider-output-resume first, then the deletion again"),
            (None, "beta_screening_attempts_deferred", "still active, or its money is not final; run the "
             "deletion again once it settles")):
        deferred = [dict(row) for row in section["deferred"] if (row["reason"] == STAGED_UNPROVEN) == bool(reason)]
        if deferred:
            rows.append({"holder": holder, "attempts": deferred, "reason": why})
    pending = [attempt["attempt_id"] for attempt in section["attempts"] if attempt["witness_pending"]]
    if pending:
        rows.append({"holder": "beta_screening_witness_unrecorded", "attempt_ids": pending,
                     "reason": "a promotion receipt's paired-witness step was still pending; a witness archive it "
                               "published may be in the artifact store without a record naming it"})
    absent = _absent_allocated(section)
    if absent:
        rows.append({"holder": "beta_screening_allocated_attempt_directory_absent",
                     "attempts": [{key: attempt[key] for key in ("attempt_id", "money")} for attempt in absent],
                     "reason": "the directory of an attempt that allocated a provider was already gone when this run "
                               "planned (an earlier deletion run, or another process, removed it); an archive or "
                               "staged copy only it named cannot be found from here"})
    if section["campaigns"]:
        rows.append({"holder": "beta_screening_money_and_plan_records", "campaigns": list(section["campaigns"]),
                     "paths": [str(Path(section["queue_root"]) / coordinator.LEDGER_FILENAME),
                               *section["retained_records"]],
                     "reason": "the ledger keeps every attempt's spend and settlement so the site budget still "
                               "reconciles; plans, planning inputs, publications, selections, holds and attempt "
                               "pointers are JSON records the coordinator reads, not capture media or renders"})
    return rows


__all__ = ["ACTIVE", "DEFAULT_QUEUE_ROOT", "DEFAULT_WORK_ROOT", "RESERVATION_OPEN", "SETTLEMENT_PENDING",
           "STAGED_UNPROVEN", "apply_screening", "incomplete", "ledger_lock", "plan_screening", "receipt_section",
           "remaining"]
