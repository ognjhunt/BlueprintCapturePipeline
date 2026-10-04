"""Optional FindAll dispatch through an existing owner's durable research ledger.

Consumes a supplied grant and a current-authority check. No credential lookup,
grant issuance, new store, default provider, or automatic dispatch is installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from typing import Any, Callable, ContextManager, Mapping, Protocol

from .paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)
from .parallel_findall import FindAllError, _validate_run_id
from .parallel_findall_execution import (
    PAID_RESOURCE_CLASS,
    AdmittedFindAllClient,
    FindAllSubmissionUnresolved,
    prepare_submission,
)

SUBMISSIONS_FIELD = "parallel_findall_submissions"


class OwnerResearchLedger(Protocol):
    """Existing Ledger/FirestoreLedger interface; rows must cover all history."""

    def lock(self) -> ContextManager[Any]: ...
    def rows(self) -> list[dict[str, Any]]: ...
    def get(self, day: str) -> dict[str, Any] | None: ...
    def put(self, row: dict[str, Any]) -> None: ...
    def write_bytes(self, name: str, value: bytes) -> None: ...
    def read_bytes(self, name: str) -> bytes: ...


def _operation_key(operation_id: str) -> str:
    return hashlib.sha256(operation_id.encode()).hexdigest()


def _slots(row: Mapping[str, Any]) -> dict[str, Any]:
    slots = row.get(SUBMISSIONS_FIELD, {})
    if not isinstance(slots, dict):
        raise FindAllError("findall_owner_journal_invalid")
    for key, entry in slots.items():
        if (not isinstance(entry, dict) or not isinstance(entry.get("operation_id"), str)
                or key != _operation_key(entry["operation_id"])):
            raise FindAllError("findall_owner_journal_invalid")
    return slots


class _OwnerJournal:
    def __init__(self, ledger, day, prepared, current_authority):
        self.ledger, self.day = ledger, day
        self.prepared = copy.deepcopy(prepared)
        self.current_authority = current_authority
        self.claimed = False
        self.known_id = None

    def _row(self):
        row = self.ledger.get(self.day)
        if (not isinstance(row, dict) or row.get("date") != self.day
                or row.get("run_key") != "blueprint-researcher:" + self.day):
            raise FindAllError("findall_existing_owner_record_required")
        return row

    def claim_submission(self, prepared):
        if prepared != self.prepared:
            raise FindAllError("findall_owner_request_binding_changed")
        row = self._row()
        key = _operation_key(prepared["operation_id"])
        history = self.ledger.rows()
        if not isinstance(history, list) or any(not isinstance(r, dict) for r in history):
            raise FindAllError("findall_owner_history_invalid")
        # The same operation cannot restart under a different daily record.
        if any(key in _slots(r) for r in [row, *history]):
            return False
        # Recheck current stop/scope/spend authority while the owner's lease is
        # held, after history reads. The check gets no API key and cannot change
        # the actual request or owner record by mutating these copies.
        if self.current_authority(copy.deepcopy(row), copy.deepcopy(prepared)) is not True:
            return False
        entry = {
            "schema_version": "blueprint.findall-owner-submission.v1",
            "operation_id": prepared["operation_id"],
            "allocation_binding_digest": prepared["allocation_binding_digest"],
            "prepared": copy.deepcopy(prepared),
            "state": "submission_unresolved",
            "findall_id": None,
        }
        row[SUBMISSIONS_FIELD] = {**_slots(row), key: entry}
        self.ledger.put(row)  # Existing durable commit BEFORE the one POST.
        self.claimed = True
        return True

    def record_created(self, operation_id, allocation_binding_digest, run):
        # Retain recovery facts before any store access: a post-response read
        # failure plus lease-release failure may replace the client's exception.
        if (not self.claimed or operation_id != self.prepared["operation_id"]
                or allocation_binding_digest != self.prepared["allocation_binding_digest"]):
            raise FindAllError("findall_owner_receipt_not_claimed")
        _validate_run_id(run.get("findall_id"))
        self.known_id = run["findall_id"]
        row = self._row()
        key = _operation_key(operation_id)
        entry = _slots(row).get(key)
        if (not isinstance(entry, dict)
                or entry.get("state") != "submission_unresolved"
                or entry.get("allocation_binding_digest") != allocation_binding_digest):
            raise FindAllError("findall_owner_receipt_not_claimed")
        entry.update(state="provider_id_recorded", findall_id=self.known_id)
        # Bind the recovery ID before serializing/writing a possibly large raw
        # receipt. Artifact failures must not make the operation replayable.
        self.ledger.put(row)
        raw = (json.dumps(run, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False) + "\n").encode()
        name = f"{self.day}-tool-findall-{key}.json"
        self.ledger.write_bytes(name, raw)
        if self.ledger.read_bytes(name) != raw:
            raise FindAllError("findall_owner_receipt_readback_failed")
        entry.update(state="receipt_retained", receipt_file=name,
                     receipt_sha256=hashlib.sha256(raw).hexdigest())
        self.ledger.put(row)


def create_with_owner_ledger(
    client: AdmittedFindAllClient,
    spec: Mapping[str, Any],
    *,
    ledger: OwnerResearchLedger,
    day: str,
    operation_id: str,
    maximum_cost_usd: str,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    current_authority: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """One explicitly authorized dispatch using existing owner state and lease.

    Call outside an already-held ledger lock. The owner supplies an existing
    daily record, a preconfigured client, an exact grant and its current action-
    time authority check. This function never creates a row or releases a claim.
    It holds the existing lock through claim, POST and raw-receipt persistence.
    Current authority must check scope, expiry, stop state, disclosure, pricing
    and remaining shared allowance; a historical reference alone is insufficient.
    """
    if not isinstance(client, AdmittedFindAllClient) or not callable(current_authority):
        raise FindAllError("findall_owner_controller_required")
    try:
        valid_day = isinstance(day, str) and date.fromisoformat(day).isoformat() == day
    except ValueError:
        valid_day = False
    if not valid_day:
        raise FindAllError("findall_owner_date_invalid")
    prepared = prepare_submission(spec, operation_id=operation_id, maximum_cost_usd=maximum_cost_usd)
    require_paid_resource_admission_grant(
        paid_resource_admission_grant,
        resource_class=PAID_RESOURCE_CLASS,
        allocation_binding_digest=prepared["allocation_binding_digest"],
        require_allocation_binding=True,
    )
    journal = _OwnerJournal(ledger, day, prepared, current_authority)
    try:
        with ledger.lock():
            return client.create(
                spec, operation_id=operation_id, maximum_cost_usd=maximum_cost_usd,
                paid_resource_admission_grant=paid_resource_admission_grant, journal=journal,
            )
    except (FindAllError, PaidResourceAdmissionBlocked):
        raise
    except Exception:
        # A lease-release/storage failure after the provider attempt must still
        # expose its known ID and retain the claim, never invite a replay.
        if journal.claimed:
            raise FindAllSubmissionUnresolved(findall_id=journal.known_id) from None
        raise FindAllError("findall_owner_ledger_failed") from None
