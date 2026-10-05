"""Optional FindAll dispatch through an existing owner's durable research ledger.

Consumes a supplied grant and a current-authority check. No credential lookup,
grant issuance, new store, default provider, or automatic dispatch is installed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from datetime import date
from typing import Any, Protocol

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

    def lock(self) -> AbstractContextManager[Any]: ...
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


# A snapshot is ONE immutable store file: one write, one readback check and its
# SHA-256 in the receipt. A snapshot too large for one tool output is paged at
# read time by slicing that same file, so continuation never fetches a new one.
SNAPSHOT_SCHEMA = "blueprint.findall-snapshot.v2"
SNAPSHOT_FRAGMENT_CHARS = 24_000  # Escaped output remains below the 500 kB tool ceiling.
SMALL_SNAPSHOT_BYTES = 200_000
# The Firestore bridge stores at most 8 MiB per file (firestore_bridge.mjs MAX_BYTES,
# checked in blobPut); 7 MiB of raw JSON keeps 1 MiB of headroom below it.
MAX_SNAPSHOT_BYTES = 7 * 1024 * 1024
_PAGING_FIELDS = ("schema_version", "page_chars", "page_count")


class FindAllSnapshotTooLarge(FindAllError):
    """A snapshot above MAX_SNAPSHOT_BYTES, refused before any store write."""

    def __init__(self, snapshot_bytes: int) -> None:
        self.snapshot_bytes = snapshot_bytes
        super().__init__("findall_snapshot_too_large")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _page_count(text):
    return -(-len(text) // SNAPSHOT_FRAGMENT_CHARS)


def encode_snapshot(snapshot):
    """The exact file bytes and receipt facts of one snapshot, with no store access.

    Above MAX_SNAPSHOT_BYTES this raises FindAllSnapshotTooLarge, so nothing is
    written. A snapshot too large for one tool output also fixes its page layout.
    """
    raw = _json_bytes(snapshot)
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise FindAllSnapshotTooLarge(len(raw))
    facts = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    # Size the inline path using the tool's ASCII-escaped serialization too.
    if (len(raw) > SMALL_SNAPSHOT_BYTES or len(json.dumps(
            snapshot, ensure_ascii=True, allow_nan=False).encode()) > SMALL_SNAPSHOT_BYTES):
        facts.update(schema_version=SNAPSHOT_SCHEMA, page_chars=SNAPSHOT_FRAGMENT_CHARS,
                     page_count=_page_count(raw.decode("utf-8")))
    return raw, facts


def store_snapshot(ledger, name, raw, facts):
    """Write the encoded snapshot once and check one readback; returns its receipt."""
    ledger.write_bytes(name, raw)
    if ledger.read_bytes(name) != raw:
        raise FindAllError("findall_snapshot_readback_failed")
    return {"file": name, **facts}


def retain_snapshot(ledger, name, snapshot):
    """Retain the whole snapshot as one immutable, read-back-checked file."""
    raw, facts = encode_snapshot(snapshot)
    return store_snapshot(ledger, name, raw, facts)


def _page_layout(receipt):
    """None for an inline snapshot, else its page count; other receipt shapes are refused."""
    if (not isinstance(receipt, Mapping) or "parts" in receipt  # The retired multi-file format.
            or not isinstance(receipt.get("file"), str) or not isinstance(receipt.get("sha256"), str)
            or type(receipt.get("bytes")) is not int):
        raise FindAllError("findall_snapshot_binding_invalid")
    present = [key for key in _PAGING_FIELDS if key in receipt]
    if not present:
        return None
    count = receipt.get("page_count")
    if (len(present) != len(_PAGING_FIELDS) or receipt["schema_version"] != SNAPSHOT_SCHEMA
            or receipt["page_chars"] != SNAPSHOT_FRAGMENT_CHARS or type(count) is not int or count < 1):
        raise FindAllError("findall_snapshot_binding_invalid")
    return count


def page_view(receipt, raw, page=0):
    """One page of these exact receipt-bound bytes, with no store access."""
    if type(page) is not int or page < 0:
        raise FindAllError("findall_snapshot_page_invalid")
    count = _page_layout(receipt)
    if (not isinstance(raw, bytes) or len(raw) != receipt["bytes"]
            or hashlib.sha256(raw).hexdigest() != receipt["sha256"]):
        raise FindAllError("findall_snapshot_binding_invalid")
    shown = {key: receipt[key] for key in ("file", "sha256", "bytes")}
    if count is None:
        if page != 0:
            raise FindAllError("findall_snapshot_page_invalid")
        return {"snapshot": json.loads(raw), "receipt": shown}
    text = raw.decode("utf-8")
    if _page_count(text) != count:
        raise FindAllError("findall_snapshot_binding_invalid")
    if page >= count:
        raise FindAllError("findall_snapshot_page_invalid")
    start = page * SNAPSHOT_FRAGMENT_CHARS
    return {"json_fragment": text[start:start + SNAPSHOT_FRAGMENT_CHARS], "encoding": "utf-8",
            "page": page, "page_count": count, "next_page": page + 1 if page + 1 < count else None,
            "snapshot_sha256": receipt["sha256"], "snapshot_bytes": receipt["bytes"],
            "receipt": shown}


def snapshot_page(receipt, read_bytes, page=0):
    """An immutable receipt-bound page; continuation never fetches a new snapshot.

    Each page is sliced from the one retained file (one store read), whose size
    and SHA-256 must match the receipt.
    """
    if type(page) is not int or page < 0:
        raise FindAllError("findall_snapshot_page_invalid")
    _page_layout(receipt)
    return page_view(receipt, read_bytes(receipt["file"]), page)


def validate_snapshot(receipt, read_bytes):
    """Export verifies the whole file: exact bytes, page layout and complete JSON."""
    _page_layout(receipt)
    raw = read_bytes(receipt["file"])
    page_view(receipt, raw)
    json.loads(raw)  # A complete JSON snapshot, not a truncation or independent partial records.


class _OwnerJournal:
    def __init__(self, ledger, day, prepared, current_authority, assert_current_lease=lambda: True):
        self.ledger, self.day = ledger, day
        self.prepared = copy.deepcopy(prepared)
        self.current_authority = current_authority
        self.assert_current_lease = assert_current_lease
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

    def authorize_submission(self, prepared):
        """Fresh post-commit authority, excluding only this exact reservation."""
        if (not self.claimed or prepared != self.prepared
                or self.assert_current_lease() is not True):
            return False
        row = self._row()
        key = _operation_key(prepared["operation_id"])
        entry = _slots(row).get(key)
        if (not isinstance(entry, dict) or entry.get("prepared") != prepared
                or entry.get("allocation_binding_digest") != prepared["allocation_binding_digest"]
                or entry.get("state") != "submission_unresolved" or entry.get("findall_id") is not None):
            return False
        # The reservation is already durable. Remove it only from the copied
        # admission view, so the exact request is counted once with all others.
        current = copy.deepcopy(row)
        current[SUBMISSIONS_FIELD].pop(key)
        return (self.current_authority(current, copy.deepcopy(prepared)) is True
                and self.assert_current_lease() is True)

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
        name = f"{self.day}-tool-findall-{key}.json"
        receipt = retain_snapshot(self.ledger, name, run)
        entry.update(state="receipt_retained", receipt_file=name,
                     receipt_sha256=receipt["sha256"], receipt=receipt)
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
    except Exception:  # noqa: BLE001 - retain known IDs and sanitize release/store failures
        # A lease-release/storage failure after the provider attempt must still
        # expose its known ID and retain the claim, never invite a replay.
        if journal.claimed:
            raise FindAllSubmissionUnresolved(findall_id=journal.known_id) from None
        raise FindAllError("findall_owner_ledger_failed") from None


def create_under_owner_lease(
    client: AdmittedFindAllClient,
    spec: Mapping[str, Any],
    *,
    ledger: OwnerResearchLedger,
    day: str,
    operation_id: str,
    maximum_cost_usd: str,
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None,
    current_authority: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    assert_current_lease: Callable[[], bool],
) -> dict[str, Any]:
    """Dispatch inside the daily consumer's existing owner lease.

    The trusted owner must freshly assert its actual process lock/fenced lease.
    No lock is acquired or released here. The same durable journal, complete
    history, exact grant and current-authority checks govern the single POST.
    """
    if (not isinstance(client, AdmittedFindAllClient) or not callable(current_authority)
            or not callable(assert_current_lease) or assert_current_lease() is not True):
        raise FindAllError("findall_owner_current_lease_required")
    prepared = prepare_submission(spec, operation_id=operation_id,
                                  maximum_cost_usd=maximum_cost_usd)
    require_paid_resource_admission_grant(
        paid_resource_admission_grant, resource_class=PAID_RESOURCE_CLASS,
        allocation_binding_digest=prepared["allocation_binding_digest"],
        require_allocation_binding=True,
    )
    journal = _OwnerJournal(ledger, day, prepared, current_authority, assert_current_lease)
    try:
        return client.create(
            spec, operation_id=operation_id, maximum_cost_usd=maximum_cost_usd,
            paid_resource_admission_grant=paid_resource_admission_grant, journal=journal,
        )
    except (FindAllError, PaidResourceAdmissionBlocked):
        raise
    except Exception:  # noqa: BLE001 - uncertain starts must remain recoverable without exposing store prose
        if journal.claimed:
            raise FindAllSubmissionUnresolved(findall_id=journal.known_id) from None
        raise FindAllError("findall_owner_ledger_failed") from None
