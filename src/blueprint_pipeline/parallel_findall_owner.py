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


SNAPSHOT_SCHEMA = "blueprint.findall-snapshot-parts.v1"
SNAPSHOT_FRAGMENT_CHARS = 24_000  # Escaped output remains below the 500 kB tool ceiling.
SMALL_SNAPSHOT_BYTES = 200_000


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def retain_snapshot(ledger, name, snapshot):
    """Retain the whole snapshot in immutable, read-back-checked bounded artifacts."""
    raw = _json_bytes(snapshot)
    # Size the small path using the tool's ASCII-escaped serialization too.
    escaped = json.dumps(snapshot, ensure_ascii=True, allow_nan=False).encode()
    if max(len(raw), len(escaped)) <= SMALL_SNAPSHOT_BYTES:
        ledger.write_bytes(name, raw)
        if ledger.read_bytes(name) != raw:
            raise FindAllError("findall_snapshot_readback_failed")
        return {"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    snapshot_sha = hashlib.sha256(raw).hexdigest()
    text, parts = raw.decode("utf-8"), []
    for index, start in enumerate(range(0, len(text), SNAPSHOT_FRAGMENT_CHARS)):
        part = {"schema_version": SNAPSHOT_SCHEMA, "snapshot_sha256": snapshot_sha,
                "page": index, "json_fragment": text[start:start + SNAPSHOT_FRAGMENT_CHARS]}
        value = _json_bytes(part)
        filename = name[:-5] + f"-part-{index:05d}.json"
        ledger.write_bytes(filename, value)
        if ledger.read_bytes(filename) != value:
            raise FindAllError("findall_snapshot_readback_failed")
        parts.append({"file": filename, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)})
    manifest = {"schema_version": SNAPSHOT_SCHEMA, "snapshot_sha256": snapshot_sha,
                "snapshot_bytes": len(raw), "parts": parts}
    value = _json_bytes(manifest)
    ledger.write_bytes(name, value)
    if ledger.read_bytes(name) != value:
        raise FindAllError("findall_snapshot_readback_failed")
    return {"file": name, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value), **manifest}


def snapshot_page(receipt, read_bytes, page=0):
    """An immutable receipt-bound page; continuation never fetches a new snapshot."""
    if type(page) is not int or page < 0:
        raise FindAllError("findall_snapshot_page_invalid")
    raw = read_bytes(receipt["file"])
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"] or len(raw) != receipt["bytes"]:
        raise FindAllError("findall_snapshot_binding_invalid")
    value = json.loads(raw)
    shown = {key: receipt[key] for key in ("file", "sha256", "bytes")}
    if "parts" not in receipt:
        if page != 0:
            raise FindAllError("findall_snapshot_page_invalid")
        return {"snapshot": value, "receipt": shown}
    expected = {key: receipt[key] for key in ("schema_version", "snapshot_sha256", "snapshot_bytes", "parts")}
    if value != expected or page >= len(receipt["parts"]):
        raise FindAllError("findall_snapshot_binding_invalid")
    ref = receipt["parts"][page]
    part_raw = read_bytes(ref["file"])
    if hashlib.sha256(part_raw).hexdigest() != ref["sha256"] or len(part_raw) != ref["bytes"]:
        raise FindAllError("findall_snapshot_binding_invalid")
    part = json.loads(part_raw)
    if (set(part) != {"schema_version", "snapshot_sha256", "page", "json_fragment"}
            or part["schema_version"] != SNAPSHOT_SCHEMA or part["snapshot_sha256"] != receipt["snapshot_sha256"]
            or type(part["page"]) is not int or part["page"] != page
            or not isinstance(part["json_fragment"], str)
            or not 0 < len(part["json_fragment"]) <= SNAPSHOT_FRAGMENT_CHARS):
        raise FindAllError("findall_snapshot_binding_invalid")
    return {"json_fragment": part["json_fragment"], "encoding": "utf-8",
            "page": page, "page_count": len(receipt["parts"]),
            "next_page": page + 1 if page + 1 < len(receipt["parts"]) else None,
            "snapshot_sha256": receipt["snapshot_sha256"], "snapshot_bytes": receipt["snapshot_bytes"],
            "receipt": shown}


def validate_snapshot(receipt, read_bytes):
    """Export verifies the full byte sequence, including every unknown field."""
    first = snapshot_page(receipt, read_bytes)
    if "parts" not in receipt:
        return
    raw = "".join(snapshot_page(receipt, read_bytes, page)["json_fragment"]
                  for page in range(first["page_count"])).encode("utf-8")
    if len(raw) != receipt["snapshot_bytes"] or hashlib.sha256(raw).hexdigest() != receipt["snapshot_sha256"]:
        raise FindAllError("findall_snapshot_binding_invalid")
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
