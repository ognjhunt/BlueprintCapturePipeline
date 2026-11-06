"""Retire one parked website preparation delivery, with the owner's approval on the record.

D-05 (2026-11-02) parks a status callback whose read has failed on every attempt
for a day: it stays pending, retries every six hours and is counted apart from
the due backlog, so nothing is dropped. Some never recover. On 2026-11-06 seven
owner test captures from 10-28/29 had been parked for over a week: WebApp #1113
changed the digest of the unconfirmed job context their status reads are bound
to, so no read can succeed again. The owner approved retiring them.

This is the general operation for any such delivery. Without ``--apply`` it
plans: it reads the delivery and says what it would retire and why, or why not.
With ``--apply`` it re-checks the same things under the capture's ledger lock
(the lock every controller writer of this record holds) and moves the record to
the terminal ``retired`` state, keeping every field it had and adding when,
why, on whose approval and through which door request. The controller's own
index update then removes its hint, so it is never due, retried or counted
again. A newer producer revision still starts a new delivery, as it does today.

Only a delivery that is pending and parked for at least a day
(``PARK_AFTER_SECONDS``) for ``PARKED_REASON`` can be retired. A live, delivered
or unreadable record is refused. Nothing else in the capture is read for
authority, changed or deleted, and no network is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "website_preparation_delivery_retirement.v1"
#: ``<bucket>/scenes/site-<id>/captures/(walkthrough|supplement)-<id>``: the shape the index keys.
CAPTURE_PATH = re.compile(
    r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]/scenes/site-[A-Za-z0-9][A-Za-z0-9._-]{0,122}"
    r"/captures/(?:walkthrough|supplement)-[A-Za-z0-9][A-Za-z0-9._-]{0,114}")
_CALLER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_REQUEST_ID = re.compile(r"\d{8}T\d{6}Z-retire-parked-delivery-[0-9a-f]{8}")


class DeliveryRetirementRefused(ValueError):
    """A typed reason this delivery is not retired."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _refuse(condition: bool, code: str) -> None:
    if not condition:
        raise DeliveryRetirementRefused(code)


def _evidence(value: Any) -> str:
    from .website_preparation_status import MAX_RETIREMENT_EVIDENCE

    _refuse(isinstance(value, str) and value == value.strip() and 1 <= len(value) <= MAX_RETIREMENT_EVIDENCE
            and value.isprintable(), "parked_delivery_approval_evidence_invalid")
    return value


def _capture_root(storage_root: Path, relative: Any) -> Path:
    from .website_preparation_delivery_index import _location

    _refuse(isinstance(relative, str) and CAPTURE_PATH.fullmatch(relative) is not None
            and ".." not in relative.split("/"), "parked_delivery_capture_invalid")
    root = Path(storage_root).absolute() / relative
    try:
        storage, keyed = _location(root)
    except (ValueError, IndexError):
        raise DeliveryRetirementRefused("parked_delivery_capture_invalid") from None
    _refuse(storage == Path(storage_root).absolute() and keyed == relative, "parked_delivery_capture_invalid")
    return root


def _summary(delivery: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("state", "revision", "pending_since_epoch", "parked_at_epoch", "parked_reason",
            "status_unavailable_since_epoch", "last_unavailable_reason", "delivery_attempt_count",
            "delivery_failure_count", "last_error_type", "retired_at_epoch", "retired_reason")
    return {key: delivery[key] for key in keys if key in delivery}


def retire_parked_delivery(*, storage_root: Path, capture_relative_path: str, approval_evidence: str,
                           apply: bool = False, door_caller: str | None = None,
                           door_request_id: str | None = None, now: float | None = None) -> dict[str, Any]:
    """Plan, or with ``apply`` perform, one parked delivery's retirement. Raises ``DeliveryRetirementRefused``."""
    from .common import write_json
    from .handoff_job_state import _existing_job_ledger_lock
    from .website_preparation_delivery_index import update as update_index
    from .website_preparation_status import (DELIVERY_FILE, PARK_AFTER_SECONDS, PARKED_REASON, RETIRED_REASON,
                                             RETIRED_STATE, _read_delivery)

    _refuse(type(apply) is bool, "parked_delivery_apply_invalid")
    evidence = _evidence(approval_evidence)
    _refuse(door_caller is None or (isinstance(door_caller, str) and _CALLER.fullmatch(door_caller) is not None),
            "parked_delivery_caller_invalid")
    _refuse(door_request_id is None
            or (isinstance(door_request_id, str) and _REQUEST_ID.fullmatch(door_request_id) is not None),
            "parked_delivery_request_id_invalid")
    root = _capture_root(Path(storage_root), capture_relative_path)
    now = time.time() if now is None else float(now)
    result: dict[str, Any] = {"schema_version": SCHEMA, "capture_relative_path": capture_relative_path,
                              "apply": apply}
    with _existing_job_ledger_lock(root) as lock_state:
        if lock_state == "capture_absent":
            # The capture's workspace is gone (retired, 2026-11-06: four of the seven), so only its
            # parked index hint is left. That hint is archived instead; see _retire_orphaned_hint.
            return _retire_orphaned_hint(Path(storage_root), capture_relative_path, evidence=evidence,
                                         apply=apply, door_caller=door_caller, door_request_id=door_request_id,
                                         now=now, result=result)
        _refuse(lock_state == "ledger_present", f"parked_delivery_{lock_state}")
        path = root / DELIVERY_FILE
        _refuse(path.is_file() and not path.is_symlink(), "parked_delivery_record_missing")
        before = path.read_bytes()
        try:
            delivery = _read_delivery(path)
        except (ValueError, OSError):  # PreparationStatusUnavailable and malformed JSON alike
            raise DeliveryRetirementRefused("parked_delivery_record_invalid") from None
        result.update(prior_record_sha256="sha256:" + hashlib.sha256(before).hexdigest(),
                      delivery=_summary(delivery))
        if delivery["state"] == RETIRED_STATE:
            if apply:
                update_index(root, delivery)  # A hint left by an interrupted apply goes now.
            return {**result, "outcome": "already_retired"}
        _refuse(delivery["state"] == "pending", "parked_delivery_not_pending")
        parked_at = delivery.get("parked_at_epoch")
        _refuse(delivery.get("parked_reason") == PARKED_REASON and type(parked_at) in (int, float)
                and math.isfinite(parked_at), "parked_delivery_not_parked")
        _refuse(now - parked_at >= PARK_AFTER_SECONDS, "parked_delivery_parked_under_a_day")
        retired = {**delivery, "state": RETIRED_STATE, "retired_at_epoch": now, "retired_reason": RETIRED_REASON,
                   "retirement_approval_evidence": evidence}
        if door_caller is not None:
            retired["retired_by_door_caller"] = door_caller
        if door_request_id is not None:
            retired["retirement_door_request_id"] = door_request_id
        if not apply:
            return {**result, "outcome": "planned", "would_set": {key: retired[key] for key in retired
                                                                 if key not in delivery or key == "state"}}
        write_json(path, retired)
        update_index(root, retired)
        after = path.read_bytes()
    return {**result, "outcome": "retired", "record_sha256": "sha256:" + hashlib.sha256(after).hexdigest(),
            "delivery": _summary(retired)}


DIRECTORY_NAME = ".website_preparation_pending"
ORPHAN_SCHEMA = "website_preparation_delivery_orphan_retirement.v1"
ORPHAN_REASON = "owner_retired_orphaned_parked_delivery"
#: Retired orphan hints are kept here, beside the index; the index scan reads only top-level hints.
ORPHAN_DIRECTORY = "retired"


def _retire_orphaned_hint(storage_root: Path, relative: str, *, evidence: str, apply: bool,
                          door_caller: str | None, door_request_id: str | None, now: float,
                          result: dict[str, Any]) -> dict[str, Any]:
    """A parked hint whose capture no longer exists: archive it with the approval, never delete it.

    With no capture there is no delivery record to move, and no status read can succeed, so the
    hint would retry every six hours and count as parked forever. Under the index lock it must
    still be the same capture's hint, parked for at least a day, and the capture must still be
    absent. The archive keeps the whole hint, plus when, why, on whose approval and through which
    door request. A restored workspace whose record is still pending makes a new hint, which can be
    retired again.
    """
    from .common import write_json
    from .website_preparation_delivery_index import SCHEMA_VERSION, _locked
    from .website_preparation_status import PARK_AFTER_SECONDS, PARKED_REASON

    key = hashlib.sha256(relative.encode()).hexdigest()
    with _locked(storage_root) as directory:
        archive = directory / ORPHAN_DIRECTORY / f"{key}.json"
        hint_path = directory / f"{key}.json"
        if not hint_path.exists() and archive.is_file() and not archive.is_symlink():
            return {**result, "outcome": "already_retired", "capture_absent": True}
        _refuse(hint_path.is_file() and not hint_path.is_symlink(), "parked_delivery_capture_absent")
        try:
            hint = json.loads(hint_path.read_text())
        except (ValueError, OSError):
            raise DeliveryRetirementRefused("parked_delivery_hint_invalid") from None
        _refuse(isinstance(hint, dict) and hint.get("schema_version") == SCHEMA_VERSION
                and hint.get("capture_relative_path") == relative, "parked_delivery_hint_invalid")
        parked_at = hint.get("parked_at_epoch")
        _refuse(type(parked_at) in (int, float) and math.isfinite(parked_at) and parked_at >= 0,
                "parked_delivery_not_parked")
        _refuse(now - parked_at >= PARK_AFTER_SECONDS, "parked_delivery_parked_under_a_day")
        _refuse(not (storage_root / relative).exists(), "parked_delivery_capture_reappeared")
        record = {"schema_version": ORPHAN_SCHEMA, "capture_relative_path": relative, "hint": hint,
                  "parked_reason_assumed": PARKED_REASON, "capture_absent": True, "retired_at_epoch": now,
                  "retired_reason": ORPHAN_REASON, "retirement_approval_evidence": evidence}
        if door_caller is not None:
            record["retired_by_door_caller"] = door_caller
        if door_request_id is not None:
            record["retirement_door_request_id"] = door_request_id
        summary = {**result, "capture_absent": True, "hint": {k: hint[k] for k in (
            "revision", "pending_since_epoch", "parked_at_epoch") if k in hint}}
        if not apply:
            return {**summary, "outcome": "planned",
                    "would_archive_to": f"{DIRECTORY_NAME}/{ORPHAN_DIRECTORY}/{key}.json"}
        archive.parent.mkdir(mode=0o700, exist_ok=True)
        _refuse(not archive.parent.is_symlink(), "parked_delivery_archive_unsafe")
        write_json(archive, record)
        hint_path.unlink()
    return {**summary, "outcome": "retired", "archive_sha256": "sha256:" + hashlib.sha256(
        archive.read_bytes()).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    from .website_preparation_delivery_index import DEFAULT_STORAGE_ROOT, STORAGE_ROOT_ENV

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", help="<bucket>/scenes/site-*/captures/<capture>")
    parser.add_argument("--approval-evidence", help="names the owner's approval (<= 500 chars)")
    parser.add_argument("--apply", action="store_true", help="retire it; without this only a plan is made")
    parser.add_argument("--request-json", help="the operator door's validated request, in place of the three above")
    parser.add_argument("--door-caller")
    parser.add_argument("--door-request-id")
    args = parser.parse_args(argv)
    storage = Path(os.environ.get(STORAGE_ROOT_ENV) or DEFAULT_STORAGE_ROOT)
    try:
        if args.request_json is not None:
            _refuse(args.capture is None and args.approval_evidence is None and not args.apply,
                    "parked_delivery_request_invalid")
            try:
                request = json.loads(args.request_json)
            except ValueError:
                raise DeliveryRetirementRefused("parked_delivery_request_invalid") from None
            _refuse(type(request) is dict and request.get("kind") == "retire-parked-delivery"
                    and set(request) == {"kind", "capture", "approval_evidence", "apply"},
                    "parked_delivery_request_invalid")
            capture, evidence, apply = request["capture"], request["approval_evidence"], request["apply"]
        else:
            capture, evidence, apply = args.capture, args.approval_evidence, args.apply
        row = retire_parked_delivery(storage_root=storage, capture_relative_path=capture,
                                     approval_evidence=evidence, apply=apply,
                                     door_caller=args.door_caller, door_request_id=args.door_request_id)
    except DeliveryRetirementRefused as refusal:
        print(json.dumps({"schema_version": SCHEMA, "outcome": "refused", "code": refusal.code}, sort_keys=True))
        return 2
    print(json.dumps({**row, "code": None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
