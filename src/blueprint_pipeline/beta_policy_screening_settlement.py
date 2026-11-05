"""Settle a finished beta screening attempt down to its own run's measured provider cost.

Beta journey step: L5 candidate screening. The coordinator reserves each candidate's cap from the
site's allowance (``SITE_LIMIT``, $15 then) and commits the whole cap once the provider allocated. Nothing
settled those commitments: capture-447ae4f4's three learned-policy attempts (2026-11-04) kept $12 of
the $15 while their Vast budget ledgers estimated $0.44, $1.31 and $0.91 (runtime plus priced
transfer), so no further $4 candidate could be funded at the site.

``attempt_settlement`` settles one committed attempt once its own records prove nothing can spend:

- the allocator result says nothing continues to spend and every staged object is gone, and its
  provider closeout confirmed provider zero with no warm session kept;
- a retained scoped provider-absence receipt (``beta_policy_screening_closeout``) for this attempt's
  prepared record and exactly this result is verified: the instance is absent and nothing live
  carries it or its resource name;
- the teardown manifest and adapter result the closeout names, re-hashed inside the attempt's own
  directory, destroyed exactly that one instance;
- the adapter's budget ledger beside the teardown is complete for that instance, within the
  attempt's provider cap, and is the whole session's estimate.

Vast's instance API reports no bill for these runs (the ledger's ``actual_cost_usd`` is null), so
the cost settles at the ledger's estimate: the observed all-in hourly rate (compute and disk) over
the observed live time, plus the priced transfer, rounded up to the cent. A candidate with a
Blueprint-paid hosted policy endpoint adds that endpoint's whole bound, which no record measures.
The basis is ``adapter_budget_ledger_estimate``, as in the rigid robot-free probe lane
(``rigid_robot_free_probe_settlement``), never a provider charge, and the figure never exceeds the
reserved maximum. A ledger naming a charge this module does not know, or anything else the records
do not prove, raises ``SettlementEvidenceError``: the attempt keeps its full maximum.

``settled_cost`` reads a recorded settlement back for the site budget; one that does not verify
counts the attempt at its maximum again.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SETTLEMENT_SCHEMA = "blueprint.beta_policy_screening_settlement.v1"
SETTLEMENT_BASIS = "adapter_budget_ledger_estimate"
RESULT_FILENAME = "allocator_result.json"
LEDGER_FILENAME = "vast_budget_ledger.json"
ABSENCE_DIRNAME = "provider_absence"
LEDGER_COST_SOURCE = "not_available_from_instance_probe_api"
#: Every reconciliation pass of an attempt retains one receipt; far fewer are ever written.
MAX_ABSENCE_RECEIPTS = 64
_RUNTIME = "estimated_runtime_cost_usd"
#: The provider charges the ledger records beside runtime. Disk storage is billed inside the
#: all-in hourly rate the runtime figure uses (``vast_all_in_cost_binding.json``).
_NON_RUNTIME = ("projected_provider_transfer_cost_usd",)
#: The ledger's cost fields: its total, its components and the bill it cannot read. Any other
#: ``*_cost_usd`` field is a charge this module would not count, so the attempt is not settled.
_LEDGER_COST_FIELDS = frozenset({"estimated_cost_usd", "actual_cost_usd", _RUNTIME, *_NON_RUNTIME})
_CENT = Decimal("0.01")
_MICRO = Decimal("0.000001")
# The ledger rounds its total and each component to six places separately.
_TOLERANCE = Decimal("0.000002")
_MONEY = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,6})?")


class SettlementEvidenceError(ValueError):
    """The attempt's records do not prove one finished, torn-down run; nothing is settled."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SettlementEvidenceError(reason)


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _text(value: Decimal) -> str:
    """Wire money: at most six places, rounded up, never down."""
    value = value.quantize(_MICRO, rounding=ROUND_CEILING)
    return format(value, "f").rstrip("0").rstrip(".") if value else "0"


def _dollars(value: Any, reason: str) -> Decimal:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
             and value >= 0, reason)
    return Decimal(str(value))


def _regular(path: Path, *, inside: Path) -> bytes:
    """One regular file inside the attempt's own directory, read without following a link."""
    _require(path.is_absolute() and ".." not in path.parts and not path.is_symlink() and path.is_file()
             and path.resolve().is_relative_to(inside), f"beta_screening_settlement_record_missing:{path.name}")
    return path.read_bytes()


def _object(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    _require(isinstance(value, dict), f"beta_screening_settlement_record_invalid:{name}")
    return value


def _bound(record: Any, *, inside: Path, name: str) -> tuple[dict[str, Any], str]:
    """A record the closeout names by path, size and digest, re-read and re-hashed in the attempt."""
    _require(isinstance(record, Mapping) and isinstance(record.get("path"), str)
             and isinstance(record.get("sha256"), str), f"beta_screening_settlement_reference_invalid:{name}")
    raw = _regular(Path(record["path"]), inside=inside)
    _require(len(raw) == record.get("size_bytes") and _sha256(raw) == record["sha256"],
             f"beta_screening_settlement_reference_changed:{name}")
    return _object(raw, name), record["sha256"]


def _absence_receipt(attempt_dir: Path, *, inside: Path, prepared_digest: str, result: Mapping[str, Any],
                     instance_id: int) -> dict[str, Any]:
    """The retained, verified scoped closeout of exactly this attempt's result and instance."""
    from .beta_policy_screening_closeout import SCHEMA, SCOPE, _observation_blockers

    directory = attempt_dir / ABSENCE_DIRNAME
    _require(directory.is_dir() and not directory.is_symlink(), "beta_screening_settlement_provider_zero_unproven")
    adapter_digest = canonical_digest(dict(result))
    for path in sorted(directory.glob("*.json"))[:MAX_ABSENCE_RECEIPTS]:
        try:
            receipt = _object(_regular(path, inside=inside), path.name)
            if (receipt.get("schema_version") == SCHEMA and receipt.get("scope") == SCOPE
                    and receipt.get("provider") == "vast" and receipt.get("status") == "verified"
                    and receipt.get("provider_absence_confirmed") is True and receipt.get("blockers") == []
                    and receipt.get("global_provider_zero_claimed") is False
                    and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
                    and path.name == canonical_digest(receipt)[7:] + ".json"
                    and receipt.get("prepared_digest") == prepared_digest
                    and receipt.get("adapter_digest") == adapter_digest
                    and receipt.get("instance_id") == instance_id
                    and not _observation_blockers(receipt, now=receipt.get("observed_at_epoch"))):
                return receipt
        except (SettlementEvidenceError, ValueError, TypeError, KeyError, OverflowError):
            continue
    raise SettlementEvidenceError("beta_screening_settlement_provider_zero_unproven")


def _hosted_maximum(candidate: Mapping[str, Any]) -> Decimal:
    """A Blueprint-paid hosted endpoint's whole bound: nothing measures what it charged."""
    execution = candidate.get("execution_spec") or {}
    if "credential_file" not in execution and "hosted_inference_maximum_usd" not in execution:
        return Decimal(0)
    value = execution.get("hosted_inference_maximum_usd")
    _require(isinstance(value, str) and _MONEY.fullmatch(value) is not None,
             "beta_screening_settlement_hosted_cost_unknown")
    return Decimal(value)


def attempt_settlement(*, attempt_dir: Path, attempt: Mapping[str, Any], candidate: Mapping[str, Any],
                       settled_at: str) -> dict[str, Any]:
    """The settlement this committed attempt's own records prove, as fields for the attempt.

    Raises ``SettlementEvidenceError`` when they prove less, and ``FileNotFoundError`` when the
    attempt's directory cannot be found at all (a volume not mounted yet), so a caller can retry.
    """
    attempt_dir = Path(attempt_dir)
    if not attempt_dir.is_dir():
        raise FileNotFoundError(str(attempt_dir))
    try:
        return _settlement(attempt_dir, attempt, candidate, settled_at)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, SettlementEvidenceError):
            raise
        raise SettlementEvidenceError("beta_screening_settlement_record_invalid") from exc


def _settlement(attempt_dir: Path, attempt: Mapping[str, Any], candidate: Mapping[str, Any],
                settled_at: str) -> dict[str, Any]:
    inside = attempt_dir.resolve()
    prepared = attempt.get("prepared")
    _require(attempt_dir.is_absolute() and not attempt_dir.is_symlink()
             and attempt_dir.name == attempt.get("attempt_id")
             and attempt.get("reservation_status") == "committed"
             and attempt.get("provider_allocation_observed") is True
             and isinstance(prepared, Mapping)
             and prepared.get("prepared_digest") == canonical_digest(dict(prepared), digest_field="prepared_digest")
             and prepared.get("attempt_dir") == str(inside)
             and prepared.get("candidate_id") == candidate.get("candidate_id"),
             "beta_screening_settlement_attempt_unbound")
    maximum = attempt.get("maximum_cost_usd")
    _require(isinstance(maximum, str) and _MONEY.fullmatch(maximum) is not None,
             "beta_screening_settlement_attempt_unbound")
    maximum = Decimal(maximum)

    raw = _regular(attempt_dir / RESULT_FILENAME, inside=inside)
    result = _object(raw, RESULT_FILENAME)
    closeout = result.get("provider_closeout")
    _require(result.get("continuing_spend_from_this_run") is False and result.get("all_staged_objects_absent") is True
             and isinstance(closeout, Mapping) and closeout.get("provider_zero_confirmed") is True
             and closeout.get("warm_session_retained") is False and closeout.get("all_staged_objects_absent") is True,
             "beta_screening_settlement_teardown_unproven")
    teardown, teardown_digest = _bound(closeout.get("teardown_manifest"), inside=inside, name="teardown")
    ids, actions = teardown.get("vast_instance_ids"), teardown.get("teardown_actions_performed")
    _require(teardown.get("schema_version") == "vast_teardown_manifest.v1" and teardown.get("status") == "completed"
             and teardown.get("runner_gpu_teardown_completed") is True
             and teardown.get("continuing_spend_from_this_run") is False
             and teardown.get("retention_authorized") is False
             and isinstance(ids, list) and len(ids) == 1 and type(ids[0]) is int and ids[0] > 0
             and isinstance(actions, list) and bool(actions)
             and all(isinstance(row, Mapping) and row.get("status") == "completed" and row.get("instance_id") == ids[0]
                     for row in actions)
             and any(row.get("action") == "destroy_instance" for row in actions),
             "beta_screening_settlement_teardown_unproven")
    instance_id = ids[0]
    adapter, adapter_digest = _bound(closeout.get("adapter_result"), inside=inside, name="adapter_result")
    _require(adapter.get("vast_instance_ids") == ids and adapter.get("continuing_spend_from_this_run") is False,
             "beta_screening_settlement_adapter_mismatch")
    absence = _absence_receipt(attempt_dir, inside=inside, prepared_digest=prepared["prepared_digest"],
                               result=result, instance_id=instance_id)

    ledger_raw = _regular(Path(closeout["teardown_manifest"]["path"]).parent / LEDGER_FILENAME, inside=inside)
    ledger = _object(ledger_raw, LEDGER_FILENAME)
    _require(ledger.get("schema_version") == "vast_budget_ledger.v1" and ledger.get("status") == "completed"
             and ledger.get("continuing_spend_from_this_run") is False and ledger.get("vast_instance_ids") == ids
             and ledger.get("provider_transfer_cost_priced") is True
             and ledger.get("estimated_spend_under_hard_cap") is True
             and ledger.get("actual_cost_usd") is None and ledger.get("actual_cost_source") == LEDGER_COST_SOURCE,
             "beta_screening_settlement_ledger_invalid")
    _require(not [key for key in ledger if key.endswith("_cost_usd") and key not in _LEDGER_COST_FIELDS],
             "beta_screening_settlement_ledger_charge_unknown")
    runtime = _dollars(ledger.get(_RUNTIME), "beta_screening_settlement_ledger_invalid")
    charges = {key: _dollars(ledger.get(key), "beta_screening_settlement_ledger_invalid") for key in _NON_RUNTIME}
    total = _dollars(ledger.get("estimated_cost_usd"), "beta_screening_settlement_ledger_invalid")
    components = runtime + sum(charges.values(), Decimal(0))
    _require(abs(total - components) <= _TOLERANCE and runtime > 0, "beta_screening_settlement_ledger_inconsistent")
    # One allocation: the session's own figures are this ledger's, or it is not this run's whole cost.
    for figure in (result.get("estimated_cost_usd"), closeout.get("estimated_cost_usd")):
        _require(abs(_dollars(figure, "beta_screening_settlement_estimate_not_this_run") - total) <= _TOLERANCE,
                 "beta_screening_settlement_estimate_not_this_run")
    hosted = _hosted_maximum(candidate)
    _require(_dollars(ledger.get("hard_cap_usd"), "beta_screening_settlement_ledger_invalid") <= maximum - hosted,
             "beta_screening_settlement_ledger_cap_not_this_attempt")

    settled = max(total, components).quantize(_CENT, rounding=ROUND_CEILING) + hosted
    _require(settled <= maximum, "beta_screening_settlement_above_reservation")
    receipt = {
        "schema_version": SETTLEMENT_SCHEMA, "attempt_id": attempt["attempt_id"],
        "candidate_id": candidate["candidate_id"], "prepared_digest": prepared["prepared_digest"],
        "provider": "vast", "instance_id": instance_id, "maximum_cost_usd": attempt["maximum_cost_usd"],
        "settlement_basis": SETTLEMENT_BASIS, "settled_cost_usd": _text(settled), "settled_at": settled_at,
        "estimated_runtime_cost_usd": _text(runtime),
        "non_runtime_charges_usd": {key: _text(value) for key, value in charges.items()},
        "ledger_estimated_cost_usd": _text(total), "hosted_inference_maximum_usd": _text(hosted),
        "actual_cost_usd": None, "actual_cost_source": LEDGER_COST_SOURCE,
        "budget_ledger_digest": _sha256(ledger_raw), "execution_result_digest": _sha256(raw),
        "teardown_receipt_digest": teardown_digest, "adapter_result_digest": adapter_digest,
        "provider_absence_receipt_digest": absence["receipt_digest"],
        "continuing_spend_from_this_run": False, "all_staged_objects_absent": True, "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return {"settled_cost_usd": receipt["settled_cost_usd"], "settlement_basis": SETTLEMENT_BASIS,
            "settled_at": settled_at, "settlement_receipt_digest": receipt["receipt_digest"],
            "settlement_receipt": receipt}


def settled_cost(attempt: Mapping[str, Any]) -> Decimal | None:
    """What a committed attempt's recorded settlement counts, or None to count its whole maximum."""
    settled = attempt.get("settled_cost_usd")
    if settled is None:
        return None
    receipt = attempt.get("settlement_receipt")
    try:
        if (not isinstance(settled, str) or not _MONEY.fullmatch(settled) or not isinstance(receipt, Mapping)
                or attempt.get("settlement_basis") != SETTLEMENT_BASIS
                or receipt.get("schema_version") != SETTLEMENT_SCHEMA
                or receipt.get("settlement_basis") != SETTLEMENT_BASIS
                or receipt.get("settled_cost_usd") != settled
                or receipt.get("settled_at") != attempt.get("settled_at")
                or receipt.get("attempt_id") != attempt.get("attempt_id")
                or receipt.get("maximum_cost_usd") != attempt.get("maximum_cost_usd")
                or receipt.get("receipt_digest") != attempt.get("settlement_receipt_digest")
                or receipt["receipt_digest"] != canonical_digest(dict(receipt), digest_field="receipt_digest")):
            return None
        value, maximum = Decimal(settled), Decimal(str(attempt["maximum_cost_usd"]))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return None
    return value if 0 <= value <= maximum else None


__all__ = ["SETTLEMENT_BASIS", "SETTLEMENT_SCHEMA", "SettlementEvidenceError", "attempt_settlement", "settled_cost"]
